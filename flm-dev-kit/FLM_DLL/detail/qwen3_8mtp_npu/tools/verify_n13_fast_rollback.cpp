// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify_n13_fast_rollback.cpp
/// \brief N13 -- recovering from a rejected draft **without re-running the
///        model**: `qwen38::decoder_layer::commit_prefix()` on
///        `q38npu::npu_delta_layer`.
///
/// Build:
///   source ~/setup.sh
///   g++-13 -O2 -march=native -fopenmp -std=c++20
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n13_fast_rollback.cpp common/cpu_ref_delta_net_layer.cpp
///       verify/npu/delta_sequence.cpp speculate/npu/proj_sequence.cpp
///       ../q4nx/q4_npu_eXpress.cpp ../q4nx/safe_tensors.cpp
///       -L../../../../src/lib/xrt -L../../build/lib -L/opt/xilinx/xrt/lib
///       -lxrt_coreutil -laiebu -luuid -lpthread -o /tmp/verify_n13_fast_rollback
///   (one line; the continuations are dropped here because a backslash at the
///    end of a // comment splices the next line into it)
///
/// Run:
///   /tmp/verify_n13_fast_rollback [model_dir] [deltanet_layer.xclbin]
///
/// ---------------------------------------------------------------------------
/// What this phase removes
/// ---------------------------------------------------------------------------
/// A speculative batch has to run all `k+1` rows before anyone can know how
/// many of them are good, so by the time `commit` is known the stack has
/// already folded the rejected tokens into its state. The three carried
/// tensors are then wrong in two quite different ways:
///
///   * an attention KV cache **is a list** -- rows above the cursor were
///     appended but are never read again, so rewinding the cursor is the
///     entire correction, and it is one `int`;
///   * a delta-net block's conv window and recurrent state **have no length**.
///     `S_t = a_t S_{t-1} + k_t (x) u_t` sums every token it ever saw into one
///     tensor and the fold has no inverse, and the conv window is a 3-tap
///     slider that has already slid.
///
/// Until this phase the only way to advance those two by n tokens was to run
/// the model on n tokens, so `prefill_verify()` restored a snapshot and
/// re-prefilled the accepted prefix: a full 64-layer weight stream, ~14 GB,
/// paid purely to recover two tensors. On a 2-cycle 11-token session it was
/// **3347.6 ms, 46.2% of the cycle**, for a single rejection of 3 accepted
/// tokens -- more than the entire 16-row verify phase beside it.
///
/// N-DELTA-VERIFY made that unnecessary without anyone noticing. The bitstream
/// writes `L_TOKENS + conv_kernel_size - 2` conv rows, which is one window per
/// accept length, and the triplet arena it writes for its own transpose holds
/// the post-conv, post-l2norm, post-sigmoid k / alpha / v / beta. Between them
/// the state at *every* prefix 1..8 is recoverable on the host for the price of
/// a memcpy and a fold.
///
/// ---------------------------------------------------------------------------
/// Why the bar here is bit identity and not a cosine
/// ---------------------------------------------------------------------------
/// `commit_prefix(slot, j)` claims to land exactly where `forward()` of only
/// the first `j` rows would land, from the same starting state. That is an
/// *exact* claim, not an approximate one, and it should be checkable as one:
///
///   * the device's rows 0..j-1 are causally independent of rows j..7 -- causal
///     conv, causal recurrence, and the padding rows a short window repeats sit
///     strictly above them -- so the arena blocks and `conv_out_at(j)` are the
///     same bytes in an 8-row dispatch and a j-row dispatch;
///   * both paths then fold those same bytes from the same S_0 with the same
///     code.
///
/// So every difference is a bug and none is precision, which makes `memcmp`
/// the right comparison. A cosine here would pass on a fold that read the
/// wrong accept length -- check 3's control is the evidence for that: the state
/// at j against the state at j+1 still scores 0.959.
///
/// This is the same discipline the overlapped replay is gated on
/// (memory: gate-an-overlapped-replay-on-bit-identity), and for the same
/// reason: when both sides run identical arithmetic, exact equality is the only
/// bar with signal in it.
///
/// ---------------------------------------------------------------------------
/// What is NOT covered here
/// ---------------------------------------------------------------------------
/// The attention half. `npu_attn_layer::commit_prefix()` is `seen_ = cursor +
/// j`, the same one-line claim its `restore_state()` already makes, and what
/// exercises it is `verify_cpu::check_invariant()` -- called from
/// `prefill_verify()` under `DEBUG_BLOCK(1)`, so a build at `DEBUG_LEVEL >= 1`
/// asserts every cycle that the base KV cache holds exactly one row per
/// committed token and that the head's cache stays behind it -- and, via
/// `attention_cache_len()`, that all 16 layers agree on the count. Nor does this
/// cover the engine-level bookkeeping that replaces `_prefill_with_mm`'s; that
/// is the `FLM_Q38_FAST_ROLLBACK` A/B in REPLACEMENTWORK.md section 7.
///
/// Timing printed at the end is only meaningful from a binary built with the
/// engine's own flags (memory: host-vs-device-ab-needs-shipping-flags); at -O2
/// the host arm reads far slower than it is.
#include "verify/npu/delta_verify_layer.hpp"

#include "common/cpu_ref_delta_net_layer.hpp"
#include "common/cpu_ref_common.hpp"
#include "npu_utils/npu_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kLayer = 4;   ///< first delta layer whose input is a dumped tensor
constexpr int kL     = 8;   ///< the bitstream's -DL_TOKENS

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-52s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
    if (ok) n_pass++; else n_fail++;
}

/// The whole of a layer's carried history, as bytes. Everything
/// `commit_prefix()` is allowed to touch and nothing else.
struct carried {
    std::vector<float> state;
    std::vector<float> conv;
    int                seen = 0;
};

carried snapshot_of(q38npu::npu_delta_layer& l) {
    cpu_ref::delta_net_layer& c = l.inner();   // joins the pending fold
    return carried{c.recurrent_state(), c.conv_window(), c.tokens_seen()};
}

/// Words that differ, not a norm. `!=` and not an epsilon: the two paths run
/// the same code over the same bytes, so one differing word is a bug however
/// small the difference is.
size_t words_differ(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return (size_t)-1;
    size_t n = 0;
    for (size_t i = 0; i < a.size(); i++) if (a[i] != b[i]) n++;
    return n;
}

/// Compare two carried histories and report. `want_equal` false makes it a
/// control -- a check that the comparison can fail at all.
bool same(const carried& a, const carried& b, size_t& s_diff, size_t& c_diff) {
    s_diff = words_differ(a.state, b.state);
    c_diff = words_differ(a.conv,  b.conv);
    return s_diff == 0 && c_diff == 0 && a.seen == b.seen;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

/// Largest relative gap between two states, for the controls -- a control that
/// merely "differs" could differ in one ulp, which would say nothing.
double max_rel_gap(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        const double m = std::max(std::fabs((double)a[i]), std::fabs((double)b[i]));
        if (m > 1e-12) worst = std::max(worst, d / m);
    }
    return worst;
}

} // namespace

int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string xclbin =
        argc > 2 ? argv[2]
                 : "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/"
                   "gate_delta_net_verify/build/QWEN3_8_27B/xclbins/"
                   "deltanet_layer.xclbin";

    cpu_ref::delta_net_geometry geo;          // defaults are Qwen3.8-27B
    geo.validate();
    const int D = geo.model_dim;

    std::printf("== N13: rejected-draft recovery without a forward pass ==\n");
    std::printf("   model  %s\n   xclbin %s\n   layer  %d\n\n",
                model_dir.c_str(), xclbin.c_str(), kLayer);

    const q38npu::delta_verify_layout t = q38npu::delta_verify_layout::build(geo, kL);

    cpu_ref::safetensors q4nx(model_dir + "/model.q4nx");
    cpu_ref::safetensors ref(model_dir + "/ref.safetensors");

    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager*   layer_mgr = mgr.register_xclbin(xclbin);

    q38npu::delta_engine     eng(layer_mgr->create_app(), t);
    q38npu::npu_delta_layer  dut(&eng, q4nx, geo, kLayer);
    cpu_ref::delta_net_layer host(q4nx, geo, kLayer);   // for the cost table only

    // Real activations, bf16-rounded: the dumped output of layer 3, which is
    // this layer's input (memory: random-kv-cache-cannot-test-attention).
    char key[64];
    std::snprintf(key, sizeof key, "prefill.layer_%02d_out", kLayer - 1);
    std::vector<float> x_all = ref.f32(key);
    if (x_all.size() < (size_t)2 * kL * D) {
        std::printf("  [FAIL] %s has %zu floats, need %d rows of %d\n",
                    key, x_all.size(), 2 * kL, D);
        return 1;
    }
    for (float& v : x_all) v = cpu_ref::round_bf16(v);
    const float* x0 = x_all.data();                       // warm-up window
    const float* x1 = x_all.data() + (size_t)kL * D;      // the batch under test

    std::vector<float> y((size_t)kL * D, 0.0f);

    // ======================================================================
    // 0. a non-trivial starting state
    // ======================================================================
    // A zeroed S makes the fold agree with almost anything: the first token's
    // `S_{t-1} @ k` term vanishes and every accept length shares a prefix that
    // never had to be right. Run a window first so the state under test is one
    // the recurrence actually built.
    dut.forward(x0, kL, nullptr, nullptr, y.data());
    const carried pre = snapshot_of(dut);
    dut.save_state(qwen38::state_slot::turn);
    {
        double n2 = 0.0;
        for (float v : pre.state) n2 += (double)v * v;
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "S after 8 warm-up tokens: |S| = %.4f over %zu words, "
                      "seen = %d", std::sqrt(n2), pre.state.size(), pre.seen);
        report("the state under test is not the zero state",
               n2 > 0.0 && pre.seen == kL, buf);
    }

    // ======================================================================
    // 1. commit_prefix(j) == forward(first j rows), bit for bit, every j
    // ======================================================================
    // The whole claim, and the reason it is checkable exactly is at the top of
    // this file. `truth` is what the engine used to spend a 64-layer weight
    // stream to reach.
    std::vector<carried> fast(kL + 1), truth(kL + 1);

    for (int j = 1; j <= kL; j++) {
        // -- the fast path: one 8-row batch, then keep j of it -------------
        dut.restore_state(qwen38::state_slot::turn);
        dut.save_state(qwen38::state_slot::speculative);      // the pre-batch snapshot
        dut.forward(x1, kL, nullptr, nullptr, y.data());
        dut.commit_prefix(qwen38::state_slot::speculative, j);
        fast[j] = snapshot_of(dut);

        // -- the truth: run only the j tokens that committed ---------------
        dut.restore_state(qwen38::state_slot::turn);
        dut.forward(x1, j, nullptr, nullptr, y.data());
        truth[j] = snapshot_of(dut);

        size_t sd = 0, cd = 0;
        const bool ok = same(fast[j], truth[j], sd, cd);
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "accept %d: S %zu/%zu words differ, conv %zu/%zu, "
                      "seen %d vs %d",
                      j, sd, fast[j].state.size(), cd, fast[j].conv.size(),
                      fast[j].seen, truth[j].seen);
        char tag[64];
        std::snprintf(tag, sizeof tag, "accept %d lands exactly where a %d-row pass does",
                      j, j);
        report(tag, ok, buf);
    }

    // ======================================================================
    // 2. the control: accept j must NOT equal accept j+1
    // ======================================================================
    // Check 1 is a chain of equalities, and a `commit_prefix` that ignored
    // `accept_len` entirely would fail it -- but a `forward(x, j)` that ignored
    // its own L would make both sides wrong together. This separates them, and
    // it is the accept-length control the conv indexing needs: `conv_out_at` is
    // a single multiply-add and getting it off by one is the way this breaks.
    for (int j = 1; j < kL; j++) {
        size_t sd = 0, cd = 0;
        (void)same(fast[j], fast[j + 1], sd, cd);
        const double gap = max_rel_gap(fast[j].state, fast[j + 1].state);
        const bool ok = sd > 0 && cd > 0 && gap > 1e-3;
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "accept %d vs %d: S %zu words differ (worst %.3f rel), "
                      "conv %zu", j, j + 1, sd, gap, cd);
        char tag[64];
        std::snprintf(tag, sizeof tag, "control: accept %d differs from accept %d",
                      j, j + 1);
        report(tag, ok, buf);
    }

    // ======================================================================
    // 3. accept 0 is exactly restore_state
    // ======================================================================
    // The degenerate end of the range, and it has to be the identity or a
    // total rejection would quietly advance the state by nothing-in-particular.
    {
        dut.restore_state(qwen38::state_slot::turn);
        dut.save_state(qwen38::state_slot::speculative);
        dut.forward(x1, kL, nullptr, nullptr, y.data());
        dut.commit_prefix(qwen38::state_slot::speculative, 0);
        const carried zero = snapshot_of(dut);

        size_t sd = 0, cd = 0;
        const bool ok = same(zero, pre, sd, cd);
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "S %zu words differ, conv %zu, seen %d vs %d "
                      "(the pre-batch state)", sd, cd, zero.seen, pre.seen);
        report("accept 0 is the pre-batch state, unchanged", ok, buf);
    }

    // ======================================================================
    // 4. the deferred fold does not race the next dispatch
    // ======================================================================
    // `commit_prefix` hands its fold to a worker and joins it a whole cycle
    // later, across the next draft. A fold reading an arena the next dispatch
    // had begun to overwrite would be wrong by a few ulp and no cosine would
    // see it, so the check is the same fold run inline, compared by word
    // (memory: gate-an-overlapped-replay-on-bit-identity).
    {
        eng.overlap = false;
        size_t worst_s = 0, worst_c = 0;
        int    worst_j = 0;
        for (int j = 1; j <= kL; j++) {
            dut.restore_state(qwen38::state_slot::turn);
            dut.save_state(qwen38::state_slot::speculative);
            dut.forward(x1, kL, nullptr, nullptr, y.data());
            dut.commit_prefix(qwen38::state_slot::speculative, j);
            const carried sync = snapshot_of(dut);
            size_t sd = 0, cd = 0;
            (void)same(sync, fast[j], sd, cd);
            if (sd > worst_s) { worst_s = sd; worst_j = j; }
            worst_c = std::max(worst_c, cd);
        }
        eng.overlap = true;
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "inline vs deferred over all 8 accept lengths: worst S "
                      "%zu/%zu words (accept %d), worst conv %zu",
                      worst_s, fast[1].state.size(), worst_j, worst_c);
        report("the deferred fold is bit-identical to the inline one",
               worst_s == 0 && worst_c == 0, buf);
    }

    // ======================================================================
    // 5. the guards
    // ======================================================================
    // `arena_stage_` holds the last tile only. An accept length past what the
    // last window ran would fold padding rows -- which are copies of the last
    // real row, so the result would be plausible and wrong.
    {
        dut.restore_state(qwen38::state_slot::turn);
        dut.save_state(qwen38::state_slot::speculative);
        dut.forward(x1, 3, nullptr, nullptr, y.data());     // a 3-row window

        std::string got;
        try { dut.commit_prefix(qwen38::state_slot::speculative, 5); }
        catch (const std::exception& e) { got = e.what(); }
        report("accept past the last window's length is refused",
               !got.empty(), got.empty() ? "accepted 5 after a 3-row window"
                                         : got);

        got.clear();
        try { dut.commit_prefix(qwen38::state_slot::speculative, kL + 1); }
        catch (const std::exception& e) { got = e.what(); }
        report("accept past L_TOKENS is refused",
               !got.empty(), got.empty() ? "accepted L_TOKENS+1" : got);
    }

    // ======================================================================
    // 6. what it costs
    // ======================================================================
    // Three arms, per layer, at accept 3 -- the length the measured session
    // actually committed on its one rejection.
    //
    //   ms_fast   commit_prefix + join: the memcpy and the fold, nothing else.
    //   ms_pass   one NPU dispatch of 3 rows: what re-running this layer costs
    //             even on the fastest path the engine has for it.
    //   ms_host   cpu_ref::delta_net_layer::forward of 3 rows: what re-running
    //             it cost before N-DELTA-VERIFY, and what the re-prefill path
    //             still pays per delta layer for its recurrence.
    //
    // The engine's replay is 48 of these plus 16 attention layers plus an
    // lm_head, so x48 is a floor on the saving, not an estimate of it.
    {
        const int j = 3;
        const int reps = 5;

        double ms_fast = 0.0;
        for (int r = 0; r < reps; r++) {
            dut.restore_state(qwen38::state_slot::turn);
            dut.save_state(qwen38::state_slot::speculative);
            dut.forward(x1, kL, nullptr, nullptr, y.data());
            const auto t0 = std::chrono::steady_clock::now();
            dut.commit_prefix(qwen38::state_slot::speculative, j);
            (void)dut.inner();                     // join, so the fold is counted
            ms_fast += ms_since(t0);
        }
        ms_fast /= reps;

        double ms_pass = 0.0;
        for (int r = 0; r < reps; r++) {
            dut.restore_state(qwen38::state_slot::turn);
            const auto t0 = std::chrono::steady_clock::now();
            dut.forward(x1, j, nullptr, nullptr, y.data());
            (void)dut.inner();
            ms_pass += ms_since(t0);
        }
        ms_pass /= reps;

        double ms_host = 0.0;
        for (int r = 0; r < reps; r++) {
            host.reset();
            const auto t0 = std::chrono::steady_clock::now();
            host.forward(x1, j, y.data());
            ms_host += ms_since(t0);
        }
        ms_host /= reps;

        std::printf("\n   per layer, accept %d of 8:\n", j);
        std::printf("     commit_prefix + join        %8.3f ms\n", ms_fast);
        std::printf("     one %d-row NPU dispatch      %8.3f ms   (%.1fx)\n",
                    j, ms_pass, ms_fast > 0 ? ms_pass / ms_fast : 0.0);
        std::printf("     one %d-row host block        %8.3f ms   (%.1fx)\n",
                    j, ms_host, ms_fast > 0 ? ms_host / ms_fast : 0.0);
        std::printf("     x48 delta layers            %8.1f ms  vs %8.1f ms "
                    "re-dispatched\n", ms_fast * 48, ms_pass * 48);
    }

    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
