// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify_n5_mtp_prefill.cpp
/// \brief N5 -- the MTP draft head's step 0 as a prefill: one `attn.xclbin`
///        dispatch for the whole window, on the head's own cache BO, with
///        steps 1-6 still on `MTP_layer.xclbin` behind the same cursor.
///
/// Build:
///   source ~/setup.sh
///   g++-13 -O2 -march=native -fopenmp -std=c++20
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n5_mtp_prefill.cpp prefill/npu/prefill_attn_sequence.cpp
///       speculate/npu/mtp_sequence.cpp speculate/mtp_head.cpp
///       ../q4nx/q4_npu_eXpress.cpp ../q4nx/safe_tensors.cpp
///       -L../../../../src/lib/xrt -L../../build/lib -L/opt/xilinx/xrt/lib
///       -lxrt_coreutil -laiebu -luuid -lpthread -o /tmp/verify_n5_mtp_prefill
///   (one line; the continuations are dropped here because a backslash at the
///    end of a // comment splices the next line into it)
///
/// Takes both xclbins as FILES. Runs ~5 minutes: three MTP layers are staged
/// (~250 MB each) and the CPU reference projects a 221-row window twice.
///
/// ---------------------------------------------------------------------------
/// What is under test, and what is not
/// ---------------------------------------------------------------------------
/// A draft is seven steps of TWO shapes. Step 0 is a window of T tokens -- the
/// catch-up that brings the head's private cache level with the base stack --
/// and steps 1-6 are one token each. N3 covers the decode bitstream; this suite
/// covers the step-0 shape and, more importantly, the SEAM between them: one
/// cache BO, one cursor, two bitstreams, written by `append_rows()` on the host
/// and by `forward()` on the device.
///
/// Only the attention core moves to `attn.xclbin`. Fusion, both projections'
/// norms, q/k-norm, RoPE, the output gate, o_proj and the MLP run on the CPU in
/// `cpu_ref::mtp_head`'s OWN helpers -- not a copy of them. That is what makes
/// the comparison below meaningful: `speculate_cpu` is the reference
/// `speculate_npu` is scored against, so the two share the code rather than
/// mirror it, and the only thing that can differ is the core.
///
/// The suite is in two halves and they fail for different reasons:
///
///   checks 1-7   the pieces, driven directly: the geometry guard, the apron,
///                step 0 against `cpu_ref::mtp_head::step()` at five widths,
///                the stale-row mask, the cross-head control, the per-segment
///                profile, and a ragged cursor. Addressing bugs live here.
///   checks 8-9   the engine's own composition: `speculate_npu::draft()`
///                against `speculate_cpu::draft()` across all seven steps, and
///                against a second `speculate_npu` built WITHOUT the prefill
///                bitstream, which falls back to the per-row decode path.
///
/// The second half is the gate. Everything in the first can pass while
/// `_prefill_step` hands the op the post-append cursor, or drops the last row,
/// or leaves the cursor somewhere steps 1-6 cannot resume from -- and a drafted
/// id sequence that matches the reference's for all seven steps can survive
/// none of those.
///
/// ---------------------------------------------------------------------------
/// Why the k/v rows are scored separately from the output
/// ---------------------------------------------------------------------------
/// Each query's own key dominates its softmax, so an error in the cache is
/// heavily diluted by the time it reaches `norm_out`. A plane-pitch error --
/// every v row read 256 rows low, which is exactly what forgetting
/// `set_layout()` after a grow produces -- moves the output by less than the
/// bf16 accumulation floor does. So the rows that landed in the BO are read
/// back and scored against the CPU head's own cache, every time.
/// (mirrors verify_n4_prefill_attn's Part B, and for the same reason)
#include "speculate/speculate_npu.hpp"
#include "speculate/speculate_cpu.hpp"
#include "speculate/mtp_head.hpp"
#include "speculate/npu/mtp_weights.hpp"
#include "prefill/npu/prefill_attn_op.hpp"
#include "verify/npu/attn_weights.hpp"
#include "common/cpu_ref_common.hpp"
#include "common/embedding.hpp"
#include "common/lm_head.hpp"
#include "common/rope.hpp"

#include "npu_utils/npu_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

/// Small, so the deep-pass check has a reachable ceiling. The head's live
/// MAX_L is the engine's, thousands of rows higher; what matters here is that
/// the apron is exercised, and the apron is a constant.
constexpr int kMaxL = 1024;

/// The head's rows start at position 1, not 0 -- step 0 feeds [input_ids[1:],
/// x_P] at positions 1..P. Both speculators count off pos0 the same way, so
/// this is only the window's origin, but it is the origin the live path uses.
/// (README section 5.3: do not "fix" the off-by-one)
constexpr int kPos0 = 1;

/// When a drafted id is evidence, in units of the logit distribution's own
/// standard deviation (see `draft_margins`).
///
/// The number comes from the perturbation the attention core actually applies.
/// Step 0's norm_out scores cos ~= 0.99985 against the reference, i.e. a
/// relative L2 error of sqrt(2 * 1.5e-4) ~= 1.7e-2. A logit is w_i . h, so a
/// relative wobble eps in h moves the top-1/top-2 gap by roughly sqrt(2) * eps
/// standard deviations -- about 0.024 here. 0.10 leaves a factor of four, which
/// is enough to separate "the two paths disagree" from "the reference had no
/// opinion" without excusing a gap a real bug would open.
constexpr double kDecisive = 0.10;

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-44s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
    if (ok) n_pass++; else n_fail++;
}

double cosine(const float* a, const float* b, size_t n) {
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < n; i++) {
        num += (double)a[i] * b[i];
        da  += (double)a[i] * a[i];
        db  += (double)b[i] * b[i];
    }
    if (da == 0 || db == 0) return 0.0;
    return num / (std::sqrt(da) * std::sqrt(db));
}

double report_cos(const char* tag, double cos, double gate, const char* note = "") {
    char buf[200];
    std::snprintf(buf, sizeof buf, "cos = %.8f  (gate %.4f) %s", cos, gate, note);
    report(tag, cos >= gate, buf);
    return cos;
}

/// Mean exp(entropy) of the softmax rows this window actually produces.
///
/// 1.0 means the distribution collapsed onto a single key, and then the
/// comparison is vacuous -- any addressing bug in the replay would score
/// perfectly. The head's own projections of a real prompt should be nowhere
/// near that, but "should be" is not a measurement.
/// (memory: random-kv-cache-cannot-test-attention)
double eff_keys(const float* qh, const float* k, int L, int base) {
    using namespace cpu_ref::q38;
    double sum = 0.0;
    long   n   = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+:sum,n)
    for (int t = 0; t < L; t++) {
        for (int h = 0; h < NUM_ATTN_HEADS; h++) {
            const int kvh = h / GQA_GROUP, nkey = base + t + 1;
            const float* qv = qh + (size_t)t * DQ + h * HEAD_DIM;
            std::vector<float> s((size_t)nkey);
            float smax = -INFINITY;
            for (int j = 0; j < nkey; j++) {
                const float* kv = k + (size_t)j * DKV + kvh * HEAD_DIM;
                float acc = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) acc += qv[d] * kv[d];
                s[(size_t)j] = acc * ATTN_SCALE;
                smax = std::max(smax, s[(size_t)j]);
            }
            float tot = 0.0f;
            for (int j = 0; j < nkey; j++) {
                s[(size_t)j] = std::exp(s[(size_t)j] - smax);
                tot += s[(size_t)j];
            }
            double ent = 0.0;
            for (int j = 0; j < nkey; j++) {
                const double p = s[(size_t)j] / tot;
                if (p > 0.0) ent -= p * std::log(p);
            }
            sum += std::exp(ent);
            n++;
        }
    }
    return n ? sum / (double)n : 0.0;
}

/// The kv rows a pass at (L_begin, L_pad) streams, from the sequence
/// generator's own arithmetic rather than from an assumption about it.
int kv_reach(int L_begin, int L_pad) {
    const int C = q38npu::prefill_attn_tiles::Q_CHUNK;
    const int Lq_last = L_begin + L_pad - C;
    return (Lq_last + 2 * C - 1) / C * C;
}

std::string id_list(const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++)
        s += (i ? " " : "") + std::to_string(v[i]);
    return s;
}

/// How decisive each of the reference's own argmaxes was.
///
/// Replays speculate_cpu's draft loop on the CPU and returns, per step, the
/// top-1 minus top-2 logit divided by the logit distribution's standard
/// deviation. Scale-free on purpose: the lm_head's rows have no fixed norm, so
/// a raw gap in logit units says nothing across steps.
///
/// This exists because comparing drafted ids across two arithmetics is only a
/// test at steps where the argmax is actually decided. Drafting is a 7-step
/// autoregression -- each step's output id and hidden state are the next
/// step's input -- so the attention core's error floor is fed back through the
/// head six more times. At a step whose top two logits are a hair apart, which
/// id falls out is a coin toss that reports the last bit of the accumulator,
/// not the correctness of either path.
/// (memory: qwen38-lm-head-untied-and-argmax-ties)
std::vector<double> draft_margins(cpu_ref::mtp_head& m, cpu_ref::lm_head& head,
                                  int V, const std::vector<int>& ids,
                                  const float* prev_hidden, int T, int pos0,
                                  int k, std::vector<int>& out_ids) {
    const int            D = cpu_ref::q38::MODEL_DIM;
    std::vector<double>  margins;
    std::vector<float>   logits((size_t)V);
    std::vector<int32_t> step_ids(ids.begin(), ids.end());
    std::vector<float>   hidden(prev_hidden, prev_hidden + (size_t)T * D);
    out_ids.clear();
    m.reset();

    int pos = pos0, width = T;
    for (int s = 0; s < k; s++) {
        const std::vector<int32_t> pids = cpu_ref::text_position_ids(pos, width);
        cpu_ref::mtp_taps          tp;
        m.step(step_ids.data(), pids.data(), hidden.data(), width, tp);
        const float* last = tp.norm_out.data() + (size_t)(width - 1) * D;
        head.forward(last, 1, logits.data());

        int top = 0;
        for (int i = 1; i < V; i++)
            if (logits[(size_t)i] > logits[(size_t)top]) top = i;

        float  second = -INFINITY;
        double sum = 0.0, sum2 = 0.0;
        for (int i = 0; i < V; i++) {
            const double x = logits[(size_t)i];
            if (i != top && logits[(size_t)i] > second) second = logits[(size_t)i];
            sum += x;
            sum2 += x * x;
        }
        const double mean = sum / V;
        const double sd   = std::sqrt(std::max(0.0, sum2 / V - mean * mean));
        margins.push_back(sd > 0.0 ? (logits[(size_t)top] - second) / sd : 0.0);

        out_ids.push_back(top);
        hidden.assign(last, last + D);
        step_ids.assign(1, (int32_t)top);
        pos += width;
        width = 1;
    }
    return margins;
}

} // namespace

// ==========================================================================
int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string mtp_xclbin =
        argc > 2 ? argv[2]
                 : "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/"
                   "MTP_decoding/build/QWEN3_8_27B/xclbins/MTP_layer.xclbin";
    const std::string prefill_xclbin =
        argc > 3 ? argv[3]
                 : "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/"
                   "gated_attention_prefill/build/xclbins/attn.xclbin";
    // Not under test here, but `speculate_npu` requires both: step 0's five
    // projections are a device op and so is the draft head's projection, so
    // there is no configuration of it that omits them.
    const std::string proj_xclbin =
        argc > 4 ? argv[4]
                 : "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/"
                   "dequant_mm/build/xclbins/dequant_mm.xclbin";
    const std::string lm_xclbin =
        argc > 5 ? argv[5]
                 : "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/"
                   "lm_head_npu_bin/build/QWEN3_8_27B/xclbins/lm_head.xclbin";

    using namespace cpu_ref::q38;
    const int D = MODEL_DIM, DKV_ = DKV, DQ_ = DQ;
    const int C = q38npu::mtp_prefill_attn_op::CHUNK;

    std::printf("== N5: the MTP draft head's step 0 as a prefill ==\n");
    std::printf("   model    %s\n   decode   %s\n   prefill  %s\n"
                "   proj     %s\n   lm_head  %s\n\n",
                model_dir.c_str(), mtp_xclbin.c_str(), prefill_xclbin.c_str(),
                proj_xclbin.c_str(), lm_xclbin.c_str());

    // ======================================================================
    // 1. the geometry guard
    // ======================================================================
    // One bitstream serves two designs' caches, so `gen_prefill_attn_seq` is a
    // template over the layout. That is only sound while the two layouts agree
    // on every field it reads. They are maintained independently -- different
    // headers, different owners -- so a future divergence would arrive as a
    // mask or a plane-hop bug on hardware, with nothing pointing at the cause.
    //
    // Neither layout has an apron any more: `MAX_L` IS the plane pitch and half
    // the BO in both, as it is in qwen3_5vl_npu. So the check is stronger than
    // it was -- the two are now required to agree on the addressing outright,
    // with nothing forced equal first.
    const q38npu::mtp_layout t = q38npu::mtp_layout::build(kMaxL);
    {
        q38npu::attn_layout a = q38npu::attn_layout::build(kMaxL, /*L=*/8);
        std::string bad;
        auto want = [&](const char* name, long long m, long long v) {
            if (m != v) bad += std::string(bad.empty() ? "" : ", ") + name + " " +
                               std::to_string(m) + " vs " + std::to_string(v);
        };
        want("DH", t.DH, a.DH);
        want("DQ", t.DQ, a.DQ);
        want("DK", t.DK, a.DK);
        want("DV", t.DV, a.DV);
        want("num_kv_heads", t.num_kv_heads, a.num_kv_heads);
        want("MAX_L", t.MAX_L, a.MAX_L);

        // The addressing. Nothing is adjusted first -- if these ever differ at
        // the same MAX_L, one of the two grew a term the generator cannot see.
        want("plane_rows", (long long)t.plane_rows(), (long long)a.plane_rows());
        want("k_at(1)", (long long)t.k_at(1), (long long)a.k_at(1));
        want("v_at(0)", (long long)t.v_at(0), (long long)a.v_at(0));
        want("v_at(1)", (long long)t.v_at(1), (long long)a.v_at(1));
        // And that the pitch really is just MAX_L, in both.
        want("v_at(0) == MAX_L * DK", (long long)t.v_at(0),
             (long long)t.MAX_L * t.DK);
        want("kv_cache_size == 2 * MAX_L * DK", (long long)t.kv_cache_size,
             2LL * t.MAX_L * t.DK);

        char buf[220];
        std::snprintf(buf, sizeof buf,
                      "mtp_layout and attn_layout agree on all 10 fields "
                      "gen_prefill_attn_seq reads, and both are two bare planes "
                      "of MAX_L = %d", t.MAX_L);
        report("the two layouts are interchangeable", bad.empty(),
               bad.empty() ? buf : ("they differ: " + bad));
    }

    // ======================================================================
    // 2. the top of the buffer, by arithmetic
    // ======================================================================
    // The kv broadcast walks whole 128-row blocks and overshoots the query
    // chunk by one, so a pass reads through `round_up(L_begin + L_pad, 128)`.
    // There is no apron under that: what keeps it inside the BO is that
    // `prefill_attn_op::run` guards the PADDED window against MAX_L, and that
    // MAX_L is a whole PREFILL_CHUNK so the round-up is a fixed point at the
    // top. Short is silent here -- a DDR read past a BO does not fault, it
    // returns whatever the allocator put there. The pass below is run for real
    // at the end of check 6, once there is a cache worth reading.
    {
        int worst = 0, worst_begin = 0;
        for (int L_begin = 0; L_begin < t.MAX_L; L_begin++)
            for (int L : {1, 7, 128, 129, 256}) {
                const int L_pad = (L + C - 1) / C * C;
                if (L_begin + L_pad > t.MAX_L) continue;   // run() refuses it
                const int r = kv_reach(L_begin, L_pad);
                if (r > worst) { worst = r; worst_begin = L_begin; }
            }
        char buf[220];
        std::snprintf(buf, sizeof buf,
                      "deepest reach %d rows (at L_begin %d), plane holds %zu = "
                      "MAX_L %d, no apron",
                      worst, worst_begin, t.plane_rows(), t.MAX_L);
        report("every window the guard admits stays inside the plane",
               t.MAX_L % C == 0 && (size_t)worst <= t.plane_rows(), buf);
    }

    // ======================================================================
    // inputs
    // ======================================================================
    cpu_ref::safetensors ref(model_dir + "/ref.safetensors");
    std::vector<int32_t> ids32;
    std::vector<float>   hid;
    try {
        ids32 = ref.i32("mtp.step0.input_ids");
        hid   = ref.f32("mtp.step0.prev_hidden");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ref.safetensors: %s\n", e.what());
        return 1;
    }
    const int P = (int)ids32.size();
    if ((int)(hid.size() / D) < P) {
        std::fprintf(stderr, "prev_hidden is %zu rows, ids are %d\n",
                     hid.size() / D, P);
        return 1;
    }
    hid.resize((size_t)P * D);
    // Round to what the array will actually receive. Letting the reference
    // start from extra precision charges the NPU for a narrowing it cannot
    // avoid, and moves the gate for no reason.
    for (float& f : hid) f = cpu_ref::round_bf16(f);

    // The dump's position_ids are [3][P] and channel-major, and this window's
    // tail is a vision seam where they diverge -- but neither speculator can
    // express that: speculate_cpu calls text_position_ids and speculate_npu
    // counts pos + r. Both are scored here, so both must be driven the way the
    // engine drives them, which is text positions from kPos0.
    // (memory: position-ids-dump-is-channel-major -- it is why this is stated)
    std::printf("   window %d rows, text positions %d..%d\n\n",
                P, kPos0, kPos0 + P - 1);

    cpu_ref::safetensors     q4nx(model_dir + "/model.q4nx");
    cpu_ref::embedding_table embed(q4nx);
    cpu_ref::lm_head         head(q4nx);

    // The speculators read exactly two fields off the descriptor, so building
    // one by hand here avoids dragging LM_Config and the whole engine config
    // path into a hardware driver. If either class grows a third, this breaks
    // loudly at the member rather than quietly at run time.
    qwen3_8mtp_desc desc;
    desc.D                     = (uint32_t)D;
    desc.vocab_size            = (uint32_t)head.vocab();
    desc.mtp_num_hidden_layers = 1;
    desc.mtp_fc_in             = (uint32_t)cpu_ref::mtp::FC_IN;

    // ======================================================================
    // the two speculators, and the CPU halves
    // ======================================================================
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);

    std::printf("-- staging (three head layers, ~250 MB each, + 1.35 GB) --\n");
    q38npu::npu_lm_head lmh(mgr.register_xclbin(lm_xclbin)->create_app(), q4nx);

    qwen38::speculate_cpu spec_cpu(q4nx, &desc, &embed, &head, kMaxL);
    qwen38::speculate_npu spec_npu(q4nx, &desc, &embed, kMaxL, &mgr,
                                   mtp_xclbin, prefill_xclbin, proj_xclbin,
                                   &lmh);
    // The same backend with the prefill bitstream withheld: `op_` stays null
    // and every step takes the per-row decode path. This is the A/B the
    // FLM_Q38_MTP_PREFILL kill switch selects, built as an object rather than
    // as an environment variable so both are live at once. An empty path is
    // how that arm is spelled now -- both arms are hardware, which is why
    // this switch survived the removal of the host ones.
    qwen38::speculate_npu spec_fb(q4nx, &desc, &embed, kMaxL, &mgr,
                                  mtp_xclbin, /*prefill=*/"", proj_xclbin,
                                  &lmh);

    if (!spec_npu.prefill_op()) {
        std::fprintf(stderr, "speculate_npu did not open %s -- nothing to test\n",
                     prefill_xclbin.c_str());
        return 1;
    }
    report("the prefill path opened", spec_fb.prefill_op() == nullptr,
           "op_ is live with attn.xclbin and null without it");

    q38npu::npu_mtp_layer&       layer = spec_npu.layer();
    q38npu::mtp_prefill_attn_op& op    = *spec_npu.prefill_op();

    // The CPU halves, standing in for the ones speculate_npu holds privately.
    // max_len 0 for the same reason it is 0 there: only fuse /
    // project_qkv_rope / gate_out_mlp are ever called on this object, and
    // attn_core() is the one helper that reads a cache -- it is the one the
    // device replaces. step() on it would throw, which is the intent.
    cpu_ref::mtp_head half(q4nx, &embed, /*max_len=*/0);
    cpu_ref::mtp_head cpu(q4nx, &embed, kMaxL);          // the scored reference
    const std::vector<float>& out_norm = half.out_norm();

    // ---- project the whole window once -------------------------------------
    // fuse() and project_qkv_rope() are per row, so the prefix for any T is
    // literally the first T rows of these -- which is what lets checks 3-6
    // share one projection pass. The reference below recomputes from scratch
    // every time regardless; that is the comparison.
    std::vector<float> cos_all, sin_all;
    cpu_ref::text_rope_tables(kPos0, P, cos_all, sin_all);
    cpu_ref::mtp_taps tp_all;
    half.fuse(ids32.data(), hid.data(), P, tp_all);
    std::vector<float> qg_all, kk_all, vv_all, qh_all;
    half.project_qkv_rope(tp_all.fc_out.data(), P, cos_all.data(), sin_all.data(),
                          qg_all, kk_all, vv_all, qh_all);

    /// The engine's `_prefill_step`, on the pieces: append at the cursor, one
    /// dispatch, then the CPU tail. Returns `norm_out` for the window.
    auto hybrid = [&](int begin, int T, std::vector<float>& norm_out) {
        const int base = layer.cache_len();       // BEFORE the append -- this is
                                                  // the window's L_begin
        layer.append_rows(kk_all.data() + (size_t)begin * DKV_,
                          vv_all.data() + (size_t)begin * DKV_, T);

        std::vector<float> attn((size_t)T * DQ_, 0.0f);
        op.run(qh_all.data() + (size_t)begin * DQ_, T, base, layer.cache(),
               attn.data());

        // gate_out_mlp consumes `attn` -- it applies the output gate IN PLACE
        // -- so the core's own answer has to be taken before the tail runs.
        // Checks 4-6 score that copy; scoring the gated one would be scoring
        // sigmoid(z) * o as if it were the attention.
        const std::vector<float> core = attn;

        std::vector<float> layer_out((size_t)T * D, 0.0f);
        half.gate_out_mlp(qg_all.data() + (size_t)begin * 2 * DQ_, attn.data(),
                          tp_all.fc_out.data() + (size_t)begin * D, T,
                          layer_out.data());

        norm_out.assign((size_t)T * D, 0.0f);
        for (int r = 0; r < T; r++)
            cpu_ref::rms_norm(layer_out.data() + (size_t)r * D, out_norm.data(),
                              D, norm_out.data() + (size_t)r * D);
        return core;
    };

    // ======================================================================
    // 3. step 0 against cpu_ref::mtp_head::step(), five widths
    // ======================================================================
    // 1 and 2 straddle the width at which `_use_prefill` switches paths, 8 is a
    // decode-sized window, 128 is exactly one query chunk, and 221 is what the
    // live path actually hands step 0 -- the last prefill chunk, since
    // automodel.cpp calls speculate() before any forward().
    std::printf("\n-- step 0 vs cpu_ref::mtp_head::step() --\n");
    std::vector<float> attn_221;
    for (int T : {1, 2, 8, 128, P}) {
        cpu.reset();
        cpu_ref::mtp_taps tp;
        const std::vector<int32_t> pids = cpu_ref::text_position_ids(kPos0, T);
        cpu.step(ids32.data(), pids.data(), hid.data(), T, tp);

        layer.reset();
        std::vector<float> norm_npu;
        std::vector<float> attn = hybrid(0, T, norm_npu);
        if (T == P) attn_221 = attn;

        char tag[64], note[96];
        std::snprintf(tag, sizeof tag, "T = %3d  norm_out", T);
        std::snprintf(note, sizeof note, "-- ONE dispatch (decode path: %d)", T);
        report_cos(tag, cosine(norm_npu.data(), tp.norm_out.data(), norm_npu.size()),
                   0.999, note);

        // The rows that landed in the BO. Scored separately because norm_out
        // cannot see a plane-pitch error clearly -- each query's own row
        // dominates its softmax, so the error is diluted before it reaches the
        // output. This is the check that catches v_at().
        double wk = 2.0, wv = 2.0;
        for (int r = 0; r < T; r++) {
            wk = std::min(wk, cosine(layer.read_k(r).data(),
                                     cpu.cache_k().data() + (size_t)r * DKV_,
                                     (size_t)DKV_));
            wv = std::min(wv, cosine(layer.read_v(r).data(),
                                     cpu.cache_v().data() + (size_t)r * DKV_,
                                     (size_t)DKV_));
        }
        std::snprintf(tag, sizeof tag, "T = %3d  k|v in the BO", T);
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "worst row: k %.8f, v %.8f (gate 0.9990); cursor %d",
                      wk, wv, layer.cache_len());
        report(tag, wk >= 0.999 && wv >= 0.999 && layer.cache_len() == T, buf);
    }

    // ======================================================================
    // 4. rows above the window are inert
    // ======================================================================
    // A window of T rounds up to a whole 128-row query chunk and the kv
    // broadcast walks whole 128-row blocks, so a dispatch at T = 8 STREAMS 128
    // cache rows and attends over 8. The other 120 are excluded by the causal
    // mask -- if the mask is right.
    //
    // Nothing above tests that. Every window so far ran on a cache whose rows
    // above the cursor were zeros, which a leak cannot be seen through: zero
    // keys land at the same logit for every query and a leaked zero row
    // contributes a zero value.
    //
    // And this is not hypothetical. `crop()` rolls the cursor back after a
    // rejected draft and leaves the rows where they are -- deliberately, it is
    // what makes the draft loop snapshot-free -- so the very next step 0 runs
    // over a plane still holding the rejected draft's real keys.
    std::printf("\n-- rows above the cursor cannot reach the window --\n");
    {
        const int T8 = 8;

        layer.cache().memset((bf16)0);       // clean: zeros above row 8
        layer.cache().sync_to_device();
        layer.reset();
        std::vector<float> n_clean;
        const std::vector<float> a_clean = hybrid(0, T8, n_clean);

        layer.reset();                       // dirty: 221 real rows, then a
        layer.mirror_from_cpu(kk_all, vv_all, P);   // cursor that says they are
        layer.reset();                              // gone -- crop()'s state
        std::vector<float> n_dirty;
        const std::vector<float> a_dirty = hybrid(0, T8, n_dirty);

        report_cos("a stale row above the window is masked",
                   cosine(a_clean.data(), a_dirty.data(), a_clean.size()),
                   1.0 - 1e-9, "-- must be 1.0 exactly");
    }

    // ======================================================================
    // 5. the cross-head control
    // ======================================================================
    // Matching our own reference does not prove the head <-> CU <-> kv-stream
    // mapping: a consistent permutation agrees with itself. So the reference is
    // recomputed with the kv-head assignment rotated by one, and the device is
    // required to agree with THAT much less well.
    // (memory: cross-head-control-catches-gqa-mapping)
    std::printf("\n-- the cross-head control, at T = %d --\n", P);
    std::vector<float> attn_ref;
    {
        const double eff = eff_keys(qh_all.data(), kk_all.data(), P, 0);
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "mean exp(entropy) = %.2f keys  (1.0 == degenerate, and "
                      "then nothing below means anything)", eff);
        report("the softmax is not collapsed", eff >= 4.0, buf);

        // The real reference, through cpu_ref's own attn_core over a seeded
        // cache -- the same six lines speculate_cpu runs, not a copy.
        cpu_ref::mtp_head aux(q4nx, &embed, kMaxL);
        aux.seed_cache(kk_all, vv_all, P);
        aux.attn_core(qh_all.data(), P, 0, attn_ref);
        const double real = cosine(attn_221.data(), attn_ref.data(),
                                   attn_221.size());

        std::vector<float> k_rot(kk_all.size()), v_rot(vv_all.size());
        for (int r = 0; r < P; r++)
            for (int h = 0; h < NUM_KV_HEADS; h++) {
                const int src = (h + 1) % NUM_KV_HEADS;
                const size_t dst_o = (size_t)r * DKV_ + (size_t)h * HEAD_DIM;
                const size_t src_o = (size_t)r * DKV_ + (size_t)src * HEAD_DIM;
                std::memcpy(k_rot.data() + dst_o, kk_all.data() + src_o,
                            HEAD_DIM * sizeof(float));
                std::memcpy(v_rot.data() + dst_o, vv_all.data() + src_o,
                            HEAD_DIM * sizeof(float));
            }
        cpu_ref::mtp_head ctl(q4nx, &embed, kMaxL);
        ctl.seed_cache(k_rot, v_rot, P);
        std::vector<float> attn_ctl;
        ctl.attn_core(qh_all.data(), P, 0, attn_ctl);
        const double rot = cosine(attn_221.data(), attn_ctl.data(),
                                  attn_221.size());

        std::snprintf(buf, sizeof buf,
                      "real %.6f vs rotated-kv %.6f -- the control must be MUCH "
                      "worse", real, rot);
        report("cross-head control separates", real > rot + 0.05, buf);

        // Per head, on the same pair. A uniform sag is bf16 accumulation; ONE
        // head adrift is the mapping, and only this printout tells them apart.
        double worst = 1.0; int worst_h = -1;
        std::printf("        per-head cosine:");
        for (int h = 0; h < NUM_ATTN_HEADS; h++) {
            double dot = 0, na = 0, nb = 0;
            for (int r = 0; r < P; r++) {
                const size_t b = (size_t)r * DQ_ + (size_t)h * HEAD_DIM;
                for (int i = 0; i < HEAD_DIM; i++) {
                    const double x = attn_221[b + i], y = attn_ref[b + i];
                    dot += x * y; na += x * x; nb += y * y;
                }
            }
            const double c = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
            if (h % 8 == 0) std::printf("\n         ");
            std::printf(" h%02d=%.4f", h, c);
            if (c < worst) { worst = c; worst_h = h; }
        }
        std::printf("\n");
        std::snprintf(buf, sizeof buf,
                      "worst h%d at %.5f, aggregate %.5f", worst_h, worst, real);
        report("no head is an outlier", worst >= real - 0.05, buf);
    }

    // ======================================================================
    // 6. per-segment cosine by absolute token
    // ======================================================================
    // The online softmax accumulates in bf16, so accuracy falls smoothly with
    // causal depth -- a deep window scores worse than a shallow one for reasons
    // that are not bugs. Printing it by segment is what separates "the tail is
    // legitimately noisier" from "one block of rows is addressed wrong".
    // (memory: prefill-attention-error-grows-with-kv-blocks)
    {
        double worst = 1.0; int worst_s = -1;
        std::printf("        per-segment cosine (absolute token):");
        for (int s0 = 0; s0 < P; s0 += C) {
            const int s1 = std::min(s0 + C, P);
            const double c = cosine(attn_221.data() + (size_t)s0 * DQ_,
                                    attn_ref.data() + (size_t)s0 * DQ_,
                                    (size_t)(s1 - s0) * DQ_);
            std::printf("  [%d,%d)=%.5f", s0, s1, c);
            if (c < worst) { worst = c; worst_s = s0; }
        }
        std::printf("\n");
        char buf[160];
        std::snprintf(buf, sizeof buf, "worst segment starts at %d, %.5f",
                      worst_s, worst);
        report("every segment clears the gate", worst >= 0.99, buf);
    }

    // ======================================================================
    // 7. a ragged cursor, and the deepest legal window
    // ======================================================================
    // `attn.cc` masks on ABSOLUTE position, so a cursor that is not a multiple
    // of 128 is legal -- and it is what a second speculation cycle hands step 0
    // every time. A pass that works at 0 and fails at 22 is addressing, not
    // math, and nothing above would have caught it: every window so far started
    // at the cursor 0 the sequence is generated for by default.
    std::printf("\n-- a ragged cursor --\n");
    {
        const int pre = 22, win = 64;
        layer.reset();
        layer.append_rows(kk_all.data(), vv_all.data(), pre);   // cursor 22
        std::vector<float> norm_npu;
        const std::vector<float> core_npu = hybrid(pre, win, norm_npu);

        cpu_ref::mtp_head aux(q4nx, &embed, kMaxL);
        aux.seed_cache(kk_all, vv_all, pre + win);
        std::vector<float> attn_r;
        aux.attn_core(qh_all.data() + (size_t)pre * DQ_, win, pre, attn_r);

        char note[120];
        std::snprintf(note, sizeof note,
                      "-- L_begin %d, %d rows, cursor now %d", pre, win,
                      layer.cache_len());
        // The core, scored directly. Both sides run the SAME CPU tail here, so
        // the layer output below is the core's error diluted through the gate
        // and the MLP -- informative, but this is the number with teeth.
        report_cos("a window at a non-128-aligned cursor",
                   cosine(core_npu.data(), attn_r.data(), core_npu.size()),
                   0.99, note);

        std::vector<float> lo((size_t)win * D, 0.0f), ref_norm((size_t)win * D, 0.0f);
        half.gate_out_mlp(qg_all.data() + (size_t)pre * 2 * DQ_, attn_r.data(),
                          tp_all.fc_out.data() + (size_t)pre * D, win, lo.data());
        for (int r = 0; r < win; r++)
            cpu_ref::rms_norm(lo.data() + (size_t)r * D, out_norm.data(), D,
                              ref_norm.data() + (size_t)r * D);
        report_cos("  and its norm_out",
                   cosine(norm_npu.data(), ref_norm.data(), norm_npu.size()),
                   0.999, "-- the CPU tail is shared, so this only dilutes");
    }

    // And the deepest window the cache can hold. The 221 real rows are tiled to
    // fill it: synthetic N(0,1) keys would put ~all the softmax mass on the
    // appended row and make the replay untestable, whereas repeated real rows
    // keep the queries and keys correlated. The point of the pass is the
    // ADDRESS it reaches, not its content.
    std::printf("\n-- the deepest legal window --\n");
    //
    // It is the boundary to the row, now that there is no apron: 127 queries
    // from 896 pad to a whole chunk ending at exactly MAX_L, so run()'s guard
    // admits it and the broadcast's round-up lands ON the top of the plane.
    // One row higher and the pass is refused.
    {
        const int top = t.MAX_L - 1;                 // _set_context stops at MAX_L-1
        const int win = C - 1, begin = top - win;    // 896 .. 1022
        std::vector<float> kb((size_t)top * DKV_), vb((size_t)top * DKV_);
        std::vector<float> qb((size_t)top * DQ_);
        for (int r = 0; r < top; r++) {
            const int s = r % P;
            std::memcpy(kb.data() + (size_t)r * DKV_,
                        kk_all.data() + (size_t)s * DKV_, DKV_ * sizeof(float));
            std::memcpy(vb.data() + (size_t)r * DKV_,
                        vv_all.data() + (size_t)s * DKV_, DKV_ * sizeof(float));
            std::memcpy(qb.data() + (size_t)r * DQ_,
                        qh_all.data() + (size_t)s * DQ_, DQ_ * sizeof(float));
        }

        layer.reset();
        layer.mirror_from_cpu(kb, vb, begin);
        layer.append_rows(kb.data() + (size_t)begin * DKV_,
                          vb.data() + (size_t)begin * DKV_, win);

        std::vector<float> a((size_t)win * DQ_, 0.0f);
        op.run(qb.data() + (size_t)begin * DQ_, win, begin, layer.cache(),
               a.data());

        cpu_ref::mtp_head aux(q4nx, &embed, kMaxL);
        aux.seed_cache(kb, vb, top);
        std::vector<float> r;
        aux.attn_core(qb.data() + (size_t)begin * DQ_, win, begin, r);

        char note[160];
        std::snprintf(note, sizeof note,
                      "-- L_begin %d, reach %d rows into a plane of %zu",
                      begin, kv_reach(begin, C), t.plane_rows());
        report_cos("the deepest legal window runs",
                   cosine(a.data(), r.data(), a.size()), 0.99, note);
    }

    // ======================================================================
    // 8. continuity: one cache, one cursor, two bitstreams
    // ======================================================================
    // THE gate. A full seven-step draft through the engine's own code:
    // `speculate_npu::draft()` takes the prefill shape on step 0 and the decode
    // bitstream on steps 1-6, on the SAME cache BO, and must produce the
    // reference's token ids -- all seven, exactly, since drafting is greedy by
    // construction and acceptance is an integer compare.
    //
    // Everything above can pass while `_prefill_step` hands the op the
    // post-append cursor, or keeps the wrong row as h2, or leaves the cursor
    // somewhere step 1 cannot resume from. None of those survive this.
    std::printf("\n-- a full 7-step draft --\n");
    const std::vector<int>   ids(ids32.begin(), ids32.end());
    const int k = spec_cpu.max_steps();
    std::vector<int> d_cpu, d_npu;
    {
        spec_cpu.reset();
        d_cpu = spec_cpu.draft(ids, hid.data(), P, kPos0, k);
        spec_npu.reset();
        d_npu = spec_npu.draft(ids, hid.data(), P, kPos0, k);

        std::printf("        cpu   %s\n        npu   %s\n",
                    id_list(d_cpu).c_str(), id_list(d_npu).c_str());

        // Scored decisively-only, on the same rule as check 9 -- a 7-deep
        // autoregression is not required to be stable at a step the reference
        // itself could not call. It has always come out exact at this width,
        // and the message says so when it does, which is the useful signal.
        std::vector<int>          m_ids;
        const std::vector<double> mg =
            draft_margins(cpu, head, desc.vocab_size, ids, hid.data(), P, kPos0,
                          k, m_ids);
        int agree = 0;
        for (size_t i = 0; i < d_cpu.size() && i < d_npu.size(); i++)
            if (d_cpu[i] == d_npu[i]) agree++; else break;
        const bool decisive_ok =
            m_ids == d_cpu && (agree == k || mg[(size_t)agree] < kDecisive);

        char buf[200];
        if (agree == k)
            std::snprintf(buf, sizeof buf, "all %d ids match, margins %.2f..%.2f",
                          k, *std::min_element(mg.begin(), mg.end()),
                          *std::max_element(mg.begin(), mg.end()));
        else
            std::snprintf(buf, sizeof buf,
                          "%d of %d from the start; step %d margin %.3f (%s)",
                          agree, k, agree, mg[(size_t)agree],
                          mg[(size_t)agree] < kDecisive ? "undecided"
                                                        : "DECISIVE");
        report("the drafted ids match the reference", decisive_ok, buf);

        std::snprintf(buf, sizeof buf,
                      "npu %d, cpu %d (want %d = %d + %d decode steps)",
                      spec_npu.cache_len(), spec_cpu.cache_len(), P + k - 1, P,
                      k - 1);
        report("the cursors agree across the seam",
               spec_npu.cache_len() == spec_cpu.cache_len() &&
               spec_npu.cache_len() == P + k - 1, buf);
    }

    // ======================================================================
    // 9. the fallback is the same computation
    // ======================================================================
    // `FLM_Q38_MTP_PREFILL=0` -- here, a speculator built without the bitstream
    // -- must not change the answer, only the shape. At T = 1 both take the
    // per-row path and the two are running literally the same code, so the ids
    // must match exactly. At T = 8 they diverge in arithmetic: the decode
    // bitstream appends and replays row by row while the prefill path appends
    // the whole window and attends once.
    //
    // Both are exact, so what is being tested is that the difference stays
    // below the argmax -- but only where there IS an argmax to stay below.
    // `draft_margins` supplies the reference's own top-1/top-2 gap per step, and
    // a mismatch is a failure at a step the reference decided and a reported
    // coin toss at a step it did not. Without that split this check asserts
    // that a 7-deep autoregression is stable to 1e-2 relative, which is not a
    // property either path has or should be made to have.
    std::printf("\n-- the kill switch changes the shape, not the answer --\n");
    for (int T : {1, 8}) {
        // Both caches zeroed, not just rewound. The two speculators have run
        // different histories by now, so `reset()` alone would leave different
        // rows above the cursor -- and this comparison is not the place to find
        // out what those cost. Check 4 owns that question; here the two paths
        // must start from the same bytes or the A/B means nothing.
        spec_npu.layer().cache().memset((bf16)0);
        spec_npu.layer().cache().sync_to_device();
        spec_fb.layer().cache().memset((bf16)0);
        spec_fb.layer().cache().sync_to_device();
        spec_npu.reset();
        spec_fb.reset();
        spec_cpu.reset();
        const std::vector<int> sub(ids.begin(), ids.begin() + T);
        const std::vector<int> a = spec_npu.draft(sub, hid.data(), T, kPos0, k);
        const std::vector<int> b = spec_fb.draft(sub, hid.data(), T, kPos0, k);
        const std::vector<int> c = spec_cpu.draft(sub, hid.data(), T, kPos0, k);

        // The reference's own confidence at each of the seven steps, from the
        // same trajectory `c` came from -- `m_ids` must equal `c`, and it is
        // checked below, because a margin read off a different trajectory would
        // excuse the wrong steps.
        std::vector<int>          m_ids;
        const std::vector<double> mg =
            draft_margins(cpu, head, desc.vocab_size, sub, hid.data(), T, kPos0,
                          k, m_ids);

        std::string decided;
        for (int s = 0; s < k; s++) {
            char c8[32];
            std::snprintf(c8, sizeof c8, "%s%.3f%s", s ? " " : "", mg[(size_t)s],
                          mg[(size_t)s] >= kDecisive ? "" : "?");
            decided += c8;
        }
        std::printf("        T = %3d  reference margin/sd per step:  %s"
                    "   ('?' = below %.2f, an undecided argmax)\n",
                    T, decided.c_str(), kDecisive);

        char tag[64], buf[256];
        std::snprintf(tag, sizeof tag, "T = %d: the margins are the reference's",
                      T);
        std::snprintf(buf, sizeof buf, "%s  vs speculate_cpu %s",
                      id_list(m_ids).c_str(), id_list(c).c_str());
        report(tag, m_ids == c, buf);

        // A disagreement counts only where the reference was decisive.
        int first_bad = -1, undecided = 0;
        for (int s = 0; s < k; s++) {
            const bool same = a[(size_t)s] == b[(size_t)s] &&
                              a[(size_t)s] == c[(size_t)s];
            if (same) continue;
            if (mg[(size_t)s] >= kDecisive) { first_bad = s; break; }
            undecided++;
            break;   // past the first coin toss the trajectories are unrelated
        }

        std::snprintf(tag, sizeof tag, "T = %d: prefill == fallback == CPU", T);
        if (first_bad >= 0)
            std::snprintf(buf, sizeof buf,
                          "step %d differs at margin %.3f -- decisive:  %s | %s | %s",
                          first_bad, mg[(size_t)first_bad], id_list(a).c_str(),
                          id_list(b).c_str(), id_list(c).c_str());
        else if (undecided)
            std::snprintf(buf, sizeof buf,
                          "agree until the reference's first coin toss:  %s | %s | %s",
                          id_list(a).c_str(), id_list(b).c_str(),
                          id_list(c).c_str());
        else
            std::snprintf(buf, sizeof buf, "all %d ids identical:  %s%s", k,
                          id_list(a).c_str(),
                          T == 1 ? "  (both on the decode path)" : "");
        report(tag, first_bad < 0, buf);
    }

    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
