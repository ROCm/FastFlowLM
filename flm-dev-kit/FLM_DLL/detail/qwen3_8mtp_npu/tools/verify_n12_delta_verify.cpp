// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify_n12_delta_verify.cpp
/// \brief N12 -- the WHOLE gated-delta-net block on the NPU, for the verify
///        phase: `gate_delta_net_verify`'s `deltanet_layer.xclbin` driven
///        through `q38npu::npu_delta_layer`.
///
/// Build:
///   source ~/setup.sh
///   g++-13 -O2 -march=native -fopenmp -std=c++20
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n12_delta_verify.cpp common/cpu_ref_delta_net_layer.cpp
///       verify/npu/delta_sequence.cpp speculate/npu/proj_sequence.cpp
///       ../q4nx/q4_npu_eXpress.cpp ../q4nx/safe_tensors.cpp
///       -L../../../../src/lib/xrt -L../../build/lib -L/opt/xilinx/xrt/lib
///       -lxrt_coreutil -laiebu -luuid -lpthread -o /tmp/verify_n12_delta_verify
///   (one line; the continuations are dropped here because a backslash at the
///    end of a // comment splices the next line into it)
///
/// Run:
///   /tmp/verify_n12_delta_verify [model_dir] [deltanet_layer.xclbin]
///
/// ---------------------------------------------------------------------------
/// What moved
/// ---------------------------------------------------------------------------
/// N9 and N10 put the 48 delta layers' matmuls and then their recurrence on the
/// device **for prefill**. The verify phase -- the L = 8 pass that scores a
/// draft -- was still running those 48 layers whole on the host, which was the
/// last real gap in this engine. This phase closes it with one bitstream that
/// runs the entire block: six projections, the causal conv, the gated delta
/// rule, the output gate and the MLP residual, one dispatch per layer.
///
/// ---------------------------------------------------------------------------
/// The two tensors that are this layer's "KV cache"
/// ---------------------------------------------------------------------------
/// The conv window and the recurrent state S, and they are carried by two
/// different mechanisms on purpose.
///
/// The **conv window** comes back off the device, and not one window but one
/// per accept length: at dispatch time nobody knows how many of the 8 drafted
/// tokens verification will keep, and accept-j needs the last
/// `conv_kernel_size - 1` pre-conv rows *as of token j*. Those runs overlap, so
/// their union is one contiguous block the host indexes (check 5).
///
/// **S does not come back at all.** The only state the array could report is
/// the accept-*all* one, which is exactly the case speculative decode cannot
/// rely on. So the host replays
///
///     u_t = beta_t * (v_t - alpha_t * (S_{t-1} @ k_t))
///     S_t = alpha_t * S_{t-1} + k_t (x) u_t
///
/// over the triplet arena the pass already had to write for its own
/// token-major -> head-major transpose, and that replay overlaps the *next*
/// layer's dispatch (check 9 is what says the overlap is safe).
///
/// ---------------------------------------------------------------------------
/// How it is scored
/// ---------------------------------------------------------------------------
/// Against `cpu_ref::delta_net_layer` reading the same `model.q4nx`, so Q4_K's
/// own loss is common mode on both sides. Inputs are the dumped output of
/// layer 3 -- real activations, bf16-rounded, never N(0,1)
/// (memory: random-kv-cache-cannot-test-attention).
///
/// Every score is paired with a control that must COLLAPSE, because a cosine
/// near 1 is evidence about arithmetic and says nothing about addressing:
///
///   h2 vs a rotated input   proves h2 is this window's and not a stale BO.
///   conv window j vs j+1    the accept-length index is one add; an off-by-one
///                           there is the single most plausible mistake in the
///                           whole carry and no cosine against j alone sees it.
///   S cross-head            device head h against reference head h+1. A
///                           consistent permutation agrees with itself.
///                           (memory: cross-head-control-catches-gqa-mapping)
///   S alpha <-> beta        decay and write strength are both per-head
///                           scalars in (0,1) living two words apart in the
///                           same arena block. Swapping them is how that
///                           scatter gets mis-wired, and only this separates it.
///
/// Two more are about properties rather than arithmetic. The pass must survive
/// a **second dispatch** -- the recurring failure on this hardware is a design
/// that is right once and parks or drifts afterwards
/// (memory: a-prologue-bd-breaks-the-second-dispatch) -- and two identical
/// dispatches from the same state must be **bit-identical**.
///
/// What this driver does NOT cover is the 64-layer stack. It used to not cover
/// the `FLM_Q38_DELTA_VERIFY=0` A/B either, which `test/qwen3_8mtp_npu` ran
/// with the switch flipped; that switch is gone -- `deltanet_layer.xclbin` is
/// required and `layer_stack::build` throws without it -- so this driver is
/// now the whole of what tests the delta-net verify layer, and the A/B's
/// recorded number (a 20% verify-phase win) is not re-measurable end to end.
/// (memory: host-vs-device-ab-needs-shipping-flags)
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

/// A gated-delta-net layer. The full-attention ones are (i + 1) % 4 == 0, so 4
/// is the first delta layer whose input is a dumped `layer_XX_out`.
constexpr int kLayer = 4;

/// The bitstream's -DL_TOKENS. Not a preference: the sequence and the kernels
/// are compiled at it.
constexpr int kL = 8;

/// The bar for a cosine against a same-bytes reference. Everything on the
/// device is bf16 -- the projections' output, the conv taps, the arena, the
/// residual -- so relative L2 lands near 2e-2 where the cosine is at four
/// nines, and it is the cosine that is the gate. Relative L2 is printed beside
/// every score because it is the number that moves when something is wrong by
/// a gain rather than by a rotation.
constexpr double kGate = 0.999;

/// The ceiling a control must fall under. Generous on purpose -- a control at
/// 0.98 has still told us the addressing is wrong.
constexpr double kControl = 0.99;

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-54s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
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

/// Relative L2, which is what a flat additive error should be judged on. A
/// per-row cosine on rows of unequal norm reports the norm, not the error.
/// (memory: per-row-cosine-reports-the-norm)
double rel_l2(const float* a, const float* b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)a[i] - b[i];
        num += d * d;
        den += (double)b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : 0.0;
}

void report_score(const char* tag, const std::vector<float>& got,
                  const std::vector<float>& ref, const char* note = "") {
    const double c = cosine(got.data(), ref.data(), got.size());
    const double e = rel_l2(got.data(), ref.data(), ref.size());
    char buf[240];
    std::snprintf(buf, sizeof buf, "cos %.8f (gate %.4f), rel L2 %.3e %s",
                  c, kGate, e, note);
    report(tag, c >= kGate, buf);
}

void report_control(const char* tag, double cos, const char* note = "") {
    char buf[200];
    std::snprintf(buf, sizeof buf, "cos %.6f  (must be < %.3f) %s",
                  cos, kControl, note);
    report(tag, cos < kControl, buf);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

// --------------------------------------------------------------------------
/// The delta rule over the device's own triplet arena, as an IMPOSTER.
///
/// `npu_delta_layer::replay_state_at()` is the production fold and is what the
/// scores below measure. This is a second, independent copy whose only purpose
/// is to be wrong in one specific way -- `swap_gates` exchanges alpha and beta,
/// which is the mis-wiring the arena's two-word holes invite. A control has to
/// come from outside the thing it controls, so it does not call into the layer.
///
/// \param arena the layer's arg-0 BO at `arena_offset`, bf16
std::vector<float> imposter_fold(const bf16* arena,
                                 const q38npu::delta_verify_layout& t,
                                 const std::vector<float>& S0, int n,
                                 bool swap_gates) {
    const int dk = t.qk_head_dim, dv = t.v_head_dim;
    const size_t count = (size_t)dk * dv;
    std::vector<float> S = S0;

#pragma omp parallel
    {
        std::vector<float> u(dv);
#pragma omp for schedule(static)
        for (int h = 0; h < t.v_heads; h++) {
            float* Sh = S.data() + (size_t)h * count;
            for (int tok = 0; tok < n; tok++) {
                const bf16* blk = arena + t.arena_at(h, tok);
                const bf16* v_v = blk + t.triplet_stride;
                float alpha, beta;
                std::memcpy(&alpha, blk + dk, sizeof(float));
                std::memcpy(&beta,  v_v + dv, sizeof(float));
                if (swap_gates) std::swap(alpha, beta);

                for (int j = 0; j < dv; j++) u[j] = 0.0f;
                for (int i = 0; i < dk; i++) {
                    const float kk = (float)blk[i];
                    if (kk == 0.0f) continue;
                    const float* Sr = Sh + (size_t)i * dv;
                    for (int j = 0; j < dv; j++) u[j] += Sr[j] * kk;
                }
                for (int j = 0; j < dv; j++)
                    u[j] = beta * ((float)v_v[j] - alpha * u[j]);
                for (int i = 0; i < dk; i++) {
                    const float kk = (float)blk[i];
                    float* Sr = Sh + (size_t)i * dv;
                    for (int j = 0; j < dv; j++) Sr[j] = alpha * Sr[j] + kk * u[j];
                }
            }
        }
    }
    return S;
}

/// The CPU layer's two carried states after exactly `n` tokens from `snap`.
struct ref_window {
    std::vector<float> y;        ///< [n][D]
    std::vector<float> state;    ///< [v_heads][dk][dv]
    std::vector<float> conv;     ///< the DEVICE's window: (K-1) * conv_dim
};

ref_window run_reference(cpu_ref::delta_net_layer& c,
                         const std::vector<uint8_t>& snap,
                         const float* x, int n) {
    const cpu_ref::delta_net_geometry& g = c.geometry();
    ref_window r;
    c.load_state(snap.data());
    r.y.assign((size_t)n * g.model_dim, 0.0f);
    c.forward(x, n, r.y.data());
    r.state = c.recurrent_state();
    // Rows 1 .. conv_k-1 of the CPU window, which is the device's whole one:
    // row 0 is shifted out by the next conv_silu() before it is ever
    // multiplied by a tap, so neither side carries it.
    const size_t CD = (size_t)g.conv_dim();
    r.conv.assign((size_t)(g.conv_k - 1) * CD, 0.0f);
    std::memcpy(r.conv.data(), c.conv_window().data() + CD, r.conv.size() * sizeof(float));
    return r;
}

} // namespace

// ==========================================================================
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

    const int D   = geo.model_dim;
    const int CD  = geo.conv_dim();
    const int NV  = geo.v_heads;
    const int DKH = geo.k_head_dim, DVH = geo.v_head_dim;
    const size_t SN = (size_t)NV * DKH * DVH;

    std::printf("== N12: the whole delta-net block on deltanet_layer.xclbin ==\n");
    std::printf("   model  %s\n   xclbin %s\n   layer  %d  (%s)\n\n",
                model_dir.c_str(), xclbin.c_str(), kLayer, geo.str().c_str());

    // ======================================================================
    // 1. the geometry guard
    // ======================================================================
    // `delta_verify_layout::build()` throws on most of this, but it throws
    // with one message from inside a constructor, so a violated precondition
    // would take the driver down without saying which condition it was.
    q38npu::delta_verify_layout t;
    {
        std::string bad;
        try {
            t = q38npu::delta_verify_layout::build(geo, kL);
        } catch (const std::exception& e) {
            bad = e.what();
        }
        char buf[300];
        if (bad.empty())
            std::snprintf(buf, sizeof buf,
                          "L %d, D %d, conv_dim %d (K %d), %d heads x %dx%d; "
                          "arena %zu w, kv %zu w, weights %zu MB",
                          t.L_TOKENS, t.D, t.conv_dim, t.conv_kernel_size,
                          t.v_heads, t.qk_head_dim, t.v_head_dim,
                          (size_t)t.arena_heads * t.L_TOKENS * t.arena_block,
                          t.kv_cache_words, t.w.bo_bytes >> 20);
        report("the layout the bitstream was compiled for", bad.empty(),
               bad.empty() ? buf : bad);
        if (!bad.empty()) return 1;
    }

    // ======================================================================
    // the device, the layer, and the reference
    // ======================================================================
    cpu_ref::safetensors q4nx(model_dir + "/model.q4nx");
    cpu_ref::safetensors ref(model_dir + "/ref.safetensors");

    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager*   layer_mgr = mgr.register_xclbin(xclbin);

    q38npu::delta_engine eng(layer_mgr->create_app(), t);
    // No shared weight BO here: the driver is the only consumer, so the layer
    // stages its own 212 MB. In the engine the stack hands it the prefill
    // phase's, which is what the a/b tail past proj_bytes exists for.
    q38npu::npu_delta_layer dut(&eng, q4nx, geo, kLayer);
    cpu_ref::delta_net_layer cpu(q4nx, geo, kLayer);

    // Real activations: the dumped output of layer 3, which is this layer's
    // input. Rounded to bf16 because that is what the device will receive --
    // letting the reference start from extra precision charges the device for
    // a narrowing it cannot avoid.
    char key[64];
    std::snprintf(key, sizeof key, "prefill.layer_%02d_out", kLayer - 1);
    std::vector<float> x_all;
    try {
        x_all = ref.f32(key);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ref.safetensors: %s\n", e.what());
        return 1;
    }
    const int P = (int)(x_all.size() / D);
    if (P < 2 * kL) {
        std::fprintf(stderr, "%s has only %d rows; need %d\n", key, P, 2 * kL);
        return 1;
    }
    for (float& f : x_all) f = cpu_ref::round_bf16(f);
    const float* x1 = x_all.data();                       // window 1
    const float* x2 = x_all.data() + (size_t)kL * D;      // window 2

    std::printf("   window %d rows from %s, first %d used\n\n", P, key, 2 * kL);

    // ======================================================================
    // 2. the small weights, byte for byte
    // ======================================================================
    // The one thing here with no end-to-end proof of its own layout. A wrong
    // a/b offset makes S garbage and check 6 catches it, but a wrong *length*
    // -- staging alpha where beta belongs, say -- can still leave S plausible
    // for one window. These are plain memcmps against the checkpoint, so they
    // pin the staging exactly and cost nothing.
    {
        const std::string p = "model.layers." + std::to_string(kLayer) + ".";
        const std::string l = p + "linear_attn.";
        const uint8_t* w = dut.delta_weights()->weights().data();
        const size_t PH = (size_t)t.v_heads_padded;
        const size_t half = q38npu::q80_bytes(PH * (size_t)D);

        std::string bad;
        auto want = [&](bool ok, const std::string& why) {
            if (!ok) bad += (bad.empty() ? "" : ", ") + why;
        };
        want(std::memcmp(w + t.w.w_ab_offset, q4nx.raw(l + "ssm_alpha_proj.weight"),
                         half) == 0, "alpha q8_0 differs");
        want(std::memcmp(w + t.w.w_ab_offset + half,
                         q4nx.raw(l + "ssm_beta_proj.weight"), half) == 0,
             "beta q8_0 differs");

        // The bias block: A in the first v_heads_padded f32 lanes, ssm_dt.bias
        // in the next, everything above zero so a pad head's alpha comes out
        // exp(0 * softplus(0)) = 1 and not a garbage exponential.
        const float* bias =
            reinterpret_cast<const float*>(w + t.w.w_ab_bias_offset);
        const std::vector<float> a = q4nx.f32(l + "ssm_a");
        const std::vector<float> dt = q4nx.f32(l + "ssm_dt.bias");
        want(a.size() == (size_t)NV && dt.size() == (size_t)NV,
             "ssm_a / ssm_dt.bias are not v_heads long");
        for (int h = 0; h < NV; h++) {
            if (bias[h] != a[h])       { want(false, "ssm_a lane " + std::to_string(h)); break; }
            if (bias[PH + h] != dt[h]) { want(false, "dt lane " + std::to_string(h)); break; }
        }
        for (size_t i = NV; i < PH; i++)
            if (bias[i] != 0.0f || bias[PH + i] != 0.0f) {
                want(false, "a pad head's bias lane is not zero"); break;
            }

        char buf[220];
        std::snprintf(buf, sizeof buf,
                      "alpha|beta %zu B each at +%zu, bias block %zu B at +%zu",
                      half, t.w.w_ab_offset,
                      q38npu::delta_layout::AB_W_BLOCK_BYTES, t.w.w_ab_bias_offset);
        report("the a/b tail is staged byte for byte", bad.empty(),
               bad.empty() ? buf : bad);
    }

    // ======================================================================
    // 3. the layer, window 1
    // ======================================================================
    dut.reset();
    cpu.reset();
    std::vector<uint8_t> snap0(cpu.state_bytes());
    cpu.save_state(snap0.data());                       // both sides are zero

    std::vector<float> y_dev((size_t)kL * D);
    const auto tp0 = std::chrono::steady_clock::now();
    dut.forward(x1, kL, nullptr, nullptr, y_dev.data());
    const double ms_pass = ms_since(tp0);

    const ref_window r8 = run_reference(cpu, snap0, x1, kL);
    report_score("h2: the whole block, 8 rows", y_dev, r8.y, "(vs cpu_ref)");

    // ------------------------------------------------------------------
    // 4. the h2 control
    // ------------------------------------------------------------------
    // The reference for a DIFFERENT window. h2 sits in the same BO the next
    // dispatch overwrites, and a pass that silently did not run leaves the
    // previous contents there looking entirely reasonable
    // (memory: unchecked-ert-state-looks-like-a-pass).
    {
        const ref_window rot = run_reference(cpu, snap0, x2, kL);
        report_control("h2 control: window 2's reference",
                       cosine(y_dev.data(), rot.y.data(), y_dev.size()));
    }

    // ======================================================================
    // 5. the conv window, one accept length at a time
    // ======================================================================
    // The carry the host owns. Scored at every accept length, because the
    // device writes one block covering all of them and the host indexes it --
    // `conv_out_at(j) = (j - 1) * conv_dim`, a single add, and a cosine at
    // j = 8 alone would pass with that add wrong.
    {
        double worst = 1.0; int worst_j = 0;
        double best_off = 0.0; int off_j = 0;
        for (int j = 1; j <= kL; j++) {
            const ref_window rj = run_reference(cpu, snap0, x1, j);
            const std::vector<float> got = dut.read_conv_out(j);
            const double c = cosine(got.data(), rj.conv.data(), got.size());
            if (c < worst) { worst = c; worst_j = j; }
            if (j < kL) {
                const ref_window rn = run_reference(cpu, snap0, x1, j + 1);
                const double n = cosine(got.data(), rn.conv.data(), got.size());
                if (n > best_off) { best_off = n; off_j = j; }
            }
        }
        char buf[220];
        std::snprintf(buf, sizeof buf,
                      "worst cos %.8f at accept %d (gate %.4f), %d lengths",
                      worst, worst_j, kGate, kL);
        report("the conv window at every accept length", worst >= kGate, buf);

        std::snprintf(buf, sizeof buf, "(best offender accept %d)", off_j);
        report_control("conv control: accept j vs reference j+1", best_off, buf);
    }

    // ======================================================================
    // 6. the recurrent state, one accept length at a time
    // ======================================================================
    // The half that never crosses DDR. `replay_state_at()` is the production
    // fold; its input arena is the device's own.
    {
        double worst = 1.0; int worst_j = 0;
        double worst_e = 0.0;
        double best_off = 0.0; int off_j = 0;
        for (int j = 1; j <= kL; j++) {
            const ref_window rj = run_reference(cpu, snap0, x1, j);
            // The pass started from a zeroed state, so that is the S0 the
            // replay has to be given -- the layer's own S has already been
            // advanced past it.
            const std::vector<float> S =
                dut.replay_state_at(std::vector<float>(SN, 0.0f), j);
            const double c = cosine(S.data(), rj.state.data(), SN);
            const double e = rel_l2(S.data(), rj.state.data(), SN);
            if (c < worst) { worst = c; worst_j = j; worst_e = e; }
            if (j < kL) {
                const ref_window rn = run_reference(cpu, snap0, x1, j + 1);
                const double n = cosine(S.data(), rn.state.data(), SN);
                if (n > best_off) { best_off = n; off_j = j; }
            }
        }
        char buf[240];
        std::snprintf(buf, sizeof buf,
                      "worst cos %.8f at accept %d (gate %.4f), rel L2 %.3e",
                      worst, worst_j, kGate, worst_e);
        report("the replayed state at every accept length", worst >= kGate, buf);

        std::snprintf(buf, sizeof buf, "(best offender accept %d)", off_j);
        report_control("S control: accept j vs reference j+1", best_off, buf);
    }

    // ------------------------------------------------------------------
    // 7. the state controls
    // ------------------------------------------------------------------
    // The state the layer itself now holds, which is the one the next window
    // and any rollback will use -- not a driver-side replay.
    std::vector<float> S_dev_8;
    {
        const std::vector<float>& S = dut.inner().recurrent_state();
        S_dev_8 = S;
        report_score("the layer's own state after the window", S, r8.state,
                     "(vs cpu_ref)");

        // Cross-head: device head h against reference head h+1. A consistent
        // permutation agrees with itself, so a whole-tensor cosine cannot see
        // one. (memory: cross-head-control-catches-gqa-mapping)
        const size_t count = (size_t)DKH * DVH;
        std::vector<float> rot(SN);
        for (int h = 0; h < NV; h++)
            std::memcpy(rot.data() + (size_t)h * count,
                        r8.state.data() + (size_t)((h + 1) % NV) * count,
                        count * sizeof(float));
        report_control("S control: cross-head (h vs reference h+1)",
                       cosine(S.data(), rot.data(), SN));

        // alpha <-> beta. Both are per-head scalars in (0,1) and they live two
        // words apart in the same arena block; this is how that scatter gets
        // mis-wired, and nothing else here separates it.
        const bf16* arena = dut.hidden_states().data() + t.arena_offset;
        const std::vector<float> swapped =
            imposter_fold(arena, t, std::vector<float>(SN, 0.0f), kL, true);
        report_control("S control: alpha and beta exchanged",
                       cosine(swapped.data(), r8.state.data(), SN));

        // And the same imposter with the gates the right way round must AGREE,
        // which is what says the control above is a control and not just a
        // second broken fold.
        const std::vector<float> straight =
            imposter_fold(arena, t, std::vector<float>(SN, 0.0f), kL, false);
        report_score("the imposter fold, gates unswapped", straight, r8.state,
                     "(the control's own control)");
    }

    // ======================================================================
    // 8. the second dispatch
    // ======================================================================
    // The recurring failure on this hardware is a design that is right once
    // and parks or drifts on dispatch 2 -- a prologue BD, an unreceived tap
    // eating its lock credits, a ring the host never re-primes.
    // (memory: a-prologue-bd-breaks-the-second-dispatch,
    //          unrouted-mm2s-chain-still-fires)
    //
    // This one also carries: window 2 starts from window 1's conv window and
    // window 1's replayed S, both written back into the CPU half. So it scores
    // the carry as much as the dispatch.
    std::vector<float> y2((size_t)kL * D);
    {
        std::vector<uint8_t> snap8(cpu.state_bytes());
        cpu.load_state(snap0.data());
        std::vector<float> ytmp((size_t)kL * D);
        cpu.forward(x1, kL, ytmp.data());
        cpu.save_state(snap8.data());

        dut.forward(x2, kL, nullptr, nullptr, y2.data());
        const ref_window r2 = run_reference(cpu, snap8, x2, kL);

        report_score("h2: dispatch 2, carried state", y2, r2.y, "(vs cpu_ref)");
        report_score("the state after dispatch 2", dut.inner().recurrent_state(),
                     r2.state, "(vs cpu_ref)");
        report_score("the conv window after dispatch 2", dut.read_conv_out(kL),
                     r2.conv, "(vs cpu_ref)");

        // It moved at all. A dispatch that silently did nothing leaves the
        // state exactly as the host wrote it in, and every score above is
        // against a reference that also advanced -- so this is the one check
        // that would fail on a no-op with a correct-looking S.
        const double self = cosine(dut.inner().recurrent_state().data(),
                                   S_dev_8.data(), SN);
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "cos(S after 16, S after 8) = %.6f (must be < %.3f)",
                      self, kControl);
        report("dispatch 2 moved the state", self < kControl, buf);
    }

    // ======================================================================
    // 9. determinism, and the overlap
    // ======================================================================
    // Same window, same input state, twice -- bit-identical, not merely close.
    // A fold that races the next dispatch's arena write is wrong by a few ulp
    // and a cosine will never see it; a memcmp will.
    {
        std::vector<uint8_t> snapA(cpu.state_bytes());
        dut.inner().save_state(snapA.data());

        std::vector<float> ya((size_t)kL * D), yb((size_t)kL * D);
        dut.forward(x1, kL, nullptr, nullptr, ya.data());
        const std::vector<float> Sa = dut.inner().recurrent_state();
        const std::vector<float> Ca = dut.read_conv_out(kL);

        dut.inner().load_state(snapA.data());
        dut.forward(x1, kL, nullptr, nullptr, yb.data());
        const std::vector<float> Sb = dut.inner().recurrent_state();
        const std::vector<float> Cb = dut.read_conv_out(kL);

        size_t dy = 0, ds = 0, dc = 0;
        for (size_t i = 0; i < ya.size(); i++) if (ya[i] != yb[i]) dy++;
        for (size_t i = 0; i < Sa.size(); i++) if (Sa[i] != Sb[i]) ds++;
        for (size_t i = 0; i < Ca.size(); i++) if (Ca[i] != Cb[i]) dc++;
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "h2 %zu/%zu, S %zu/%zu, conv %zu/%zu words differ",
                      dy, ya.size(), ds, Sa.size(), dc, Ca.size());
        report("two identical dispatches are bit-identical",
               dy == 0 && ds == 0 && dc == 0, buf);

        // The same window with the replay run INLINE. If the worker ever read
        // an arena the next dispatch had begun to overwrite, this is where the
        // two would part -- and they must part by nothing at all, because the
        // arithmetic is the same code either way.
        eng.overlap = false;
        dut.inner().load_state(snapA.data());
        std::vector<float> yc((size_t)kL * D);
        dut.forward(x1, kL, nullptr, nullptr, yc.data());
        const std::vector<float> Sc = dut.inner().recurrent_state();
        eng.overlap = true;

        size_t dsy = 0;
        for (size_t i = 0; i < Sa.size(); i++) if (Sa[i] != Sc[i]) dsy++;
        std::snprintf(buf, sizeof buf,
                      "S %zu/%zu words differ from the overlapped run",
                      dsy, Sa.size());
        report("FLM_Q38_DELTA_SYNC changes nothing", dsy == 0, buf);
    }

    // ======================================================================
    // 10. what it costs
    // ======================================================================
    // Not a check, and the numbers below are only meaningful from a binary
    // built at the engine's own flags -- the host layer this replaces is
    // -O3 -ffast-math -mavx512f, and an -O2 build of it has been enough to
    // flip the sign of a verdict before.
    // (memory: host-vs-device-ab-needs-shipping-flags)
    //
    // Three numbers matter: what a dispatch costs, what the replay costs
    // (it is the whole reason the conv comes back off the device and S does
    // not), and what the host layer this displaces costs.
    {
        const int reps = 8;
        std::vector<uint8_t> snapT(cpu.state_bytes());
        dut.inner().save_state(snapT.data());

        eng.overlap = false;
        std::vector<float> yt((size_t)kL * D);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) {
            dut.inner().load_state(snapT.data());
            dut.forward(x1, kL, nullptr, nullptr, yt.data());
        }
        const double ms_sync = ms_since(t0) / reps;

        eng.overlap = true;
        const auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) {
            dut.inner().load_state(snapT.data());     // joins the pending fold
            dut.forward(x1, kL, nullptr, nullptr, yt.data());
        }
        const double ms_async = ms_since(t1) / reps;

        // The layer this displaces, on the same 8 rows.
        std::vector<float> yh((size_t)kL * D);
        const auto t2 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) {
            cpu.load_state(snap0.data());
            cpu.forward(x1, kL, yh.data());
        }
        const double ms_host = ms_since(t2) / reps;

        std::printf("\n-- what a pass costs, layer %d, L = %d --\n", kLayer, kL);
        std::printf("   first pass (cold)            %8.2f ms\n", ms_pass);
        std::printf("   dispatch + inline replay     %8.2f ms\n", ms_sync);
        std::printf("   dispatch, replay deferred    %8.2f ms   (the replay is "
                    "%.2f ms and overlaps the NEXT layer)\n",
                    ms_async, ms_sync - ms_async);
        std::printf("   the host layer it replaces   %8.2f ms   (%.2fx)\n",
                    ms_host, ms_host / ms_async);
        std::printf("   48 layers: device %.0f ms, host %.0f ms per verify pass\n",
                    48.0 * ms_async, 48.0 * ms_host);
    }

    // ======================================================================
    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
