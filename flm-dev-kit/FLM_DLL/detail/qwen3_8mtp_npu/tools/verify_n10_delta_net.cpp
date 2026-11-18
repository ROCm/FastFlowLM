// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify_n10_delta_net.cpp
/// \brief N10 -- the gated-delta-net RECURRENCE on the NPU: the depthwise
///        conv1d on `conv.xclbin` and the fold on
///        `GateDeltaNet_prefill.xclbin`.
///
/// Build:
///   source ~/setup.sh
///   g++-13 -O2 -march=native -fopenmp -std=c++20
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n10_delta_net.cpp common/cpu_ref_delta_net_layer.cpp
///       prefill/npu/delta_net_sequence.cpp speculate/npu/proj_sequence.cpp
///       ../q4nx/q4_npu_eXpress.cpp ../q4nx/safe_tensors.cpp
///       -L../../external_libs -L../../build/lib -L/opt/xilinx/xrt/lib
///       -lxrt_coreutil -laiebu -luuid -lpthread -o /tmp/verify_n10_delta_net
///   (one line; the continuations are dropped here because a backslash at the
///    end of a // comment splices the next line into it)
///
/// Run:
///   /tmp/verify_n10_delta_net [model_dir] [conv.xclbin] [GateDeltaNet.xclbin]
///
/// ---------------------------------------------------------------------------
/// What moved
/// ---------------------------------------------------------------------------
/// N9 put the 48 delta layers' six matmuls on `dequant_mm.xclbin` and left the
/// recurrence -- 0.8% of the block's MACs -- on the host. Measured afterwards,
/// on the `-s 0` prompt: `conv_silu` 243 ms and `delta_rule` 449 ms out of a
/// 5.95 s instrumented prefill window, i.e. **11.6% of prefill and 15.0% of the
/// 48 linear layers**. That is what this phase is spending, and it is the
/// ceiling on what it can win.
///
/// The two stages are exactly the two that MUTATE
/// (`cpu_ref::delta_net_layer`'s `conv_state_` and `state_`), so moving them
/// moves the layer's history onto the device and back every window. The host
/// stays authoritative -- decode reaches the same block at L = 1 on the CPU
/// path and speculation's save_state/load_state read those two vectors -- so
/// `delta_net_op::run()` uploads S, folds, and downloads S, once per layer per
/// window. 3 MB each way, 48 layers.
///
/// ---------------------------------------------------------------------------
/// The finding this file exists to protect
/// ---------------------------------------------------------------------------
/// **Both bitstreams scale q by 1/sqrt(head_dim).** `conv1d_prefill`'s NORM_Q
/// mode folds `LINEAR_ATTN_SCALE = 0.08838835` into q's l2norm, and the
/// delta-net kernel multiplies q by its own `inv_sqrt_DH` (its host feeds it q
/// l2-normalised with NO scale). Chained at their defaults, q comes out scaled
/// by 1/128 rather than 1/sqrt(128).
///
/// So the engine's conv sequence drives the q columns in NORM_K -- l2norm,
/// scale 1.0 -- and the fold stays the only place the scale is applied.
/// A cosine cannot see the mistake: a uniform scale on q passes through the
/// fold linearly and then through a per-head RMSNorm that divides most of it
/// straight back out. Check 3 is the only thing here that catches it, and it
/// catches it by measuring the device q band's per-head L2 norm and demanding
/// **1.0, not 0.0884**.
///
/// ---------------------------------------------------------------------------
/// How the two dispatches are scored
/// ---------------------------------------------------------------------------
/// Each against its OWN input, not against the pair's
/// (memory: score-a-projection-against-its-own-input). `delta_net_op` fuses
/// them -- the conv's output never reaches the host on the live path -- so it
/// exposes `conv_out()` for this driver alone.
///
/// Inputs are real: a real layer's `qkv_proj` driven by the dumped hidden
/// states, never N(0,1). A random q/k makes a delta-rule test pass on nothing
/// (memory: random-kv-cache-cannot-test-attention).
///
/// Every score is paired with a control that must COLLAPSE, because a cosine
/// near 1 is evidence about arithmetic and says nothing about addressing:
///
///   q band L2 == 1        the double-scale above. Not a cosine at all.
///   cross-head            device head h against reference head h+1. A
///                         consistent permutation agrees with itself.
///                         (memory: cross-head-control-catches-gqa-mapping)
///   alpha <-> beta        decay and write-strength are both per-head scalars
///                         in (0,1) on one BD; swapping them is the single
///                         most plausible way to mis-wire that buffer.
///   S_out != S_in         proves the dispatch moved the state at all.
///
/// And two that are about the fold's defining property rather than its
/// arithmetic: one dispatch of 1152 tokens must equal nine of 128 (the op
/// chunks internally at 1024, so this exercises `L_start` as well), and the
/// same window run twice from the same S must be bit-identical -- the recurring
/// failure on this hardware is a design that is right once and parks or drifts
/// on dispatch 2 (memory: one-shot-bench-cannot-see-dispatch-state).
///
/// What this driver does NOT cover is the 64-layer stack; `test/qwen3_8mtp_npu`
/// is what tests the seam, read off a binary built at the engine's own flags.
/// (memory: host-vs-device-ab-needs-shipping-flags) The `FLM_Q38_DELTA_NET=0`
/// A/B is gone with the host arm it switched to; its measurements are in
/// REPLACEMENTWORK.md.
#include "prefill/npu/delta_net_op.hpp"
#include "common/cpu_ref_delta_net_layer.hpp"
#include "common/cpu_ref_common.hpp"
#include "verify/npu/q4nx.hpp"

#include "npu_utils/npu_utils.hpp"

#include <algorithm>
#include <array>
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

/// The bar for a cosine against a same-inputs reference.
///
/// Both designs' own out.log measured better than this on their own harnesses:
/// the conv at 0.999785 overall, the fold at 0.999925 per-head on `o` and
/// 1.000000 on `S`. Everything below is bf16 end to end -- the conv taps, the
/// l2norm, the fold's operands and its readback -- so relative L2 lands near
/// 2e-2 where the cosine is at 4 nines, and it is the cosine that is the gate.
/// Relative L2 is printed beside every score because it is the number that
/// moves when something is wrong by a gain rather than by a rotation.
constexpr double kGate = 0.999;

/// The ceiling a control must fall under. Generous on purpose -- a control at
/// 0.98 has still told us the permutation is wrong.
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

void report_control(const char* tag, double cos) {
    char buf[180];
    std::snprintf(buf, sizeof buf, "cos %.6f  (must be < %.3f)", cos, kControl);
    report(tag, cos < kControl, buf);
}

/// One band of a `[L][stride]` matrix as its own contiguous matrix, so a fused
/// score cannot hide a narrow tensor inside a wide one.
/// (memory: fused-matmul-cosine-hides-narrow-tensors)
std::vector<float> band(const float* src, int L, int stride, int off, int w) {
    std::vector<float> v((size_t)L * w);
    for (int l = 0; l < L; l++)
        std::memcpy(v.data() + (size_t)l * w, src + (size_t)l * stride + off,
                    (size_t)w * sizeof(float));
    return v;
}

std::vector<float> band_bf16(const bf16* src, int L, int stride, int off, int w) {
    std::vector<float> v((size_t)L * w);
    for (int l = 0; l < L; l++)
        for (int j = 0; j < w; j++)
            v[(size_t)l * w + j] = (float)src[(size_t)l * stride + off + j];
    return v;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

/// What `conv.xclbin` computes, in fp32: the layer's own causal conv + SiLU,
/// then the per-head l2norm on the q and k bands **with no extra scale**. The
/// v band is left alone.
///
/// `conv_silu` is the reference's own code and advances `c`'s window, so this
/// doubles as the check-4 reference.
void conv_reference(cpu_ref::delta_net_layer& c, std::vector<float>& qkv, int L) {
    const cpu_ref::delta_net_geometry& g = c.geometry();
    const int CD = g.conv_dim(), DQ = g.dq(), DKH = g.k_head_dim;
    c.conv_silu(qkv.data(), L);
    for (int l = 0; l < L; l++) {
        float* row = qkv.data() + (size_t)l * CD;
        for (int h = 0; h < g.qk_heads; h++) {
            cpu_ref::l2_norm(row + h * DKH,      DKH, row + h * DKH,      1.0f, g.l2_eps);
            cpu_ref::l2_norm(row + DQ + h * DKH, DKH, row + DQ + h * DKH, 1.0f, g.l2_eps);
        }
    }
}

/// Fill the op's conv input BO from an fp32 `[L][conv_dim]` matrix. Rows below
/// `qkv_in_row()` are the history and belong to `run()`.
void stage_qkv(q38npu::delta_net_op& dn, const std::vector<float>& qkv,
               int L, int CD) {
    buffer<bf16>& x = dn.qkv_in(L);
    bf16* d = x.data() + (size_t)dn.qkv_in_row() * CD;
    for (size_t i = 0; i < (size_t)L * CD; i++) d[i] = (bf16)qkv[i];
}

} // namespace

// ==========================================================================
int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string conv_xclbin =
        argc > 2 ? argv[2]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "conv1d_prefill/build/xclbins/conv.xclbin";
    const std::string dn_xclbin =
        argc > 3 ? argv[3]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "gate_delta_net_prefill/build/xclbins/GateDeltaNet_prefill.xclbin";

    cpu_ref::delta_net_geometry geo;          // defaults are Qwen3.8-27B
    geo.validate();

    const int D   = geo.model_dim;
    const int CD  = geo.conv_dim();
    const int DV  = geo.dv();
    const int DQ  = geo.dq(), DK = geo.dk();
    const int NV  = geo.v_heads;
    const int DKH = geo.k_head_dim, DVH = geo.v_head_dim;

    std::printf("== N10: the delta-net recurrence on conv.xclbin + "
                "GateDeltaNet_prefill.xclbin ==\n");
    std::printf("   model  %s\n   conv   %s\n   fold   %s\n   layer  %d  (%s)\n\n",
                model_dir.c_str(), conv_xclbin.c_str(), dn_xclbin.c_str(),
                kLayer, geo.str().c_str());

    // ======================================================================
    // 1. the geometry guard
    // ======================================================================
    // Everything the two sequences assume, restated here rather than trusted
    // from `delta_net_op`'s constructor -- that one throws at load, so a
    // violated precondition would take the engine down with one message and
    // nothing saying which condition it was.
    //
    // Each is silent if it fails. A column that straddles the q/k split takes
    // one norm mode for channels that need two; a head width the fold was not
    // built for reads past its own buffers; a ragged L emits a short sequence
    // and the dispatch parks in wait().
    {
        std::string bad;
        auto want = [&](bool ok, const std::string& why) {
            if (!ok) bad += (bad.empty() ? "" : ", ") + why;
        };
        want(CD % q38npu::conv_seq::COL_CHANNELS == 0,
             "conv_dim " + std::to_string(CD) + " is not a whole conv column");
        want(DQ % q38npu::conv_seq::COL_CHANNELS == 0,
             "dq is not a whole conv column -- a column would straddle q/k");
        want(DQ == DK, "DQ != DK, which the fold's sequence asserts");
        want(DKH == 128 && DVH == 128,
             "the fold bitstream is built for 128-wide heads");
        want(NV % 8 == 0, "v_heads does not fill the array's 8 columns");
        want(geo.expand() == NV / geo.qk_heads,
             "QK_HEAD_GROUP is not v_heads / qk_heads");
        want(q38npu::delta_net_op::CHUNK % q38npu::delta_net_op::MT_L == 0,
             "CHUNK is not a whole number of memtile chunks");

        char buf[240];
        std::snprintf(buf, sizeof buf,
                      "conv_dim %d = %d cols; q|k|v = %d|%d|%d; %d heads x "
                      "%dx%d; max window %d",
                      CD, CD / (int)q38npu::conv_seq::COL_CHANNELS, DQ, DK, DV,
                      NV, DKH, DVH, q38npu::delta_net_op::MAX_WINDOW);
        report("both sequences' geometry is legal", bad.empty(),
               bad.empty() ? buf : bad);
    }

    // ======================================================================
    // the device and the inputs
    // ======================================================================
    cpu_ref::safetensors q4nx(model_dir + "/model.q4nx");
    cpu_ref::safetensors ref(model_dir + "/ref.safetensors");

    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager*   conv_mgr = mgr.register_xclbin(conv_xclbin);
    npu_app_manager*   dn_mgr   = mgr.register_xclbin(dn_xclbin);

    q38npu::delta_net_op dn(conv_mgr->create_app(), dn_mgr, geo,
                            q38npu::conv1d_rtp_addresses{});

    // The conv weights, straight out of the layer's own reference.
    cpu_ref::delta_net_layer w_src(q4nx, geo, kLayer);
    npu_app                  stage_app = conv_mgr->create_app();
    q38npu::npu_delta_ssm_weights ssm(stage_app, w_src);

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
    const int P = (int)(x_all.size() / D);          // 221
    for (float& f : x_all) f = cpu_ref::round_bf16(f);

    // The reference layer has already resolved these, so take them from it
    // rather than naming the checkpoint's keys a second time here.
    const cpu_ref::q4nx_tensor& W_qkv = w_src.weights().qkv;
    const cpu_ref::q4nx_tensor& W_z   = w_src.weights().z;
    const cpu_ref::q4nx_tensor& W_out = w_src.weights().out;

    // The conv's input: the real qkv projection of the real hidden states,
    // through the same quantised bytes the device path would use. Q4_K's own
    // loss is upstream of everything measured here and is common mode.
    std::vector<float> xn;
    w_src.in_norm(x_all.data(), P, xn);
    for (float& f : xn) f = cpu_ref::round_bf16(f);

    std::vector<float> qkv_raw((size_t)P * CD), z_raw((size_t)P * DV);
    cpu_ref::q4nx_matmul(W_qkv, xn.data(), P, qkv_raw.data());
    cpu_ref::q4nx_matmul(W_z,   xn.data(), P, z_raw.data());
    for (float& f : qkv_raw) f = cpu_ref::round_bf16(f);

    std::vector<float> aa, bb, alpha, beta;
    w_src.alpha_beta(xn.data(), P, aa, bb);
    w_src.gates(aa.data(), bb.data(), P, alpha, beta);

    std::printf("   window %d rows from %s\n\n", P, key);
    std::printf("-- the conv dispatch, against its own input --\n");

    // ======================================================================
    // 2. the conv, per band
    // ======================================================================
    // A fresh layer on each side, so both start from a zero conv window.
    // `dn.run()` does both dispatches; `conv_out()` is how the first one is
    // charged for its own error rather than for the pair's.
    std::vector<float> conv_dev;               // [P][CD], kept for check 3
    {
        cpu_ref::delta_net_layer dev_l(q4nx, geo, kLayer);
        cpu_ref::delta_net_layer cpu_l(q4nx, geo, kLayer);
        dev_l.reset(); cpu_l.reset();

        stage_qkv(dn, qkv_raw, P, CD);
        std::vector<float> core;
        dn.run(&dev_l, P, ssm.conv_w(), alpha.data(), beta.data(), core);

        buffer<bf16>& co = dn.conv_out();
        co.sync_from_device();
        conv_dev.assign((size_t)P * CD, 0.0f);
        for (size_t i = 0; i < conv_dev.size(); i++) conv_dev[i] = (float)co.data()[i];

        std::vector<float> want = qkv_raw;
        conv_reference(cpu_l, want, P);

        report_score("conv+l2norm: the q band",
                     band(conv_dev.data(), P, CD, 0, DQ),
                     band(want.data(), P, CD, 0, DQ), "(16 heads)");
        report_score("conv+l2norm: the k band",
                     band(conv_dev.data(), P, CD, DQ, DK),
                     band(want.data(), P, CD, DQ, DK), "(16 heads)");
        report_score("conv+silu: the v band",
                     band(conv_dev.data(), P, CD, DQ + DK, DV),
                     band(want.data(), P, CD, DQ + DK, DV), "(48 heads, no norm)");

        // ------------------------------------------------------------------
        // 4. the conv window the device advanced
        // ------------------------------------------------------------------
        // `delta_net_op` takes it out of the x BO -- the rows the device
        // actually convolved -- rather than from the caller's fp32 copy, so
        // this should be exact and not merely close. Both sides hold the same
        // bf16-rounded values in fp32.
        const std::vector<float>& gw = dev_l.conv_window();
        const std::vector<float>& rw = cpu_l.conv_window();
        size_t diff = 0;
        for (size_t i = 0; i < gw.size(); i++) if (gw[i] != rw[i]) diff++;
        char buf[220];
        std::snprintf(buf, sizeof buf,
                      "%zu of %zu taps differ (rel L2 %.2e); seen %d vs %d",
                      diff, gw.size(),
                      rel_l2(gw.data(), rw.data(), gw.size()),
                      dev_l.tokens_seen(), cpu_l.tokens_seen());
        report("the device advanced conv_state_ exactly", diff == 0, buf);
    }

    // ======================================================================
    // 3. the q scale -- the one check a cosine cannot make
    // ======================================================================
    // See the file header. If the conv is left in NORM_Q, q arrives at the fold
    // scaled by 1/sqrt(128) and the fold scales it again: every per-head norm
    // below reads 0.0884 instead of 1.0, the block's output moves by a uniform
    // gain that the downstream per-head RMSNorm mostly divides back out, and
    // every cosine in this file still passes.
    {
        auto head_norms = [&](int off, int heads) {
            double lo = 1e30, hi = 0, sum = 0;
            for (int l = 0; l < P; l++)
                for (int h = 0; h < heads; h++) {
                    const float* v = conv_dev.data() + (size_t)l * CD + off + h * DKH;
                    double acc = 0;
                    for (int j = 0; j < DKH; j++) acc += (double)v[j] * v[j];
                    const double n = std::sqrt(acc);
                    lo = std::min(lo, n); hi = std::max(hi, n); sum += n;
                }
            return std::array<double,3>{sum / (P * (double)heads), lo, hi};
        };
        const auto q = head_norms(0,  geo.qk_heads);
        const auto k = head_norms(DQ, geo.qk_heads);
        char buf[240];
        std::snprintf(buf, sizeof buf,
                      "q %.6f [%.4f, %.4f], k %.6f -- 1/sqrt(128) = %.6f would "
                      "mean the scale is applied twice",
                      q[0], q[1], q[2], k[0], geo.qk_scale());
        report("the device q band is l2-normalised to 1, not scaled",
               std::fabs(q[0] - 1.0) < 0.01 && std::fabs(k[0] - 1.0) < 0.01, buf);
    }

    // ======================================================================
    // the fold, from a warm state
    // ======================================================================
    // A fold scored from S = 0 is a fold scored on one rank-1 update. Warm both
    // sides with the same 96 tokens through the reference's own stages, then
    // measure the next 125.
    std::printf("\n-- the fold dispatch, from a warm state --\n");

    constexpr int WARM = 96;
    const int T = P - WARM;                      // 125 measured tokens

    cpu_ref::delta_net_layer dev_l(q4nx, geo, kLayer);
    cpu_ref::delta_net_layer cpu_l(q4nx, geo, kLayer);
    dev_l.reset(); cpu_l.reset();
    {
        std::vector<float> w1 = qkv_raw, w2 = qkv_raw, c1, c2;
        dev_l.conv_silu(w1.data(), WARM);
        dev_l.delta_rule(w1.data(), alpha.data(), beta.data(), WARM, c1);
        dev_l.advance(WARM);
        cpu_l.conv_silu(w2.data(), WARM);
        cpu_l.delta_rule(w2.data(), alpha.data(), beta.data(), WARM, c2);
        cpu_l.advance(WARM);
    }

    // The snapshot both the control and check 8 restart from, plus a typed copy
    // of just S for the "did it move" control.
    std::vector<uint8_t> warm_snap(dev_l.state_bytes());
    dev_l.save_state(warm_snap.data());
    const std::vector<float> warm_S = dev_l.recurrent_state();

    const float* qkv_t   = qkv_raw.data() + (size_t)WARM * CD;
    const float* alpha_t = alpha.data()   + (size_t)WARM * NV;
    const float* beta_t  = beta.data()    + (size_t)WARM * NV;

    std::vector<float> core_dev, core_ref;
    {
        std::vector<float> tail((size_t)T * CD);
        std::memcpy(tail.data(), qkv_t, tail.size() * sizeof(float));

        stage_qkv(dn, tail, T, CD);
        dn.run(&dev_l, T, ssm.conv_w(), alpha_t, beta_t, core_dev);
        dev_l.advance(T);

        std::vector<float> ref_in = tail;
        cpu_l.conv_silu(ref_in.data(), T);
        cpu_l.delta_rule(ref_in.data(), alpha_t, beta_t, T, core_ref);
        cpu_l.advance(T);

        report_score("the fold's output o", core_dev, core_ref,
                     ("(" + std::to_string(T) + " tokens x 48 heads)").c_str());
        report_score("the recurrent state S it left behind",
                     dev_l.recurrent_state(), cpu_l.recurrent_state(),
                     "(48 x 128 x 128)");

        // ---- control: cross-head ----------------------------------------
        // A GQA-shaped mapping error -- q/k head gq = h / 3 feeding the wrong
        // v head -- is consistent with itself and agrees with its own
        // reference. Only a rotated reference separates them.
        std::vector<float> rot((size_t)T * DV);
        for (int l = 0; l < T; l++)
            for (int h = 0; h < NV; h++)
                std::memcpy(rot.data() + (size_t)l * DV + h * DVH,
                            core_ref.data() + (size_t)l * DV + ((h + 1) % NV) * DVH,
                            DVH * sizeof(float));
        report_control("control: o against the next head's o",
                       cosine(core_dev.data(), rot.data(), core_dev.size()));

        // ---- control: alpha and beta swapped -----------------------------
        // Both are [L][48] fp32 in one buffer on one BD, told apart only by an
        // offset. Swapping them is the single most plausible way to mis-wire
        // it, and both are in (0, 1) so nothing blows up to give it away.
        {
            cpu_ref::delta_net_layer swp(q4nx, geo, kLayer);
            swp.load_state(warm_snap.data());
            std::vector<float> s_in = tail, s_core;
            swp.conv_silu(s_in.data(), T);
            swp.delta_rule(s_in.data(), beta_t, alpha_t, T, s_core);
            report_control("control: alpha and beta swapped",
                           cosine(core_dev.data(), s_core.data(), s_core.size()));
        }

        // ---- control: the state actually moved ---------------------------
        {
            const std::vector<float>& after = dev_l.recurrent_state();
            const bool moved = std::memcmp(after.data(), warm_S.data(),
                                           warm_S.size() * sizeof(float)) != 0;
            const double c = cosine(after.data(), warm_S.data(), warm_S.size());
            char buf[200];
            std::snprintf(buf, sizeof buf,
                          "S_out %s S_in (cos %.6f over %d tokens)",
                          moved ? "differs from" : "is IDENTICAL to", c, T);
            report("control: the dispatch moved S at all", moved, buf);
        }
    }

    // ======================================================================
    // 8. the same window twice
    // ======================================================================
    // The recurring failure on this hardware is a design that is correct on
    // dispatch 1 and parks or drifts on dispatch 2 -- an MM2S chain running on
    // its initial lock credit, a BD repeat leaving chunks queued. A one-shot
    // driver cannot see it. Same S_in, same rows, twice: bit-identical or the
    // design has state it should not have.
    // (memory: unrouted-mm2s-chain-still-fires,
    //          one-shot-bench-cannot-see-dispatch-state)
    {
        cpu_ref::delta_net_layer a(q4nx, geo, kLayer), b(q4nx, geo, kLayer);
        a.load_state(warm_snap.data());
        b.load_state(warm_snap.data());

        std::vector<float> tail((size_t)T * CD), c1, c2;
        std::memcpy(tail.data(), qkv_t, tail.size() * sizeof(float));

        stage_qkv(dn, tail, T, CD);
        dn.run(&a, T, ssm.conv_w(), alpha_t, beta_t, c1);
        stage_qkv(dn, tail, T, CD);
        dn.run(&b, T, ssm.conv_w(), alpha_t, beta_t, c2);

        const bool same_o = std::memcmp(c1.data(), c2.data(),
                                        c1.size() * sizeof(float)) == 0;
        const bool same_s = std::memcmp(a.recurrent_state().data(),
                                        b.recurrent_state().data(),
                                        a.recurrent_state().size() * sizeof(float)) == 0;
        char buf[220];
        std::snprintf(buf, sizeof buf,
                      "o %s, S %s (o rel L2 %.2e, S rel L2 %.2e)",
                      same_o ? "bit-identical" : "DIFFERS",
                      same_s ? "bit-identical" : "DIFFERS",
                      rel_l2(c1.data(), c2.data(), c1.size()),
                      rel_l2(a.recurrent_state().data(), b.recurrent_state().data(),
                             a.recurrent_state().size()));
        report("dispatch 2 == dispatch 1 from the same S_in", same_o && same_s, buf);
    }

    // ======================================================================
    // 9. chunking: one dispatch of 1152 == nine of 128
    // ======================================================================
    // The op chunks internally at CHUNK = 1024, so 1152 tokens is two chunks
    // and exercises the sequence's `L_start` -- the offset that makes a long
    // window chunkable, and the one thing about the fold's addressing that the
    // 221-row window above never touches.
    //
    // The rows are the real window tiled with a per-copy gain, which keeps the
    // dynamic range the conv taps and the l2norm were fitted to. The fold does
    // not care that the tokens repeat; what is being measured is that the
    // decomposition is exact, and separately that both agree with the CPU fold
    // over the same 1152 tokens.
    {
        constexpr int LONG = 1152;
        std::vector<float> q_long((size_t)LONG * CD), a_long((size_t)LONG * NV),
                           b_long((size_t)LONG * NV);
        for (int l = 0; l < LONG; l++) {
            const int   src  = l % P;
            const float gain = 1.0f + 0.05f * (float)(l / P);
            for (int c = 0; c < CD; c++)
                q_long[(size_t)l * CD + c] =
                    cpu_ref::round_bf16(qkv_raw[(size_t)src * CD + c] * gain);
            std::memcpy(a_long.data() + (size_t)l * NV,
                        alpha.data() + (size_t)src * NV, NV * sizeof(float));
            std::memcpy(b_long.data() + (size_t)l * NV,
                        beta.data()  + (size_t)src * NV, NV * sizeof(float));
        }

        cpu_ref::delta_net_layer one(q4nx, geo, kLayer),
                                 many(q4nx, geo, kLayer), cpu(q4nx, geo, kLayer);
        one.load_state(warm_snap.data());
        many.load_state(warm_snap.data());
        cpu.load_state(warm_snap.data());

        std::vector<float> o_one, o_many((size_t)LONG * DV), o_cpu;

        stage_qkv(dn, q_long, LONG, CD);
        dn.run(&one, LONG, ssm.conv_w(), a_long.data(), b_long.data(), o_one);

        constexpr int STEP = 128;
        for (int t = 0; t < LONG; t += STEP) {
            std::vector<float> piece((size_t)STEP * CD);
            std::memcpy(piece.data(), q_long.data() + (size_t)t * CD,
                        piece.size() * sizeof(float));
            std::vector<float> op;
            stage_qkv(dn, piece, STEP, CD);
            dn.run(&many, STEP, ssm.conv_w(),
                   a_long.data() + (size_t)t * NV, b_long.data() + (size_t)t * NV, op);
            std::memcpy(o_many.data() + (size_t)t * DV, op.data(),
                        op.size() * sizeof(float));
        }

        std::vector<float> ref_in = q_long;
        cpu.conv_silu(ref_in.data(), LONG);
        cpu.delta_rule(ref_in.data(), a_long.data(), b_long.data(), LONG, o_cpu);

        const double e  = rel_l2(o_one.data(), o_many.data(), o_one.size());
        const double es = rel_l2(one.recurrent_state().data(),
                                 many.recurrent_state().data(),
                                 one.recurrent_state().size());
        const bool bitwise = std::memcmp(o_one.data(), o_many.data(),
                                         o_one.size() * sizeof(float)) == 0;
        char buf[240];
        std::snprintf(buf, sizeof buf,
                      "o rel L2 %.2e%s, S rel L2 %.2e (%d tokens: 2 chunks vs %d "
                      "dispatches)", e, bitwise ? " (bit-identical)" : "", es,
                      LONG, LONG / STEP);
        report("1 dispatch of 1152 == 9 of 128", e < 1e-6 && es < 1e-6, buf);

        report_score("the 1152-token fold against the CPU fold", o_one, o_cpu,
                     "(2 chunks, L_start 0 and 1024)");
        report_score("and the S it left behind", one.recurrent_state(),
                     cpu.recurrent_state(), "");
    }

    // ======================================================================
    // 10. the whole block, device recurrence vs host recurrence
    // ======================================================================
    // Everything else held equal: both sides run the reference's own norms and
    // the same CPU matmuls, and differ only in where `conv_silu` + `delta_rule`
    // ran. So this is N10's error and nothing else's -- it does not carry N9's
    // bf16 projections and cannot be improved by them.
    //
    // The error is expected to be larger than the fold's own, and to compound:
    // the recurrence is downstream of a bf16 conv and its state persists, so a
    // window's error is the sum of every token's before it.
    {
        auto run_block = [&](cpu_ref::delta_net_layer& l, bool device,
                             std::vector<float>& y) {
            std::vector<float> n1, core;
            l.in_norm(x_all.data(), P, n1);
            for (float& f : n1) f = cpu_ref::round_bf16(f);

            std::vector<float> qk((size_t)P * CD), zg((size_t)P * DV);
            cpu_ref::q4nx_matmul(W_qkv, n1.data(), P, qk.data());
            cpu_ref::q4nx_matmul(W_z,   n1.data(), P, zg.data());
            for (float& f : qk) f = cpu_ref::round_bf16(f);

            std::vector<float> g1, g2, al, be;
            l.alpha_beta(n1.data(), P, g1, g2);
            l.gates(g1.data(), g2.data(), P, al, be);

            if (device) {
                stage_qkv(dn, qk, P, CD);
                dn.run(&l, P, ssm.conv_w(), al.data(), be.data(), core);
            } else {
                l.conv_silu(qk.data(), P);
                l.delta_rule(qk.data(), al.data(), be.data(), P, core);
            }
            l.gated_rmsnorm(core.data(), zg.data(), P);
            for (float& f : core) f = cpu_ref::round_bf16(f);

            y.assign((size_t)P * D, 0.0f);
            cpu_ref::q4nx_matmul(W_out, core.data(), P, y.data());
            for (size_t i = 0; i < y.size(); i++) y[i] += x_all[i];
            l.advance(P);
        };

        cpu_ref::delta_net_layer a(q4nx, geo, kLayer), b(q4nx, geo, kLayer);
        a.reset(); b.reset();
        std::vector<float> y_dev, y_host;
        run_block(a, true,  y_dev);
        run_block(b, false, y_host);

        std::printf("\n-- the mixer half of the block, both paths --\n");
        report_score("h1 (mixer out + residual), device recurrence", y_dev, y_host,
                     "(N10's error alone; N9's bf16 projections not in play)");
        report_score("the block's S after the window", a.recurrent_state(),
                     b.recurrent_state(), "");
    }

    // ======================================================================
    // 11. throughput
    // ======================================================================
    // Not a gate -- the number this phase exists to produce. Both columns are
    // ONE layer's two stages over L tokens; the engine pays this 48 times per
    // prefill window. The device column includes the 3 MB S upload and the
    // 3 MB download, which is the fixed cost the host path does not have and
    // the reason the crossover is not at L = 1.
    {
        std::printf("\n-- one layer's conv + fold, ms --\n");
        std::printf("   %6s  %10s  %10s  %9s  %9s\n",
                    "L", "NPU", "CPU", "NPU us/tok", "speedup");
        for (int L : {128, 256, 512, 1024, 2048}) {
            std::vector<float> qk((size_t)L * CD), al((size_t)L * NV),
                               be((size_t)L * NV);
            for (int l = 0; l < L; l++) {
                const int src = l % P;
                std::memcpy(qk.data() + (size_t)l * CD,
                            qkv_raw.data() + (size_t)src * CD, CD * sizeof(float));
                std::memcpy(al.data() + (size_t)l * NV,
                            alpha.data() + (size_t)src * NV, NV * sizeof(float));
                std::memcpy(be.data() + (size_t)l * NV,
                            beta.data() + (size_t)src * NV, NV * sizeof(float));
            }

            cpu_ref::delta_net_layer d1(q4nx, geo, kLayer), h1(q4nx, geo, kLayer);
            d1.reset(); h1.reset();
            std::vector<float> c1, c2;

            stage_qkv(dn, qk, L, CD);
            auto t0 = std::chrono::steady_clock::now();
            dn.run(&d1, L, ssm.conv_w(), al.data(), be.data(), c1);
            const double npu = ms_since(t0);

            std::vector<float> host_in = qk;
            auto t1 = std::chrono::steady_clock::now();
            h1.conv_silu(host_in.data(), L);
            h1.delta_rule(host_in.data(), al.data(), be.data(), L, c2);
            const double cpu = ms_since(t1);

            std::printf("   %6d  %10.2f  %10.2f  %9.1f  %8.2fx\n",
                        L, npu, cpu, npu * 1000.0 / L, cpu / npu);
        }
    }

    // ======================================================================
    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
