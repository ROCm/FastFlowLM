// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file verify_n9_delta_proj.cpp
/// \brief N9 -- the 48 gated-delta-net layers' projections on
///        `dequant_mm.xclbin`, with the recurrence left on the host.
///
/// Build:
///   source ~/setup.sh
///   g++-13 -O2 -march=native -fopenmp -std=c++20
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n9_delta_proj.cpp common/cpu_ref_delta_net_layer.cpp
///       speculate/npu/proj_sequence.cpp
///       ../q4nx/q4_npu_eXpress.cpp ../q4nx/safe_tensors.cpp
///       -L../../../../src/lib/xrt -L../../build/lib -L/opt/xilinx/xrt/lib
///       -lxrt_coreutil -laiebu -luuid -lpthread -o /tmp/verify_n9_delta_proj
///   (one line; the continuations are dropped here because a backslash at the
///    end of a // comment splices the next line into it)
///
/// Takes `dequant_mm.xclbin` and the model dir. One delta-net layer is staged
/// (211 MB) and the CPU builds a same-bytes reference for each of its 383 M
/// parameters several times over.
///
/// ---------------------------------------------------------------------------
/// What moved
/// ---------------------------------------------------------------------------
/// N8 put the 16 full-attention layers' seven projections on dequant_mm and
/// took prefill from 54.3 s to 42.1 s. What was left was almost entirely the
/// OTHER 48 layers, which were wholly on the host. Their six Q4_K matmuls are
/// 382.8 M MACs per token per layer -- 18.4 G/token across the stack, against
/// 5.96 G for all 16 attention layers -- and the recurrence they wrap is 3.1 M,
/// **0.8% of the layer**. So the six matmuls move and the sequential fold
/// stays, which is both what was asked for and what the arithmetic wants.
///
/// Six matmuls, FOUR dispatches: `up|gate` is one woven matrix and `qkv|z` one
/// adjacent pair. Everything else -- both norms, the depthwise conv, the gates,
/// the delta rule, the gated RMSNorm, both residuals -- is
/// `cpu_ref::delta_net_layer`'s own code called by name.
///
/// The weight BO is NEW here, and that is the one structural difference from
/// N8. An attention layer already had a staged 209 MB BO for its core to read;
/// a delta layer had no device presence at all. So `delta_layout` +
/// `stage_delta_proj_weights` build one, in **`qwen3_5vl_npu`'s linear weight
/// order** (`up|gate, down, qkv, z, ssm_out`) rather than this engine's
/// attention order -- see delta_weights.hpp on why that costs nothing and buys
/// something. 48 of them is 10.62 GB resident, which is legal because XRT's
/// 4 GB limit is per-BO. (memory: xrt-bo-4gb-addressing-limit)
///
/// ---------------------------------------------------------------------------
/// How the dispatches are scored
/// ---------------------------------------------------------------------------
/// Against a reference built from the SAME QUANTIZED BYTES, through
/// `cpu_ref::q4nx_matmul` on a bf16-rounded x. Q4_K's own ~0.997 loss is then
/// common mode and cancels, so the bar is 0.999 and what is left to measure is
/// the device's arithmetic: `aie::mmul<8,8,8,bf16>` floors both operands onto a
/// shared-exponent-per-8 grid, and every fp32 -> bf16 readback truncates toward
/// -inf. (memory: score-dequant-mm-against-its-own-quantized-bytes,
///                aie-mmul-bf16-truncates-to-bfp16)
///
/// Every score is paired with a control that must COLLAPSE, because a cosine
/// near 1 is evidence about arithmetic and says nothing about addressing:
///
///   rotate by 64        one core column. Catches an off-by-one core mapping.
///   z | qkv instead     both are legal 16384-row matrices over the same
///   of qkv | z          bytes and the op cannot tell them apart.
///   concat instead      up|gate is WOVEN 512 output rows at a time.
///   of weave
///   Din = D for         ssm_out reduces over the gated delta row (6144), not
///   ssm_out             over D. Same byte count, strided wrong -- which is
///                       exactly the trap `reorder_cpy`'s `col` carries, and
///                       this file is the only place it is measured.
///   silu(up) * gate     just as plausible a reading of a SwiGLU.
///                       (memory: score-swiglu-by-swapping-up-and-gate)
///
/// ---------------------------------------------------------------------------
/// The two halves
/// ---------------------------------------------------------------------------
///   checks 1-8     the operator, driven directly. Geometry, the weight-layout
///                  memcmp, the four dispatches with their controls, and the
///                  host-side tiling.
///   checks 9-12    the composition: one hybrid layer against
///                  `cpu_ref::delta_net_layer::forward()`, what the bf16
///                  projections did to the conv window and the recurrent state,
///                  the L = 1 / L = n equivalence that is the delta rule's
///                  defining property, and a sabotage that must break check 9.
///   check 13       throughput. Not a gate; the number this phase exists for.
///
/// **Check 11 is the one that has no counterpart in N8.** An attention layer is
/// row-parallel: a wrong cursor shows up immediately. A delta layer is a fold
/// with no inverse, so `conv_silu`, `delta_rule` and `advance` must each run
/// exactly once per window and in order. Running the same tokens one at a time
/// and all at once is the only check that sees a double-advance, and it sees it
/// as a large error rather than a small one.
///
/// What this driver does NOT cover is the 64-layer stack; `test/qwen3_8mtp_npu`
/// is what tests the seam, read off a binary built at the engine's own flags.
/// (memory: host-vs-device-ab-needs-shipping-flags) The `FLM_Q38_DELTA_PROJ=0`
/// A/B is gone with the host arm it switched to; its measurements are in
/// REPLACEMENTWORK.md.
#include "prefill/npu/delta_proj.hpp"
#include "prefill/npu/delta_weights.hpp"
#include "verify/npu/q4nx.hpp"
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

/// A gated-delta-net layer. The full-attention ones are (i + 1) % 4 == 0, i.e.
/// 3, 7, ... 63, so 4 is the first delta layer whose input is a dumped
/// `layer_XX_out` rather than the embedding.
constexpr int kLayer = 4;

/// The bar for a same-bytes score. See the file header: Q4_K cancels, so what
/// is left is the bfp16 mmul and the bf16 readback.
constexpr double kGate = 0.999;

/// The ceiling a control must fall under. Generous on purpose -- a control that
/// lands at 0.98 has still told us the permutation is wrong.
constexpr double kControl = 0.99;

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-52s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
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

double report_cos(const char* tag, double cos, double gate,
                  const char* note = "") {
    char buf[220];
    std::snprintf(buf, sizeof buf, "cos = %.8f  (gate %.4f) %s", cos, gate, note);
    report(tag, cos >= gate, buf);
    return cos;
}

void report_control(const char* tag, double cos) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "cos = %.6f  (must be < %.3f)", cos, kControl);
    report(tag, cos < kControl, buf);
}

/// Score `got` against `ref`, then against `ref` with its columns rotated by
/// one core column. The rotation is the negative control every dispatch gets.
void score_with_rotation(const char* tag, const char* ctl_tag,
                         const std::vector<float>& got,
                         const std::vector<float>& ref, int L, int N) {
    report_cos(tag, cosine(got.data(), ref.data(), got.size()), kGate);

    std::vector<float> rot(ref.size());
    for (int l = 0; l < L; l++)
        for (int n = 0; n < N; n++)
            rot[(size_t)l * N + n] = ref[(size_t)l * N + (n + 64) % N];
    report_control(ctl_tag, cosine(got.data(), rot.data(), got.size()));
}

/// A deterministic activation-shaped fill, for the checks that need more rows
/// than the dump has. Rounded to bf16 because that is what the device will
/// receive; letting the reference start from extra precision charges the device
/// for a narrowing it cannot avoid.
void fill_synth(std::vector<float>& v, uint64_t seed) {
    uint64_t s = seed | 1;
    for (float& f : v) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        f = cpu_ref::round_bf16(((float)(int32_t)(uint32_t)s / 2147483648.0f) * 0.5f);
    }
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

} // namespace

// ==========================================================================
int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string proj_xclbin =
        argc > 2 ? argv[2]
                 : "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/"
                   "dequant_mm/build/xclbins/dequant_mm.xclbin";

    cpu_ref::delta_net_geometry geo;          // defaults are Qwen3.8-27B
    geo.validate();

    const int D  = geo.model_dim;
    const int DI = geo.intermediate_size;
    const int CD = geo.conv_dim();
    const int DV = geo.dv();
    const int DQ = geo.dq(), DK = geo.dk();

    std::printf("== N9: the delta-net projections on dequant_mm ==\n");
    std::printf("   model  %s\n   proj   %s\n   layer  %d  (%s)\n\n",
                model_dir.c_str(), proj_xclbin.c_str(), kLayer,
                geo.str().c_str());

    const q38npu::delta_layout t = q38npu::delta_layout::build(geo);

    // ======================================================================
    // 1. the geometry guard
    // ======================================================================
    // Everything the four dispatches assume about the layout, restated here
    // rather than trusted from `delta_proj`'s constructor -- that one throws at
    // load, so a violated precondition would take the engine down with one
    // message and nothing saying which of the conditions it was.
    //
    // Each of these is silent if it fails at the wrong layer: a ragged Dout or
    // Din hangs the dispatch; a Din that is not a whole number of 256-wide
    // blocks, or a row count that is not a multiple of 64, makes reorder_cpy
    // fold the buffer's head back in as padding with nothing asserting.
    {
        struct { const char* name; int Dout, Din; } shapes[4] = {
            {"up|gate", 2 * DI,       D},
            {"down",    D,            DI},
            {"qkv|z",   t.QKVZ_OUT(), D},
            {"ssm_out", D,            DV},
        };
        std::string bad;
        auto want = [&](bool ok, const std::string& why) {
            if (!ok) bad += (bad.empty() ? "" : ", ") + why;
        };
        for (auto& s : shapes) {
            want(s.Dout % q38npu::dequant_mm_tiles::PASS_COLS == 0,
                 std::string(s.name) + " Dout " + std::to_string(s.Dout) +
                 " is not a whole array pass");
            want(s.Din % q38npu::dequant_mm_tiles::K_TILE == 0,
                 std::string(s.name) + " Din " + std::to_string(s.Din) +
                 " is not a whole reduction tile");
            want(s.Din % 256 == 0, std::string(s.name) +
                 " Din is not a whole q4_k column block");
        }
        // The 64-row locality is per TENSOR -- 2*DI passing says nothing about
        // up and gate individually, and QKVZ_OUT passing says nothing about k.
        const int rows[7] = {DI, D, DQ, DK, DV, DV, D};
        const char* rn[7] = {"up/gate", "down", "q", "k", "v", "z", "ssm_out"};
        for (int i = 0; i < 7; i++)
            want(rows[i] % q38npu::dequant_mm_tiles::N_TILE == 0,
                 std::string(rn[i]) + " is " + std::to_string(rows[i]) +
                 " rows, not a multiple of 64");
        // The fused dispatch reads across the qkv -> z boundary and stops where
        // ssm_out begins.
        want(t.w_z_offset ==
                 t.w_qkv_offset + q38npu::q4k_bytes((size_t)CD * D),
             "z does not immediately follow qkv");
        want(t.w_out_offset ==
                 t.w_z_offset + q38npu::q4k_bytes((size_t)DV * D),
             "ssm_out does not immediately follow z");
        // Every weight byte offset reaches the DMA, which folds a 1-byte
        // element type to u32 and shifts the offset right by two.
        const size_t offs[5] = {t.w_upgate_offset, t.w_down_offset,
                                t.w_qkv_offset, t.w_z_offset, t.w_out_offset};
        for (size_t o : offs)
            want(o % 4 == 0, "a weight offset is not a multiple of 4");
        want(DI % 512 == 0, "intermediate_size is not a whole weave band");

        char buf[240];
        std::snprintf(buf, sizeof buf,
                      "4 shapes, 7 row counts, 2 adjacencies, 5 offsets "
                      "(proj_bytes %.1f MB; 48 layers = %.2f GB)",
                      t.proj_bytes / 1048576.0,
                      48.0 * t.proj_bytes / 1073741824.0);
        report("every dispatch's geometry is legal", bad.empty(),
               bad.empty() ? buf : bad);
    }

    // ======================================================================
    // the device
    // ======================================================================
    cpu_ref::safetensors q4nx(model_dir + "/model.q4nx");
    cpu_ref::safetensors ref(model_dir + "/ref.safetensors");

    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager*   pm = mgr.register_xclbin(proj_xclbin);

    std::printf("-- staging one delta-net layer (%.0f MB) --\n",
                t.proj_bytes / 1048576.0);
    auto t_stage = std::chrono::steady_clock::now();
    npu_app stage_app = pm->create_app();
    q38npu::npu_delta_weights dw(stage_app, t, q4nx, kLayer);
    const double stage_ms = ms_since(t_stage);
    q38npu::delta_proj proj(pm, t, q38npu::dequant_mm_rtp_addresses{});
    buffer<u8>&        W = dw.weights();
    std::printf("   staged in %.0f ms  (x48 = %.1f s of load time)\n",
                stage_ms, 48.0 * stage_ms / 1000.0);

    // The CPU sides. `cpu` is the scored reference; `half` drives the hybrid
    // path. Unlike N8 there is no weights-only companion -- a delta layer's
    // weights and its recurrent state live in the same object, and the hybrid
    // path needs both, so the two differ only in which one ran which pass.
    cpu_ref::delta_net_layer cpu(q4nx, geo, kLayer);
    cpu_ref::delta_net_layer half(q4nx, geo, kLayer);

    // The layer's own tensors, for the same-bytes references. `get_q4nx`
    // returns a view into the mmap -- no copy, and the exact bytes the BO was
    // staged from.
    const std::string pl = "model.layers." + std::to_string(kLayer) + ".";
    const cpu_ref::q4nx_tensor W_qkv  = cpu_ref::get_q4nx(q4nx, pl + "linear_attn.qkv_proj.weight");
    const cpu_ref::q4nx_tensor W_z    = cpu_ref::get_q4nx(q4nx, pl + "self_attn.gate_proj.weight");
    const cpu_ref::q4nx_tensor W_out  = cpu_ref::get_q4nx(q4nx, pl + "linear_attn.ssm_out_proj.weight");
    const cpu_ref::q4nx_tensor W_up   = cpu_ref::get_q4nx(q4nx, pl + "mlp.up_proj.weight");
    const cpu_ref::q4nx_tensor W_gate = cpu_ref::get_q4nx(q4nx, pl + "mlp.gate_proj.weight");
    const cpu_ref::q4nx_tensor W_down = cpu_ref::get_q4nx(q4nx, pl + "mlp.down_proj.weight");

    // ======================================================================
    // 2. the weight layout, by memcmp
    // ======================================================================
    // The claim the fused `qkv|z` dispatch rests on: reorder_cpy's permutation
    // never crosses a 64-weight-row boundary, so staging qkv and z through two
    // separate calls produces the same bytes as one call over their
    // concatenation -- which is what the op will read. Same for the up|gate
    // weave, which the staging lays down 512 rows at a time.
    //
    // These are the two checks in the suite that can fail with no hardware
    // involved, and whose failure invalidates everything below rather than
    // merely joining it.
    {
        const size_t qb = q38npu::q4k_bytes((size_t)CD * D);
        const size_t zb = q38npu::q4k_bytes((size_t)DV * D);
        const size_t total = qb + zb;

        std::vector<uint8_t> cat(total);
        std::memcpy(cat.data(),      q4nx.raw(pl + "linear_attn.qkv_proj.weight"), qb);
        std::memcpy(cat.data() + qb, q4nx.raw(pl + "self_attn.gate_proj.weight"),  zb);

        std::vector<uint8_t> wantb(total);
        q38npu::reorder_cpy(wantb.data(), cat.data(), total, D);

        const bool same =
            std::memcmp(wantb.data(), W.data() + t.w_qkv_offset, total) == 0;
        size_t first = total;
        if (!same)
            for (size_t i = 0; i < total; i++)
                if (wantb[i] != W.data()[t.w_qkv_offset + i]) { first = i; break; }

        char buf[240];
        if (same)
            std::snprintf(buf, sizeof buf,
                          "%.1f MB of qkv|z identical to one reorder_cpy over "
                          "the concatenation", total / 1048576.0);
        else
            std::snprintf(buf, sizeof buf,
                          "diverges at byte %zu of %zu (block %zu, %s)",
                          first, total, first / q38npu::W_BLOCK_BYTES,
                          first < qb ? "in qkv" : "in z");
        report("per-tensor staging == concatenated staging", same, buf);
    }
    {
        // The weave, the same way. `stage_delta_proj_weights` alternates one
        // 512-row chunk of up with one of gate and reorders each chunk on its
        // own; the dispatch reads the result as one 34816-row matrix.
        const size_t total = q38npu::q4k_bytes((size_t)DI * D);
        const size_t chunk = q38npu::q4k_bytes((size_t)512 * D);
        const size_t phases = total / chunk;
        const uint8_t* up = q4nx.raw(pl + "mlp.up_proj.weight");
        const uint8_t* gt = q4nx.raw(pl + "mlp.gate_proj.weight");

        std::vector<uint8_t> cat(2 * total);
        for (size_t i = 0; i < phases; i++) {
            std::memcpy(cat.data() + (2 * i)     * chunk, up + i * chunk, chunk);
            std::memcpy(cat.data() + (2 * i + 1) * chunk, gt + i * chunk, chunk);
        }
        std::vector<uint8_t> wantb(2 * total);
        q38npu::reorder_cpy(wantb.data(), cat.data(), 2 * total, D);

        const bool same = std::memcmp(wantb.data(), W.data() + t.w_upgate_offset,
                                      2 * total) == 0;
        size_t first = 2 * total;
        if (!same)
            for (size_t i = 0; i < 2 * total; i++)
                if (wantb[i] != W.data()[t.w_upgate_offset + i]) { first = i; break; }

        char buf[240];
        if (same)
            std::snprintf(buf, sizeof buf,
                          "%.1f MB of the up|gate weave identical to one "
                          "reorder_cpy over %zu alternating chunks",
                          2 * total / 1048576.0, 2 * phases);
        else
            std::snprintf(buf, sizeof buf,
                          "diverges at byte %zu of %zu (chunk %zu, %s band)",
                          first, 2 * total, first / chunk,
                          (first / chunk) % 2 ? "gate" : "up");
        report("per-chunk weave == concatenated weave", same, buf);
    }

    // ======================================================================
    // the activations every dispatch is scored on
    // ======================================================================
    // Real ones, from the dumped prefill window: the output of layer 3, which
    // is this layer's input. A random x would score the arithmetic honestly but
    // would not exercise the dynamic range Q4_K's per-group scales were fitted
    // to. Unlike N8 there is no RoPE to reproduce -- a delta layer has no
    // positional rotation, the position enters through the fold.
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
    std::printf("   window %d rows from %s\n\n", P, key);

    // The stage inputs, each scored against ITS OWN input rather than charged
    // for the one upstream. (memory: score-a-projection-against-its-own-input)
    std::vector<float> xn;
    half.in_norm(x_all.data(), P, xn);
    for (float& f : xn) f = cpu_ref::round_bf16(f);

    std::vector<float> qkv_ref((size_t)P * CD), z_ref((size_t)P * DV);
    cpu_ref::q4nx_matmul(W_qkv, xn.data(), P, qkv_ref.data());
    cpu_ref::q4nx_matmul(W_z,   xn.data(), P, z_ref.data());

    // ssm_out's input: the post-gated-RMSNorm delta row, through the
    // reference's own recurrence, so `out` is scored on the activation the live
    // path will hand it rather than on a synthetic one with the wrong dynamic
    // range. The two mutating stages run on a SCRATCH layer -- `half` must
    // reach check 9 with a clean history.
    std::vector<float> core_in;
    {
        cpu_ref::delta_net_layer scratch(q4nx, geo, kLayer);
        std::vector<float> aa, bb, alpha, beta, qkv_c = qkv_ref;
        scratch.alpha_beta(xn.data(), P, aa, bb);
        scratch.conv_silu(qkv_c.data(), P);
        scratch.gates(aa.data(), bb.data(), P, alpha, beta);
        scratch.delta_rule(qkv_c.data(), alpha.data(), beta.data(), P, core_in);
        scratch.gated_rmsnorm(core_in.data(), z_ref.data(), P);
        for (float& f : core_in) f = cpu_ref::round_bf16(f);
    }

    std::vector<float> h1((size_t)P * D);
    cpu_ref::q4nx_matmul(W_out, core_in.data(), P, h1.data());
    for (size_t i = 0; i < h1.size(); i++) h1[i] += x_all[i];

    std::vector<float> hn;
    half.post_norm(h1.data(), P, hn);
    for (float& f : hn) f = cpu_ref::round_bf16(f);

    std::vector<float> up_ref((size_t)P * DI), gate_ref((size_t)P * DI);
    cpu_ref::q4nx_matmul(W_up,   hn.data(), P, up_ref.data());
    cpu_ref::q4nx_matmul(W_gate, hn.data(), P, gate_ref.data());
    std::vector<float> g;
    half.swiglu(gate_ref.data(), up_ref.data(), P, g);
    for (float& f : g) f = cpu_ref::round_bf16(f);

    std::printf("-- the four dispatches, each against its own same-bytes "
                "reference --\n");

    // ======================================================================
    // 3. qkv | z, one fused dispatch
    // ======================================================================
    {
        std::vector<float> gqkv, gz;
        proj.qkv_z(xn.data(), P, W, gqkv, gz);

        const int QZ = t.QKVZ_OUT();
        std::vector<float> got((size_t)P * QZ), want((size_t)P * QZ);
        for (int l = 0; l < P; l++) {
            float* gd = got.data()  + (size_t)l * QZ;
            float* wd = want.data() + (size_t)l * QZ;
            std::memcpy(gd,      gqkv.data()    + (size_t)l * CD, CD * 4);
            std::memcpy(gd + CD, gz.data()      + (size_t)l * DV, DV * 4);
            std::memcpy(wd,      qkv_ref.data() + (size_t)l * CD, CD * 4);
            std::memcpy(wd + CD, z_ref.data()   + (size_t)l * DV, DV * 4);
        }
        score_with_rotation("qkv|z: one 16384-row dispatch",
                            "qkv|z: rotated by one core column",
                            got, want, P, QZ);

        // Per sub-tensor, because the aggregate above cannot see q and k: they
        // are 4096 of 16384 columns, so v and z carry 75% of the norm and a
        // cosine over the whole matrix reports them.
        // (memory: fused-matmul-cosine-hides-narrow-tensors)
        {
            const char* nm[4] = {"q", "k", "v", "z"};
            const int   w[4]  = {DQ, DK, DV, DV};
            const int   o[4]  = {0, DQ, DQ + DK, CD};
            std::printf("         per tensor, rel L2:");
            for (int i = 0; i < 4; i++) {
                std::vector<float> a((size_t)P * w[i]), b((size_t)P * w[i]);
                for (int l = 0; l < P; l++) {
                    std::memcpy(a.data() + (size_t)l * w[i],
                                got.data() + (size_t)l * QZ + o[i], w[i] * 4);
                    std::memcpy(b.data() + (size_t)l * w[i],
                                want.data() + (size_t)l * QZ + o[i], w[i] * 4);
                }
                std::printf("  %s %.5f (cos %.6f)", nm[i],
                            rel_l2(a.data(), b.data(), a.size()),
                            cosine(a.data(), b.data(), a.size()));
            }
            std::printf("\n");
        }

        // The ordering control. `z | qkv` is a legal 16384-row matrix over the
        // same bytes and the op cannot tell it from `qkv | z` -- the whole
        // reason `delta_layout` and the staging both assert the adjacency, and
        // the reason a scatter copied from another engine's order would produce
        // plausible garbage. (memory: rope-tile-consumes-kv-before-q)
        std::vector<float> zq((size_t)P * QZ);
        for (int l = 0; l < P; l++) {
            float* d = zq.data() + (size_t)l * QZ;
            std::memcpy(d,      z_ref.data()   + (size_t)l * DV, DV * 4);
            std::memcpy(d + DV, qkv_ref.data() + (size_t)l * CD, CD * 4);
        }
        report_control("qkv|z: read as z|qkv", cosine(got.data(), zq.data(),
                                                      got.size()));

        // And the scatter itself: the split is a column offset, so a `qkv` that
        // began at column DV instead of 0 would still assemble back into a
        // matrix scoring 0.999 above.
        report_cos("qkv|z: qkv is columns [0, conv_dim)",
                   cosine(gqkv.data(), qkv_ref.data(), gqkv.size()), kGate);
        report_cos("qkv|z: z is columns [conv_dim, +dv)",
                   cosine(gz.data(), z_ref.data(), gz.size()), kGate);
    }

    // ======================================================================
    // 4. ssm_out -- and the Din that is not D
    // ======================================================================
    {
        std::vector<float> got((size_t)P * D, 0.0f);
        proj.out(core_in.data(), P, W, got.data());
        std::vector<float> want((size_t)P * D);
        cpu_ref::q4nx_matmul(W_out, core_in.data(), P, want.data());
        score_with_rotation("ssm_out: y = W_out . delta_row  (Din = dv = 6144)",
                            "ssm_out: rotated by one core column", got, want, P, D);

        // The trap `reorder_cpy`'s `col` parameter carries, made explicit:
        // ssm_out reduces over the gated delta row, not over D. A [5120][5120]
        // matrix at the same offset is a legal shape over bytes that are really
        // [5120][6144], and nothing between here and the cores notices. This is
        // also the check that would catch `stage_delta_proj_weights` passing
        // `t.D` where it passes `t.DV`.
        q38npu::dequant_mm_op wrong(pm->create_app(), D, D, t.w_out_offset);
        std::vector<float> bad((size_t)P * D, 0.0f);
        wrong.run(core_in.data(), P, W, bad.data());   // core's first D columns
        report_control("ssm_out: read as Din = D instead of dv",
                       cosine(bad.data(), want.data(), bad.size()));
    }

    // ======================================================================
    // 5. up | gate, woven -- and the two-pass RTP write that silu's it
    // ======================================================================
    //
    // The production op runs TWO passes over the same matrix, `{0,2,none}` for
    // up and `{1,2,silu}` for gate, so the kernel's fused activation lands on
    // the gate bands alone. Three things have to be separated, because each can
    // fail while the others look fine:
    //
    //   a  the strided `j` and the mid-sequence drain address the same bytes a
    //      single dense pass does        -- activation held out entirely
    //   b  what the device's bf16 silu costs against an fp32 one
    //   c  the silu landed on the gate bands and not the up bands
    //
    // (a) is scored bit-for-bit and (c) bit-for-bit too, so neither can be paid
    // for by arithmetic slack in (b).
    {
    constexpr size_t BAND = 512;          // the weave grain, = PASS_COLS
    const size_t WD    = 2 * (size_t)DI;
    const size_t bands = (size_t)DI / BAND;

    q38npu::dequant_mm_op raw_op(pm->create_app(), 2 * DI, D, t.w_upgate_offset);
    std::vector<float> raw((size_t)P * WD, 0.0f);
    raw_op.run(hn.data(), P, W, raw.data());
    {
        std::vector<float> woven((size_t)P * WD), concat((size_t)P * WD);
        for (int l = 0; l < P; l++) {
            float* wv = woven.data()  + (size_t)l * WD;
            float* cc = concat.data() + (size_t)l * WD;
            const float* u  = up_ref.data()   + (size_t)l * DI;
            const float* g2 = gate_ref.data() + (size_t)l * DI;
            for (size_t p = 0; p < bands; p++) {
                std::memcpy(wv + (2 * p)     * BAND, u  + p * BAND, BAND * 4);
                std::memcpy(wv + (2 * p + 1) * BAND, g2 + p * BAND, BAND * 4);
            }
            std::memcpy(cc,      u,  (size_t)DI * 4);
            std::memcpy(cc + DI, g2, (size_t)DI * 4);
        }
        report_cos("up|gate: woven at 512 output rows",
                   cosine(raw.data(), woven.data(), raw.size()), kGate);
        report_control("up|gate: concatenated instead of woven",
                       cosine(raw.data(), concat.data(), raw.size()));
    }

    // ---- 5a. the strided j and the mid-sequence drain, activation off -----
    // Two passes that between them cover exactly what one dense pass covers,
    // both asking for `none`. Any difference at all is the split itself: a
    // wrong w_offset or y_offset under the stride, a ping/pong bank reused
    // while in flight, or a column block written by neither pass and left
    // holding the previous tile. So the bar is EQUALITY, not a cosine.
    {
        q38npu::dequant_mm_op split(
            pm->create_app(), 2 * DI, D, t.w_upgate_offset,
            std::vector<q38npu::dequant_mm_pass>{
                {0, 2, q38npu::dequant_mm_act_none},
                {1, 2, q38npu::dequant_mm_act_none},
            });
        std::vector<float> got((size_t)P * WD, 0.0f);
        split.run(hn.data(), P, W, got.data());
        const bool same = std::memcmp(got.data(), raw.data(),
                                      got.size() * sizeof(float)) == 0;
        size_t first = got.size();
        for (size_t i = 0; i < got.size() && !same; i++)
            if (got[i] != raw[i]) { first = i; break; }
        char buf[192];
        if (same)
            std::snprintf(buf, sizeof buf, "bit-identical over %zu elements",
                          got.size());
        else
            std::snprintf(buf, sizeof buf,
                          "diverges at element %zu (row %zu, col %zu of %zu)",
                          first, first / WD, first % WD, WD);
        report("up|gate 5a: two none-passes == one dense pass", same, buf);
    }

    // The production op: `{0,2,none}` then `{1,2,silu}`.
    std::vector<float> act((size_t)P * WD, 0.0f);
    proj.upgate_op().run(hn.data(), P, W, act.data());

    // ---- 5b. what the device's bf16 silu costs ----------------------------
    // Score the odd bands against fp32 silu OF THE RAW ODD BANDS -- not of
    // gate_ref -- so the matmul's own error is common mode and this number is
    // the activation alone.
    {
        std::vector<float> dvec, rvec;
        dvec.reserve((size_t)P * DI); rvec.reserve((size_t)P * DI);
        for (int l = 0; l < P; l++)
            for (size_t p = 0; p < bands; p++) {
                const size_t o = (size_t)l * WD + (2 * p + 1) * BAND;
                for (size_t j = 0; j < BAND; j++) {
                    dvec.push_back(act[o + j]);
                    const float x = raw[o + j];
                    rvec.push_back(x / (1.0f + std::exp(-x)));
                }
            }
        const double c = cosine(dvec.data(), rvec.data(), dvec.size());
        const double e = rel_l2(dvec.data(), rvec.data(), dvec.size());
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "cos %.6f, rel L2 %.5f vs fp32 silu of the same rows", c, e);
        report("up|gate 5b: the device's bf16 silu", c >= 0.999 && e < 0.02, buf);
    }

    // ---- 5c. the silu landed on the gate bands only -----------------------
    // The even bands must come back untouched, bit for bit. A cosine would not
    // do here: silu(x) ~ x for x >> 0, so an activation wrongly applied to up
    // still scores high on a band whose values happen to be positive.
    {
        size_t nbad = 0, first = 0;
        for (int l = 0; l < P; l++)
            for (size_t p = 0; p < bands; p++) {
                const size_t o = (size_t)l * WD + (2 * p) * BAND;
                for (size_t j = 0; j < BAND; j++)
                    if (act[o + j] != raw[o + j]) {
                        if (!nbad) first = o + j;
                        nbad++;
                    }
            }
        char buf[160];
        if (nbad)
            std::snprintf(buf, sizeof buf,
                          "%zu up elements moved, first at %zu", nbad, first);
        else
            std::snprintf(buf, sizeof buf, "all %zu up elements unchanged",
                          (size_t)P * DI);
        report("up|gate 5c: the up bands are untouched", nbad == 0, buf);
    }

    // ---- the fused readback: silu on device, the product in the un-weave --
    {
        std::vector<float> got;
        proj.up_gate_swiglu(hn.data(), P, W, got);
        std::vector<float> want;
        half.swiglu(gate_ref.data(), up_ref.data(), P, want);
        report_cos("up|gate: fused to silu(gate) * up",
                   cosine(got.data(), want.data(), got.size()), kGate);

        // If the un-weave read one band twice, the scores above would still be
        // whatever that band deserved. Both degenerate readings must collapse.
        std::vector<float> gg, uu;
        half.swiglu(gate_ref.data(), gate_ref.data(), P, gg);
        half.swiglu(up_ref.data(),   up_ref.data(),   P, uu);
        report_control("up|gate: silu(gate) * gate instead",
                       cosine(got.data(), gg.data(), got.size()));
        report_control("up|gate: silu(up) * up instead",
                       cosine(got.data(), uu.data(), got.size()));

        // And the SwiGLU orientation, which is the one bug this design can
        // plausibly have: silu(up) * gate is just as plausible a reading and
        // scores nearly as well. (memory: score-swiglu-by-swapping-up-and-gate)
        std::vector<float> swapped;
        half.swiglu(up_ref.data(), gate_ref.data(), P, swapped);
        report_control("up|gate: silu(up) * gate instead",
                       cosine(got.data(), swapped.data(), got.size()));
    }

    // ---- the parity swap, from the device side ----------------------------
    // `{0,2,silu}, {1,2,none}` silu's the UP bands instead. Both halves are the
    // same shape, so nothing about the dispatch complains; only the arithmetic
    // says which parity the RTP write landed on.
    {
        q38npu::dequant_mm_op sw_op(
            pm->create_app(), 2 * DI, D, t.w_upgate_offset,
            std::vector<q38npu::dequant_mm_pass>{
                {0, 2, q38npu::dequant_mm_act_silu},   // up, wrongly
                {1, 2, q38npu::dequant_mm_act_none},   // gate, raw
            });
        std::vector<float> sw((size_t)P * WD, 0.0f);
        sw_op.run(hn.data(), P, W, sw.data());

        std::vector<float> got((size_t)P * DI);
        for (int l = 0; l < P; l++)
            for (size_t p = 0; p < bands; p++) {
                const size_t o = (size_t)l * WD + (2 * p) * BAND;
                for (size_t j = 0; j < BAND; j++)
                    got[(size_t)l * DI + p * BAND + j] = sw[o + BAND + j] * sw[o + j];
            }
        std::vector<float> want;
        half.swiglu(gate_ref.data(), up_ref.data(), P, want);
        report_control("up|gate: silu on the up bands instead of the gate bands",
                       cosine(got.data(), want.data(), got.size()));
    }
    }

    // ======================================================================
    // 6. down_proj
    // ======================================================================
    {
        std::vector<float> got((size_t)P * D, 0.0f);
        proj.down(g.data(), P, W, got.data());
        std::vector<float> want((size_t)P * D);
        cpu_ref::q4nx_matmul(W_down, g.data(), P, want.data());
        score_with_rotation("down: y = W_down . swiglu  (Din = 17408)",
                            "down: rotated by one core column", got, want, P, D);
    }

    // ======================================================================
    // 7. the host-side tiling is inert
    // ======================================================================
    // `dequant_mm_op` pins L at 512 and tiles a window on the host. That is only
    // sound if a row's result depends on its own x row and nothing else -- so
    // the same rows, computed as part of a two-tile window and as part of a
    // ragged one, must come back BIT FOR BIT identical. Anything less means a
    // pad row bled into a real one, or a per-tile offset is wrong.
    //
    // Bit-for-bit and not by cosine on purpose: a tail-tile pad row leaking into
    // row 2 of a 3-row window moves that row a long way, but a cosine over
    // 1024 x 5120 values would never see it.
    //
    // Run on `ssm_out`, whose Din is the one that is not D: a per-tile x offset
    // computed with the wrong stride is invisible on a square-ish matrix.
    //
    // This check also underwrites check 11 -- if the projections are row-local
    // then the only thing that can make L = 1 differ from L = n is the fold.
    {
        constexpr int TMAX = 1024;
        std::vector<float> x((size_t)TMAX * DV);
        fill_synth(x, 0x51b0c0de);

        std::vector<float> full((size_t)TMAX * D, 0.0f);
        proj.out(x.data(), TMAX, W, full.data());

        std::string bad;
        for (int T : {1, 3, 128, 221, 511, 512, 513, 1000}) {
            std::vector<float> part((size_t)T * D, 0.0f);
            proj.out(x.data(), T, W, part.data());
            if (std::memcmp(part.data(), full.data(),
                            (size_t)T * D * sizeof(float)) != 0) {
                size_t i = 0;
                while (i < (size_t)T * D && part[i] == full[i]) i++;
                bad += (bad.empty() ? "" : ", ") + std::to_string(T) +
                       " (row " + std::to_string(i / D) + ")";
            }
        }
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "8 ragged widths reproduce their rows out of a %d-row "
                      "(%d-tile) run exactly", TMAX, TMAX / 512);
        report("host-side tiling changes no row", bad.empty(),
               bad.empty() ? buf : ("widths that differ: " + bad));
    }

    // ======================================================================
    // 8. the weight BO crosses apps
    // ======================================================================
    // Checks 3-7 all ran against a BO created from one app in dispatches
    // belonging to others, and the engine allocates all 48 from a single
    // staging app for exactly that reason. BOs are device-scoped, so that is
    // legal -- but "legal" here means the dispatch completes, and a BO from the
    // wrong place does not fault, it simply never completes or reads stale
    // bytes. So: the same matrix, copied into a BO from the DISPATCHING app,
    // must give the identical answer.
    {
        npu_app      a  = pm->create_app();
        const size_t ob = q38npu::q4k_bytes((size_t)D * DV);

        buffer<u8> priv = a.create_bo_buffer<u8>(ob);
        std::memcpy(priv.data(), W.data() + t.w_out_offset, ob);
        priv.sync_to_device();

        q38npu::dequant_mm_op own(pm->create_app(), D, DV, 0);
        std::vector<float> a_out((size_t)P * D, 0.0f), b_out((size_t)P * D, 0.0f);
        own.run(core_in.data(), P, priv, a_out.data());
        proj.out(core_in.data(), P, W, b_out.data());

        const bool same = std::memcmp(a_out.data(), b_out.data(),
                                      a_out.size() * sizeof(float)) == 0;
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "%.1f MB private copy vs the shared %.0f MB BO, %s",
                      ob / 1048576.0, t.proj_bytes / 1048576.0,
                      same ? "bit for bit" : "DIFFERENT");
        report("a BO from another app is not degraded", same, buf);
    }

    // ======================================================================
    // Part B: the hybrid layer
    // ======================================================================
    std::printf("\n-- the hybrid layer: cpu_ref's recurrence around four "
                "dispatches --\n");

    // `prefill_linear_block::forward` rebuilt here from the pieces, because
    // the engine only hands back a hidden state and this is the one place each
    // stage can be seen. What it is scored against is
    // `cpu_ref::delta_net_layer::forward()`, which shares every elementwise
    // helper with it.
    //
    // \param keep_residual  check 12's sabotage switch.
    auto run_hybrid = [&](cpu_ref::delta_net_layer& L_,
                          const float* x, int L, float* y, bool keep_residual) {
        std::vector<float> lxn;
        L_.in_norm(x, L, lxn);

        std::vector<float> qkv, zz;
        proj.qkv_z(lxn.data(), L, W, qkv, zz);

        std::vector<float> aa, bb;
        L_.alpha_beta(lxn.data(), L, aa, bb);

        L_.conv_silu(qkv.data(), L);                   // advances conv_state_

        std::vector<float> alpha, beta;
        L_.gates(aa.data(), bb.data(), L, alpha, beta);

        std::vector<float> core;
        L_.delta_rule(qkv.data(), alpha.data(), beta.data(), L, core);  // state_

        L_.gated_rmsnorm(core.data(), zz.data(), L);

        std::vector<float> lh1((size_t)L * D, 0.0f);
        proj.out(core.data(), L, W, lh1.data());
        if (keep_residual)
            for (size_t i = 0; i < lh1.size(); i++) lh1[i] += x[i];

        std::vector<float> lhn;
        L_.post_norm(lh1.data(), L, lhn);
        std::vector<float> lg;
        proj.up_gate_swiglu(lhn.data(), L, W, lg);
        proj.down(lg.data(), L, W, y);
        for (size_t i = 0; i < (size_t)L * D; i++) y[i] += lh1[i];

        L_.advance(L);
    };

    // The reference: the same layer, entirely on the host, through the very
    // helpers the path above calls.
    cpu.reset();
    std::vector<float> y_ref((size_t)P * D), y_hyb((size_t)P * D);
    cpu.forward(x_all.data(), P, y_ref.data());

    // ======================================================================
    // 9. one hybrid layer against the CPU reference
    // ======================================================================
    {
        half.reset();
        run_hybrid(half, x_all.data(), P, y_hyb.data(), /*keep_residual=*/true);

        // Two statistics, because they answer different questions and neither
        // substitutes for the other:
        //
        //   RELATIVE L2 is the magnitude. The error is a flat relative
        //   perturbation -- bfp16 in the mmul, bf16 on every readback -- so
        //   this is the number with a P-independent bar.
        //   WORST-ROW COSINE is the addressing check, and only that. A minimum
        //   over P samples drifts down with P at uniform error, so it gets a
        //   loose bar; what it discriminates is a row holding the wrong
        //   content, which scores near zero rather than near 0.998.
        // (memory: worst-row-cosine-is-a-minimum-over-n,
        //          per-row-cosine-reports-the-norm)
        const double c = cosine(y_hyb.data(), y_ref.data(), y_hyb.size());
        const double e = rel_l2(y_hyb.data(), y_ref.data(), y_hyb.size());
        double worst = 2.0; int worst_row = -1;
        for (int r = 0; r < P; r++) {
            const double rc = cosine(y_hyb.data() + (size_t)r * D,
                                     y_ref.data() + (size_t)r * D, (size_t)D);
            if (rc < worst) { worst = rc; worst_row = r; }
        }
        char buf[260];
        std::snprintf(buf, sizeof buf,
                      "cos %.6f [0.999], rel L2 %.5f [0.06], worst row %.6f "
                      "(row %d) [0.99] -- L = %d through one fold",
                      c, e, worst, worst_row, P);
        report("hybrid layer vs cpu_ref::forward",
               c >= 0.999 && e < 0.06 && worst >= 0.99, buf);
    }

    // ======================================================================
    // 10. what the bf16 projections did to the recurrent state
    // ======================================================================
    // Scored separately from the layer output, and read differently from N8's
    // equivalent. There the k/v rows made a round trip through a device cache
    // and the question was whether storage was inert. Here **the state never
    // leaves the host** -- there is no device cache to be wrong about -- so
    // what this measures is how far the bf16 error on `qkv` propagates into a
    // fold that has no inverse.
    //
    // So it is reported first and gated loosely. What it discriminates is not
    // precision but structure: a conv window holding the wrong taps, or an S
    // that absorbed a token twice, is off by a lot and not a little.
    // (memory: k-v-cache-roundtrip-is-exactly-inert)
    {
        const std::vector<float>& cw_h = half.conv_window();
        const std::vector<float>& cw_c = cpu.conv_window();
        const std::vector<float>& s_h  = half.recurrent_state();
        const std::vector<float>& s_c  = cpu.recurrent_state();

        const double ec = rel_l2(cw_h.data(), cw_c.data(), cw_h.size());
        const double es = rel_l2(s_h.data(),  s_c.data(),  s_h.size());
        const double cc = cosine(cw_h.data(), cw_c.data(), cw_h.size());
        const double cs = cosine(s_h.data(),  s_c.data(),  s_h.size());

        // Per head, because S is [v_heads][k][v] and one head folded with the
        // wrong gate would be diluted 48:1 in the aggregate above.
        const size_t hsz = (size_t)geo.k_head_dim * geo.v_head_dim;
        double worst = 2.0; int worst_head = -1;
        for (int h = 0; h < geo.v_heads; h++) {
            const double ch = cosine(half.head_state(h), cpu.head_state(h), hsz);
            if (ch < worst) { worst = ch; worst_head = h; }
        }

        char buf[300];
        std::snprintf(buf, sizeof buf,
                      "conv rel L2 %.5f (cos %.6f); S rel L2 %.5f (cos %.6f); "
                      "worst head %.6f (h%d) [0.99]; seen %d/%d",
                      ec, cc, es, cs, worst, worst_head,
                      half.tokens_seen(), cpu.tokens_seen());
        report("the fold tracks the reference's",
               worst >= 0.99 && cc >= 0.99 && cs >= 0.99 &&
               half.tokens_seen() == P && cpu.tokens_seen() == P, buf);
    }

    // ======================================================================
    // 11. L = 1 repeated == L = n in one go
    // ======================================================================
    // The delta rule's defining property, and the check with no counterpart in
    // N8. An attention layer is row-parallel; this one is a fold, and
    // `conv_silu`, `delta_rule` and `advance` must each run exactly once per
    // token and in order. A double `advance`, a conv window shifted by the
    // wrong amount, or a `delta_rule` that re-reads a row shows up here and
    // essentially nowhere else -- check 9 would still pass with a state that is
    // internally consistent but historically wrong.
    //
    // Check 7 already established that the projections are row-local and
    // bit-identical across tile widths, so any difference this check finds is
    // the fold.
    //
    // Run over a short window: this is 32 separate 4-dispatch rounds and the
    // property does not need 221 tokens to hold.
    {
        constexpr int SHORT = 32;
        std::vector<float> y_one((size_t)SHORT * D), y_many((size_t)SHORT * D);

        cpu_ref::delta_net_layer a(q4nx, geo, kLayer), b(q4nx, geo, kLayer);
        a.reset(); b.reset();

        run_hybrid(a, x_all.data(), SHORT, y_many.data(), true);
        for (int i = 0; i < SHORT; i++)
            run_hybrid(b, x_all.data() + (size_t)i * D, 1,
                       y_one.data() + (size_t)i * D, true);

        const double e  = rel_l2(y_one.data(), y_many.data(), y_one.size());
        const double es = rel_l2(b.recurrent_state().data(),
                                 a.recurrent_state().data(),
                                 a.recurrent_state().size());
        const double ec = rel_l2(b.conv_window().data(),
                                 a.conv_window().data(),
                                 a.conv_window().size());
        const bool bitwise = std::memcmp(y_one.data(), y_many.data(),
                                         y_one.size() * sizeof(float)) == 0;
        char buf[260];
        std::snprintf(buf, sizeof buf,
                      "y rel L2 %.2e%s; S %.2e, conv %.2e; seen %d/%d (want %d)",
                      e, bitwise ? " (bit-identical)" : "", es, ec,
                      b.tokens_seen(), a.tokens_seen(), SHORT);
        report("token-at-a-time == the whole window at once",
               e < 1e-5 && es < 1e-5 && ec < 1e-5 &&
               a.tokens_seen() == SHORT && b.tokens_seen() == SHORT, buf);
    }

    // ======================================================================
    // 12. the sabotage
    // ======================================================================
    // Check 9's gate has to be able to fail. The mixer residual is the cheapest
    // thing to drop and the hardest to see: y is still the right shape, still
    // built from the right weights, the fold still advances correctly, and
    // every dispatch above still scores 0.999.
    {
        cpu_ref::delta_net_layer bad_(q4nx, geo, kLayer);
        bad_.reset();
        std::vector<float> y_bad((size_t)P * D);
        run_hybrid(bad_, x_all.data(), P, y_bad.data(), /*keep_residual=*/false);
        const double c = cosine(y_bad.data(), y_ref.data(), y_bad.size());
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "cos %.6f without the mixer residual (check 9 was %.6f)",
                      c, cosine(y_hyb.data(), y_ref.data(), y_hyb.size()));
        report("dropping the mixer residual breaks check 9", c < 0.999, buf);
    }

    // ======================================================================
    // 13. throughput
    // ======================================================================
    // Not a gate -- it is the number this phase exists to produce. Both paths
    // read the whole 211 MB of weights at any width, so the device should win
    // from a handful of rows. The hybrid prefill path only fires at L >= 128,
    // so the crossover matters as a sanity check on that choice.
    //
    // The CPU column stops at 512: a host pass over 383 M parameters says
    // nothing new above that.
    {
        std::printf("\n-- one layer's projections, ms (4 dispatches per 512-row "
                    "tile) --\n");
        std::printf("   %6s  %10s  %10s  %9s  %9s\n",
                    "L", "NPU", "CPU", "NPU us/row", "speedup");

        for (int T : {1, 128, 221, 512, 1024}) {
            std::vector<float> xs((size_t)T * D), cs((size_t)T * DV),
                               gs((size_t)T * DI);
            fill_synth(xs, 0xbeef ^ (uint64_t)T);
            fill_synth(cs, 0xfeed ^ (uint64_t)T);
            fill_synth(gs, 0xf00d ^ (uint64_t)T);
            std::vector<float> y1((size_t)T * D), q2, z2, g2;

            auto t0 = std::chrono::steady_clock::now();
            proj.qkv_z(xs.data(), T, W, q2, z2);
            proj.out(cs.data(), T, W, y1.data());
            proj.up_gate_swiglu(xs.data(), T, W, g2);
            proj.down(gs.data(), T, W, y1.data());
            const double npu = ms_since(t0);

            double cpu_ms = 0.0;
            if (T <= 512) {
                std::vector<float> o1((size_t)T * D), o2((size_t)T * CD),
                                   o3((size_t)T * DV), o5((size_t)T * DI),
                                   o6((size_t)T * DI), o7;
                auto c0 = std::chrono::steady_clock::now();
                cpu_ref::q4nx_matmul(W_qkv,  xs.data(), T, o2.data());
                cpu_ref::q4nx_matmul(W_z,    xs.data(), T, o3.data());
                cpu_ref::q4nx_matmul(W_out,  cs.data(), T, o1.data());
                cpu_ref::q4nx_matmul(W_up,   xs.data(), T, o5.data());
                cpu_ref::q4nx_matmul(W_gate, xs.data(), T, o6.data());
                cpu_ref::q4nx_matmul(W_down, gs.data(), T, o1.data());
                // The host SwiGLU the fused readback removes, which the NPU
                // column above never pays.
                half.swiglu(o6.data(), o5.data(), T, o7);
                cpu_ms = ms_since(c0);
            }

            if (cpu_ms > 0.0)
                std::printf("   %6d  %10.1f  %10.1f  %9.1f  %8.2fx\n",
                            T, npu, cpu_ms, npu * 1000.0 / T, cpu_ms / npu);
            else
                std::printf("   %6d  %10.1f  %10s  %9.1f  %9s\n",
                            T, npu, "-", npu * 1000.0 / T, "-");
        }
    }

    // ======================================================================
    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
