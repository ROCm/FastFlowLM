// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file verify_n8_prefill_proj.cpp
/// \brief N8 -- the prefill attention layers' projections on `dequant_mm.xclbin`:
///        fused Q4_K dequant + GEMM, over `npu_attn_layer`'s existing weight BO.
///
/// Build:
///   source ~/setup.sh
///   g++-13 -O2 -march=native -fopenmp -std=c++20
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n8_prefill_proj.cpp speculate/npu/proj_sequence.cpp
///       prefill/npu/prefill_attn_sequence.cpp verify/npu/attn_sequence.cpp
///       ../q4nx/q4_npu_eXpress.cpp ../q4nx/safe_tensors.cpp
///       -L../../external_libs -L../../build/lib -L/opt/xilinx/xrt/lib
///       -lxrt_coreutil -laiebu -luuid -lpthread -o /tmp/verify_n8_prefill_proj
///   (one line; the continuations are dropped here because a backslash at the
///    end of a // comment splices the next line into it)
///
/// Takes all three xclbins as FILES plus the model dir. One attention layer is
/// staged (~209 MB) and the CPU builds a same-bytes reference for each of its
/// 372 M parameters several times over.
///
/// ---------------------------------------------------------------------------
/// What moved
/// ---------------------------------------------------------------------------
/// N4 put the prefill window's ATTENTION CORE on `attn.xclbin` and left the
/// seven projections -- q|z, k, v, o, up, gate, down -- on the host, which is
/// much the larger share of the FLOPs. Those now run on `dequant_mm.xclbin` as
/// FOUR dispatches per 512-row tile, because the staged q|k|v|z is one fused
/// matrix and the woven up|gate another. The host keeps the two norms, the
/// RoPE, the output gate and the two residuals -- every one of them the CPU
/// reference's own helper, called by name.
///
/// The four read `npu_attn_layer`'s existing 209 MB weight BO at four byte
/// offsets. Nothing is restaged and nothing is copied, and that is the load
/// bearing claim of the whole design: `stage_attn_proj_weights` lays the
/// layer's tensors down through `reorder_cpy(..., vertical_blocks = 2)`, which
/// is the same permutation dequant_mm's own host driver applies, and that
/// permutation is LOCAL TO 64 WEIGHT ROWS. Every sub-tensor here is a multiple
/// of 64 rows, so reordering each separately is byte for byte reordering the
/// concatenation. It is an argument, not a measurement -- check 2 turns it
/// into one.
///
/// ---------------------------------------------------------------------------
/// How the dispatches are scored
/// ---------------------------------------------------------------------------
/// Against a reference built from the SAME QUANTIZED BYTES, through
/// `cpu_ref::q4nx_matmul` on a bf16-rounded x. Q4_K's own ~0.997 loss is then
/// common mode and cancels, so the bar is 0.999 and what is left to measure is
/// the device's arithmetic: `aie::mmul<8,8,8,bf16>` on npu2 floors both
/// operands onto a shared-exponent-per-8 grid, and every fp32 -> bf16 on the
/// way out truncates toward -inf.
/// (memory: score-dequant-mm-against-its-own-quantized-bytes,
///          aie-mmul-bf16-truncates-to-bfp16)
///
/// Every score is paired with a control that must COLLAPSE, because a cosine
/// near 1 is evidence about arithmetic and says nothing about addressing:
///
///   rotate by 64        one core column. Catches an off-by-one core mapping,
///                       which a whole-matrix cosine cannot see.
///   k|v|q|z instead     **the draft head's staging order, not this one.**
///   of q|k|v|z          `stage_mtp_proj_weights` writes k, then v, then q,
///                       because the RoPE tile consumes kv before q;
///                       `stage_attn_proj_weights` writes q first. Both are
///                       legal 14336-row matrices over the same bytes and the
///                       op cannot tell them apart, so a `prefill_proj` that
///                       copied `mtp_proj`'s scatter would produce plausible
///                       garbage. (memory: the-rope-tile-consumes-kv-before-q)
///   concat instead      up|gate is WOVEN 512 output rows at a time, not
///   of weave            concatenated. Both are legal 34816-row matrices.
///   Din = D for o       o reduces over the gated row (6144), not over D. A
///                       matrix of identical byte count, strided wrong.
///   silu(up) * gate     just as plausible a reading of a SwiGLU, and scores
///                       nearly as well. (memory: score-swiglu-by-swapping-
///                       up-and-gate)
///
/// ---------------------------------------------------------------------------
/// The two halves, and what is NOT here
/// ---------------------------------------------------------------------------
///   checks 1-8     the operator, driven directly. Geometry, the weight-layout
///                  memcmp, the four dispatches with their controls, the
///                  host-side tiling, and the cross-app BO.
///   checks 9-13    the composition: one hybrid layer against
///                  `cpu_ref::attention_layer::forward()`, the k/v rows that
///                  reached the cache BO, the causal-depth segmentation, an
///                  L = 8 pass by the OTHER bitstream over the cache this path
///                  just filled, and a sabotage that must break check 9.
///   check 14       throughput. Not a gate; the number this phase exists for.
///
/// The second half is where a dispatch scoring 0.9999 still fails: it can do
/// that while the scatter reads q where the gate is, while o is fed the
/// ungated attention, or while a residual is dropped.
///
/// What this driver deliberately does NOT cover is the 64-layer stack. It
/// needs the whole engine -- 16 staged layers and the 48 delta-net ones -- and
/// the engine's own binary already composes exactly that. Running it here would
/// be a second transcription of `prefill_attn_block::forward`, which is the one
/// thing a reference must not be. Check 9 rebuilds that function from its
/// pieces so that a failure localises to a stage; `test/qwen3_8mtp_npu` is what
/// tests the seam itself, and any timing there must be read off a binary built
/// at the engine's flags. (memory: host-vs-device-ab-needs-shipping-flags)
///
/// The `FLM_Q38_PREFILL_PROJ=0` A/B that used to be the other half of this
/// paragraph is gone with the host arm it switched to; the measurements it
/// produced are in REPLACEMENTWORK.md.
#include "prefill/npu/prefill_proj.hpp"
#include "prefill/npu/prefill_attn_op.hpp"
#include "verify/npu/attn_layer.hpp"
#include "verify/npu/attn_weights.hpp"
#include "verify/npu/q4nx.hpp"
#include "common/cpu_ref_attention_layer.hpp"
#include "common/cpu_ref_common.hpp"
#include "common/rope.hpp"

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

using namespace cpu_ref::q38;

/// The layout's MAX_L for this suite. The dump is 221 rows and check 13 adds
/// eight more; the tiling checks do not touch the cache.
constexpr int kMaxL = 1024;

/// The first full-attention layer: (3 + 1) % 4 == 0. Same layer N4 uses, so
/// the two suites' numbers are comparable.
constexpr int kLayer = 3;

/// The bar for a same-bytes score. See the file header: Q4_K cancels, so what
/// is left is the bfp16 mmul and the bf16 readback.
constexpr double kGate = 0.999;

/// The ceiling a control must fall under. Generous on purpose -- a control
/// that lands at 0.98 has still told us the permutation is wrong, and pinning
/// it tighter would only make the suite brittle against the next checkpoint.
constexpr double kControl = 0.99;

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-48s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
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
    char buf[200];
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
/// receive; letting the reference start from extra precision charges the
/// device for a narrowing it cannot avoid.
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
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "dequant_mm/build/xclbins/dequant_mm.xclbin";
    const std::string prefill_xclbin =
        argc > 3 ? argv[3]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "gated_attention_prefill/build/xclbins/attn.xclbin";
    const std::string verify_xclbin =
        argc > 4 ? argv[4]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "gated_attention_verify/build/QWEN3_8_27B/xclbins/layer.xclbin";

    const int D = MODEL_DIM, DQ_ = DQ, DKV_ = DKV, DI = INTERMEDIATE_SIZE;

    std::printf("== N8: the prefill projections on dequant_mm ==\n");
    std::printf("   model    %s\n   proj     %s\n   prefill  %s\n   verify   %s\n\n",
                model_dir.c_str(), proj_xclbin.c_str(), prefill_xclbin.c_str(),
                verify_xclbin.c_str());

    const q38npu::attn_layout t = q38npu::attn_layout::build(kMaxL, /*L=*/8);

    // ======================================================================
    // 1. the geometry guard
    // ======================================================================
    // Everything the four dispatches assume about the layout, restated here
    // rather than trusted from prefill_proj's constructor -- that one throws at
    // load, so a violated precondition would take the engine down with a
    // message and nothing pointing at which of the conditions it was.
    //
    // Each of these is silent if it fails at the wrong layer: a ragged Dout or
    // Din hangs the dispatch; a Din that is not a whole number of 256-wide
    // blocks, or a row count that is not a multiple of 64, makes reorder_cpy
    // fold the buffer's head back in as padding with nothing asserting.
    {
        const int QKVZ = t.qkv_out + t.DQ;
        struct { const char* name; int Dout, Din; } shapes[4] = {
            {"q|k|v|z", QKVZ, D},
            {"o",       D,    t.DQ},
            {"up|gate", 2*DI, D},
            {"down",    D,    DI},
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
        // up and gate individually, and QKVZ passing says nothing about k.
        const int rows[7] = {t.DQ, t.DK, t.DV, t.DQ, D, DI, D};
        const char* rn[7] = {"q", "k", "v", "z", "o", "up/gate", "down"};
        for (int i = 0; i < 7; i++)
            want(rows[i] % q38npu::dequant_mm_tiles::N_TILE == 0,
                 std::string(rn[i]) + " is " + std::to_string(rows[i]) +
                 " rows, not a multiple of 64");
        // The fused dispatch reads from byte 0 across the q|k|v -> z boundary
        // and stops where o begins.
        want(t.w_z_offset == q38npu::q4k_bytes((size_t)t.qkv_out * D),
             "z does not immediately follow q|k|v");
        want(t.w_o_offset ==
                 t.w_z_offset + q38npu::q4k_bytes((size_t)t.DQ * D),
             "o does not immediately follow z");
        // Every weight byte offset reaches the DMA, which folds a 1-byte
        // element type to u32 and shifts the offset right by two.
        const size_t offs[4] = {0, t.w_o_offset, t.w_upgate_offset, t.w_down_offset};
        for (size_t o : offs)
            want(o % 4 == 0, "a weight offset is not a multiple of 4");
        want(DI % 512 == 0, "intermediate_size is not a whole weave band");

        char buf[220];
        std::snprintf(buf, sizeof buf,
                      "4 shapes, 7 row counts, 2 adjacencies, 4 offsets "
                      "(proj_bytes %.1f MB)", t.proj_bytes / 1048576.0);
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

    npu_app_manager* pm      = mgr.register_xclbin(proj_xclbin);
    npu_app_manager* pre_mgr = mgr.register_xclbin(prefill_xclbin);
    npu_app_manager* ver_mgr = mgr.register_xclbin(verify_xclbin);

    std::printf("-- staging one attention layer (%.0f MB) --\n",
                t.proj_bytes / 1048576.0);
    q38npu::attn_engine    eng(ver_mgr->create_app(), t, q38npu::attn_rtp_addresses{});
    q38npu::npu_attn_layer dev_layer(&eng, q4nx, kLayer);
    q38npu::prefill_attn_op op(pre_mgr->create_app(), t,
                               q38npu::prefill_attn_rtp_addresses{});
    q38npu::prefill_proj   proj(pm, t, q38npu::dequant_mm_rtp_addresses{});
    buffer<u8>&            W = dev_layer.weights();

    // The CPU sides. `half` carries the weights and none of the 32 MB cache --
    // only its const elementwise helpers are called, exactly as the engine does
    // it; `cpu` is the scored reference and keeps its own cache.
    cpu_ref::attention_layer cpu(q4nx, kLayer, t.MAX_L);
    cpu_ref::attention_layer half(q4nx, kLayer, /*max_len=*/0);

    // The layer's own tensors, for the same-bytes references. `get_q4nx`
    // returns a view into the mmap -- no copy, and the exact bytes the BO was
    // staged from.
    const std::string pl = "model.layers." + std::to_string(kLayer) + ".";
    const cpu_ref::q4nx_tensor W_q    = cpu_ref::get_q4nx(q4nx, pl + "self_attn.q_proj.weight");
    const cpu_ref::q4nx_tensor W_k    = cpu_ref::get_q4nx(q4nx, pl + "self_attn.k_proj.weight");
    const cpu_ref::q4nx_tensor W_v    = cpu_ref::get_q4nx(q4nx, pl + "self_attn.v_proj.weight");
    const cpu_ref::q4nx_tensor W_o    = cpu_ref::get_q4nx(q4nx, pl + "self_attn.o_proj.weight");
    const cpu_ref::q4nx_tensor W_up   = cpu_ref::get_q4nx(q4nx, pl + "mlp.up_proj.weight");
    const cpu_ref::q4nx_tensor W_gate = cpu_ref::get_q4nx(q4nx, pl + "mlp.gate_proj.weight");
    const cpu_ref::q4nx_tensor W_down = cpu_ref::get_q4nx(q4nx, pl + "mlp.down_proj.weight");

    // ======================================================================
    // 2. the weight layout, by memcmp
    // ======================================================================
    // The claim the two fused dispatches rest on: reorder_cpy's permutation
    // never crosses a 64-weight-row boundary, so staging q, k, v and z through
    // four separate calls produces the same bytes as one call over their
    // concatenation -- which is what the op will read. Same for the up|gate
    // weave, which the staging lays down 512 rows at a time.
    //
    // These are the two checks in the suite that can fail with no hardware
    // involved, and whose failure invalidates everything below rather than
    // merely joining it.
    {
        const size_t qb = q38npu::q4k_bytes((size_t)t.DQ * D);
        const size_t kb = q38npu::q4k_bytes((size_t)t.DK * D);
        const size_t vb = q38npu::q4k_bytes((size_t)t.DV * D);
        const size_t total = 2 * qb + kb + vb;

        // q, then k, then v, then the output gate -- which is the second half
        // of the SAME q_proj tensor, [q | z] contiguous in the checkpoint.
        const uint8_t* wq = q4nx.raw(pl + "self_attn.q_proj.weight");
        std::vector<uint8_t> cat(total);
        std::memcpy(cat.data(),                wq,                                qb);
        std::memcpy(cat.data() + qb,           q4nx.raw(pl + "self_attn.k_proj.weight"), kb);
        std::memcpy(cat.data() + qb + kb,      q4nx.raw(pl + "self_attn.v_proj.weight"), vb);
        std::memcpy(cat.data() + qb + kb + vb, wq + qb,                           qb);

        std::vector<uint8_t> wantb(total);
        q38npu::reorder_cpy(wantb.data(), cat.data(), total, D);

        const bool same = std::memcmp(wantb.data(), W.data(), total) == 0;
        size_t first = total;
        if (!same)
            for (size_t i = 0; i < total; i++)
                if (wantb[i] != W.data()[i]) { first = i; break; }

        char buf[240];
        if (same)
            std::snprintf(buf, sizeof buf,
                          "%.1f MB of q|k|v|z identical to one reorder_cpy over "
                          "the concatenation", total / 1048576.0);
        else
            std::snprintf(buf, sizeof buf,
                          "diverges at byte %zu of %zu (block %zu, %s)",
                          first, total, first / q38npu::W_BLOCK_BYTES,
                          first < qb ? "in q" : first < qb + kb ? "in k"
                          : first < qb + kb + vb ? "in v" : "in z");
        report("per-tensor staging == concatenated staging", same, buf);
    }
    {
        // The weave, the same way. `stage_attn_proj_weights` alternates one
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
    // Real ones, from the dumped prefill window, produced by the CPU
    // reference's own helpers in the order the layer needs them. A random x
    // would score the arithmetic honestly but would not exercise the dynamic
    // range Q4_K's per-group scales were fitted to.
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
    // Round to what the array will receive.
    for (float& f : x_all) f = cpu_ref::round_bf16(f);

    // mRoPE from the dump's own position ids, which is what the engine builds:
    // this prompt carries an image block, so the three channels diverge and
    // text positions would be a different rotation.
    // (memory: position-ids-dump-is-channel-major)
    std::vector<int32_t> pos_thw = ref.i32("prefill.position_ids");
    if ((int)pos_thw.size() != 3 * P) {
        std::fprintf(stderr, "prefill.position_ids is %zu, want %d\n",
                     pos_thw.size(), 3 * P);
        return 1;
    }
    // Eight more positions for check 13's verify pass, continuing all three
    // channels from the window's own maximum -- what a decode step would use.
    constexpr int VER = 8;
    int pmax = 0;
    for (int32_t p : pos_thw) pmax = std::max(pmax, (int)p);
    std::vector<int32_t> pos_ext((size_t)3 * (P + VER));
    for (int c = 0; c < 3; c++) {
        std::memcpy(pos_ext.data() + (size_t)c * (P + VER),
                    pos_thw.data() + (size_t)c * P, (size_t)P * sizeof(int32_t));
        for (int i = 0; i < VER; i++)
            pos_ext[(size_t)c * (P + VER) + P + i] = pmax + 1 + i;
    }
    std::vector<float> cos_all((size_t)(P + VER) * ROPE_DIM),
                       sin_all((size_t)(P + VER) * ROPE_DIM);
    cpu_ref::mrope_table(pos_ext.data(), P + VER, cos_all.data(), sin_all.data(),
                         /*round_to_bf16=*/true);

    std::printf("   window %d rows, mrope positions 0..%d (+%d for the verify "
                "pass)\n\n", P, pmax, VER);

    // The stage inputs, each scored against ITS OWN input rather than charged
    // for the one upstream. (memory: score-a-projection-against-its-own-input)
    std::vector<float> xn;
    half.in_norm(x_all.data(), P, xn);
    for (float& f : xn) f = cpu_ref::round_bf16(f);

    std::vector<float> qz((size_t)P * 2 * DQ_), kr((size_t)P * DKV_),
                       vr((size_t)P * DKV_);
    cpu_ref::q4nx_matmul(W_q, xn.data(), P, qz.data());
    cpu_ref::q4nx_matmul(W_k, xn.data(), P, kr.data());
    cpu_ref::q4nx_matmul(W_v, xn.data(), P, vr.data());

    // The gated attention row, o's input: through the reference's own core, so
    // o is scored on the activation the live path will hand it rather than on
    // a synthetic one with the wrong dynamic range.
    //
    // `qkv_post_rope` rotates k IN PLACE and normalises q into qh, so both its
    // inputs are copies -- qz and kr are still needed below, un-rotated, as the
    // references for check 3.
    std::vector<float> attn;
    {
        std::vector<float> qg_c = qz, kk_c = kr, qh_c;
        half.qkv_post_rope(qg_c.data(), kk_c.data(), P, cos_all.data(),
                           sin_all.data(), qh_c);
        cpu.reset();
        cpu.seed_cache(kk_c, vr, P);
        cpu.attn_core(qh_c.data(), P, /*base=*/0, attn);
        half.apply_gate(qz.data(), attn.data(), P);
        for (float& f : attn) f = cpu_ref::round_bf16(f);
    }

    std::vector<float> h1((size_t)P * D);
    cpu_ref::q4nx_matmul(W_o, attn.data(), P, h1.data());
    for (size_t i = 0; i < h1.size(); i++) h1[i] += x_all[i];

    std::vector<float> hn;
    half.post_norm(h1.data(), P, hn);
    for (float& f : hn) f = cpu_ref::round_bf16(f);

    std::vector<float> up_ref((size_t)P * DI), gate_ref((size_t)P * DI);
    cpu_ref::q4nx_matmul(W_up,   hn.data(), P, up_ref.data());
    cpu_ref::q4nx_matmul(W_gate, hn.data(), P, gate_ref.data());
    std::vector<float> g((size_t)P * DI);
    half.swiglu(gate_ref.data(), up_ref.data(), P, g.data());
    for (float& f : g) f = cpu_ref::round_bf16(f);

    std::printf("\n-- the four dispatches, each against its own same-bytes "
                "reference --\n");

    // ======================================================================
    // 3. q | k | v | z, one fused dispatch
    // ======================================================================
    {
        std::vector<float> gk, gv, gqg;
        proj.qkv_z(xn.data(), P, W, gk, gv, gqg);

        // Assemble what the device returned in its own staging order, and the
        // reference the same way.
        const int QZ = proj.QKVZ_OUT();
        std::vector<float> got((size_t)P * QZ), want((size_t)P * QZ);
        for (int l = 0; l < P; l++) {
            float* gd = got.data()  + (size_t)l * QZ;
            float* wd = want.data() + (size_t)l * QZ;
            const float* qgr = gqg.data() + (size_t)l * 2 * DQ_;
            const float* qzr = qz.data()  + (size_t)l * 2 * DQ_;
            std::memcpy(gd,                          qgr,                     DQ_ * 4);
            std::memcpy(gd + DQ_,                    gk.data() + (size_t)l * DKV_, DKV_ * 4);
            std::memcpy(gd + DQ_ + DKV_,             gv.data() + (size_t)l * DKV_, DKV_ * 4);
            std::memcpy(gd + DQ_ + 2 * DKV_,         qgr + DQ_,               DQ_ * 4);
            std::memcpy(wd,                          qzr,                     DQ_ * 4);
            std::memcpy(wd + DQ_,                    kr.data() + (size_t)l * DKV_, DKV_ * 4);
            std::memcpy(wd + DQ_ + DKV_,             vr.data() + (size_t)l * DKV_, DKV_ * 4);
            std::memcpy(wd + DQ_ + 2 * DKV_,         qzr + DQ_,               DQ_ * 4);
        }
        score_with_rotation("q|k|v|z: one 14336-row dispatch",
                            "q|k|v|z: rotated by one core column",
                            got, want, P, QZ);

        // Per sub-tensor, because the aggregate above cannot see k and v: they
        // are 2048 of 14336 columns, so q and z carry 86% of the norm and a
        // cosine over the whole matrix reports them. The four numbers are also
        // the ones check 10 needs -- it scores the same k and v after a RoPE
        // and a bf16 cache, and without these it could not say which stage an
        // error came from.
        {
            const char* nm[4] = {"q", "k", "v", "z"};
            const int   w[4]  = {DQ_, DKV_, DKV_, DQ_};
            const int   o[4]  = {0, DQ_, DQ_ + DKV_, DQ_ + 2 * DKV_};
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

        // The ordering control, and the reason this file exists as its own
        // suite rather than a copy of N6's: the DRAFT HEAD stages k|v|q|z, and
        // a prefill_proj that inherited mtp_proj's scatter would read q where
        // k is. Both are 14336-row matrices over the same bytes.
        // (memory: the-rope-tile-consumes-kv-before-q)
        std::vector<float> kvqz((size_t)P * QZ);
        for (int l = 0; l < P; l++) {
            float* d = kvqz.data() + (size_t)l * QZ;
            const float* qzr = qz.data() + (size_t)l * 2 * DQ_;
            std::memcpy(d,                    kr.data() + (size_t)l * DKV_, DKV_ * 4);
            std::memcpy(d + DKV_,             vr.data() + (size_t)l * DKV_, DKV_ * 4);
            std::memcpy(d + 2 * DKV_,         qzr,                          DQ_ * 4);
            std::memcpy(d + 2 * DKV_ + DQ_,   qzr + DQ_,                    DQ_ * 4);
        }
        report_control("q|k|v|z: read as the draft head's k|v|q|z",
                       cosine(got.data(), kvqz.data(), got.size()));

        // And the scatter itself: qkv_z hands back [q | gate] contiguously,
        // which is the layout cpu_ref's own q4nx_matmul(w_.q, ...) produces and
        // therefore what apply_gate's offset indexes into. A scatter that split
        // q and gate the other way would still score above, because the fused
        // vector is assembled from the same pieces.
        report_cos("q|k|v|z: qg is [q | gate], not [gate | q]",
                   cosine(gqg.data(), qz.data(), gqg.size()), kGate);
        std::vector<float> swapped((size_t)P * 2 * DQ_);
        for (int l = 0; l < P; l++) {
            const float* s = qz.data() + (size_t)l * 2 * DQ_;
            float* d = swapped.data() + (size_t)l * 2 * DQ_;
            std::memcpy(d,       s + DQ_, DQ_ * 4);
            std::memcpy(d + DQ_, s,       DQ_ * 4);
        }
        report_control("q|k|v|z: qg halves swapped",
                       cosine(gqg.data(), swapped.data(), gqg.size()));
    }

    // ======================================================================
    // 4. o_proj -- and the Din that is not D
    // ======================================================================
    {
        std::vector<float> got((size_t)P * D, 0.0f);
        proj.o(attn.data(), P, W, got.data());
        std::vector<float> want((size_t)P * D);
        cpu_ref::q4nx_matmul(W_o, attn.data(), P, want.data());
        score_with_rotation("o: y = W_o . gated_attn  (Din = DQ = 6144)",
                            "o: rotated by one core column", got, want, P, D);

        // The trap reorder_cpy's `col` parameter carries, made explicit: o
        // reduces over the gated row, not over D. A [5120][5120] matrix at the
        // same offset is a legal shape over bytes that are really [5120][6144],
        // and nothing between here and the cores notices.
        q38npu::dequant_mm_op wrong(pm->create_app(), D, D, t.w_o_offset);
        std::vector<float> bad((size_t)P * D, 0.0f);
        wrong.run(attn.data(), P, W, bad.data());   // attn's first D columns
        report_control("o: read as Din = D instead of DQ",
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

    // A one-pass op over the same matrix: today's sequence, activation off. It
    // is the baseline every sub-check below is measured against.
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
    // holding the previous tile. So the bar is EQUALITY, not a cosine -- the
    // two sequences ask the device for the same arithmetic in the same order
    // and bf16 is deterministic.
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
    // the activation alone. Reported at a loose bar on purpose: the gate that
    // matters is check 9, and this exists to say what check 9 is paying.
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
        std::vector<float> want((size_t)P * DI);
        half.swiglu(gate_ref.data(), up_ref.data(), P, want.data());
        report_cos("up|gate: fused to silu(gate) * up",
                   cosine(got.data(), want.data(), got.size()), kGate);

        // If the un-weave read one band twice, the scores above would still be
        // whatever that band deserved. Both degenerate readings must collapse.
        std::vector<float> gg((size_t)P * DI), uu((size_t)P * DI);
        half.swiglu(gate_ref.data(), gate_ref.data(), P, gg.data());
        half.swiglu(up_ref.data(),   up_ref.data(),   P, uu.data());
        report_control("up|gate: silu(gate) * gate instead",
                       cosine(got.data(), gg.data(), got.size()));
        report_control("up|gate: silu(up) * up instead",
                       cosine(got.data(), uu.data(), got.size()));

        // And the SwiGLU orientation, which is the one bug this design can
        // plausibly have: silu(up) * gate is just as plausible a reading and
        // scores nearly as well. (memory: score-swiglu-by-swapping-up-and-gate)
        std::vector<float> wrong((size_t)P * DI);
        half.swiglu(up_ref.data(), gate_ref.data(), P, wrong.data());
        report_control("up|gate: silu(up) * gate instead",
                       cosine(got.data(), wrong.data(), got.size()));
    }

    // ---- the parity swap, from the device side ----------------------------
    // `{0,2,silu}, {1,2,none}` silu's the UP bands instead. Both halves are the
    // same shape, so nothing about the dispatch complains; only the arithmetic
    // says which parity the RTP write landed on. Un-weave it exactly as
    // `up_gate_swiglu` does -- odd band times even band -- so the only
    // difference from the production path is which band got the activation.
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
        std::vector<float> want((size_t)P * DI);
        half.swiglu(gate_ref.data(), up_ref.data(), P, want.data());
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
    // `dequant_mm_op` pins L at 512 and tiles a window on the host, because
    // every DDR term in the engine's cost is linear in L and a wide dispatch
    // amortises nothing. That is only sound if a row's result depends on its
    // own x row and nothing else -- so the same rows, computed as part of a
    // two-tile window and as part of a ragged one, must come back BIT FOR BIT
    // identical. Anything less means a pad row bled into a real one, or a
    // per-tile offset is wrong.
    //
    // Bit-for-bit and not by cosine on purpose: a tail-tile pad row leaking
    // into row 2 of a 3-row window moves that row a long way, but a cosine over
    // 1024 x 5120 values would never see it.
    //
    // Run on `o`, whose Din is the one that is not D: a per-tile x offset
    // computed with the wrong stride is invisible on a square-ish matrix.
    {
        constexpr int TMAX = 1024;
        std::vector<float> x((size_t)TMAX * DQ_);
        fill_synth(x, 0x51b0c0de);

        std::vector<float> full((size_t)TMAX * D, 0.0f);
        proj.o(x.data(), TMAX, W, full.data());

        std::string bad;
        for (int T : {1, 3, 128, 221, 511, 512, 513, 1000}) {
            std::vector<float> part((size_t)T * D, 0.0f);
            proj.o(x.data(), T, W, part.data());
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
    // Checks 3-7 all ran against a BO allocated from the attention_layer app,
    // in dispatches belonging to dequant_mm's. BOs are device-scoped, so that
    // is legal -- but "legal" here means the dispatch completes, and a BO from
    // the wrong place does not fault, it simply never completes or reads stale
    // bytes. So: the same matrix, copied into a BO from dequant_mm's OWN app,
    // must give the identical answer.
    {
        npu_app      a  = pm->create_app();
        const size_t ob = q38npu::q4k_bytes((size_t)D * t.DQ);

        buffer<u8> priv = a.create_bo_buffer<u8>(ob);
        std::memcpy(priv.data(), W.data() + t.w_o_offset, ob);
        priv.sync_to_device();

        q38npu::dequant_mm_op own(pm->create_app(), D, t.DQ, 0);
        std::vector<float> a_out((size_t)P * D, 0.0f), b_out((size_t)P * D, 0.0f);
        own.run(attn.data(), P, priv, a_out.data());
        proj.o(attn.data(), P, W, b_out.data());

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
    std::printf("\n-- the hybrid layer: cpu_ref's elementwise halves around "
                "five dispatches --\n");

    // `prefill_attn_block::forward` rebuilt here from the pieces, because the
    // engine only hands back a hidden state and this is the one place each
    // stage can be seen. It is NOT a second transcription of the production
    // body for scoring purposes -- what it is scored against is
    // `cpu_ref::attention_layer::forward()`, which shares every elementwise
    // helper with it.
    //
    // \param keep_o_residual  check 12's sabotage switch.
    //
    // Leaves the rows it handed to `append_rows` in fed_k / fed_v, so check 10
    // can ask what the CACHE did to them separately from what the projection
    // did. Without that split, one number would be charged for two stages and
    // a bf16 storage loss would be indistinguishable from a bad dispatch.
    std::vector<float> fed_k, fed_v;
    auto run_hybrid = [&](const float* x, int L, float* y, bool keep_o_residual) {
        dev_layer.reset();

        std::vector<float> lxn;
        half.in_norm(x, L, lxn);

        std::vector<float> kk, vv, qg, qh;
        proj.qkv_z(lxn.data(), L, W, kk, vv, qg);
        half.qkv_post_rope(qg.data(), kk.data(), L, cos_all.data(),
                           sin_all.data(), qh);

        // The cursor BEFORE the append: this window's L_begin, and the core
        // masks on absolute position. The post-append value would let every
        // query attend to its own future.
        const int base = dev_layer.tokens_seen();
        dev_layer.append_rows(kk.data(), vv.data(), L);
        fed_k = kk;
        fed_v = vv;

        std::vector<float> a((size_t)L * DQ_, 0.0f);
        op.run(qh.data(), L, base, dev_layer.cache(), a.data());
        half.apply_gate(qg.data(), a.data(), L);

        std::vector<float> lh1((size_t)L * D, 0.0f);
        proj.o(a.data(), L, W, lh1.data());
        if (keep_o_residual)
            for (size_t i = 0; i < lh1.size(); i++) lh1[i] += x[i];

        std::vector<float> lhn;
        half.post_norm(lh1.data(), L, lhn);
        std::vector<float> lg;
        proj.up_gate_swiglu(lhn.data(), L, W, lg);
        proj.down(lg.data(), L, W, y);
        for (size_t i = 0; i < (size_t)L * D; i++) y[i] += lh1[i];
    };

    // The reference: the same layer, entirely on the host, through the very
    // helpers the path above calls.
    cpu.reset();
    std::vector<float> y_ref((size_t)P * D), y_hyb((size_t)P * D);
    cpu.forward(x_all.data(), P, cos_all.data(), sin_all.data(), y_ref.data());

    // ======================================================================
    // 9. one hybrid layer against the CPU reference
    // ======================================================================
    {
        run_hybrid(x_all.data(), P, y_hyb.data(), /*keep_o_residual=*/true);

        // Two statistics, because they answer different questions and neither
        // substitutes for the other:
        //
        //   RELATIVE L2 is the magnitude. The error here is a flat relative
        //   perturbation -- bfp16 in the mmul, bf16 on every readback -- so
        //   this is the number that has a T-independent bar.
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
        char buf[240];
        std::snprintf(buf, sizeof buf,
                      "cos %.6f [0.999], rel L2 %.5f [0.06], worst row %.6f "
                      "(row %d) [0.99] -- L = %d in one attention dispatch",
                      c, e, worst, worst_row, P);
        report("hybrid layer vs cpu_ref::forward",
               c >= 0.999 && e < 0.06 && worst >= 0.99, buf);

        // What the same window costs with the projections on the host: N4's
        // path, and the difference between the two numbers is what moving the
        // matmuls onto the device is worth in accuracy terms. Reported, not
        // gated -- it is the baseline, not a requirement.
        std::vector<float> y_cpu((size_t)P * D);
        {
            std::vector<float> cqg, ckk, cvv, cqh;
            dev_layer.reset();
            half.project_qkv_rope(x_all.data(), P, cos_all.data(), sin_all.data(),
                                  cqg, ckk, cvv, cqh);
            dev_layer.append_rows(ckk.data(), cvv.data(), P);
            std::vector<float> a((size_t)P * DQ_, 0.0f);
            op.run(cqh.data(), P, 0, dev_layer.cache(), a.data());
            half.gate_out_mlp(cqg.data(), a.data(), x_all.data(), P, y_cpu.data());
        }
        std::printf("         (host projections, same core: cos %.6f, "
                    "rel L2 %.5f)\n",
                    cosine(y_cpu.data(), y_ref.data(), y_cpu.size()),
                    rel_l2(y_cpu.data(), y_ref.data(), y_cpu.size()));
    }

    // ======================================================================
    // 10. the k / v rows that reached the cache BO
    // ======================================================================
    // Scored separately from the layer output because the output cannot see a
    // cache addressing error clearly: each query's own row dominates its
    // softmax and dilutes the error before it reaches y.
    //
    // Two bars again, and for the same reason as check 9. The rel-L2 one is
    // against a SAME-BYTES reference over this branch's own bf16-rounded xn,
    // carried through the reference's own RoPE -- k is rotated on the way into
    // the cache, so the reference has to take the same turn before it can be
    // compared to what is in the BO.
    {
        run_hybrid(x_all.data(), P, y_hyb.data(), true);   // refill the cache

        const std::vector<float> k_dev = dev_layer.read_k(0, P);
        const std::vector<float> v_dev = dev_layer.read_v(0, P);

        double wk = 2.0, wv = 2.0; int wk_row = -1, wv_row = -1;
        for (int r = 0; r < P; r++) {
            const double ck = cosine(k_dev.data() + (size_t)r * DKV_,
                                     cpu.cache_k().data() + (size_t)r * DKV_,
                                     (size_t)DKV_);
            const double cv = cosine(v_dev.data() + (size_t)r * DKV_,
                                     cpu.cache_v().data() + (size_t)r * DKV_,
                                     (size_t)DKV_);
            if (ck < wk) { wk = ck; wk_row = r; }
            if (cv < wv) { wv = cv; wv_row = r; }
        }

        std::vector<float> rk_ref((size_t)P * DKV_), rv_ref((size_t)P * DKV_);
        cpu_ref::q4nx_matmul(W_k, xn.data(), P, rk_ref.data());
        cpu_ref::q4nx_matmul(W_v, xn.data(), P, rv_ref.data());
        {
            std::vector<float> qg_copy = qz, qh_dummy;
            half.qkv_post_rope(qg_copy.data(), rk_ref.data(), P, cos_all.data(),
                               sin_all.data(), qh_dummy);
        }
        const double ek = rel_l2(k_dev.data(), rk_ref.data(), k_dev.size());
        const double ev = rel_l2(v_dev.data(), rv_ref.data(), v_dev.size());

        // The same two rows against WHAT WAS HANDED TO append_rows, which is
        // the only part of this check the cache is responsible for. The
        // projection's own error is common mode here, so what is left is the
        // round trip: fp32 -> bf16 -> the BO -> back. The plane stride, the
        // apron and the row base all live in that round trip, and any of them
        // being wrong moves this number a long way rather than a little.
        const double sk = rel_l2(k_dev.data(), fed_k.data(), k_dev.size());
        const double sv = rel_l2(v_dev.data(), fed_v.data(), v_dev.size());

        char buf[300];
        std::snprintf(buf, sizeof buf,
                      "storage k %.5f v %.5f [0.005]; end-to-end k %.5f v %.5f; "
                      "worst row k %.6f (r%d) v %.6f (r%d) [0.99]; cursor %d",
                      sk, sv, ek, ev, wk, wk_row, wv, wv_row,
                      dev_layer.tokens_seen());
        report("k/v survive the cache round trip",
               sk < 0.005 && sv < 0.005 && wk >= 0.99 && wv >= 0.99 &&
               dev_layer.tokens_seen() == P && cpu.cache_len() == P, buf);
    }

    // ======================================================================
    // 11. cosine by causal depth
    // ======================================================================
    // A prefill window's error grows with how many keys a row attends over, so
    // the tail scores worse than the head and that is arithmetic, not
    // addressing. Segmenting says which: a monotone decay is the expected
    // shape, and a single segment out of line is a chunk boundary.
    // (memory: prefill-attention-error-grows-with-kv-blocks)
    {
        const int SEG = 64;
        double worst = 2.0; int worst_seg = -1;
        std::printf("         by causal depth:");
        for (int s0 = 0; s0 < P; s0 += SEG) {
            const int n = std::min(SEG, P - s0);
            const double c = cosine(y_hyb.data() + (size_t)s0 * D,
                                    y_ref.data() + (size_t)s0 * D,
                                    (size_t)n * D);
            std::printf("  [%d,%d) %.5f", s0, s0 + n, c);
            if (c < worst) { worst = c; worst_seg = s0; }
        }
        std::printf("\n");
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "worst segment %.6f at row %d (gate 0.999)", worst, worst_seg);
        report("every causal segment clears the gate", worst >= 0.999, buf);
    }

    // ======================================================================
    // 12. the sabotage
    // ======================================================================
    // Check 9's gate has to be able to fail. The attention residual is the
    // cheapest thing to drop and the hardest to see: y is still the right
    // shape, still built from the right weights, and every dispatch above
    // still scores 0.999.
    {
        std::vector<float> y_bad((size_t)P * D);
        run_hybrid(x_all.data(), P, y_bad.data(), /*keep_o_residual=*/false);
        const double c = cosine(y_bad.data(), y_ref.data(), y_bad.size());
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "cos %.6f without the o residual (check 9 was %.6f)",
                      c, cosine(y_hyb.data(), y_ref.data(), y_hyb.size()));
        report("dropping the o residual breaks check 9", c < 0.999, buf);
        run_hybrid(x_all.data(), P, y_hyb.data(), true);   // restore the cache
    }

    // ======================================================================
    // 13. cache continuity: this path wrote, the other bitstream reads
    // ======================================================================
    // The integration gate. The projections ran on dequant_mm, the rows went
    // into the BO through append_rows, and a verify pass now reads them through
    // a THIRD bitstream's replay on the same BO and the same cursor. If any two
    // of the three disagree about the layout -- the plane base, the apron, the
    // cursor -- this is where it shows and nowhere earlier.
    {
        char buf[200];
        std::snprintf(buf, sizeof buf, "device %d, cpu %d",
                      dev_layer.tokens_seen(), cpu.cache_len());
        report("cursors agree after the prefill window",
               dev_layer.tokens_seen() == P && cpu.cache_len() == P, buf);

        std::snprintf(key, sizeof key, "verify.layer_%02d_out", kLayer - 1);
        std::vector<float> xv;
        try {
            xv = ref.f32(key);
        } catch (const std::exception&) {
            xv.assign(x_all.end() - (size_t)VER * D, x_all.end());
        }
        xv.resize((size_t)VER * D);
        for (float& f : xv) f = cpu_ref::round_bf16(f);

        const float* cv = cos_all.data() + (size_t)P * ROPE_DIM;
        const float* sv = sin_all.data() + (size_t)P * ROPE_DIM;

        std::vector<float> yv_ref((size_t)VER * D), yv_dev((size_t)VER * D);
        cpu.forward(xv.data(), VER, cv, sv, yv_ref.data());
        dev_layer.forward(xv.data(), VER, cv, sv, yv_dev.data());

        report_cos("an L=8 verify pass on the prefilled cache",
                   cosine(yv_dev.data(), yv_ref.data(), yv_dev.size()), 0.99,
                   "-- a third bitstream, the same BO");
        std::snprintf(buf, sizeof buf, "device %d, cpu %d (want %d)",
                      dev_layer.tokens_seen(), cpu.cache_len(), P + VER);
        report("cursors agree after it",
               dev_layer.tokens_seen() == P + VER && cpu.cache_len() == P + VER,
               buf);
    }

    // ======================================================================
    // 14. throughput
    // ======================================================================
    // Not a gate -- it is the number this phase exists to produce. Both paths
    // read the whole 209 MB of weights at any width, so the device should win
    // from a handful of rows; that is a prediction, and this is what refutes
    // it. The hybrid prefill path only fires at L >= 128, so the crossover
    // matters only as a sanity check on the choice.
    //
    // The CPU column stops at 512: a host pass over 372 M parameters is ~0.4
    // TFLOP per 512 rows and says nothing new above that.
    {
        std::printf("\n-- one layer's projections, ms (4 dispatches per 512-row "
                    "tile) --\n");
        std::printf("   %6s  %10s  %10s  %9s  %9s\n",
                    "L", "NPU", "CPU", "NPU us/row", "speedup");

        for (int T : {1, 128, 221, 512, 1024}) {
            std::vector<float> xs((size_t)T * D), as((size_t)T * DQ_),
                               gs((size_t)T * DI);
            fill_synth(xs, 0xbeef ^ (uint64_t)T);
            fill_synth(as, 0xfeed ^ (uint64_t)T);
            fill_synth(gs, 0xf00d ^ (uint64_t)T);
            std::vector<float> y1((size_t)T * D), k2, v2, q2, g2;

            auto t0 = std::chrono::steady_clock::now();
            proj.qkv_z(xs.data(), T, W, k2, v2, q2);
            proj.o(as.data(), T, W, y1.data());
            proj.up_gate_swiglu(xs.data(), T, W, g2);
            proj.down(gs.data(), T, W, y1.data());
            const double npu = ms_since(t0);

            double cpu_ms = 0.0;
            if (T <= 512) {
                std::vector<float> o1((size_t)T * D), o2((size_t)T * 2 * DQ_),
                                   o3((size_t)T * DKV_), o4((size_t)T * DKV_),
                                   o5((size_t)T * DI), o6((size_t)T * DI);
                auto c0 = std::chrono::steady_clock::now();
                cpu_ref::q4nx_matmul(W_q,    xs.data(), T, o2.data());
                cpu_ref::q4nx_matmul(W_k,    xs.data(), T, o3.data());
                cpu_ref::q4nx_matmul(W_v,    xs.data(), T, o4.data());
                cpu_ref::q4nx_matmul(W_o,    as.data(), T, o1.data());
                cpu_ref::q4nx_matmul(W_up,   xs.data(), T, o5.data());
                cpu_ref::q4nx_matmul(W_gate, xs.data(), T, o6.data());
                cpu_ref::q4nx_matmul(W_down, gs.data(), T, o1.data());
                // The host SwiGLU the fused readback removes, which the NPU
                // column above never pays.
                half.swiglu(o6.data(), o5.data(), T, o5.data());
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
