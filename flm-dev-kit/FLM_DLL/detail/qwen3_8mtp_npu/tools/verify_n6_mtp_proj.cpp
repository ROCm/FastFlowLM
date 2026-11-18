// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file verify_n6_mtp_proj.cpp
/// \brief N6 -- the MTP draft head's step-0 projections on `dequant_mm.xclbin`:
///        fused Q4_K dequant + GEMM, over the head's existing weight BO.
///
/// Build:
///   source ~/setup.sh
///   g++-13 -O2 -march=native -fopenmp -std=c++20
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n6_mtp_proj.cpp speculate/npu/proj_sequence.cpp
///       prefill/npu/prefill_attn_sequence.cpp speculate/npu/mtp_sequence.cpp
///       speculate/mtp_head.cpp ../q4nx/q4_npu_eXpress.cpp
///       ../q4nx/safe_tensors.cpp
///       -L../../external_libs -L../../build/lib -L/opt/xilinx/xrt/lib
///       -lxrt_coreutil -laiebu -luuid -lpthread -o /tmp/verify_n6_mtp_proj
///   (one line; the continuations are dropped here because a backslash at the
///    end of a // comment splices the next line into it)
///
/// Takes all three xclbins as FILES. Runs ~8 minutes: three head layers are
/// staged (~250 MB each) and the CPU builds a same-bytes reference for every
/// one of the head's 424 M parameters several times over.
///
/// ---------------------------------------------------------------------------
/// What moved
/// ---------------------------------------------------------------------------
/// N5 put step 0's ATTENTION CORE on `attn.xclbin` and left the seven
/// projections -- fc, q|z, k, v, o, up, gate, down -- on the host, which is
/// much the larger share of the FLOPs. Those now run on `dequant_mm.xclbin` as
/// FIVE dispatches per 512-row tile, because k|v|q|z is one fused matrix and
/// the woven up|gate another.
///
/// The five read `npu_mtp_layer`'s existing 245 MB weight BO at five byte
/// offsets. Nothing is restaged and nothing is copied, and that is the load
/// bearing claim of the whole design: `stage_mtp_proj_weights` lays the head's
/// tensors down through `reorder_cpy(..., vertical_blocks = 2)`, which is the
/// same permutation dequant_mm's own host driver applies, and that permutation
/// is LOCAL TO 64 WEIGHT ROWS. Every sub-tensor here is a multiple of 64 rows,
/// so reordering each separately is byte for byte reordering the concatenation.
/// It is an argument, not a measurement -- check 2 turns it into one.
///
/// ---------------------------------------------------------------------------
/// How the dispatches are scored
/// ---------------------------------------------------------------------------
/// Against a reference built from the SAME QUANTIZED BYTES, through
/// `cpu_ref::q4nx_matmul` on a bf16-rounded x. Q4_K's own ~0.997 loss is then
/// common mode and cancels, so the bar is 0.999 and what is left to measure is
/// the device's arithmetic: `aie::mmul<...,bf16>` on npu2 floors both operands
/// onto a shared-exponent-per-8 grid, and every fp32 -> bf16 on the way out
/// truncates toward -inf. That is why the bar is 0.999 and not 0.9999.
/// (memory: score-dequant-mm-against-its-own-quantized-bytes,
///          aie-mmul-bf16-truncates-to-bfp16)
///
/// Every score is paired with a control that must COLLAPSE, because a cosine
/// near 1 is evidence about arithmetic and says nothing about addressing:
///
///   rotate by 64        one core column. Catches an off-by-one core mapping,
///                       which a whole-matrix cosine cannot see.
///   q|k|v instead       the checkpoint stages k, then v, then q -- the RoPE
///   of k|v|q            tile consumes kv before q. The op cannot tell.
///                       (memory: the-rope-tile-consumes-kv-before-q)
///   concat instead      up|gate is WOVEN 512 output rows at a time, not
///   of weave            concatenated. Both are legal 34816-row matrices.
///   Din = D for o       o reduces over the gated row (6144), not over D. A
///                       matrix of identical byte count, strided wrong.
///   silu(up) * gate     just as plausible a reading of a SwiGLU, and scores
///                       nearly as well. (memory: score-swiglu-by-swapping-
///                       up-and-gate)
///
/// ---------------------------------------------------------------------------
/// The two halves
/// ---------------------------------------------------------------------------
///   checks 1-9    the operator, driven directly. Geometry, the weight-layout
///                 memcmp, the five dispatches with their controls, tiling,
///                 and the cross-app BO.
///   checks 10-14  the composition: step 0 against `cpu_ref::mtp_head::step()`,
///                 the SwiGLU control, stale rows, the seven-step draft loop
///                 by argmax margin, and the throughput this phase exists to
///                 produce.
///
/// There used to be a check 14 for the `FLM_Q38_MTP_PROJ=0` A/B, and a guard
/// that the projection path had opened at all. Both are gone with the switch:
/// `dequant_mm.xclbin` is a required argument to `speculate_npu` now, and a
/// driver cannot build the arm they compared against. What is still scored
/// here, and is the better half of what that A/B bought, is the device body
/// against `cpu_ref::mtp_head`'s own helpers -- the two share every norm, the
/// RoPE, the gate and the SwiGLU, so a disagreement is the seven matmuls.
///
/// The second half is the gate. Every dispatch can score 0.9999 while
/// `_prefill_step_npu_proj` scatters k|v|q|z wrong, or feeds o the ungated
/// attention, or drops the residual -- and none of those survives a drafted id
/// sequence.
#include "speculate/speculate_npu.hpp"
#include "speculate/speculate_cpu.hpp"
#include "speculate/mtp_head.hpp"
#include "speculate/npu/mtp_proj.hpp"
#include "speculate/npu/mtp_weights.hpp"
#include "prefill/npu/prefill_attn_op.hpp"
#include "verify/npu/q4nx.hpp"
#include "common/cpu_ref_common.hpp"
#include "common/embedding.hpp"
#include "common/lm_head.hpp"
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

/// Room for the widest composition check plus the dump's 221 rows. The tiling
/// checks do not touch the cache, so this does not have to reach 4096.
constexpr int kMaxL = 1024;

/// The head's rows start at position 1 -- step 0 feeds [input_ids[1:], x_P] at
/// positions 1..P. (README section 5.3: do not "fix" the off-by-one)
constexpr int kPos0 = 1;

/// The bar for a same-bytes score. See the file header: Q4_K cancels, so what
/// is left is the bfp16 mmul and the bf16 readback.
constexpr double kGate = 0.999;

/// The ceiling a control must fall under. Generous on purpose -- a control
/// that lands at 0.98 has still told us the permutation is wrong, and pinning
/// it tighter would only make the suite brittle against the next checkpoint.
constexpr double kControl = 0.99;

/// When a drafted id is evidence, in units of the logit distribution's own
/// standard deviation. Same derivation as N5's, one term larger: the
/// projections now carry bf16 rounding too, so step 0's norm_out sits a little
/// further from the reference than it did with only the attention core moved.
/// (memory: score-a-draft-loop-by-argmax-margin)
constexpr double kDecisive = 0.15;

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-46s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
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

/// Relative L2, which is what a flat additive error should be judged on.
/// A per-row cosine on rows of unequal norm reports the norm, not the error.
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

double report_cos(const char* tag, double cos, double gate) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "cos = %.8f  (gate %.4f)", cos, gate);
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

std::string id_list(const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++)
        s += (i ? " " : "") + std::to_string(v[i]);
    return s;
}

/// How decisive each of the reference's own argmaxes was, per draft step.
///
/// Verbatim in intent from N5: comparing drafted ids across two arithmetics is
/// only a test at steps where the argmax was actually decided. Drafting is a
/// 7-step autoregression, so every step's error is fed back through the head
/// six more times; at a step whose top two logits are a hair apart, which id
/// falls out reports the last bit of the accumulator.
/// (memory: score-a-draft-loop-by-argmax-margin,
///          qwen38-lm-head-untied-and-argmax-ties)
std::vector<double> draft_margins(cpu_ref::mtp_head& m, cpu_ref::lm_head& head,
                                  int V, const std::vector<int>& ids,
                                  const float* prev_hidden, int T, int pos0,
                                  int k, std::vector<int>& out_ids) {
    const int            D = MODEL_DIM;
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

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

} // namespace

// ==========================================================================
int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string mtp_xclbin =
        argc > 2 ? argv[2]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "MTP_decoding/build/QWEN3_8_27B/xclbins/MTP_layer.xclbin";
    const std::string prefill_xclbin =
        argc > 3 ? argv[3]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "gated_attention_prefill/build/xclbins/attn.xclbin";
    const std::string proj_xclbin =
        argc > 4 ? argv[4]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "dequant_mm/build/xclbins/dequant_mm.xclbin";
    const std::string lm_xclbin =
        argc > 5 ? argv[5]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "lm_head_npu_bin/build/QWEN3_8_27B/xclbins/lm_head.xclbin";

    const int D = MODEL_DIM, DKV_ = DKV, DQ_ = DQ, DI = INTERMEDIATE_SIZE;

    std::printf("== N6: the MTP head's step-0 projections on dequant_mm ==\n");
    std::printf("   model    %s\n   decode   %s\n   prefill  %s\n   proj     %s\n"
                "   lm_head  %s\n\n",
                model_dir.c_str(), mtp_xclbin.c_str(), prefill_xclbin.c_str(),
                proj_xclbin.c_str(), lm_xclbin.c_str());

    const q38npu::mtp_layout t = q38npu::mtp_layout::build(kMaxL);

    // ======================================================================
    // 1. the geometry guard
    // ======================================================================
    // Everything the five dispatches assume about the layout, restated here
    // rather than trusted from mtp_proj's constructor -- that one throws at
    // load, so a violated precondition would abort the engine with a message
    // and nothing pointing at which of the six conditions it was.
    //
    // Each of these is silent if it fails at the wrong layer: a ragged Dout or
    // Din hangs the dispatch; a Din that is not a whole number of 256-wide
    // blocks, or a row count that is not a multiple of 64, makes reorder_cpy
    // fold the buffer's head back in as padding with nothing asserting.
    {
        const int KVQZ = t.qkv_out + t.DQ;
        struct { const char* name; int Dout, Din; } shapes[5] = {
            {"fc",      D,    t.FC_IN},
            {"k|v|q|z", KVQZ, D},
            {"o",       D,    t.DQ},
            {"up|gate", 2*DI, D},
            {"down",    D,    DI},
        };
        std::string bad;
        auto want = [&](bool ok, const std::string& why) {
            if (!ok) bad += (bad.empty() ? "" : ", ") + why;
        };
        for (auto& s : shapes) {
            want(s.Dout % 512 == 0, std::string(s.name) + " Dout " +
                 std::to_string(s.Dout) + " is not a whole array pass");
            want(s.Din % 512 == 0, std::string(s.name) + " Din " +
                 std::to_string(s.Din) + " is not a whole reduction tile");
            want(s.Din % 256 == 0, std::string(s.name) +
                 " Din is not a whole q4_k column block");
        }
        // The 64-row locality is per TENSOR -- 2*DI passing says nothing about
        // up and gate individually, and KVQZ passing says nothing about k.
        const int rows[8] = {D, t.DK, t.DV, t.DQ, t.DQ, D, DI, D};
        const char* rn[8] = {"fc", "k", "v", "q", "z", "o", "up/gate", "down"};
        for (int i = 0; i < 8; i++)
            want(rows[i] % 64 == 0, std::string(rn[i]) + " is " +
                 std::to_string(rows[i]) + " rows, not a multiple of 64");
        // The fused qkv_z dispatch reads across the k|v|q -> z boundary.
        want(t.w_gate_offset ==
                 t.w_qkv_offset + q38npu::q4k_bytes((size_t)t.qkv_out * D),
             "z does not immediately follow k|v|q");
        want(t.w_o_offset ==
                 t.w_gate_offset + q38npu::q4k_bytes((size_t)t.DQ * D),
             "o does not immediately follow z");
        // Every weight byte offset reaches the DMA, which folds a 1-byte
        // element type to u32 and shifts the offset right by two.
        const size_t offs[5] = {t.w_fc_offset, t.w_qkv_offset, t.w_o_offset,
                                t.w_upgate_offset, t.w_down_offset};
        for (size_t o : offs)
            want(o % 4 == 0, "a weight offset is not a multiple of 4");
        want(DI % 512 == 0, "intermediate_size is not a whole weave band");

        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "5 shapes, 8 row counts, 2 adjacencies, 5 offsets "
                      "(proj_bytes %.1f MB)", t.proj_bytes / 1048576.0);
        report("every dispatch's geometry is legal", bad.empty(),
               bad.empty() ? buf : bad);
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
    hid.resize((size_t)P * D);
    for (float& f : hid) f = cpu_ref::round_bf16(f);

    std::printf("   window %d rows, text positions %d..%d\n\n",
                P, kPos0, kPos0 + P - 1);

    cpu_ref::safetensors     q4nx(model_dir + "/model.q4nx");
    cpu_ref::embedding_table embed(q4nx);
    cpu_ref::lm_head         head(q4nx);

    qwen3_8mtp_desc desc;
    desc.D                     = (uint32_t)D;
    desc.vocab_size            = (uint32_t)head.vocab();
    desc.mtp_num_hidden_layers = 1;
    desc.mtp_fc_in             = (uint32_t)cpu_ref::mtp::FC_IN;

    // The head's own tensors, for the same-bytes references. `get_q4nx` returns
    // a view into the mmap -- no copy, and the exact bytes the BO was staged
    // from.
    const std::string pl = "model.mtp.layers.0.";
    const cpu_ref::q4nx_tensor W_fc   = cpu_ref::get_q4nx(q4nx, "model.mtp.fc.weight");
    const cpu_ref::q4nx_tensor W_q    = cpu_ref::get_q4nx(q4nx, pl + "self_attn.q_proj.weight");
    const cpu_ref::q4nx_tensor W_k    = cpu_ref::get_q4nx(q4nx, pl + "self_attn.k_proj.weight");
    const cpu_ref::q4nx_tensor W_v    = cpu_ref::get_q4nx(q4nx, pl + "self_attn.v_proj.weight");
    const cpu_ref::q4nx_tensor W_o    = cpu_ref::get_q4nx(q4nx, pl + "self_attn.o_proj.weight");
    const cpu_ref::q4nx_tensor W_up   = cpu_ref::get_q4nx(q4nx, pl + "mlp.up_proj.weight");
    const cpu_ref::q4nx_tensor W_gate = cpu_ref::get_q4nx(q4nx, pl + "mlp.gate_proj.weight");
    const cpu_ref::q4nx_tensor W_down = cpu_ref::get_q4nx(q4nx, pl + "mlp.down_proj.weight");

    // ======================================================================
    // the speculators
    // ======================================================================
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);

    std::printf("-- staging (two head layers, ~250 MB each, + 1.35 GB) --\n");
    // Not under test here, but `speculate_npu` requires it: the draft head
    // projects on the device or not at all.
    q38npu::npu_lm_head lmh(mgr.register_xclbin(lm_xclbin)->create_app(), q4nx);

    qwen38::speculate_cpu spec_cpu(q4nx, &desc, &embed, &head, kMaxL);
    qwen38::speculate_npu spec_npu(q4nx, &desc, &embed, kMaxL, &mgr,
                                   mtp_xclbin, prefill_xclbin, proj_xclbin,
                                   &lmh);

    q38npu::npu_mtp_layer& layer = spec_npu.layer();
    q38npu::mtp_proj&      proj  = *spec_npu.proj();
    buffer<u8>&            W     = layer.weights();

    // ======================================================================
    // 2. the weight layout, by memcmp
    // ======================================================================
    // The claim the fused k|v|q|z dispatch rests on: reorder_cpy's permutation
    // never crosses a 64-weight-row boundary, so staging k, v, q and z through
    // four separate calls produces the same bytes as one call over their
    // concatenation -- which is what the op will read.
    //
    // This is the one check in the suite that can fail without any hardware
    // being involved, and the one whose failure invalidates checks 4 and 10-14
    // rather than merely joining them.
    {
        const size_t kb = q38npu::q4k_bytes((size_t)t.DK * D);
        const size_t vb = q38npu::q4k_bytes((size_t)t.DV * D);
        const size_t qb = q38npu::q4k_bytes((size_t)t.DQ * D);
        const size_t total = kb + vb + 2 * qb;

        // k, then v, then the whole of q_proj -- which is [q | z] already
        // contiguous in the checkpoint, so three source blocks make four
        // matrices.
        std::vector<uint8_t> cat(total);
        std::memcpy(cat.data(),           q4nx.raw(pl + "self_attn.k_proj.weight"), kb);
        std::memcpy(cat.data() + kb,      q4nx.raw(pl + "self_attn.v_proj.weight"), vb);
        std::memcpy(cat.data() + kb + vb, q4nx.raw(pl + "self_attn.q_proj.weight"), 2 * qb);

        std::vector<uint8_t> want(total);
        q38npu::reorder_cpy(want.data(), cat.data(), total, D);

        const bool same = std::memcmp(want.data(), W.data() + t.w_qkv_offset,
                                      total) == 0;
        size_t first = total;
        if (!same)
            for (size_t i = 0; i < total; i++)
                if (want[i] != W.data()[t.w_qkv_offset + i]) { first = i; break; }

        char buf[220];
        if (same)
            std::snprintf(buf, sizeof buf,
                          "%.1f MB of k|v|q|z identical to one reorder_cpy over "
                          "the concatenation", total / 1048576.0);
        else
            std::snprintf(buf, sizeof buf,
                          "diverges at byte %zu of %zu (block %zu, %s)",
                          first, total, first / q38npu::W_BLOCK_BYTES,
                          first < kb ? "in k" : first < kb + vb ? "in v"
                          : first < kb + vb + qb ? "in q" : "in z");
        report("per-tensor staging == concatenated staging", same, buf);
    }

    // ======================================================================
    // the activations every dispatch is scored on
    // ======================================================================
    // Real ones, from the dumped window, produced by the CPU reference's own
    // helpers in the order the layer needs them. A random x would score the
    // arithmetic honestly but would not exercise the dynamic range Q4_K's
    // per-group scales were fitted to.
    cpu_ref::mtp_head half(q4nx, &embed, /*max_len=*/0);   // no cache, like the engine's
    cpu_ref::mtp_head cpu(q4nx, &embed, kMaxL);            // the scored reference

    std::vector<float> cos_all, sin_all;
    cpu_ref::text_rope_tables(kPos0, P, cos_all, sin_all);

    cpu_ref::mtp_taps  tp;
    std::vector<float> fc_in;
    half.fuse_pre(ids32.data(), hid.data(), P, tp, fc_in);
    for (float& f : fc_in) f = cpu_ref::round_bf16(f);

    // fc_out on the CPU, so that every later stage's x is the same in both
    // arithmetics and each dispatch is scored against ITS OWN input rather than
    // charged for the one upstream. (memory: score-a-projection-against-its-
    // own-input)
    std::vector<float> fc_out((size_t)P * D);
    cpu_ref::q4nx_matmul(W_fc, fc_in.data(), P, fc_out.data());

    std::vector<float> xn;
    half.in_norm(fc_out.data(), P, xn);
    for (float& f : xn) f = cpu_ref::round_bf16(f);

    std::vector<float> qg((size_t)P * 2 * DQ_), kk((size_t)P * DKV_),
                       vv((size_t)P * DKV_), qh;
    cpu_ref::q4nx_matmul(W_q, xn.data(), P, qg.data());
    cpu_ref::q4nx_matmul(W_k, xn.data(), P, kk.data());
    cpu_ref::q4nx_matmul(W_v, xn.data(), P, vv.data());
    half.qkv_post_rope(qg.data(), kk.data(), P, cos_all.data(), sin_all.data(), qh);

    // The gated attention row, o's input. Through the reference's own core so
    // that o is scored on the activation the live path will hand it.
    cpu.seed_cache(kk, vv, P);
    std::vector<float> attn;
    cpu.attn_core(qh.data(), P, /*base=*/0, attn);
    half.apply_gate(qg.data(), attn.data(), P);
    for (float& f : attn) f = cpu_ref::round_bf16(f);

    std::vector<float> h1((size_t)P * D);
    cpu_ref::q4nx_matmul(W_o, attn.data(), P, h1.data());
    for (size_t i = 0; i < h1.size(); i++) h1[i] += fc_out[i];

    std::vector<float> hn;
    half.post_norm(h1.data(), P, hn);
    for (float& f : hn) f = cpu_ref::round_bf16(f);

    std::vector<float> up_ref((size_t)P * DI), gate_ref((size_t)P * DI);
    cpu_ref::q4nx_matmul(W_up,   hn.data(), P, up_ref.data());
    cpu_ref::q4nx_matmul(W_gate, hn.data(), P, gate_ref.data());
    std::vector<float> g((size_t)P * DI);
    half.swiglu(gate_ref.data(), up_ref.data(), P, g.data());
    for (float& f : g) f = cpu_ref::round_bf16(f);

    // ======================================================================
    // 3. fc
    // ======================================================================
    {
        std::vector<float> got((size_t)P * D, 0.0f);
        proj.fc(fc_in.data(), P, W, got.data());
        score_with_rotation("fc: y = W_fc . cat(norm_e, norm_h)",
                            "fc: rotated by one core column", got, fc_out, P, D);
    }

    // ======================================================================
    // 4. k | v | q | z, one fused dispatch
    // ======================================================================
    {
        std::vector<float> gk, gv, gqg;
        proj.qkv_z(xn.data(), P, W, gk, gv, gqg);

        // Assemble what the device returned in its own staging order, and the
        // reference the same way.
        const int KV = proj.KVQZ_OUT();
        std::vector<float> got((size_t)P * KV), want((size_t)P * KV);
        // The reference's q|z comes straight out of q_proj -- rows [0, DQ) are
        // q and [DQ, 2*DQ) the output gate, already de-interleaved by the
        // converter, which is what makes the split a byte offset on the device.
        std::vector<float> qz((size_t)P * 2 * DQ_);
        cpu_ref::q4nx_matmul(W_q, xn.data(), P, qz.data());
        std::vector<float> kr((size_t)P * DKV_), vr((size_t)P * DKV_);
        cpu_ref::q4nx_matmul(W_k, xn.data(), P, kr.data());
        cpu_ref::q4nx_matmul(W_v, xn.data(), P, vr.data());
        for (int l = 0; l < P; l++) {
            float* gd = got.data()  + (size_t)l * KV;
            float* wd = want.data() + (size_t)l * KV;
            std::memcpy(gd,                  gk.data()  + (size_t)l * DKV_,      DKV_ * 4);
            std::memcpy(gd + DKV_,           gv.data()  + (size_t)l * DKV_,      DKV_ * 4);
            std::memcpy(gd + 2 * DKV_,       gqg.data() + (size_t)l * 2 * DQ_, 2 * DQ_ * 4);
            std::memcpy(wd,                  kr.data()  + (size_t)l * DKV_,      DKV_ * 4);
            std::memcpy(wd + DKV_,           vr.data()  + (size_t)l * DKV_,      DKV_ * 4);
            std::memcpy(wd + 2 * DKV_,       qz.data()  + (size_t)l * 2 * DQ_, 2 * DQ_ * 4);
        }
        score_with_rotation("k|v|q|z: one 14336-row dispatch",
                            "k|v|q|z: rotated by one core column",
                            got, want, P, KV);

        // The ordering control. The staging is k, then v, then q, because the
        // RoPE tile consumes kv before q; q|k|v is the attention port's order
        // and just as plausible a reading of "qkv". The op cannot tell the two
        // apart -- both are 14336-row matrices over the same bytes.
        // (memory: the-rope-tile-consumes-kv-before-q)
        std::vector<float> qkv((size_t)P * KV);
        for (int l = 0; l < P; l++) {
            float* d = qkv.data() + (size_t)l * KV;
            std::memcpy(d,                 qz.data() + (size_t)l * 2 * DQ_, DQ_ * 4);
            std::memcpy(d + DQ_,           kr.data() + (size_t)l * DKV_,   DKV_ * 4);
            std::memcpy(d + DQ_ + DKV_,    vr.data() + (size_t)l * DKV_,   DKV_ * 4);
            std::memcpy(d + DQ_ + 2*DKV_,  qz.data() + (size_t)l * 2 * DQ_ + DQ_,
                        DQ_ * 4);
        }
        report_control("k|v|q|z: staged as q|k|v instead",
                       cosine(got.data(), qkv.data(), got.size()));

        // And the scatter itself: qkv_z hands back [q | gate] contiguously,
        // which is the layout cpu_ref's own q4nx_matmul(w_.q, ...) produces and
        // therefore what apply_gate's q_gate_off indexes into. A scatter that
        // split q and gate the other way would still score above, because the
        // fused vector above is assembled from the same three pieces.
        report_cos("k|v|q|z: qg is [q | gate], not interleaved",
                   cosine(gqg.data(), qz.data(), gqg.size()), kGate);
    }

    // ======================================================================
    // 5. o_proj -- and the Din that is not D
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
        npu_app_manager* pm = mgr.register_xclbin(proj_xclbin);
        q38npu::dequant_mm_op wrong(pm->create_app(), D, D, t.w_o_offset);
        std::vector<float> bad((size_t)P * D, 0.0f);
        wrong.run(attn.data(), P, W, bad.data());   // feeding it attn's first D columns
        report_control("o: read as Din = D instead of DQ",
                       cosine(bad.data(), want.data(), bad.size()));
    }

    // ======================================================================
    // 6. up | gate, woven -- and the two-pass RTP write that silu's it
    // ======================================================================
    //
    // The production op runs TWO passes over the same matrix, `{0,2,none}` for
    // up and `{1,2,silu}` for gate, so the kernel's fused activation lands on
    // the gate bands alone. Three things have to be separated, because each
    // can fail while the others look fine:
    //
    //   a  the strided `j` and the mid-sequence drain address the same bytes
    //      a single dense pass does           -- activation held out entirely
    //   b  what the device's bf16 silu costs against an fp32 one
    //   c  the silu landed on the gate bands and not the up bands
    //
    // (a) is scored bit-for-bit and (c) bit-for-bit too, so neither can be
    // paid for by arithmetic slack in (b).
    {
    constexpr size_t BAND = 512;          // the weave grain, = PASS_COLS
    const size_t WD    = 2 * (size_t)DI;
    const size_t bands = (size_t)DI / BAND;

    // A one-pass op over the same matrix: today's sequence, activation off.
    // It is the baseline every sub-check below is measured against.
    npu_app_manager* pm = mgr.register_xclbin(proj_xclbin);
    q38npu::dequant_mm_op raw_op(pm->create_app(), 2 * DI, D, t.w_upgate_offset);
    std::vector<float> raw((size_t)P * WD, 0.0f);
    raw_op.run(hn.data(), P, W, raw.data());
    {
        std::vector<float> woven((size_t)P * WD), concat((size_t)P * WD);
        for (int l = 0; l < P; l++) {
            float* wv = woven.data()  + (size_t)l * WD;
            float* cc = concat.data() + (size_t)l * WD;
            const float* u = up_ref.data()   + (size_t)l * DI;
            const float* g2 = gate_ref.data() + (size_t)l * DI;
            for (size_t p = 0; p < bands; p++) {
                std::memcpy(wv + (2 * p)     * BAND, u  + p * BAND, BAND * 4);
                std::memcpy(wv + (2 * p + 1) * BAND, g2 + p * BAND, BAND * 4);
            }
            std::memcpy(cc,            u,  (size_t)DI * 4);
            std::memcpy(cc + DI,       g2, (size_t)DI * 4);
        }
        report_cos("up|gate: woven at 512 output rows",
                   cosine(raw.data(), woven.data(), raw.size()), kGate);
        report_control("up|gate: concatenated instead of woven",
                       cosine(raw.data(), concat.data(), raw.size()));
    }

    // ---- 6a. the strided j and the mid-sequence drain, activation off -----
    // Two passes that between them cover exactly what one dense pass covers,
    // both asking for `none`. Any difference at all is the split itself: a
    // wrong `w_offset` or `y_offset` under the stride, a ping/pong bank reused
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
        report("up|gate 6a: two none-passes == one dense pass", same, buf);
    }

    // The production op: `{0,2,none}` then `{1,2,silu}`.
    std::vector<float> act((size_t)P * WD, 0.0f);
    proj.upgate_op().run(hn.data(), P, W, act.data());

    // ---- 6b. what the device's bf16 silu costs ----------------------------
    // `getSiluBf16_nonLUT` is x*(tanh(x/2)+1)/2 with three truncating bf16
    // roundings; `cpu_ref::mtp_head::swiglu` computes silu in fp32. Score the
    // odd bands against fp32 silu OF THE RAW ODD BANDS -- not of `gate_ref` --
    // so the matmul's own error is common mode and this number is the
    // activation alone. It is reported at a loose bar on purpose: the gate
    // that matters is check 10, and this exists to say what check 10 is paying.
    {
        std::vector<float> dev, ref;
        dev.reserve((size_t)P * DI); ref.reserve((size_t)P * DI);
        for (int l = 0; l < P; l++)
            for (size_t p = 0; p < bands; p++) {
                const size_t o = (size_t)l * WD + (2 * p + 1) * BAND;
                for (size_t j = 0; j < BAND; j++) {
                    dev.push_back(act[o + j]);
                    const float x = raw[o + j];
                    ref.push_back(x / (1.0f + std::exp(-x)));
                }
            }
        const double c = cosine(dev.data(), ref.data(), dev.size());
        const double e = rel_l2(dev.data(), ref.data(), dev.size());
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "cos %.6f, rel L2 %.5f vs fp32 silu of the same rows", c, e);
        report("up|gate 6b: the device's bf16 silu", c >= 0.999 && e < 0.02, buf);
    }

    // ---- 6c. the silu landed on the gate bands only -----------------------
    // The even bands must come back untouched, bit for bit. A cosine would not
    // do here: silu(x) ~ x for x >> 0, so an activation wrongly applied to up
    // still scores high on a band whose values happen to be positive.
    {
        size_t bad = 0, first = 0;
        for (int l = 0; l < P; l++)
            for (size_t p = 0; p < bands; p++) {
                const size_t o = (size_t)l * WD + (2 * p) * BAND;
                for (size_t j = 0; j < BAND; j++)
                    if (act[o + j] != raw[o + j]) {
                        if (!bad) first = o + j;
                        bad++;
                    }
            }
        char buf[160];
        if (bad)
            std::snprintf(buf, sizeof buf,
                          "%zu up elements moved, first at %zu", bad, first);
        else
            std::snprintf(buf, sizeof buf, "all %zu up elements unchanged",
                          (size_t)P * DI);
        report("up|gate 6c: the up bands are untouched", bad == 0, buf);
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
    }

    // ---- the parity swap: the one bug this design can plausibly have -------
    // `{0,2,silu}, {1,2,none}` silu's the UP bands instead. Both halves are the
    // same shape, so nothing about the dispatch complains; only the arithmetic
    // says which parity the RTP write landed on. Un-weave it exactly as
    // `up_gate_swiglu` does -- odd band times even band -- so the only
    // difference from the production path is which band got the activation.
    {
        q38npu::dequant_mm_op swapped(
            pm->create_app(), 2 * DI, D, t.w_upgate_offset,
            std::vector<q38npu::dequant_mm_pass>{
                {0, 2, q38npu::dequant_mm_act_silu},   // up, wrongly
                {1, 2, q38npu::dequant_mm_act_none},   // gate, raw
            });
        std::vector<float> sw((size_t)P * WD, 0.0f);
        swapped.run(hn.data(), P, W, sw.data());

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
    // 7. down_proj
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
    // 8. the tiling is inert
    // ======================================================================
    // `dequant_mm_op` pins L at 512 and tiles a window on the host, because
    // every DDR term in the engine's cost is linear in L and a wide dispatch
    // amortises nothing. That is only sound if a row's result depends on its
    // own x row and nothing else -- so the same rows, computed as part of an
    // 8-tile window and as part of a ragged 1-tile one, must come back BIT FOR
    // BIT identical. Anything less means a pad row bled into a real one, or a
    // per-tile offset is wrong.
    //
    // Bit-for-bit and not by cosine on purpose: a tail-tile pad row leaking
    // into row 2 of a 3-row window moves that row a long way, but a cosine over
    // 4096 x 5120 values would never see it.
    {
        constexpr int TMAX = 4096;
        std::vector<float> x((size_t)TMAX * cpu_ref::mtp::FC_IN);
        fill_synth(x, 0x51b0c0de);

        std::vector<float> full((size_t)TMAX * D, 0.0f);
        proj.fc(x.data(), TMAX, W, full.data());

        std::string bad;
        for (int T : {3, 130, 221, 512, 513, 1000, 2048}) {
            std::vector<float> part((size_t)T * D, 0.0f);
            proj.fc(x.data(), T, W, part.data());
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
                      "7 ragged widths reproduce their rows out of a %d-row "
                      "(%d-tile) run exactly", TMAX, TMAX / 512);
        report("host-side tiling changes no row", bad.empty(),
               bad.empty() ? buf : ("widths that differ: " + bad));
    }

    // ======================================================================
    // 9. the weight BO crosses apps
    // ======================================================================
    // Checks 3-7 all ran against a BO allocated from the MTP_layer app, in
    // dispatches belonging to dequant_mm's. BOs are device-scoped, so that is
    // legal -- but "legal" here means the dispatch completes, and a BO from the
    // wrong place does not fault, it simply never completes or reads stale
    // bytes. So: the same matrix, copied into a BO from dequant_mm's OWN app,
    // must give the identical answer.
    {
        npu_app_manager* pm = mgr.register_xclbin(proj_xclbin);
        npu_app          a  = pm->create_app();
        const size_t     fb = q38npu::q4k_bytes((size_t)D * cpu_ref::mtp::FC_IN);

        buffer<u8> priv = a.create_bo_buffer<u8>(fb);
        std::memcpy(priv.data(), W.data() + t.w_fc_offset, fb);
        priv.sync_to_device();

        q38npu::dequant_mm_op own(pm->create_app(), D, cpu_ref::mtp::FC_IN, 0);
        std::vector<float> a_out((size_t)P * D, 0.0f), b_out((size_t)P * D, 0.0f);
        own.run(fc_in.data(), P, priv, a_out.data());
        proj.fc(fc_in.data(), P, W, b_out.data());

        const bool same = std::memcmp(a_out.data(), b_out.data(),
                                      a_out.size() * sizeof(float)) == 0;
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "%.1f MB private copy vs the shared 245 MB BO, %s",
                      fb / 1048576.0, same ? "bit for bit" : "DIFFERENT");
        report("a BO from another app is not degraded", same, buf);
    }

    // ======================================================================
    // 10. step 0, composed
    // ======================================================================
    // `_prefill_step_npu_proj` rebuilt here from the pieces, because the engine
    // only hands back drafted ids and this is the one place norm_out can be
    // seen. Checks 13-14 then score the engine's own composition; this one
    // localises a failure to a stage.
    //
    // Widths chosen to cross the tile boundary in both directions: 600 is two
    // tiles with a 88-row tail, 512 is exactly one.
    {
        q38npu::mtp_prefill_attn_op& op = *spec_npu.prefill_op();

        // A window longer than the dump, by repeating it. Fine here -- this is
        // a numerics test, not a determinism one, and repeated ids still carry
        // the activation range Q4_K was fitted to.
        constexpr int PW = 600;
        std::vector<int32_t> ids_w((size_t)PW);
        std::vector<float>   hid_w((size_t)PW * D);
        for (int i = 0; i < PW; i++) {
            ids_w[(size_t)i] = ids32[(size_t)(i % P)];
            std::memcpy(hid_w.data() + (size_t)i * D,
                        hid.data() + (size_t)(i % P) * D, (size_t)D * 4);
        }

        for (int Twin : {1, 2, 8, 128, 221, 512, 600}) {
            const int32_t* wi = Twin <= P ? ids32.data() : ids_w.data();
            const float*   wh = Twin <= P ? hid.data()   : hid_w.data();

            // ---- the NPU path, exactly as the engine composes it ------------
            layer.reset();
            std::vector<float> cs, sn;
            cpu_ref::text_rope_tables(kPos0, Twin, cs, sn);

            cpu_ref::mtp_taps  ntp;
            std::vector<float> nfc_in;
            half.fuse_pre(wi, wh, Twin, ntp, nfc_in);
            proj.fc(nfc_in.data(), Twin, W, ntp.fc_out.data());

            std::vector<float> nxn;
            half.in_norm(ntp.fc_out.data(), Twin, nxn);

            std::vector<float> nk, nv, nqg, nqh;
            proj.qkv_z(nxn.data(), Twin, W, nk, nv, nqg);
            half.qkv_post_rope(nqg.data(), nk.data(), Twin, cs.data(), sn.data(), nqh);

            const int base = layer.cache_len();
            layer.append_rows(nk.data(), nv.data(), Twin);

            std::vector<float> nattn((size_t)Twin * DQ_, 0.0f);
            op.run(nqh.data(), Twin, base, layer.cache(), nattn.data());
            half.apply_gate(nqg.data(), nattn.data(), Twin);

            std::vector<float> nh1((size_t)Twin * D, 0.0f);
            proj.o(nattn.data(), Twin, W, nh1.data());
            for (size_t i = 0; i < nh1.size(); i++) nh1[i] += ntp.fc_out[i];

            std::vector<float> nhn;
            half.post_norm(nh1.data(), Twin, nhn);
            std::vector<float> ng;
            proj.up_gate_swiglu(nhn.data(), Twin, W, ng);

            std::vector<float> nlo((size_t)Twin * D, 0.0f);
            proj.down(ng.data(), Twin, W, nlo.data());
            for (size_t i = 0; i < nlo.size(); i++) nlo[i] += nh1[i];

            std::vector<float> nout((size_t)Twin * D);
            for (int r = 0; r < Twin; r++)
                cpu_ref::rms_norm(nlo.data() + (size_t)r * D,
                                  half.out_norm().data(), D,
                                  nout.data() + (size_t)r * D);

            // ---- the reference ----------------------------------------------
            cpu.reset();
            const std::vector<int32_t> pids = cpu_ref::text_position_ids(kPos0, Twin);
            cpu_ref::mtp_taps ctp;
            cpu.step(wi, pids.data(), wh, Twin, ctp);

            char tag[80];
            std::snprintf(tag, sizeof tag, "step 0 at T = %d: norm_out", Twin);
            report_cos(tag, cosine(nout.data(), ctp.norm_out.data(), nout.size()),
                       0.995);

            // The k/v rows that actually landed in the cache BO, on two
            // separate bars because two different things can go wrong and they
            // are not visible to the same statistic.
            //
            // WORST-ROW COSINE, against the CPU head's cache, is the addressing
            // check -- N5's, carried forward, but at 0.99 rather than N5's
            // 0.999. A cosine over a single 1024-wide row is nearly blind to a
            // uniform relative error and very sensitive to a row holding the
            // wrong content, which is exactly the discrimination a plane-pitch
            // bug needs: norm_out cannot see one clearly, because each query's
            // own row dominates its softmax and dilutes the error before it
            // reaches the output.
            //
            // Why the bar moved off N5's 0.999: a WORST row is a minimum over
            // T samples, and a minimum drifts down with T even when the
            // per-row error is uniform. Measured here, k's worst row walks
            // 0.99975 (T=1), 0.99903 (8), 0.99852 (128), 0.99848 (221) and
            // then stops dead -- because T = 512 and 600 repeat the 221-row
            // dump and find the same row 146 again. That flat tail is the
            // evidence it is a sampling floor and not a tiling or pad-row
            // fault, which would keep growing past 512.
            //
            // N5 held 0.999 because its projections ran on the host and the
            // only per-row loss was the bf16 cache write. With the matmul on
            // the device the per-row error is ~8x that, so a min over a few
            // hundred rows clears 0.999 and no fixed worst-row bar is
            // T-independent. The bar that IS well posed is the rel-L2 one
            // below; this one stays only for its addressing discrimination,
            // where 0.99 is as good as 0.999 -- a swapped or mis-strided row
            // scores near zero, not near 0.998.
            //
            // The norm ratio is printed to separate these two readings at a
            // glance. It came back 0.99x mean, so row 146 is an ordinary row
            // having an unlucky draw, NOT the small-norm row a per-row cosine
            // would otherwise be flagging.
            //
            // RELATIVE L2, against a SAME-BYTES reference over this branch's
            // own bf16-rounded xn, is the arithmetic check. It gets its own
            // bar because a per-row cosine cannot supply one: the error here is
            // a flat relative perturbation, so a per-row score would report
            // which rows have the smallest norm rather than how large it is.
            // (memory: per-row-cosine-reports-the-norm)
            //
            // The bar is 0.04, not bf16's ~0.002, and the gap is the device's:
            // `aie::mmul<8,8,8,bf16>` floors BOTH operands onto a shared-
            // exponent-per-8 grid, so an element well below its block's max
            // keeps far fewer than bf16's 8 mantissa bits. Measured 0.023-0.030
            // for k across these widths, 0.010-0.023 for v; the bar sits above
            // that with room and well below the ~0.5 a wrong reduction gives.
            // (memory: aie-mmul-bf16-truncates-to-bfp16)
            std::vector<float> bk((size_t)Twin * DKV_), bv((size_t)Twin * DKV_);
            double wk = 2.0, wv = 2.0, mean_n2 = 0.0, worst_n2 = 0.0;
            int    wk_row = -1;
            for (int r = 0; r < Twin; r++) {
                const std::vector<float> rk = layer.read_k(r);
                const std::vector<float> rv = layer.read_v(r);
                std::memcpy(bk.data() + (size_t)r * DKV_, rk.data(), (size_t)DKV_ * 4);
                std::memcpy(bv.data() + (size_t)r * DKV_, rv.data(), (size_t)DKV_ * 4);
                const float* ck = cpu.cache_k().data() + (size_t)r * DKV_;
                const double c  = cosine(rk.data(), ck, (size_t)DKV_);
                double n2 = 0.0;
                for (int i = 0; i < DKV_; i++) n2 += (double)ck[i] * ck[i];
                mean_n2 += n2;
                if (c < wk) { wk = c; wk_row = r; worst_n2 = n2; }
                wv = std::min(wv, cosine(rv.data(),
                                         cpu.cache_v().data() + (size_t)r * DKV_,
                                         (size_t)DKV_));
            }
            mean_n2 /= Twin;

            std::vector<float> xb = nxn;
            for (float& f : xb) f = cpu_ref::round_bf16(f);
            std::vector<float> rk_ref((size_t)Twin * DKV_), rv_ref((size_t)Twin * DKV_);
            cpu_ref::q4nx_matmul(W_k, xb.data(), Twin, rk_ref.data());
            cpu_ref::q4nx_matmul(W_v, xb.data(), Twin, rv_ref.data());
            // RoPE is applied to k on the way into the cache, so the reference
            // has to take the same turn before it can be compared to the BO.
            std::vector<float> rq_dummy;
            std::vector<float> qg_copy = nqg;
            half.qkv_post_rope(qg_copy.data(), rk_ref.data(), Twin, cs.data(),
                               sn.data(), rq_dummy);

            const double ek = rel_l2(bk.data(), rk_ref.data(), bk.size());
            const double ev = rel_l2(bv.data(), rv_ref.data(), bv.size());

            std::snprintf(tag, sizeof tag, "step 0 at T = %d: k/v in the BO", Twin);
            char buf[240];
            std::snprintf(buf, sizeof buf,
                          "worst row k %.6f (row %d, norm %.2fx mean) v %.6f "
                          "[gate 0.99]; rel L2 k %.5f v %.5f [gate 0.04]; cursor %d",
                          wk, wk_row,
                          mean_n2 > 0 ? std::sqrt(worst_n2 / mean_n2) : 0.0,
                          wv, ek, ev, layer.cache_len());
            report(tag, wk >= 0.99 && wv >= 0.99 && ek < 0.04 && ev < 0.04 &&
                        layer.cache_len() == Twin, buf);
        }
    }

    // ======================================================================
    // 11. the SwiGLU control
    // ======================================================================
    // silu(up) * gate is just as plausible a reading of a SwiGLU as
    // silu(gate) * up, and it scores nearly as well -- the two differ only in
    // which factor got the nonlinearity. The device returns both halves woven
    // and the host separates them, so getting the un-weave backwards produces
    // exactly this, with no other symptom.
    //
    // Check 6's parity swap is the same ambiguity one stage earlier, seen from
    // the device side. This one scores it through `down_proj`, because that is
    // where a wrong reading actually reaches the residual -- and it takes its
    // two halves from a RAW op rather than the production one, so the
    // orientation being tested is the host's un-weave and not the RTP.
    // (memory: score-swiglu-by-swapping-up-and-gate)
    {
        npu_app_manager* pm = mgr.register_xclbin(proj_xclbin);
        q38npu::dequant_mm_op raw_op(pm->create_app(), 2 * DI, D, t.w_upgate_offset);
        const size_t WD = 2 * (size_t)DI;
        constexpr size_t BAND = 512;
        const size_t bands = (size_t)DI / BAND;
        std::vector<float> raw((size_t)P * WD, 0.0f);
        raw_op.run(hn.data(), P, W, raw.data());

        std::vector<float> gu((size_t)P * DI), gg((size_t)P * DI);
        for (int l = 0; l < P; l++)
            for (size_t p = 0; p < bands; p++) {
                const size_t o = (size_t)l * WD + (2 * p) * BAND;
                std::memcpy(gu.data() + (size_t)l * DI + p * BAND, raw.data() + o,
                            BAND * sizeof(float));
                std::memcpy(gg.data() + (size_t)l * DI + p * BAND,
                            raw.data() + o + BAND, BAND * sizeof(float));
            }

        std::vector<float> right((size_t)P * DI), wrong((size_t)P * DI);
        half.swiglu(gg.data(), gu.data(), P, right.data());   // silu(gate) * up
        half.swiglu(gu.data(), gg.data(), P, wrong.data());   // silu(up) * gate

        std::vector<float> y_right((size_t)P * D, 0.0f), y_wrong((size_t)P * D, 0.0f);
        proj.down(right.data(), P, W, y_right.data());
        proj.down(wrong.data(), P, W, y_wrong.data());

        std::vector<float> want((size_t)P * D);
        cpu_ref::q4nx_matmul(W_down, g.data(), P, want.data());
        report_cos("MLP: silu(gate) * up through down_proj",
                   cosine(y_right.data(), want.data(), want.size()), kGate);
        report_control("MLP: silu(up) * gate instead",
                       cosine(y_wrong.data(), want.data(), want.size()));
    }

    // ======================================================================
    // 12. rows above the cursor stay out
    // ======================================================================
    // N5's stale-row check, re-run because the projections moved: the cache BO
    // is written by `append_rows` on the host and read by two bitstreams, and a
    // crop only moves a cursor -- the rows above it are still there. A window
    // run from a cropped cursor must be bit-identical to the same window run
    // from a reset one, or the mask is not masking.
    {
        constexpr int TW = 96;
        layer.reset();
        std::vector<float> clean;
        auto run_window = [&](std::vector<float>& out) {
            std::vector<float> cs, sn;
            cpu_ref::text_rope_tables(kPos0, TW, cs, sn);
            cpu_ref::mtp_taps  wtp;
            std::vector<float> wfc;
            half.fuse_pre(ids32.data(), hid.data(), TW, wtp, wfc);
            proj.fc(wfc.data(), TW, W, wtp.fc_out.data());
            std::vector<float> wxn;
            half.in_norm(wtp.fc_out.data(), TW, wxn);
            std::vector<float> wk, wv, wqg, wqh;
            proj.qkv_z(wxn.data(), TW, W, wk, wv, wqg);
            half.qkv_post_rope(wqg.data(), wk.data(), TW, cs.data(), sn.data(), wqh);
            const int b = layer.cache_len();
            layer.append_rows(wk.data(), wv.data(), TW);
            out.assign((size_t)TW * DQ_, 0.0f);
            spec_npu.prefill_op()->run(wqh.data(), TW, b, layer.cache(), out.data());
        };
        run_window(clean);

        // Fill the cache well past the window, then crop back to nothing.
        layer.reset();
        std::vector<float> junk;
        run_window(junk);
        run_window(junk);
        layer.crop(0);
        std::vector<float> after;
        run_window(after);

        const bool same = std::memcmp(clean.data(), after.data(),
                                      clean.size() * sizeof(float)) == 0;
        report("rows above the cursor are masked out", same,
               same ? "a cropped cursor reproduces a reset one bit for bit"
                    : "the window sees rows it should not");
    }

    // ======================================================================
    // 13. the seven-step draft loop
    // ======================================================================
    // The gate. Step 0 with NPU projections, steps 1-6 on MTP_layer.xclbin,
    // against speculate_cpu's own loop -- the full composition, including the
    // seam between the two bitstreams and the cursor they share.
    //
    // Scored by the REFERENCE's argmax margin, not by id equality: drafting is
    // a 7-step autoregression, so a step whose top two logits are a hair apart
    // is a coin toss reporting the last bit of the accumulator rather than
    // anything about either path.
    std::vector<int> cpu_ids, npu_ids;
    {
        const int V = (int)head.vocab();
        cpu_ref::mtp_head margin_head(q4nx, &embed, kMaxL);
        const std::vector<double> m = draft_margins(
            margin_head, head, V, std::vector<int>(ids32.begin(), ids32.end()),
            hid.data(), P, kPos0, cpu_ref::mtp::MTP_STEPS, cpu_ids);

        spec_cpu.reset();
        spec_npu.reset();
        const std::vector<int> ids_v(ids32.begin(), ids32.end());
        const std::vector<int> c = spec_cpu.draft(ids_v, hid.data(), P, kPos0,
                                                  cpu_ref::mtp::MTP_STEPS);
        npu_ids = spec_npu.draft(ids_v, hid.data(), P, kPos0,
                                 cpu_ref::mtp::MTP_STEPS);

        std::string detail, undecided;
        int checked = 0, agreed = 0;
        for (size_t s = 0; s < npu_ids.size(); s++) {
            if (m[s] < kDecisive) {
                undecided += (undecided.empty() ? "" : ",") + std::to_string(s);
                continue;
            }
            checked++;
            if (c[s] == npu_ids[s]) agreed++;
        }
        char buf[260];
        std::snprintf(buf, sizeof buf,
                      "%d/%d decisive steps agree%s%s  [cpu %s | npu %s]",
                      agreed, checked,
                      undecided.empty() ? "" : "; undecided: ",
                      undecided.c_str(), id_list(c).c_str(),
                      id_list(npu_ids).c_str());
        report("draft(): NPU projections vs speculate_cpu",
               checked > 0 && agreed == checked, buf);
    }

    // ======================================================================
    // 14. throughput
    // ======================================================================
    // Not a gate -- it is the number this phase exists to produce. Both paths
    // read the whole 245 MB of weights at any width, so the device should win
    // from width 1; that is a prediction, and this is what refutes it.
    //
    // The CPU side is measured only at the two smaller widths. A 4096-row host
    // pass over 424 M parameters is ~3.5 TFLOP and would dominate the suite's
    // runtime without changing what the per-row cost says.
    {
        std::printf("\n-- step-0 projections, ms (5 dispatches per 512-row tile) --\n");
        std::printf("   %6s  %10s  %10s  %9s  %9s  %10s\n",
                    "T", "NPU", "CPU", "NPU us/row", "speedup", "swiglu cut");

        for (int T : {1, 221, 1024, 4096}) {
            std::vector<float> x((size_t)T * cpu_ref::mtp::FC_IN);
            fill_synth(x, 0xa11ce ^ (uint64_t)T);
            std::vector<float> y1((size_t)T * D), y2((size_t)T * D);
            std::vector<float> k2, v2, q2, g2, xh((size_t)T * D);
            fill_synth(xh, 0xbeef ^ (uint64_t)T);
            std::vector<float> at((size_t)T * DQ_), gi((size_t)T * DI);
            fill_synth(at, 0xfeed ^ (uint64_t)T);
            fill_synth(gi, 0xf00d ^ (uint64_t)T);

            auto t0 = std::chrono::steady_clock::now();
            proj.fc(x.data(), T, W, y1.data());
            proj.qkv_z(xh.data(), T, W, k2, v2, q2);
            proj.o(at.data(), T, W, y2.data());
            proj.up_gate_swiglu(xh.data(), T, W, g2);
            proj.down(gi.data(), T, W, y1.data());
            const double npu = ms_since(t0);

            // What the fuse removed: the host silu-and-multiply pass that used
            // to sit between up|gate and down, over two [T][DI] fp32 buffers.
            // `up_gate_swiglu`'s un-weave does the multiply for free -- it was
            // already touching both bands -- so this column is pure saving, and
            // it is the only part of the NPU number above that got smaller.
            std::vector<float> su((size_t)T * DI), sg((size_t)T * DI);
            fill_synth(su, 0x51100 ^ (uint64_t)T);
            fill_synth(sg, 0x51101 ^ (uint64_t)T);
            auto s0 = std::chrono::steady_clock::now();
            half.swiglu(sg.data(), su.data(), T, sg.data());
            const double swig = ms_since(s0);

            double cpu_ms = 0.0;
            if (T <= 1024) {
                std::vector<float> o1((size_t)T * D), o2((size_t)T * 2 * DQ_),
                                   o3((size_t)T * DKV_), o4((size_t)T * DKV_),
                                   o5((size_t)T * DI), o6((size_t)T * DI);
                auto c0 = std::chrono::steady_clock::now();
                cpu_ref::q4nx_matmul(W_fc,   x.data(),  T, o1.data());
                cpu_ref::q4nx_matmul(W_q,    xh.data(), T, o2.data());
                cpu_ref::q4nx_matmul(W_k,    xh.data(), T, o3.data());
                cpu_ref::q4nx_matmul(W_v,    xh.data(), T, o4.data());
                cpu_ref::q4nx_matmul(W_o,    at.data(), T, o1.data());
                cpu_ref::q4nx_matmul(W_up,   xh.data(), T, o5.data());
                cpu_ref::q4nx_matmul(W_gate, xh.data(), T, o6.data());
                cpu_ref::q4nx_matmul(W_down, gi.data(), T, o1.data());
                cpu_ms = ms_since(c0);
            }

            if (cpu_ms > 0.0)
                std::printf("   %6d  %10.1f  %10.1f  %9.1f  %8.2fx  %10.2f\n",
                            T, npu, cpu_ms, npu * 1000.0 / T, cpu_ms / npu, swig);
            else
                std::printf("   %6d  %10.1f  %10s  %9.1f  %9s  %10.2f\n",
                            T, npu, "-", npu * 1000.0 / T, "-", swig);
        }
        std::printf("   (CPU omitted above 1024: a host pass there is ~3.5 TFLOP "
                    "and says nothing new)\n");
    }

    // ======================================================================
    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
