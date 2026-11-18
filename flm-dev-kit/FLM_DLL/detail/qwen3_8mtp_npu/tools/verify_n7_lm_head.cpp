// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify_n7_lm_head.cpp
/// \brief N7 -- the final norm + vocabulary projection on `lm_head.xclbin`,
///        at L = 1, and the draft loop that now runs on it.
///
/// Build:
///   source ~/setup.sh
///   g++-13 -O2 -march=native -fopenmp -std=c++20
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n7_lm_head.cpp common/npu/lm_head_sequence.cpp
///       speculate/npu/proj_sequence.cpp speculate/npu/mtp_sequence.cpp
///       prefill/npu/prefill_attn_sequence.cpp speculate/mtp_head.cpp
///       ../q4nx/q4_npu_eXpress.cpp ../q4nx/safe_tensors.cpp
///       -L../../external_libs -L../../build/lib -L/opt/xilinx/xrt/lib
///       -lxrt_coreutil -laiebu -luuid -lpthread -o /tmp/verify_n7_lm_head
///   (one line; the continuations are dropped here because a backslash at the
///    end of a // comment splices the next line into it)
///
/// Takes the four xclbins as FILES. Runs in ~3 s but burns ~27 CPU-seconds:
/// the table is 1.35 GB and the host reads all of it once per same-bytes
/// reference row, at about 50 GB/s across 32 threads.
///
/// ---------------------------------------------------------------------------
/// What moved
/// ---------------------------------------------------------------------------
/// The draft loop's seven projections and the engine's `get_logits()`. Both
/// are L = 1, which is the shape this bitstream has; module 3's verify pass is
/// L = 8 and stays on the host, because this design streams the whole table
/// per row and eight dispatches would cost ~207 ms against the host's ~70.
///
/// Those two callers are scored separately, because they are not the same
/// claim. Check 8 scores the DRAFT loop, where a different token is
/// output-neutral and only moves acceptance. Check 9 scores PREFILL, where the
/// projected token is emitted -- so there the argmax is the product, and it is
/// gated against `prefill.next_token` rather than against a margin.
///
/// **This is not a speedup and the suite is not written as if it were.** The
/// projection is DRAM-bound on both engines -- ~26 ms on the device, ~28 on 32
/// host threads, both ~50 GB/s over 1.35 GB. What changes is that a draft step
/// stops occupying every CPU core. So the numbers this suite has to get right
/// are correctness and the ACCEPTANCE RATE, not milliseconds; throughput is
/// reported at the end and gated at "not a regression".
///
/// ---------------------------------------------------------------------------
/// Why bf16 logits are the risk
/// ---------------------------------------------------------------------------
/// The device returns bf16 -- about three decimal digits. Top-1 ties that fp32
/// never saw become common: the design's own host driver hit one on its second
/// reference row, two tokens at 24.75. On the draft path a different token is
/// output-neutral, because `speculate()` verifies every draft against the real
/// stack, but it moves acceptance, which is the only thing drafting buys.
///
/// So checks 7 and 8 measure the top-2 margin distribution and compare both
/// backends' drafted ids against the reference dump's own, rather than
/// asserting the two backends agree. A driver that gated on agreement here
/// would be gating on a coin toss.
/// (memory: score-a-draft-loop-by-argmax-margin,
///          qwen38-lm-head-untied-and-argmax-ties)
///
/// ---------------------------------------------------------------------------
/// The controls
/// ---------------------------------------------------------------------------
/// A cosine near 1 is evidence about arithmetic and says nothing about
/// addressing, so every score is paired with something that must COLLAPSE:
///
///   rotate the row      row i scored against row (i+1)%8's reference. Catches
///                       a dispatch that replayed a stale y BO -- which is
///                       exactly how an unchecked timeout presents.
///                       (memory: unchecked-ert-state-looks-like-a-pass)
///   swap the norm       the same x through model.norm.weight and
///                       model.mtp.norm.weight. This is the one check the
///                       draft path's correctness rests on: the norm lives in
///                       the second half of the input BO, the kernel is the
///                       only thing that reads it, and if it were ignored the
///                       draft would silently project through the base model's
///                       norm and still score 0.99 against everything else.
///   permuted repeat     eight more dispatches in reverse row order. A design
///                       whose memtile x fill is a one-shot BD passes a single
///                       dispatch and returns NaN on the second -- which is
///                       what the L = 8 sibling does today.
///                       (memory: one-shot-bench-cannot-see-dispatch-state,
///                                unrouted-mm2s-chain-still-fires)
#include "speculate/speculate_npu.hpp"
#include "speculate/speculate_cpu.hpp"
#include "speculate/mtp_head.hpp"
#include "common/npu/lm_head_op.hpp"
#include "common/npu/lm_head_weights.hpp"
#include "common/cpu_ref_common.hpp"
#include "common/embedding.hpp"
#include "common/lm_head.hpp"

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

/// The bar for a same-bytes score. Q8_0 is common mode between the device and
/// the reference, so what is left is the device's own arithmetic: the bfp16
/// mmul floors both operands onto a shared-exponent-per-8 grid and every
/// fp32 -> bf16 on the way out truncates toward -inf.
/// (memory: score-dequant-mm-against-its-own-quantized-bytes,
///          aie-mmul-bf16-truncates-to-bfp16)
constexpr double kGate = 0.999;

/// Against the reference dump, which is numpy in fp32 over unquantised
/// weights, so Q8_0's own loss is NOT common mode and shows up here.
constexpr double kGateRef = 0.999;

/// The ceiling a control must fall under.
constexpr double kControl = 0.99;

/// Rows in the verify dump.
constexpr int kRows = 8;

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-46s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
    if (ok) n_pass++; else n_fail++;
}

void info(const char* tag, const std::string& detail) {
    std::printf("  [info] %-46s %s\n", tag, detail.c_str());
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

double rel_l2(const float* a, const float* b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)a[i] - b[i];
        num += d * d;
        den += (double)b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : 0.0;
}

void report_cos(const char* tag, double cos, double gate) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "cos = %.8f  (gate %.4f)", cos, gate);
    report(tag, cos >= gate, buf);
}

void report_control(const char* tag, double cos) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "cos = %.6f  (must be < %.3f)", cos, kControl);
    report(tag, cos < kControl, buf);
}

/// The kernel's own normalisation, modelled exactly: x arrives bf16, the norm
/// weight arrives bf16, and the result is written back into a bf16 buffer. A
/// reference that norms in fp32 charges the device for rounding it did not
/// choose. (memory: aie-to-vector-bf16-floors)
void device_norm(const float* x, const float* norm_w, int D, float* out) {
    double ss = 0.0;
    std::vector<float> xb((size_t)D);
    for (int j = 0; j < D; j++) {
        xb[j] = cpu_ref::round_bf16(x[j]);
        ss += (double)xb[j] * xb[j];
    }
    const float inv = 1.0f / std::sqrt((float)(ss / (double)D) + RMS_EPS);
    for (int j = 0; j < D; j++)
        out[j] = cpu_ref::round_bf16(xb[j] * inv * cpu_ref::round_bf16(norm_w[j]));
}

int argmax_f(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/// The reference logits are bf16-rounded upstream, so its top-1 is sometimes
/// an exact tie. Which of a tied pair wins is arbitrary, so score against the
/// reference's own VALUE at the token the device picked rather than against
/// the index numpy happened to break the tie toward.
bool argmax_agrees(const float* ref, int n, int got) {
    return ref[got] >= ref[argmax_f(ref, n)];
}

} // namespace

int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string lm_xclbin =
        argc > 2 ? argv[2]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "lm_head_npu_bin/build/QWEN3_8_27B/xclbins/lm_head.xclbin";
    const std::string mtp_xclbin =
        argc > 3 ? argv[3]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "MTP_decoding/build/QWEN3_8_27B/xclbins/MTP_layer.xclbin";
    const std::string prefill_xclbin =
        argc > 4 ? argv[4]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "gated_attention_prefill/build/xclbins/attn.xclbin";
    const std::string proj_xclbin =
        argc > 5 ? argv[5]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "dequant_mm/build/xclbins/dequant_mm.xclbin";

    const int D = MODEL_DIM;

    std::printf("== N7: the lm_head on lm_head.xclbin, L = 1 ==\n");
    std::printf("   model    %s\n   lm_head  %s\n\n",
                model_dir.c_str(), lm_xclbin.c_str());

    // ======================================================================
    // inputs
    // ======================================================================
    cpu_ref::safetensors ref(model_dir + "/ref.safetensors");
    cpu_ref::safetensors q4nx(model_dir + "/model.q4nx");

    const std::vector<float> x_ref      = ref.f32("verify.layer_63_out");
    const std::vector<float> logits_ref = ref.f32("verify.logits");

    const std::vector<float> norm_base = q4nx.f32("model.norm.weight");
    const std::vector<float> norm_mtp  = q4nx.f32("model.mtp.norm.weight");

    const cpu_ref::q4nx_tensor W = cpu_ref::get_q4nx(q4nx, "lm_head.weight");
    const int vocab = W.rows;

    if ((int)(x_ref.size() / D) != kRows ||
        logits_ref.size() != (size_t)kRows * vocab) {
        std::fprintf(stderr, "ref.safetensors: verify.layer_63_out / "
                             "verify.logits do not match %d x %d\n", kRows, vocab);
        return 1;
    }

    // ======================================================================
    // 1. geometry
    // ======================================================================
    std::printf("-- 1. geometry --\n");
    {
        const int padded = q38npu::lm_head_vocab_padded(vocab);
        char buf[200];

        // The kernel hardcodes 1e-6. A divergence here is invisible in a
        // cosine and shows up only as a slightly wrong distribution.
        std::snprintf(buf, sizeof buf, "kernel 1e-6 vs q38::RMS_EPS %g",
                      (double)RMS_EPS);
        report("rms epsilon matches the kernel", RMS_EPS == 1e-6f, buf);

        std::snprintf(buf, sizeof buf, "%d -> %d (%d rounds of %d)", vocab,
                      padded, padded / q38npu::lm_head_tiles::M_PER_ROUND,
                      q38npu::lm_head_tiles::M_PER_ROUND);
        report("padded vocab is a whole number of rounds",
               padded % q38npu::lm_head_tiles::M_PER_ROUND == 0 && padded >= vocab,
               buf);

        // Upstream sizes the BO as D*VP + D*VP/32*2; lm_head_weight_bytes
        // writes the same thing as VP*D*17/16. If they ever disagree the
        // sequence walks off the end of the BO, which does not fault.
        const size_t ours = q38npu::lm_head_weight_bytes(padded, D);
        const size_t upstream = (size_t)D * padded + (size_t)D * padded / 32 * 2;
        std::snprintf(buf, sizeof buf, "%zu B, upstream %zu B", ours, upstream);
        report("weight BO size matches the design's host", ours == upstream, buf);

        // The staging permutation has to be a bijection onto the BO. A
        // collision would overwrite one core's blocks with another's and
        // still fill every byte.
        const int padded_nrb = padded / cpu_ref::Q4NX_ROW_BLOCK;
        std::vector<uint8_t> seen((size_t)padded_nrb * W.ncb, 0);
        bool bij = true;
        for (int rb = 0; rb < padded_nrb && bij; rb++)
            for (int cb = 0; cb < W.ncb; cb++) {
                const size_t slot =
                    q38npu::lm_head_block_offset(rb, cb, W.ncb) /
                    cpu_ref::Q80_BLOCK_BYTES;
                if (slot >= seen.size() || seen[slot]) { bij = false; break; }
                seen[slot] = 1;
            }
        std::snprintf(buf, sizeof buf, "%d x %d blocks", padded_nrb, W.ncb);
        report("block offsets are a bijection onto the BO", bij, buf);

        // The closed form against upstream's running pointer -- the two are
        // independent derivations of the same permutation, which is the only
        // way to check a staging routine that would otherwise check itself.
        bool same = true;
        size_t running = 0;
        const int V = q38npu::LM_HEAD_VERTICAL_BLOCKS;
        for (int r = 0; r < padded_nrb && same; r += V)
            for (int half = 0; half < 2 && same; half++)
                for (int cb = 0; cb < W.ncb && same; cb++)
                    for (int i = 0; i < V / 2; i++) {
                        const int rb = r + half * (V / 2) + i;
                        if (q38npu::lm_head_block_offset(rb, cb, W.ncb) != running)
                            { same = false; break; }
                        running += cpu_ref::Q80_BLOCK_BYTES;
                    }
        report("closed form == upstream's reorder_cpy_q80 order", same,
               "all blocks");
    }

    // ======================================================================
    // 2. the device
    // ======================================================================
    std::printf("\n-- 2. staging (1.35 GB) --\n");
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);

    const auto t_stage0 = std::chrono::steady_clock::now();
    npu_app_manager* lm_mgr = mgr.register_xclbin(lm_xclbin);
    q38npu::npu_lm_head lmh(lm_mgr->create_app(), q4nx);
    const double stage_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_stage0).count();
    info("staged", std::to_string((long)stage_ms) + " ms");

    {
        char buf[200];
        std::snprintf(buf, sizeof buf, "%d x %d, padded to %d", lmh.vocab(),
                      lmh.dim(), lmh.vocab_padded());
        report("op geometry matches the checkpoint",
               lmh.vocab() == vocab && lmh.dim() == D, buf);

        // Every staged block against its source. The whole BO, not a sample:
        // an addressing bug that only touches one core's slots would survive
        // any sampling scheme that did not happen to land on it.
        // (memory: move-weights-offset-is-in-elements)
        const int padded_nrb = lmh.vocab_padded() / cpu_ref::Q4NX_ROW_BLOCK;
        const uint8_t* dst = lmh.weight_bo().data();
        long bad_real = 0, bad_pad = 0;
#pragma omp parallel for schedule(static) reduction(+:bad_real,bad_pad)
        for (int rb = 0; rb < padded_nrb; rb++)
            for (int cb = 0; cb < W.ncb; cb++) {
                const uint8_t* src =
                    (rb < W.nrb)
                        ? W.blocks + ((size_t)rb * W.ncb + cb) * W.block_bytes
                        : W.blocks;
                const uint8_t* got =
                    dst + q38npu::lm_head_block_offset(rb, cb, W.ncb);
                if (std::memcmp(got, src, cpu_ref::Q80_BLOCK_BYTES) != 0) {
                    if (rb < W.nrb) bad_real++; else bad_pad++;
                }
            }
        char b2[200];
        std::snprintf(b2, sizeof b2, "%ld of %ld real blocks differ", bad_real,
                      (long)W.nrb * W.ncb);
        report("staged bytes == lm_head.weight", bad_real == 0, b2);
        std::snprintf(b2, sizeof b2, "%ld of %ld pad blocks differ", bad_pad,
                      (long)(padded_nrb - W.nrb) * W.ncb);
        report("pad rows hold the replicated block", bad_pad == 0, b2);
    }

    // ======================================================================
    // 3. the eight reference rows
    // ======================================================================
    std::printf("\n-- 3. the reference rows, through model.norm.weight --\n");

    std::vector<std::vector<float>> got((size_t)kRows,
                                        std::vector<float>((size_t)vocab));
    double dispatch_ms = 0;
    for (int i = 0; i < kRows; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        const bf16* lg = lmh.run(x_ref.data() + (size_t)i * D, norm_base.data());
        dispatch_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        for (int j = 0; j < vocab; j++) got[i][j] = (float)lg[j];
    }

    {
        // A dispatch that timed out leaves the output BO holding the previous
        // row, and run() would have thrown -- but a run that came back in 2 ms
        // did not stream 1.35 GB and is a different kind of lie.
        const double per = dispatch_ms / kRows;
        char buf[200];
        std::snprintf(buf, sizeof buf, "%.1f ms/row (1.35 GB => %.1f GB/s)", per,
                      1.3536e9 / (per * 1e-3) / 1e9);
        report("dispatch moved the whole table", per > 10.0 && per < 200.0, buf);
    }

    int hits = 0;
    double worst_ref = 1.0, worst_same = 1.0, worst_rel = 0.0;
    std::vector<float> same_bytes((size_t)vocab), xn((size_t)D);
    for (int i = 0; i < kRows; i++) {
        const float* r = logits_ref.data() + (size_t)i * vocab;
        const int got_tok = argmax_f(got[i].data(), vocab);
        if (argmax_agrees(r, vocab, got_tok)) hits++;

        device_norm(x_ref.data() + (size_t)i * D, norm_base.data(), D, xn.data());
        cpu_ref::q4nx_matmul(W, xn.data(), 1, same_bytes.data());

        const double c_ref  = cosine(got[i].data(), r, vocab);
        const double c_same = cosine(got[i].data(), same_bytes.data(), vocab);
        const double rl     = rel_l2(got[i].data(), same_bytes.data(), vocab);
        worst_ref  = std::min(worst_ref, c_ref);
        worst_same = std::min(worst_same, c_same);
        worst_rel  = std::max(worst_rel, rl);

        std::printf("     row %d  argmax npu=%-7d ref=%-7d %-10s "
                    "cos(ref)=%.6f cos(same)=%.6f relL2=%.5f\n",
                    i, got_tok, argmax_f(r, vocab),
                    argmax_agrees(r, vocab, got_tok)
                        ? (got_tok == argmax_f(r, vocab) ? "OK" : "OK (tie)")
                        : "MISMATCH",
                    c_ref, c_same, rl);
    }
    {
        char buf[200];
        std::snprintf(buf, sizeof buf, "%d/%d", hits, kRows);
        report("argmax matches verify.logits", hits == kRows, buf);
    }
    report_cos("worst cos vs verify.logits (fp32 numpy)", worst_ref, kGateRef);
    report_cos("worst cos vs a same-bytes CPU reference", worst_same, kGate);
    {
        char buf[200];
        std::snprintf(buf, sizeof buf, "worst relative L2 = %.5f", worst_rel);
        report("relative L2 vs the same-bytes reference", worst_rel < 0.05, buf);
    }

    // ======================================================================
    // 4. the rotate control
    // ======================================================================
    std::printf("\n-- 4. controls --\n");
    {
        double worst = 0.0;
        for (int i = 0; i < kRows; i++)
            worst = std::max(worst,
                             cosine(got[i].data(),
                                    logits_ref.data() + (size_t)((i + 1) % kRows) * vocab,
                                    vocab));
        report_control("rotate the row by one (must collapse)", worst);
    }

    // ======================================================================
    // 5. the norm-weight control
    // ======================================================================
    {
        // The whole draft path rests on this: the norm lives in the second
        // half of the input BO and the kernel is the only thing that reads it.
        // If it were ignored -- stale from a previous call, say -- the draft
        // would project through the base model's norm and every other check
        // here would still pass.
        //
        // This control is NOT gated on an absolute cosine, and the reason is
        // measured rather than assumed: the two norm vectors are themselves
        // cos 0.999434 apart, differing mostly in scale (mean 1.944 vs 2.252).
        // Projections through two nearly-parallel norms are nearly parallel
        // too, so no honest kernel could push this under 0.99 -- an absolute
        // bar here would fail a correct device. What separates "read it" from
        // "ignored it" is the SIZE of the move against the device's own noise:
        // if the second half were ignored the swap would move nothing and this
        // cosine would sit at the error floor.
        std::vector<float> with_mtp((size_t)vocab);
        const bf16* lg = lmh.run(x_ref.data(), norm_mtp.data());
        for (int j = 0; j < vocab; j++) with_mtp[j] = (float)lg[j];

        // The floor: the same row, same norm, device against its own bytes.
        device_norm(x_ref.data(), norm_base.data(), D, xn.data());
        cpu_ref::q4nx_matmul(W, xn.data(), 1, same_bytes.data());
        const double floor_d = 1.0 - cosine(got[0].data(), same_bytes.data(), vocab);
        const double swap_d  = 1.0 - cosine(with_mtp.data(), got[0].data(), vocab);
        const double ratio   = floor_d > 0 ? swap_d / floor_d : 0.0;

        char buf[220];
        std::snprintf(buf, sizeof buf,
                      "swap moves %.2e, floor %.2e -- %.0fx (need 20x)",
                      swap_d, floor_d, ratio);
        report("swapping the norm weight moves the logits", ratio >= 20.0, buf);

        // And it is the right different: score it against its own same-bytes
        // reference, so a garbled second half cannot pass by merely differing.
        device_norm(x_ref.data(), norm_mtp.data(), D, xn.data());
        cpu_ref::q4nx_matmul(W, xn.data(), 1, same_bytes.data());
        report_cos("mtp.norm scores against its own reference",
                   cosine(with_mtp.data(), same_bytes.data(), vocab), kGate);

        std::snprintf(buf, sizeof buf, "model.norm -> %d, mtp.norm -> %d",
                      argmax_f(got[0].data(), vocab),
                      argmax_f(with_mtp.data(), vocab));
        info("argmax under each norm", buf);
    }

    // ======================================================================
    // 6. the permuted repeat
    // ======================================================================
    {
        // Eight more dispatches, rows reversed. A design whose memtile x fill
        // is a one-shot BD passes dispatch 1 and returns NaN afterwards, and a
        // design that replayed the previous y would agree with itself here.
        int agree = 0;
        bool finite = true;
        for (int i = kRows - 1; i >= 0; i--) {
            const bf16* lg = lmh.run(x_ref.data() + (size_t)i * D, norm_base.data());
            double worst_d = 0.0;
            for (int j = 0; j < vocab; j++) {
                const float v = (float)lg[j];
                if (!std::isfinite(v)) { finite = false; break; }
                worst_d = std::max(worst_d, (double)std::fabs(v - got[i][j]));
            }
            if (!finite) break;
            if (worst_d == 0.0) agree++;
        }
        char buf[200];
        std::snprintf(buf, sizeof buf, "%d/%d rows reproduced bit-exactly", agree,
                      kRows);
        report("a second dispatch pass is finite and repeatable",
               finite && agree == kRows, buf);
    }

    // ======================================================================
    // 7. the bf16 top-2 margin
    // ======================================================================
    std::printf("\n-- 7. how much resolution bf16 left --\n");
    {
        // bf16 carries 8 bits of mantissa, so the spacing at magnitude m is
        // about m * 2^-8. A top-2 gap below that was decided by the rounding.
        int decided_by_rounding = 0;
        for (int i = 0; i < kRows; i++) {
            const float* g = got[i].data();
            float best = g[0], second = -1e30f;
            for (int j = 1; j < vocab; j++) {
                if (g[j] > best)        { second = best; best = g[j]; }
                else if (g[j] > second) { second = g[j]; }
            }
            const float ulp = std::fabs(best) * (1.0f / 256.0f);
            const float gap = best - second;
            if (gap <= ulp) decided_by_rounding++;
            std::printf("     row %d  top1 %.4f  gap %.4f  bf16 ulp %.4f  %s\n",
                        i, best, gap, ulp, gap <= ulp ? "<- a coin toss" : "");
        }
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "%d of %d rows' top-1 is inside one bf16 ulp",
                      decided_by_rounding, kRows);
        info("bf16 tie exposure", buf);
        // Not a gate: a tie is a property of the distribution, not a defect.
        // It is here because it is the number that predicts the acceptance
        // rate, and check 8 is where that gets measured.
    }

    // ======================================================================
    // 8. the draft loop, both speculators
    // ======================================================================
    //
    // `speculate_npu` no longer has a host projection to fall back to -- the
    // draft head's lm_head is the device op or nothing, so the arm this check
    // used to build (the same backend constructed without `&lmh`) is not a
    // configuration the engine can reach any more. The comparator is now
    // `speculate_cpu`, the host reference the whole module is scored against
    // in N5 and N6, which makes this a claim about the draft LOOP rather than
    // about the projection alone. The projection alone is check 9: same row,
    // `npu_lm_head` against `cpu_ref::lm_head`, both gated on the reference.
    std::printf("\n-- 8. the draft loop --\n");
    {
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

        cpu_ref::embedding_table embed(q4nx);
        cpu_ref::lm_head         head(q4nx);

        qwen3_8mtp_desc desc;
        desc.D                     = (uint32_t)D;
        desc.vocab_size            = (uint32_t)vocab;
        desc.mtp_num_hidden_layers = 1;
        desc.mtp_fc_in             = (uint32_t)cpu_ref::mtp::FC_IN;

        const int kMaxL = 1024;
        std::printf("   staging two draft heads (~250 MB each)\n");
        qwen38::speculate_npu spec_npu(q4nx, &desc, &embed, kMaxL, &mgr,
                                       mtp_xclbin, prefill_xclbin, proj_xclbin,
                                       &lmh);
        qwen38::speculate_cpu spec_cpu(q4nx, &desc, &embed, &head, kMaxL);

        report("the NPU speculator took the lm_head op",
               spec_npu.lm_head_op() == &lmh,
               "the device head, borrowed from the engine's own context");

        // ------------------------------------------------------------------
        // what a 5th hw_context costs
        // ------------------------------------------------------------------
        // In the live engine the draft step got SLOWER on the device (40.5 vs
        // 35.8 ms/step) even though the projection alone measures 26 ms -- so
        // something is charged for the move that is not the projection. The
        // two candidates are the number of contexts (firmware time-slices
        // them) and the SWITCHING between two of them every draft step.
        //
        // Check 3 already timed this dispatch with lm_head.xclbin as the only
        // context. Repeating it verbatim now that the two speculators have
        // registered four more separates the two: if the solo number moves,
        // it is the context count; if it does not, the cost is in alternating
        // with MTP_layer.xclbin, which is what the draft loop does.
        const double solo1 = dispatch_ms / kRows;
        double solo5 = 0;
        for (int i = 0; i < kRows; i++) {
            const auto t = std::chrono::steady_clock::now();
            lmh.run(x_ref.data() + (size_t)i * D, norm_base.data());
            solo5 += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t).count();
        }
        solo5 /= kRows;
        char tb[220];
        std::snprintf(tb, sizeof tb,
                      "%.1f ms/row with 1 context, %.1f with 5 (%+.1f ms)",
                      solo1, solo5, solo5 - solo1);
        info("the same dispatch, more contexts", tb);

        const std::vector<int> ids(ids32.begin(), ids32.end());
        const int K = cpu_ref::mtp::MTP_STEPS;

        // Timed twice each, and only the second is reported. The first draft
        // after a reset takes step 0's prefill shape over P rows and warms
        // both contexts, so timing it measures the wrong thing -- the engine's
        // steady state is the repeat.
        auto timed_draft = [&](auto& s, double& ms) {
            s.reset();
            std::vector<int> out = s.draft(ids, hid.data(), P, 1, K);
            s.reset();
            const auto t = std::chrono::steady_clock::now();
            out = s.draft(ids, hid.data(), P, 1, K);
            ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t).count();
            return out;
        };
        double npu_ms = 0, cpu_ms = 0;
        const std::vector<int> d_npu = timed_draft(spec_npu, npu_ms);
        const std::vector<int> d_cpu = timed_draft(spec_cpu, cpu_ms);

        // The reference's own drafted ids, from its per-step logits. This is
        // the target both backends are scored against -- comparing them to
        // EACH OTHER would gate on which way bf16 broke a tie.
        std::vector<int> d_ref;
        for (int s = 0; s < K; s++) {
            const std::vector<float> l =
                ref.f32("mtp.step" + std::to_string(s) + ".logits_last");
            d_ref.push_back(argmax_f(l.data(), vocab));
        }

        int hit_npu = 0, hit_cpu = 0;
        std::printf("     step  ref       npu       cpu\n");
        for (int s = 0; s < K; s++) {
            if (d_npu[s] == d_ref[s]) hit_npu++;
            if (d_cpu[s] == d_ref[s]) hit_cpu++;
            std::printf("     %4d  %-8d  %-8d%c %-8d%c\n", s, d_ref[s],
                        d_npu[s], d_npu[s] == d_ref[s] ? ' ' : '*',
                        d_cpu[s], d_cpu[s] == d_ref[s] ? ' ' : '*');
        }
        char buf[200];
        std::snprintf(buf, sizeof buf, "npu %d/%d, cpu %d/%d", hit_npu, K,
                      hit_cpu, K);
        // The device loop must not draft WORSE than the host reference. That
        // is the acceptance-rate claim, and it is the only gate on the loop --
        // an exact match with the reference is not available to either
        // backend once 7 autoregressive steps have compounded their error.
        // (memory: score-a-draft-loop-by-argmax-margin)
        report("the NPU loop drafts no worse than speculate_cpu",
               hit_npu >= hit_cpu, buf);

        std::snprintf(buf, sizeof buf,
                      "npu %.0f ms (%.1f/step), cpu %.0f ms (%.1f/step)",
                      npu_ms, npu_ms / K, cpu_ms, cpu_ms / K);
        info("draft wall clock, whole module", buf);
        // Reported, not gated, and it is no longer the projection's number:
        // these two differ in every stage, not just the head. What the head
        // alone costs is the solo dispatch above against check 9's host
        // projection on the same row.
    }

    // ======================================================================
    // 9. module 1: the prefill projection
    // ======================================================================
    //
    // Everything above this point is driven from `verify.*`, which is module
    // 3's dump. Prefill is a different row of a different sweep, and it is the
    // caller whose argmax is actually EMITTED -- `_prefill_with_mm` projects
    // the last row of the window and the runtime samples the result. So this
    // section gates on the token, not on a cosine: `prefill.next_token` is
    // what the reference implementation emitted for this prompt, and the
    // device has to reproduce it.
    //
    // It also scores the host backend on the same row. Turning the device path
    // on by default is only defensible if the two agree here, and agreement is
    // a claim about THIS distribution rather than a property of bf16 -- the
    // margin below says how much room there was.
    std::printf("\n-- 9. module 1: the prefill projection --\n");
    {
        std::vector<float> x_pre, logits_pre;
        std::vector<int32_t> next_tok;
        bool have = true;
        try {
            x_pre      = ref.f32("prefill.layer_63_out");
            logits_pre = ref.f32("prefill.logits_last");
            next_tok   = ref.i32("prefill.next_token");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "ref.safetensors: %s\n", e.what());
            have = false;
        }

        if (!have || next_tok.empty() ||
            logits_pre.size() != (size_t)vocab ||
            x_pre.size() % (size_t)D != 0) {
            report("prefill reference rows are present", false,
                   "prefill.layer_63_out / logits_last / next_token");
        } else {
            const int  P    = (int)(x_pre.size() / (size_t)D);
            const int  last = P - 1;
            const int  want = next_tok[0];
            const float* xr = x_pre.data() + (size_t)last * D;

            char buf[240];
            std::snprintf(buf, sizeof buf, "%d rows, projecting row %d", P, last);
            info("prefill window", buf);

            // ---- the device -------------------------------------------
            const bf16* lg = lmh.run(xr, norm_base.data());
            std::vector<float> dev((size_t)vocab);
            for (int j = 0; j < vocab; j++) dev[j] = (float)lg[j];
            const int dev_tok = argmax_f(dev.data(), vocab);

            // ---- the host, on the same row -----------------------------
            //
            // The engine has no host projection left -- `lm_head.xclbin` is
            // required and `Impl::project()` is gone -- so this is a scored
            // reference, not the other arm of a switch. It stays because it
            // is what "the device is right" was originally measured against.
            cpu_ref::lm_head head_cpu(q4nx);
            std::vector<float> host((size_t)vocab);
            head_cpu.norm_and_forward(xr, 1, norm_base.data(), 1, host.data(),
                                      RMS_EPS);
            const int host_tok = argmax_f(host.data(), vocab);

            std::snprintf(buf, sizeof buf, "ref=%d  npu=%d  cpu=%d", want,
                          dev_tok, host_tok);
            // The gate. Not "the two backends agree" -- both have to be right.
            report("prefill next_token, device backend", dev_tok == want, buf);
            report("prefill next_token, host backend",   host_tok == want, buf);

            report_cos("cos vs prefill.logits_last (fp32 numpy)",
                       cosine(dev.data(), logits_pre.data(), vocab), kGateRef);

            // Same-bytes: the device's own Q8_0 weights and its own bf16 norm,
            // so Q8_0 is common mode and what is left is the array's
            // arithmetic. (memory: score-dequant-mm-against-its-own-quantized-bytes)
            std::vector<float> same((size_t)vocab), xnp((size_t)D);
            device_norm(xr, norm_base.data(), D, xnp.data());
            cpu_ref::q4nx_matmul(W, xnp.data(), 1, same.data());
            report_cos("cos vs a same-bytes CPU reference",
                       cosine(dev.data(), same.data(), vocab), kGate);

            // ---- the control ------------------------------------------
            //
            // A cosine near 1 says nothing about WHICH row was projected, and
            // row P-2 is the single most plausible off-by-one. It has to
            // collapse against row P-1's reference, and it has to pick a
            // different token -- 64 chained layers leave neighbouring rows
            // correlated, so the cosine alone is the weaker of the two.
            // (memory: q38-prefill-cosine-decays-but-argmax-exact)
            const bf16* lg2 = lmh.run(x_pre.data() + (size_t)(last - 1) * D,
                                      norm_base.data());
            std::vector<float> prev((size_t)vocab);
            for (int j = 0; j < vocab; j++) prev[j] = (float)lg2[j];
            report_control("row P-2 does not pass as row P-1",
                           cosine(prev.data(), logits_pre.data(), vocab));
            std::snprintf(buf, sizeof buf, "row %d -> %d, row %d -> %d",
                          last, dev_tok, last - 1, argmax_f(prev.data(), vocab));
            report("the two rows choose different tokens",
                   argmax_f(prev.data(), vocab) != dev_tok, buf);

            // ---- how much room bf16 had -------------------------------
            float best = dev[0], second = -1e30f;
            for (int j = 1; j < vocab; j++) {
                if (dev[j] > best)        { second = best; best = dev[j]; }
                else if (dev[j] > second) { second = dev[j]; }
            }
            const float ulp = std::fabs(best) * (1.0f / 256.0f);
            std::snprintf(buf, sizeof buf,
                          "top1 %.4f  gap %.4f  bf16 ulp %.4f%s", best,
                          best - second, ulp,
                          (best - second) <= ulp ? "  <- a coin toss" : "");
            // Reported, not gated. A narrow margin here does not make the
            // token wrong; it makes the AGREEMENT above luck rather than
            // structure, and that is worth seeing when it changes.
            info("prefill top-2 margin", buf);
        }
    }

    std::printf("\n== N7: %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
