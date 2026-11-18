// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify_n11_lm_head8.cpp
/// \brief N11 -- the verify phase's vocabulary projection on
///        `lm_head_8.xclbin`, all k+1 rows in one dispatch.
///
/// Build (ONE line; the continuations are dropped because a backslash at the
/// end of a // comment splices the next line into it):
///   source ~/setup.sh
///   g++-13 -O3 -march=native -ffast-math -fopenmp -std=c++20
///       -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n11_lm_head8.cpp common/npu/lm_head8_sequence.cpp
///       ../q4nx/q4_npu_eXpress.cpp ../q4nx/safe_tensors.cpp
///       -L../../../../src/lib/xrt -L../../build/lib -L/opt/xilinx/xrt/lib
///       -lxrt_coreutil -laiebu -luuid -lpthread -o /tmp/verify_n11_lm_head8
///
/// **The flags matter.** This driver times the host against the device, and at
/// -O2 the host reference runs 1.7x slow and the comparison changes sign. The
/// line above is the engine's own flag set.
/// (memory: host-vs-device-ab-needs-shipping-flags)
///
/// ---------------------------------------------------------------------------
/// What this is checking, and what N7 already checked
/// ---------------------------------------------------------------------------
/// N7 covers the L = 1 head on `lm_head.xclbin`, the backend for prefill and
/// the draft loop. This is the other shape: module 3 projects k+1 rows at once
/// and `lm_head_8.xclbin` reads the 1.35 GB table ONCE for all of them.
///
/// The two designs share an array topology, so they share a weight layout --
/// `reorder_cpy_q80` is byte-for-byte the same function in both host.cpp files
/// and `stage_lm_head_weights()` is reused unchanged. Check 1 re-derives that
/// rather than assuming it.
///
/// What is genuinely new, and therefore what most of this driver is about:
///
///   * the L = 8 sequence -- four shim BDs a round instead of three, x
///     streamed per round instead of broadcast once, y landing interleaved
///   * the on-core rms_norm. Like the L = 1 bitstream this one takes **raw**
///     x, but it normalises differently: no core holds a whole row, so the
///     sum of squares is accumulated across round 0's k loop, the per-column
///     `model.norm.weight` rides in as a ninth lane of the x stream, and the
///     reciprocal lands on the finished accumulator. Feeding it already-normed
///     x does not fault, it produces fluent wrong logits, so check 4 separates
///     them.
///   * `_scatter_y`'s un-interleave of [round][col][core][token][32]
///   * the short-batch path. `keep < 8` is engine code with no counterpart in
///     the design's harness: module 3 runs at k+1, and k is not always 7.
///
/// ---------------------------------------------------------------------------
/// The controls
/// ---------------------------------------------------------------------------
/// A cosine near 1 is evidence about arithmetic and says nothing about
/// addressing, so every score is paired with something that must COLLAPSE:
///
///   rotate the row      row i scored against row (i+1)%8's reference. Catches
///                       a dispatch that replayed a stale y BO, which is
///                       exactly how an unchecked timeout presents.
///                       (memory: unchecked-ert-state-looks-like-a-pass)
///   rotated re-dispatch the decisive one here. This design's first version
///                       filled its memtile x from a one-shot PROLOGUE BD; the
///                       ring steps over a prologue forever after the first
///                       pass, so dispatch 2's x landed in a weight BD and
///                       came back all-NaN -- while ERT still reported
///                       COMPLETED. Module 3 dispatches once per speculation
///                       cycle, so dispatch 2 is the normal case, not an edge
///                       case. Eight re-dispatches, token slots rotated each
///                       time, each scored against the row actually fed in.
///                       (memory: a-prologue-bd-breaks-the-second-dispatch,
///                                one-shot-bench-cannot-see-dispatch-state)
///   skip the norm       the same x projected raw. Scored by RELATIVE L2, not
///                       cosine: RMSNorm's rms division is a positive scalar,
///                       which cosine cannot see, and its w is nearly uniform
///                       (cos(w, ones) = 0.9975), so a cosine control here is
///                       unsatisfiable by construction. The norm is in the
///                       scale, so the score has to be.
///                       (memory: gate-a-control-against-the-error-floor,
///                                per-row-cosine-reports-the-norm)
///   row 0 into every    all eight slots fed row 0. If the token lanes were
///   slot                not independent -- a design that computes row 0 eight
///                       times -- the main check still scores 8/8, because the
///                       eight reference argmaxes would each be compared to a
///                       lane that happened to hold the right thing. Here
///                       every output must become row 0's.
///
/// ---------------------------------------------------------------------------
/// What this is NOT claiming
/// ---------------------------------------------------------------------------
/// Not a large speedup. The honest baseline is the host at the SAME L, not
/// eight times the host at L = 1: `q4nx_matmul` keeps its row loop inside its
/// weight tile loop, so the table streams once there too. Check 6 measures
/// both at the same flags and reports the difference without dressing it up.
/// (memory: host-lm-head-is-sublinear-in-rows)
///
/// And unlike the draft head, this backend is NOT output-neutral by
/// construction. Module 3's argmax is the committed token, and the device
/// returns bf16. Check 5 measures how often the top-2 gap is inside a bf16
/// ulp instead of assuming it never is.
/// (memory: qwen38-lm-head-untied-and-argmax-ties)
#include "common/cpu_ref_common.hpp"
#include "common/lm_head.hpp"
#include "common/npu/lm_head8_op.hpp"
#include "common/npu/lm_head_weights.hpp"

#include "npu_utils/npu_utils.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace cpu_ref::q38;

/// Rows in the verify dump, and the bitstream's L.
constexpr int kRows = q38npu::npu_lm_head_8::ROWS;

/// Against a same-bytes CPU reference: Q8_0 is common mode, so what is left is
/// the device's own arithmetic -- the bfp16 mmul floors both operands onto a
/// shared-exponent-per-8 grid, and every fp32 -> bf16 out truncates toward
/// -inf. (memory: score-dequant-mm-against-its-own-quantized-bytes,
///          aie-mmul-bf16-truncates-to-bfp16)
constexpr double kGate = 0.999;

/// Against the reference dump, which is numpy fp32 over unquantised weights,
/// so Q8_0's own loss is not common mode and shows up here too.
constexpr double kGateRef = 0.999;

/// The ceiling a control must fall under.
constexpr double kControl = 0.99;

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

/// ||a - b|| / ||b||. The scale-sensitive score: a cosine cannot see a
/// per-row rescaling at all, and the RMSNorm this driver has to check is
/// mostly a per-row rescaling. (memory: per-row-cosine-reports-the-norm)
double rel_l2(const float* a, const float* b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)a[i] - b[i];
        num += d * d;
        den += (double)b[i] * b[i];
    }
    if (den == 0) return 1e30;
    return std::sqrt(num / den);
}

int argmax_f(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/// The reference logits are bf16-rounded upstream, so its top-1 is sometimes
/// an exact tie (verify row 1 has two tokens at 24.75). Which of a tied pair
/// wins is arbitrary, so score against the reference's own VALUE at the token
/// the device picked rather than against the index numpy broke the tie toward.
bool argmax_agrees(const float* ref, int n, int got) {
    return ref[got] >= ref[argmax_f(ref, n)];
}

/// Gap between the top two entries of a row.
double top2_gap(const float* v, int n) {
    double best = v[0], second = -1e30;
    for (int i = 1; i < n; i++) {
        const double x = v[i];
        if (x > best)        { second = best; best = x; }
        else if (x > second) { second = x; }
    }
    return best - second;
}

/// A bf16 ulp at magnitude `m`: 8 bits of mantissa.
double bf16_ulp(double m) {
    return std::abs(m) * std::pow(2.0, -8.0);
}

} // namespace

int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string lm8_xclbin =
        argc > 2 ? argv[2]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "lm_head_8/build/QWEN3_8_27B/xclbins/lm_head.xclbin";

    std::printf("== N11: the verify lm_head on lm_head_8.xclbin, L = %d ==\n",
                kRows);
    std::printf("   model    %s\n   lm_head  %s\n\n",
                model_dir.c_str(), lm8_xclbin.c_str());

    // ======================================================================
    // inputs
    // ======================================================================
    cpu_ref::safetensors ref(model_dir + "/ref.safetensors");
    cpu_ref::safetensors q4nx(model_dir + "/model.q4nx");

    const std::vector<float> x_ref      = ref.f32("verify.layer_63_out");
    const std::vector<float> logits_ref = ref.f32("verify.logits");
    const std::vector<float> norm_base  = q4nx.f32("model.norm.weight");

    const cpu_ref::q4nx_tensor W = cpu_ref::get_q4nx(q4nx, "lm_head.weight");
    const int vocab = W.rows;
    const int D     = W.cols;

    if ((int)(x_ref.size() / D) != kRows ||
        logits_ref.size() != (size_t)kRows * vocab) {
        std::fprintf(stderr, "ref.safetensors: verify.layer_63_out / "
                             "verify.logits do not match %d x %d\n", kRows, vocab);
        return 1;
    }

    // ======================================================================
    // 1. geometry, and the claim that the weight layout is shared
    // ======================================================================
    std::printf("-- 1. geometry --\n");
    {
        char buf[200];
        const int padded = q38npu::lm_head_vocab_padded(vocab);

        // The L = 8 design pads to the same 32*32 round as L = 1. This is the
        // load-bearing fact behind reusing stage_lm_head_weights(): if the two
        // tilings differed the staging would be silently wrong for one of
        // them, and nothing would fault.
        std::snprintf(buf, sizeof buf, "%d -> %d (%d rounds of %d)", vocab,
                      padded, padded / q38npu::lm_head8_tiles::M_PER_ROUND,
                      q38npu::lm_head8_tiles::M_PER_ROUND);
        report("padded vocab is a whole number of rounds",
               padded % q38npu::lm_head8_tiles::M_PER_ROUND == 0 &&
                   padded >= vocab, buf);

        std::snprintf(buf, sizeof buf, "L=8 %d vs L=1 %d",
                      q38npu::lm_head8_tiles::M_PER_ROUND,
                      q38npu::lm_head_tiles::M_PER_ROUND);
        report("the two designs share a round, so a weight layout",
               q38npu::lm_head8_tiles::M_PER_ROUND ==
                   q38npu::lm_head_tiles::M_PER_ROUND, buf);

        // Upstream sizes the BO as D*VP + D*VP/32*2; lm_head_weight_bytes
        // writes the same thing as VP*D*17/16. If they ever disagree the
        // sequence walks off the end of the BO, which does not fault.
        const size_t ours     = q38npu::lm_head_weight_bytes(padded, D);
        const size_t upstream = (size_t)D * padded + (size_t)D * padded / 32 * 2;
        std::snprintf(buf, sizeof buf, "%zu B, upstream %zu B", ours, upstream);
        report("weight BO size matches the design's host", ours == upstream, buf);

        // x goes out as [k_block][token][256]; D must divide into whole
        // 256-wide blocks or the interleave is not expressible.
        std::snprintf(buf, sizeof buf, "D = %d, %d blocks of %d", D,
                      D / q38npu::lm_head8_tiles::K_CHUNK,
                      q38npu::lm_head8_tiles::K_CHUNK);
        report("D tiles into whole Q8_0 column blocks",
               D % q38npu::lm_head8_tiles::K_CHUNK == 0, buf);
    }

    // ======================================================================
    // 2. the device
    // ======================================================================
    std::printf("\n-- 2. staging (1.35 GB) --\n");
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);

    const auto t_stage0 = std::chrono::steady_clock::now();
    npu_app_manager* lm_mgr = mgr.register_xclbin(lm8_xclbin);
    q38npu::npu_lm_head_8 lmh(lm_mgr->create_app(), q4nx);
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
        // an addressing bug touching one core's slots would survive any
        // sampling scheme that did not happen to land on it.
        // (memory: move-weights-offset-is-in-elements)
        const int      padded_nrb = lmh.vocab_padded() / cpu_ref::Q4NX_ROW_BLOCK;
        const uint8_t* dst        = lmh.weight_bo().data();
        long bad = 0;
#pragma omp parallel for schedule(static) reduction(+ : bad)
        for (int rb = 0; rb < padded_nrb; rb++)
            for (int cb = 0; cb < W.ncb; cb++) {
                const uint8_t* src =
                    (rb < W.nrb)
                        ? W.blocks + ((size_t)rb * W.ncb + cb) * W.block_bytes
                        : W.blocks;
                if (std::memcmp(dst + q38npu::lm_head_block_offset(rb, cb, W.ncb),
                                src, cpu_ref::Q80_BLOCK_BYTES) != 0)
                    bad++;
            }
        std::snprintf(buf, sizeof buf, "%d x %d blocks, %ld wrong",
                      padded_nrb, W.ncb, bad);
        report("every staged block equals its source", bad == 0, buf);
    }

    // ======================================================================
    // 3. the projection -- all 8 rows, one dispatch
    // ======================================================================
    std::printf("\n-- 3. the projection, %d rows in one dispatch --\n", kRows);

    std::vector<float> got((size_t)kRows * vocab);
    double dev_ms = 0;
    {
        const auto t0 = std::chrono::steady_clock::now();
        lmh.run(x_ref.data(), kRows, norm_base.data(), kRows, got.data(),
                RMS_EPS);
        dev_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0).count();
        info("dispatch", std::to_string((long)dev_ms) + " ms for " +
                             std::to_string(kRows) + " rows");
    }

    // A same-bytes reference: the host reads the SAME Q8_0 blocks, so Q8_0's
    // loss is common mode and what is left is the device's own arithmetic.
    // x is rounded to bf16 first because that is what the upload sends -- a
    // reference that projects fp32 x charges the device for a rounding it did
    // not choose. (memory: aie-to-vector-bf16-floors)
    std::vector<float> same_bytes((size_t)kRows * vocab);
    {
        std::vector<float> xn((size_t)kRows * D);
        for (int t = 0; t < kRows; t++) {
            cpu_ref::rms_norm(x_ref.data() + (size_t)t * D, norm_base.data(), D,
                              xn.data() + (size_t)t * D, RMS_EPS);
            for (int j = 0; j < D; j++) {
                float& v = xn[(size_t)t * D + j];
                v = cpu_ref::round_bf16(v);
            }
        }
        cpu_ref::q4nx_matmul(W, xn.data(), kRows, same_bytes.data());
    }

    {
        int hits = 0, hits_same = 0;
        double worst_ref = 1.0, worst_same = 1.0;
        for (int t = 0; t < kRows; t++) {
            const float* g = got.data() + (size_t)t * vocab;
            const float* r = logits_ref.data() + (size_t)t * vocab;
            const float* s = same_bytes.data() + (size_t)t * vocab;
            const int    a = argmax_f(g, vocab);
            if (argmax_agrees(r, vocab, a)) hits++;
            if (argmax_agrees(s, vocab, a)) hits_same++;
            worst_ref  = std::min(worst_ref,  cosine(g, r, vocab));
            worst_same = std::min(worst_same, cosine(g, s, vocab));
        }
        char buf[200];
        std::snprintf(buf, sizeof buf, "%d/%d rows", hits, kRows);
        report("argmax matches verify.logits", hits == kRows, buf);
        std::snprintf(buf, sizeof buf, "%d/%d rows", hits_same, kRows);
        report("argmax matches its own Q8_0 bytes", hits_same == kRows, buf);
        report_cos("worst-row cos vs verify.logits", worst_ref, kGateRef);
        report_cos("worst-row cos vs the same-bytes host", worst_same, kGate);
    }

    // Control: score row i against row (i+1)'s reference. This is what a
    // stale y BO or a lane mix-up looks like, and it must collapse.
    {
        double worst = 0;
        for (int t = 0; t < kRows; t++)
            worst = std::max(worst,
                             cosine(got.data() + (size_t)t * vocab,
                                    logits_ref.data() +
                                        (size_t)((t + 1) % kRows) * vocab,
                                    vocab));
        report_control("control: rows rotated against the reference", worst);
    }

    // ======================================================================
    // 4. the on-core norm is real
    // ======================================================================
    std::printf("\n-- 4. the norm (this bitstream takes RAW x and norms on the core) --\n");
    {
        // A COSINE CANNOT CHECK THIS, and the first version of this driver
        // failed here for that reason rather than because anything was wrong.
        //
        // RMSNorm is (x / rms(x)) (*) w. The division is a positive scalar and
        // cosine is invariant to it, so the only part that can rotate the
        // vector is the elementwise w -- and w is nearly uniform. Measured on
        // this checkpoint: mean 1.944, sd 0.138, cos(w, ones) = 0.9975. So
        // cos(normed, raw) is pinned near 1 by construction and no cosine gate
        // can separate them.
        // (memory: gate-a-control-against-the-error-floor)
        //
        // The norm lives in the SCALE: rms(x) runs 4.5 .. 6.5 across these
        // eight rows, a different factor for each. So the control is relative
        // L2. (memory: per-row-cosine-reports-the-norm,
        //          score-elementwise-op-with-a-control)
        std::vector<float> raw((size_t)kRows * vocab);
        std::vector<float> xb((size_t)kRows * D);
        for (size_t i = 0; i < xb.size(); i++)
            xb[i] = cpu_ref::round_bf16(x_ref[i]);
        cpu_ref::q4nx_matmul(W, xb.data(), kRows, raw.data());

        char   buf[220];
        double worst_cos = 0, lo_ratio = 1e30, hi_ratio = 0, worst_rel_ok = 0;
        int    argmax_survives = 0;
        for (int t = 0; t < kRows; t++) {
            const float* g = got.data() + (size_t)t * vocab;
            const float* r = raw.data() + (size_t)t * vocab;
            const float* s = same_bytes.data() + (size_t)t * vocab;
            worst_cos = std::max(worst_cos, cosine(g, r, vocab));
            worst_rel_ok = std::max(worst_rel_ok, rel_l2(g, s, vocab));
            if (argmax_f(g, vocab) == argmax_f(r, vocab)) argmax_survives++;

            // The quantity the norm actually moves: ||raw|| / ||normed||,
            // which is rms(x) / mean(w) and so differs per row. Exactly 1
            // would mean the norm never ran.
            double ng = 0, nr = 0;
            for (int i = 0; i < vocab; i++) {
                ng += (double)g[i] * g[i];
                nr += (double)r[i] * r[i];
            }
            const double ratio = std::sqrt(nr / ng);
            lo_ratio = std::min(lo_ratio, ratio);
            hi_ratio = std::max(hi_ratio, ratio);
        }

        std::snprintf(buf, sizeof buf,
                      "cos = %.6f -- near 1 by construction, not a result",
                      worst_cos);
        info("un-normed x, scored by cosine", buf);

        // rel L2 is the wrong shape here too: normalised by the raw vector it
        // saturates just under 1 whichever way the scale went. The ratio says
        // which way and by how much.
        std::snprintf(buf, sizeof buf,
                      "||raw|| / ||normed|| = %.2f .. %.2f (must clear 1.5)",
                      lo_ratio, hi_ratio);
        report("control: x projected without the norm", lo_ratio > 1.5, buf);

        std::snprintf(buf, sizeof buf, "rel L2 <= %.5f (must be < 0.01)",
                      worst_rel_ok);
        report("the device's scale matches a normed reference",
               worst_rel_ok < 0.01, buf);

        // The sobering number, and the reason check 3 is the real guard here:
        // argmax is scale-invariant and w barely rotates, so dropping the norm
        // entirely would still emit mostly the same tokens. A driver that only
        // compared emitted tokens would not notice a missing rms_norm.
        std::snprintf(buf, sizeof buf,
                      "%d/%d argmaxes unchanged with the norm dropped entirely",
                      argmax_survives, kRows);
        info("what a token-level check would miss", buf);
    }

    // ======================================================================
    // 5. the token lanes are independent
    // ======================================================================
    std::printf("\n-- 5. the eight lanes carry eight different rows --\n");
    {
        // Feed row 0 into all eight slots. A design that silently computes one
        // row eight times passes check 3 -- each reference argmax would be
        // compared against a lane that happened to hold the right thing. Here
        // every lane must come back as row 0's.
        std::vector<float> rep_x((size_t)kRows * D);
        for (int t = 0; t < kRows; t++)
            std::memcpy(rep_x.data() + (size_t)t * D, x_ref.data(),
                        (size_t)D * sizeof(float));

        std::vector<float> rep((size_t)kRows * vocab);
        lmh.run(rep_x.data(), kRows, norm_base.data(), kRows, rep.data(),
                RMS_EPS);

        const int a0 = argmax_f(got.data(), vocab);
        int same = 0;
        for (int t = 0; t < kRows; t++)
            if (argmax_f(rep.data() + (size_t)t * vocab, vocab) == a0) same++;

        char buf[200];
        std::snprintf(buf, sizeof buf, "%d/%d lanes returned row 0's argmax %d",
                      same, kRows, a0);
        report("row 0 in every slot comes back as row 0", same == kRows, buf);

        // And the eight real argmaxes are distinct, which is the other half of
        // the same claim: if they were not, check 3 would be weaker than it
        // looks.
        int distinct = 0;
        for (int t = 0; t < kRows; t++) {
            const int a = argmax_f(got.data() + (size_t)t * vocab, vocab);
            bool seen = false;
            for (int u = 0; u < t; u++)
                if (argmax_f(got.data() + (size_t)u * vocab, vocab) == a)
                    seen = true;
            if (!seen) distinct++;
        }
        std::snprintf(buf, sizeof buf, "%d/%d distinct", distinct, kRows);
        report("the reference rows have distinct argmaxes", distinct == kRows,
               buf);
    }

    // ======================================================================
    // 6. re-dispatch -- the check a one-shot driver cannot make
    // ======================================================================
    std::printf("\n-- 6. eight more dispatches, token slots rotated --\n");
    {
        // The whole reason this design was not usable until now. Module 3
        // dispatches once per speculation cycle, so a fault that only appears
        // on dispatch 2 is a fault on every real call. Rotating matters as
        // much as repeating: a design replaying dispatch 1's x would keep
        // returning dispatch 1's argmaxes and score 8/8 against an unrotated
        // expectation.
        // (memory: a-prologue-bd-breaks-the-second-dispatch)
        std::vector<float> rot_x((size_t)kRows * D);
        std::vector<float> out((size_t)kRows * vocab);
        int    bad_rows = 0, nan_rows = 0;
        double worst    = 1.0, best_ms = 1e30, total_ms = 0;

        for (int it = 1; it <= kRows; it++) {
            for (int t = 0; t < kRows; t++) {
                const int row = (t + it) % kRows;
                std::memcpy(rot_x.data() + (size_t)t * D,
                            x_ref.data() + (size_t)row * D,
                            (size_t)D * sizeof(float));
            }
            const auto t0 = std::chrono::steady_clock::now();
            lmh.run(rot_x.data(), kRows, norm_base.data(), kRows, out.data(),
                    RMS_EPS);
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count();
            total_ms += ms;
            best_ms = std::min(best_ms, ms);

            for (int t = 0; t < kRows; t++) {
                const int    row = (t + it) % kRows;
                const float* g   = out.data() + (size_t)t * vocab;
                const float* r   = logits_ref.data() + (size_t)row * vocab;
                if (!std::isfinite(g[0]) || !std::isfinite(g[vocab / 2]))
                    nan_rows++;
                if (!argmax_agrees(r, vocab, argmax_f(g, vocab))) bad_rows++;
                worst = std::min(worst, cosine(g, r, vocab));
            }
        }

        char buf[200];
        std::snprintf(buf, sizeof buf, "%d rows over %d dispatches", nan_rows,
                      kRows);
        report("no dispatch returns NaN", nan_rows == 0, buf);
        std::snprintf(buf, sizeof buf, "%d/%d rows wrong", bad_rows,
                      kRows * kRows);
        report("every slot still carries the row fed into it", bad_rows == 0,
               buf);
        report_cos("worst row over all re-dispatches", worst, kGateRef);
        std::snprintf(buf, sizeof buf, "%.0f ms best, %.0f ms mean over %d",
                      best_ms, total_ms / kRows, kRows);
        info("steady-state dispatch", buf);
        dev_ms = best_ms;   // the number the engine actually sees
    }

    // ======================================================================
    // 7. the short batch -- k+1 is not always 8
    // ======================================================================
    std::printf("\n-- 7. keep < %d (module 3 runs at k+1) --\n", kRows);
    {
        // `k = min(max_draft, max_steps())`, so a caller asking for fewer
        // drafts gives verify fewer rows. The array has no shorter shape, so
        // run() pads the spare lanes and trims the output; this is engine code
        // with no counterpart in the design's own harness.
        //
        // norm_and_forward's contract is a SUFFIX: keep = 3 of 8 rows means
        // rows 5, 6, 7. Getting that off by the offset would be invisible in
        // any cosine that did not check which row it got.
        bool all_ok = true;
        char buf[200];
        for (int keep = 1; keep <= kRows; keep++) {
            std::vector<float> part((size_t)keep * vocab);
            lmh.run(x_ref.data(), kRows, norm_base.data(), keep, part.data(),
                    RMS_EPS);
            for (int t = 0; t < keep; t++) {
                const int    row = kRows - keep + t;
                const float* g   = part.data() + (size_t)t * vocab;
                const float* r   = logits_ref.data() + (size_t)row * vocab;
                if (!argmax_agrees(r, vocab, argmax_f(g, vocab)) ||
                    cosine(g, r, vocab) < kGateRef) {
                    std::snprintf(buf, sizeof buf,
                                  "keep=%d slot %d (row %d) cos %.6f", keep, t,
                                  row, cosine(g, r, vocab));
                    all_ok = false;
                }
            }
            if (!all_ok) break;
        }
        if (all_ok)
            std::snprintf(buf, sizeof buf, "keep = 1 .. %d, suffix rows", kRows);
        report("a short batch projects the right suffix", all_ok, buf);
    }

    // ======================================================================
    // 8. bf16 and the committed token
    // ======================================================================
    std::printf("\n-- 8. what bf16 costs module 3 --\n");
    {
        // Module 3's argmax IS the emitted token -- there is no downstream
        // verification the way there is on the draft path. So the honest
        // statement is not "the two backends agree" but "here is how often the
        // top-1 gap is inside the resolution that decides it".
        int inside = 0;
        double worst_ratio = 1e30;
        for (int t = 0; t < kRows; t++) {
            const float* s   = same_bytes.data() + (size_t)t * vocab;
            const double gap = top2_gap(s, vocab);
            const double ulp = bf16_ulp(s[argmax_f(s, vocab)]);
            if (gap < ulp) inside++;
            worst_ratio = std::min(worst_ratio, gap / ulp);
        }
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "%d/%d rows have a top-2 gap inside a bf16 ulp "
                      "(narrowest %.2f ulp)", inside, kRows, worst_ratio);
        info("top-2 margin", buf);

        // The claim that IS gateable: every row the device disagrees with the
        // fp32 host on must be a row where fp32 itself had a tie or a
        // sub-ulp gap. A disagreement on a well-separated row is a bug, not
        // rounding.
        int unexplained = 0;
        for (int t = 0; t < kRows; t++) {
            const float* g = got.data() + (size_t)t * vocab;
            const float* s = same_bytes.data() + (size_t)t * vocab;
            const int    a = argmax_f(g, vocab), b = argmax_f(s, vocab);
            if (a == b) continue;
            const double gap = top2_gap(s, vocab);
            const double ulp = bf16_ulp(s[b]);
            if (gap > 2.0 * ulp) unexplained++;
        }
        std::snprintf(buf, sizeof buf, "%d rows", unexplained);
        report("no disagreement on a well-separated row", unexplained == 0, buf);
    }

    // ======================================================================
    // 9. what it is worth
    // ======================================================================
    std::printf("\n-- 9. against the host, at the same L --\n");
    {
        // The baseline is the host at L = 8, NOT eight times the host at
        // L = 1. q4nx_matmul keeps its row loop inside its weight tile loop,
        // so the 1.35 GB table streams once there too and eight rows cost
        // ~3x one row rather than 8x.
        // (memory: host-lm-head-is-sublinear-in-rows)
        cpu_ref::lm_head host(q4nx);
        std::vector<float> h((size_t)kRows * vocab);

        host.norm_and_forward(x_ref.data(), kRows, norm_base.data(), kRows,
                              h.data(), RMS_EPS);   // warm
        double host_ms = 1e30;
        for (int i = 0; i < 3; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            host.norm_and_forward(x_ref.data(), kRows, norm_base.data(), kRows,
                                  h.data(), RMS_EPS);
            host_ms = std::min(host_ms,
                               std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - t0).count());
        }

        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "device %.0f ms, host %.0f ms -- %.0f ms/cycle, %.2fx",
                      dev_ms, host_ms, host_ms - dev_ms, host_ms / dev_ms);
        info("8 rows", buf);
        report("the device is not slower than the host", dev_ms <= host_ms, buf);

        // And the host's own scaling, stated so the ratio above cannot be
        // mistaken for 8x.
        std::vector<float> h1((size_t)vocab);
        host.norm_and_forward(x_ref.data(), kRows, norm_base.data(), 1,
                              h1.data(), RMS_EPS);
        double host1_ms = 1e30;
        for (int i = 0; i < 3; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            host.norm_and_forward(x_ref.data(), kRows, norm_base.data(), 1,
                                  h1.data(), RMS_EPS);
            host1_ms = std::min(host1_ms,
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0).count());
        }
        std::snprintf(buf, sizeof buf, "L=1 %.0f ms, L=%d %.0f ms (%.1fx, not %dx)",
                      host1_ms, kRows, host_ms, host_ms / host1_ms, kRows);
        info("the host is sublinear in rows", buf);
    }

    std::printf("\n== N11: %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
