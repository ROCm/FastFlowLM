// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify_n16_vision_mm.cpp
/// \brief N16 -- the vision tower's matmuls on the NPU, against the host op
///        they replace and against the captured HF reference.
///
/// Build (one line; a backslash at the end of a // comment splices the next
/// line into it, so the continuations are dropped here):
///
///   source ~/setup.sh
///   g++-13 -O3 -march=native -fopenmp -std=c++20 -ffast-math
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n16_vision_mm.cpp image/npu/vision_mm_sequence.cpp
///       image/npu/vision_attn_sequence.cpp image/image_cpu.cpp
///       image/vision_ops.cpp image/vision_glue.cpp image/vision_rope.cpp
///       image/mrope.cpp
///       -L../../../../src/lib/xrt -L../../build/lib -L/opt/xilinx/xrt/lib
///       -lxrt_coreutil -laiebu -luuid -lpthread
///       -o /tmp/verify_n16_vision_mm
///
/// **-O3, not -O2, and that is not cosmetic.** The Makefile builds the engine
/// at -O3 and the host arm of check 10 is `image/vision_ops.cpp` compiled into
/// THIS binary. N15 measured what the wrong flag does to the same source: a
/// 2.6-3x handicap, enough to flip a verdict's sign. The off-arm has to be
/// built the way it ships.
/// (memory: host-vs-device-ab-needs-shipping-flags, ab-off-arm-is-not-the-host)
///
/// Run:
///   /tmp/verify_n16_vision_mm [model_dir] [mm_a.xclbin] [mm_b.xclbin]
///                             [vision_attn.xclbin]
///   FLM_Q38_VISION_FP32=1 /tmp/verify_n16_vision_mm          # precision arm
///   FLM_Q38_VISION_MM_PREROUND=0 /tmp/verify_n16_vision_mm   # pre-round off
///
/// ---------------------------------------------------------------------------
/// What is under test
/// ---------------------------------------------------------------------------
/// `vision_mm` computes ONLY
///
///     y[M][N] = act(x[M][K] W[N][K]^T + bias[N])
///
/// with `act` in {none, silu, gelu_tanh}. That is 111 of the tower's ops and,
/// at the real operating point, essentially all of its time: at L = 1200 the
/// attention is ~0.2 s of an ~8 s image and these are the other ~7 s.
///
/// The reference is `vision::cpu_mm_op` on the SAME bf16 weight bytes -- not
/// torch, and not an fp32 recomputation from a different source. The weights
/// are raw bf16 in the checkpoint and `cpu_ops_emulate_bf16()` rounds the
/// activations too, so the only thing left between the two arms is the device's
/// arithmetic and its addressing.
/// (memory: score-dequant-mm-against-its-own-quantized-bytes)
///
/// ---------------------------------------------------------------------------
/// Why nothing here is gated on cosine
/// ---------------------------------------------------------------------------
/// Same reason as N15, and worse. These outputs are 1152 to 5120 columns of
/// smooth activations; two independent such tensors routinely score cos 0.9,
/// and N14 measured a dropped rotary at cos 0.9975. Worse, a FUSED output hides
/// its narrow tensors: `qkv` is q|k|v side by side, and a broken k rides behind
/// a healthy q at an aggregate cosine of 0.9999.
///
/// So every magnitude claim is gated on **relative L2**, the per-N-band relL2
/// is broken out at the design's `tile_N` granularity -- one band is one array
/// column's output, which is where an addressing error lives -- and `qkv` is
/// additionally broken out per third.
/// (memory: fused-matmul-cosine-hides-narrow-tensors, per-row-cosine-reports-the-norm)
///
/// ---------------------------------------------------------------------------
/// The floor this is gated against
/// ---------------------------------------------------------------------------
/// The kernel is built with -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16, so both
/// mmul operands are floored onto a shared-exponent-per-8 grid -- 7 mantissa
/// bits, not bf16's 8 -- and the accumulator's every fp32 -> bf16 truncates
/// toward -inf rather than rounding.
/// (memory: aie-mmul-bf16-truncates-to-bfp16, aie-to-vector-bf16-floors)
///
/// **`bind()` pre-rounds the weights onto that grid**, which turns the
/// hardware's truncation into a no-op and removes its bias. Measured here on
/// `patch_embed`, the worst-conditioned shape: relL2 0.1194 -> 0.0063, cosine
/// 0.9929 -> 0.99998. Check 5 is what holds that in place, and it holds it by
/// asking which of two host references the device agrees with rather than by
/// putting a bar on an error.
///
/// What pre-rounding does NOT fix is a band whose weights have a wide dynamic
/// range inside one group of 8 -- there the shared exponent destroys the small
/// entries outright, and rounding rather than truncating only removes the bias,
/// not the variance. `merger.fc2`'s band 3 is that band, and it is the reason
/// that one shape's bar is looser than the rest.
///
/// That floor is large enough to swallow a subtle bug, which is why five of the
/// ten checks are controls rather than scores:
///
///   a rotated weight                  must COLLAPSE, at TWO shifts. A
///       transposed or permuted array mapping agrees with the true reference
///       AND with itself, so only a reference computed from different weights
///       separates them -- and the offset must not be a symmetry of the tensor
///       being rotated, which N/2 turns out to be for `merger.fc1`.
///   the pre-round                     the device must agree with the EXACT
///       weights and disagree with the floored ones. With the pre-round off it
///       is the other way round, and the check flips with it.
///   fused gelu vs none + host gelu    must AGREE, and both must DIFFER from
///       the unactivated output. Dividing the activation back out is
///       conditioning-limited; the number that proves the RTP reached the core
///       is the comparison against the UN-applied result.
///       (memory: score-an-elementwise-op-with-a-control)
///   the merger's erf arm              must differ from the fused tanh arm and
///       must equal a host erf of the device's own unactivated output. The
///       DIRECTION (erf beats tanh) cannot be gated here: the device output is
///       bf16 either way, and N14 already measured that the two forms are
///       indistinguishable below that floor -- tanh was ahead by 6e-7.
///       (memory: gate-a-control-against-the-error-floor)
///   M = 300 after M = 1200            the pad rows are not inert. There is no
///       mask anywhere in this design, so a previous image's rows left above
///       the cursor are multiplied and accumulated. Bit identity against a
///       fresh op is the right bar for a question about stale memory.
///       (memory: kv-rows-above-cursor-are-not-inert,
///                gate-an-overlapped-replay-on-bit-identity)
///
/// And the sizes are deliberate. The captured dump is a 16x16 grid, L = 256, so
/// every real shape fits in ONE array pass of design A (tile_M 256) and the
/// round loop, the BD ping-pong and the lagged wait are entirely untested by
/// it. 1200 is the real operating point: a 639x480 image smart_resizes to a
/// 30x40 grid.
/// (memory: vision-tower-operating-point-is-l1200)
#include "image/image_cpu.hpp"
#include "image/npu/vision_mm_op.hpp"
#include "image/npu/vision_op_factory.hpp"
#include "image/vision_model_def.hpp"
#include "image/vision_ops.hpp"
#include "image/vision_weights.hpp"

#include "common/cpu_ref_common.hpp"
#include "npu_utils/npu_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using qwen38::vision::activation;

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-46s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
    if (ok) n_pass++; else n_fail++;
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[640];
    va_list ap; va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return std::string(buf);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

/// relL2 over one column band, all rows.
///
/// An aggregate relL2 is a norm-weighted mean over N columns, so a band 6x
/// worse than the rest moves it by well under the floor. One band of `tile_N`
/// columns is what ONE array column produced, which is exactly where an
/// addressing error lives.
/// (memory: fused-matmul-cosine-hides-narrow-tensors)
double band_rel_l2(const float* a, const float* b, int rows, int N,
                   int c0, int c1) {
    double num = 0, den = 0;
    for (int r = 0; r < rows; r++) {
        const float* pa = a + (size_t)r * N;
        const float* pb = b + (size_t)r * N;
        for (int c = c0; c < c1; c++) {
            const double d = (double)pa[c] - pb[c];
            num += d * d;
            den += (double)pb[c] * pb[c];
        }
    }
    return den > 0 ? std::sqrt(num / den) : 0.0;
}

/// Worst band of `width`, and which one, for diagnosis.
double worst_band(const float* a, const float* b, int rows, int N, int width,
                  int* which) {
    double worst = 0.0;
    if (width <= 0 || width > N) width = N;
    for (int c = 0; c + width <= N; c += width) {
        const double e = band_rel_l2(a, b, rows, N, c, c + width);
        if (e > worst) { worst = e; if (which) *which = c / width; }
    }
    return worst;
}

/// Smallest per-row cosine. Reported, never gated: a minimum over N rows drifts
/// down with N even at uniform error.
/// (memory: worst-row-cosine-is-a-minimum-over-n)
double worst_row_cos(const float* a, const float* b, int rows, int cols,
                     int* which) {
    double worst = 2.0;
    for (int r = 0; r < rows; r++) {
        const cpu_ref::err_stats s =
            cpu_ref::error_matrix(a + (size_t)r * cols, b + (size_t)r * cols,
                                  (size_t)cols);
        if (s.cosine < worst) { worst = s.cosine; if (which) *which = r; }
    }
    return worst;
}

/// Stretch a real [rows][cols] tensor to `want` rows with a small jitter.
///
/// The captured image is 16x16 patches, so the dump alone cannot reach a second
/// array pass. Repeating rows is not a substitute for a real 30x40 capture, but
/// it is the right KIND of input: real activation magnitudes and real column
/// structure, which is what the bf16 range floor cares about. What it tests is
/// the round loop and the BD ping-pong, which are pure addressing and do not
/// care that row 300 is row 44 with noise on it.
std::vector<float> tile_rows(const std::vector<float>& in, int rows, int cols,
                             int want, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> jitter(0.0f, 0.02f);
    std::vector<float> out((size_t)want * cols);
    for (int r = 0; r < want; r++) {
        const int src = r % rows;
        for (int c = 0; c < cols; c++)
            out[(size_t)r * cols + c] =
                in[(size_t)src * cols + c] + jitter(rng);
    }
    return out;
}

/// Time one op, as the **minimum** over several runs.
///
/// Not the mean. A first pass of this driver took the mean of 3 and the host
/// arm at L=256 swung 2.2x between two runs of the same binary -- scheduling
/// and thermal noise is one-sided, so a mean of a handful of samples is an
/// estimate of the noise as much as of the work. The minimum is the cleanest
/// run that actually happened, which is what an A/B between two implementations
/// wants. Both arms are measured the same way, so neither is flattered.
double bench_min(qwen38::vision::mm_op& dev, qwen38::vision::mm_op& host,
                 const std::vector<float>& x, int M, bool device) {
    qwen38::vision::mm_op& op = device ? dev : host;
    std::vector<float> y((size_t)M * op.N());
    auto once = [&] { op.run(x.data(), M, y.data()); };
    once(); once();                       // warm the BOs and the page cache
    const int iters = device ? 10 : 6;
    double best = 1e30;
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::steady_clock::now();
        once();
        best = std::min(best, ms_since(t0));
    }
    return best;
}

/// One matmul under test.
struct shape {
    const char*     tag;
    const uint16_t* w;
    const uint16_t* b;
    int             K, N;
    activation      act;
    bool            want_b;   ///< expected to land on design B
};

}  // namespace

// ===========================================================================
int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string mm_root =
        "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/vision_mm/"
        "build/";
    const std::string xclbin_a =
        argc > 2 ? argv[2] : mm_root + "VISION_MM_A/xclbins/mm.xclbin";
    const std::string xclbin_b =
        argc > 3 ? argv[3] : mm_root + "VISION_MM_B/xclbins/mm.xclbin";
    const std::string xclbin_attn =
        argc > 4 ? argv[4]
                 : "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/"
                   "vision_attn/build/xclbins/attn.xclbin";
    const std::string ref_path = model_dir + "/vision_ref.safetensors";

    const bool bf16 = qwen38::vision::cpu_ops_emulate_bf16();

    std::printf("== N16: the vision tower's matmuls on the NPU ==\n");
    std::printf("   model  %s\n   mm A   %s\n   mm B   %s\n   ref    %s\n"
                "   mode   %s\n\n",
                model_dir.c_str(), xclbin_a.c_str(), xclbin_b.c_str(),
                ref_path.c_str(),
                bf16 ? "bf16 host reference (shipping default)"
                     : "fp32 host reference (FLM_Q38_VISION_FP32=1)");

    // The bar. `vision_mm`'s own harness gates at cosine 0.999 against this
    // same kind of reference; expressed as relL2 on these widths that is about
    // 0.045, and the measured floor sits below it. 0.05 leaves the floor
    // headroom and still collapses on anything structural -- the rotated-weight
    // control below lands above 1.0, twenty times the bar.
    //
    // In the fp32 run the host reference stops rounding its activations to
    // bf16 while the device cannot, so the gap grows by exactly that rounding
    // and the bar moves with it, for the same reason N14's and N15's do.
    const double bar_rel_l2 = bf16 ? 0.05 : 0.07;
    const double kCollapse  = 0.40;   ///< a control must exceed this relL2

    // ---- config and weights ------------------------------------------------
    LM_Config config;
    config.model_path = model_dir;
    config.model_name = "Qwen3.8-27B-NPU2";
    config.exec_path  = ".";
    {
        std::ifstream f(model_dir + "/config.json");
        if (!f) { std::fprintf(stderr, "cannot read config.json\n"); return 1; }
        f >> config._json_config;
    }

    qwen3_8mtp_vision_desc desc;
    desc.build(config);
    if (!desc.present) {
        std::fprintf(stderr, "config.json has no vision_config\n");
        return 1;
    }

    cpu_ref::safetensors ref(ref_path);
    const std::vector<int32_t> thw = ref.i32("image_grid_thw");
    const qwen38::image_grid g{thw[0], thw[1], thw[2]};
    const int L_dump    = desc.patch_count(g.t, g.h, g.w);
    const int rows_dump = desc.token_count(g.t, g.h, g.w);

    // Constructing `weights` is what corrects D_ff from the config's 4304 to
    // the shipped 4608, so nothing may size a buffer off desc before this.
    qwen38::vision::weights w(desc);

    const int D  = (int)desc.D;
    const int F  = (int)desc.D_ff;
    const int PD = (int)desc.patch_dim();
    const int MI = (int)desc.merger_in();
    const int OH = (int)desc.out_hidden_size;

    std::printf("   D %d  D_ff %d  patch_dim %d  merger_in %d  out_hidden %d\n"
                "   captured image %dx%dx%d -> L %d patches, %d merged tokens\n"
                "   under test at L=1200 (a 639x480 image), %d merged\n\n",
                D, F, PD, MI, OH, g.t, g.h, g.w, L_dump, rows_dump, 1200 / 4);

    // =======================================================================
    std::printf("-- check 1: the two per-build RTP addresses --\n");
    // =======================================================================
    // A stale address does not fault. It writes the trip counts into some other
    // core buffer, the cores never see them, and the dispatch hangs until
    // ERT_CMD_STATE_TIMEOUT. Checked FIRST and against the files the bitstream
    // builds actually wrote, so a rebuild is a readable FAIL here rather than a
    // two-minute hang somewhere below.
    // (memory: rtp-addresses-are-per-build)
    const q38npu::vision_mm_rtp_addresses rtp;
    {
        auto from_build = [](const std::string& xclbin) {
            const std::string f =
                xclbin.substr(0, xclbin.rfind("/xclbins/")) + "/rtp_addr.txt";
            std::ifstream in(f);
            long v = -1;
            if (in) in >> v;
            return v;
        };
        const long ba = from_build(xclbin_a), bb = from_build(xclbin_b);
        const bool have = ba >= 0 && bb >= 0;
        report("rtp addresses match the builds",
               !have || ((uint32_t)ba == rtp.a && (uint32_t)bb == rtp.b),
               fmt("header A %u B %u, %s A %ld B %ld%s", rtp.a, rtp.b,
                   have ? "rtp_addr.txt" : "rtp_addr.txt missing --", ba, bb,
                   have ? "" : " (not checked; a mismatch hangs in "
                               "qds_device::wait)"));
    }

    // ---- the device ---------------------------------------------------------
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* mgr_a = mgr.register_xclbin(xclbin_a);
    npu_app_manager* mgr_b = mgr.register_xclbin(xclbin_b);

    q38npu::vision_mm_arena arena_a(mgr_a, q38npu::vision_mm_tiles::A, rtp.a);
    q38npu::vision_mm_arena arena_b(mgr_b, q38npu::vision_mm_tiles::B, rtp.b);

    auto make_dev = [&](const char* tag) {
        return std::make_unique<q38npu::npu_vision_mm_op>(&arena_a, &arena_b,
                                                          tag);
    };

    // ---- the seven real shapes ----------------------------------------------
    const qwen38::vision::block_weights& b0 = w.block(0);
    const std::vector<shape> shapes = {
        {"patch_embed", w.patch_w(),      w.patch_b(),      PD, D,     activation::none,     false},
        {"block.qkv",   b0.qkv_w,         b0.qkv_b,         D,  3 * D, activation::none,     false},
        {"block.proj",  b0.proj_w,        b0.proj_b,        D,  D,     activation::none,     false},
        {"block.fc1",   b0.fc1_w,         b0.fc1_b,         D,  F,     activation::gelu,     false},
        {"block.fc2",   b0.fc2_w,         b0.fc2_b,         F,  D,     activation::none,     false},
        {"merger.fc1",  w.merger_fc1_w(), w.merger_fc1_b(), MI, MI,    activation::gelu_erf, false},
        {"merger.fc2",  w.merger_fc2_w(), w.merger_fc2_b(), MI, OH,    activation::none,     true },
    };

    // =======================================================================
    std::printf("\n-- check 2: bind() picks a design, or says why it cannot --\n");
    // =======================================================================
    // The design is chosen from the SHAPE, never from the tag, so a renamed
    // tensor cannot pick the wrong bitstream. What must not happen is a shape
    // neither design serves being accepted: the kernel's trip counts are
    // integers it reads out of an RTP, so a remainder is not a short last round
    // -- it is a round the cores wait for and never get.
    {
        auto refuses = [&](int K, int N) {
            auto p = make_dev("probe");
            try { p->bind(b0.proj_w, b0.proj_b, K, N, activation::none); }
            catch (const std::exception&) { return true; }
            return false;
        };
        // 1024 is a multiple of neither A's tile_N (384) nor, as a K, of 384.
        const bool guards = refuses(1000, D) && refuses(D, 1000) &&
                            refuses(D, 512);

        bool placed = true;
        std::string where;
        for (const shape& s : shapes) {
            auto op = make_dev(s.tag);
            op->bind(s.w, s.b, s.K, s.N, s.act);
            const bool on_b = op->on_design_b();
            if (on_b != s.want_b) placed = false;
            where += fmt("%s%s:%c", where.empty() ? "" : " ", s.tag,
                         on_b ? 'B' : 'A');
        }
        report("geometry guards and design placement", guards && placed,
               fmt("K=1000, N=1000 and N=512 are refused; %s", where.c_str()));
    }

    // ---- inputs, one per distinct K -----------------------------------------
    // Real activations of the right width, tiled out to the operating point.
    // A Gaussian would be the wrong test: bf16's range floor and the bfp16
    // shared exponent both care about the actual magnitude spread of a row.
    const int M_BIG = 1200;
    std::map<int, std::vector<float>> xin;
    {
        const std::vector<float> pv = ref.f32("pixel_values");     // [L][1536]
        const std::vector<float> n1 = ref.f32("block_00.norm1");   // [L][1152]
        const std::vector<float> mn = ref.f32("merger_norm");      // [rows][4608]
        xin[PD] = tile_rows(pv, L_dump,    PD, M_BIG, 1536);
        xin[D]  = tile_rows(n1, L_dump,    D,  M_BIG, 1152);
        xin[MI] = tile_rows(mn, rows_dump, MI, M_BIG, 4608);
    }
    // F == MI == 4608 on this checkpoint, so fc2's input is covered too; if a
    // future checkpoint separates them this would silently feed the wrong
    // width, so it is asserted rather than assumed.
    if (F != MI) {
        std::fprintf(stderr, "D_ff %d != merger_in %d -- add an input for it\n",
                     F, MI);
        return 1;
    }

    // =======================================================================
    std::printf("\n-- checks 3-4: every shape, device vs host, at M=1200 --\n");
    // =======================================================================
    bool all_ok = true, all_controls_ok = true;
    std::string worst_line;
    double worst_seen = 0.0;

    for (const shape& s : shapes) {
        const int tileN = s.want_b
            ? (int)q38npu::vision_mm_tiles::B.tile_N()
            : (int)q38npu::vision_mm_tiles::A.tile_N();
        const std::vector<float>& x = xin.at(s.K);

        auto op = make_dev(s.tag);
        op->bind(s.w, s.b, s.K, s.N, s.act);
        qwen38::vision::cpu_mm_op host;
        host.bind(s.w, s.b, s.K, s.N, s.act);

        std::vector<float> a((size_t)M_BIG * s.N), h((size_t)M_BIG * s.N);
        op->run(x.data(), M_BIG, a.data());
        host.run(x.data(), M_BIG, h.data());

        const cpu_ref::err_stats e =
            cpu_ref::error_matrix(a.data(), h.data(), a.size());
        int wb = -1, wr = -1;
        const double wband = worst_band(a.data(), h.data(), M_BIG, s.N, tileN,
                                        &wb);
        const double wrow  = worst_row_cos(a.data(), h.data(), M_BIG, s.N, &wr);

        // qkv is q|k|v side by side. The aggregate is dominated by whichever
        // third has the largest norm, so a broken k rides behind a healthy q.
        // (memory: fused-matmul-cosine-hides-narrow-tensors)
        std::string thirds;
        if (std::string(s.tag) == "block.qkv") {
            const int t = s.N / 3;
            for (int i = 0; i < 3; i++)
                thirds += fmt("%s%c %.5f", i ? "  " : "  q|k|v ", "qkv"[i],
                              band_rel_l2(a.data(), h.data(), M_BIG, s.N,
                                          i * t, (i + 1) * t));
        }

        const bool ok = wband <= bar_rel_l2;
        all_ok = all_ok && ok;
        if (wband > worst_seen) {
            worst_seen = wband;
            worst_line = fmt("%s band %d", s.tag, wb);
        }
        report(fmt("%s  K=%d N=%d %s", s.tag, s.K, s.N,
                   qwen38::vision::act_name(s.act)).c_str(),
               ok,
               fmt("relL2 %.5f  worst band %.5f @%d of %d  cos %.6f  "
                   "worst row %.6f @%d%s",
                   e.rel_l2, wband, wb, s.N / tileN, e.cosine, wrow, wr,
                   thirds.c_str()));

        // -- the control: a rotated weight ---------------------------------
        // Rotating W's rows permutes the OUTPUT columns, so the true answer and
        // the rotated answer have identical norms, identical column statistics
        // and identical everything a summary statistic can see. A device that
        // transposed its array mapping would agree with the true reference AND
        // with itself; only a reference computed from different weights
        // separates them.
        //
        // TWO shifts, and both must collapse. `vision_mm`'s own harness rotates
        // by exactly N/2, and on this checkpoint that is degenerate for
        // `merger.fc1`: its weight has a genuine period of N/2, so the rotated
        // reference lands at relL2 0.098 while shifts of 1, 7 and N/3 all land
        // at 1.33. A rotation control must not be a symmetry of the tensor it
        // rotates, and the cheapest way not to depend on knowing which offsets
        // are symmetries is to require two coprime-ish ones.
        {
            double weakest = 1e9;
            int weakest_shift = 0;
            for (const int shift : {s.N / 2 + 1, s.N / 3 + 1}) {
                std::vector<uint16_t> wrot((size_t)s.N * s.K);
                for (int j = 0; j < s.N; j++)
                    std::memcpy(wrot.data() + (size_t)j * s.K,
                                s.w + (size_t)((j + shift) % s.N) * s.K,
                                (size_t)s.K * sizeof(uint16_t));
                std::vector<uint16_t> brot;
                if (s.b) {
                    brot.resize((size_t)s.N);
                    for (int j = 0; j < s.N; j++)
                        brot[(size_t)j] = s.b[(j + shift) % s.N];
                }
                qwen38::vision::cpu_mm_op rot;
                rot.bind(wrot.data(), s.b ? brot.data() : nullptr, s.K, s.N,
                         s.act);
                // 256 rows is enough for a control; the full 1200 would cost
                // more scalar host time than the rest of this driver together.
                std::vector<float> r((size_t)256 * s.N);
                rot.run(x.data(), 256, r.data());
                const cpu_ref::err_stats c =
                    cpu_ref::error_matrix(a.data(), r.data(), r.size());
                if (c.rel_l2 < weakest) {
                    weakest = c.rel_l2;
                    weakest_shift = shift;
                }
            }
            const bool cok = weakest > kCollapse;
            all_controls_ok = all_controls_ok && cok;
            if (!cok)
                report(fmt("CONTROL %s rotated weight", s.tag).c_str(), false,
                       fmt("weakest of the two shifts is %d at relL2 %.4f "
                           "(must be > %.2f)", weakest_shift, weakest,
                           kCollapse));
        }
    }
    report("CONTROL a rotated weight collapses, all 7", all_controls_ok,
           fmt("every shape scores relL2 > %.2f against a reference built from "
               "W's rows rotated by N/2", kCollapse));
    if (!all_ok)
        std::printf("     worst band overall: %s at relL2 %.5f\n",
                    worst_line.c_str(), worst_seen);

    // =======================================================================
    std::printf("\n-- check 5: the weight pre-round made the floor a no-op --\n");
    // =======================================================================
    // The kernel is built with -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16, so
    // the mmul floors each 8-consecutive-k block of W onto a shared exponent
    // and TRUNCATES toward -inf. A floor is a biased error, so it accumulates
    // linearly down a 1536-long dot product instead of cancelling as sqrt(K);
    // `bind()` therefore pre-rounds W onto that same grid to nearest, which
    // makes the hardware's own truncation a no-op.
    // (memory: aie-mmul-bf16-truncates-to-bfp16)
    //
    // `preround_enabled()` caches in a static, so a single process cannot run
    // both arms. The single-process test is better anyway: score the device
    // against TWO host references, the exact bf16 weights and the same weights
    // floored here, and see which one it agrees with. That is a direct read of
    // what the array multiplied, not an inference from an error magnitude.
    //
    //   pre-round on   -> the device should match the EXACT reference
    //   pre-round off  -> the device should match the FLOORED reference
    //
    // Either way one of the two must win by a wide margin. A device agreeing
    // with neither would mean the grid model itself is wrong, which is the one
    // outcome no error bar would have caught.
    {
        const int M = 256, K = PD, N = D;
        const std::vector<float>& x = xin.at(K);

        auto op = make_dev("patch_embed.preround");
        op->bind(w.patch_w(), w.patch_b(), K, N, activation::none);
        std::vector<float> a((size_t)M * N);
        op->run(x.data(), M, a.data());

        qwen38::vision::cpu_mm_op exact;
        exact.bind(w.patch_w(), w.patch_b(), K, N, activation::none);
        std::vector<float> h_exact((size_t)M * N);
        exact.run(x.data(), M, h_exact.data());

        // The hardware's grid: step = 2^(floor(log2 max) + 1 - 7) over each
        // run of 8 consecutive k. Rows are contiguous K, so no group straddles
        // a row boundary.
        std::vector<uint16_t> wf(w.patch_w(), w.patch_w() + (size_t)N * K);
        for (size_t b = 0; b + 8 <= wf.size(); b += 8) {
            float v[8], mx = 0.0f;
            for (int i = 0; i < 8; i++) {
                v[i] = cpu_ref::bf16_to_f32(wf[b + i]);
                mx = std::max(mx, std::fabs(v[i]));
            }
            if (!(mx > 0.0f)) continue;
            const float st =
                std::ldexp(1.0f, (int)std::floor(std::log2(mx)) - 6);
            for (int i = 0; i < 8; i++)
                wf[b + i] = cpu_ref::f32_to_bf16(std::floor(v[i] / st) * st);
        }
        qwen38::vision::cpu_mm_op floored;
        floored.bind(wf.data(), w.patch_b(), K, N, activation::none);
        std::vector<float> h_floor((size_t)M * N);
        floored.run(x.data(), M, h_floor.data());

        const double e_exact =
            cpu_ref::error_matrix(a.data(), h_exact.data(), a.size()).rel_l2;
        const double e_floor =
            cpu_ref::error_matrix(a.data(), h_floor.data(), a.size()).rel_l2;
        const bool on = q38npu::preround_enabled();
        // 4x is a wide margin and the measured separation is 18x either way.
        const bool ok = on ? (e_exact * 4 < e_floor) : (e_floor * 4 < e_exact);
        report(on ? "pre-round on: the device matches the EXACT weights"
                  : "pre-round off: the device matches the FLOORED weights",
               ok,
               fmt("patch_embed (kappa ~300, the worst-conditioned shape): vs "
                   "exact relL2 %.5f, vs floored %.5f -- %.0fx apart",
                   e_exact, e_floor,
                   e_exact > 0 && e_floor > 0
                       ? std::max(e_exact, e_floor) / std::min(e_exact, e_floor)
                       : 0.0));
        if (!on)
            std::printf("     FLM_Q38_VISION_MM_PREROUND=0 is set; this is the "
                        "off-arm, and the shipping default is on\n");
    }

    // =======================================================================
    std::printf("\n-- check 6: the fused gelu is the RTP's, not the host's --\n");
    // =======================================================================
    // Three arms on the same input and the same weight:
    //
    //   fused   bind(gelu)  -> rtp_act = 2, the kernel's getGeluBf16_nonLUT
    //   raw     bind(none)  -> rtp_act = 0
    //   host    gelu_tanh(raw) applied here
    //
    // fused == host is the score. fused != raw is what says the RTP actually
    // reached the core -- without it, a kernel that ignored rtp_act entirely
    // would pass the score whenever the activation happened to be near
    // identity, and gelu is near identity over most of its range.
    // (memory: score-an-elementwise-op-with-a-control)
    //
    // The two arms are also a check that `rtp_act` is not leaking ACROSS ops:
    // the kernel re-reads it every round, so it is per-round, not per-dispatch,
    // and this driver dispatches a `none` op between two `gelu` ones.
    // (memory: rtp-act-is-per-round-not-per-dispatch)
    {
        const std::vector<float>& x = xin.at(D);
        const int M = 512, N = F;

        auto fused = make_dev("fc1.gelu");
        fused->bind(b0.fc1_w, b0.fc1_b, D, N, activation::gelu);
        auto raw = make_dev("fc1.none");
        raw->bind(b0.fc1_w, b0.fc1_b, D, N, activation::none);

        std::vector<float> a((size_t)M * N), r((size_t)M * N);
        fused->run(x.data(), M, a.data());
        raw->run(x.data(), M, r.data());

        std::vector<float> host_applied(r);
        for (float& v : host_applied)
            v = qwen38::vision::gelu_tanh(v);

        const cpu_ref::err_stats e_same =
            cpu_ref::error_matrix(a.data(), host_applied.data(), a.size());
        const cpu_ref::err_stats e_diff =
            cpu_ref::error_matrix(a.data(), r.data(), a.size());

        // The bar is `bar_rel_l2`, not something tighter, and that is not
        // slack. The two arms do not share an intermediate: the fused arm
        // applies gelu to the fp32 ACCUMULATOR inside the core, while this one
        // applies it to the accumulator after it has been rounded to bf16 on
        // the way out. gelu is steep where its output is small, so that one
        // rounding is amplified to about the same size as the matmul's own
        // floor -- measured 0.0246 against block.fc1's own 0.0254, which is
        // the tell that it IS that floor and not a different function.
        // The separation that carries the claim is the 40x against the
        // unapplied arm, not the size of the agreement.
        report("fused gelu == host gelu_tanh of the raw output",
               e_same.rel_l2 <= bar_rel_l2 && e_diff.rel_l2 > 0.10,
               fmt("fused vs host-applied relL2 %.6f (bar %.3f); fused vs "
                   "UNapplied %.5f (must be > 0.100, or the RTP never landed)",
                   e_same.rel_l2, bar_rel_l2, e_diff.rel_l2));
    }

    // =======================================================================
    std::printf("\n-- check 7: the merger's erf epilogue stayed on the host --\n");
    // =======================================================================
    // `vision_mm` has no erf gelu, so `merger.fc1` binds rtp_act = none and
    // applies `vision::gelu_erf` here. What is gated:
    //
    //   the erf arm equals a host erf of the device's own raw output  -- the
    //       wiring. This is exact up to the bf16 rounding both paths do.
    //   the erf arm DIFFERS from the fused tanh arm                   -- that
    //       the substitution forbidden by the seam did not quietly happen.
    //
    // What is NOT gated is the direction, erf beating tanh against the
    // reference. The device output is bf16 whichever form is used, and N14
    // already measured that below that floor the two are indistinguishable --
    // tanh was ahead by 6e-7. A check that cannot separate its arms is not a
    // check; the numbers are printed instead.
    // (memory: gate-a-control-against-the-error-floor)
    {
        const std::vector<float>& x = xin.at(MI);
        const int M = 256;

        auto erf_op = make_dev("merger.fc1.erf");
        erf_op->bind(w.merger_fc1_w(), w.merger_fc1_b(), MI, MI,
                     activation::gelu_erf);
        auto raw = make_dev("merger.fc1.none");
        raw->bind(w.merger_fc1_w(), w.merger_fc1_b(), MI, MI, activation::none);
        auto tanh_op = make_dev("merger.fc1.tanh");
        tanh_op->bind(w.merger_fc1_w(), w.merger_fc1_b(), MI, MI,
                      activation::gelu);

        std::vector<float> a((size_t)M * MI), r((size_t)M * MI),
                           t((size_t)M * MI);
        erf_op->run(x.data(), M, a.data());
        raw->run(x.data(), M, r.data());
        tanh_op->run(x.data(), M, t.data());

        std::vector<float> host_erf(r);
        for (float& v : host_erf) {
            const float u = qwen38::vision::gelu_erf(v);
            v = bf16 ? cpu_ref::round_bf16(u) : u;
        }

        const cpu_ref::err_stats e_wire =
            cpu_ref::error_matrix(a.data(), host_erf.data(), a.size());
        const cpu_ref::err_stats e_forms =
            cpu_ref::error_matrix(a.data(), t.data(), a.size());
        const bool bound_none = erf_op->host_epilogue();

        report("merger.fc1 runs erf on the host, not the kernel's tanh",
               bound_none && e_wire.rel_l2 <= 1e-6 && e_forms.rel_l2 > 1e-4,
               fmt("rtp_act=none: %s; vs a host erf of the raw output relL2 "
                   "%.2e; vs the FUSED TANH arm %.2e (must be > 1e-4)",
                   bound_none ? "yes" : "NO", e_wire.rel_l2, e_forms.rel_l2));

        // Reported, not gated. See above.
        {
            const std::vector<float> mref = ref.f32("merger_fc1");
            const int mr = rows_dump;
            const std::vector<float>& mn = xin.at(MI);
            std::vector<float> ae((size_t)mr * MI), at((size_t)mr * MI);
            // The dump's own rows, untiled -- tile_rows put jitter on them, so
            // rebuild the first `mr` rows from the reference tap directly.
            const std::vector<float> mn0 = ref.f32("merger_norm");
            (void)mn;
            erf_op->run(mn0.data(), mr, ae.data());
            tanh_op->run(mn0.data(), mr, at.data());
            const cpu_ref::err_stats se =
                cpu_ref::error_matrix(ae.data(), mref.data(), ae.size());
            const cpu_ref::err_stats st =
                cpu_ref::error_matrix(at.data(), mref.data(), at.size());
            std::printf("     vs the HF merger_fc1 tap: erf relL2 %.6f cos "
                        "%.7f | tanh %.6f cos %.7f %s\n",
                        se.rel_l2, se.cosine, st.rel_l2, st.cosine,
                        se.rel_l2 < st.rel_l2 ? "(erf ahead)" : "(tanh ahead "
                        "-- below the device's bf16 floor, as expected)");
        }
    }

    // =======================================================================
    std::printf("\n-- check 8: a short call after a long one, and redispatch --\n");
    // =======================================================================
    // Two different failures, both of which produce a correct FIRST answer.
    //
    // The pad rows: design A's tile_M is 256, so M=300 runs 512 rows and 212 of
    // them are whatever the previous, longer call left in the shared x BO.
    // Nothing masks them -- there is no mask in this design at all -- so they
    // are multiplied and accumulated before their output rows are dropped.
    // (memory: kv-rows-above-cursor-are-not-inert)
    //
    // The redispatch: a prologue BD runs once per xclbin load, so a design that
    // put its bias fill outside the round loop is right on dispatch 1 and wrong
    // on dispatch 2 -- and ERT still reports COMPLETED.
    // (memory: a-prologue-bd-breaks-the-second-dispatch,
    //          unchecked-ert-state-looks-like-a-pass)
    //
    // Bit identity is the bar for both. Stale memory and a missing bias are
    // questions about WHAT RAN, not about how much error is acceptable.
    // (memory: gate-an-overlapped-replay-on-bit-identity)
    {
        const std::vector<float>& x = xin.at(D);
        const int M = 300;   // 300 -> 512 padded on design A, and 4 | 1200

        auto op = make_dev("proj.reuse");
        op->bind(b0.proj_w, b0.proj_b, D, D, activation::none);
        std::vector<float> big((size_t)M_BIG * D), a((size_t)M * D),
                           a2((size_t)M * D), fresh((size_t)M * D);
        op->run(x.data(), M_BIG, big.data());   // the long call first
        op->run(x.data(), M,     a.data());
        op->run(x.data(), M,     a2.data());    // the same call again

        auto clean = make_dev("proj.fresh");
        clean->bind(b0.proj_w, b0.proj_b, D, D, activation::none);
        clean->run(x.data(), M, fresh.data());

        const size_t bytes = a.size() * sizeof(float);
        const bool no_residue = std::memcmp(a.data(), fresh.data(), bytes) == 0;
        const bool stable     = std::memcmp(a.data(), a2.data(), bytes) == 0;

        qwen38::vision::cpu_mm_op host;
        host.bind(b0.proj_w, b0.proj_b, D, D, activation::none);
        std::vector<float> h((size_t)M * D);
        host.run(x.data(), M, h.data());
        int wb = -1;
        const double wband = worst_band(a.data(), h.data(), M, D,
                                        (int)q38npu::vision_mm_tiles::A.tile_N(),
                                        &wb);

        report("M=300 after M=1200, twice, leaves no residue",
               no_residue && stable && wband <= bar_rel_l2,
               fmt("bit-identical to a fresh op: %s; second dispatch identical "
                   "to the first: %s; worst band vs host %.5f @%d (bar %.3f)",
                   no_residue ? "yes" : "NO -- the pad rows leaked",
                   stable ? "yes" : "NO -- dispatch 2 differs",
                   wband, wb, bar_rel_l2));
    }

    // =======================================================================
    std::printf("\n-- check 9: the whole tower on the device factory --\n");
    // =======================================================================
    // N14's stage-by-stage checks with the device backend under them, which
    // N14 itself cannot run because it links no XRT. This is the only check
    // here that scores against the captured HF reference rather than against
    // the host op, and it is what says the 111 ops are wired into the tower in
    // the right ORDER and the right LAYOUT -- a transposed output that matched
    // the host op would still be caught by the 27 blocks that consume it.
    //
    // It allocates 959 MB of weight BOs, which is the real shipping cost of
    // `bind()` and the reason the seam is weight-resident at all.
    if (!std::filesystem::exists(xclbin_attn)) {
        std::printf("     skipped: no %s (the factory needs the attention "
                    "xclbin too)\n", xclbin_attn.c_str());
    } else {
        npu_app_manager* attn_mgr = mgr.register_xclbin(xclbin_attn);
        auto factory = std::make_unique<q38npu::vision_op_factory>(attn_mgr);
        factory->enable_mm(mgr_a, mgr_b, rtp);
        const std::string backend_name = factory->name();

        auto t_load = std::chrono::steady_clock::now();
        qwen38::image_cpu tower(desc, std::move(factory));
        const double load_ms = ms_since(t_load);

        std::map<std::string, std::vector<float>> got;
        std::map<std::string, std::pair<int, int>> dim;
        tower.set_tap([&](const char* name, const float* p, int r, int c) {
            got[name].assign(p, p + (size_t)r * c);
            dim[name] = {r, c};
        });

        const std::vector<float> pixel_values = ref.f32("pixel_values");
        std::vector<float> out((size_t)rows_dump * OH, 0.0f);
        auto t_enc = std::chrono::steady_clock::now();
        tower.encode(pixel_values.data(), g, out.data());
        const double enc_ms = ms_since(t_enc);

        // Gated on relL2, not on N14's cosine bars: those were set for a host
        // tower whose only error was bf16 activation rounding. The device adds
        // the bfp16 mmul floor at all 111 op boundaries and it compounds down
        // 27 blocks, so the right bar is the one this port's own error budget
        // implies, and the shape of the decay is what says it is a floor and
        // not a bug.
        // (memory: q38-prefill-cosine-decays-but-argmax-exact)
        auto score = [&](const char* tag, const std::string& key, double bar) {
            auto it = got.find(key);
            if (it == got.end() || !ref.has(key)) {
                report(tag, false, "stage '" + key + "' was never tapped");
                return;
            }
            const std::vector<float> r = ref.f32(key);
            if (r.size() != it->second.size()) {
                report(tag, false, fmt("size %zu vs reference %zu",
                                       it->second.size(), r.size()));
                return;
            }
            const cpu_ref::err_stats e =
                cpu_ref::error_matrix(it->second.data(), r.data(), r.size());
            int wr = -1;
            const double wrow = worst_row_cos(it->second.data(), r.data(),
                                              dim[key].first, dim[key].second,
                                              &wr);
            report(tag, e.rel_l2 <= bar,
                   fmt("relL2 %.5f  cos %.7f  worst row %.6f @%d  (bar %.3f)",
                       e.rel_l2, e.cosine, wrow, wr, bar));
        };

        std::printf("     backend \"%s\", load %.0f ms, encode %.0f ms "
                    "(L=%d)\n", backend_name.c_str(), load_ms, enc_ms, L_dump);

        // patch_embed is ONE matmul deep, so its bar is the single-op floor;
        // block_26.out is 27 blocks of accumulation and merger output is that
        // plus two more.
        score("patch_embed (one matmul deep)", "patch_embed", bar_rel_l2);
        score("block_00.attn_proj",            "block_00.attn_proj", 2 * bar_rel_l2);
        score("block_26.out (27 blocks deep)", "block_26.out", 4 * bar_rel_l2);
        score("merger_fc1",                    "merger_fc1", 4 * bar_rel_l2);
        {
            const std::vector<float> r = ref.f32("image_embeds");
            const cpu_ref::err_stats e =
                cpu_ref::error_matrix(out.data(), r.data(), out.size());
            int wr = -1;
            const double wrow = worst_row_cos(out.data(), r.data(), rows_dump,
                                              OH, &wr);
            report("image_embeds (tower output)", e.rel_l2 <= 4 * bar_rel_l2,
                   fmt("relL2 %.5f  cos %.7f  worst row %.6f @%d  (bar %.3f)",
                       e.rel_l2, e.cosine, wrow, wr, 4 * bar_rel_l2));
        }
    }

    // =======================================================================
    std::printf("\n-- check 10: what the move costs, at both sizes --\n");
    // =======================================================================
    // Not gated, and reported at TWO sizes because a fixed-cost dispatch
    // against a linear host path crosses somewhere, and one number would have
    // the wrong sign on the other side of it.
    // (memory: ab-at-one-size-can-have-the-wrong-sign)
    //
    // The per-image total is the number that matters and it is not the sum of
    // the per-op numbers: the tower runs patch_embed once, four ops per block
    // times 27 blocks, and the two merger layers at L/4 rows.
    //
    // Build this driver at -O2 and the host arm slows 2.6-3x. See the header.
    {
        std::printf("     %-12s %5s %5s  %9s  %9s  %6s\n",
                    "tag", "K", "N", "device ms", "host ms", "x");
        double dev_img = 0, host_img = 0;
        for (const shape& s : shapes) {
            const bool merger = std::string(s.tag).rfind("merger", 0) == 0;
            const int  M      = merger ? M_BIG / 4 : M_BIG;
            const int  count  = std::string(s.tag).rfind("block", 0) == 0
                                    ? (int)desc.depth : 1;
            const std::vector<float>& x = xin.at(s.K);

            auto op = make_dev(s.tag);
            op->bind(s.w, s.b, s.K, s.N, s.act);
            qwen38::vision::cpu_mm_op host;
            host.bind(s.w, s.b, s.K, s.N, s.act);

            const double d = bench_min(*op, host, x, M, true);
            const double h = bench_min(*op, host, x, M, false);
            dev_img  += d * count;
            host_img += h * count;
            std::printf("     %-12s %5d %5d  %9.2f  %9.2f  %5.2fx  x%d\n",
                        s.tag, s.K, s.N, d, h, h / d, count);
        }
        std::printf("     per image at L=%d: device %.0f ms, host %.0f ms, "
                    "%.2fx\n", M_BIG, dev_img, host_img, host_img / dev_img);

        // And the dump's size, where the host has less to do and the device's
        // fixed dispatch cost is a larger share.
        double dev_small = 0, host_small = 0;
        for (const shape& s : shapes) {
            const bool merger = std::string(s.tag).rfind("merger", 0) == 0;
            const int  M      = merger ? L_dump / 4 : L_dump;
            const int  count  = std::string(s.tag).rfind("block", 0) == 0
                                    ? (int)desc.depth : 1;
            const std::vector<float>& x = xin.at(s.K);
            auto op = make_dev(s.tag);
            op->bind(s.w, s.b, s.K, s.N, s.act);
            qwen38::vision::cpu_mm_op host;
            host.bind(s.w, s.b, s.K, s.N, s.act);
            dev_small  += bench_min(*op, host, x, M, true)  * count;
            host_small += bench_min(*op, host, x, M, false) * count;
        }
        std::printf("     per image at L=%d: device %.0f ms, host %.0f ms, "
                    "%.2fx\n", L_dump, dev_small, host_small,
                    host_small / dev_small);

        // ------------------------------------------------------------------
        // The same A/B IN SITU, which is the number that is actually true.
        //
        // Everything above uses `bench_min`: warm BOs, min of 10. The tower
        // dispatches each of the 111 ops ONCE per image, with a cold sequence
        // and a cold x upload, so the per-op table OVERSTATES the win by about
        // 1.8x -- it predicted 1.17 s and the real encode A/B is 0.66 s.
        // Measured through the real `image_cpu`, so every host-side norm, rope,
        // gelu epilogue and gather is in both arms.
        //
        // This is also why `make test`'s prefill cannot confirm the move on its
        // own: one prefill sample per arm has a run-to-run spread of ~0.6 s,
        // which is the size of the whole effect.
        // (memory: ab-off-arm-is-not-the-host, host-vs-device-ab-needs-shipping-flags)
        if (std::filesystem::exists(xclbin_attn)) {
            npu_app_manager* attn_mgr = mgr.register_xclbin(xclbin_attn);
            qwen38::image_grid gg; gg.t = 1; gg.h = 30; gg.w = 40;   // panda.png
            const int Lfull = gg.t * gg.h * gg.w;
            const int rows   = Lfull / (desc.spatial_merge_size *
                                        desc.spatial_merge_size);

            // real patch rows tiled out to the operating point, same jitter as
            // checks 3-4 (memory: vision-tower-operating-point-is-l1200)
            const std::vector<float> pv = ref.f32("pixel_values");
            std::vector<float> px = tile_rows(pv, L_dump, PD, Lfull, 11);

            std::printf("     in situ, the real tower at L=%d:\n", Lfull);
            double first = 0;
            for (int arm = 0; arm < 3; arm++) {
                const bool npu_attn = arm < 2, npu_mm = arm == 0;
                std::unique_ptr<qwen38::vision::op_factory> f;
                if (npu_attn) {
                    auto nf = std::make_unique<q38npu::vision_op_factory>(attn_mgr);
                    if (npu_mm) nf->enable_mm(mgr_a, mgr_b, rtp);
                    f = std::move(nf);
                } else {
                    f = std::make_unique<qwen38::vision::cpu_op_factory>();
                }
                const std::string nm = f->name();
                qwen38::image_cpu tw(desc, std::move(f));
                std::vector<float> o((size_t)rows * OH, 0.0f);
                tw.encode(px.data(), gg, o.data());              // warm
                double best = 1e30;
                for (int i = 0; i < 3; i++) {
                    auto t0 = std::chrono::steady_clock::now();
                    tw.encode(px.data(), gg, o.data());
                    best = std::min(best, ms_since(t0));
                }
                if (!first) first = best;
                std::printf("       %-9s encode %8.0f ms   %+8.0f ms\n",
                            nm.c_str(), best, best - first);
            }
        }
    }

    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
