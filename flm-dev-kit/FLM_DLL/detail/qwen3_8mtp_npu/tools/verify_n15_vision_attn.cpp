// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify_n15_vision_attn.cpp
/// \brief N15 -- the vision tower's attention on the NPU, against the host op
///        it replaces and against the captured HF reference.
///
/// Build (one line; a backslash at the end of a // comment splices the next
/// line into it, so the continuations are dropped here):
///
///   source ~/setup.sh
///   g++-13 -O3 -march=native -fopenmp -std=c++20 -ffast-math
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n15_vision_attn.cpp image/npu/vision_attn_sequence.cpp
///       image/vision_ops.cpp image/vision_rope.cpp
///       ../q4nx/q4_npu_eXpress.cpp ../q4nx/safe_tensors.cpp
///       -L../../../../src/lib/xrt -L../../build/lib -L/opt/xilinx/xrt/lib
///       -lxrt_coreutil -laiebu -luuid -lpthread
///       -o /tmp/verify_n15_vision_attn
///
/// **-O3, not -O2, and that is not cosmetic.** The Makefile builds the engine
/// at -O3, and the host arm of check 9 is `image/vision_ops.cpp` compiled into
/// THIS binary. At -O2 the same source measures 4.48 -> 11.58 ms at L=256 and
/// 41.7 -> 124.9 at L=1200: a 2.6-3x handicap that makes the device look three
/// times better than it is. The off-arm has to be built the way it ships.
/// (memory: host-vs-device-ab-needs-shipping-flags, ab-off-arm-is-not-the-host)
///
/// Run:
///   /tmp/verify_n15_vision_attn [model_dir] [vision_attn.xclbin]
///   FLM_Q38_VISION_FP32=1 /tmp/verify_n15_vision_attn   # the precision control
///
/// ---------------------------------------------------------------------------
/// What is under test, and what is not
/// ---------------------------------------------------------------------------
/// `vision_attn.xclbin` computes ONLY
///
///     o = softmax(q K^T / sqrt(72)) V
///
/// over one image, fully bidirectional. No projections, no RoPE, no mask but
/// the padding tail. Everything around it -- the fused qkv matmul, the 2-D
/// rotary, the output projection -- is the host's, is N14's business, and is
/// driven here from the SAME code N14 drives it from, so that a difference
/// between the two suites is the device and nothing else.
///
/// The reference is `vision::cpu_attn_op` on the same inputs, not torch.
/// `cpu_ops_emulate_bf16()` is on by default, so the host op already rounds its
/// activations the way the device does and the bf16 error is common-mode; what
/// is left to measure is the device's arithmetic and its addressing.
/// (memory: score-dequant-mm-against-its-own-quantized-bytes)
///
/// ---------------------------------------------------------------------------
/// Why nothing here is gated on cosine
/// ---------------------------------------------------------------------------
/// The vision softmax is diffuse -- 205 effective keys out of 256 at block 0 --
/// so the output sits near a weighted average of 256 smooth rows and points in
/// nearly the same direction however the scores are tilted. N14 measured what
/// that costs: dropping the rotary entirely still scores cos 0.9975 while relL2
/// moves 14x. A cosine bar here would pass a broken kernel.
///
/// So every magnitude claim is gated on **relative L2**, the per-head relL2 is
/// broken out rather than aggregated (a fused output hides its worst head
/// exactly the way a fused matmul hides its narrow tensors), and the per-row
/// cosine is reported for ADDRESSING only.
/// (memory: per-row-cosine-reports-the-norm, fused-matmul-cosine-hides-narrow-tensors)
///
/// ---------------------------------------------------------------------------
/// The floor this is gated against
/// ---------------------------------------------------------------------------
/// ~0.03 relL2, and it is the design's, not a defect. The kernel is built with
/// -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16, so both mmul operands are
/// floored onto a shared-exponent-per-8 grid -- 7 mantissa bits, not bf16's 8
/// -- inside BOTH the QK and the PV products. `vision_attn`'s own host harness
/// reports the same number (0.0329 at L=256, 0.0354 at L=1200) against its own
/// fp32 reference, which is what says the gap is the mmul and not this port.
/// (memory: aie-mmul-bf16-truncates-to-bfp16)
///
/// That floor is large enough to swallow a subtle bug, which is why four of the
/// nine checks below are controls rather than scores:
///
///   rotated head_perm vs the straight reference   must COLLAPSE. A consistent
///       permutation agrees with itself, so only a cross-head reference sees a
///       head <-> CU mapping error.
///       (memory: cross-head-control-catches-gqa-mapping)
///   rotated head_perm vs the ROTATED reference    must AGREE. Without this,
///       a device that silently ignored head_perm would pass the collapse test
///       for the wrong reason, and N14's control would be dead on the device.
///   L = 1200, three rounds                        the real image. 256 pads to
///       512, which is ONE round, so the round loop and the BD ping-pong are
///       entirely untested by the captured dump.
///   a short run after a long one                  the padding rows are not
///       inert (memory: kv-rows-above-cursor-are-not-inert): the mask covers
///       the KEY tail, so a previous image's k/v left above the cursor would be
///       multiplied before being masked. This is the check that the per-call
///       memset is load-bearing.
#include "image/npu/vision_attn_op.hpp"
#include "image/vision_model_def.hpp"
#include "image/vision_ops.hpp"
#include "image/vision_rope.hpp"
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
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using qwen38::vision::activation;

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-44s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
    if (ok) n_pass++; else n_fail++;
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap; va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return std::string(buf);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

/// relL2 of one head's column band, over all rows.
///
/// The aggregate is a norm-weighted mean over 16 heads, so a single head 6x
/// worse than the rest moves it by well under the floor. Breaking the band out
/// is the only way that shows up, and a head <-> CU mapping error is exactly
/// the bug that lives in ONE band.
/// (memory: fused-matmul-cosine-hides-narrow-tensors)
double head_rel_l2(const float* a, const float* b,
                   int rows, int heads, int dh, int head) {
    double num = 0, den = 0;
    const size_t D = (size_t)heads * dh;
    for (int r = 0; r < rows; r++) {
        const float* pa = a + (size_t)r * D + (size_t)head * dh;
        const float* pb = b + (size_t)r * D + (size_t)head * dh;
        for (int c = 0; c < dh; c++) {
            const double d = (double)pa[c] - pb[c];
            num += d * d;
            den += (double)pb[c] * pb[c];
        }
    }
    return den > 0 ? std::sqrt(num / den) : 0.0;
}

/// Worst head, for gating, and which one it was, for diagnosis.
double worst_head_rel_l2(const float* a, const float* b,
                         int rows, int heads, int dh, int* which) {
    double worst = 0.0;
    for (int h = 0; h < heads; h++) {
        const double e = head_rel_l2(a, b, rows, heads, dh, h);
        if (e > worst) { worst = e; if (which) *which = h; }
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

/// Block 0's real q, k and v, rotary already applied -- N14's check 12, and
/// deliberately the same code so the two suites differ only in the backend.
struct qkv_rows {
    std::vector<float> q, k, v;
    int L = 0;
};

/// Stretch L real rows to `want` rows, keeping the softmax diffuse.
///
/// The captured image is 16x16 patches, so the dump alone cannot reach a second
/// round. Repeating the rows with a small jitter is not a substitute for a real
/// 30x40 capture, but it is the right KIND of input: the topic structure that
/// makes keys compete is preserved, so the effective key count stays in the
/// hundreds instead of collapsing onto one key the way independent Gaussians do
/// (memory: random-kv-cache-cannot-test-attention). What it tests is the round
/// loop and the BD ping-pong, which are pure addressing and do not care that
/// row 300 is row 44 with noise on it.
qkv_rows tile_rows(const qkv_rows& in, int want, int D, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> jitter(0.0f, 0.05f);
    qkv_rows out;
    out.L = want;
    out.q.resize((size_t)want * D);
    out.k.resize((size_t)want * D);
    out.v.resize((size_t)want * D);
    for (int r = 0; r < want; r++) {
        const int src = r % in.L;
        for (int c = 0; c < D; c++) {
            const size_t d = (size_t)r * D + c, s = (size_t)src * D + c;
            out.q[d] = in.q[s] + jitter(rng);
            out.k[d] = in.k[s] + jitter(rng);
            out.v[d] = in.v[s] + jitter(rng);
        }
    }
    return out;
}

}  // namespace

// ===========================================================================
int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string xclbin =
        argc > 2 ? argv[2]
                 : "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/"
                   "vision_attn/build/xclbins/attn.xclbin";
    const std::string ref_path = model_dir + "/vision_ref.safetensors";

    const bool bf16 = qwen38::vision::cpu_ops_emulate_bf16();

    std::printf("== N15: the vision tower's attention on the NPU ==\n");
    std::printf("   model  %s\n   xclbin %s\n   ref    %s\n   mode   %s\n\n",
                model_dir.c_str(), xclbin.c_str(), ref_path.c_str(),
                bf16 ? "bf16 host reference (shipping default)"
                     : "fp32 host reference (FLM_Q38_VISION_FP32=1)");

    // The bar. Not a taste: `vision_attn`'s own harness reports 0.0329 relL2 at
    // L=256 and 0.0354 at L=1200 against its fp32 reference, and that is the
    // bfp16 mmul floor on BOTH products. 0.06 leaves the floor most of a factor
    // of two of headroom and still collapses on anything structural -- the
    // rotated-head control below lands at ~1.0, sixteen times the bar.
    //
    // In the fp32 run the host reference stops rounding to bf16 and the gap
    // grows by that rounding, so the bar moves with the mode for the same
    // reason N14's does.
    const double bar_rel_l2 = bf16 ? 0.06 : 0.08;
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

    const int H  = (int)desc.num_heads;
    const int DH = (int)desc.DH;
    const int D  = (int)desc.D;

    cpu_ref::safetensors ref(ref_path);
    const std::vector<int32_t> thw = ref.i32("image_grid_thw");
    const int L = desc.patch_count(thw[0], thw[1], thw[2]);

    std::printf("   %d heads x %d = %d wide; captured image %dx%dx%d -> L %d "
                "(padded %d, %d round%s)\n\n",
                H, DH, D, thw[0], thw[1], thw[2], L,
                (L + 511) / 512 * 512, (L + 511) / 512,
                (L + 511) / 512 == 1 ? "" : "s");

    // =======================================================================
    std::printf("-- check 1: the per-build RTP address --\n");
    // =======================================================================
    // A stale address does not fault. It writes L_pad and L_valid into some
    // other core buffer, the cores never see their trip count, and the dispatch
    // hangs until ERT_CMD_STATE_TIMEOUT. Checked FIRST and against the file the
    // bitstream build actually wrote, so a rebuild is a readable FAIL here
    // rather than a two-minute hang somewhere below.
    // (memory: rtp-addresses-are-per-build)
    const q38npu::vision_attn_rtp_addresses rtp;
    {
        const std::string addr_file =
            xclbin.substr(0, xclbin.rfind("/xclbins/")) + "/rtp_addr.txt";
        std::ifstream f(addr_file);
        long from_build = -1;
        if (f) f >> from_build;
        const bool have = from_build >= 0;
        report("rtp address matches the build",
               !have || (uint32_t)from_build == rtp.l,
               fmt("header %u, %s %ld%s", rtp.l,
                   have ? "rtp_addr.txt" : "rtp_addr.txt missing --",
                   from_build,
                   have ? "" : " (not checked; a mismatch hangs in "
                               "qds_device::wait)"));
    }

    // ---- the device op -----------------------------------------------------
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* attn_mgr = mgr.register_xclbin(xclbin);

    q38npu::npu_vision_attn_op npu_attn(attn_mgr->create_app(), rtp);
    qwen38::vision::cpu_attn_op cpu_attn;

    // =======================================================================
    std::printf("\n-- check 2: configure() refuses a geometry the bitstream "
                "cannot serve --\n");
    // =======================================================================
    // dh and scale are `constexpr`/`const` in kernel/common.h, so a checkpoint
    // that disagrees would be computed WRONG rather than rejected -- and the
    // diffuse softmax means a mistempered one still scores cos 0.99. These are
    // load-time throws for that reason.
    // (memory: score-an-elementwise-op-with-a-control)
    {
        auto refuses = [&](int heads, int dh, float scale) {
            q38npu::npu_vision_attn_op probe(attn_mgr->create_app(), rtp);
            try { probe.configure(512, heads, dh, scale); }
            catch (const std::exception&) { return true; }
            return false;
        };
        const float good = desc.attn_scale;
        const bool ok = refuses(H, 64, 1.0f / std::sqrt(64.0f)) &&
                        refuses(6, DH, good) &&
                        refuses(H, DH, good * 1.05f) &&
                        !refuses(H, DH, good);
        report("geometry guards", ok,
               fmt("dh=64, heads=6 and a 5%% scale are all refused; "
                   "heads=%d dh=%d scale=%.9f is accepted", H, DH, good));
    }

    npu_attn.configure(L, H, DH, desc.attn_scale);
    cpu_attn.configure(1536, H, DH, desc.attn_scale);

    // ---- block 0's real q/k/v, exactly as N14 builds them ------------------
    qwen38::vision::weights w(desc);
    qwen38::vision::cpu_mm_op qkv_op, proj_op;
    qkv_op.bind(w.block(0).qkv_w, w.block(0).qkv_b, D, 3 * D, activation::none);
    proj_op.bind(w.block(0).proj_w, w.block(0).proj_b, D, D, activation::none);

    qkv_rows in;
    in.L = L;
    {
        const std::vector<float> norm1 = ref.f32("block_00.norm1");
        std::vector<float> qkv((size_t)L * 3 * D);
        qkv_op.run(norm1.data(), L, qkv.data());
        in.q.resize((size_t)L * D);
        in.k.resize((size_t)L * D);
        in.v.resize((size_t)L * D);
        for (int t = 0; t < L; t++) {
            const float* r0 = qkv.data() + (size_t)t * 3 * D;
            std::memcpy(in.q.data() + (size_t)t * D, r0,         sizeof(float) * D);
            std::memcpy(in.k.data() + (size_t)t * D, r0 + D,     sizeof(float) * D);
            std::memcpy(in.v.data() + (size_t)t * D, r0 + 2 * D, sizeof(float) * D);
        }
        const qwen38::vision::rope_tables rope = qwen38::vision::build_rope(
            thw[0], thw[1], thw[2], (int)desc.spatial_merge_size, DH,
            desc.rope_theta);
        qwen38::vision::apply_rope(in.q.data(), L, H, rope);
        qwen38::vision::apply_rope(in.k.data(), L, H, rope);
    }

    // =======================================================================
    std::printf("\n-- checks 3-5: block 0's attention, device vs host --\n");
    // =======================================================================
    std::vector<float> o_npu((size_t)L * D), o_cpu((size_t)L * D);

    double base_rel = 0.0;
    {
        const double ekc = qwen38::vision::effective_key_count(
            in.q.data(), in.k.data(), L, H, DH, desc.attn_scale);

        cpu_attn.run(in.q.data(), in.k.data(), in.v.data(), L, o_cpu.data());
        npu_attn.run(in.q.data(), in.k.data(), in.v.data(), L, o_npu.data());

        const cpu_ref::err_stats e =
            cpu_ref::error_matrix(o_npu.data(), o_cpu.data(), o_npu.size());
        base_rel = e.rel_l2;
        int worst_h = -1, worst_r = -1;
        const double wh = worst_head_rel_l2(o_npu.data(), o_cpu.data(),
                                            L, H, DH, &worst_h);
        const double wr = worst_row_cos(o_npu.data(), o_cpu.data(), L, D,
                                        &worst_r);
        report(fmt("device == host at L=%d", L).c_str(),
               wh <= bar_rel_l2,
               fmt("relL2 %.5f  worst head %.5f @h%d  (bar %.3f)  "
                   "cos %.6f  worst row %.6f @%d  eff keys %.1f of %d",
                   e.rel_l2, wh, worst_h, bar_rel_l2, e.cosine, wr, worst_r,
                   ekc, L));
    }

    // The two halves of the head_perm control. The collapse alone is not
    // enough: a device that dropped head_perm on the floor would produce the
    // STRAIGHT answer, which also collapses against a rotated reference. Only
    // the agreement half says the rotation reached the k/v descriptors, and
    // without it N14's cross-head control cannot be run against hardware.
    {
        std::vector<int> perm((size_t)H);
        for (int h = 0; h < H; h++) perm[(size_t)h] = (h + 1) % H;

        std::vector<float> o_npu_rot((size_t)L * D), o_cpu_rot((size_t)L * D);
        npu_attn.run(in.q.data(), in.k.data(), in.v.data(), L,
                     o_npu_rot.data(), perm.data());
        cpu_attn.run(in.q.data(), in.k.data(), in.v.data(), L,
                     o_cpu_rot.data(), perm.data());

        const cpu_ref::err_stats e_vs_straight =
            cpu_ref::error_matrix(o_npu_rot.data(), o_cpu.data(),
                                  o_npu_rot.size());
        report("CONTROL rotated q/kv collapses",
               e_vs_straight.rel_l2 > kCollapse,
               fmt("relL2 %.4f vs the straight reference (must be > %.2f); "
                   "cos %.5f, which is why cosine is not the gate",
                   e_vs_straight.rel_l2, kCollapse, e_vs_straight.cosine));

        int worst_h = -1;
        const double wh = worst_head_rel_l2(o_npu_rot.data(), o_cpu_rot.data(),
                                            L, H, DH, &worst_h);
        report("device honours head_perm", wh <= bar_rel_l2,
               fmt("relL2 %.5f  worst head %.5f @h%d against the ROTATED "
                   "reference (bar %.3f)",
                   cpu_ref::error_matrix(o_npu_rot.data(), o_cpu_rot.data(),
                                         o_npu_rot.size()).rel_l2,
                   wh, worst_h, bar_rel_l2));
    }

    // =======================================================================
    std::printf("\n-- check 6: through the output projection, against HF --\n");
    // =======================================================================
    // The only check scored against the captured reference rather than against
    // the host op, and it is the one that says the device's output is in the
    // layout the rest of the tower reads. A transposed or head-shuffled o would
    // still match the host op if the host op were driven the same wrong way;
    // it cannot survive a projection whose weight came from the checkpoint.
    // (memory: mmul-c-operand-is-dim-block-major)
    //
    // Scored against N14's own number for the same tap, so this charges the
    // device for its own arithmetic plus the qkv/rope/proj chain that N14
    // already passed -- an end-to-end relL2 is a sum, and the pair is printed
    // so the device's share is visible.
    // (memory: score-a-projection-against-its-own-input)
    {
        const std::vector<float> ref_proj = ref.f32("block_00.attn_proj");
        std::vector<float> p_npu((size_t)L * D), p_cpu((size_t)L * D);
        proj_op.run(o_npu.data(), L, p_npu.data());
        proj_op.run(o_cpu.data(), L, p_cpu.data());

        const cpu_ref::err_stats e_n =
            cpu_ref::error_matrix(p_npu.data(), ref_proj.data(), p_npu.size());
        const cpu_ref::err_stats e_c =
            cpu_ref::error_matrix(p_cpu.data(), ref_proj.data(), p_cpu.size());
        report("attn_proj vs the HF reference",
               e_n.rel_l2 <= e_c.rel_l2 + bar_rel_l2,
               fmt("device relL2 %.5f cos %.6f; host %.5f / %.6f "
                   "(device may exceed host by up to %.3f)",
                   e_n.rel_l2, e_n.cosine, e_c.rel_l2, e_c.cosine,
                   bar_rel_l2));
    }

    // =======================================================================
    std::printf("\n-- check 7: L=1200, three rounds --\n");
    // =======================================================================
    // The captured 16x16 image is ONE round, so everything the round loop does
    // -- the BD ping-pong, the q row offset advancing by 512*D, the trailing
    // wait that lags a round -- is untested above. 1200 is not an arbitrary
    // bigger number: it is the real operating point, the patch count a 639x480
    // image smart_resizes to.
    {
        const int L2 = 1200;
        const qkv_rows big = tile_rows(in, L2, D, /*seed=*/1200);
        std::vector<float> a((size_t)L2 * D), b((size_t)L2 * D);

        const double ekc = qwen38::vision::effective_key_count(
            big.q.data(), big.k.data(), L2, H, DH, desc.attn_scale);

        cpu_attn.run(big.q.data(), big.k.data(), big.v.data(), L2, b.data());
        npu_attn.run(big.q.data(), big.k.data(), big.v.data(), L2, a.data());

        const cpu_ref::err_stats e =
            cpu_ref::error_matrix(a.data(), b.data(), a.size());
        int worst_h = -1, worst_r = -1;
        const double wh = worst_head_rel_l2(a.data(), b.data(), L2, H, DH,
                                            &worst_h);
        const double wr = worst_row_cos(a.data(), b.data(), L2, D, &worst_r);
        report("device == host at L=1200",
               wh <= bar_rel_l2,
               fmt("relL2 %.5f  worst head %.5f @h%d  (bar %.3f)  "
                   "worst row %.6f @%d  eff keys %.1f of %d",
                   e.rel_l2, wh, worst_h, bar_rel_l2, wr, worst_r, ekc, L2));
    }

    // =======================================================================
    std::printf("\n-- check 8: a short image after a long one --\n");
    // =======================================================================
    // The BOs are grow-only, so after L=1200 they hold 1536 rows of the
    // previous image. A short call must not see them. The mask covers the KEY
    // tail inside the kernel, but the QK mmul still READS those rows, and the
    // v tail is not masked at all -- so if the per-call memset were dropped,
    // this is the check that would catch it, and nothing above would.
    // (memory: kv-rows-above-cursor-are-not-inert)
    //
    // L=200 also lands mid-chunk, which is the only test here of L_valid's
    // masking: 256 and 1200 both happen to be exactly representable choices,
    // 200 leaves 312 masked key rows carrying a previous image's data.
    {
        const int L3 = 200;
        std::vector<float> a((size_t)L3 * D), b((size_t)L3 * D), fresh((size_t)L3 * D);

        cpu_attn.run(in.q.data(), in.k.data(), in.v.data(), L3, b.data());
        npu_attn.run(in.q.data(), in.k.data(), in.v.data(), L3, a.data());

        // The same call on an op that has never seen a long image. If the two
        // device results differ AT ALL, the tail leaked; bit identity is the
        // right bar for a question about stale memory, not a cosine.
        // (memory: gate-an-overlapped-replay-on-bit-identity)
        q38npu::npu_vision_attn_op clean(attn_mgr->create_app(), rtp);
        clean.configure(L3, H, DH, desc.attn_scale);
        clean.run(in.q.data(), in.k.data(), in.v.data(), L3, fresh.data());

        const bool identical =
            std::memcmp(a.data(), fresh.data(), a.size() * sizeof(float)) == 0;
        int worst_h = -1;
        const double wh = worst_head_rel_l2(a.data(), b.data(), L3, H, DH,
                                            &worst_h);
        report("L=200 after L=1200 leaves no residue",
               identical && wh <= bar_rel_l2,
               fmt("bit-identical to a fresh op: %s; relL2 vs host %.5f, "
                   "worst head %.5f @h%d (bar %.3f)",
                   identical ? "yes" : "NO -- the tail leaked",
                   cpu_ref::error_matrix(a.data(), b.data(), a.size()).rel_l2,
                   wh, worst_h, bar_rel_l2));
    }

    // =======================================================================
    std::printf("\n-- check 9: what the move costs, at both sizes --\n");
    // =======================================================================
    // Not gated, and reported at TWO sizes because a fixed-cost dispatch
    // against a linear host path crosses somewhere, and one number would have
    // the wrong sign on the other side of it.
    // (memory: ab-at-one-size-can-have-the-wrong-sign)
    //
    // Measured at the engine's own flags, the crossover is below the smallest
    // image this checkpoint produces: 2.8x at L=256 and 5.3x at L=1200. The op
    // therefore dispatches at every size with no host threshold -- one code
    // path, so an end-to-end A/B measures exactly what ships.
    //
    // Build this driver at -O2 and the host arm slows 2.6-3x and those ratios
    // read 6.5x and 13.7x. See the file header.
    {
        auto bench = [&](int rows, const qkv_rows& x, bool device) {
            std::vector<float> o((size_t)rows * D);
            auto once = [&] {
                if (device) npu_attn.run(x.q.data(), x.k.data(), x.v.data(),
                                         rows, o.data());
                else        cpu_attn.run(x.q.data(), x.k.data(), x.v.data(),
                                         rows, o.data());
            };
            for (int i = 0; i < 3; i++) once();          // warm up
            const int iters = rows > 512 ? 10 : 20;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++) once();
            return ms_since(t0) / iters;
        };

        const qkv_rows big = tile_rows(in, 1200, D, /*seed=*/1200);
        const double d256 = bench(L,    in,  true),  h256 = bench(L,    in,  false);
        const double d12  = bench(1200, big, true),  h12  = bench(1200, big, false);

        std::printf("     L=%-5d device %7.2f ms   host %7.2f ms   %.2fx\n",
                    L,    d256, h256, h256 / d256);
        std::printf("     L=%-5d device %7.2f ms   host %7.2f ms   %.2fx\n",
                    1200, d12,  h12,  h12 / d12);
        std::printf("     per image (27 blocks): device %.0f ms, host %.0f ms "
                    "at L=1200\n", d12 * 27, h12 * 27);
    }

    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
