// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify_n14_vision.cpp
/// \brief N14 -- the vision tower (`image/`) against a captured HF reference,
///        stage by stage, with the controls that a cosine alone cannot supply.
///
/// Build:
///   source ~/setup.sh
///   g++-13 -O2 -march=native -fopenmp -std=c++20
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n14_vision.cpp image/image_cpu.cpp image/vision_ops.cpp
///       image/vision_glue.cpp image/vision_rope.cpp image/mrope.cpp
///       -lpthread -o /tmp/verify_n14_vision
///   (one line; the continuations are dropped here because a backslash at the
///    end of a // comment splices the next line into it)
///
/// Run:
///   /tmp/verify_n14_vision [model_dir] [vision_ref.safetensors]
///   FLM_Q38_VISION_FP32=1 /tmp/verify_n14_vision      # the precision control
///
/// No NPU. Everything in `image/` is host code today; this driver exists so
/// that when the matmuls and the attention move behind `vision::op_factory`
/// there is already a per-stage reference to move them against.
///
/// ---------------------------------------------------------------------------
/// Where the reference comes from
/// ---------------------------------------------------------------------------
/// `models_simple/capture_qwen3_8_vision.py`. It loads ONLY `model.visual.*`
/// (333 tensors, 879 MiB, all in shard 1 of 18), runs one image through
/// `Qwen3_5VisionModel` with `_attn_implementation = "eager"`, and hooks every
/// stage. The weights are the checkpoint's bf16 VALUES cast to fp32 and the
/// arithmetic is fp32, which is deliberately the same contract as
/// `vision::cpu_mm_op` -- so with FLM_Q38_VISION_FP32=1 the two run the same
/// function and any gap is a real difference, not a rounding budget.
///
/// The shipped `ref.safetensors` cannot do this job: it has the tower's OUTPUT
/// (`prefill.image_embeds`) and its grid, but no `pixel_values`, so nothing in
/// it can drive an encoder comparison. It is still used here, for one thing the
/// vision dump does not carry -- the m-rope positions of a real multimodal
/// prompt.
///
/// ---------------------------------------------------------------------------
/// The two runs, and what each one measures
/// ---------------------------------------------------------------------------
/// `vision::cpu_ops_emulate_bf16()` caches its environment read in a function
/// static, so a process is one mode or the other for its whole life and the
/// two runs are two invocations:
///
///   default                 activations round through bf16 at every op
///                           boundary, which is what the device will do. This
///                           is the mode the engine ships in, so it is the one
///                           whose numbers matter.
///   FLM_Q38_VISION_FP32=1   the same arithmetic the reference ran. A stage
///                           that only passes here has a RANGE problem; a stage
///                           that fails in both has an addressing problem. That
///                           separation is the entire reason the switch exists.
///
/// ---------------------------------------------------------------------------
/// Every score is paired with a control that must collapse
/// ---------------------------------------------------------------------------
/// A cosine near 1 is evidence about arithmetic and says nothing about
/// addressing, so each claim here is made twice -- once as itself and once as
/// the nearest wrong thing:
///
///   pos_embed in raster order   the merge-block walk
///     instead of merge-block    `reshape(h/m,m,w/m,m).permute(0,2,1,3)` is a
///                               pure permutation, and a permuted [L][1152]
///                               is still a plausible tensor. At a square grid
///                               it is also the ONLY thing separating the two.
///   attention head h against    a consistent permutation agrees with itself,
///     k/v head (h+1)%16         so only a cross-head reference sees it.
///                               (memory: cross-head-control-catches-gqa-mapping)
///   image_embeds shifted one    catches "the rows are right but they start in
///     merged row                the wrong place", which the aggregate cosine
///                               of a smooth image will not.
///
/// Two of the controls do NOT collapse, and both are gated on something else
/// because of what they revealed rather than in spite of it.
///
/// **Dropping the vision RoPE entirely still scores cos 0.9975.** The softmax
/// over an image is diffuse -- 18.6 effective keys out of 256 at block 0 -- so
/// the output is close to an average of v whichever way the scores are tilted,
/// and an average of 256 smooth rows points in nearly the same direction
/// however it is weighted. What moves is the LENGTH, not the angle: relL2 goes
/// from 0.0018 to 0.026, fourteen-fold. So that control is gated on relL2, and
/// the cosine is printed beside it as the evidence that cosine was the wrong
/// instrument. (memory: per-row-cosine-reports-the-norm)
///
/// **The merger's two gelus are indistinguishable under bf16 rounding.** The
/// tower uses both: `Qwen3_5VisionMLP` takes `vision_config.hidden_act`
/// ("gelu_pytorch_tanh") while `Qwen3_5VisionPatchMerger` hardcodes
/// `nn.GELU()`, the exact erf form. They agree to ~1e-3 absolute, which is
/// right at the bf16 activation floor -- scored through `image_embeds` the two
/// land 6e-7 apart with the WRONG one very slightly ahead, which is noise, not
/// a finding. The check is therefore made where it has signal: against the
/// `merger_fc1` tap, which is `act_fn`'s own output rather than that output
/// mixed through a 4608 -> 5120 matmul, and the direction is gated only in the
/// fp32 run where the two sides share their arithmetic and the function is the
/// only thing left that differs.
/// (memory: gate-a-control-against-the-error-floor)
///
/// The m-rope check has the same shape of problem. The captured prompt's image
/// is 16x16 patches -- square -- and for a square grid `st + max(mh, mw)` and
/// 5VL's `st + mw` are the same number, so the real dump cannot separate the
/// two rules. Check 14 is therefore a synthetic PORTRAIT grid, which is the
/// only thing that sees it.
#include "image/image_cpu.hpp"
#include "image/mrope.hpp"
#include "image/vision_glue.hpp"
#include "image/vision_model_def.hpp"
#include "image/vision_ops.hpp"
#include "image/vision_rope.hpp"
#include "image/vision_weights.hpp"

#include "common/cpu_ref_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

using qwen38::image_grid;
using qwen38::vision::activation;

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-46s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
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

/// One tapped stage, kept whole so it can be scored, permuted and re-fed.
struct stage {
    std::vector<float> v;
    int rows = 0, cols = 0;
};

/// The largest per-row shortfall, alongside the aggregate. A flat additive
/// error makes the smallest-norm rows the worst-scoring ones, so this is
/// reported for ADDRESSING and the aggregate relL2 is what gates magnitude.
/// (memory: per-row-cosine-reports-the-norm, worst-row-cosine-is-a-minimum-over-n)
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

}  // namespace

// ===========================================================================
int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string ref_path =
        argc > 2 ? argv[2] : model_dir + "/vision_ref.safetensors";

    const bool bf16 = qwen38::vision::cpu_ops_emulate_bf16();

    std::printf("== N14: the vision tower against its HF reference ==\n");
    std::printf("   model %s\n   ref   %s\n   mode  %s\n\n",
                model_dir.c_str(), ref_path.c_str(),
                bf16 ? "bf16 activations (shipping default)"
                     : "fp32 activations (FLM_Q38_VISION_FP32=1)");

    // The bar moves with the mode, and only with the mode. In fp32 the driver
    // and the reference run the same function over the same weight values, so
    // anything below 0.99999 is a difference and not a rounding budget. With
    // bf16 activation rounding at every one of ~110 op boundaries the error
    // compounds down the 27 blocks; the bar is set from what that actually
    // measures rather than from taste, and the fp32 run is what says the gap
    // is precision.
    const double bar_early = bf16 ? 0.9990 : 0.999990;   // through the blocks
    const double bar_final = bf16 ? 0.9980 : 0.999980;   // merger output

    // ---- config ------------------------------------------------------------
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
        std::fprintf(stderr, "config.json has no vision_config -- nothing to verify\n");
        return 1;
    }

    cpu_ref::safetensors ref(ref_path);

    // =======================================================================
    std::printf("-- checks 1-2: geometry --\n");
    // =======================================================================
    const std::vector<int32_t> thw = ref.i32("image_grid_thw");
    const image_grid g{thw[0], thw[1], thw[2]};
    const int L    = desc.patch_count(g.t, g.h, g.w);
    const int rows = desc.token_count(g.t, g.h, g.w);
    const int D    = (int)desc.D;

    {
        const cpu_ref::tensor_info& pv = ref.info("pixel_values");
        const cpu_ref::tensor_info& ie = ref.info("image_embeds");
        const bool ok = pv.shape.size() == 2 &&
                        pv.shape[0] == L &&
                        pv.shape[1] == (int64_t)desc.patch_dim() &&
                        ie.shape[0] == rows &&
                        ie.shape[1] == (int64_t)desc.out_hidden_size;
        report("grid agrees with the dump", ok,
               fmt("thw %dx%dx%d -> %d patches, %d tokens; pixel_values "
                   "[%lld,%lld] embeds [%lld,%lld]",
                   g.t, g.h, g.w, L, rows,
                   (long long)pv.shape[0], (long long)pv.shape[1],
                   (long long)ie.shape[0], (long long)ie.shape[1]));
    }

    // ---- the tower ---------------------------------------------------------
    // Constructing it opens vision_weight.q4nx, which is what corrects D_ff
    // from the config's 4304 to the shipped 4608. Reported rather than
    // asserted: the padding is the checkpoint's business, the STRIDE is ours.
    auto t_load = std::chrono::steady_clock::now();
    qwen38::image_cpu tower(desc,
        std::make_unique<qwen38::vision::cpu_op_factory>());
    const double load_ms = ms_since(t_load);

    {
        const bool ok = desc.D_ff >= desc.D_ff_config &&
                        desc.merger_in() == desc.D * desc.merge_unit();
        report("tensor widths adopted", ok,
               fmt("D_ff config %u -> tensor %u, merger_in %u, backend %s, "
                   "load %.0f ms",
                   desc.D_ff_config, desc.D_ff, desc.merger_in(),
                   tower.backend(), load_ms));
    }

    // =======================================================================
    std::printf("\n-- checks 3-9: the tower, stage by stage --\n");
    // =======================================================================
    std::map<std::string, stage> got;
    tower.set_tap([&](const char* name, const float* p, int r, int c) {
        stage& s = got[name];
        s.rows = r; s.cols = c;
        s.v.assign(p, p + (size_t)r * c);
    });

    const std::vector<float> pixel_values = ref.f32("pixel_values");
    std::vector<float>       out((size_t)rows * desc.out_hidden_size, 0.0f);

    auto t_enc = std::chrono::steady_clock::now();
    tower.encode(pixel_values.data(), g, out.data());
    const double enc_ms = ms_since(t_enc);

    /// Score a tapped stage against the dump. Returns the aggregate stats so a
    /// caller can compare two candidates against each other.
    auto score = [&](const char* tag, const std::string& stage_name,
                     const std::string& ref_key, double bar) {
        auto it = got.find(stage_name);
        if (it == got.end()) {
            report(tag, false, "stage '" + stage_name + "' was never tapped");
            return cpu_ref::err_stats{};
        }
        const stage& s = it->second;
        const std::vector<float> r = ref.f32(ref_key);
        if (r.size() != s.v.size()) {
            report(tag, false,
                   fmt("size %zu vs reference %zu", s.v.size(), r.size()));
            return cpu_ref::err_stats{};
        }
        const cpu_ref::err_stats e =
            cpu_ref::error_matrix(s.v.data(), r.data(), s.v.size());
        int worst_at = -1;
        const double wr = worst_row_cos(s.v.data(), r.data(), s.rows, s.cols,
                                        &worst_at);
        report(tag, e.cosine >= bar,
               fmt("cos %.7f  relL2 %.5f  worst row %.6f @%d  (bar %.6f)",
                   e.cosine, e.rel_l2, wr, worst_at, bar));
        return e;
    };

    score("patch_embed",  "patch_embed",  "patch_embed",  bar_early);
    score("pos_embed",    "pos_embed",    "pos_embed",    bar_early);
    score("embed",        "embed",        "embed",        bar_early);

    // Every block is scored; only the extremes are printed, because 27 blocks
    // times three stages is 81 lines of noise around the two that matter.
    {
        double worst_cos = 2.0; std::string worst_key;
        bool all_ok = true;
        for (uint32_t i = 0; i < desc.depth; i++) {
            char pre[32];
            std::snprintf(pre, sizeof pre, "block_%02u", i);
            for (const char* sub : {".norm1", ".attn_proj", ".out"}) {
                const std::string key = std::string(pre) + sub;
                auto it = got.find(key);
                if (it == got.end() || !ref.has(key)) { all_ok = false; continue; }
                const std::vector<float> r = ref.f32(key);
                const cpu_ref::err_stats e = cpu_ref::error_matrix(
                    it->second.v.data(), r.data(), it->second.v.size());
                if (e.cosine < worst_cos) { worst_cos = e.cosine; worst_key = key; }
            }
        }
        report("all 27 blocks (norm1, attn_proj, out)",
               all_ok && worst_cos >= bar_early,
               fmt("worst of 81 stages: %s cos %.7f  (bar %.6f)",
                   worst_key.c_str(), worst_cos, bar_early));
    }

    score("block_26.out (last block)", "block_26.out", "block_26.out", bar_early);
    score("merger_norm",  "merger_norm",  "merger_norm",  bar_early);
    score("merger_fc1",   "merger_fc1",   "merger_fc1",   bar_final);

    cpu_ref::err_stats final_e;
    {
        const std::vector<float> r = ref.f32("image_embeds");
        final_e = cpu_ref::error_matrix(out.data(), r.data(), out.size());
        int worst_at = -1;
        const double wr = worst_row_cos(out.data(), r.data(), rows,
                                        (int)desc.out_hidden_size, &worst_at);
        report("image_embeds (tower output)", final_e.cosine >= bar_final,
               fmt("cos %.7f  relL2 %.5f  worst row %.6f @%d  (bar %.6f)",
                   final_e.cosine, final_e.rel_l2, wr, worst_at, bar_final));
    }

    // =======================================================================
    std::printf("\n-- checks 10-11: controls that must collapse --\n");
    // =======================================================================
    // A control is only evidence if it is FAR away. 0.99 is not a collapse at
    // these dimensions -- two independent smooth tensors of 1152 columns
    // routinely score 0.9; the bar is set where a real permutation lands.
    constexpr double kCollapse = 0.98;

    // ---- 10: pos_embed walked in raster order ------------------------------
    // Rebuilt by permuting the correct answer rather than by reimplementing the
    // interpolation, so the control tests the ORDER and nothing else.
    {
        const int m        = (int)desc.spatial_merge_size;
        const int w_blocks = g.w / m;
        const stage& pe    = got["pos_embed"];
        std::vector<int> blk_of_rc((size_t)g.h * g.w, 0);
        for (int by = 0; by < g.h / m; by++)
            for (int bx = 0; bx < w_blocks; bx++)
                for (int dy = 0; dy < m; dy++)
                    for (int dx = 0; dx < m; dx++) {
                        const int s = ((by * w_blocks + bx) * m + dy) * m + dx;
                        blk_of_rc[(size_t)(by * m + dy) * g.w + (bx * m + dx)] = s;
                    }
        std::vector<float> cand((size_t)L * D);
        for (int t = 0; t < g.t; t++)
            for (int s = 0; s < g.h * g.w; s++)
                std::memcpy(cand.data() + ((size_t)t * g.h * g.w + s) * D,
                            pe.v.data() + ((size_t)t * g.h * g.w +
                                           blk_of_rc[(size_t)s]) * D,
                            sizeof(float) * D);
        const std::vector<float> r = ref.f32("pos_embed");
        const cpu_ref::err_stats e =
            cpu_ref::error_matrix(cand.data(), r.data(), cand.size());
        report("CONTROL pos_embed in raster order", e.cosine < kCollapse,
               fmt("cos %.6f, must be < %.2f", e.cosine, kCollapse));
    }

    // ---- 11: image_embeds shifted one merged row ---------------------------
    {
        const int OD = (int)desc.out_hidden_size;
        std::vector<float> cand((size_t)rows * OD);
        for (int r0 = 0; r0 < rows; r0++)
            std::memcpy(cand.data() + (size_t)r0 * OD,
                        out.data() + (size_t)((r0 + 1) % rows) * OD,
                        sizeof(float) * OD);
        const std::vector<float> r = ref.f32("image_embeds");
        const cpu_ref::err_stats e =
            cpu_ref::error_matrix(cand.data(), r.data(), cand.size());
        report("CONTROL image_embeds shifted one row", e.cosine < kCollapse,
               fmt("cos %.6f, must be < %.2f", e.cosine, kCollapse));
    }

    // =======================================================================
    std::printf("\n-- checks 12-13: attention and the merger, driven directly --\n");
    // =======================================================================
    // A second view of the same mmap'd checkpoint. The tower owns its weights
    // privately, and these checks need to bind ops of their own.
    qwen38::vision::weights w(desc);

    // ---- 12: block 0's attention, with a rotated-head control --------------
    {
        const int H  = (int)desc.num_heads;
        const int DH = (int)desc.DH;
        const std::vector<float> norm1 = ref.f32("block_00.norm1");

        qwen38::vision::cpu_mm_op qkv_op, proj_op;
        qkv_op.bind(w.block(0).qkv_w, w.block(0).qkv_b, D, 3 * D, activation::none);
        proj_op.bind(w.block(0).proj_w, w.block(0).proj_b, D, D, activation::none);

        std::vector<float> qkv((size_t)L * 3 * D);
        qkv_op.run(norm1.data(), L, qkv.data());

        std::vector<float> q((size_t)L * D), k((size_t)L * D), v((size_t)L * D);
        for (int t = 0; t < L; t++) {
            const float* r0 = qkv.data() + (size_t)t * 3 * D;
            std::memcpy(q.data() + (size_t)t * D, r0,         sizeof(float) * D);
            std::memcpy(k.data() + (size_t)t * D, r0 + D,     sizeof(float) * D);
            std::memcpy(v.data() + (size_t)t * D, r0 + 2 * D, sizeof(float) * D);
        }
        const std::vector<float> q_raw = q, k_raw = k;

        const qwen38::vision::rope_tables rope = qwen38::vision::build_rope(
            g.t, g.h, g.w, (int)desc.spatial_merge_size, DH, desc.rope_theta);
        qwen38::vision::apply_rope(q.data(), L, H, rope);
        qwen38::vision::apply_rope(k.data(), L, H, rope);

        // A random q/k pair puts almost all the softmax mass on a handful of
        // keys, and then any addressing scores well. This says how many keys
        // the comparison is actually about.
        // (memory: random-kv-cache-cannot-test-attention)
        const double ekc = qwen38::vision::effective_key_count(
            q.data(), k.data(), L, H, DH, desc.attn_scale);

        const std::vector<float> ref_proj = ref.f32("block_00.attn_proj");
        std::vector<float> o((size_t)L * D), p((size_t)L * D);

        qwen38::vision::cpu_attn_op attn;
        attn.configure(L, H, DH, desc.attn_scale);

        attn.run(q.data(), k.data(), v.data(), L, o.data());
        proj_op.run(o.data(), L, p.data());
        const cpu_ref::err_stats e_ok =
            cpu_ref::error_matrix(p.data(), ref_proj.data(), p.size());
        report("block 0 attention rebuilt from norm1",
               e_ok.cosine >= bar_early,
               fmt("cos %.7f  relL2 %.5f  eff keys %.1f of %d  (bar %.6f)",
                   e_ok.cosine, e_ok.rel_l2, ekc, L, bar_early));

        std::vector<int> perm((size_t)H);
        for (int h = 0; h < H; h++) perm[(size_t)h] = (h + 1) % H;
        attn.run(q.data(), k.data(), v.data(), L, o.data(), perm.data());
        proj_op.run(o.data(), L, p.data());
        const cpu_ref::err_stats e_rot =
            cpu_ref::error_matrix(p.data(), ref_proj.data(), p.size());
        report("CONTROL q head h against k/v head h+1",
               e_rot.cosine < kCollapse,
               fmt("cos %.6f, must be < %.2f", e_rot.cosine, kCollapse));

        // Gated on relL2, not on cosine, and that is a finding rather than a
        // convenience. Dropping the rotation entirely still scores cos 0.9975
        // here: the softmax is diffuse (18.6 effective keys of 256), so the
        // output is close to a uniform average of v whichever way the scores
        // are tilted, and an average of 256 smooth rows points in nearly the
        // same direction no matter what the weights are. Cosine cannot see a
        // 14x error that does not rotate the result. relL2 can, and the
        // separation there is not marginal.
        attn.run(q_raw.data(), k_raw.data(), v.data(), L, o.data());
        proj_op.run(o.data(), L, p.data());
        const cpu_ref::err_stats e_norope =
            cpu_ref::error_matrix(p.data(), ref_proj.data(), p.size());
        report("CONTROL rope not applied",
               e_norope.rel_l2 > 0.05 && e_norope.rel_l2 > 20.0 * e_ok.rel_l2,
               fmt("relL2 %.5f vs %.5f with rope (%.0fx); cos %.6f, which is "
                   "why cosine is not the gate",
                   e_norope.rel_l2, e_ok.rel_l2,
                   e_ok.rel_l2 > 0 ? e_norope.rel_l2 / e_ok.rel_l2 : 0.0,
                   e_norope.cosine));
    }

    // ---- 13: the merger's two gelus ----------------------------------------
    // Not a collapse, and not scored through fc2 either. See the file header:
    // the tanh and erf forms agree to ~1e-3, so what is gated is the
    // DIRECTION. That only has signal if the measurement is close to the
    // activation -- `merger_fc1` is a direct tap of `act_fn`'s output, whereas
    // `image_embeds` is that same tensor mixed through a 4608 -> 5120 matmul,
    // which averages the discrepancy away to nothing.
    //
    // Even at fc1 the difference is small, and in the shipping bf16 mode it is
    // BELOW THE ROUNDING FLOOR: the two score within 1e-6 of each other and
    // which one wins is noise. So the direction is gated only in fp32, where
    // the reference and this code run the same arithmetic and the only
    // remaining difference is the function itself. In bf16 the numbers are
    // printed and the check passes on magnitude alone, which is the honest
    // thing to do rather than gating on a coin toss.
    // (memory: gate-a-control-against-the-error-floor)
    {
        const int MI = (int)desc.merger_in();
        const stage& mn = got["merger_norm"];
        const std::vector<float> r = ref.f32("merger_fc1");

        auto run_with = [&](activation act) {
            qwen38::vision::cpu_mm_op fc1;
            fc1.bind(w.merger_fc1_w(), w.merger_fc1_b(), MI, MI, act);
            std::vector<float> mid((size_t)rows * MI);
            fc1.run(mn.v.data(), rows, mid.data());
            return cpu_ref::error_matrix(mid.data(), r.data(), mid.size());
        };

        const cpu_ref::err_stats e_erf  = run_with(activation::gelu_erf);
        const cpu_ref::err_stats e_tanh = run_with(activation::gelu);
        const bool  discriminating = !bf16;
        const bool  ok = e_erf.cosine >= bar_final &&
                         (!discriminating || e_erf.rel_l2 < e_tanh.rel_l2);
        report("merger uses gelu_erf, not gelu_tanh", ok,
               fmt("relL2 erf %.6f vs tanh %.6f%s",
                   e_erf.rel_l2, e_tanh.rel_l2,
                   discriminating
                       ? ""
                       : " -- indistinguishable under bf16 rounding; the "
                         "direction is gated in the fp32 run"));
    }

    // =======================================================================
    std::printf("\n-- checks 14-16: m-rope --\n");
    // =======================================================================
    // The one thing the vision dump cannot carry: where an image's rows sit in
    // a real prompt's position walk. `ref.safetensors` has the whole 221-token
    // multimodal prefill, so this is an EXACT check -- both sides are integers
    // and every difference is a bug.
    {
        cpu_ref::safetensors lm(model_dir + "/ref.safetensors");
        const std::vector<int32_t> ids  = lm.i32("prefill.input_ids");
        const std::vector<int32_t> pos  = lm.i32("prefill.position_ids");
        const std::vector<int32_t> lthw = lm.i32("prefill.image_grid_thw");
        const int P = (int)ids.size();

        const image_grid lg{lthw[0], lthw[1], lthw[2]};
        qwen38::mrope_cursor cur;
        std::vector<qwen38::image_span> spans;
        const std::vector<int32_t> mine = qwen38::build_mrope(
            ids.data(), P, desc.image_token_id, &lg, 1,
            (int)desc.spatial_merge_size, cur, &spans);

        size_t diff = 0; int first = -1;
        for (size_t i = 0; i < mine.size() && i < pos.size(); i++)
            if (mine[i] != pos[i]) { diff++; if (first < 0) first = (int)i; }
        report("position_ids match the dumped prefill",
               mine.size() == pos.size() && diff == 0,
               fmt("%d tokens x 3 channels, %zu words differ%s",
                   P, diff,
                   first < 0 ? "" : fmt(" (first at [%d][%d])",
                                        first / P, first % P).c_str()));

        const bool span_ok = spans.size() == 1 &&
                             spans[0].count == desc.token_count(lg.t, lg.h, lg.w) &&
                             spans[0].image_index == 0 &&
                             spans[0].token_offset == 0;
        report("one span covering the image's rows", span_ok,
               spans.empty()
                   ? std::string("no span produced")
                   : fmt("begin %d count %d (grid %dx%dx%d -> %d tokens)",
                         spans[0].begin, spans[0].count,
                         lg.t, lg.h, lg.w,
                         desc.token_count(lg.t, lg.h, lg.w)));
    }

    // ---- 16: the portrait rule ---------------------------------------------
    // The dumped image is 16x16 patches, so its merged grid is 8x8 and
    // `st + max(mh, mw)` equals `st + mw`. Check 14 therefore passes under
    // BOTH rules and says nothing about which one is implemented. A portrait
    // grid is the only thing that separates them, and there isn't one in any
    // dump -- so it is built here.
    {
        const int m  = (int)desc.spatial_merge_size;
        const int gh = 8 * m, gw = 2 * m;           // merged 8 x 2, portrait
        const int mh = gh / m, mw = gw / m;
        const int n_img = (gh * gw) / (m * m);

        std::vector<int32_t> toks;
        toks.push_back(1);                                   // one text token
        for (int i = 0; i < n_img; i++) toks.push_back(desc.image_token_id);
        toks.push_back(2);                                   // and one after
        const int P = (int)toks.size();

        const image_grid pg{1, gh, gw};
        qwen38::mrope_cursor cur;
        const std::vector<int32_t> p = qwen38::build_mrope(
            toks.data(), P, desc.image_token_id, &pg, 1, m, cur, nullptr);

        const int st   = 1;                 // image starts right after token 0
        const int next = p[(size_t)(P - 1)];            // t-channel of the tail
        const bool ok  = next == st + std::max(mh, mw) && next != st + mw;

        // Spot-check the interior rule too: image token s -> (st, st+s/mw, st+s%mw)
        bool rows_ok = true;
        for (int s = 0; s < n_img; s++) {
            const int i = 1 + s;
            rows_ok &= p[(size_t)i]             == st &&
                       p[(size_t)P + i]         == st + s / mw &&
                       p[(size_t)2 * P + i]     == st + s % mw;
        }
        report("portrait image advances by max(mh, mw)", ok && rows_ok,
               fmt("merged %dx%d: next text pos %d, max rule %d, width rule %d"
                   "%s", mh, mw, next, st + std::max(mh, mw), st + mw,
                   rows_ok ? "" : "; INTERIOR ROWS WRONG"));
    }

    // =======================================================================
    std::printf("\n-- cost (not a gate) --\n");
    // =======================================================================
    std::printf("  %-46s %.1f ms\n", "tower load (mmap + norms)", load_ms);
    std::printf("  %-46s %.1f ms  (%d patches -> %d tokens)\n",
                "encode, whole tower", enc_ms, L, rows);
    std::printf("  %-46s %.2f ms\n", "per merged token", enc_ms / rows);

    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    if (n_fail && bf16)
        std::printf("   re-run with FLM_Q38_VISION_FP32=1: if it passes there,\n"
                    "   the gap is activation rounding and not addressing.\n");
    return n_fail == 0 ? 0 : 1;
}
