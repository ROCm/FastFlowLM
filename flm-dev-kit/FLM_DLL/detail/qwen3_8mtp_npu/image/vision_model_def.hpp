// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/vision_model_def.hpp
/// \brief Shape definition for the Qwen3.8-27B vision tower.
///
/// Built from config.json's `vision_config`, which uses HF-native key names
/// (`hidden_size`, `depth`, `num_heads`, ...). This is NOT the qwen3_5vl_npu
/// convention: that engine reads renamed `QWEN3_5_VISION_*` keys written by a
/// converter. The tower itself is the same architecture; only the key names
/// differ, so do not copy the 5VL config reader.
///
/// Mirrors qwen3_8mtp_model_def.hpp: one place reads config.json, and anything
/// that disagrees with the shipped tensors throws at load rather than reading
/// out of bounds.
#ifndef __QWEN3_8MTP_VISION_MODEL_DEF_HPP__
#define __QWEN3_8MTP_VISION_MODEL_DEF_HPP__
#pragma once

#include "lm_config.hpp"
#include "utils/utils.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

/// \brief Geometry of the 27-block Qwen3.5 vision tower shipped with the 27B.
struct qwen3_8mtp_vision_desc {
    /// False when config.json has no `vision_config`. The engine stays
    /// text-only in that case instead of failing to load.
    bool present = false;

    // ---- tower ------------------------------------------------------------
    uint32_t depth       = 0;   ///< 27 blocks
    uint32_t D           = 0;   ///< vision hidden_size, 1152
    uint32_t num_heads   = 0;   ///< 16
    uint32_t DH          = 0;   ///< D / num_heads, 72
    float    attn_scale  = 0.0f;

    /// The MLP width the config declares (4304) and the width the tensors
    /// actually carry (4608). They differ: the checkpoint is zero-padded out to
    /// a multiple of the NPU MM tile. The pad is exactly zero in fc1.weight,
    /// fc1.bias and in fc2.weight's columns, so gelu(0)=0 and the padded
    /// columns contribute nothing -- the two widths give identical output.
    /// Everything downstream uses `D_ff`, the tensor width, because that is
    /// what the strides must match.
    uint32_t D_ff_config = 0;   ///< 4304, kept only to report the mismatch
    uint32_t D_ff        = 0;   ///< 4608, the shipped width

    // ---- patch embedding --------------------------------------------------
    uint32_t in_channels         = 0;   ///< 3
    uint32_t patch_size          = 0;   ///< 16
    uint32_t temporal_patch_size = 0;   ///< 2
    uint32_t spatial_merge_size  = 0;   ///< 2

    // ---- position embedding ------------------------------------------------
    uint32_t num_position_embeddings = 0;   ///< 2304
    /// sqrt(num_position_embeddings), 48. The learned table is a square grid
    /// that gets bilinearly resampled onto whatever grid the image produced.
    uint32_t num_grid_per_side = 0;

    // ---- merger ------------------------------------------------------------
    uint32_t out_hidden_size = 0;   ///< 5120, the language model's D

    // ---- numerics ----------------------------------------------------------
    /// LayerNorm epsilon. Not in this config.json; HF's Qwen3VL vision blocks
    /// construct nn.LayerNorm with eps=1e-6, and the merger norm uses the same
    /// value. Overridable via `layer_norm_eps` if a checkpoint ever ships one.
    float ln_eps = 1e-6f;
    /// Vision RoPE base. HF hardcodes 10000 for the vision tower; it is NOT
    /// the language model's rope_theta, which is three orders of magnitude
    /// larger -- see cpu_ref::ROPE_THETA, and do not restate the number here.
    float rope_theta = 10000.0f;

    // ---- token ids ---------------------------------------------------------
    /// Top-level keys, not under vision_config.
    int32_t image_token_id        = -1;   ///< 248056
    int32_t video_token_id        = -1;   ///< 248057
    int32_t vision_start_token_id = -1;
    int32_t vision_end_token_id   = -1;

    /// The tower's own checkpoint, a sibling of model.q4nx. Despite the
    /// extension it is an ordinary safetensors file whose tensors are all BF16
    /// -- nothing in the vision tower is quantised -- so cpu_ref::safetensors
    /// reads it directly and no Q4NX machinery is involved.
    std::string weight_path;

    // ---- derived ----------------------------------------------------------
    /// Flattened conv3d kernel volume: 3*2*16*16 = 1536. One row of
    /// pixel_values, and the K of the patch-embed matmul.
    inline uint32_t patch_dim() const {
        return in_channels * temporal_patch_size * patch_size * patch_size;
    }
    /// Patches folded into one language token: 4.
    inline uint32_t merge_unit() const {
        return spatial_merge_size * spatial_merge_size;
    }
    /// The merger's input width, 4608 = D * merge_unit(). Numerically equal to
    /// D_ff here, which is a coincidence of this checkpoint -- keep them apart.
    inline uint32_t merger_in() const { return D * merge_unit(); }

    /// Patches an image of this grid produces, before the merger.
    inline int patch_count(int t, int h, int w) const { return t * h * w; }
    /// Language tokens it produces, after the merger. This is how many rows the
    /// prompt must reserve with image_token_id, and the merged grid is
    /// (h/m) x (w/m) -- which is why an image advances m-rope position by w/m,
    /// not by its token count.
    inline int token_count(int t, int h, int w) const {
        return patch_count(t, h, w) / (int)merge_unit();
    }

    qwen3_8mtp_vision_desc() {}

    void build(LM_Config& config) {
        const nlohmann::json& vc = config.sub("vision_config");
        if (vc.is_null() || vc.empty() || !vc.contains("depth")) {
            present = false;
            return;
        }
        present = true;

        depth       = cfg_get<uint32_t>(vc, "depth", 0);
        D           = cfg_get<uint32_t>(vc, "hidden_size", 0);
        num_heads   = cfg_get<uint32_t>(vc, "num_heads", 0);
        D_ff_config = cfg_get<uint32_t>(vc, "intermediate_size", 0);

        in_channels         = cfg_get<uint32_t>(vc, "in_channels", 3);
        patch_size          = cfg_get<uint32_t>(vc, "patch_size", 16);
        temporal_patch_size = cfg_get<uint32_t>(vc, "temporal_patch_size", 2);
        spatial_merge_size  = cfg_get<uint32_t>(vc, "spatial_merge_size", 2);

        num_position_embeddings = cfg_get<uint32_t>(vc, "num_position_embeddings", 0);
        out_hidden_size         = cfg_get<uint32_t>(vc, "out_hidden_size", 0);

        ln_eps     = cfg_get<float>(vc, "layer_norm_eps", 1e-6f);
        rope_theta = cfg_get<float>(vc, "rope_theta", 10000.0f);

        image_token_id        = config.get<int32_t>("image_token_id", -1);
        video_token_id        = config.get<int32_t>("video_token_id", -1);
        vision_start_token_id = config.get<int32_t>("vision_start_token_id", -1);
        vision_end_token_id   = config.get<int32_t>("vision_end_token_id", -1);

        weight_path = config.model_path + "/vision_weight.q4nx";

        _derive();
        _validate();
    }

    /// Called after the weight file is open: the shipped tensors are the
    /// authority on the padded MLP width, and on nothing else.
    void adopt_tensor_widths(uint32_t fc1_rows) {
        D_ff = fc1_rows;
        if (D_ff < D_ff_config)
            throw std::runtime_error(
                "vision mlp.linear_fc1 has " + std::to_string(D_ff) +
                " rows but vision_config.intermediate_size is " +
                std::to_string(D_ff_config) + " -- the checkpoint is narrower "
                "than the config, which cannot be a zero pad");
    }

    std::string str() const {
        std::ostringstream o;
        auto row = [&](const char* k, auto v) {
            o << std::left << std::setw(28) << k << " = " << v << "\n";
        };
        o << "============ qwen3_8mtp_vision_desc ============\n";
        if (!present) {
            o << "(no vision_config -- text-only checkpoint)\n";
            o << "================================================\n";
            return o.str();
        }
        row("depth",                 depth);
        row("D (vision hidden)",     D);
        row("num_heads",             num_heads);
        row("DH (head_dim)",         DH);
        row("D_ff (tensor)",         D_ff);
        row("D_ff (config)",         D_ff_config);
        row("patch_size",            patch_size);
        row("temporal_patch_size",   temporal_patch_size);
        row("spatial_merge_size",    spatial_merge_size);
        row("patch_dim",             patch_dim());
        row("merger_in",             merger_in());
        row("num_position_embeddings", num_position_embeddings);
        row("num_grid_per_side",     num_grid_per_side);
        row("out_hidden_size",       out_hidden_size);
        row("ln_eps",                ln_eps);
        row("rope_theta",            rope_theta);
        row("image_token_id",        image_token_id);
        row("weight_path",           weight_path);
        o << "================================================\n";
        return o.str();
    }

private:
    void _derive() {
        DH = num_heads ? D / num_heads : 0;
        attn_scale = DH ? 1.0f / std::sqrt((float)DH) : 0.0f;
        // Until the weight file is open, assume the config width. build()
        // callers overwrite this through adopt_tensor_widths().
        D_ff = D_ff_config;

        const double s = std::sqrt((double)num_position_embeddings);
        num_grid_per_side = (uint32_t)std::llround(s);
    }

    void _validate() const {
        auto req = [](const char* what, bool ok) {
            if (!ok) throw std::runtime_error(std::string("vision_config: ") + what);
        };
        req("depth is 0", depth > 0);
        req("hidden_size is 0", D > 0);
        req("num_heads is 0", num_heads > 0);
        req("hidden_size is not a multiple of num_heads", D % num_heads == 0);
        req("out_hidden_size is 0", out_hidden_size > 0);
        req("spatial_merge_size is 0", spatial_merge_size > 0);
        // The 2-D vision RoPE splits head_dim in half and then splits that half
        // between the row and column frequency tables, so head_dim must be a
        // multiple of 4 or the concatenation silently truncates.
        req("head_dim is not a multiple of 4", DH % 4 == 0);
        // num_grid_per_side must be exact: the learned table is indexed as a
        // square, and a non-square count would read across rows.
        req("num_position_embeddings is not a perfect square",
            (uint64_t)num_grid_per_side * num_grid_per_side == num_position_embeddings);

        if (image_token_id < 0)
            throw std::runtime_error(
                "config.json has a vision_config but no image_token_id -- "
                "there is no way to tell which prompt rows the tower feeds");
    }
};

#endif  // __QWEN3_8MTP_VISION_MODEL_DEF_HPP__
