// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/vision_model_def.hpp
/// \brief Shape definition for MiniCPM-V-4.7-1B's SigLIP vision tower.
///
/// Read from the packaged config.json's `vision_config`, which carries
/// ENGINE-named keys (`MINICPM_V_VISION_EMBED_DIM`, ...) like every shipped
/// FLM config; each falls back to the upstream HF name and then to the
/// checkpoint's literal, the same chain `minicpm_v_4_7_npu::
/// load_vision_preprocess_parameters()` uses.
///
/// The geometry is Qwen3.8-27B's tower exactly -- 27 blocks, D 1152, 16 heads x
/// 72, MLP 4304 padded to 4608 -- which is why that tower's bitstreams serve
/// this one unchanged. What is MiniCPM's own:
///
///   * 2-D patches (Conv2d 14x14, no temporal axis): patch_dim 3*14*14 = 588,
///     which is NOT a multiple of vision_mm's k tile of 384. The matmul op pads
///     K to 768 with zeros; see `npu/vision_mm_op.hpp`.
///   * a learned 70x70 position table looked up by NaViT bucketize, not
///     resampled bilinearly (`vision_glue.hpp`);
///   * the `vit_merger` after block `insert_layer_id` (6): windowed 2x2
///     attention, then a 2x2 merge into a 4608 -> 17216 -> 1152 MLP. Blocks
///     7..26 run on a quarter of the tokens;
///   * `post_layernorm`, then the projector's own 2x2 merge into 4608 -> 4608 ->
///     1024 (`merger.*` in vision_weight.q4nx).
///
/// No vision RoPE: SigLIP has none, so attention is permutation-equivariant
/// within a view and the only thing that knows where a patch sits is the
/// position table and the two 2x2 regroups.
#ifndef __MINICPM_V_VISION_MODEL_DEF_HPP__
#define __MINICPM_V_VISION_MODEL_DEF_HPP__
#pragma once

#include "lm_config.hpp"
#include "utils/utils.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

struct minicpm_v_moe_vision_desc {
    /// False when config.json has no `vision_model_weight`: the engine stays
    /// text-only and rejects images, which is the V1 scope.
    bool present = false;

    // ---- tower ------------------------------------------------------------
    uint32_t depth      = 0;   ///< 27
    uint32_t D          = 0;   ///< 1152
    uint32_t num_heads  = 0;   ///< 16
    uint32_t DH         = 0;   ///< 72
    float    attn_scale = 0.0f;

    /// MLP width the config declares (4304) vs the width the shipped tensors
    /// carry (4608, zero-padded onto vision_mm's 384 tile by S7a). The tensors
    /// are the authority; see `adopt_tensor_widths`.
    uint32_t D_ff_config = 0;
    uint32_t D_ff        = 0;

    // ---- patch embedding ----------------------------------------------------
    uint32_t in_channels = 3;
    uint32_t patch_size  = 14;

    // ---- position embedding -------------------------------------------------
    uint32_t num_position_embeddings = 0;   ///< 4900
    uint32_t num_patches_per_side    = 0;   ///< 70

    // ---- vit_merger ---------------------------------------------------------
    /// Index of the block AFTER which the vit_merger runs. -1 disables it,
    /// which is `downsample_mode: "4x"`.
    int32_t  insert_layer_id = -1;
    uint32_t window          = 2;   ///< window_kernel_size, square
    /// vit_merger.mlp.linear_fc1 rows: 17216 declared, 17280 shipped.
    uint32_t vm_ff_config = 0;
    uint32_t vm_ff        = 0;

    // ---- projector ----------------------------------------------------------
    uint32_t merge          = 2;   ///< merge_kernel_size, square
    uint32_t out_hidden_size = 0;  ///< 1024, the language model's D

    float ln_eps = 1e-6f;

    std::string weight_path;

    // ---- derived ----------------------------------------------------------
    /// One row of pixel_values and the K of the patch matmul: 3*14*14 = 588.
    inline uint32_t patch_dim() const { return in_channels * patch_size * patch_size; }
    inline bool     use_vit_merger() const { return insert_layer_id >= 0; }
    /// 4608 = D * window^2: the vit_merger MLP's input.
    inline uint32_t vm_in() const { return D * window * window; }
    /// 4608 = D * merge^2: the projector's input.
    inline uint32_t merger_in() const { return D * merge * merge; }
    /// Patches folded into one language-model row: 16 with the vit_merger, 4
    /// without. This is the processor's `image_token_divisor`.
    inline uint32_t token_divisor() const {
        return (use_vit_merger() ? window * window : 1u) * merge * merge;
    }
    /// LLM rows one view of h x w patches produces.
    inline int tokens_for_view(int h, int w) const { return h * w / (int)token_divisor(); }
    /// The spatial factor a view's grid must divide by: 4 with the vit_merger.
    inline int grid_multiple() const {
        return (int)((use_vit_merger() ? window : 1u) * merge);
    }

    void build(LM_Config& config) {
        weight_path = config.get<std::string>("vision_model_weight", "");
        if (weight_path.empty()) { present = false; return; }
        present = true;
        // A bare filename is relative to the model folder, as model.q4nx is.
        if (weight_path.find('/') == std::string::npos &&
            weight_path.find('\\') == std::string::npos)
            weight_path = utils::path_join(config.model_path, weight_path);

        const nlohmann::json& vc = config.sub("vision_config");
        auto u = [&](const char* eng, const char* hf, uint32_t lit) {
            return cfg_get<uint32_t>(vc, eng, cfg_get<uint32_t>(vc, hf, lit));
        };
        depth       = u("MINICPM_V_VISION_NUM_LAYERS", "num_hidden_layers", 27);
        D           = u("MINICPM_V_VISION_EMBED_DIM", "hidden_size", 1152);
        num_heads   = u("MINICPM_V_VISION_NUM_HEADS", "num_attention_heads", 16);
        D_ff_config = u("MINICPM_V_VISION_MLP_INTERMEDIATE_SIZE", "intermediate_size", 4304);
        patch_size  = u("MINICPM_V_PATCH_SIZE", "patch_size", 14);
        in_channels = u("MINICPM_V_VISION_NUM_CHANNELS", "num_channels", 3);
        num_position_embeddings =
            u("MINICPM_V_VISION_NUM_POSITION_EMBEDDINGS", "num_position_embeddings", 4900);
        out_hidden_size = u("MINICPM_V_VISION_OUT_HIDDEN_SIZE", "out_hidden_size", 1024);
        vm_ff_config    = u("MINICPM_V_VIT_MERGER_INTERMEDIATE_SIZE", "window_intermediate_size",
                            D_ff_config * 4);
        ln_eps = cfg_get<float>(vc, "MINICPM_V_VISION_LAYER_NORM_EPSILON",
                                cfg_get<float>(vc, "layer_norm_eps", 1e-6f));

        // downsample_mode "4x" skips the vit_merger (modeling_minicpmv4_7.py:665).
        const uint32_t factor = cfg_get<uint32_t>(vc, "MINICPM_V_DOWNSAMPLE_FACTOR", 0u);
        const std::string mode = cfg_get<std::string>(config._json_config, "downsample_mode",
                                                      std::string("16x"));
        const bool vm = factor ? (factor == 16u) : (mode != "4x");
        insert_layer_id = vm ? (int32_t)cfg_get<uint32_t>(
                                   vc, "MINICPM_V_INSERT_LAYER_ID",
                                   cfg_get<uint32_t>(config._json_config, "insert_layer_id", 6u))
                             : -1;

        DH         = num_heads ? D / num_heads : 0;
        attn_scale = DH ? 1.0f / std::sqrt((float)DH) : 0.0f;
        D_ff  = D_ff_config;
        vm_ff = vm_ff_config;
        num_patches_per_side = (uint32_t)std::llround(std::sqrt((double)num_position_embeddings));
        _validate();
    }

    /// Called once the weight file is open: the shipped tensors are the
    /// authority on the two padded MLP widths, and on nothing else.
    void adopt_tensor_widths(uint32_t fc1_rows, uint32_t vm_fc1_rows) {
        if (fc1_rows < D_ff_config || (use_vit_merger() && vm_fc1_rows < vm_ff_config))
            throw std::runtime_error(
                "vision_weight.q4nx MLP is narrower than vision_config (fc1 " +
                std::to_string(fc1_rows) + " vs " + std::to_string(D_ff_config) +
                ", vit_merger fc1 " + std::to_string(vm_fc1_rows) + " vs " +
                std::to_string(vm_ff_config) + ") -- that cannot be a zero pad");
        D_ff  = fc1_rows;
        vm_ff = vm_fc1_rows;
    }

    std::string str() const {
        std::ostringstream o;
        auto row = [&](const char* k, auto v) {
            o << std::left << std::setw(28) << k << " = " << v << "\n";
        };
        o << "============ minicpm_v_moe_vision_desc ============\n";
        if (!present) {
            o << "(no vision_model_weight -- text-only)\n";
            return o.str();
        }
        row("depth", depth);
        row("D", D);
        row("num_heads x DH", std::to_string(num_heads) + " x " + std::to_string(DH));
        row("D_ff (tensor/config)", std::to_string(D_ff) + " / " + std::to_string(D_ff_config));
        row("patch_dim", patch_dim());
        row("pos table side", num_patches_per_side);
        row("insert_layer_id", insert_layer_id);
        row("vm_ff (tensor/config)", std::to_string(vm_ff) + " / " + std::to_string(vm_ff_config));
        row("merger_in", merger_in());
        row("out_hidden_size", out_hidden_size);
        row("token_divisor", token_divisor());
        row("ln_eps", ln_eps);
        row("weight_path", weight_path);
        return o.str();
    }

private:
    void _validate() const {
        auto req = [](const char* what, bool ok) {
            if (!ok) throw std::runtime_error(std::string("vision_config: ") + what);
        };
        req("depth is 0", depth > 0);
        req("hidden size is not a multiple of num_heads", num_heads && D % num_heads == 0);
        req("out_hidden_size is 0", out_hidden_size > 0);
        req("num_position_embeddings is not a perfect square",
            (uint64_t)num_patches_per_side * num_patches_per_side == num_position_embeddings);
        req("insert_layer_id is past the last block", insert_layer_id < (int32_t)depth);
    }
};

#endif  // __MINICPM_V_VISION_MODEL_DEF_HPP__
