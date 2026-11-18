// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/vision_tower.hpp
/// \brief MiniCPM-V-4.7's SigLIP tower, driven through the vision_ops seam.
///
/// Qwen3.8's `image_cpu`, renamed because it never was a CPU implementation:
/// this is the tower's control flow, and every matmul and the full-view
/// attention go through `mm_op` / `attn_op`, so swapping the factory moves them
/// to the device without a line changing here.
///
///   patch_embed (matmul, K 588)  + bucketized position table
///   blocks 0..6                  on every patch
///   vit_merger                   LN -> qkv -> 2x2 WINDOWED attention (host) ->
///                                proj -> +residual -> 2x2 merge -> LN(4608) ->
///                                fc1 (gelu_tanh) -> fc2 -> + mean-of-4
///   blocks 7..26                 on a quarter of the patches
///   post_layernorm, 2x2 merge, LN(4608), fc1 (erf gelu, host epilogue), fc2
#ifndef __MINICPM_V_VISION_TOWER_HPP__
#define __MINICPM_V_VISION_TOWER_HPP__
#pragma once

#include "image/image_module.hpp"
#include "image/vision_model_def.hpp"
#include "image/vision_ops.hpp"
#include "image/vision_weights.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mcvm {

class vision_tower final : public image_module {
public:
    /// \param desc taken by reference: the checkpoint corrects the padded MLP
    ///        widths, and everything downstream must see the corrected values.
    /// \param factory which backend the 115 matmuls and the attention use.
    vision_tower(minicpm_v_moe_vision_desc& desc, std::unique_ptr<vision::op_factory> factory);

    int tokens_for(view_grid v) const override { return desc_.tokens_for_view(v.h, v.w); }

    void encode(const float* pixel_values, const std::vector<view_grid>& views,
                float* out) override;

    void set_tap(tap_fn fn) override { tap_ = std::move(fn); }
    const char* backend() const override { return backend_.c_str(); }
    const minicpm_v_moe_vision_desc& desc() const { return desc_; }

private:
    void _ensure_scratch(int L);
    void _emit(const char* name, const float* p, int rows, int cols) const {
        if (tap_) tap_(name, p, rows, cols);
    }

    /// Build every op's device state for this call's shapes, in parallel.
    void _prepare(const std::vector<view_grid>& views, int L);
    /// One pre-LN encoder block, in place on x_ over L rows of `views`.
    void _block(int idx, const std::vector<view_grid>& views, int L);
    /// The vit_merger: x_ goes from L rows to L/4, views halve.
    void _vit_merger(std::vector<view_grid>& views, int L);

    minicpm_v_moe_vision_desc&              desc_;
    std::unique_ptr<vision::op_factory> factory_;
    std::unique_ptr<vision::weights>    w_;
    std::string                         backend_;
    tap_fn                              tap_;

    // ---- ops ---------------------------------------------------------------
    std::unique_ptr<vision::mm_op> patch_mm_;
    struct block_ops { std::unique_ptr<vision::mm_op> qkv, proj, fc1, fc2; };
    std::vector<block_ops>           block_ops_;
    block_ops                        vm_ops_;
    std::unique_ptr<vision::mm_op>   merger_fc1_, merger_fc2_;
    std::unique_ptr<vision::attn_op> attn_;
    int                              attn_configured_L_ = 0;

    // ---- scratch, sized for the largest call so far ---------------------------
    int scratch_L_ = 0;
    std::vector<float> x_;        ///< [L][D]   the residual stream
    std::vector<float> h_;        ///< [L][D]   normed input to a sublayer
    std::vector<float> qkv_;      ///< [L][3D], read in place by both attentions
    std::vector<float> attn_o_;   ///< [L][D]
    std::vector<float> proj_;     ///< [L][D]
    std::vector<float> ff_;       ///< [L][D_ff]; reused as [L/4][vm_ff]
    std::vector<float> merged_;   ///< [L/4][4D]
    std::vector<float> merged_n_; ///< [L/4][4D]
    std::vector<float> mean_;     ///< [L/4][D]
};

}  // namespace mcvm

#endif  // __MINICPM_V_VISION_TOWER_HPP__
