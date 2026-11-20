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

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mcv {

class vision_tower final : public image_module {
public:
    /// \param desc taken by reference: the checkpoint corrects the padded MLP
    ///        widths, and everything downstream must see the corrected values.
    /// \param factory which backend the 115 matmuls and the attention use.
    vision_tower(minicpm_v_vision_desc& desc, std::unique_ptr<vision::op_factory> factory);

    int tokens_for(view_grid v) const override { return desc_.tokens_for_view(v.h, v.w); }

    void encode(const float* pixel_values, const std::vector<view_grid>& views,
                float* out) override;

    void set_tap(tap_fn fn) override { tap_ = std::move(fn); }
    const char* backend() const override { return backend_.c_str(); }
    const minicpm_v_vision_desc& desc() const { return desc_; }

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

    // ---- the streamed encoder (backends whose ops are streamable) -----------
    // Same arithmetic as _block, row for row, with the row-independent stretch of
    // each block (proj, fc1, fc2 and the next block's norm + qkv) run as chunks of
    // rows a few at a time: while the device runs one chunk the host works on
    // another. See mm_op's streaming interface.
    struct stream_step {
        vision::mm_op* op;
        std::function<void(int slot, int r0, int n)> load;    ///< fill the slot's x
        std::function<void(int slot, int r0, int n)> store;   ///< consume the slot's y
    };
    bool _can_stream() const;
    int  _chunk_rows(int rows) const;
    void _stream_prepare(int rows, vision::mm_op* const* ops, int n_ops);
    void _run_stream(const std::vector<stream_step>& steps, int rows,
                     const std::function<void(int)>& done = {});
    std::function<void(int)> _attn_stager(const std::vector<view_grid>& views);
    std::function<void(int)> _per_view(const std::vector<view_grid>& views,
                                       std::function<void(const view_grid&, size_t)> fn);
    std::function<void(int)> _vm_wattn_hook(const std::vector<view_grid>& views);
    stream_step _head_step(int block);
    void _tail_steps(int block, std::vector<stream_step>& out);
    stream_step _vm_qkv_step();
    void _vit_merger_streamed(std::vector<view_grid>& views, int L, int next);
    void _encode_blocks_streamed(std::vector<view_grid>& views, int& L);

    minicpm_v_vision_desc&              desc_;
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
    std::vector<uint16_t> qkv16_;     ///< [L][3D] bf16, the streamed path's qkv
    std::vector<uint16_t> attn_o16_;  ///< [L][D] bf16, its attention output
    bool streamed_ = false;       ///< this encode takes the streamed path
};

}  // namespace mcv

#endif  // __MINICPM_V_VISION_TOWER_HPP__
