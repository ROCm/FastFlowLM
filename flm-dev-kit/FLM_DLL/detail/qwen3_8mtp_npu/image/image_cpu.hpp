// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/image_cpu.hpp
/// \brief The vision tower, driven through the vision_ops seam.
///
/// This class is NOT a CPU implementation of the tower. It is the tower's
/// control flow, which is backend-agnostic: every matmul and the attention go
/// through mm_op / attn_op, and swapping the factory for image/npu's moves them
/// to the device without a line changing here. The class name says "cpu"
/// because the default factory is the CPU one.
#ifndef __QWEN3_8MTP_IMAGE_CPU_HPP__
#define __QWEN3_8MTP_IMAGE_CPU_HPP__
#pragma once

#include "image/image_module.hpp"
#include "image/vision_model_def.hpp"
#include "image/vision_ops.hpp"
#include "image/vision_rope.hpp"
#include "image/vision_weights.hpp"

#include <memory>
#include <string>
#include <vector>

namespace qwen38 {

class image_cpu final : public image_module {
public:
    /// \param desc taken by reference: the checkpoint corrects the padded MLP
    ///        width, and everything downstream must see the corrected value.
    /// \param factory which backend the 110 matmuls and the attention use.
    image_cpu(qwen3_8mtp_vision_desc& desc,
              std::unique_ptr<vision::op_factory> factory);

    int tokens_for(image_grid g) const override {
        return desc_.token_count(g.t, g.h, g.w);
    }

    void encode(const float* pixel_values, image_grid g, float* out) override;

    void set_tap(tap_fn fn) override { tap_ = std::move(fn); }

    const char* backend() const override { return backend_.c_str(); }

    const qwen3_8mtp_vision_desc& desc() const { return desc_; }

private:
    /// Grow the scratch to hold L patches. Called once per image; an image
    /// smaller than the last one reuses the buffers.
    void _ensure_scratch(int L);

    void _emit(const char* name, const float* p, int rows, int cols) const {
        if (tap_) tap_(name, p, rows, cols);
    }

    /// One pre-LN block, in place on x_.
    void _block(int idx, int L, const vision::rope_tables& rope);

    qwen3_8mtp_vision_desc&             desc_;
    std::unique_ptr<vision::op_factory> factory_;
    std::unique_ptr<vision::weights>    w_;
    std::string                         backend_;
    tap_fn                              tap_;

    // ---- ops ---------------------------------------------------------------
    /// The patch embedding is a matmul too: pixel_values is already flattened
    /// into [patches][in_channels*Tp*Ps*Ps], so the conv3d is [L][1536] against
    /// [1152][1536]^T. No separate conv kernel is needed, on host or device.
    std::unique_ptr<vision::mm_op> patch_mm_;
    struct block_ops {
        std::unique_ptr<vision::mm_op> qkv, proj, fc1, fc2;
    };
    std::vector<block_ops>           block_ops_;
    std::unique_ptr<vision::mm_op>   merger_fc1_, merger_fc2_;
    std::unique_ptr<vision::attn_op> attn_;
    int                              attn_configured_L_ = 0;

    // ---- scratch -----------------------------------------------------------
    int scratch_L_ = 0;
    std::vector<float> x_;        ///< [L][D]   the residual stream
    std::vector<float> h_;        ///< [L][D]   normed input to a sublayer
    std::vector<float> qkv_;      ///< [L][3D]  the fused projection
    std::vector<float> q_, k_, v_;///< [L][D]   its thirds, split into planes
    std::vector<float> attn_o_;   ///< [L][D]
    std::vector<float> proj_;     ///< [L][D]
    std::vector<float> ff_;       ///< [L][D_ff]
    std::vector<float> pos_;      ///< [L][D]
    std::vector<float> merged_;   ///< [L/4][merger_in]
};

}  // namespace qwen38

#endif  // __QWEN3_8MTP_IMAGE_CPU_HPP__
