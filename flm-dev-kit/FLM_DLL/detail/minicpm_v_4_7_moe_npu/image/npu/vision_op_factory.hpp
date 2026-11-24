// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/npu/vision_op_factory.hpp
/// \brief Which backend each of the tower's ops gets.
///
/// Ported from Qwen3.8-27B's tower. `vision_tower.cpp` asks a factory for its
/// 115 matmuls and its one attention op and never learns what it got.
///
/// `make_attn` always returns the device op -- the attention xclbin is required
/// to construct this factory at all. `make_mm` returns the device op once
/// `enable_mm` has been given both `vision_mm` designs, and `cpu_mm_op`
/// otherwise; the engine always enables both, so a shipping tower is "npu" or it
/// does not start. The half-configured factory exists for the verify driver.
///
/// `name()` says which halves are live: "npu" or "npu-attn", so a half-ported
/// tower cannot be read as a full one from a log line.
///
/// The arenas -- the x/y staging the matmul ops of one design share -- belong
/// here because they must outlive every op, and the engine holds the factory
/// for the tower's lifetime.
#ifndef __MINICPM_V_4_7_NPU_VISION_OP_FACTORY_HPP__
#define __MINICPM_V_4_7_NPU_VISION_OP_FACTORY_HPP__

#include "image/npu/vision_attn_op.hpp"
#include "image/npu/vision_mm_op.hpp"
#include "image/vision_ops.hpp"

#include "npu_utils/npu_utils.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace mcvm_npu {

class vision_op_factory final : public mcvm::vision::op_factory {
public:
    /// \param attn_mgr manager for vision_attn.xclbin, owned by the model's
    ///        npu_xclbin_manager, which outlives the tower.
    explicit vision_op_factory(npu_app_manager* attn_mgr,
                               const vision_attn_rtp_addresses& rtp = vision_attn_rtp_addresses{})
        : attn_mgr_(attn_mgr), rtp_(rtp) {
        if (!attn_mgr_)
            throw std::runtime_error(
                "vision_op_factory: null vision_attn app manager -- construct "
                "cpu_op_factory instead when the xclbin is absent, so the off-arm "
                "of an A/B is the real host path");
    }

    /// Put the matmuls on the device too. Both designs together: A serves
    /// every shape but the projector's fc2 (N 1024), which only B can.
    void enable_mm(npu_app_manager* mm_a, npu_app_manager* mm_b,
                   const vision_mm_rtp_addresses& mm_rtp = vision_mm_rtp_addresses{}) {
        if (!mm_a || !mm_b)
            throw std::runtime_error(
                "vision_op_factory::enable_mm: both vision_mm designs are required "
                "(A serves every shape but merger.fc2, whose N of 1024 is not a "
                "multiple of A's tile_N of 384)");
        arena_a_ = std::make_unique<vision_mm_arena>(mm_a, vision_mm_tiles::A, mm_rtp.a);
        arena_b_ = std::make_unique<vision_mm_arena>(mm_b, vision_mm_tiles::B, mm_rtp.b);
    }

    bool mm_on_device() const { return arena_a_ && arena_b_; }

    std::unique_ptr<mcvm::vision::mm_op> make_mm(const char* tag) override {
        if (!mm_on_device()) return std::make_unique<mcvm::vision::cpu_mm_op>();
        return std::make_unique<npu_vision_mm_op>(arena_a_.get(), arena_b_.get(), tag ? tag : "");
    }

    std::unique_ptr<mcvm::vision::attn_op> make_attn(const char*) override {
        return std::make_unique<npu_vision_attn_op>(attn_mgr_, rtp_);
    }

    const char* name() const override { return mm_on_device() ? "npu" : "npu-attn"; }

private:
    npu_app_manager*                 attn_mgr_ = nullptr;
    vision_attn_rtp_addresses        rtp_;
    std::unique_ptr<vision_mm_arena> arena_a_, arena_b_;
};

} // namespace mcvm_npu

#endif  // __MINICPM_V_4_7_NPU_VISION_OP_FACTORY_HPP__
