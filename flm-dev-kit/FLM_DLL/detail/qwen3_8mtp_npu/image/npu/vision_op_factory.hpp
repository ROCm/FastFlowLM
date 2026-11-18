// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/npu/vision_op_factory.hpp
/// \brief Which backend each of the tower's ops gets.
///
/// This is the whole of phase 2's wiring. `image_cpu.cpp` asks a factory for
/// its 111 matmuls and its one attention and never learns what it got; swapping
/// this in for `cpu_op_factory` at `qwen3_8mtp_npu.cpp`'s module 4 is the only
/// edit outside this folder.
///
/// ### Both halves, independently
///
/// `make_attn` always returns the device op -- the attention xclbin is required
/// to construct this factory at all. `make_mm` returns the device op when the
/// two `vision_mm` arenas are present and `cpu_mm_op` when they are not.
///
/// **The engine no longer reaches that second case.** Module 4 resolves all
/// three vision xclbins or throws naming the missing one, and always calls
/// `enable_mm`, so a shipping tower is "npu" or it does not start. The
/// half-configured factory exists for a check that scores the attention op
/// against `cpu_mm_op`-driven surroundings and would otherwise have to stage
/// two more bitstreams it tests nothing in.
///
/// `name()` says which halves are live, because `image_cpu::backend()` prints
/// it: **"npu"** when the matmuls are on the device too, **"npu-attn"** when
/// only the attention is. No one should be able to read a half-ported tower as
/// a full one from a log line.
///
/// ### The arenas belong here
///
/// The 111 matmul ops share two x/y staging BOs, one pair per design, and those
/// have to outlive every op. The factory owns them: `image_cpu` holds its ops
/// for its own lifetime and the model holds the factory, so "the arenas outlive
/// the ops" is a statement about this object's lifetime and nothing subtler.
#ifndef __QWEN3_8MTP_NPU_VISION_OP_FACTORY_HPP__
#define __QWEN3_8MTP_NPU_VISION_OP_FACTORY_HPP__

#include "image/npu/vision_attn_op.hpp"
#include "image/npu/vision_mm_op.hpp"
#include "image/vision_ops.hpp"

#include "npu_utils/npu_utils.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace q38npu {

class vision_op_factory final : public qwen38::vision::op_factory {
public:
    /// \param attn_mgr  the manager for vision_attn.xclbin. Not owned; it
    ///        belongs to the `npu_xclbin_manager` the model holds, which
    ///        outlives the tower.
    /// \param rtp  the per-build attention RTP address, overridable so a driver
    ///        can pass one read out of `build/rtp_addr.txt` rather than relying
    ///        on the header's baked default.
    explicit vision_op_factory(npu_app_manager* attn_mgr,
                               const vision_attn_rtp_addresses& rtp =
                                   vision_attn_rtp_addresses{})
        : attn_mgr_(attn_mgr), rtp_(rtp) {
        if (!attn_mgr_)
            throw std::runtime_error(
                "vision_op_factory: null vision_attn app manager -- construct "
                "cpu_op_factory instead when the xclbin is absent, so the "
                "off-arm of the A/B is the real host path "
                "(memory: ab-off-arm-is-not-the-host)");
    }

    /// Put the matmuls on the device as well.
    ///
    /// Both managers are required together: design A serves every shape but
    /// `merger.fc2`, and a tower that ran 110 matmuls on the device and threw
    /// on the 111th would be worse than one that ran none. The caller (module 4)
    /// resolves both xclbins or neither.
    ///
    /// \param mm_a  manager for vision_mm_a.xclbin (m=64,k=384,n=48)
    /// \param mm_b  manager for vision_mm_b.xclbin (m=32,k=384,n=128)
    /// \param mm_rtp  the two per-build RTP addresses. A stale one does not
    ///        fault; it hangs in `qds_device::wait()`
    ///        (memory: rtp-addresses-are-per-build)
    void enable_mm(npu_app_manager* mm_a, npu_app_manager* mm_b,
                   const vision_mm_rtp_addresses& mm_rtp =
                       vision_mm_rtp_addresses{}) {
        if (!mm_a || !mm_b)
            throw std::runtime_error(
                "vision_op_factory::enable_mm: both vision_mm designs are "
                "required (A serves every shape but merger.fc2, whose N of "
                "5120 is not a multiple of A's tile_N of 384) -- leave the "
                "matmuls on the host instead by not calling this");
        arena_a_ = std::make_unique<vision_mm_arena>(
                       mm_a, vision_mm_tiles::A, mm_rtp.a);
        arena_b_ = std::make_unique<vision_mm_arena>(
                       mm_b, vision_mm_tiles::B, mm_rtp.b);
    }

    bool mm_on_device() const { return arena_a_ && arena_b_; }

    /// One app per op, and one weight BO per op. See `vision_mm_op.hpp` --
    /// sharing an app across the 111 would rebuild its ELF on every dispatch.
    ///
    /// `tag` is carried into the op so a dispatch failure can name which of the
    /// 111 it was. It is not used to select anything: the design comes from the
    /// shape at `bind()`, so a renamed tensor cannot silently pick the wrong
    /// bitstream.
    std::unique_ptr<qwen38::vision::mm_op> make_mm(const char* tag) override {
        if (!mm_on_device())
            return std::make_unique<qwen38::vision::cpu_mm_op>();
        return std::make_unique<npu_vision_mm_op>(
                   arena_a_.get(), arena_b_.get(), tag ? tag : "");
    }

    /// One app per op. `image_cpu` asks for exactly one, once, in its
    /// constructor -- an app carries its own instruction sequence and its own
    /// BOs, so a second one would be a second copy of both.
    std::unique_ptr<qwen38::vision::attn_op> make_attn(const char*) override {
        return std::make_unique<npu_vision_attn_op>(attn_mgr_->create_app(),
                                                    rtp_);
    }

    const char* name() const override {
        return mm_on_device() ? "npu" : "npu-attn";
    }

private:
    npu_app_manager*                 attn_mgr_ = nullptr;
    vision_attn_rtp_addresses        rtp_;
    std::unique_ptr<vision_mm_arena> arena_a_, arena_b_;
};

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_VISION_OP_FACTORY_HPP__
