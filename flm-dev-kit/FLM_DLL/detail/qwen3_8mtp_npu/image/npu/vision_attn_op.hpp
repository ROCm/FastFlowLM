// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/npu/vision_attn_op.hpp
/// \brief The vision tower's attention on the NPU: one dispatch per block per
///        image, behind `qwen38::vision::attn_op`.
///
/// This is the device half of the seam `image/vision_ops.hpp` exists to
/// protect. It holds no weights -- vision attention has none -- and no cache,
/// because the tower attends within one image and never across calls. What it
/// owns is four staging BOs, and they must come from THIS app: a BO from
/// another app does not fault, the dispatch simply never completes.
///
/// One instance serves all 27 blocks. `image_cpu` makes exactly one
/// `attn_op` and calls it once per block, so an image is 27 dispatches and the
/// sequence is emitted once (the shape does not move between blocks).
///
/// ### What it is NOT
///
/// It is not weight-resident and it does not stream. `vision_attn.xclbin` has
/// no residency at all: k and v are re-read from DDR on every one of the
/// `L_pad/512 * Heads/4` rounds. That makes the cost close to linear in
/// `L_pad` rather than in `L_pad^2`, and it makes a short image a fixed cost.
///
/// ### What it costs
///
/// N15 check 9, against `cpu_attn_op` in `image/vision_ops.cpp` on 16 threads
/// (it parallelises over heads, so 16 is its ceiling, not 32):
///
///     L = 256 (padded to 512)    device 1.62 ms   host  4.48 ms   2.8x
///     L = 1200 (padded to 1536)  device 7.92 ms   host 41.68 ms   5.3x
///
/// i.e. 214 ms per image against 1125 ms, over 27 blocks at the real operating
/// point -- a 639x480 image smart_resizes to a 30x40 grid, so L = 1200.
///
/// **Both arms are compiled at the engine's -O3.** At -O2 the identical host
/// source measures 11.58 and 124.9 ms and those ratios read 6.5x and 13.7x. An
/// earlier estimate of "the device loses at L=256" came from a driver built
/// that way and was wrong by a factor of three in the host's favour; the
/// crossover is in fact below the smallest image this checkpoint produces.
///
/// There is no length threshold and no host fallback inside `run()`. One code
/// path is what makes an end-to-end A/B measure what actually ships; the CPU
/// arm is selected once, at load, by choosing a different factory.
#ifndef __QWEN3_8MTP_NPU_VISION_ATTN_OP_HPP__
#define __QWEN3_8MTP_NPU_VISION_ATTN_OP_HPP__

#include "image/npu/vision_attn_sequence.hpp"
#include "image/npu/vision_bfp16.hpp"
#include "image/vision_ops.hpp"
#include "image/vision_profile.hpp"

#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace q38npu {

class npu_vision_attn_op final : public qwen38::vision::attn_op {
public:
    /// \param a  an app over vision_attn.xclbin
    explicit npu_vision_attn_op(npu_app a,
                                const vision_attn_rtp_addresses& r =
                                    vision_attn_rtp_addresses{})
        : app_(std::move(a)), rtp_(r) {}

    /// Rows an image is rounded up to.
    static constexpr int CHUNK = vision_attn_tiles::ROWS_PER_ROUND;   // 512

    const vision_attn_rtp_addresses& rtp() const { return rtp_; }

    /// The geometry the bitstream is compiled for is checked HERE rather than
    /// at the first dispatch, so a checkpoint the design cannot serve is a
    /// load-time throw with a readable reason instead of a hang.
    ///
    /// `scale` is checked too, and it is not pedantry: `kernel/common.h` bakes
    /// `1/sqrt(72)` in as a `const float`, so a caller passing anything else
    /// would get a silently differently-tempered softmax. Nothing downstream
    /// could tell -- the vision softmax is diffuse enough that a wrong
    /// temperature still scores cos 0.99.
    void configure(int max_L, int heads, int dh, float scale) override {
        if (dh != vision_attn_tiles::DH_BAKED)
            throw std::runtime_error(
                "npu_vision_attn_op: head_dim " + std::to_string(dh) +
                ", but vision_attn.xclbin is compiled for " +
                std::to_string(vision_attn_tiles::DH_BAKED));
        if (heads <= 0 || heads % vision_attn_tiles::HEADS_PER_PASS != 0)
            throw std::runtime_error(
                "npu_vision_attn_op: " + std::to_string(heads) +
                " heads is not a positive multiple of " +
                std::to_string(vision_attn_tiles::HEADS_PER_PASS));
        const float baked = 1.0f / std::sqrt((float)vision_attn_tiles::DH_BAKED);
        if (std::fabs(scale - baked) > 1e-6f)
            throw std::runtime_error(
                "npu_vision_attn_op: attn_scale " + std::to_string(scale) +
                " but the kernel bakes " + std::to_string(baked) +
                " (kernel/common.h's `scale`)");
        if (max_L <= 0)
            throw std::runtime_error("npu_vision_attn_op::configure: max_L <= 0");

        heads_ = heads;
        dh_    = dh;
        D_     = (size_t)heads * (size_t)dh;
        _ensure_bos(_pad(max_L));
    }

    /// Run one image. **One dispatch, whatever L is.**
    ///
    /// \param q,k,v  [L][heads*dh] fp32, head-major within a row
    /// \param L      patches in this image. Any positive length; padded here
    /// \param o      [L][heads*dh] fp32 out
    /// \param head_perm  null for the real run. Non-null scores q head h
    ///        against k/v head `head_perm[h]`, which the sequence implements as
    ///        a different DDR offset on the k and v descriptors. Honoured
    ///        rather than refused so the cross-head control can run against the
    ///        device and not only against the CPU reference.
    void run(const float* q, const float* k, const float* v,
             int L, float* o, const int* head_perm = nullptr) override {
        if (heads_ == 0)
            throw std::runtime_error(
                "npu_vision_attn_op::run before configure()");
        if (L <= 0) return;

        const int L_pad = _pad(L);
        _ensure_bos(L_pad);

        // Rows [L, L_pad) of q, k and v are zeroed on every call, not just at
        // allocation. The key mask makes the k/v tail inert in the SCORES, but
        // the QK mmul still reads it, and a previous (longer) image's rows left
        // there would be multiplied before being masked. Zeroing is one memset
        // on <= 511 rows and removes the question.
        //
        // The q tail is different: those rows are computed, not masked, and
        // their outputs are dropped below. Zero q means every score is 0, the
        // softmax is uniform over the L valid keys and the row is a finite mean
        // of v -- no inf, no NaN to propagate into a neighbouring row's block.
        {
            VPROF(attn_host_us);
            _fill(q_, q, L, L_pad);
            _fill(k_, k, L, L_pad);
            _fill(v_, v, L, L_pad);
            q_.sync_to_device();
            k_.sync_to_device();
            v_.sync_to_device();
        }

        if (!have_seq_ || L_pad != seq_pad_ || L != seq_valid_ ||
            !_same_perm(head_perm)) {
            gen_vision_attn_seq(app_.seq(), L_pad, L, dh_, heads_, rtp_,
                                head_perm);
            seq_pad_   = L_pad;
            seq_valid_ = L;
            _keep_perm(head_perm);
            have_seq_  = true;
        }

        const ert_cmd_state st = VPROF_EXPR(attn_dev_us, app_(o_, q_, k_, v_));
        if (st != ERT_CMD_STATE_COMPLETED) {
            have_seq_ = false;   // do not reuse a sequence that did not land
            throw std::runtime_error(
                "npu_vision_attn_op::run: dispatch state " +
                std::to_string((int)st) + " at L " + std::to_string(L) +
                " (padded " + std::to_string(L_pad) + ", " +
                std::to_string(L_pad / CHUNK) + " rounds x " +
                std::to_string(heads_ / vision_attn_tiles::HEADS_PER_PASS) +
                " passes) -- a timeout is a sequence bug or a stale RTP "
                "address (diff vision_attn_rtp_addresses against "
                "FLM_Xclbin/Qwen3_8/vision_attn/build/rtp_addr.txt)");
        }

        VPROF(attn_host_us);
        o_.sync_from_device();
        widen_f32_par(o, o_.data(), (size_t)L * D_);
    }

    const char* backend() const override { return "npu"; }

private:
    static int _pad(int L) { return (L + CHUNK - 1) / CHUNK * CHUNK; }

    /// Convert `L` rows and zero the tail out to `L_pad`.
    void _fill(buffer<bf16>& dst, const float* src, int L, int L_pad) {
        bf16* d = dst.data();
        narrow_bf16_par(d, src, (size_t)L * D_);
        const size_t tail = ((size_t)L_pad - (size_t)L) * D_;
        if (tail) std::memset(d + (size_t)L * D_, 0, tail * sizeof(bf16));
    }

    /// Grow-only. An image larger than the last one costs one reallocation and
    /// a sequence regeneration it would have paid for anyway.
    void _ensure_bos(int L_pad) {
        if (L_pad <= bo_rows_) return;
        const size_t words = (size_t)L_pad * D_;
        q_ = app_.create_bo_buffer<bf16>(words);
        k_ = app_.create_bo_buffer<bf16>(words);
        v_ = app_.create_bo_buffer<bf16>(words);
        o_ = app_.create_bo_buffer<bf16>(words);
        q_.memset((bf16)0); k_.memset((bf16)0);
        v_.memset((bf16)0); o_.memset((bf16)0);
        q_.sync_to_device(); k_.sync_to_device();
        v_.sync_to_device(); o_.sync_to_device();
        bo_rows_  = L_pad;
        have_seq_ = false;   // the BOs moved; the sequence's handles did not
    }

    bool _same_perm(const int* p) const {
        if (!p) return perm_.empty();
        if (perm_.size() != (size_t)heads_) return false;
        for (int h = 0; h < heads_; h++)
            if (perm_[(size_t)h] != p[h]) return false;
        return true;
    }

    void _keep_perm(const int* p) {
        if (!p) { perm_.clear(); return; }
        perm_.assign(p, p + heads_);
    }

    npu_app                   app_;
    vision_attn_rtp_addresses rtp_;
    buffer<bf16>              q_, k_, v_, o_;
    std::vector<int>          perm_;
    size_t                    D_        = 0;
    int                       heads_    = 0;
    int                       dh_       = 0;
    int                       bo_rows_  = 0;
    int                       seq_pad_  = -1;
    int                       seq_valid_= -1;
    bool                      have_seq_ = false;
};

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_VISION_ATTN_OP_HPP__
