// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/npu/vision_attn_op.hpp
/// \brief The vision tower's full-view attention on the NPU: one dispatch per
///        view per block, behind `mco::vision::attn_op`.
///
/// Ported from Qwen3.8-27B's tower over the same `vision_attn.xclbin`,
/// unchanged. What is different is how it is called. Qwen3.8 attends one image
/// per block; MiniCPM attends one VIEW per dispatch and an image has up to ten,
/// and they are not all the same size -- a 1280x720 image is a 24x44 source
/// view (1056 patches) and six 28x36 slices (1008). The sequence bakes L_pad
/// and L_valid in, and `npu_app::operator()` rebuilds its ELF, module and
/// kernel whenever the sequence moves, so a single app would rebuild twice per
/// block (source -> slice -> next block's source).
///
/// So this op keeps ONE APP PER SHAPE: a small cache keyed on (L_pad, L_valid,
/// segments), each emitting its sequence once. The four q/k/v/o BOs are shared
/// by all of them -- a BO is device-scoped and bound per run
/// (`npu_app::operator()` passes `args.bo()` each call), so only the sequence
/// is per-shape.
///
/// And it BATCHES: `run_views()` puts consecutive views of the same length into
/// one dispatch, one segment each (`gen_vision_attn_seq`'s n_seg). A 720p image
/// is then two dispatches per block -- the source view and the six slices --
/// instead of seven. On Qwen3.8's own numbers a dispatch carries ~0.8 ms of
/// fixed cost (1.62 ms at L 256 vs 7.92 ms at L 1200 fits 0.83 ms + 0.2 ms per
/// 512x512 round), and MiniCPM's post-merge views are all small.
///
/// q/k/v are read with a row stride (`ld`), so the tower hands over the fused
/// [L][3D] qkv projection as it came off the matmul: no split pass.
///
/// Everything else is Qwen3.8's and still load-bearing:
///
///   * rows [L, L_pad) of q, k AND v are zeroed on every call: the key mask
///     covers the k tail in the scores, but the QK mmul still reads it and the
///     v tail is not masked at all (memory: kv-rows-above-cursor-are-not-inert);
///   * head_dim 72, 16 heads and scale 1/sqrt(72) are baked into the kernel and
///     are checked at configure() rather than at the first hang;
///   * head_perm is honoured, for the cross-head control.
#ifndef __MINICPM_OCR_NPU_VISION_ATTN_OP_HPP__
#define __MINICPM_OCR_NPU_VISION_ATTN_OP_HPP__

#include "image/npu/vision_attn_sequence.hpp"
#include "image/npu/vision_bfp16.hpp"
#include "image/npu/vision_bo.hpp"
#include "image/npu/vision_overlay_state.hpp"
#include "image/vision_ops.hpp"
#include "image/vision_profile.hpp"

#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <cmath>
#include <cstring>
#include <map>
#include <tuple>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mco_npu {

class npu_vision_attn_op final : public mco::vision::attn_op {
public:
    /// \param mgr   the manager for vision_attn.xclbin (or vision_overlay.xclbin); apps are
    ///              made from it on demand, one per view shape. Not owned.
    /// \param state the overlay's state when mgr is vision_overlay.xclbin, else null
    explicit npu_vision_attn_op(npu_app_manager* mgr,
                                const vision_attn_rtp_addresses& r = vision_attn_rtp_addresses{},
                                overlay_state* state = nullptr)
        : mgr_(mgr), rtp_(r), state_(state) {
        if (!mgr_) throw std::runtime_error("npu_vision_attn_op: null vision_attn manager");
    }

    static constexpr int CHUNK = vision_attn_tiles::ROWS_PER_ROUND;   // 512

    const vision_attn_rtp_addresses& rtp() const { return rtp_; }
    /// Distinct view shapes seen so far, i.e. apps built. For the verify driver.
    size_t shapes() const { return slots_.size(); }

    void configure(int max_L, int heads, int dh, float scale) override {
        if (dh != vision_attn_tiles::DH_BAKED)
            throw std::runtime_error("npu_vision_attn_op: head_dim " + std::to_string(dh) +
                                     ", but vision_attn.xclbin is compiled for " +
                                     std::to_string(vision_attn_tiles::DH_BAKED));
        if (heads <= 0 || heads % vision_attn_tiles::HEADS_PER_PASS != 0)
            throw std::runtime_error("npu_vision_attn_op: " + std::to_string(heads) +
                                     " heads is not a positive multiple of " +
                                     std::to_string(vision_attn_tiles::HEADS_PER_PASS));
        // kernel/common.h bakes 1/sqrt(72) as a const float. Anything else is a
        // silently re-tempered softmax that still scores cos 0.99.
        const float baked = 1.0f / std::sqrt((float)vision_attn_tiles::DH_BAKED);
        if (std::fabs(scale - baked) > 1e-6f)
            throw std::runtime_error("npu_vision_attn_op: attn_scale " + std::to_string(scale) +
                                     " but the kernel bakes " + std::to_string(baked));
        if (max_L <= 0) throw std::runtime_error("npu_vision_attn_op::configure: max_L <= 0");

        heads_ = heads;
        dh_    = dh;
        D_     = (size_t)heads * (size_t)dh;
        _ensure_bos(_pad(max_L));
    }

    /// Run one view. One dispatch, whatever L is.
    void run(const float* q, const float* k, const float* v,
             int L, float* o, const int* head_perm = nullptr, int ld = 0) override {
        if (heads_ == 0) throw std::runtime_error("npu_vision_attn_op::run before configure()");
        if (L <= 0) return;
        _dispatch(q, k, v, ld > 0 ? (size_t)ld : D_, L, 1, o, head_perm);
    }

    /// Consecutive views of equal length share a dispatch, one segment each.
    void run_views(const float* q, const float* k, const float* v, int ld,
                   const std::vector<int>& lengths, float* o) override {
        if (heads_ == 0) throw std::runtime_error("npu_vision_attn_op::run_views before configure()");
        const size_t S = ld > 0 ? (size_t)ld : D_;
        size_t row = 0;
        for (size_t i = 0; i < lengths.size();) {
            size_t j = i + 1;
            while (j < lengths.size() && lengths[j] == lengths[i]) j++;
            const int n = (int)(j - i);
            _dispatch(q + row * S, k + row * S, v + row * S, S, lengths[i], n, o + row * D_, nullptr);
            row += (size_t)n * (size_t)lengths[i];
            i = j;
        }
    }

    /// Build the sequence (and the ELF behind it) for each run of equal
    /// lengths now, so the first run_views() of a new image size does not.
    void prepare(const std::vector<int>& lengths) override {
        if (heads_ == 0) return;
        // on the overlay, a block's first dispatch follows its qkv matmul, the others this op
        int from = state_ ? vision_overlay::OP_MM : NOT_OVERLAY;
        for (size_t i = 0; i < lengths.size();) {
            size_t j = i + 1;
            while (j < lengths.size() && lengths[j] == lengths[i]) j++;
            const int L = lengths[i], n = (int)(j - i);
            _ensure_bos(_pad(L) * n);
            slot& s = _slot(_pad(L), L, n, from);
            if (!s.have_seq || !s.perm.empty()) {
                _gen_seq(s, _pad(L), L, n, nullptr, from);
                s.perm.clear();
                s.have_seq = true;
                s.app.update_ctrl_seq();
            }
            if (state_) from = vision_overlay::OP_ATTN;
            i = j;
        }
    }

    const char* backend() const override { return "npu"; }

protected:
    int o_cols() const override { return (int)D_; }

private:
    /// One dispatch over n segments of L rows each: segment s's rows are read
    /// at q/k/v + s*L*S (stride S) and land at o + s*L*D.
    void _dispatch(const float* q, const float* k, const float* v, size_t S, int L, int n,
                   float* o, const int* head_perm) {
        const int L_pad = _pad(L);
        const size_t used = (size_t)n * (size_t)L_pad * D_;
        _ensure_bos(L_pad * n);
        {
            VPROF(attn_host_us);
            for (int s = 0; s < n; s++) {
                const size_t src = (size_t)s * (size_t)L * S, dst = (size_t)s * (size_t)L_pad;
                _fill(q_, dst, q + src, S, L, L_pad);
                _fill(k_, dst, k + src, S, L, L_pad);
                _fill(v_, dst, v + src, S, L, L_pad);
            }
            sync_to_device_n(q_, used);
            sync_to_device_n(k_, used);
            sync_to_device_n(v_, used);
        }

        const int from = state_ ? state_->current : NOT_OVERLAY;
        slot& sl = _slot(L_pad, L, n, from);
        if (!sl.have_seq || !_same_perm(sl, head_perm)) {
            _gen_seq(sl, L_pad, L, n, head_perm, from);
            _keep_perm(sl, head_perm);
            sl.have_seq = true;
        }

        auto run_as_is = [&] { return VPROF_EXPR(attn_dev_us, sl.app(o_, q_, k_, v_)); };
        ert_cmd_state st;
        if (state_) {
            const std::string what = "attention L " + std::to_string(L) + " x " + std::to_string(n);
            st = state_->run_checked(what, run_as_is,
                                     [&] { return _run_from_full_switch(L_pad, L, n, head_perm); });
            state_->current = st == ERT_CMD_STATE_COMPLETED ? vision_overlay::OP_ATTN : vision_overlay::OP_NONE;
        } else {
            st = run_as_is();
        }
        if (st != ERT_CMD_STATE_COMPLETED) {
            sl.have_seq = false;
            throw std::runtime_error(
                "npu_vision_attn_op: dispatch state " + std::to_string((int)st) + " at L " +
                std::to_string(L) + " x " + std::to_string(n) + " views (padded " +
                std::to_string(L_pad) + ") -- a timeout is a sequence bug or a stale RTP "
                "address (diff vision_attn_rtp_addresses against "
                "FLM_Xclbin/Qwen3_8/vision_attn/build/rtp_addr.txt)");
        }

        VPROF(attn_host_us);
        sync_from_device_n(o_, used);
        for (int s = 0; s < n; s++)
            widen_f32_par(o + (size_t)s * L * D_, o_.data() + (size_t)s * L_pad * D_, (size_t)L * D_);
    }

    struct slot {
        npu_app          app;
        bool             have_seq = false;
        std::vector<int> perm;
    };

    static int _pad(int L) { return (L + CHUNK - 1) / CHUNK * CHUNK; }

    /// The slot key's `from` off the overlay. On it, `from` is the op the overlay is set up
    /// for when the dispatch starts, and each value gets its own app (see vision_mm_op.hpp).
    static constexpr int NOT_OVERLAY = -1;

    void _gen_seq(slot& s, int L_pad, int L, int n, const int* head_perm, int from) {
        gen_vision_attn_seq(s.app.seq(), L_pad, L, dh_, heads_, rtp_, head_perm, n);
        if (state_) {
            npu_sequence prefix(device_npu2);
            const uint32_t calls = (uint32_t)(n * (heads_ / vision_attn_tiles::HEADS_PER_PASS));
            vision_overlay::gen_prefix(prefix, vision_overlay::OP_ATTN, from, calls);
            vision_overlay::prepend(*s.app.seq(), prefix);
        }
    }

    /// The run again of overlay_state::run_checked: this dispatch from a full switch.
    ert_cmd_state _run_from_full_switch(int L_pad, int L, int n, const int* head_perm) {
        slot& f = _slot(L_pad, L, n, vision_overlay::OP_NONE);
        if (!f.have_seq || !_same_perm(f, head_perm)) {
            _gen_seq(f, L_pad, L, n, head_perm, vision_overlay::OP_NONE);
            _keep_perm(f, head_perm);
            f.have_seq = true;
        }
        return f.app(o_, q_, k_, v_);
    }

    slot& _slot(int L_pad, int L, int n, int from) {
        auto key = std::make_tuple(L_pad, L, n, from);
        auto it = slots_.find(key);
        if (it == slots_.end()) {
            slot s;
            s.app = mgr_->create_app();
            it = slots_.emplace(key, std::move(s)).first;
        }
        return it->second;
    }

    /// Rows [row0, row0+L) of the BO from `src` (stride S), and rows
    /// [row0+L, row0+L_pad) zeroed -- on every call (see the file comment).
    void _fill(buffer<bf16>& dst, size_t row0, const float* src, size_t S, int L, int L_pad) {
        bf16* d = dst.data() + row0 * D_;
        if (S == D_) {
            narrow_bf16_par(d, src, (size_t)L * D_);
        } else {
            const size_t cols = D_;
#pragma omp parallel for schedule(static) num_threads(mco::vision::par_threads())
            for (int r = 0; r < L; r++)
                mco::simd::f32_to_bf16(d + (size_t)r * cols, src + (size_t)r * S, cols);
        }
        const size_t tail = ((size_t)L_pad - (size_t)L) * D_;
        if (tail) std::memset(d + (size_t)L * D_, 0, tail * sizeof(bf16));
    }

    /// Grow-only. A reallocation does not touch any slot's sequence: the
    /// descriptors are offsets and the BOs are bound per run.
    void _ensure_bos(int L_pad) {
        if (L_pad <= bo_rows_) return;
        const size_t words = (size_t)L_pad * D_;
        q_ = mgr_->create_bo_buffer<bf16>(words);
        k_ = mgr_->create_bo_buffer<bf16>(words);
        v_ = mgr_->create_bo_buffer<bf16>(words);
        o_ = mgr_->create_bo_buffer<bf16>(words);
        q_.memset((bf16)0); k_.memset((bf16)0);
        v_.memset((bf16)0); o_.memset((bf16)0);
        q_.sync_to_device(); k_.sync_to_device();
        v_.sync_to_device(); o_.sync_to_device();
        bo_rows_ = L_pad;
    }

    bool _same_perm(const slot& s, const int* p) const {
        if (!p) return s.perm.empty();
        if (s.perm.size() != (size_t)heads_) return false;
        for (int h = 0; h < heads_; h++)
            if (s.perm[(size_t)h] != p[h]) return false;
        return true;
    }

    void _keep_perm(slot& s, const int* p) {
        if (!p) { s.perm.clear(); return; }
        s.perm.assign(p, p + heads_);
    }

    npu_app_manager*                    mgr_ = nullptr;
    vision_attn_rtp_addresses           rtp_;
    std::map<std::tuple<int, int, int, int>, slot> slots_;
    overlay_state* state_ = nullptr;
    buffer<bf16>                        q_, k_, v_, o_;
    size_t                              D_       = 0;
    int                                 heads_   = 0;
    int                                 dh_      = 0;
    int                                 bo_rows_ = 0;
};

} // namespace mco_npu

#endif  // __MINICPM_OCR_NPU_VISION_ATTN_OP_HPP__
