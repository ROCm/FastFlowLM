// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/npu/vision_attn_op.hpp
/// \brief The vision tower's full-view attention on the NPU: one dispatch per
///        view per block, behind `mcv::vision::attn_op`.
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
///     v tail is not masked at all;
///   * head_dim 72, 16 heads and scale 1/sqrt(72) are baked into the kernel and
///     are checked at configure() rather than at the first hang;
///   * head_perm is honoured, for the cross-head control.
#ifndef __MINICPM_V_4_7_NPU_VISION_ATTN_OP_HPP__
#define __MINICPM_V_4_7_NPU_VISION_ATTN_OP_HPP__

#include "image/npu/vision_attn_reconfig.hpp"
#include "image/npu/vision_attn_sequence.hpp"
#include "image/npu/vision_bfp16.hpp"
#include "image/npu/vision_bo.hpp"
#include "image/vision_ops.hpp"
#include "image/vision_par.hpp"
#include "image/vision_profile.hpp"

#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <cstring>
#include <map>
#include <tuple>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mcv_npu {

class npu_vision_attn_op final : public mcv::vision::attn_op {
public:
    /// \param mgr  the manager for vision_attn.xclbin; apps are made from it on
    ///             demand, one per view shape. Not owned.
    /// \param sc   optional (FLM_MINICPM_V_VISION_SINGLE_CTX): the in-stream reconfiguration the dispatches may run through
    explicit npu_vision_attn_op(npu_app_manager* mgr,
                                const vision_attn_rtp_addresses& r = vision_attn_rtp_addresses{},
                                vision_attn_reconfig* sc = nullptr)
        : mgr_(mgr), rtp_(r) {
        if (!mgr_) throw std::runtime_error("npu_vision_attn_op: null vision_attn manager");
#if FLM_MINICPM_RECONFIG_SUPPORTED
        sc_ = sc;
#else
        (void)sc;
#endif
    }

#if FLM_MINICPM_RECONFIG_SUPPORTED
    /// Which hw_context path a dispatch takes. AUTO (production): the round trip when sc_->usable(), else the vision_attn context.
    /// The load-time self-test pins the two arms to compare them.
    enum class path_mode { AUTO, CONTEXT_ONLY, SINGLE_CONTEXT_ONLY };
    void set_path_mode(path_mode m) { mode_ = m; }
#endif

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
        _run_groups(lengths, [&](int L, int n) {
            _dispatch(q + row * S, k + row * S, v + row * S, S, L, n, o + row * D_, nullptr);
            row += (size_t)n * (size_t)L;
        });
    }

    /// bf16 in, bf16 out (see attn_op::run_views_bf16): the qkv rows are copied
    /// into the q/k/v BOs as they are, the output rows copied out as they are.
    bool supports_bf16() const override { return heads_ != 0; }
    void run_views_bf16(const uint16_t* qkv, int ld, const std::vector<int>& lengths,
                        uint16_t* o) override {
        if (heads_ == 0) throw std::runtime_error("npu_vision_attn_op::run_views_bf16 before configure()");
        const size_t S = ld > 0 ? (size_t)ld : D_;
        size_t row = 0;
        const int sub = _sub_views();
        _run_groups(lengths, [&](int L, int n) {
            if (sub > 0 && n > sub) _dispatch16_piped(qkv + row * S, S, L, n, sub, o + row * D_);
            else                    _dispatch16(qkv + row * S, S, L, n, o + row * D_);
            row += (size_t)n * (size_t)L;
        });
    }

    /// The first group's first sub-dispatch, when run_views_bf16 pipes it.
    int stage_rows(const std::vector<int>& lengths) const override {
        int L = 0, sub = 0;
        return _first_sub(lengths, L, sub) ? L * sub : 0;
    }

    void stage_bf16(const uint16_t* qkv, int ld, const std::vector<int>& lengths) override {
        int L = 0, sub = 0;
        if (heads_ == 0 || !_first_sub(lengths, L, sub)) return;
        const size_t S = ld > 0 ? (size_t)ld : D_;
        _ensure_bos(_pad(L) * sub);
        _fill16(q_, k_, v_, qkv, S, L, sub);
        staged_ = {qkv, S, L, sub};
    }

    /// Build the sequence (and the ELF behind it) for each run of equal
    /// lengths now, so the first run_views() of a new image size does not.
    void prepare(const std::vector<int>& lengths) override {
        if (heads_ == 0) return;
        for (size_t i = 0; i < lengths.size();) {
            size_t j = i + 1;
            while (j < lengths.size() && lengths[j] == lengths[i]) j++;
            const int L = lengths[i], n = (int)(j - i);
            _ensure_bos(_pad(L) * n);
            slot& s = _slot(_pad(L), L, n);
            if (!s.have_seq || !s.perm.empty()) {
                gen_vision_attn_seq(s.app.seq(), _pad(L), L, dh_, heads_, rtp_, nullptr, n);
                s.perm.clear();
                s.have_seq = true;
                s.app.update_ctrl_seq();
            }
#if FLM_MINICPM_RECONFIG_SUPPORTED
            if (sc_ && sc_->built && sc_->validated) _prepare_sc(s, _pad(L), L, n, nullptr);
#endif
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

        slot& sl = _slot(L_pad, L, n);
        if (!sl.have_seq || !_same_perm(sl, head_perm)) {
            gen_vision_attn_seq(sl.app.seq(), L_pad, L, dh_, heads_, rtp_, head_perm, n);
            _keep_perm(sl, head_perm);
            sl.have_seq = true;
        }

        const ert_cmd_state st = VPROF_EXPR(attn_dev_us, _launch(sl, L_pad, L, n, head_perm));
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

    /// `_dispatch` for bf16 data: segment s's rows are read at qkv + s*L*S (q at
    /// column 0, k at D, v at 2D of each stride-S row) and land at o + s*L*D.
    void _dispatch16(const uint16_t* qkv, size_t S, int L, int n, uint16_t* o) {
        const int L_pad = _pad(L);
        _ensure_bos(L_pad * n);
        _fill16(q_, k_, v_, qkv, S, L, n);
        _launch16(L, n, q_, k_, v_, o_);
        _drain16(o_, L, n, o);
    }

    /// `_dispatch16` over n views in sub-dispatches of `sub` views, on two BO sets: while the
    /// device runs sub-dispatch j the host drains j-1 and fills j+1, so the copies (and their
    /// syncs) hide behind the device instead of idling it. Each view is its own segment either
    /// way, so the result is the same bits; the reconfiguration enters once and leaves after
    /// the last sub-dispatch, like one dispatch.
    void _dispatch16_piped(const uint16_t* qkv, size_t S, int L, int n, int sub, uint16_t* o) {
        const int L_pad = _pad(L);
        _ensure_bos(L_pad * sub);
        _ensure_alt(L_pad * sub);
        buffer<bf16>* Q[2] = {&q_, &aq_};
        buffer<bf16>* K[2] = {&k_, &ak_};
        buffer<bf16>* V[2] = {&v_, &av_};
        buffer<bf16>* O[2] = {&o_, &ao_};
        const int nsub = (n + sub - 1) / sub;
        auto cnt = [&](int j) { return std::min(sub, n - j * sub); };
        auto src = [&](int j) { return qkv + (size_t)j * sub * (size_t)L * S; };
        auto dst = [&](int j) { return o + (size_t)j * sub * (size_t)L * D_; };
#if FLM_MINICPM_RECONFIG_SUPPORTED
        const bool last_of_call = grp_last_;
#endif
        std::future<void> run;
        // stage_bf16 may have filled set 0 already
        const staged st0 = staged_;
        staged_ = {};
        try {
            if (!(st0.qkv == qkv && st0.S == S && st0.L == L && st0.n == cnt(0)))
                _fill16(*Q[0], *K[0], *V[0], src(0), S, L, cnt(0));
            for (int j = 0; j < nsub; j++) {
                const int b = j & 1;
#if FLM_MINICPM_RECONFIG_SUPPORTED
                grp_last_ = last_of_call && j + 1 == nsub;
#endif
                run = std::async(std::launch::async, [&, j, b] {
                    _launch16(L, cnt(j), *Q[b], *K[b], *V[b], *O[b]);
                });
                if (j > 0)        _drain16(*O[b ^ 1], L, cnt(j - 1), dst(j - 1));
                if (j + 1 < nsub) _fill16(*Q[b ^ 1], *K[b ^ 1], *V[b ^ 1], src(j + 1), S, L, cnt(j + 1));
                { VPROF(wait_us); run.get(); }
            }
            _drain16(*O[(nsub - 1) & 1], L, cnt(nsub - 1), dst(nsub - 1));
        } catch (...) {
            if (run.valid()) { try { run.wait(); } catch (...) {} }
#if FLM_MINICPM_RECONFIG_SUPPORTED
            grp_last_ = last_of_call;
#endif
            throw;
        }
#if FLM_MINICPM_RECONFIG_SUPPORTED
        grp_last_ = last_of_call;
#endif
    }

    /// What stage_bf16 left in the set-0 BOs; consumed (cleared) by the next piped run.
    struct staged { const uint16_t* qkv = nullptr; size_t S = 0; int L = 0, n = 0; };
    staged staged_;

    /// The first group of `lengths` (L rows, its view count) when run_views_bf16 pipes it, and its sub size.
    static bool _first_sub(const std::vector<int>& lengths, int& L, int& sub) {
        sub = _sub_views();
        if (lengths.empty() || sub <= 0) return false;
        L = lengths[0];
        int n = 1;
        while (n < (int)lengths.size() && lengths[(size_t)n] == L) n++;
        return n > sub;
    }

    /// Sub-dispatch size in views for run_views_bf16 (FLM_MINICPM_V_VISION_ATTN_SUB, default 15; 0 = one dispatch
    /// per group). 15 hides the next fill and the previous drain behind the device at N=120 (-0.4 s, bit-identical).
    static int _sub_views() {
        static const int v = [] {
            if (const char* e = std::getenv("FLM_MINICPM_V_VISION_ATTN_SUB")) return std::max(0, std::atoi(e));
            return 15;
        }();
        return v;
    }

    /// n views of L rows from qkv (stride S) into q/k/v, each segment padded to L_pad with zeros, then synced.
    void _fill16(buffer<bf16>& qb, buffer<bf16>& kb, buffer<bf16>& vb,
                 const uint16_t* qkv, size_t S, int L, int n) {
        VPROF(attn_host_us);
        const int L_pad = _pad(L);
        const size_t D = D_;
        for (int s = 0; s < n; s++) {
            const uint16_t* src = qkv + (size_t)s * (size_t)L * S;
            const size_t    dst = (size_t)s * (size_t)L_pad;
            uint16_t* qd = reinterpret_cast<uint16_t*>(qb.data()) + dst * D;
            uint16_t* kd = reinterpret_cast<uint16_t*>(kb.data()) + dst * D;
            uint16_t* vd = reinterpret_cast<uint16_t*>(vb.data()) + dst * D;
#pragma omp parallel for schedule(static) num_threads(mcv::vision::par_threads())
            for (int r = 0; r < L; r++) {
                const uint16_t* row = src + (size_t)r * S;
                std::memcpy(qd + (size_t)r * D, row,         D * sizeof(uint16_t));
                std::memcpy(kd + (size_t)r * D, row + D,     D * sizeof(uint16_t));
                std::memcpy(vd + (size_t)r * D, row + 2 * D, D * sizeof(uint16_t));
            }
            const size_t tail = ((size_t)L_pad - (size_t)L) * D;
            if (tail) {
                std::memset(qd + (size_t)L * D, 0, tail * sizeof(uint16_t));
                std::memset(kd + (size_t)L * D, 0, tail * sizeof(uint16_t));
                std::memset(vd + (size_t)L * D, 0, tail * sizeof(uint16_t));
            }
        }
        const size_t used = (size_t)n * (size_t)L_pad * D_;
        sync_to_device_n(qb, used);
        sync_to_device_n(kb, used);
        sync_to_device_n(vb, used);
    }

    /// The device part of one bf16 dispatch of n views of L rows on the given BOs.
    void _launch16(int L, int n, buffer<bf16>& qb, buffer<bf16>& kb, buffer<bf16>& vb, buffer<bf16>& ob) {
        const int L_pad = _pad(L);
        slot& sl = _slot(L_pad, L, n);
        if (!sl.have_seq || !_same_perm(sl, nullptr)) {
            gen_vision_attn_seq(sl.app.seq(), L_pad, L, dh_, heads_, rtp_, nullptr, n);
            _keep_perm(sl, nullptr);
            sl.have_seq = true;
        }
        const ert_cmd_state st = VPROF_EXPR(attn_dev_us, _launch(sl, L_pad, L, n, nullptr, &qb, &kb, &vb, &ob));
        if (st != ERT_CMD_STATE_COMPLETED) {
            sl.have_seq = false;
            throw std::runtime_error(
                "npu_vision_attn_op: dispatch state " + std::to_string((int)st) + " at L " +
                std::to_string(L) + " x " + std::to_string(n) + " views (padded " +
                std::to_string(L_pad) + ")");
        }
    }

    /// n views' output rows out of ob (L_pad stride) into dst (L stride).
    void _drain16(buffer<bf16>& ob, int L, int n, uint16_t* o) {
        VPROF(attn_host_us);
        const int L_pad = _pad(L);
        sync_from_device_n(ob, (size_t)n * (size_t)L_pad * D_);
        for (int s = 0; s < n; s++) {
            const uint16_t* od = reinterpret_cast<const uint16_t*>(ob.data()) + (size_t)s * L_pad * D_;
            uint16_t*       dd = o + (size_t)s * L * D_;
            const size_t    total = (size_t)L * D_;
            constexpr size_t CH = 1u << 16;
            const int64_t nchunk = (int64_t)((total + CH - 1) / CH);
#pragma omp parallel for schedule(static) num_threads(mcv::vision::par_threads())
            for (int64_t c = 0; c < nchunk; c++) {
                const size_t i0 = (size_t)c * CH;
                std::memcpy(dd + i0, od + i0, std::min(total - i0, CH) * sizeof(uint16_t));
            }
        }
    }

    struct slot {
        npu_app          app;
        bool             have_seq = false;
        std::vector<int> perm;
#if FLM_MINICPM_RECONFIG_SUPPORTED
        npu_app          app_sc;            //!< the same sequence for vision_mm_a's hw_context (single-context path)
        bool             sc_created = false;
        bool             sc_have_seq = false;
        std::vector<int> sc_perm;
#endif
    };

#if FLM_MINICPM_RECONFIG_SUPPORTED
    /// the slot's attention sequence for the vision_mm_a context (built once per shape, like the vision_attn one)
    void _prepare_sc(slot& s, int L_pad, int L, int n, const int* head_perm) {
        if (!s.sc_created) { s.app_sc = sc_->home->create_app(); s.sc_created = true; }
        const bool same = head_perm == nullptr ? s.sc_perm.empty()
                                               : (s.sc_perm.size() == (size_t)heads_ && std::equal(s.sc_perm.begin(), s.sc_perm.end(), head_perm));
        if (!s.sc_have_seq || !same) {
            gen_vision_attn_seq(s.app_sc.seq(), L_pad, L, dh_, heads_, rtp_, head_perm, n);
            if (head_perm) s.sc_perm.assign(head_perm, head_perm + heads_); else s.sc_perm.clear();
            s.sc_have_seq = true;
            s.app_sc.update_ctrl_seq();
        }
    }

    /// One submission for one dispatch. The array is reconfigured to vision_attn first unless a previous dispatch of this call left it
    /// there (armed_), and back to vision_mm_a after the last dispatch of the call (grp_last_): a call with several groups (a 720p image
    /// attends its source view and its slices separately) enters and leaves the attention configuration once, like the two-context path
    /// switches once. Returns false when the submission failed: the single-context path is then off for good (armed_ cleared) and the caller
    /// runs this dispatch through vision_attn's own context, which also makes the firmware load that PDI, so the next vision_mm_a dispatch
    /// reloads design A's instead of trusting a half-reconfigured array. Redoing the dispatch is harmless whichever part of the submission
    /// failed, and so is a failed leave: the attention already ran and is simply run again.
    bool _launch_single_ctx(slot& sl, int L_pad, int L, int n, const int* head_perm,
                            buffer<bf16>& q, buffer<bf16>& k, buffer<bf16>& v, buffer<bf16>& o) {
        try {
            _prepare_sc(sl, L_pad, L, n, head_perm);
            xrt::runlist rl = sc_->home->create_runlist();
            const bool enter = !armed_;
            if (enter) {
#ifdef FLM_MINICPM_RECONFIG_FAULT_INJECT
                if (sc_->validated && sc_->fault_at > 0 && ++sc_->enters == sc_->fault_at) sc_->to_attn_unquiesced.add_to(rl);
                else
#endif
                sc_->to_attn.add_to(rl);
            }
            xrt::run attn_run = sl.app_sc.create_run(o, q, k, v);
            rl.add(attn_run);
            if (grp_last_) sc_->to_mm.add_to(rl);
            rl.execute();
            // the driver ends a hung command by itself (ERT timeout -> command_error); this bound is a backstop for a wait that never returns
            if (rl.wait(std::chrono::milliseconds(30000)) == std::cv_status::timeout) throw std::runtime_error("round trip did not complete in 30 s");
            if (enter) {
                sc_->to_attn.release();
#ifdef FLM_MINICPM_RECONFIG_FAULT_INJECT
                sc_->to_attn_unquiesced.release();
#endif
            }
            if (grp_last_) sc_->to_mm.release();
            armed_ = !grp_last_;
            if (sc_->trips.fetch_add(1) == 0) header_print_r("INFO", "minicpm_v: vision attention runs through single-context reconfiguration (first round trip done)");
            return true;
        } catch (const std::exception& e) {
            sc_->to_attn.release();
#ifdef FLM_MINICPM_RECONFIG_FAULT_INJECT
            sc_->to_attn_unquiesced.release();
#endif
            sc_->to_mm.release();
            armed_ = false;
            minicpm_reconfig::disable(std::string("vision attention round trip failed: ") + e.what());
            return false;
        }
    }

    /// An exception left a call (or the previous call ended) with the array still in the attention configuration of vision_mm_a's context:
    /// the firmware believes design A is loaded, so the next matmul would run on the wrong array. Disable the path and run one dispatch in
    /// vision_attn's own context so that the firmware reloads design A's PDI for the next matmul.
    void _abort_armed() {
        if (!armed_) return;
        armed_ = false;
        minicpm_reconfig::disable("array left in the attention configuration between dispatches");
        try {
            if (last_slot_ != nullptr) (void)last_slot_->app(o_, q_, k_, v_);
        } catch (const std::exception& e) {
            header_print_r("WARNING", "minicpm_v: recovery dispatch after an aborted vision attention call failed: " << e.what());
        }
    }
#endif

    /// The groups of one run_views call: consecutive views of equal length, one dispatch each. `fn(L, n)` makes one dispatch.
    template <class Fn>
    void _run_groups(const std::vector<int>& lengths, Fn&& fn) {
        int groups = 0;
        for (size_t i = 0; i < lengths.size();) {
            size_t j = i + 1;
            while (j < lengths.size() && lengths[j] == lengths[i]) j++;
            groups++;
            i = j;
        }
#if FLM_MINICPM_RECONFIG_SUPPORTED
        _abort_armed();   // a previous call must not have left the array armed
        try {
#endif
            int g = 0;
            for (size_t i = 0; i < lengths.size();) {
                size_t j = i + 1;
                while (j < lengths.size() && lengths[j] == lengths[i]) j++;
#if FLM_MINICPM_RECONFIG_SUPPORTED
                grp_last_ = (++g == groups);
#else
                (void)g;
#endif
                fn(lengths[i], (int)(j - i));
                i = j;
            }
#if FLM_MINICPM_RECONFIG_SUPPORTED
        } catch (...) {
            grp_last_ = true;
            _abort_armed();
            throw;
        }
        grp_last_ = true;
#endif
    }

    /// The attention dispatch of one (L_pad, L, n) shape on the BOs already filled.
    ert_cmd_state _launch(slot& sl, int L_pad, int L, int n, const int* head_perm,
                          buffer<bf16>* qb = nullptr, buffer<bf16>* kb = nullptr,
                          buffer<bf16>* vb = nullptr, buffer<bf16>* ob = nullptr) {
        buffer<bf16>& q = qb ? *qb : q_;
        buffer<bf16>& k = kb ? *kb : k_;
        buffer<bf16>& v = vb ? *vb : v_;
        buffer<bf16>& o = ob ? *ob : o_;
#if FLM_MINICPM_RECONFIG_SUPPORTED
        last_slot_ = &sl;
        if (sc_ != nullptr && mode_ != path_mode::CONTEXT_ONLY) {
            const bool ok = mode_ == path_mode::SINGLE_CONTEXT_ONLY ? (sc_->built && minicpm_reconfig::npu_exclusive()) : sc_->usable();
            if (ok && _launch_single_ctx(sl, L_pad, L, n, head_perm, q, k, v, o)) return ERT_CMD_STATE_COMPLETED;
            if (mode_ == path_mode::SINGLE_CONTEXT_ONLY) throw std::runtime_error("single-context attention unavailable or failed");
        }
        armed_ = false;   // this dispatch runs through vision_attn's own context
#else
        (void)L_pad; (void)L; (void)n; (void)head_perm;
#endif
        return sl.app(o, q, k, v);
    }

    static int _pad(int L) { return (L + CHUNK - 1) / CHUNK * CHUNK; }

    slot& _slot(int L_pad, int L, int n) {
        auto key = std::make_tuple(L_pad, L, n);
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
#pragma omp parallel for schedule(static) num_threads(mcv::vision::par_threads())
            for (int r = 0; r < L; r++)
                mcv::simd::f32_to_bf16(d + (size_t)r * cols, src + (size_t)r * S, cols);
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

    /// The second BO set of `_dispatch16_piped`, grow-only like the first.
    void _ensure_alt(int L_pad) {
        if (L_pad <= alt_rows_) return;
        const size_t words = (size_t)L_pad * D_;
        aq_ = mgr_->create_bo_buffer<bf16>(words);
        ak_ = mgr_->create_bo_buffer<bf16>(words);
        av_ = mgr_->create_bo_buffer<bf16>(words);
        ao_ = mgr_->create_bo_buffer<bf16>(words);
        aq_.memset((bf16)0); ak_.memset((bf16)0);
        av_.memset((bf16)0); ao_.memset((bf16)0);
        aq_.sync_to_device(); ak_.sync_to_device();
        av_.sync_to_device(); ao_.sync_to_device();
        alt_rows_ = L_pad;
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
    std::map<std::tuple<int, int, int>, slot> slots_;
    buffer<bf16>                        q_, k_, v_, o_;
    buffer<bf16>                        aq_, ak_, av_, ao_;   //!< _dispatch16_piped's second set
    int                                 alt_rows_ = 0;
    size_t                              D_       = 0;
    int                                 heads_   = 0;
    int                                 dh_      = 0;
    int                                 bo_rows_ = 0;
#if FLM_MINICPM_RECONFIG_SUPPORTED
    vision_attn_reconfig*               sc_      = nullptr;
    path_mode                           mode_    = path_mode::AUTO;
    bool                                armed_   = false;   //!< the array is in vision_attn's configuration inside vision_mm_a's context
    bool                                grp_last_ = true;   //!< the dispatch being made is the last of its call (leaves the configuration)
    slot*                               last_slot_ = nullptr;
#endif
};

} // namespace mcv_npu

#endif  // __MINICPM_V_4_7_NPU_VISION_ATTN_OP_HPP__
