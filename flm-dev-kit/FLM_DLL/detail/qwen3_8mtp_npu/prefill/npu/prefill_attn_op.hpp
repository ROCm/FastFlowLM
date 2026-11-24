// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file prefill/npu/prefill_attn_op.hpp
/// \brief The prefill attention core on the NPU: one dispatch per window per
///        layer, over a KV cache some other object owns.
///
/// This is deliberately NOT a `qwen38::decoder_layer`. `attn.xclbin` computes
/// only `o = softmax(q K^T / sqrt(d) + causal) V`; the projections, both
/// RMSNorms, RoPE, the output gate, o_proj and the MLP stay on the CPU in
/// `cpu_ref::attention_layer`'s own helpers. So the natural shape is an
/// operator the prefill block calls between two CPU halves, not a layer.
///
/// It holds no weights and no cache. One instance is shared by all 16
/// attention layers -- the only per-layer state an attention core has is the
/// cache, and that belongs to `npu_attn_layer`. The caller hands the cache BO
/// in; BOs are device-scoped (`npu_app::create_bo_buffer` builds
/// `buffer<T>(*device, size)`), so a BO allocated from the verify xclbin's app
/// is legal here.
///
/// What it does own is the q and o staging BOs, which must come from THIS
/// app -- a BO from another app does not fault, the dispatch simply never
/// completes.
#ifndef __QWEN3_8MTP_NPU_PREFILL_ATTN_OP_HPP__
#define __QWEN3_8MTP_NPU_PREFILL_ATTN_OP_HPP__

#include "prefill/npu/prefill_attn_sequence.hpp"
#include "speculate/npu/mtp_weights.hpp"    // q38npu::mtp_layout
#include "speculate/npu/proj_op.hpp"   // PROF8_DEV, host_par_min
#include "common/cpu_simd.hpp"         // the vectorised fp32 <-> bf16 marshalling

#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace q38npu {

/// Generic over the layout for the same reason `gen_prefill_attn_seq` is: the
/// base stack's `attn_layout` and the draft head's `mtp_layout` describe the
/// same cache geometry, so one bitstream serves both. The two aliases below are
/// the only instantiations; see the note on the sequence generator.
template <class Layout>
class prefill_attn_op_t {
public:
    /// \param a  an app over attn.xclbin
    /// \param layout  the SAME layout the cache BOs were built from. Only the
    ///        geometry and plane pitch are read; nothing here is sized by
    ///        `L_TOKENS`, which belongs to the other bitstream.
    prefill_attn_op_t(npu_app a, const Layout& layout,
                      const prefill_attn_rtp_addresses& r = prefill_attn_rtp_addresses{})
        : app_(std::move(a)), t_(layout), rtp_(r) {
        DQ_ = (size_t)t_.DQ;
    }

    /// Rows a window is rounded up to. The caller does not have to be aligned;
    /// `run()` pads. There is no *upper* bound to match it: the sequence is
    /// emitted per call, one round of ~40 descriptors per CHUNK rows, so a long
    /// window costs a longer sequence and nothing else. `qwen3_5vl_prefill.hpp`
    /// drives its attention the same way -- one dispatch for the whole padded
    /// prompt, per layer.
    static constexpr int CHUNK = prefill_attn_tiles::Q_CHUNK;   // 128

    const Layout& layout() const { return t_; }
    const prefill_attn_rtp_addresses& rtp() const { return rtp_; }

    /// Adopt a grown layout. The op has no BO sized by `MAX_L`, so this only
    /// changes the plane pitch the next sequence is emitted with -- but that
    /// pitch IS `MAX_L`, and it is where the v plane starts, so forgetting it
    /// reads every v row out of the k plane's tail.
    void set_layout(const Layout& layout) {
        if (layout.plane_rows() == t_.plane_rows() && layout.DQ == t_.DQ) return;
        t_ = layout;
        DQ_ = (size_t)t_.DQ;
        have_seq_ = false;
    }

    /// Run the attention core for one window. **Any positive L, one dispatch.**
    ///
    /// Short windows are padded up to CHUNK and long ones simply emit more
    /// rounds; neither is the caller's problem and neither is a reason to go
    /// anywhere but the device. The caller may hand any `L_begin`: attn.cc
    /// masks on absolute position, so the queries carry positions
    /// `L_begin .. L_begin + L` and reach exactly the keys they should. The
    /// window's own k|v must already be in the cache -- the caller appends
    /// before calling.
    ///
    /// \param q_rows  [L][DQ] fp32, post-q_norm post-RoPE, head-major in a row
    /// \param L       query rows. Any positive length; padded to CHUNK here
    /// \param L_begin absolute position of `q_rows[0]`. The cache must already
    ///                hold rows [0, L_begin + L)
    /// \param cache   the layer's cache BO, `attn_layout`'s two planes
    /// \param o_rows  [L][DQ] fp32 out, ungated
    void run(const float* q_rows, int L, int L_begin,
             buffer<bf16>& cache, float* o_rows) {
        if (L <= 0) return;
        if (cache.size() != t_.kv_cache_size)
            throw std::runtime_error(
                "prefill_attn_op::run: the cache BO is " +
                std::to_string(cache.size()) + " bf16, the layout wants " +
                std::to_string(t_.kv_cache_size) +
                " -- set_layout() was not applied after a grow");
        // **Against the PADDED window, not L.** The pass is generated for
        // `L_pad` queries and its kv broadcast walks 128-row blocks through
        // `round_up(L_begin + L_pad, CHUNK)` -- see gen_prefill_attn_seq's
        // kv_end. Rounding up is monotone and MAX_L is a fixed point of it
        // (attn_layout::build requires a whole CHUNK), so `L_begin + L_pad <=
        // MAX_L` is exactly the condition that keeps kv_end inside the buffer.
        //
        // The cache has no apron above MAX_L to absorb the padding, and it
        // cannot be clamped here either: attn.cc derives the same block count
        // itself (`kv_blocks`), so a host-only clamp would leave blocks queued
        // for the next dispatch.
        const int L_pad = (L + CHUNK - 1) / CHUNK * CHUNK;
        if (L_begin < 0 || (size_t)L_begin + (size_t)L_pad > (size_t)t_.MAX_L)
            throw std::runtime_error(
                "prefill_attn_op::run: [" + std::to_string(L_begin) + ", " +
                std::to_string(L_begin + L) + ") pads to [" +
                std::to_string(L_begin) + ", " + std::to_string(L_begin + L_pad) +
                "), outside a cache of MAX_L " + std::to_string(t_.MAX_L));

        _ensure_bos(L_pad);

        // q in. Padding rows repeat the last real one rather than zeroing:
        // a zero query is not harmful here -- there is no norm downstream --
        // but repeating keeps the pad rows' scores in the same range as the
        // real ones, so a NaN in the output is unambiguously a real row's.
        //
        // The pad rows do attend to cache positions above the cursor, which
        // hold zeros. That is sound for the same reason the verify path's tail
        // padding is: the mask is causal, so a row at index >= L reaches only
        // outputs at index >= L, and those are dropped below.
        bf16* qd = q_.data();
        {
            // Vectorised and threaded on the same terms as every other
            // marshalling loop on this path (common/cpu_simd.hpp, and
            // q38npu::host_par_min for why the threads are conditional). 16
            // layers x L x 6144 per window, and it was the fpclassify
            // constructor one element at a time.
            const size_t n = (size_t)L * DQ_;
#pragma omp parallel for schedule(static) if (n >= q38npu::host_par_min())
            for (int r = 0; r < L; r++)
                q38simd::f32_to_bf16(qd + (size_t)r * DQ_,
                                     q_rows + (size_t)r * DQ_, DQ_);
        }
        for (int r = L; r < L_pad; r++)
            std::memcpy(qd + (size_t)r * DQ_, qd + (size_t)(L - 1) * DQ_,
                        DQ_ * sizeof(bf16));
        q_.sync_to_device();

        if (!have_seq_ || L_begin != seq_begin_ || L_pad != seq_rows_) {
            gen_prefill_attn_seq(app_.seq(), t_, L_begin, L_pad, rtp_);
            seq_begin_ = L_begin;
            seq_rows_  = L_pad;
            have_seq_  = true;
        }

        const ert_cmd_state st = PROF8_DEV(app_(o_, q_, cache));
        if (st != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error(
                "prefill_attn_op::run: dispatch state " +
                std::to_string((int)st) + " at L_begin " +
                std::to_string(L_begin) + ", " + std::to_string(L_pad / CHUNK) +
                " rounds -- a timeout is a sequence bug or a stale RTP address "
                "(diff attn.mlir.prj/input_with_addresses.mlir against "
                "prefill_attn_rtp_addresses)");

        o_.sync_from_device();
        const bf16* od = o_.data();
        {
            const size_t n = (size_t)L * DQ_;
#pragma omp parallel for schedule(static) if (n >= q38npu::host_par_min())
            for (int r = 0; r < L; r++)
                q38simd::bf16_to_f32(o_rows + (size_t)r * DQ_,
                                     od + (size_t)r * DQ_, DQ_);
        }
    }

private:
    /// Grow-only, and never shrink: prefill windows arrive largest-first in
    /// practice, and a reallocation invalidates nothing but costs a sequence
    /// regeneration through the L_pad change anyway.
    void _ensure_bos(int L_pad) {
        if (L_pad <= bo_rows_) return;
        const size_t words = (size_t)L_pad * DQ_;
        q_ = app_.create_bo_buffer<bf16>(words);
        o_ = app_.create_bo_buffer<bf16>(words);
        q_.memset((bf16)0);
        o_.memset((bf16)0);
        q_.sync_to_device();
        o_.sync_to_device();
        bo_rows_ = L_pad;
    }

    npu_app                    app_;
    Layout                     t_;
    prefill_attn_rtp_addresses rtp_;
    buffer<bf16>               q_, o_;
    size_t                     DQ_        = 0;
    int                        bo_rows_   = 0;
    int                        seq_begin_ = -1;
    int                        seq_rows_  = -1;
    bool                       have_seq_  = false;
};

/// Module 1: the 16 base full-attention layers, over `npu_attn_layer`'s caches.
using prefill_attn_op = prefill_attn_op_t<attn_layout>;
/// Module 2: the MTP draft head's step 0, over `npu_mtp_layer`'s single cache.
using mtp_prefill_attn_op = prefill_attn_op_t<mtp_layout>;

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_PREFILL_ATTN_OP_HPP__
