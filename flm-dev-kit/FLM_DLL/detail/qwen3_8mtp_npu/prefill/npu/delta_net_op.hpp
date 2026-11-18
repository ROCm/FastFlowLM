// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file prefill/npu/delta_net_op.hpp
/// \brief The gated-delta-net recurrence on the NPU: conv1d then the fold.
///
/// N9 moved a delta layer's six projections onto `dequant_mm.xclbin`, which is
/// 99.2% of the block's MACs. This moves what was left -- the two *mutating*
/// stages of `cpu_ref::delta_net_layer`, `conv_silu` and `delta_rule` -- onto
/// two more bitstreams:
///
///   * `conv.xclbin`            causal depthwise conv1d(k=4) + SiLU over all
///                              10240 channels, and the per-head l2norm on q
///                              and k that `delta_rule` would otherwise do
///   * `GateDeltaNet_prefill.xclbin`  the fold itself, 48 heads of
///                              `S += k (beta (v - alpha S^T k))^T`, `o = S^T q`
///
/// Measured on the `-s 0` prompt before this existed: the two host stages are
/// 692 ms of a 5.95 s prefill window, 11.6%. That is the ceiling for this file.
///
/// ## The host stays authoritative for state
///
/// Not a preference -- a requirement. Decode reaches the same block at L = 1
/// through the CPU path, and speculation's `save_state`/`load_state` read
/// `cpu_ref::delta_net_layer`'s `state_` and `conv_state_` directly. So every
/// dispatch uploads S and downloads it again, 3 MB each way per layer per
/// window, and there is exactly one dispatch per layer per window to amortise
/// it over. At L = 346 that round trip is a real fraction of the win; at
/// L = 1024 it is not. It is the first thing to measure if the A/B disappoints.
///
/// This is also why `run()` takes the `delta_net_layer` rather than raw state
/// pointers: it *is* the layer's `conv_silu` + `delta_rule` + the l2norms, and
/// it owes the same contract -- advance the conv window by L and the recurrent
/// state by L, exactly once, in order. The caller must then NOT call those two.
///
/// ## Where the 1/sqrt(head_dim) goes
///
/// `conv.xclbin`'s q columns are driven in `NORM_K` mode -- l2norm, no extra
/// scale -- so the only place q is scaled is the delta-net kernel's own
/// `inv_sqrt_DH`. See conv1d_sequence.hpp; chaining the two designs at their
/// defaults scales q by 1/128 instead, and a cosine cannot see it.
///
/// ## Chunking a fold is safe
///
/// S is updated in place on the device and stays there between chunks, so N
/// dispatches over [0,1024), [1024,2048), ... compute exactly what one
/// dispatch over [0, N*1024) would. The sequence generator takes `L_start` for
/// precisely this. The conv is not chunked: it is one pass over the window.
#ifndef __QWEN3_8MTP_NPU_DELTA_NET_OP_HPP__
#define __QWEN3_8MTP_NPU_DELTA_NET_OP_HPP__

#include "prefill/npu/conv1d_sequence.hpp"
#include "prefill/npu/delta_net_sequence.hpp"
#include "common/cpu_ref_delta_net_layer.hpp"
#include "speculate/npu/proj_op.hpp"   // PROF8_DEV, host_par_min
#include "common/cpu_simd.hpp"         // the vectorised fp32 <-> bf16 marshalling

#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace q38npu {

// ==========================================================================
// one layer's conv weights
// ==========================================================================
/// `[conv_k][conv_dim]` bf16, tap-major -- the layout `gen_seq_conv1d`'s weight
/// descriptor walks and the one `cpu_ref::delta_net_weights::conv_w` already
/// has, so this is a convert-and-copy and nothing more.
///
/// 80 KB a layer, 3.8 MB for all 48. Kept separate from `npu_delta_weights`
/// (the 211 MB of q4_k projections) because the two are read by different
/// bitstreams and staged at different times; nothing is shared but the layer
/// index.
class npu_delta_ssm_weights {
public:
    /// \param app   an app over conv.xclbin
    /// \param half  the layer's CPU reference, already loaded. Its `conv_w` is
    ///              the source -- no second path from the checkpoint to keep
    ///              in step with `delta_net_weights::load`.
    npu_delta_ssm_weights(npu_app& app, const cpu_ref::delta_net_layer& half)
        : layer_(half.layer()) {
        const std::vector<float>& w = half.weights().conv_w;
        const size_t n = (size_t)half.geometry().conv_k *
                         (size_t)half.geometry().conv_dim();
        if (w.size() != n)
            throw std::runtime_error(
                "npu_delta_ssm_weights: conv_w is " + std::to_string(w.size()) +
                " floats, the geometry wants " + std::to_string(n));
        conv_w_ = app.create_bo_buffer<bf16>(n);
        bf16* d = conv_w_.data();
        for (size_t i = 0; i < n; i++) d[i] = (bf16)w[i];
        conv_w_.sync_to_device();
    }

    buffer<bf16>& conv_w() { return conv_w_; }
    int           layer() const { return layer_; }

private:
    int          layer_;
    buffer<bf16> conv_w_;
};

// ==========================================================================
// the two dispatches
// ==========================================================================
class delta_net_op {
public:
    /// Tokens per memtile chunk in the delta-net design. `L_PADDED` is a
    /// multiple of this, and the kernel's alpha/beta and qkv descriptors move
    /// whole chunks whether or not the rows in them are real.
    static constexpr int MT_L = 128;

    /// Tokens per delta-net dispatch. The design is proven at L = 1120
    /// (its own out.log), and 1024 is the power of two under that -- which
    /// matters because the engine's prefill windows are powers of two
    /// (`automodel.cpp` `_chunked_insert`), so every chunk but possibly the
    /// last is full and one emitted sequence serves them all.
    static constexpr int CHUNK = 1024;

    /// The conv's x descriptor has a 6-bit iteration field over 128-token
    /// blocks, and it needs conv_k - 1 history rows on top of the window.
    static constexpr int MAX_WINDOW = (int)conv_seq::MAX_L_IN - 3;

    /// \param conv_app  an app over conv.xclbin -- one dispatch per layer per
    ///                  window, so one app is enough
    /// \param dn_mgr    a manager over GateDeltaNet_prefill.xclbin. Apps are
    ///                  taken one per *chunk*, because a sequence is baked into
    ///                  an app and each chunk has its own `L_start`.
    /// \param g         the block geometry; all 48 layers share it
    /// \param rtp       the conv design's per-build core RTP address
    delta_net_op(npu_app conv_app, npu_app_manager* dn_mgr,
                 const cpu_ref::delta_net_geometry& g,
                 const conv1d_rtp_addresses& rtp = conv1d_rtp_addresses{})
        : conv_(std::move(conv_app)), dn_mgr_(dn_mgr), g_(g), rtp_(rtp) {
        if (!dn_mgr_) throw std::runtime_error("delta_net_op: no app manager");
        _check_geometry();
        S_ = conv_.create_bo_buffer<float>(_S_floats());
    }

    const cpu_ref::delta_net_geometry& geometry() const { return g_; }
    const conv1d_rtp_addresses&        rtp()      const { return rtp_; }

    /// The staging BO the qkv projection must write into, and the row it must
    /// start at.
    ///
    /// The conv reads `conv_k - 1` history rows before the window, so the
    /// projection's output belongs at row `conv_k - 1`, not row 0 -- the same
    /// arrangement `qwen3_5vl_prefill.hpp` gets by passing that offset to
    /// `gen_dequant_mm_512`. Call this before filling it; it is what sizes
    /// every BO for the window.
    ///
    /// \param L window length. Must be the same L the following `run()` gets.
    buffer<bf16>& qkv_in(int L) {
        _ensure_bos(L);
        staged_L_ = L;
        return x_;
    }
    /// Row of `qkv_in()` the projection's first output token belongs at.
    int qkv_in_row() const { return g_.conv_k - 1; }

    /// conv1d + SiLU + the q/k l2norms, then the gated delta rule.
    ///
    /// **Advances `half`'s conv window and recurrent state by L**, exactly as
    /// `conv_silu` followed by `delta_rule` would. The caller owes `advance(L)`
    /// and owes it exactly once, and must not also call those two.
    ///
    /// \param half   the layer, for its state and its geometry
    /// \param L      window length; must match the `qkv_in(L)` that staged it
    /// \param conv_w this layer's `npu_delta_ssm_weights::conv_w()`
    /// \param alpha  [L][v_heads] from `delta_net_layer::gates`
    /// \param beta   [L][v_heads] likewise
    /// \param core   [L][dv] out, resized here -- `delta_rule`'s `core`
    void run(cpu_ref::delta_net_layer* half, int L, buffer<bf16>& conv_w,
             const float* alpha, const float* beta, std::vector<float>& core) {
        if (L <= 0) return;
        if (L != staged_L_)
            throw std::runtime_error(
                "delta_net_op::run: L " + std::to_string(L) +
                " but qkv_in() staged " + std::to_string(staged_L_) +
                " -- the x BO's history rows and the sequence would disagree");
        if (L > MAX_WINDOW)
            throw std::runtime_error(
                "delta_net_op::run: window of " + std::to_string(L) +
                " past MAX_WINDOW " + std::to_string(MAX_WINDOW) + " -- chunk it");

        const int    K   = g_.conv_k;
        const int    CD  = g_.conv_dim();
        const int    NV  = g_.v_heads;
        const int    DV  = g_.dv();
        const int    L_pad = (L + MT_L - 1) / MT_L * MT_L;

        // ---- conv: history rows in, window out ------------------------------
        // conv_state_ holds the last conv_k *pre-conv* inputs, newest last, so
        // the conv_k - 1 rows the window needs are its newest ones. Row 0 is
        // the oldest and is already past.
        {
            bf16* xd = x_.data();
            const std::vector<float>& win = half->conv_window();
            // Rows 1..K-1 of the window land at rows 0..K-2 of the BO, so both
            // sides are one contiguous run of (K - 1) * CD and the nested loop
            // was only ever expressing the shift.
            q38simd::f32_to_bf16(xd, win.data() + (size_t)CD,
                                 (size_t)(K - 1) * CD);
            x_.sync_to_device();
        }

        // The conv writes rows [0, L); the delta-net's descriptors walk L_pad.
        // The pad rows are never folded (the kernel's trip count is the real L)
        // but they are still DMA'd, so leave nothing from a longer previous
        // window there. This has to happen *before* the conv dispatch: a
        // sync_to_device is whole-buffer, so doing it after would push the
        // stale host copy of rows [0, L) over the conv's own output.
        //
        // And it is whole-buffer that makes the guard worth having. This BO is
        // the widest thing in the op -- 20 MB at L = 1024 -- and the host has
        // nothing to put in rows [0, L): the conv fills them on the device.
        // So the only reason to upload at all is to clear the tail, and the
        // tail is only dirty if a *longer* window already convolved into it.
        // `_ensure_bos` zeroed the whole BO when it allocated, so until that
        // happens there is nothing to clear and 20 MB of DDR traffic per layer
        // per window to save.
        if (L_pad > L && conv_written_ > L) {
            std::memset(qkv_.data() + (size_t)L * CD, 0,
                        (size_t)(L_pad - L) * CD * sizeof(bf16));
            qkv_.sync_to_device();
        }
        conv_written_ = std::max(conv_written_, L);

        if (conv_rows_ != L) {
            gen_seq_conv1d(conv_.seq(), (uint32_t)(L + K - 1), (uint32_t)CD,
                           (uint32_t)L, (uint32_t)g_.dq(), rtp_.l_buffer,
                           0, conv_seq::NORM_K);
            conv_rows_ = L;
        }
        _dispatch(PROF8_DEV(conv_(qkv_, x_, conv_w)), "conv", L, 0);

        // The layer's conv window advances to the last conv_k *pre-conv* rows
        // of this window -- which are still sitting in the x BO, because that
        // is what the x BO holds. Taking them from there rather than from the
        // caller's fp32 copy keeps the host's window bit-identical to what the
        // device convolved.
        {
            const bf16* xd = x_.data();
            std::vector<float>& win = half->conv_window_mut();
            const int base = (K - 1) + L - K;    // first of the last conv_k rows
            // K consecutive rows out of the BO into K consecutive rows of the
            // window: contiguous on both sides.
            q38simd::bf16_to_f32(win.data(), xd + (size_t)base * CD,
                                 (size_t)K * CD);
        }

        // ---- alpha | beta ---------------------------------------------------
        // One fp32 buffer, alpha's L_pad x NV block then beta's. The kernel
        // reads them on a single descriptor whose inner axis of 2 strides by
        // exactly the distance between the two blocks.
        //
        // Pad rows get alpha = 1, beta = 0 -- an exact no-op fold, not zeros.
        // The kernel never reaches them (its trip count is the real L), but
        // qwen3_5vl hit NaN in its tail chunk leaving them raw, and a decay of
        // 0 is not a value this recurrence has a meaning for.
        {
            float* a = ab_.data();
            float* b = a + _ab_half();
            std::memcpy(a, alpha, (size_t)L * NV * sizeof(float));
            std::memcpy(b, beta,  (size_t)L * NV * sizeof(float));
            for (size_t i = (size_t)L * NV; i < (size_t)L_pad * NV; i++) {
                a[i] = 1.0f;
                b[i] = 0.0f;
            }
            ab_.sync_to_device();
        }

        // ---- S in, the chunks, S out ---------------------------------------
        std::memcpy(S_.data(), half->recurrent_state().data(),
                    _S_floats() * sizeof(float));
        S_.sync_to_device();

        const int chunks = (L + CHUNK - 1) / CHUNK;
        _ensure_apps(chunks);
        for (int c = 0; c < chunks; c++) {
            const int L_start = c * CHUNK;
            const int Lc      = (L - L_start) < CHUNK ? (L - L_start) : CHUNK;
            const int Lc_pad  = (Lc + MT_L - 1) / MT_L * MT_L;

            if (dn_shape_[c].first != Lc || dn_shape_[c].second != L_start) {
                generate_gate_delta_net_prefill_sequence(
                    *dn_[c]->seq(),
                    /*NUM_OF_COLUMN=*/8, /*NUM_CT_PER_COLUMN=*/4,
                    (uint32_t)g_.expand(), (uint32_t)NV, (uint32_t)MT_L,
                    (uint32_t)Lc, (uint32_t)Lc_pad,
                    (uint32_t)g_.k_head_dim, (uint32_t)g_.k_head_dim,
                    (uint32_t)g_.v_head_dim,
                    (uint32_t)L_start,
                    /*S_in_out_offset_fp32=*/0,
                    (uint32_t)_ab_half());
                dn_shape_[c] = {Lc, L_start};
            }
            _dispatch(PROF8_DEV((*dn_[c])(S_, qkv_, ab_, o_)), "deltanet", Lc, L_start);
        }

        S_.sync_from_device();
        std::memcpy(half->recurrent_state_mut().data(), S_.data(),
                    _S_floats() * sizeof(float));

        // ---- o ---------------------------------------------------------------
        o_.sync_from_device();
        // `resize`, not `assign`: the loop below writes all L x DV of it, so
        // the zero-fill is a 24 MB memset per layer whose every byte is
        // overwritten before it is read. 48 layers x L = 1024 is 1.2 GB.
        core.resize((size_t)L * DV);
        const bf16* od = o_.data();
        {
            // The widest widen left on the path: 48 layers x L x 6144.
            const size_t n = (size_t)L * DV;
#pragma omp parallel for schedule(static) if (n >= q38npu::host_par_min())
            for (int r = 0; r < L; r++)
                q38simd::bf16_to_f32(core.data() + (size_t)r * DV,
                                     od + (size_t)r * DV, (size_t)DV);
        }
    }

    // For the hardware driver, which scores the two dispatches separately.
    // The op fuses them -- the conv's output never reaches the host on the
    // live path -- so this is the only way to charge each one for its own
    // error instead of for the pair's.

    /// The conv's output and the fold's input, `[L_pad][conv_dim]` bf16. The
    /// caller owes a `sync_from_device()`; nothing on the live path reads it.
    buffer<bf16>& conv_out() { return qkv_; }
    /// The state BO, `[v_heads][k_head_dim][v_head_dim]` fp32, already synced
    /// back by `run()`.
    buffer<float>& state_bo() { return S_; }
    /// `[L_pad][dv]` bf16, the fold's raw output before the fp32 widening.
    buffer<bf16>& o_bo() { return o_; }

private:
    size_t _S_floats() const {
        return (size_t)g_.v_heads * g_.k_head_dim * g_.v_head_dim;
    }
    /// fp32 elements from alpha's block to beta's. Sized by the BO, not by the
    /// chunk: the generator does not re-base it by `L_start`.
    size_t _ab_half() const { return (size_t)bo_L_pad_ * g_.v_heads; }

    void _dispatch(ert_cmd_state st, const char* what, int L, int L_start) {
        if (st == ERT_CMD_STATE_COMPLETED) return;
        throw std::runtime_error(
            std::string("delta_net_op::run: ") + what + " dispatch state " +
            std::to_string((int)st) + " at L " + std::to_string(L) +
            ", L_start " + std::to_string(L_start) +
            " -- a timeout is a sequence bug or a stale RTP address (diff the "
            "design's input_with_addresses.mlir against conv1d_rtp_addresses)");
    }

    /// Grow-only. Every BO is sized by the padded window, so a shrink would
    /// only force a reallocation the next long window undoes.
    void _ensure_bos(int L) {
        const int L_pad = (L + MT_L - 1) / MT_L * MT_L;
        if (L_pad <= bo_L_pad_) return;

        const size_t CD = (size_t)g_.conv_dim();
        x_   = conv_.create_bo_buffer<bf16>((size_t)(L_pad + g_.conv_k - 1) * CD);
        qkv_ = conv_.create_bo_buffer<bf16>((size_t)L_pad * CD);
        o_   = conv_.create_bo_buffer<bf16>((size_t)L_pad * g_.dv());
        ab_  = conv_.create_bo_buffer<float>(2 * (size_t)L_pad * g_.v_heads);
        x_  .memset((bf16)0.0f);
        qkv_.memset((bf16)0.0f);
        o_  .memset((bf16)0.0f);
        ab_ .memset(0.0f);
        x_.sync_to_device();
        qkv_.sync_to_device();
        o_.sync_to_device();
        ab_.sync_to_device();

        bo_L_pad_ = L_pad;
        // Both offsets the sequences carry moved: the conv's row count and the
        // alpha/beta block distance.
        conv_rows_    = -1;
        conv_written_ = 0;       // the fresh BO is zeroed, so nothing is dirty
        for (auto& s : dn_shape_) s = {-1, -1};
    }

    void _ensure_apps(int chunks) {
        while ((int)dn_.size() < chunks) {
            dn_.push_back(std::make_unique<npu_app>(dn_mgr_->create_app()));
            dn_shape_.push_back({-1, -1});
        }
    }

    /// Everything the two sequences assume, in one place. Each is silent if
    /// violated: a ragged shape emits a short sequence and the dispatch parks.
    void _check_geometry() const {
        auto need = [](bool ok, const std::string& what) {
            if (!ok) throw std::runtime_error("delta_net_op: " + what);
        };
        need(g_.conv_dim() % conv_seq::COL_CHANNELS == 0,
             "conv_dim " + std::to_string(g_.conv_dim()) + " is not a whole "
             "number of conv columns (" +
             std::to_string(conv_seq::COL_CHANNELS) + ")");
        need(g_.dq() % conv_seq::COL_CHANNELS == 0,
             "dq is not a whole number of conv columns -- a column would "
             "straddle the q/k split and take the wrong norm mode");
        need(g_.dq() == g_.dk(),
             "the delta-net sequence asserts DQ == DK");
        need(g_.k_head_dim == 128 && g_.v_head_dim == 128,
             "the delta-net bitstream is built for 128-wide heads");
        need(g_.v_heads % 8 == 0,
             "v_heads " + std::to_string(g_.v_heads) + " does not fill the "
             "array's 8 columns evenly");
        need(CHUNK % MT_L == 0,
             "CHUNK is not a whole number of memtile chunks");
    }

    npu_app                        conv_;
    npu_app_manager*               dn_mgr_ = nullptr;
    cpu_ref::delta_net_geometry    g_;
    conv1d_rtp_addresses           rtp_;

    /// One app per chunk: a sequence is baked into an app and `L_start` differs.
    std::vector<std::unique_ptr<npu_app>> dn_;
    /// (L, L_start) each app's sequence was emitted for.
    std::vector<std::pair<int,int>>       dn_shape_;

    buffer<bf16>  x_;     ///< [(L_pad + conv_k - 1)][conv_dim] conv input
    buffer<bf16>  qkv_;   ///< [L_pad][conv_dim] conv output == delta-net input
    buffer<float> ab_;    ///< [2][L_pad][v_heads]
    buffer<bf16>  o_;     ///< [L_pad][dv]
    buffer<float> S_;     ///< [v_heads][k_head_dim][v_head_dim]

    int bo_L_pad_  = 0;
    int staged_L_  = -1;
    int conv_rows_ = -1;
    /// Longest window the conv has written into `qkv_` since it was allocated:
    /// how far past the current L the device copy can be dirty.
    int conv_written_ = 0;
};

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_DELTA_NET_OP_HPP__
