// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hy_mem_prefill_overlay.hpp
/// \brief The prefill overlay: dequant_mm and attention in one xclbin
///        (FLM_Xclbin/Hy_mem_30BA3B/prefill_overlay, its README has the design)
/// \note An xclbin fixes its stream routes, DMA BDs and lock values at load time,
///       and the two ops need different ones. So the overlay xclbin holds only what
///       both share (core programs, memory arenas), and the host writes the
///       active op's configuration into the array when the op changes: ~0.12 ms,
///       instead of a ~2 ms hw context switch between two xclbins.
///
///       Whenever another xclbin runs (decode layers, lm_head), the NPU reloads the
///       overlay on its next run and everything written into it is gone:
///       invalidate() then, and the next run writes the full configuration again.
#pragma once
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include "npu_utils/npu_utils.hpp"
#include "hy_mem_model_def.hpp"

class hy_mem_prefill_overlay {
public:
    /// ops of the overlay (iron/layout.py OPS); NONE = configuration unknown
    enum op_t { OP_NONE = 0, OP_DQ = 1, OP_ATTN = 2 };

    hy_mem_prefill_overlay(npu_xclbin_manager* npu, const std::string& xclbin, const hy_mem_desc& desc);

    /// \brief another xclbin ran: the next run writes the full configuration
    void invalidate() { configured = OP_NONE; }

    /// \brief builds the attention runs for a chunk of L tokens at pos0. Once per
    ///        chunk: every layer's attention is the same run on its own KV cache.
    ///        The q rows are zero-padded to a multiple of 128 (one attention
    ///        round); the cache needs hy_mem_desc::KV_TAIL_ROWS behind it.
    void prepare_attention(int L, int pos0, uint32_t MAX_L);

    /// \brief out[L][DQ] = attention of q over one layer's decode KV cache
    /// \param q        [L] rows of DQ (normed, roped), row stride q_stride
    /// \param kv_cache the layer's cache, rows 0 .. pos0 + L - 1 filled and on the device
    void attention(bf16* out, const bf16* q, int q_stride, buffer<bf16>& kv_cache);

    /// \brief y[L][Dout] = x[L][Din] W^T, W the Q4_K matrix at `w_offset` bytes into
    ///        `weights`, in the overlay's (int8-mmul) block order
    /// \param y   [L][Dout] bf16, row major; rows past L are not written
    /// \param x   [L][Din] bf16, row major
    /// \param weights the layer's prefill projection buffer, on the device
    /// \note L is padded up to DQ_ROWS_PER_ROUND with zero rows, whose outputs are
    ///       dropped. Din % 256 and Dout % 512 must hold; fits() says so.
    /// \param channel_bytes 0 for a `pairs` matrix; a routed expert's channel stride
    ///        for an `expert_channels` one (hy_mem_cpu::q4k_matrix)
    void dequant_mm(bf16* y, const bf16* x, int L, buffer<u8>& weights, size_t w_offset, int Din, int Dout,
                    size_t channel_bytes = 0);

    /// \brief can dequant_mm take this shape? (the host version takes any)
    static bool fits(int Din, int Dout) { return Din % 256 == 0 && Dout % 512 == 0; }

    /// \brief largest chunk the q / o buffers hold
    static constexpr int MAX_ROWS = 4096;
    /// \brief tokens of one dequant_mm round (4 core rows x NPU_M_TILE 64)
    static constexpr int DQ_ROWS_PER_ROUND = 256;
private:
    /// \brief x / y staging big enough for these element counts, kept across calls
    void _ensure_mm_buffers(size_t x_elems, size_t y_elems);

    const hy_mem_desc& desc;
    npu_app_manager* mgr;
    int configured = OP_NONE;

    // attention of the prepared chunk, one app per op the array may come from
    // (the switch is part of the run's instruction sequence)
    std::map<int, npu_app> attn_runs;
    int attn_L = 0, attn_L_pad = 0, attn_pos0 = 0;
    uint32_t attn_MAX_L = 0;
    npu_app buf_owner;    // the app the buffers below were created with
    buffer<bf16> q_buf;   // [MAX_ROWS][DQ]: q rows packed, padded to 128 rows
    buffer<bf16> o_buf;   // [MAX_ROWS][DQ]

    // dequant_mm: one run per (shape, weight offset, op the array comes from).
    // The weight buffer is a call argument, so all 47 MoE layers share one run.
    struct dq_key {
        int L_pad, Din, Dout, from;
        size_t w_offset, channel_bytes;
        bool operator<(const dq_key& o) const {
            return std::tie(L_pad, Din, Dout, from, w_offset, channel_bytes)
                 < std::tie(o.L_pad, o.Din, o.Dout, o.from, o.w_offset, o.channel_bytes);
        }
    };
    std::map<dq_key, npu_app> dq_runs;
    /// The routed experts give every (expert, token count) its own run, so the cache
    /// can grow without bound over a long session; past this many it is dropped and
    /// rebuilt. Each entry is an npu_app (an XRT run plus its instruction buffer).
    static constexpr size_t MAX_DQ_RUNS = 2048;
    buffer<bf16> mm_x;    // [L_pad][Din] staging, grown as needed
    buffer<bf16> mm_y;    // [L_pad][Dout]
};
