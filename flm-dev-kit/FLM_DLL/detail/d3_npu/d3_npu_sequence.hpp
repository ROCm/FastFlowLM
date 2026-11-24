// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
/// \file d3_npu_sequence.hpp
/// \brief Instruction-sequence generator of the d3 decision engine.
/// \note  d3 never decodes, has no lm_head and keeps no KV cache, so of the Qwen3.5 sequence
///        generators only the prefill GEMM survives. The RTP addresses are mm.xclbin's
///        (FLM_Xclbin/Qwen3_5 MM build) and must move with it.

#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "d3_model_def.hpp"

struct d3_npu_sequence{
    // mm.xclbin RTP addresses; the CT kernel differs between even and odd rows.
    static constexpr uint32_t CT_rtp_address_even_row = 54784;
    static constexpr uint32_t act_rtp_address_even_row = 54848;
    static constexpr uint32_t CT_rtp_address_odd_row = 61696;
    static constexpr uint32_t act_rtp_address_odd_row = 61760;

    /// @brief rows of x one gen_dequant_mm_512() run consumes per weight pass
    static constexpr int MM_ROWS = 512;

    const d3_desc* desc = nullptr;

    d3_npu_sequence(){}
    explicit d3_npu_sequence(d3_desc& desc) : desc(&desc) {}

    /// @brief y[L][Dout] (+ out_offset) = x[L][Din] * W^T for a q4k weight at weight_offset of the
    ///        layer's proj buffer; L % 512 == 0. rtp_act = 1 fuses a silu.
    /// @param weight_offset in **bytes** into the proj buffer (the weight descriptors' offsets)
    /// @param out_offset    in **bf16 elements** into y
    void gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act);

    // attn.xclbin RTP addresses (FLM_Xclbin/Qwen3_5/bidirectional_attention_prefill, the same as
    // gated_attention_prefill's): q chunk start, q chunk count, and L_valid in the window slot.
    static constexpr uint32_t attn_l_begin_address = 61568;
    static constexpr uint32_t attn_l_chunks_address = 11904;
    static constexpr uint32_t attn_window_address = 61600;
    /// @brief rows of one attention q chunk; L_valid is padded to it
    static constexpr int ATTN_ROWS = 128;

    /// @brief bidirectional attention of q_rows query rows over L_valid keys on attn.xclbin: every q
    ///        chunk reads keys [0, ceil128(L_valid)) and the kernel masks keys >= L_valid.
    ///        Arguments: o (0), q (1), kv (2). q and o are [*, num_heads * DH] from element q_base /
    ///        o_base; kv is K [ceil128(L_valid), num_kv_heads * DH] then V, V at element kv_half.
    ///        Writes ceil128(q_rows) rows of o. q_rows = 0 means L_valid (one prompt over itself).
    ///        The query chunk count (RTP) and the key length are independent: that is what lets
    ///        prefix_bidir run the question rows (q_base = o_base = state rows, q_rows = question
    ///        rows) over all L_valid = state + question keys.
    ///        Any GQA group is run: a kv head's q heads go four at a time, a short last pass is padded
    ///        with duplicates of its real heads (group 6: 4 + 2 real heads, 8 passes per layer for
    ///        4 kv heads).
    void gen_bidirectional_attention(npu_sequence* seq, const uint32_t L_valid, const size_t q_base,
                                     const size_t o_base, const size_t kv_half, const uint32_t q_rows = 0);

    // conv1d.xclbin (FLM_Xclbin/Qwen3_8/conv1d_prefill, the build with a norm mode per core): the
    // depthwise causal conv1d + silu of the gated delta net, q and k l2-normalised on the way out.
    /// @brief core RTP address of the conv's L_buffer (length, norm mode)
    static constexpr uint32_t conv_l_buffer_address = 49664;
    /// @brief longest conv input: the x descriptor's 6-bit iteration field over 128-row blocks
    static constexpr uint32_t CONV_MAX_L_IN = 64 * 128;

    /// @brief y[L_out][D] = silu(conv1d(x[L_out + 3][D])) with q (first DQ channels) and k (next DQ)
    ///        l2-normalised per 128-channel head, no scale (the delta rule applies 1/sqrt(DH) itself).
    ///        The 3 rows ahead of the L_out rows are the conv state (zeros: a sequence start).
    ///        Arguments: y (0), x (1), w (2): w is [4][D] bf16 tap-major from element weight_offset.
    ///        D % 1024 == 0 is not needed, but DQ % 1024 must be 0 (a core column never straddles q / k).
    void gen_conv1d(npu_sequence* seq, const uint32_t L_out, const uint32_t D, const uint32_t DQ, const uint32_t weight_offset);
};
