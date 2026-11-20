// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hy_mem_npu_sequence.hpp
/// \brief instruction sequences for the Hy-Memory-30B-A3B decode overlays
/// \author FastFlowLM Team
/// \date 2026-10-26
/// \note Three bitstreams, one sequence family each:
///         dense_layer.xclbin  gen_dense_layer_seq   (layer 0)
///         moe_layer.xclbin    gen_moe_layer_seq     (layers 1..47, one sequence for all)
///                             gen_router_rtp_seq    (one per MoE layer, built once)
///         lm_head.xclbin      gen_lm_head_seq
///       The layer sequences depend on the context length and are rebuilt once per
///       token; the router one only carries the layer's number and expert bias,
///       which never change, so it runs as its own small run in front of every MoE
///       layer (the router reads both long after the layer run has started).
///       Ported from FLM_Xclbin/Hy_mem_30BA3B/{dense,moe}_decoding/host/decoding_layer.cpp
///       and lm_head_npu_bin/host/host.cpp.
#pragma once
#include <string>
#include <vector>
#include "npu_utils/npu_utils.hpp"
#include "hy_mem_model_def.hpp"

class hy_mem_npu_sequence {
public:
    // kernel argument order of both layer overlays: (x, proj, rms, rope_rms, kv_cache)
    static constexpr int xy_arg_idx = 0;
    static constexpr int w_proj_arg_idx = 1;
    static constexpr int w_rms_arg_idx = 2;
    static constexpr int w_rope_rms_arg_idx = 3;
    static constexpr int kv_cache_arg_idx = 4;
    // lm_head overlay: (y, w, x)
    static constexpr int lm_y_arg_idx = 0;
    static constexpr int lm_w_arg_idx = 1;
    static constexpr int lm_x_arg_idx = 2;

    hy_mem_npu_sequence(hy_mem_desc& desc, uint32_t MAX_L);

    /// \brief the KV block offsets depend on MAX_L; call before regenerating
    void set_max_length(uint32_t MAX_L) { this->MAX_L = MAX_L; }

    /// \param L tokens in the cache after this call (the new token is row L - 1)
    void gen_dense_layer_seq(npu_sequence* seq, int L);
    void gen_moe_layer_seq(npu_sequence* seq, int L);
    /// \brief the router's layer number and fp32 expert bias, for one MoE layer
    void gen_router_rtp_seq(npu_sequence* seq, int layer_idx, const std::vector<float>& bias);
    void gen_lm_head_seq(npu_sequence* seq);

private:
    hy_mem_desc& desc;
    uint32_t MAX_L;

    // ---- floorplan (same on both layer overlays unless noted) ----
    static constexpr npu_tiles xyr_tile = IT2;                       // x / y / rms / rope
    static constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};  // proj weight shims
    static constexpr int attn_cus = 4;                               // CU i holds KV head i
    static constexpr int attn_cols = hy_mem_desc::ATTN_COLS;
    static constexpr int attn_qk_rc[attn_cus][2] = {{2, 3}, {4, 3}, {2, 4}, {4, 4}};
    static constexpr int attn_kv_rc[attn_cus][2] = {{3, 3}, {5, 3}, {3, 4}, {5, 4}};
    static constexpr npu_tiles attn_shims[attn_cols] = {IT3, IT4};
    // dense: RoPE writes the new K row through shim 3 and the new V row through shim 4
    static constexpr npu_tiles dense_kv_write_shim[2] = {IT3, IT4};
    // proj (and dense rms) cores wait on lock 6 for their RTP every call
    static constexpr int proj_rc[16][2] = {{2, 0}, {3, 0}, {4, 0}, {5, 0},
                                           {2, 1}, {3, 1}, {4, 1}, {5, 1},
                                           {2, 6}, {3, 6}, {4, 6}, {5, 6},
                                           {2, 7}, {3, 7}, {4, 7}, {5, 7}};
    static constexpr int dense_rms_rc[2] = {2, 2};
    static constexpr int router_rc[2] = {4, 5};                      // MoE router CT
    static constexpr size_t CT_lock_address_base = 0x000001F000;
    static constexpr int proj_rtp_lock_id = 6;
    static constexpr int router_bd_ready_lock_id = 10;               // expert BDs written
    static constexpr uint32_t moe_proj_rtp_address = 17408;          // moe proj cores' RTP word

    // ---- RTP addresses of the two layer overlays ----
    // Each xclbin build writes these to its own xclbins/rtp_addresses.txt; they are
    // copied here as constants, the way every other engine keeps them (compare
    // qwen3_6_moe_npu_sequence.hpp). Every core of a kind holds its RTP at the same
    // offset inside its own data memory, so one value covers all of them -- the
    // files list it once per core, all identical.
    //
    // RE-CHECK THESE AFTER REBUILDING dense_decoding OR moe_decoding. If a rebuild
    // moves an RTP, the host writes to a stale address and the result is wrong
    // output or a hang, not an error: the host writes RTPs at fixed addresses, and
    // the kernel's bitstream rule records where the build put them in
    // xclbins/rtp_addresses.txt (a Gemma4 decode hang was exactly this mismatch).
    // Diff the new rtp_addresses.txt against the values below.
    static constexpr uint32_t l_attn_qk_address = 5120;              // RTP_L_attn_qk_core_*, both overlays
    static constexpr uint32_t l_attn_kv_address = 34816;             // RTP_L_attn_kv_core_*, both overlays
    static constexpr uint32_t dense_proj_is_attn_address = 33280;    // RTP_PROJ_IS_ATTN_BUFFER_*, dense
    static constexpr uint32_t dense_rms_is_attn_address = 53248;     // RTP_RMS_IS_ATTN_BUFFER, dense
    static constexpr uint32_t router_bias_address = 53248;           // RTP_router_expert_bias, moe
    static constexpr uint32_t router_layer_address = 53760;          // RTP_router_layer, moe

    void _send_hidden_states(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _move_weights_q4k(npu_sequence* seq, size_t Dout, size_t Din, size_t w_offset, bool wait_last_round = true);
    void _wait_proj_weights(npu_sequence* seq);
    void _setup_expert_q4k(npu_sequence* seq, int col, int bd_id, size_t const_w_byte_offset);
};
