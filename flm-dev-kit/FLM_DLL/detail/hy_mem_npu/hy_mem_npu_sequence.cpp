// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hy_mem_npu_sequence.cpp
/// \brief instruction sequences for the Hy-Memory-30B-A3B decode overlays
/// \note See hy_mem_npu_sequence.hpp. The DMA order inside each layer sequence is
///       the one the kernels were checked with; do not reorder it casually (the
///       MoE comments say what broke when it was).
#include "hy_mem_npu_sequence.hpp"

hy_mem_npu_sequence::hy_mem_npu_sequence(hy_mem_desc& desc, uint32_t MAX_L)
    : desc(desc), MAX_L(MAX_L) {}

// =============================================================================
// dense layer (layer 0)
// =============================================================================
void hy_mem_npu_sequence::gen_dense_layer_seq(npu_sequence* seq, int L) {
    assert(L > 0 && L <= (int)MAX_L);
    hy_mem_dense_layer_def& W = desc.dense;
    const size_t kv_col_row = desc.kv_col_row();
    seq->clear_cmds();

    for (int i = 0; i < attn_cus; i++) {
        seq->rtp_write(get_tile(attn_qk_rc[i][0], attn_qk_rc[i][1]), l_attn_qk_address, L);
        seq->rtp_write(get_tile(attn_kv_rc[i][0], attn_kv_rc[i][1]), l_attn_kv_address, L);
    }
    // IS_ATTN = 1: decoder layer (0 would be this overlay's LM-head mode, unused: lm_head is Q8_0)
    for (int i = 0; i < 16; i++) {
        npu_tiles tile = get_tile(proj_rc[i][0], proj_rc[i][1]);
        seq->rtp_write(tile, dense_proj_is_attn_address, 1);
        seq->rtp_write(tile, CT_lock_address_base + 16 * proj_rtp_lock_id, 1);   // set lock to 1
    }
    npu_tiles rms_tile = get_tile(dense_rms_rc[0], dense_rms_rc[1]);
    seq->rtp_write(rms_tile, dense_rms_is_attn_address, 1);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * proj_rtp_lock_id, 1);

    _send_hidden_states(seq);
    // input_layernorm then post_attention_layernorm. Not waited on: the rms tile
    // holds one D-long weight at a time, so the second half only lands after
    // attention, long after the weights below.
    seq->npu_dma_memcpy_nd(sizeof(bf16), w_rms_arg_idx, MM2S, xyr_tile, bd_1, it_channel_1,
        {0, 0, 0, 0}, {1, 1, 1, desc.D * 2}, {0, 0, 0, 1},
        0, 0, false, aggressive_cache);
    _send_rope_rms_weights(seq);

    // receive y
    seq->npu_dma_memcpy_nd(sizeof(bf16), xy_arg_idx, S2MM, xyr_tile, bd_10, it_channel_1,
        {0, 0, 0, 0}, {1, 1, 1, desc.D}, {0, 0, 0, 1},
        -1, 0, true);

    // the new K/V row: K through shim 3, V through shim 4, each as two halves
    // (heads 0, 1 then heads 2, 3) into its column's block, row L - 1. These
    // receive BDs go first, as in the production sequence.
    const uint32_t row_offset = (uint32_t)((L - 1) * kv_col_row);
    for (int kv = 0; kv < 2; kv++) {
        for (int col = 0; col < attn_cols; col++) {
            const uint32_t block_offset = (uint32_t)desc.kv_block_offset(col, kv, MAX_L);
            seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_idx, S2MM, dense_kv_write_shim[kv], npu_bd_id(col), it_channel_0,
                {0, 0, 0, block_offset + row_offset}, {1, 1, 1, (uint32_t)kv_col_row}, {0, 0, 0, 1},
                -1, 0, true);
        }
    }
    _move_weights_q4k(seq, desc.DK + desc.DV + desc.DQ, desc.D, W.attn_k.offset);   // k, v, q

    // each column's shim reads its own K block (channel 0) and V block (channel 1);
    // K reads start once both K writes of the new row are done (shim 3), V reads
    // once both V writes are (shim 4). 16-token chunks: the tail past L is masked.
    {
        const int L_padded = (L + hy_mem_desc::KV_CHUNK - 1) / hy_mem_desc::KV_CHUNK * hy_mem_desc::KV_CHUNK;
        assert(L_padded <= (int)MAX_L);
        const uint32_t data2move = (uint32_t)(L_padded * kv_col_row);
        const npu_bd_id read_bd[2] = {bd_4, bd_5};   // shims 3, 4 use BDs 0, 1 to write
        for (int kv = 0; kv < 2; kv++) {
            for (int col = 0; col < attn_cols; col++) {
                seq->npu_dma_wait(dense_kv_write_shim[kv], S2MM, it_channel_0);
            }
            for (int col = 0; col < attn_cols; col++) {
                seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_idx, MM2S, attn_shims[col], read_bd[kv],
                    kv == 0 ? it_channel_0 : it_channel_1,
                    {0, 0, 0, (uint32_t)desc.kv_block_offset(col, kv, MAX_L)}, {1, 1, 1, data2move}, {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache);
            }
        }
    }
    _move_weights_q4k(seq, desc.D, desc.DQ, W.attn_output.offset);

    // MLP: up/gate (27 rounds into the GLU), then down
    _move_weights_q4k(seq, 2 * desc.dense_width, desc.D, W.ffn_up.offset);
    _move_weights_q4k(seq, desc.D, desc.dense_width, W.ffn_down.offset);

    seq->npu_dma_wait(xyr_tile, S2MM, it_channel_1);   // y
    seq->cmds2seq();
}

// =============================================================================
// MoE layer (layers 1..47), one sequence for all of them
// =============================================================================
void hy_mem_npu_sequence::gen_moe_layer_seq(npu_sequence* seq, int L) {
    assert(L > 0 && L <= (int)MAX_L);
    hy_mem_moe_layer_def& W = desc.moe;
    const size_t kv_col_row = desc.kv_col_row();
    seq->clear_cmds();

    for (int i = 0; i < attn_cus; i++) {
        seq->rtp_write(get_tile(attn_qk_rc[i][0], attn_qk_rc[i][1]), l_attn_qk_address, L);
        seq->rtp_write(get_tile(attn_kv_rc[i][0], attn_kv_rc[i][1]), l_attn_kv_address, L);
    }
    for (int i = 0; i < 16; i++) {
        npu_tiles tile = get_tile(proj_rc[i][0], proj_rc[i][1]);
        seq->rtp_write(tile, moe_proj_rtp_address, 1);
        seq->rtp_write(tile, CT_lock_address_base + 16 * proj_rtp_lock_id, 1);   // set lock to 1
    }
    // (the router's layer number and expert bias: gen_router_rtp_seq, its own run)

    _send_hidden_states(seq);
    // input_layernorm | post_attention_layernorm (the router weight behind them goes on IT5)
    seq->npu_dma_memcpy_nd(sizeof(bf16), w_rms_arg_idx, MM2S, xyr_tile, bd_1, it_channel_1,
        {0, 0, 0, 0}, {1, 1, 1, desc.D * 2}, {0, 0, 0, 1},
        0, 0, true, aggressive_cache);
    seq->npu_dma_wait(xyr_tile, MM2S, it_channel_1);
    _send_rope_rms_weights(seq);

    // Router weight first: MT5 holds all of it (512 KB), so IT5 fills it while
    // attention runs, and the router + helper start on it as soon as x is there.
    // The host stored it in read order, so this is one straight read.
    seq->npu_dma_memcpy_nd(sizeof(bf16), w_rms_arg_idx, MM2S, IT5, bd_12, it_channel_0,
        {0, 0, 0, (uint32_t)(W.ffn_gate_inp.offset / sizeof(bf16))}, {1, 1, 1, desc.num_experts * desc.D}, {0, 0, 0, 1},
        10, 0, false, aggressive_cache);

    // receive y
    seq->npu_dma_memcpy_nd(sizeof(bf16), xy_arg_idx, S2MM, xyr_tile, bd_10, it_channel_1,
        {0, 0, 0, 0}, {1, 1, 1, desc.D}, {0, 0, 0, 1},
        -1, 0, true);

    // RoPE sends the new token's K (all KV heads) then V (all KV heads) through
    // IT3. Write each column's heads into that column's K / V block, row L - 1:
    // BDs 0..3 = K col 0, K col 1, V col 0, V col 1, in stream order.
    const uint32_t row_offset = (uint32_t)((L - 1) * kv_col_row);
    for (int kv = 0; kv < 2; kv++) {
        for (int col = 0; col < attn_cols; col++) {
            const uint32_t block_offset = (uint32_t)desc.kv_block_offset(col, kv, MAX_L);
            seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_idx, S2MM, attn_shims[0], npu_bd_id(kv * attn_cols + col), it_channel_0,
                {0, 0, 0, block_offset + row_offset}, {1, 1, 1, (uint32_t)kv_col_row}, {0, 0, 0, 1},
                -1, 0, true);
        }
    }
    _move_weights_q4k(seq, desc.DK + desc.DV + desc.DQ, desc.D, W.attn_k.offset);   // k, v, q

    // K reads start once both K writes of the new row are done, V reads once both V writes are
    {
        const int L_padded = (L + hy_mem_desc::KV_CHUNK - 1) / hy_mem_desc::KV_CHUNK * hy_mem_desc::KV_CHUNK;
        assert(L_padded <= (int)MAX_L);
        const uint32_t data2move = (uint32_t)(L_padded * kv_col_row);
        const npu_bd_id read_bd[2][attn_cols] = {{bd_4, bd_0}, {bd_5, bd_1}};   // IT3 uses BDs 0..3 to write
        for (int kv = 0; kv < 2; kv++) {
            for (int col = 0; col < attn_cols; col++) {
                seq->npu_dma_wait(attn_shims[0], S2MM, it_channel_0);
            }
            for (int col = 0; col < attn_cols; col++) {
                seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_idx, MM2S, attn_shims[col], read_bd[kv][col],
                    kv == 0 ? it_channel_0 : it_channel_1,
                    {0, 0, 0, (uint32_t)desc.kv_block_offset(col, kv, MAX_L)}, {1, 1, 1, data2move}, {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache);
            }
        }
    }
    _move_weights_q4k(seq, desc.D, desc.DQ, W.attn_output.offset);

    // Routed expert weight BDs on shim 0, 1, 6, 7: one linear BD per channel,
    // 4 / 5 for round 0 and 12 / 13 for round 1. Written here but not started;
    // the router CT patches in the chosen expert's address and starts them with
    // control packets.
    for (int i = 0; i < 4; i++) {
        _setup_expert_q4k(seq, i, 4, W.ffn_up_exps.offset);
        _setup_expert_q4k(seq, i, 12, W.ffn_up_exps.offset);
    }

    // shared expert up-gate, then down (down's last round pushed but not yet waited for)
    _move_weights_q4k(seq, 2 * desc.expert_width, desc.D, W.ffn_up_shexp.offset);
    _move_weights_q4k(seq, desc.D, desc.expert_width, W.ffn_down_shexp.offset, false);

    // Tell the router the expert BDs are ready. Only now: the router pushes the
    // routed-expert weight tasks onto the same shim queues (0, 1, 6, 7) as the
    // shared-expert ones above, and a router that got here first queued them in
    // front of the last shared-down weights (wrong output from the 2nd call on).
    npu_tiles router_CT = get_tile(router_rc[0], router_rc[1]);
    seq->npu_maskwrite(router_CT, 0x32038, 0x1, 0x1);
    seq->rtp_write(router_CT, CT_lock_address_base + 16 * router_bd_ready_lock_id, 1);   // set lock to 1
    _wait_proj_weights(seq);

    seq->npu_dma_wait(xyr_tile, S2MM, it_channel_1);   // y
    seq->cmds2seq();
}

void hy_mem_npu_sequence::gen_router_rtp_seq(npu_sequence* seq, int layer_idx, const std::vector<float>& bias) {
    assert(bias.size() == desc.num_experts);
    npu_tiles router_CT = get_tile(router_rc[0], router_rc[1]);
    seq->clear_cmds();
    // which layer this is: the router keeps one weight base address per layer
    seq->rtp_write(router_CT, router_layer_address, (uint32_t)layer_idx);
    // the expert bias, one write per expert (fp32 bit pattern). The router reads
    // it only after its logits, i.e. well inside the layer run that follows.
    for (uint32_t e = 0; e < desc.num_experts; e++) {
        uint32_t bits;
        memcpy(&bits, &bias[e], sizeof(bits));
        seq->rtp_write(router_CT, router_bias_address + 4 * e, bits);
    }
    seq->cmds2seq();
}

// =============================================================================
// shared helpers of the two layer overlays
// =============================================================================
void hy_mem_npu_sequence::_send_hidden_states(npu_sequence* seq) {
    // packet 0: to the rms tile
    seq->npu_dma_memcpy_nd(sizeof(bf16), xy_arg_idx, MM2S, xyr_tile, bd_0, it_channel_0,
        {0, 0, 0, 0}, {1, 1, 1, desc.D}, {0, 0, 0, 1},
        0, 0, false, aggressive_cache);
}

void hy_mem_npu_sequence::_send_rope_rms_weights(npu_sequence* seq) {
    // cos | sin | q_norm | k_norm (packet 1: to the RoPE tile, which holds all of it)
    seq->npu_dma_memcpy_nd(sizeof(bf16), w_rope_rms_arg_idx, MM2S, xyr_tile, bd_2, it_channel_0,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)desc.get_rope_rms_elems()}, {0, 0, 0, 1},
        1, 0, true, no_cache);
    seq->npu_dma_wait(xyr_tile, MM2S, it_channel_0);
}

/// One matrix through the 16 proj cores, 32 rows each per round: channel 0 of a
/// column's shim feeds its cores 0, 1 and channel 1 cores 2, 3 (reorder_cpy put
/// those blocks next to each other). Two BD sets ping-pong, so round r + 1 is
/// queued before round r is waited for.
/// \param wait_last_round false: the last round's tasks are pushed but not waited for
void hy_mem_npu_sequence::_move_weights_q4k(npu_sequence* seq, size_t Dout, size_t Din, size_t w_offset, bool wait_last_round) {
    const uint32_t m = QXNX_ROW_BLOCK_SIZE;
    const uint32_t a_block_size = hy_mem_desc::Q4K_BLOCK_BYTES / sizeof(bf16);   // in bf16 elements
    const uint32_t blocks_per_row = Din / QXNX_COL_BLOCK_SIZE;
    const uint32_t cores = 16;
    const uint32_t w_base = (uint32_t)(w_offset / sizeof(bf16));
    assert(Dout % (m * cores) == 0);
    for (size_t round = 0; round < Dout / m / cores; round++) {
        const uint32_t bd_offset = (round % 2) * 8;
        for (uint32_t col = 0; col < 4; col++) {
            const uint32_t first_row = round * cores + col * 4;   // this column's first block row
            seq->npu_dma_memcpy_nd(sizeof(bf16), w_proj_arg_idx, MM2S, mvm_tiles[col], npu_bd_id(bd_1 + bd_offset), it_channel_0,
                {0, 0, 0, first_row * a_block_size * blocks_per_row + w_base}, {1, 1, 1, 2 * blocks_per_row * a_block_size}, {0, 0, 0, 1},
                -1, 0, true, aggressive_cache);
            seq->npu_dma_memcpy_nd(sizeof(bf16), w_proj_arg_idx, MM2S, mvm_tiles[col], npu_bd_id(bd_2 + bd_offset), it_channel_1,
                {0, 0, 0, (first_row + 2) * a_block_size * blocks_per_row + w_base}, {1, 1, 1, 2 * blocks_per_row * a_block_size}, {0, 0, 0, 1},
                -1, 0, true, aggressive_cache);
        }
        if (round > 0) {
            _wait_proj_weights(seq);
        }
    }
    if (wait_last_round) {
        _wait_proj_weights(seq);
    }
}

/// Wait for one weight task on both channels of each proj shim (0, 1, 6, 7).
void hy_mem_npu_sequence::_wait_proj_weights(npu_sequence* seq) {
    for (uint32_t col = 0; col < 4; col++) {
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
    }
}

/// One routed expert's weight BDs on proj shim `col` (0-3): channel 0 gets BD
/// bd_id, channel 1 BD bd_id + 1, each one linear read of half an expert (see
/// hy_mem_desc::load_moe_layer). Points at expert 0; the router patches in the
/// chosen expert's address and pushes the BDs onto the queues.
void hy_mem_npu_sequence::_setup_expert_q4k(npu_sequence* seq, int col, int bd_id, size_t const_w_byte_offset) {
    const size_t gate_up_bytes = 2 * (size_t)(desc.expert_width / QXNX_ROW_BLOCK_SIZE) * (desc.D / QXNX_COL_BLOCK_SIZE) * hy_mem_desc::Q4K_BLOCK_BYTES;
    const size_t down_bytes = (size_t)(desc.D / QXNX_ROW_BLOCK_SIZE) * (desc.expert_width / QXNX_COL_BLOCK_SIZE) * hy_mem_desc::Q4K_BLOCK_BYTES;
    const uint32_t channel_bytes = (uint32_t)((gate_up_bytes + down_bytes) / 2);
    constexpr uint32_t SHIM_6_CONTROL_RES_PKT = 30;   // the router's control response packet id (kernels)
    for (int ch = 0; ch < 2; ch++) {
        seq->npu_dma_memcpy_nd(sizeof(uint8_t), w_proj_arg_idx, MM2S, mvm_tiles[col], npu_bd_id(bd_id + ch),
            ch == 0 ? it_channel_0 : it_channel_1,
            {0, 0, 0, (uint32_t)(const_w_byte_offset + (size_t)ch * channel_bytes)}, {1, 1, 1, channel_bytes}, {0, 0, 0, 1},
            -1, 0, false, aggressive_cache, SHIM_6_CONTROL_RES_PKT, false);
    }
}

// =============================================================================
// LM head (lm_head.xclbin): final RMS norm + Q8_0 logits over 8 columns
// =============================================================================
void hy_mem_npu_sequence::gen_lm_head_seq(npu_sequence* seq) {
    static constexpr npu_tiles ITs[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    static constexpr int COLS = 8;
    static constexpr int M_PER_COL = 32 * 4;             // 4 cores x 32 rows
    static constexpr int M_PER_ROUND = M_PER_COL * COLS;
    // the two flows sharing IT0's MM2S 0 (X_PKT_ID / W_PKT_ID in iron/q4k_mm.py);
    // every other shim MM2S port is circuit-switched
    static constexpr int X_PKT_ID = 0;
    static constexpr int W_PKT_ID = 1;
    static constexpr int NO_PKT = -1;
    // rounds of shim BD ping-pong; also how far the y waits lag the y issues
    static constexpr int LOOKAHEAD = 3;
    const uint32_t vocab = desc.vocab_size_padded;
    const uint32_t D = desc.D;
    assert(vocab % M_PER_ROUND == 0);
    const int rounds = vocab / M_PER_ROUND;
    // Q8_0: 8.5 bits per weight; in bf16 elements
    const size_t W_PER_COL = size_t(M_PER_COL * D * 8.5 / 8) / sizeof(bf16);
    const size_t W_PER_ROUND = W_PER_COL * COLS;
    // each column's weights are split over two shim MM2S channels, one per pair
    // of cores; load_lm_head laid the two halves out back to back
    const size_t W_PER_STREAM = W_PER_COL / 2;

    seq->clear_cmds();
    // x | final norm weight
    seq->npu_dma_memcpy_nd(sizeof(bf16), lm_x_arg_idx, MM2S, ITs[0], bd_0, it_channel_0,
        {0, 0, 0, 0}, {1, 1, 1, D * 2}, {0, 0, 0, 1},
        X_PKT_ID, 0, false, aggressive_cache);

    for (int r = 0; r < rounds; r++) {
        const int bd_offset = (r % (LOOKAHEAD + 1)) * 3;
        const npu_bd_id bd_y = npu_bd_id(bd_1 + bd_offset);
        const npu_bd_id bd_wa = npu_bd_id(bd_2 + bd_offset);
        const npu_bd_id bd_wb = npu_bd_id(bd_3 + bd_offset);
        for (int col = 0; col < COLS; col++) {
            const size_t w_col_offset = r * W_PER_ROUND + col * W_PER_COL;
            const uint32_t y_offset = r * M_PER_ROUND + col * M_PER_COL;
            seq->npu_dma_memcpy_nd(sizeof(bf16), lm_y_arg_idx, S2MM, ITs[col], bd_y, it_channel_0,
                {0, 0, 0, y_offset}, {1, 1, 1, (uint32_t)M_PER_COL}, {0, 0, 0, 1},
                -1, 0, true, aggressive_cache);
            // group a (cores 0/1) on channel 1, group b (cores 2/3) on channel 0;
            // only IT0's channel 0 is shared with the x broadcast and needs a packet id
            seq->npu_dma_memcpy_nd(sizeof(bf16), lm_w_arg_idx, MM2S, ITs[col], bd_wa, it_channel_1,
                {0, 0, 0, (uint32_t)w_col_offset}, {1, 1, 1, (uint32_t)W_PER_STREAM}, {0, 0, 0, 1},
                NO_PKT, 0, false, aggressive_cache);
            seq->npu_dma_memcpy_nd(sizeof(bf16), lm_w_arg_idx, MM2S, ITs[col], bd_wb, it_channel_0,
                {0, 0, 0, (uint32_t)(w_col_offset + W_PER_STREAM)}, {1, 1, 1, (uint32_t)W_PER_STREAM}, {0, 0, 0, 1},
                col == 0 ? W_PKT_ID : NO_PKT, 0, false, aggressive_cache);
            // four BD sets in flight: a round waits for the y of three rounds back,
            // which keeps three rounds of weights queued ahead of the sequencer
            if (r > LOOKAHEAD - 1) {
                seq->npu_dma_wait(ITs[col], S2MM, it_channel_0);
            }
        }
    }
    for (int col = 0; col < COLS; col++) {
        for (int i = 0; i < LOOKAHEAD; i++) {
            seq->npu_dma_wait(ITs[col], S2MM, it_channel_0);
        }
    }
    seq->cmds2seq();
}
