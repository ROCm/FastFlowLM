// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "minicpm_v_4_7_npu_sequence.hpp"

minicpm_v_4_7_npu_sequence::minicpm_v_4_7_npu_sequence(minicpm_v_desc& desc, LM_Config config, uint32_t MAX_L)
    : desc(&desc){
    conv_w_size_bf16 = desc.get_conv_w_size() / sizeof(bf16);

    s_offset = desc.get_s_offset();
    this->kv_cache_size = (uint32_t)desc.get_kv_cache_size(MAX_L);
    this->MAX_L = MAX_L;

}

void minicpm_v_4_7_npu_sequence::set_max_length(const uint32_t MAX_L){
    this->MAX_L = MAX_L;
    this->kv_cache_size = MAX_L * (desc->DK + desc->DV);
}

void minicpm_v_4_7_npu_sequence::gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_attn_layer, bool is_first){
    if (is_attn_layer) {
        _gen_sequence(seq, L);
    }
    else {
        _gen_linear_sequence(seq, is_first);
    }
}

void minicpm_v_4_7_npu_sequence::gen_seq_conv1d(
    npu_sequence* seq,
    const uint32_t L_PADDED, // assume padded, L_OUT + (window_size-1)
    const uint32_t D,
    const uint32_t L_OUT,
    const uint32_t weight_offset
){

    auto round_up_to_multiple = [](int x, int multiple) -> int {
        if (multiple == 0) {
            return x; // Cannot divide by zero
        }
        // This uses integer division to achieve the rounding
        return ((x + multiple - 1) / multiple) * multiple;
    };

    npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    seq->clear_cmds();

    // Note: this is bind to be specific per xclbin

    const int l_address = 49664;
    uint32_t window_size = 4;
    const uint32_t d_per_column = 1024;

    assert(D % d_per_column == 0);

    uint32_t D_div_d_per_column = D / d_per_column;

    uint32_t num_compute_row = 4;
    uint32_t num_compute_col = 8;

    constexpr uint32_t conv1d_arg_idx = 2;
    constexpr uint32_t conv1d_in_out_arg_idx = 1;

    uint32_t X_transaction_block_size = 256;
    uint32_t Y_transaction_block_size = 1024;

    uint32_t L_DIV_PADD_REQUIREMENT = L_PADDED;
    int num_input_L_blocks = L_DIV_PADD_REQUIREMENT / X_transaction_block_size;
    int input_L_remainder  = L_DIV_PADD_REQUIREMENT % X_transaction_block_size;

    uint32_t w_bd_id = 2;
    uint32_t x_input_bd_id = 3;

    std::vector<int> y_output_ping_pong(8, 0);
    std::vector<int> y_ping_pong_flag(8, 0);

    uint32_t num_round = round_up_to_multiple(D_div_d_per_column, num_compute_col) / num_compute_col;
    assert(num_round >= 1);

    for(int whole_array_round = 0; whole_array_round< num_round; whole_array_round++){

        auto num_valid_col_this_round = std::min(num_compute_col, D_div_d_per_column - whole_array_round * num_compute_col);

        // first write the rtp command for this round
        for(uint32_t row = 2; row < 2 + num_compute_row; row++){
            for(uint32_t col_idx = 0; col_idx < num_valid_col_this_round; col_idx++){
                npu_tiles tile = get_tile(row, col_idx);
                seq->rtp_write(tile, l_address, L_OUT);
            }
        }

        for(int col_idx = 0; col_idx < num_valid_col_this_round; col_idx ++){
            size_t w_offset = (whole_array_round * num_compute_col + col_idx) * d_per_column + weight_offset;

            seq->npu_dma_memcpy_nd(
                sizeof(bf16), conv1d_arg_idx,
                MM2S, IT[col_idx],
                (npu_bd_id)(w_bd_id), it_channel_1,
                {0,0,0, (uint32_t)w_offset},
                {1, 1, window_size, (uint32_t)d_per_column},
                {0, 0, D, 1},
                -1, 0, false
            );
        }

        // Something special, we do a single transaction
        // now, send x and receive the output
        for(int col_idx = 0; col_idx < num_valid_col_this_round; col_idx++){

            uint32_t x_offset = (whole_array_round * num_compute_col + col_idx) * d_per_column;
            // NOTE: the logic will be more complexed if num_input_L_blocks ends up overflow, which probably not possible with small models
            if (num_input_L_blocks > 0) {
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), conv1d_in_out_arg_idx,
                    MM2S, IT[col_idx],
                    (npu_bd_id)(x_input_bd_id), it_channel_0,
                    {0,0,0, x_offset},
                    {1, (uint32_t)num_input_L_blocks, (uint32_t)X_transaction_block_size, (uint32_t)d_per_column},
                    {0, (uint32_t)(X_transaction_block_size * D), (uint32_t)D, 1},
                    -1, 0, false
                );

                uint32_t x_remainder_input_offfset = x_offset + (X_transaction_block_size * D) * num_input_L_blocks;
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), conv1d_in_out_arg_idx,
                    MM2S, IT[col_idx],
                    (npu_bd_id)(x_input_bd_id + 8), it_channel_0,
                    {0,0,0, x_remainder_input_offfset},
                    {1, 1, (uint32_t)input_L_remainder, (uint32_t)d_per_column},
                    {0, 0, (uint32_t)D, 1},
                    -1, 0, false
                );
            } else {
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), conv1d_in_out_arg_idx,
                    MM2S, IT[col_idx],
                    (npu_bd_id)(x_input_bd_id), it_channel_0,
                    {0,0,0, x_offset},
                    {1, 1, L_DIV_PADD_REQUIREMENT, (uint32_t)d_per_column},
                    {0, 0, (uint32_t)D, 1},
                    -1, 0, false
                );
            }
        }
        // now, receive Y

        uint32_t receive_round = round_up_to_multiple(L_OUT, Y_transaction_block_size) / Y_transaction_block_size;
        for(int receive_r = 0; receive_r< receive_round; receive_r++){

            int bd_offset = (receive_r % 2) * 8;
            for(int col_idx = 0; col_idx < num_valid_col_this_round; col_idx++){
                size_t o_offset = (col_idx + whole_array_round* num_compute_col)* d_per_column + receive_r * Y_transaction_block_size * D;

                if(receive_r +1 == receive_round && L_OUT % Y_transaction_block_size != 0){
                    uint32_t L_remain = L_OUT  % Y_transaction_block_size;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[col_idx],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)o_offset},
                        {1, 1,  (uint32_t)L_remain, (uint32_t)d_per_column},
                        {0, 0,  (uint32_t)D,        (uint32_t)1},
                        -1, 0, true
                    );
                }else{
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[col_idx],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)o_offset},
                        {1, 1,  (uint32_t)Y_transaction_block_size, (uint32_t)d_per_column},
                        {0, 0,  (uint32_t)D, (uint32_t)1},
                        -1, 0, true
                    );
                }
            }
            if (receive_r > 0){
                for (int col = 0; col < num_valid_col_this_round; col++){
                    seq->npu_dma_wait(
                        IT[col],
                        S2MM,
                        it_channel_0
                    );
                }
            }// round > 0
        }
        for (int col = 0; col < num_valid_col_this_round; col++){
            seq->npu_dma_wait(
                IT[col],
                S2MM,
                it_channel_0
            );
        }
    }

    seq->cmds2seq();
}

void minicpm_v_4_7_npu_sequence::gen_mha_engine_seq_bfp16(
    npu_sequence* seq,
    const uint32_t L_begin,
    const uint32_t L_wanted,
    const uint32_t kv_bfp16_rows
){
    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};

    const int Heads = desc->num_heads;
    const int Num_of_d_in_KV_cache = desc->num_kv_heads;
    // a round: 8 cores per q head x 32 query rows each
    const int data_per_round = 32 * 8;
    const int rows_per_memtile = data_per_round / 2;
    const uint32_t row_bytes = desc->DH * 9 / 8;
    assert((L_wanted) % data_per_round == 0);
    const int using_window_size = MAX_L;
    const int Q_chunks = L_wanted / data_per_round;
    seq->clear_cmds();

    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, mha_bfp16_l_begin, L_begin);
            seq->rtp_write(tile, mha_bfp16_l_end, Q_chunks);
            seq->rtp_write(tile, mha_bfp16_window_size, using_window_size);
        }
    }
    // each CU has 8 CTs and works on 1 head.
    const int num_cu = 4;
    for (int head = 0; head < Heads / num_cu; head++){
        for (int round = 0; round < Q_chunks; round++){
            const int Lq_current = L_begin + round * data_per_round;
            const int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            const int kv_end = (Lq_current + data_per_round + 128 - 1) / 128 * 128;
            const int kv_length = kv_end - kv_begin;

            const int bd_offset = (round % 2) * 8;
            for (int cu = 0; cu < num_cu; cu++){
                const int head_offset = head * num_cu + cu;
                for (int col = 0; col < 2; col++){
                    // receive y
                    size_t y_offset = head_offset * desc->DH + (round * data_per_round + col * rows_per_memtile) * desc->DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)rows_per_memtile, (uint32_t)(desc->DH)},
                        {0, 0, (uint32_t)desc->DH * Heads, 1},
                        -1, 0, true
                    );
                }
                // send q: tiles [head][L_wanted / 16][DH / 8][2][72 B], the two halves of a round
                const uint32_t q_bytes = (head_offset * L_wanted + round * data_per_round) * row_bytes;
                for (int half = 0; half < 2; half++) {
                    seq->npu_dma_memcpy_nd(
                        1, 1,
                        MM2S, IT[cu][0],
                        (npu_bd_id)(bd_offset + 1 + half), half == 0 ? it_channel_0 : it_channel_1,
                        {0, 0, 0, q_bytes + half * (uint32_t)rows_per_memtile * row_bytes},
                        {1, 1, (uint32_t)rows_per_memtile * row_bytes / 512, 512},
                        {0, 0, 512, 1},
                        -1, 0, false
                    );
                }
            } // cu

            // K tiles [kv head][kv_bfp16_rows / 16][DH / 8][2][72 B], then V^T tiles in the same layout
            const int kv_head_offset = head;
            const int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;
            for (int part = 0; part < 2; part++) {
                const uint32_t kv_bytes = ((part * Num_of_d_in_KV_cache + kv_head_offset_in_chunk) * kv_bfp16_rows + kv_begin) * row_bytes;
                seq->npu_dma_memcpy_nd(
                    1, 2,
                    MM2S, IT[1][1],
                    (npu_bd_id)(bd_offset + 3 + part), part == 0 ? it_channel_0 : it_channel_1,
                    {0, 0, 0, kv_bytes},
                    {1, (uint32_t)kv_length / 128, 128 * row_bytes / 512, 512},
                    {0, 128 * row_bytes, 512, 1},
                    -1, 0, false
                );
            }
            if (round > 0){
                for (int cu = 0; cu < num_cu; cu++){
                    for (int col = 0; col < 2; col++){
                        seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
                    }
                }
            }
        } // round
        for (int cu = 0; cu < num_cu; cu++){
            for (int col = 0; col < 2; col++){
                seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
            }
        }
    } // head
    seq->cmds2seq();
}

void minicpm_v_4_7_npu_sequence::gen_mha_engine_seq(
    npu_sequence* seq,
    const uint32_t L_begin,
    const uint32_t L_wanted
){
    static constexpr int lc = 16;
    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};

    const int Heads = desc->num_heads;
    const int Num_of_d_in_KV_cache = desc->num_kv_heads;
    assert((L_wanted) % (lc * 8) == 0);
    int using_window_size = MAX_L;
    const int Q_chunks = L_wanted / (lc * 8);
    seq->clear_cmds();

    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, Q_chunks);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }
    // each CU has 8 CTs and works on 1 head.
    const int data_per_round = lc * 8;
    const int down_rounds = Q_chunks;

    int num_cu = 4;
    for (int head = 0; head < Heads / num_cu; head++){
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_end = (Lq_current + 2 * data_per_round - 1) / data_per_round * data_per_round;
            int kv_length = kv_end - kv_begin;

            int bd_offset = (round % 2) * 8;
            for (int cu = 0; cu < num_cu; cu++){
                int head_offset = head * num_cu + cu;
                for (int col = 0; col < 2; col++){
                    // receive y
                    size_t y_offset = head_offset * desc->DH + (round * data_per_round + col * lc * 4) * desc->DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)4 * lc, (uint32_t)(desc->DH)},
                        {0, 0, (uint32_t)desc->DH * Heads, 1},
                        -1, 0, true
                    );
                }
                // send q
                size_t q_offset = head_offset * desc->DH + round * data_per_round * desc->DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(desc->DH)},
                    {0, 0, (uint32_t)desc->DH * Heads, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * desc->DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(desc->DH)},
                    {0, 0, (uint32_t)desc->DH * Heads, 1},
                    -1, 0, false
                );
            } // cu

            int kv_head_offset = head;
            int kv_chunk_offset = kv_head_offset / Num_of_d_in_KV_cache;
            int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;

            size_t k_offset = kv_chunk_offset * MAX_L * desc->DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * desc->DH + kv_begin * desc->DH * Num_of_d_in_KV_cache;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[1][1],
                (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(desc->DH)},
                {0, (uint32_t)128 * desc->DH * Num_of_d_in_KV_cache, (uint32_t)desc->DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            size_t v_offset = kv_chunk_offset * MAX_L * desc->DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * desc->DH + kv_begin * desc->DH * Num_of_d_in_KV_cache + this->kv_cache_size / 2;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[1][1],
                (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(desc->DH)},
                {0, (uint32_t)128 * desc->DH * Num_of_d_in_KV_cache, (uint32_t)desc->DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            if (round > 0){
                for (int cu = 0; cu < num_cu; cu++){
                    for (int col = 0; col < 2; col++){
                        seq->npu_dma_wait(
                            IT[cu][col],
                            S2MM,
                            it_channel_0
                        );
                    } // col
                } // cu
            }// round > 0
        }// round
        for (int cu = 0; cu < num_cu; cu++){
            for (int col = 0; col < 2; col++){
                seq->npu_dma_wait(
                    IT[cu][col],
                    S2MM,
                    it_channel_0
                );
            } // col
        } // cu
    } // head
    seq->cmds2seq();
}

// ---------------------------------------------------------------------------------------------------
// Runtime-m mm. The mm array works on blocks of 4 core rows x m rows; the fixed design is built for
// m = 128 (512-row blocks). The runtime-m xclbin takes m per dispatch (16 <= m <= 128, steps of 16):
//   * RTP word 1 of every core = m / 16 (the kernel reads it after its first weight tile arrived),
//   * the shim x / y descriptors (rows),
//   * the memtile DMA descriptors, which the design programs for m = 128: the lengths (and the row wrap of
//     the y output descriptors) are rewritten. Channels park on a descriptor they have already fetched, so
//     the memtile x / y channels are reset, their locks put back to the initial values (a reset does not
//     cancel an acquire pending in the lock module; writing the maximum value grants it first), the
//     descriptors rewritten and the channels started again as the design's CDO starts them.
// A dispatch with m < 128 does that first and puts the design's configuration back after its last
// transfer, so m = 128 dispatches (the common case of long prompts) carry no extra commands and the
// array is always left as the design makes it. FLM_Xclbin/dequant_mm_bfp16_rtm2_glu_512x512x512/
// check_memtile_bds.py verifies the ids / values used here in the built CDO.
// ---------------------------------------------------------------------------------------------------
minicpm_v_4_7_npu_sequence::mm_geometry minicpm_v_4_7_npu_sequence::mm_pick_geometry(uint32_t rows){
    mm_geometry g;
    if (rows == 0) rows = 1;
    g.blocks = (rows + 511) / 512;
    g.m = 16 * ((rows + g.blocks * 64 - 1) / (g.blocks * 64));
    g.covered = g.blocks * 4 * g.m;
    // A reduced-row dispatch pays ~50 us of memtile reconfiguration (before and after); below ~128 rows saved
    // that is a loss (measured: 400 rows -> 448 covered was 4% slower), so keep the zero-overhead full blocks.
    if (g.blocks * 512 - g.covered < 128) { g.m = 128; g.covered = g.blocks * 512; }
    return g;
}

namespace {
    constexpr uint32_t MT_BD_BASE = 0xA0000, MT_BD_STRIDE = 0x20, MT_LOCK_BASE = 0xC0000, MT_LOCK_STRIDE = 0x10;
    struct mt_channel { uint32_t ctrl, start_bd; };
    constexpr mt_channel MT_X_CH[] = {{0xA0600, 0}, {0xA0630, 2}};                                                   // S2MM0, MM2S0
    constexpr mt_channel MT_Y_CH[] = {{0xA0608, 24}, {0xA0610, 6}, {0xA0618, 26}, {0xA0620, 8}, {0xA0638, 28}};      // S2MM1-4, MM2S1
    constexpr uint32_t MT_X_S2MM[] = {0, 1}, MT_X_MM2S[] = {2, 3, 4, 5}, MT_Y_S2MM[] = {6, 7, 8, 9, 24, 25, 26, 27}, MT_Y_MM2S[] = {28, 29};
    constexpr uint32_t MT_X_LOCKS[] = {0, 1, 2, 3, 4}, MT_Y_LOCKS[] = {5, 6, 7, 8, 9}, MT_LOCK_INIT[] = {2, 0, 0, 0, 0};
    // mm_rtm2.xclbin / mm_rtm2_glu.xclbin (dequant_mm_bfp16_rtm2_glu): the y join takes 4 producer credits per buffer, locks 5..8 only
    constexpr uint32_t MT_Y_LOCKS_PAR[] = {5, 6, 7, 8}, MT_Y_INIT_PAR[] = {4, 4, 0, 0};
}

// GLU mode (rtp_act 3, mm_rtm_glu.xclbin): a core sends 32 columns per row (silu(gate) * up of its two n slices) instead
// of 64, so three words of the y descriptors differ from the design's: S2MM D2 wrap (the 8x8 n tiles of a row group,
// 8 -> 4), MM2S D0 wrap (row length in words, 32 -> 16) and the buffer lengths. Strides and base addresses stay.
void minicpm_v_4_7_npu_sequence::mm_rearm(npu_sequence* seq, const uint32_t m, const bool yjoin_par, const uint32_t ymode){
    auto bd_word = [](uint32_t bd, uint32_t word){ return MT_BD_BASE + MT_BD_STRIDE * bd + 4 * word; };
    auto lock_reg = [](uint32_t id){ return MT_LOCK_BASE + MT_LOCK_STRIDE * id; };
    for (uint32_t col = 0; col < 8; col++){
        const npu_tiles mt = get_tile(1, col);
        const bool has_x = (col % 2) == 0;     // the x of core row r enters through memtile 2r
        if (has_x) for (auto& c : MT_X_CH) seq->rtp_write(mt, c.ctrl, 2);
        for (auto& c : MT_Y_CH) seq->rtp_write(mt, c.ctrl, 2);
        if (has_x) for (uint32_t i = 0; i < 5; i++){ seq->rtp_write(mt, lock_reg(MT_X_LOCKS[i]), 63); seq->rtp_write(mt, lock_reg(MT_X_LOCKS[i]), MT_LOCK_INIT[i]); }
        if (yjoin_par) for (uint32_t i = 0; i < 4; i++){ seq->rtp_write(mt, lock_reg(MT_Y_LOCKS_PAR[i]), 63); seq->rtp_write(mt, lock_reg(MT_Y_LOCKS_PAR[i]), MT_Y_INIT_PAR[i]); }
        else for (uint32_t i = 0; i < 5; i++){ seq->rtp_write(mt, lock_reg(MT_Y_LOCKS[i]), 63); seq->rtp_write(mt, lock_reg(MT_Y_LOCKS[i]), MT_LOCK_INIT[i]); }
        if (has_x){
            for (uint32_t bd : MT_X_S2MM) seq->rtp_write(mt, bd_word(bd, 0), m * 256);
            for (uint32_t bd : MT_X_MM2S) seq->rtp_write(mt, bd_word(bd, 0), m * 128);
        }
        const bool glu = ymode == MM_Y_GLU;
        for (uint32_t bd : MT_Y_S2MM) {
            seq->rtp_write(mt, bd_word(bd, 0), glu ? m * 16 : m * 32);
            if (ymode != MM_Y_PLAIN) seq->rtp_write(mt, bd_word(bd, 4), glu ? ((4u << 17) | 3u) : ((8u << 17) | 3u));
        }
        for (uint32_t bd : MT_Y_MM2S) {
            seq->rtp_write(mt, bd_word(bd, 0), glu ? m * 64 : m * 128);
            if (ymode != MM_Y_PLAIN) seq->rtp_write(mt, bd_word(bd, 2), glu ? (16u << 17) : (32u << 17));
            seq->rtp_write(mt, bd_word(bd, 4), (m << 17) | 0x1F);
        }
        if (has_x) for (auto& c : MT_X_CH){ seq->rtp_write(mt, c.ctrl, 0); seq->rtp_write(mt, c.ctrl + 4, c.start_bd); }
        for (auto& c : MT_Y_CH){ seq->rtp_write(mt, c.ctrl, 0); seq->rtp_write(mt, c.ctrl + 4, c.start_bd); }
    }
}

void minicpm_v_4_7_npu_sequence::mm_set_core_rows(npu_sequence* seq, const uint32_t m){
    for (int row = 2; row < 6; row++){
        const uint32_t ctp = (row % 2 == 0) ? CT_rtp_address_even_row : CT_rtp_address_odd_row;
        for (uint32_t col = 0; col < 8; col++) seq->rtp_write(get_tile(row, col), ctp + 4, m / 16);
    }
}

void minicpm_v_4_7_npu_sequence::gen_dequant_mm_rtm(npu_sequence* seq, const uint32_t rows, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act, const uint32_t row_base){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    static constexpr int qm = QXNX_ROW_BLOCK_SIZE;
    static constexpr uint32_t desired_k = 512;
    static constexpr uint32_t desired_n = 64;
    const int K_div_k = Din / desired_k;
    const uint32_t columns = 8;
    assert(Dout / qm / columns > 0);
    assert(Dout % (qm * columns) == 0);

    const mm_geometry g = mm_pick_geometry(rows);
    const uint32_t m = g.m;
    const int M_div_m = g.blocks;
    const int N_div_n = Dout / desired_n / columns;
    // rtp_act 3: the gate/up weights are the glu_repack()ed copy (pairs of [up block | gate block] per core), y =
    // silu(gate) * up, Dout / 2 columns wide (see minicpm_v_desc::glu_repack)
    const bool glu = rtp_act == 3;
    const uint32_t Dy = glu ? Dout / 2 : Dout;                                          // row pitch of y
    const uint32_t pass_cols = glu ? desired_n * columns / 2 : desired_n * columns;     // output columns per pass
    const uint32_t half_cols = pass_cols / 2;                                           // columns of one memtile group

    seq->clear_cmds();

    for (int row = 2; row < 6; row++){
        const bool even = (row % 2) == 0;
        const uint32_t ctp = even ? CT_rtp_address_even_row : CT_rtp_address_odd_row;
        const uint32_t atp = even ? act_rtp_address_even_row : act_rtp_address_odd_row;
        for (uint32_t col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, ctp, K_div_k);
            seq->rtp_write(tile, atp, rtp_act);
        }
    }
    mm_set_core_rows(seq, m);
    if (glu) mm_rearm(seq, m, this->mm_yjoin_par, MM_Y_GLU); else if (m != 128) mm_rearm(seq, m, this->mm_yjoin_par);

    for (int i = 0; i < M_div_m; i++){
        for (int j = 0; j < N_div_n; j++){
            int round = i * N_div_n + j;
            int is_pong = round % 2;
            int bd_offset = is_pong * 8;
            uint32_t y_offset = (row_base + i * m * 4) * Dy + j * pass_cols;
            uint32_t x_offset = (row_base + i * m * 4) * Din;
            int w_offset = get_quantization_byte_size((size_t)j * desired_n * Din * columns, flm_q4k) + weight_offset;

            npu_bd_id y_bd_0 = npu_bd_id(0 + bd_offset);
            npu_bd_id y_bd_1 = npu_bd_id(3 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd_0 = npu_bd_id(2 + bd_offset);
            npu_bd_id w_bd_1 = npu_bd_id(4 + bd_offset);
            for (uint32_t col = 0; col < 4; col++){
                seq->npu_dma_memcpy_nd(
                    sizeof(char), 2, MM2S, IT[col], w_bd_0, it_channel_1,
                    {0, 0, 0, (uint32_t)(w_offset + get_quantization_byte_size((size_t)col * desired_n * Din, flm_q4k))},
                    {1, 1, 1, (uint32_t)get_quantization_byte_size(desired_n * Din, flm_q4k)},
                    {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
                seq->npu_dma_memcpy_nd(
                    sizeof(char), 2, MM2S, IT[col + 4], w_bd_1, it_channel_1,
                    {0, 0, 0, (uint32_t)(w_offset + get_quantization_byte_size((size_t)col * desired_n * Din, flm_q4k) + get_quantization_byte_size(4 * desired_n * Din, flm_q4k))},
                    {1, 1, 1, (uint32_t)get_quantization_byte_size(desired_n * Din, flm_q4k)},
                    {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
                // receive y
                seq->npu_dma_memcpy_nd(
                    2, 0, S2MM, IT[col], y_bd_0, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * m * Dy + out_offset)},
                    {1, 1, m, half_cols}, {0, 0, Dy, 1}, -1, 0, true);
                seq->npu_dma_memcpy_nd(
                    2, 0, S2MM, IT[col + 4], y_bd_1, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * m * Dy + half_cols + out_offset)},
                    {1, 1, m, half_cols}, {0, 0, Dy, 1}, -1, 0, true);
                // send x
                seq->npu_dma_memcpy_nd(
                    2, 1, MM2S, IT[2 * col], x_bd, it_channel_0,
                    {0, 0, 0, x_offset + col * m * Din},
                    {1, (uint32_t)K_div_k, m, desired_k}, {0, desired_k, Din, 1}, -1, 0, false);
            }
            if (round > 0){
                for (int r = 0; r < 4; r++){
                    seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                    seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int r = 0; r < 4; r++){
        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
        seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
    }
    if (glu){ mm_rearm(seq, 128, this->mm_yjoin_par, MM_Y_RESTORE); mm_set_core_rows(seq, 128); }
    else if (m != 128){ mm_rearm(seq, 128, this->mm_yjoin_par); mm_set_core_rows(seq, 128); }   // leave the design's configuration behind
    seq->cmds2seq();
}

void minicpm_v_4_7_npu_sequence::gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act, const uint32_t row_base, const uint32_t valid_rows){
    if (mm_rtm){
        gen_dequant_mm_rtm(seq, valid_rows ? std::min(valid_rows, L) : L, Din, Dout, weight_offset, out_offset, rtp_act, row_base);
        return;
    }
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr int m = QXNX_ROW_BLOCK_SIZE;
    static constexpr uint32_t desired_m = 128;
    static constexpr uint32_t desired_k = 512;
    static constexpr uint32_t desired_n = 64;
    //  %K_div_k_2_7 = aie.buffer(%tile_7_2) {address = 54784 : i32, sym_name = "K_div_k_2_7"} : memref<16xi32>
    // %rtp_act_2_7 = aie.buffer(%tile_7_2) {address = 54848 : i32, sym_name = "rtp_act_2_7"} : memref<16xi32>
    // %K_div_k_3_0 = aie.buffer(%tile_0_3) {address = 61696 : i32, sym_name = "K_div_k_3_0"} : memref<16xi32>
    // %rtp_act_3_0 = aie.buffer(%tile_0_3) {address = 61760 : i32, sym_name = "rtp_act_3_0"} : memref<16xi32>

    int K_div_k = Din / desired_k;

    const uint32_t columns = 8;
    assert(Dout / m / columns > 0);
    assert(Dout % (m * columns) == 0);

    seq->clear_cmds();

    int M_div_m = L / (desired_m * 4); // 4 row-blocks processed per i iteration (IT[col], col=0..3)
    int N_div_n = Dout / desired_n / columns;
    assert(L % (desired_m * 4) == 0);

    for (int row = 2; row < 6; row+=2){
        for (int col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, CT_rtp_address_even_row, K_div_k);
            seq->rtp_write(tile, act_rtp_address_even_row, rtp_act);
        }
    }
    for (int row = 3; row < 6; row+=2){
        for (int col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, CT_rtp_address_odd_row, K_div_k);
            seq->rtp_write(tile, act_rtp_address_odd_row, rtp_act);
        }
    }

    for (int i = 0; i < M_div_m; i++){
        for (int j = 0; j < N_div_n; j++){
            int round = i * N_div_n + j;
            int is_pong = round % 2;
            int bd_offset = is_pong * 8;
            uint32_t y_offset = (row_base + i * desired_m * 4) * Dout + j * desired_n * columns;
            uint32_t x_offset = (row_base + i * desired_m * 4) * Din;
            int w_offset = get_quantization_byte_size((size_t)j * desired_n * Din * columns, flm_q4k) + weight_offset;

            npu_bd_id y_bd_0 = npu_bd_id(0 + bd_offset);
            npu_bd_id y_bd_1 = npu_bd_id(3 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd_0 = npu_bd_id(2 + bd_offset);
            npu_bd_id w_bd_1 = npu_bd_id(4 + bd_offset);
            for (uint32_t col = 0; col < 4; col++){
                seq->npu_dma_memcpy_nd(
                    sizeof(char),
                    2,
                    MM2S,
                    IT[col],
                    w_bd_0,
                    it_channel_1,
                    {0, 0, 0, (uint32_t)(w_offset + get_quantization_byte_size((size_t)col * desired_n * Din, flm_q4k))},
                    {1, 1, 1, (uint32_t)get_quantization_byte_size(desired_n * Din, flm_q4k)},
                    {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache
                );
                seq->npu_dma_memcpy_nd(
                    sizeof(char),
                    2,
                    MM2S,
                    IT[col + 4],
                    w_bd_1,
                    it_channel_1,
                    {0, 0, 0, (uint32_t)(w_offset + get_quantization_byte_size((size_t)col * desired_n * Din, flm_q4k) + get_quantization_byte_size(4 * desired_n * Din, flm_q4k))},
                    {1, 1, 1, (uint32_t)get_quantization_byte_size(desired_n * Din, flm_q4k)},
                    {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache
                );
                // receive y
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    y_bd_0, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * desired_m * Dout + out_offset)},
                    {1, 1, desired_m, desired_n * 4},
                    {0, 0, Dout, 1},
                    -1, 0, true
                );
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col + 4],
                    y_bd_1, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * desired_m * Dout + 4 * desired_n + out_offset)},
                    {1, 1, desired_m, desired_n * 4},
                    {0, 0, Dout, 1},
                    -1, 0, true
                );
                // send x
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[2 * col],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + col * desired_m * Din},
                    {1, (uint32_t)K_div_k, desired_m, desired_k},
                    {0, desired_k, Din, 1},
                    -1, 0, false
                );
            }
            if (round > 0){
                for (int r = 0; r < 4; r++){
                    seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                    seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
                }
            }
        }// end j
    } // end i
    for (int r = 0; r < 4; r++){
        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
        seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}

void minicpm_v_4_7_npu_sequence::gen_lm_head_seq(npu_sequence* seq, int VOCAB_SIZE, int D){
    static constexpr int y_arg_id = 0;
    static constexpr int w_arg_id = 1;
    static constexpr int x_arg_id = 2;
    static constexpr int M_PER_ROUND = 32 * 32;
    static constexpr int M    = 32;
    static constexpr int M_PER_COL    = M * 4;
    static constexpr int COLS = 8;
    static npu_tiles ITs[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    // Packet ids for the two flows that share IT0's MM2S 0 -- must match X_PKT_ID /
    // W_PKT_ID in FLM_Xclbin/MiniCPM_V/lm_head_npu_bin/iron/q4k_mm.py. Every other
    // shim MM2S port carries a single circuit-switched flow.
    static constexpr int X_PKT_ID = 0;
    static constexpr int W_PKT_ID = 1;
    static constexpr int NO_PKT   = -1;
    // Rounds of shim BD ping-pong; also how far the y waits lag the y issues. Three
    // sets in flight ahead of the sequencer was the knee, four bought nothing.
    static constexpr int LOOKAHEAD = 3;
    assert(VOCAB_SIZE % M_PER_ROUND == 0);
    int rounds = VOCAB_SIZE / M_PER_ROUND;
    size_t W_PER_COL = q8_0_size(M_PER_COL * D) / sizeof(bf16);
    size_t W_PER_ROUND = W_PER_COL * COLS;
    // A column's four cores are fed by two independent shim MM2S channels, one per
    // pair of cores: a single channel saturates at roughly a quarter of what the
    // column can consume, which is what made this kernel purely data-movement bound.
    // reorder_cpy_lm_head has already laid the two halves out back to back in DDR.
    size_t W_PER_STREAM = W_PER_COL / 2;

    seq->clear_cmds();
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        MM2S,
        ITs[0],
        npu_bd_id(bd_0),
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D * 2},
        {0, 0, 0, 1},
        X_PKT_ID, 0, false, aggressive_cache
    );

    for (int r = 0; r < rounds; r++) {
        int bd_offset = (r % (LOOKAHEAD + 1)) * 3;
        npu_bd_id bd_y  = npu_bd_id(bd_1 + bd_offset);
        npu_bd_id bd_wa = npu_bd_id(bd_2 + bd_offset);
        npu_bd_id bd_wb = npu_bd_id(bd_3 + bd_offset);
        for (size_t col = 0; col < std::size(ITs); col++){
            size_t w_col_offset = r * W_PER_ROUND + col * W_PER_COL;
            uint32_t y_offset = r * M_PER_ROUND + col * M_PER_COL;
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                y_arg_id,
                S2MM,
                ITs[col],
                bd_y,
                it_channel_0,
                {0, 0, 0, (uint32_t)(y_offset)},
                {1, 1, 1, (uint32_t)M_PER_COL},
                {0, 0, 0, 1},
                NO_PKT, 0, true, aggressive_cache
            );
            // Group a: cores 0 and 1.
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_arg_id,
                MM2S,
                ITs[col],
                bd_wa,
                it_channel_1,
                {0, 0, 0, (uint32_t)(w_col_offset)},
                {1, 1, 1, (uint32_t)W_PER_STREAM},
                {0, 0, 0, 1},
                NO_PKT, 0, false, aggressive_cache
            );
            // Group b: cores 2 and 3. On column 0 this port also carries x, so both
            // flows on it have to be packet-switched.
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_arg_id,
                MM2S,
                ITs[col],
                bd_wb,
                it_channel_0,
                {0, 0, 0, (uint32_t)(w_col_offset + W_PER_STREAM)},
                {1, 1, 1, (uint32_t)W_PER_STREAM},
                {0, 0, 0, 1},
                col == 0 ? W_PKT_ID : NO_PKT, 0, false, aggressive_cache
            );
            if (r > LOOKAHEAD - 1){
                seq->npu_dma_wait(
                    ITs[col],
                    S2MM,
                    it_channel_0
                );
            }
        }
    }
    for (size_t col = 0; col < std::size(ITs); col++){
        for (int i = 0; i < LOOKAHEAD; i++){
            seq->npu_dma_wait(
                ITs[col],
                S2MM,
                it_channel_0
            );
        }
    }

    seq->cmds2seq();
}

void minicpm_v_4_7_npu_sequence::_set_modes(npu_sequence* seq, int proj_mode, int rms_mode){
    for (npu_tiles tile : proj_tiles){
        seq->rtp_write(tile, proj_mode_addr, proj_mode);
        seq->rtp_write(tile, ct_lock_base + 16 * proj_rtp_lock_id, 1);
    }
    seq->rtp_write(rms_tile, rms_mode_addr, rms_mode);
    seq->rtp_write(rms_tile, ct_lock_base + 16 * proj_rtp_lock_id, 1);
    if (stateless && proj_mode != proj_mode_lm_head){   // the lm_head run does not use the gate core: it stays parked on its RTP lock
        seq->rtp_write(gate_tile, gate_mode_addr, proj_mode == proj_mode_attn ? gate_mode_attn : gate_mode_linear);
        seq->rtp_write(gate_tile, ct_lock_base + 16 * proj_rtp_lock_id, 1);
    }
}

/// @note  Same dataflow as the Qwen3.5-2B flash engine's lm_head phase (one linear BD per weight channel, 8 channels, logits on IT1 S2MM 0), sized by
///        the MiniCPM shape: D = 1024, 485 rounds of 512 rows.
void minicpm_v_4_7_npu_sequence::gen_lm_head_in_layer_seq(npu_sequence* seq){
    const uint32_t rows = desc->get_lm_head_in_layer_rows();
    const uint32_t stream_size = desc->get_lm_head_in_layer_w_size() / minicpm_v_desc::lm_head_streams / sizeof(bf16);
    const uint32_t columns = 4;
    seq->clear_cmds();
    _set_modes(seq, proj_mode_lm_head, rms_mode_lm_head);

    _send_hidden_states(seq);
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)desc->D},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        logits_arg_idx,
        S2MM,
        lm_head_logits_tile,
        bd_3,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, rows},
        {0, 0, 0, 1},
        -1, 0, true
    );
    for (uint32_t col = 0; col < columns; col++){
        for (uint32_t ch = 0; ch < 2; ch++){
            const uint32_t stream = col * 2 + ch;
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_proj_arg_idx,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_1 + ch),
                ch == 0 ? it_channel_0 : it_channel_1,
                {0, 0, 0, stream * stream_size},
                {1, 1, 1, stream_size},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
        }
    }
    seq->npu_dma_wait(lm_head_logits_tile, S2MM, it_channel_0);
    for (uint32_t col = 0; col < columns; col++){
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
    }
    seq->cmds2seq();
}

void minicpm_v_4_7_npu_sequence::_gen_sequence(npu_sequence* seq, int L) {
    minicpm_v_attn_layer& _weight_desc = static_cast<minicpm_v_attn_layer&>(*desc->weight_desc[desc->attn_idx]);
    seq->clear_cmds();
    mvm_pending_ = 0;
    mvm_set_ = 0;
    if (lm_in_layer) _set_modes(seq, proj_mode_attn, rms_mode_attn);
    if (stateless){
        // the context length is not left to the set_layer_rtp run at the start of the token (a design reload since then would have wiped it), and
        // x comes from DDR like it does for the first linear run
        gen_set_context(seq, L);
        _send_hidden_states(seq);
    }

    _send_rms_weights(seq);
    _send_rope_rms_weights(seq);
    // receive y
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx,
        S2MM,
        xyr_tile,
        bd_10,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)desc->D},
        {0, 0, 0, 1},
        -1, 0, true
    );

    _receive_kv_cache(seq, L); // also move qkv weights
    _move_weights(seq, desc->DK + desc->DV, desc->D, _weight_desc.attn_k);
    _move_weights(seq, desc->DQ, desc->D, _weight_desc.attn_q);
    _move_kv_cache(seq, L);
    _move_weights(seq, desc->DQ, desc->D, _weight_desc.attn_gate);
    _move_weights(seq, desc->D, desc->DQ, _weight_desc.attn_output);
    _move_weights(seq, 2 * desc->intermediate_size, desc->D, _weight_desc.ffn_up);
    _move_weights(seq, desc->D, desc->intermediate_size, _weight_desc.ffn_down);

    _flush_mvm(seq);
    // wait for receiving y
    seq->npu_dma_wait(
        xyr_tile,
        S2MM,
        it_channel_1
    );

    //
    seq->cmds2seq();
}

void minicpm_v_4_7_npu_sequence::_send_hidden_states(npu_sequence* seq){
    // send x
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx,
        MM2S,
        xyr_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)desc->D},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

void minicpm_v_4_7_npu_sequence::_send_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)desc->D * 2},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

void minicpm_v_4_7_npu_sequence::_send_rope_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_2,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(desc->RoPE_DIM + desc->DH * 2)},
        {0, 0, 0, 1},
        1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        xyr_tile,
        MM2S,
        it_channel_0
    );
}

void minicpm_v_4_7_npu_sequence::_receive_kv_cache(npu_sequence* seq, const int L){
    uint32_t data_per_chunk = desc->num_kv_heads * desc->DH;
    uint32_t L_offset = (L - 1) * data_per_chunk;
    uint32_t v_offset = this->kv_cache_size / 2;

    // receive k
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        attn_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, (uint32_t)L_offset},
        {1, 1, 1, data_per_chunk},
        {0, 0, 0, 1},
        -1, 0, true
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        attn_tile,
        bd_1,
        it_channel_0,
        {0, 0, 0, (uint32_t)(L_offset + v_offset)},
        {1, 1, 1, data_per_chunk},
        {0, 0, 0, 1},
        -1, 0, true
    );
}

void minicpm_v_4_7_npu_sequence::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const weight_desc_t& weight){
    const size_t w_offset = weight.offset;
    const flm_dtype_t dtype = weight.dtype;
    const int m = 32;
    const int k = 256;
    const uint32_t columns = 4;
    const uint32_t a_block_size = get_quantization_byte_size(m * k, dtype) / sizeof(bf16);
    const uint32_t blocks_per_row = Din / k;
    const uint32_t cores = columns * 4;
    assert(Dout / m / cores > 0);
    assert(Dout % (m * cores) == 0);
    // One strided BD per (column, channel) for the whole matrix instead of one per 512-row
    // round: the shim queue never waits for the sequencer to issue the next round. Same
    // bytes in the same order, so results are bit-identical. dim0 <= 1024 words, the round
    // is the iteration (< 64); anything that does not factor falls back to the round loop.
    const int wmode = wstream_mode();
    if (wmode != 0){
        const uint32_t rounds = (uint32_t)(Dout / m / cores);
        const uint32_t block_words = a_block_size / 2;        // 32-bit words per 32x256 block... a_block_size is in bf16
        const uint32_t bd_words = 2 * blocks_per_row * block_words;
        uint32_t d0 = 0;
        for (uint32_t c = 1023; c >= 16; c--){
            if (bd_words % c == 0 && bd_words / c < 1024 && bd_words / c > 1){ d0 = c; break; }
        }
        if (d0 != 0 && rounds < 64){
            const uint32_t d1 = bd_words / d0;
            const size_t round_stride = (size_t)cores * a_block_size * blocks_per_row;  // bf16
            assert(round_stride / 2 < (1u << 20));
            // mode 2 alternates two BD id sets so the next matrix can be queued while this one drains
            const int bd_off = (wmode == 2) ? (mvm_set_ ^= 8) : 0;
            for (uint32_t col = 0; col < columns; col++){
                for (int ch = 0; ch < 2; ch++){
                    seq->npu_dma_memcpy_nd(
                        sizeof(bf16),
                        w_proj_arg_idx,
                        MM2S,
                        mvm_tiles[col],
                        npu_bd_id((ch == 0 ? bd_1 : bd_2) + bd_off),
                        ch == 0 ? it_channel_0 : it_channel_1,
                        {0, 0, 0, (uint32_t)((size_t)(col * 4 + ch * 2) * a_block_size * blocks_per_row + (uint32_t)w_offset / sizeof(bf16))},
                        {rounds, 1, d1, d0 * 2},
                        {(uint32_t)round_stride, 0, d0 * 2, 1},
                        -1, 0, true, aggressive_cache
                    );
                }
            }
            if (wmode == 2){
                // the previous matrix's tokens come first on each channel
                if (mvm_pending_) _flush_mvm(seq);
                mvm_pending_ = 1;
            }
            else{
                for (uint32_t col = 0; col < columns; col++){
                    seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
                    seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
                }
            }
            return;
        }
    }
    for (size_t round = 0; round < Dout / m / cores; round++){
        uint32_t bd_offset = (round % 2) * 8;
        for (uint32_t col = 0; col < columns; col++){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_proj_arg_idx,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_1 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * a_block_size * blocks_per_row + (uint32_t)w_offset / sizeof(bf16))},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_proj_arg_idx,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_2 + bd_offset),
                it_channel_1,
                {0, 0, 0, (uint32_t)((round * cores + col * 4 + 2) * a_block_size * blocks_per_row + (uint32_t)w_offset / sizeof(bf16))},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
        }
        if (round > 0){
            for (uint32_t col = 0; col < columns; col++){
                seq->npu_dma_wait(
                    mvm_tiles[col],
                    MM2S,
                    it_channel_0
                );
                seq->npu_dma_wait(
                    mvm_tiles[col],
                    MM2S,
                    it_channel_1
                );
            }
        }
    }
    for (uint32_t col = 0; col < columns; col++){
        seq->npu_dma_wait(
            mvm_tiles[col],
            MM2S,
            it_channel_0
        );
        seq->npu_dma_wait(
            mvm_tiles[col],
            MM2S,
            it_channel_1
        );
    }
}

int minicpm_v_4_7_npu_sequence::wstream_mode(){
    static const int m = [](){ const char* e = std::getenv("FLM_MCV_WSTREAM"); return e ? std::atoi(e) : 2; }();
    return m;
}

void minicpm_v_4_7_npu_sequence::_flush_mvm(npu_sequence* seq){
    if (!mvm_pending_) return;
    for (uint32_t col = 0; col < 4; col++){
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
    }
    mvm_pending_ = 0;
}

void minicpm_v_4_7_npu_sequence::_move_kv_cache(npu_sequence* seq, const size_t L){
    const int L_padded = (L + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    const uint32_t data2move = L_padded * desc->DH * desc->num_kv_heads;
    const uint32_t v_offset = this->kv_cache_size / 2;

    seq->npu_dma_wait(
        attn_tile,
        S2MM,
        it_channel_0
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        attn_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
    seq->npu_dma_wait(
        attn_tile,
        S2MM,
        it_channel_0
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        attn_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, (uint32_t)(v_offset)},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
}

void minicpm_v_4_7_npu_sequence::_gen_linear_sequence(npu_sequence* seq, bool is_first) {
    minicpm_v_linear_layer& _weight_desc = static_cast<minicpm_v_linear_layer&>(*desc->weight_desc[desc->lin_idx]);
    seq->clear_cmds();
    mvm_pending_ = 0;
    mvm_set_ = 0;
    if (lm_in_layer) _set_modes(seq, proj_mode_linear, is_first ? rms_mode_first_linear : rms_mode_linear);

    if (is_first || stateless) {
        _send_hidden_states(seq);
    }
    if (stateless){
        // receive the new x (the packet route the attention run uses); nothing stays on the array for the next run
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            xy_arg_idx,
            S2MM,
            xyr_tile,
            bd_10,
            it_channel_1,
            {0, 0, 0, 0},
            {1, 1, 1, (uint32_t)desc->D},
            {0, 0, 0, 1},
            -1, 0, true
        );
    }
    _send_rms_weights(seq);
    _send_linear_conv_weights(seq);

    _move_linear_kv_cache(seq);
    _move_alpha_beta_weights(seq, desc->linear_num_v_heads_padded * 2, desc->D, _weight_desc.ssm_alpha.offset);
    _move_weights(seq, desc->Linear_DQ + desc->Linear_DK + desc->Linear_DV, desc->D, _weight_desc.linear_attn_qkv);
    _receive_linear_kv_cache(seq); // also move qkv weights
    _move_s(seq);
    _move_weights(seq, desc->Linear_DV, desc->D, _weight_desc.attn_gate);
    _move_weights(seq, desc->D, desc->Linear_DV, _weight_desc.ssm_out);
    _move_weights(seq, 2 * desc->intermediate_size, desc->D, _weight_desc.ffn_up);
    _move_weights(seq, desc->D, desc->intermediate_size, _weight_desc.ffn_down);
    _flush_mvm(seq);
    // _move_s pushes the GDN state write-back (IT4 S2MM ch0) with a task-complete token that nothing waited for: every linear layer run left one
    // token behind (18 per decode token). A PDI load (a hw-context switch) forgets them; a decode that stays in one hw_context (reconfiguration
    // instead of a context switch) keeps accumulating them and hangs after ~70 tokens. The write-back has long finished by
    // the end of the layer, so this wait costs nothing.
    if (stateless){
        seq->npu_dma_wait(xyr_tile, S2MM, it_channel_1);
    }
#ifndef FLM_MINICPM_DEC_FAULT_NO_GDN_WAIT   // test-only: compile with it to reproduce the old hang and exercise the single-context self-test's failure path
    seq->npu_dma_wait(IT4, S2MM, it_channel_0);
#endif
    seq->cmds2seq();
}

void minicpm_v_4_7_npu_sequence::_send_linear_conv_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_rms_arg_idx,
        MM2S,
        IT5,
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)conv_w_size_bf16 + desc->linear_v_head_dim + desc->linear_num_v_heads_padded * 2 * 2},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_1
    );
}

void minicpm_v_4_7_npu_sequence::_receive_linear_kv_cache(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        IT5,
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(desc->conv_kernel_size - 1) * (desc->Linear_DQ + desc->Linear_DK + desc->Linear_DV)},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        IT5,
        S2MM,
        it_channel_1
    );
}

void minicpm_v_4_7_npu_sequence::_move_alpha_beta_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    // currently, only on DMA channel 0 of IT5
    const int m = 32;
    const int k = 256;
    const uint32_t block_size = q8_0_size(m * k) / sizeof(bf16);
    const uint32_t total_blocks = (Dout / m) * (Din / k);
    const uint32_t data2move = block_size * total_blocks;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_rms_arg_idx,
        MM2S,
        IT5,
        npu_bd_id(bd_13),
        it_channel_0,
        {0, 0, 0, (uint32_t)(w_offset / sizeof(bf16))},
        {1, 1, 1, (uint32_t)data2move},
        {0, 0, 0, 1},
        -1, 0, true, aggressive_cache
    );
    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_0
    );
}

void minicpm_v_4_7_npu_sequence::_move_linear_kv_cache(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        IT5,
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(desc->conv_kernel_size - 1) * (desc->Linear_DQ + desc->Linear_DK + desc->Linear_DV)},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_1
    );
}

void minicpm_v_4_7_npu_sequence::_move_s(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        IT4,
        bd_2,
        it_channel_0,
        {0, 0, 0, (uint32_t)s_offset},
        {1, 1, 1, (uint32_t)(desc->linear_num_v_heads * desc->linear_v_head_dim * desc->linear_qk_head_dim * 2)},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        IT4,
        bd_3,
        it_channel_0,
        {0, 0, 0, (uint32_t)s_offset},
        {1, 1, 1, (uint32_t)(desc->linear_num_v_heads * desc->linear_v_head_dim * desc->linear_qk_head_dim * 2)},
        {0, 0, 0, 1},
        -1, 0, false, no_cache
    );

    // seq->npu_dma_wait(
    //     IT4,
    //     S2MM,
    //     it_channel_0
    // );
}
minicpm_v_4_7_npu_sequence::~minicpm_v_4_7_npu_sequence() = default;
