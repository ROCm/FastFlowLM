// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "minicpm_ocr_npu_sequence.hpp"
#include "lm_head_in_layer.hpp"

minicpm_ocr_npu_sequence::minicpm_ocr_npu_sequence(minicpm_ocr_desc& desc, LM_Config config, uint32_t MAX_L)
    : desc(&desc){
    conv_w_size_bf16 = desc.get_conv_w_size() / sizeof(bf16);

    s_offset = desc.get_s_offset();
    this->kv_cache_size = (uint32_t)desc.get_kv_cache_size(MAX_L);
    this->MAX_L = MAX_L;

}

void minicpm_ocr_npu_sequence::set_max_length(const uint32_t MAX_L){
    this->MAX_L = MAX_L;
    this->kv_cache_size = MAX_L * (desc->DK + desc->DV);
}

void minicpm_ocr_npu_sequence::_set_modes(npu_sequence* seq, int proj_mode, int rms_mode){
    if (!write_modes){
        return;
    }
    for (npu_tiles tile : proj_tiles){
        seq->rtp_write(tile, proj_mode_addr, proj_mode);
        seq->rtp_write(tile, ct_lock_base + 16 * mode_rtp_lock_id, 1);
    }
    seq->rtp_write(rms_tile, rms_mode_addr, rms_mode);
    seq->rtp_write(rms_tile, ct_lock_base + 16 * mode_rtp_lock_id, 1);
}

void minicpm_ocr_npu_sequence::gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_attn_layer, bool is_first){
    if (is_attn_layer) {
        _gen_sequence(seq, L);
    }
    else {
        _gen_linear_sequence(seq, is_first);
    }
}

// Runtime sequence for conv8.xclbin (FLM_Xclbin/minicpm_ocr/conv1d_prefill). The same
// code is that folder's host/seq_gen.hpp, which its standalone test runs; keep the two
// in step.
//
// x is [L_PADDED, D] bf16 (the 3 conv-state rows first), w is [4, D] at weight_offset in
// its own argument, o is [L_OUT, D]. D = q (2048) + k (2048) + v (2048) = 6144.
//
// Layout: all 8 columns, every core owns one 128-wide q/k head and 64 v channels.
// Column col therefore takes q/k channels [512 col, 512 col + 512) and v channels
// [4096 + 256 col, ... + 256), sent as two streams:
//     shim MM2S 0: the 4 q/k weight rows, then the q/k part of every x row
//     shim MM2S 1: the 4 v weight rows,   then the v part of every x row
//     shim S2MM 0 / 1: the q/k / v part of every output row
// The weight rows lead the stream so they reach the cores like any other row.
void minicpm_ocr_npu_sequence::gen_seq_conv1d(
    npu_sequence* seq,
    const uint32_t L_PADDED, // L_OUT + (window_size - 1)
    const uint32_t D,
    const uint32_t L_OUT,
    const uint32_t weight_offset
){
    npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    seq->clear_cmds();

    // address of the L rtp buffer in conv8.xclbin (the same in all 32 cores)
    constexpr int l_address = 33152;

    constexpr uint32_t num_col = 8;
    constexpr uint32_t num_row = 4;
    constexpr uint32_t window_size = 4;
    constexpr uint32_t d_qk_col = 512;            // 4 cores x one 128-wide head
    constexpr uint32_t d_v_col = 256;             // 4 cores x 64
    constexpr uint32_t D_qk = num_col * d_qk_col; // 4096 = q + k
    assert(D == D_qk + num_col * d_v_col && "conv8.xclbin is laid out for D = 6144 (q 2048, k 2048, v 2048)");

    constexpr uint32_t x_arg = 1;
    constexpr uint32_t w_arg = 2;
    constexpr uint32_t o_arg = 0;

    // input rows go in blocks of 256 (the BD's row count field), plus a remainder
    constexpr uint32_t X_block = 256;
    const uint32_t num_x_blocks = L_PADDED / X_block;
    const uint32_t x_remainder = L_PADDED % X_block;
    constexpr uint32_t Y_block = 1024;

    for (uint32_t row = 2; row < 2 + num_row; row++) {
        for (uint32_t col = 0; col < num_col; col++) {
            seq->rtp_write(get_tile(row, col), l_address, L_OUT);
        }
    }

    // one input stream: its 4 weight rows, then the x rows
    auto send_stream = [&](uint32_t col, npu_it_channel ch, uint32_t c_off, uint32_t width,
                           uint32_t bd_w, uint32_t bd_x, uint32_t bd_rem) {
        seq->npu_dma_memcpy_nd(
            sizeof(bf16), w_arg, MM2S, IT[col], (npu_bd_id)bd_w, ch,
            {0, 0, 0, weight_offset + c_off},
            {1, 1, window_size, width},
            {0, 0, D, 1},
            -1, 0, false);
        if (num_x_blocks > 0) {
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), x_arg, MM2S, IT[col], (npu_bd_id)bd_x, ch,
                {0, 0, 0, c_off},
                {1, num_x_blocks, X_block, width},
                {0, X_block * D, D, 1},
                -1, 0, false);
        }
        if (x_remainder > 0) {
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), x_arg, MM2S, IT[col], (npu_bd_id)bd_rem, ch,
                {0, 0, 0, c_off + X_block * D * num_x_blocks},
                {1, 1, x_remainder, width},
                {0, 0, D, 1},
                -1, 0, false);
        }
    };
    for (uint32_t col = 0; col < num_col; col++) {
        send_stream(col, it_channel_0, col * d_qk_col, d_qk_col, 1, 2, 3);
        send_stream(col, it_channel_1, D_qk + col * d_v_col, d_v_col, 4, 5, 6);
    }

    // outputs, Y_block rows at a time; each round waits for the one before it
    const uint32_t receive_round = (L_OUT + Y_block - 1) / Y_block;
    for (uint32_t r = 0; r < receive_round; r++) {
        const uint32_t rows = (r + 1 == receive_round && L_OUT % Y_block != 0) ? L_OUT % Y_block : Y_block;
        for (uint32_t col = 0; col < num_col; col++) {
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), o_arg, S2MM, IT[col], (npu_bd_id)((r % 2) ? 8 : 0), it_channel_0,
                {0, 0, 0, col * d_qk_col + r * Y_block * D},
                {1, 1, rows, d_qk_col},
                {0, 0, D, 1},
                -1, 0, true);
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), o_arg, S2MM, IT[col], (npu_bd_id)((r % 2) ? 9 : 7), it_channel_1,
                {0, 0, 0, D_qk + col * d_v_col + r * Y_block * D},
                {1, 1, rows, d_v_col},
                {0, 0, D, 1},
                -1, 0, true);
        }
        if (r > 0) {
            for (uint32_t col = 0; col < num_col; col++) {
                seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
                seq->npu_dma_wait(IT[col], S2MM, it_channel_1);
            }
        }
    }
    for (uint32_t col = 0; col < num_col; col++) {
        seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
        seq->npu_dma_wait(IT[col], S2MM, it_channel_1);
    }

    seq->cmds2seq();
}

void minicpm_ocr_npu_sequence::gen_mha_engine_seq(
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

// attn_bfp16.xclbin: the kernels_2b path of qwen3_5vl_flash_sequence::gen_mha_engine_seq (origin/main). q tiles
// [head][L_wanted / 16][DH / 8][2][72 B]; K tiles [kv head][kv_bfp16_rows / 16][...], then V^T tiles the same.
void minicpm_ocr_npu_sequence::gen_mha_bfp16_seq(
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
    int using_window_size = MAX_L;
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
    const int down_rounds = Q_chunks;

    int num_cu = 4;
    for (int head = 0; head < Heads / num_cu; head++){
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_end = (Lq_current + data_per_round + 128 - 1) / 128 * 128;
            int kv_length = kv_end - kv_begin;

            int bd_offset = (round % 2) * 8;
            for (int cu = 0; cu < num_cu; cu++){
                int head_offset = head * num_cu + cu;
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
                // send q, two halves of a round
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

            int kv_head_offset = head;
            for (int part = 0; part < 2; part++) {
                const uint32_t kv_bytes = ((part * Num_of_d_in_KV_cache + kv_head_offset) * kv_bfp16_rows + kv_begin) * row_bytes;
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
        }
        for (int cu = 0; cu < num_cu; cu++){
            for (int col = 0; col < 2; col++){
                seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
            }
        }
    }
    seq->cmds2seq();
}

// mm_bfp16.xclbin: x arrives as bfp16 tiles [16-row group][Din / 8][2 row blocks][72 B] (minicpm_ocr_mm_bfp16::tiles),
// y as 4 cores' 128 x 64 tiles one after another per mem tile. Weights as in gen_dequant_mm_512. The 2B flash engine's
// kernels_2b path (origin/main qwen3_5vl_flash_sequence.cpp).
// x_from_glu_slots: x is the GLU slots gen_dequant_mm_bfp16_glu wrote (the down projection), read as the same stream.
// x_row_begin: a run over one row chunk (prefill_chunks.hpp) reads x from that row on; out_offset points at its y rows.
void minicpm_ocr_npu_sequence::gen_dequant_mm_bfp16(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act, const bool x_from_glu_slots, const uint32_t x_row_begin){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr int m = QXNX_ROW_BLOCK_SIZE;
    static constexpr uint32_t desired_m = 128;
    static constexpr uint32_t desired_k = 512;
    static constexpr uint32_t desired_n = 64;

    int K_div_k = Din / desired_k;

    const uint32_t columns = 8;
    assert(Dout / m / columns > 0);
    assert(Dout % (m * columns) == 0);
    assert(Din % desired_k == 0);

    seq->clear_cmds();

    int M_div_m = L / (desired_m * 4);
    int N_div_n = Dout / desired_n / columns;
    assert(L % (desired_m * 4) == 0);
    assert(x_row_begin % (desired_m * 4) == 0);

    const std::vector<uint32_t> y_sizes = {1, 4, desired_m, desired_n};
    const std::vector<uint32_t> y_strides = {0, desired_n, Dout, 1};

    for (int row = 2; row < 6; row+=2){
        for (int col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, mm_bfp16_ct_even, K_div_k);
            seq->rtp_write(tile, mm_bfp16_act_even, rtp_act);
        }
    }
    for (int row = 3; row < 6; row+=2){
        for (int col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, mm_bfp16_ct_odd, K_div_k);
            seq->rtp_write(tile, mm_bfp16_act_odd, rtp_act);
        }
    }

    const uint32_t group_bytes = Din * 18;
    for (int i = 0; i < M_div_m; i++){
        for (int j = 0; j < N_div_n; j++){
            int round = i * N_div_n + j;
            int is_pong = round % 2;
            int bd_offset = is_pong * 8;
            uint32_t y_offset = i * desired_m * 4 * Dout + j * desired_n * columns;
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
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    y_bd_0, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * desired_m * Dout + out_offset)},
                    y_sizes,
                    y_strides,
                    -1, 0, true
                );
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col + 4],
                    y_bd_1, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * desired_m * Dout + 4 * desired_n + out_offset)},
                    y_sizes,
                    y_strides,
                    -1, 0, true
                );
                const uint32_t first_group = (x_row_begin + i * desired_m * 4 + col * desired_m) / 16;
                if (x_from_glu_slots) {
                    const uint32_t rb = x_row_begin / desired_m + i * 4 + col;
                    seq->npu_dma_memcpy_nd(
                        1, 1,
                        MM2S, IT[2 * col],
                        x_bd, it_channel_0,
                        {0, 0, 0, rb * (Din / glu_slot_features) * glu_slot_bytes},
                        {Din / 256, desired_m / 16, 8, glu_slot_used / 8},
                        {8 * glu_slot_bytes, glu_slot_used / 8, glu_slot_bytes, 1},
                        -1, 0, false
                    );
                }
                else {
                    seq->npu_dma_memcpy_nd(
                        1, 1,
                        MM2S, IT[2 * col],
                        x_bd, it_channel_0,
                        {0, 0, 0, first_group * group_bytes},
                        {Din / 256, desired_m / 16, 9, 512},
                        {4608, group_bytes, 512, 1},
                        -1, 0, false
                    );
                }
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
    seq->cmds2seq();
}

// mm_bfp16_glu.xclbin, up/gate with the GLU epilogue (rtp_act 2). The weights are stored [up 512 rows][gate 512 rows] per
// 512 hid features, each 64 rows as [256 k][2 x 32-row blocks]. Round j, core column c makes hid features
// j * 256 + c * 32 .. + 32: its shim sends, per 256 k, the 32 up rows' block, then the matching gate rows' block. Each
// core's 16 KB of y lands in its slot; the 4 cores of a shim are 4 consecutive slots. x as in gen_dequant_mm_bfp16.
void minicpm_ocr_npu_sequence::gen_dequant_mm_bfp16_glu(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t IS, const size_t weight_offset, const uint32_t x_row_begin){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    static constexpr uint32_t desired_m = 128;
    static constexpr uint32_t desired_k = 512;
    const uint32_t columns = 8;
    const uint32_t feats_per_round = columns * glu_slot_features;
    const int K_div_k = Din / desired_k;
    assert(IS % 512 == 0);
    assert(Din % desired_k == 0);
    assert(L % (desired_m * 4) == 0);
    assert(x_row_begin % (desired_m * 4) == 0);

    seq->clear_cmds();
    const int M_div_m = L / (desired_m * 4);
    const int N_div_n = IS / feats_per_round;
    const uint32_t block = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, flm_q4k);
    const uint32_t up_to_gate = get_quantization_byte_size(512 * Din, flm_q4k);
    const uint32_t slots_per_rb = IS / glu_slot_features;

    for (int row = 2; row < 6; row += 2){
        for (uint32_t col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, mm_bfp16_ct_even, K_div_k);
            seq->rtp_write(tile, mm_bfp16_act_even, mm_bfp16_act_glu);
        }
    }
    for (int row = 3; row < 6; row += 2){
        for (uint32_t col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, mm_bfp16_ct_odd, K_div_k);
            seq->rtp_write(tile, mm_bfp16_act_odd, mm_bfp16_act_glu);
        }
    }

    const uint32_t group_bytes = Din * 18;
    for (int i = 0; i < M_div_m; i++){
        for (int j = 0; j < N_div_n; j++){
            const int round = i * N_div_n + j;
            const int bd_offset = (round % 2) * 8;
            npu_bd_id y_bd_0 = npu_bd_id(0 + bd_offset);
            npu_bd_id y_bd_1 = npu_bd_id(3 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd_0 = npu_bd_id(2 + bd_offset);
            npu_bd_id w_bd_1 = npu_bd_id(4 + bd_offset);
            for (uint32_t col = 0; col < 4; col++){
                for (uint32_t side = 0; side < 2; side++){
                    const uint32_t c = col + 4 * side;
                    const uint32_t f = j * feats_per_round + c * glu_slot_features;
                    const uint32_t u = (f / 512) * 1024 + f % 512;
                    const uint32_t up = weight_offset + get_quantization_byte_size((size_t)(u / 64) * 64 * Din, flm_q4k) + (size_t)((u % 64) / 32) * block;
                    seq->npu_dma_memcpy_nd(
                        1, 2,
                        MM2S, IT[c],
                        side ? w_bd_1 : w_bd_0, it_channel_1,
                        {0, 0, 0, up},
                        {Din / 256, 2, 2, block / 2},
                        {2 * block, up_to_gate, block / 2, 1},
                        -1, 0, false, aggressive_cache
                    );
                }
                const uint32_t rb = x_row_begin / desired_m + i * 4 + col;
                for (uint32_t side = 0; side < 2; side++){
                    seq->npu_dma_memcpy_nd(
                        1, 0,
                        S2MM, IT[col + 4 * side],
                        side ? y_bd_1 : y_bd_0, it_channel_0,
                        {0, 0, 0, (rb * slots_per_rb + j * columns + 4 * side) * glu_slot_bytes},
                        {1, 1, 1, 4 * glu_slot_bytes},
                        {0, 0, 0, 1},
                        -1, 0, true
                    );
                }
                const uint32_t first_group = (x_row_begin + i * desired_m * 4 + col * desired_m) / 16;
                seq->npu_dma_memcpy_nd(
                    1, 1,
                    MM2S, IT[2 * col],
                    x_bd, it_channel_0,
                    {0, 0, 0, first_group * group_bytes},
                    {Din / 256, desired_m / 16, 9, 512},
                    {4608, group_bytes, 512, 1},
                    -1, 0, false
                );
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
    seq->cmds2seq();
}

void minicpm_ocr_npu_sequence::gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act){
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
            uint32_t y_offset = i * desired_m * 4 * Dout + j * desired_n * columns;
            uint32_t x_offset = i * desired_m * 4 * Din;
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

void minicpm_ocr_npu_sequence::gen_lm_head_seq(npu_sequence* seq, int VOCAB_SIZE, int D){
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

void minicpm_ocr_npu_sequence::gen_lm_head_in_layer_seq(npu_sequence* seq){
    const uint32_t rows = minicpm_ocr_lm_head::rows(*desc);
    const uint32_t stream_size = minicpm_ocr_lm_head::weight_bytes(*desc) / minicpm_ocr_lm_head::streams / sizeof(bf16);
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
    seq->npu_dma_wait(
        lm_head_logits_tile,
        S2MM,
        it_channel_0
    );
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
    seq->cmds2seq();
}

void minicpm_ocr_npu_sequence::_gen_sequence(npu_sequence* seq, int L) {
    minicpm_ocr_attn_layer& _weight_desc = static_cast<minicpm_ocr_attn_layer&>(*desc->weight_desc[desc->attn_idx]);
    seq->clear_cmds();
    _set_modes(seq, proj_mode_attn, rms_mode_attn);

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
    if (kv_bfp16){
        _move_kv_cache_bfp16(seq, L);
    }
    else {
        _move_kv_cache(seq, L);
    }
    _move_weights(seq, desc->DQ, desc->D, _weight_desc.attn_gate);
    _move_weights(seq, desc->D, desc->DQ, _weight_desc.attn_output);
    _move_weights(seq, 2 * desc->intermediate_size, desc->D, _weight_desc.ffn_up);
    _move_weights(seq, desc->D, desc->intermediate_size, _weight_desc.ffn_down);

    // wait for receiving y
    seq->npu_dma_wait(
        xyr_tile,
        S2MM,
        it_channel_1
    );

    //
    seq->cmds2seq();
}

void minicpm_ocr_npu_sequence::_send_hidden_states(npu_sequence* seq){
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

void minicpm_ocr_npu_sequence::_send_rms_weights(npu_sequence* seq){
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

void minicpm_ocr_npu_sequence::_send_rope_rms_weights(npu_sequence* seq){
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

void minicpm_ocr_npu_sequence::_receive_kv_cache(npu_sequence* seq, const int L){
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

void minicpm_ocr_npu_sequence::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const weight_desc_t& weight){
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

void minicpm_ocr_npu_sequence::_move_kv_cache(npu_sequence* seq, const size_t L){
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

// layer_with_lm_head_kvb.xclbin (kv_bfp16): the attention cores read the finished 16-key chunks as bfp16 units from
// the decode copy in arg 3 (kv_bfp16_k_base / kv_bfp16_v_base, kv_bfp16_cap units each), then the newest chunk as two
// bf16 units, one per KV head: 18 cache rows of that head (2 rows before the chunk when there are older rows), read
// after the array wrote the new row. Every unit is 9,216 B; the cores pack the bf16 tail themselves.
void minicpm_ocr_npu_sequence::_move_kv_cache_bfp16(npu_sequence* seq, const size_t L){
    const uint32_t row = desc->num_kv_heads * desc->DH;
    const uint32_t n = (uint32_t)((L - 1) / L_CHUNK);
    const uint32_t unit_elems = kv_bfp16_unit_bytes / sizeof(bf16);
    const uint32_t v_offset = this->kv_cache_size / 2;
    const uint32_t lead = n > 0 ? kv_bfp16_lead_rows : 0;
    const uint32_t tail_offset = (n * L_CHUNK - lead) * row;
    const uint32_t tail_rows = L_CHUNK + kv_bfp16_lead_rows;

    if (n > 0){
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            w_rope_rms_arg_idx,
            MM2S,
            attn_tile,
            bd_2,
            it_channel_0,
            {0, 0, 0, (uint32_t)(kv_bfp16_k_base / sizeof(bf16))},
            {1, 1, 1, n * unit_elems},
            {0, 0, 0, 1},
            -1, 0, false, aggressive_cache
        );
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            w_rope_rms_arg_idx,
            MM2S,
            attn_tile,
            bd_3,
            it_channel_1,
            {0, 0, 0, (uint32_t)(kv_bfp16_v_base() / sizeof(bf16))},
            {1, 1, 1, n * unit_elems},
            {0, 0, 0, 1},
            -1, 0, false, aggressive_cache
        );
    }
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
        bd_4,
        it_channel_0,
        {0, 0, 0, tail_offset},
        {1, (uint32_t)desc->num_kv_heads, tail_rows, (uint32_t)desc->DH},
        {0, (uint32_t)desc->DH, row, 1},
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
        bd_5,
        it_channel_1,
        {0, 0, 0, tail_offset + v_offset},
        {1, (uint32_t)desc->num_kv_heads, tail_rows, (uint32_t)desc->DH},
        {0, (uint32_t)desc->DH, row, 1},
        -1, 0, false, aggressive_cache
    );
}

void minicpm_ocr_npu_sequence::_gen_linear_sequence(npu_sequence* seq, bool is_first) {
    minicpm_ocr_linear_layer& _weight_desc = static_cast<minicpm_ocr_linear_layer&>(*desc->weight_desc[desc->lin_idx]);
    seq->clear_cmds();
    _set_modes(seq, proj_mode_linear, is_first ? rms_mode_first_linear : rms_mode_linear);

    if (is_first) {
        _send_hidden_states(seq);
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
    seq->cmds2seq();
}

void minicpm_ocr_npu_sequence::_send_linear_conv_weights(npu_sequence* seq){
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

void minicpm_ocr_npu_sequence::_receive_linear_kv_cache(npu_sequence* seq){
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

void minicpm_ocr_npu_sequence::_move_alpha_beta_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
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

void minicpm_ocr_npu_sequence::_move_linear_kv_cache(npu_sequence* seq){
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

void minicpm_ocr_npu_sequence::_move_s(npu_sequence* seq){
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
minicpm_ocr_npu_sequence::~minicpm_ocr_npu_sequence() = default;
