// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "minicpm_v_4_7_moe_npu_sequence.hpp"

minicpm_v_4_7_moe_npu_sequence::minicpm_v_4_7_moe_npu_sequence(minicpm_v_4_7_moe_desc& desc, LM_Config config, uint32_t MAX_L)
    : desc(&desc){
    conv_w_size_bf16 = desc.get_conv_w_size() / sizeof(bf16);

    s_offset = desc.get_s_offset();
    this->kv_cache_size = (uint32_t)desc.get_kv_cache_size(MAX_L);
    this->MAX_L = MAX_L;
}

void minicpm_v_4_7_moe_npu_sequence::set_max_length(const uint32_t MAX_L){
    this->MAX_L = MAX_L;
    this->kv_cache_size = MAX_L * (desc->DK + desc->DV);
}

void minicpm_v_4_7_moe_npu_sequence::gen_lm_head_seq(npu_sequence* seq, int VOCAB_SIZE, int D){
    static constexpr int y_arg_id = 0;
    static constexpr int w_arg_id = 1;
    static constexpr int x_arg_id = 2;
    static constexpr int M_PER_ROUND = 32 * 32;
    static constexpr int M    = 32;
    static constexpr int M_PER_COL    = M * 4;
    static constexpr int COLS = 8;
    static npu_tiles ITs[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    // Packet ids for the two flows that share IT0's MM2S 0 -- must match X_PKT_ID /
    // W_PKT_ID in FLM_Xclbin/Qwen3_5/lm_head_npu_bin/iron/q4k_mm.py, which is where
    // this model's lm_head.xclbin is built from. Every other shim MM2S port carries
    // a single circuit-switched flow.
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
    // mcvmoe_reorder_cpy_lm_head has already laid the two halves out back to back.
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

void minicpm_v_4_7_moe_npu_sequence::gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act, flm_dtype_t dtype){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr int m = QXNX_ROW_BLOCK_SIZE;
    static constexpr int k = QXNX_COL_BLOCK_SIZE;
    static constexpr uint32_t elems = m * k;
    static constexpr uint32_t desired_m = 128;
    static constexpr uint32_t desired_k = 512;
    static constexpr uint32_t desired_n = 64;

    uint32_t a_block_size = get_quantization_byte_size(m * k, flm_q4k);

    const uint32_t blocks_per_row = Din / k;
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
            uint32_t y_offset = i * desired_m * 4 * Dout + j * desired_n * columns + out_offset;
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
                    {0, 0, 0, y_offset + col * desired_m * Dout},
                    {1, 1, desired_m, desired_n * 4},
                    {0, 0, Dout, 1},
                    -1, 0, true
                );
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col + 4],
                    y_bd_1, it_channel_0,
                    {0, 0, 0, y_offset + col * desired_m * Dout + 4 * desired_n},
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

void minicpm_v_4_7_moe_npu_sequence::gen_dequant_mm(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const int rtp_act, flm_dtype_t dtype){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr int m = QXNX_ROW_BLOCK_SIZE;
    static constexpr int k = QXNX_COL_BLOCK_SIZE;
    static constexpr uint32_t elems = m * k;
    static constexpr uint32_t desired_m = 64;
    static constexpr uint32_t desired_k = 512;
    static constexpr uint32_t desired_n = 128;

    uint32_t a_block_size = get_quantization_byte_size(m * k, flm_q4k);

    const uint32_t blocks_per_row = Din / k;
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
                    {0, 0, 0, y_offset + col * desired_m * Dout},
                    {1, 1, desired_m, desired_n * 4},
                    {0, 0, Dout, 1},
                    -1, 0, true
                );
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col + 4],
                    y_bd_1, it_channel_0,
                    {0, 0, 0, y_offset + col * desired_m * Dout + 4 * desired_n},
                    {1, 1, desired_m, desired_n * 4},
                    {0, 0, Dout, 1},
                    -1, 0, true
                );
                // send x
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[col * 2],
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

void minicpm_v_4_7_moe_npu_sequence::gen_mm_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t weight_offset, uint32_t out_offset, int32_t rtp_act){
    const size_t m = 64;
    const size_t k = 512;
    const size_t n = 64;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    int K_div_k = K / k;
    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr int CT_rtp_sync_lock_id = 6;
    assert(N % 256 == 0);
    seq->clear_cmds();

    for (int row = 2; row < 6; row++){
        for (int col = 0; col < total_cols; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, mm_CT_rtp_address, K_div_k);
            seq->rtp_write(tile, mm_act_rtp_address, rtp_act);
            seq->rtp_write(tile, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1); // set lock to 1
        }
    }

    // One tiled MM pass. `split` selects between the two layouts:
    //   split == true : 4 active columns, the weights are broadcast to both
    //                   column groups (BDs 2 and 5) and the second group reads
    //                   x/y from the M*K/2 / M*N/2 half of DDR.
    //   split == false: 8 active columns, a single weight BD per column and the
    //                   second group is the adjacent 4*n slice of the same row.
    auto run_tiles = [&](bool split, int M_div_m, int N_div_n, int y_base, uint32_t w_base){
        const int n_cols_used = split ? 4 : total_cols;
        // When split, the two column groups interleave the M dimension block by
        // block (block = m * total_rows rows): group 0 takes even blocks, group 1
        // the odd block immediately after. So the i stride doubles and the bd1
        // offset is a single block gap rather than the contiguous M/2 half.
        const int i_stride = split ? 2 : 1;
        const int y_bd1_extra = split ? (int)(m * N * total_rows) : (int)(4 * n);
        const int x_bd1_extra = split ? (int)(m * K * total_rows) : 0;
        for (int j = 0; j < N_div_n; j++){
            for (int i = 0; i < M_div_m; i++){
                int round = j * M_div_m + i;
                int bd_offset = (round % 2) * 8;
                int y_offset = (size_t)i * i_stride * m * N * total_rows + j * n * n_cols_used + y_base;
                int x_offset = (size_t)i * i_stride * m * K * total_rows;
                uint32_t w_offset = j * n * K * n_cols_used + w_base;
                npu_bd_id y_bd_0 = npu_bd_id(0 + bd_offset);
                npu_bd_id y_bd_1 = npu_bd_id(3 + bd_offset);
                npu_bd_id x_bd_0 = npu_bd_id(1 + bd_offset);
                npu_bd_id x_bd_1 = npu_bd_id(4 + bd_offset);
                npu_bd_id w_bd_0 = npu_bd_id(2 + bd_offset);
                npu_bd_id w_bd_1 = npu_bd_id(5 + bd_offset);
                // send w
                for (int c = 0; c < n_cols_used; c++){
                    seq->npu_dma_memcpy_nd(
                        2, 2,
                        MM2S, IT[c],
                        w_bd_0, it_channel_1,
                        {0, 0, 0, (uint32_t)(w_offset + c * n * K)},
                        {1, 1, 1, (uint32_t)(K_div_k * n * k)},
                        {0, 0, 0, 1},
                        -1, 0, false
                    );
                    if (split){
                        seq->npu_dma_memcpy_nd(
                            2, 2,
                            MM2S, IT[c + 4],
                            w_bd_1, it_channel_1,
                            {0, 0, 0, (uint32_t)(w_offset + c * n * K)},
                            {1, 1, 1, (uint32_t)(K_div_k * n * k)},
                            {0, 0, 0, 1},
                            -1, 0, false
                        );
                    }
                }
                for (int r = 0; r < total_rows; r++){
                    // receive y
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[r],
                        y_bd_0, it_channel_0,
                        {0, 0, 0, (uint32_t)(y_offset + r * m * N)},
                        {1, 1, (uint32_t)m, (uint32_t)(n * 4)},
                        {0, 0, N, 1},
                        -1, 0, true
                    );
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[r + 4],
                        y_bd_1, it_channel_0,
                        {0, 0, 0, (uint32_t)(y_offset + r * m * N + y_bd1_extra)},
                        {1, 1, (uint32_t)m, (uint32_t)(n * 4)},
                        {0, 0, N, 1},
                        -1, 0, true
                    );
                    // send x
                    seq->npu_dma_memcpy_nd(
                        2, 1,
                        MM2S, IT[r],
                        x_bd_0, it_channel_0,
                        {0, 0, 0, (uint32_t)(x_offset + r * m * K)},
                        {1, (uint32_t)K_div_k, (uint32_t)m, (uint32_t)k},
                        {0, (uint32_t)k, K, 1},
                        -1, 0, false
                    );
                    seq->npu_dma_memcpy_nd(
                        2, 1,
                        MM2S, IT[r + 4],
                        x_bd_1, it_channel_0,
                        {0, 0, 0, (uint32_t)(x_offset + r * m * K + x_bd1_extra)},
                        {1, (uint32_t)K_div_k, (uint32_t)m, (uint32_t)k},
                        {0, (uint32_t)k, K, 1},
                        -1, 0, false
                    );
                }
                if (round > 0){
                    for (int r = 0; r < total_rows; r++){
                        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                        seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
                    }
                }
            }
        }
        for (int r = 0; r < total_rows; r++){
            seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
            seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
        }
    };

    // N is always a multiple of 256, so N % 512 is either 0 or 256. The general
    // layout is an 8-column pass over the 512-aligned region followed by a
    // 4-column split pass over the trailing 256 (if any). The two historical
    // "only 4 columns" (N < 512) and "only 8 columns" (N % 512 == 0) cases are
    // just the degenerate ends where one of the two passes has zero width.
    const int M_div_m = M / m / total_rows;
    const int D_main  = (N / 512) * 512;  // 8-column region (multiple of 512)
    const int D_rem   = N - D_main;       // 0 or 256
    if (D_main > 0){
        run_tiles(false, M_div_m, D_main / n / total_cols, out_offset, weight_offset);
    }
    if (D_rem > 0){
        run_tiles(true, M_div_m / 2, D_rem / n / 4, D_main + out_offset, weight_offset + D_main * K);
    }
    seq->cmds2seq();
}

void minicpm_v_4_7_moe_npu_sequence::gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_attn_layer, bool is_first){
    if (is_attn_layer) {
        _gen_sequence(seq, L);
    }
    else {
        _gen_linear_sequence(seq, is_first);
    }
}

void minicpm_v_4_7_moe_npu_sequence::_gen_linear_sequence(npu_sequence* seq, bool is_first) {
    minicpm_v_4_7_moe_linear_layer& _weight_desc = static_cast<minicpm_v_4_7_moe_linear_layer&>(*desc->weight_desc[desc->lin_idx]);
    constexpr size_t CT_lock_address_base = 0x000001F000;
    int router_shimtile_configure_ready_lock = 10;
    seq->clear_cmds();

    _send_hidden_states(seq);
    _send_rms_weights(seq);
    _send_linear_conv_weights(seq);

    // // receive y
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx,
        S2MM,
        IT2,
        bd_10,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)desc->D},
        {0, 0, 0, 1},
        -1, 0, true
    );

    _move_linear_kv_cache(seq);
    _move_alpha_beta_weights(seq, desc->linear_num_v_heads_padded * 2, desc->D, _weight_desc.ssm_alpha.offset);
    _move_weights(seq, desc->Linear_DQ + desc->Linear_DK + desc->Linear_DV, desc->D, _weight_desc.linear_attn_qkv);
    _receive_linear_kv_cache(seq); // also move qkv weights
    _move_s(seq);
    _move_weights(seq, desc->Linear_DV, desc->D, _weight_desc.attn_gate);
    _move_weights(seq, desc->D, desc->Linear_DV, _weight_desc.ssm_out, w_rope_rms_arg_idx);

    for(int i = 0; i < 4; i++){
        setup_expert_up_gate_q4k(
            seq, i,
            desc->intermediate_size, desc->D,
            4, 5,
            _weight_desc.ffn_up_exps.offset);

        setup_expert_down_gate_q4k(
            seq, i,
            desc->intermediate_size, desc->D,
            6, 7,
            _weight_desc.ffn_up_exps.offset);
    }

    for(int i = 0; i < 4; i++){
        setup_expert_up_gate_q4k(
            seq, i,
            desc->intermediate_size, desc->D,
            12, 13,
            _weight_desc.ffn_up_exps.offset);

        setup_expert_down_gate_q4k(
            seq, i,
            desc->intermediate_size, desc->D,
            14, 15,
            _weight_desc.ffn_up_exps.offset);
    }

    auto router_CT = get_tile(4, 5);
    seq->npu_maskwrite(router_CT, 0x32038, 0x1, 0x1);
    seq->rtp_write(router_CT, CT_lock_address_base + 16 * router_shimtile_configure_ready_lock, 1); // set lock to 1

    _send_router_w_and_share_exp_gate(seq, desc->num_experts, desc->D, _weight_desc.ffn_gate_inp.offset , &_weight_desc);

    _move_weights(seq, 2 * desc->intermediate_size, desc->D, _weight_desc.ffn_up_shexp);
    _move_weights(seq, desc->D, desc->intermediate_size, _weight_desc.ffn_down_shexp);

    // // wait for receiving y
    seq->npu_dma_wait(
        IT2,
        S2MM,
        it_channel_1
    );
    seq->cmds2seq();
}

void minicpm_v_4_7_moe_npu_sequence::_gen_sequence(npu_sequence* seq, int L) {

    minicpm_v_4_7_moe_attn_layer& _weight_desc = static_cast<minicpm_v_4_7_moe_attn_layer&>(*desc->weight_desc[desc->attn_idx]);

    constexpr size_t CT_lock_address_base = 0x000001F000;
    int router_shimtile_configure_ready_lock = 10;
    seq->clear_cmds();
    _send_hidden_states(seq);
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

    for(int i = 0; i < 4; i++){
        setup_expert_up_gate_q4k(
            seq, i,
            desc->intermediate_size, desc->D,
            4, 5,
            _weight_desc.ffn_up_exps.offset);

        setup_expert_down_gate_q4k(
            seq, i,
            desc->intermediate_size, desc->D,
            6, 7,
            _weight_desc.ffn_up_exps.offset);
    }

    for(int i = 0; i < 4; i++){
        setup_expert_up_gate_q4k(
            seq, i,
            desc->intermediate_size, desc->D,
            12, 13,
            _weight_desc.ffn_up_exps.offset);

        setup_expert_down_gate_q4k(
            seq, i,
            desc->intermediate_size, desc->D,
            14, 15,
            _weight_desc.ffn_up_exps.offset);
    }

    auto router_CT = get_tile(4, 5);
    seq->npu_maskwrite(router_CT, 0x32038, 0x1, 0x1);
    seq->rtp_write(router_CT, CT_lock_address_base + 16 * router_shimtile_configure_ready_lock, 1); // set lock to 1

    _send_router_w_and_share_exp_gate(seq, desc->num_experts, desc->D, _weight_desc.ffn_gate_inp.offset , &_weight_desc);

    //TODO: consider send router weights and share_expert vector here

    // send share-expert up-gate weights
    _move_weights(seq, 2 * desc->intermediate_size, desc->D, _weight_desc.ffn_up_shexp);

    // send share-expert down weights
    _move_weights(seq, desc->D, desc->intermediate_size, _weight_desc.ffn_down_shexp);

    // wait for receiving y
    seq->npu_dma_wait(
        xyr_tile,
        S2MM,
        it_channel_1
    );

    seq->cmds2seq();
}

void minicpm_v_4_7_moe_npu_sequence::_send_hidden_states(npu_sequence* seq){
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

void minicpm_v_4_7_moe_npu_sequence::_send_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)desc->D * 3},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

void minicpm_v_4_7_moe_npu_sequence::_send_rope_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_2,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, desc->RoPE_DIM + desc->DH * 2},
        {0, 0, 0, 1},
        1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        xyr_tile,
        MM2S,
        it_channel_0
    );
}

void minicpm_v_4_7_moe_npu_sequence::_receive_kv_cache(npu_sequence* seq, const int L){
    uint32_t data_per_chunk = desc->num_kv_heads *  desc->DH;
    uint32_t v_offset = this->kv_cache_size / 2;
    uint32_t L_offset = (L - 1) * data_per_chunk;

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

void minicpm_v_4_7_moe_npu_sequence::_send_router_w_and_share_exp_gate(npu_sequence* seq, const size_t NUM_expert, const size_t D, const size_t W_offset_in_byte, mcvmoe_layer_weright_def* _weight_desc_ptr){

    //TODO: have this aligned with the MT parametesr
    // router weights is NUM_expert x D in row major

    // but transfer router_weight in block sof router_weight_row_block x router_wieght_col_block
    // each block is column major order
    // but the blocks are row-block major order

    //TODO: need to share same file with decode for sanity
    uint32_t router_weight_row_block = 32;
    uint32_t router_weight_col_block = 256;
    mcvmoe_layer_weright_def& _weight_desc = *_weight_desc_ptr;

    // assert(router_weight_row_block *router_weight_col_block == D );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        IT5,
        npu_bd_id(bd_12),
        it_channel_0,
        {0, 0, 0, (uint32_t)((W_offset_in_byte/2)  )},
        {       (uint32_t)NUM_expert/router_weight_row_block                      ,(uint32_t)D/router_weight_col_block           ,(uint32_t)router_weight_col_block    ,(uint32_t)router_weight_row_block},
        {       (uint32_t)router_weight_row_block,                            (uint32_t)NUM_expert*router_weight_col_block          ,(uint32_t)NUM_expert                 ,1},
        10, 0, false, aggressive_cache
    );
}

void minicpm_v_4_7_moe_npu_sequence::send_manual_expert_down_gate_q4k(

    npu_sequence*seq,
    int col, // the shimtile for this expert, 0-3
    int intermediate, int D,
    int expert_id, size_t const_w_byte_offset
){

    const int m = 32;
    const int k = 256;
    const int num_core_per_column=4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 37 / 64 ;

    //TODO: parameterized it
    //size_t expert_up_gate_block =  ((2*intermediate)/(m)) * ( D/k);
    size_t expert_down_block = (size_t)(D/m)*(intermediate/k);

    size_t cur_expert_w_offset_in_byte = const_w_byte_offset + expert_id*a_block_size * (expert_down_block );

    size_t INT_Q4nx_BLOCK_PER_ROW = intermediate/k;
    size_t down_col_number = D/ (num_core_per_column*m);

    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_6),
        it_channel_0,
        {0,0,0,(uint32_t)cur_expert_w_offset_in_byte},
        {(uint32_t)down_col_number,                        (uint32_t)INT_Q4nx_BLOCK_PER_ROW,    (uint32_t)(2*a_block_size)/256 ,256},
        {(uint32_t)(4*a_block_size*INT_Q4nx_BLOCK_PER_ROW),       (uint32_t)4*a_block_size,             (uint32_t)256,1},
        -1,0,true
    );// TODO: change l

    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_7),
        it_channel_1,
        {0,0,0,(uint32_t)(cur_expert_w_offset_in_byte + 2*a_block_size)},
        {(uint32_t)down_col_number,                        (uint32_t)INT_Q4nx_BLOCK_PER_ROW,    (uint32_t)(2*a_block_size)/256 ,256},
        {(uint32_t)(4*a_block_size*INT_Q4nx_BLOCK_PER_ROW),       (uint32_t)4*a_block_size,             (uint32_t)256,1},
        -1,0,true
    );// TODO: change l
}

void minicpm_v_4_7_moe_npu_sequence::setup_expert_down_gate_q4k(

    npu_sequence*seq,
    int col, // the shimtile for this expert, 0-3
    int intermediate, int D,
    int bd_id_channel_0, int bd_id_channel_1,
    size_t const_w_byte_offset
){

    const int m = 32;
    const int k = 256;
    const int num_core_per_column=4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 37 / 64 ;

    //TODO: parameterized it
    //size_t expert_up_gate_block =  ((2*intermediate)/(m)) * ( D/k);
    uint32_t expert_down_block = (D/m)*(intermediate/k);

    uint32_t cur_expert_w_offset_in_byte = const_w_byte_offset;// + expert_id*a_block_size * (expert_down_block );

    uint32_t INT_Q4nx_BLOCK_PER_ROW = intermediate/k;
    uint32_t down_col_number = D/ (num_core_per_column*m);
    //TODO: sync with the kernel through share hpp
    constexpr uint32_t SHIM_6_CONTROL_RES_PKT = 30;

    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_id_channel_0),
        it_channel_0,
        {0,0,0,cur_expert_w_offset_in_byte},
        {down_col_number,                        INT_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*INT_Q4nx_BLOCK_PER_ROW,       4*a_block_size,             256,1},
        -1,0,false,
        aggressive_cache,
        SHIM_6_CONTROL_RES_PKT, false
    );// TODO: change l

    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_id_channel_1),
        it_channel_1,
        {0,0,0,cur_expert_w_offset_in_byte },
        {down_col_number,                        INT_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*INT_Q4nx_BLOCK_PER_ROW,       4*a_block_size,             256,1},
        -1,0,false,
        aggressive_cache,
        SHIM_6_CONTROL_RES_PKT, false
    );// TODO: change l
}

void minicpm_v_4_7_moe_npu_sequence::setup_expert_up_gate_q4k(
    npu_sequence*seq,
    int col, // the shimtile for this expert, 0-3
    int intermediate, int D,
    int bd_id_channel_0, int bd_id_channel_1,
    size_t const_w_byte_offset
){

    const int m = 32;
    const int k = 256;
    const int num_core_per_column=4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 37 / 64 ;

    //TODO: parameterized it
    uint32_t expert_up_gate_block =  ((2*intermediate)/(m)) * ( D/k);
    //size_t expert_down_block = (D/m)*(intermediate/k);

    uint32_t cur_expert_w_offset_in_byte = const_w_byte_offset;// + expert_id* size_t(a_block_size* expert_up_gate_block );

    uint32_t D_Q4nx_BLOCK_PER_ROW = D/k;
    uint32_t gate_up_col_number = intermediate/ (num_core_per_column*m);
    //TODO: sync with the kernel through share hpp
    constexpr uint32_t SHIM_6_CONTROL_RES_PKT = 30;
    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_id_channel_0),
        it_channel_0,
        {0,0,0,cur_expert_w_offset_in_byte},
        {gate_up_col_number*2,                        D_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*D_Q4nx_BLOCK_PER_ROW, 4*a_block_size, 256,1},

        -1,0,false,
        aggressive_cache,
        SHIM_6_CONTROL_RES_PKT, false
    );// TODO: change later, not submit to queue by default

    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_id_channel_1), //TODO: different number for down?
        it_channel_1,
        {0,0,0,cur_expert_w_offset_in_byte },
        {gate_up_col_number*2,                        D_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*D_Q4nx_BLOCK_PER_ROW, 4*a_block_size, 256,1},

        -1,0,false,
        aggressive_cache,
        SHIM_6_CONTROL_RES_PKT, false
    );
}

void minicpm_v_4_7_moe_npu_sequence::send_manual_expert_up_gate_q4k(
    npu_sequence*seq,
    int col, // the shimtile for this expert, 0-3
    int intermediate, int D,
    int expert_id, size_t const_w_byte_offset
){

    const int m = 32;
    const int k = 256;
    const int num_core_per_column=4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 37 / 64 ;

    //TODO: parameterized it
    uint32_t expert_up_gate_block =  ((2*intermediate)/(m)) * ( D/k);
    //size_t expert_down_block = (D/m)*(intermediate/k);

    uint32_t cur_expert_w_offset_in_byte = const_w_byte_offset + expert_id* size_t(a_block_size* expert_up_gate_block );

    uint32_t D_Q4nx_BLOCK_PER_ROW = D/k;
    uint32_t gate_up_col_number = intermediate/ (num_core_per_column*m);

    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_4),
        it_channel_0,
        {0,0,0,cur_expert_w_offset_in_byte},
        {gate_up_col_number*2,                        D_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*D_Q4nx_BLOCK_PER_ROW, 4*a_block_size, 256,1},

        -1,0,true
    );// TODO: change later, not submit to queue by default

    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_5), //TODO: different number for down?
        it_channel_1,
        {0,0,0,cur_expert_w_offset_in_byte +(2*a_block_size) },
        {gate_up_col_number*2,                        D_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*D_Q4nx_BLOCK_PER_ROW, 4*a_block_size, 256,1},

        -1,0,true
    );
}

void minicpm_v_4_7_moe_npu_sequence::_move_weights(npu_sequence* seq, const uint32_t Dout, const uint32_t Din, const weight_desc_t& weight, const int arg_id){
    const size_t w_offset = weight.offset;
    const flm_dtype_t dtype = weight.dtype;
    const int m = 32;
    const int k = 256;
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
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
                arg_id,
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
                arg_id,
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

void minicpm_v_4_7_moe_npu_sequence::_move_kv_cache(npu_sequence* seq, const size_t L){
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
        bd_1,
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
        bd_2,
        it_channel_1,
        {0, 0, 0, (uint32_t)(v_offset)},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
}

void minicpm_v_4_7_moe_npu_sequence::_send_linear_conv_weights(npu_sequence* seq){
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

void minicpm_v_4_7_moe_npu_sequence::_receive_linear_kv_cache(npu_sequence* seq){
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

void minicpm_v_4_7_moe_npu_sequence::_move_alpha_beta_weights(npu_sequence* seq, const uint32_t Dout, const uint32_t Din, const uint32_t w_offset){
    // currently, only on DMA channel 0 of IT5
    size_t router_weight_row_block = 32;
    size_t router_weight_col_block = 256;

    assert(Dout %2 == 0);

    // send alpha
    const size_t alpha_beta_Dout = Dout/2;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_rms_arg_idx,
        MM2S,
        IT5,
        npu_bd_id(bd_13),
        it_channel_0,
        {0, 0, 0, (uint32_t)w_offset / sizeof(bf16)},
        {       (uint32_t)(alpha_beta_Dout/router_weight_row_block)                      ,(uint32_t)(Din/router_weight_col_block)           ,(uint32_t)router_weight_col_block    ,(uint32_t)router_weight_row_block},
        {       (uint32_t)router_weight_row_block,                            (uint32_t)(alpha_beta_Dout*router_weight_col_block)          ,(uint32_t)alpha_beta_Dout                 ,1},
        3, 0, false, aggressive_cache
    );

    // send beta
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_rms_arg_idx,
        MM2S,
        IT5,
        npu_bd_id(bd_14),
        it_channel_0,
        {0, 0, 0, (uint32_t)((uint32_t)w_offset / sizeof(bf16) + alpha_beta_Dout*Din)},
        {       (uint32_t)(alpha_beta_Dout/router_weight_row_block)                      ,(uint32_t)(Din/router_weight_col_block)           ,(uint32_t)router_weight_col_block    ,(uint32_t)router_weight_row_block},
        {       (uint32_t)router_weight_row_block,                            (uint32_t)(alpha_beta_Dout*router_weight_col_block)          ,(uint32_t)alpha_beta_Dout                 ,1},
        3, 0, false, aggressive_cache
    );

    // seq->npu_dma_wait(
    //     IT5,
    //     MM2S,
    //     it_channel_0
    // );
}

void minicpm_v_4_7_moe_npu_sequence::_move_linear_kv_cache(npu_sequence* seq){
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

void minicpm_v_4_7_moe_npu_sequence::_move_s(npu_sequence* seq){
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
}

void minicpm_v_4_7_moe_npu_sequence::gen_mha_engine_seq(
    npu_sequence* seq,
    const uint32_t L_begin,
    const uint32_t L
){

    static constexpr int lc = 16;
    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};
    assert((L) % (lc * 8) == 0);
    int using_window_size = MAX_L;
    const int Q_chunks = L / (lc * 8);
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
    int Heads = desc->num_heads;
    int DH = desc->DH;
    int Num_of_d_in_KV_cache = desc->num_kv_heads;

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
                    size_t y_offset = head_offset * DH + (round * data_per_round + col * lc * 4) * DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)4 * lc, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, true
                    );
                }
                // send q
                size_t q_offset = head_offset * DH + round * data_per_round * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu

            int kv_head_offset = head / 2;
            int kv_chunk_offset = kv_head_offset / Num_of_d_in_KV_cache;
            int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;

            size_t k_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[1][1],
                (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
                -1, 0, false
            );
            size_t v_offset = kv_chunk_offset * MAX_L * DH * Num_of_d_in_KV_cache + kv_head_offset_in_chunk * DH + kv_begin * DH * Num_of_d_in_KV_cache + kv_cache_size / 2;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[1][1],
                (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                {0, (uint32_t)128 * DH * Num_of_d_in_KV_cache, (uint32_t)DH * Num_of_d_in_KV_cache, 1},
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

void minicpm_v_4_7_moe_npu_sequence::gen_seq_conv1d(
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
minicpm_v_4_7_moe_npu_sequence::~minicpm_v_4_7_moe_npu_sequence() = default;
