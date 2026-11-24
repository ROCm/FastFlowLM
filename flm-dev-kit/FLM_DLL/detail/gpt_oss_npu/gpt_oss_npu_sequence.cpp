// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "gpt_oss_npu_sequence_detail.hpp"

void gpt_oss_npu_sequence::Impl::_receive_kv_cache(npu_sequence* seq_ptr, const int L, bool is_sliding_window){
    const int sliding_window = 128;
    int L_local = is_sliding_window ? (L - 1) % sliding_window : L - 1;
    const uint32_t kv_cache_offset = L_local * 4 * DH;

    uint32_t local_k03_offset = is_sliding_window ? K03_sliding_offset : K03_offset;
    uint32_t local_v03_offset = is_sliding_window ? V03_sliding_offset : V03_offset;
    uint32_t local_k47_offset = is_sliding_window ? K47_sliding_offset : K47_offset;
    uint32_t local_v47_offset = is_sliding_window ? V47_sliding_offset : V47_offset;
    npu_sequence& seq = *seq_ptr;

    //receive k, V
    seq.npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx, S2MM, // put it to the this buffer for now
        IT0,
        (npu_bd_id)0, it_channel_0,
        {0,0,0, kv_cache_offset + (uint32_t)local_k03_offset},  // this is a ring buffer like
        {1, 1, 1, GPTOSS_DK / 2},
        {0,0,0,1},
        -1, 0, true
    );

    seq.npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx, S2MM, // put it to the this buffer for now
        IT7,
        (npu_bd_id)10, it_channel_0,
        {0,0,0, kv_cache_offset + (uint32_t)local_v03_offset},// this is a ring buffer like
        {1,1,1, GPTOSS_DV / 2},
        {0,0,0,1},
        -1, 0, true
    );

    seq.npu_dma_wait( IT0, S2MM, it_channel_0);
    seq.npu_dma_wait( IT7, S2MM, it_channel_0);
    seq.npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx, S2MM, // put it to the this buffer for now
        IT0,
        (npu_bd_id)0, it_channel_0,
        {0,0,0, kv_cache_offset + (uint32_t)local_k47_offset},  // this is a ring buffer like
        {1, 1, 1, GPTOSS_DK / 2},
        {0,0,0,1},
        -1, 0, true
    );

    seq.npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx, S2MM, // put it to the this buffer for now
        IT7,
        (npu_bd_id)10, it_channel_0,
        {0,0,0, kv_cache_offset + (uint32_t)local_v47_offset},// this is a ring buffer like
        {1,1,1, GPTOSS_DV / 2},
        {0,0,0,1},
        -1, 0, true
    );

    seq.npu_dma_wait( IT0, S2MM, it_channel_0);
    seq.npu_dma_wait( IT7, S2MM, it_channel_0);
}

void gpt_oss_npu_sequence::Impl::_move_weights(npu_sequence* seq,
    const size_t Dout, const size_t Din, const size_t w_offset){
    static constexpr npu_tiles mvm_tiles[] = {IT2, IT3, IT4, IT5};

    const uint32_t total_round = Dout / (Q4NX_ROW * MVM_cores);
    constexpr int columns = MVM_cores/4;

    constexpr uint32_t Q4NX_BLOCK_SIZE_BF16 = Q4NX_BLOCK_BYTE_SIZE / 2;
    //This should align with gpt_dequant_q41
    uint32_t blocks_per_row = Din / Q4NX_COL;
    for (size_t round = 0; round < total_round; round++){
        uint32_t bd_offset = (round % 2) * 4;
        for (uint32_t col = 0; col < columns; col++){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_rms_arg_idx,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_1 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)((round * MVM_cores + col * 4) * ( Q4NX_BLOCK_SIZE_BF16*blocks_per_row)  + (uint32_t)w_offset)},
                { 1,    blocks_per_row ,       (2*Q4NX_BLOCK_SIZE_BF16)/128 , 128},
                { 0,    4*Q4NX_BLOCK_SIZE_BF16 ,                      128,    1},
                -1, 0, true,
                aggressive_cache
            );

            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_rms_arg_idx,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_2 + bd_offset),
                it_channel_1,
                {0, 0, 0, (uint32_t)((round * MVM_cores + col * 4) * ( Q4NX_BLOCK_SIZE_BF16*blocks_per_row)  + (uint32_t)w_offset + 2*Q4NX_BLOCK_SIZE_BF16)},
                { 1,    blocks_per_row ,       (2*Q4NX_BLOCK_SIZE_BF16)/128 , 128},
                { 0,    4*Q4NX_BLOCK_SIZE_BF16 ,                      128,    1},
                -1, 0, true,
                aggressive_cache
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
    }
}
void gpt_oss_npu_sequence::Impl::_move_kv_cache(npu_sequence* seq, const int L, bool is_sliding_window){
    // cases:

    // 1. not sliding window, for 0 -> L - 1
    // 2. sliding window:
    // 2.1 L <= Sliding_Window_Size, for 0 -> L - 1
    // 2.2 L > Sliding_Window_Size:
    // 2.2.1 L % Sliding_Window_Size == 0, for 0 -> L - 1
    // 2.2.2 L % Sliding_Window_Size != 0, L % Sliding_Window_Size -> Sliding_Window_Size - 1, 0 -> (L % Sliding_Window_Size) - 1
    const uint32_t DH = 64;
    const int DK = 512;
    const int DV = 512;
    const int Sliding_Window_Size = 128;
    const int using_L = is_sliding_window ? ((L > Sliding_Window_Size) ? Sliding_Window_Size : L) : L;
    const int padded_window_size = (using_L + 15) / 16 * 16;

    npu_tiles attn_qk_cores[4] = {CT00, CT20, CT07, CT27};
    npu_tiles attn_kv_cores[4] = {CT10, CT30, CT17, CT37};

    if (is_sliding_window){
        if (L % Sliding_Window_Size == 0){ // for 0 -> L - 1
            uint32_t kv_offset = 0;
            uint32_t kv_length = Sliding_Window_Size * DK / 2;
            // move in k03
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                dma_direction::MM2S,
                IT0,
                bd_1,
                it_channel_0,
                {0, 0, 0, kv_offset},
                {1, 1, 1, kv_length},
                {0, 0, 0, 1},
                -1, 0, false
            );

            // move in v03
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                dma_direction::MM2S,
                IT0,
                bd_2,
                it_channel_1,
                {0, 0, 0, (uint32_t)V03_sliding_offset + kv_offset},
                {1, 1, 1, kv_length},
                {0, 0, 0, 1},
                -1, 0, false
            );

            // move in k47
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                dma_direction::MM2S,
                IT7,
                bd_2,
                it_channel_0,
                {0, 0, 0, (uint32_t)K47_sliding_offset + kv_offset},
                {1, 1, 1, kv_length},
                {0, 0, 0, 1},
                -1, 0, false
            );

            // move in v47
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                dma_direction::MM2S,
                IT7,
                bd_3,
                it_channel_1,
                {0, 0, 0, (uint32_t)V47_sliding_offset + kv_offset},
                {1, 1, 1, kv_length},
                {0, 0, 0, 1},
                -1, 0, false
            );
            return;
        }
        else if (L > Sliding_Window_Size){
            uint32_t cut = L % Sliding_Window_Size;
            uint32_t kv_offset_0 = cut * DK / 2;
            uint32_t kv_offset_1 = 0;
            int L_phase_0 = Sliding_Window_Size - cut;
            int L_phase_1 = cut;
            uint32_t kv_length_0 = L_phase_0 * DK / 2;
            uint32_t kv_length_1 = L_phase_1 * DK / 2;
            // phase 0
            // move in k03
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                dma_direction::MM2S,
                IT0,
                bd_5,
                it_channel_0,
                {0, 0, 0, kv_offset_0},
                {1, 1, 1, kv_length_0},
                {0, 0, 0, 1},
                -1, 0, false
            );

            // move in v03
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                dma_direction::MM2S,
                IT0,
                bd_6,
                it_channel_1,
                {0, 0, 0, (uint32_t)V03_sliding_offset + kv_offset_0},
                {1, 1, 1, kv_length_0},
                {0, 0, 0, 1},
                -1, 0, false
            );
            // move in k47
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                dma_direction::MM2S,
                IT7,
                bd_7,
                it_channel_0,
                {0, 0, 0, (uint32_t)K47_sliding_offset + kv_offset_0},
                {1, 1, 1, kv_length_0},
                {0, 0, 0, 1},
                -1, 0, false
            );

            // move in v47
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                dma_direction::MM2S,
                IT7,
                bd_8,
                it_channel_1,
                {0, 0, 0, (uint32_t)V47_sliding_offset + kv_offset_0},
                {1, 1, 1, kv_length_0},
                {0, 0, 0, 1},
                -1, 0, false
            );

            // phase 1

            // move in k03
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                dma_direction::MM2S,
                IT0,
                bd_1,
                it_channel_0,
                {0, 0, 0, kv_offset_1},
                {1, 1, 1, kv_length_1},
                {0, 0, 0, 1},
                -1, 0, false
            );

            // move in v03
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                dma_direction::MM2S,
                IT0,
                bd_2,
                it_channel_1,
                {0, 0, 0, (uint32_t)V03_sliding_offset + kv_offset_1},
                {1, 1, 1, kv_length_1},
                {0, 0, 0, 1},
                -1, 0, false
            );

            // move in k47
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                dma_direction::MM2S,
                IT7,
                bd_2,
                it_channel_0,
                {0, 0, 0, (uint32_t)K47_sliding_offset + kv_offset_1},
                {1, 1, 1, kv_length_1},
                {0, 0, 0, 1},
                -1, 0, false
            );

            // move in v47
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                dma_direction::MM2S,
                IT7,
                bd_3,
                it_channel_1,
                {0, 0, 0, (uint32_t)V47_sliding_offset + kv_offset_1},
                {1, 1, 1, kv_length_1},
                {0, 0, 0, 1},
                -1, 0, false
            );
            return;
        }
    }
    // fall back to no sliding window, for 0 -> L - 1
    uint32_t kv_offset = 0;
    uint32_t kv_length = padded_window_size * DK / 2;
    uint32_t k47_offset = is_sliding_window ? K47_sliding_offset : K47_offset;
    uint32_t v47_offset = is_sliding_window ? V47_sliding_offset : V47_offset;
    uint32_t v03_offset = is_sliding_window ? V03_sliding_offset : V03_offset;
    // move in k03
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        dma_direction::MM2S,
        IT0,
        bd_1,
        it_channel_0,
        {0, 0, 0, kv_offset},
        {1, 1, 1, kv_length},
        {0, 0, 0, 1},
        -1, 0, false
    );

    // move in v03
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        dma_direction::MM2S,
        IT0,
        bd_2,
        it_channel_1,
        {0, 0, 0, v03_offset + kv_offset},
        {1, 1, 1, kv_length},
        {0, 0, 0, 1},
        -1, 0, false
    );

    // move in k47
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        dma_direction::MM2S,
        IT7,
        bd_2,
        it_channel_0,
        {0, 0, 0, k47_offset + kv_offset},
        {1, 1, 1, kv_length},
        {0, 0, 0, 1},
        -1, 0, false
    );

    // move in v47
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        dma_direction::MM2S,
        IT7,
        bd_3,
        it_channel_1,
        {0, 0, 0, v47_offset + kv_offset},
        {1, 1, 1, kv_length},
        {0, 0, 0, 1},
        -1, 0, false
    );
    return;
}

gpt_oss_npu_sequence::Impl::Impl(LM_Config config, uint32_t MAX_L){

    this->MAX_L = MAX_L;

    this->kv_cache_size = MAX_L * (GPTOSS_DK + GPTOSS_DV);

    K03_offset = 0;
    K47_offset = kv_cache_size / 4;
    V03_offset = kv_cache_size / 2;
    V47_offset = 3 * kv_cache_size / 4;
    size_t kv_cache_size_sliding = 128 * GPTOSS_DV * 2;
    this->K03_sliding_offset = 0;
    this->K47_sliding_offset = kv_cache_size_sliding / 4;
    this->V03_sliding_offset = kv_cache_size_sliding / 2;
    this->V47_sliding_offset = 3 * kv_cache_size_sliding / 4;
}

void gpt_oss_npu_sequence::Impl::gen_layer_seq(npu_sequence* seq_ptr, const int L, bool is_sliding_window){
    assert(L <= MAX_L + 1);
    assert(L > 0);

    static constexpr npu_tiles shim_tiles[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    static constexpr npu_tiles mvm_tiles[] = {IT2, IT3, IT4, IT5};

    npu_sequence& seq = *seq_ptr;
    seq.clear_cmds();
    seq.npu_preemption(0);

    const int Sliding_Window_Size = 128;
    const int using_L = is_sliding_window ? ((L > Sliding_Window_Size) ? Sliding_Window_Size : L) : L;

    const uint32_t L_qk_addr = 20480;
    const uint32_t L_kv_addr = 51200;

    size_t decode_MVM_rtp_address = 19040;
    size_t decode_MVM_rtp_lock_id = 6;

    size_t rms_rtp_address = 58752;
    size_t rms_rtp_lock_id = 8;

    size_t CT_lock_address_base = 0x000001F000;
    npu_tiles attn_qk_cores[4] = {CT00, CT20, CT07, CT27};
    npu_tiles attn_kv_cores[4] = {CT10, CT30, CT17, CT37};

    auto rms_tile = get_tile(2, 1);
    seq.rtp_write(rms_tile, rms_rtp_address, 1);
    seq.rtp_write(rms_tile, CT_lock_address_base+16*(rms_rtp_lock_id), 1);

    for(int row_idx = 0; row_idx < 4; row_idx++){
        for(int col_idx = 2; col_idx < 6; col_idx++){
            auto tile = get_tile(row_idx + 2, col_idx);
            // disable the control packet sending first

            seq.rtp_write(tile, decode_MVM_rtp_address, 1);
            seq.rtp_write(tile, CT_lock_address_base+16*(decode_MVM_rtp_lock_id), 1); // set lock to 1
        }
    }

    {
        uint32_t shim_control_id [4] = {SHIM_2_CONTROL_PKT, SHIM_3_CONTROL_PKT, SHIM_4_CONTROL_PKT,  SHIM_5_CONTROL_PKT};

        // This should align with gpt_dequant_mxfp4

        constexpr uint32_t D_Q4NX_BLOCK_PER_ROW = D_PADDED_Q4NX_COL/Q4NX_COL;
        constexpr uint32_t INT_Q4NX_BLOCK_PER_ROW = INTERMEDIATE_PADDED_Q4NX_COL/Q4NX_COL;
        constexpr uint32_t Down_col_number = D_PADDED_FOUR_Q4NX_ROW/(4*Q4NX_ROW);
        constexpr uint32_t GATE_UP_col_number = INTERMEDIATE_PADDED_FOUR_Q4NX_ROW/(4*Q4NX_ROW);

        //TODO: also the bias later
        for(int k = 2; k < 6; k++){

            //  send gate, up configuration
            seq.npu_dma_memcpy_nd(
                sizeof(uint8_t),
                w_proj_arg_idx, MM2S,
                shim_tiles[k],
                (npu_bd_id)SHIM_X_gateup_BD_ID, (npu_it_channel)(0), //channel 0
                {  0,0,0, 0 }, //has to be 0
                { GATE_UP_col_number*2,                                   D_Q4NX_BLOCK_PER_ROW ,       (2*Q4NX_BLOCK_BYTE_SIZE)/128 , 128},
                { 4*Q4NX_BLOCK_BYTE_SIZE * D_Q4NX_BLOCK_PER_ROW,     4*Q4NX_BLOCK_BYTE_SIZE ,                      128,            1},
                -1, 0, false,
                aggressive_cache,
                shim_control_id[k-2], false
            );

            seq.npu_dma_memcpy_nd(
                sizeof(uint8_t),
                w_proj_arg_idx, MM2S,
                shim_tiles[k],
                (npu_bd_id)SHIM_X2_gateup_BD_ID, (npu_it_channel)(1), //channel 1
                {  0,0,0, 0 }, //has to be 0, offset will be given by control packet
                { GATE_UP_col_number*2,                                   D_Q4NX_BLOCK_PER_ROW ,       (2*Q4NX_BLOCK_BYTE_SIZE)/128 , 128},
                { 4*Q4NX_BLOCK_BYTE_SIZE * D_Q4NX_BLOCK_PER_ROW,     4*Q4NX_BLOCK_BYTE_SIZE ,                      128,            1},
                -1, 0, false,
                aggressive_cache,
                shim_control_id[k-2], false
            );

            seq.npu_dma_memcpy_nd(
                sizeof(uint8_t),
                w_proj_arg_idx, MM2S,
                shim_tiles[k],
                (npu_bd_id)SHIM_X_down_BD_ID, (npu_it_channel)(0), //channel 0
                {  0,0,0, 0 }, //has to be 0
                { Down_col_number,                                  INT_Q4NX_BLOCK_PER_ROW ,       (2*Q4NX_BLOCK_BYTE_SIZE)/128 , 128},
                { 4*Q4NX_BLOCK_BYTE_SIZE*INT_Q4NX_BLOCK_PER_ROW,    4*Q4NX_BLOCK_BYTE_SIZE ,                      128,             1},
                -1, 0, false,
                aggressive_cache,
                shim_control_id[k-2], false
            );

            seq.npu_dma_memcpy_nd(
                sizeof(uint8_t),
                w_proj_arg_idx, MM2S,
                shim_tiles[k],
                (npu_bd_id)SHIM_X2_down_BD_ID, (npu_it_channel)(1), //channel 1
                {  0,0,0, 0 }, //has to be 0, offset will be given by control packet
                { Down_col_number,                                  INT_Q4NX_BLOCK_PER_ROW ,       (2*Q4NX_BLOCK_BYTE_SIZE)/128 , 128},
                { 4*Q4NX_BLOCK_BYTE_SIZE*INT_Q4NX_BLOCK_PER_ROW,    4*Q4NX_BLOCK_BYTE_SIZE ,                      128,             1},
                -1, 0, false,
                aggressive_cache,
                shim_control_id[k-2], false
            );
            // send down configuration
        }
    }

    auto CT_6_3 = get_tile(3,6);
    seq.npu_maskwrite(CT_6_3, 0x32038, 0x1, 0x1);
    // at this point, set the lock tha allow CT_6_3 to send control packet
    seq.npu_maskwrite(CT_6_3, 0x000001F0F0, 0x1, 0x1F);// set lock 15 of CT_6_#

    // set rtp
    for (int i = 0; i < 4; i++){
        seq.rtp_write(attn_qk_cores[i], L_qk_addr, using_L);
        seq.rtp_write(attn_kv_cores[i], L_kv_addr, using_L);
    }

    // receive final X( after doing residual_2)
    seq.npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx, S2MM,
        shim_tiles[1],
        (npu_bd_id)12, it_channel_1, //bd_id=2, channle_1,
        {0,0,0,0},
        {1,1,1, D_ATTEN_O_PADDED_MVM_SIZE},
        {0,0,0,1},
        -1, 0, true
    );

    // send the x
    seq.npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx, MM2S,
        shim_tiles[1],
        (npu_bd_id)0, it_channel_0,
        {0,0,0,0},
        {1,1,1, D_ATTEN_O_PADDED_MVM_SIZE},
        {0,0,0,1},
        0,0,false
    );

    // send rms weight and o-bias
    seq.npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx, MM2S,
        shim_tiles[1],
        (npu_bd_id)1, it_channel_1,//send the input weight
        {0,0,0, rms_o_bias_offset_in_byte / 2},
        {1,1,1, 3 * GPTOSS_HIDDEN_SIZE},
        {0,0,0,1},
        0, 0, false
    );

    // seq.npu_dma_wait( shim_tiles[1], MM2S, it_channel_0);
    // send the q,k,v,bias
    seq.npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx, MM2S,
        shim_tiles[1],
        (npu_bd_id)2, it_channel_0,
        {0, 0, 0, PROJ_BIAS_OFFSET_IN_BYTE / 2},
        {1,1,1, GPTOSS_DQ + GPTOSS_DK + GPTOSS_DV + DH},
        {0,0,0,1},
        1, 0, false
    );

    // seq.npu_dma_wait( shim_tiles[1], MM2S, it_channel_0);

    // rope
    seq.npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_arg_idx,MM2S,
        shim_tiles[1],
        (npu_bd_id)3, it_channel_0,
        {0,0,0,0},
        {1,1,1, GPTOSS_HEAD_DIM},
        {0,0,0,1},
        1, 0, false
    );

    //
    _move_weights(
        seq_ptr,
        GPTOSS_DQ + GPTOSS_DK + GPTOSS_DV, GPTOSS_HIDDEN_SIZE_PADDED,
        (qkv_weight_offset_in_byte/2)
    );

    _receive_kv_cache(seq_ptr, L, is_sliding_window);

    _move_kv_cache(seq_ptr, L, is_sliding_window); // this is not used now.

    // Send O weight
    _move_weights(
        seq_ptr,
        D_ATTEN_O_PADDED_MVM_SIZE, GPTOSS_DQ,
        (qkv_weight_offset_in_byte + Q_WEIGHT_BYTE + K_WEIGHT_BYTE + V_WEIGHT_BYTE)/2
    );

    // send router W&B
    seq.npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx, MM2S,
        IT6,
        (npu_bd_id)11, it_channel_0,
        {0,0,0, (PROJ_BIAS_OFFSET_IN_BYTE) / 2 + GPTOSS_DQ + GPTOSS_DK + GPTOSS_DV + NUM_ATTN_HEADS},
        {1,1,1, EXPERT_WEIGHT_OFFSET_IN_BYTE / 2},
        {0,0,0,1},
        -1,0,false
    );

    //NOTE: for now

    // now, wait for issue token from control packet

    //DEBUG: check if finish all attention result
    // seq.npu_dma_wait(shim_tiles[6],MM2S,  it_channel_1 );  // check if consue all attention result

    // seq.npu_dma_wait(shim_tiles[1], MM2S, it_channel_1); // ensure rms weight and o-bias is finished

    seq.npu_dma_wait( shim_tiles[1], S2MM, it_channel_1);
    // seq.npu_maskwrite(CT_6_3, 0x32038, 0x0, 0x1);
    seq.cmds2seq();
}

void gpt_oss_npu_sequence::Impl::gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, bf16* sinks, bool is_sliding_window){
    const int lc = 16;
    const int WINDOW_SIZE = SLIDING_WINDOW;
    const int Heads = NUM_ATTN_HEADS;
    const int Num_of_d_in_KV_cache = 4;
    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};
    assert(L_begin % (lc * 8) == 0);
    assert(L_end % (lc * 8) == 0);
    int using_window_size = is_sliding_window ? WINDOW_SIZE : (L_end);
    seq->clear_cmds();

    // each CU has 8 CTs and works on 1 head.
    int all_data_size = L_end - L_begin;
    const int data_per_round = lc * 8;
    const int down_rounds = (all_data_size + data_per_round - 1) / data_per_round;

    int num_cu = 4;
    for (int head = 0; head < Heads / num_cu; head++){
        for (int cu = 0; cu < num_cu; cu++){
            int head_offset = head * num_cu + cu;
            for(int row = 2; row < 6; row++){
                for (int col = 0; col < 2; col++){
                    npu_tiles tile = get_tile(row, cu * 2 + col);
                    seq->rtp_write(tile, l_begin_mha_address, L_begin);
                    seq->rtp_write(tile, l_end_mha_address, L_end);
                    seq->rtp_write(tile, window_size_address, using_window_size);
                    float float_sink = (float)(sinks[head_offset]);
                    LOG_VERBOSE(1, "float_sink: " << float_sink << std::endl);
                    seq->rtp_write(tile, sinks_address, *(uint32_t*)(&(float_sink)));
                }
            }
        }
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_length = Lq_current - kv_begin + data_per_round;
            LOG_VERBOSE(1, "Round " << round << " Lq_current " << Lq_current << " kv_begin " << kv_begin << " kv_length " << kv_length << std::endl);

            int bd_offset = (round % 2) * 8;
            LOG_VERBOSE(1, "bd_offset: " << bd_offset << std::endl);

            for (int cu = 0; cu < num_cu; cu++){
                int head_offset = head * num_cu + cu;
                LOG_VERBOSE(1, "head_offset: " << head_offset << std::endl);
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
                    {0, 0, 0, (uint32_t)(q_offset + lc * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu

            int kv_head_offset = head / 2;
            int kv_chunk_offset = kv_head_offset / Num_of_d_in_KV_cache;
            int kv_head_offset_in_chunk = kv_head_offset % Num_of_d_in_KV_cache;
            LOG_VERBOSE(1, " kv_head_offset: " << kv_head_offset << " kv_chunk_offset: " << kv_chunk_offset << " kv_head_offset_in_chunk: " << kv_head_offset_in_chunk << std::endl);

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

void gpt_oss_npu_sequence::Impl::generate_lm_head_seq(npu_sequence* seq_ptr){
    static constexpr npu_tiles lm_head_tiles[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr int lm_head_y_arg_idx = 0;
    static constexpr int lm_head_w_arg_idx = 1;
    static constexpr int lm_head_x_arg_idx = 2;
    static constexpr int lm_head_rms_weight_arg_idx = 3;

    uint32_t q4nx_block_per_row = D_PADDED_DIV_Q4NX_COL;
    npu_sequence& seq = *seq_ptr;
    seq.clear_cmds();
    seq.npu_preemption(0);

    uint32_t blocks_per_row= q4nx_block_per_row;
    size_t decode_MVM_rtp_address = 19040;
    size_t decode_MVM_rtp_lock_id = 6;

    size_t rms_rtp_address = 58752;
    size_t rms_rtp_lock_id = 8;

    size_t CT_lock_address_base = 0x000001F000;

    auto rms_tile = get_tile(2, 1);
    seq.rtp_write(rms_tile, rms_rtp_address, 0);
    seq.rtp_write(rms_tile, CT_lock_address_base+16*(rms_rtp_lock_id), 1);

    for(int row_idx = 0; row_idx < 4; row_idx++){
        for(int col_idx = 2; col_idx < 6; col_idx++){
            auto tile = get_tile(row_idx + 2, col_idx);
            seq.rtp_write(tile, decode_MVM_rtp_address, 0);
            seq.rtp_write(tile, CT_lock_address_base+16*(decode_MVM_rtp_lock_id), 1); // set lock to 1
        }
    }
    static_assert(VOCAB_SIZE_PADDED %D_ATTEN_O_PADDED_MVM_SIZE == 0 );
    static constexpr npu_tiles shim_tiles[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    // first, setup y to receive the final data
    seq.npu_dma_memcpy_nd(
        sizeof(bf16),
        lm_head_y_arg_idx,S2MM,
        shim_tiles[1],
        (npu_bd_id)(12), it_channel_1,
        {0,0,0,0},
        {1, 1, 1, VOCAB_SIZE_PADDED},
        {0,0,0,1},
        -1, 0, true
    );

    // send the x
    seq.npu_dma_memcpy_nd(
        sizeof(bf16),
        lm_head_x_arg_idx, MM2S,
        shim_tiles[1],
        (npu_bd_id)0, it_channel_0,
        {0,0,0,0},
        {1,1,1, D_ATTEN_O_PADDED_MVM_SIZE},
        {0,0,0,1},
        0,0,false
    );
    //send the weight
    seq.npu_dma_memcpy_nd(
        sizeof(bf16),
        lm_head_rms_weight_arg_idx, MM2S,
        shim_tiles[1],
        (npu_bd_id)1, it_channel_1,//send the input weight
        {0,0,0, 0},
        {1,1,1, GPTOSS_HIDDEN_SIZE},
        {0,0,0,1},
        0, 0, false
    );
    uint32_t issue_token_queue = 0;
    static_assert(VOCAB_SIZE_PADDED %(MVM_cores*Q4NX_ROW) == 0 );
    for(int round = 0; round < VOCAB_SIZE_PADDED /(MVM_cores*Q4NX_ROW)  ; round++){

        uint32_t bd_offset = (round%2) * 8;
        if(issue_token_queue == 2){
            for(int mvm_col_idx = 0; mvm_col_idx< (MVM_DECODE_CORES/4); mvm_col_idx++){
                seq.npu_dma_wait( shim_tiles[mvm_col_idx+ 2], MM2S, it_channel_0);
                seq.npu_dma_wait( shim_tiles[mvm_col_idx+ 2], MM2S, it_channel_1);
            }
            issue_token_queue-- ;
        }

        for(int mvm_col_idx = 0; mvm_col_idx< (MVM_DECODE_CORES/4); mvm_col_idx++){

            seq.npu_dma_memcpy_nd(
                sizeof(uint8_t),
                lm_head_w_arg_idx,
                MM2S,
                shim_tiles[mvm_col_idx+2],
                npu_bd_id(0+ bd_offset),
                it_channel_0,
                {0,0,0,  (uint32_t) (round * MVM_cores + mvm_col_idx * 4) * ( Q4NX_BLOCK_BYTE_SIZE*blocks_per_row) },
                {1, 1, 1,       q4nx_block_per_row*(2*Q4NX_BLOCK_BYTE_SIZE) }, //NOTE: slight different with rest with lm_head being reorder different, 2 blocks in each col-stride instead of 4
                {0,0,0,1},
                -1, 0, true,
                aggressive_cache
            );

            seq.npu_dma_memcpy_nd(
                sizeof(uint8_t),
                lm_head_w_arg_idx,
                MM2S,
                shim_tiles[mvm_col_idx+2],
                npu_bd_id(1+ bd_offset),
                it_channel_1,
                {0,0,0,  (uint32_t) (round * MVM_cores + mvm_col_idx * 4) * ( Q4NX_BLOCK_BYTE_SIZE*blocks_per_row) + 2*Q4NX_BLOCK_BYTE_SIZE*q4nx_block_per_row },
                {1, 1, 1,       q4nx_block_per_row*(2*Q4NX_BLOCK_BYTE_SIZE) },
                {0,0,0,1},
                -1, 0, true,
                aggressive_cache
            );
        }

        issue_token_queue++;
    }

    while(issue_token_queue > 0){

        for(int mvm_col_idx = 0; mvm_col_idx< (MVM_DECODE_CORES/4); mvm_col_idx++){
            seq.npu_dma_wait( shim_tiles[mvm_col_idx+ 2], MM2S, it_channel_0);
            seq.npu_dma_wait( shim_tiles[mvm_col_idx+ 2], MM2S, it_channel_1);
        }
        issue_token_queue-- ;
    }

    seq.npu_dma_wait( shim_tiles[1], S2MM, it_channel_1);

    seq.cmds2seq();
}

void gpt_oss_npu_sequence::Impl::set_max_length(const uint32_t MAX_L)
{
    this->MAX_L = MAX_L;
    this->kv_cache_size = MAX_L * (GPTOSS_DK + GPTOSS_DV);
    this->K03_offset = 0;
    this->K47_offset = this->kv_cache_size / 4;
    this->V03_offset = this->kv_cache_size / 2;
    this->V47_offset = 3 * this->kv_cache_size / 4;
    size_t kv_cache_size_sliding = 128 * GPTOSS_DV * 2;
    this->K03_sliding_offset = 0;
    this->K47_sliding_offset = kv_cache_size_sliding / 4;
    this->V03_sliding_offset = kv_cache_size_sliding / 2;
    this->V47_sliding_offset = 3 * kv_cache_size_sliding / 4;
}

// ===============================================
// Externals for gpt_oss_npu_sequence
// ===============================================
/// @brief
/// @param rtp_seq : npu_sequence ptr for layer rtp write (set L)
/// @param layer_seq : npu_sequence ptr for layer sequence
/// @param dequant_seq : npu_sequence ptr for dequant sequence
/// @param mm_engine_seq : npu_sequence ptr for mm engine sequence
/// @param mha_engine_seq : npu_sequence ptr for mha engine sequence
/// @param config : gpt_oss config
/// @param MAX_L : max length
gpt_oss_npu_sequence::gpt_oss_npu_sequence(LM_Config config, uint32_t MAX_L){
    this->_impl = new Impl(config, MAX_L);
}

void gpt_oss_npu_sequence::gen_layer_seq(npu_sequence* seq, const int L, bool is_sliding_window){
    this->_impl->gen_layer_seq(seq, L, is_sliding_window);
}

void gpt_oss_npu_sequence::gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, bf16* sinks, bool is_sliding_window){
    this->_impl->gen_mha_engine_seq(seq, L_begin, L_end, sinks, is_sliding_window);
}

void gpt_oss_npu_sequence::set_max_length(const uint32_t MAX_L){
    this->_impl->set_max_length(MAX_L);
}

void gpt_oss_npu_sequence::generate_lm_head_seq(npu_sequence* seq){
    this->_impl->generate_lm_head_seq(seq);
}

size_t gpt_oss_npu_sequence::get_k03_offset() const{
    return this->_impl->K03_offset;
}

size_t gpt_oss_npu_sequence::get_k47_offset() const{
    return this->_impl->K47_offset;
}

size_t gpt_oss_npu_sequence::get_v03_offset() const{
    return this->_impl->V03_offset;
}

size_t gpt_oss_npu_sequence::get_v47_offset() const{
    return this->_impl->V47_offset;
}

gpt_oss_npu_sequence::~gpt_oss_npu_sequence() = default;
