#pragma once
#include "npu_utils/npu_instr_utils.hpp"
#include "npu_sequences/image_attention_sequence.hpp"

void _gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_padded, const uint32_t L,
    const int QKV_buffer_offset_bf,
    const int O_buffer_offset_bf
){
    constexpr int DQ = 64 * 16;
    constexpr int DK = 64 * 16;
    constexpr int DV = DK;
    constexpr int DH = 64;
    int K_OFFSET = DQ;
    int V_OFFSET = DQ + DK;
    int D_TOTAL = DQ + DK + DV;
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    // each function call init a new list
    int k_ping_pong_flag[8] = {0,0,0,0,0,0,0,0};
    int v_ping_pong_flag[8] = {0,0,0,0,0,0,0,0};
    int k_bd_queue[8] = {0,0,0,0,0,0,0,0};
    int v_bd_queue[8] = {0,0,0,0,0,0,0,0};

    constexpr int lc = 32;
    assert(L_padded % (32) == 0);
    constexpr int l_qk_mha_address = 21760;
    constexpr int l_kv_mha_address = 28672;
    // seq->clear_cmds();

    for (int row = 2; row < total_rows + 2; row++){
        for (int col = 0; col < 4; col++){
            npu_tiles tile_qk = get_tile(row, col * 2);
            npu_tiles tile_kv = get_tile(row, col * 2 + 1);
            seq->rtp_write(tile_qk, l_qk_mha_address, L); // 32 is lk
            seq->rtp_write(tile_kv, l_kv_mha_address, L); // 32 is lk
        }
    }

    for (int round = 0; round < L_padded / 32; round++){
        int bd_offset = (round % 2) * 8;
        for (int col = 0; col < 4; col++){
            // receive y
            size_t y_offset = round * lc * DQ + col * 4 * DH + O_buffer_offset_bf; // y does not have history
            seq->npu_dma_memcpy_nd(
                2, 0,
                S2MM, IT[col * 2 + 1],
                (npu_bd_id)(bd_offset + 0), it_channel_0,
                {0, 0, 0, (uint32_t)y_offset},
                {1, 1, (uint32_t)lc, (uint32_t)DH * 4},
                {0, 0, (uint32_t)DQ, 1},
                -1, 0, true
            );
            // send q
            size_t q_offset = round * lc * D_TOTAL + col * 4 * DH  +QKV_buffer_offset_bf; // q does not have history
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[col * 2],
                (npu_bd_id)(bd_offset + 1), it_channel_0,
                {0, 0, 0, (uint32_t)q_offset},
                {1, 1, (uint32_t)lc, (uint32_t)DH * 4},
                {0, 0, (uint32_t)D_TOTAL, 1},
                -1, 0, false
            );

            uint32_t L_padded_div_32 = L_padded / 32;
            uint32_t max_L_per_chunk = 512;
            // send k
            size_t k_offset = col * 4 * DH + K_OFFSET + QKV_buffer_offset_bf;
            size_t v_offset = col * 4 * DH + V_OFFSET + QKV_buffer_offset_bf;
            for(int chunk_idx = 0; chunk_idx < L_padded_div_32; chunk_idx += max_L_per_chunk){

                uint32_t cur_chunk_seqlen = std::min(max_L_per_chunk,  L_padded_div_32 - chunk_idx);
                int k_shim_id  = col*2;
                int k_bd_offset ;
                if(k_ping_pong_flag[k_shim_id] == 0){
                    k_bd_offset = 0;
                    k_ping_pong_flag[k_shim_id] = 1;
                }else{
                    k_bd_offset = 8;
                    k_ping_pong_flag[k_shim_id] = 0;
                }
                if(k_bd_queue[k_shim_id] == 2){
                    // wait k
                    seq->npu_dma_wait(
                        IT[k_shim_id],
                        MM2S,
                        it_channel_1
                    );
                    k_bd_queue[k_shim_id]--;
                }
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[k_shim_id],
                    (npu_bd_id)(k_bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)k_offset + chunk_idx * D_TOTAL * 32},  //row major of q, k, v per row
                    {1, cur_chunk_seqlen, (uint32_t)32, (uint32_t)DH * 4},
                    {0, (uint32_t)32 * D_TOTAL, (uint32_t)D_TOTAL, 1},
                    -1, 0, true,
                    aggressive_cache
                );
                k_bd_queue[k_shim_id]++;

                int v_shim_id  = col * 2 + 1;
                int v_bd_offset ;
                if(v_ping_pong_flag[v_shim_id] == 0){
                    v_bd_offset = 0;
                    v_ping_pong_flag[v_shim_id] = 1;
                }else{
                    v_bd_offset = 8;
                    v_ping_pong_flag[v_shim_id] = 0;
                }
                if(v_bd_queue[v_shim_id] == 2){
                    // wait v
                    seq->npu_dma_wait(
                        IT[v_shim_id],
                        MM2S,
                        it_channel_1
                    );
                    v_bd_queue[v_shim_id]--;
                }
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[v_shim_id],
                    (npu_bd_id)(v_bd_offset + 3), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset + chunk_idx * D_TOTAL * 32},
                    {1, cur_chunk_seqlen, (uint32_t)32, (uint32_t)DH * 4},
                    {0, (uint32_t)32 * D_TOTAL, (uint32_t)D_TOTAL, 1},
                    -1, 0, true,
                    aggressive_cache
                );
                v_bd_queue[v_shim_id]++;
            }
        } // col loop

        if (round > 0){
            for (int col = 0; col < 4; col++){
                seq->npu_dma_wait(
                    IT[col * 2 + 1],
                    S2MM,
                    it_channel_0
                );
            }
        }
    }

    int max_k_queue_remaing = -1; //should be same with v
    for(int col = 0; col < 8; col++){
        if( k_bd_queue[col] > max_k_queue_remaing){
            max_k_queue_remaing = k_bd_queue[col];
        }
    }

    for(int queue_size = 0; queue_size <max_k_queue_remaing; queue_size++){
        for(int col  = 0; col < 8; col++){
            if(k_bd_queue[col] > 0){
                seq->npu_dma_wait(
                    IT[col],
                    MM2S,
                    it_channel_1
                );
                k_bd_queue[col]--;
            }
            if(v_bd_queue[col] > 0){
                seq->npu_dma_wait(
                    IT[col],
                    MM2S,
                    it_channel_1
                );
                v_bd_queue[col]--;
            }
        }
    }

    for (int col = 0; col < 4; col++){
        seq->npu_dma_wait(
            IT[col * 2 + 1],
            S2MM,
            it_channel_0
        );
    }
    // seq->cmds2seq();
}

//deprecated
void gen_mha_main(

    npu_sequence* seq,
    std::vector<std::vector<int32_t>> image_grid_thw,
    int pad_requirement_for_attention,
    int QWEN3_5_VISION_HIDDEN_SIZE

){

    // support of multiple batch mha

    seq->clear_cmds();
    seq->npu_preemption(0);

    auto round_up_to_multiple_lambda = [](int x, int multiple) -> int {
            return ((x + multiple - 1) / multiple) * multiple;
        };

    int cur_seq_len = 0;
    for(int b = 0; b < image_grid_thw.size(); b++){

        int start_seq_len = cur_seq_len;
        int end_seq_len = 1;
        for(int v: image_grid_thw[b]){
            end_seq_len *= v;
        }

        _gen_mha_engine_seq(
            seq, round_up_to_multiple_lambda(end_seq_len, pad_requirement_for_attention),
            end_seq_len,
            start_seq_len * (3*QWEN3_5_VISION_HIDDEN_SIZE), //q,k,v offser
            start_seq_len* QWEN3_5_VISION_HIDDEN_SIZE// offset

        );
        cur_seq_len += end_seq_len;
    }

    seq->cmds2seq();
}

void gen_mha_vision_attention(

    npu_sequence* seq,
    std::vector<int> &seq_len_per_image,
    int vision_L_padded_requirement_for_attention,
    int vision_S_padded_requirement_for_attention,
    uint32_t vision_num_of_columns,
    uint32_t vision_num_of_rows,
    uint32_t vision_CU_mode,
    uint32_t vision_LQ_per_CT,
    uint32_t vision_LK_per_CT,
    uint32_t vision_LQ_internal,
    uint32_t vision_LK_internal,
    bool REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH,
    int VISION_HIDDEN_SIZE,
    int Padded_VISION_HIDDEN_SIZE,
    int VISION_HEAD_DIM,
    int VISION_NUM_ATTENTION_HEADS
){

    if(vision_S_padded_requirement_for_attention >vision_L_padded_requirement_for_attention ){
        std::cerr << "vision S padded requirement greater than L padded requirement"<< std::endl;
        exit(1);
    }

    // support of multiple batch mha

    auto round_up_to_multiple_lambda = [](int x, int multiple) -> int {
            return ((x + multiple - 1) / multiple) * multiple;
        };

    std::vector<int> L_seq_list;
    std::vector<int> S_seq_list;
    std::vector<int> S_seq_padded_list;

    std::vector<int> Q_batch_offset_list;
    std::vector<int> K_batch_offset_list;
    std::vector<int> V_batch_offset_list;
    std::vector<int> O_batch_offset_list;

    int cur_seq_len = 0;
    for(int b = 0; b < seq_len_per_image.size(); b++){

        int start_seq_len = cur_seq_len;
        int end_seq_len = seq_len_per_image[b];

        uint32_t l_seq_padded = round_up_to_multiple_lambda(end_seq_len, vision_L_padded_requirement_for_attention);// same for L, and S
        uint32_t s_seq_padded = round_up_to_multiple_lambda(end_seq_len, vision_S_padded_requirement_for_attention);
        L_seq_list.push_back(l_seq_padded);
        S_seq_list.push_back(end_seq_len);
        S_seq_padded_list.push_back(s_seq_padded);

        // but since
        if(!REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH){
            Q_batch_offset_list.push_back(start_seq_len * (Padded_VISION_HIDDEN_SIZE) );
            K_batch_offset_list.push_back(start_seq_len * Padded_VISION_HIDDEN_SIZE);
            V_batch_offset_list.push_back(start_seq_len * Padded_VISION_HIDDEN_SIZE);
            O_batch_offset_list.push_back(start_seq_len * Padded_VISION_HIDDEN_SIZE);
        }else{
            std::cout << "Currently the code only support REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH = false, please set it to false" << std::endl;
            exit(-1);
            // // The buffer is now in [3*QWEN3_VISION_NUM_HEAD, batch, L_Seq_per_batch, parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM]
        }

        cur_seq_len += end_seq_len;
    }

    // because q, k, v buffer all in on buffer after MM
    constexpr int ARG_Q = 1;
    constexpr int ARG_K = 2;
    constexpr int ARG_V = 3;
    constexpr int ARG_O = 0;

    AttentionConfig config = {
        (uint32_t)VISION_HEAD_DIM, (uint32_t)VISION_NUM_ATTENTION_HEADS,
        vision_LQ_per_CT, vision_LK_per_CT,
        vision_LQ_internal, vision_LK_internal,
        vision_num_of_columns, vision_num_of_rows,
        vision_CU_mode,
        (uint32_t)seq_len_per_image.size(),

        ARG_Q, ARG_K, ARG_V, ARG_O,

        Padded_VISION_HIDDEN_SIZE, Padded_VISION_HIDDEN_SIZE, Padded_VISION_HIDDEN_SIZE,
        Padded_VISION_HIDDEN_SIZE,
        REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH,
        0,0,0 // since this only vailid if REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH == true, and currently we only support false, so set it to -1 to avoid misuse
    };

    BatchMetadata batch_data = {
        L_seq_list, S_seq_list, S_seq_padded_list,
        Q_batch_offset_list, K_batch_offset_list, V_batch_offset_list, O_batch_offset_list
    };

    setup_SHM_configuration<bf16, bf16>(
        *seq,
        config,
        batch_data,
        1.0f// NOTE: very special for gemma4e
    );
}
