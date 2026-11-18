// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
/// \file d3_vision_seq.hpp
/// \brief The vision attention sequence of the d3 vision tower: gen_mha_vision_attention of
///        qwen3_5vl_npu/seq_gen.hpp, for D3VisionEncoder (internal linkage).
#include "npu_utils/npu_instr_utils.hpp"
#include "npu_sequences/image_attention_sequence.hpp"

class D3VisionEncoder;

template <typename Encoder>
static void gen_mha_vision_attention(

    npu_sequence* seq,
    std::vector<std::vector<int32_t>> image_grid_thw,
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
    Encoder* image_encoder_ptr

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

    uint32_t total_seq_len = 0;
    for(int b= 0; b < image_grid_thw.size(); b++){
        int end_seq_len = 1;
        for(int v: image_grid_thw[b]){
            end_seq_len *= v;
        }
        total_seq_len += end_seq_len;
    }
    //total_seq_len is now batch*L_Seq_per_batch

    uint32_t Q_total_seq_len_per_head = round_up_to_multiple_lambda( total_seq_len,vision_L_padded_requirement_for_attention );
    uint32_t K_total_seq_len_per_head = round_up_to_multiple_lambda( total_seq_len,vision_L_padded_requirement_for_attention );
    uint32_t V_total_seq_len_per_head = K_total_seq_len_per_head;

    int cur_seq_len = 0;
    for(int b = 0; b < image_grid_thw.size(); b++){

        int start_seq_len = cur_seq_len;
        int end_seq_len = 1;
        for(int v: image_grid_thw[b]){
            end_seq_len *= v;
        }

        uint32_t l_seq_padded = round_up_to_multiple_lambda(end_seq_len, vision_L_padded_requirement_for_attention);// same for L, and S
        uint32_t s_seq_padded = round_up_to_multiple_lambda(end_seq_len, vision_S_padded_requirement_for_attention);
        L_seq_list.push_back(l_seq_padded);
        S_seq_list.push_back(end_seq_len);
        S_seq_padded_list.push_back(s_seq_padded);

        // but since
        if(!REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH){
            Q_batch_offset_list.push_back(start_seq_len * (3*image_encoder_ptr->Padded_QWEN3_5_VISION_HIDDEN_SIZE) +0 );
            K_batch_offset_list.push_back(start_seq_len * (3*image_encoder_ptr->Padded_QWEN3_5_VISION_HIDDEN_SIZE) + image_encoder_ptr->_QWEN3_5_VISION_HIDDEN_SIZE );
            V_batch_offset_list.push_back(start_seq_len * (3*image_encoder_ptr->Padded_QWEN3_5_VISION_HIDDEN_SIZE) +2*image_encoder_ptr->_QWEN3_5_VISION_HIDDEN_SIZE);
            O_batch_offset_list.push_back(start_seq_len * image_encoder_ptr->Padded_QWEN3_5_VISION_HIDDEN_SIZE );
        }else{

            // The buffer is now in [3*QWEN3_VISION_NUM_HEAD, batch, L_Seq_per_batch, QWEN3_5_VISION_HEAD_DIM]
            Q_batch_offset_list.push_back(start_seq_len * (image_encoder_ptr->QWEN3_5_VISION_HEAD_DIM) );
            K_batch_offset_list.push_back( (image_encoder_ptr->_QWEN3_5_VISION_HIDDEN_SIZE * Q_total_seq_len_per_head*image_encoder_ptr->_QWEN3_5_VISION_HIDDEN_SIZE)
                + start_seq_len * (image_encoder_ptr->QWEN3_5_VISION_HEAD_DIM));
            V_batch_offset_list.push_back( 2*(image_encoder_ptr->_QWEN3_5_VISION_HIDDEN_SIZE * Q_total_seq_len_per_head*image_encoder_ptr->_QWEN3_5_VISION_HIDDEN_SIZE)
                + start_seq_len * (image_encoder_ptr->QWEN3_5_VISION_HEAD_DIM));
            O_batch_offset_list.push_back(start_seq_len * image_encoder_ptr->Padded_QWEN3_5_VISION_HIDDEN_SIZE );
        }

        cur_seq_len += end_seq_len;
    }
    if(total_seq_len != cur_seq_len){
        std::cerr << "total_seq_len != cur_seq_len" << std::endl;
        exit(-1);
    }

    // because q, k, v buffer all in on buffer after MM
    constexpr int ARG_Q = 1;
    constexpr int ARG_K = 1;
    constexpr int ARG_V = 1;
    constexpr int ARG_O = 0;

    AttentionConfig config = {
        (uint32_t)image_encoder_ptr->QWEN3_5_VISION_HEAD_DIM, (uint32_t)image_encoder_ptr->QWEN3_5_VISION_NUM_HEADS,
        vision_LQ_per_CT, vision_LK_per_CT,
        vision_LQ_internal, vision_LK_internal,
        vision_num_of_columns, vision_num_of_rows,
        vision_CU_mode,
        (uint32_t)image_grid_thw.size(),

        ARG_Q, ARG_K, ARG_V, ARG_O,

        (uint32_t)(image_encoder_ptr->Padded_QWEN3_5_VISION_HIDDEN_SIZE*3), (uint32_t)(image_encoder_ptr->Padded_QWEN3_5_VISION_HIDDEN_SIZE*3), (uint32_t)(image_encoder_ptr->Padded_QWEN3_5_VISION_HIDDEN_SIZE*3),
        (uint32_t)image_encoder_ptr->Padded_QWEN3_5_VISION_HIDDEN_SIZE,

        REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH,
        Q_total_seq_len_per_head, K_total_seq_len_per_head, V_total_seq_len_per_head
    };

    BatchMetadata batch_data = {
        L_seq_list, S_seq_list, S_seq_padded_list,
        Q_batch_offset_list, K_batch_offset_list, V_batch_offset_list, O_batch_offset_list
    };

    setup_SHM_configuration<bf16, bf16>(
        *seq,
        config,
        batch_data,
        1.0/std::sqrt(image_encoder_ptr->QWEN3_5_VISION_HEAD_DIM)
    );
}
