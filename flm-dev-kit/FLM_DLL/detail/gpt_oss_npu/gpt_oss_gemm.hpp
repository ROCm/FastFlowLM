// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __gemm_detail__
#define __gemm_detail__
#include "npu_utils/npu_utils.hpp"
#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"

class GPT_OSS_Gemm
{
private:
    static constexpr int mm_y_group_id = 3;
    static constexpr int mm_x_group_id = 4;
    static constexpr int mm_w_group_id = 5;

    static constexpr npu_tiles IT[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr uint32_t CT_rtp_address = 4096; // for 128

    static constexpr int CT_rtp_sync_lock_id = 10;// for now hard coded

public:

    static constexpr uint32_t total_cols = 8;
    static constexpr uint32_t total_rows = 4;
    uint32_t m = 64;
    uint32_t k = 128;
    static constexpr uint32_t n = 128;
    static constexpr uint32_t r = 8;
    static constexpr uint32_t s = 8;
    static constexpr uint32_t t = 8;

    GPT_OSS_Gemm();

    ~GPT_OSS_Gemm();

    inline int y_id() { return mm_y_group_id; };
    inline int x_id() { return mm_x_group_id; };
    inline int w_id() { return mm_w_group_id; };

    bool IS_B_ROW_MAJOR = false; // for language model, B is always in col-major order
    bool B_in_K_N_block_col_major_order = true; // for language model, B is default in kxn block col-major order
    bool ENABLE_AXI4 = true; // for language model, B is always in kxn block col-major order

    template <typename T_in, typename T_out>
    void generate_shimtile_sequence_per_k_block(
        uint32_t shim_index, uint32_t total_npu_row, uint32_t total_npu_col,
        uint32_t mega_block_row_idx, uint32_t mega_block_col_idx,
        uint32_t M_size, uint32_t K_size, uint32_t N_size,
        uint32_t m, uint32_t k, uint32_t n,
        uint32_t Arg_A, uint32_t  Arg_B, uint32_t Arg_C, uint32_t Arg_Bias,
        uint32_t A_const_offset, uint32_t B_const_offset,  uint32_t C_const_offset, uint32_t Bias_const_offset,

        std::vector<int> &list_A_shim_queue, std::vector<int> &list_B_shim_queue, std::vector<int> &list_C_shim_queue,
        std::vector<int> &list_A_bd_pingpong_flag, std::vector<int> &list_B_bd_pingpong_flag, std::vector<int> &list_C_bd_pingpong_flag,

        bool IS_B_ROW_MAJOR, bool ENABLE_AXI4,
        bool B_in_K_N_block_col_major_order,
        bool VALID_COLUMN,
        bool ADD_BIAS, bool SEND_BIAS,
        std::map<int, int>& valid_A_MT_shimtile_index,
        npu_sequence*seq,
        std::vector<npu_tiles> & shimtile_list
    ){

        if(B_in_K_N_block_col_major_order){
            assert(IS_B_ROW_MAJOR== false); // on valid for B in col major order
        }
        // When B_in_K_N_block_col_major_order is set to true, it mean
        // B is col-major order &&
        // B is rearrange into kxn blocks, where blocks are in col-major. Moreover, the data in each blocks is
        // also in col-major order.

        // Basically,B as a col-major matrix goes through
        // stride: [N_size/n,K_size/k  ,n,     k]
        // offset: [K_size*n,k         ,K_SIZE, 1]

        uint32_t AXI_FLAG = 0x2;// 0b1110<<24;
        ;
        uint32_t K_div_k = K_size/k;

        npu_tiles cur_shimtile = shimtile_list.at(shim_index);

        if (valid_A_MT_shimtile_index.contains(shim_index) && valid_A_MT_shimtile_index[shim_index] < total_npu_row){

            if (list_A_shim_queue.at(shim_index) == 2) {
                seq->npu_dma_wait(
                    cur_shimtile, MM2S, it_channel_0
                );
                list_A_shim_queue.at(shim_index)--;
            }

            uint32_t A_offset = mega_block_row_idx *(total_npu_row*m) *K_size;
            A_offset += valid_A_MT_shimtile_index[shim_index]*(m*K_size);
            npu_bd_id A_bd_id;
            if (list_A_bd_pingpong_flag.at(shim_index) ==0){
                A_bd_id = bd_0;
                list_A_bd_pingpong_flag.at(shim_index) =1;
            }else{
                A_bd_id = bd_1;
                list_A_bd_pingpong_flag.at(shim_index) =0;
            }

            seq->npu_dma_memcpy_nd(
                sizeof(T_in),   // bfloat16
                Arg_A,
                MM2S,
                cur_shimtile,
                A_bd_id,
                it_channel_0,
                {0,0,0,A_offset+ A_const_offset},
                {1, K_div_k, m,k},
                {0, k, K_size, 1},
                -1, 0, true,
                aggressive_cache
            );
            list_A_shim_queue.at(shim_index)++;
        }

        if(shim_index < total_npu_col && VALID_COLUMN){

            if(SEND_BIAS){
                if (list_B_shim_queue[shim_index] == 2){

                    seq->npu_dma_wait(
                        cur_shimtile, MM2S, it_channel_1
                    );
                    list_B_shim_queue[shim_index] -= 1;
                }
                uint32_t _BIAS_DATA_OFFSET = Bias_const_offset +  mega_block_col_idx * (total_npu_col*n) +  shim_index *n;
                seq->npu_dma_memcpy_nd(
                    sizeof(T_in),
                    Arg_Bias,
                    MM2S,
                    cur_shimtile,
                    npu_bd_id(bd_6), //reserved for sending bias
                    it_channel_1,
                    {0,0,0,_BIAS_DATA_OFFSET},
                    {1, 1,1, k*n},
                    {0, 0, 0, 1},
                    -1, 0, true,
                    aggressive_cache
                );
                list_B_shim_queue[shim_index]++;
            }

            npu_bd_id b_bd_id;
            if (list_B_shim_queue[shim_index] == 2){

                seq->npu_dma_wait(
                    cur_shimtile, MM2S, it_channel_1
                );
                list_B_shim_queue[shim_index] -= 1;
            }

            if (list_B_bd_pingpong_flag[shim_index] == 0) {
                b_bd_id = bd_2;
                list_B_bd_pingpong_flag[shim_index] = 1;
            } else {
                b_bd_id = bd_3;
                list_B_bd_pingpong_flag[shim_index] = 0;
            }

            if (IS_B_ROW_MAJOR){
                uint32_t B_offset = mega_block_col_idx* (total_npu_col) * n;
                B_offset += shim_index * n;
                seq->npu_dma_memcpy_nd(
                    sizeof(T_in),
                    Arg_B,
                    MM2S,
                    cur_shimtile,
                    b_bd_id,
                    it_channel_1,
                    {0,0,0,B_offset+ B_const_offset },
                    {1, K_div_k, k, n},
                    {0, k*N_size, N_size, 1},
                    -1, 0, true,
                    aggressive_cache
                );
            }else{
                uint32_t B_offset = mega_block_col_idx*(total_npu_col*n)*K_size;
                B_offset += shim_index * n*K_size;
                if(B_in_K_N_block_col_major_order){
                    seq->npu_dma_memcpy_nd(
                        sizeof(T_in),
                        Arg_B,
                        MM2S,
                        cur_shimtile,
                        b_bd_id,
                        it_channel_1,
                        {0,0,0,B_offset+ B_const_offset },
                        {1, 1,1, K_div_k* n*k},
                        {0, 0, 0, 1},
                        -1, 0, true,
                        aggressive_cache
                    );
                }else{
                    seq->npu_dma_memcpy_nd(
                        sizeof(T_in),
                        Arg_B,
                        MM2S,
                        cur_shimtile,
                        b_bd_id,
                        it_channel_1,
                        {0,0,0,B_offset+ B_const_offset },
                        {1, K_div_k, n, k},
                        {0, k, K_size, 1},
                        -1, 0, true,
                        aggressive_cache
                    );
                }
            }
            list_B_shim_queue[shim_index]++;
        }

        uint32_t C_offset = mega_block_col_idx*n*total_npu_col;
        C_offset += mega_block_row_idx* m*total_npu_row*N_size;
        C_offset += shim_index *n;

        if (shim_index < total_npu_col && VALID_COLUMN){

            if (list_C_shim_queue.at(shim_index) == 2) {
                seq->npu_dma_wait(
                    cur_shimtile, S2MM, it_channel_0
                );
                list_C_shim_queue.at(shim_index)--;
            }

            npu_bd_id c_bd_id;
            if(list_C_bd_pingpong_flag.at(shim_index) ==0){
                c_bd_id = bd_14;
                list_C_bd_pingpong_flag.at(shim_index) =1;
            }else{
                c_bd_id = bd_15;
                list_C_bd_pingpong_flag.at(shim_index) =0;
            }

            seq->npu_dma_memcpy_nd(
                sizeof(T_out),
                Arg_C,
                S2MM,
                cur_shimtile,
                c_bd_id,
                it_channel_0,
                {0,0,0, C_offset+ C_const_offset},
                {1,1,4*m, n},
                {0,0,N_size, 1},
                -1, 0, true,
                normal_cache
            );
            list_C_shim_queue.at(shim_index)++;
        }
    }

    template <typename T_in, typename T_out>
    void generate_runtime_sequence(
        uint32_t Arg_A, uint32_t Arg_B, uint32_t Arg_C, uint32_t Arg_Bias,
        uint32_t A_const_offset, uint32_t B_const_offset,  uint32_t C_const_offset, uint32_t Bias_const_offset,
        uint32_t M_size, uint32_t N_size, uint32_t K_size,
        uint32_t m, uint32_t n, uint32_t k,
        uint32_t total_npu_row, uint32_t total_npu_col,
        std::vector<int> &list_A_shim_queue,
        std::vector<int> &list_B_shim_queue,
        std::vector<int> &list_C_shim_queue,

        bool IS_B_ROW_MAJOR,  bool ENABLE_AXI4, bool B_in_K_N_block_col_major_order,
        bool ADD_BIAS,
        std::map<int, int>& valid_A_MT_shimtile_index,
        npu_sequence* seq,
        std::vector<npu_tiles> &shim_tiles
    ){

        uint32_t M_div_num_row_m = M_size/(m*total_npu_row);
        uint32_t N_div_num_col_n = N_size/(n*total_npu_col);

        uint32_t N_div_num_col_n_remainder_blocks =  (N_size % (n*total_npu_col))/ n;

        std::vector<int> list_A_BD_pingpong_flag(std::max(total_npu_row, total_npu_col), 0);
        std::vector<int> list_B_BD_pingpong_flag(std::max(total_npu_row, total_npu_col), 0);
        std::vector<int> list_C_BD_pingpong_flag(std::max(total_npu_row, total_npu_col), 0);

        uint32_t col_block_range = N_div_num_col_n;
        if (N_div_num_col_n_remainder_blocks!= 0){
            col_block_range += 1;
        }

        for(uint32_t mega_block_col_idx=0; mega_block_col_idx<col_block_range; mega_block_col_idx++){

            for (uint32_t mega_block_row_idx = 0; mega_block_row_idx < M_div_num_row_m; mega_block_row_idx++) {

                for (uint32_t shim_index = 0; shim_index < std::max(total_npu_col, total_npu_row); shim_index++) {
                    bool SEND_ADD_BIAS = false;
                    if (mega_block_row_idx == 0 &&ADD_BIAS){
                        SEND_ADD_BIAS = true;
                    }
                    if (N_div_num_col_n_remainder_blocks!= 0 && mega_block_col_idx == N_div_num_col_n){

                        if (shim_index < N_div_num_col_n_remainder_blocks){

                            generate_shimtile_sequence_per_k_block<T_in, T_out>(
                                shim_index,
                                total_npu_row, total_npu_col,
                                mega_block_row_idx, mega_block_col_idx,
                                M_size, K_size, N_size,
                                m, k, n,
                                Arg_A, Arg_B, Arg_C, Arg_Bias,
                                A_const_offset, B_const_offset, C_const_offset, Bias_const_offset,
                                list_A_shim_queue, list_B_shim_queue, list_C_shim_queue,
                                list_A_BD_pingpong_flag, list_B_BD_pingpong_flag, list_C_BD_pingpong_flag,
                                IS_B_ROW_MAJOR, ENABLE_AXI4,
                                B_in_K_N_block_col_major_order,
                                true,
                                ADD_BIAS, SEND_ADD_BIAS,
                                valid_A_MT_shimtile_index,
                                seq, shim_tiles

                            );
                        }

                        else{
                            generate_shimtile_sequence_per_k_block<T_in, T_out>(
                                shim_index,
                                total_npu_row, total_npu_col,
                                mega_block_row_idx, mega_block_col_idx,
                                M_size, K_size, N_size,
                                m, k, n,
                                Arg_A, Arg_B, Arg_C, Arg_Bias,
                                A_const_offset, B_const_offset, C_const_offset, Bias_const_offset,
                                list_A_shim_queue, list_B_shim_queue, list_C_shim_queue,
                                list_A_BD_pingpong_flag, list_B_BD_pingpong_flag, list_C_BD_pingpong_flag,
                                IS_B_ROW_MAJOR, ENABLE_AXI4,
                                B_in_K_N_block_col_major_order,
                                false,
                                ADD_BIAS, SEND_ADD_BIAS,
                                valid_A_MT_shimtile_index,
                                seq, shim_tiles
                            );
                        }
                    }
                    else{
                        generate_shimtile_sequence_per_k_block<T_in, T_out>(
                            shim_index,
                            total_npu_row, total_npu_col,
                            mega_block_row_idx, mega_block_col_idx,
                            M_size, K_size, N_size,
                            m, k, n,
                            Arg_A, Arg_B, Arg_C, Arg_Bias,
                            A_const_offset, B_const_offset, C_const_offset, Bias_const_offset,
                            list_A_shim_queue, list_B_shim_queue, list_C_shim_queue,
                            list_A_BD_pingpong_flag, list_B_BD_pingpong_flag, list_C_BD_pingpong_flag,
                            IS_B_ROW_MAJOR, ENABLE_AXI4,
                            B_in_K_N_block_col_major_order,
                            true,
                            ADD_BIAS, SEND_ADD_BIAS,
                            valid_A_MT_shimtile_index,
                            seq, shim_tiles
                        );
                    }
                }
            }
        }
    }

    void generate_seq(npu_sequence *seq, uint32_t M, uint32_t K, uint32_t N,
    const uint32_t weight_offset,
    bool ADD_BIAS, int OUTPUT_MODE, const uint32_t bias_offset
    );
};
#endif
