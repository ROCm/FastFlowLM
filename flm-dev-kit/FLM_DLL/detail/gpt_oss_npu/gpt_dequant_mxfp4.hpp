// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"
#include "gpt_oss_params.hpp"

template<int ARG_QUANT_IN, int ARG_DEQUANT_OUT, bool IS_UP_PROJ=false,
    int Q4NX_ROW,  int Q4NX_COL,  int Q4NX_BLOCK_BYTE_SIZE,
    int INTERMEDIATE_SIZE_PADDED, int D_SIZE_PADDED>  // Should be INTERMEDIATE_PADDED_FOUR_Q4NX_ROW, D_PADDED_Q4NX_COL>
void dequant_gate_up_matrix(
    npu_sequence& seq,
    std::vector<npu_tiles>& shim_tiles,
    uint32_t input_offset,
    uint32_t output_offset
){
    constexpr uint32_t Q4NX_BLOCK_SIZE = Q4NX_ROW*Q4NX_COL;
    int bd_wait_counter[8] = {0,0,0,0,0,0,0,0};

    // Gate up matrix is INTERMEDIATE_SIZE_PADDED x  D_SIZE_PADDED

    constexpr uint32_t gate_up_column = INTERMEDIATE_SIZE_PADDED/(4*Q4NX_ROW);

    constexpr uint32_t D_Q4NX_BLOCK_PER_ROW = D_SIZE_PADDED/Q4NX_COL;
    constexpr uint32_t quant_in_per_CT_column  = 4* Q4NX_BLOCK_BYTE_SIZE * D_Q4NX_BLOCK_PER_ROW;
    constexpr uint32_t dequant_out_per_CT_coulmn = 4*Q4NX_ROW* D_SIZE_PADDED;

    if (IS_UP_PROJ){
        /// additional offset to up_proj due to interleafe stuff
        input_offset += quant_in_per_CT_column;
    }
    // first, the dequant of down
    for(int i = 0; i < gate_up_column; i++){

        for(int col = 0; col < 8; col++){
            uint32_t bd_offset = (i % 2) * 8;

            if( i*8 + col < gate_up_column  ){
                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    ARG_QUANT_IN,
                    MM2S,
                    shim_tiles[col],
                    (npu_bd_id)(0+bd_offset),
                    it_channel_0,
                    {0,0,0, input_offset+ (i*8+col) *quant_in_per_CT_column *2   }, // times 2 due to gate and up is interleaved
                    { 1,    D_Q4NX_BLOCK_PER_ROW ,       (2*Q4NX_BLOCK_BYTE_SIZE)/128 , 128},
                    { 0,    4*Q4NX_BLOCK_BYTE_SIZE ,                      128,    1},
                    -1,0, false,
                    aggressive_cache

                );
                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    ARG_QUANT_IN,
                    MM2S,
                    shim_tiles[col],
                    (npu_bd_id)(2+bd_offset),
                    it_channel_1,
                    {0,0,0, input_offset+ (i*8+col) *quant_in_per_CT_column *2 + 2*Q4NX_BLOCK_BYTE_SIZE   }, // times 2 due to gate and up is interleaved
                    { 1,    D_Q4NX_BLOCK_PER_ROW ,       (2*Q4NX_BLOCK_BYTE_SIZE)/128 , 128},
                    { 0,    4*Q4NX_BLOCK_BYTE_SIZE ,                      128,    1},
                    -1,0, false,
                    aggressive_cache

                );

                // Each port receive 2*Q4NX_ROWx D_Q4NX_BLOCK_PER_ROW*Q4NX_COL
                seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),
                    ARG_DEQUANT_OUT,
                    S2MM,
                    shim_tiles[col],
                    (npu_bd_id)(1+bd_offset),
                    it_channel_0,
                    {0,0,0,  output_offset +(i*8+col) *dequant_out_per_CT_coulmn },
                    {1,1,1, D_Q4NX_BLOCK_PER_ROW *(4*Q4NX_BLOCK_SIZE)},
                    {0,0,0,1},
                    -1,0,true
                );

                bd_wait_counter[col]++;
            }
        }

        // note: for now
        for(int col = 0; col < 8; col++){
            if(bd_wait_counter[col] == 2){
                seq.npu_dma_wait(shim_tiles[col], S2MM, it_channel_0);
                bd_wait_counter[col] --;
            }
        }
    }

    for(int col = 0; col < 8; col++){
        while(bd_wait_counter[col] != 0){
            seq.npu_dma_wait(shim_tiles[col], S2MM, it_channel_0);
            bd_wait_counter[col] --;
        }
    }
}

template<int ARG_QUANT_IN, int ARG_DEQUANT_OUT, bool IS_UP_PROJ=false,
    int Q4NX_ROW,  int Q4NX_COL,  int Q4NX_BLOCK_BYTE_SIZE,
    int INTERMEDIATE_SIZE_PADDED, int D_SIZE_PADDED>  // Should be INTERMEDIATE_PADDED_FOUR_Q4NX_ROW, D_PADDED_Q4NX_COL>
void dequant_gate_up_matrix_reorder_for_padding(
    npu_sequence& seq,
    std::vector<npu_tiles>& shim_tiles,
    uint32_t input_offset,
    uint32_t output_offset,
    uint32_t D_size_mem_padded, // D_size is padded to multiple of MM_K_size
    uint32_t MM_K_size
){
    constexpr uint32_t Q4NX_BLOCK_SIZE = Q4NX_ROW*Q4NX_COL;
    int bd_wait_counter[8] = {0,0,0,0,0,0,0,0};

    // Gate up matrix is INTERMEDIATE_SIZE_PADDED x  D_SIZE_PADDED

    constexpr uint32_t gate_up_column = INTERMEDIATE_SIZE_PADDED/(4*Q4NX_ROW);

    constexpr uint32_t D_Q4NX_BLOCK_PER_ROW = D_SIZE_PADDED/Q4NX_COL;
    constexpr uint32_t quant_in_per_CT_column  = 4* Q4NX_BLOCK_BYTE_SIZE * D_Q4NX_BLOCK_PER_ROW;
    // constexpr uint32_t dequant_out_per_CT_coulmn = 4*Q4NX_ROW* D_SIZE_PADDED;

    if (IS_UP_PROJ){
        /// additional offset to up_proj due to interleafe stuff
        input_offset += quant_in_per_CT_column;
    }
    // first, the dequant of down
    for(int i = 0; i < gate_up_column; i++){

        for(int col = 0; col < 8; col++){
            uint32_t bd_offset = (i % 2) * 8;

            if( i*8 + col < gate_up_column  ){
                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    ARG_QUANT_IN,
                    MM2S,
                    shim_tiles[col],
                    (npu_bd_id)(0+bd_offset),
                    it_channel_0,
                    {0,0,0, input_offset+ (i*8+col) *quant_in_per_CT_column *2   }, // times 2 due to gate and up is interleaved
                    { 1,    D_Q4NX_BLOCK_PER_ROW ,       (2*Q4NX_BLOCK_BYTE_SIZE)/128 , 128},
                    { 0,    4*Q4NX_BLOCK_BYTE_SIZE ,                      128,    1},
                    -1,0, false,
                    aggressive_cache

                );
                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    ARG_QUANT_IN,
                    MM2S,
                    shim_tiles[col],
                    (npu_bd_id)(2+bd_offset),
                    it_channel_1,
                    {0,0,0, input_offset+ (i*8+col) *quant_in_per_CT_column *2 + 2*Q4NX_BLOCK_BYTE_SIZE   }, // times 2 due to gate and up is interleaved
                    { 1,    D_Q4NX_BLOCK_PER_ROW ,       (2*Q4NX_BLOCK_BYTE_SIZE)/128 , 128},
                    { 0,    4*Q4NX_BLOCK_BYTE_SIZE ,                      128,    1},
                    -1,0, false,
                    aggressive_cache

                );

                // Each port receive 2*Q4NX_ROWx D_Q4NX_BLOCK_PER_ROW*Q4NX_COL
                seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),
                    ARG_DEQUANT_OUT,
                    S2MM,
                    shim_tiles[col],
                    (npu_bd_id)(1+bd_offset),
                    it_channel_0,
                    {0,0,0,  output_offset +(i*8+col) * ( 4*Q4NX_ROW* D_size_mem_padded) },
                    {D_SIZE_PADDED/MM_K_size,                MM_K_size/Q4NX_COL           ,Q4NX_ROW*4           ,Q4NX_COL},//n==128
                    {MM_K_size*(Q4NX_ROW * 4),              Q4NX_COL                    ,MM_K_size              ,1},
                    -1,0,true
                );
                seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),
                    ARG_DEQUANT_OUT,
                    S2MM,
                    shim_tiles[col],
                    (npu_bd_id)(3+bd_offset),
                    it_channel_0,
                    {0,0,0,  output_offset +(i*8+col) * ( 4*Q4NX_ROW* D_size_mem_padded)  +  ( D_SIZE_PADDED/MM_K_size ) * MM_K_size * Q4NX_ROW * 4},
                    {1,         (D_SIZE_PADDED%MM_K_size)/Q4NX_COL ,Q4NX_ROW*4           ,Q4NX_COL},//n==128
                    {0,         Q4NX_COL                            ,MM_K_size           ,1},
                    -1,0,true
                );
                bd_wait_counter[col]++;
            }
        }

        // note: for now
        for(int col = 0; col < 8; col++){
            if(bd_wait_counter[col] == 2){
                seq.npu_dma_wait(shim_tiles[col], S2MM, it_channel_0);
                seq.npu_dma_wait(shim_tiles[col], S2MM, it_channel_0);
                bd_wait_counter[col] --;
            }
        }
    }

    for(int col = 0; col < 8; col++){
        while(bd_wait_counter[col] != 0){
            seq.npu_dma_wait(shim_tiles[col], S2MM, it_channel_0);
            seq.npu_dma_wait(shim_tiles[col], S2MM, it_channel_0);
            bd_wait_counter[col] --;
        }
    }
}

template<int ARG_QUANT_IN, int ARG_DEQUANT_OUT,
    int Q4NX_ROW,  int Q4NX_COL,  int Q4NX_BLOCK_BYTE_SIZE,
    int D_SIZE_PADDED, int INTERMEDIATE_SIZE_PADDED>
void dequant_down_matrix(
    npu_sequence& seq,
    std::vector<npu_tiles>& shim_tiles,
    uint32_t expert_offest,
    uint32_t output_offset
){
    int bd_wait_counter[8] = {0,0,0,0,0,0,0,0};

    // matrix is D_SIZE_PADDED x INTERMEDIATE_SIZE_PADDED

    constexpr uint32_t INT_Q4NX_BLOCK_PER_ROW = INTERMEDIATE_SIZE_PADDED/Q4NX_COL;
    constexpr uint32_t quant_in_per_CT_column  = 4* Q4NX_BLOCK_BYTE_SIZE *  (INTERMEDIATE_SIZE_PADDED/Q4NX_COL);
    constexpr uint32_t dequant_out_per_CT_coulmn = 4*Q4NX_ROW* INTERMEDIATE_SIZE_PADDED;

    //D_PADDED_FOUR_Q4NX_ROW
    constexpr uint32_t down_column = D_SIZE_PADDED/(4*Q4NX_ROW);

    constexpr uint32_t Q4NX_BLOCK_SIZE = Q4NX_ROW*Q4NX_COL;
    if((2*Q4NX_BLOCK_BYTE_SIZE)%128 !=0){
        std::cout << "dequant mxfp4: block size not aligned" << std::endl;
        exit(1);
    }
    // first, the dequant of down
    for(int i = 0; i < down_column; i++){

        for(int col = 0; col < 8; col++){
            uint32_t bd_offset = (i % 2) * 8;

            if( i*8 + col < down_column  ){
                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    ARG_QUANT_IN,
                    MM2S,
                    shim_tiles[col],
                    (npu_bd_id)(0+bd_offset),
                    it_channel_0,
                    {0,0,0, expert_offest+ (i*8+col) *quant_in_per_CT_column },
                    { 1,    INT_Q4NX_BLOCK_PER_ROW ,       (2*Q4NX_BLOCK_BYTE_SIZE)/128 , 128},
                    { 0,    4*Q4NX_BLOCK_BYTE_SIZE ,                      128,             1},
                    -1,0, false,
                    aggressive_cache

                );
                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    ARG_QUANT_IN,
                    MM2S,
                    shim_tiles[col],
                    (npu_bd_id)(2+bd_offset),
                    it_channel_1,
                    {0,0,0, expert_offest+ (i*8+col) *quant_in_per_CT_column + 2*Q4NX_BLOCK_BYTE_SIZE },
                    { 1,    INT_Q4NX_BLOCK_PER_ROW ,       (2*Q4NX_BLOCK_BYTE_SIZE)/128 , 128},
                    { 0,    4*Q4NX_BLOCK_BYTE_SIZE ,                      128,             1},
                    -1,0, false,
                    aggressive_cache

                );

                // Each port receive 2*Q4NX_ROWx D_Q4NX_BLOCK_PER_ROW*Q4NX_COL
                seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),
                    ARG_DEQUANT_OUT,
                    S2MM,
                    shim_tiles[col],
                    (npu_bd_id)(1+bd_offset),
                    it_channel_0,
                    {0,0,0,  output_offset +(i*8+col) *dequant_out_per_CT_coulmn },
                    {1,1,1, INT_Q4NX_BLOCK_PER_ROW *(4*Q4NX_BLOCK_SIZE)},
                    {0,0,0,1},
                    -1,0,true
                );
                bd_wait_counter[col]++;
            }
        }

        // note: for now
        for(int col = 0; col < 8; col++){
            if(bd_wait_counter[col] == 2){
                seq.npu_dma_wait(shim_tiles[col], S2MM, it_channel_0);
                bd_wait_counter[col] --;
            }
        }
    }

    for(int col = 0; col < 8; col++){
        while(bd_wait_counter[col] != 0){
            seq.npu_dma_wait(shim_tiles[col], S2MM, it_channel_0);
            bd_wait_counter[col] --;
        }
    }
}

template<int ARG_QUANT_IN, int ARG_DEQUANT_OUT,
    int Q4NX_ROW,  int Q4NX_COL,  int Q4NX_BLOCK_BYTE_SIZE,
    int D_SIZE_PADDED, int INTERMEDIATE_SIZE_PADDED>
void dequant_down_matrix_reorder_for_padding(
    npu_sequence& seq,
    std::vector<npu_tiles>& shim_tiles,
    uint32_t expert_offest,
    uint32_t output_offset,
    uint32_t INTERMEDIATE_size_mem_padded,
    uint32_t MM_K_size
){
    int bd_wait_counter[8] = {0,0,0,0,0,0,0,0};

    // matrix is D_SIZE_PADDED x INTERMEDIATE_SIZE_PADDED

    constexpr uint32_t INT_Q4NX_BLOCK_PER_ROW = INTERMEDIATE_SIZE_PADDED/Q4NX_COL;
    constexpr uint32_t quant_in_per_CT_column  = 4* Q4NX_BLOCK_BYTE_SIZE *  (INTERMEDIATE_SIZE_PADDED/Q4NX_COL);
    // constexpr uint32_t dequant_out_per_CT_coulmn = 4*Q4NX_ROW* INTERMEDIATE_SIZE_PADDED;

    //D_PADDED_FOUR_Q4NX_ROW
    constexpr uint32_t down_column = D_SIZE_PADDED/(4*Q4NX_ROW);

    constexpr uint32_t Q4NX_BLOCK_SIZE = Q4NX_ROW*Q4NX_COL;
    if((2*Q4NX_BLOCK_BYTE_SIZE)%128 !=0){
        std::cout << "dequant mxfp4: block size not aligned" << std::endl;
        exit(1);
    }
    // first, the dequant of down
    for(int i = 0; i < down_column; i++){

        for(int col = 0; col < 8; col++){
            uint32_t bd_offset = (i % 2) * 8;

            if( i*8 + col < down_column  ){
                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    ARG_QUANT_IN,
                    MM2S,
                    shim_tiles[col],
                    (npu_bd_id)(0+bd_offset),
                    it_channel_0,
                    {0,0,0, expert_offest+ (i*8+col) *quant_in_per_CT_column },
                    { 1,    INT_Q4NX_BLOCK_PER_ROW ,       (2*Q4NX_BLOCK_BYTE_SIZE)/128 , 128},
                    { 0,    4*Q4NX_BLOCK_BYTE_SIZE ,                      128,             1},
                    -1,0, false,
                    aggressive_cache

                );
                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    ARG_QUANT_IN,
                    MM2S,
                    shim_tiles[col],
                    (npu_bd_id)(2+bd_offset),
                    it_channel_1,
                    {0,0,0, expert_offest+ (i*8+col) *quant_in_per_CT_column + 2*Q4NX_BLOCK_BYTE_SIZE },
                    { 1,    INT_Q4NX_BLOCK_PER_ROW ,       (2*Q4NX_BLOCK_BYTE_SIZE)/128 , 128},
                    { 0,    4*Q4NX_BLOCK_BYTE_SIZE ,                      128,             1},
                    -1,0, false,
                    aggressive_cache

                );

                // Each port receive 2*Q4NX_ROWx D_Q4NX_BLOCK_PER_ROW*Q4NX_COL
                seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),
                    ARG_DEQUANT_OUT,
                    S2MM,
                    shim_tiles[col],
                    (npu_bd_id)(1+bd_offset),
                    it_channel_0,
                    {0,0,0,  output_offset +(i*8+col) *( 4*Q4NX_ROW*INTERMEDIATE_size_mem_padded ) },
                    {INTERMEDIATE_SIZE_PADDED/MM_K_size,                MM_K_size/Q4NX_COL        ,Q4NX_ROW*4           ,Q4NX_COL},//n==128
                    {MM_K_size*(Q4NX_ROW * 4),                          Q4NX_COL                 ,MM_K_size     ,1},
                    -1,0,true
                );

                seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),
                    ARG_DEQUANT_OUT,
                    S2MM,
                    shim_tiles[col],
                    (npu_bd_id)(3+bd_offset),
                    it_channel_0,
                    {0,0,0,  output_offset +(i*8+col) *( 4*Q4NX_ROW*INTERMEDIATE_size_mem_padded ) +   ( INTERMEDIATE_SIZE_PADDED/MM_K_size ) * MM_K_size * Q4NX_ROW * 4 },
                    {1,         (INTERMEDIATE_SIZE_PADDED%MM_K_size)/Q4NX_COL ,Q4NX_ROW*4           ,Q4NX_COL},//n==128
                    {0,         Q4NX_COL                                      ,MM_K_size           ,1},
                    -1,0,true
                );
                bd_wait_counter[col]++;
            }
        }

        // note: for now
        for(int col = 0; col < 8; col++){
            if(bd_wait_counter[col] == 2){
                seq.npu_dma_wait(shim_tiles[col], S2MM, it_channel_0);
                seq.npu_dma_wait(shim_tiles[col], S2MM, it_channel_0);
                bd_wait_counter[col] --;
            }
        }
    }

    for(int col = 0; col < 8; col++){
        while(bd_wait_counter[col] != 0){
            seq.npu_dma_wait(shim_tiles[col], S2MM, it_channel_0);
                seq.npu_dma_wait(shim_tiles[col], S2MM, it_channel_0);
            bd_wait_counter[col] --;
        }
    }
}

// template<  int Q4NX_ROW,  int Q4NX_COL,  int Q4NX_BLOCK_BYTE_SIZE,
//     npu_sequence& seq,

//     // for now, just dequant the4 first 4 column

//     seq.clear_cmds();

//     uint32_t input_per_expert =  (gate_up_input_per_expert + down_input_per_expert);

//     uint32_t output_per_expert = 2*mat_size_gate_up_pad + mat_size_down_pad;

//     dequant_gate_up_matrix<ARG_QUANT_IN, ARG_DEQUANT_OUT, false,
//     Q4NX_ROW,  Q4NX_COL,   Q4NX_BLOCK_BYTE_SIZE,
//     GATE_UP_INTERMEDIATE_SIZE_PADDED, GATE_UP_D_SIZE_PADDED
//     >(
//         seq,
//         shim_tiles,
//         expert_id*input_per_expert,0
//     );

//     dequant_gate_up_matrix<ARG_QUANT_IN, ARG_DEQUANT_OUT, true,
//         Q4NX_ROW,  Q4NX_COL,   Q4NX_BLOCK_BYTE_SIZE,
//         GATE_UP_INTERMEDIATE_SIZE_PADDED, GATE_UP_D_SIZE_PADDED
//     >(
//         seq,
//         shim_tiles,
//         expert_id*input_per_expert, mat_size_gate_up_pad
//     );

//     dequant_down_matrix<ARG_QUANT_IN, ARG_DEQUANT_OUT,
//     Q4NX_ROW,  Q4NX_COL,   Q4NX_BLOCK_BYTE_SIZE,
//     DOWN_D_SIZE_PADDED, DOWN_INTERMEDIATE_SIZE_PADDED
//     >(
//         seq,
//         shim_tiles,
//         expert_id*input_per_expert +  gate_up_input_per_expert,
//          2*mat_size_gate_up_pad
//     );

//     seq.cmds2seq();

// }

// template<  int Q4NX_ROW,  int Q4NX_COL,  int Q4NX_BLOCK_BYTE_SIZE,
//     >
//     npu_sequence& seq,

//     // for now, just dequant the4 first 4 column

//     seq.clear_cmds();

//     uint32_t input_per_expert =  (gate_up_input_per_expert + down_input_per_expert);

//         dequant_gate_up_matrix_reorder_for_padding<ARG_QUANT_IN, ARG_DEQUANT_OUT, false,
//         Q4NX_ROW,  Q4NX_COL,   Q4NX_BLOCK_BYTE_SIZE,
//         GATE_UP_INTERMEDIATE_SIZE_PADDED, GATE_UP_D_SIZE_PADDED
//         >(
//             seq,
//             shim_tiles,
//             i*input_per_expert,i*output_per_expert,
//             GATE_UP_D_SIZE_MEM_PADDED, MM_K_size
//         );

//         dequant_gate_up_matrix_reorder_for_padding<ARG_QUANT_IN, ARG_DEQUANT_OUT, true,
//             Q4NX_ROW,  Q4NX_COL,   Q4NX_BLOCK_BYTE_SIZE,
//             GATE_UP_INTERMEDIATE_SIZE_PADDED, GATE_UP_D_SIZE_PADDED
//         >(
//             seq,
//             shim_tiles,
//             i*input_per_expert, i*output_per_expert +mat_size_gate_up_pad,
//             GATE_UP_D_SIZE_MEM_PADDED, MM_K_size
//         );

//         dequant_down_matrix_reorder_for_padding<ARG_QUANT_IN, ARG_DEQUANT_OUT,
//         Q4NX_ROW,  Q4NX_COL,   Q4NX_BLOCK_BYTE_SIZE,
//         DOWN_D_SIZE_PADDED, DOWN_INTERMEDIATE_SIZE_PADDED
//         >(
//             seq,
//             shim_tiles,
//             i*input_per_expert +  gate_up_input_per_expert,
//             i*output_per_expert+ 2*mat_size_gate_up_pad,
//             DOWN_INTERMEDIATE_SIZE_MEM_PADDED, MM_K_size
//         );

//     }

//     seq.cmds2seq();

// }

template<  int Q4NX_ROW,  int Q4NX_COL,  int Q4NX_BLOCK_BYTE_SIZE,
    int GATE_UP_INTERMEDIATE_SIZE_PADDED, int GATE_UP_D_SIZE_PADDED,
    int DOWN_D_SIZE_PADDED, int DOWN_INTERMEDIATE_SIZE_PADDED,
    int NUM_OF_EXPERT
    >
void generate_mxfp4_gpotss_dequant_sequence(
    npu_sequence& seq,
    uint32_t mat_size_gate_up_pad,
    uint32_t mat_size_down_pad
){

    // for now, just dequant the4 first 4 column

    const int ARG_QUANT_IN = 0;
    const int ARG_DEQUANT_OUT = 1;

    seq.clear_cmds();
    seq.npu_preemption(0);

    constexpr uint32_t dequant_rtp_address  = 35328;
    constexpr int  CT_lock_address_base = 0x000001F000;
    constexpr int rtp_lock_id = 6;

    constexpr uint32_t  rtp_lock_address =  CT_lock_address_base+16*(rtp_lock_id);

    for(size_t row_idx = 0; row_idx < 4; row_idx++){
        for(size_t col_idx = 0; col_idx< 8; col_idx++){
            auto CT_tile = get_tile(row_idx+2, col_idx);
            // set RTP value
            seq.rtp_write( CT_tile, dequant_rtp_address, 1 );
        }
    }

    std::vector<npu_tiles> shim_tiles;
    for(int i = 0; i < 8; i++){
        shim_tiles.push_back( get_tile(0, i));
    }

    uint32_t gate_up_input_per_expert = 2*GATE_UP_INTERMEDIATE_SIZE_PADDED* GATE_UP_D_SIZE_PADDED;
    gate_up_input_per_expert =   (gate_up_input_per_expert / (Q4NX_ROW*Q4NX_COL))*Q4NX_BLOCK_BYTE_SIZE;

    uint32_t down_input_per_expert = DOWN_D_SIZE_PADDED * GATE_UP_D_SIZE_PADDED;
    down_input_per_expert = (down_input_per_expert / (Q4NX_ROW*Q4NX_COL))*Q4NX_BLOCK_BYTE_SIZE;

    uint32_t input_per_expert =  (gate_up_input_per_expert + down_input_per_expert);

    uint32_t output_per_expert = 2*mat_size_gate_up_pad + mat_size_down_pad;
    for(int i = 0; i < NUM_OF_EXPERT; i++){

        dequant_gate_up_matrix<ARG_QUANT_IN, ARG_DEQUANT_OUT, false,
        Q4NX_ROW,  Q4NX_COL,   Q4NX_BLOCK_BYTE_SIZE,
        GATE_UP_INTERMEDIATE_SIZE_PADDED, GATE_UP_D_SIZE_PADDED
        >(
            seq,
            shim_tiles,
            i*input_per_expert,i*output_per_expert
        );

        dequant_gate_up_matrix<ARG_QUANT_IN, ARG_DEQUANT_OUT, true,
            Q4NX_ROW,  Q4NX_COL,   Q4NX_BLOCK_BYTE_SIZE,
            GATE_UP_INTERMEDIATE_SIZE_PADDED, GATE_UP_D_SIZE_PADDED
        >(
            seq,
            shim_tiles,
            i*input_per_expert, i*output_per_expert +mat_size_gate_up_pad
        );

        dequant_down_matrix<ARG_QUANT_IN, ARG_DEQUANT_OUT,
        Q4NX_ROW,  Q4NX_COL,   Q4NX_BLOCK_BYTE_SIZE,
        DOWN_D_SIZE_PADDED, DOWN_INTERMEDIATE_SIZE_PADDED
        >(
            seq,
            shim_tiles,
            i*input_per_expert +  gate_up_input_per_expert,
            i*output_per_expert+ 2*mat_size_gate_up_pad
        );
    }

    seq.cmds2seq();
}
