// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

// #include "gemm_detail.hpp"
// #include <cassert>
// #include <algorithm> // Required for std::max
// #include <string>
// #include "npu_utils.hpp"

// // // constructors
// // Gemm::Impl::Impl() {
// //     kk_address =40960; // for 64x512x64
// // }

// Gemm::Impl::~Impl() = default;


// //TODO: could also try generate for each k block, 
// void Gemm::Impl::generate_shimtile_sequence_per_mega_block(
//     uint32_t shim_index,
//     uint32_t mega_block_row_idx, uint32_t mega_block_col_idx,
//     uint32_t M_size, uint32_t K_size, uint32_t N_size,
//     uint32_t Arg_A, uint32_t  Arg_B, uint32_t Arg_C,
//     int &shim_wait_queue,
//     uint32_t weight_offset, 
//     std::map<int, int>& valid_A_MT_shimtile_index,
//     npu_sequence* seq,
//     npu_tiles& cur_shimtile
// ){

   
//     uint32_t AXI_FLAG = 0b1110<<24; 
//     uint32_t DMA_BD_x_5 = 0x000001D014;
//     uint32_t K_div_k = K_size/k;
//     if (shim_wait_queue == 2){
//         // trigger a npu wait 
//         seq->npu_dma_wait(
//             cur_shimtile, S2MM, it_channel_0
//         );
//         shim_wait_queue--;
//     }

//     if (valid_A_MT_shimtile_index.contains(shim_index) && valid_A_MT_shimtile_index[shim_index] < total_rows){
//         uint32_t A_offset = mega_block_row_idx *(total_rows*m) *K_size;
//         A_offset += valid_A_MT_shimtile_index[shim_index]*(m*K_size);
//         npu_bd_id A_bd_id = shim_wait_queue%2 ==0 ? bd_0 : bd_1;
//         seq->npu_dma_memcpy_nd(
//             sizeof(uint16_t),   // bfloat16
//             Arg_A, 
//             MM2S,
//             cur_shimtile,
//             A_bd_id,
//             it_channel_0,
//             {0,0,0,A_offset},
//             {1, K_div_k, m,k},
//             {0, k, K_size, 1},
//             -1, 0, false
//         );
//         seq->npu_maskwrite(
//             cur_shimtile,  DMA_BD_x_5 + 32*(uint32_t)(A_bd_id), AXI_FLAG, 0xF000000
//         );
        
//     }

//     if(shim_index < total_cols){
//         npu_bd_id b_bd_id = shim_wait_queue %2 == 0? bd_2 : bd_3;        

//         uint32_t B_offset = mega_block_col_idx*(total_cols*n)*K_size + weight_offset;
//         B_offset += shim_index * n*K_size;

//         seq->npu_dma_memcpy_nd(
//             sizeof(uint16_t),
//             Arg_B,
//             MM2S,
//             cur_shimtile,
//             b_bd_id,
//             it_channel_1,
//             {0,0,0,B_offset},
//             // {1, K_div_k, n, k},
//             // {0, k, K_size, 1},
//             {1, 1,1, K_div_k* n*k},
//             {0, 0, 0, 1},            
//             -1, 0, false
//         );
        
//         seq->npu_maskwrite(
//             cur_shimtile, DMA_BD_x_5 + 32*(uint32_t)(b_bd_id), AXI_FLAG, 0xF000000
//         );
    

//     }

//     uint32_t C_offset = mega_block_col_idx*n*total_cols;
//     C_offset += mega_block_row_idx* m*total_rows*N_size;
//     C_offset += shim_index *n;

//     if (shim_index < total_cols){
//         npu_bd_id c_bd_id = shim_wait_queue %2 == 0? bd_14 : bd_15;
//         seq->npu_dma_memcpy_nd(
//             sizeof(uint16_t),
//             Arg_C,
//             S2MM, 
//             cur_shimtile,
//             shim_wait_queue %2 == 0? bd_14 : bd_15,
//             it_channel_0,
//             {0,0,0, C_offset},
//             {1,1,4*m, n},
//             {0,0,N_size, 1},
//             -1, 0, true
//         );
//         seq->npu_maskwrite(
//             cur_shimtile, DMA_BD_x_5 + 32*(uint32_t)(c_bd_id), AXI_FLAG, 0xF000000
//         );
        
//         shim_wait_queue++;
//     }

// }



// void Gemm::Impl::generate_one_mega_block_sequence(
//     uint32_t mega_block_row_idx, uint32_t mega_block_col_idx,
//     uint32_t M_size, uint32_t N_size, uint32_t K_size,
//     uint32_t Arg_A, uint32_t  Arg_B, uint32_t Arg_C,
//     std::vector<int> &list_shim_queue,
//     uint32_t weight_offset,
//     std::map<int, int>& valid_A_MT_shimtile_index,
//     npu_sequence* seq,
//     std::vector<npu_tiles> &shim_tiles
// ){
    
//     for (size_t shim_index= 0; shim_index <  std::max(total_cols, total_rows); shim_index++){
//         generate_shimtile_sequence_per_mega_block(
//             shim_index,
//             mega_block_row_idx, mega_block_col_idx,
//             M_size, K_size, N_size,
//             Arg_A, Arg_B, Arg_C,
//             list_shim_queue.at(shim_index),
//             weight_offset,
//             valid_A_MT_shimtile_index,
//             seq,
//             shim_tiles.at(shim_index)
//         );

//     }

// }



// void Gemm::Impl::generate_runtime_sequence(
//     uint32_t Arg_A, uint32_t Arg_B, uint32_t Arg_C,
//     uint32_t M_size, uint32_t N_size, uint32_t K_size,
//     std::vector<int> &list_shim_queue,
//     uint32_t weight_offset,
//     std::map<int, int>& valid_A_MT_shimtile_index,
//     npu_sequence* seq,
//     std::vector<npu_tiles> &shim_tiles
// ){

//     uint32_t M_div_num_row_m = M_size/ (m*total_rows);
//     uint32_t N_div_num_col_n = N_size / (n*total_cols);
//     //MEGA block is 
//     //total_row*m x total_col *n
//     uint32_t mega_block_size = (total_rows*m) * (total_cols*n);
//     assert( (M_size*N_size)% mega_block_size == 0 );

//     uint32_t num_mega_block = (M_size*N_size) / mega_block_size;
//     assert (num_mega_block == (N_div_num_col_n* M_div_num_row_m));

//     for(size_t mega_block_row_idx = 0; mega_block_row_idx <  M_div_num_row_m; mega_block_row_idx++){
//         for(size_t mega_block_col_idx = 0; mega_block_col_idx< N_div_num_col_n; mega_block_col_idx++){
//             generate_one_mega_block_sequence(
//                 mega_block_row_idx, mega_block_col_idx,
//                 M_size, N_size, K_size,
//                 Arg_A, Arg_B, Arg_C,
//                 list_shim_queue,
//                 weight_offset,
//                 valid_A_MT_shimtile_index,
//                 seq,
//                 shim_tiles

//             );

//         }
//     }



// }

// // methods
// // weight offset is constant offset on B matrix
// void Gemm::Impl::generate_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t weight_offset){
//     constexpr int CT_lock_address_base = 0x000001F000;

//     const int K_div_k  = K/k;
//     const int M_div_num_row_m = M/(m*total_rows);
//     const int N_div_num_col_n = N/(n*total_cols);

//     std::cout << "K_divb_k" << K_div_k << std::endl;

//     constexpr int CT_rtp_sync_lock_id =10;// for now hard coded

//     const int Arg_A = 0;
//     const int Arg_B = 1;
//     const int Arg_C = 2;

    
//     seq->clear_cmds();

//     // mapping of valid shimtile index for A -> index offset
//     std::map<int, int> valid_A_MT_shimtile_index;
//     valid_A_MT_shimtile_index[0] = 0;
//     valid_A_MT_shimtile_index[2] = 1;
//     valid_A_MT_shimtile_index[4] = 2;
//     valid_A_MT_shimtile_index[6] = 3;


//     //create list of tiles
//     uint32_t shimtile_size = std::max(total_cols, total_rows);
//     std::vector<npu_tiles> shim_tiles;

//     std::vector<int> list_shim_queue;
//     std::cout <<"shimtile size " << shimtile_size <<std::endl;
//     for(size_t i = 0; i < shimtile_size; i++){
//         shim_tiles.push_back((get_tile(0, i)));
//         list_shim_queue.push_back(0);
//     }
 
//     // first, setup the rtp buffer and the rtp locks
    
//     for(size_t row_idx = 0; row_idx < total_rows; row_idx++){
//         for(size_t col_idx = 0; col_idx< total_cols; col_idx++){
//             auto CT_tile = get_tile(row_idx+2, col_idx);

//             // set RTP value
//             seq->rtp_write( CT_tile, kk_address, K_div_k );
//             seq->rtp_write( CT_tile, kk_address+4, M_div_num_row_m );
//             seq->rtp_write( CT_tile, kk_address+8, N_div_num_col_n );                        
//             // set RTP lock
//             seq->rtp_write(CT_tile, CT_lock_address_base+16*(CT_rtp_sync_lock_id), 1); // set lock to 1
//         }
//     }

//     Gemm::Impl::generate_runtime_sequence(
//         Arg_A, Arg_B, Arg_C,
//         M, N, K,
//         list_shim_queue, weight_offset,
//         valid_A_MT_shimtile_index, seq, shim_tiles
//     );

//     int max_C_remain =  0;
//     for( auto li: list_shim_queue){
//         max_C_remain = std::max(max_C_remain, li);
//     }

//     for(size_t k = 0; k< max_C_remain; k++){
//         for(size_t shim_index = 0; shim_index < total_cols; shim_index++){
            
//             if(list_shim_queue.at(shim_index) > 0){
//                 seq->npu_dma_wait(
//                     shim_tiles.at(shim_index),
//                     S2MM,
//                     it_channel_0

//                 );
//                 list_shim_queue.at(shim_index)--;
//             }
//         }

//     }

//     seq->cmds2seq();

// }
// // wrappers

// Gemm::~Gemm(){
//     delete _impl;
// }

// void Gemm::generate_seq(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t weight_offset){
//     _impl->generate_seq(seq, M, K, N, weight_offset);
// }

// int Gemm::y_id(){
//     return _impl->y_id();
// }

// int Gemm::x_id(){
//     return _impl->x_id();
// }

// int Gemm::w_id(){
//     return _impl->w_id();
// }