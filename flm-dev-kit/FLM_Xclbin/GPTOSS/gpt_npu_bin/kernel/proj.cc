// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "common.h"
#include "q4_k.h"
#include "packet_control_gen.hpp"
#include "gptossParam.hpp"
#include "common_external.hpp"
#include "passThroughCommon.hpp"

#include "zero.h"
#include "mvm_MXFP4.hpp"

// template<int m>
// void zero_y(float * y) {
//     zero_vectorized_512<float, m,1>(y);
// }

// ///@brief pseduo q projection: y = linear(w, x)
// ///@param y: output
// ///@param x: input
// ///@param w: weight
// template<int m, int k>
// void pseduo_proj(
//     float * y,
//     const bf16 * x,
//     const bf16 * w,
//     const int offset
// ) {

//     for (int i = 0; i < m; i++){
//         float acc = y[i];
//         for (int j = 0; j < k; j++){
//             acc += w[i * k + j] * x[j + offset];
//         }
//         y[i] = acc;
//     }
// }



// template<int m>
// void copy_y(bf16 * y, const float * y_acc) {
//     // for (int i = 0; i < m / 16; i++){
//     //     aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + i * 16);
//     //     aie::accum<accfloat, 16> acc;
//     //     acc.from_vector(y_vec);
//     //     aie::store_v(y + i * 16, acc.template to_vector<bf16>());
//     // }
//     // write a scalar version

//     for(int i = 0; i < m ; i++){
//         y[i] = static_cast<bfloat16>(y_acc[i]);
//     }
// }

// phase 1: qkv proj, D -> (D + D // 2) to RoPE
// phase 2: o_proj, D -> D to rms_residual
// phase 3: up/gate proj, D -> D * 8 to MT_GLU
// phase 4: down proj, D * 4 -> D, to rms_residual
// phase 4 is special, do not clear y to zero
// static bool is_y_ping = false;
// static bool is_wx_ping = false;



template<int M, int K, int m, int k>
void linear_proj_iD(
    bf16 * x_ping,
    bf16 * x_pong,
    q4k_block * w_ping,
    q4k_block * w_pong,
    bf16 * y_ping,
    bf16 * y_pong,
    const uint32 pkt_id,
    float* y_acc,
    int32_t*  y_ping_pong_flag,
    int32_t* wx_ping_pong_flag       
) {

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
    
    uint32* pkt_id_ping = reinterpret_cast<uint32*>(y_ping + 14);
    uint32* pkt_id_pong = reinterpret_cast<uint32*>(y_pong + 14);
    //TODO: optimize flags
    for(int i = 0; i <M / m; i++ ){
        zero_vectorized_512<float, m, 1>(y_acc);
        for(int j = 0; j <K / k; j++ ){
            q4k_block* w_using;
            bf16* x_using;
            _lock_acquire(w_cons_lock);
            _lock_acquire(x_cons_lock);
            if(*wx_ping_pong_flag == 0 ){
                w_using = w_ping;
                x_using = x_ping;
                *wx_ping_pong_flag=1;
            }else{
                w_using = w_pong;
                x_using = x_pong;
                *wx_ping_pong_flag=0;
            }


  
            _qmm_q4k_bf16<m, k>(w_using, x_using, y_acc);
            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
    
        } 
        _lock_acquire(y_prod_lock);

        if(*y_ping_pong_flag == 0){
            *pkt_id_ping= packet_flow_gen(0,pkt_id);
            
            copy_float_to_bf16<m>(y_ping + 16, y_acc);
            *y_ping_pong_flag = 1;
        }else{
            *pkt_id_pong = packet_flow_gen(0,pkt_id);
            copy_float_to_bf16<m>(y_pong + 16, y_acc);
            *y_ping_pong_flag = 0;            
        }
        _lock_release(y_cons_lock);
   
    } 
}




// template<int M, int K, int m, int k>
// void debug(
//     bf16 * x_ping,
//     bf16 * x_pong,
//     q4k_block * w_ping,
//     q4k_block * w_pong,
//     bf16 * y_ping,
//     bf16 * y_pong,
//     const uint32 pkt_id,
//     float* y_acc,
//     int32_t*  y_ping_pong_flag,
//     int32_t* wx_ping_pong_flag       
// ) {

// constexpr int x_prod_lock = 0;
// constexpr int x_cons_lock = 1;
// constexpr int w_prod_lock = 2;
// constexpr int w_cons_lock = 3;
// constexpr int y_prod_lock = 4;
// constexpr int y_cons_lock = 5;
    
//     uint32* pkt_id_ping = reinterpret_cast<uint32*>(y_ping + 14);
//     uint32* pkt_id_pong = reinterpret_cast<uint32*>(y_pong + 14);
//     //TODO: optimize flags
//     for(int i = 0; i <M / m; i++ ){
//         zero_vectorized_512<float, m, 1>(y_acc);
//         for(int j = 0; j <K / k; j++ ){
//             q4k_block* w_using;
//             bf16* x_using;
//             //_lock_acquire(w_cons_lock);
//             _lock_acquire(x_cons_lock);
//             if(*wx_ping_pong_flag == 0 ){
//                 w_using = w_ping;
//                 x_using = x_ping;
//                 *wx_ping_pong_flag=1;
//             }else{
//                 w_using = w_pong;
//                 x_using = x_pong;
//                 *wx_ping_pong_flag=0;
//             }


  
//             _qmm_q4k_bf16<m, k>(w_using, x_using, y_acc);
//             //_lock_release(w_prod_lock);
//             _lock_release(x_prod_lock);
    
//         } 
//         _lock_acquire(y_prod_lock);

//         if(*y_ping_pong_flag == 0){
//             *pkt_id_ping= packet_flow_gen(0,pkt_id);
            
//             copy_float_to_bf16<m>(y_ping + 16, y_acc);
//             *y_ping_pong_flag = 1;
//         }else{
//             *pkt_id_pong = packet_flow_gen(0,pkt_id);
//             copy_float_to_bf16<m>(y_pong + 16, y_acc);
//             *y_ping_pong_flag = 0;            
//         }
//         _lock_release(y_cons_lock);
   
//     } 
// }







template<int32_t q4nx_mat_row, int32_t q4nx_mat_col, bool enable_BIAS=true> 
void MVM_Q4NX_MFP4_block(
    uint8_t * scalar_data_ping, uint8_t *scalar_data_pong,

    bfloat16* vec_in_ping, bfloat16* vec_in_pong, // assume single buffer stores 1x q4nx_mat_col size already
    bfloat16* output_ping, bfloat16* output_pong,  // Each is size of 1xQ4NX_ROW
    
    float* result_acc_ptr,


    int32_t packet_id,
    int32_t * input_ping_pong_flag, int32_t *output_ping_pong_flag

){

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;

    constexpr int  data_block_offset =   Q4NX_ROW * (Q4NX_COL/32) * 4;
    
    uint8_t* cur_scalar_ptr;
    uint8_t* cur_data_block_ptr;
    bfloat16 * cur_output_ptr;

    bfloat16* cur_vec_in;

    uint32* pkt_id_ping = reinterpret_cast<uint32*>(output_ping + 14);
    uint32* pkt_id_pong = reinterpret_cast<uint32*>(output_pong + 14);

    for(int row_idx = 0; row_idx <q4nx_mat_row; row_idx +=Q4NX_ROW  ){
        
        // zero pit the acc for new row
        if(enable_BIAS == false){
            zero_vectorized_512<float, Q4NX_ROW, 1>(result_acc_ptr);
        }

        _lock_acquire(y_prod_lock, 1);            
        if(*output_ping_pong_flag == 0 ){
            cur_output_ptr = output_ping;
            *pkt_id_ping= packet_flow_gen(0,packet_id);  
            *output_ping_pong_flag = 1;
        }else{
            cur_output_ptr = output_pong;
            *pkt_id_pong= packet_flow_gen(0,packet_id);
            *output_ping_pong_flag = 0;
        }



        if (enable_BIAS){
            if(*input_ping_pong_flag == 0){
                cur_scalar_ptr = scalar_data_ping;
                cur_data_block_ptr= scalar_data_ping + data_block_offset;
                cur_vec_in = vec_in_ping;
                *input_ping_pong_flag = 1;
            }else{
                cur_scalar_ptr = scalar_data_pong;
                cur_data_block_ptr= scalar_data_pong + data_block_offset;
                cur_vec_in = vec_in_pong;
                *input_ping_pong_flag=0;
            }                

            _lock_acquire(w_cons_lock, 1);
            _lock_acquire(x_cons_lock, 1);

            uint8_t * _bias_ptr = cur_scalar_ptr + Q4NX_ROW * (Q4NX_COL/32);
            bfloat16* bias_ptr = (bfloat16*)_bias_ptr;
            copy_bf16_to_float<Q4NX_ROW>(result_acc_ptr,bias_ptr );

            mvm_MXFP4<Q4NX_ROW, Q4NX_COL, Q4NX_ROWL_TILE >(
                cur_scalar_ptr,  cur_data_block_ptr,
                cur_vec_in,
                result_acc_ptr
            );
        
            _lock_release(w_prod_lock, 1);
            _lock_release(x_prod_lock,1);
        }else{
            if(*input_ping_pong_flag == 0){
                cur_scalar_ptr = scalar_data_ping;
                cur_data_block_ptr= scalar_data_ping + data_block_offset;
                cur_vec_in = vec_in_ping;
                *input_ping_pong_flag = 1;
            }else{
                cur_scalar_ptr = scalar_data_pong;
                cur_data_block_ptr= scalar_data_pong + data_block_offset;
                cur_vec_in = vec_in_pong;
                *input_ping_pong_flag=0;
            }                

            _lock_acquire(w_cons_lock, 1);
            _lock_acquire(x_cons_lock, 1);


            mvm_MXFP4<Q4NX_ROW, Q4NX_COL, Q4NX_ROWL_TILE >(
                cur_scalar_ptr,  cur_data_block_ptr,
                cur_vec_in,
                result_acc_ptr
            );
        
            _lock_release(w_prod_lock, 1);
            _lock_release(x_prod_lock,1);
        }
        for(int col_idx = Q4NX_COL; col_idx <q4nx_mat_col; col_idx += Q4NX_COL ){

            if(*input_ping_pong_flag == 0){
                cur_scalar_ptr = scalar_data_ping;
                cur_data_block_ptr= scalar_data_ping + data_block_offset;
                cur_vec_in = vec_in_ping;
                *input_ping_pong_flag = 1;
            }else{
                cur_scalar_ptr = scalar_data_pong;
                cur_data_block_ptr= scalar_data_pong + data_block_offset;
                cur_vec_in = vec_in_pong;
                *input_ping_pong_flag=0;
            }                

            _lock_acquire(w_cons_lock, 1);
            _lock_acquire(x_cons_lock, 1);
            if (col_idx == 0 && enable_BIAS){
                uint8_t * _bias_ptr = cur_scalar_ptr + Q4NX_ROW * (Q4NX_COL/32);
                bfloat16* bias_ptr = (bfloat16*)_bias_ptr;
                copy_bf16_to_float<Q4NX_ROW>(result_acc_ptr,bias_ptr );
            }

            mvm_MXFP4<Q4NX_ROW, Q4NX_COL, Q4NX_ROWL_TILE >(
                cur_scalar_ptr,  cur_data_block_ptr,
                cur_vec_in,
                result_acc_ptr
            );
        
            _lock_release(w_prod_lock, 1);
            _lock_release(x_prod_lock,1);
        }
        
        // now, save to result
        copy_float_to_bf16<Q4NX_ROW>(cur_output_ptr+16, result_acc_ptr );
        _lock_release(y_cons_lock, 1);
    }



}

extern "C" {

void proj(
    bf16 * y_ping,
    q4k_block * w_ping,
    bf16 * x_ping,
    bf16 * y_pong,
    q4k_block * w_pong,
    bf16 * x_pong,
    float* y_acc,
    int32_t*  y_ping_pong_flag,
    int32_t* wx_ping_pong_flag,
    int32_t* rtp_buffer
#if !defined(__chess__)
    , uint32_t* lut_ab
    , uint32_t* lut_cd
#endif
)
{
#if !defined(__chess__)
    mxfp4_lut_ab = lut_ab;
    mxfp4_lut_cd = lut_cd;
#endif
    constexpr int rtp_lock = 6;
    
    _lock_acquire(rtp_lock, 1);
    if(*rtp_buffer == 0){ // first one is whether we enable/disable the last layer lm_head

        //assume it satisfy the condition
        static_assert(VOCAB_SIZE_PADDED % MVM_DECODE_CORES ==0);
         // phase 5: lm_head proj, D -> VOCAB_SIZE
        linear_proj_iD< VOCAB_SIZE_PADDED / MVM_DECODE_CORES, D_PADDED_Q4NX_COL, Q4NX_ROW, Q4NX_COL>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 4,
            y_acc, y_ping_pong_flag,wx_ping_pong_flag );
       
    }else{



        // phase 1: qkv proj, D -> (D + D / 2) to RoPE
        linear_proj_iD<(GPTOSS_DQ + GPTOSS_DK + GPTOSS_DV) / MVM_DECODE_CORES, D_PADDED_Q4NX_COL, Q4NX_ROW, Q4NX_COL>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 1, 
            
            y_acc, y_ping_pong_flag,wx_ping_pong_flag );

        static_assert(D_ATTEN_O_PADDED_MVM_SIZE %MVM_DECODE_CORES == 0);
        // // phase 2: o proj, D -> D to  RMS
        linear_proj_iD<D_ATTEN_O_PADDED_MVM_SIZE / MVM_DECODE_CORES, GPTOSS_DQ, Q4NX_ROW, Q4NX_COL>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 4,
            y_acc, y_ping_pong_flag,wx_ping_pong_flag 
        );

        // //testing 
        // linear_proj_iD<2*INTERMEDIATE_PADDED_FOUR_Q4NX_ROW / 4, D_PADDED_Q4NX_COL, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 8,
        //      y_acc, y_ping_pong_flag,wx_ping_pong_flag 
        // );

        MVM_Q4NX_MFP4_block< (2*INTERMEDIATE_PADDED_FOUR_Q4NX_ROW)/4,  D_PADDED_Q4NX_COL>(
            reinterpret_cast<uint8_t*>(w_ping),reinterpret_cast<uint8_t*>(w_pong),
            x_ping, x_pong, y_ping, y_pong, y_acc, 8, 
            wx_ping_pong_flag, y_ping_pong_flag

        );

        //NOTE CHANGE to true fot real test
        MVM_Q4NX_MFP4_block< (D_PADDED_FOUR_Q4NX_ROW)/4,  INTERMEDIATE_PADDED_Q4NX_COL>( //DEBUG:
            reinterpret_cast<uint8_t*>(w_ping),reinterpret_cast<uint8_t*>(w_pong),
            x_ping, x_pong, y_ping, y_pong, y_acc, 10, 
            wx_ping_pong_flag, y_ping_pong_flag

        );
        // // // phase 3: up/gate proj, D -> D * 8 to MT_GLU
        // linear_proj_iD<2 * HIDDEN_SIZE / total_cores, D, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 8);

        // // // phase 4.1 down-> 4D -> D, first D -> D
        // linear_proj_iD<D / total_cores, HIDDEN_SIZE, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 5);


    }





}


}