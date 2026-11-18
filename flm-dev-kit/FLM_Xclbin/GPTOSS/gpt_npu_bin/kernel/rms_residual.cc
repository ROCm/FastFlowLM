// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "gptossParam.hpp"
#include "common.h"
#include "packet_control_gen.hpp"
#include "passThroughCommon.hpp"
#include "aie_kernel_utils.h" //TODO: a better method?
#include "common_external.hpp"
#include "zero.h"
#include "flm_rsqrt.h"

///@brief  rms norm: y = nrom(x) * w;
///@brief                    x_buf = x
///@param y: output
///@param x_buf: copy of x
///@param x: input
///@param w: weight
void rms_norm(
    bf16 *restrict y,
    const bf16 *restrict x,
    const bf16 *restrict w,
    const float epsilon
){
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_w = const_cast<bf16*>(w);
    

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)D;

    aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = D / vector_size;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(F, F)
    for (int i = 0; i < F; i++){
        auto x_vec = aie::load_v<vector_size>(it_x);
        sum_squares = aie::mac_square(sum_squares, x_vec);
        it_x += vector_size;
    }

    it_x = const_cast<bf16*>(x);

    mx_vec = sum_squares.template to_vector<float>();
    float sum = aie::reduce_add(mx_vec);
#if defined(__chess__)
    sum = sum * one_over_D;
#else
    sum = AIE_FMUL(sum, one_over_D);
#endif
#if defined(__chess__)
    sum = sum + epsilon;
#else
    // Same math as the chess branch, with each scalar op an IEEE softfloat
    // call so Peano rounds like chess (no fused multiply-add).
    sum = AIE_FADD(sum, epsilon);
#endif
    const float divrms = flm_rsqrt(sum);

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(F, F)
    for (int i = 0; i < F; i++){
        aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
        aie::vector<bf16, vector_size> w_vec = aie::load_v<vector_size>(it_w);
        aie::vector<float, vector_size> wx_vec = aie::mul(x_vec, w_vec);
        aie::vector<bf16, vector_size> o_vec = aie::mul(wx_vec, divrms);
        aie::store_v(it_y, o_vec);
        it_x += vector_size;
        it_y += vector_size;
        it_w += vector_size;
    }
  
}


///@brief residual add: y = x + x_buf, and x_buf = y
///@param y: output
///@param x: input
///@param x_buf: copy of x
void residual_add(
    bf16 *restrict y,
    const bf16 *restrict x_buf,
    const bf16 *restrict x
){
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x_buf = const_cast<bf16*>(x_buf);

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(D/16, D/16)
    for (int i = 0; i < D / 16; i++) {
        aie::vector<bf16, 16> x_vec = aie::load_v<16>(it_x);
        aie::vector<bf16, 16> x_buf_vec = aie::load_v<16>(it_x_buf);
        auto out_vec = aie::add(x_vec, x_buf_vec);
        aie::store_v(it_y, out_vec);
        aie::store_v(it_x_buf, out_vec);
        it_x += 16;
        it_y += 16;
        it_x_buf += 16;
    }
}




void add_bias(bfloat16*out,  const bfloat16* x, const bfloat16* bias_in){
    
    static_assert( D%16 == 0);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE( D/16, D/16)
    for(int i = 0; i < D; i+=16 ){
        aie::vector<bfloat16, 16> x_vec = aie::load_v<16>(x+i);
        aie::vector<bfloat16, 16> bias_vec = aie::load_v<16>(bias_in + i );

        aie::vector<bfloat16, 16> res = aie::add(x_vec, bias_vec );

        aie::store_v( out+i, res );
    }

}

inline bfloat16* determine_w_ping_pong_flag(int32_t*w_ping_pong_flag, 
    bfloat16 * w_ping, 
    bfloat16 * w_pong
){
    if(*w_ping_pong_flag==0){

        *w_ping_pong_flag= 1;
        return w_ping;
    }else{

        *w_ping_pong_flag= 0;
        return w_pong;
    }
}



///@brief


extern "C" {

void rms_residual(
    bfloat16 * y, // output of 1XD_PADDED_Q4NX_COL
    bfloat16 * x_final_0, // 1XD, the final output
    bfloat16 * x_final_1, // 1XD, the final output    
    const bfloat16 * x, //1xD
    bfloat16 * _w_ping, 
    bfloat16 * _w_pong,

    bfloat16* x_buf, // 1xD
    bfloat16* temp_buf, //1XD
    bfloat16* temp_buf2, // 1xD another temporary buffer
    uint32_t pkt_id_to_projection, // packet_id_to memtile for doing MVM projection
    uint32_t pkt_id_to_router,
    int32_t* w_ping_pong_flag,
    int32_t* x_out_ping_pong_flag,
    int32_t* rtp_buf
){
    constexpr int w_prod_lock = 0;
    constexpr int w_cons_lock = 1;
    constexpr int y_prod_lock = 2;
    constexpr int y_cons_lock = 3;
    constexpr int x_prod_lock = 4;
    constexpr int x_cons_lock = 5;
    constexpr int x_final_prod_lock = 6;
    constexpr int x_final_cons_lock = 7;
    constexpr int rms_rtp_lock = 8;

    _lock_acquire(rms_rtp_lock, 1);

    uint32_t* pkt_id_ptr = reinterpret_cast<uint32_t*>(y + 14);
    bfloat16* cur_w_ptr;    
    if(*rtp_buf == 1){




        // input later norm + residul
        _lock_acquire(x_cons_lock);
        _lock_acquire(w_cons_lock);

        cur_w_ptr = determine_w_ping_pong_flag(w_ping_pong_flag, _w_ping, _w_pong);
        packet_flow_gen(0, pkt_id_to_projection);
        //memcpy(x_buf, x, D * sizeof(bfloat16)); // copy x for residual later
        copy_bf16_to_bf16<D>(x_buf, x);
        rms_norm(y + 16, x_buf, cur_w_ptr, GPTOSS_RMS_NORM_EP);
        _lock_release(x_prod_lock);
        _lock_release(w_prod_lock);
        _lock_release(y_cons_lock, X_QKV_REPEAT);



        // now, receive the x add the residual
        _lock_acquire(y_prod_lock, X_QKV_REPEAT);
        _lock_acquire(x_cons_lock);    
        _lock_acquire(w_cons_lock);
        cur_w_ptr = determine_w_ping_pong_flag(w_ping_pong_flag, _w_ping, _w_pong);
        add_bias(temp_buf2, x, cur_w_ptr); // temp_buf2 has the update o with bias
        _lock_release(w_prod_lock);

        // then do residual nd norm stuff
        _lock_acquire(w_cons_lock);
        cur_w_ptr = determine_w_ping_pong_flag(w_ping_pong_flag, _w_ping, _w_pong);    
        residual_add(temp_buf, temp_buf2, x_buf); // add residual    
        //memcpy(x_buf, temp_buf, sizeof(bfloat16)*D);
        copy_bf16_to_bf16<D>(x_buf, temp_buf);
        rms_norm(y+16, x_buf, cur_w_ptr, GPTOSS_RMS_NORM_EP);

        _lock_release(w_prod_lock);
        _lock_release(x_prod_lock);

        // first send one copy to router
        *pkt_id_ptr = packet_flow_gen(0, pkt_id_to_router);       
        _lock_release(y_cons_lock, 1); // send one copy off doing MLP stuff
        _lock_acquire(y_prod_lock, 1);

        // then send to mlp
        *pkt_id_ptr = packet_flow_gen(0, pkt_id_to_projection);   
        _lock_release(y_cons_lock, X_UP_GATE_REPEAT); // send off doing MLP stuff

        

        _lock_acquire(y_prod_lock, X_UP_GATE_REPEAT); // after this, y_prod_lock go to 0
        _lock_acquire(x_cons_lock);



        _lock_acquire(x_final_prod_lock, 1);  //y_prod_lock start with 0
        if(*x_out_ping_pong_flag == 0){
            *x_out_ping_pong_flag = 1;
            residual_add(x_final_0, x_buf, x);
        }else{
            *x_out_ping_pong_flag = 0;
            residual_add(x_final_1, x_buf, x);
        }

        _lock_release(x_final_cons_lock, 1);
        _lock_release(x_prod_lock); 
    }else{
        // For now, simply receive the VOCAB_PADDED_SIZE, 
        // zero out the padding datas
        zero_vectorized_512<bf16,  D_PADDED_Q4NX_COL-D, 1>(y+16+D);

        // _lock_acquire(x_cons_lock);
        // copy_bf16_to_bf16<D>(y+16, x);
        // *pkt_id_ptr = packet_flow_gen(0, pkt_id_to_projection);   
        // _lock_release(x_prod_lock); 

        _lock_acquire(x_cons_lock);
        _lock_acquire(w_cons_lock); // final layer rms weight
        cur_w_ptr = determine_w_ping_pong_flag(w_ping_pong_flag, _w_ping, _w_pong);
        // copy_bf16_to_bf16<D>(y+16, x);
        rms_norm(y + 16, x, cur_w_ptr, GPTOSS_RMS_NORM_EP);        
        *pkt_id_ptr = packet_flow_gen(0, pkt_id_to_projection);   
        _lock_release(x_prod_lock); 
        _lock_release(w_prod_lock);


        // do xrepeat for sending out

        // now, receive the VOCAB_SIZE_PADDED size
        static_assert(VOCAB_SIZE_PADDED %D_ATTEN_O_PADDED_MVM_SIZE == 0 );
        static_assert(D_ATTEN_O_PADDED_MVM_SIZE %(MVM_DECODE_CORES*Q4NX_ROW) == 0);


        for(int i = 0; i <VOCAB_SIZE_PADDED/D_ATTEN_O_PADDED_MVM_SIZE; i++  ){


            constexpr uint32_t Y_REPEAT = D_ATTEN_O_PADDED_MVM_SIZE/(MVM_DECODE_CORES*Q4NX_ROW);
            _lock_release(y_cons_lock, Y_REPEAT); // send off doing MLP stuff

            _lock_acquire(x_cons_lock); // from projection tile
            _lock_acquire(x_final_prod_lock, 1);  //y_prod_lock start with 0
            if(*x_out_ping_pong_flag == 0){
                *x_out_ping_pong_flag = 1;
                copy_bf16_to_bf16<D_ATTEN_O_PADDED_MVM_SIZE>(x_final_0, x);
            }else{
                *x_out_ping_pong_flag = 0;
                copy_bf16_to_bf16<D_ATTEN_O_PADDED_MVM_SIZE>(x_final_1, x);
            }

            _lock_release(x_final_cons_lock, 1);
            _lock_release(x_prod_lock); 
            _lock_acquire(y_prod_lock, Y_REPEAT);


        }



    }

   
}

}