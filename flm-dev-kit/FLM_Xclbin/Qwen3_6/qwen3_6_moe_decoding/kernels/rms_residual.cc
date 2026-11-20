// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"

#include "passThroughCommon.hpp"
#include "nonLinearFunction.h"


void share_expert_output_gate(
    bf16* restrict hidden_state_in, //1xModel_DIM
    bf16* shared_expert_gate_weight, //1xModel_DIM
    bf16* shared_expert_output //1xModel_DIM
    

){


    // // For faster computation, convert the shared_expert_gate_weight from fp32 to bf16
    // alignas(aie::vector_decl_align) bfloat16 shared_expert_gate_weight_bf16[MODEL_DIM];
    // copy_bf16_to_bf16<MODEL_DIM>(
    //     shared_expert_gate_weight_bf16, shared_expert_gate_weight
    // );

    // equation is sigmoid( shared_expert_gate_weight dot_prod hidden_state_in  ) * shared_expert_output
    static constexpr int vec_size = 32;
    static_assert(MODEL_DIM%vec_size == 0 );
    

    static_assert(MODEL_DIM >= vec_size); 
    // manually unroll first iteration
    aie::vector<bf16, vec_size> hidd_vec = aie::load_v<vec_size>(hidden_state_in);
    aie::vector<bf16, vec_size> gate_vec = aie::load_v<vec_size>(shared_expert_gate_weight);
    aie::accum<accfloat, vec_size> mul_acc = aie::mul(hidd_vec, gate_vec);


    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(MODEL_DIM/vec_size,MODEL_DIM/vec_size)    
    for(int i = vec_size; i <MODEL_DIM; i+=vec_size ){
        hidd_vec = aie::load_v<vec_size>(hidden_state_in + i);
        gate_vec = aie::load_v<vec_size>(shared_expert_gate_weight + i);

        mul_acc = aie::mac(mul_acc, hidd_vec, gate_vec);
    }
    bf16 gate_scale = aie::reduce_add(mul_acc.template to_vector<float>());

    aie::vector<bf16, vec_size> gate_scale_vec = aie::broadcast<bf16, vec_size>((bf16)gate_scale);
    aie::vector<bf16, vec_size> gate_scale_sigmoid_vec = getSigmoidBf16_nonLUT<vec_size, bf16>(gate_scale_vec);


    // now, apply this scalar to // shared_expert_output
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(MODEL_DIM/vec_size,MODEL_DIM/vec_size)
    for(int i = 0; i < MODEL_DIM; i+= vec_size){
        aie::vector<bf16, vec_size>shared_expert_vec = aie::load_v<vec_size>(shared_expert_output + i);

        aie::vector<bf16, vec_size> res = aie::mul(shared_expert_vec, gate_scale_sigmoid_vec);
        aie::store_v(shared_expert_output+i,res );
    }



}







///@brief  rms norm: y = nrom(x) * w;
///@brief                    x_buf = x
///@param y: output
///@param x_buf: copy of x
///@param x: input
///@param w: weight
void rms_norm(
    bf16 *restrict y1,
    bf16 *restrict y2,
    const bf16 *restrict x,
    const bf16 *restrict w
){
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)MODEL_DIM;

    bf16 *it_y1_0 = const_cast<bf16*>(y1);
    bf16 *it_y1_1 = const_cast<bf16*>(y1) + vector_size;
    bf16 *it_y2_0 = const_cast<bf16*>(y2);
    bf16 *it_y2_1 = const_cast<bf16*>(y2) + vector_size;
    bf16 *it_x_0 = const_cast<bf16*>(x);
    bf16 *it_x_1 = const_cast<bf16*>(x) + vector_size;
    bf16 *it_w_0 = const_cast<bf16*>(w);
    bf16 *it_w_1 = const_cast<bf16*>(w) + vector_size;

    aie::accum<accfloat, vector_size> sum_squares_0 = aie::zeros<accfloat>();
    aie::accum<accfloat, vector_size> sum_squares_1 = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = MODEL_DIM / (2 * vector_size);
    for (int i = 0; i < F; i++){
        auto x_vec_0 = aie::load_v<vector_size>(it_x_0);
        auto x_vec_1 = aie::load_v<vector_size>(it_x_1);
        sum_squares_0 = aie::mac_square(sum_squares_0, x_vec_0);
        sum_squares_1 = aie::mac_square(sum_squares_1, x_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
    }

    it_x_0 = const_cast<bf16*>(x);
    it_x_1 = const_cast<bf16*>(x) + vector_size;

    mx_vec = aie::add(sum_squares_0.template to_vector<float>(),
                      sum_squares_1.template to_vector<float>());
    float sum = aie::reduce_add(mx_vec);
#if defined(__chess__)
    sum = sum * one_over_D + epsilon;
#else
    sum = AIE_FADD(AIE_FMUL(sum, one_over_D), epsilon);  // round like chess, no fused multiply-add
#endif
    aie::vector<float, vector_size> sum_vec = aie::broadcast<float, vector_size>(sum);
    float divrms = aie::invsqrt(sum_vec)[0];

    for (int i = 0; i < F; i++){
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::vector<bf16, vector_size> w_vec_0 = aie::load_v<vector_size>(it_w_0);
        aie::vector<bf16, vector_size> w_vec_1 = aie::load_v<vector_size>(it_w_1);
        aie::vector<float, vector_size> wx_vec_0 = aie::mul(x_vec_0, w_vec_0);
        aie::vector<float, vector_size> wx_vec_1 = aie::mul(x_vec_1, w_vec_1);
        aie::vector<bf16, vector_size> o_vec_0 = aie::mul(wx_vec_0, divrms);
        aie::vector<bf16, vector_size> o_vec_1 = aie::mul(wx_vec_1, divrms);
        aie::store_v(it_y1_0, o_vec_0);
        aie::store_v(it_y1_1, o_vec_1);
        aie::store_v(it_y2_0, o_vec_0);
        aie::store_v(it_y2_1, o_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_y1_0 += 2 * vector_size;
        it_y1_1 += 2 * vector_size;
        it_y2_0 += 2 * vector_size;
        it_y2_1 += 2 * vector_size;
        it_w_0 += 2 * vector_size;
        it_w_1 += 2 * vector_size;
    }
    event1();
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
    constexpr int vector_size = 16;
    bf16 *it_x_0 = const_cast<bf16*>(x);
    bf16 *it_x_1 = const_cast<bf16*>(x) + vector_size;
    bf16 *it_y_0 = const_cast<bf16*>(y);
    bf16 *it_y_1 = const_cast<bf16*>(y) + vector_size;
    bf16 *it_x_buf_0 = const_cast<bf16*>(x_buf);
    bf16 *it_x_buf_1 = const_cast<bf16*>(x_buf) + vector_size;

    for (int i = 0; i < MODEL_DIM / (2 * vector_size); i++) {
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::vector<bf16, vector_size> x_buf_vec_0 = aie::load_v<vector_size>(it_x_buf_0);
        aie::vector<bf16, vector_size> x_buf_vec_1 = aie::load_v<vector_size>(it_x_buf_1);
        auto out_vec_0 = aie::add(x_vec_0, x_buf_vec_0);
        auto out_vec_1 = aie::add(x_vec_1, x_buf_vec_1);
        aie::store_v(it_y_0, out_vec_0);
        aie::store_v(it_y_1, out_vec_1);
        aie::store_v(it_x_buf_0, out_vec_0);
        aie::store_v(it_x_buf_1, out_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_y_0 += 2 * vector_size;
        it_y_1 += 2 * vector_size;
        it_x_buf_0 += 2 * vector_size;
        it_x_buf_1 += 2 * vector_size;
    }
}


///@brief


extern "C" {

void rms_residual(
    bf16 *restrict y,
    bf16 *restrict x_ping,
    bf16 *restrict x_pong,
    bf16 *restrict w,
    bf16 *restrict y_out,
    bf16 *restrict x_buf
){
    constexpr int w_prod_lock = 0;
    constexpr int w_cons_lock = 1;
    constexpr int y_prod_lock = 2;
    constexpr int y_cons_lock = 3;
    constexpr int x_prod_lock = 4;
    constexpr int x_cons_lock = 5;
    constexpr int rtp_available_lock = 6;
    constexpr int ab_out_prod_lock = 7;
    constexpr int ab_out_cons_lock = 8;
#if defined(__chess__)
    uint32_t* pkt_id_ptr = reinterpret_cast<uint32_t*>(y + 14);
#else
    // y is restrict, so Peano assumes the lock calls cannot read the packet id and
    // drops a store that a later one overwrites. The packet then goes to the
    // wrong consumer and the layer hangs. volatile keeps every store.
    volatile uint32_t* pkt_id_ptr = reinterpret_cast<volatile uint32_t*>(y + 14);
#endif
    bf16* x;

    AIE_PERSIST_ZERO static bool is_x_ping = false;
    AIE_PERSIST_ZERO static bool is_ab_ping = false;
    AIE_PERSIST_ZERO static bool is_w_ping = false;



    alignas(aie::vector_decl_align) bfloat16 post_attn_norm_res[MODEL_DIM];

    bf16* w_input_norm = w;
    bf16* w_post_attention_norm = w_input_norm+ MODEL_DIM;
    bf16* w_share_exp_gate = w_post_attention_norm + MODEL_DIM;
 
    for (int layer = 0; layer < LAYER_PERIOD; layer++){

    if (layer == LAYER_PERIOD - 1) {
        // input later norm + residul
        _lock_acquire(x_cons_lock);
        _lock_acquire(w_cons_lock);
        *pkt_id_ptr = pkt_id_rms_to_proj;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        _memcpy<MODEL_DIM * 2>((uint8*)x_buf, (uint8*)x);
        rms_norm(y + 16, y_out, x_buf,w_input_norm);
        _lock_release(x_prod_lock);

        _lock_release(y_cons_lock, QKV_REPEATS);


        _lock_acquire(x_cons_lock);

        _lock_acquire(y_prod_lock, QKV_REPEATS);
      
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        residual_add(x_buf, x_buf, x);
        // pre_feed_forward
   
        rms_norm(y + 16, post_attn_norm_res, x_buf, w_post_attention_norm);

        _lock_release(x_prod_lock);


        // send a copy of x to router
        *pkt_id_ptr = pkt_id_rms_to_glu;        
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);

        *pkt_id_ptr = pkt_id_rms_to_proj;
        _lock_release(y_cons_lock, UP_GATE_REPEATS); // sending for gate-up


        // a simple acquire of share expert
        _lock_acquire(x_cons_lock);
        _lock_acquire(y_prod_lock, UP_GATE_REPEATS);
  
        // holder place to prevent kernel stuck
        *pkt_id_ptr = pkt_id_rms_to_glu;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        share_expert_output_gate(
            post_attn_norm_res, w_share_exp_gate, x
        );
        copy_bf16_to_bf16<MODEL_DIM>(
            y+16, x
        );
        _lock_release(w_prod_lock);
        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, 1);
        
        
        
        // doing final residual add
        _lock_acquire(x_cons_lock);
        _lock_acquire(y_prod_lock, 1);
        *pkt_id_ptr = pkt_id_rms_to_it;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        
        // for(int i = 0; i <2048; i++ ){
        //     *(y+16+i) = *(x+i);
        // }
        
        residual_add(y + 16, x_buf, x);
        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
    }
    else {
        // input later norm + residul
        _lock_acquire(x_cons_lock);
        _lock_acquire(w_cons_lock);
        _lock_acquire(ab_out_prod_lock);
        *pkt_id_ptr = pkt_id_rms_to_proj;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        _memcpy<MODEL_DIM * 2>((uint8*)x_buf, (uint8*)x);
        rms_norm(y + 16, y_out, x_buf, w_input_norm);
        _lock_release(x_prod_lock);
    
        _lock_release(y_cons_lock, L_QKV_REPEATS);
        _lock_release(ab_out_cons_lock);


        _lock_acquire(x_cons_lock);

        _lock_acquire(y_prod_lock, L_QKV_REPEATS);
        *pkt_id_ptr = pkt_id_rms_to_proj;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        residual_add(x_buf, x_buf, x);
        // pre_feed_forward
      
        rms_norm(y + 16, post_attn_norm_res, x_buf,w_post_attention_norm );

        _lock_release(x_prod_lock);


       *pkt_id_ptr = pkt_id_rms_to_glu;        
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);

      
        *pkt_id_ptr = pkt_id_rms_to_proj;

        _lock_release(y_cons_lock, UP_GATE_REPEATS);


        // a simple acquire of share expert
        _lock_acquire(x_cons_lock);
        _lock_acquire(y_prod_lock, UP_GATE_REPEATS);
  
        // holder place to prevent kernel stuck
        *pkt_id_ptr = pkt_id_rms_to_glu;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        share_expert_output_gate(
            post_attn_norm_res, w_share_exp_gate, x
        );
        copy_bf16_to_bf16<MODEL_DIM>(
            y+16, x
        );
        _lock_release(w_prod_lock);
        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, 1);
        
        
        
        // doing final residual add
        _lock_acquire(x_cons_lock);
        _lock_acquire(y_prod_lock, 1);
        *pkt_id_ptr = pkt_id_rms_to_it;
        is_x_ping = !is_x_ping;
        x = is_x_ping ? x_ping : x_pong;
        residual_add(y + 16, x_buf, x);
        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock, 1);
        _lock_acquire(y_prod_lock, 1);
    }
    }
}

}