// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "lut_based_ops.h"
#include "nonLinearFunction.h"
#include "zero.h"
// template<int L>
// void pseduo_glu(bf16 * y, const bf16 * x) {
//     bf16* gate_ptr = const_cast<bf16*>(x) + (L / 2);
//     bf16* hid_ptr = const_cast<bf16*>(x);
//     bf16* y_ptr = y;
//     for (int i = 0; i < L / 2; i += 16) {
//         aie::vector<bf16, 16> gate_vec = aie::load_v<16>(gate_ptr + i);
//         aie::vector<bf16, 16> hid_vec = aie::load_v<16>(hid_ptr + i);
//         gate_vec = getActivationBf16(gate_vec);

//         aie::vector<bf16, 16> y_vec = aie::mul(gate_vec, hid_vec);
//         aie::store_v(y_ptr + i, y_vec);
//     }
// }




template<int L>
void pseduo_glu(bf16 * y_ptr, bf16 * gate_ptr, bf16 * hid_ptr) {
    
    constexpr int vec_Size = 32;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(L/vec_Size, L/vec_Size)
    for (int i = 0; i < L ; i += vec_Size) {
        aie::vector<bf16, vec_Size> gate_vec = aie::load_v<vec_Size>(gate_ptr + i);
        aie::vector<bf16, vec_Size> hid_vec = aie::load_v<vec_Size>(hid_ptr + i);
        //gate_vec = getActivationBf16(gate_vec);
        gate_vec = getSiluBf16_nonLUT<vec_Size>(gate_vec);
        aie::vector<bf16, vec_Size> y_vec = aie::mul(gate_vec, hid_vec);
        aie::store_v(y_ptr + i, y_vec);
    }
}

// does the following operation
// output =    vec*scalar + output

template<int L>
void mul_scalar_accum(float*output_ptr,  bf16* in_ptr, float scalar){
    constexpr int vecSize=32;
    static_assert(L%vecSize == 0);

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(L/vecSize, L/vecSize)
    for(int i = 0; i < L; i+=vecSize){
        aie::vector<bf16, vecSize> in_vec = aie::load_v<vecSize>(in_ptr+i);
        
        aie::accum<accfloat, vecSize> in_vec_acc;
        in_vec_acc.from_vector(in_vec);

        aie::vector<float, vecSize> in_vec_mul = aie::mul(
        in_vec_acc.template to_vector<float>(), scalar);


        aie::vector<float, vecSize> acc_vec = aie::load_v<vecSize>(output_ptr+i);

        aie::vector<float, vecSize> res = aie::add(in_vec_mul, acc_vec);

        aie::store_v( output_ptr+i, res );
    }
}



template<int OUT_SIZE>
void send_to_proj_x(
    int out_pkt_id,
    bf16* out_buffer,
    bool *is_to_proj_x_slice_ping_ptr,
    bf16* to_proj_x_slice_With_pkt_ping,
    bf16* to_proj_x_slice_With_pkt_pong,
    int to_proj_x_slice_prod_lock,
    int to_proj_x_slice_cons_lock
){


    uint32_t* to_proj_x_slice_pkt_ping = reinterpret_cast<uint32_t*>(to_proj_x_slice_With_pkt_ping+14);
    uint32_t* to_proj_x_slice_pkt_pong = reinterpret_cast<uint32_t*>(to_proj_x_slice_With_pkt_pong+14);
    bf16* to_proj_x_slice_data_ping = to_proj_x_slice_With_pkt_ping+16;
    bf16* to_proj_x_slice_data_pong = to_proj_x_slice_With_pkt_pong+ 16;



    // setup the pkt_id for this 
    *to_proj_x_slice_pkt_ping = out_pkt_id;
    *to_proj_x_slice_pkt_pong = out_pkt_id;    
    static_assert(OUT_SIZE % PROJ_X_SLICE == 0);
    for(int j = 0; j <OUT_SIZE; j+=PROJ_X_SLICE ){
        _lock_acquire(to_proj_x_slice_prod_lock,1);


        *is_to_proj_x_slice_ping_ptr = !(*is_to_proj_x_slice_ping_ptr);
        if(*is_to_proj_x_slice_ping_ptr){
            copy_bf16_to_bf16<PROJ_X_SLICE>(
                to_proj_x_slice_data_ping, out_buffer+j
            );
        }else{
            copy_bf16_to_bf16<PROJ_X_SLICE>(
                to_proj_x_slice_data_pong, out_buffer+j
            );
        } 

        _lock_release(to_proj_x_slice_cons_lock,1);
    }




}


constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int to_proj_x_slice_prod_lock = 2;
constexpr int to_proj_x_slice_cons_lock = 3;
constexpr int D_out_prod_lock = 4;
constexpr int D_out_con_lock = 5;
constexpr int D_in_prod_lock = 6;
constexpr int D_in_con_lock = 7;

      
constexpr int glu_request_router_send_lock = 12;




// phase 1: qkv proj, D -> (D + D // 2) to RoPE
// phase 2: o_proj, D -> D to rms_residual
// phase 3: up/gate proj, D -> D * 8 to MT_GLU
// phase 4: down proj, D * 4 -> D, to rms_residual
// phase 4 is special, do not clear y to zero
extern "C" {
AIE_PERSIST_ZERO static bool is_glu_slice_in_ping = false;
AIE_PERSIST_ZERO static bool is_to_proj_x_slice_ping = false;


void receive_down(
    size_t &share_expert_result_offset,
    float* share_expert_result,

    float* expert_score, // assume already in the right offset
    bool *is_glu_slice_in_ping_ptr,
    const bf16 * glu_slice_in_buffer_ping,
    const bf16 * glu_slice_in_buffer_pong
){

    static_assert( (NUM_MVM_COLUMN*MODEL_DIM)% GLU_SLICE == 0  );
    static_assert(   NUM_PROJ_OUT_PER_GLU_SLICE *PER_PROJ_OUTPUT_SLICE ==GLU_SLICE);
    // With this assertion, if GLU_SLICE =1024,NUM_MVM_COLUMN=4,  NUM_PROJ_OUT_PER_GLU_SLICE==2

    // It means the receive data is order as
    // expert-0-down[0-127], expert-1-down[0-127], expert-2-down[0-127], expert-3-down[0-127]
    // expert-0-down[128-255], expert-1-down, expert-2-down, expert-3-down

    *(is_glu_slice_in_ping_ptr) = !*(is_glu_slice_in_ping_ptr);
    bf16* x_using = *(is_glu_slice_in_ping_ptr) ? const_cast<bf16*>(glu_slice_in_buffer_ping) : const_cast<bf16*>(glu_slice_in_buffer_pong);
    _lock_acquire(x_cons_lock);  


    for(int exp_idx = 0; exp_idx< NUM_MVM_COLUMN; exp_idx ++){

        for(int l = 0; l < NUM_PROJ_OUT_PER_GLU_SLICE; l++){

            static_assert(PER_PROJ_COLUMN_OUTPUT_SLICE * NUM_MVM_COLUMN == PER_PROJ_OUTPUT_SLICE);
            mul_scalar_accum<PER_PROJ_COLUMN_OUTPUT_SLICE>(
                share_expert_result + share_expert_result_offset + l*PER_PROJ_COLUMN_OUTPUT_SLICE,
                x_using + (exp_idx*PER_PROJ_COLUMN_OUTPUT_SLICE) + l*(PER_PROJ_OUTPUT_SLICE),
               *(expert_score+exp_idx)
            );
            //TODO: FIXME: remove later debug, simply copied 

            // copy_bf16_to_bf16<PER_PROJ_COLUMN_OUTPUT_SLICE>(
            //     share_expert_result + share_expert_result_offset + l*PER_PROJ_COLUMN_OUTPUT_SLICE + exp_idx* MODEL_DIM ,
            //     x_using + (exp_idx*PER_PROJ_COLUMN_OUTPUT_SLICE) + l*(PER_PROJ_OUTPUT_SLICE)
            // );

            // simply copy it to the output


        }
    }
    share_expert_result_offset+= NUM_PROJ_OUT_PER_GLU_SLICE* PER_PROJ_COLUMN_OUTPUT_SLICE;


    _lock_release(x_prod_lock);       

}


void receive_expert_up_gate(
    size_t &exp_glu_res_offset,
    int32_t &slice_idx,
    bf16* exp_glu_res,
    bool *is_glu_slice_in_ping_ptr,
    const bf16 * glu_slice_in_buffer_ping,
    const bf16 * glu_slice_in_buffer_pong

){

    // some assertion
    static_assert( GLU_SLICE ==NUM_MVM_COLUMN * EXP_UP_GATE_OUTDIM_REORDER*2 );
    // With this assertion, if GLU_SLICE =1024,NUM_MVM_COLUMN=4,  EXP_UP_GATE_OUTDIM_REORDER=128

    // It means the receive data is order as
    // expert-0-up[0-127], expert-1-up[0-127], expert-2-up[0-127], expert-3-up[0-127]
    // expert-0-gate[0-127], similar for expert 1, 2, 3

    // NOTE: for exp_glu_res
    // organized as exp0-glu[0-255], exp1-glu[0-255], exp2, exp3,
    // then exp0-glu[256-511], 

    constexpr int X_SLICE_DIV_EXP_UP_GATE_OUTDIM_REORDER = PROJ_X_SLICE/ EXP_UP_GATE_OUTDIM_REORDER;

    *(is_glu_slice_in_ping_ptr) = !*(is_glu_slice_in_ping_ptr);
    const bf16* x_using = *(is_glu_slice_in_ping_ptr) ? glu_slice_in_buffer_ping : glu_slice_in_buffer_pong;
    _lock_acquire(x_cons_lock);      
    for(int exp_idx = 0; exp_idx < NUM_MVM_COLUMN; exp_idx++){
        bf16* exp_up = const_cast<bf16*>(x_using+ exp_idx* EXP_UP_GATE_OUTDIM_REORDER);
        bf16* exp_gate =exp_up +  NUM_MVM_COLUMN*EXP_UP_GATE_OUTDIM_REORDER;

        pseduo_glu<EXP_UP_GATE_OUTDIM_REORDER>(
            exp_glu_res + exp_glu_res_offset + exp_idx*PROJ_X_SLICE ,
            exp_gate, exp_up  
        );
     
    }

    slice_idx = (slice_idx + 1) % X_SLICE_DIV_EXP_UP_GATE_OUTDIM_REORDER;
    exp_glu_res_offset = slice_idx == 0
        ? exp_glu_res_offset + (NUM_MVM_COLUMN * PROJ_X_SLICE - EXP_UP_GATE_OUTDIM_REORDER)
        : exp_glu_res_offset + EXP_UP_GATE_OUTDIM_REORDER;


    _lock_release(x_prod_lock);   
     


}


void process_glu(
    bf16* y_it,    
    const bf16* glu_slice_in_buffer_ping,
    const bf16* glu_slice_in_buffer_pong

){

    
    static_assert((2 * INTERMEDIATE_SIZE) % GLU_SLICE == 0, "GLU_SLICE must divide 2 * INTERMEDIATE_SIZE");
    constexpr int chunks = 2 * INTERMEDIATE_SIZE / GLU_SLICE; // number of chunks in one GLU, each chunk is GLU_SLICE in size

    for (int i = 0; i < chunks; i++) {
        is_glu_slice_in_ping = !is_glu_slice_in_ping;
        const bf16* x_using = is_glu_slice_in_ping ? glu_slice_in_buffer_ping : glu_slice_in_buffer_pong;
        
        bf16* gate_ptr = const_cast<bf16*>(x_using+UP_GATE_OUTDIM_REORDEER);
        bf16* up_ptr = const_cast<bf16*>(x_using);

        _lock_acquire(x_cons_lock);
        pseduo_glu<UP_GATE_OUTDIM_REORDEER>(y_it, gate_ptr,up_ptr );
        static_assert(UP_GATE_OUTDIM_REORDEER == GLU_SLICE/2);
        y_it += UP_GATE_OUTDIM_REORDEER;
        _lock_release(x_prod_lock);
    }
        
}


void glu(
    float* topk_router_score,
    int32_t* top_k_expert_index_buffer,
    bf16 * intermediate_buffer,
    bf16* D_mlp_final_to_rms,
    bf16* D_in,
    float* shared_expert_output,
    const bf16 * glu_slice_in_buffer_ping,
    const bf16 * glu_slice_in_buffer_pong,
    bf16* to_proj_x_slice_With_pkt_ping,
    bf16* to_proj_x_slice_With_pkt_pong
)
{

    // buffer to save the intermediate result
    alignas(aie::vector_decl_align) bfloat16 exp_glu_res[INTERMEDIATE_SIZE * NUM_MVM_COLUMN];
   
    //zero out shared_expert_output
    zero_vectorized_512<float, MODEL_DIM, 1>(
        shared_expert_output
    );
    process_glu(
        intermediate_buffer,
        glu_slice_in_buffer_ping, glu_slice_in_buffer_pong
    );


    {

        size_t share_expert_output_in_offset = 0;

        for(int i = 0; i <O_DOWN_REPEATS; i++ ){

            // Each i iteration produce a complete

            send_to_proj_x<INTERMEDIATE_SIZE>(
                aux_to_proj_x_in_pkt_id_glu_to_aux_x_in,
                intermediate_buffer,
                &is_to_proj_x_slice_ping,
                to_proj_x_slice_With_pkt_ping,to_proj_x_slice_With_pkt_pong,
                to_proj_x_slice_prod_lock, to_proj_x_slice_cons_lock
            );
            
            static_assert(MODEL_DIM% GLU_SLICE == 0);
            static_assert(O_DOWN_REPEATS % NUM_PROJ_OUT_PER_GLU_SLICE ==0);
            // if( (i+1)%NUM_PROJ_OUT_PER_GLU_SLICE == 0 ){

            //     is_glu_slice_in_ping = !is_glu_slice_in_ping;
            //     const bf16* x_using = is_glu_slice_in_ping ? glu_slice_in_buffer_ping : glu_slice_in_buffer_pong;
            //     _lock_acquire(x_cons_lock);

            //     copy_bf16_to_float<GLU_SLICE>(
            //         shared_expert_output + share_expert_output_in_offset, x_using
            //     );
            //     _lock_release(x_prod_lock);
            //     share_expert_output_in_offset+= GLU_SLICE;
            // }


        }
            
    }   




    // at this point ,trigger the router to send the x again(before the  )
    _lock_acquire(D_in_con_lock,1);
    // within this scope, have access to D_in
    //D_in contains the x input to MLP layer

    {
        // Release shared_expert_output buffer for router ct to do share_expert_gate

        // NOTE: for now, the score and topk is already done
    
        {
            //TODO:FIXME: for now, only 1 round, later NUM_ACT_EXPERTS/NUM_MVM_COLUMN
            for(int round = 0; round <  NUM_ACT_EXPERTS/NUM_MVM_COLUMN; round++){ //TODO:don't hardcode this


                _lock_release(glu_request_router_send_lock );

                // First, send D to repeat for up-gate
                //NOTELFIXME:TODO: request send NUM_MVM_COLUMN expert 
                {   
                    size_t exp_glu_res_offset = 0;
                    int32_t slice_idx = 0;
                    for(int i =0; i <UP_GATE_EXPERT_REPEATS; i++){
                        send_to_proj_x<MODEL_DIM>(
                        aux_to_proj_x_in_pkt_id_glu_to_aux_x_in,
                        D_in,
                        &is_to_proj_x_slice_ping,
                        to_proj_x_slice_With_pkt_ping,to_proj_x_slice_With_pkt_pong,
                        to_proj_x_slice_prod_lock, to_proj_x_slice_cons_lock
                        );


                        static_assert( (NUM_MVM_COLUMN* INTERMEDIATE_SIZE) %GLU_SLICE == 0 );
                        static_assert( UP_GATE_EXPERT_REPEATS % NUM_PROJ_OUT_PER_GLU_SLICE == 0);
                        if((i+1)%NUM_PROJ_OUT_PER_GLU_SLICE == 0){
                        

                            receive_expert_up_gate(
                                exp_glu_res_offset,
                                slice_idx,
                                exp_glu_res,
                                &is_glu_slice_in_ping,
                                glu_slice_in_buffer_ping, glu_slice_in_buffer_pong
                            );

                        }
                    }
                    // sanity checks
            
 
                }

                // NOTE: TODO: move to receive gate up later
                // do a reorder
                // current data is 
                // exp0-glu[0:127] exp1-glu[0:127] exp2-glu[0:127] exp3-glu[0:127],
                // exp0-glu[128:255]...... 
                // for debug: COPY to debug_buffer
                // bf16* debug_ptr_offset =  debug_buffer + round* INTERMEDIATE_SIZE * NUM_MVM_COLUMN;


                // reorder to exp0-glu[0:255],  exp1-glu[0:255],  exp2-glu[0:255],  exp3-glu[0:255]
                // Then follow by exp0-glu[256:512]....


                // for(int j = 0; j < INTERMEDIATE_SIZE * NUM_MVM_COLUMN; j++){
                //     // for simplicyt, copyied into the debug_ptr_offset for now, 
                //     // do the reorder on cpu 
                //     debug_ptr_offset[j] = exp_glu_res[j];
                // }   

                // DEBUG:TODO: REMOVE ME LATER FIXME:

                // reorder to exp0-glu[0:255],  exp1-glu[0:255],  exp2-glu[0:255],  exp3-glu[0:255]
                // Then follow by exp0-glu[256:512]....
                // for(int k = 0; k < 256; k++){
                //     exp_glu_res[k+256] = exp_glu_res[k];
                //     exp_glu_res[k+256*2] = exp_glu_res[k];
                //     exp_glu_res[k+256*3] = exp_glu_res[k];
                // }


                // for(int k = 0; k < 256; k++){
                //     exp_glu_res[k+256 + 256*4] = exp_glu_res[k + 256*4];
                //     exp_glu_res[k+256*2+ 256*4] = exp_glu_res[k + 256*4];
                //     exp_glu_res[k+256*3+ 256*4] = exp_glu_res[k + 256*4];
                // }


        
                // now, the intermediate is ready, attempt to send it out to down projection
                {
                    size_t share_expert_result_offset = 0;
                    for(int i =0; i < DOWN_EXPERT_REPEATS; i++){

                        send_to_proj_x<NUM_MVM_COLUMN*INTERMEDIATE_SIZE>(
                            x_repeat_to_aux_x_proj_id,
                            exp_glu_res, 
                            &is_to_proj_x_slice_ping,
                            to_proj_x_slice_With_pkt_ping,to_proj_x_slice_With_pkt_pong,
                            to_proj_x_slice_prod_lock, to_proj_x_slice_cons_lock

                        );

                        
                        static_assert( (NUM_MVM_COLUMN*MODEL_DIM)%GLU_SLICE == 0);
                        static_assert( DOWN_EXPERT_REPEATS %NUM_PROJ_OUT_PER_GLU_SLICE == 0  );
                        if((i +1)%NUM_PROJ_OUT_PER_GLU_SLICE ==0  ){
                            // some debug info
                            // if(round == 0 ){
                            //     receive_down(
                            //         share_expert_result_offset,
                            //         debug_buffer,
                            //         topk_router_score+ round*NUM_MVM_COLUMN,
                            //         &is_glu_slice_in_ping,
                            //         glu_slice_in_buffer_ping, glu_slice_in_buffer_pong                                
                            //         );
                            // }else{
                            //     receive_down(
                            //     share_expert_result_offset,
                            //     shared_expert_output,
                            //     topk_router_score+ round*NUM_MVM_COLUMN,
                            //     &is_glu_slice_in_ping,
                            //     glu_slice_in_buffer_ping, glu_slice_in_buffer_pong                                
                            //     );
                            // }
                            receive_down(
                                share_expert_result_offset,
                                shared_expert_output,
                                topk_router_score+ round*NUM_MVM_COLUMN,
                                &is_glu_slice_in_ping,
                                glu_slice_in_buffer_ping, glu_slice_in_buffer_pong                                
                            );
                        }

                    }

                }

            }



        }


    }



    _lock_release(D_in_prod_lock,1);

    _lock_acquire(D_in_con_lock,1);
    // TODO: add it up 
    //TODO: optimize it laer
    // for(int i = 0; i < MODEL_DIM; i++){
    //     shared_expert_output[i] += D_in[i];
    // }
    constexpr int vec_size = 32;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(MODEL_DIM/vec_size,MODEL_DIM/vec_size)
    for(int i = 0; i < MODEL_DIM; i+=vec_size ){
        aie::vector<bf16, vec_size> share_gate_sigmoid = aie::load_v<vec_size>(D_in + i);
        aie::vector<float, vec_size> shared_expert_vec = aie::load_v<vec_size>(shared_expert_output + i);

        aie::accum<accfloat, vec_size> temp_acc;
        temp_acc.from_vector( shared_expert_vec);

        aie::vector<bf16, vec_size> result = aie::add( share_gate_sigmoid, temp_acc.template to_vector<bf16>() );
        aie::store_v(D_mlp_final_to_rms+i,result );

    }

    _lock_release(D_in_prod_lock,1);

    //TODO:FIXME: uncomment this after debug
    // copy_float_to_bf16<MODEL_DIM>(
    //     D_mlp_final_to_rms, shared_expert_output
    // );
    _lock_release(D_out_con_lock, 1);
    // now, send the final stuff
        
    _lock_acquire(D_out_prod_lock, 1);


}


}