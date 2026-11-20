// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "lut_based_ops.h"
#include "nonLinearFunction.h"
#include "zero.h"

// y = silu(gate) * up
template<int L>
void pseduo_glu(bf16 * y_ptr, bf16 * gate_ptr, bf16 * hid_ptr) {

    constexpr int vec_Size = 32;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(L/vec_Size, L/vec_Size)
    for (int i = 0; i < L ; i += vec_Size) {
        aie::vector<bf16, vec_Size> gate_vec = aie::load_v<vec_Size>(gate_ptr + i);
        aie::vector<bf16, vec_Size> hid_vec = aie::load_v<vec_Size>(hid_ptr + i);
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
    for(AIE_LOCK_LOOP_CTR j = 0; j <OUT_SIZE; j+=PROJ_X_SLICE ){
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
    // One slice is NUM_PROJ_OUT_PER_GLU_SLICE proj rounds (1 with GLU_SLICE = 512),
    // each ordered by column, i.e. by expert:
    // expert-0-down[0-127], expert-1-down[0-127], expert-2-down[0-127], expert-3-down[0-127]
    // then the next round's expert-0-down[128-255], expert-1-down, ...

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
        }
    }
    share_expert_result_offset+= NUM_PROJ_OUT_PER_GLU_SLICE* PER_PROJ_COLUMN_OUTPUT_SLICE;

    _lock_release(x_prod_lock);
}


void receive_expert_up_gate(
    int slice,          // which GLU slice of this round, 0 .. 2 * INTERMEDIATE_SIZE / PER_PROJ_COLUMN_OUTPUT_SLICE - 1
    bf16* exp_glu_res,
    bool *is_glu_slice_in_ping_ptr,
    const bf16 * glu_slice_in_buffer_ping,
    const bf16 * glu_slice_in_buffer_pong

){
    constexpr int E = EXP_UP_GATE_OUTDIM_REORDER;   // h values per expert per slice
    static_assert( GLU_SLICE == NUM_MVM_COLUMN * E * 2 );

    // In: one slice is one proj round, column c = expert c, each column's part is
    //   [up E | gate E]   (the host orders the expert weight rows this way)
    //
    // Out: exp_glu_res is the x of the routed down step, which the proj engine
    // reads in PROJ_X_SLICE pieces, round robin over the 4 experts:
    //   exp0 h[0:256], exp1 h[0:256], exp2, exp3, exp0 h[256:512], ...
    // Slice s gives h[s*E, s*E + E) of every expert; find where that lands.
    const int h_pos    = slice * E;                     // position inside one expert's h
    const int chunk    = h_pos / PROJ_X_SLICE;          // which x piece of that expert
    const int in_chunk = h_pos % PROJ_X_SLICE;          // where inside the piece

    *(is_glu_slice_in_ping_ptr) = !*(is_glu_slice_in_ping_ptr);
    const bf16* x_using = *(is_glu_slice_in_ping_ptr) ? glu_slice_in_buffer_ping : glu_slice_in_buffer_pong;
    _lock_acquire(x_cons_lock);
    for(int exp_idx = 0; exp_idx < NUM_MVM_COLUMN; exp_idx++){
        bf16* exp_up   = const_cast<bf16*>(x_using + exp_idx * 2 * E);
        bf16* exp_gate = exp_up + E;
        bf16* exp_h    = exp_glu_res + chunk * NUM_MVM_COLUMN * PROJ_X_SLICE + exp_idx * PROJ_X_SLICE + in_chunk;

        pseduo_glu<E>(exp_h, exp_gate, exp_up);
    }
    _lock_release(x_prod_lock);
}


void process_glu(
    bf16* y_it,    
    const bf16* glu_slice_in_buffer_ping,
    const bf16* glu_slice_in_buffer_pong

){

    
    static_assert((2 * INTERMEDIATE_SIZE) % GLU_SLICE == 0, "GLU_SLICE must divide 2 * INTERMEDIATE_SIZE");
    constexpr int chunks = 2 * INTERMEDIATE_SIZE / GLU_SLICE; // number of chunks in one GLU, each chunk is GLU_SLICE in size

    for (AIE_LOCK_LOOP_CTR i = 0; i < chunks; i++) {
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
    // NOTE: despite its name, shared_expert_output is the fp32 sum of the
    // *routed* experts. The shared expert's result comes back from rms into
    // D_in (the second D_in fill below).

    // buffer to save the intermediate result
    alignas(aie::vector_decl_align) bfloat16 exp_glu_res[INTERMEDIATE_SIZE * NUM_MVM_COLUMN];

    zero_vectorized_512<float, MODEL_DIM, 1>(
        shared_expert_output
    );

    // shared expert: silu(gate) * up, then send it to the proj engine for down
    process_glu(
        intermediate_buffer,
        glu_slice_in_buffer_ping, glu_slice_in_buffer_pong
    );

    for(int i = 0; i <O_DOWN_REPEATS; i++ ){
        send_to_proj_x<INTERMEDIATE_SIZE>(
            pkt_id_glu_to_proj_x,
            intermediate_buffer,
            &is_to_proj_x_slice_ping,
            to_proj_x_slice_With_pkt_ping,to_proj_x_slice_With_pkt_pong,
            to_proj_x_slice_prod_lock, to_proj_x_slice_cons_lock
        );
        static_assert(MODEL_DIM% GLU_SLICE == 0);
        static_assert(O_DOWN_REPEATS % NUM_PROJ_OUT_PER_GLU_SLICE ==0);
    }

    // First D_in fill: the post_attention_layernorm x, the MoE input.
    _lock_acquire(D_in_con_lock,1);

    // Routed experts: each round runs NUM_MVM_COLUMN experts, one per proj column.
    for(AIE_LOCK_LOOP_CTR round = 0; round <  NUM_ACT_EXPERTS/NUM_MVM_COLUMN; round++){

        // let the router point the shim weight BDs at this round's experts
        _lock_release(glu_request_router_send_lock );

        // gate/up: send x once per row-block round, collect silu(gate) * up
        {
            for(int i =0; i <UP_GATE_EXPERT_REPEATS; i++){
                send_to_proj_x<MODEL_DIM>(
                    pkt_id_glu_to_proj_x,
                    D_in,
                    &is_to_proj_x_slice_ping,
                    to_proj_x_slice_With_pkt_ping,to_proj_x_slice_With_pkt_pong,
                    to_proj_x_slice_prod_lock, to_proj_x_slice_cons_lock
                );

                static_assert( (NUM_MVM_COLUMN* INTERMEDIATE_SIZE) %GLU_SLICE == 0 );
                static_assert( UP_GATE_EXPERT_REPEATS % NUM_PROJ_OUT_PER_GLU_SLICE == 0);
                if((i+1)%NUM_PROJ_OUT_PER_GLU_SLICE == 0){
                    receive_expert_up_gate(
                        i / NUM_PROJ_OUT_PER_GLU_SLICE,
                        exp_glu_res,
                        &is_glu_slice_in_ping,
                        glu_slice_in_buffer_ping, glu_slice_in_buffer_pong
                    );
                }
            }
        }

        // down: exp_glu_res is laid out as
        //   exp0-glu[0:255], exp1-glu[0:255], exp2-glu[0:255], exp3-glu[0:255], exp0-glu[256:511], ...
        // add score * down of each expert into shared_expert_output
        {
            size_t share_expert_result_offset = 0;
            for(int i =0; i < DOWN_EXPERT_REPEATS; i++){

                send_to_proj_x<NUM_MVM_COLUMN*INTERMEDIATE_SIZE>(
                    pkt_id_glu_to_proj_x,
                    exp_glu_res,
                    &is_to_proj_x_slice_ping,
                    to_proj_x_slice_With_pkt_ping,to_proj_x_slice_With_pkt_pong,
                    to_proj_x_slice_prod_lock, to_proj_x_slice_cons_lock
                );

                static_assert( (NUM_MVM_COLUMN*MODEL_DIM)%GLU_SLICE == 0);
                static_assert( DOWN_EXPERT_REPEATS %NUM_PROJ_OUT_PER_GLU_SLICE == 0  );
                if((i +1)%NUM_PROJ_OUT_PER_GLU_SLICE ==0  ){
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

    _lock_release(D_in_prod_lock,1);

    // Second D_in fill: the shared expert's down output. MoE out = shared + routed.
    _lock_acquire(D_in_con_lock,1);
    constexpr int vec_size = 32;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(MODEL_DIM/vec_size,MODEL_DIM/vec_size)
    for(int i = 0; i < MODEL_DIM; i+=vec_size ){
        aie::vector<bf16, vec_size> shared_vec = aie::load_v<vec_size>(D_in + i);
        aie::vector<float, vec_size> routed_vec = aie::load_v<vec_size>(shared_expert_output + i);

        aie::accum<accfloat, vec_size> temp_acc;
        temp_acc.from_vector( routed_vec);

        aie::vector<bf16, vec_size> result = aie::add( shared_vec, temp_acc.template to_vector<bf16>() );
        aie::store_v(D_mlp_final_to_rms+i,result );
    }
    _lock_release(D_in_prod_lock,1);

    // send the MoE output to rms
    _lock_release(D_out_con_lock, 1);
    _lock_acquire(D_out_prod_lock, 1);
}


}
