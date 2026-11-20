// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "common.h"
#include "aie_kernel_utils.h"
#include "flm_rsqrt.h"


#define GEMMA4_E2B       0
#define GEMMA4_E4B       1

#ifndef MODEL_TYPE
#define MODEL_TYPE GEMMA4_E2B
#endif


constexpr int VOCAB_SIZE = 262144;
constexpr int NUM_CTS    = 32;

#if MODEL_TYPE == GEMMA4_E2B
constexpr int MODEL_DIM  = 1536;
#elif MODEL_TYPE == GEMMA4_E4B
constexpr int MODEL_DIM  = 2560;
#else
#error "Unsupported MODEL_TYPE"
#endif

constexpr int   Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int   Q4NX_COL_BLOCK_SIZE = 256;
constexpr int   Q4NX_GROUP_SIZE     = 32;
constexpr int   Q4NX_ROW_STRIDE = 16;
constexpr int   Q4NX_ROW_STRIDE_SIZE = Q4NX_ROW_STRIDE; // an alias
constexpr int   GGML_GROUP_SIZE     = 32;

constexpr int M_PER_ROUND = NUM_CTS * Q4NX_ROW_BLOCK_SIZE;
constexpr int VOCAB_SIZE_PADDED = ((VOCAB_SIZE + M_PER_ROUND - 1) / M_PER_ROUND) * M_PER_ROUND;
constexpr int MODEL_DIM_PADDED  = ((MODEL_DIM + Q4NX_COL_BLOCK_SIZE - 1) / Q4NX_COL_BLOCK_SIZE) * Q4NX_COL_BLOCK_SIZE;


constexpr int ROUNDS   = VOCAB_SIZE_PADDED / M_PER_ROUND;
constexpr int K_BLOCKS = MODEL_DIM_PADDED  / Q4NX_COL_BLOCK_SIZE;
void rms_norm(
    bf16 *restrict y,
    bf16 *restrict x,
    bf16 *restrict w
){
    bf16 *it_y = const_cast<bf16*>(y);
    bf16 *it_x = const_cast<bf16*>(x);
    bf16 *it_w = const_cast<bf16*>(w);
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)MODEL_DIM;

    aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = MODEL_DIM / vector_size;
    for (int i = 0; i < F; i++){
        auto x_vec = aie::load_v<vector_size>(it_x);
        sum_squares = aie::mac_square(sum_squares, x_vec);
        it_x += vector_size;
    }

    it_x = const_cast<bf16*>(x);

    mx_vec = sum_squares.template to_vector<float>();
    float sum = aie::reduce_add(mx_vec);
    sum = sum * one_over_D;
    sum = sum + epsilon;
    const float divrms = flm_rsqrt(sum);

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
    event1();
}

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;


///@brief Q4K block
///@param scales: scales and mins, bf16
///@param mins: scales and mins, bf16
///@param qs: 4-bit quantized values
typedef struct {
    bf16 scales[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE]; // per-group scales and mins
    bf16 mins[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE];   // per-group scales and mins
    uint4 qs[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE];                      // 4-bit quantized values
} q4k_block_t;

template<int m>
void copy_y(bf16 * y, const float * y_acc) {
    for (int i = 0; i < m / 16; i++){
        aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + i * 16);
        aie::accum<accfloat, 16> acc;
        acc.from_vector(y_vec);
        aie::store_v(y + i * 16, acc.template to_vector<bf16>());
    }
}

template <int M, int N>
 void _qmm_q4k_bf16(
    const q4k_block_t * A,
    const bf16 * B,
    float * c,
    bf16* b_col_reduce_buffer
) { 

    constexpr int pr = 16;
    constexpr int pc = 8;
    const q4k_block_t * A_q4k = (q4k_block_t *)A;
    const uint4* qs_ptr = A_q4k->qs;
    // precompute chunk sum of B;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(M/pr, M/pr)
    for (int row = 0; row < M; row += pr){
        aie::accum<accfloat, pr> c_accum;
        c_accum.from_vector(aie::load_v<pr>(c + row));
        const bf16* it_B =B;

        uint32_t scale_min_offset = row;
        bf16* cur_b_reduce_Buf = b_col_reduce_buffer;

        
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE( N / 32, N / 32 )
        for (int sub_chunk = 0; sub_chunk < N / 32; sub_chunk++)AIE_LOOP_FLATTEN  {

            aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);
            it_B += 32;

            
            aie::vector<bf16, pr> a_scales = aie::load_v<pr>(A_q4k->scales +scale_min_offset);
            aie::vector<bf16, pr> a_mins = aie::load_v<pr>(A_q4k->mins + scale_min_offset);

            scale_min_offset+=M;
            bf16 reduce_b = *(cur_b_reduce_Buf);
            cur_b_reduce_Buf++;

            aie::accum<accfloat, pr> temp_acc;  
                   
            {

                aie::accum<accfloat, pr * 8> a_cc_f32_accum_0;
                aie::accum<accfloat, pr * 8> a_cc_f32_accum_1;
                aie::accum<accfloat, pr * 8> a_cc_f32_accum_2;  
                aie::accum<accfloat, pr * 8> a_cc_f32_accum_3;   
                // Load all a vectors
                aie::vector<uint4, pr * 8> a_cc_0 = aie::load_v<pr * 8>(qs_ptr);
                qs_ptr += pr * 4;
                aie::vector<float, pr * 8> a_cc_f32_0 = aie::to_float(a_cc_0, 0);     

                a_cc_f32_accum_0.from_vector(a_cc_f32_0);
                aie::vector<bf16, pr * 8> a_cc_bf16_0 = a_cc_f32_accum_0.template to_vector<bf16>();

                aie::vector<uint4, pr * 8> a_cc_1 = aie::load_v<pr * 8>(qs_ptr);
                qs_ptr += pr * 4;
                aie::vector<float, pr * 8> a_cc_f32_1 = aie::to_float(a_cc_1, 0);

                a_cc_f32_accum_1.from_vector(a_cc_f32_1);
                aie::vector<bf16, pr * 8> a_cc_bf16_1 = a_cc_f32_accum_1.template to_vector<bf16>();                

                aie::vector<uint4, pr * 8> a_cc_2 = aie::load_v<pr * 8>(qs_ptr);
                qs_ptr += pr * 4;
                aie::vector<float, pr * 8> a_cc_f32_2 = aie::to_float(a_cc_2, 0);

                a_cc_f32_accum_2.from_vector(a_cc_f32_2);
                aie::vector<bf16, pr * 8> a_cc_bf16_2 = a_cc_f32_accum_2.template to_vector<bf16>();


                aie::vector<uint4, pr * 8> a_cc_3 = aie::load_v<pr * 8>(qs_ptr);
                qs_ptr += pr * 4;
                aie::vector<float, pr * 8> a_cc_f32_3 = aie::to_float(a_cc_3, 0);

                a_cc_f32_accum_3.from_vector(a_cc_f32_3);
                aie::vector<bf16, pr * 8> a_cc_bf16_3 = a_cc_f32_accum_3.template to_vector<bf16>();


  

                // Extract all columns
                aie::vector<bf16, pr> a_col_0_0 = a_cc_bf16_0.extract<pr>(0);
                aie::vector<bf16, pr> a_col_1_0 = a_cc_bf16_0.extract<pr>(1);
                aie::vector<bf16, pr> a_col_2_0 = a_cc_bf16_0.extract<pr>(2);
                aie::vector<bf16, pr> a_col_3_0 = a_cc_bf16_0.extract<pr>(3);
                aie::vector<bf16, pr> a_col_4_0 = a_cc_bf16_0.extract<pr>(4);
                aie::vector<bf16, pr> a_col_5_0 = a_cc_bf16_0.extract<pr>(5);
                aie::vector<bf16, pr> a_col_6_0 = a_cc_bf16_0.extract<pr>(6);
                aie::vector<bf16, pr> a_col_7_0 = a_cc_bf16_0.extract<pr>(7);

                aie::vector<bf16, pr> a_col_0_1 = a_cc_bf16_1.extract<pr>(0);
                aie::vector<bf16, pr> a_col_1_1 = a_cc_bf16_1.extract<pr>(1);
                aie::vector<bf16, pr> a_col_2_1 = a_cc_bf16_1.extract<pr>(2);
                aie::vector<bf16, pr> a_col_3_1 = a_cc_bf16_1.extract<pr>(3);
                aie::vector<bf16, pr> a_col_4_1 = a_cc_bf16_1.extract<pr>(4);
                aie::vector<bf16, pr> a_col_5_1 = a_cc_bf16_1.extract<pr>(5);
                aie::vector<bf16, pr> a_col_6_1 = a_cc_bf16_1.extract<pr>(6);
                aie::vector<bf16, pr> a_col_7_1 = a_cc_bf16_1.extract<pr>(7);

                aie::vector<bf16, pr> a_col_0_2 = a_cc_bf16_2.extract<pr>(0);
                aie::vector<bf16, pr> a_col_1_2 = a_cc_bf16_2.extract<pr>(1);
                aie::vector<bf16, pr> a_col_2_2 = a_cc_bf16_2.extract<pr>(2);
                aie::vector<bf16, pr> a_col_3_2 = a_cc_bf16_2.extract<pr>(3);
                aie::vector<bf16, pr> a_col_4_2 = a_cc_bf16_2.extract<pr>(4);
                aie::vector<bf16, pr> a_col_5_2 = a_cc_bf16_2.extract<pr>(5);
                aie::vector<bf16, pr> a_col_6_2 = a_cc_bf16_2.extract<pr>(6);
                aie::vector<bf16, pr> a_col_7_2 = a_cc_bf16_2.extract<pr>(7);

                aie::vector<bf16, pr> a_col_0_3 = a_cc_bf16_3.extract<pr>(0);
                aie::vector<bf16, pr> a_col_1_3 = a_cc_bf16_3.extract<pr>(1);
                aie::vector<bf16, pr> a_col_2_3 = a_cc_bf16_3.extract<pr>(2);
                aie::vector<bf16, pr> a_col_3_3 = a_cc_bf16_3.extract<pr>(3);
                aie::vector<bf16, pr> a_col_4_3 = a_cc_bf16_3.extract<pr>(4);
                aie::vector<bf16, pr> a_col_5_3 = a_cc_bf16_3.extract<pr>(5);
                aie::vector<bf16, pr> a_col_6_3 = a_cc_bf16_3.extract<pr>(6);
                aie::vector<bf16, pr> a_col_7_3 = a_cc_bf16_3.extract<pr>(7);

                // Load all b scalars
                bfloat16 scalar_0_0 = b_col.get(0);
                bfloat16 scalar_1_0 = b_col.get(1);
                bfloat16 scalar_2_0 = b_col.get(2);
                bfloat16 scalar_3_0 = b_col.get(3);
                bfloat16 scalar_4_0 = b_col.get(4);
                bfloat16 scalar_5_0 = b_col.get(5);
                bfloat16 scalar_6_0 = b_col.get(6);
                bfloat16 scalar_7_0 = b_col.get(7);

                bfloat16 scalar_0_1 = b_col.get(8);
                bfloat16 scalar_1_1 = b_col.get(9);
                bfloat16 scalar_2_1 = b_col.get(10);
                bfloat16 scalar_3_1 = b_col.get(11);
                bfloat16 scalar_4_1 = b_col.get(12);
                bfloat16 scalar_5_1 = b_col.get(13);
                bfloat16 scalar_6_1 = b_col.get(14);
                bfloat16 scalar_7_1 = b_col.get(15);

                bfloat16 scalar_0_2 = b_col.get(16);
                bfloat16 scalar_1_2 = b_col.get(17);
                bfloat16 scalar_2_2 = b_col.get(18);
                bfloat16 scalar_3_2 = b_col.get(19);
                bfloat16 scalar_4_2 = b_col.get(20);
                bfloat16 scalar_5_2 = b_col.get(21);
                bfloat16 scalar_6_2 = b_col.get(22);
                bfloat16 scalar_7_2 = b_col.get(23);

                bfloat16 scalar_0_3 = b_col.get(24);
                bfloat16 scalar_1_3 = b_col.get(25);
                bfloat16 scalar_2_3 = b_col.get(26);
                bfloat16 scalar_3_3 = b_col.get(27);
                bfloat16 scalar_4_3 = b_col.get(28);
                bfloat16 scalar_5_3 = b_col.get(29);
                bfloat16 scalar_6_3 = b_col.get(30);
                bfloat16 scalar_7_3 = b_col.get(31);

                // All MAC operations
                temp_acc = aie::mul(a_col_0_0, scalar_0_0);
                temp_acc = aie::mac(temp_acc, a_col_1_0, scalar_1_0);
                temp_acc = aie::mac(temp_acc, a_col_2_0, scalar_2_0);
                temp_acc = aie::mac(temp_acc, a_col_3_0, scalar_3_0);
                temp_acc = aie::mac(temp_acc, a_col_4_0, scalar_4_0);
                temp_acc = aie::mac(temp_acc, a_col_5_0, scalar_5_0);
                temp_acc = aie::mac(temp_acc, a_col_6_0, scalar_6_0);
                temp_acc = aie::mac(temp_acc, a_col_7_0, scalar_7_0);
                temp_acc = aie::mac(temp_acc, a_col_0_1, scalar_0_1);
                temp_acc = aie::mac(temp_acc, a_col_1_1, scalar_1_1);
                temp_acc = aie::mac(temp_acc, a_col_2_1, scalar_2_1);
                temp_acc = aie::mac(temp_acc, a_col_3_1, scalar_3_1);
                temp_acc = aie::mac(temp_acc, a_col_4_1, scalar_4_1);
                temp_acc = aie::mac(temp_acc, a_col_5_1, scalar_5_1);
                temp_acc = aie::mac(temp_acc, a_col_6_1, scalar_6_1);
                temp_acc = aie::mac(temp_acc, a_col_7_1, scalar_7_1);
                temp_acc = aie::mac(temp_acc, a_col_0_2, scalar_0_2);
                temp_acc = aie::mac(temp_acc, a_col_1_2, scalar_1_2);
                temp_acc = aie::mac(temp_acc, a_col_2_2, scalar_2_2);
                temp_acc = aie::mac(temp_acc, a_col_3_2, scalar_3_2);
                temp_acc = aie::mac(temp_acc, a_col_4_2, scalar_4_2);
                temp_acc = aie::mac(temp_acc, a_col_5_2, scalar_5_2);
                temp_acc = aie::mac(temp_acc, a_col_6_2, scalar_6_2);
                temp_acc = aie::mac(temp_acc, a_col_7_2, scalar_7_2);
                temp_acc = aie::mac(temp_acc, a_col_0_3, scalar_0_3);
                temp_acc = aie::mac(temp_acc, a_col_1_3, scalar_1_3);
                temp_acc = aie::mac(temp_acc, a_col_2_3, scalar_2_3);
                temp_acc = aie::mac(temp_acc, a_col_3_3, scalar_3_3);
                temp_acc = aie::mac(temp_acc, a_col_4_3, scalar_4_3);
                temp_acc = aie::mac(temp_acc, a_col_5_3, scalar_5_3);
                temp_acc = aie::mac(temp_acc, a_col_6_3, scalar_6_3);
                temp_acc = aie::mac(temp_acc, a_col_7_3, scalar_7_3);

            } // end of cc, 32 columns (manually unrolled);

            // c_accum = aie::mac(c_accum, c_local_accum.template to_vector<bf16>(), a_scales);
            // c_accum = aie::mac(c_accum, a_mins, (bf16)reduce_b);
            c_accum = aie::mac(c_accum, temp_acc.template to_vector<bf16>(), a_scales);
            c_accum = aie::mac(c_accum, a_mins, reduce_b);
            // 
        } // end of sub_chunk, 256 columns;
        aie::vector<float, pr> c_accum_f32 = c_accum.template to_vector<float>();
        aie::store_v(c + row, c_accum_f32);
    }
}


extern "C" {
void qmm(
    bf16* y_ping, bf16* y_pong,
    q4k_block_t* w_ping, q4k_block_t* w_pong,
    bf16* x,
    float* final_scale
){
    // Force into .data (file-backed) rather than .bss: under Peano (llvm-aie)
    // the aie-rt CDO loader does not reliably zero a .bss tail (FileSiz<MemSiz
    // PT_LOAD), leaving these persistent ping-pong toggles as garbage on load.
    // Placing them in .data keeps the initial values file-backed. (chess is fine.)
    AIE_PERSIST_ZERO static bool is_y_ping = false;
    AIE_PERSIST_ZERO static bool is_w_ping = false;

    alignas(aie::vector_decl_align) float  y_acc[Q4NX_ROW_BLOCK_SIZE];
    alignas(aie::vector_decl_align) bfloat16 b_col_reduce_add[MODEL_DIM/ Q4NX_GROUP_SIZE];
    
    _lock_acquire(x_cons_lock);

    rms_norm(x, x, x + MODEL_DIM);

    for (int k = 0; k < K_BLOCKS; k++) {
        bf16* b_col_reduce_add_ptr = b_col_reduce_add + k *(Q4NX_COL_BLOCK_SIZE / 32);
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_UNROLL_FULL
        for(int l = 0; l < Q4NX_COL_BLOCK_SIZE / 32; l++)AIE_LOOP_FLATTEN{
            *(b_col_reduce_add_ptr + l) = bf16(aie::reduce_add(aie::load_v<32>(x + k * Q4NX_COL_BLOCK_SIZE + 32*l)));
        }
    }
    for (int r = 0; r < ROUNDS; r++) {

        zero_vectorized<float, Q4NX_ROW_BLOCK_SIZE>(y_acc);
        static_assert(K_BLOCKS % 2 == 0, "K_BLOCKS should be even for better performance, otherwise the pipeline will have bubbles");
        for (int k = 0; k < K_BLOCKS; k++) {
            bf16* x_using = x + k * Q4NX_COL_BLOCK_SIZE;
            _lock_acquire(w_cons_lock);

            is_w_ping = !is_w_ping;
            q4k_block_t* w_using = is_w_ping ? w_ping : w_pong;

            _qmm_q4k_bf16<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>(w_using, x_using, y_acc, b_col_reduce_add + k *(Q4NX_COL_BLOCK_SIZE / 32));
            
            _lock_release(w_prod_lock);
        }

        _lock_acquire(y_prod_lock);

        is_y_ping = !is_y_ping;
        bf16* y_using = is_y_ping ? y_ping : y_pong;
        copy_y<Q4NX_ROW_BLOCK_SIZE>(y_using, y_acc);

        aie::vector<bf16, Q4NX_ROW_BLOCK_SIZE> y_vec = aie::load_v<Q4NX_ROW_BLOCK_SIZE>(y_using);
        bf16 final_scale_bf16 = (bf16)(*final_scale);
        bf16 final_scale_inv  = aie::inv(final_scale_bf16);
        aie::accum<accfloat, Q4NX_ROW_BLOCK_SIZE> y_accum = aie::mul(y_vec, final_scale_inv); // apply a global scaling to prevent overflow, this can be tuned
        y_vec = aie::tanh(y_accum.template to_vector<float>());
        y_vec = aie::mul(y_vec, final_scale_bf16); // scale back
        aie::store_v(y_using, y_vec);

        _lock_release(y_cons_lock);
    }
    _lock_release(x_prod_lock);
}
}
