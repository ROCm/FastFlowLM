// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "model_spec.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "q4_core.h"
#include "q8_core.h"
#include "lut_based_ops.h"

float chess_storage(% chess_alignof(v32int8)) softplus_lut_ab[256] = {
    0.00000000f, 0.00000000f, 0.00048859f, 0.00421473f,
    0.00062727f, 0.00525430f, 0.00080529f, 0.00654419f,
    0.00000000f, 0.00000000f, 0.00048859f, 0.00421473f,
    0.00062727f, 0.00525430f, 0.00080529f, 0.00654419f,
    0.00103378f, 0.00814266f, 0.00132700f, 0.01012074f,
    0.00170326f, 0.01256489f, 0.00218597f, 0.01557985f,
    0.00103378f, 0.00814266f, 0.00132700f, 0.01012074f,
    0.00170326f, 0.01256489f, 0.00218597f, 0.01557985f,
    0.00280509f, 0.01929206f, 0.00359894f, 0.02385342f,
    0.00461639f, 0.02944529f, 0.00591978f, 0.03628279f,
    0.00280509f, 0.01929206f, 0.00359894f, 0.02385342f,
    0.00461639f, 0.02944529f, 0.00591978f, 0.03628279f,
    0.00758835f, 0.04461889f, 0.00972261f, 0.05474805f,
    0.01244960f, 0.06700855f, 0.01592910f, 0.08178260f,
    0.00758835f, 0.04461889f, 0.00972261f, 0.05474805f,
    0.01244960f, 0.06700855f, 0.01592910f, 0.08178260f,
    0.02036098f, 0.09949259f, 0.02599324f, 0.12059159f,
    0.03313067f, 0.14554508f, 0.04214291f, 0.17480073f,
    0.02036098f, 0.09949259f, 0.02599324f, 0.12059159f,
    0.03313067f, 0.14554508f, 0.04214291f, 0.17480073f,
    0.05347073f, 0.20874225f, 0.06762788f, 0.24762348f,
    0.08519513f, 0.29148072f, 0.10680177f, 0.34002431f,
    0.05347073f, 0.20874225f, 0.06762788f, 0.24762348f,
    0.08519513f, 0.29148072f, 0.10680177f, 0.34002431f,
    0.13308923f, 0.39251766f, 0.16465192f, 0.44766199f,
    0.20195318f, 0.50351777f, 0.24522074f, 0.55750524f,
    0.13308923f, 0.39251766f, 0.16465192f, 0.44766199f,
    0.20195318f, 0.50351777f, 0.24522074f, 0.55750524f,
    0.29433542f, 0.60652873f, 0.34873826f, 0.64725330f,
    0.40738816f, 0.67652228f, 0.46879933f, 0.69184721f,
    0.29433542f, 0.60652873f, 0.34873826f, 0.64725330f,
    0.40738816f, 0.67652228f, 0.46879933f, 0.69184721f,
    0.53116956f, 0.69185110f, 0.59258169f, 0.67653359f,
    0.65123337f, 0.64727104f, 0.70563863f, 0.60655144f,
    0.53116956f, 0.69185110f, 0.59258169f, 0.67653359f,
    0.65123337f, 0.64727104f, 0.70563863f, 0.60655144f,
    0.75475614f, 0.55753126f, 0.79802668f, 0.50354547f,
    0.83533089f, 0.44768993f, 0.86689635f, 0.39254470f,
    0.75475614f, 0.55753126f, 0.79802668f, 0.50354547f,
    0.83533089f, 0.44768993f, 0.86689635f, 0.39254470f,
    0.89318631f, 0.34004965f, 0.91479513f, 0.29150386f,
    0.93236424f, 0.24764418f, 0.94652294f, 0.20876044f,
    0.89318631f, 0.34004965f, 0.91479513f, 0.29150386f,
    0.93236424f, 0.24764418f, 0.94652294f, 0.20876044f,
    0.95785204f, 0.17481651f, 0.96686532f, 0.14555859f,
    0.97400359f, 0.12060306f, 0.97963653f, 0.09950226f,
    0.95785204f, 0.17481651f, 0.96686532f, 0.14555859f,
    0.97400359f, 0.12060306f, 0.97963653f, 0.09950226f,
    0.98406894f, 0.08179068f, 0.98754887f, 0.06701528f,
    0.99027618f, 0.05475362f, 0.99241071f, 0.04462348f,
    0.98406894f, 0.08179068f, 0.98754887f, 0.06701528f,
    0.99027618f, 0.05475362f, 0.99241071f, 0.04462348f,
    0.99407949f, 0.03628656f, 0.99538303f, 0.02944838f,
    0.99640062f, 0.02385594f, 0.99719456f, 0.01929412f,
    0.99407949f, 0.03628656f, 0.99538303f, 0.02944838f,
    0.99640062f, 0.02385594f, 0.99719456f, 0.01929412f,
    0.99781376f, 0.01558152f, 0.99829653f, 0.01256624f,
    0.99867283f, 0.01012184f, 0.99896610f, 0.00814354f,
    0.99781376f, 0.01558152f, 0.99829653f, 0.01256624f,
    0.99867283f, 0.01012184f, 0.99896610f, 0.00814354f,
    0.99919461f, 0.00654491f, 0.99937265f, 0.00525487f,
    0.99951135f, 0.00421520f, 1.00000000f, 0.00000000f,
    0.99919461f, 0.00654491f, 0.99937265f, 0.00525487f,
    0.99951135f, 0.00421520f, 1.00000000f, 0.00000000f,
};




float chess_storage(% chess_alignof(v32int8)) softplus_lut_cd[256] = {
    0.00000000f, 0.00000000f, 0.00048859f, 0.00421473f,
    0.00062727f, 0.00525430f, 0.00080529f, 0.00654419f,
    0.00000000f, 0.00000000f, 0.00048859f, 0.00421473f,
    0.00062727f, 0.00525430f, 0.00080529f, 0.00654419f,
    0.00103378f, 0.00814266f, 0.00132700f, 0.01012074f,
    0.00170326f, 0.01256489f, 0.00218597f, 0.01557985f,
    0.00103378f, 0.00814266f, 0.00132700f, 0.01012074f,
    0.00170326f, 0.01256489f, 0.00218597f, 0.01557985f,
    0.00280509f, 0.01929206f, 0.00359894f, 0.02385342f,
    0.00461639f, 0.02944529f, 0.00591978f, 0.03628279f,
    0.00280509f, 0.01929206f, 0.00359894f, 0.02385342f,
    0.00461639f, 0.02944529f, 0.00591978f, 0.03628279f,
    0.00758835f, 0.04461889f, 0.00972261f, 0.05474805f,
    0.01244960f, 0.06700855f, 0.01592910f, 0.08178260f,
    0.00758835f, 0.04461889f, 0.00972261f, 0.05474805f,
    0.01244960f, 0.06700855f, 0.01592910f, 0.08178260f,
    0.02036098f, 0.09949259f, 0.02599324f, 0.12059159f,
    0.03313067f, 0.14554508f, 0.04214291f, 0.17480073f,
    0.02036098f, 0.09949259f, 0.02599324f, 0.12059159f,
    0.03313067f, 0.14554508f, 0.04214291f, 0.17480073f,
    0.05347073f, 0.20874225f, 0.06762788f, 0.24762348f,
    0.08519513f, 0.29148072f, 0.10680177f, 0.34002431f,
    0.05347073f, 0.20874225f, 0.06762788f, 0.24762348f,
    0.08519513f, 0.29148072f, 0.10680177f, 0.34002431f,
    0.13308923f, 0.39251766f, 0.16465192f, 0.44766199f,
    0.20195318f, 0.50351777f, 0.24522074f, 0.55750524f,
    0.13308923f, 0.39251766f, 0.16465192f, 0.44766199f,
    0.20195318f, 0.50351777f, 0.24522074f, 0.55750524f,
    0.29433542f, 0.60652873f, 0.34873826f, 0.64725330f,
    0.40738816f, 0.67652228f, 0.46879933f, 0.69184721f,
    0.29433542f, 0.60652873f, 0.34873826f, 0.64725330f,
    0.40738816f, 0.67652228f, 0.46879933f, 0.69184721f,
    0.53116956f, 0.69185110f, 0.59258169f, 0.67653359f,
    0.65123337f, 0.64727104f, 0.70563863f, 0.60655144f,
    0.53116956f, 0.69185110f, 0.59258169f, 0.67653359f,
    0.65123337f, 0.64727104f, 0.70563863f, 0.60655144f,
    0.75475614f, 0.55753126f, 0.79802668f, 0.50354547f,
    0.83533089f, 0.44768993f, 0.86689635f, 0.39254470f,
    0.75475614f, 0.55753126f, 0.79802668f, 0.50354547f,
    0.83533089f, 0.44768993f, 0.86689635f, 0.39254470f,
    0.89318631f, 0.34004965f, 0.91479513f, 0.29150386f,
    0.93236424f, 0.24764418f, 0.94652294f, 0.20876044f,
    0.89318631f, 0.34004965f, 0.91479513f, 0.29150386f,
    0.93236424f, 0.24764418f, 0.94652294f, 0.20876044f,
    0.95785204f, 0.17481651f, 0.96686532f, 0.14555859f,
    0.97400359f, 0.12060306f, 0.97963653f, 0.09950226f,
    0.95785204f, 0.17481651f, 0.96686532f, 0.14555859f,
    0.97400359f, 0.12060306f, 0.97963653f, 0.09950226f,
    0.98406894f, 0.08179068f, 0.98754887f, 0.06701528f,
    0.99027618f, 0.05475362f, 0.99241071f, 0.04462348f,
    0.98406894f, 0.08179068f, 0.98754887f, 0.06701528f,
    0.99027618f, 0.05475362f, 0.99241071f, 0.04462348f,
    0.99407949f, 0.03628656f, 0.99538303f, 0.02944838f,
    0.99640062f, 0.02385594f, 0.99719456f, 0.01929412f,
    0.99407949f, 0.03628656f, 0.99538303f, 0.02944838f,
    0.99640062f, 0.02385594f, 0.99719456f, 0.01929412f,
    0.99781376f, 0.01558152f, 0.99829653f, 0.01256624f,
    0.99867283f, 0.01012184f, 0.99896610f, 0.00814354f,
    0.99781376f, 0.01558152f, 0.99829653f, 0.01256624f,
    0.99867283f, 0.01012184f, 0.99896610f, 0.00814354f,
    0.99919461f, 0.00654491f, 0.99937265f, 0.00525487f,
    0.99951135f, 0.00421520f, 1.00000000f, 0.00000000f,
    0.99919461f, 0.00654491f, 0.99937265f, 0.00525487f,
    0.99951135f, 0.00421520f, 1.00000000f, 0.00000000f,
};

template<typename T, int vec_size>
aie::vector<T, vec_size> getSoftPlus(aie::vector<bfloat16, vec_size> vInput) {
    aie::vector<bfloat16, vec_size> input = vInput;

    int step_bits = -2;
    int bias = 32;
    int data_size = 16;
    int LUT_elems = 64;
    int shift_offset = 0; // unused

    using lut_type = aie::lut<4, float, bfloat16>;

    lut_type test_lut(LUT_elems, (bfloat16 *)softplus_lut_ab, (bfloat16 *)softplus_lut_cd);

    aie::linear_approx<bfloat16, lut_type> lin_aprox(test_lut, step_bits, bias, shift_offset);

    aie::vector<T, vec_size> output = lin_aprox.compute(input).template to_vector<T>();

    return output;
}


constexpr int w_prod_lock = 0;
constexpr int w_cons_lock = 1;
constexpr int x_prod_lock = 2;
constexpr int x_cons_lock = 3;
constexpr int abw_prod_lock = 4;
constexpr int abw_cons_lock = 5;
constexpr int y_prod_lock = 6;
constexpr int y_cons_lock = 7;

constexpr int ab_prod_lock = 4;
constexpr int ab_cons_lock = 5;

template<int vec_size>
aie::vector<float, vec_size> getSigmoidF32_nonLUT(aie::vector<float, vec_size> x){
  

  const float half = 0.5f;
  const float one = 1.0f;
  aie::vector<float, vec_size> v_half = aie::broadcast<float, vec_size>(half);
  aie::vector<float, vec_size> v_one = aie::broadcast<float, vec_size>(one);
  aie::accum<accfloat, vec_size> x_mul_half =aie::mul(x, v_half);
  aie::vector<bf16, vec_size> tanh_x_half = aie::tanh<bf16>(x_mul_half.template to_vector<float>());
  aie::accum<accfloat, vec_size> tanh_x_half_acc;
  tanh_x_half_acc.from_vector(tanh_x_half);
  aie::vector<float, vec_size> tanh_x_half_plus_one = aie::add(tanh_x_half_acc.template to_vector<float>(), v_one);
  aie::vector<float, vec_size> tanh_x_half_plus_one_div_2 = aie::mul(tanh_x_half_plus_one, v_half);

  return tanh_x_half_plus_one_div_2;


}
template<int M, int K>
void small_proj(
    float* y_acc,
    AB_QUANT_BLOCK * w_ping,
    AB_QUANT_BLOCK * w_pong,
    bf16 * x
) {

#if AB_QUANT_TYPE == __Q4_1__
    // REDUCE_LIMBS bf16 limbs per group, interleaved, see the fill loop below
    alignas(aie::vector_decl_align) bf16 b_col_reduce_add[REDUCE_LIMBS * MODEL_DIM / GGML_GROUP_SIZE];
#endif
    constexpr int m = QXNX_ROW_BLOCK_SIZE;
    constexpr int k = QXNX_COL_BLOCK_SIZE;

    zero_vectorized<float, M>(y_acc);
#if AB_QUANT_TYPE == __Q4_1__
    // the deferred nibble bias for one row block, see NIBBLE_BIAS
    alignas(aie::vector_decl_align) float y_bias[m];
#endif
    int i = 0;
    AIE_PERSIST_ZERO static bool is_w_ping = false;

    do {
        float* y_acc_using = y_acc + i;
        #if AB_QUANT_TYPE == __Q4_1__
        zero_vectorized<float, m>(y_bias);
        #endif
        for (int j = 0; j < K; j += k) {
            bf16* x_using = x + j;
            #if AB_QUANT_TYPE == __Q4_1__
            bf16* b_col_reduce_add_ptr = b_col_reduce_add + REDUCE_LIMBS * (j / GGML_GROUP_SIZE);
            // special logic for i == 0, avoid recompute if it is repeat.
            if (i == 0) {
                // Fill the b_col_reduce_add.  aie::reduce_add over a bf16
                // vector already sums in accfloat, but returns bf16 and throws
                // 16 of those mantissa bits away; that error rides the whole
                // min path.  Reduce in float32 instead and hand the kernel
                // REDUCE_LIMBS bf16 limbs, the same layout proj_main.cc uses.
                AIE_PREPARE_FOR_PIPELINING
                AIE_LOOP_UNROLL_FULL
                for(int l = 0; l < k / GGML_GROUP_SIZE; l++)AIE_LOOP_FLATTEN{
                    aie::accum<accfloat, GGML_GROUP_SIZE> x_acc;
                    x_acc.from_vector(aie::load_v<GGML_GROUP_SIZE>(x_using + GGML_GROUP_SIZE * l));
                    float sum = aie::reduce_add(x_acc.template to_vector<float>());
                    bf16 l0 = bf16(sum);
                    *(b_col_reduce_add_ptr + REDUCE_LIMBS * l)     = l0;
                    *(b_col_reduce_add_ptr + REDUCE_LIMBS * l + 1) =
                        bf16(sum - float(l0));
                }
            }
            _lock_acquire(w_cons_lock);
            is_w_ping = !is_w_ping;
            AB_QUANT_BLOCK* w_using = is_w_ping? w_ping : w_pong;

            _Q41_x_bf16(y_acc_using, y_bias, x_using, w_using, b_col_reduce_add_ptr);

            _lock_release(w_prod_lock);
            #elif AB_QUANT_TYPE == __Q8_0__
            _lock_acquire(w_cons_lock);
            is_w_ping = !is_w_ping;
            AB_QUANT_BLOCK* w_using = is_w_ping? w_ping : w_pong;

            _Q80_x_bf16(y_acc_using, x_using, w_using);

            _lock_release(w_prod_lock);
            #endif
        }
        #if AB_QUANT_TYPE == __Q4_1__
        _Q41_finalize(y_acc_using, y_bias);
        #endif
        i += m;
    } while (i < M);
}

void activate_a(float * a, float * A_BIAS){
    float* A = A_BIAS;
    float* BIAS = A_BIAS + L_V_HEADS_PADDED;
    for (int i = 0; i < L_V_HEADS_PADDED; i += 16){
        aie::vector<float, 16> a_vec = aie::load_v<16>(a + i);
        aie::vector<float, 16> A_vec = aie::load_v<16>(A + i);
        aie::vector<float, 16> bias_vec = aie::load_v<16>(BIAS + i);
        aie::vector<float, 16> dt = aie::add(a_vec, bias_vec);
        aie::accum<accfloat, 16> dt_acc;
        dt_acc.from_vector(dt);
        aie::vector<bf16, 16> dt_bf16 = dt_acc.template to_vector<bf16>();
        aie::vector<float, 16> a_activated_vec = getSoftPlus<float, 16>(dt_bf16);
        a_activated_vec = aie::mul(a_activated_vec, A_vec);
        aie::accum<accfloat, 16> a_activated_vec_acc;
        a_activated_vec_acc.from_vector(a_activated_vec);
        dt_bf16 = a_activated_vec_acc.template to_vector<bf16>();
        aie::accum<accfloat, 16> final_activation = getExpBf16(dt_bf16);
        aie::store_v(a + i, final_activation.template to_vector<float>());
    }
}

void activate_b(float * b){
    for (int i = 0; i < L_V_HEADS_PADDED / 16; i++){
        aie::vector<float, 16> b_vec = aie::load_v<16>(b + i * 16);
        b_vec = getSigmoidF32_nonLUT<16>(b_vec);
        aie::store_v(b + i * 16, b_vec);
    }
}


extern "C" {

void ab_proj(
    float * alpha_beta,
    AB_QUANT_BLOCK * w_ping,
    AB_QUANT_BLOCK * w_pong,
    bf16 * x,
    float* A_BIAS
)
{
    _down_lock_acquire(ab_prod_lock);
    _lock_acquire(x_cons_lock);
    _lock_acquire(abw_cons_lock);
    
    small_proj<L_V_HEADS_PADDED * 2, MODEL_DIM>(alpha_beta, w_ping, w_pong, x);
    activate_a(alpha_beta, A_BIAS);
    activate_b(alpha_beta + L_V_HEADS_PADDED);
    
    _lock_release(abw_prod_lock);
    _lock_release(x_prod_lock);
    _down_lock_release(ab_cons_lock);


} 


}