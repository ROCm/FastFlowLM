// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __LFM2_CPU_FUNCTIONS_HPP__
#define __LFM2_CPU_FUNCTIONS_HPP__
#include <cmath>
#include "typedef.hpp"
#include "buffer.hpp"
#include "tensor_2d.hpp"

/// @brief Host-side kernels used by the lfm2 prefill path.
/// @note  These used to be private members of lfm2_npu::Impl; they are free
///        functions here so the prefill blocks can call them without a handle
///        on the model. Everything they used to read off `this` (DH, D,
///        inv_freq, the context length) is now an argument.
namespace lfm2_cpu_func {

inline void _elementwise_mul(buffer<bf16>& dest, buffer<bf16>& a, buffer<bf16>& b){
    int n = a.size();
    const int simd_width = 16;
    const int unroll = simd_width * 4;

    int i = 0;
    // 4-way unrolled loop for maximum throughput
    for (; i <= n - unroll; i += unroll){
        __m256i bf16_vals_a0 = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i bf16_vals_a1 = _mm256_loadu_si256((__m256i*)&a[i + simd_width]);
        __m256i bf16_vals_a2 = _mm256_loadu_si256((__m256i*)&a[i + simd_width * 2]);
        __m256i bf16_vals_a3 = _mm256_loadu_si256((__m256i*)&a[i + simd_width * 3]);

        __m256i bf16_vals_b0 = _mm256_loadu_si256((__m256i*)&b[i]);
        __m256i bf16_vals_b1 = _mm256_loadu_si256((__m256i*)&b[i + simd_width]);
        __m256i bf16_vals_b2 = _mm256_loadu_si256((__m256i*)&b[i + simd_width * 2]);
        __m256i bf16_vals_b3 = _mm256_loadu_si256((__m256i*)&b[i + simd_width * 3]);

        __m512 fp32_vals_a0 = bf16o_fp32_512(bf16_vals_a0);
        __m512 fp32_vals_a1 = bf16o_fp32_512(bf16_vals_a1);
        __m512 fp32_vals_a2 = bf16o_fp32_512(bf16_vals_a2);
        __m512 fp32_vals_a3 = bf16o_fp32_512(bf16_vals_a3);

        __m512 fp32_vals_b0 = bf16o_fp32_512(bf16_vals_b0);
        __m512 fp32_vals_b1 = bf16o_fp32_512(bf16_vals_b1);
        __m512 fp32_vals_b2 = bf16o_fp32_512(bf16_vals_b2);
        __m512 fp32_vals_b3 = bf16o_fp32_512(bf16_vals_b3);

        __m512 result_vec0 = _mm512_mul_ps(fp32_vals_a0, fp32_vals_b0);
        __m512 result_vec1 = _mm512_mul_ps(fp32_vals_a1, fp32_vals_b1);
        __m512 result_vec2 = _mm512_mul_ps(fp32_vals_a2, fp32_vals_b2);
        __m512 result_vec3 = _mm512_mul_ps(fp32_vals_a3, fp32_vals_b3);

        __m256i result_bf16_0 = f32o_bf16_512(result_vec0);
        __m256i result_bf16_1 = f32o_bf16_512(result_vec1);
        __m256i result_bf16_2 = f32o_bf16_512(result_vec2);
        __m256i result_bf16_3 = f32o_bf16_512(result_vec3);

        _mm256_storeu_si256((__m256i*)&dest[i], result_bf16_0);
        _mm256_storeu_si256((__m256i*)&dest[i + simd_width], result_bf16_1);
        _mm256_storeu_si256((__m256i*)&dest[i + simd_width * 2], result_bf16_2);
        _mm256_storeu_si256((__m256i*)&dest[i + simd_width * 3], result_bf16_3);
    }

    // Handle remaining elements
    for (; i <= n - simd_width; i += simd_width){
        __m256i bf16_vals_a = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i bf16_vals_b = _mm256_loadu_si256((__m256i*)&b[i]);
        __m512 fp32_vals_a = bf16o_fp32_512(bf16_vals_a);
        __m512 fp32_vals_b = bf16o_fp32_512(bf16_vals_b);
        __m512 result_vec = _mm512_mul_ps(fp32_vals_a, fp32_vals_b);
        __m256i result_bf16 = f32o_bf16_512(result_vec);
        _mm256_storeu_si256((__m256i*)&dest[i], result_bf16);
    }
}

inline void _rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& rms_weight){
    int n = x.size();
    const int simd_width = 8;

    __m256 sum_xx_vec = _mm256_setzero_ps();

    // Step 1: Compute sum of squares using AVX
    int i = 0;
    for (; i <= n - simd_width; i += simd_width) {
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);

        // Convert BF16 to FP32
        __m256 fp32_vals = bf16o_fp32(bf16_vals);
        sum_xx_vec = _mm256_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);  // sum_xx += x[i] * x[i];
    }

    // Horizontal sum of the AVX register
    f32 sum_xx_arr[simd_width];
    _mm256_storeu_ps(sum_xx_arr, sum_xx_vec);
    f32 sum_xx = sum_xx_arr[0] + sum_xx_arr[1] + sum_xx_arr[2] + sum_xx_arr[3] +
                      sum_xx_arr[4] + sum_xx_arr[5] + sum_xx_arr[6] + sum_xx_arr[7];

    sum_xx /= (f32)n;
    f32 rms_x = sqrtf(sum_xx + 1e-5f);

    // Step 2: Compute element-wise operation y[i] = w[i] * x[i] / rms_x
    __m256 rms_x_vec = _mm256_set1_ps(rms_x);
    for (i = 0; i <= n - simd_width; i += simd_width) {
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);
        __m256 fp32_vals = bf16o_fp32(bf16_vals);
        __m128i bf16_vals_w = _mm_loadu_si128((__m128i*)&rms_weight[i]);
        __m256 fp32_vals_w = bf16o_fp32(bf16_vals_w);
        __m256 y_vec = _mm256_div_ps(_mm256_mul_ps(fp32_vals_w, fp32_vals), rms_x_vec);
        __m128i result_bf16 = f32o_bf16(y_vec);
        _mm_storeu_si128((__m128i*)&y[i], result_bf16);
    }
}

inline void _rope_rms(buffer<bf16>& x, int idx, buffer<bf16>& rms_weight, uint32_t DH, const f32* inv_freq){
    int heads = x.size() / DH;
    buffer<f32> cos_buf(DH / 2);
    buffer<f32> sin_buf(DH / 2);
    for (uint32_t i = 0; i < DH / 2; i++){
        cos_buf[i] = std::cos(inv_freq[i] * idx);
        sin_buf[i] = std::sin(inv_freq[i] * idx);
    }
    for (int h = 0; h < heads; h++){
        buffer<bf16> x_head(x.data() + h * DH, DH);
        _rms_norm(x_head, x_head, rms_weight);
        buffer<bf16> x_left(x_head.data(), DH / 2);
        buffer<bf16> x_right(x_head.data() + DH / 2, DH / 2);
        int i = 0, simd = 8;
        // vectorized main loop
        for (; i + simd <= (int)(DH / 2); i += simd) {
            // load 8 BF16 → float32
            __m128i left_bf16  = _mm_loadu_si128((__m128i*)(x_left.data()  + i));
            __m128i right_bf16 = _mm_loadu_si128((__m128i*)(x_right.data() + i));
            __m256  L = bf16o_fp32(left_bf16);
            __m256  R = bf16o_fp32(right_bf16);

            // load sin/cos
            __m256 C = _mm256_loadu_ps(cos_buf.data() + i);
            __m256 S = _mm256_loadu_ps(sin_buf.data() + i);

            // new_left  = L*C - R*S
            __m256 newL = _mm256_sub_ps(_mm256_mul_ps(L, C),
                                        _mm256_mul_ps(R, S));
            // new_right = L*S + R*C
            __m256 newR = _mm256_add_ps(_mm256_mul_ps(L, S),
                                        _mm256_mul_ps(R, C));
            // store back
            __m128i newL_bf16 = f32o_bf16(newL);
            __m128i newR_bf16 = f32o_bf16(newR);
            _mm_storeu_si128((__m128i*)(x_left.data()  + i), newL_bf16);
            _mm_storeu_si128((__m128i*)(x_right.data() + i), newR_bf16);
        }
    }
}

inline void _residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual){
    int n = src.size();
    const int simd_width = 8;

    for (int i = 0; i < n; i += simd_width){
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)&src[i]);
        __m128i bf16_vals_r = _mm_loadu_si128((__m128i*)&residual[i]);
        __m256 fp32_vals = bf16o_fp32(bf16_vals);
        __m256 fp32_vals_r = bf16o_fp32(bf16_vals_r);
        __m256 result_vec = _mm256_add_ps(fp32_vals, fp32_vals_r);
        __m128i result_bf16 = f32o_bf16(result_vec);
        _mm_storeu_si128((__m128i*)&dest[i], result_bf16);
    }
}

inline __m256 _mm256_exp_ps(__m256 x){
    // clamp x to avoid overflow
    const __m256 max_val = _mm256_set1_ps(88.3762626647949f);
    const __m256 min_val = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(x, max_val);
    x = _mm256_max_ps(x, min_val);

    // express exp(x) = 2^(x / ln2)
    const __m256 ln2_inv = _mm256_set1_ps(1.44269504088896341f);  // 1/ln(2)
    __m256 fx = _mm256_mul_ps(x, ln2_inv);
    // round to nearest int
    fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
    __m256i emm0 = _mm256_cvttps_epi32(fx);
    fx = _mm256_cvtepi32_ps(emm0);

    // rebuild x' = x - fx*ln2
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    __m256 tmp = _mm256_mul_ps(fx, ln2_hi);
    __m256 z   = _mm256_mul_ps(fx, ln2_lo);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);

    // polynomial approximation on reduced x
    const __m256 c0 = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 c1 = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 c2 = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 c3 = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 c4 = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 c5 = _mm256_set1_ps(5.0000001201E-1f);

    __m256 y = c0;
    y = _mm256_fmadd_ps(y, x, c1);
    y = _mm256_fmadd_ps(y, x, c2);
    y = _mm256_fmadd_ps(y, x, c3);
    y = _mm256_fmadd_ps(y, x, c4);
    y = _mm256_fmadd_ps(y, x, c5);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.0f));

    // build 2^n from the integer part
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7F));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(emm0);

    // combine
    return _mm256_mul_ps(y, pow2n);
}

inline void _glu(buffer<bf16>& dest, buffer<bf16>& up_gate, int L_offset, int L_effective, uint32_t INTERMEDIATE_SIZE){
    const int simd_width = 8;
    const int GLU_SLICE = 1024;

    const int n = L_effective * 2 * (int)INTERMEDIATE_SIZE;
    const int chunks = n / GLU_SLICE;
    bf16 * in_ptr = up_gate.data() + L_offset * 2 * INTERMEDIATE_SIZE;
    bf16 * out_ptr = dest.data() + L_offset * INTERMEDIATE_SIZE;
    for (int i = 0; i < chunks; i++){
        bf16* up_ptr = in_ptr;
        bf16* gate_ptr = up_ptr + GLU_SLICE / 2;

        for (int j = 0; j < GLU_SLICE / 2; j += simd_width){
            __m128i up_vec = _mm_loadu_si128((__m128i*)&up_ptr[j]);
            __m128i gate_vec = _mm_loadu_si128((__m128i*)&gate_ptr[j]);
            __m256 up_vec_fp32 = bf16o_fp32(up_vec);
            __m256 gate_vec_fp32 = bf16o_fp32(gate_vec);
            // ---- SiLU(gate):  gate * sigmoid(gate)  ----
            __m256 one    = _mm256_set1_ps(1.0f);
            __m256 neg    = _mm256_sub_ps(_mm256_setzero_ps(), gate_vec_fp32);
            __m256 exp_n  = lfm2_cpu_func::_mm256_exp_ps(neg);
            __m256 sig    = _mm256_div_ps(one, _mm256_add_ps(one, exp_n));
            __m256 silu   = _mm256_mul_ps(gate_vec_fp32, sig);

            __m256 result_vec = _mm256_mul_ps(silu, up_vec_fp32);
            __m128i result_bf16 = f32o_bf16(result_vec);
            _mm_storeu_si128((__m128i*)&out_ptr[j], result_bf16);
        }
        in_ptr += GLU_SLICE;
        out_ptr += GLU_SLICE / 2;
    }
}

inline void _compute_cbx(buffer<bf16>& dest, buffer<bf16>& c, buffer<bf16>& bx, int L_effective, int L_offset, uint32_t D){
    tensor_2d<bf16> tensor_bcx(c, D, L_offset * 3); // 3k + 0: b, 3k + 1: c, 3k + 2: x
    tensor_2d<bf16> tensor_bx(bx, D, 0); // no offset here, already considered whenn filling the conv in buffer
    tensor_2d<bf16> tensor_dest(dest, D, L_offset); // put the offset back to match the attntion layers

    for (int i = 0; i < L_effective; i++){
        bf16* bx_ptr = tensor_bx[i].data();
        bf16* c_ptr = tensor_bcx[3 * i + 1].data();
        bf16* dest_ptr = tensor_dest[i].data();

        // AVX-optimized element-wise multiplication
        const int simd_width = 8;
        int j = 0;
        for (; j + simd_width <= (int)D; j += simd_width) {
            // Load 8 bf16 values from bx_ptr
            __m128i bx_bf16 = _mm_loadu_si128((__m128i*)(bx_ptr + j));
            __m256 bx_fp32 = bf16o_fp32(bx_bf16);

            // Load 8 bf16 values from c_ptr
            __m128i c_bf16 = _mm_loadu_si128((__m128i*)(c_ptr + j));
            __m256 c_fp32 = bf16o_fp32(c_bf16);

            // Multiply in float32
            __m256 result_fp32 = _mm256_mul_ps(bx_fp32, c_fp32);

            // Convert back to bf16 and store
            __m128i result_bf16 = f32o_bf16(result_fp32);
            _mm_storeu_si128((__m128i*)(dest_ptr + j), result_bf16);
        }

        // Scalar fallback for remaining elements
        for (; j < (int)D; j++) {
            dest_ptr[j] = bx_ptr[j] * c_ptr[j];
        }
    }
}

inline void _prepare_conv_in(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& kv_cache, int L_effective, int L_offset, uint32_t D, int L_begin){

    tensor_2d<bf16> tensor_bcx(src, D, L_offset * 3); // 3k + 0: b, 3k + 1: c, 3k + 2: x
    tensor_2d<bf16> tensor_dest(dest, D, 0);
    kv_cache.sync_from_device(); //actually conv cache
    if (L_begin % 2 == 0){ // in order
        memcpy(tensor_dest[0].data(), kv_cache.data(), D * sizeof(bf16));
        memcpy(tensor_dest[1].data(), kv_cache.data() + D, D * sizeof(bf16));
    }
    else{ // reverse of order
        memcpy(tensor_dest[0].data(), kv_cache.data() + D, D * sizeof(bf16));
        memcpy(tensor_dest[1].data(), kv_cache.data(), D * sizeof(bf16));
    }

    for (int i = 0; i < L_effective; i++) {
        bf16* b_ptr = tensor_bcx[3 * i].data();
        bf16* x_ptr = tensor_bcx[3 * i + 2].data();
        bf16* bx_ptr = tensor_dest[i + 2].data();

        // AVX-optimized element-wise multiplication
        const int simd_width = 8;
        int j = 0;
        for (; j + simd_width <= (int)D; j += simd_width) {
            // Load 8 bf16 values from b_ptr
            __m128i b_bf16 = _mm_loadu_si128((__m128i*)(b_ptr + j));
            __m256 b_fp32 = bf16o_fp32(b_bf16);

            // Load 8 bf16 values from x_ptr
            __m128i x_bf16 = _mm_loadu_si128((__m128i*)(x_ptr + j));
            __m256 x_fp32 = bf16o_fp32(x_bf16);

            // Multiply in float32
            __m256 result_fp32 = _mm256_mul_ps(b_fp32, x_fp32);

            // Convert back to bf16 and store
            __m128i result_bf16 = f32o_bf16(result_fp32);
            _mm_storeu_si128((__m128i*)(bx_ptr + j), result_bf16);
        }

        // Scalar fallback for remaining elements
        for (; j < (int)D; j++) {
            bx_ptr[j] = b_ptr[j] * x_ptr[j];
        }
    }

    int t_end = L_begin + L_effective;
    if (t_end % 2 == 0) { // even time index, only in kv cache [0]s
        memcpy(kv_cache.data(), tensor_dest[L_effective + 2 - 1].data(), D * sizeof(bf16));
        memcpy(kv_cache.data() + D, tensor_dest[L_effective + 2 - 2].data(), D * sizeof(bf16));
    }
    else{
        memcpy(kv_cache.data() + D, tensor_dest[L_effective + 2 - 1].data(), D * sizeof(bf16));
        memcpy(kv_cache.data(), tensor_dest[L_effective + 2 - 2].data(), D * sizeof(bf16));
    }
    kv_cache.sync_to_device();
}

} // namespace lfm2_cpu_func

#endif
