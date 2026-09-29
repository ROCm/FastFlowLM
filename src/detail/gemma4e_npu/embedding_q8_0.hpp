#ifndef __EMBEDDING_Q8_0_HPP__
#define __EMBEDDING_Q8_0_HPP__
#include "buffer.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include <immintrin.h>

class embedding_q8_0{
private:
    static constexpr int Q8_0_GROUP_SIZE = 32;
    buffer<float> scale;
    buffer<int8_t> qweight;
    size_t vocabe_size;
    size_t dim;
    tensor_2d<int8_t> tensor_qweight;
    tensor_2d<float> tensor_scale;

    buffer<bf16> out_buffer;

    public:
    embedding_q8_0(size_t vocab_size, size_t dim) {
        this->vocabe_size = vocab_size;
        this->dim = dim;
        this->scale = buffer<float>(vocab_size * dim / Q8_0_GROUP_SIZE);
        this->qweight = buffer<int8_t>(vocab_size * dim);
        this->out_buffer = buffer<bf16>(dim);
        this->tensor_qweight = tensor_2d<int8_t>(qweight, dim);
        this->tensor_scale = tensor_2d<float>(scale, dim / Q8_0_GROUP_SIZE);
    }

    void init_weights(Q4NX& q4nx, const std::string& weight_name){
        q4nx.load_weights(this->scale, weight_name + ".weight.scale");
        q4nx.load_weights(this->qweight, weight_name + ".weight");
    }

    inline void dequant_row_avx512(const int8_t* __restrict qw, const float* __restrict sc, bf16* __restrict dst) {
        const size_t num_groups = dim / Q8_0_GROUP_SIZE;
        for (size_t i = 0; i < num_groups; i++) {
            const int8_t* group_ptr = qw + i * Q8_0_GROUP_SIZE;
            bf16* out_ptr = dst + i * Q8_0_GROUP_SIZE;
            __m512 scale_vec = _mm512_set1_ps(sc[i]);

            // First 16 elements
            __m128i q8_lo = _mm_loadu_si128((const __m128i*)group_ptr);
            __m512i q32_lo = _mm512_cvtepi8_epi32(q8_lo);
            __m512 f32_lo = _mm512_cvtepi32_ps(q32_lo);
            f32_lo = _mm512_mul_ps(f32_lo, scale_vec);
            __m512i bf16_lo = _mm512_srli_epi32(_mm512_castps_si512(
                _mm512_add_ps(f32_lo, _mm512_castsi512_ps(
                    _mm512_add_epi32(_mm512_set1_epi32(0x7FFF),
                        _mm512_and_si512(_mm512_srli_epi32(_mm512_castps_si512(f32_lo), 16),
                            _mm512_set1_epi32(1)))))), 16);
            __m256i out_lo = _mm512_cvtepi32_epi16(bf16_lo);
            _mm256_storeu_si256((__m256i*)out_ptr, out_lo);

            // Next 16 elements
            __m128i q8_hi = _mm_loadu_si128((const __m128i*)(group_ptr + 16));
            __m512i q32_hi = _mm512_cvtepi8_epi32(q8_hi);
            __m512 f32_hi = _mm512_cvtepi32_ps(q32_hi);
            f32_hi = _mm512_mul_ps(f32_hi, scale_vec);
            __m512i bf16_hi = _mm512_srli_epi32(_mm512_castps_si512(
                _mm512_add_ps(f32_hi, _mm512_castsi512_ps(
                    _mm512_add_epi32(_mm512_set1_epi32(0x7FFF),
                        _mm512_and_si512(_mm512_srli_epi32(_mm512_castps_si512(f32_hi), 16),
                            _mm512_set1_epi32(1)))))), 16);
            __m256i out_hi = _mm512_cvtepi32_epi16(bf16_hi);
            _mm256_storeu_si256((__m256i*)(out_ptr + 16), out_hi);
        }
    }

    buffer<bf16> forward(int idx){
        buffer<int8_t> qweight_row = tensor_qweight[idx];
        buffer<float> scale_row = tensor_scale[idx];
        dequant_row_avx512(qweight_row.data(), scale_row.data(), out_buffer.data());
        return out_buffer;
    }

    void forward(int idx, buffer<bf16>& out){
        buffer<int8_t> qweight_row = tensor_qweight[idx];
        buffer<float> scale_row = tensor_scale[idx];
        dequant_row_avx512(qweight_row.data(), scale_row.data(), out.data());
    }
};

#endif
