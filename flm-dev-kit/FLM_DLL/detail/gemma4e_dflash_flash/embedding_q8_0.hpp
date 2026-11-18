// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __EMBEDDING_Q8_0_HPP__
#define __EMBEDDING_Q8_0_HPP__
#include "buffer.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_utils/mapped_file.hpp"
#include "tensor_2d.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <vector>

/// Row-wise q8_0 lookup table: int8 [vocab, dim] plus one F32 scale per 32 values.
/// Read one row per token, so both tensors are mapped read-only from model.q4nx instead of
/// copied (the per-layer table of a Gemma4 E4B is 2.7 GiB). owned_* is the fallback when a
/// range cannot be mapped.
class embedding_q8_0{
private:
    static constexpr int Q8_0_GROUP_SIZE = 32;
    size_t vocabe_size;
    size_t dim;

    mapped_file mapped_qweight;
    mapped_file mapped_scale;
    buffer<int8_t> owned_qweight;
    buffer<float> owned_scale;
    const int8_t* qweight = nullptr;
    const uint8_t* scale = nullptr; // may be unaligned inside the file, read with memcpy

    buffer<bf16> out_buffer;

    public:
    embedding_q8_0(size_t vocab_size, size_t dim) {
        this->vocabe_size = vocab_size;
        this->dim = dim;
        this->out_buffer = buffer<bf16>(dim);
    }

    /// \brief Bind the table named weight_name (+ ".weight", + ".weight.scale") of model.q4nx
    /// \param q4nx the opened model.q4nx
    /// \param q4nx_path path of that file, for mapping
    void init_weights(Q4NX& q4nx, const std::string& q4nx_path, const std::string& weight_name){
        const std::string wname = weight_name + ".weight";
        const std::string sname = weight_name + ".weight.scale";
        tensor_metadata wm = q4nx.get_tensor_metadata(wname);
        tensor_metadata sm = q4nx.get_tensor_metadata(sname);
        if (wm.offsets[1] - wm.offsets[0] >= vocabe_size * dim
            && this->mapped_qweight.open(q4nx_path, wm.offsets[0], wm.offsets[1] - wm.offsets[0])){
            this->qweight = reinterpret_cast<const int8_t*>(this->mapped_qweight.data());
        }
        else{
            this->owned_qweight = buffer<int8_t>(vocabe_size * dim);
            q4nx.load_weights(this->owned_qweight, wname);
            this->qweight = this->owned_qweight.begin();
        }
        if (sm.offsets[1] - sm.offsets[0] >= vocabe_size * dim / Q8_0_GROUP_SIZE * sizeof(float)
            && this->mapped_scale.open(q4nx_path, sm.offsets[0], sm.offsets[1] - sm.offsets[0])){
            this->scale = this->mapped_scale.data();
        }
        else{
            this->owned_scale = buffer<float>(vocabe_size * dim / Q8_0_GROUP_SIZE);
            q4nx.load_weights(this->owned_scale, sname);
            this->scale = reinterpret_cast<const uint8_t*>(this->owned_scale.begin());
        }
    }

    /// \brief Hint that these token rows are about to be looked up (a prompt about to be prefilled).
    /// \note Matters only when the mapped pages were dropped (memory pressure): the rows of a
    ///       whole prompt are then read in one batch instead of faulting in one page at a time.
    void prefetch(const int* ids, size_t n){
        if (!this->mapped_qweight && !this->mapped_scale) return;
        std::vector<size_t> rows;
        rows.reserve(n);
        for (size_t i = 0; i < n; i++){
            if (ids[i] >= 0 && (size_t)ids[i] < vocabe_size) rows.push_back((size_t)ids[i]);
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        const size_t groups = dim / Q8_0_GROUP_SIZE;
        for (size_t r : rows){
            this->mapped_qweight.will_need(r * dim, dim);
            this->mapped_scale.will_need(r * groups * sizeof(float), groups * sizeof(float));
        }
    }

    inline void dequant_row_avx512(const int8_t* __restrict qw, const uint8_t* __restrict sc, bf16* __restrict dst) {
        const size_t num_groups = dim / Q8_0_GROUP_SIZE;
        for (size_t i = 0; i < num_groups; i++) {
            const int8_t* group_ptr = qw + i * Q8_0_GROUP_SIZE;
            bf16* out_ptr = dst + i * Q8_0_GROUP_SIZE;
            float sc_i;
            std::memcpy(&sc_i, sc + i * sizeof(float), sizeof(float));
            __m512 scale_vec = _mm512_set1_ps(sc_i);
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

    /// \note an id outside the vocabulary gives a zero row (the table is mapped, not padded)
    void dequant_row(int idx, bf16* dst){
        if (idx < 0 || (size_t)idx >= vocabe_size){
            std::memset(dst, 0, dim * sizeof(bf16));
            return;
        }
        const size_t groups = dim / Q8_0_GROUP_SIZE;
        dequant_row_avx512(qweight + (size_t)idx * dim, scale + (size_t)idx * groups * sizeof(float), dst);
    }

    buffer<bf16> forward(int idx){
        dequant_row(idx, out_buffer.data());
        return out_buffer;
    }

    void forward(int idx, buffer<bf16>& out){
        dequant_row(idx, out.data());
    }
};

#endif
