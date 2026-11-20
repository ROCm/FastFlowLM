// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once

#include <immintrin.h> // Intel intrinsics
// #include <stdfloat>    // For bf16 in C++23
#include <cstddef>     // For std::size_t
#include <array>
#include <limits>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include "typedef.hpp"

// Cross-platform count trailing zeros implementation
#ifdef _MSC_VER
    #include <intrin.h>
    inline unsigned count_trailing_zeros(unsigned int value) {
        unsigned long index;
        _BitScanForward(&index, value);
        return index;
    }
#else
    inline unsigned count_trailing_zeros(unsigned int value) {
        return __builtin_ctz(value);
    }
#endif

// something specific to this file

namespace {

inline __m512 exp512_approx(__m512 x) {
    const __m512 max_logf = _mm512_set1_ps(88.3762626647949f);
    const __m512 min_logf = _mm512_set1_ps(-88.3762626647949f);
    x = _mm512_max_ps(_mm512_min_ps(x, max_logf), min_logf);

    const __m512 log2ef = _mm512_set1_ps(1.44269504088896341f);
    const __m512 c1 = _mm512_set1_ps(0.693359375f);
    const __m512 c2 = _mm512_set1_ps(-2.12194440e-4f);

    __m512 fx = _mm512_mul_ps(x, log2ef);
    fx = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m512 r = _mm512_fnmadd_ps(fx, c2, _mm512_fnmadd_ps(fx, c1, x));

    const __m512 p0 = _mm512_set1_ps(1.9875691500E-4f);
    const __m512 p1 = _mm512_set1_ps(1.3981999507E-3f);
    const __m512 p2 = _mm512_set1_ps(8.3334519073E-3f);
    const __m512 p3 = _mm512_set1_ps(4.1665795894E-2f);
    const __m512 p4 = _mm512_set1_ps(1.6666665459E-1f);
    const __m512 p5 = _mm512_set1_ps(5.0000001201E-1f);

    __m512 y = p0;
    y = _mm512_fmadd_ps(y, r, p1);
    y = _mm512_fmadd_ps(y, r, p2);
    y = _mm512_fmadd_ps(y, r, p3);
    y = _mm512_fmadd_ps(y, r, p4);
    y = _mm512_fmadd_ps(y, r, p5);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.0f));

    __m512i emm0 = _mm512_cvtps_epi32(fx);
    emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(0x7f));
    emm0 = _mm512_slli_epi32(emm0, 23);
    __m512 pow2n = _mm512_castsi512_ps(emm0);

    return _mm512_mul_ps(y, pow2n);
}

constexpr __mmask16 lane_mask_table[17] = {
    static_cast<__mmask16>(1u << 0),  static_cast<__mmask16>(1u << 1),
    static_cast<__mmask16>(1u << 2),  static_cast<__mmask16>(1u << 3),
    static_cast<__mmask16>(1u << 4),  static_cast<__mmask16>(1u << 5),
    static_cast<__mmask16>(1u << 6),  static_cast<__mmask16>(1u << 7),
    static_cast<__mmask16>(1u << 8),  static_cast<__mmask16>(1u << 9),
    static_cast<__mmask16>(1u << 10), static_cast<__mmask16>(1u << 11),
    static_cast<__mmask16>(1u << 12), static_cast<__mmask16>(1u << 13),
    static_cast<__mmask16>(1u << 14), static_cast<__mmask16>(1u << 15),
    static_cast<__mmask16>(0)
};

constexpr __mmask16 group_masks[4] = {
    static_cast<__mmask16>(0x000F),
    static_cast<__mmask16>(0x00F0),
    static_cast<__mmask16>(0x0F00),
    static_cast<__mmask16>(0xF000)
};

inline unsigned find_max_lane(const __m512& values, float& out_value) {
    const float max_val = _mm512_reduce_max_ps(values);
    __mmask16 mask = _mm512_cmp_ps_mask(values, _mm512_set1_ps(max_val), _CMP_EQ_OQ);

    if (mask == 0) {
        out_value = -std::numeric_limits<float>::infinity();
        return 16;
    }

    const unsigned lane = count_trailing_zeros(static_cast<unsigned int>(mask));
    out_value = max_val;
    return lane;
}
}

template<int K>
void topk_expert_select_avx512_bf16_out(
    const bf16* scores,
    int32_t* top_indices,
    bf16* top_values
) {
    static_assert(K > 0 && K <= 32, "K must be in [1, 32]");
    const int items = 32;

    // 1. Load all 32 bfloat16 scores into one 512-bit integer vector.
    __m512i scores_bf16 = _mm512_loadu_si512(scores);

    // 2. Split into two 16-element chunks (256-bit each).
    __m256i scores0_bf16_chunk = _mm512_castsi512_si256(scores_bf16);
    __m256i scores1_bf16_chunk = _mm512_extracti64x4_epi64(scores_bf16, 1);

    // 3. Convert to fp32.
    __m512 scores0_fp32 = _mm512_cvtpbh_ps((__m256bh)scores0_bf16_chunk);
    __m512 scores1_fp32 = _mm512_cvtpbh_ps((__m256bh)scores1_bf16_chunk);

    const __m512 neg_inf_vec = _mm512_set1_ps(-std::numeric_limits<float>::infinity());

    float top_vals[K];
    int top_idx[K];

    for (int k = 0; k < K; ++k) {
        float chunk0_val, chunk1_val;
        const unsigned lane0 = find_max_lane(scores0_fp32, chunk0_val);
        const unsigned lane1 = find_max_lane(scores1_fp32, chunk1_val);

        const int take0_mask = -static_cast<int>(chunk0_val >= chunk1_val);
        const int lane0_valid_mask = -static_cast<int>(lane0 < 16);
        const int lane1_valid_mask = -static_cast<int>(lane1 < 16);

        const int lane0_idx = (lane0_valid_mask & static_cast<int>(lane0)) |
                               (~lane0_valid_mask & -1);
        const int lane1_idx = (lane1_valid_mask & (static_cast<int>(lane1) + 16)) |
                               (~lane1_valid_mask & -1);

        const int val0_bits = std::bit_cast<int>(chunk0_val);
        const int val1_bits = std::bit_cast<int>(chunk1_val);
        const int selected_bits = (take0_mask & val0_bits) | (~take0_mask & val1_bits);

        top_vals[k] = std::bit_cast<float>(selected_bits);
        top_idx[k] = (take0_mask & lane0_idx) | (~take0_mask & lane1_idx);

        const __mmask16 select0_mask = static_cast<__mmask16>(take0_mask & 0xFFFF);
        const __mmask16 select1_mask = static_cast<__mmask16>((~take0_mask) & 0xFFFF);
        const __mmask16 lane_mask0 = lane_mask_table[lane0];
        const __mmask16 lane_mask1 = lane_mask_table[lane1];

        scores0_fp32 = _mm512_mask_mov_ps(scores0_fp32, select0_mask & lane_mask0, neg_inf_vec);
        scores1_fp32 = _mm512_mask_mov_ps(scores1_fp32, select1_mask & lane_mask1, neg_inf_vec);
    }

    // Copy indices out
    for (int i = 0; i < K; ++i) {
        top_indices[i] = (top_idx[i] >= 0)
            ? static_cast<int32_t>(top_idx[i])
            : static_cast<int32_t>(-1);
    }

    // Convert top values to bf16
    for (int i = 0; i < K; i += 4) {
        int remaining = (K - i >= 4) ? 4 : (K - i);
        float tmp[4] = {0, 0, 0, 0};
        for (int j = 0; j < remaining; ++j)
            tmp[j] = top_vals[i + j];

        __m128 vals_fp32 = _mm_loadu_ps(tmp);
        __m128bh vals_bf16 = _mm_cvtneps_pbh(vals_fp32);
        _mm_storeu_si64(top_values + i, (__m128i)vals_bf16);
    }
}

template<int K>
void topk_expert_select_bf16_out(
    const bf16* scores,
    int32_t* top_indices,
    bf16* top_values
) {
    static_assert(K > 0 && K <= 32, "K must be in [1, 32]");
    const int items = 32;

    // Create pairs of (value, index) for sorting
    std::vector<std::pair<bf16, int32_t>> score_index_pairs;
    score_index_pairs.reserve(items);

    for (int i = 0; i < items; ++i) {
        score_index_pairs.emplace_back(scores[i], i);
    }

    // Sort in descending order by score
    std::partial_sort(
        score_index_pairs.begin(),
        score_index_pairs.begin() + K,
        score_index_pairs.end(),
        [](const std::pair<bf16, int32_t>& a, const std::pair<bf16, int32_t>& b) {
            return a.first > b.first;
        }
    );

    // Extract top K values and indices
    for (int i = 0; i < K; ++i) {
        top_values[i] = score_index_pairs[i].first;
        top_indices[i] = score_index_pairs[i].second;
    }
}

template <std::size_t TOP_K>
void softmax_topk_bf16_avx512(bf16* values, std::size_t rows) {
    if constexpr (TOP_K == 0) {
        return;
    }

    std::array<float, TOP_K> temp_exp_values{};

    for (std::size_t r = 0; r < rows; ++r) {
        bf16* row_ptr = values + r * TOP_K;
        float* temp_ptr = temp_exp_values.data();

        __m512 max_vec = _mm512_set1_ps(std::numeric_limits<float>::lowest());
        for (std::size_t j = 0; j < TOP_K; j += 16) {
            const std::size_t remaining = TOP_K - j;
            const __mmask16 mask = (remaining >= 16)
                ? static_cast<__mmask16>(0xFFFF)
                : static_cast<__mmask16>((1u << remaining) - 1u);

            __m256i bf16_chunk = _mm256_maskz_loadu_epi16(mask, row_ptr + j);
            __m512 fp32_chunk = _mm512_cvtpbh_ps((__m256bh)bf16_chunk);

            max_vec = _mm512_mask_max_ps(max_vec, mask, max_vec, fp32_chunk);
        }
        const float max_val = _mm512_reduce_max_ps(max_vec);

        double sum = 0.0;
        const __m512 max_val_vec = _mm512_set1_ps(max_val);
        const __m512 zero_vec = _mm512_setzero_ps();
        for (std::size_t j = 0; j < TOP_K; j += 16) {
            const std::size_t remaining = TOP_K - j;
            const __mmask16 mask = (remaining >= 16)
                ? static_cast<__mmask16>(0xFFFF)
                : static_cast<__mmask16>((1u << remaining) - 1u);

            __m256i bf16_chunk = _mm256_maskz_loadu_epi16(mask, row_ptr + j);
            __m512 fp32_chunk = _mm512_cvtpbh_ps((__m256bh)bf16_chunk);

            __m512 shifted_vals = _mm512_mask_sub_ps(zero_vec, mask, fp32_chunk, max_val_vec);
            __m512 exp_vals = exp512_approx(shifted_vals);
            exp_vals = _mm512_mask_mov_ps(zero_vec, mask, exp_vals);

            _mm512_mask_storeu_ps(temp_ptr + j, mask, exp_vals);
            sum += static_cast<double>(_mm512_mask_reduce_add_ps(mask, exp_vals));
        }

    const float inv_sum = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
    const __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
        for (std::size_t j = 0; j < TOP_K; j += 16) {
            const std::size_t remaining = TOP_K - j;
            const __mmask16 mask = (remaining >= 16)
                ? static_cast<__mmask16>(0xFFFF)
                : static_cast<__mmask16>((1u << remaining) - 1u);

            __m512 exp_vals = _mm512_maskz_loadu_ps(mask, temp_ptr + j);
            __m512 norm_vals = _mm512_mask_mul_ps(zero_vec, mask, exp_vals, inv_sum_vec);

            __m256bh out_bf16 = _mm512_mask_cvtneps_pbh((__m256bh)_mm256_setzero_si256(), mask, norm_vals);
            _mm256_mask_storeu_epi16(row_ptr + j, mask, (__m256i)out_bf16);
        }
    }
}

template<int TOP_K, int NUM_OF_EXPERT>
void moeRouterPrefill(
    const bf16* input_logits, // L * NUM_OF_EXPERT
    std::vector<std::vector<std::pair<int, bf16>>>& expert_to_index_map,
    uint32_t L
){

    expert_to_index_map.resize(NUM_OF_EXPERT);
    for (int i = 0; i < NUM_OF_EXPERT; i++){
        expert_to_index_map[i].reserve(L);
    }
    std::vector<int> router_index_out_vec(L * TOP_K);
    std::vector<bf16> router_score_out_vec(L * TOP_K);

    for(uint32_t i = 0; i < L; i++){
        uint32_t logits_offest = i * NUM_OF_EXPERT;

        uint32_t res_offset = i*TOP_K;
        topk_expert_select_avx512_bf16_out<TOP_K>(
            input_logits + logits_offest,
            router_index_out_vec.data() + res_offset,
            router_score_out_vec.data() + res_offset
        );
    }
    softmax_topk_bf16_avx512<TOP_K>(router_score_out_vec.data(), L);
    for(uint32_t i = 0; i < L; i++){
        uint32_t res_offset = i*TOP_K;
        for(int k = 0; k < TOP_K; k++){
            int expert_id = router_index_out_vec[res_offset + k];
            bf16 score = router_score_out_vec[res_offset + k];
            std::pair<int, bf16> pair = std::make_pair(i, score);
            expert_to_index_map[expert_id].push_back(pair);
        }
    }
}

