// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "typedef.hpp"

// ============================================================================
// AVX-512 Helper Functions
// ============================================================================

// Helper: AVX-512 horizontal sum for 16 floats
inline float _mm512_reduce_add_ps_manual(__m512 v) {
    __m256 low = _mm512_castps512_ps256(v);
    __m256 high = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(low, high);
    __m128 low128 = _mm256_castps256_ps128(sum256);
    __m128 high128 = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(low128, high128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

// Helper: AVX-512 horizontal max for 16 floats
inline float _mm512_reduce_max_ps_manual(__m512 v) {
    __m256 low = _mm512_castps512_ps256(v);
    __m256 high = _mm512_extractf32x8_ps(v, 1);
    __m256 max256 = _mm256_max_ps(low, high);
    __m128 low128 = _mm256_castps256_ps128(max256);
    __m128 high128 = _mm256_extractf128_ps(max256, 1);
    __m128 max128 = _mm_max_ps(low128, high128);
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(max128);
}

// Optimized dot product using AVX-512
inline float dot_product_avx512(const std::bfloat16_t* a, const std::bfloat16_t* b, int len) {
    __m512 sum = _mm512_setzero_ps();
    int i = 0;
    
    // Process 16 elements at a time
    for (; i + 15 < len; i += 16) {
        // Load bfloat16 and convert to float
        __m256i a_bf16 = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i b_bf16 = _mm256_loadu_si256((__m256i*)&b[i]);
        
        // Convert bfloat16 to float by shifting left 16 bits
        __m512i a_int = _mm512_cvtepu16_epi32(a_bf16);
        __m512i b_int = _mm512_cvtepu16_epi32(b_bf16);
        __m512i a_shifted = _mm512_slli_epi32(a_int, 16);
        __m512i b_shifted = _mm512_slli_epi32(b_int, 16);
        __m512 a_float = _mm512_castsi512_ps(a_shifted);
        __m512 b_float = _mm512_castsi512_ps(b_shifted);
        
        // Fused multiply-add
        sum = _mm512_fmadd_ps(a_float, b_float, sum);
    }
    
    float result = _mm512_reduce_add_ps_manual(sum);
    
    // Handle remaining elements
    for (; i < len; i++) {
        result += (float)a[i] * (float)b[i];
    }
    
    return result;
}

// ============================================================================
// AVX-512 Optimized Attention Reference Implementation
// ============================================================================

inline void attention_reference_avx512(
    std::bfloat16_t* q_buffer,
    std::bfloat16_t* k_buffer,
    std::bfloat16_t* v_buffer,
    std::bfloat16_t* output_buffer,
    int L_SEQ,
    int S_SEQ,
    int D_H,
    int num_head
) {
    float scale = 1.0f / std::sqrt((float)D_H);
    int total_dim = D_H * num_head;

    // Process each head independently
    #pragma omp parallel for schedule(dynamic)
    for (int h = 0; h < num_head; ++h) {
        int head_offset = h * D_H;

        // 1. Compute Scores S = Q * K^T / sqrt(D_H) for this head
        std::vector<float> S(L_SEQ * S_SEQ);

        for (int i = 0; i < L_SEQ; ++i) {
            for (int j = 0; j < S_SEQ; ++j) {
                float dot = dot_product_avx512(
                    &q_buffer[i * total_dim + head_offset],
                    &k_buffer[j * total_dim + head_offset],
                    D_H
                );
                S[i * S_SEQ + j] = dot * scale;
            }
        }

        // 2. Softmax row-wise with AVX-512
        std::vector<float> P(L_SEQ * S_SEQ);
        for (int i = 0; i < L_SEQ; ++i) {
            // Find max using AVX-512
            float max_val = -std::numeric_limits<float>::infinity();
            int j = 0;
            __m512 vmax = _mm512_set1_ps(max_val);
            
            for (; j + 15 < S_SEQ; j += 16) {
                __m512 vals = _mm512_loadu_ps(&S[i * S_SEQ + j]);
                vmax = _mm512_max_ps(vmax, vals);
            }
            max_val = _mm512_reduce_max_ps_manual(vmax);
            
            for (; j < S_SEQ; ++j) {
                if (S[i * S_SEQ + j] > max_val) max_val = S[i * S_SEQ + j];
            }

            // Compute exp(x - max) and sum
            __m512 vmax_broadcast = _mm512_set1_ps(max_val);
            __m512 sum_vec = _mm512_setzero_ps();
            j = 0;
            
            for (; j + 15 < S_SEQ; j += 16) {
                __m512 vals = _mm512_loadu_ps(&S[i * S_SEQ + j]);
                vals = _mm512_sub_ps(vals, vmax_broadcast);
                
                // Compute exp - using scalar for accuracy
                alignas(64) float temp[16];
                _mm512_storeu_ps(temp, vals);
                for (int k = 0; k < 16; k++) {
                    temp[k] = std::exp(temp[k]);
                }
                __m512 exp_vals = _mm512_loadu_ps(temp);
                _mm512_storeu_ps(&P[i * S_SEQ + j], exp_vals);
                sum_vec = _mm512_add_ps(sum_vec, exp_vals);
            }
            
            float sum_exp = _mm512_reduce_add_ps_manual(sum_vec);
            
            for (; j < S_SEQ; ++j) {
                float val = std::exp(S[i * S_SEQ + j] - max_val);
                P[i * S_SEQ + j] = val;
                sum_exp += val;
            }

            // Normalize
            __m512 sum_inv = _mm512_set1_ps(1.0f / sum_exp);
            j = 0;
            for (; j + 15 < S_SEQ; j += 16) {
                __m512 vals = _mm512_loadu_ps(&P[i * S_SEQ + j]);
                vals = _mm512_mul_ps(vals, sum_inv);
                _mm512_storeu_ps(&P[i * S_SEQ + j], vals);
            }
            
            for (; j < S_SEQ; ++j) {
                P[i * S_SEQ + j] /= sum_exp;
            }
        }

        // 3. Output O = P * V for this head
        for (int i = 0; i < L_SEQ; ++i) {
            for (int d = 0; d < D_H; ++d) {
                __m512 val_vec = _mm512_setzero_ps();
                int j = 0;
                
                for (; j + 15 < S_SEQ; j += 16) {
                    __m512 p_vals = _mm512_loadu_ps(&P[i * S_SEQ + j]);
                    
                    // Load V values and convert from bfloat16
                    alignas(64) float v_vals[16];
                    for (int k = 0; k < 16; k++) {
                        v_vals[k] = (float)v_buffer[(j + k) * total_dim + head_offset + d];
                    }
                    __m512 v_vec = _mm512_loadu_ps(v_vals);
                    
                    val_vec = _mm512_fmadd_ps(p_vals, v_vec, val_vec);
                }
                
                float val = _mm512_reduce_add_ps_manual(val_vec);
                
                for (; j < S_SEQ; ++j) {
                    val += P[i * S_SEQ + j] * (float)v_buffer[j * total_dim + head_offset + d];
                }
                output_buffer[i * total_dim + head_offset + d] = (std::bfloat16_t)val;
            }
        }
    }
}

// ============================================================================
// AVX-512 Optimized Flash Attention Progressive Implementation
// ============================================================================

inline void flash_attention_progressive_avx512(
    std::bfloat16_t* q_buffer,
    std::bfloat16_t* k_buffer,
    std::bfloat16_t* v_buffer,
    std::bfloat16_t* output_buffer,
    int L_SEQ,
    int S_SEQ,
    int D_H,
    const int BLOCK_L_Q,
    const int BLOCK_L_V,
    int num_head,
    std::vector<float> & S_ij_row,
    std::vector<float> & m_i,
    std::vector<float> & m_new_vec,
    std::vector<float> &exp_old_vec,
    std::vector<float> & l_i,
    std::vector<float> &P_ij_unnorm_flat,
    std::vector<float> &O_accum
) {
    float scale = 1.0f / std::sqrt((float)D_H);
    int total_dim = D_H * num_head;
    
    #pragma omp parallel for schedule(dynamic)
    for (int h = 0; h < num_head; ++h) {
        int head_offset = h * D_H;
        
        // Thread-local storage for each head
        std::vector<float> local_S_ij_row;
        std::vector<float> local_m_i;
        std::vector<float> local_m_new_vec;
        std::vector<float> local_exp_old_vec;
        std::vector<float> local_l_i;
        std::vector<float> local_P_ij_unnorm_flat;
        std::vector<float> local_O_accum;
        
        for (int i_start = 0; i_start < L_SEQ; i_start += BLOCK_L_Q) {
            int i_end = std::min(i_start + BLOCK_L_Q, L_SEQ);
            int current_block_q_len = i_end - i_start;

            local_O_accum.resize(current_block_q_len * D_H);
            std::fill(local_O_accum.begin(), local_O_accum.end(), 0.0f);
            
            local_m_i.resize(current_block_q_len);
            local_l_i.resize(current_block_q_len);
            std::fill(local_m_i.begin(), local_m_i.end(), -std::numeric_limits<float>::infinity());
            std::fill(local_l_i.begin(), local_l_i.end(), 0.0f);

            for (int j_start = 0; j_start < S_SEQ; j_start += BLOCK_L_V) {
                int j_end = std::min(j_start + BLOCK_L_V, S_SEQ);
                int current_block_v_len = j_end - j_start;

                // STEP 1: Compute scores with AVX-512
                local_S_ij_row.resize(current_block_q_len * current_block_v_len);

                for (int i = 0; i < current_block_q_len; ++i) {
                    int global_i = i_start + i;
                    for (int j = 0; j < current_block_v_len; ++j) {
                        int global_j = j_start + j;
                        
                        float dot = dot_product_avx512(
                            &q_buffer[global_i * total_dim + head_offset],
                            &k_buffer[global_j * total_dim + head_offset],
                            D_H
                        );
                        local_S_ij_row[i * current_block_v_len + j] = dot * scale;
                    }
                }

                // STEP 2: FlashAttention tiling
                
                // 2.1 Find max with AVX-512
                std::vector<float> m_ij_block_vec(current_block_q_len);
                for (int i = 0; i < current_block_q_len; ++i) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int j = 0;
                    __m512 vmax = _mm512_set1_ps(max_val);
                    
                    for (; j + 15 < current_block_v_len; j += 16) {
                        __m512 vals = _mm512_loadu_ps(&local_S_ij_row[i * current_block_v_len + j]);
                        vmax = _mm512_max_ps(vmax, vals);
                    }
                    max_val = _mm512_reduce_max_ps_manual(vmax);
                    
                    for (; j < current_block_v_len; j++) {
                        if (local_S_ij_row[i * current_block_v_len + j] > max_val) {
                            max_val = local_S_ij_row[i * current_block_v_len + j];
                        }
                    }
                    m_ij_block_vec[i] = max_val;
                }

                // 2.2 Compute m_new and exp_old
                local_m_new_vec.resize(current_block_q_len);
                local_exp_old_vec.resize(current_block_q_len);
                for (int i = 0; i < current_block_q_len; ++i) {
                    float m_old = local_m_i[i];
                    float m_new = std::max(m_old, m_ij_block_vec[i]);
                    local_m_new_vec[i] = m_new;
                    local_exp_old_vec[i] = std::exp(m_old - m_new);
                }

                // 2.3 Compute P_ij_unnormalized with AVX-512
                local_P_ij_unnorm_flat.resize(current_block_q_len * current_block_v_len);
                for (int i = 0; i < current_block_q_len; ++i) {
                    float m_new = local_m_new_vec[i];
                    __m512 m_new_vec_avx = _mm512_set1_ps(m_new);
                    int j = 0;
                    
                    for (; j + 15 < current_block_v_len; j += 16) {
                        __m512 s = _mm512_loadu_ps(&local_S_ij_row[i * current_block_v_len + j]);
                        s = _mm512_sub_ps(s, m_new_vec_avx);
                        
                        // Compute exp
                        alignas(64) float temp[16];
                        _mm512_storeu_ps(temp, s);
                        for (int k = 0; k < 16; k++) {
                            temp[k] = std::exp(temp[k]);
                        }
                        _mm512_storeu_ps(&local_P_ij_unnorm_flat[i * current_block_v_len + j], _mm512_loadu_ps(temp));
                    }
                    
                    for (; j < current_block_v_len; ++j) {
                        float s = local_S_ij_row[i * current_block_v_len + j];
                        local_P_ij_unnorm_flat[i * current_block_v_len + j] = std::exp(s - m_new);
                    }
                }

                // 2.4 Update L_i with AVX-512
                for (int i = 0; i < current_block_q_len; ++i) {
                    float row_sum = 0.0f;
                    int j = 0;
                    __m512 sum_vec = _mm512_setzero_ps();
                    
                    for (; j + 15 < current_block_v_len; j += 16) {
                        __m512 vals = _mm512_loadu_ps(&local_P_ij_unnorm_flat[i * current_block_v_len + j]);
                        sum_vec = _mm512_add_ps(sum_vec, vals);
                    }
                    row_sum = _mm512_reduce_add_ps_manual(sum_vec);
                    
                    for (; j < current_block_v_len; ++j) {
                        row_sum += local_P_ij_unnorm_flat[i * current_block_v_len + j];
                    }
                    local_l_i[i] = local_exp_old_vec[i] * local_l_i[i] + row_sum;
                }

                // 2.5 Update O_accum with AVX-512
                for (int i = 0; i < current_block_q_len; ++i) {
                    int global_i = i_start + i;
                    float exp_old = local_exp_old_vec[i];
                    __m512 exp_old_vec_avx = _mm512_set1_ps(exp_old);
                    
                    // Scale old O
                    int d = 0;
                    for (; d + 15 < D_H; d += 16) {
                        __m512 o = _mm512_loadu_ps(&local_O_accum[i * D_H + d]);
                        o = _mm512_mul_ps(o, exp_old_vec_avx);
                        _mm512_storeu_ps(&local_O_accum[i * D_H + d], o);
                    }
                    for (; d < D_H; ++d) {
                        local_O_accum[i * D_H + d] *= exp_old;
                    }
      
                    // Add P * V contribution with AVX-512
                    for (d = 0; d < D_H; ++d) {
                        float pv_sum = 0.0f;
                        int j = 0;
                        __m512 pv_vec = _mm512_setzero_ps();
                        
                        for (; j + 15 < current_block_v_len; j += 16) {
                            __m512 p_vals = _mm512_loadu_ps(&local_P_ij_unnorm_flat[i * current_block_v_len + j]);
                            
                            alignas(64) float v_vals[16];
                            for (int k = 0; k < 16; k++) {
                                int global_j = j_start + j + k;
                                v_vals[k] = (float)v_buffer[global_j * total_dim + head_offset + d];
                            }
                            __m512 v_vec = _mm512_loadu_ps(v_vals);
                            
                            pv_vec = _mm512_fmadd_ps(p_vals, v_vec, pv_vec);
                        }
                        pv_sum = _mm512_reduce_add_ps_manual(pv_vec);
                        
                        for (; j < current_block_v_len; ++j) {
                            int global_j = j_start + j;
                            float p_val = local_P_ij_unnorm_flat[i * current_block_v_len + j];
                            float v_val = (float)v_buffer[global_j * total_dim + head_offset + d];
                            pv_sum += p_val * v_val;
                        }
                        local_O_accum[i * D_H + d] += pv_sum;
                    }
                }

                // 2.6 Update m_i state
                for (int i = 0; i < current_block_q_len; ++i) {
                    local_m_i[i] = local_m_new_vec[i];
                }
            }

            // 4. Final normalization
            for (int i = 0; i < current_block_q_len; ++i) {
                int global_i = i_start + i;
                float inv_l = 1.0f / local_l_i[i];
                __m512 inv_l_vec = _mm512_set1_ps(inv_l);
                int d = 0;
                
                for (; d + 15 < D_H; d += 16) {
                    __m512 o = _mm512_loadu_ps(&local_O_accum[i * D_H + d]);
                    o = _mm512_mul_ps(o, inv_l_vec);
                    
                    alignas(64) float temp[16];
                    _mm512_storeu_ps(temp, o);
                    for (int k = 0; k < 16; k++) {
                        output_buffer[global_i * total_dim + head_offset + d + k] = (std::bfloat16_t)temp[k];
                    }
                }
                
                for (; d < D_H; ++d) {
                    output_buffer[global_i * total_dim + head_offset + d] = (std::bfloat16_t)(local_O_accum[i * D_H + d] * inv_l);
                }
            }
        }
    }
}
