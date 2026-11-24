// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef GEMMA_IMAGE_UTIL_HPP
#define GEMMA_IMAGE_UTIL_HPP

// This is copy file from https://github.com/joeldushouyu/Template/commits/gemma3-mega-kernel-branch/
// commit 2fb4269ad927e16938708c641346a7b0c1710826

#include <vector>
#include <iostream>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <immintrin.h>
#include <algorithm> // For std::min
#include "typedef.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <type_traits> // Required for std::is_same_v
#include "biovault_bfloat16.h"

// from error_measure.hpp
template<typename Ta,typename Tb >
float get_relativeL2(Ta* y, Tb* y_ref, int batch_size, int seq_len, int hidden_size, int seq_len_padded, int hidden_size_padded){

    float rmse = 0;
    float ref_sum = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < hidden_size; ++h) {
                long long y_idx = (long long)b * seq_len_padded * hidden_size_padded + (long long)s * hidden_size_padded + h;
                long long y_ref_idx = (long long)b * seq_len * hidden_size + (long long)s * hidden_size + h;
                ref_sum += static_cast<float>(y_ref[y_ref_idx]) * static_cast<float>(y_ref[y_ref_idx]);
                rmse += (static_cast<float>(y[y_idx]) - static_cast<float>(y_ref[y_ref_idx])) * (static_cast<float>(y[y_idx]) - static_cast<float>(y_ref[y_ref_idx]));
            }
        }
    }
    int y_size = batch_size * seq_len * hidden_size;
    return sqrt(rmse / y_size) / sqrt(ref_sum / y_size);
}

// y is [batch_size, seq_len_padded, hidden_size_padded]
// y_ref is [batch_size, seq_len, hidden_size]
template<typename Ta,typename Tb >
float get_relativeL1(Ta* y, Tb *y_ref, int batch_size, int seq_len, int hidden_size, int seq_len_padded, int hidden_size_padded)
{
    float l1 = 0;
    float ref_sum = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < hidden_size; ++h) {
                long long y_idx = (long long)b * seq_len_padded * hidden_size_padded + (long long)s * hidden_size_padded + h;
                long long y_ref_idx = (long long)b * seq_len * hidden_size + (long long)s * hidden_size + h;
                ref_sum += std::fabs(static_cast<float>(y_ref[y_ref_idx]));
                l1 += std::fabs(static_cast<float>(y[y_idx]) - static_cast<float>(y_ref[y_ref_idx]));
            }
        }
    }
    return l1 / ref_sum;
}

template<typename Ta,typename Tb >
float get_rmse(Ta *y, Tb *y_ref, int batch_size, int seq_len, int hidden_size, int seq_len_padded, int hidden_size_padded)
{
    float rmse = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < hidden_size; ++h) {
                long long y_idx = (long long)b * seq_len_padded * hidden_size_padded + (long long)s * hidden_size_padded + h;
                long long y_ref_idx = (long long)b * seq_len * hidden_size + (long long)s * hidden_size + h;
                rmse += ((float)y[y_idx] - (float)y_ref[y_ref_idx]) * ((float)y[y_idx] - (float)y_ref[y_ref_idx]);
            }
        }
    }
    int y_size = batch_size * seq_len * hidden_size;
    return sqrt(rmse / y_size);
}

template<typename T>
float cal_rms_value(T* a, int a_size) {
    float sum_sq = 0.0f;
    for (int i = 0; i < a_size; ++i) {
        sum_sq += static_cast<float>(a[i]) * static_cast<float>(a[i]);
    }
    return std::sqrt(sum_sq / a_size);
}
template<typename Ta,typename Tb >
float get_cosine_similarity(Ta *y, Tb *y_ref, int batch_size, int seq_len, int hidden_size, int seq_len_padded, int hidden_size_padded)
{
    float dot_product = 0;
    float norm_y = 0;
    float norm_y_ref = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < hidden_size; ++h) {
                long long y_idx = (long long)b * seq_len_padded * hidden_size_padded + (long long)s * hidden_size_padded + h;
                long long y_ref_idx = (long long)b * seq_len * hidden_size + (long long)s * hidden_size + h;
                dot_product += static_cast<float>(y[y_idx]) * static_cast<float>(y_ref[y_ref_idx]);
                norm_y += static_cast<float>(y[y_idx]) * static_cast<float>(y[y_idx]);
                norm_y_ref += static_cast<float>(y_ref[y_ref_idx]) * static_cast<float>(y_ref[y_ref_idx]);
            }
        }
    }
    return dot_product / (sqrt(norm_y) * sqrt(norm_y_ref));
}

template<typename T_a, typename T_b>
void print_error_metrics(T_a*a, T_b*b, int batch_size, int seq_len, int hidden_size, int seq_len_padded, int hidden_size_padded){

    float rmse = get_rmse(a, b, batch_size, seq_len, hidden_size, seq_len_padded, hidden_size_padded);
    float relativeL1 = get_relativeL1(a, b, batch_size, seq_len, hidden_size, seq_len_padded, hidden_size_padded);
    float relativeL2 = get_relativeL2(a, b, batch_size, seq_len, hidden_size, seq_len_padded, hidden_size_padded);
    float cosine_similarity = get_cosine_similarity(a, b, batch_size, seq_len, hidden_size, seq_len_padded, hidden_size_padded);

    header_print("info", "Relative L1: " << relativeL1);
    header_print("info", "Relative L2: " << relativeL2);
    header_print("info", "Cosine similarity: " << cosine_similarity);
    header_print("info", "RMSE: " << rmse);
}

// from conv2d
void conv2d_avx512_bf16_ptr_bias_hwc_out(
    const  biovault::bfloat16_t* image,
    const biovault::bfloat16_t* kernel,
    const biovault::bfloat16_t* bias,
    biovault::bfloat16_t* output,
    int C_in, int H, int W,
    int C_out, int K,
    int stride);

// for embedding
void add_embedding(
    biovault::bfloat16_t* __restrict data,
    const biovault::bfloat16_t* __restrict positional_embedding,
    const int batch_size,
    const int sequence_length,
    const int hidden_size,
    const int padded_hidden_size);

void layernorm_avx512_bf16_single_core(
     const biovault::bfloat16_t* input,   // <-- MODIFIED
     biovault::bfloat16_t* output,        // <-- MODIFIED
     const biovault::bfloat16_t* weight,
     const biovault::bfloat16_t* bias,
     const int batch_size,
     const int seq_len,
     const int seq_len_padded,
     const int hidden_size,
     const int hidden_size_padded,
     const float eps ) ;
void rmsnorm_avx512_bf16_single_core(
    biovault::bfloat16_t* output,           // MODIFIED: Separate output buffer
    const biovault::bfloat16_t* input,      // MODIFIED: Separate const input buffer
    const biovault::bfloat16_t* weight,
    const int batch_size,
    const int seq_len,
    const int seq_len_padded,
    const int hidden_size,
    const int hidden_size_padded,
    const float eps);

inline int round_up_to_multiple (int x, int multiple)
{
    return ((x + multiple - 1) / multiple) * multiple;
};

void vector_add_bfloat16(biovault::bfloat16_t *dest,biovault::bfloat16_t *a,  biovault::bfloat16_t *b, size_t size);

// from average pooling

int average_pooling_avx512(
    biovault::bfloat16_t* __restrict input_output, // [batch_size, seq_len_padded, hidden_size_padded]
    int batch_size,
    int seq_len,
    int seq_len_padded,
    int hidden_size,
    int hidden_size_padded,
    int kernel,
    int stride,
    int padding
);

#endif // GEMMA_IMAGE_UTIL_HPP

