#pragma once



#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <immintrin.h>
#include "typedef.hpp"





/**
 * @brief High-precision Layer Normalization using scalar float32 operations (no AVX).
 * 
 * Uses double precision for mean/variance accumulation and standard sqrt.
 * This is slower but more accurate than AVX version - ideal for debugging.
 */
void layernorm_high_precision(
    size_t N,
    bf16* hidden_states,
    bf16* weight,
    bf16* bias,
    float eps,
    bf16* output);


/**
 * @brief Parallel Layer Normalization across multiple sequences with OpenMP.
 *
 * Applies layer normalization to multiple sequences in parallel using up to 4 threads.
 * Uses chunked distribution (static scheduling) for better cache locality.
 * Automatically disables parallelization for small sequence counts.
 */
void layernorm_parallel(
    int seq_len,
    size_t hidden_dim,
    size_t hidden_dim_padded,
    bf16* hidden_states_base,
    bf16* weight,
    bf16* bias,
    float eps,
    bf16* output_base);


