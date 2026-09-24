#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdint>
#include "typedef.hpp"
#pragma once





void simd_add(
    bf16* input1,
    bf16* input2,
    bf16* output,
    size_t size
);



// Optimized AVX-512 version that combines bias addition and GELU activation
// bias is broadcast across sequence positions (size = hidden_dim, not total_size)
void simd_bias_add_gelu(
    bf16* input,
    const bf16* bias,
    bf16* output,
    size_t total_size,
    size_t hidden_dim
);


void gelu_bfloat16_ref(
    const bf16* input,
    bf16* output,
    size_t size
);


// GELU activation function with tanh approximation (same as PytorchGELUTanh)
inline float gelu_tanh(float x) {
    return 0.5f * x * (1 + std::tanh(std::sqrt(2.0f / 3) * (x + 0.044715f * x * x * x)));
}