#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdint>
#include "typedef.hpp"
#pragma once

void simd_conv2d(
    bf16* input,
    const bf16* kernel,
    bf16* output,
    int C_in, int H_in, int W_in,
    int C_out, int K, int stride, int padding
);
void scalar_conv2d(
    const bf16* input,
    const bf16* kernel,
    bf16* output,
    int C_in, int H_in, int W_in,
    int C_out, int K, int stride, int padding
);

void simd_layernorm(
    bf16* input, // shape of [seq_len, D_padded] in row major ordr, but only D is valud
    bf16* output,
    const bf16* weights,
    int D,
    int D_padded,
    int seq_len,
    float eps = 1e-6f
);

// NCHW channel-wise operations (acting across C dimension, strided by HW)
void layernorm_relu_nchw(bf16* data, const bf16* norm_weight, int C, int HW, float eps = 1e-6f);
void layernorm_gelu_nchw(bf16* data, const bf16* norm_weight, int C, int HW, float eps = 1e-6f);
void rmsnorm_gelu_nchw(bf16* data, const bf16* norm_weight, int C, int HW, float eps = 1e-6f);

void simd_relu(
    const bf16* input,
    bf16* output,
    size_t size
);

void simd_silu(
    const bf16* input,
    bf16* output,
    size_t size
);

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

// RMS Norm without scale weights
// Input layout: [seq_len_padded x X_padded], only processes seq_len rows and X cols per row
// Formula: output = input * rsqrt(mean(input^2) + eps)
void simd_rms_norm(
    const bf16* input,
    bf16* output,
    size_t seq_len,
    size_t X,
    size_t seq_len_padded,
    size_t X_padded,
    float eps = 1e-6f
);

// RMS Norm with scale weights (Gemma4RMSNorm with with_scale=True)
// Input layout: [seq_len_padded x X_padded], only processes seq_len rows and X cols per row
// norm_weight: bf16 pointer of size [X]
// Formula: output = (input * rsqrt(mean(input^2) + eps)) * weight
void simd_rms_norm(
    const bf16* input,
    const bf16* norm_weight,
    bf16* output,
    size_t seq_len,
    size_t X,
    size_t seq_len_padded,
    size_t X_padded,
    float eps = 1e-6f
);

void simd_clamp(
    const bf16* input,
    bf16* output,
    bf16 min_val,
    bf16 max_val,
    size_t size
);

void simd_mul(
    const bf16* input1,
    const bf16* input2,
    bf16* output,
    size_t size
);

void simd_mul(
    const bf16* input1,
    bf16 input2_scalar,
    bf16* output,
    size_t size
);

void transpose_2d(
    const bf16* input,
    bf16* output,
    size_t rows,
    size_t cols
);

// General bf16 matmul: C = A @ B^T  with float32 accumulation
// A: [M, K] row-major with stride lda
// B: [N, K] row-major with stride ldb  (B^T gives [K, N])
// C: [M, N] row-major with stride ldc
void simd_gemm_abt_bf16(
    const bf16* A, const bf16* B, bf16* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
);

void simd_glu(
    const bf16* input,  //[seq_len, 2,hidden_dim]
    bf16* output,   // [seq_len, hidden_dim]
    size_t seq_len,
    size_t hidden_dim
);

void conv1d(
    int conv_kernel_size,
    int conv_stride,
    const bf16* input,   /// [seq_len+ (conv_kernel_size-conv_stride), hidden_dim]
    const bf16* kernel_weights, // [conv_kernel_size, hidden_dim]
    bf16* output, //  [seq_len, hidden_dim]
    int seq_len,
    int hidden_dim

);

void scalar_conv1d(
    int conv_kernel_size,
    int conv_stride,
    const bf16* input,   /// [seq_len+ (conv_kernel_size-conv_stride), hidden_dim]
    const bf16* kernel_weights, // [conv_kernel_size, hidden_dim]
    bf16* output, //  [seq_len, hidden_dim]
    int seq_len,
    int hidden_dim
);
