
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>

#include "typedef.hpp"
#pragma once

/**
 * @brief Performs a non-overlapping 3D convolution patch embedding.
 *
 * This function mimics the forward pass of:
 * nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)
 * where kernel_size = stride = [temporal_patch_size, patch_size, patch_size].
 *
 * The input is treated as a flattened sequence of non-overlapping patches. The operation
 * simplifies to a large dot product between the flattened input patch and the flattened
 * convolution kernel, followed by adding the bias.
 *
 * @param hidden_states The flattened input tensor [B, Cin*Tp*Ps*Ps] (row major, pointer to data).
 * @param weights The flattened weights tensor [Ed, Cin*Tp*Ps*Ps] (row major, pointer to data).
 * @param bias The bias vector [Ed] (pointer to data), or nullptr if the conv has no bias.
 * @param seq_len Sequence length (batch size, B).
 * @param in_channels Input channels (Cin).
 * @param embed_dim Embedding dimension (output channels, Ed).
 * @param temporal_patch_size Temporal patch size (kernel depth, Tp).
 * @param patch_size Spatial patch size (kernel height/width, Ps).
 * @return The flattened output tensor [B, Ed].
 */
std::vector<bf16> conv3d_patch_embed(
    const bf16* hidden_states,
    const bf16* weights,
    const bf16* bias,
    size_t seq_len,
    size_t in_channels,
    size_t embed_dim,
    size_t temporal_patch_size,
    size_t patch_size);



/**
 * @brief Performs a batched 3D convolution patch embedding using AVX-512 intrinsics.
 *
 * This function computes: output[b][e] = sum(hidden_states[b][i] * weights[e][i]) + bias[e]
 * where the sum runs over the 'kernel_volume'.
 *
 * @param hidden_states Input tensor (seq_len * kernel_volume)
 * @param weights Convolution kernels (embed_dim * kernel_volume)
 * @param bias Bias vector (embed_dim), or nullptr if the conv has no bias
 * @param seq_len Batch size (B)
 * @param in_channels Input channels (C_in)
 * @param embed_dim Output embedding dimension (Ed) / Output channels (C_out)
 * @param temporal_patch_size Time dimension of the patch (Kt)
 * @param patch_size Spatial dimension of the patch (Ks)
 * @return std::vector<bf16> The resulting embeddings (seq_len * embed_dim)
 */
std::vector<bf16> conv3d_patch_embed_avx512(
    const bf16* hidden_states,
    const bf16* weights,
    const bf16* bias,
    size_t seq_len,
    size_t in_channels,
    size_t embed_dim,
    size_t temporal_patch_size,
    size_t patch_size);

/**
 * @brief Optimized multi-threaded version with loop unrolling and prefetching.
 * 
 * Optimizations applied:
 * - Multi-threading with OpenMP (up to 4 threads)
 * - Loop unrolling (2x) to reduce loop overhead
 * - Software prefetching for better cache utilization
 * - Better memory access patterns
 *
 * @param hidden_states Input tensor (seq_len * kernel_volume)
 * @param weights Convolution kernels (embed_dim * kernel_volume)
 * @param bias Bias vector (embed_dim), or nullptr if the conv has no bias
 * @param seq_len Batch size (B)
 * @param in_channels Input channels (C_in)
 * @param embed_dim Output embedding dimension (Ed) / Output channels (C_out)
 * @param temporal_patch_size Time dimension of the patch (Kt)
 * @param patch_size Spatial dimension of the patch (Ks)
 * @return std::vector<bf16> The resulting embeddings (seq_len * embed_dim)
 */
std::vector<bf16> conv3d_patch_embed_avx512_optimized(
    const bf16* hidden_states,
    const bf16* weights,
    const bf16* bias,
    size_t seq_len,
    size_t in_channels,
    size_t embed_dim,
    size_t temporal_patch_size,
    size_t patch_size);

/**
 * @brief Maximum performance version with 4x loop unrolling and aggressive optimizations.
 * 
 * Optimizations applied:
 * - Multi-threading with OpenMP (up to 4 threads)
 * - 4x loop unrolling (64 elements per iteration) for maximum throughput
 * - Four independent accumulator chains to maximize instruction-level parallelism
 * - Aggressive software prefetching
 * - Optimized for modern CPUs with deep pipelines
 *
 * @param hidden_states Input tensor (seq_len * kernel_volume)
 * @param weights Convolution kernels (embed_dim * kernel_volume)
 * @param bias Bias vector (embed_dim), or nullptr if the conv has no bias
 * @param seq_len Batch size (B)
 * @param in_channels Input channels (C_in)
 * @param embed_dim Output embedding dimension (Ed) / Output channels (C_out)
 * @param temporal_patch_size Time dimension of the patch (Kt)
 * @param patch_size Spatial dimension of the patch (Ks)
 * @return std::vector<bf16> The resulting embeddings (seq_len * embed_dim)
 */
std::vector<bf16> conv3d_patch_embed_avx512_max_performance(
    const bf16* hidden_states,
    const bf16* weights,
    const bf16* bias,
    size_t seq_len,
    size_t in_channels,
    size_t embed_dim,
    size_t temporal_patch_size,
    size_t patch_size);