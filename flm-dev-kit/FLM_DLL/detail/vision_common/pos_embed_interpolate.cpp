// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "vision/pos_embed_interpolate.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

std::vector<bf16> fast_pos_embed_interpolate(
    const std::vector<std::vector<int32_t>>& grid_thw,
    const bf16* pos_embed_weight,
    int num_grid_per_side,
    int hidden_size,
    int spatial_merge_size
) {
    // Extract t, h, w from grid_thw
    std::vector<int32_t> grid_ts, grid_hs, grid_ws;
    for (const auto& thw : grid_thw) {
        grid_ts.push_back(thw[0]);
        grid_hs.push_back(thw[1]);
        grid_ws.push_back(thw[2]);
    }

    // Initialize index and weight lists (4 corners for bilinear interpolation)
    std::vector<std::vector<int64_t>> idx_list(4);
    std::vector<std::vector<float>> weight_list(4);

    // Process each image/video
    for (size_t i = 0; i < grid_thw.size(); ++i) {
        int32_t h = grid_hs[i];
        int32_t w = grid_ws[i];

        // Create linearly spaced indices for height and width
        std::vector<float> h_idxs(h);
        std::vector<float> w_idxs(w);

        for (int j = 0; j < h; ++j) {
            h_idxs[j] = static_cast<float>(j) * (num_grid_per_side - 1) / std::max(h - 1, 1);
        }
        for (int j = 0; j < w; ++j) {
            w_idxs[j] = static_cast<float>(j) * (num_grid_per_side - 1) / std::max(w - 1, 1);
        }

        // Compute floor and ceil indices
        std::vector<int32_t> h_idxs_floor(h);
        std::vector<int32_t> w_idxs_floor(w);
        std::vector<int32_t> h_idxs_ceil(h);
        std::vector<int32_t> w_idxs_ceil(w);
        std::vector<float> dh(h);
        std::vector<float> dw(w);

        for (int j = 0; j < h; ++j) {
            h_idxs_floor[j] = static_cast<int32_t>(h_idxs[j]);
            h_idxs_ceil[j] = std::min(h_idxs_floor[j] + 1, num_grid_per_side - 1);
            dh[j] = h_idxs[j] - h_idxs_floor[j];
        }

        for (int j = 0; j < w; ++j) {
            w_idxs_floor[j] = static_cast<int32_t>(w_idxs[j]);
            w_idxs_ceil[j] = std::min(w_idxs_floor[j] + 1, num_grid_per_side - 1);
            dw[j] = w_idxs[j] - w_idxs_floor[j];
        }

        // Compute base indices
        std::vector<int32_t> base_h(h);
        std::vector<int32_t> base_h_ceil(h);
        for (int j = 0; j < h; ++j) {
            base_h[j] = h_idxs_floor[j] * num_grid_per_side;
            base_h_ceil[j] = h_idxs_ceil[j] * num_grid_per_side;
        }

        // Generate indices and weights for 4 corners (bilinear interpolation)
        // indices[0]: top-left, indices[1]: top-right
        // indices[2]: bottom-left, indices[3]: bottom-right
        for (int hi = 0; hi < h; ++hi) {
            for (int wi = 0; wi < w; ++wi) {
                // Top-left corner
                idx_list[0].push_back(base_h[hi] + w_idxs_floor[wi]);
                weight_list[0].push_back((1.0f - dh[hi]) * (1.0f - dw[wi]));

                // Top-right corner
                idx_list[1].push_back(base_h[hi] + w_idxs_ceil[wi]);
                weight_list[1].push_back((1.0f - dh[hi]) * dw[wi]);

                // Bottom-left corner
                idx_list[2].push_back(base_h_ceil[hi] + w_idxs_floor[wi]);
                weight_list[2].push_back(dh[hi] * (1.0f - dw[wi]));

                // Bottom-right corner
                idx_list[3].push_back(base_h_ceil[hi] + w_idxs_ceil[wi]);
                weight_list[3].push_back(dh[hi] * dw[wi]);
            }
        }
    }

    // Get total number of spatial positions
    size_t total_positions = idx_list[0].size();

    // Gather embeddings and apply weights
    std::vector<bf16> patch_pos_embeds(total_positions * hidden_size, bf16(0.0f));

    // For each position, interpolate from 4 corners
#ifdef __AVX512F__

    // AVX-512 optimized bilinear interpolation
    for (size_t pos = 0; pos < total_positions; ++pos) {
        int dim = 0;

        // Process 16 dimensions at a time
        for (; dim + 15 < hidden_size; dim += 16) {
            __m512 accum = _mm512_setzero_ps();

            // Sum weighted embeddings from 4 corners
            for (int corner = 0; corner < 4; ++corner) {
                int64_t embed_idx = idx_list[corner][pos];
                __m512 v_weight = _mm512_set1_ps(weight_list[corner][pos]);

                // Load 16 bfloat16 values and convert to float
                const bf16* embed_ptr = &pos_embed_weight[embed_idx * hidden_size + dim];

                #ifdef __AVX512BF16__
                // Use hardware bfloat16 to float conversion if available
                __m256i bf16_vals_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(embed_ptr));
                __m512 float_vals = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(bf16_vals_raw));
                #else
                // Manual conversion
                float temp[16];
                for (int i = 0; i < 16; ++i) {
                    temp[i] = static_cast<float>(embed_ptr[i]);
                }
                __m512 float_vals = _mm512_loadu_ps(temp);
                #endif

                // Multiply by weight and accumulate
                accum = _mm512_fmadd_ps(float_vals, v_weight, accum);
            }

            // Convert back to bfloat16 and store
            #ifdef __AVX512BF16__
            __m256bh result_bf16 = _mm512_cvtneps_pbh(accum);
            __m256i result_raw = reinterpret_cast<__m256i>(result_bf16);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&patch_pos_embeds[pos * hidden_size + dim]),
                                result_raw);
            #else
            float temp[16];
            _mm512_storeu_ps(temp, accum);
            for (int i = 0; i < 16; ++i) {
                patch_pos_embeds[pos * hidden_size + dim + i] = bf16(temp[i]);
            }
            #endif
        }

        // Handle remaining dimensions
        for (; dim < hidden_size; ++dim) {
            float accum = 0.0f;

            // Sum weighted embeddings from 4 corners
            for (int corner = 0; corner < 4; ++corner) {
                int64_t embed_idx = idx_list[corner][pos];
                float weight = weight_list[corner][pos];

                // Access pos_embed_weight[embed_idx, dim] (row-major)
                bf16 embed_val = pos_embed_weight[embed_idx * hidden_size + dim];
                accum += static_cast<float>(embed_val) * weight;
            }

            patch_pos_embeds[pos * hidden_size + dim] = bf16(accum);
        }
    }
#else
    // Scalar fallback
    for (size_t pos = 0; pos < total_positions; ++pos) {
        for (int dim = 0; dim < hidden_size; ++dim) {
            float accum = 0.0f;

            // Sum weighted embeddings from 4 corners
            for (int corner = 0; corner < 4; ++corner) {
                int64_t embed_idx = idx_list[corner][pos];
                float weight = weight_list[corner][pos];

                // Access pos_embed_weight[embed_idx, dim] (row-major)
                bf16 embed_val = pos_embed_weight[embed_idx * hidden_size + dim];
                accum += static_cast<float>(embed_val) * weight;
            }

            patch_pos_embeds[pos * hidden_size + dim] = bf16(accum);
        }
    }
#endif

    // Split into separate patches based on h*w sizes
    std::vector<size_t> split_sizes;
    for (size_t i = 0; i < grid_thw.size(); ++i) {
        split_sizes.push_back((size_t)grid_hs[i] * grid_ws[i]);
    }

    // Prepare for permutation
    std::vector<bf16> patch_pos_embeds_permute;

    size_t offset = 0;
    for (size_t img_idx = 0; img_idx < grid_thw.size(); ++img_idx) {
        int32_t t = grid_ts[img_idx];
        int32_t h = grid_hs[img_idx];
        int32_t w = grid_ws[img_idx];
        size_t hw = (size_t)h * w;

        // Repeat for t frames
        std::vector<bf16> pos_embed_repeated(t * hw * hidden_size);
        for (int32_t ti = 0; ti < t; ++ti) {
            for (size_t pos = 0; pos < hw; ++pos) {
                for (int dim = 0; dim < hidden_size; ++dim) {
                    pos_embed_repeated[ti * hw * hidden_size + pos * hidden_size + dim] =
                        patch_pos_embeds[offset * hidden_size + pos * hidden_size + dim];
                }
            }
        }

        // Permute: view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
        //          .permute(0, 1, 3, 2, 4, 5)
        //          .flatten(0, 4)
        int32_t h_blocks = h / spatial_merge_size;
        int32_t w_blocks = w / spatial_merge_size;

        for (int32_t ti = 0; ti < t; ++ti) {
            for (int32_t hi = 0; hi < h_blocks; ++hi) {
                for (int32_t wi = 0; wi < w_blocks; ++wi) {
                    for (int32_t mh = 0; mh < spatial_merge_size; ++mh) {
                        for (int32_t mw = 0; mw < spatial_merge_size; ++mw) {
                            // Original position in repeated array
                            int32_t orig_h = hi * spatial_merge_size + mh;
                            int32_t orig_w = wi * spatial_merge_size + mw;
                            size_t orig_idx = ti * hw * hidden_size +
                                            (size_t)orig_h * w * hidden_size +
                                            (size_t)orig_w * hidden_size;

                            // Copy embedding
                            for (int dim = 0; dim < hidden_size; ++dim) {
                                patch_pos_embeds_permute.push_back(
                                    pos_embed_repeated[orig_idx + dim]
                                );
                            }
                        }
                    }
                }
            }
        }

        offset += hw;
    }

    return patch_pos_embeds_permute;
}
