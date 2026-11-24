// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file sinusoids_position_embedding.hpp
/// \brief Sinusoidal positional embedding for the audio encoder.
///
/// Matches Python:
///   class SinusoidsPositionEmbedding(nn.Module):
///       def __init__(self, length, channels, max_timescale=10000):
///           log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
///           inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
///           scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
///           positional_embedding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
///       def forward(self, seqlen):
///           return self.positional_embedding[:seqlen, :]
#pragma once
#include <cmath>
#include <vector>
#include <cstdint>
#include <cassert>
#include <immintrin.h>
#include <omp.h>
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "../avx512_util.hpp"

/// Precomputed sinusoidal position embedding table, shape [max_len, channels].
/// Row-major: embedding for position p starts at table[p * channels].
class SinusoidsPositionEmbedding {
public:
    SinusoidsPositionEmbedding() = default;

    /// Build the table.
    /// \param length        Maximum sequence length (max_source_positions).
    /// \param channels      Embedding dimension (d_model). Must be even.
    /// \param max_timescale Whisper default is 10000.
    void build(uint32_t length, uint32_t channels, float max_timescale = 10000.f) {
        assert(channels % 2 == 0 && "SinusoidsPositionEmbedding requires even channel count");
        length_   = length;
        channels_ = channels;
        table_.resize(static_cast<size_t>(length) * channels);

        const uint32_t half = channels / 2;
        // log_timescale_increment = log(max_timescale) / (half - 1)
        const float log_ts_inc = std::log(max_timescale) / static_cast<float>(half - 1);

        for (uint32_t p = 0; p < length; ++p) {
            float* row = table_.data() + static_cast<size_t>(p) * channels;
            for (uint32_t i = 0; i < half; ++i) {
                // inv_timescale = exp(-log_ts_inc * i)
                float inv_ts  = std::exp(-log_ts_inc * static_cast<float>(i));
                float angle   = static_cast<float>(p) * inv_ts;
                row[i]        = std::sin(angle);          // sin half
                row[i + half] = std::cos(angle);          // cos half
            }
        }
    }

    /// Add positional embedding in-place, matching Python:
    ///   positional_embedding[:t, :].unsqueeze(0)  broadcast-added to [b, t, d]
    /// \p data is row-major [num_blocks * seqlen_per_block, channels] in bf16.
    /// The same positions 0..seqlen_per_block-1 are added to every block.
    /// Uses AVX-512 (16 bf16 lanes at a time) + OpenMP across blocks.
    void apply(bf16* data, uint32_t seqlen_per_block, uint32_t num_blocks) const {
        assert(seqlen_per_block <= length_);
        const uint32_t vec_width  = 16;                   // AVX-512: 16 floats
        const uint32_t c_aligned  = (channels_ / vec_width) * vec_width;

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
            for (uint32_t p = 0; p < seqlen_per_block; ++p) {
                const float* pe  = table_.data() + static_cast<size_t>(p) * channels_;
                bf16*        out = data + (static_cast<size_t>(b) * seqlen_per_block + p) * channels_;

                uint32_t c = 0;
                for (; c < c_aligned; c += vec_width) {
                    __m512 v_out = load_bfloat16_to_m512(out + c);
                    __m512 v_pe  = _mm512_loadu_ps(pe + c);
                    store_m512_to_bfloat16_rne(out + c, _mm512_add_ps(v_out, v_pe));
                }
                // scalar tail for channels not divisible by 16
                for (; c < channels_; ++c)
                    out[c] = static_cast<float>(out[c]) + pe[c];
            }
        }
    }

    /// Read-only access to a single row of the table (fp32).
    const float* row(uint32_t pos) const {
        assert(pos < length_);
        return table_.data() + static_cast<size_t>(pos) * channels_;
    }

    uint32_t length()   const { return length_;   }
    uint32_t channels() const { return channels_; }

private:
    uint32_t          length_   = 0;
    uint32_t          channels_ = 0;
    std::vector<float> table_;   // [length, channels], fp32
};
