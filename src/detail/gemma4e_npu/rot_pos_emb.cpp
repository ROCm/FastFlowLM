#include "rot_pos_emb.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdfloat>
#include <cstring>
#include "avx512_util.hpp"
#include <immintrin.h>
#include <math.h>

void generate_gemma4_audio_rotary_pos_emb(
    int hidden_size,
    int attention_chunk_size,
    int attention_context_left,
    int attention_context_right,

    std::vector<bf16>& position_embedding  // shape of [13, hidden_size]
){

    int context_size = attention_chunk_size + (attention_context_left-1) + attention_context_right;

    float min_timescale = 1.0;
    float max_timescale = 10000.0;

    int num_timescales = hidden_size / 2;
    float log_timescale_increment = std::log(

        max_timescale/min_timescale

    )/  std::max(  num_timescales - 1, 1) ;

    std::vector<float> inv_timescales(num_timescales);

    for(int i = 0; i < num_timescales; i++){
        inv_timescales[i] = min_timescale * std::exp(i * -log_timescale_increment);
    }

    for(int i = 0; i <= 12; i++){
        int pos_val = 12 - i;

        for(int j = 0; j < num_timescales; j++){
            float angle = pos_val * inv_timescales[j];

            position_embedding[i *hidden_size + j] = bf16(std::sin(angle));
            position_embedding[i *hidden_size + j + num_timescales] = bf16(std::cos(angle));
        }
    }
}

/**
 * Apply rotary position embeddings to Q and K in-place within mm_res buffer.
 * New Layout: [3 * num_heads, seq_len, head_dim]
 * * @param mm_res_ptr Pointer to buffer of shape [3 * num_heads, seq_len, head_dim]
 * Structure: [Block Q (Heads 0..H-1)] | [Block K (Heads 0..H-1)] | [Block V...]
 * @param cos_emb_ptr Pointer to cosine embeddings of shape [seq_len, head_dim] (float)
 * @param sin_emb_ptr Pointer to sine embeddings of shape [seq_len, head_dim] (float)
 * @param seq_len The L_Seq dimension size
 * @param hidden_size Hidden dimension size (total across all heads)
 * @param num_heads Number of attention heads
 */
void generate_gemma4_vision_rotary_pos_emb(
    const std::vector<std::vector<int>>& grid_pairs_per_image,
    const std::vector<int>& seq_len_per_image,
    const std::vector<int>& start_seq_len_index_per_image,
    int seq_len_padded,
    int head_dim,
    float theta,
    float scale,
    std::vector<bf16>& cos_emb,
    std::vector<bf16>& sin_emb
) {
    const int spatial_dim   = head_dim / 2;      // 32 for head_dim=64
    const int inv_freq_len  = spatial_dim / 2;   // 16

    // inv_freq[j] = 1.0 / (theta ^ (j*2 / spatial_dim))
    std::vector<float> inv_freq(inv_freq_len);
    for (int j = 0; j < inv_freq_len; j++) {
        inv_freq[j] = 1.0f / std::pow(theta, (float)(j * 2) / (float)spatial_dim);
    }

    cos_emb.assign(seq_len_padded * head_dim, bf16(0.0f));
    sin_emb.assign(seq_len_padded * head_dim, bf16(0.0f));

    for (int img = 0; img < (int)grid_pairs_per_image.size(); img++) {
        const int num_patches = seq_len_per_image[img];
        const int start       = start_seq_len_index_per_image[img];
        const auto& grid_pairs = grid_pairs_per_image[img];
        int compact_patch_idx = 0;

        for (int pair_idx = 0; pair_idx + 1 < (int)grid_pairs.size() && compact_patch_idx < num_patches; pair_idx += 2) {
            const int x_val = grid_pairs[pair_idx];
            const int y_val = grid_pairs[pair_idx + 1];

            // Gemma4 position ids are padded with (-1, -1). The encoder state is compacted to
            // valid patches only, so rotary embeddings must compact the valid coordinates too.
            if (x_val < 0 || y_val < 0) {
                continue;
            }

            const int global_s = start + compact_patch_idx;

            bf16* cos_row = cos_emb.data() + global_s * head_dim;
            bf16* sin_row = sin_emb.data() + global_s * head_dim;

            for (int j = 0; j < inv_freq_len; j++) {
                const float x_freq = x_val * inv_freq[j];
                const float y_freq = y_val * inv_freq[j];

                // x-axis: channels [0..inv_freq_len-1] and [inv_freq_len..spatial_dim-1] (duplication)
                cos_row[j]               = bf16(std::cos(x_freq) * scale);
                cos_row[j + inv_freq_len]= bf16(std::cos(x_freq) * scale);
                sin_row[j]               = bf16(std::sin(x_freq) * scale);
                sin_row[j + inv_freq_len]= bf16(std::sin(x_freq) * scale);

                // y-axis: channels [spatial_dim..spatial_dim+inv_freq_len-1] and [spatial_dim+inv_freq_len..head_dim-1]
                cos_row[spatial_dim + j]               = bf16(std::cos(y_freq) * scale);
                cos_row[spatial_dim + j + inv_freq_len]= bf16(std::cos(y_freq) * scale);
                sin_row[spatial_dim + j]               = bf16(std::sin(y_freq) * scale);
                sin_row[spatial_dim + j + inv_freq_len]= bf16(std::sin(y_freq) * scale);
            }

            compact_patch_idx++;
        }

        assert(compact_patch_idx == num_patches);
    }
}

void apply_multidimensional_rope(
    bf16* qkv,
    const bf16* cos_emb,
    const bf16* sin_emb,
    int seq_len,
    int num_head,
    int head_dim,
    int ndim
) {
    // Python: num_rotated_channels_per_dim = 2 * (head_dim // (2 * ndim))
    const int spatial_dim = 2 * (head_dim / (2 * ndim)); // channels per spatial dim (32 for head_dim=64, ndim=2)
    const int quarter_dim = spatial_dim / 2;              // rotate_half midpoint (16 for spatial_dim=32)

#ifdef __AVX512F__
    // Fast path: AVX-512, processes 16 bf16 values per register.
    // Requires quarter_dim == 16 (head_dim == 64).
    if (quarter_dim == 16) {
        for (int s = 0; s < seq_len; s++) {
            const bf16* cos_row = cos_emb + s * head_dim;
            const bf16* sin_row = sin_emb + s * head_dim;

            // Load cos/sin for x-spatial dim (cos[0..15], duplicated at [16..31])
            const __m512 cos_x = load_bfloat16_to_m512(cos_row);
            const __m512 sin_x = load_bfloat16_to_m512(sin_row);
            // Load cos/sin for y-spatial dim (cos[32..47], duplicated at [48..63])
            const __m512 cos_y = load_bfloat16_to_m512(cos_row + spatial_dim);
            const __m512 sin_y = load_bfloat16_to_m512(sin_row + spatial_dim);

            bf16* row = qkv + s * num_head * head_dim;
            for (int h = 0; h < num_head; h++) {
                bf16* x = row + h * head_dim;

                // --- x-spatial half: channels [0..spatial_dim-1] ---
                const __m512 x_lo = load_bfloat16_to_m512(x);
                const __m512 x_hi = load_bfloat16_to_m512(x + quarter_dim);
                // out_lo = x_lo * cos_x - x_hi * sin_x
                const __m512 out_lo = _mm512_fmsub_ps(x_lo, cos_x, _mm512_mul_ps(x_hi, sin_x));
                // out_hi = x_hi * cos_x + x_lo * sin_x
                const __m512 out_hi = _mm512_fmadd_ps(x_hi, cos_x, _mm512_mul_ps(x_lo, sin_x));
                store_m512_to_bfloat16_rne(x,              out_lo);
                store_m512_to_bfloat16_rne(x + quarter_dim, out_hi);

                // --- y-spatial half: channels [spatial_dim..head_dim-1] ---
                bf16* y = x + spatial_dim;
                const __m512 y_lo = load_bfloat16_to_m512(y);
                const __m512 y_hi = load_bfloat16_to_m512(y + quarter_dim);
                // out_lo = y_lo * cos_y - y_hi * sin_y
                const __m512 y_out_lo = _mm512_fmsub_ps(y_lo, cos_y, _mm512_mul_ps(y_hi, sin_y));
                // out_hi = y_hi * cos_y + y_lo * sin_y
                const __m512 y_out_hi = _mm512_fmadd_ps(y_hi, cos_y, _mm512_mul_ps(y_lo, sin_y));
                store_m512_to_bfloat16_rne(y,              y_out_lo);
                store_m512_to_bfloat16_rne(y + quarter_dim, y_out_hi);
            }
        }
        return;
    }
#endif // __AVX512F__

    // Scalar fallback: works for any head_dim divisible by 4.
    for (int s = 0; s < seq_len; s++) {
        const bf16* cos_row = cos_emb + s * head_dim;
        const bf16* sin_row = sin_emb + s * head_dim;

        bf16* row = qkv + s * num_head * head_dim;
        for (int h = 0; h < num_head; h++) {
            bf16* x = row + h * head_dim;

            // x-spatial half [0..spatial_dim-1]
            for (int j = 0; j < quarter_dim; j++) {
                const float x_lo = float(x[j]);
                const float x_hi = float(x[j + quarter_dim]);
                const float c    = float(cos_row[j]);
                const float s_   = float(sin_row[j]);
                x[j]              = bf16(x_lo * c - x_hi * s_);
                x[j + quarter_dim]= bf16(x_hi * c + x_lo * s_);
            }

            // y-spatial half [spatial_dim..head_dim-1]
            for (int j = 0; j < quarter_dim; j++) {
                const float y_lo = float(x[spatial_dim + j]);
                const float y_hi = float(x[spatial_dim + j + quarter_dim]);
                const float c    = float(cos_row[spatial_dim + j]);
                const float s_   = float(sin_row[spatial_dim + j]);
                x[spatial_dim + j]              = bf16(y_lo * c - y_hi * s_);
                x[spatial_dim + j + quarter_dim]= bf16(y_hi * c + y_lo * s_);
            }
        }
    }
}
