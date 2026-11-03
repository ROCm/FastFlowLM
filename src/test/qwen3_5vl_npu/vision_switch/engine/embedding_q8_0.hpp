#ifndef __EMBEDDING_Q8_0_HPP__
#define __EMBEDDING_Q8_0_HPP__
#include "buffer.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include <immintrin.h>

class embedding_q8_0{
private:
    static constexpr int Q8_0_GROUP_SIZE = 32;
    buffer<bf16> embed;
    size_t vocabe_size;
    size_t dim;

    buffer<bf16> out_buffer;

    public:
    embedding_q8_0(size_t vocab_size, size_t dim) {
        this->vocabe_size = vocab_size;
        this->dim = dim;
        this->embed = buffer<bf16>(vocab_size * dim);
        this->out_buffer = buffer<bf16>(dim);
    }

    void init_weights(Q4NX& q4nx, const std::string& weight_name){
        // Packing config (must match _pack_q8nx / dequant_q8_0 in the converter):
        //   ROW_BLOCK_SIZE=32, COL_BLOCK_SIZE=256, PARALLEL_SIZE=32, Q8_GROUP_SIZE=32
        static constexpr int m = 32;   // ROW_BLOCK_SIZE
        static constexpr int k = 256;  // COL_BLOCK_SIZE
        static constexpr int parallel_size = 32;
        static constexpr int q8_group_size = Q8_0_GROUP_SIZE; // 32

        static constexpr int c = k / q8_group_size;      // 8 scale groups per block col
        static constexpr int r = m;                      // 32 rows per block
        static constexpr int r_div_parallel = r / parallel_size; // 1

        // Per-block byte layout: [ scales_bf16 (c*r*2) | data_int8 (r*k) ]
        static constexpr size_t scale_bytes = (size_t)c * r * sizeof(bf16); // 512
        static constexpr size_t data_bytes  = (size_t)r * k;                // 8192
        static constexpr size_t block_bytes = scale_bytes + data_bytes;     // 8704

        buffer<u8> lm_head_w;
        q4nx.load_weights(lm_head_w, weight_name);

        // dequantize the lm_head_w to get the embedding, for checkpoints that tie the embedding and lm_head.
        // The packed weight is padded up to a whole number of (m x k) blocks on both dims
        // (rows padded to vocab_size_padded, cols padded to a multiple of k), so derive the
        // block grid from the buffer and only write back the valid [vocabe_size, dim] region.
        size_t b_cols = (dim + k - 1) / k;
        assert(lm_head_w.size() % (b_cols * block_bytes) == 0);
        size_t b_rows = lm_head_w.size() / (b_cols * block_bytes);
        assert(b_rows * m >= vocabe_size);

        const u8* base = lm_head_w.begin();

        for (size_t br = 0; br < b_rows; br++){
            for (size_t bc = 0; bc < b_cols; bc++){
                const u8* blk = base + (br * b_cols + bc) * block_bytes;
                const bf16* scales = reinterpret_cast<const bf16*>(blk);
                const int8_t* data = reinterpret_cast<const int8_t*>(blk + scale_bytes);

                for (int ri = 0; ri < r; ri++){
                    const size_t global_row = br * (size_t)m + ri;
                    if (global_row >= vocabe_size) break; // skip padded rows

                    // data was reordered '(r_div_parallel parallel) c -> (r_div_parallel c parallel)'
                    const int rp = ri / parallel_size;   // 0..r_div_parallel-1
                    const int p  = ri % parallel_size;   // 0..parallel_size-1
                    bf16* out_row = this->embed.begin() + global_row * this->dim + bc * (size_t)k;

                    const size_t col_base = bc * (size_t)k;
                    for (int ci = 0; ci < k; ci++){
                        if (col_base + ci >= this->dim) break; // skip padded cols
                        // scales packed as (c r) within block -> flat = group*r + ri
                        const int group = ci / q8_group_size;
                        const float scale = (float)scales[group * r + ri];

                        // data packed as (rp, col, p) within block
                        const size_t d_idx = (size_t)rp * (k * parallel_size)
                                           + (size_t)ci * parallel_size + p;
                        out_row[ci] = (bf16)(scale * (float)data[d_idx]);
                    }
                }
            }
        }
        (void)r_div_parallel;
    }

    /// \brief Forward pass
    /// \param x the input tensor
    /// \return the output tensor
    vdtype forward(int x){
        this->out_buffer.copy_from(this->embed.begin() + (size_t)x * this->dim, this->dim);
        return this->out_buffer;
    }

    /// \brief Forward pass
    /// \param x the input tensor
    /// \param y the output tensor
    /// \return the output tensor
    vdtype forward(int x, vdtype& y){
        y.copy_from(this->embed.begin() + (size_t)x * this->dim, this->dim);
        return y;
    }
};

#endif
