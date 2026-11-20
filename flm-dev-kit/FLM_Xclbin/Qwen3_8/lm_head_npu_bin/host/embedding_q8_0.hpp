// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __EMBEDDING_Q8_0_HPP__
#define __EMBEDDING_Q8_0_HPP__
#include "buffer.hpp"
#include "typedef.hpp"
#include <cassert>

/// \brief Recover embedding rows by dequantizing the packed q8_0 lm_head.
/// \note Newer converters tie the word embeddings and emit only the quantized
///       lm_head, so model.embed_tokens.weight is simply absent. Reading the
///       embedding back out of lm_head.weight works either way, which is why
///       the DLL (FLM_DLL/detail/qwen3_5vl_npu/embedding_q8_0.hpp) does the
///       same. This version dequantizes one row at a time instead of building
///       the whole [vocab, dim] table: the test only ever looks up a handful of
///       tokens, and the full table would be over a gigabyte.
class embedding_q8_0 {
private:
    // Packing config, must match _pack_q8nx / dequant_q8_0 in the converter.
    static constexpr int m             = 32;   //!< ROW_BLOCK_SIZE
    static constexpr int k             = 256;  //!< COL_BLOCK_SIZE
    static constexpr int parallel_size = 32;
    static constexpr int q8_group_size = 32;

    static constexpr int    c           = k / q8_group_size;              //!< scale groups per block col
    static constexpr size_t scale_bytes = (size_t)c * m * sizeof(bf16);   //!< 512
    static constexpr size_t data_bytes  = (size_t)m * k;                  //!< 8192
    static constexpr size_t block_bytes = scale_bytes + data_bytes;       //!< 8704

    const u8* base;
    size_t    vocab_size;
    size_t    dim;
    size_t    b_cols;

public:
    /// \param packed lm_head.weight exactly as it comes out of the q4nx file,
    ///        i.e. before reorder_cpy_q80 shuffles it for the NPU
    embedding_q8_0(buffer<u8>& packed, size_t vocab_size, size_t dim)
        : base(packed.begin()), vocab_size(vocab_size), dim(dim)
    {
        // The packed weight is padded to whole (m x k) blocks on both dims, so
        // derive the block grid from the buffer rather than from vocab_size.
        this->b_cols = (dim + k - 1) / k;
        assert(packed.size() % (this->b_cols * block_bytes) == 0);
        assert(packed.size() / (this->b_cols * block_bytes) * m >= vocab_size);
    }

    /// \brief Dequantize one vocab row into out[0..dim).
    void forward(int token, bf16* out) const
    {
        assert(token >= 0 && (size_t)token < this->vocab_size);
        const size_t br = (size_t)token / m;
        const int    ri = token % m;
        // data was reordered '(r_div_parallel parallel) c -> (r_div_parallel c parallel)';
        // r_div_parallel is 1 here, so rp is always 0 and p is just the row.
        const int    p  = ri % parallel_size;

        for (size_t bc = 0; bc < this->b_cols; bc++) {
            const u8*     blk    = this->base + (br * this->b_cols + bc) * block_bytes;
            const bf16*   scales = reinterpret_cast<const bf16*>(blk);
            const int8_t* data   = reinterpret_cast<const int8_t*>(blk + scale_bytes);

            const size_t col_base = bc * (size_t)k;
            for (int ci = 0; ci < k; ci++) {
                if (col_base + ci >= this->dim) break; // skip padded cols
                // scales packed as (c m) within the block -> flat = group * m + ri
                const float scale = (float)scales[(ci / q8_group_size) * m + ri];
                out[col_base + ci] = (bf16)(scale * (float)data[(size_t)ci * parallel_size + p]);
            }
        }
    }
};

#endif
