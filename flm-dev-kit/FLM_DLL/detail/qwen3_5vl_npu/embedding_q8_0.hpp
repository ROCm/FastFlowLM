// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __EMBEDDING_Q8_0_HPP__
#define __EMBEDDING_Q8_0_HPP__
#include "buffer.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_utils/mapped_file.hpp"
#include "tensor_2d.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cassert>
#include <vector>

class embedding_q8_0{
private:
    static constexpr int Q8_0_GROUP_SIZE = 32;
    // Packing config (must match _pack_q8nx / dequant_q8_0 in the converter):
    //   ROW_BLOCK_SIZE=32, COL_BLOCK_SIZE=256, PARALLEL_SIZE=32, Q8_GROUP_SIZE=32
    static constexpr int m = 32;   // ROW_BLOCK_SIZE
    static constexpr int k = 256;  // COL_BLOCK_SIZE
    static constexpr int parallel_size = 32;
    // Per-block byte layout: [ scales_bf16 (c*r*2) | data_int8 (r*k) ]
    static constexpr size_t scale_bytes = (size_t)(k / Q8_0_GROUP_SIZE) * m * sizeof(bf16); // 512
    static constexpr size_t data_bytes  = (size_t)m * k;                                     // 8192
    static constexpr size_t block_bytes = scale_bytes + data_bytes;                          // 8704

    size_t vocabe_size;
    size_t dim;
    size_t b_cols = 0;

    // The tied lm_head is only read one row per token, so it is mapped from model.q4nx
    // instead of copied (and instead of dequantized into a vocab x dim bf16 table).
    // owned is the fallback when the range cannot be mapped.
    mapped_file mapped;
    buffer<u8> owned;
    const u8* base = nullptr;

    buffer<bf16> out_buffer;

    void dequant_row(size_t row, bf16* dst) const {
        const size_t br = row / m;
        const int ri = (int)(row % m);
        for (size_t bc = 0; bc < b_cols; bc++){
            const u8* blk = base + (br * b_cols + bc) * block_bytes;
            const bf16* scales = reinterpret_cast<const bf16*>(blk);
            // data was reordered '(r_div_parallel parallel) c -> (r_div_parallel c parallel)'
            // with r_div_parallel == 1, so element ci of this row sits at ci * parallel_size + ri.
            const int8_t* data = reinterpret_cast<const int8_t*>(blk + scale_bytes) + ri;
            const size_t col_base = bc * (size_t)k;
            const size_t n = std::min((size_t)k, dim - col_base); // skip padded cols
            for (size_t g = 0; g * Q8_0_GROUP_SIZE < n; g++){
                // scales packed as (c r) within block -> flat = group*r + ri
                const float scale = (float)scales[g * m + ri];
                const size_t ce = std::min((size_t)Q8_0_GROUP_SIZE, n - g * Q8_0_GROUP_SIZE);
                for (size_t cj = 0; cj < ce; cj++){
                    const size_t ci = g * Q8_0_GROUP_SIZE + cj;
                    dst[col_base + ci] = (bf16)(scale * (float)data[ci * parallel_size]);
                }
            }
        }
    }

    public:
    embedding_q8_0(size_t vocab_size, size_t dim) {
        this->vocabe_size = vocab_size;
        this->dim = dim;
        this->b_cols = (dim + k - 1) / k;
        this->out_buffer = buffer<bf16>(dim);
    }

    /// \brief Bind the packed lm_head weight of the tied checkpoint as the embedding table.
    /// \param q4nx the opened model.q4nx
    /// \param q4nx_path path of that file, for mapping
    /// \param weight_name the packed q8_0 tensor (lm_head.weight)
    void init_weights(Q4NX& q4nx, const std::string& q4nx_path, const std::string& weight_name){
        tensor_metadata meta = q4nx.get_tensor_metadata(weight_name);
        const size_t nbytes = meta.offsets[1] - meta.offsets[0];
        if (this->mapped.open(q4nx_path, meta.offsets[0], nbytes)){
            this->base = this->mapped.data();
        }
        else{
            q4nx.load_weights(this->owned, weight_name);
            this->base = this->owned.begin();
        }
        // The packed weight is padded up to a whole number of (m x k) blocks on both dims
        // (rows padded to vocab_size_padded, cols padded to a multiple of k).
        assert(nbytes % (b_cols * block_bytes) == 0);
        assert(nbytes / (b_cols * block_bytes) * m >= vocabe_size);
    }

    /// \brief Hint that these token rows are about to be looked up (a prompt about to be prefilled).
    /// \note Matters only when the mapped pages were dropped (memory pressure): the rows of a
    ///       whole prompt are then read in one batch instead of faulting in one page at a time.
    ///       The 32 rows of a block row share one contiguous b_cols * block_bytes range.
    void prefetch(const int* ids, size_t n){
        if (!this->mapped) return;
        size_t last = (size_t)-1;
        std::vector<size_t> block_rows;
        block_rows.reserve(n);
        for (size_t i = 0; i < n; i++) block_rows.push_back((size_t)ids[i] / m);
        std::sort(block_rows.begin(), block_rows.end());
        for (size_t br : block_rows){
            if (br == last) continue;
            last = br;
            this->mapped.will_need(br * b_cols * block_bytes, b_cols * block_bytes);
        }
    }

    /// \brief Forward pass
    /// \param x the input tensor
    /// \return the output tensor
    vdtype forward(int x){
        this->dequant_row((size_t)x, this->out_buffer.begin());
        return this->out_buffer;
    }

    /// \brief Forward pass
    /// \param x the input tensor
    /// \param y the output tensor
    /// \return the output tensor
    vdtype forward(int x, vdtype& y){
        this->dequant_row((size_t)x, this->out_buffer.begin());
        y.copy_from(this->out_buffer.begin(), this->dim);
        return y;
    }
};

#endif
