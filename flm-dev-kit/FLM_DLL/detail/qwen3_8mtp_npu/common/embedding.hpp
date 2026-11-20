// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/embedding.hpp
/// \brief Q8_0 token-embedding lookup, shared by the base model and the MTP
///        head (mtp_use_dedicated_embeddings is false -- there is no
///        mtp.embed_tokens in the checkpoint, so both read this one table).
#pragma once

#include "common/cpu_ref_common.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cpu_ref {

/// model.embed_tokens.weight as [7760][20] blocks of Q8_0, i.e. 248320 x 5120.
/// Rows are dequantised on demand: the packed table is 1.35 GB and a prefill
/// touches at most a few hundred of its 248320 rows.
class embedding_table {
public:
    explicit embedding_table(const safetensors& q4nx,
                             const std::string& name = "model.embed_tokens.weight")
        : w_(get_q4nx(q4nx, name)) {
        if (w_.fmt != q4nx_format::q8_0)
            throw std::runtime_error("embed_tokens: expected Q8_0, got " +
                                     std::string(format_name(w_.fmt)));
    }

    int rows() const { return w_.rows; }
    int cols() const { return w_.cols; }

    /// One row of the table into out[cols()].
    /// Carried across from MTP_decoding/host/cpu_ref_mtp_head.cpp:204.
    void embed_row(int32_t token, float* out) const {
        if (token < 0 || token >= w_.rows)
            throw std::runtime_error("embed_row: token " + std::to_string(token) +
                                     " out of range");
        const int rb = token / Q4NX_ROW_BLOCK;
        const int rl = token % Q4NX_ROW_BLOCK;
        std::vector<float> tile((size_t)Q4NX_COL_BLOCK * Q4NX_ROW_BLOCK);
        for (int cb = 0; cb < w_.ncb; cb++) {
            w_.dequant_block(rb, cb, tile.data());
            float* dst = out + (size_t)cb * Q4NX_COL_BLOCK;
            for (int c = 0; c < Q4NX_COL_BLOCK; c++)
                dst[c] = tile[(size_t)c * Q4NX_ROW_BLOCK + rl];
        }
    }

    /// L rows into out[L][cols()], row-major.
    void embed_rows(const int32_t* tokens, int L, float* out) const {
        const int D = w_.cols;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < L; i++)
            embed_row(tokens[i], out + (size_t)i * D);
    }

private:
    q4nx_tensor w_;
};

} // namespace cpu_ref
