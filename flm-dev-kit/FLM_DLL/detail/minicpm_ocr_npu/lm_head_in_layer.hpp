// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file lm_head_in_layer.hpp
/// \brief lm_head weights for layer_with_lm_head.xclbin (FLM_Xclbin/minicpm_ocr/decoding), see README.md
///        "lm_head inside layer.xclbin".
#pragma once

#include <cstring>
#include "minicpm_ocr_model_def.hpp"

namespace minicpm_ocr_lm_head {

constexpr int streams = 8;
constexpr int rows_per_round = 512;

inline int rows(const minicpm_ocr_desc& desc) {
    return (desc.vocab_size + rows_per_round - 1) / rows_per_round * rows_per_round;
}

inline size_t padded_block_bytes() {
    return get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, flm_q4k);
}

inline size_t weight_bytes(const minicpm_ocr_desc& desc) {
    const size_t half_blocks_per_row_block = 2 * (desc.D / QXNX_COL_BLOCK_SIZE);
    return (size_t)rows(desc) / QXNX_ROW_BLOCK_SIZE * half_blocks_per_row_block * padded_block_bytes();
}

inline void reorder_cpy(const minicpm_ocr_desc& desc, u8* dst, const buffer<u8>& src) {
    const size_t chunk = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, flm_q80);
    const size_t padded_block = padded_block_bytes();
    const size_t scale_bytes = QXNX_ROW_BLOCK_SIZE * (QXNX_COL_BLOCK_SIZE / 32) * sizeof(bf16);
    const size_t half_scale_bytes = scale_bytes / 2;
    const size_t half_q_bytes = (chunk - scale_bytes) / 2;
    const int chunks_per_row_block = desc.D / QXNX_COL_BLOCK_SIZE;
    const int stored_row_blocks = src.size() / chunk / chunks_per_row_block;
    const int rounds = rows(desc) / rows_per_round;
    const int row_blocks_per_pair = 2;

    memset(dst, 0, weight_bytes(desc));
    u8* dst_ptr = dst;
    for (int stream = 0; stream < streams; stream++) {
        for (int r = 0; r < rounds; r++) {
            const int pair = r * streams + stream;
            for (int c = 0; c < chunks_per_row_block; c++) {
                for (int half = 0; half < 2; half++) {
                    for (int i = 0; i < row_blocks_per_pair; i++) {
                        const int row_block = pair * row_blocks_per_pair + i;
                        if (row_block < stored_row_blocks) {
                            const u8* src_chunk = src.data() + ((size_t)row_block * chunks_per_row_block + c) * chunk;
                            memcpy(dst_ptr, src_chunk + half * half_scale_bytes, half_scale_bytes);
                            memcpy(dst_ptr + half_scale_bytes, src_chunk + scale_bytes + half * half_q_bytes, half_q_bytes);
                        }
                        dst_ptr += padded_block;
                    }
                }
            }
        }
    }
}

} // namespace minicpm_ocr_lm_head
