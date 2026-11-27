// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_vision_seq.hpp
/// \brief the control sequence of the vision attention kernel for one image of the Inkling-2 tower
///        (16 heads x 64, q / k / v side by side in one [L][3 * DIM] buffer, output [L][DIM])
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "npu_sequences/image_attention_sequence.hpp"
#include "typedef.hpp"

namespace inkling2_vision {

/// \brief tile shape of the kernel (fixed by the xclbin)
struct attn_shape {
    static constexpr int L_PAD = 512;        // L and S are padded to this; the real S is masked
    static constexpr uint32_t COLS = 8, ROWS = 4, CU_MODE = 2;
    static constexpr uint32_t LQ_PER_CT = 32, LK_PER_CT = 512, LQ_INT = 32, LK_INT = 32;
};

/// \brief fill seq for one image of L tokens (L = 4 * N + 1): the attention of every head over all L rows
inline void gen_attention_sequence(npu_sequence* seq, int L, int heads, int head_dim) {
    const int dim = heads * head_dim;
    const int l_pad = (L + attn_shape::L_PAD - 1) / attn_shape::L_PAD * attn_shape::L_PAD;
    const uint32_t total = (uint32_t)l_pad;

    AttentionConfig config = {
        (uint32_t)head_dim, (uint32_t)heads,
        attn_shape::LQ_PER_CT, attn_shape::LK_PER_CT,
        attn_shape::LQ_INT, attn_shape::LK_INT,
        attn_shape::COLS, attn_shape::ROWS,
        attn_shape::CU_MODE,
        1u,
        1, 1, 1, 0,   // q, k, v come from argument 1 (the qkv buffer), o goes to argument 0
        (uint32_t)(3 * dim), (uint32_t)(3 * dim), (uint32_t)(3 * dim),
        (uint32_t)dim,
        false,
        total, total, total};

    const std::vector<int> l_list{l_pad}, s_list{L}, s_pad_list{l_pad}, q_off{0}, k_off{dim}, v_off{2 * dim}, o_off{0};
    BatchMetadata batch = {l_list, s_list, s_pad_list, q_off, k_off, v_off, o_off};

    setup_SHM_configuration<bf16, bf16>(*seq, config, batch, 1.0 / std::sqrt((double)head_dim));
}

}  // namespace inkling2_vision
