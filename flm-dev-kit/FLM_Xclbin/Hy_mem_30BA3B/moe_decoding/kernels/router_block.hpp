// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __ROUTER_BLOCK_HPP__
#define __ROUTER_BLOCK_HPP__

#include "aie_array_layout.h"
#include "mvm_col_major.hpp"

// The router weight arrives in blocks of router_weight_row_block (32) experts x
// router_weight_col_block (256) inputs, column major (the 32 experts of one input
// adjacent). The router CT takes experts 0..63 from IT5 channel 0 and the router
// helper takes 64..127 from IT5 channel 1; both run this per block.
//
// logits[0..31] += block * x[0..255], in fp32.
// Done as two 16-row halves with the 16-row matvec that was already in use (a
// 32-row instantiation would be new code in the router CT, which has ~330 B of
// program memory left).
static inline void router_block_matvec(bfloat16* block, bf16* x, float* logits)
{
    constexpr int R = router_weight_row_block;
    constexpr int C = router_weight_col_block;
    matvec_vectorized_col_major<R / 2, C, R / 2, bfloat16, bfloat16, float, R>(block, x, logits);
    matvec_vectorized_col_major<R / 2, C, R / 2, bfloat16, bfloat16, float, R>(block + R / 2, x, logits + R / 2);
}

#endif // __ROUTER_BLOCK_HPP__
