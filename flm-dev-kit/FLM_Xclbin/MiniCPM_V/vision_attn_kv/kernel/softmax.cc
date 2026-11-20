// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <aie_api/aie.hpp>
#include "common.h"

// exp(s - m) in place over N 16x16 blocks. The blocks are contiguous 16-element rows, row t of every block being
// vector t mod 16, so this is one flat loop of 16 N vectors reading each row's max from a pre-broadcast table.
// Per element the same ops as attn_kernel.h's update() (sub, lower clamp, * log2 e, exp2): bit-identical.
//
// Compiled on its own with -mllvm --aie-force-postpipeliner (kernel.mk). That flag, the post-increment iterators and
// pS being __restrict together make the loop II 3 (one vector per 3 bundles); drop any one and it stays serial at
// ~23 bundles per vector (L=1056 kernel 3278 -> 2627 us).
alignas(aie::vector_decl_align) static bf16 m_bcast[16 * 16];

extern "C" void attn_softmax(bf16*__restrict pS, const bf16* new_m_local, int N) {
    for (int t = 0; t < 16; t++)
        aie::store_v(m_bcast + t * 16, aie::broadcast<bf16, 16>(new_m_local[t]));
    constexpr bf16 min_clamp = -87.0f;
    const aie::vector<bf16, 16> lo = aie::broadcast<bf16, 16>(min_clamp);
    const int n = N * 16;
    auto in = aie::begin_vector<16>(pS);
    auto out = aie::begin_vector<16>(pS);
    _Pragma("clang loop min_iteration_count(16)")
    for (int v = 0; v < n; v++){
        aie::vector<bf16, 16> V = aie::sub(*in++, aie::load_v<16>(m_bcast + (v & 15) * 16));
        V = aie::max(V, lo);
        aie::accum<accfloat, 16> A = aie::mul(V, exp_scale);
        *out++ = aie::exp2<bf16>(A.template to_vector<float>());
    }
}
