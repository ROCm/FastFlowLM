// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// MiniCPM-OCR vision overlay core program: vision mm (design A) + vision attention + dispatcher.
// See ../README.md.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <type_traits>
#include <aie_api/aie.hpp>
#include "overlay_layout.h"

namespace vmm {
#include "../../vision_mm/kernel/mm.cc"
}

namespace vat {
#include "../../vision_attn/kernel/attn.cc"
}

template <typename T>
static inline T* at(int8_t* arena, int address) {
    return reinterpret_cast<T*>(arena + (address - CT_ARENA_BASE));
}

extern "C" {
void fused(int8_t* arena, int32_t col_parity) {
    volatile int32_t* words = at<volatile int32_t>(arena, CT_DISPATCH);
    acquire_greater_equal(OP_LOCK + 48, 1);
    const int op = words[0];
    const int calls = words[1];

    if (op == OP_MM) {
        aie::set_rounding(aie::rounding_mode::floor);
        for (int c = 0; c < calls; c++) {
            vmm::mm(at<bfloat16>(arena, CT_MM_X_0), at<bfloat16>(arena, CT_MM_X_1),
                    at<bfloat16>(arena, CT_MM_W_0), at<bfloat16>(arena, CT_MM_W_1),
                    at<bfloat16>(arena, CT_MM_Y_0), at<bfloat16>(arena, CT_MM_Y_1),
                    at<float>(arena, CT_MM_Y_ACC), at<float>(arena, CT_MM_B),
                    at<int>(arena, CT_MM_RTP));
        }
    } else if (op == OP_ATTN) {
        for (int c = 0; c < calls; c++) {
            vat::attn(at<bfloat16>(arena, CT_AT_O_0), at<bfloat16>(arena, CT_AT_O_1),
                      at<bfloat16>(arena, CT_AT_Q_0), at<bfloat16>(arena, CT_AT_Q_1),
                      at<bfloat16>(arena, CT_AT_Q),
                      at<bfloat16>(arena, CT_AT_IN_0), at<bfloat16>(arena, CT_AT_IN_1),
                      at<bfloat16>(arena, CT_AT_S), at<float>(arena, CT_AT_Y),
                      at<int>(arena, CT_AT_L_ATTN_CORE), col_parity);
        }
    }
}
}
