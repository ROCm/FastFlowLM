// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#define GEMM_EVEN 0
#include "gemm.h"

extern "C" {
void gemm_o(float *acc, uint8 *x0, uint8 *x1, uint8 *w_lo0, uint8 *w_lo1, uint8 *w_hi0, uint8 *w_hi1, int *rtp) {
    gemm_body(acc, x0, x1, w_lo0, w_lo1, w_hi0, w_hi1, rtp);
}
}
