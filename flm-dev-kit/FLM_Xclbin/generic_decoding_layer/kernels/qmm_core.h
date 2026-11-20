// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __QMM_CORE_H__
#define __QMM_CORE_H__

#include "model_spec.h"

/// @brief The quantized MVM kernel for the model's QUANT_TYPE.
///
/// Every format exposes the same four calls -- qmm_accum_t, _QMM_zero(),
/// _QMM_x_bf16<m, k>(), _QMM_finalize() -- so proj_main.cc includes this header
/// and never branches on the format. q4_core.h covers the three 4-bit formats,
/// q8_core.h the 8-bit one.

/// @brief Two-bf16-limb splits on the accumulators that reach the output.
///
/// 0 (default) rounds a group sum to bf16 on its way into the scale MAC, the way
/// this kernel always has: 8 mantissa bits, ~2e-03 relative, taken once per group
/// and correlated across the groups that share a scale. 1 splits it into two bf16
/// limbs instead (~17 bits) and does the same on the min path, at ~28% more
/// cycles in the projection loop -- measured on QWEN3_0_6B Q4_1, 1228 vs 963
/// bundles per row pass. It is off by default because the projections are what
/// the decode rate is made of, and at 0 this kernel schedules to the same 963
/// bundles and the same opcode mix as the hand-written q4_k.h it replaced. Set it
/// to 1 in a model header if that model needs the bits.
#ifndef QMM_LIMB_PRECISION
#define QMM_LIMB_PRECISION 0
#endif

#if QUANT_TYPE == __Q8_0__
#include "q8_core.h"
#elif QUANT_TYPE == __Q41_Q80_MIX__
#include "mix_core.h"
#else
#include "q4_core.h"
#endif

#endif // __QMM_CORE_H__
