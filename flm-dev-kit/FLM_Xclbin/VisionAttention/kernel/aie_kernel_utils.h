/*
    Copyright (C) 2014 - 2022 Xilinx, Inc. All rights reserved.
    Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
    Portions of this file consist of AI-generated content.
    SPDX-License-Identifier: MIT
*/

#ifndef _AIE_KERNEL_UTILS_
#define _AIE_KERNEL_UTILS_

#if defined(__chess__)
#define AIE_LOOP_UNROLL(x) [[chess::unroll_loop(x)]]
#define AIE_LOOP_UNROLL_FULL [[chess::unroll_loop()]]
#define AIE_LOOP_NO_UNROLL [[chess::no_unroll]]
#define AIE_LOOP_MIN_ITERATION_COUNT(x) [[chess::min_loop_count(x)]]
#define AIE_LOOP_MAX_ITERATION_COUNT(x) [[chess::max_loop_count(x)]]
#define AIE_LOOP_RANGE(a, ...)                                                 \
  [[chess::min_loop_count(a)]] __VA_OPT__(                                     \
      [[chess::max_loop_count(__VA_ARGS__)]])
#define AIE_PREPARE_FOR_PIPELINING [[chess::prepare_for_pipelining]]
#define AIE_NO_PREPARE_FOR_PIPELINING [[chess::no_prepare_for_pipelining]]
#define AIE_MODULO_SCHEDULING_BUDGET_RATIO(x)                                  \
  [[chess::modulo_scheduling_budget_ratio(x)]]
#define AIE_KEEP_SW_LOOP [[chess::keep_sw_loop]]
#define AIE_PEEL_PIPELINED_LOOP(x) [[chess::peel_pipelined_loop(x)]]
#define AIE_KEEP_FREE_FOR_PIPELINING(x) [[chess::keep_free_for_pipelining(x)]]
#define AIE_ALLOCATE(x) [[chess::allocate(x)]]
#define AIE_NO_HW_LOOP [[chess::no_hw_loop]]
#define AIE_TRY_INITIATION_INTERVAL(x)
#define AIE_PREPARE_FOR_POSTPIPELINING
#define AIE_LOOP_FLATTEN chess_flatten_loop

#elif defined(__AIECC__)
#ifndef __STRINGIFY
#define __STRINGIFY(a) #a
#endif
#define AIE_LOOP_UNROLL(x) _Pragma(__STRINGIFY(clang loop unroll_count(x)))
#define AIE_LOOP_UNROLL_FULL _Pragma("clang loop unroll(full)")
#define AIE_LOOP_NO_UNROLL _Pragma("clang loop unroll(disable)")
#define AIE_LOOP_MIN_ITERATION_COUNT(x)                                        \
  _Pragma(__STRINGIFY(clang loop min_iteration_count(x)))
#define AIE_LOOP_MAX_ITERATION_COUNT(x)                                        \
  _Pragma(__STRINGIFY(clang loop max_iteration_count(x)))
#define AIE_LOOP_RANGE(a, ...)                                                 \
  AIE_LOOP_MIN_ITERATION_COUNT(a)                                              \
  __VA_OPT__(AIE_LOOP_MAX_ITERATION_COUNT(__VA_ARGS__))
#define AIE_PREPARE_FOR_PIPELINING
#define AIE_NO_PREPARE_FOR_PIPELINING
#define AIE_MODULO_SCHEDULING_BUDGET_RATIO(x)
#define AIE_KEEP_SW_LOOP
#define AIE_PEEL_PIPELINED_LOOP(x)
#define AIE_KEEP_FREE_FOR_PIPELINING(x)
#define AIE_ALLOCATE(x)
#define AIE_NO_HW_LOOP
#define AIE_TRY_INITIATION_INTERVAL(x)                                         \
  _Pragma(__STRINGIFY(clang loop pipeline_initiation_interval(x)))
#define AIE_PREPARE_FOR_POSTPIPELINING _Pragma("clang loop pipeline(disable)")
#define AIE_LOOP_FLATTEN

#else
#define AIE_LOOP_UNROLL(x)
#define AIE_LOOP_UNROLL_FULL
#define AIE_LOOP_NO_UNROLL
#define AIE_LOOP_MIN_ITERATION_COUNT(x)
#define AIE_LOOP_MAX_ITERATION_COUNT(x)
#define AIE_LOOP_RANGE(a, ...)
#define AIE_PREPARE_FOR_PIPELINING
#define AIE_NO_PREPARE_FOR_PIPELINING
#define AIE_MODULO_SCHEDULING_BUDGET_RATIO(x)
#define AIE_KEEP_SW_LOOP
#define AIE_PEEL_PIPELINED_LOOP(x)
#define AIE_KEEP_FREE_FOR_PIPELINING(x)
#define AIE_ALLOCATE(x)
#define AIE_NO_HW_LOOP
#define AIE_TRY_INITIATION_INTERVAL(x)
#define AIE_PREPARE_FOR_POSTPIPELINING
#define AIE_LOOP_FLATTEN
#endif

// Loop hints that only Peano needs. AIE_PEANO_NO_PIPELINE keeps the
// MachinePipeliner off a loop it miscompiles; AIE_PEANO_UNROLL_FULL unrolls
// loops whose vector insert/extract index is the induction variable, which
// Peano otherwise lowers to a jump table per element.
#if defined(__chess__) || !defined(__AIECC__)
#define AIE_PEANO_NO_PIPELINE
#define AIE_PEANO_UNROLL_FULL
#else
#define AIE_PEANO_NO_PIPELINE _Pragma("clang loop pipeline(disable)")
#define AIE_PEANO_UNROLL_FULL _Pragma("clang loop unroll(full)")
#endif

// Scalar float add for the softmax row sums. chess calls IEEE
// round-to-nearest-even softfloat for it; Peano runs it on the vector unit,
// which rounds some sums differently. The operands are sums of exponentials
// (finite and >= 0), so under Peano an inline IEEE add for that case gives
// chess's results without the cost of a softfloat call.
#if defined(__chess__)
#define AIE_FADD_NONNEG(a, b) ((a) + (b))
#else
#include <stdint.h>
__attribute__((always_inline)) inline float aie_fadd_nonneg(float fx, float fy) {
    uint32_t x = __builtin_bit_cast(uint32_t, fx), y = __builtin_bit_cast(uint32_t, fy);
    uint32_t a = x > y ? x : y, b = x > y ? y : x;
    int ea = a >> 23, eb = b >> 23;
    // 6 extra low bits hold guard/round/sticky through alignment and carry
    uint32_t ma = ((a & 0x7fffff) | (ea ? 0x800000u : 0u)) << 6;
    uint32_t mb = ((b & 0x7fffff) | (eb ? 0x800000u : 0u)) << 6;
    ea = ea ? ea : 1;
    eb = eb ? eb : 1;
    int d = ea - eb;
    d = d < 31 ? d : 31;
    mb = (mb >> d) | ((mb & ((1u << d) - 1)) != 0);
    uint32_t m = ma + mb;
    uint32_t c = m >> 30;
    m = (m >> c) | (m & c);
    ea += c;
    uint32_t rem = m & 0x3f;
    m >>= 6;
    m += (rem > 0x20) | ((rem == 0x20) & (m & 1));
    c = m >> 24;
    m >>= c;
    ea += c;
    uint32_t r = (m & 0x800000) ? (((uint32_t)(ea - 1) << 23) + m) : m;
    r = ea >= 255 ? 0x7f800000u : r;
    return __builtin_bit_cast(float, r);
}
#define AIE_FADD_NONNEG(a, b) aie_fadd_nonneg((a), (b))
#endif

#endif