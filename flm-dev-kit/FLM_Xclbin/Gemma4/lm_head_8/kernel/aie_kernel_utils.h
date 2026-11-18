/*
    Copyright (C) 2014 - 2022 Xilinx, Inc. All rights reserved.
    Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
    SPDX-License-Identifier: MIT
*/

#ifndef _AIE_KERNEL_UTILS_
#define _AIE_KERNEL_UTILS_

// Peano's CDO loader does not zero .bss, so a zero-initialized static that must
// carry its value across kernel invocations has to be forced into .data
// (AIE_PERSIST_ZERO); and a static that does NOT need to persist is better left
// as an automatic (AIE_TRANSIENT), because the Peano ldscript's data region is
// tight. chess zeroes .bss and has room, so both macros are inert there and the
// chess build is byte-for-byte what it was.

// Some kernels still use the raw chess loop pragmas directly. Peano (llvm-aie,
// non-chess) does not provide them; make them no-ops so those kernels compile.
#ifndef __chess__
#define chess_prepare_for_pipelining
#define chess_loop_range(...)
#define chess_flatten_loop
#endif

#if defined(__chess__)
#define AIE_PERSIST_ZERO
#define AIE_TRANSIENT static
#else
#define AIE_PERSIST_ZERO __attribute__((section(".data")))
#define AIE_TRANSIENT
#endif

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

#endif