// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
// SPDX-License-Identifier: MIT

#ifndef _AIE_KERNEL_UTILS_
#define _AIE_KERNEL_UTILS_

#include <aie_api/aie.hpp>
typedef bfloat16 bf16;

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

// Some lock utilities
/// \brief Acquire a lock with given lock ID on this CT and number of locks
/// \param lock_id The lock ID to acquire
/// \param num_locks The number of locks to acquire
inline void _lock_acquire(int lock_id, int num_locks = 1){
    acquire_greater_equal(lock_id + 48, num_locks);
}

/// \brief Release a lock with given lock ID on this CT and number of locks
/// \param lock_id The lock ID to release
/// \param num_locks The number of locks to release
inline void _lock_release(int lock_id, int num_locks = 1){
    release(lock_id + 48, num_locks);
}


/// \brief Acquire a lock with given lock ID on left CT and number of locks
/// \param lock_id The lock ID to acquire
/// \param num_locks The number of locks to acquire
inline void _left_lock_acquire(int lock_id, int num_locks = 1){
    acquire_greater_equal(lock_id + 16, num_locks);
}

/// \brief Release a lock with given lock ID on left CT and number of locks
/// \param lock_id The lock ID to release
/// \param num_locks The number of locks to release
inline void _left_lock_release(int lock_id, int num_locks = 1){
    release(lock_id + 16, num_locks);
}

/// \brief Acquire a lock with given lock ID on up CT and number of locks
/// \param lock_id The lock ID to acquire
/// \param num_locks The number of locks to acquire
inline void _up_lock_acquire(int lock_id, int num_locks = 1){
    acquire_greater_equal(lock_id + 32, num_locks);
}

/// \brief Release a lock with given lock ID on up CT and number of locks
/// \param lock_id The lock ID to release
/// \param num_locks The number of locks to release
inline void _up_lock_release(int lock_id, int num_locks = 1){
    release(lock_id + 32, num_locks);
}

/// \brief Acquire a lock with given lock ID on down CT and number of locks
/// \param lock_id The lock ID to acquire
/// \param num_locks The number of locks to acquire
inline void _down_lock_acquire(int lock_id, int num_locks = 1){
    acquire_greater_equal(lock_id, num_locks);
}

/// \brief Release a lock with given lock ID on down CT and number of locks
/// \param lock_id The lock ID to release
/// \param num_locks The number of locks to release
inline void _down_lock_release(int lock_id, int num_locks = 1){
    release(lock_id, num_locks);
}

template <typename T, int M>
void zero_vectorized(T *__restrict c) {
#if defined(__chess__)
    constexpr int r = 256 / (sizeof(T) * 8); // one 256 bit store unit
    static_assert((M) % r == 0);
    const aie::vector<T, r> zeros = aie::zeros<T, r>();
    const T *__restrict c_end = c + M;
    for (; c < c_end; c += r) {
        aie::store_v(c, zeros);
    }
#else
    // Peano left the loop above unpipelined (one store per 3 cycles); 512-bit
    // stores unrolled by 4 run at one store per cycle. c must be 64-byte aligned.
    constexpr int r = 512 / (sizeof(T) * 8);
    static_assert((M) % (4 * r) == 0);
    const aie::vector<T, r> zeros = aie::zeros<T, r>();
    AIE_LOOP_UNROLL(4)
    for (int i = 0; i < M; i += r) {
        aie::store_v(c + i, zeros);
    }
#endif
}

template<int SIZE>
void _memcpy(uint8* dst, uint8* src){
    static constexpr int vec_size = 256;
    for (int i = 0; i < SIZE; i += vec_size){
        aie::vector<uint8, vec_size> data = aie::load_v<vec_size>(src + i);
        aie::store_v(dst + i, data);
    }
}

// Ping/pong buffer selectors. Peano's loader does not zero .bss, so under Peano
// they live in .data. They are volatile there too: Peano does not treat a lock
// acquire as a memory barrier, so if it can prove two picks give the same buffer
// it reuses the earlier load, even though the DMA has refilled the buffer in between.
#if defined(__chess__)
#define AIE_PING_FLAG bool
#else
#define AIE_PING_FLAG __attribute__((section(".data"))) volatile bool
#endif

// bf16 8x8x8 mmul with the BFP16 emulation, for Peano. It issues the same
// intrinsic as aie_api's emulated mac but converts B with the accumulator
// constructor rather than a multiply by one, which would take a multiplier slot
// from the macs. The conversion is exact, so results are unchanged.
#if !defined(__chess__) && defined(AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16)
#include <aie_api/aie.hpp>
struct aie_mmul_bf16_888 {
    aie::accum<accfloat, 64> acc;
    explicit aie_mmul_bf16_888(const aie::vector<float, 64> &c) { acc.from_vector(c); }
    __attribute__((always_inline)) void mac(const aie::vector<bfloat16, 64> &a, const aie::vector<bfloat16, 64> &b) {
        aie::accum<accfloat, 64> acc_a(a);
        aie::accum<accfloat, 64> acc_b(aie::transpose(b, 8, 8));
        acc = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(acc_a), ::to_v64bfp16ebs8(acc_b), acc, 0, 0, 0);
    }
    template <typename T> aie::vector<T, 64> to_vector() const { return acc.template to_vector<T>(); }
};
#endif

#endif
