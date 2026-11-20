/*
    Copyright (C) 2014 - 2022 Xilinx, Inc. All rights reserved.
    Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
    Portions of this file consist of AI-generated content.
    SPDX-License-Identifier: MIT
*/
#include "zero.h"
#include "passThroughCommon.hpp"
#ifndef _AIE_KERNEL_UTILS_
#define _AIE_KERNEL_UTILS_
#include "typedef.hpp"
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

// Loop-counter type for loops whose body ends in a per-iteration lock release.
// Peano/llvm-aie (non-chess) sinks the final lock release past the loop
// back-edge, so it runs once on loop exit instead of every iteration and
// deadlocks any multi-iteration lock loop. A volatile counter makes the trip
// count opaque and keeps the release inside the loop. chess schedules correctly.
#if defined(__chess__)
#define AIE_LOCK_LOOP_CTR int
#else
#define AIE_LOCK_LOOP_CTR volatile int
#endif

// State that must start at zero and keep its value between calls (ping/pong
// flags, "already known" tables). Peano's CDO loader does not zero .bss, so
// under Peano such variables go to .data, which is loaded with its initial value.
#if defined(__chess__)
#define AIE_PERSIST_ZERO
#else
#define AIE_PERSIST_ZERO __attribute__((section(".data")))
#endif

// Scalar float +, -, * rounded like chess. chess computes these with IEEE
// round-to-nearest-even softfloat; Peano fuses a*b+c and runs scalar add/sub on
// the vector unit, which rounds some results differently. Under Peano, call
// compiler-rt's IEEE softfloat directly so both compilers round the same way.
#if defined(__chess__)
#define AIE_FADD(a, b) ((a) + (b))
#define AIE_FMUL(a, b) ((a) * (b))
#else
extern "C" float __addsf3(float, float);
extern "C" float __mulsf3(float, float);
#define AIE_FADD(a, b) __addsf3((a), (b))
#define AIE_FMUL(a, b) __mulsf3((a), (b))
#endif

// Buffer parameters of a kernel entry function that also takes and gives back
// locks. The DMA reads and writes these buffers between the lock calls, so they
// are not "restrict" in the C sense. chess keeps every access where it is
// written; Peano sees that a lock call cannot reach a restrict buffer and drops
// or moves accesses across it (it deleted packet-id writes in rms_residual).
#if defined(__chess__)
#define AIE_DMA_RESTRICT restrict
#else
#define AIE_DMA_RESTRICT
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

// template <typename T, int M>
// void zero_vectorized(T *__restrict c) {
//     constexpr int r = 256 / (sizeof(T) * 8); // one 256 bit store unit
//     static_assert((M) % r == 0);
//     const aie::vector<T, r> zeros = aie::zeros<T, r>();
//     const T *__restrict c_end = c + M;
//     for (; c < c_end; c += r) {
//         aie::store_v(c, zeros);
//     }
// }





// template<int m>
// void copy_float_to_bf16(bfloat16 * y, const float * y_acc) {
//     static_assert(m%16 == 0);
//     AIE_PREPARE_FOR_PIPELINING
//     AIE_LOOP_UNROLL_FULL
//     for (int i = 0; i < m / 16; i++){
//         aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + i * 16);
//         aie::accum<accfloat, 16> acc;
//         acc.from_vector(y_vec);
//         aie::vector<bf16, 16> y_vec_bf16 = acc.template to_vector<bfloat16>();
//         // y_vec_bf16 = aie::mul(y_vec_bf16, (bf16)0.5); // convert from float32 to bfloat16 by dividing 256 and truncating
//         aie::store_v(y + i * 16, y_vec_bf16);
//     }
// }

// template<int m>
// void copy_bf16_to_float(float * y, const bfloat16 * y_bf16) {
//     static_assert(m%16 == 0);
//     AIE_PREPARE_FOR_PIPELINING
//     for (int i = 0; i < m / 16; i++){
//         aie::vector<bfloat16, 16> y_vec = aie::load_v<16>(y_bf16 + i * 16);
//         aie::accum<accfloat, 16> acc;
//         acc.from_vector(y_vec);
//         aie::store_v(y + i * 16, acc.template to_vector<float>());
//     }
// }


// template<int m>
// void copy_bf16_to_bf16( bfloat16 *y, const bfloat16 * y_bf16) {
//     static_assert(m%16 == 0);
//     AIE_PREPARE_FOR_PIPELINING
//     for (int i = 0; i < m / 16; i++){
//         aie::vector<bfloat16, 16> y_vec = aie::load_v<16>(y_bf16 + i * 16);
//         aie::store_v(y + i * 16, y_vec);
//     }

// }


template<int SIZE, int vec_size = 32>
void _memcpy(uint8* dst, uint8* src){
    static_assert(SIZE%vec_size == 0);
    if constexpr(SIZE/vec_size <= 3){
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < SIZE; i += vec_size){
            aie::vector<uint8, vec_size> data = aie::load_v<vec_size>(src + i);
            aie::store_v(dst + i, data);
        }
    } else {
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(SIZE/vec_size,SIZE/vec_size)
        for (int i = 0; i < SIZE; i += vec_size){
            aie::vector<uint8, vec_size> data = aie::load_v<vec_size>(src + i);
            aie::store_v(dst + i, data);
        }
    }
}
// template<int vector_size = 16>
// inline aie::vector<bf16, vector_size> getExpBf16(aie::vector<bf16, vector_size>& x) {
//     constexpr bfloat16 exp_scale = 1.44269504089;
//     aie::vector<float, vector_size> y_scaled = aie::mul(x, aie::broadcast<bf16, vector_size>(exp_scale));
//     aie::vector<bf16, vector_size> y_exp = aie::exp2(y_scaled);
//     return y_exp;
// }


#endif