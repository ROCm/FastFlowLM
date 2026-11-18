/*
    Copyright (C) 2014 - 2022 Xilinx, Inc. All rights reserved.
    Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
    SPDX-License-Identifier: MIT
*/

#ifndef _AIE_KERNEL_UTILS_
#define _AIE_KERNEL_UTILS_
#include "typedef.hpp"
#include "flm_rsqrt.h"
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
    constexpr int r = 256 / (sizeof(T) * 8); // one 256 bit store unit
    static_assert((M) % r == 0);
    const aie::vector<T, r> zeros = aie::zeros<T, r>();
    const T *__restrict c_end = c + M;
    for (; c < c_end; c += r) {
        aie::store_v(c, zeros);
    }
}





///////////////////////////////////////////////////////////////////////////////
// Vector width
///////////////////////////////////////////////////////////////////////////////

// FLM_VECTOR_BITS -- the widest vector op (load, store, mul, mac) the
// elementwise helpers below, and the kernels that share their width constants,
// are allowed to emit.  Override from the build:
//
//     make FLM_VECTOR_BITS=512 ...
//
// The default is 256: AIE2P registers are 512 bit, but the 512-bit forms are
// avoided in this tree, so a bf16 op covers 16 lanes and fills half a register
// and half the 1024-bit accumulator.  512 restores the full-register forms.
//
// This caps the *elementwise* path only.  aie::mmul operand widths are fixed by
// the matmul intrinsic's r/s/t shape and are deliberately left alone.
#ifndef FLM_VECTOR_BITS
#define FLM_VECTOR_BITS 256
#endif
static_assert(FLM_VECTOR_BITS == 128 || FLM_VECTOR_BITS == 256 ||
                  FLM_VECTOR_BITS == 512,
              "FLM_VECTOR_BITS must be 128, 256 or 512");

// Element count one FLM_VECTOR_BITS-wide vector covers, per element type.
constexpr int AIE_VECTOR_BITS = FLM_VECTOR_BITS;
constexpr int AIE_BF16_LANES  = AIE_VECTOR_BITS / 16;  // bf16 / int16
constexpr int AIE_BYTE_LANES  = AIE_VECTOR_BITS / 8;   // uint8 / int8
constexpr int AIE_F32_LANES   = AIE_VECTOR_BITS / 32;  // float / int32

/// \brief Largest lane count in {W, W/2, W/4} that divides m, so a helper
/// widens to full registers when it can and still compiles when m is odd.
template<int m, int W>
constexpr int _fit_lanes() {
    if constexpr (m % W == 0)          return W;
    else if constexpr (m % (W/2) == 0) return W / 2;
    else                               return W / 4;
}


template<int m>
void copy_float_to_bf16(bfloat16 * y, const float * y_acc) {
    constexpr int V = _fit_lanes<m, AIE_BF16_LANES>();
    static_assert(m % V == 0);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < m / V; i++){
        aie::vector<float, V> y_vec = aie::load_v<V>(y_acc + i * V);
        aie::accum<accfloat, V> acc;
        acc.from_vector(y_vec);
        aie::store_v(y + i * V, acc.template to_vector<bfloat16>());
    }
}

template<int m>
void copy_bf16_to_float(float * y, const bfloat16 * y_bf16) {
    constexpr int V = _fit_lanes<m, AIE_BF16_LANES>();
    static_assert(m % V == 0);
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < m / V; i++){
        aie::vector<bfloat16, V> y_vec = aie::load_v<V>(y_bf16 + i * V);
        aie::accum<accfloat, V> acc;
        acc.from_vector(y_vec);
        aie::store_v(y + i * V, acc.template to_vector<float>());
    }
}


template<int m>
void copy_bf16_to_bf16( bfloat16 *y, const bfloat16 * y_bf16) {
    constexpr int V = _fit_lanes<m, AIE_BF16_LANES>();
    static_assert(m % V == 0);
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < m / V; i++){
        aie::vector<bfloat16, V> y_vec = aie::load_v<V>(y_bf16 + i * V);
        aie::store_v(y + i * V, y_vec);
    }
}


/// \brief Vectorized copy of a compile-time byte count.
///
/// The libc memcpy the core would otherwise call is a scalar loop, and every
/// use of it in this tree sits inside a lock window where the whole array is
/// stalled waiting on it.  This moves FLM_VECTOR_BITS per instruction, and
/// unrolls away the loop entirely when the copy is only a few of them.
///
/// SIZE is in bytes; src and dst must be vector aligned, as every buffer the
/// object FIFOs hand a kernel is.
template<int SIZE, int vec_size = AIE_BYTE_LANES>
void _memcpy(uint8 *restrict dst, const uint8 *restrict src){
    constexpr int V = _fit_lanes<SIZE, vec_size>();
    static_assert(SIZE % V == 0, "_memcpy: SIZE must be a whole number of vectors");
    constexpr int N = SIZE / V;
    if constexpr (N <= 4){
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < N; i++){
            aie::store_v(dst + i * V, aie::load_v<V>(src + i * V));
        }
    } else {
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(4)
        for (int i = 0; i < N; i++){
            aie::store_v(dst + i * V, aie::load_v<V>(src + i * V));
        }
    }
}

/// \brief _memcpy in units of bf16 elements rather than bytes.
template<int N>
inline void _copy_bf16(bf16 *restrict dst, const bf16 *restrict src){
    _memcpy<N * (int)sizeof(bf16)>((uint8*)dst, (const uint8*)src);
}


///////////////////////////////////////////////////////////////////////////////
// RMS norm
///////////////////////////////////////////////////////////////////////////////

// Scalar float +, -, * whose result has to match chess bit for bit. chess
// computes these with IEEE round-to-nearest-even softfloat. Peano fuses a*b+c
// and runs scalar add/sub on the vector unit, which rounds some results
// differently and makes the decode output differ from chess. Under Peano, call
// compiler-rt's IEEE softfloat directly so both compilers round the same way.
#if defined(__chess__)
#define AIE_FADD(a, b) ((a) + (b))
#define AIE_FSUB(a, b) ((a) - (b))
#define AIE_FMUL(a, b) ((a) * (b))
#else
extern "C" float __addsf3(float, float);
extern "C" float __subsf3(float, float);
extern "C" float __mulsf3(float, float);
#define AIE_FADD(a, b) __addsf3((a), (b))
#define AIE_FSUB(a, b) __subsf3((a), (b))
#define AIE_FMUL(a, b) __mulsf3((a), (b))
#endif

/// \brief 1 / sqrt(mean(x[0..D]^2) + epsilon), the RMS norm scale factor.
///
/// Two independent accumulators so the mac chain is not serialized on a single
/// one; they are folded together once, after the loop.
template<int D, int VS = _fit_lanes<D / 2, AIE_BF16_LANES>()>
inline float _rms_scale(const bf16 *restrict x, float epsilon){
    static_assert(D % (2 * VS) == 0, "_rms_scale: D must be a multiple of 2*VS");
    constexpr float one_over_D = 1.0f / (float)D;

    const bf16 *restrict it_x_0 = x;
    const bf16 *restrict it_x_1 = x + VS;

    aie::accum<accfloat, VS> sum_squares_0 = aie::zeros<accfloat, VS>();
    aie::accum<accfloat, VS> sum_squares_1 = aie::zeros<accfloat, VS>();

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(1)
    for (int i = 0; i < D / (2 * VS); i++){
        sum_squares_0 = aie::mac_square(sum_squares_0, aie::load_v<VS>(it_x_0));
        sum_squares_1 = aie::mac_square(sum_squares_1, aie::load_v<VS>(it_x_1));
        it_x_0 += 2 * VS;
        it_x_1 += 2 * VS;
    }

    aie::vector<float, VS> mx_vec = aie::add(sum_squares_0.template to_vector<float>(),
                                             sum_squares_1.template to_vector<float>());
    return flm_rsqrt(AIE_FADD(AIE_FMUL(aie::reduce_add(mx_vec), one_over_D), epsilon));
}

/// \brief RMS norm, out of place: y = w * x / rms(x), over D elements.
///
/// y must not overlap x or w -- use _rms_norm_inplace for the per-head Q/K
/// norms, which normalize a vector where it lies.
template<int D, int VS = _fit_lanes<D / 2, AIE_BF16_LANES>()>
inline void _rms_norm(bf16 *restrict y, const bf16 *restrict x, const bf16 *restrict w,
                      float epsilon){
    static_assert(D % (2 * VS) == 0, "_rms_norm: D must be a multiple of 2*VS");
    const float divrms = _rms_scale<D, VS>(x, epsilon);

    const bf16 *restrict it_x_0 = x;
    const bf16 *restrict it_x_1 = x + VS;
    const bf16 *restrict it_w_0 = w;
    const bf16 *restrict it_w_1 = w + VS;
    bf16 *restrict it_y_0 = y;
    bf16 *restrict it_y_1 = y + VS;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(1)
    for (int i = 0; i < D / (2 * VS); i++){
        aie::vector<bf16, VS> x_vec_0 = aie::load_v<VS>(it_x_0);
        aie::vector<bf16, VS> x_vec_1 = aie::load_v<VS>(it_x_1);
        aie::vector<bf16, VS> w_vec_0 = aie::load_v<VS>(it_w_0);
        aie::vector<bf16, VS> w_vec_1 = aie::load_v<VS>(it_w_1);
        aie::vector<float, VS> wx_vec_0 = aie::mul(x_vec_0, w_vec_0);
        aie::vector<float, VS> wx_vec_1 = aie::mul(x_vec_1, w_vec_1);
        aie::store_v(it_y_0, aie::vector<bf16, VS>(aie::mul(wx_vec_0, divrms)));
        aie::store_v(it_y_1, aie::vector<bf16, VS>(aie::mul(wx_vec_1, divrms)));
        it_x_0 += 2 * VS;
        it_x_1 += 2 * VS;
        it_w_0 += 2 * VS;
        it_w_1 += 2 * VS;
        it_y_0 += 2 * VS;
        it_y_1 += 2 * VS;
    }
}

/// \brief RMS norm, in place: x = w * x / rms(x), over D elements.
///
/// A per-head vector is small enough to sit in registers across both passes, so
/// the sum-of-squares pass hands its loads straight to the scaling pass and the
/// second read of x disappears.  Above that it falls back to re-reading x,
/// without restrict, since the stores land on the same buffer.
template<int D, int VS = _fit_lanes<D / 2, AIE_BF16_LANES>()>
inline void _rms_norm_inplace(bf16 *x, const bf16 *restrict w, float epsilon){
    static_assert(D % (2 * VS) == 0, "_rms_norm_inplace: D must be a multiple of 2*VS");
    constexpr int NV = D / VS;
    constexpr float one_over_D = 1.0f / (float)D;

    // 256 bytes -- four full registers -- is what a per-head vector costs.
    if constexpr (D * (int)sizeof(bf16) <= 256){
        aie::vector<bf16, VS> x_vec[NV];
        aie::accum<accfloat, VS> sum_squares_0 = aie::zeros<accfloat, VS>();
        aie::accum<accfloat, VS> sum_squares_1 = aie::zeros<accfloat, VS>();

        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < NV; i++) x_vec[i] = aie::load_v<VS>(x + i * VS);

        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < NV / 2; i++){
            sum_squares_0 = aie::mac_square(sum_squares_0, x_vec[2 * i]);
            sum_squares_1 = aie::mac_square(sum_squares_1, x_vec[2 * i + 1]);
        }
        aie::vector<float, VS> mx_vec = aie::add(sum_squares_0.template to_vector<float>(),
                                                 sum_squares_1.template to_vector<float>());
        const float divrms = flm_rsqrt(AIE_FADD(AIE_FMUL(aie::reduce_add(mx_vec), one_over_D), epsilon));

        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < NV; i++){
            aie::vector<float, VS> wx_vec = aie::mul(x_vec[i], aie::load_v<VS>(w + i * VS));
            aie::store_v(x + i * VS, aie::vector<bf16, VS>(aie::mul(wx_vec, divrms)));
        }
    } else {
        const float divrms = _rms_scale<D, VS>(x, epsilon);
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(1)
        for (int i = 0; i < NV; i++){
            aie::vector<float, VS> wx_vec = aie::mul(aie::load_v<VS>(x + i * VS),
                                                     aie::load_v<VS>(w + i * VS));
            aie::store_v(x + i * VS, aie::vector<bf16, VS>(aie::mul(wx_vec, divrms)));
        }
    }
}



// Peano's CDO loader does not zero .bss, so a zero-initialized static that must
// keep its value across kernel invocations is placed in .data instead. chess
// zeroes .bss, so the macro expands to nothing there.
#if defined(__chess__)
#define AIE_PERSIST_ZERO
#else
#define AIE_PERSIST_ZERO __attribute__((section(".data")))
#endif

#endif