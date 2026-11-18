// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef COMMON_H
#define COMMON_H

#include <aie_api/aie.hpp>
typedef bfloat16 bf16;

// Peano's loader does not zero .bss. A zero-initialised static that must keep
// its value between calls goes in .data (AIE_PERSIST_ZERO); one that need not is
// better as an automatic (AIE_TRANSIENT), since a function-local static's guard
// byte is in .bss too. Both are no-ops under chess.
#if defined(__chess__)
#define AIE_PERSIST_ZERO
#define AIE_TRANSIENT static
#else
#define AIE_PERSIST_ZERO __attribute__((section(".data")))
#define AIE_TRANSIENT
#endif

// The chess loop pragmas do not exist under Peano.
#ifndef __chess__
#define chess_prepare_for_pipelining
#define chess_loop_range(...)
#define chess_flatten_loop
#endif

inline void _lock_acquire(int lock_id){
    acquire_greater_equal(lock_id + 48, 1);
}

inline void _lock_release(int lock_id){
    release(lock_id + 48, 1);
}

inline void _left_lock_acquire(int lock_id){
    acquire_greater_equal(lock_id + 16, 1);
}

inline void _left_lock_release(int lock_id){
    release(lock_id + 16, 1);
}

inline void _up_lock_acquire(int lock_id){
    acquire_greater_equal(lock_id + 32, 1);
}

inline void _up_lock_release(int lock_id){
    release(lock_id + 32, 1);
}

inline void _down_lock_acquire(int lock_id){
    acquire_greater_equal(lock_id, 1);
}

inline void _down_lock_release(int lock_id){
    release(lock_id, 1);
}

template <typename T, int M>
void zero_vectorized(T *__restrict c) {
    constexpr int r = 16; // one 256 bit store unit
    static_assert((M) % r == 0);
    const aie::vector<T, r> zeros = aie::zeros<T, r>();
    const T *__restrict c_end = c + M;
    for (; c < c_end; c += r) {
        aie::store_v(c, zeros);
    }
}



template <typename T_in, typename T_out>
void passThrough_aie(T_in *restrict in, T_out *restrict out) {
    event0();
    const int vec_factor = 64;

    aie::accum<accfloat, vec_factor> c_acc_in;
    c_acc_in.from_vector(aie::load_v<vec_factor>(in));
    aie::store_v(out, c_acc_in.template to_vector<T_out>());
    event1();
}



constexpr int lq = 16;
constexpr int lk = 16;
constexpr int dh = 256;

const float scale = 0.0625f;

constexpr bf16 exp_scale = (bf16)1.4426950408889634f;


// bf16 8x8x8 mmul with the BFP16 emulation, for Peano. It issues the same
// intrinsic as aie_api's emulated mac but converts B with the accumulator
// constructor rather than a multiply by one, which would take a multiplier slot
// from the macs. The conversion is exact, so results are unchanged.
// mmul_c_t<MMUL, T, r, s, t> is this type where it applies and MMUL elsewhere.
#include <type_traits>
#if !defined(__chess__) && defined(AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16)
struct aie_mmul_bf16_888 {
    aie::accum<accfloat, 64> acc;
    template <typename T> explicit aie_mmul_bf16_888(const aie::vector<T, 64> &c) { acc.from_vector(c); }
    __attribute__((always_inline)) void mac(const aie::vector<bfloat16, 64> &a, const aie::vector<bfloat16, 64> &b) {
        aie::accum<accfloat, 64> acc_a(a);
        aie::accum<accfloat, 64> acc_b(aie::transpose(b, 8, 8));
        acc = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(acc_a), ::to_v64bfp16ebs8(acc_b), acc, 0, 0, 0);
    }
    template <typename T> aie::vector<T, 64> to_vector() const { return acc.template to_vector<T>(); }
};
template <typename M, typename T, unsigned r, unsigned s, unsigned t>
using mmul_c_t = std::conditional_t<std::is_same_v<T, bfloat16> && r == 8 && s == 8 && t == 8, aie_mmul_bf16_888, M>;
#else
template <typename M, typename T, unsigned r, unsigned s, unsigned t>
using mmul_c_t = M;
#endif

#endif
