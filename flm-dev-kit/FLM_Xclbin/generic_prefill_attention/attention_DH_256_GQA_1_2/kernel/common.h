// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef COMMON_H
#define COMMON_H

#include <aie_api/aie.hpp>
typedef bfloat16 bf16;

// Peano's CDO loader does not zero .bss, so a zero-initialized static that must
// carry its value across kernel invocations has to be forced into .data
// (AIE_PERSIST_ZERO); and a static that does NOT need to persist is better left
// as an automatic (AIE_TRANSIENT), because a lazily-initialized function-local
// static depends on a .bss guard byte the loader may not zero. chess zeroes .bss,
// so both macros are inert there and the chess build is byte-for-byte what it was.
#if defined(__chess__)
#define AIE_PERSIST_ZERO
#define AIE_TRANSIENT static
#else
#define AIE_PERSIST_ZERO __attribute__((section(".data")))
#define AIE_TRANSIENT
#endif

// Peano (llvm-aie, non-chess) does not provide the chess loop pragmas; make
// them no-ops so the kernels compile under both toolchains.
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

#endif
