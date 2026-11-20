// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef COMMON_H
#define COMMON_H

#include <aie_api/aie.hpp>
typedef bfloat16 bf16;

// .bss is not zeroed when a Peano-built core is loaded. AIE_PERSIST_ZERO puts a
// zero-initialized static that must keep its value across calls in .data, and
// AIE_PING_FLAG makes a ping/pong selector volatile there too: under Peano a lock
// acquire is not a memory barrier. Both do nothing under chess.
#if defined(__chess__)
#define AIE_PERSIST_ZERO
#define AIE_PING_FLAG bool
#else
#define AIE_PERSIST_ZERO __attribute__((section(".data")))
#define AIE_PING_FLAG volatile bool
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

template <typename T_in, typename T_out, int N>
void passThrough_aie(T_in *restrict in0, T_out *restrict out) {
    const int vec_factor = 64;

    aie::vector<T_out, vec_factor> Out;

    const int F = N / vec_factor;
    for (int i = 0; i < F; i++){
        aie::accum<accfloat, vec_factor> c_acc_in;
        c_acc_in.from_vector(aie::load_v<vec_factor>(in0));
        aie::store_v(out, c_acc_in.template to_vector<T_out>());
        in0 += vec_factor;
        out += vec_factor;
    }
}

constexpr int L = 1536;
// Whisper has 1500 encoder frames; rows from here to L are padding, and keys
// there must get no attention weight.
constexpr int L_valid = 1500;
constexpr int lk_mt = 512;
constexpr int chunk_size = lk_mt;
constexpr int chunks = L / chunk_size;
constexpr int lq = 64;
constexpr int lk = 16;
constexpr int dh = 64;

constexpr bf16 exp_scale = (bf16)1.4426950408889634f;
const float scale = 0.125f;

#endif
