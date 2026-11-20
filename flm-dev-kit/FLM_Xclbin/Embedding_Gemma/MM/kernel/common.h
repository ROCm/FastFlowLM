// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef COMMON_H
#define COMMON_H

#include <aie_api/aie.hpp>
#include <type_traits>
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

// The chess loop pragmas do nothing under Peano.
#ifndef __chess__
#define chess_prepare_for_pipelining
#define chess_loop_range(...)
#define chess_flatten_loop
#endif

// W reaches the core as 8x8 blocks in (n, k) order, i.e. rows of the [N][K] weight
// the runtime sends (x W^T), while aie::mmul takes B in (k, n) order. mmul_wt_t<MMUL, T>
// is an 8x8x8 bf16 mmul whose mac() takes such a W block: under Peano with BFP16
// emulation the block goes straight to mac_8x8_8x8T (its B operand is transposed
// already), elsewhere it is transposed first.
#if !defined(__chess__) && defined(AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16)
struct aie_mmul_bf16_888_wt {
    aie::accum<accfloat, 64> acc;
    template <typename T> explicit aie_mmul_bf16_888_wt(const aie::vector<T, 64> &c) { acc.from_vector(c); }
    __attribute__((always_inline)) void mac(const aie::vector<bfloat16, 64> &a, const aie::vector<bfloat16, 64> &w_nk) {
        aie::accum<accfloat, 64> acc_a(a);
        aie::accum<accfloat, 64> acc_w(w_nk);
        acc = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(acc_a), ::to_v64bfp16ebs8(acc_w), acc, 0, 0, 0);
    }
    template <typename T> aie::vector<T, 64> to_vector() const { return acc.template to_vector<T>(); }
};
template <typename M, typename T>
using mmul_wt_t = aie_mmul_bf16_888_wt;
#else
template <typename M>
struct aie_mmul_wt : M {
    using M::M;
    template <typename V> __attribute__((always_inline)) void mac(const V &a, const V &w_nk) { M::mac(a, aie::transpose(w_nk, 8, 8)); }
};
template <typename M, typename T>
using mmul_wt_t = aie_mmul_wt<M>;
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
    event0();
    const int vec_factor = 32;

    aie::vector<T_out, vec_factor> Out;

    const int F = N / vec_factor;
    for (int i = 0; i < F; i++)
    chess_flatten_loop{
            aie::accum<accfloat, vec_factor> c_acc_in;
            c_acc_in.from_vector(aie::load_v<vec_factor>(in0));
            aie::store_v(out, c_acc_in.template to_vector<T_out>());
            in0 += vec_factor;
            out += vec_factor;
        }
    event1();
}

constexpr int NPU_M_TILE = 16;
constexpr int NPU_K_TILE = 128;
constexpr int NPU_N_TILE = 64;
constexpr int m_slice = 16;
constexpr int k_slice = 64;
constexpr int n_slice = 64;
constexpr int y_slice = 64;
constexpr int y_num = 16;
#endif
