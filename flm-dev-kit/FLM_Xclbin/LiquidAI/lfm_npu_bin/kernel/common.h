// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef COMMON_H
#define COMMON_H

// Model type definitions
#define LFM_1_2B 1
#define LFM_2_6B 2

#ifndef MODEL_TYPE
#define MODEL_TYPE LFM_1_2B
#endif

#include <aie_api/aie.hpp>
// #include <aie2p_locks.h>
typedef bfloat16 bf16;

// Peano (llvm-aie, non-chess) does not provide the raw chess loop pragmas; make
// them no-ops so kernels that use them directly still compile.
#ifndef __chess__
#define chess_prepare_for_pipelining
#define chess_loop_range(...)
#define chess_flatten_loop
#define chess_unroll_loop(...)
#endif

// Lock-loop counter type: peano hoists lock releases out of the do{}while(i>0)
// lock loops unless the counter is volatile. Chess keeps a plain int.
#if defined(__chess__)
#define AIE_LOCK_LOOP_CTR int
#else
#define AIE_LOCK_LOOP_CTR volatile int
#endif

#if MODEL_TYPE == LFM_1_2B
// 1B model parameters
constexpr int D = 2048;
constexpr int NUM_ATTN_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;

constexpr int HIDDEN_SIZE = 8192;
const float scale = 0.125f;

#elif MODEL_TYPE == LFM_2_6B
// 3B model parameters
constexpr int D = 2048;
constexpr int NUM_ATTN_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;

constexpr int HIDDEN_SIZE = 10752;
const float scale = 0.125f;
#else
    #error "Invalid model type. Please define MODEL_TYPE as LFM_1_2B or LFM_2_6B"
#endif

#define GLU_SLICE 1024

constexpr int ATTN_GROUPS = NUM_ATTN_HEADS / NUM_KV_HEADS;
constexpr int DK = D / ATTN_GROUPS;
constexpr int DV = DK;
constexpr int DH = D / NUM_ATTN_HEADS;
constexpr int ATTN_GROUPS_PADDING = ((ATTN_GROUPS + 4 - 1) / 4) * 4 - ATTN_GROUPS;
// Preprocessor definition for ATTN_GROUPS_PADDING
#define ATTN_GROUPS_PADDING_VALUE (((ATTN_GROUPS + 4 - 1) / 4) * 4 - ATTN_GROUPS)

constexpr int CONV_D3 = D * 3;

constexpr int cores = 16;
constexpr int conv_d3_repeats = CONV_D3 / cores / 32;

constexpr int qkv_repeats = (D + DK + DV) / cores / 32;
constexpr int up_gate_repeats = (2 * HIDDEN_SIZE) / cores / 32;
constexpr int o_repeats = D / cores / 32;
const bf16 exp_scale = 1.4426950408889634f;

inline void _lock_acquire(int lock_id, int num_locks = 1){
    acquire_greater_equal(lock_id + 48, num_locks);
}

inline void _lock_release(int lock_id, int num_locks = 1){
    release(lock_id + 48, num_locks);
}

inline void _left_lock_acquire(int lock_id, int num_locks = 1){
    acquire_greater_equal(lock_id + 16, num_locks);
}

inline void _left_lock_release(int lock_id, int num_locks = 1){
    release(lock_id + 16, num_locks);
}

inline void _up_lock_acquire(int lock_id, int num_locks = 1){
    acquire_greater_equal(lock_id + 32, num_locks);
}

inline void _up_lock_release(int lock_id, int num_locks = 1){
    release(lock_id + 32, num_locks);
}

inline void _down_lock_acquire(int lock_id, int num_locks = 1){
    acquire_greater_equal(lock_id, num_locks);
}

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



template <typename T_in, typename T_out, int N>
void passThrough_aie(T_in *restrict in0, T_out *restrict out) {
    event0();
    const int vec_factor = 16;

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
#endif
