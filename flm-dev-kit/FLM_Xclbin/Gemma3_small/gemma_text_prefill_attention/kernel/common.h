// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef COMMON_H
#define COMMON_H

#include <aie_api/aie.hpp>
typedef bfloat16 bf16;

// .bss is not zeroed when a Peano-built core is loaded. AIE_PERSIST_ZERO puts a
// zero-initialized static that must keep its value across calls in .data.
// AIE_TRANSIENT drops `static` where the value does not need to persist, so no
// init guard in .bss is involved. Both do nothing under chess.
#if defined(__chess__)
#define AIE_PERSIST_ZERO
#define AIE_TRANSIENT static
#else
#define AIE_PERSIST_ZERO __attribute__((section(".data")))
#define AIE_TRANSIENT
#endif


// Model type definitions
#define GEMMA_1B 1
#define GEMMA_4B 2

#ifndef MODEL_TYPE
#define MODEL_TYPE GEMMA_1B
#endif


#if MODEL_TYPE == GEMMA_1B
// 1B model parameters
constexpr int D = 1152;
constexpr int NUM_ATTN_HEADS = 4;
constexpr int NUM_KV_HEADS = 1;

constexpr int HIDDEN_SIZE = 6912;
const float scale = 0.0625f;

#elif MODEL_TYPE == GEMMA_4B
// 4B model parameters
constexpr int D = 2560;
constexpr int NUM_ATTN_HEADS = 8;
constexpr int NUM_KV_HEADS = 4;

constexpr int HIDDEN_SIZE = 10240;
const float scale = 0.0625f;

#else
    #error "Invalid model type. Please define MODEL_TYPE as GEMMA_1B or GEMMA_4B"
#endif

#define GLU_SLICE 1024

constexpr int ATTN_GROUPS = NUM_ATTN_HEADS / NUM_KV_HEADS;
constexpr int DQ = 1024;
constexpr int DK = DQ / ATTN_GROUPS;
constexpr int DV = DK;
constexpr int DH = DQ / NUM_ATTN_HEADS;
constexpr int ATTN_GROUPS_PADDING = ((ATTN_GROUPS + 4 - 1) / 4) * 4 - ATTN_GROUPS;
// Preprocessor definition for ATTN_GROUPS_PADDING
#define ATTN_GROUPS_PADDING_VALUE (((ATTN_GROUPS + 4 - 1) / 4) * 4 - ATTN_GROUPS)

constexpr int cores = 16;
constexpr int qkv_repeats = (DQ + DK + DV) / cores / 32;
constexpr int up_gate_repeats = (2 * HIDDEN_SIZE) / cores / 32;


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



constexpr int NPU_M_TILE = 16;
constexpr int NPU_K_TILE = 256;
constexpr int NPU_N_TILE = 16;

constexpr bf16 exp_scale = (bf16)1.4426950408889634f;

#endif
