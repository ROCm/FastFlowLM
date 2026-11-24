// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef COMMON_H
#define COMMON_H

// Model type definitions
#define LFM_1_2B 1
#define LFM_2_6B 2
#define LFM3_1_7B 3

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
#elif MODEL_TYPE == LFM3_1_7B
// LFM3-1.7B: 16 q heads, 4 kv heads, head dim 128, conv width 7, sandwich norms
constexpr int D = 2048;
constexpr int NUM_ATTN_HEADS = 16;
constexpr int NUM_KV_HEADS = 4;

constexpr int HIDDEN_SIZE = 7680;
const float scale = 0.08838834764831845f; // 128^-0.5
#else
    #error "Invalid model type. Please define MODEL_TYPE as LFM_1_2B, LFM_2_6B, or LFM3_1_7B"
#endif

// Per-family constants (LFM2: conv width 3, 2 norms/layer, vocab 65536;
// LFM3: conv width 7, 4 norms/layer (sandwich), vocab 128000 padded to a multiple of D).
#if MODEL_TYPE == LFM3_1_7B
// S buffer of an attention core pair (bf16 elements): P [8 rows][16 tokens] | c [8 floats] | gates [4 bf16, padded to 8]
constexpr int GATE_OFFSET = 8 * 16 + 16;
constexpr int ATTN_S_LEN = GATE_OFFSET + 16;
constexpr int CONV_K = 7;
constexpr float QK_NORM_EPS = 1e-5f;      // LFM3 norm_eps
// Sandwich norm: the rms tile holds all NORMS_PER_LAYER norm weights of a layer in one buffer (one DMA per layer) and the
// kernel indexes them: operator, post_operator, ffn, post_ffn. (A one-weight buffer refilled between the norms deadlocks:
// the later weights can only land after earlier ones are consumed, which blocks the stream packets behind them.)
constexpr bool SANDWICH_NORM = true;
constexpr int NORMS_PER_LAYER = 4;
constexpr int VOCAB_PADDED = ((128000 + D - 1) / D) * D; // 129024
constexpr bool LM_HEAD_HAS_MIN = true;  // tied lm_head is Q4_1
#else
constexpr int CONV_K = 3;
constexpr float QK_NORM_EPS = 1e-6f;
constexpr bool SANDWICH_NORM = false;
constexpr int NORMS_PER_LAYER = 1;   // one weight buffer, refilled between the two norms
constexpr int VOCAB_PADDED = 65536;
constexpr bool LM_HEAD_HAS_MIN = false;
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
