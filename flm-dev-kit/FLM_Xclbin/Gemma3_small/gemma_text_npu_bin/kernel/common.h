// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef COMMON_H
#define COMMON_H

#include <aie_api/aie.hpp>
// #include <aie2p_locks.h>
typedef bfloat16 bf16;

// Ping/pong buffer selectors that persist across calls. Under Peano they live in
// .data because .bss is not zeroed at load.
#if defined(__chess__)
#define AIE_PING_FLAG bool
#else
#define AIE_PING_FLAG __attribute__((section(".data"))) volatile bool
#endif

// Packet-id word written into an output buffer just before its lock is released.
// Volatile under Peano so the store is not moved past the release.
#if defined(__chess__)
typedef uint32_t aie_pkt_word;
#else
typedef volatile uint32_t aie_pkt_word;
#endif

// Lock-loop counter type: peano hoists lock releases out of the do{}while(i>0)
// lock loops unless the counter is volatile. Chess keeps a plain int.
#if defined(__chess__)
#define AIE_LOCK_LOOP_CTR int
#else
#define AIE_LOCK_LOOP_CTR volatile int
#endif

// Scalar float +, -, * for the RMS-norm scale. chess rounds each one as IEEE
// round-to-nearest-even softfloat; Peano would fuse a*b+c and round differently,
// so under Peano these call compiler-rt's IEEE softfloat.
#if defined(__chess__)
#define AIE_FADD(a, b) ((a) + (b))
#define AIE_FMUL(a, b) ((a) * (b))
#else
extern "C" float __addsf3(float, float);
extern "C" float __mulsf3(float, float);
#define AIE_FADD(a, b) __addsf3((a), (b))
#define AIE_FMUL(a, b) __mulsf3((a), (b))
#endif

// Model type definitions
#define GEMMA3_1B 1
#define GEMMA3_270M 2

#ifndef MODEL_TYPE
#define MODEL_TYPE GEMMA3_270M
#endif

#include <aie_api/aie.hpp>
// #include <aie2p_locks.h>
typedef bfloat16 bf16;

#if MODEL_TYPE == GEMMA3_1B
// 1B model parameters
constexpr int D = 1152;
constexpr int NUM_ATTN_HEADS = 4;
constexpr int NUM_KV_HEADS = 1;
constexpr int HEADS = 4;
constexpr int DQ = 1024;
constexpr int DK = DQ / 4;
constexpr int DV = DK;
const int DH = DQ / HEADS;
constexpr int HIDDEN_SIZE = 6912;
const float scale = 0.0625f;
#elif MODEL_TYPE == GEMMA3_270M
// 270M model parameters
constexpr int D = 640;
constexpr int NUM_ATTN_HEADS = 4;
constexpr int NUM_KV_HEADS = 1;
constexpr int HEADS = 4;
constexpr int DQ = 1024;
constexpr int DK = DQ / 4;
constexpr int DV = DK;
const int DH = DQ / HEADS;
constexpr int HIDDEN_SIZE = 2048;
const float scale = 0.0625f;
#else
#error "Invalid model type. Please define MODEL_TYPE as GEMMA3_4B"
#endif

constexpr int GELU_SLICE = 256;
const bf16 exp_scale = 1.4426950408889634f;

constexpr int ATTN_GROUPS = NUM_ATTN_HEADS / NUM_KV_HEADS;
constexpr int PADDED_D = (D + 255) / 256 * 256;

const int cores = 16;
const int qkv_repeats = (DQ + DK + DV) / cores / 16;
const int up_gate_repeats = (2 * HIDDEN_SIZE) / cores / 16;


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
