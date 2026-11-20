// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef COMMON_H
#define COMMON_H

#include <aie_api/aie.hpp>
typedef bfloat16 bf16;

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

// Qwen3-VL vision encoder attention: 16 heads, head_dim 72, no GQA, full
// (bidirectional) mask. L is not a compile-time constant here - it arrives per
// dispatch through the L_buffer RTP, so only the chunk size is fixed.
constexpr int lk_mt = 512;
constexpr int chunk_size = lk_mt;
constexpr int lq = 64;
constexpr int lk = 16;
constexpr int dh = 72;

// 9, and odd: the f*v column loop steps two blocks at a time and so needs a
// tail, and the k/v memtile DMA size must track this rather than the old 8.
constexpr int d_blocks = dh / 8;

constexpr bf16 exp_scale = (bf16)1.4426950408889634f;
const float scale = 0.11785113019775793f; // 1 / sqrt(72)
constexpr bf16 neg_inf_bf16 = -0x1.FEp127f;

#endif
