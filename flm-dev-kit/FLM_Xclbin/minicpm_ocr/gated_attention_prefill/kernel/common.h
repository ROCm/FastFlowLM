// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

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

constexpr int lq = 16;
constexpr int lk = 16;
constexpr int dh = 256;
// a core takes rows_per_core query rows (halves of lq) per round; a round is 256 rows per q head
constexpr int halves = 2;
constexpr int rows_per_core = halves * lq;
constexpr int rows_per_round = 8 * rows_per_core;
constexpr int lk_block = 128;

const float scale = 0.0625f;

// bfp16 tiles: 8 x 8 values, each row one block of an exponent byte + 8 mantissas
constexpr int bfp16_tile_bytes = 72;
constexpr int q_bfp_bytes = (lq / 8) * (dh / 8) * bfp16_tile_bytes;
constexpr int kv_bfp_bytes = (lk / 8) * (dh / 8) * bfp16_tile_bytes;

constexpr bf16 exp_scale = (bf16)1.4426950408889634f;
constexpr bf16 neg_exp_scale = (bf16)-1.4426950408889634f;

// The running max only moves when a block's max is this far above it, so exp(s - m) <= e^5.5 (< 2^8).
constexpr bf16 lazy_max_step = (bf16)5.5f;

// Peano's CDO loader does not zero .bss, so a zero-initialized static that must
// carry its value across kernel invocations has to be forced into .data. chess's
// loader does zero .bss, so this expands to nothing there and the chess memory
// layout stays byte-identical.
#if defined(__chess__)
#define AIE_PERSIST_ZERO
#else
#define AIE_PERSIST_ZERO __attribute__((section(".data")))
#endif

#endif
