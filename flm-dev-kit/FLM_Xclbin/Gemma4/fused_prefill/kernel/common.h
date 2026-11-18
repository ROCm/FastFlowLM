// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef COMMON_H
#define COMMON_H

#include <aie_api/aie.hpp>
typedef bfloat16 bf16;

// Peano's CDO loader does not zero .bss, so a zero-initialized static that must
// carry its value across kernel invocations has to be forced into .data
// (AIE_PERSIST_ZERO); and a static that does NOT need to persist is better left
// as an automatic (AIE_TRANSIENT), because the Peano ldscript's data region is
// tight. chess zeroes .bss and has room, so both macros are inert there and the
// chess build is byte-for-byte what it was.
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
// `r` is the store width of the zeroing loop.  It was hard-coded per design (8
// for the DH512 build, 16 for the DH256 one); the fused build compiles both
// variants from these sources, so it comes from the variant macro block below.
#ifndef ATTN_ZERO_R
#define ATTN_ZERO_R 8
#endif

template <typename T, int M>
void zero_vectorized(T *__restrict c) {
    constexpr int r = ATTN_ZERO_R;
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


// ---------------------------------------------------------------------------
// Per-variant geometry.  The standalone designs baked these in; here each of
// attn_full.cc / attn_swa.cc defines them before including this header, so one
// set of sources produces both cores of the fused array.
// ---------------------------------------------------------------------------
#if !defined(ATTN_LQ) || !defined(ATTN_LK) || !defined(ATTN_DH) || !defined(ATTN_LK_MT)
#error "attention variant must define ATTN_LQ / ATTN_LK / ATTN_DH / ATTN_LK_MT"
#endif

constexpr int lq = ATTN_LQ;
constexpr int lk = ATTN_LK;
constexpr int dh = ATTN_DH;

// Tokens of k (and of v) the memtile stages per block.  The standalone designs
// both landed on 128 and wrote `16` / `8` (= lk_mt / lk) as literals in the
// microkernels; the fused full-attention column has to drop to 64 to fit its
// 512-wide k, v, q and o in one 512 KB memtile, so the microkernels now derive
// the count instead.
constexpr int lk_mt = ATTN_LK_MT;
constexpr int n_kblk = lk_mt / lk;

// A fused column owns 4 cores (one per compute row) instead of the 16 (DH512)
// or 8 (DH256) of a multi-column CU, so a q round is 4 * lq tokens rather than
// the 128 the standalone kernels hard-coded.
constexpr int n_rows = 4;
constexpr int q_block = n_rows * lq;

constexpr bf16 exp_scale = (bf16)1.4426950408889634f;

#endif
