// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef COMMON_H
#define COMMON_H

#include <aie_api/aie.hpp>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


typedef bfloat16 bf16;

// Peano (llvm-aie, non-chess) does not provide the raw chess loop pragmas; make
// them no-ops so kernels that use them directly still compile.
#ifndef __chess__
#define chess_prepare_for_pipelining
#define chess_loop_range(...)
#define chess_flatten_loop
#endif

// Keep lock-loop trip counters opaque on peano so lock releases are not hoisted
// out of the loop (which desyncs the static lock protocol -> deadlock). chess is
// unaffected and uses a plain int.
#if defined(__chess__)
#define AIE_LOCK_LOOP_CTR int
#else
#define AIE_LOCK_LOOP_CTR volatile int
#endif





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



// Common to slave and master CT for router

constexpr uint32_t M_WEIGHT_PROD_LOCK = 2;
constexpr uint32_t M_WEIGHT_CON_LOCK = 3;
constexpr uint32_t M_D_PROD_LOCK = 0;
constexpr uint32_t M_D_CON_LOCK = 1;
constexpr uint32_t M_SCORE_PROD_LOCK = 4;
constexpr uint32_t M_SCORE_CON_LOCK = 5;
constexpr uint32_t M_Prod_S_Con_LOCK = 10;
constexpr uint32_t M_Con_S_Prod_LOCK = 11;

constexpr float scale = 0.125f;

#endif