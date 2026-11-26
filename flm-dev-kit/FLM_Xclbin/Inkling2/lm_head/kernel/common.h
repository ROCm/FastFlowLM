// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef COMMON_H
#define COMMON_H

#include <aie_api/aie.hpp>
// #include <aie2p_locks.h>
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
    constexpr int r = 256 / (sizeof(T) * 8); // one 256 bit store unit
    static_assert((M) % r == 0);
    const aie::vector<T, r> zeros = aie::zeros<T, r>();
    const T *__restrict c_end = c + M;
    for (; c < c_end; c += r) {
        aie::store_v(c, zeros);
    }
}

#endif
