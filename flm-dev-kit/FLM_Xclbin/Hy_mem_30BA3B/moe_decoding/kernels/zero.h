// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef __ZERO_H__
#define __ZERO_H__
#define NOCPP
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

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



template <typename T, int M, int N>
void zero_vectorized_512(T *__restrict c) {
  constexpr int r = 512 / (sizeof(T) * 8); // 512 bit store units for AIE2P
  static_assert((M * N) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M * N;
  event0();
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
  event1();
}
#endif
