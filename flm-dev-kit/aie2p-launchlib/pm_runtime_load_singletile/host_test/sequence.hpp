//===- sequence.hpp - the runtime sequence of each design ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// Portions of this file consist of AI-generated content.
//
//===----------------------------------------------------------------------===//
//
// All runtime sequences of the host program are in this file: one C++ function
// for each design. It is NOT part of the launch library. The host gives the
// library the map `sequences()` (app name -> function): `launch::build(list,
// sequences())`. The library calls the function of the app that a kernel runs.
//
// A function is the C++ form of the Python `runtime_sequence` of that design
// (`<name>_fp32/<name>_fp32_placed.py`), with the same calls in the same order.
// IRON's task API (shim_dma_single_bd_task, dma_start_task, dma_await_task)
// lowers to: every BD (written, not pushed; BD ids 0, 1, 2, ... per shim in task
// order), then every push in task order, then a wait per awaited task. `Tasks`
// below does the same with the calls of `launch::Rt`.
//
// `make check` (checks/check_sequence.cpp) builds every function and compares
// it with the package's insts.bin (aiecc's lowering of the Python sequence),
// word for word.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/rt/rt.hpp"

#include <format>
#include <map>
#include <string>
#include <vector>

#ifndef COLS
#define COLS 4
#endif

namespace hosttest {

using launch::Arg;
using launch::Rt;
using launch::ShimChannel;

// All three designs use COLS whole columns (0 .. COLS-1) of TILES = 4 cores; one
// core works on N = 256 fp32. Column k has its own shim channels (the Python
// names A_k, B_k, C_k) and owns slice k of B and C (COL_ELEMS elements at
// offset k * COL_ELEMS).
constexpr uint32_t kCols = COLS, TILES = 4, N = 256, COL_ELEMS = TILES * N;
constexpr uint32_t kElems = kCols * COL_ELEMS; ///< B and C (and A of vmul, vminus)

/// IRON's task API on Rt: add() writes a BD (the next free id of that shim),
/// start() pushes them all in order, await() waits on the tasks with a token.
class Tasks {
public:
  explicit Tasks(Rt &rt) : rt_(rt) {}

  /// One 1-d transfer of `len` fp32 of argument `arg`, from element `off`, on
  /// the shim channel of the object FIFO (or shim_dma_allocation) `fifo`.
  void add(const std::string &fifo, unsigned arg, uint32_t off, uint32_t len, bool token = false) {
    const ShimChannel ch = rt_.alloc(fifo);
    const uint32_t bd = next_bd_[ch.col]++;
    rt_.npu_dma_memcpy_nd({.metadata = ch, .bd_id = bd, .mem = Arg{arg, sizeof(float)}, .offsets = {0, 0, 0, off},
                           .sizes = {1, 1, 1, len}, .strides = {0, 0, 0, 1}, .issue_token = token,
                           .push = false});
    tasks_.push_back({ch, bd, token});
  }
  void start() {
    for (const Task &t : tasks_)
      rt_.npu_push_bd(t.ch, t.bd, t.token);
  }
  /// The waits, in task order: a column's C comes from its inputs, so its
  /// token covers them.
  void await() {
    for (const Task &t : tasks_)
      if (t.token)
        rt_.npu_dma_wait(t.ch);
  }

private:
  struct Task {
    ShimChannel ch;
    uint32_t bd;
    bool token;
  };
  Rt &rt_;
  std::map<int, uint32_t> next_bd_;
  std::vector<Task> tasks_;
};

inline std::string col(const char *fifo, uint32_t k) { return std::format("{}_{}", fifo, k); }

// vadd_fp32_placed.py (A: N fp32, sent once from the shim of column 0 to every
// core; B, C: kElems fp32):
//   task A_0: A, N                               (column 0's shim)
//   for k: task B_k: B, offset k*COL_ELEMS, COL_ELEMS
//          task C_k: C, offset k*COL_ELEMS, COL_ELEMS, issue_token
//   dma_start_task(all); dma_await_task(every C_k)
inline void vadd_sequence(Rt &rt) {
  Tasks t(rt);
  t.add("A_0", 0, 0, N);
  for (uint32_t k = 0; k < kCols; ++k) {
    t.add(col("B", k), 1, k * COL_ELEMS, COL_ELEMS);
    t.add(col("C", k), 2, k * COL_ELEMS, COL_ELEMS, true);
  }
  t.start();
  t.await();
}

// A second sequence of vadd (key "vadd_rev", not an app): the same transfers,
// the columns in reverse order. A config selects it with `"sequence": "vadd_rev"`
// (sequence.json). Same result as vadd.
inline void vadd_rev_sequence(Rt &rt) {
  Tasks t(rt);
  t.add("A_0", 0, 0, N);
  for (uint32_t k = kCols; k-- > 0;) {
    t.add(col("B", k), 1, k * COL_ELEMS, COL_ELEMS);
    t.add(col("C", k), 2, k * COL_ELEMS, COL_ELEMS, true);
  }
  t.start();
  t.await();
}

// vmul_fp32_placed.py (A, B, C: kElems fp32): column k moves slice k of A, B
// and C, then the sequence waits for every C_k.
inline void vmul_sequence(Rt &rt) {
  Tasks t(rt);
  for (uint32_t k = 0; k < kCols; ++k) {
    t.add(col("A", k), 0, k * COL_ELEMS, COL_ELEMS);
    t.add(col("B", k), 1, k * COL_ELEMS, COL_ELEMS);
    t.add(col("C", k), 2, k * COL_ELEMS, COL_ELEMS, true);
  }
  t.start();
  t.await();
}

// vminus_fp32_placed.py: the transfers of vmul (A goes from the shim's MM2S 1
// to the stream port of every core of the column, B to the core DMA: a route,
// not a different shim transfer). Before the transfers of column k, each core
// r of it gets its RTP:
//   npu_rtp_write("ct_rtp_k_r", 0, 256); set_lock_value(ct_rtplock_k_r, 1)
inline void vminus_sequence(Rt &rt) {
  Tasks t(rt);
  for (uint32_t k = 0; k < kCols; ++k) {
    for (uint32_t r = 0; r < TILES; ++r) {
      rt.npu_rtp_write(std::format("ct_rtp_{}_{}", k, r), 0, N);
      rt.set_lock_value(std::format("ct_rtplock_{}_{}", k, r), 1);
    }
    t.add(col("A", k), 0, k * COL_ELEMS, COL_ELEMS);
    t.add(col("B", k), 1, k * COL_ELEMS, COL_ELEMS);
    t.add(col("C", k), 2, k * COL_ELEMS, COL_ELEMS, true);
  }
  t.start();
  t.await();
}

/// The registry: the key of a runtime sequence -> the function. The key is the
/// app name, or the `sequence` of a config (a second sequence of an app).
inline launch::SequenceMap sequences() {
  return {{"vadd", vadd_sequence},
          {"vadd_rev", vadd_rev_sequence},
          {"vmul", vmul_sequence},
          {"vminus", vminus_sequence}};
}

} // namespace hosttest
