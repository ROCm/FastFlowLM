//===- elementwise_sequence.hpp - the runtime sequence of both apps -*- C++ -*-===//
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
// The C++ copy of the runtime_sequence in iron/vadd_vmul_bf16_aie.py, the same
// calls in the same order (IRON's task API: all BDs, then all pushes, then the
// waits), for any length. vadd and vmul have the same floorplan, so one
// function serves both: out = lhs (op) rhs on whichever program the cores run.
// `make check` compares it with aiecc's instructions (the package's insts.bin).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/rt/rt.hpp"

#include "elementwise_config.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace ew {

// Kernel arguments of the sequence: lhs, rhs, out (XRT arguments 3, 4, 5).
constexpr unsigned kArgLhs = 0, kArgRhs = 1, kArgOut = 2;
constexpr uint32_t kLhsBd = 0, kRhsBd = 1, kOutBd = 2;

// The smallest length: one block for every core of the array.
constexpr int kBlockElements = EW_N * EW_COLS * EW_CORES;

/// A partial block leaves a DMA holding a buffer whose lock is never released,
/// and a core waits on it forever.
inline void check_length(int elements) {
  if (elements <= 0 || elements % kBlockElements != 0)
    throw std::invalid_argument("length " + std::to_string(elements) + " must be a positive multiple of " +
                                std::to_string(kBlockElements) + " (EW_N x columns x cores)");
}

/// Blocks each core computes in one launch of `elements`.
inline int blocks_per_core(int elements) { return elements / kBlockElements; }

/// Two inputs in, one out. Column c moves slice c of each buffer. The DMAs
/// count i32 words, two bf16 each, as the IRON design. The waits make the
/// output reach DDR before the kernel ends, so the next launch can read it.
/// `preempt`: start with a preemption point, as FLM's sequences on Windows.
inline void stage(launch::Rt &rt, int elements, bool preempt = false) {
  if (preempt)
    rt.npu_preempt(0);
  const uint32_t col_words = (uint32_t)elements / 2 / EW_COLS;
  std::vector<std::pair<launch::ShimChannel, uint32_t>> pushes;
  for (int c = 0; c < EW_COLS; ++c) {
    const uint32_t off = (uint32_t)c * col_words;
    const std::string col = std::to_string(c);
    auto copy = [&](const std::string &fifo, uint32_t bd, unsigned arg, bool token) {
      const launch::ShimChannel ch = rt.alloc(fifo + col);
      rt.npu_dma_memcpy_nd({.metadata = ch, .bd_id = bd, .mem = launch::Arg{arg, 4}, .offsets = {0, 0, 0, off},
                            .sizes = {1, 1, 1, col_words}, .strides = {0, 0, 0, 1}, .issue_token = token,
                            .push = false});
      pushes.push_back({ch, bd});
    };
    copy("lhs", kLhsBd, kArgLhs, false);
    copy("rhs", kRhsBd, kArgRhs, false);
    copy("out", kOutBd, kArgOut, true);
  }
  for (const auto &[ch, bd] : pushes)
    rt.npu_push_bd(ch, bd, bd == kOutBd);
  // A column's output comes from its inputs: its token covers them.
  for (int c = 0; c < EW_COLS; ++c)
    rt.npu_dma_wait(rt.alloc("out" + std::to_string(c)));
}

} // namespace ew
