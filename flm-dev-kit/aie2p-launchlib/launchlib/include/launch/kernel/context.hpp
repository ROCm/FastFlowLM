//===- context.hpp - what the parts of one kernel share ------------*- C++ -*-===//
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
// `KernelContext` is the one argument of every part of a kernel: the op list
// the parts append to, the configuration, and the packages of the app it runs
// (`run`) and of the app before it (`prev`, warm only), found once by name.
// The runtime sequence is built first (the pipeline appends it last): a
// transport that sends its own shim DMA tasks keeps clear of the sequence's
// BDs and arguments. Such a transport puts its control packets in `packets`,
// for kernel argument `packet_arg`.
//
// `write_runs` writes a set of register values of one tile with as few ops as
// it can: one block write for each run of consecutive 32-bit registers, a
// Write32 for a register alone.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/config.hpp"
#include "launch/formats/package.hpp"
#include "launch/txn/txn.hpp"

#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace launch::detail {

using Apps = std::map<std::string, Package>;

struct KernelContext {
  std::vector<txn::Op> &ops;
  const Config &c;
  const Package *run = nullptr;  ///< null for a reset kernel
  const Package *prev = nullptr; ///< warm only
  const std::vector<txn::Op> *sequence = nullptr; ///< the runtime sequence's ops (null: none)
  std::set<unsigned> sequence_args;               ///< the kernel arguments the sequence uses
  int free_arg = -1;                              ///< a kernel argument the sequence leaves free (-1: none)
  std::vector<uint32_t> packets;                  ///< control packets the transport sends (its buffer)
  int packet_arg = -1;                            ///< their kernel argument (-1: no packets)
  std::set<uint32_t> drain_by_load;               ///< reset drain locks the transport's load drains instead

  KernelContext(std::vector<txn::Op> &out, const Apps &apps, const Config &config) : ops(out), c(config) {
    if (!config.app().empty())
      run = &find(apps, config.app());
    if (!config.prev().empty())
      prev = &find(apps, config.prev());
  }

  [[noreturn]] void fail(const std::string &m) const {
    throw std::runtime_error("configuration '" + c.name + "': " + m);
  }

private:
  const Package &find(const Apps &apps, const std::string &app) const {
    auto it = apps.find(app);
    if (it == apps.end())
      fail("app '" + app + "' is not loaded");
    return it->second;
  }
};

/// Write `regs` (tile offset -> value) of tile `t`: one block write per run of
/// consecutive registers (4 bytes apart), a Write32 for a register alone.
inline void write_runs(std::vector<txn::Op> &ops, Tile t, const std::map<uint32_t, uint32_t> &regs) {
  auto it = regs.begin();
  while (it != regs.end()) {
    const uint32_t first = it->first;
    std::vector<uint32_t> words{it->second};
    auto next = std::next(it);
    while (next != regs.end() && next->first == first + 4 * (uint32_t)words.size()) {
      words.push_back(next->second);
      ++next;
    }
    if (words.size() == 1)
      ops.push_back(txn::Write32{t.addr(first), words[0]});
    else
      ops.push_back(txn::BlockWrite{t.addr(first), std::move(words)});
    it = next;
  }
}

/// Core_Control writes, as mlir-aie test/npu-xrt/local_reset/core: each one
/// changes one bit field.
inline void core_hold(std::vector<txn::Op> &ops, Tile t) {
  ops.push_back(txn::MaskWrite32{t.addr(hw::kCoreControl), hw::kCoreReset, hw::kCoreReset | hw::kCoreEnable});
}
inline void core_start(std::vector<txn::Op> &ops, Tile t) {
  ops.push_back(txn::MaskWrite32{t.addr(hw::kCoreControl), 0, hw::kCoreReset});            // unreset: PC 0
  ops.push_back(txn::MaskWrite32{t.addr(hw::kCoreControl), hw::kCoreEnable, hw::kCoreEnable}); // enable
}

} // namespace launch::detail
