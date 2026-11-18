//===- txn_strategy.hpp - load with command-processor writes -------*- C++ -*-===//
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
// Transport::txn: the kernel's own instruction stream writes every register.
// The command processor runs the ops in order, so a write has landed before
// the next op starts: no barrier, no answer, no token is needed (AIE4 needs
// them for its six uCs and its control packets).
//
// load(k), class by class over all tiles of k.run (each class in the order of
// aiecc's configure sequence):
//
//   program   PM of every core (block writes; on AIE2P there is no ping-pong,
//             so every load writes the program)
//   data      initial values of buffers
//   locks     the lock values of the app (required after the DMA reset, see
//             kernel/reset.hpp): the locks it declares or its cores acquire
//             (Package::needs_lock; aiecc also writes 0 to every other lock of
//             a core tile, which no one of the app uses), and any lock aiecc
//             sets to a value other than 0
//   bds       core and mem-tile BDs
//   mux       shim mux / demux
//   starts    the chain starts (Start_Queue, Ctrl) of the core and mem-tile channels
//   cores     unreset, enable: each core starts at PC 0
//
// load_program(k): per core, Core_Control reset, the program, unreset, enable.
// Nothing else changes. Measured on npu5: ~1.4 us per KB of program
// (aie2-launch/vadd_vmul_bf16 README).
//
// Command-processor writes run at about 0.7 KB/us (measured). Control packets
// on the shim DMAs could write the columns in parallel: a later transport.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/transport/strategy.hpp"

namespace launch::detail {

class TxnStrategy final : public TransportStrategy {
public:
  void load(KernelContext &k) override {
    const Package &p = *k.run;
    auto each = [&](auto member) {
      for (const auto &[tile, init] : p.tiles)
        for (const txn::Op &op : init.*member)
          k.ops.push_back(op);
    };
    each(&TileInit::program);
    each(&TileInit::data);
    for (const auto &[tile, init] : p.tiles)
      for (const auto &[off, value] : init.locks)
        if (value != 0 || p.needs_lock(tile, off))
          k.ops.push_back(txn::Write32{tile.addr(off), value});
    each(&TileInit::bds);
    each(&TileInit::mux);
    each(&TileInit::starts);
    for (const Tile &c : p.cores())
      core_start(k.ops, c);
  }

  void load_program(KernelContext &k) override {
    const Package &p = *k.run;
    for (const Tile &c : p.cores()) {
      core_hold(k.ops, c);
      for (const txn::Op &op : p.tiles.at(c).program)
        k.ops.push_back(op);
      core_start(k.ops, c);
    }
  }
};

} // namespace launch::detail
