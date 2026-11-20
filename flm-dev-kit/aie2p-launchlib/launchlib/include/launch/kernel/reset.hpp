//===- reset.hpp - the reset part of a kernel ---------------------*- C++ -*-===//
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
// WHICH registers (`reset_targets`), by scope:
//
//   whole_array  (reset, cold)  every core, every core and mem-tile DMA channel,
//                               every stream-switch routing register and shim
//                               mux of the 8 columns (hw_gen.hpp), the locks of
//                               every core tile. Any kernel, also of another ELF
//                               or xclbin, may have run before.
//   used_by      (warm)         the cores, the started DMA channels, the switch
//                               registers and the shim muxes of `prev` and `run`;
//                               the locks prev's cores acquire (the package's
//                               census; without one, every lock aiecc writes in
//                               prev's core tiles). Only `prev`
//                               can have state in the array (the host keeps the
//                               order). A switch register that both set to the
//                               same value is not written.
//
// HOW (`reset_part`), in this order, the same for every transport:
//
//   1  Core_Control: reset on, enable off       the cores stop; a core that waits
//                                               on a lock (cores loop forever)
//                                               stops too, but see 5
//   2  DMA channel Ctrl: reset bit on           the chains of the old app stop
//   3  switch registers: run's value, or 0      block writes of consecutive runs
//      shim mux and demux: 0                    aiecc sets them with mask writes,
//                                               so the old app's bits would stay
//   4  DMA channel Ctrl: 0                      reset bit off, and the bits the old
//                                               app set on a channel run does not
//                                               use (the load writes run's)
//   5  core-tile locks: 63, then 0              the lock drain, below
//
// The lock drain. A core stopped in step 1 while it waits in a lock acquire
// leaves the acquire pending in the lock's tile; the core reset does not drop
// it. Without step 5 the load's lock writes grant it: one token of the next app
// is gone, and its cores and DMAs get out of step (wrong data, or a hang). A
// core waits only on locks its code acquires (of its own tile or of a core
// tile next to it), so a warm kernel drains those of prev's cores: the census
// make_package.py takes from the core ELFs (2 per core in vadd_vmul_bf16: 64
// locks on 32 cores, not 512). Writing
// the largest value grants every pending "acquire >= n"; 0 after it leaves the
// locks as a reset does. It must come after step 4: before it, the old app's
// DMA chains would take the tokens and move data (measured: every run wrong).
// Measured on the Whisper encoder (one matmul and one attention app): without
// it the attention after a matmul is always wrong and the engine hangs; with
// it, the same output as two hw_contexts (whisper_encoder/README.md). Not
// covered: a core that waits for a lock to equal a value (an acquire_eq; IRON's
// object FIFOs use acquire >= n). A program-only warm kernel has no reset part,
// so no drain (without the DMA reset before it, the running chains would take
// the tokens): its program runs on the same floorplan and waits on the same
// locks. Measured: ~10^5 program-only switches in the stress and pressure runs,
// no wrong result. Why the stopped core's acquire does no harm there is not
// known (one pending acquire per core port, taken over by the new program's,
// would explain it; not checked).
//
// The load part then writes the locks again. That is required: an idle S2MM
// channel holds the lock of its next BD (it acquires it when the BD starts,
// before data comes), and the channel reset drops it without a release. A
// reset without the lock writes after it hung the next launch (measured,
// aie2-launch/vadd_vmul_bf16 README).
//
// Shim DMA channels are not reset: each runtime sequence writes and starts its
// own shim BDs, and waits for them. The AIE2P shim channel has no reset bit, so
// no kernel can clear a shim task that never finishes. A sequence that runs on
// the wrong array (a warm or rerun kernel after the wrong prev, or after another
// hw_context put its xclbin back) can leave one: its stream has no route. It
// stays through every later kernel, cold ones too, and the first app that routes
// that channel gets its data first (measured, pm_runtime_load_singletile S5).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/kernel/context.hpp"

#include <map>
#include <set>
#include <vector>

namespace launch::detail {

enum class ResetScope { none, whole_array, used_by };

struct ResetTargets {
  std::set<Tile> cores;
  std::set<uint32_t> channels;                    ///< array addresses of DMA channel Ctrl registers
  RouteMap routes;                                ///< switch registers to write, with their values
  std::set<Tile> muxes;                           ///< shim tiles whose mux and demux go to 0
  std::set<uint32_t> drain;                       ///< array addresses of the core-tile locks to drain
};

/// The Ctrl register of the channel a chain-start write (Start_Queue or Ctrl)
/// belongs to. Both checks need the 8-byte step: 4 bytes before the first
/// MM2S Ctrl is the last S2MM Start_Queue, not a Ctrl. (A reset mask write on a
/// Start_Queue pushes a BD: measured, wrong data.)
inline uint32_t channel_ctrl_of(uint32_t addr) {
  const hw::Where w = hw::locate(addr);
  auto is_ctrl = [](const hw::Bank &b, uint32_t off) { return b.holds(off) && (off - b.base) % b.step == 0; };
  for (hw::Dir d : {hw::Dir::s2mm, hw::Dir::mm2s}) {
    const hw::Bank b = hw::channels(w.kind, d);
    if (is_ctrl(b, w.offset))
      return addr;
    if (w.offset >= hw::kQueueFromCtrl && is_ctrl(b, w.offset - hw::kQueueFromCtrl))
      return addr - hw::kQueueFromCtrl;
  }
  throw std::logic_error("channel_ctrl_of: not a DMA channel register");
}

/// The value of register `off` of `tile` in `m`, or 0.
inline uint32_t route_value(const RouteMap *m, Tile tile, uint32_t off) {
  if (!m)
    return 0;
  auto it = m->find(tile);
  if (it == m->end())
    return 0;
  auto r = it->second.find(off);
  return r == it->second.end() ? 0 : r->second;
}

/// The registers of the reset part. The switch registers come from the
/// transport (transport/strategy.hpp): `final_prev`, what prev's load left in
/// the array (warm only); `load_run`, what run's load needs (null for a reset
/// kernel). With TXN writes both are the apps' routes.
inline ResetTargets reset_targets(ResetScope scope, const KernelContext &k, const RouteMap *final_prev,
                                  const RouteMap *load_run) {
  ResetTargets t;
  if (scope == ResetScope::whole_array) {
    for (int col = 0; col < hw::kCols; ++col)
      for (int row = 0; row < hw::kRows; ++row) {
        const Tile tile{col, row};
        const hw::Kind kind = tile.kind();
        if (kind == hw::Kind::core) {
          t.cores.insert(tile);
          for (uint32_t i = 0; i < hw::locks(kind).count; ++i)
            t.drain.insert(tile.addr(hw::locks(kind).at(i)));
        }
        if (kind != hw::Kind::shim) // a shim channel has no reset bit (bit 1 is Pause_Mem)
          for (hw::Dir d : {hw::Dir::s2mm, hw::Dir::mm2s})
            for (uint32_t i = 0; i < hw::channels(kind, d).count; ++i)
              t.channels.insert(tile.addr(hw::channels(kind, d).at(i)));
        std::map<uint32_t, uint32_t> &regs = t.routes[tile];
        auto all = [&](const auto &list) {
          for (uint32_t off : list)
            regs[off] = route_value(load_run, tile, off);
        };
        if (kind == hw::Kind::core)
          all(hw::kSwitchRegs_core);
        else if (kind == hw::Kind::mem)
          all(hw::kSwitchRegs_mem);
        else {
          all(hw::kSwitchRegs_shim);
          t.muxes.insert(tile);
        }
        if (load_run && load_run->count(tile))
          for (const auto &[off, value] : load_run->at(tile))
            regs[off] = value;
      }
  } else if (scope == ResetScope::used_by) {
    for (const Package *p : {k.prev, k.run}) {
      for (const Tile &c : p->cores())
        t.cores.insert(c);
      for (const auto &[tile, init] : p->tiles) {
        if (tile.kind() != hw::Kind::shim) // as whole_array: bit 1 of a shim Ctrl is Pause_Mem
          for (const txn::Op &op : init.starts)
            t.channels.insert(channel_ctrl_of(txn::addr_of(op)));
        if (!init.mux.empty())
          t.muxes.insert(tile);
      }
    }
    for (const RouteMap *m : {final_prev, load_run})
      for (const auto &[tile, regs] : *m)
        for (const auto &[off, value] : regs)
          t.routes[tile][off] = route_value(load_run, tile, off);
    // Skip what prev already set to run's value.
    for (auto &[tile, regs] : t.routes) {
      const auto pt = final_prev->find(tile), rt = load_run->find(tile);
      if (pt == final_prev->end() || rt == load_run->end())
        continue;
      std::erase_if(regs, [&](const auto &reg) {
        const auto a = pt->second.find(reg.first), b = rt->second.find(reg.first);
        return a != pt->second.end() && b != rt->second.end() && a->second == b->second;
      });
    }
    std::erase_if(t.routes, [](const auto &e) { return e.second.empty(); });
    // A core waits only on a lock its code acquires: the census of prev's
    // package (make_package.py), else every lock aiecc writes in prev's core
    // tiles.
    if (k.prev->census)
      t.drain = k.prev->waits;
    else
      for (const auto &[tile, init] : k.prev->tiles)
        if (tile.kind() == hw::Kind::core)
          for (const auto &[off, value] : init.locks)
            t.drain.insert(tile.addr(off));
  }
  return t;
}

/// The writes of the reset part, in the order of the table above.
inline void write_reset(KernelContext &k, const ResetTargets &t) {
  for (const Tile &c : t.cores)
    core_hold(k.ops, c);
  for (uint32_t ctrl : t.channels)
    k.ops.push_back(txn::MaskWrite32{ctrl, hw::kDmaReset, hw::kDmaReset});
  for (const auto &[tile, values] : t.routes)
    write_runs(k.ops, tile, values);
  for (const Tile &s : t.muxes) {
    k.ops.push_back(txn::Write32{s.addr(hw::kShimMux), 0});
    k.ops.push_back(txn::Write32{s.addr(hw::kShimDemux), 0});
  }
  for (uint32_t ctrl : t.channels)
    k.ops.push_back(txn::Write32{ctrl, 0});
  for (uint32_t value : {hw::kLockMax, 0u})
    for (uint32_t lock : t.drain)
      k.ops.push_back(txn::Write32{lock, value});
}

} // namespace launch::detail
