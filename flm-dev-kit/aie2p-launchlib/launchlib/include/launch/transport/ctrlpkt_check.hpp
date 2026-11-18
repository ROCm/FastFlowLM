//===- ctrlpkt_check.hpp - replay a control-packet kernel ---------*- C++ -*-===//
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
// `check_kernel` runs a kernel of Transport::ctrlpkt on a model of the
// registers (sw::Registers: switch registers, and every other word written)
// and the same kernel of Transport::txn on another, from the same array, and
// says what is wrong:
//
//   pushes    a push of a shim MM2S queue whose BD is patched to the packet
//             argument starts a chain: its BDs (length, next) must stay in the
//             packets; the packets must decode (ctrlpkt_packets.hpp); each one,
//             routed with the switch registers of that moment (sw::route) from
//             the lane's SOUTH slave, must end in exactly the TILE_CTRL its id
//             names (8..11 a core row, 12 the mem tile, 13 every core row of the
//             lane), with no stall, illegal join or dropped header. The shim mux
//             must give that SOUTH slave to the DMA, the channel's Ctrl must send
//             the token to the tile's controller id, and a core whose program
//             memory a packet writes must be held (Core_Control reset bit).
//   ordering  a TXN write between a push and the column's Sync into a register
//             the chain writes, or one of the lane's switch registers (other
//             than through the push of a new chain); a chain never synced;
//             two chains on one column without a Sync between them.
//   end state every word the TXN kernel leaves equals the control-packet
//             kernel's, but for: the lane's residents (switch registers of
//             `final_routes` the app does not set), the lane's DMA bit of a shim
//             mux, the shim BDs of the chains, and the token field of a lane
//             channel's Ctrl. Every switch register of the control-packet model
//             holds final_routes(run) (or 0 where that has none).
//
// The start (CheckInput::start, txn_start): for a reset or cold kernel, an
// empty array (the reset part writes every switch register); for a warm
// kernel, what cold(prev) without a sequence leaves in each model (build()
// replays it once per app, launch.hpp): the shim muxes, lane bits and token
// fields a warm reset does not write come from there.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/hw/switch.hpp"
#include "launch/transport/ctrlpkt_lane.hpp"
#include "launch/transport/ctrlpkt_packets.hpp"
#include "launch/txn/txn.hpp"

#include <bitset>
#include <format>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace launch::ctrlpkt {

namespace check_detail {

inline void apply(sw::Registers &r, const txn::Op &op) {
  std::visit(txn::Overload{
                 [&](const txn::Write32 &o) { r.write(o.addr, o.value); },
                 [&](const txn::MaskWrite32 &o) { r.write(o.addr, o.value, o.mask); },
                 [&](const txn::BlockWrite &o) {
                   for (size_t i = 0; i < o.words.size(); ++i)
                     r.write(o.addr + 4 * (uint32_t)i, o.words[i]);
                 },
                 [](const auto &) {},
             },
             op);
}

/// The shim MM2S channel whose queue (Ctrl + 4) is at `off` of a shim tile, or -1.
inline int mm2s_queue(uint32_t off) {
  const hw::Bank c = hw::channels(hw::Kind::shim, hw::Dir::mm2s);
  if (off < c.base + hw::kQueueFromCtrl)
    return -1;
  const uint32_t ctrl = off - hw::kQueueFromCtrl;
  return c.holds(ctrl) && (ctrl - c.base) % c.step == 0 ? (int)((ctrl - c.base) / c.step) : -1;
}

} // namespace check_detail

/// The model of a control-packet array: its registers, and the words the
/// transports may leave different (the chains' shim BDs, the lane channels' Ctrl).
struct Replay {
  sw::Registers r;
  sw::AddrSet chain_bds, lane_ctrls;
};

/// Run the ops of a control-packet kernel of app `run` (null: a reset kernel)
/// on `st`; what is wrong goes to `fail`.
template <class Fail>
void replay(Replay &st, const Package *run, const Plan *plan, bool memtile_data, const std::vector<txn::Op> &ops,
            const std::vector<uint32_t> &packets, int packet_arg, Fail &&fail) {
  using namespace check_detail;
  sw::Registers &cp = st.r;
  struct Pending {
    int ch;
    sw::AddrSet writes; ///< what the chain writes
    sw::AddrSet lane;   ///< the switch registers of the lane
  };
  std::map<int, Pending> pending;     // column -> its chain, until the Sync
  sw::AddrSet raced;                  // written by a chain already synced
  std::map<uint32_t, uint32_t> patch; // BD address field -> packet offset (bytes)
  for (size_t at = 0; at < ops.size(); ++at) {
    const txn::Op &op = ops[at];
    auto where = [&] { return std::format("op {}", at); }; // built only for an error
    if (const auto *a = std::get_if<txn::AddressPatch>(&op)) {
      if ((int)a->arg == packet_arg)
        patch[a->addr] = a->offset;
      else
        patch.erase(a->addr);
      continue;
    }
    if (const auto *s = std::get_if<txn::Sync>(&op)) {
      auto p = pending.find((int)s->col);
      if (p != pending.end() && s->row == 0 && s->dir == (uint32_t)hw::Dir::mm2s && (int)s->channel == p->second.ch) {
        raced.insert(p->second.writes);
        pending.erase(p);
      }
      continue;
    }
    // A TXN write into what a chain in flight writes.
    for (uint32_t a : txn::written(op))
      for (const auto &[col, p] : pending)
        if (p.writes.count(a) || p.lane.count(a))
          fail(std::format("{}: a write of {:#010x} while column {}'s chain is in flight", where(), a, col));
    // A push of a packet chain.
    const auto *w = std::get_if<txn::Write32>(&op);
    if (w) {
      const hw::Where loc = hw::locate(w->addr);
      const int ch = loc.kind == hw::Kind::shim ? mm2s_queue(loc.offset) : -1;
      const hw::Bank bdb = hw::bds(hw::Kind::shim);
      const uint32_t first = hw::address(loc.col, 0, bdb.at(w->value & 0xF));
      if (ch >= 0 && patch.count(first + 4)) {
        if (!run || !plan) {
          fail(std::format("{}: a packet chain in a kernel without an app", where()));
          continue;
        }
        if (pending.count(loc.col))
          fail(std::format("{}: a second chain on column {} before its Sync", where(), loc.col));
        Pending p{ch, {}, {}};
        if (!(w->value & hw::kQueueToken))
          fail(std::format("{}: column {}'s chain issues no token", where(), loc.col));
        const uint32_t ctrl = w->addr - hw::kQueueFromCtrl;
        st.lane_ctrls.insert(ctrl);
        auto id = run->controller_id.find(Tile{loc.col, 0});
        if (id == run->controller_id.end() ||
            (cp.get(ctrl) & hw::kTokenControllerMask) != id->second << hw::kTokenControllerShift)
          fail(std::format("{}: column {}'s channel sends its token to the wrong controller", where(), loc.col));
        const uint32_t mux = cp.get(hw::address(loc.col, 0, hw::kShimMux));
        if ((mux >> mux_shift(ch) & 3) != kMuxDma)
          fail(std::format("{}: column {}'s shim mux does not give SOUTH {} to the DMA", where(), loc.col,
                           lane_slave(ch).index));
        const Lane *lane = plan->lane(loc.col);
        if (!lane || lane->ch != ch)
          fail(std::format("{}: column {}: a chain on a channel that is not the lane's", where(), loc.col));
        if (lane)
          for (const auto &[t, regs] : lane->regs)
            for (const auto &[off, v] : regs)
              p.lane.insert(t.addr(off));
        // The chain.
        uint32_t bd = w->value & 0xF;
        std::set<uint32_t> seen;
        // Packet id -> its route. Only the chain's packets write the model until
        // the Sync, and a packet `allowed` writes no switch register; one that is
        // not allowed drops them all.
        std::map<uint32_t, sw::Reach> routes;
        while (true) {
          const uint32_t ba = hw::address(loc.col, 0, bdb.at(bd));
          if (!seen.insert(bd).second) {
            fail(std::format("{}: column {}'s chain loops at BD {}", where(), loc.col, bd));
            break;
          }
          for (uint32_t i = 0; i < bdb.step / 4; ++i)
            st.chain_bds.insert(ba + 4 * i);
          auto pa = patch.find(ba + 4);
          if (pa == patch.end()) {
            fail(std::format("{}: column {} BD {} is not patched to the packets", where(), loc.col, bd));
            break;
          }
          const uint32_t len = cp.get(ba) & 0x3FFFFFF, off = pa->second;
          if (off % 4 || off / 4 + len > packets.size() || len == 0) {
            fail(std::format("{}: column {} BD {}: {} words at byte {} are not in the packets ({} words)", where(),
                             loc.col, bd, len, off, packets.size()));
            break;
          }
          std::vector<Packet> pk;
          try {
            pk = decode(packets.data() + off / 4, len);
          } catch (const std::exception &e) {
            fail(std::format("{}: column {} BD {}: {}", where(), loc.col, bd, e.what()));
            break;
          }
          for (const Packet &k : pk) {
            std::vector<Tile> want;
            if (k.id == kBroadcastId) {
              for (int row = hw::kFirstCoreRow; lane && row <= lane->top; ++row)
                want.push_back({loc.col, row});
            } else if (k.id == kMemId) {
              want.push_back({loc.col, 1});
            } else if (k.id >= row_id(hw::kFirstCoreRow) && k.id < row_id(hw::kRows)) {
              want.push_back({loc.col, hw::kFirstCoreRow + (int)(k.id - row_id(hw::kFirstCoreRow))});
            } else {
              fail(std::format("{}: column {}: packet id {} names no tile", where(), loc.col, k.id));
              continue;
            }
            auto rt = routes.find(k.id);
            if (rt == routes.end())
              rt = routes.emplace(k.id, sw::route(cp, loc.col, 0, lane_slave(ch), (int)k.id)).first;
            const sw::Reach &r = rt->second;
            for (const std::string &e : r.errors)
              fail(std::format("{}: column {} id {}: {}", where(), loc.col, k.id, e));
            std::set<Tile> got;
            for (const sw::Hop &h : r.sinks) {
              if (h.port.port != sw::Port::tile_ctrl)
                fail(std::format("{}: column {} id {} ends in ({}, {}) {}", where(), loc.col, k.id, h.col, h.row,
                                 sw::port_name(h.port)));
              else
                got.insert({h.col, h.row});
            }
            if (got != std::set<Tile>(want.begin(), want.end()))
              fail(std::format("{}: column {} id {} reaches {} tiles, not the {} it names", where(), loc.col, k.id,
                               got.size(), want.size()));
            for (const Tile &t : want) {
              if (!allowed(t.kind(), k.offset, k.words.size(), memtile_data)) {
                fail(std::format("{}: a packet into {} at {:#x} it may not write", where(), t.describe(), k.offset));
                routes.clear(); // r is not used below
              }
              if (t.kind() == hw::Kind::core && k.offset >= hw::kPm && k.offset < hw::kPm + hw::kPmBytes &&
                  !(cp.get(t.addr(hw::kCoreControl)) & hw::kCoreReset))
                fail(std::format("{}: a packet into the program of {} while its core runs", where(), t.describe()));
              for (size_t i = 0; i < k.words.size(); ++i) {
                const uint32_t a = t.addr(k.offset + 4 * (uint32_t)i);
                cp.write(a, k.words[i]);
                p.writes.insert(a);
              }
            }
          }
          const uint32_t w7 = cp.get(ba + 28);
          if (!(w7 >> 26 & 1))
            break;
          bd = w7 >> 27 & 0xF;
        }
        pending[loc.col] = std::move(p);
        continue;
      }
    }
    // A TXN write into a word a synced chain wrote: the packets may be in flight.
    for (uint32_t a : txn::written(op))
      if (raced.count(a))
        fail(std::format("{}: a write of {:#010x} that a synced chain also wrote (the packets may be in flight)",
                         where(), a));
    apply(cp, op);
  }
  for (const auto &[col, p] : pending)
    fail(std::format("column {}'s chain is never synced", col));
}

struct CheckInput {
  const Package *run = nullptr;  ///< null: a reset kernel
  const Plan *plan = nullptr;    ///< run's lanes
  bool memtile_data = false;
  const std::vector<txn::Op> &ops;    ///< the control-packet kernel
  const std::vector<uint32_t> &packets;
  int packet_arg = -1;
  const std::vector<txn::Op> &txn_ops; ///< the same kernel with Transport::txn
  const Replay *start = nullptr;       ///< the control-packet array before (null: empty)
  const sw::Registers *txn_start = nullptr; ///< the TXN array before
};

/// What is wrong with a control-packet kernel (empty: nothing). `end`, when
/// given, gets the control-packet array after the kernel.
inline std::vector<std::string> check_kernel(const CheckInput &in, Replay *end = nullptr) {
  using namespace check_detail;
  std::vector<std::string> errors;
  auto fail = [&](std::string m) {
    if (errors.size() < 20)
      errors.push_back(std::move(m));
  };
  Replay st = in.start ? *in.start : Replay{};
  sw::Registers tx = in.txn_start ? *in.txn_start : sw::Registers{};
  sw::Registers &cp = st.r;
  replay(st, in.run, in.plan, in.memtile_data, in.ops, in.packets, in.packet_arg, fail);

  // The TXN kernel.
  for (const txn::Op &op : in.txn_ops)
    apply(tx, op);

  // The end states.
  std::map<uint32_t, uint32_t> lane_mux; // shim mux address -> the lane bits it may have more
  static const Plan no_plan;
  const Plan &plan = in.plan ? *in.plan : no_plan;
  for (const Lane &l : plan.lanes)
    lane_mux[hw::address(l.col, 0, hw::kShimMux)] = 3u << mux_shift(l.ch);
  auto final_value = [&](uint32_t a) -> std::optional<uint32_t> {
    const hw::Where w = hw::locate(a);
    auto t = plan.final_map.find(Tile{w.col, w.row});
    if (t == plan.final_map.end())
      return std::nullopt;
    auto r = t->second.find(w.offset);
    return r == t->second.end() ? std::nullopt : std::optional<uint32_t>(r->second);
  };
  // Every word either model wrote, in address order.
  std::set<uint32_t> pages;
  for (const auto &[pg, p] : cp.pages)
    pages.insert(pg);
  for (const auto &[pg, p] : tx.pages)
    pages.insert(pg);
  static const sw::Registers::Page none{};
  for (const uint32_t pg : pages) {
    auto ci = cp.pages.find(pg), xi = tx.pages.find(pg);
    const sw::Registers::Page &cpg = ci == cp.pages.end() ? none : ci->second;
    const sw::Registers::Page &xpg = xi == tx.pages.end() ? none : xi->second;
    const std::bitset<sw::kPageWords> on = cpg.on | xpg.on;
    for (uint32_t i = 0; i < sw::kPageWords; ++i) {
      if (!on.test(i))
        continue;
      const uint32_t a = 4 * (pg * sw::kPageWords + i), c = cpg.v[i], x = xpg.v[i];
      hw::Cls cls;
      try {
        // Equal words pass, but a switch register must also hold its final value.
        if (c == x) {
          const hw::Where w = hw::locate(a);
          if (!hw::is_switch(w.kind, w.offset))
            continue;
        }
        cls = hw::classify(a);
      } catch (const std::exception &) {
        if (c != x)
          fail(std::format("{:#010x}: {:#x}, the TXN kernel {:#x}", a, c, x));
        continue;
      }
      if (cls == hw::Cls::route) {
        const uint32_t want = final_value(a).value_or(0);
        if (c != want)
          fail(std::format("{} {:#x}: switch register {:#x}, not the final {:#x}", hw::locate(a).col,
                           hw::locate(a).offset, c, want));
        continue;
      }
      if (c == x)
        continue;
      if (st.chain_bds.count(a) && !xpg.on.test(i))
        continue;
      if (st.lane_ctrls.count(a) && (c & ~hw::kTokenControllerMask) == (x & ~hw::kTokenControllerMask))
        continue;
      if (auto m = lane_mux.find(a); m != lane_mux.end() && (c & ~m->second) == (x & ~m->second))
        continue;
      const hw::Where w = hw::locate(a);
      fail(std::format("({}, {}) {:#x} ({}): {:#x}, the TXN kernel {:#x}", w.col, w.row, w.offset,
                       hw::cls_name(cls), c, x));
    }
  }
  if (end)
    *end = std::move(st);
  return errors;
}

} // namespace launch::ctrlpkt
