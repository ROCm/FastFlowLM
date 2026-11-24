//===- ctrlpkt_lane.hpp - the packet routes control packets take ---*- C++ -*-===//
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
// A lane is one column's chain of packet routes from a shim MM2S channel to the
// TILE_CTRL of its mem tile and of its core tiles up to the app's top core:
//
//   shim       slave SOUTH3 (MM2S 0) or SOUTH7 (MM2S 1), the mux on DMA
//              slot (8, mask 0x18) -> master NORTH k
//   mem tile   slave SOUTH k: id 12 -> TILE_CTRL; ids 8..11 and 13 -> NORTH k
//              (a mem tile joins SOUTH k only to NORTH k)
//   core row r slave SOUTH j(r-1): its id (6 + r) -> TILE_CTRL (msel 0); id 13,
//              the broadcast, -> TILE_CTRL and NORTH j(r) (msel 2); the ids of the
//              rows above -> NORTH j(r) (msel 1)
//
// ryzenai-corelib puts the same tree in the PDI of its own image (one lane on
// NORTH4/SOUTH4, ids 1, 3 and 6 + row). Here the array holds the app's routes
// (its xclbin's): the lane is planned around them, per package.
//
// The slots of a slave never overlap (no two match one id), and each tile's
// lane takes one arbiter: the lowest one the app does not use in that tile, so
// an app master never takes a lane packet. All 4 slots of a lane slave are
// written (unused ones 0).
//
// Each lane register falls in one of three classes, by the app's value:
//
//   free       the app has the lane's value                      cost 0
//   resident   the app has 0: the lane value stays after the     cost 1
//              load (no app stream reaches a lane slave the app
//              does not use, no app packet takes the lane's arbiter)
//   toggle     the app has another value: lane value at load     shim 100,
//              time, the app's after it (TXN writes)             others 1000
//
// A circuit master the app joins to a lane slave would take the packets too: it
// is a toggle to 0. The planner tries both channels, all six k and all six
// j(r) of each row, takes the cheapest lane whose routes `sw::route` checks
// (each id reaches exactly its TILE_CTRLs, nothing else), and else the next
// one. A column with no lane (both channels' muxes taken by something other
// than the DMA, or the app's shim has no controller id for the Sync) loads with
// TXN writes.
//
// Two route maps per app follow (both tile -> offset -> value):
//
//   final   what the array holds after the app's load: its routes, the residents
//   load    what the lane needs while it carries packets: final with the
//           toggles at their lane values (the reset part writes these)
//
// The plan depends only on the package's routes, mux, cores and controller ids:
// two apps of one floorplan get the same lanes (a program-only switch needs it).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/formats/package.hpp"
#include "launch/hw/switch.hpp"

#include <algorithm>
#include <format>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace launch::ctrlpkt {

inline constexpr uint32_t kMemId = 12, kBroadcastId = 13;
inline constexpr uint32_t row_id(int row) { return 8 + (uint32_t)(row - hw::kFirstCoreRow); }
inline constexpr int kArbiters = 6;
inline constexpr int kCostResident = 1, kCostToggleShim = 100, kCostToggle = 1000;

/// The shim slave of MM2S channel `ch`, and its 2-bit field in the shim mux.
inline sw::PortId lane_slave(int ch) { return {sw::Port::south, ch == 0 ? 3 : 7}; }
inline unsigned mux_shift(int ch) { return ch == 0 ? 10 : 14; }
inline constexpr uint32_t kMuxDma = 1;

struct Lane {
  int col = 0;
  int ch = 0;              ///< shim MM2S channel
  int k = 0;               ///< shim NORTH k = mem tile SOUTH k, NORTH k = row 2 SOUTH k
  std::vector<int> north;  ///< NORTH j of core rows 2 .. top - 1
  int top = 1;             ///< top row (1: the mem tile only)
  std::map<int, uint32_t> arb; ///< row -> arbiter
  RouteMap regs;           ///< every register the lane writes, at its lane value
  int cost = 0;
  bool mux_resident = false; ///< the app's mux field is 0: the load sets it to DMA
};

struct Plan {
  std::vector<Lane> lanes;             ///< columns that load by packet
  std::map<int, std::string> txn_cols; ///< columns with app state and no lane: why
  RouteMap final_map, load_map;
  RouteMap toggles;                    ///< register -> the app's value (restored after the load)
  bool shim_only_toggles = true;       ///< no toggle above row 0: DMA and core starts may go by packet

  const Lane *lane(int col) const {
    for (const Lane &l : lanes)
      if (l.col == col)
        return &l;
    return nullptr;
  }
};

namespace detail {

inline uint32_t app_value(const Package &p, Tile t, uint32_t off) {
  auto it = p.tiles.find(t);
  if (it == p.tiles.end())
    return 0;
  auto r = it->second.routes.find(off);
  return r == it->second.routes.end() ? 0 : r->second;
}

/// The app's shim mux value: its mux ops from 0 (the reset part zeroes it).
inline uint32_t app_mux(const Package &p, int col) {
  auto it = p.tiles.find(Tile{col, 0});
  sw::Registers r;
  const uint32_t a = hw::address(col, 0, hw::kShimMux);
  if (it != p.tiles.end())
    for (const txn::Op &op : it->second.mux) {
      if (const auto *w = std::get_if<txn::Write32>(&op); w && w->addr == a)
        r.write(a, w->value);
      else if (const auto *m = std::get_if<txn::MaskWrite32>(&op); m && m->addr == a)
        r.write(a, m->value, m->mask);
    }
  return r.get(a);
}

/// The arbiters the app's switch in `t` uses (its packet masters and slots).
inline std::set<uint32_t> app_arbiters(const Package &p, Tile t) {
  std::set<uint32_t> out;
  auto it = p.tiles.find(t);
  if (it == p.tiles.end())
    return out;
  const hw::Kind kind = t.kind();
  for (const auto &[off, v] : it->second.routes) {
    const uint32_t b = sw::base(kind);
    if (off < b + 0x100) { // master
      if ((v & sw::kOn) && (v & sw::kPacket))
        out.insert(v & 7);
    } else if (off >= b + 0x200 && (v & sw::kSlotOn)) {
      out.insert(v & 7);
    }
  }
  return out;
}

/// Whether the column has state a packet would write: a core, or core or mem
/// tile BDs, locks, data or chain starts.
inline bool has_state(const Package &p, Tile t) {
  auto it = p.tiles.find(t);
  if (it == p.tiles.end() || t.kind() == hw::Kind::shim)
    return false;
  const TileInit &i = it->second;
  return i.core || !i.program.empty() || !i.data.empty() || !i.bds.empty() || !i.locks.empty() ||
         !i.starts.empty();
}

/// The lane registers of tile (col, row): `in` the slave it enters by, `out`
/// the NORTH master it leaves by (-1: the top), circuit masters of `in` to 0.
inline std::map<uint32_t, uint32_t> tile_regs(const Package &p, int col, int row, sw::PortId in, int out, int top,
                                              uint32_t arb) {
  using sw::Port;
  const Tile t{col, row};
  const hw::Kind kind = t.kind();
  std::map<uint32_t, uint32_t> r;
  const int in_index = sw::index_of(sw::slaves(kind), in);
  for (const sw::PortId &m : sw::masters(kind)) {
    const uint32_t off = sw::master_reg(kind, m), v = app_value(p, t, off);
    if ((v & sw::kOn) && !(v & sw::kPacket) && (int)(v & 0x7F) == in_index)
      r[off] = 0;
  }
  std::vector<uint32_t> slots;
  r[sw::slave_reg(kind, in)] = sw::kPacketSlave;
  if (kind == hw::Kind::shim) {
    slots.push_back(sw::slot_value(row_id(hw::kFirstCoreRow), 0x18, arb, 0));
    r[sw::master_reg(kind, {Port::north, out})] = sw::packet_master(arb, 0b1);
  } else if (kind == hw::Kind::mem) {
    slots.push_back(sw::slot_value(kMemId, 0x1F, arb, 0));
    r[sw::master_reg(kind, {Port::tile_ctrl, 0})] = sw::packet_master(arb, 0b01);
    if (out >= 0) {
      slots.push_back(sw::slot_value(row_id(hw::kFirstCoreRow), 0x1C, arb, 1));
      slots.push_back(sw::slot_value(kBroadcastId, 0x1F, arb, 1));
      r[sw::master_reg(kind, {Port::north, out})] = sw::packet_master(arb, 0b10);
    }
  } else {
    slots.push_back(sw::slot_value(row_id(row), 0x1F, arb, 0));
    slots.push_back(sw::slot_value(kBroadcastId, 0x1F, arb, 2));
    const int above = top - row;
    if (above == 3) { // row 2 of a 4-row lane: rows 3, then 4 and 5 (ids 10, 11) in one slot
      slots.push_back(sw::slot_value(row_id(row + 1), 0x1F, arb, 1));
      slots.push_back(sw::slot_value(row_id(row + 2), 0x1E, arb, 1));
    } else {
      for (int a = 1; a <= above; ++a)
        slots.push_back(sw::slot_value(row_id(row + a), 0x1F, arb, 1));
    }
    r[sw::master_reg(kind, {Port::tile_ctrl, 0})] = sw::packet_master(arb, 0b0101);
    if (out >= 0)
      r[sw::master_reg(kind, {Port::north, out})] = sw::packet_master(arb, 0b0110);
  }
  slots.resize(sw::kSlots, 0);
  for (int s = 0; s < sw::kSlots; ++s)
    r[sw::slot_reg(kind, in, s)] = slots[(size_t)s];
  return r;
}

inline int tile_cost(const Package &p, Tile t, const std::map<uint32_t, uint32_t> &regs) {
  int cost = 0;
  for (const auto &[off, v] : regs) {
    const uint32_t a = app_value(p, t, off);
    if (a == v)
      continue;
    cost += a == 0 ? kCostResident : t.row == 0 ? kCostToggleShim : kCostToggle;
  }
  return cost;
}

/// The ids a lane carries, and the TILE_CTRLs each must reach.
inline std::map<uint32_t, std::set<Tile>> expected_sinks(int col, int top) {
  std::map<uint32_t, std::set<Tile>> out;
  out[kMemId] = {Tile{col, 1}};
  for (int r = hw::kFirstCoreRow; r <= top; ++r) {
    out[row_id(r)] = {Tile{col, r}};
    out[kBroadcastId].insert(Tile{col, r});
  }
  return out;
}

/// Empty if every id of the lane reaches exactly its TILE_CTRLs in `regs`, else why not.
inline std::string check_lane(const sw::Registers &regs, int col, int ch, int top) {
  for (const auto &[id, want] : expected_sinks(col, top)) {
    const sw::Reach r = sw::route(regs, col, 0, lane_slave(ch), (int)id);
    if (!r.errors.empty())
      return std::format("id {}: {}", id, r.errors.front());
    std::set<Tile> got;
    for (const sw::Hop &h : r.sinks) {
      if (h.port.port != sw::Port::tile_ctrl)
        return std::format("id {} reaches ({}, {}) {}", id, h.col, h.row, sw::port_name(h.port));
      got.insert(Tile{h.col, h.row});
    }
    if (got != want || r.sinks.size() != want.size())
      return std::format("id {} reaches {} TILE_CTRLs, not the {} it is for", id, r.sinks.size(), want.size());
  }
  return "";
}

inline RouteMap routes_of(const Package &p) {
  RouteMap m;
  for (const auto &[t, init] : p.tiles)
    if (!init.routes.empty())
      m[t] = init.routes;
  return m;
}

} // namespace detail

/// The lanes of an app. `no_lane_cols`: columns to load by TXN writes anyway (tests).
inline Plan plan(const Package &p, const std::set<int> &no_lane_cols = {}) {
  using namespace detail;
  Plan out;
  out.final_map = routes_of(p);
  for (int col = 0; col < hw::kCols; ++col) {
    int top = 0;
    for (int row = hw::kMemRows; row < hw::kRows; ++row)
      if (has_state(p, Tile{col, row}))
        top = row;
    if (top == 0)
      continue;
    if (no_lane_cols.count(col)) {
      out.txn_cols[col] = "asked for";
      continue;
    }
    if (!p.controller_id.count(Tile{col, 0})) {
      out.txn_cols[col] = "no controller id for the shim (the Sync needs one)";
      continue;
    }
    const uint32_t mux = app_mux(p, col);
    std::map<int, uint32_t> arb;
    bool arb_ok = true;
    for (int row = 0; row <= top; ++row) {
      const std::set<uint32_t> used = app_arbiters(p, Tile{col, row});
      int a = 0;
      while (a < kArbiters && used.count((uint32_t)a))
        ++a;
      if (a == kArbiters)
        arb_ok = false;
      arb[row] = (uint32_t)a;
    }
    if (!arb_ok) {
      out.txn_cols[col] = "a tile of the column has no free arbiter";
      continue;
    }
    // The cost of each tile for each pair of ports, then every lane in cost order.
    const int ncore = std::max(0, top - hw::kFirstCoreRow + 1);
    struct Candidate {
      int cost, ch, k;
      std::vector<int> north;
      auto operator<=>(const Candidate &) const = default;
    };
    std::vector<Candidate> cands;
    std::map<std::tuple<int, int, int>, int> core_cost; // (row, in, out) -> cost
    auto core_regs = [&](int row, int in, int outp) {
      return tile_regs(p, col, row, {sw::Port::south, in}, outp, top, arb[row]);
    };
    for (int row = hw::kFirstCoreRow; row <= top; ++row)
      for (int in = 0; in < 6; ++in)
        for (int o = row < top ? 0 : -1; o < (row < top ? 6 : 0); ++o)
          core_cost[{row, in, o}] = tile_cost(p, Tile{col, row}, core_regs(row, in, o));
    for (int ch = 0; ch < 2; ++ch) {
      const uint32_t field = mux >> mux_shift(ch) & 3;
      if (field != 0 && field != kMuxDma)
        continue;
      for (int k = 0; k < 6; ++k) {
        const int base_cost = (field == 0 ? kCostResident : 0) +
                              tile_cost(p, Tile{col, 0}, tile_regs(p, col, 0, lane_slave(ch), k, top, arb[0])) +
                              tile_cost(p, Tile{col, 1},
                                        tile_regs(p, col, 1, {sw::Port::south, k}, top >= 2 ? k : -1, top, arb[1]));
        // Every choice of NORTH j per core row below the top.
        const int links = std::max(0, ncore - 1);
        int combos = 1;
        for (int i = 0; i < links; ++i)
          combos *= 6;
        for (int n = 0; n < combos; ++n) {
          std::vector<int> north;
          for (int i = 0, x = n; i < links; ++i, x /= 6)
            north.push_back(x % 6);
          int cost = base_cost;
          for (int row = hw::kFirstCoreRow; row <= top; ++row) {
            const int in = row == hw::kFirstCoreRow ? k : north[(size_t)(row - hw::kFirstCoreRow - 1)];
            const int o = row < top ? north[(size_t)(row - hw::kFirstCoreRow)] : -1;
            cost += core_cost[{row, in, o}];
          }
          cands.push_back({cost, ch, k, north});
        }
      }
    }
    if (cands.empty()) {
      out.txn_cols[col] = "the shim mux takes both MM2S channels' ports for something other than the DMA";
      continue;
    }
    std::sort(cands.begin(), cands.end());
    std::string why;
    for (const Candidate &c : cands) {
      Lane l;
      l.col = col;
      l.ch = c.ch;
      l.k = c.k;
      l.north = c.north;
      l.top = top;
      l.arb = arb;
      l.cost = c.cost;
      l.mux_resident = (mux >> mux_shift(c.ch) & 3) == 0;
      l.regs[Tile{col, 0}] = tile_regs(p, col, 0, lane_slave(c.ch), c.k, top, arb[0]);
      l.regs[Tile{col, 1}] = tile_regs(p, col, 1, {sw::Port::south, c.k}, top >= 2 ? c.k : -1, top, arb[1]);
      for (int row = hw::kFirstCoreRow; row <= top; ++row) {
        const int in = row == hw::kFirstCoreRow ? c.k : c.north[(size_t)(row - hw::kFirstCoreRow - 1)];
        const int o = row < top ? c.north[(size_t)(row - hw::kFirstCoreRow)] : -1;
        l.regs[Tile{col, row}] = core_regs(row, in, o);
      }
      // The load-time map of the whole array with this lane: the app's routes, the lane's values.
      sw::Registers regs;
      for (const auto &[t, m] : out.final_map)
        for (const auto &[off, v] : m)
          regs.write(t.addr(off), v);
      for (const auto &[t, m] : l.regs)
        for (const auto &[off, v] : m)
          regs.write(t.addr(off), v);
      why = check_lane(regs, col, c.ch, top);
      if (why.empty()) {
        out.lanes.push_back(std::move(l));
        break;
      }
    }
    if (!out.lane(col))
      out.txn_cols[col] = "no lane passes the route check (the cheapest: " + why + ")";
  }
  // The maps: residents stay, toggles go back to the app's values.
  out.load_map = out.final_map;
  for (const Lane &l : out.lanes)
    for (const auto &[t, m] : l.regs)
      for (const auto &[off, v] : m) {
        const uint32_t a = detail::app_value(p, t, off);
        if (a == v)
          continue;
        out.load_map[t][off] = v;
        if (a == 0) {
          out.final_map[t][off] = v;
        } else {
          out.toggles[t][off] = a;
          if (t.row != 0)
            out.shim_only_toggles = false;
        }
      }
  return out;
}

} // namespace launch::ctrlpkt
