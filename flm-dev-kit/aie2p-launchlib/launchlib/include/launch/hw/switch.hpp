//===- switch.hpp - the AIE2P stream switches: ports, registers, routes -*- C++ -*-===//
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
// What the control-packet transport needs to know about the stream switches
// (transport/ctrlpkt_*.hpp): the ports of each tile kind, their registers, the
// packet routing rules, and a model that follows a packet through them.
// Sources: aie-rt xaie2pgbl_params.h (port order), mlir-aie
// AIE2TargetModel::isLegalTileConnection (legal joins); the same tables as
// ryzenai-corelib tools/npu_direct/aie2p_routes.py.
//
//   registers   (base 0x3F000; mem tile 0xB0000)
//     master i      base + 4i         bit 31 on, bit 30 packet; circuit: bits 6:0
//                                     the slave; packet: bit 7 drop the header,
//                                     bits 6:3 the msel values it takes, bits 2:0
//                                     its arbiter
//     slave i       base + 0x100 + 4i bit 31 on, bit 30 packet
//     slot s of i   base + 0x200 + 0x10i + 4s   (4 per slave) bits 28:24 id,
//                                     20:16 mask, 8 on, 5:4 msel, 2:0 arbiter
//
//   A packet entering a packet slave takes the first slot that is on and whose
//   (id & mask) equals the packet's (id & mask); it goes to every packet master
//   of the slot's arbiter that takes its msel. A circuit master takes all of
//   the stream of its slave.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/hw/aie2p.hpp"

#include <array>
#include <bitset>
#include <format>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace launch::sw {

enum class Port : uint8_t { core, dma, tile_ctrl, fifo, south, west, north, east, trace };

struct PortId {
  Port port;
  int index = 0;
  auto operator<=>(const PortId &) const = default;
};

inline std::string port_name(PortId p) {
  static const char *names[] = {"CORE", "DMA", "TILE_CTRL", "FIFO", "SOUTH", "WEST", "NORTH", "EAST", "TRACE"};
  return p.port == Port::tile_ctrl ? "TILE_CTRL" : std::format("{}{}", names[(int)p.port], p.index);
}

namespace detail {
struct Group {
  Port port;
  int count;
};
inline std::vector<PortId> expand(std::initializer_list<Group> groups) {
  std::vector<PortId> out;
  for (const Group &g : groups)
    for (int i = 0; i < g.count; ++i)
      out.push_back({g.port, i});
  return out;
}
} // namespace detail

/// The master ports of a tile kind, in register order.
inline const std::vector<PortId> &masters(hw::Kind k) {
  using P = Port;
  static const std::vector<PortId> core = detail::expand(
      {{P::core, 1}, {P::dma, 2}, {P::tile_ctrl, 1}, {P::fifo, 1}, {P::south, 4}, {P::west, 4}, {P::north, 6}, {P::east, 4}});
  static const std::vector<PortId> mem = detail::expand({{P::dma, 6}, {P::tile_ctrl, 1}, {P::south, 4}, {P::north, 6}});
  static const std::vector<PortId> shim = detail::expand(
      {{P::tile_ctrl, 1}, {P::fifo, 1}, {P::south, 6}, {P::west, 4}, {P::north, 6}, {P::east, 4}});
  return k == hw::Kind::core ? core : k == hw::Kind::mem ? mem : shim;
}

/// The slave ports of a tile kind, in register order.
inline const std::vector<PortId> &slaves(hw::Kind k) {
  using P = Port;
  static const std::vector<PortId> core = detail::expand({{P::core, 1}, {P::dma, 2}, {P::tile_ctrl, 1}, {P::fifo, 1},
                                                          {P::south, 6}, {P::west, 4}, {P::north, 4}, {P::east, 4},
                                                          {P::trace, 2}});
  static const std::vector<PortId> mem =
      detail::expand({{P::dma, 6}, {P::tile_ctrl, 1}, {P::south, 6}, {P::north, 4}, {P::trace, 1}});
  static const std::vector<PortId> shim = detail::expand(
      {{P::tile_ctrl, 1}, {P::fifo, 1}, {P::south, 8}, {P::west, 4}, {P::north, 4}, {P::east, 4}, {P::trace, 1}});
  return k == hw::Kind::core ? core : k == hw::Kind::mem ? mem : shim;
}

/// Index of a port in its list, or -1.
inline int index_of(const std::vector<PortId> &ports, PortId p) {
  for (size_t i = 0; i < ports.size(); ++i)
    if (ports[i] == p)
      return (int)i;
  return -1;
}

inline constexpr int kSlots = 4;
inline constexpr uint32_t kOn = 0x80000000, kPacket = 0x40000000, kPacketSlave = kOn | kPacket;
inline constexpr uint32_t kSlotOn = 1u << 8, kDropHeader = 1u << 7;

inline uint32_t base(hw::Kind k) { return k == hw::Kind::mem ? 0xB0000 : 0x3F000; }

/// Tile offsets of the registers of a port (throws for a port the kind has not).
inline uint32_t master_reg(hw::Kind k, PortId m) {
  const int i = index_of(masters(k), m);
  if (i < 0)
    throw std::invalid_argument(std::format("a {} has no master {}", hw::kind_name(k), port_name(m)));
  return base(k) + 4 * (uint32_t)i;
}
inline uint32_t slave_reg(hw::Kind k, PortId s) {
  const int i = index_of(slaves(k), s);
  if (i < 0)
    throw std::invalid_argument(std::format("a {} has no slave {}", hw::kind_name(k), port_name(s)));
  return base(k) + 0x100 + 4 * (uint32_t)i;
}
inline uint32_t slot_reg(hw::Kind k, PortId s, int slot) {
  return base(k) + 0x200 + 0x10 * (uint32_t)index_of(slaves(k), s) + 4 * (uint32_t)slot;
}

inline uint32_t slot_value(uint32_t id, uint32_t mask, uint32_t arb, uint32_t msel) {
  return (id & 0x1F) << 24 | (mask & 0x1F) << 16 | kSlotOn | (msel & 3) << 4 | (arb & 7);
}
/// A packet master of arbiter `arb` that takes the msel values in `msels` (bit i: msel i).
inline uint32_t packet_master(uint32_t arb, uint32_t msels, bool drop = false) {
  return kPacketSlave | (drop ? kDropHeader : 0) | (msels & 0xF) << 3 | (arb & 7);
}
inline uint32_t circuit_master(int slave_index) { return kOn | (uint32_t)slave_index; }

/// Whether a switch may join `slave` to `master` (mlir-aie isLegalTileConnection, AIE2).
inline bool legal(hw::Kind k, PortId slave, PortId master) {
  using P = Port;
  const Port src = slave.port, dst = master.port;
  const int s = slave.index, m = master.index;
  auto side = [](Port p) { return p == P::south || p == P::north || p == P::west || p == P::east; };
  if (k == hw::Kind::mem) {
    if (src == P::dma)
      return dst == P::dma ? m == s : dst == P::tile_ctrl || dst == P::south || dst == P::north;
    if (src == P::tile_ctrl)
      return dst == P::dma ? m == 5 : dst == P::south || dst == P::north;
    if (src == P::south || src == P::north)
      return dst == P::dma || dst == P::tile_ctrl || ((dst == P::south || dst == P::north) && m == s);
    return src == P::trace && (dst == P::south || (dst == P::dma && m == 5));
  }
  if (k == hw::Kind::shim) {
    if (src == P::tile_ctrl)
      return dst != P::tile_ctrl;
    if (src == P::fifo || src == P::south)
      return true;
    if (src == P::west || src == P::north || src == P::east)
      return src == dst ? m == s : true;
    return src == P::trace && (dst == P::fifo || dst == P::south || ((dst == P::west || dst == P::east) && m == 0));
  }
  if (src == P::dma || src == P::fifo || side(src))
    return src == dst ? m == s : dst != P::trace;
  if (src == P::core)
    return dst != P::core;
  if (src == P::tile_ctrl)
    return dst != P::tile_ctrl && dst != P::dma;
  return src == P::trace && (dst == P::fifo || dst == P::south || (dst == P::dma && m == 0));
}

struct Hop {
  int col, row;
  PortId port;
  auto operator<=>(const Hop &) const = default;
};

/// The slave a master's stream enters in the next tile, or nullopt where it
/// leaves the array (a shim's SOUTH) or stays in its tile (DMA, TILE_CTRL, ...).
inline std::optional<Hop> neighbour(int col, int row, PortId master) {
  switch (master.port) {
  case Port::north:
    if (row + 1 < hw::kRows)
      return Hop{col, row + 1, {Port::south, master.index}};
    return std::nullopt;
  case Port::south:
    if (row > 0)
      return Hop{col, row - 1, {Port::north, master.index}};
    return std::nullopt;
  case Port::east:
    if (col + 1 < hw::kCols)
      return Hop{col + 1, row, {Port::west, master.index}};
    return std::nullopt;
  case Port::west:
    if (col > 0)
      return Hop{col - 1, row, {Port::east, master.index}};
    return std::nullopt;
  default: return std::nullopt;
  }
}

/// Word addresses of the array, in pages of 256 words: a model of a kernel holds
/// every program and data word it writes, a check copies one per kernel and
/// routes one per packet; a node per word made that the build's main cost.
inline constexpr uint32_t kPageWords = 256;
inline uint32_t page_of(uint32_t addr) { return addr / (4 * kPageWords); }
inline uint32_t word_of(uint32_t addr) { return addr / 4 % kPageWords; }
/// `addr`, which a write or an insert must give word-aligned.
inline uint32_t word_addr(uint32_t addr) {
  if (addr & 3)
    throw std::invalid_argument(std::format("{:#010x}: not a word address", addr));
  return addr;
}

/// A set of word addresses.
struct AddrSet {
  std::map<uint32_t, std::bitset<kPageWords>> pages;
  void insert(uint32_t addr) { pages[page_of(word_addr(addr))].set(word_of(addr)); }
  void insert(const AddrSet &o) {
    for (const auto &[p, b] : o.pages)
      pages[p] |= b;
  }
  bool count(uint32_t addr) const {
    auto it = pages.find(page_of(addr));
    return !(addr & 3) && it != pages.end() && it->second.test(word_of(addr));
  }
};

/// Switch (and other) register values by array address; a register never
/// written is 0, as after the reset part.
struct Registers {
  struct Page {
    std::array<uint32_t, kPageWords> v{};
    std::bitset<kPageWords> on; ///< written
  };
  std::map<uint32_t, Page> pages;
  uint32_t get(uint32_t addr) const {
    auto it = pages.find(page_of(addr));
    return (addr & 3) || it == pages.end() ? 0 : it->second.v[word_of(addr)];
  }
  /// Whether `addr` was ever written.
  bool has(uint32_t addr) const {
    auto it = pages.find(page_of(addr));
    return !(addr & 3) && it != pages.end() && it->second.on.test(word_of(addr));
  }
  void write(uint32_t addr, uint32_t value, uint32_t mask = 0xFFFFFFFF) {
    Page &p = pages[page_of(word_addr(addr))];
    const uint32_t w = word_of(addr);
    p.v[w] = (p.v[w] & ~mask) | (value & mask);
    p.on.set(w);
  }
  uint32_t master(int col, int row, PortId m) const {
    return get(hw::address(col, row, master_reg(hw::kind_of_row(row), m)));
  }
  uint32_t slave(int col, int row, PortId s) const {
    return get(hw::address(col, row, slave_reg(hw::kind_of_row(row), s)));
  }
  uint32_t slot(int col, int row, PortId s, int k) const {
    return get(hw::address(col, row, slot_reg(hw::kind_of_row(row), s, k)));
  }
};

/// Where a packet (or a stream without a header: id < 0) entering a slave goes.
struct Reach {
  std::vector<Hop> sinks;           ///< masters it ends in: TILE_CTRL, DMA, CORE, FIFO, or a port out of the array
  std::vector<std::string> errors;  ///< a stall, an illegal join, a dropped header
};

/// Follow a packet with stream id `id` from slave `s` of tile (col, row).
inline Reach route(const Registers &r, int col, int row, PortId s, int id) {
  Reach out;
  std::set<Hop> seen;
  std::vector<Hop> queue{{col, row, s}};
  while (!queue.empty()) {
    const Hop at = queue.back();
    queue.pop_back();
    if (!seen.insert(at).second)
      continue;
    const hw::Kind kind = hw::kind_of_row(at.row);
    // Built only for an error: a check routes every packet of a kernel.
    auto where = [&] { return std::format("({}, {}) slave {}", at.col, at.row, port_name(at.port)); };
    const int index = index_of(slaves(kind), at.port);
    if (index < 0) {
      out.errors.push_back(where() + ": no such port");
      continue;
    }
    // The registers by port index (master_reg and slot_reg look the port up).
    const std::vector<PortId> &ms = masters(kind);
    const uint32_t sw_base = base(kind);
    auto master_at = [&](size_t i) { return r.get(hw::address(at.col, at.row, sw_base + 4 * (uint32_t)i)); };
    const uint32_t config = r.get(hw::address(at.col, at.row, sw_base + 0x100 + 4 * (uint32_t)index));
    if (!(config & kOn)) {
      out.errors.push_back(where() + ": off (the stream stalls)");
      continue;
    }
    std::vector<size_t> to; // indexes in ms
    for (size_t i = 0; i < ms.size(); ++i) {
      const uint32_t v = master_at(i);
      if ((v & kOn) && !(v & kPacket) && (int)(v & 0x7F) == index)
        to.push_back(i); // a circuit master of this slave takes everything
    }
    if (config & kPacket) {
      if (id < 0) {
        out.errors.push_back(where() + ": a packet slave, and the stream has no header");
        continue;
      }
      auto slot_at = [&](int k) {
        return r.get(hw::address(at.col, at.row, sw_base + 0x200 + 0x10 * (uint32_t)index + 4 * (uint32_t)k));
      };
      int hit = -1;
      for (int k = 0; k < kSlots && hit < 0; ++k) {
        const uint32_t v = slot_at(k);
        if ((v & kSlotOn) && (((uint32_t)id & (v >> 16 & 0x1F)) == ((v >> 24) & (v >> 16) & 0x1F)))
          hit = k;
      }
      if (hit < 0) {
        out.errors.push_back(std::format("{}: no slot takes id {}", where(), id));
        continue;
      }
      const uint32_t slot = slot_at(hit), arb = slot & 7, msel = slot >> 4 & 3;
      for (size_t i = 0; i < ms.size(); ++i) {
        const uint32_t v = master_at(i);
        if ((v & kOn) && (v & kPacket) && (v & 7) == arb && (v >> (3 + msel) & 1))
          to.push_back(i);
      }
    }
    if (to.empty())
      out.errors.push_back(where() + ": no master takes it (the stream stalls)");
    for (const size_t i : to) {
      const PortId &m = ms[i];
      if (!legal(kind, at.port, m))
        out.errors.push_back(std::format("{} -> master {}: not a legal join", where(), port_name(m)));
      const uint32_t v = master_at(i);
      if ((v & kPacket) && (v & kDropHeader) && id >= 0)
        out.errors.push_back(std::format("{} -> master {}: drops the header", where(), port_name(m)));
      const std::optional<Hop> next = neighbour(at.col, at.row, m);
      if (next)
        queue.push_back(*next);
      else
        out.sinks.push_back({at.col, at.row, m});
    }
  }
  return out;
}

} // namespace launch::sw
