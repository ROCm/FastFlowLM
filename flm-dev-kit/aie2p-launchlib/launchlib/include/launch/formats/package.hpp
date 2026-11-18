//===- package.hpp - read the package of one app ------------------*- C++ -*-===//
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
// The only file of the library that reads files. tools/make_package.py writes
// PKG/<app>/: init.txn, design.xclbin, manifest.txt, insts.bin.
//
// `load_package` parses init.txn (aiecc's configure sequence) and sorts its
// writes per tile and register class (hw::classify). The sort keeps the order
// of the writes within a class. What the loader does with each class:
//
//   program, data   kept as block writes (the program goes into PM on every
//                   load: there is no PM ping-pong on AIE2P)
//   lock            the last value written to each lock (aiecc writes 0 to
//                   every core lock, then the initial values)
//   bd              kept as written
//   channel         the Start_Queue and Ctrl value writes (the chain starts) are
//                   kept in order; the reset and unreset mask writes of aiecc
//                   are dropped (the reset part of a kernel writes its own)
//   route           the value of each switch register
//   mux             kept as written
//   core_control    dropped (the kernel stops and starts the cores itself);
//                   it marks the tile as a core with a program
//
// A write without a register class, a Sync or an address patch in init.txn,
// or a block write that crosses a class border stops the load with an error
// that names the file.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/hw/aie2p.hpp"
#include "launch/txn/txn.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace launch {

/// A tile of the array: column and row.
struct Tile {
  int col = 0, row = 0;
  auto operator<=>(const Tile &) const = default;
  hw::Kind kind() const { return hw::kind_of_row(row); }
  uint32_t addr(uint32_t offset) const { return hw::address(col, row, offset); }
  std::string describe() const { return std::format("{} ({}, {})", hw::kind_name(kind()), col, row); }
};

/// A shim DMA channel: what IRON calls a shim_dma_allocation.
struct ShimChannel {
  int col = 0;
  hw::Dir dir = hw::Dir::mm2s;
  int channel = 0;
  auto operator<=>(const ShimChannel &) const = default;
};

struct Buffer {
  Tile tile;
  uint32_t address, bytes;
};
struct Lock {
  Tile tile;
  uint32_t id;
  int init;
};

/// Switch register values of an array: tile -> tile offset -> value.
using RouteMap = std::map<Tile, std::map<uint32_t, uint32_t>>;

/// The init writes of one tile, by class.
struct TileInit {
  std::vector<txn::Op> program, data, bds, starts, mux;
  std::map<uint32_t, uint32_t> locks;  ///< offset -> value
  std::map<uint32_t, uint32_t> routes; ///< offset -> value
  bool core = false;                   ///< a core with a program
};

struct Package {
  std::string name, dir;
  std::string xclbin;  ///< path of design.xclbin
  txn::Header header;  ///< of init.txn: the device
  std::map<Tile, TileInit> tiles;
  std::map<Tile, uint32_t> controller_id; ///< packet id of the tile's controller
  std::map<std::string, Buffer> buffers;
  std::map<std::string, Lock> lock_names;
  std::map<std::string, ShimChannel> allocs;
  /// The lock census of the manifest (make_package.py): the array addresses of
  /// the locks the cores' code acquires, the only ones a stopped core can wait
  /// on. Without a census (an older package) every lock of a core tile counts.
  bool census = false;
  std::set<uint32_t> waits;
  std::set<uint32_t> declared_locks; ///< array addresses of every aie.lock (manifest "declare", named or not)

  /// The locks whose values the app needs: the ones it declares and the ones
  /// its cores acquire (aiecc also writes 0 to every other lock of a core
  /// tile; no one of this app uses those). Without a census or the declare
  /// lines, every lock aiecc writes. The loader writes a lock of a value other
  /// than 0 in any case.
  bool needs_lock(Tile t, uint32_t offset) const {
    if (!census || declared_locks.empty())
      return true;
    const uint32_t a = t.addr(offset);
    return declared_locks.count(a) || waits.count(a);
  }

  std::vector<Tile> cores() const {
    std::vector<Tile> out;
    for (const auto &[t, init] : tiles)
      if (init.core)
        out.push_back(t);
    return out;
  }
  const Buffer &buffer(const std::string &n) const { return find(buffers, n, "buffer"); }
  const Lock &lock(const std::string &n) const { return find(lock_names, n, "lock"); }
  const ShimChannel &alloc(const std::string &n) const { return find(allocs, n, "shim_dma_allocation"); }

private:
  template <class M> const typename M::mapped_type &find(const M &m, const std::string &n, const char *what) const {
    auto it = m.find(n);
    if (it == m.end())
      throw std::invalid_argument(std::format("app '{}': no {} '{}' in its manifest", name, what, n));
    return it->second;
  }
};

namespace detail {

inline std::vector<char> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

inline void read_manifest(Package &p, const std::string &path) {
  const std::vector<char> bytes = read_file(path);
  std::istringstream in(std::string(bytes.begin(), bytes.end()));
  std::string line;
  int n = 0;
  while (std::getline(in, line)) {
    ++n;
    std::istringstream l(line);
    std::string key;
    if (!(l >> key))
      continue;
    auto bad = [&] { throw std::runtime_error(std::format("{}:{}: cannot read '{}'", path, n, line)); };
    if (key == "design") {
      std::string d;
      l >> d;
    } else if (key == "tile") {
      Tile t;
      uint32_t id;
      if (!(l >> t.col >> t.row >> id))
        bad();
      p.controller_id[t] = id;
    } else if (key == "core") {
      Tile t;
      if (!(l >> t.col >> t.row))
        bad();
      p.tiles[t].core = true;
    } else if (key == "buffer") {
      std::string name;
      Buffer b;
      if (!(l >> name >> b.tile.col >> b.tile.row >> b.address >> b.bytes))
        bad();
      p.buffers[name] = b;
    } else if (key == "lock") {
      std::string name;
      Lock k;
      if (!(l >> name >> k.tile.col >> k.tile.row >> k.id >> k.init))
        bad();
      p.lock_names[name] = k;
    } else if (key == "declare") {
      Tile t;
      uint32_t id;
      if (!(l >> t.col >> t.row >> id) || t.col < 0 || t.col >= hw::kCols || t.row < 0 || t.row >= hw::kRows ||
          id >= hw::locks(t.kind()).count)
        bad();
      p.declared_locks.insert(t.addr(hw::locks(t.kind()).at(id)));
    } else if (key == "census") {
      p.census = true;
    } else if (key == "acquire") {
      Tile core, at;
      uint32_t id;
      if (!(l >> core.col >> core.row >> at.col >> at.row >> id) || at.kind() != hw::Kind::core ||
          id >= hw::locks(hw::Kind::core).count)
        bad();
      p.waits.insert(at.addr(hw::locks(hw::Kind::core).at(id)));
    } else if (key == "acquire_any") {
      // An acquire through a register: any lock of the core's tile and of the
      // core tiles next to it.
      Tile core;
      if (!(l >> core.col >> core.row))
        bad();
      for (auto [dc, dr] : {std::pair{0, 0}, {0, -1}, {-1, 0}, {0, 1}, {1, 0}}) {
        const Tile t{core.col + dc, core.row + dr};
        if (t.col < 0 || t.col >= hw::kCols || t.row >= hw::kRows || t.kind() != hw::Kind::core)
          continue;
        for (uint32_t i = 0; i < hw::locks(hw::Kind::core).count; ++i)
          p.waits.insert(t.addr(hw::locks(hw::Kind::core).at(i)));
      }
    } else if (key == "alloc") {
      std::string name, dir;
      ShimChannel c;
      if (!(l >> name >> dir >> c.channel >> c.col) || (dir != "mm2s" && dir != "s2mm"))
        bad();
      c.dir = dir == "mm2s" ? hw::Dir::mm2s : hw::Dir::s2mm;
      p.allocs[name] = c;
    } else {
      bad();
    }
  }
}

/// Sort the ops of init.txn per tile and class (see the top of this file).
inline void sort_init(Package &p, const txn::Stream &s, const std::string &path) {
  for (size_t i = 0; i < s.ops.size(); ++i) {
    const txn::Op &op = s.ops[i];
    auto fail = [&](const std::string &m) { throw std::runtime_error(std::format("{}: op {}: {}", path, i, m)); };
    if (!std::holds_alternative<txn::Write32>(op) && !std::holds_alternative<txn::MaskWrite32>(op) &&
        !std::holds_alternative<txn::BlockWrite>(op))
      fail("only writes may be in the init sequence");
    const uint32_t addr = txn::addr_of(op);
    hw::Cls cls;
    hw::Where w;
    try {
      cls = hw::classify(addr);
      w = hw::locate(addr);
      if (const auto *b = std::get_if<txn::BlockWrite>(&op); b && !b->words.empty())
        if (hw::classify(addr + 4 * (uint32_t)(b->words.size() - 1)) != cls)
          fail("a block write crosses a register class border");
    } catch (const std::logic_error &e) { // invalid_argument (no class), out_of_range (no tile)
      fail(e.what());
    }
    TileInit &t = p.tiles[Tile{w.col, w.row}];
    switch (cls) {
    case hw::Cls::program: t.program.push_back(op); break;
    case hw::Cls::data: t.data.push_back(op); break;
    case hw::Cls::bd: t.bds.push_back(op); break;
    case hw::Cls::mux: t.mux.push_back(op); break;
    case hw::Cls::core_control: t.core = true; break;
    case hw::Cls::channel:
      if (const auto *m = std::get_if<txn::MaskWrite32>(&op)) {
        if (m->mask != hw::kDmaReset)
          fail(std::format("a channel mask write with mask {:#x}: only the reset bit is known", m->mask));
      } else if (const auto *wr = std::get_if<txn::Write32>(&op)) {
        t.starts.push_back(*wr);
      } else {
        fail("a block write into DMA channel registers");
      }
      break;
    case hw::Cls::lock:
    case hw::Cls::route: {
      const auto *wr = std::get_if<txn::Write32>(&op);
      if (!wr)
        fail(std::format("a {} write that is not a write32", hw::cls_name(cls)));
      (cls == hw::Cls::lock ? t.locks : t.routes)[w.offset] = wr->value;
      break;
    }
    }
  }
}

} // namespace detail

/// Read PKG/<app>/. Throws std::runtime_error naming the file.
inline Package load_package(const std::string &package_dir, const std::string &app) {
  Package p;
  p.name = app;
  p.dir = (std::filesystem::path(package_dir) / app).string();
  p.xclbin = (std::filesystem::path(p.dir) / "design.xclbin").string();
  if (!std::filesystem::exists(p.xclbin))
    throw std::runtime_error(std::format("app '{}': no {} (tools/make_package.py)", app, p.xclbin));
  detail::read_manifest(p, p.dir + "/manifest.txt");
  const std::string init = p.dir + "/init.txn";
  txn::Stream s;
  try {
    s = txn::parse(txn::words_of(detail::read_file(init)));
  } catch (const std::exception &e) {
    throw std::runtime_error(init + ": " + e.what());
  }
  if (s.header.dev_gen != hw::kTxnDevGen || s.header.cols != hw::kCols || s.header.rows != hw::kRows)
    throw std::runtime_error(std::format("{}: device generation {} with {} x {} tiles; the library is for npu2 "
                                         "(generation {}, {} x {})",
                                         init, s.header.dev_gen, s.header.cols, s.header.rows, hw::kTxnDevGen,
                                         hw::kCols, hw::kRows));
  p.header = s.header;
  detail::sort_init(p, s, init);
  for (const Tile &t : p.cores())
    if (p.tiles.at(t).program.empty())
      throw std::runtime_error(std::format("{}: the {} has a core but no program", init, t.describe()));
  return p;
}

/// The facts of two apps that the "program only" switch needs to be equal:
/// everything but the program. Empty if equal, else the first difference.
inline std::string floorplan_difference(const Package &a, const Package &b) {
  if (a.cores() != b.cores())
    return "the cores differ";
  for (const auto &[t, ia] : a.tiles) {
    auto it = b.tiles.find(t);
    if (it == b.tiles.end())
      return std::format("{} is only in '{}'", t.describe(), a.name);
    const TileInit &ib = it->second;
    if (ia.data != ib.data) return std::format("{}: the data memory differs", t.describe());
    if (ia.bds != ib.bds) return std::format("{}: the BDs differ", t.describe());
    if (ia.starts != ib.starts) return std::format("{}: the chain starts differ", t.describe());
    if (ia.locks != ib.locks) return std::format("{}: the locks differ", t.describe());
    if (ia.routes != ib.routes) return std::format("{}: the routes differ", t.describe());
    if (ia.mux != ib.mux) return std::format("{}: the mux differs", t.describe());
  }
  for (const auto &[t, ib] : b.tiles)
    if (!a.tiles.count(t))
      return std::format("{} is only in '{}'", t.describe(), b.name);
  return "";
}

} // namespace launch
