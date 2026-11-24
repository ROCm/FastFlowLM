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
// States. An app whose cores run several ops (an overlay: one program, the op
// picked by registers the host writes) can be loaded set up for one of them.
// "APP@STATE" names that: the package of APP plus the writes of
// PKG/APP/state_STATE.txn (the op's configuration, as the host writes it
// before the op's first run; make_package.py --state), sorted as init.txn's and
// after them. A kernel that loads APP@STATE leaves the op ready, so the host's
// next run of that op needs no configuring of its own: warm(other, "APP@STATE")
// is the switch back to APP that also configures the op. A state may only add
// to the app: no program, no core APP does not have, and no register or memory
// word that init.txn sets to a value other than 0 (lock values excepted: a
// state's lock values are the ones its op starts from). Channel reset writes
// are dropped as init.txn's. The host must know that the state fits what it
// runs next; the library cannot see that (an op that keeps state of its own
// between runs, e.g. a ping/pong selector, must start from the state's values).
//
// Columns. "APP[A-B]" is the part of APP in columns A..B: its tiles there,
// nothing of the other columns. A warm kernel between it and an app inside
// those columns (a guest) resets and loads those columns only, so APP's other
// columns keep their state, cores waiting where APP's last run left them:
// warm("APP[A-B]", guest) in, warm(guest, "APP[A-B]") back. The array must hold
// APP before (as for any warm kernel); APP's columns must not depend on each
// other's progress (a DMA chain or lock across the border, data still in
// flight), since A..B start again from the init values and the others do not.
// The library checks the census (no core waits on a lock across the border);
// the rest is the host's knowledge of APP, as with states. Its next run of APP
// is a run of the whole app.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/hw/aie2p.hpp"
#include "launch/txn/txn.hpp"

#include <cstdio>
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
  std::string name, dir; ///< name: APP or APP@STATE; dir: PKG/APP
  std::string state;     ///< STATE of APP@STATE, else empty
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
  std::vector<std::pair<Tile, uint32_t>> acquires; ///< the census per core: (core, lock address) of every wait
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
      p.acquires.push_back({core, at.addr(hw::locks(hw::Kind::core).at(id))});
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
        for (uint32_t i = 0; i < hw::locks(hw::Kind::core).count; ++i) {
          p.waits.insert(t.addr(hw::locks(hw::Kind::core).at(i)));
          p.acquires.push_back({core, t.addr(hw::locks(hw::Kind::core).at(i))});
        }
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

namespace detail {

inline txn::Stream read_txn(const std::string &path) {
  txn::Stream s;
  try {
    s = txn::parse(txn::words_of(read_file(path)));
  } catch (const std::exception &e) {
    throw std::runtime_error(path + ": " + e.what());
  }
  if (s.header.dev_gen != hw::kTxnDevGen || s.header.cols != hw::kCols || s.header.rows != hw::kRows)
    throw std::runtime_error(std::format("{}: device generation {} with {} x {} tiles; the library is for npu2 "
                                         "(generation {}, {} x {})",
                                         path, s.header.dev_gen, s.header.cols, s.header.rows, hw::kTxnDevGen,
                                         hw::kCols, hw::kRows));
  return s;
}

/// The registers and memory words the writes of `ops` set: address -> last value.
inline std::map<uint32_t, uint32_t> values_of(const std::vector<txn::Op> &ops) {
  std::map<uint32_t, uint32_t> v;
  for (const txn::Op &op : ops)
    if (const auto *w = std::get_if<txn::Write32>(&op))
      v[w->addr] = w->value;
    else if (const auto *m = std::get_if<txn::MaskWrite32>(&op))
      v[m->addr] = (v[m->addr] & ~m->mask) | (m->value & m->mask);
    else if (const auto *b = std::get_if<txn::BlockWrite>(&op))
      for (size_t i = 0; i < b->words.size(); ++i)
        v[b->addr + 4 * (uint32_t)i] = b->words[i];
  return v;
}

/// A state against its app's init.txn (see the top of this file).
inline void check_state(const txn::Stream &init, const txn::Stream &state, const std::string &path) {
  const std::map<uint32_t, uint32_t> base = values_of(init.ops);
  for (size_t i = 0; i < state.ops.size(); ++i) {
    const txn::Op &op = state.ops[i];
    if (const auto *m = std::get_if<txn::MaskWrite32>(&op); m && m->mask == hw::kDmaReset)
      continue; // a channel reset: dropped as init.txn's
    for (const auto &[addr, value] : values_of({op})) {
      auto fail = [&](const std::string &m) {
        throw std::runtime_error(std::format("{}: op {}: address {:#010x}: {}", path, i, addr, m));
      };
      hw::Cls cls = hw::Cls::data;
      try {
        cls = hw::classify(addr);
      } catch (const std::logic_error &e) { // sort_init names it too; here first
        fail(e.what());
      }
      if (cls == hw::Cls::program)
        fail("a state cannot change the program");
      auto it = base.find(addr);
      if (cls != hw::Cls::lock && it != base.end() && it->second != 0 && it->second != value)
        fail(std::format("the state writes {:#x} where init.txn sets {:#x} (a state may only set what the app "
                         "leaves at 0)",
                         value, it->second));
    }
  }
}

/// "APP[A-B]": the columns A..B of the name, -1 -1 for none; `name` loses the suffix.
inline std::pair<int, int> column_range(std::string &name, const std::string &app) {
  if (name.empty() || name.back() != ']')
    return {-1, -1};
  const size_t open = name.rfind('[');
  int lo = -1, hi = -1;
  char dash = 0, end = 0;
  if (open == std::string::npos ||
      std::sscanf(name.c_str() + open, "[%d%c%d%c", &lo, &dash, &hi, &end) != 4 || dash != '-' || end != ']' ||
      lo < 0 || hi < lo || hi >= hw::kCols)
    throw std::runtime_error(std::format("app '{}': APP[A-B] expected, columns 0 <= A <= B < {}", app, hw::kCols));
  name.resize(open);
  return {lo, hi};
}

/// Keep the tiles of columns lo..hi (APP[A-B]). A core may not wait on a lock
/// across the border: the census says which ones it acquires.
inline void restrict_columns(Package &p, int lo, int hi) {
  auto in = [&](int col) { return col >= lo && col <= hi; };
  auto in_addr = [&](uint32_t a) { return in(hw::locate(a).col); };
  for (const auto &[core, lock] : p.acquires)
    if (in(core.col) != in_addr(lock))
      throw std::runtime_error(std::format("app '{}': the {} acquires a lock of column {}, across the border",
                                           p.name, core.describe(), hw::locate(lock).col));
  std::erase_if(p.tiles, [&](const auto &e) { return !in(e.first.col); });
  std::erase_if(p.controller_id, [&](const auto &e) { return !in(e.first.col); });
  std::erase_if(p.buffers, [&](const auto &e) { return !in(e.second.tile.col); });
  std::erase_if(p.lock_names, [&](const auto &e) { return !in(e.second.tile.col); });
  std::erase_if(p.allocs, [&](const auto &e) { return !in(e.second.col); });
  std::erase_if(p.waits, [&](uint32_t a) { return !in_addr(a); });
  std::erase_if(p.declared_locks, [&](uint32_t a) { return !in_addr(a); });
  std::erase_if(p.acquires, [&](const auto &e) { return !in(e.first.col); });
  if (p.cores().empty())
    throw std::runtime_error(std::format("app '{}': no core in columns {}..{}", p.name, lo, hi));
}

} // namespace detail

/// Read PKG/<app>/, or PKG/APP/ with its state for "APP@STATE", or the columns
/// A..B of either for "APP[A-B]" / "APP@STATE[A-B]". Throws std::runtime_error
/// naming the file.
inline Package load_package(const std::string &package_dir, const std::string &app) {
  Package p;
  p.name = app;
  std::string name = app;
  const auto [lo, hi] = detail::column_range(name, app);
  const size_t at = name.find('@');
  const std::string base = name.substr(0, at);
  if (at != std::string::npos) {
    p.state = name.substr(at + 1);
    if (base.empty() || p.state.empty() || p.state.find('@') != std::string::npos)
      throw std::runtime_error(std::format("app '{}': APP or APP@STATE expected", app));
  }
  p.dir = (std::filesystem::path(package_dir) / base).string();
  p.xclbin = (std::filesystem::path(p.dir) / "design.xclbin").string();
  if (!std::filesystem::exists(p.xclbin))
    throw std::runtime_error(std::format("app '{}': no {} (tools/make_package.py)", app, p.xclbin));
  detail::read_manifest(p, p.dir + "/manifest.txt");
  const std::string init = p.dir + "/init.txn";
  const txn::Stream s = detail::read_txn(init);
  p.header = s.header;
  detail::sort_init(p, s, init);
  if (!p.state.empty()) {
    const std::string path = p.dir + "/state_" + p.state + ".txn";
    if (!std::filesystem::exists(path))
      throw std::runtime_error(std::format("app '{}': no {} (tools/make_package.py --state)", app, path));
    const txn::Stream st = detail::read_txn(path);
    detail::check_state(s, st, path);
    const std::vector<Tile> cores = p.cores();
    detail::sort_init(p, st, path);
    if (p.cores() != cores)
      throw std::runtime_error(std::format("{}: the state has a core that '{}' does not have", path, base));
  }
  for (const Tile &t : p.cores())
    if (p.tiles.at(t).program.empty())
      throw std::runtime_error(std::format("{}: the {} has a core but no program", init, t.describe()));
  if (lo >= 0)
    detail::restrict_columns(p, lo, hi);
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
