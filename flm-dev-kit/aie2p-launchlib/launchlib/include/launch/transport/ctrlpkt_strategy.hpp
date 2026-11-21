//===- ctrlpkt_strategy.hpp - load with control packets -----------*- C++ -*-===//
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
// Transport::ctrlpkt: the mem and core tiles of a column get their state as
// control packets (ctrlpkt_packets.hpp) that the column's shim DMA sends down
// its lane (ctrlpkt_lane.hpp), all columns at once, while the command processor
// writes what packets may not. The packets of a kernel are one buffer, the
// kernel argument `packet_arg`; the kernel's shim BDs point into it.
//
// load(k):
//
//   TXN      the mux of every shim (and the lane's DMA bit where the app leaves
//            the field 0)
//   chain A  per lane column, one shim BD per packet id: the core rows, each
//            the drained locks (63), the program, the data, the lock values,
//            the BDs (one broadcast chunk when the rows are equal, or the
//            drain and program broadcast and the rest per row), then the mem
//            tile: its locks, BDs (and data with memtile_data)
//   TXN      meanwhile: the shim tiles, the mem-tile data, the DMA channel Ctrl
//            writes, every tile of a column without a lane
//   Sync     per lane column (the chain's last BD issues the token)
//   TXN      mask writes of data and anything else a packet may not write, and
//            what the runtime sequence writes too (an RTP lock it sets: its TXN
//            write could land before a packet still in flight)
//   chain B  per lane column: the Start_Queue pushes and Core_Control 0, 1 of
//            each core row, then the mem tile's pushes; Sync. Only with
//            CtrlPktOptions::starts_by_packet and when every toggle of the
//            app is in the shim (an app stream through a toggled mem or core
//            switch would take lane routes once its DMA runs); else TXN writes
//            as TxnStrategy, after the toggles go back.
//   TXN      the toggles back to the app's values
//
// load_program(k): the lanes' registers (with CtrlPktOptions::rewrite_lanes,
// and the lane's mux bit; else only the toggles) to lane values, per lane
// column one chain of Core_Control 2, the program, Core_Control 0, 1 per core
// row (one broadcast chunk when the rows are equal), Sync, the toggles back.
//
// A column without a lane, or with fewer free shim BDs than its chains need
// (the sequence's own BDs are not touched), loads with TXN writes as in
// TxnStrategy; so does every column of a kernel whose sequence uses all five
// kernel arguments. The reset part's lock drain of a lane column's core rows
// moves into chain A (adjust_reset).
//
// A warm kernel's reset part writes every register of its lanes, and so does
// a program-only kernel before its chain (CtrlPktOptions::rewrite_lanes): the
// lanes do not depend on what the array held, so a switch of another
// hw_context (which loads that context's xclbin) costs wrong data, as with
// TxnStrategy, not a Sync that never ends. Without rewrite_lanes the lanes'
// residents must still be in the array (`final_routes(prev)`).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/config.hpp"
#include "launch/transport/ctrlpkt_lane.hpp"
#include "launch/transport/ctrlpkt_packets.hpp"
#include "launch/transport/strategy.hpp"

#include <map>
#include <optional>
#include <set>
#include <vector>

namespace launch::detail {

class CtrlPktStrategy final : public TransportStrategy {
public:
  explicit CtrlPktStrategy(CtrlPktOptions o = {}) : opts_(o) {}

  const ctrlpkt::Plan &plan_of(const Package &p) {
    auto it = plans_.find(&p);
    if (it == plans_.end())
      it = plans_.emplace(&p, ctrlpkt::plan(p)).first;
    return it->second;
  }

  const RouteMap &final_routes(const Package &p) override { return plan_of(p).final_map; }
  const RouteMap &load_routes(const Package &p) override { return plan_of(p).load_map; }

  void adjust_reset(KernelContext &k, ResetTargets &t) override {
    for (const Package *p : {k.prev, k.run})
      if (p)
        for (const ctrlpkt::Lane &l : plan_of(*p).lanes)
          t.muxes.insert(Tile{l.col, 0});
    if (!k.run)
      return;
    for (const auto &[col, c] : packet_columns(k)) {
      if (k.prev && opts_.rewrite_lanes)
        for (const auto &[tile, regs] : c.lane->regs)
          for (const auto &[off, v] : regs)
            t.routes[tile][off] = v;
      for (int row = hw::kFirstCoreRow; row <= c.lane->top; ++row)
        for (uint32_t i = 0; i < hw::locks(hw::Kind::core).count; ++i) {
          const uint32_t a = Tile{col, row}.addr(hw::locks(hw::Kind::core).at(i));
          if (t.drain.erase(a))
            k.drain_by_load.insert(a);
        }
    }
  }

  void load(KernelContext &k) override {
    const Package &p = *k.run;
    const ctrlpkt::Plan &plan = plan_of(p);
    const std::map<int, Column> cols = packet_columns(k);
    auto by_packet = [&](Tile t) {
      auto c = cols.find(t.col);
      return t.row >= hw::kMemRows && c != cols.end() && t.row <= c->second.lane->top;
    };

    // The muxes, with the lane's DMA bit.
    for (const auto &[tile, init] : p.tiles)
      for (const txn::Op &op : init.mux)
        k.ops.push_back(op);
    for (const auto &[col, c] : cols)
      if (c.lane->mux_resident)
        k.ops.push_back(txn::MaskWrite32{hw::address(col, 0, hw::kShimMux), ctrlpkt::kMuxDma
                                                                                << ctrlpkt::mux_shift(c.lane->ch),
                                         3u << ctrlpkt::mux_shift(c.lane->ch)});

    // Chain A, and what goes by TXN writes meanwhile or after it.
    std::vector<txn::Op> late;
    for (const auto &[col, c] : cols)
      send(k, c, chain_a(k, c, late));
    for (auto member : {&TileInit::program, &TileInit::data})
      for (const auto &[tile, init] : p.tiles)
        if (!by_packet(tile) || (member == &TileInit::data && tile.kind() == hw::Kind::mem && !opts_.memtile_data))
          for (const txn::Op &op : init.*member)
            k.ops.push_back(op);
    for (const auto &[tile, init] : p.tiles)
      if (!by_packet(tile))
        for (const auto &[off, value] : init.locks)
          if (value != 0 || p.needs_lock(tile, off))
            k.ops.push_back(txn::Write32{tile.addr(off), value});
    for (const auto &[tile, init] : p.tiles)
      if (!by_packet(tile))
        for (const txn::Op &op : init.bds)
          k.ops.push_back(op);
    for (const auto &[tile, init] : p.tiles)
      if (by_packet(tile))
        for (const txn::Write32 &w : starts_of(init))
          if (is_ctrl(w.addr))
            k.ops.push_back(w);
    for (const auto &[col, c] : cols)
      sync(k, c);
    k.ops.insert(k.ops.end(), late.begin(), late.end());

    // The starts.
    if (!cols.empty() && opts_.starts_by_packet && plan.shim_only_toggles) {
      for (const auto &[col, c] : cols)
        send(k, c, chain_b(k, c));
      for (const auto &[tile, init] : p.tiles)
        if (!by_packet(tile))
          for (const txn::Write32 &w : starts_of(init))
            k.ops.push_back(w);
      for (const Tile &core : p.cores())
        if (!by_packet(core))
          core_start(k.ops, core);
      for (const auto &[col, c] : cols)
        sync(k, c);
      restore(k, plan);
    } else {
      restore(k, plan);
      for (const auto &[tile, init] : p.tiles)
        for (const txn::Write32 &w : starts_of(init))
          if (!by_packet(tile) || !is_ctrl(w.addr))
            k.ops.push_back(w);
      for (const Tile &core : p.cores())
        core_start(k.ops, core);
    }
  }

  void load_program(KernelContext &k) override {
    const Package &p = *k.run;
    const ctrlpkt::Plan &plan = plan_of(p);
    const std::map<int, Column> cols = packet_columns(k);
    if (opts_.rewrite_lanes)
      for (const auto &[col, c] : cols) {
        if (c.lane->mux_resident)
          k.ops.push_back(txn::MaskWrite32{hw::address(col, 0, hw::kShimMux),
                                           ctrlpkt::kMuxDma << ctrlpkt::mux_shift(c.lane->ch),
                                           3u << ctrlpkt::mux_shift(c.lane->ch)});
        for (const auto &[t, regs] : c.lane->regs)
          write_runs(k.ops, t, regs);
      }
    for (const auto &[t, m] : plan.toggles)
      if (cols.count(t.col) && !opts_.rewrite_lanes) {
        std::map<uint32_t, uint32_t> lane;
        for (const auto &[off, v] : m)
          lane[off] = plan.load_map.at(t).at(off);
        write_runs(k.ops, t, lane);
      }
    for (const auto &[col, c] : cols) {
      std::map<int, std::vector<ctrlpkt::Write>> rows;
      for (int row = hw::kFirstCoreRow; row <= c.lane->top; ++row) {
        const Tile t{col, row};
        auto it = p.tiles.find(t);
        if (it == p.tiles.end() || !it->second.core)
          continue;
        std::vector<ctrlpkt::Write> &w = rows[row];
        w.push_back({hw::kCoreControl, {hw::kCoreReset}});
        std::vector<txn::Op> rest;
        to_writes(hw::Kind::core, it->second.program, w, rest);
        if (!rest.empty())
          throw std::logic_error("ctrlpkt: a program write a packet may not do (packet_columns checks this)");
        w.push_back({hw::kCoreControl, {0}});
        w.push_back({hw::kCoreControl, {hw::kCoreEnable}});
      }
      send(k, c, per_row_or_broadcast(*c.lane, rows));
    }
    for (const Tile &core : p.cores())
      if (!cols.count(core.col)) {
        core_hold(k.ops, core);
        for (const txn::Op &op : p.tiles.at(core).program)
          k.ops.push_back(op);
        core_start(k.ops, core);
      }
    for (const auto &[col, c] : cols)
      sync(k, c);
    for (const auto &[t, m] : plan.toggles)
      if (cols.count(t.col))
        write_runs(k.ops, t, m);
  }

private:
  struct Column {
    const ctrlpkt::Lane *lane = nullptr;
    std::vector<uint32_t> bds; ///< free shim BDs, highest first
  };
  struct Chunk {
    uint32_t id;
    hw::Kind kind;
    std::vector<ctrlpkt::Write> writes;
  };

  static bool is_ctrl(uint32_t addr) { return channel_ctrl_of(addr) == addr; }

  /// The chain-start writes of a tile (the package keeps only Write32 there).
  static std::vector<txn::Write32> starts_of(const TileInit &init) {
    std::vector<txn::Write32> out;
    for (const txn::Op &op : init.starts)
      out.push_back(std::get<txn::Write32>(op));
    return out;
  }

  /// The shim BDs of column `col` the sequence or the app writes or pushes.
  static std::set<uint32_t> used_shim_bds(const KernelContext &k, int col) {
    std::set<uint32_t> out;
    const hw::Bank bds = hw::bds(hw::Kind::shim);
    auto is_queue = [](uint32_t off) {
      for (hw::Dir d : {hw::Dir::s2mm, hw::Dir::mm2s}) {
        const hw::Bank ch = hw::channels(hw::Kind::shim, d);
        if (off >= ch.base + hw::kQueueFromCtrl && ch.holds(off - hw::kQueueFromCtrl) &&
            (off - hw::kQueueFromCtrl - ch.base) % ch.step == 0)
          return true;
      }
      return false;
    };
    auto scan = [&](const std::vector<txn::Op> &ops) {
      for (const txn::Op &op : ops) {
        if (std::holds_alternative<txn::Sync>(op) || std::holds_alternative<txn::MaskPoll>(op) ||
            std::holds_alternative<txn::Preempt>(op))
          continue;
        const uint32_t addr = txn::addr_of(op);
        const hw::Where w = hw::locate(addr);
        if (w.col != col || w.row != 0)
          continue;
        size_t words = 1;
        if (const auto *b = std::get_if<txn::BlockWrite>(&op))
          words = b->words.size();
        for (size_t i = 0; i < words; ++i) {
          const uint32_t off = w.offset + 4 * (uint32_t)i;
          if (bds.holds(off))
            out.insert((off - bds.base) / bds.step);
          if (is_queue(off)) {
            if (const auto *s = std::get_if<txn::Write32>(&op))
              out.insert(s->value & 0xF);
            else if (const auto *b = std::get_if<txn::BlockWrite>(&op))
              out.insert(b->words[i] & 0xF);
          }
        }
      }
    };
    if (k.sequence)
      scan(*k.sequence);
    auto it = k.run->tiles.find(Tile{col, 0});
    if (it != k.run->tiles.end()) {
      scan(it->second.bds);
      scan(it->second.data);
      scan(it->second.starts);
    }
    return out;
  }

  /// The array words the kernel's sequence writes.
  static std::set<uint32_t> sequence_words(const KernelContext &k) {
    std::set<uint32_t> out;
    if (k.sequence)
      for (const txn::Op &op : *k.sequence)
        for (uint32_t a : txn::written(op))
          out.insert(a);
    return out;
  }

  /// Whether every write of `ops` can go as packets into a tile of `kind`.
  bool all_packets(hw::Kind kind, const std::vector<txn::Op> &ops) const {
    std::vector<ctrlpkt::Write> w;
    std::vector<txn::Op> rest;
    to_writes(kind, ops, w, rest);
    return rest.empty();
  }

  /// The lane columns that load by packet in this kernel, with their shim BDs.
  std::map<int, Column> packet_columns(const KernelContext &k) {
    std::map<int, Column> out;
    if (k.free_arg < 0 || !k.run)
      return out;
    for (const ctrlpkt::Lane &l : plan_of(*k.run).lanes) {
      const int ncore = std::max(0, l.top - hw::kFirstCoreRow + 1);
      const size_t need = (size_t)ncore + 2;
      const std::set<uint32_t> used = used_shim_bds(k, l.col);
      Column c{&l, {}};
      for (int bd = (int)hw::bds(hw::Kind::shim).count - 1; bd >= 0 && c.bds.size() < need; --bd)
        if (!used.count((uint32_t)bd))
          c.bds.push_back((uint32_t)bd);
      if (c.bds.size() < need)
        continue;
      // A program-only kernel writes the program by packet only: it must all fit.
      bool ok = true;
      for (int row = hw::kFirstCoreRow; row <= l.top; ++row) {
        auto it = k.run->tiles.find(Tile{l.col, row});
        if (it != k.run->tiles.end() && !all_packets(hw::Kind::core, it->second.program))
          ok = false;
      }
      if (ok)
        out.emplace(l.col, std::move(c));
    }
    return out;
  }

  /// Packet writes of `ops` into a tile of `kind`; the ops a packet may not do go to `rest`.
  void to_writes(hw::Kind kind, const std::vector<txn::Op> &ops, std::vector<ctrlpkt::Write> &w,
                 std::vector<txn::Op> &rest) const {
    for (const txn::Op &op : ops) {
      std::optional<ctrlpkt::Write> x;
      if (const auto *b = std::get_if<txn::BlockWrite>(&op))
        x = ctrlpkt::Write{hw::locate(b->addr).offset, b->words};
      else if (const auto *s = std::get_if<txn::Write32>(&op))
        x = ctrlpkt::Write{hw::locate(s->addr).offset, {s->value}};
      if (x && !x->words.empty() && fits(kind, *x))
        w.push_back(std::move(*x));
      else
        rest.push_back(op);
    }
  }

  bool fits(hw::Kind kind, const ctrlpkt::Write &w) const {
    for (size_t at = 0; at < w.words.size();) {
      const uint32_t offset = w.offset + 4 * (uint32_t)at;
      const size_t beats = std::min({w.words.size() - at, ctrlpkt::kMaxBeats, (size_t)(4 - offset % 16 / 4)});
      if (offset > ctrlpkt::kOffsetBits || !ctrlpkt::allowed(kind, offset, beats, opts_.memtile_data))
        return false;
      at += beats;
    }
    return true;
  }

  /// Chain A of a lane column: the state of its mem and core tiles. The writes
  /// a packet may not do go to `late` (TXN writes after the Sync).
  std::vector<Chunk> chain_a(const KernelContext &k, const Column &c, std::vector<txn::Op> &late) const {
    const Package &p = *k.run;
    const int col = c.lane->col;
    // A word the sequence writes (an RTP lock it sets) goes by TXN: the
    // sequence's TXN write could land before a packet still in flight.
    const std::set<uint32_t> seq = sequence_words(k);
    auto pick = [&](hw::Kind kind, const std::vector<txn::Op> &ops, std::vector<ctrlpkt::Write> &w) {
      std::vector<txn::Op> mine;
      for (const txn::Op &op : ops) {
        bool txn = false;
        for (uint32_t a : txn::written(op))
          txn = txn || seq.count(a);
        (txn ? late : mine).push_back(op);
      }
      to_writes(kind, mine, w, late);
    };
    auto lock = [&](Tile t, uint32_t off, uint32_t value, std::vector<ctrlpkt::Write> &w) {
      if (seq.count(t.addr(off)))
        late.push_back(txn::Write32{t.addr(off), value});
      else
        w.push_back({off, {value}});
    };
    std::map<int, std::vector<ctrlpkt::Write>> head, rest;
    for (int row = hw::kFirstCoreRow; row <= c.lane->top; ++row) {
      const Tile t{col, row};
      const hw::Bank lk = hw::locks(hw::Kind::core);
      std::set<uint32_t> drained;
      for (uint32_t i = 0; i < lk.count; ++i)
        if (k.drain_by_load.count(t.addr(lk.at(i))))
          drained.insert(lk.at(i));
      auto it = p.tiles.find(t);
      if (it == p.tiles.end() && drained.empty())
        continue;
      std::vector<ctrlpkt::Write> &h = head[row], &r = rest[row];
      for (uint32_t off : drained)
        lock(t, off, hw::kLockMax, h);
      if (it != p.tiles.end()) {
        pick(hw::Kind::core, it->second.program, h);
        pick(hw::Kind::core, it->second.data, r);
      }
      std::map<uint32_t, uint32_t> locks;
      for (uint32_t off : drained)
        locks[off] = 0;
      if (it != p.tiles.end())
        for (const auto &[off, value] : it->second.locks)
          if (value != 0 || p.needs_lock(t, off))
            locks[off] = value;
      for (const auto &[off, value] : locks)
        lock(t, off, value, r);
      if (it != p.tiles.end())
        pick(hw::Kind::core, it->second.bds, r);
    }
    std::vector<Chunk> out;
    // One broadcast when every row of the lane has the same writes; else the
    // head (drain, program) broadcast when equal, the rest per row.
    std::map<int, std::vector<ctrlpkt::Write>> whole;
    for (const auto &[row, h] : head) {
      whole[row] = h;
      whole[row].insert(whole[row].end(), rest[row].begin(), rest[row].end());
    }
    if (all_equal(*c.lane, whole)) {
      out.push_back({ctrlpkt::kBroadcastId, hw::Kind::core, whole.begin()->second});
    } else if (all_equal(*c.lane, head)) {
      out.push_back({ctrlpkt::kBroadcastId, hw::Kind::core, head.begin()->second});
      for (const auto &[row, r] : rest)
        if (!r.empty())
          out.push_back({ctrlpkt::row_id(row), hw::Kind::core, r});
    } else if (auto patched = patch_rows(*c.lane, head)) {
      // Programs that differ in a few words (a row constant): what the rows
      // share broadcast, then each row's other words and its rest.
      out.push_back({ctrlpkt::kBroadcastId, hw::Kind::core, std::move(patched->common)});
      for (auto &[row, w] : patched->rows) {
        w.insert(w.end(), rest[row].begin(), rest[row].end());
        if (!w.empty())
          out.push_back({ctrlpkt::row_id(row), hw::Kind::core, std::move(w)});
      }
    } else {
      for (const auto &[row, w] : whole)
        if (!w.empty())
          out.push_back({ctrlpkt::row_id(row), hw::Kind::core, w});
    }
    auto mt = p.tiles.find(Tile{col, 1});
    if (mt != p.tiles.end()) {
      std::vector<ctrlpkt::Write> w;
      for (const auto &[off, value] : mt->second.locks)
        if (value != 0 || p.needs_lock(mt->first, off))
          lock(mt->first, off, value, w);
      pick(hw::Kind::mem, mt->second.bds, w);
      if (opts_.memtile_data)
        pick(hw::Kind::mem, mt->second.data, w);
      if (!w.empty())
        out.push_back({ctrlpkt::kMemId, hw::Kind::mem, std::move(w)});
    }
    return out;
  }

  /// Chain B of a lane column: the Start_Queue pushes and the core starts of
  /// its core rows, then the mem tile's pushes.
  std::vector<Chunk> chain_b(const KernelContext &k, const Column &c) const {
    const Package &p = *k.run;
    const int col = c.lane->col;
    auto queues = [&](Tile t, std::vector<ctrlpkt::Write> &w) {
      auto it = p.tiles.find(t);
      if (it != p.tiles.end())
        for (const txn::Write32 &s : starts_of(it->second))
          if (!is_ctrl(s.addr))
            w.push_back({hw::locate(s.addr).offset, {s.value}});
    };
    std::map<int, std::vector<ctrlpkt::Write>> rows;
    for (int row = hw::kFirstCoreRow; row <= c.lane->top; ++row) {
      const Tile t{col, row};
      std::vector<ctrlpkt::Write> w;
      queues(t, w);
      auto it = p.tiles.find(t);
      if (it != p.tiles.end() && it->second.core) {
        w.push_back({hw::kCoreControl, {0}});
        w.push_back({hw::kCoreControl, {hw::kCoreEnable}});
      }
      if (!w.empty())
        rows[row] = std::move(w);
    }
    std::vector<Chunk> out = per_row_or_broadcast(*c.lane, rows);
    std::vector<ctrlpkt::Write> m;
    queues(Tile{col, 1}, m);
    if (!m.empty())
      out.push_back({ctrlpkt::kMemId, hw::Kind::mem, std::move(m)});
    return out;
  }

  /// Whether every core row of the lane has the same, non-empty writes.
  static bool all_equal(const ctrlpkt::Lane &l, const std::map<int, std::vector<ctrlpkt::Write>> &rows) {
    if ((int)rows.size() != l.top - hw::kFirstCoreRow + 1 || rows.empty())
      return false;
    const std::vector<ctrlpkt::Write> &first = rows.begin()->second;
    if (first.empty())
      return false;
    for (const auto &[row, w] : rows)
      if (w != first)
        return false;
    return true;
  }

  /// Rows that write the same words but for a few (a row constant in the
  /// program): the broadcast of the words every row writes, each with the
  /// value most rows write, then per row its words the broadcast did not leave right,
  /// in runs. None when that sends no fewer words than one chunk per row.
  struct Patched {
    std::vector<ctrlpkt::Write> common;
    std::map<int, std::vector<ctrlpkt::Write>> rows;
  };
  static std::optional<Patched> patch_rows(const ctrlpkt::Lane &l,
                                           const std::map<int, std::vector<ctrlpkt::Write>> &rows) {
    if ((int)rows.size() != l.top - hw::kFirstCoreRow + 1 || rows.empty())
      return std::nullopt;
    auto words = [](const std::vector<ctrlpkt::Write> &w) {
      std::map<uint32_t, uint32_t> m;
      for (const ctrlpkt::Write &x : w)
        for (size_t i = 0; i < x.words.size(); ++i)
          m[x.offset + 4 * (uint32_t)i] = x.words[i];
      return m;
    };
    auto append = [](std::vector<ctrlpkt::Write> &w, uint32_t off, uint32_t v) {
      if (!w.empty() && w.back().offset + 4 * (uint32_t)w.back().words.size() == off)
        w.back().words.push_back(v);
      else
        w.push_back({off, {v}});
    };
    std::map<int, std::map<uint32_t, uint32_t>> all;
    for (const auto &[row, w] : rows)
      all[row] = words(w);
    std::map<uint32_t, uint32_t> common;
    for (const auto &[off, v] : all.begin()->second) {
      std::map<uint32_t, int> votes;
      bool everywhere = true;
      for (const auto &[row, m] : all) {
        auto at = m.find(off);
        everywhere = everywhere && at != m.end();
        if (at != m.end())
          ++votes[at->second];
      }
      if (!everywhere)
        continue;
      auto most = votes.begin();
      for (auto x = votes.begin(); x != votes.end(); ++x)
        if (x->second > most->second)
          most = x;
      common[off] = most->first;
    }
    Patched p;
    for (const auto &[off, v] : common)
      append(p.common, off, v);
    size_t sent = common.size(), whole = 0;
    for (const auto &[row, m] : all) {
      whole += m.size();
      std::vector<ctrlpkt::Write> &w = p.rows[row];
      for (const auto &[off, v] : m) {
        auto c = common.find(off);
        if (c != common.end() && c->second == v)
          continue;
        append(w, off, v);
        ++sent;
      }
    }
    if (common.empty() || sent >= whole)
      return std::nullopt;
    return p;
  }

  static std::vector<Chunk> per_row_or_broadcast(const ctrlpkt::Lane &l,
                                                 const std::map<int, std::vector<ctrlpkt::Write>> &rows) {
    std::vector<Chunk> out;
    if (all_equal(l, rows))
      out.push_back({ctrlpkt::kBroadcastId, hw::Kind::core, rows.begin()->second});
    else
      for (const auto &[row, w] : rows)
        if (!w.empty())
          out.push_back({ctrlpkt::row_id(row), hw::Kind::core, w});
    return out;
  }

  /// The chunks of `chain` into the kernel's packet buffer, one shim BD each,
  /// chained, and the push of the first.
  void send(KernelContext &k, const Column &c, const std::vector<Chunk> &chain) {
    if (chain.empty())
      return;
    if (chain.size() > c.bds.size())
      throw std::logic_error("ctrlpkt: a chain longer than the column's free shim BDs");
    k.packet_arg = k.free_arg;
    const int col = c.lane->col;
    // The BDs, then their patches. BDs next to each other (c.bds is highest
    // first) go in one block write: one op per run instead of one per BD.
    std::map<uint32_t, std::array<uint32_t, 8>> bds; // BD address -> its words
    std::vector<txn::AddressPatch> patches;
    for (size_t i = 0; i < chain.size(); ++i) {
      const std::vector<uint32_t> words = ctrlpkt::encode(chain[i].id, chain[i].kind, chain[i].writes,
                                                          opts_.memtile_data);
      const uint32_t at = place(k, words);
      const uint32_t bd = hw::address(col, 0, hw::bds(hw::Kind::shim).at(c.bds[i]));
      bds[bd] = ctrlpkt::shim_bd_words((uint32_t)words.size(), i + 1 < chain.size() ? (int)c.bds[i + 1] : -1);
      patches.push_back(txn::AddressPatch{bd + 4, (uint32_t)k.packet_arg, at});
    }
    for (auto it = bds.begin(); it != bds.end();) {
      txn::BlockWrite b{it->first, {}};
      for (; it != bds.end() && it->first == b.addr + 4 * (uint32_t)b.words.size(); ++it)
        b.words.insert(b.words.end(), it->second.begin(), it->second.end());
      k.ops.push_back(std::move(b));
    }
    k.ops.insert(k.ops.end(), patches.begin(), patches.end());
    const uint32_t ctrl = hw::address(col, 0, hw::channels(hw::Kind::shim, hw::Dir::mm2s).at((uint32_t)c.lane->ch));
    k.ops.push_back(txn::MaskWrite32{ctrl, k.run->controller_id.at(Tile{col, 0}) << hw::kTokenControllerShift,
                                     hw::kTokenControllerMask});
    k.ops.push_back(txn::Write32{ctrl + hw::kQueueFromCtrl, c.bds[0] | hw::kQueueToken});
    pending_.insert(col);
  }

  void sync(KernelContext &k, const Column &c) {
    if (!pending_.erase(c.lane->col))
      return;
    k.ops.push_back(txn::Sync{(uint32_t)c.lane->col, 0, (uint32_t)hw::Dir::mm2s, (uint32_t)c.lane->ch});
  }

  /// The byte offset of `words` in the kernel's packets (equal chunks once).
  static uint32_t place(KernelContext &k, const std::vector<uint32_t> &words) {
    const size_t n = words.size();
    for (size_t at = 0; at + n <= k.packets.size(); ++at)
      if (std::equal(words.begin(), words.end(), k.packets.begin() + (ptrdiff_t)at))
        return 4 * (uint32_t)at;
    const uint32_t at = 4 * (uint32_t)k.packets.size();
    k.packets.insert(k.packets.end(), words.begin(), words.end());
    return at;
  }

  static void restore(KernelContext &k, const ctrlpkt::Plan &plan) {
    for (const auto &[t, m] : plan.toggles)
      write_runs(k.ops, t, m);
  }

  CtrlPktOptions opts_;
  std::map<const Package *, ctrlpkt::Plan> plans_;
  std::set<int> pending_; ///< columns pushed and not yet synced
};

} // namespace launch::detail
