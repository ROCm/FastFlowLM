//===- test_ctrlpkt.cpp - Transport::ctrlpkt and its check, no NPU -*- C++ -*-===//
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
//   test_ctrlpkt PKG A B
//
// A and B: two apps of one package (vadd, vmul). Each app's runtime sequence
// is one transfer per shim allocation (arguments 0..2) and a wait on each S2MM.
//   * the lanes of each app (the plan), printed;
//   * build() with every combination of CtrlPktOptions, check on: the kernels
//     pass their check; the packets are where the kernels say (packet_arg, not
//     an argument of the sequence); reset and rerun have none; the check
//     changes nothing in the kernels; each kernel converts to the ELF flow;
//     the options survive JSON;
//   * the check finds what is wrong (ctrlpkt_check.hpp, on warm(A, B)): a
//     packet corrupted, the packets cut short or on another argument, a Sync
//     missing, a TXN write into a tile while its chain is in flight;
//   * a sequence of B that sets the core locks the load writes (an RTP lock):
//     cold(B) and warm(A, B) write them by TXN, and pass their check;
//   * lanes lost (another hw_context's switch puts its xclbin in the array, so
//     the lanes' residents are gone): warm(A, B) and warm(A, B, program_only)
//     still reach every tile with CtrlPktOptions::rewrite_lanes; without it
//     the check sees packets that go nowhere.
//
//===----------------------------------------------------------------------===//

#include "check.hpp"

#include "launch/launch.hpp"
#include "launch/txn/elf.hpp"

#include <algorithm>
#include <map>

using namespace launch;
using namespace launch::detail;

namespace {

SequenceMap sequences(const Apps &apps) {
  SequenceMap seqs;
  for (const auto &[name, p] : apps) {
    const Package *pp = &p;
    seqs[name] = [pp](Rt &rt) {
      std::map<int, uint32_t> next_bd;
      std::vector<ShimChannel> waits;
      unsigned arg = 0;
      for (const auto &[n, ch] : pp->allocs) {
        const bool tok = ch.dir == hw::Dir::s2mm;
        rt.npu_dma_memcpy_nd({.metadata = ch, .bd_id = next_bd[ch.col]++, .mem = Arg{arg++ % 3, 4},
                              .offsets = {0, 0, 0, 0}, .sizes = {1, 1, 1, 256}, .strides = {0, 0, 0, 1},
                              .issue_token = tok, .push = true});
        if (tok)
          waits.push_back(ch);
      }
      for (const ShimChannel &ch : waits)
        rt.npu_dma_wait(ch);
    };
  }
  seqs[kNoSequence] = [](Rt &) {};
  return seqs;
}

/// One kernel of each transport, and the arrays the check starts a warm kernel
/// from (cold(prev) without a sequence), as build() makes them.
struct Twin {
  KernelBuild cp, txn;
  ctrlpkt::Replay cp_start;
  sw::Registers txn_start;
};

Twin twin(const Apps &apps, const SequenceMap &seqs, CtrlPktStrategy &cps, const Config &c) {
  TxnStrategy txs;
  const KernelPipeline cpp(apps, cps, seqs), txp(apps, txs, seqs);
  Twin t{cpp.build_kernel(c), txp.build_kernel(c), {}, {}};
  if (c.kind() == Config::Kind::warm) {
    Config cold_prev = cold(c.prev());
    cold_prev.sequence = kNoSequence;
    const KernelBuild b = cpp.build_kernel(cold_prev), x = txp.build_kernel(cold_prev);
    const Package &prev = apps.at(c.prev());
    const auto errors = ctrlpkt::check_kernel(
        {&prev, &cps.plan_of(prev), false, b.ops, b.packets, b.packet_arg, x.ops, nullptr, nullptr}, &t.cp_start);
    CHECK(errors.empty());
    for (const txn::Op &op : x.ops)
      ctrlpkt::check_detail::apply(t.txn_start, op);
  }
  return t;
}

std::vector<std::string> check(const Apps &apps, CtrlPktStrategy &cps, const Config &c, const Twin &t,
                               const std::vector<txn::Op> &ops, const std::vector<uint32_t> &packets,
                               int packet_arg, const ctrlpkt::Replay *cp_start = nullptr) {
  const Package &run = apps.at(c.app());
  const bool warm_kind = c.kind() == Config::Kind::warm;
  return ctrlpkt::check_kernel({&run, &cps.plan_of(run), false, ops, packets, packet_arg, t.txn.ops,
                                warm_kind ? (cp_start ? cp_start : &t.cp_start) : nullptr,
                                warm_kind ? &t.txn_start : nullptr});
}

bool any_has(const std::vector<std::string> &errors, const std::string &what) {
  return std::any_of(errors.begin(), errors.end(), [&](const std::string &e) { return e.find(what) != e.npos; });
}

void print(const std::vector<std::string> &errors) {
  for (size_t i = 0; i < errors.size() && i < 2; ++i)
    std::printf("      %s\n", errors[i].c_str());
}

/// Index of the first push of a packet chain in `ops` (a shim MM2S queue write
/// with a token), or -1.
int first_push(const std::vector<txn::Op> &ops) {
  for (size_t i = 0; i < ops.size(); ++i)
    if (const auto *w = std::get_if<txn::Write32>(&ops[i])) {
      const hw::Where loc = hw::locate(w->addr);
      if (loc.kind == hw::Kind::shim && ctrlpkt::check_detail::mm2s_queue(loc.offset) >= 0 &&
          (w->value & hw::kQueueToken))
        return (int)i;
    }
  return -1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: test_ctrlpkt PKG A B\n");
    return 2;
  }
  const std::string pkg = argv[1], a = argv[2], b = argv[3];
  Apps apps;
  apps.emplace(a, load_package(pkg, a));
  apps.emplace(b, load_package(pkg, b));
  const SequenceMap seqs = sequences(apps);
  const bool same_floorplan = floorplan_difference(apps.at(a), apps.at(b)).empty();

  // The lanes.
  {
    CtrlPktStrategy cps;
    for (const std::string &app : {a, b}) {
      const ctrlpkt::Plan &plan = cps.plan_of(apps.at(app));
      std::printf("  %s: %zu lane(s)%s\n", app.c_str(), plan.lanes.size(),
                  plan.shim_only_toggles ? "" : ", toggles above the shim");
      for (const ctrlpkt::Lane &l : plan.lanes) {
        size_t regs = 0;
        for (const auto &[t, m] : l.regs)
          regs += m.size();
        std::printf("    column %d: MM2S %d, rows 1..%d, %zu switch registers\n", l.col, l.ch, l.top, regs);
      }
      for (const auto &[col, why] : plan.txn_cols)
        std::printf("    column %d by TXN writes: %s\n", col, why.c_str());
      CHECK(!plan.lanes.empty());
    }
  }

  // build() with every combination of the options.
  std::vector<Config> configs = {reset(), cold(a), warm(a, b), warm(b, a), rerun(a)};
  if (same_floorplan)
    configs.push_back(warm(a, b, {.program_only = true}));
  for (int bits = 0; bits < 8; ++bits) {
    ConfigList list{pkg, Transport::ctrlpkt, configs};
    list.ctrlpkt.memtile_data = bits & 1;
    list.ctrlpkt.starts_by_packet = bits & 2;
    list.ctrlpkt.rewrite_lanes = bits & 4;
    Result r;
    try {
      r = build(list, seqs);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "options %d: %s\n", bits, e.what());
      ++g_failures;
      continue;
    }
    list.ctrlpkt.check = false;
    const Result unchecked = build(list, seqs);
    for (size_t i = 0; i < r.kernels.size(); ++i) {
      const Kernel &k = r.kernels[i];
      const Config::Kind kind = configs[i].kind();
      CHECK(k.words == unchecked.kernels[i].words && k.packets == unchecked.kernels[i].packets);
      if (kind == Config::Kind::reset || kind == Config::Kind::rerun) {
        CHECK(k.packets.empty());
        continue;
      }
      CHECK(!k.packets.empty());
      CHECK(k.packet_arg >= 0 && k.packet_arg < kKernelArgs);
      // The sequence uses arguments 0..2; the packets take the highest free one.
      CHECK(k.packet_arg == kKernelArgs - 1);
      size_t syncs = 0, patches = 0;
      for (const txn::Op &op : txn::parse(k.words).ops) {
        syncs += std::holds_alternative<txn::Sync>(op);
        patches += std::holds_alternative<txn::AddressPatch>(op);
      }
      CHECK(syncs >= 1);
      // The ELF flow: every patch (the chains' BDs among them) a relocation.
      try {
        CHECK(txn::elf_text(k.words).relocations.size() == patches);
      } catch (const std::exception &e) {
        std::fprintf(stderr, "options %d, %s: %s\n", bits, k.name.c_str(), e.what());
        ++g_failures;
      }
    }
    const ConfigList back = config_from_json(config_to_json(list));
    CHECK(back.transport == Transport::ctrlpkt);
    CHECK(back.ctrlpkt.memtile_data == list.ctrlpkt.memtile_data &&
          back.ctrlpkt.starts_by_packet == list.ctrlpkt.starts_by_packet &&
          back.ctrlpkt.rewrite_lanes == list.ctrlpkt.rewrite_lanes && !back.ctrlpkt.check);
  }
  CHECK(throws([] { transport_from_name("dma"); }));
  CHECK(throws([&] {
    config_from_json(R"({"package_dir": "x", "transport": "ctrlpkt", "ctrlpkt": {"lanes": 1}, "configs": []})");
  }));

  // The check finds what is wrong.
  {
    CtrlPktStrategy cps;
    const Config c = warm(a, b);
    const Twin t = twin(apps, seqs, cps, c);
    const auto ok = check(apps, cps, c, t, t.cp.ops, t.cp.packets, t.cp.packet_arg);
    CHECK(ok.empty());
    print(ok);

    std::printf("  negatives of warm(%s, %s):\n", a.c_str(), b.c_str());
    auto expect = [&](const char *what, const std::vector<std::string> &errors) {
      std::printf("    %-40s %s\n", what, errors.empty() ? "NOT FOUND" : "found");
      CHECK(!errors.empty());
      print(errors);
    };
    // A packet header with another id: it goes elsewhere (or nowhere).
    std::vector<uint32_t> bad = t.cp.packets;
    bad[0] ^= 1u << 0;
    expect("a packet corrupted", check(apps, cps, c, t, t.cp.ops, bad, t.cp.packet_arg));
    bad = t.cp.packets;
    bad.resize(bad.size() / 2);
    expect("the packets cut short", check(apps, cps, c, t, t.cp.ops, bad, t.cp.packet_arg));
    expect("the packets on another argument", check(apps, cps, c, t, t.cp.ops, t.cp.packets, t.cp.packet_arg - 1));
    // The first Sync gone.
    std::vector<txn::Op> ops = t.cp.ops;
    auto sync = std::find_if(ops.begin(), ops.end(), [](const txn::Op &op) { return std::holds_alternative<txn::Sync>(op); });
    CHECK(sync != ops.end());
    if (sync != ops.end())
      ops.erase(sync);
    expect("a Sync missing", check(apps, cps, c, t, ops, t.cp.packets, t.cp.packet_arg));
    // A TXN write into a switch register of the first chain's lane, before its Sync.
    const int push = first_push(t.cp.ops);
    const ctrlpkt::Lane *lane = push < 0 ? nullptr
                                         : cps.plan_of(apps.at(b)).lane(hw::locate(txn::addr_of(t.cp.ops[(size_t)push])).col);
    CHECK(lane);
    if (lane) {
      ops = t.cp.ops;
      const auto &[tile, regs] = *lane->regs.begin();
      ops.insert(ops.begin() + push + 1, txn::Write32{tile.addr(regs.begin()->first), 0});
      const auto e = check(apps, cps, c, t, ops, t.cp.packets, t.cp.packet_arg);
      expect("a TXN write while a chain is in flight", e);
      CHECK(any_has(e, "in flight"));
    }
  }

  // A sequence that sets the core locks the load writes (an RTP lock, as
  // pm_runtime_load_singletile's vminus): those locks go by TXN, not packets
  // (the sequence's write could land before a packet in flight), so the
  // kernels pass their check and write each such lock twice by TXN.
  {
    const Package &pb = apps.at(b);
    std::vector<uint32_t> locks;
    for (const Tile &core : pb.cores())
      for (const auto &[off, value] : pb.tiles.at(core).locks)
        if (value != 0 || pb.needs_lock(core, off))
          locks.push_back(core.addr(off));
    CHECK(!locks.empty());
    SequenceMap rtp = seqs;
    rtp[b] = [inner = seqs.at(b), locks](Rt &rt) {
      for (uint32_t addr : locks)
        rt.npu_write32(addr, 1);
      inner(rt);
    };
    for (bool starts : {true, false}) {
      ConfigList list{pkg, Transport::ctrlpkt, {cold(b), warm(a, b)}};
      list.ctrlpkt.starts_by_packet = starts;
      Result r;
      try {
        r = build(list, rtp);
      } catch (const std::exception &e) {
        std::fprintf(stderr, "a sequence that sets locks: %s\n", e.what());
        ++g_failures;
        continue;
      }
      size_t twice = 0;
      for (const Kernel &k : r.kernels) {
        std::map<uint32_t, int> writes;
        for (const txn::Op &op : txn::parse(k.words).ops)
          if (const auto *w = std::get_if<txn::Write32>(&op))
            ++writes[w->addr];
        for (uint32_t addr : locks)
          twice += writes[addr] >= 2;
      }
      std::printf("  a sequence that sets %zu locks, starts_by_packet %s: %zu of %zu written by the load and the "
                  "sequence\n",
                  locks.size(), starts ? "on" : "off", twice, 2 * locks.size());
      CHECK(twice == 2 * locks.size());
    }
  }

  // Lanes lost: the array holds the base app's switch registers (A's own, as an
  // xclbin load leaves them), not the lanes' residents.
  for (bool rewrite : {true, false}) {
    std::vector<Config> lost = {warm(a, b)};
    if (same_floorplan)
      lost.push_back(warm(a, b, {.program_only = true}));
    for (const Config &c : lost) {
      CtrlPktStrategy cps({.rewrite_lanes = rewrite});
      const Twin t = twin(apps, seqs, cps, c);
      ctrlpkt::Replay start = t.cp_start;
      const Package &pa = apps.at(a);
      size_t cleared = 0;
      for (const ctrlpkt::Lane &l : cps.plan_of(pa).lanes)
        for (const auto &[tile, regs] : l.regs)
          for (const auto &[off, v] : regs) {
            const uint32_t app_value = launch::ctrlpkt::detail::app_value(pa, tile, off);
            if (app_value != v) {
              start.r.write(tile.addr(off), app_value);
              ++cleared;
            }
          }
      // A's muxes as its xclbin sets them: without the lanes' DMA fields.
      for (const ctrlpkt::Lane &l : cps.plan_of(pa).lanes)
        start.r.write(hw::address(l.col, 0, hw::kShimMux), launch::ctrlpkt::detail::app_mux(pa, l.col));
      const auto e = check(apps, cps, c, t, t.cp.ops, t.cp.packets, t.cp.packet_arg, &start);
      std::printf("  lanes lost (%zu registers), %s, rewrite_lanes %s: %s\n", cleared, c.name.c_str(),
                  rewrite ? "on" : "off", e.empty() ? "every packet arrives" : "packets go astray");
      if (rewrite) {
        CHECK(e.empty());
        print(e);
      } else {
        CHECK(!e.empty());
        print(e);
      }
    }
  }
  return done("test_ctrlpkt");
}
