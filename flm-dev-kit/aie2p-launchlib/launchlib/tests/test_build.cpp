//===- test_build.cpp - the kernels of build(), on a register model --*- C++ -*-===//
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
//   test_build PKG A B
//
// A and B: two apps of one floorplan (vadd, vmul). The test builds
//   reset, cold(A), warm(A, B), warm(B, A), warm(A, B, program_only), rerun(A)
// and runs their ops on a model of the registers (address -> value; Write32,
// MaskWrite32, BlockWrite). Checks:
//   * cold(A) leaves every register that aiecc's configure sequence of A writes
//     with aiecc's final value (core control, DMA channels, PM, locks, BDs,
//     routes, mux): the cold kernel configures what aiecc configures.
//   * cold(A) then warm(A, B) leaves the state of aiecc's configure of B; the
//     same with the program-only warm kernel.
//   * the parts each kind has (no reset in rerun, no routes in a warm kernel
//     between equal floorplans, ...), and the errors of the list.
// A and B of different floorplans (vadd, vmul, vminus of
// pm_runtime_load_singletile): warm(A, B, program_only) is refused; cold(A),
// then warm(A, B) and warm(B, A) two times each give aiecc's state of their app
// and leave no route, mux or channel Ctrl of the other app.
//
//===----------------------------------------------------------------------===//

#include "check.hpp"

#include "launch/launch.hpp"

#include <filesystem>
#include <map>
#include <set>
#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

using namespace launch;

using Model = std::map<uint32_t, uint32_t>;

static void run_ops(Model &m, const std::vector<txn::Op> &ops) {
  for (const txn::Op &op : ops)
    std::visit(txn::Overload{
                   [&](const txn::Write32 &o) { m[o.addr] = o.value; },
                   [&](const txn::MaskWrite32 &o) { m[o.addr] = (m[o.addr] & ~o.mask) | (o.value & o.mask); },
                   [&](const txn::BlockWrite &o) {
                     for (size_t i = 0; i < o.words.size(); ++i)
                       m[o.addr + 4 * (uint32_t)i] = o.words[i];
                   },
                   [](const auto &) {},
               },
               op);
}

static std::vector<txn::Op> ops_of(const Kernel &k) { return txn::parse(k.words).ops; }

/// The bits of each address that `ops` write (a mask write: its mask).
static Model written_bits(const std::vector<txn::Op> &ops) {
  Model bits;
  for (const txn::Op &op : ops)
    std::visit(txn::Overload{
                   [&](const txn::Write32 &o) { bits[o.addr] = 0xFFFFFFFFu; },
                   [&](const txn::MaskWrite32 &o) { bits[o.addr] |= o.mask; },
                   [&](const txn::BlockWrite &o) {
                     for (size_t i = 0; i < o.words.size(); ++i)
                       bits[o.addr + 4 * (uint32_t)i] = 0xFFFFFFFFu;
                   },
                   [](const auto &) {},
               },
               op);
  return bits;
}

/// aiecc's configure of `p` as a model, and the bits it writes.
static std::pair<Model, Model> aiecc_model(const Package &p) {
  const std::vector<txn::Op> ops = txn::parse(txn::words_of(read_bytes(p.dir + "/init.txn"))).ops;
  Model m;
  run_ops(m, ops);
  return {m, written_bits(ops)};
}

/// The lock drain of a kernel's reset part (reset.hpp step 5): the number of
/// core-tile locks it writes 63 and then 0, after every DMA channel write of the
/// reset part and before the first op of the next part. -1 if the order is wrong.
static int drained_locks(const Kernel &k) {
  const std::vector<txn::Op> ops = ops_of(k);
  size_t begin = ops.size(), end = ops.size();
  for (size_t p = 0; p < k.parts.size(); ++p)
    if (k.parts[p].name == "reset") {
      begin = k.parts[p].first;
      end = p + 1 < k.parts.size() ? k.parts[p + 1].first : ops.size();
    }
  std::map<uint32_t, std::vector<uint32_t>> values; // core lock -> values written, in order
  size_t last_channel = 0, first_lock = ops.size();
  for (size_t i = begin; i < end; ++i) {
    const uint32_t addr = txn::addr_of(ops[i]);
    const hw::Cls cls = hw::classify(addr);
    if (cls == hw::Cls::channel)
      last_channel = i;
    const auto *w = std::get_if<txn::Write32>(&ops[i]);
    if (cls == hw::Cls::lock && hw::locate(addr).kind == hw::Kind::core && w) {
      values[addr].push_back(w->value);
      first_lock = std::min(first_lock, i);
    }
  }
  if (values.empty())
    return 0;
  if (last_channel > first_lock)
    return -1;
  for (const auto &[addr, v] : values)
    if (v != std::vector<uint32_t>{hw::kLockMax, 0})
      return -1;
  return (int)values.size();
}

/// Registers of `want` (aiecc_model) whose value in `got` differs in the bits
/// aiecc's configure writes; prints the first few. A mask write leaves the other
/// bits of its word as they were: aiecc does not set them, so they are not
/// compared (an app's data word can share a byte it does not use with another
/// app's block write).
static int differences(const Model &got, const std::pair<Model, Model> &want, const char *what) {
  int bad = 0;
  for (const auto &[addr, value] : want.first) {
    auto it = got.find(addr);
    const uint32_t g = it == got.end() ? 0 : it->second, bits = want.second.at(addr);
    if ((g & bits) != (value & bits) && ++bad <= 3)
      std::fprintf(stderr, "  %s: %#010x is %#x, aiecc %#x (bits %#x)\n", what, addr, g, value, bits);
  }
  return bad;
}

/// Registers that aiecc's configure of `prev` writes and that of `app` does not,
/// of the classes that steer data (routes, mux, channel Ctrl), and that still
/// hold prev's value in `got` (a reset gives them their reset value; a register
/// that prev wrote with its reset value does not count). Prints the first few.
/// Not checked: BDs and locks. prev's stay, but a BD runs only when a channel
/// queue or a BD chain names it, and aiecc writes every BD and every lock that
/// app declares (a lock of init 0 too), so the load writes all those app uses.
static int left_behind(const Model &got, const Model &prev, const Model &app, const Model &reset_values,
                       const char *what) {
  int bad = 0;
  for (const auto &[addr, value] : prev) {
    if (app.count(addr))
      continue;
    const hw::Cls c = hw::classify(addr);
    if (c != hw::Cls::route && c != hw::Cls::mux && c != hw::Cls::channel)
      continue;
    // A Start_Queue write pushes a BD; it is not a setting (the channel reset empties the queue).
    if (c == hw::Cls::channel && detail::channel_ctrl_of(addr) != addr)
      continue;
    auto r = reset_values.find(addr);
    if (value == (r == reset_values.end() ? 0 : r->second))
      continue;
    auto it = got.find(addr);
    if (it != got.end() && it->second == value && ++bad <= 3)
      std::fprintf(stderr, "  %s: %#010x (%s) still has the previous app's %#x\n", what, addr, hw::cls_name(c), value);
  }
  return bad;
}

/// Two apps of different floorplans (pm_runtime_load_singletile: vadd, vmul,
/// vminus): a full switch either way configures what aiecc configures and
/// leaves no route, mux or channel Ctrl of the other app; a program-only switch
/// is refused.
static int different_floorplans(const std::string &pkg, const Package &pa, const Package &pb, const SequenceMap &seqs) {
  const std::string a = pa.name, b = pb.name;
  std::printf("  %s and %s: different floorplans (%s)\n", a.c_str(), b.c_str(), floorplan_difference(pa, pb).c_str());
  CHECK(throws([&] { build(ConfigList{pkg, Transport::txn, {warm(a, b, {.program_only = true})}}, seqs); }));
  const Result r = build(ConfigList{pkg, Transport::txn, {reset(), cold(a), warm(a, b), warm(b, a), cold(b)}}, seqs);
  for (const Kernel &k : r.kernels)
    std::printf("  %-22s %6zu ops %8zu B  (write32 %zu, maskwrite32 %zu, blockwrite %zu: %zu B, program %zu B)\n",
                k.name.c_str(), k.stats.ops, k.stats.bytes, k.stats.write32, k.stats.maskwrite32, k.stats.blockwrite,
                k.stats.block_bytes, k.stats.program_bytes);
  const auto aiecc_a = aiecc_model(pa), aiecc_b = aiecc_model(pb);
  const Model &want_a = aiecc_a.first, &want_b = aiecc_b.first;
  Model reset_values;
  run_ops(reset_values, ops_of(r.kernels[0]));
  // cold(A), warm(A, B), warm(B, A), warm(A, B): each step gives the state of its app.
  Model m;
  run_ops(m, ops_of(r.kernels[1]));
  CHECK(differences(m, aiecc_a, "cold(A)") == 0);
  for (int round = 0; round < 2; ++round) {
    run_ops(m, ops_of(r.kernels[2]));
    CHECK(differences(m, aiecc_b, "warm(A, B)") == 0);
    CHECK(left_behind(m, want_a, want_b, reset_values, "warm(A, B)") == 0);
    run_ops(m, ops_of(r.kernels[3]));
    CHECK(differences(m, aiecc_a, "warm(B, A)") == 0);
    CHECK(left_behind(m, want_b, want_a, reset_values, "warm(B, A)") == 0);
  }
  // cold(B) after A: the same state as warm(A, B) for every register of B.
  run_ops(m, ops_of(r.kernels[4]));
  CHECK(differences(m, aiecc_b, "cold(B) after A") == 0);
  CHECK(left_behind(m, want_a, want_b, reset_values, "cold(B) after A") == 0);
  return done("test_build");
}

int main(int argc, char **argv) {
  // The list and its errors (no package needed).
  CHECK(throws([] { validate(ConfigList{}); }));
  CHECK(throws([] { validate(ConfigList{"p", Transport::txn, {cold("a"), cold("a")}}); }));
  CHECK(throws([] { validate(ConfigList{"p", Transport::txn, {warm("", "a")}}); }));
  CHECK(throws([] { config_from_json(R"({"configs":[{"kind":"cold","app":"a","prev":"b"}]})"); }));
  CHECK(throws([] { config_from_json(R"({"transport":"ucdma","configs":[{"kind":"reset"}]})"); }));
  const ConfigList json_list = config_from_json(
      R"({"package_dir":"pkg","configs":[{"kind":"cold","app":"a"},{"kind":"warm","prev":"a","app":"b","program_only":true},{"kind":"rerun","app":"b","name":"again","sequence":"b2"}]})");
  CHECK(json_list.configs.size() == 3);
  CHECK(json_list.transport == Transport::ctrlpkt); // the default
  CHECK(json_list.configs[1].name == "a_then_b_pm");
  CHECK(json_list.configs[1].program_only());
  CHECK(json_list.configs[2].sequence_key() == "b2");
  CHECK(config_to_json(config_from_json(config_to_json(json_list))) == config_to_json(json_list));
  CHECK(cycle({"a", "b", "c"}).size() == 4);

  // The Ctrl register of each chain-start write; 4 bytes before the first
  // MM2S Ctrl is the last S2MM Start_Queue (this was a bug: it reset a queue).
  using detail::channel_ctrl_of;
  for (int row : {1, 2}) {
    const hw::Kind kind = hw::kind_of_row(row);
    for (hw::Dir d : {hw::Dir::s2mm, hw::Dir::mm2s})
      for (uint32_t i = 0; i < hw::channels(kind, d).count; ++i) {
        const uint32_t ctrl = hw::address(3, row, hw::channels(kind, d).at(i));
        CHECK(channel_ctrl_of(ctrl) == ctrl);
        CHECK(channel_ctrl_of(ctrl + hw::kQueueFromCtrl) == ctrl);
      }
  }
  CHECK(channel_ctrl_of(hw::address(0, 2, 0x1DE10)) == hw::address(0, 2, 0x1DE10));
  CHECK(channel_ctrl_of(hw::address(0, 1, 0xA0630)) == hw::address(0, 1, 0xA0630));

  if (argc < 4)
    return done("test_build");
  const std::string pkg = argv[1], a = argv[2], b = argv[3];
  const Package pa = load_package(pkg, a), pb = load_package(pkg, b);
  Package other = pb;
  other.tiles.begin()->second.routes[0x3F000] ^= 1;
  CHECK(!floorplan_difference(pa, other).empty());

  // The sequences: one write per app, so that a sequence is easy to find.
  SequenceMap seqs;
  for (const std::string &app : {a, b})
    seqs[app] = [](Rt &rt) { rt.npu_write32(hw::address(0, 0, 0x1D000), 7); };
  if (!floorplan_difference(pa, pb).empty())
    return different_floorplans(pkg, pa, pb, seqs);
  ConfigList list{pkg, Transport::txn,
                  {reset(), cold(a), warm(a, b), warm(b, a), warm(a, b, {.program_only = true}), rerun(a)}};
  const Result r = build(list, seqs);
  CHECK(r.kernels.size() == 6);
  CHECK(r.base_xclbin == pa.xclbin);
  CHECK(r.base_app == a);
  // A list that starts with a warm kernel: the base is its prev (the first app it
  // names), the app whose xclbin the hw_context is opened on (Session's state).
  const Result wf = build(ConfigList{pkg, Transport::txn, {warm(a, b), rerun(b)}}, seqs);
  CHECK(wf.base_app == a && wf.base_xclbin == pa.xclbin);
  for (const Kernel &k : r.kernels)
    std::printf("  %-22s %6zu ops %8zu B  (write32 %zu, maskwrite32 %zu, blockwrite %zu: %zu B, program %zu B)\n",
                k.name.c_str(), k.stats.ops, k.stats.bytes, k.stats.write32, k.stats.maskwrite32,
                k.stats.blockwrite, k.stats.block_bytes, k.stats.program_bytes);

  // aiecc's configure sequences, as models.
  const auto want_a = aiecc_model(pa), want_b = aiecc_model(pb);

  // reset: no program, no sequence.
  CHECK(r.kernels[0].stats.program_bytes == 0 && r.kernels[0].stats.ops > 0);
  // cold(A) configures what aiecc configures.
  Model m;
  run_ops(m, ops_of(r.kernels[1]));
  CHECK(differences(m, want_a, "cold(A)") == 0);
  // then warm(A, B): the state of B.
  Model warm_full = m;
  run_ops(warm_full, ops_of(r.kernels[2]));
  CHECK(differences(warm_full, want_b, "cold(A), warm(A, B)") == 0);
  // then warm(B, A): the state of A again.
  run_ops(warm_full, ops_of(r.kernels[3]));
  CHECK(differences(warm_full, want_a, "..., warm(B, A)") == 0);
  // the program-only switch from cold(A): the state of B too.
  Model warm_pm = m;
  run_ops(warm_pm, ops_of(r.kernels[4]));
  CHECK(differences(warm_pm, want_b, "cold(A), warm(A, B, program_only)") == 0);

  // The parts of each kind.
  const KernelStats &full = r.kernels[2].stats, &pm = r.kernels[4].stats, &again = r.kernels[5].stats;
  CHECK(pm.program_bytes == full.program_bytes);
  CHECK(pm.ops < full.ops);
  CHECK(again.ops == 1); // the sequence only
  CHECK(r.kernels[1].stats.program_bytes > 0);
  // The lock drain: every core-tile lock in reset and cold; in a warm kernel the
  // locks prev's cores acquire (the census), or without a census every lock
  // aiecc writes in prev's core tiles; none without a reset part.
  const int all_core_locks = hw::kCols * (hw::kRows - hw::kFirstCoreRow) * 16;
  auto prev_core_locks = [](const Package &p) {
    if (p.census)
      return (int)p.waits.size();
    int n = 0;
    for (const auto &[tile, init] : p.tiles)
      if (tile.kind() == hw::Kind::core)
        n += (int)init.locks.size();
    return n;
  };
  CHECK(prev_core_locks(pa) > 0);
  CHECK(drained_locks(r.kernels[0]) == all_core_locks);
  CHECK(drained_locks(r.kernels[1]) == all_core_locks);
  CHECK(drained_locks(r.kernels[2]) == prev_core_locks(pa));
  CHECK(drained_locks(r.kernels[3]) == prev_core_locks(pb));
  CHECK(drained_locks(r.kernels[4]) == 0);
  CHECK(drained_locks(r.kernels[5]) == 0);
  std::printf("  census: %s, %d locks drained by warm(%s, %s)\n", pa.census ? "yes" : "no",
              drained_locks(r.kernels[2]), pa.name.c_str(), pb.name.c_str());
  // The load writes every lock the app declares or its cores acquire; a lock it
  // leaves out is one aiecc only zeroes.
  for (const Package *p : {&pa, &pb}) {
    std::set<uint32_t> written;
    for (const txn::Op &op : ops_of(r.kernels[p == &pa ? 1 : 2])) // (cold(A) loads A, warm(A, B) loads B)
      if (const auto *w = std::get_if<txn::Write32>(&op); w && hw::classify(w->addr) == hw::Cls::lock)
        written.insert(w->addr);
    for (uint32_t a : p->declared_locks)
      CHECK(written.count(a));
    for (uint32_t a : p->waits)
      CHECK(written.count(a));
    for (const auto &[tile, init] : p->tiles)
      for (const auto &[off, value] : init.locks)
        if (!written.count(tile.addr(off)))
          CHECK(value == 0 && !p->needs_lock(tile, off));
    if (!p->declared_locks.empty()) // (a package with declare lines: every named lock is declared)
      for (const auto &[n, k] : p->lock_names)
        CHECK(p->declared_locks.count(k.tile.addr(hw::locks(k.tile.kind()).at(k.id))));
  }
  // Equal floorplans: a warm kernel writes no route.
  bool route_in_warm = false;
  for (const txn::Op &op : ops_of(r.kernels[2]))
    if (!std::holds_alternative<txn::Sync>(op) && hw::classify(txn::addr_of(op)) == hw::Cls::route)
      route_in_warm = true;
  CHECK(!route_in_warm);
  // breakdown: the program-only kernel has a program part (core control and
  // program) and the sequence; the full one a reset and a load part.
  std::set<std::string> pm_parts, full_parts;
  for (const BreakdownRow &row : breakdown(r.kernels[4]))
    pm_parts.insert(row.part + ":" + row.what);
  for (const BreakdownRow &row : breakdown(r.kernels[2]))
    full_parts.insert(row.part);
  // (the test's sequence is one write to shim BD 0: class bd)
  CHECK((pm_parts == std::set<std::string>{"program:core_control", "program:program", "sequence:bd"}));
  CHECK((full_parts == std::set<std::string>{"reset", "load", "sequence"}));

  // program_only between different floorplans is refused: a copy of B whose
  // init has one route changed.
  namespace fs = std::filesystem;
  const fs::path tmp = fs::temp_directory_path() / ("launchlib_test_" + std::to_string(::getpid()));
  fs::create_directories(tmp);
  fs::copy(pa.dir, tmp / a, fs::copy_options::recursive);
  fs::copy(pb.dir, tmp / "changed", fs::copy_options::recursive);
  txn::Stream init = txn::parse(txn::words_of(read_bytes(pb.dir + "/init.txn")));
  for (txn::Op &op : init.ops)
    if (auto *w = std::get_if<txn::Write32>(&op); w && hw::classify(w->addr) == hw::Cls::route) {
      w->value ^= 1;
      break;
    }
  const std::vector<uint32_t> words = txn::encode(init);
  std::ofstream(tmp / "changed" / "init.txn", std::ios::binary)
      .write(reinterpret_cast<const char *>(words.data()), (std::streamsize)(words.size() * 4));
  seqs["changed"] = seqs[b];
  CHECK(throws([&] {
    build(ConfigList{tmp.string(), Transport::txn, {warm(a, "changed", {.program_only = true})}}, seqs);
  }));
  CHECK(!throws([&] { build(ConfigList{tmp.string(), Transport::txn, {warm(a, "changed")}}, seqs); }));
  fs::remove_all(tmp);
  return done("test_build");
}
