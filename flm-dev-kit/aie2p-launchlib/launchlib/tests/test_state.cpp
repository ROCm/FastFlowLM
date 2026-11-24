//===- test_state.cpp - APP@STATE: an app loaded with one op set up --*- C++ -*-===//
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
//   test_state PKG APP
//
// States of a copy of PKG/APP in a temporary folder (formats/package.hpp), made
// from its init.txn: what a state may write loads, and cold, warm and the warm
// back carry it, with both transports (control packets with their check); what
// it may not write stops the load with a message that names it.
//
//===----------------------------------------------------------------------===//

#include "check.hpp"

#include "launch/launch.hpp"

#include <filesystem>
#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

using namespace launch;
namespace fs = std::filesystem;

template <class F> bool says(F f, std::initializer_list<const char *> words) {
  std::string m;
  try {
    f();
  } catch (const std::exception &e) {
    m = e.what();
  }
  bool ok = !m.empty();
  for (const char *w : words)
    ok &= m.find(w) != std::string::npos;
  if (!ok)
    std::fprintf(stderr, "  message: '%s'\n", m.c_str());
  return ok;
}

static void write_state(const fs::path &p, const std::vector<txn::Op> &ops) {
  txn::Stream s;
  s.ops = ops;
  const std::vector<uint32_t> w = txn::encode(s);
  std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char *>(w.data()), (std::streamsize)(w.size() * 4));
}

/// The value the kernel leaves at `addr` (its writes, in order), if it writes it.
static std::optional<uint32_t> left_at(const Kernel &k, uint32_t addr) {
  const auto v = detail::values_of(txn::parse(k.words).ops);
  auto it = v.find(addr);
  return it == v.end() ? std::nullopt : std::optional<uint32_t>(it->second);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: test_state PKG APP\n");
    return 2;
  }
  const std::string pkg = argv[1], app = argv[2];
  const fs::path tmp = fs::temp_directory_path() / ("launchlib_state_" + std::to_string(::getpid()));
  fs::remove_all(tmp);
  fs::create_directories(tmp);
  fs::copy(fs::path(pkg) / app, tmp / app, fs::copy_options::recursive);
  const txn::Stream init = txn::parse(txn::words_of(read_bytes((tmp / app / "init.txn").string())));
  const auto base = detail::values_of(init.ops);
  const Package plain = load_package(tmp.string(), app);
  auto load = [&](const std::string &state) { return load_package(tmp.string(), app + "@" + state); };
  auto state = [&](const std::string &name, const std::vector<txn::Op> &ops) {
    write_state(tmp / app / ("state_" + name + ".txn"), ops);
  };

  // A register the app leaves at 0: the last BD of the first memory tile it does not set to another value
  // (not a shim BD: the control-packet lanes take the free ones).
  std::optional<uint32_t> free_reg;
  for (int col = 0; col < hw::kCols && !free_reg; ++col) {
    const uint32_t a = hw::address(col, 1, hw::bds(hw::Kind::mem).at(hw::bds(hw::Kind::mem).count - 1));
    if (!base.count(a) || base.at(a) == 0)
      free_reg = a;
  }
  CHECK(free_reg.has_value());
  // A register and a lock the app sets to a value other than 0.
  std::optional<std::pair<uint32_t, uint32_t>> set_reg, set_lock;
  for (const auto &[a, v] : base) {
    if (v == 0)
      continue;
    const hw::Cls c = hw::classify(a);
    if (c == hw::Cls::lock && !set_lock)
      set_lock = {{a, v}};
    if ((c == hw::Cls::bd || c == hw::Cls::route) && !set_reg)
      set_reg = {{a, v}};
  }
  CHECK(set_reg.has_value());

  // What a state may write: a register the app leaves at 0, the app's own value again, a lock value, and a
  // channel reset (dropped, as init.txn's).
  const uint32_t ctrl = hw::address(0, 0, hw::channels(hw::Kind::shim, hw::Dir::mm2s).at(0));
  std::vector<txn::Op> good = {txn::Write32{*free_reg, 0x1234}, txn::MaskWrite32{ctrl, hw::kDmaReset, hw::kDmaReset}};
  if (set_reg)
    good.push_back(txn::Write32{set_reg->first, set_reg->second});
  if (set_lock)
    good.push_back(txn::Write32{set_lock->first, set_lock->second == 1 ? 2u : 1u});
  state("s", good);
  {
    const Package p = load("s");
    CHECK(p.name == app + "@s" && p.state == "s" && p.dir == plain.dir && p.xclbin == plain.xclbin);
    CHECK(p.cores() == plain.cores());
  }
  for (Transport t : {Transport::txn, Transport::ctrlpkt}) {
    const auto none = [](Rt &) {};
    const std::string s = app + "@s";
    ConfigList list{tmp.string(), t, {cold(app), cold(s), warm(app, s), warm(s, app), rerun(s)}};
    Result r;
    try {
      r = build(list, {{app, none}, {s, none}});
    } catch (const std::exception &e) {
      std::fprintf(stderr, "  %s: %s\n", transport_name(t), e.what());
      CHECK(false);
    }
    if (r.kernels.size() != 5)
      continue;
    // with control packets the BD may go by packet: then only the TXN twin (the check) shows it
    if (t == Transport::txn) {
      CHECK(left_at(r.kernels[0], *free_reg).value_or(0) != 0x1234);
      CHECK(left_at(r.kernels[1], *free_reg) == 0x1234u);
      CHECK(left_at(r.kernels[2], *free_reg) == 0x1234u);
      CHECK(left_at(r.kernels[3], *free_reg).value_or(0) != 0x1234);
      if (set_lock)
        CHECK(left_at(r.kernels[1], set_lock->first) == (set_lock->second == 1 ? 2u : 1u));
    }
    CHECK(r.base_app == app);
  }

  // Names.
  CHECK(says([&] { load_package(tmp.string(), app + "@"); }, {"APP@STATE"}));
  CHECK(says([&] { load_package(tmp.string(), "@s"); }, {"APP@STATE"}));
  CHECK(says([&] { load_package(tmp.string(), app + "@a@b"); }, {"APP@STATE"}));
  CHECK(says([&] { load("missing"); }, {"state_missing.txn", "--state"}));
  // What a state may not write.
  const Tile core = plain.cores().empty() ? Tile{0, 2} : plain.cores().front();
  state("pm", {txn::Write32{core.addr(hw::kPm), 1}});
  CHECK(says([&] { load("pm"); }, {"state_pm.txn", "op 0", "cannot change the program"}));
  if (set_reg) {
    state("over", {txn::Write32{*free_reg, 1}, txn::Write32{set_reg->first, set_reg->second + 1}});
    CHECK(says([&] { load("over"); }, {"state_over.txn", "op 1", "where init.txn sets"}));
    state("over_block", {txn::BlockWrite{set_reg->first, {set_reg->second ^ 1}}});
    CHECK(says([&] { load("over_block"); }, {"state_over_block.txn", "where init.txn sets"}));
  }
  std::optional<Tile> not_core;
  for (int col = 0; col < hw::kCols && !not_core; ++col)
    for (int row = 2; row < hw::kRows && !not_core; ++row)
      if (!plain.tiles.count({col, row}) || !plain.tiles.at({col, row}).core)
        not_core = Tile{col, row};
  if (not_core) {
    state("core", {txn::Write32{not_core->addr(hw::kCoreControl), 1}});
    CHECK(says([&] { load("core"); }, {"state_core.txn", "a core that"}));
  }
  state("nocls", {txn::Write32{hw::address(0, 2, 0x32004), 1}});
  CHECK(says([&] { load("nocls"); }, {"state_nocls.txn", "no register class"}));
  state("wait", {txn::Sync{0, 0, 0, 0}});
  CHECK(says([&] { load("wait"); }, {"state_wait.txn", "only writes"}));

  fs::remove_all(tmp);
  return done("test_state");
}
