//===- test.cpp - host program: build the kernels at run time, launch, check ===//
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
// The AIE2P counterpart of aie4-launchlib's host_test, with three apps of three
// different floorplans (vadd, vmul, vminus):
//
//   1. the list of configurations, one kernel per entry: built in C++ at run time
//      (make_list, from --apps), or read from a JSON file (--config)
//   2. launch::Session(list, sequences(), flow): the kernels, built in memory, and
//      one hw_context on the base xclbin; per flow, what XRT runs:
//        xclbin  kernel "MLIR_AIE", the instruction stream in a BO (opcode 3)
//        elf     the instruction stream as an ELF (txn::to_elf), xrt::module, xrt::ext::kernel
//   3. per kernel its own buffers (Design)
//   4. Session::run: the launches of a row in runlists; check the outputs (row)
//
// Steps 2 to 4 run once per flow. Each row of the table is one Session::run
// (one runlist; in the ELF flow several when they need more than its budget),
// made from the kinds in the list:
//
//   reset            reset x ITERS
//   cold             cold_<app> x ITERS                         (each cold entry)
//   cold + rerun     cold_<app>, then rerun_<app> x ITERS        (each rerun entry)
//   cycle            cold_<first>, then the warm kernels x ITERS, round-robin: every
//                    launch switches to the next app (AIE4: the chain)
//   ping-pong        cold_<A>, then warm(A, B), warm(B, A) x ITERS: two apps in turn.
//                    AIE2P has no second PM bank, so each launch writes its program
//                    (AIE4's ping-pong writes none)
//
// A row runs once (not timed), then once more (timed). The Session's report must
// be "ok": with another hw_context on the NPU it runs warm and rerun launches as
// cold kernels (the row says how many), and the results must still be right.
//
//   test.exe --package DIR [--apps vadd,vmul,vminus | --config FILE.json] [--iters N]
//            [--flow xclbin|elf] [--transport txn|ctrlpkt] [--build-only] [--exclusive]
//
//   --config FILE   read the list from FILE (three_apps.json = the list of make_list for
//                   the three apps). --package replaces its package_dir.
//   --transport T   how the kernels load the apps (launch/config.hpp; default ctrlpkt, or the
//                   file's)
//   --build-only    build the kernels, print the ops of each, stop (no device)
//   --exclusive     the NPU is ours alone: without the sharing guard (Windows) warm and
//                   rerun launches run as asked, not as cold
//
// A runlist that fails or does not finish in 10 s fails its row (the Session opens
// a new hw_context; the next rows go on).
//
// The runtime sequences are in sequence.hpp; the inputs and golden result of each
// app are in designs/designs.hpp.
//
//===----------------------------------------------------------------------===//

#include "designs/designs.hpp"
#include "sequence.hpp"

#include "launch/launch.hpp"
#include "launch/xrt/session.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>

using namespace hosttest;

namespace {

/// 1a. The list, built in C++: reset; cold_<app> and rerun_<app> for each app;
/// the warm kernels of cycle(apps) (each app to the next, the last back to the
/// first); then, for the ping-pong of the first two apps A and B, warm(B, A):
/// warm(A, B) is already in the cycle. With two apps the cycle is the ping-pong.
launch::ConfigList make_list(const std::string &package, const std::vector<std::string> &apps) {
  launch::ConfigList list{package, launch::Transport::ctrlpkt, {launch::reset()}};
  for (const std::string &app : apps) {
    list.configs.push_back(launch::cold(app));
    list.configs.push_back(launch::rerun(app)); // must follow a kernel of `app`
  }
  if (apps.size() >= 2) {
    const std::vector<launch::Config> cyc = launch::cycle(apps);
    list.configs.insert(list.configs.end(), cyc.begin() + 1, cyc.end()); // its cold is already in
    if (apps.size() > 2)
      list.configs.push_back(launch::warm(apps[1], apps[0]));
  }
  return list;
}

/// 1b. The list, read from a JSON file (the format is in launch/config.hpp).
launch::ConfigList read_list(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("cannot read " + path);
  std::stringstream text;
  text << f.rdbuf();
  return launch::config_from_json(text.str());
}

/// The Session of one flow, and the buffers of each kernel.
struct Flow {
  const char *name;
  launch::Session session;
  std::map<std::string, std::unique_ptr<Design>> designs; ///< kernel name -> its buffers (not for reset)

  Flow(const char *flow, const launch::ConfigList &list, bool exclusive)
      : name(flow), session(list, sequences(), {.flow = std::string(flow) == "elf" ? launch::Flow::elf
                                                                                   : launch::Flow::xclbin,
                                                .exclusive = exclusive}) {
    const MakeBuffer mk = [this](size_t bytes, unsigned arg) { return session.buffer(bytes, arg); };
    for (const launch::Config &c : list.configs)
      if (c.kind() != launch::Config::Kind::reset)
        designs[c.name] = make_design(c.app(), mk);
  }
};

/// `first`, then `iters` launches of `rr`, round-robin. `iters` is rounded up to a
/// multiple of rr.size(), so that each kernel of `rr` runs as often as the others.
std::vector<std::string> launches(std::vector<std::string> first, const std::vector<std::string> &rr, int iters) {
  const int n = (int)rr.size();
  for (int i = 0; i < (iters + n - 1) / n * n; ++i)
    first.push_back(rr[i % n]);
  return first;
}

/// 4. One row: one Session::run of the kernels `names`, in that order. The outputs are set
/// to NaN before and checked after. Print the mean time of a launch.
bool row(Flow &f, const std::string &label, const std::vector<std::string> &names) {
  const std::set<std::string> used(names.begin(), names.end());
  for (const std::string &n : used)
    if (auto d = f.designs.find(n); d != f.designs.end()) {
      d->second->clear_outputs();
      for (xrt::bo *bo : d->second->inputs())
        bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
      for (xrt::bo *bo : d->second->outputs())
        bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
  std::vector<launch::Launch> ls;
  for (const std::string &n : names) {
    auto d = f.designs.find(n);
    ls.push_back({n, d == f.designs.end() ? std::vector<xrt::bo>{} : d->second->args()});
  }
  launch::RunReport rep = f.session.run(ls); // the first time: not timed
  const auto t0 = std::chrono::steady_clock::now();
  if (rep.status == launch::RunReport::Status::ok)
    rep = f.session.run(ls);
  const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();

  bool ok = rep.status == launch::RunReport::Status::ok;
  if (!ok)
    printf("  %s: %s (%s)\n", label.c_str(), launch::status_name(rep.status), rep.message.c_str());
  for (const std::string &n : used)
    if (auto d = f.designs.find(n); d != f.designs.end() && ok) {
      for (xrt::bo *bo : d->second->outputs())
        bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      ok = d->second->check(n) && ok;
    }
  std::string note;
  if (rep.substituted)
    note = "  (" + std::to_string(rep.substituted) + " as cold: " +
           (f.session.watching() ? "another hw_context is on the NPU)" : "no sharing guard; --exclusive: as asked)");
  printf("%-7s %-56s %6zu %9.1f  %s%s\n", f.name, label.c_str(), names.size(), us / names.size(), ok ? "PASS" : "FAIL",
         note.c_str());
  fflush(stdout);
  return ok;
}

/// All the rows of one flow, from the kinds in the list.
bool run_rows(Flow &f, const launch::ConfigList &list, int iters) {
  using Kind = launch::Config::Kind;
  bool ok = true;
  std::map<std::string, std::string> cold_of; // app -> name of its cold kernel
  std::vector<const launch::Config *> warms;
  for (const launch::Config &c : list.configs)
    if (c.kind() == Kind::reset || c.kind() == Kind::cold) {
      ok = row(f, c.name, launches({}, {c.name}, iters)) && ok;
      if (c.kind() == Kind::cold)
        cold_of.emplace(c.app(), c.name);
    } else if (c.kind() == Kind::warm)
      warms.push_back(&c);
  for (const launch::Config &c : list.configs)
    if (c.kind() == Kind::rerun) { // needs the cold kernel of its app first
      auto cold = cold_of.find(c.app());
      if (cold == cold_of.end())
        std::cout << "skip " << c.name << ": the list has no cold kernel of " << c.app() << "\n";
      else
        ok = row(f, cold->second + ", " + c.name, launches({cold->second}, {c.name}, iters)) && ok;
    }
  if (warms.empty())
    return ok;
  // A loop of warm kernels starts after the cold kernel of its first prev.
  auto loop_row = [&](const std::vector<std::string> &names, const std::string &from, const std::string &what) {
    auto cold = cold_of.find(from);
    if (cold == cold_of.end()) {
      std::cout << "skip the " << what << " row: the list has no cold kernel of " << from << "\n";
      return;
    }
    ok = row(f, cold->second + ", " + std::to_string(names.size()) + " " + what, launches({cold->second}, names, iters)) &&
         ok;
  };
  // The cycle: the warm kernels in list order, each after the one before it,
  // until one comes back to the first one's prev.
  const std::string start = warms[0]->prev();
  std::string at = start;
  std::vector<std::string> cyc;
  size_t i = 0;
  while (i < warms.size() && warms[i]->prev() == at) {
    cyc.push_back(warms[i]->name);
    at = warms[i++]->app();
    if (at == start)
      break;
  }
  if (at == start) {
    loop_row(cyc, start, "warm");
  } else {
    std::cout << "skip the cycle: the warm kernels do not come back to " << start << " in list order\n";
    i = 0;
  }
  // The ping-pong: each warm kernel after the cycle, B -> A, with the warm A -> B of the list.
  for (; i < warms.size(); ++i) {
    const launch::Config *back = warms[i];
    auto there = std::find_if(warms.begin(), warms.end(), [&](const launch::Config *c) {
      return c->prev() == back->app() && c->app() == back->prev();
    });
    if (there == warms.end())
      std::cout << "skip " << back->name << ": the list has no warm kernel " << back->app() << " -> " << back->prev()
                << "\n";
    else
      loop_row({(*there)->name, back->name}, back->app(), "ping-pong");
  }
  return ok;
}

} // namespace

int main(int argc, const char *argv[]) {
  const char *usage = "usage: test.exe --package DIR [--apps a,b,c | --config FILE.json] [--iters N] "
                      "[--flow xclbin|elf] [--transport txn|ctrlpkt] [--build-only] [--exclusive]\n";
  std::string package, apps_arg = "vadd,vmul,vminus", config, only, transport;
  int iters = 100;
  bool build_only = false, exclusive = false;
  for (int i = 1; i < argc; i++) {
    const std::string s = argv[i];
    const bool has_value = i + 1 < argc;
    if (s == "--package" && has_value) package = argv[++i];
    else if (s == "--apps" && has_value) apps_arg = argv[++i];
    else if (s == "--config" && has_value) config = argv[++i];
    else if (s == "--iters" && has_value) iters = std::stoi(argv[++i]);
    else if (s == "--flow" && has_value) only = argv[++i];
    else if (s == "--transport" && has_value) transport = argv[++i];
    else if (s == "--build-only") build_only = true;
    else if (s == "--exclusive") exclusive = true;
    else return std::cerr << usage, 2;
  }
  if ((package.empty() && config.empty()) || iters < 1)
    return std::cerr << usage, 2;

  try {
    launch::ConfigList list;
    if (!config.empty()) { // 1b. from the JSON file
      list = read_list(config);
      if (!package.empty())
        list.package_dir = package;
    } else { // 1a. built at run time
      std::vector<std::string> apps;
      std::stringstream ss(apps_arg);
      for (std::string a; std::getline(ss, a, ',');)
        apps.push_back(a);
      list = make_list(package, apps);
    }
    if (!transport.empty())
      list.transport = launch::transport_from_name(transport);
    printf("[list] %zu configurations, %s; %u columns x %u cores\n", list.configs.size(),
           config.empty() ? ("built at run time from --apps " + apps_arg).c_str() : ("read from " + config).c_str(),
           kCols, TILES);
    const launch::Result r = launch::build(list, sequences()); // 2. the kernels, in memory
    printf("[build] %zu kernels in %.2f ms, transport %s\n", r.kernels.size(), r.build_ms,
           launch::transport_name(list.transport));
    for (const launch::Kernel &k : r.kernels)
      printf("  %-28s %6zu ops %7zu B (program %6zu B, block writes %3zu, write32 %4zu, maskwrite32 %3zu, packets %5zu B)\n",
             k.name.c_str(), k.stats.ops, k.stats.bytes, k.stats.program_bytes, k.stats.blockwrite, k.stats.write32,
             k.stats.maskwrite32, 4 * k.packets.size());
    if (build_only)
      return 0;

    bool ok = true;
    for (const char *flow : {"xclbin", "elf"}) {
      if (!only.empty() && only != flow)
        continue;
      Flow f(flow, list, exclusive); // one hw_context per flow, freed before the next one
      printf("%-7s %-56s %6s %9s  %s   (sharing guard %s%s)\n", "", "runlist", "n", "mean(us)", "check",
             f.session.watching() ? "on" : "OFF: ", f.session.watch_why().c_str());
      ok = run_rows(f, list, iters) && ok;
    }
    return ok ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "test.exe: " << e.what() << "\n";
    return 2;
  }
}
