//===- stress.cpp - stress and corner-case tests of the launch kernels -*- C++ -*-===//
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
//   stress.exe --package DIR --mode MODE [--seed S] [--launches L] [--max-runlist R]
//              [--transport txn|ctrlpkt] [--flow elf|xclbin] [--hang-risk]
//
// The kernels: for each length (1, 2, 3, 4, 8 blocks per core) cold, warm full,
// warm program-only, self-switch and rerun of both apps, and one reset kernel.
// Every launch writes its own output buffer; every output is checked against
// the CPU, and the words after the launch's length must stay untouched.
//
// MODE
//   random   S1: a random walk over the kernels, in runlists of random length
//            (1 .. R). The walk knows what is in the array and where the
//            ping/pong DMA chains are, so it only picks legal kernels:
//              rerun(app)              the app in the array
//              warm(app, other)        full switch, any time
//              warm(app, app)          full switch to itself
//              warm(app, x, pm)        program only: only at ping (even blocks
//                                      since the last DMA reset)
//              cold(x), reset + cold   any time
//   lengths  S2: runlists of 1, 23, 24, 25, 48, 200 and 1000 launches (XRT
//            cuts a runlist into chains of 24 runs).
//   parity   S3: odd lengths, then a full switch (must pass); odd lengths, then
//            a program-only switch (the chains are at pong: expected to give
//            wrong data, and expected not to hang).
//   prev     S4: a warm kernel whose prev is wrong (vmul -> vadd launched
//            while vadd is in the array).
//   ctx      S5: after warm(vadd, vmul) in context A, run vmul.xclbin in context
//            B, then rerun(vmul) in A: is the switched state still there, and
//            which program does A run? The same with another process using the
//            NPU in between. S5b: another process loops on the NPU while this
//            one runs runlists of 100 switches (XRT cuts them into chains of 24).
//   special  S8: inputs with +-0, +-inf, NaN, denormals and the largest values.
//   big      S6: the largest length (1024 blocks per core: 16 MB per buffer on
//            8 x 4), every kind.
//   all      lengths, parity, prev, special, big, then random.
//   latency  P4: one-launch runlists of each kind, timed one by one (--count
//            per kind): mean, p50, p90, p99, p99.9, max.
//   churn    P7: --count times: build, open a hw_context, 100 launches, close.
//   session  S9: the random walk through launch::Session, which runs a rerun or
//            program-only switch as the cold kernel when the array state is
//            unknown (another context, or more than max_gap since its last
//            runlist); --no-guard: max_gap 0. Even --lengths only. Launches the
//            Session reports ok must be right.
//
// Options of the random walk: --lengths B,B,... (blocks per core it picks from;
// default 1,2,3,4,8), --idle-us U (sleep U us before each runlist; wrong
// launches are counted by the time since the previous runlist), --robust (only kernels that do not depend on the array's
// state: full switches, cold, reset; safe while another process uses the NPU).
//
// --transport (default ctrlpkt): how the kernels load (launch::Transport); the
// children of ctx run the same. ctx, churn and --robust (other hw_contexts on
// the NPU) default to txn: with ctrlpkt a kernel whose packets lose their route
// waits on a Sync that never ends, and that has not been tested. With
// --transport ctrlpkt they also need --hang-risk (Linux only: there a hang is a
// timeout).
//
// --flow (default elf; include/flow.hpp): an ELF per kernel or the xclbin's
// kernel with instruction BOs. ELF runlists end where launch::Session's do
// (24 MB of instructions, 1,024 runs) and the next one starts.
//
// Exit 0 when every check a mode expects to pass passes.
//
//===----------------------------------------------------------------------===//

#include "bf16.hpp"
#include "elementwise_sequence.hpp"
#include "flow.hpp"
#include "spawn.hpp"

#include "launch/launch.hpp"
#include "launch/xrt/packets.hpp"
#include "launch/xrt/session.hpp"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr unsigned kOpcode = 3;
const std::vector<int> kBlocks = {1, 2, 3, 4, 8}; // blocks per core per launch
constexpr int kBigBlocks = 1024;                   // the "big" length
constexpr int kInputs = 4;                         // input buffers to pick from
constexpr uint16_t kPoison = 0xFFFF;               // a NaN no add or multiply makes

enum class App { none = -1, vadd = 0, vmul = 1 };
const char *app_name(App a) { return a == App::vadd ? "vadd" : a == App::vmul ? "vmul" : "none"; }
App other(App a) { return a == App::vadd ? App::vmul : App::vadd; }

/// The kernels of one length.
struct LengthKernels {
  int blocks;
  // index into Result::kernels, per (from app, to app); -1 = none
  int cold[2], rerun[2], warm[2][2], pm[2][2];
};

/// One launch of a runlist, to check afterwards.
struct Launch {
  int kernel;
  App app;        ///< the program that runs (none for reset)
  int elements;   ///< 0 for reset
  int lhs, rhs;   ///< input buffers
  int out;        ///< output buffer
  std::string what;
};

struct Stats {
  std::map<std::string, long> kinds;
  long launches = 0, runlists = 0, checked = 0, bad = 0, overwrites = 0;
};

/// What a Stress object builds and how its walk picks.
struct Options {
  std::vector<int> lengths = kBlocks; ///< blocks per core the random walk picks from
  bool big = false;                   ///< also the kBigBlocks kernels (mode big)
  bool robust = false;                ///< walk: only kernels that do not depend on the array's state
  bool quiet = false;
  bool preempt = false;               ///< every sequence starts with a preemption point
  launch::Transport transport = launch::Transport::ctrlpkt;
  bool elf = true;                    ///< --flow elf (else xclbin)
  bool session = false;               ///< run through launch::Session (mode session)
  bool guard = true;                  ///< Session: its max_gap rule (--no-guard: max_gap 0)
  int idle_us = 0;                    ///< walk: sleep before each runlist (--idle-us)
};

/// Runlists by the time since the previous one ended.
struct GapBucket {
  long runlists = 0;
  long dep_start = 0;  ///< first launch a rerun or program-only switch, run as asked
  long wrong = 0;      ///< wrong launches
  long wrong_first = 0; ///< ... of which the first launch of a dep_start runlist
};
constexpr double kGapEdges[] = {2000, 10000, 100000}; // us
const char *const kGapNames[] = {"< 2 ms", "2-10 ms", "10-100 ms", ">= 100 ms"};

class Stress {
public:
  Stress(const std::string &package, unsigned seed, Options opts = {})
      : package_(package), rng_(seed), opts_(std::move(opts)) {
    build();
    open();
  }

  // --- the kernels ---------------------------------------------------------------

  void build() {
    launch::ConfigList list{package_, opts_.transport, {launch::reset()}};
    launch::SequenceMap seqs;
    auto add = [&](launch::Config c, const std::string &seq) {
      c.name += "@" + seq;
      c.sequence = seq;
      list.configs.push_back(c);
      return (int)list.configs.size() - 1;
    };
    // The walk's lengths, 1 and 2 (the fixed cases use them), and the big one
    // only when asked: every output buffer has the size of the largest.
    std::set<int> all_blocks(opts_.lengths.begin(), opts_.lengths.end());
    all_blocks.insert({1, 2});
    if (opts_.big)
      all_blocks.insert(kBigBlocks);
    for (int b : all_blocks) {
      const int n = b * ew::kBlockElements;
      const std::string seq = "b" + std::to_string(b);
      seqs[seq] = [n, p = opts_.preempt](launch::Rt &rt) { ew::stage(rt, n, p); };
      LengthKernels k{b, {}, {}, {}, {}};
      for (App a : {App::vadd, App::vmul}) {
        const int i = (int)a;
        k.cold[i] = add(launch::cold(app_name(a)), seq);
        k.rerun[i] = add(launch::rerun(app_name(a)), seq);
        for (App t : {App::vadd, App::vmul}) {
          k.warm[i][(int)t] = add(launch::warm(app_name(a), app_name(t)), seq);
          k.pm[i][(int)t] = add(launch::warm(app_name(a), app_name(t), {.program_only = true}), seq);
        }
      }
      lengths_[b] = k;
      max_elements_ = std::max(max_elements_, n);
    }
    result_ = launch::build(list, seqs);
    list_ = list;
    seqs_ = seqs;
    if (!opts_.quiet) {
      size_t packets = 0;
      for (const launch::Kernel &k : result_.kernels)
        packets += k.packets.size() * 4;
      std::printf("%zu kernels (%s) built in %.1f ms, %zu B of control packets\n", result_.kernels.size(),
                  launch::transport_name(opts_.transport), result_.build_ms, packets);
    }
  }

  // --- XRT --------------------------------------------------------------------------

  void open() {
    bytes_ = (size_t)max_elements_ * 2;
    if (opts_.session) {
      // Only the Session's hw_context: one of ours would count as another context.
      launch::SessionOptions so;
      so.flow = opts_.elf ? launch::Flow::elf : launch::Flow::xclbin;
      if (!opts_.guard)
        so.max_gap = std::chrono::microseconds(0);
      session_ = std::make_unique<launch::Session>(list_, seqs_, so);
    } else {
      xclbin_ = xrt::xclbin(result_.base_xclbin);
      device_.register_xclbin(xclbin_);
      context_ = xrt::hw_context(device_, xclbin_.get_uuid());
      kernel_ = xrt::kernel(context_, "MLIR_AIE");
      kernels_ = std::make_unique<ew::FlowKernels>(device_, context_, kernel_, result_, opts_.elf);
    }
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
    for (int i = 0; i < kInputs; ++i) {
      xrt::bo &bo = session_ ? inputs_.emplace_back(session_->buffer(bytes_, i % 2))
                             : inputs_.emplace_back(device_, bytes_, XRT_BO_FLAGS_HOST_ONLY, kernel_.group_id(3 + i % 2));
      uint16_t *p = bo.map<uint16_t *>();
      for (int e = 0; e < max_elements_; ++e)
        p[e] = ew::to_bf16(dis(rng_));
      bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
  }

  /// Fill input `i` with special values (+-0, +-inf, NaN, denormals, extremes).
  void special_inputs() {
    const uint16_t specials[] = {0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x0001, 0x8001, 0x007F,
                                 0x7F7F, 0xFF7F, 0x0080, 0x3F80, 0xBF80, 0x4000};
    std::uniform_int_distribution<int> pick(0, (int)std::size(specials) - 1);
    for (int i = 0; i < kInputs; ++i) {
      uint16_t *p = inputs_[i].map<uint16_t *>();
      for (int e = 0; e < max_elements_; ++e)
        p[e] = specials[pick(rng_)];
      inputs_[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
  }

  xrt::bo &out_bo(int i) {
    while ((int)outputs_.size() <= i)
      if (session_)
        outputs_.push_back(session_->buffer(bytes_, 2));
      else
        outputs_.emplace_back(device_, bytes_, XRT_BO_FLAGS_HOST_ONLY, kernel_.group_id(5));
    return outputs_[i];
  }

  /// Run launches as one runlist and check them. Returns the launches that failed.
  int run(const std::vector<Launch> &ls, bool expect_ok = true, bool quiet = false) {
    poison(ls.size());
    const double gap = before_runlist();
    std::vector<size_t> ks;
    for (const Launch &l : ls)
      ks.push_back((size_t)l.kernel);
    for (size_t first = 0; first < ls.size();) {
      const size_t end = kernels_->chunk_end(ks, first);
      std::vector<std::unique_ptr<xrt::run>> runs;
      xrt::runlist rl(context_);
      for (size_t i = first; i < end; ++i) {
        const Launch &l = ls[i];
        std::unique_ptr<xrt::run> &r = runs.emplace_back(kernels_->run(l.kernel));
        r->set_arg(3, inputs_[l.lhs]);
        r->set_arg(4, inputs_[l.rhs]);
        r->set_arg(5, out_bo((int)i));
        rl.add(*r);
      }
      rl.execute();
      rl.wait();
      first = end;
    }
    last_end_ = std::chrono::steady_clock::now();
    return check(ls, gap, depends_at_start(ls), expect_ok, quiet);
  }

  /// Poison the first `n` outputs.
  void poison(size_t n) {
    for (size_t i = 0; i < n; ++i) {
      xrt::bo &o = out_bo((int)i);
      std::memset(o.map<char *>(), 0xFF, bytes_);
      o.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
  }

  /// Sleep --idle-us, then the time since the last runlist ended (us; -1: none yet).
  double before_runlist() {
    if (opts_.idle_us > 0)
      std::this_thread::sleep_for(std::chrono::microseconds(opts_.idle_us));
    if (last_end_ == std::chrono::steady_clock::time_point{})
      return -1;
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - last_end_).count();
  }

  /// The first launch is a rerun or a program-only switch: it depends on the
  /// state the previous runlist left.
  static bool depends_at_start(const std::vector<Launch> &ls) {
    return !ls.empty() && (ls[0].what == "rerun" || ls[0].what == "pm" || ls[0].what == "pm self");
  }

  /// Check the outputs of a finished runlist that started `gap` us after the last one.
  int check(const std::vector<Launch> &ls, double gap, bool dep_start, bool expect_ok, bool quiet) {
    ++stats_.runlists;
    GapBucket *bucket = nullptr;
    if (gap >= 0) {
      size_t b = 0;
      while (b < std::size(kGapEdges) && gap >= kGapEdges[b])
        ++b;
      bucket = &gaps_[b];
      ++bucket->runlists;
      bucket->dep_start += dep_start;
    }
    int failed = 0;
    for (size_t i = 0; i < ls.size(); ++i) {
      const Launch &l = ls[i];
      ++stats_.launches;
      ++stats_.kinds[l.what];
      if (l.app == App::none)
        continue;
      xrt::bo &o = out_bo((int)i);
      o.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      const uint16_t *a = inputs_[l.lhs].map<uint16_t *>(), *b = inputs_[l.rhs].map<uint16_t *>();
      const uint16_t *p = o.map<uint16_t *>();
      int bad = 0, over = 0;
      for (int e = 0; e < l.elements; ++e)
        bad += !ew::bf16_ok(a[e], b[e], p[e], l.app == App::vmul);
      for (int e = l.elements; e < max_elements_; ++e)
        over += p[e] != kPoison;
      ++stats_.checked;
      if (bad || over) {
        ++failed;
        if (expect_ok) {
          stats_.bad += bad ? 1 : 0;
          stats_.overwrites += over ? 1 : 0;
          if (bucket) {
            ++bucket->wrong;
            bucket->wrong_first += i == 0 && dep_start;
          }
        }
        if (!quiet && failed <= 3)
          std::printf("    launch %zu of %zu (%s, %d elements): %d wrong, %d words written past the end; "
                      "%.0f us after the last runlist\n",
                      i, ls.size(), l.what.c_str(), l.elements, bad, over, gap);
      }
    }
    return failed;
  }

  // --- the walk -------------------------------------------------------------------------

  /// A data launch of kind `what`, kernel index `k`, running `app`, `blocks` per core.
  Launch data(int k, App app, int blocks, const std::string &what) {
    std::uniform_int_distribution<int> in(0, kInputs - 1);
    return {k, app, blocks * ew::kBlockElements, in(rng_), in(rng_), 0, what};
  }

  /// Append one random legal launch to `ls` and update the array state.
  void step(std::vector<Launch> &ls, int blocks_choice = 0) {
    std::uniform_int_distribution<int> pick_len(0, (int)opts_.lengths.size() - 1);
    const int blocks = blocks_choice ? blocks_choice : opts_.lengths[pick_len(rng_)];
    const LengthKernels &k = lengths_.at(blocks);
    std::uniform_int_distribution<int> pick_app(0, 1);
    if (state_ == App::none) {
      const App a = (App)pick_app(rng_);
      ls.push_back(data(k.cold[(int)a], a, blocks, "cold"));
      state_ = a;
      parity_ = blocks % 2;
      return;
    }
    // weights: rerun 4, warm other 3, warm self 1, pm other 3 (at ping), pm self 1, cold 1, reset 1
    const bool ping = parity_ == 0;
    std::vector<std::pair<int, std::string>> menu = {{4, "rerun"}, {3, "warm"}, {1, "warm self"}, {1, "cold"},
                                                    {1, "reset"}};
    if (opts_.robust) // no rerun, no program-only switch: they depend on the array's state
      menu = {{3, "warm"}, {1, "warm self"}, {1, "cold"}, {1, "reset"}};
    else if (ping) {
      menu.push_back({3, "pm"});
      menu.push_back({1, "pm self"});
    }
    int total = 0;
    for (auto &m : menu)
      total += m.first;
    int r = std::uniform_int_distribution<int>(0, total - 1)(rng_);
    std::string what;
    for (auto &m : menu)
      if ((r -= m.first) < 0) {
        what = m.second;
        break;
      }
    const int s = (int)state_;
    if (what == "rerun") {
      ls.push_back(data(k.rerun[s], state_, blocks, what));
      parity_ = (parity_ + blocks) % 2;
    } else if (what == "warm" || what == "warm self") {
      const App t = what == "warm" ? other(state_) : state_;
      ls.push_back(data(k.warm[s][(int)t], t, blocks, what));
      state_ = t;
      parity_ = blocks % 2;
    } else if (what == "pm" || what == "pm self") {
      const App t = what == "pm" ? other(state_) : state_;
      ls.push_back(data(k.pm[s][(int)t], t, blocks, what));
      state_ = t;
      parity_ = (parity_ + blocks) % 2;
    } else if (what == "cold") {
      const App a = (App)pick_app(rng_);
      ls.push_back(data(k.cold[(int)a], a, blocks, what));
      state_ = a;
      parity_ = blocks % 2;
    } else { // reset, then the walk must cold-start
      ls.push_back({0, App::none, 0, 0, 0, 0, "reset"});
      state_ = App::none;
    }
  }

  /// P4: one-launch runlists, each timed (execute + wait), per kernel kind; the
  /// output of every launch is checked. Prints percentiles.
  bool latency(int count) {
    std::printf("P4 latency of one-launch runlists (%d per kind, 2 blocks per core)\n", count);
    const LengthKernels &k = lengths_.at(2);
    std::vector<std::unique_ptr<xrt::run>> keep;
    // One runlist per launch spec, made once and executed again and again.
    auto runlist_of = [&](const Launch &l, int out) {
      auto rl = std::make_unique<xrt::runlist>(context_);
      std::unique_ptr<xrt::run> r = kernels_->run(l.kernel);
      r->set_arg(3, inputs_[l.lhs]);
      r->set_arg(4, inputs_[l.rhs]);
      r->set_arg(5, out_bo(out));
      rl->add(*r);
      keep.push_back(std::move(r));
      return rl;
    };
    struct Kind { std::string name; std::vector<Launch> cycle; };
    const std::vector<Kind> kinds = {
        {"rerun", {data(k.rerun[0], App::vadd, 2, "rerun")}},
        {"warm (full switch)", {data(k.warm[0][1], App::vmul, 2, "warm"), data(k.warm[1][0], App::vadd, 2, "warm")}},
        {"pm (program-only switch)", {data(k.pm[0][1], App::vmul, 2, "pm"), data(k.pm[1][0], App::vadd, 2, "pm")}},
        {"cold", {data(k.cold[0], App::vadd, 2, "cold")}},
        {"reset", {{0, App::none, 0, 0, 0, 0, "reset"}}},
    };
    bool ok = true;
    std::printf("  %-26s %8s %8s %8s %8s %8s %8s  (us per launch, execute + wait)\n", "kind", "mean", "p50", "p90",
                "p99", "p99.9", "max");
    for (size_t ki = 0; ki < kinds.size(); ++ki) {
      const Kind &kind = kinds[ki];
      if (run(start(App::vadd)) != 0) // every kind starts from vadd at ping
        ok = false;
      std::vector<std::unique_ptr<xrt::runlist>> rls;
      for (size_t i = 0; i < kind.cycle.size(); ++i)
        rls.push_back(runlist_of(kind.cycle[i], (int)(2 * ki + i)));
      std::vector<double> us;
      for (int i = -10; i < count; ++i) { // 10 untimed first
        xrt::runlist &rl = *rls[(size_t)(i + 10) % rls.size()];
        const auto t0 = std::chrono::steady_clock::now();
        rl.execute();
        rl.wait();
        const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (i >= 0)
          us.push_back(t);
      }
      // The last output of each launch of the cycle.
      for (size_t i = 0; i < kind.cycle.size(); ++i)
        if (kind.cycle[i].app != App::none && !check_out(kind.cycle[i], (int)(2 * ki + i))) {
          std::printf("  %s: WRONG output\n", kind.name.c_str());
          ok = false;
        }
      std::sort(us.begin(), us.end());
      auto pct = [&](double p) { return us[std::min(us.size() - 1, (size_t)(p * (double)us.size()))]; };
      double mean = 0;
      for (double v : us)
        mean += v / (double)us.size();
      std::printf("  %-26s %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f\n", kind.name.c_str(), mean, pct(0.5), pct(0.9),
                  pct(0.99), pct(0.999), us.back());
    }
    run(start(App::vadd)); // a known state for what follows
    std::printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok;
  }

  /// Output `out` against launch `l` (after a runlist that is done).
  bool check_out(const Launch &l, int out) {
    xrt::bo &o = out_bo(out);
    o.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const uint16_t *a = inputs_[l.lhs].map<uint16_t *>(), *b = inputs_[l.rhs].map<uint16_t *>();
    const uint16_t *p = o.map<uint16_t *>();
    for (int e = 0; e < l.elements; ++e)
      if (!ew::bf16_ok(a[e], b[e], p[e], l.app == App::vmul))
        return false;
    return true;
  }

  bool random_walk(long launches, int max_runlist) {
    if (!opts_.quiet)
      std::printf("S1 random walk%s: %ld launches, runlists of 1..%d, lengths", opts_.robust ? " (robust kinds)" : "",
                  launches, max_runlist);
    if (!opts_.quiet) {
      for (int b : opts_.lengths)
        std::printf(" %d", b);
      std::printf(" blocks per core\n");
    }
    std::uniform_int_distribution<int> len(1, max_runlist);
    const auto t0 = std::chrono::steady_clock::now();
    const long bad0 = stats_.bad + stats_.overwrites;
    long done = 0;
    while (done < launches) {
      std::vector<Launch> ls;
      const int n = (int)std::min<long>(len(rng_), launches - done);
      while ((int)ls.size() < n)
        step(ls);
      run(ls);
      done += (long)ls.size();
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const long bad = stats_.bad + stats_.overwrites - bad0;
    if (!opts_.quiet || bad)
      std::printf("  %ld launches in %.1f s (%.1f us per launch, checks included): %s\n", done, s,
                  1e6 * s / (double)done, bad ? "FAIL" : "PASS");
    if (!opts_.quiet)
      print_gaps();
    return bad == 0;
  }

  /// Wrong launches by the time since the previous runlist ended.
  void print_gaps() const {
    std::printf("  %-10s %9s %24s %7s %18s\n", "gap", "runlists", "start: rerun/pm as asked", "wrong",
                "wrong at that start");
    for (size_t b = 0; b < std::size(gaps_); ++b)
      if (gaps_[b].runlists)
        std::printf("  %-10s %9ld %24ld %7ld %18ld\n", kGapNames[b], gaps_[b].runlists, gaps_[b].dep_start,
                    gaps_[b].wrong, gaps_[b].wrong_first);
  }

  /// S9: the random walk through launch::Session. Its guard runs a rerun or
  /// program-only switch as the cold kernel when the array state is unknown
  /// (another context, or more than max_gap since the last runlist); with
  /// --no-guard max_gap is 0 (the state is trusted across any gap). Even
  /// lengths only: the Session does not know the ping/pong position, so a
  /// cold kernel in place of a rerun must not move it.
  bool session_walk(long launches, int max_runlist) {
    for (int b : opts_.lengths)
      if (b % 2)
        throw std::invalid_argument("mode session needs even --lengths (the Session does not track ping/pong)");
    std::printf("S9 random walk through launch::Session (guard %s, %s), %d us idle before each runlist: %ld "
                "launches, runlists of 1..%d, lengths",
                opts_.guard ? "on" : "off", session_->watching() ? "watching the driver" : session_->watch_why().c_str(),
                opts_.idle_us, launches, max_runlist);
    for (int b : opts_.lengths)
      std::printf(" %d", b);
    std::printf(" blocks per core\n");
    std::uniform_int_distribution<int> len(1, max_runlist);
    const auto t0 = std::chrono::steady_clock::now();
    const long bad0 = stats_.bad + stats_.overwrites;
    std::map<std::string, long> status;
    long done = 0, substituted = 0, as_asked = 0, not_ok_wrong = 0;
    while (done < launches) {
      std::vector<Launch> ls;
      const int n = (int)std::min<long>(len(rng_), launches - done);
      while ((int)ls.size() < n)
        step(ls);
      std::vector<launch::Launch> sl;
      for (size_t i = 0; i < ls.size(); ++i)
        sl.push_back({list_.configs[ls[i].kernel].name, {inputs_[ls[i].lhs], inputs_[ls[i].rhs], out_bo((int)i)}});
      poison(ls.size());
      const double gap = before_runlist();
      const launch::RunReport rep = session_->run(sl);
      last_end_ = std::chrono::steady_clock::now();
      ++status[launch::status_name(rep.status)];
      substituted += (long)rep.substituted;
      const bool asked = depends_at_start(ls) && !rep.ran.empty() && rep.ran[0] == sl[0].kernel;
      as_asked += asked;
      const bool ok = rep.status == launch::RunReport::Status::ok;
      if (!ok && rep.status != launch::RunReport::Status::other_context_ran)
        std::printf("    runlist of %zu: %s (%s)\n", ls.size(), launch::status_name(rep.status), rep.message.c_str());
      const int wrong = check(ls, gap, asked, ok, !ok);
      if (!ok)
        not_ok_wrong += wrong;
      done += (long)ls.size();
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const long bad = stats_.bad + stats_.overwrites - bad0;
    std::printf("  %ld launches in %.1f s; %ld launches run as the cold kernel, %ld runlists started with a rerun "
                "or program-only switch as asked\n",
                done, s, substituted, as_asked);
    std::printf("  runlists by status:");
    for (const auto &[k, v] : status)
      std::printf(" %s %ld", k.c_str(), v);
    std::printf("; %ld wrong launches in runlists not reported ok\n", not_ok_wrong);
    print_gaps();
    const bool failed = bad || status.count("failed") || status.count("stopped");
    std::printf("  launches the Session reports ok: %ld wrong: %s\n", bad, failed ? "FAIL" : "PASS");
    return !failed;
  }

  bool lengths() {
    std::printf("S2 runlist lengths\n");
    bool ok = true;
    for (int n : {1, 23, 24, 25, 48, 200, 1000}) {
      std::vector<Launch> ls;
      while ((int)ls.size() < n)
        step(ls);
      const int failed = run(ls);
      std::printf("  %4d launches: %s\n", n, failed ? "FAIL" : "PASS");
      ok &= failed == 0;
    }
    return ok;
  }

  /// Bring the array to app `a` at ping with a cold kernel of 2 blocks.
  std::vector<Launch> start(App a) {
    state_ = a;
    parity_ = 0;
    return {data(lengths_.at(2).cold[(int)a], a, 2, "cold")};
  }

  bool parity() {
    std::printf("S3 odd lengths before a switch\n");
    const LengthKernels &one = lengths_.at(1), &two = lengths_.at(2);
    // odd, then full: must pass
    std::vector<Launch> ls = start(App::vadd);
    ls.push_back(data(one.rerun[0], App::vadd, 1, "rerun"));
    ls.push_back(data(two.warm[0][1], App::vmul, 2, "warm"));
    const int f1 = run(ls);
    std::printf("  odd, then full switch: %s\n", f1 ? "FAIL" : "PASS");
    // odd, then program only: the chains are at pong; expected wrong data, no hang
    ls = start(App::vadd);
    ls.push_back(data(one.rerun[0], App::vadd, 1, "rerun"));
    ls.push_back(data(two.pm[0][1], App::vmul, 2, "pm at pong"));
    int f2 = -1;
    try {
      f2 = run(ls, false, true);
    } catch (const std::exception &e) {
      std::printf("  odd, then program-only switch: ERROR %s\n", e.what());
      return false;
    }
    std::printf("  odd, then program-only switch (expected wrong, no hang): %s\n",
                f2 ? "wrong data, as expected" : "UNEXPECTED: right data");
    // and a cold kernel recovers
    const int f3 = run(start(App::vmul));
    std::printf("  cold afterwards: %s\n", f3 ? "FAIL" : "PASS");
    state_ = App::vmul;
    parity_ = 0;
    return f1 == 0 && f3 == 0;
  }

  bool wrong_prev() {
    std::printf("S4 a warm kernel with a wrong prev\n");
    const LengthKernels &two = lengths_.at(2);
    // vadd is in the array; warm(vmul, vadd) and warm(vmul, vadd, pm) claim vmul.
    std::vector<Launch> ls = start(App::vadd);
    ls.push_back(data(two.warm[1][0], App::vadd, 2, "warm, wrong prev"));
    ls.push_back(data(two.pm[1][1], App::vmul, 2, "pm, wrong prev"));
    const int f = run(ls, false, true);
    std::printf("  same floorplan: %s\n", f ? "wrong data" : "right data (the reset of prev and run is the same)");
    state_ = App::vmul;
    parity_ = 0;
    return true; // informative
  }

  bool special() {
    std::printf("S8 special values (+-0, +-inf, NaN, denormals, extremes)\n");
    special_inputs();
    std::vector<Launch> ls = start(App::vadd);
    ls.push_back(data(lengths_.at(2).warm[0][1], App::vmul, 2, "warm"));
    ls.push_back(data(lengths_.at(2).pm[1][0], App::vadd, 2, "pm"));
    ls.push_back(data(lengths_.at(2).rerun[0], App::vadd, 2, "rerun"));
    const int f = run(ls, false, true);
    // Sort the differences: a denormal input or result (the NPU flushes them
    // to 0), or anything else (would be a real error).
    auto denormal = [](uint16_t h) { return (h & 0x7F80) == 0 && (h & 0x7F) != 0; };
    long diff = 0, flush = 0, other_diff = 0;
    for (size_t i = 0; i < ls.size(); ++i) {
      const Launch &l = ls[i];
      const uint16_t *a = inputs_[l.lhs].map<uint16_t *>(), *b = inputs_[l.rhs].map<uint16_t *>();
      const uint16_t *p = out_bo((int)i).map<uint16_t *>();
      for (int e = 0; e < l.elements; ++e) {
        if (ew::bf16_ok(a[e], b[e], p[e], l.app == App::vmul))
          continue;
        ++diff;
        const float fa = ew::from_bf16(a[e]), fb = ew::from_bf16(b[e]);
        const uint16_t want = ew::to_bf16(l.app == App::vmul ? fa * fb : fa + fb);
        const bool zero_out = (p[e] & 0x7FFF) == 0;
        if (denormal(a[e]) || denormal(b[e]) || (denormal(want) && zero_out)) {
          ++flush;
        } else if (++other_diff <= 5) {
          std::printf("    %s: %#06x %s %#06x = %#06x, CPU %#06x\n", l.what.c_str(), a[e],
                      l.app == App::vmul ? "*" : "+", b[e], p[e], want);
        }
      }
    }
    std::printf("  %ld elements differ: %ld from denormals (flushed to 0 on the NPU), %ld other\n", diff, flush,
                other_diff);
    std::printf("  %s\n", other_diff ? "OTHER DIFFERENCES: see above" : f ? "PASS (only denormal flushing)" : "PASS");
    state_ = App::vadd;
    parity_ = 0;
    // random values again for the next mode
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
    for (int i = 0; i < kInputs; ++i) {
      uint16_t *p = inputs_[i].map<uint16_t *>();
      for (int e = 0; e < max_elements_; ++e)
        p[e] = ew::to_bf16(dis(rng_));
      inputs_[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    return true; // informative
  }

  bool big() {
    std::printf("S6 the largest length (%d blocks per core, %d elements, %.1f MB per buffer)\n", kBigBlocks,
                kBigBlocks * ew::kBlockElements, kBigBlocks * ew::kBlockElements * 2 / 1e6);
    const LengthKernels &k = lengths_.at(kBigBlocks);
    std::vector<Launch> ls = {data(k.cold[0], App::vadd, kBigBlocks, "cold"),
                              data(k.warm[0][1], App::vmul, kBigBlocks, "warm"),
                              data(k.pm[1][0], App::vadd, kBigBlocks, "pm"),
                              data(k.rerun[0], App::vadd, kBigBlocks, "rerun")};
    const int f = run(ls);
    std::printf("  %s\n", f ? "FAIL" : "PASS");
    state_ = App::vadd;
    parity_ = 0;
    return f == 0;
  }

  /// Which program output `i` of the last runlist matches: "vadd", "vmul" or "neither".
  std::string matches(const Launch &l, int i) {
    xrt::bo &o = out_bo(i);
    o.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const uint16_t *a = inputs_[l.lhs].map<uint16_t *>(), *b = inputs_[l.rhs].map<uint16_t *>();
    const uint16_t *p = o.map<uint16_t *>();
    bool add = true, mul = true;
    for (int e = 0; e < l.elements; ++e) {
      add &= ew::bf16_ok(a[e], b[e], p[e], false);
      mul &= ew::bf16_ok(a[e], b[e], p[e], true);
    }
    return add ? "vadd" : mul ? "vmul" : "neither";
  }

  /// A child's arguments: this package and options, then `more`.
  std::vector<std::string> child_args(std::vector<std::string> more) const {
    std::vector<std::string> a = {"--package", package_};
    if (opts_.preempt)
      a.push_back("--preempt");
    a.insert(a.end(), {"--transport", launch::transport_name(opts_.transport)});
    a.insert(a.end(), {"--flow", ew::flow_name(opts_.elf)});
    a.insert(a.end(), more.begin(), more.end());
    return a;
  }

  /// S5b: a child process runs its own context in a loop while this one runs
  /// long runlists of switches. Returns the launches that came out wrong.
  long concurrent(const char *self, int runlists) {
    // Start the child and wait for its first pass, so that it is on the NPU
    // before our runlists (starting it can take longer than they do).
    const std::string progress = "stress_child_progress.txt";
    std::remove(progress.c_str());
    auto passes = [&] {
      long n = 0;
      std::ifstream(progress) >> n;
      return n;
    };
    const auto t0 = std::chrono::steady_clock::now();
    ew::Child child(self, child_args({"--mode", "child-loop", "--progress", progress}));
    while (passes() == 0 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30))
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const long first = passes();
    std::printf("  child on the NPU after %.0f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    // Two kinds of runlist: full switches only (they do not depend on what is in
    // the array when the floorplans are equal), and kernels that do depend on
    // it (cold, then program-only switches and reruns).
    const LengthKernels &two = lengths_.at(2);
    long wrong_full = 0, wrong_state = 0, total_full = 0, total_state = 0;
    for (int r = 0; r < runlists; ++r) {
      std::vector<Launch> full = start(App::vadd), state = start(App::vadd);
      for (int i = 0; i < 99; ++i) {
        const App from = i % 2 ? App::vmul : App::vadd, to = other(from);
        full.push_back(data(two.warm[(int)from][(int)to], to, 2, "warm, concurrent"));
        if (i % 3 == 0)
          state.push_back(data(two.pm[(int)from][(int)to], to, 2, "pm, concurrent"));
        else
          state.push_back(data(two.rerun[(int)from], from, 2, "rerun, concurrent"));
      }
      // keep `state` legal: replay it to know which app each launch runs
      App now = App::vadd;
      for (size_t i = 1; i < state.size(); ++i) {
        if (state[i].what == "pm, concurrent") {
          const App to = other(now);
          state[i] = data(two.pm[(int)now][(int)to], to, 2, "pm, concurrent");
          now = to;
        } else {
          state[i] = data(two.rerun[(int)now], now, 2, "rerun, concurrent");
        }
      }
      wrong_full += run(full, false, true);
      total_full += (long)full.size();
      wrong_state += run(state, false, true);
      total_state += (long)state.size();
    }
    const long last = passes();
    child.stop();
    child.wait();
    std::remove(progress.c_str());
    std::printf("  while another process loops on the NPU (%ld of its passes meanwhile), %d runlists of 100 launches "
                "each:\n",
                last - first, runlists);
    if (first == 0)
      std::printf("    the child never ran: no overlap measured\n");
    std::printf("    full switches only:              %ld of %ld launches wrong\n", wrong_full, total_full);
    std::printf("    program-only switches and reruns: %ld of %ld launches wrong\n", wrong_state, total_state);
    return wrong_full + wrong_state;
  }

  bool context(const char *self) {
    std::printf("S5 another hw_context or process between kernels\n");
    const LengthKernels &two = lengths_.at(2);
    // In process: context B on vmul.xclbin runs one rerun(vmul) between.
    xrt::xclbin xb(package_ + "/vmul/design.xclbin");
    device_.register_xclbin(xb);
    xrt::hw_context ctx_b(device_, xb.get_uuid());
    xrt::kernel kb(ctx_b, "MLIR_AIE");
    const launch::Kernel &rk = result_.kernels[two.rerun[1]];
    if (!rk.packets.empty())
      throw std::logic_error("context: a rerun kernel with control packets");
    xrt::bo ib(device_, rk.words.size() * 4, XCL_BO_FLAGS_CACHEABLE, kb.group_id(1));
    std::memcpy(ib.map<void *>(), rk.words.data(), rk.words.size() * 4);
    ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // In context A the array is vadd.xclbin's, switched to vmul by our kernels.
    std::vector<Launch> ls = start(App::vadd);
    ls.push_back(data(two.warm[0][1], App::vmul, 2, "warm"));
    int f = run(ls);
    auto r = kb(kOpcode, ib, (uint32_t)rk.words.size(), inputs_[0], inputs_[1], out_bo(0));
    r.wait();
    const Launch after_b = data(two.rerun[1], App::vmul, 2, "rerun after context B");
    const int after_ctx = run({after_b}, false, true);
    std::printf("  rerun(vmul) in A after a run in context B: %s (the output is %s's)\n",
                after_ctx ? "WRONG: the switched state of A was lost" : "right: state kept",
                matches(after_b, 0).c_str());
    // If A came back as its xclbin (vadd), a warm kernel from vadd recovers.
    f += run(start(App::vadd));
    f += run({data(two.warm[0][1], App::vmul, 2, "warm")});
    r = kb(kOpcode, ib, (uint32_t)rk.words.size(), inputs_[0], inputs_[1], out_bo(0));
    r.wait();
    const int warm_from_base =
        run({data(two.warm[0][1], App::vmul, 2, "warm(vadd, vmul) after context B")}, false, true);
    std::printf("  warm(vadd, vmul) after context B (as if A is its xclbin again): %s\n",
                warm_from_base ? "WRONG" : "right");
    // Another process: this program in mode "child" runs vadd.xclbin's cold kernel.
    f += run(start(App::vadd));
    f += run({data(two.warm[0][1], App::vmul, 2, "warm")});
    const int status = ew::Child(self, child_args({"--mode", "child"})).wait();
    const Launch after_p = data(two.rerun[1], App::vmul, 2, "rerun after another process");
    const int after_proc = run({after_p}, false, true);
    std::printf("  child process exit %d; rerun(vmul) afterwards: %s (the output is %s's)\n", status,
                after_proc ? "WRONG: the switched state was lost" : "right: state kept",
                matches(after_p, 0).c_str());
    // A cold kernel always recovers.
    const int cold_after = run(start(App::vadd));
    std::printf("  cold afterwards: %s\n", cold_after ? "FAIL" : "PASS");
    // S5b: another process at the same time as long runlists.
    concurrent(self, 10);
    const int cold_end = run(start(App::vadd));
    std::printf("  cold afterwards: %s\n", cold_end ? "FAIL" : "PASS");
    return f == 0 && cold_after == 0 && cold_end == 0;
  }

  /// The child of `context`: one cold(vmul) launch on its own context.
  bool child() {
    return run({data(lengths_.at(2).cold[1], App::vmul, 2, "child cold")}) == 0;
  }

  /// The child of `concurrent`: cold(vmul) and reruns in a loop until killed;
  /// after each pass it writes the number of passes to `progress`.
  bool child_loop(const std::string &progress) {
    for (long pass = 1;; ++pass) {
      run({data(lengths_.at(2).cold[1], App::vmul, 2, "child cold"),
           data(lengths_.at(2).rerun[1], App::vmul, 2, "child rerun")},
          false, true);
      std::ofstream(progress, std::ios::trunc) << pass << "\n";
    }
  }

  void report() const {
    std::printf("\n%ld launches in %ld runlists, %ld outputs checked; unexpected: %ld wrong, %ld written past the "
                "end\n",
                stats_.launches, stats_.runlists, stats_.checked, stats_.bad, stats_.overwrites);
    for (const auto &[k, v] : stats_.kinds)
      std::printf("  %-36s %8ld\n", k.c_str(), v);
  }

private:
  std::string package_;
  std::mt19937 rng_;
  Options opts_;
  launch::Result result_;
  launch::ConfigList list_;
  launch::SequenceMap seqs_;
  std::unique_ptr<launch::Session> session_;
  std::chrono::steady_clock::time_point last_end_{}; ///< the last runlist ended
  GapBucket gaps_[std::size(kGapNames)];
  std::map<int, LengthKernels> lengths_;
  int max_elements_ = 0;
  size_t bytes_ = 0;
  xrt::device device_{0};
  xrt::xclbin xclbin_;
  xrt::hw_context context_;
  xrt::kernel kernel_;
  std::unique_ptr<ew::FlowKernels> kernels_;
  std::vector<xrt::bo> inputs_, outputs_;
  App state_ = App::none;
  int parity_ = 0; ///< blocks per core since the last DMA reset, mod 2
  Stats stats_;
};

} // namespace

int main(int argc, char **argv) {
  std::string package, mode = "all", progress;
  unsigned seed = 1;
  long launches = 20000;
  int max_runlist = 64, count = 2000;
  Options opts;
  bool hang_risk = false, transport_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string opt = argv[i];
    if (opt == "--robust") {
      opts.robust = true;
      continue;
    }
    if (opt == "--preempt") {
      opts.preempt = true;
      continue;
    }
    if (opt == "--no-guard") {
      opts.guard = false;
      continue;
    }
    if (opt == "--hang-risk") {
      hang_risk = true;
      continue;
    }
    if (i + 1 >= argc)
      break;
    const char *v = argv[++i];
    if (opt == "--package") package = v;
    else if (opt == "--mode") mode = v;
    else if (opt == "--seed") seed = (unsigned)std::atoi(v);
    else if (opt == "--launches") launches = std::atol(v);
    else if (opt == "--max-runlist") max_runlist = std::atoi(v);
    else if (opt == "--count") count = std::atoi(v);
    else if (opt == "--idle-us") opts.idle_us = std::atoi(v);
    else if (opt == "--progress") progress = v;
    else if (opt == "--transport") opts.transport = launch::transport_from_name(v), transport_set = true;
    else if (opt == "--flow") opts.elf = ew::elf_flow(v);
    else if (opt == "--lengths") {
      opts.lengths.clear();
      std::stringstream ss(v);
      for (std::string b; std::getline(ss, b, ',');)
        opts.lengths.push_back(std::atoi(b.c_str()));
    }
  }
  if (package.empty() || opts.lengths.empty()) {
    std::cerr << "usage: stress.exe --package DIR [--mode MODE] [--seed S] [--launches L] [--max-runlist R]\n"
                 "                  [--lengths B,B,...] [--robust] [--preempt] [--count C] [--transport txn|ctrlpkt]\n"
                 "                  [--flow elf|xclbin] [--hang-risk] [--idle-us U] [--no-guard]\n";
    return 2;
  }
  // Other hw_contexts on the NPU: txn unless asked (the children of ctx are).
  const bool shared = mode == "ctx" || mode == "churn" || opts.robust;
  if (shared && !transport_set)
    opts.transport = launch::Transport::txn;
  if (shared && opts.transport == launch::Transport::ctrlpkt && !hang_risk) {
    std::cerr << "--mode " << mode << (opts.robust ? " --robust" : "")
              << " with --transport ctrlpkt may hang the NPU: add --hang-risk (Linux only)\n";
    return 2;
  }
  opts.big = mode == "big" || mode == "all";
  opts.session = mode == "session";
  try {
    if (mode == "churn") {
      // P7: a whole Stress (build, hw_context, buffers, a short walk) made and
      // destroyed `count` times in this process.
      std::printf("P7 hw_context churn: %d times build + hw_context + 100 launches + destroy, transport %s\n", count,
                  launch::transport_name(opts.transport));
      Options brief = opts; // not "small": a macro in Windows' rpcndr.h
      brief.lengths = {2};
      brief.quiet = true;
      int failed = 0;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < count; ++i) {
        Stress s(package, seed + (unsigned)i, brief);
        failed += s.random_walk(100, 16) ? 0 : 1;
      }
      const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("  %d of %d rounds failed, %.1f ms per round\n", failed, count, 1e3 * sec / count);
      std::printf("stress churn: %s\n", failed ? "FAIL" : "PASS");
      return failed ? 1 : 0;
    }
    Stress s(package, seed, opts);
    if (mode == "child")
      return s.child() ? 0 : 1;
    if (mode == "child-loop")
      return s.child_loop(progress) ? 0 : 1;
    std::printf("%d x %d cores, seed %u, transport %s, flow %s\n", EW_COLS, EW_CORES, seed,
                launch::transport_name(opts.transport), ew::flow_name(opts.elf));
    bool ok = true;
    if (mode == "lengths" || mode == "all") ok &= s.lengths();
    if (mode == "parity" || mode == "all") ok &= s.parity();
    if (mode == "prev" || mode == "all") ok &= s.wrong_prev();
    if (mode == "special" || mode == "all") ok &= s.special();
    if (mode == "big" || mode == "all") ok &= s.big();
    if (mode == "ctx") ok &= s.context(argv[0]);
    if (mode == "latency") ok &= s.latency(count);
    if (mode == "random" || mode == "all") ok &= s.random_walk(launches, max_runlist);
    if (mode == "session") ok &= s.session_walk(launches, max_runlist);
    s.report();
    std::printf("stress %s: %s\n", mode.c_str(), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
