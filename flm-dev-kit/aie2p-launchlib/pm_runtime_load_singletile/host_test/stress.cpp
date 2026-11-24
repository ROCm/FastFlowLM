//===- stress.cpp - stress and pressure tests of the three apps ---*- C++ -*-===//
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
//   stress.exe --package DIR --mode MODE [--seed S] [--launches L] [--max-runlist R] [--count C]
//              [--flow xclbin|elf] [--policy cold_only|refuse|allow] [--no-guard] [--robust]
//              [--max-gap US] [--child loop|churn|churn-empty|churn-register] [--pause US] [--life US] [--exclusive] [--blind]
//              [--max-failures N] [--transport txn|ctrlpkt] [--hang-risk]
//
// The kernels: reset; per app (vadd, vmul, vminus: three floorplans) cold and
// rerun; warm(p, t) for all nine pairs (a switch to itself too); and the cold
// and rerun of vadd with its second sequence (vadd_rev). They run through a
// launch::Session (launchlib/include/launch/xrt/session.hpp; --transport, below): order check,
// sharing guard (--policy, default cold_only; --max-gap, default 2000 us, 1000 on Windows;
// --no-guard: off, and the NPU taken as ours alone), recovery, ELF chunks.
// --blind: no guard, as on Windows (every warm and rerun runs as cold);
// --exclusive: on Windows, the NPU is ours alone (warm and rerun run as asked). Every launch writes its own output buffer, checked against the CPU
// (designs/design.hpp bounds); the words after the output must stay untouched.
// Launches the Session reports "ok" must be right; the other reports are counted.
//
// MODE
//   random   S1: a random walk over the kernels, in runlists of random length
//            (1 .. R). The walk knows which app is in the array, so it only picks
//            legal kernels: rerun(app in the array), warm(app in the array, any),
//            cold(any), reset (then a cold). --robust: cold and reset only.
//   lengths  S2: runlists of 1, 23, 24, 25, 48, 200 and 1000 launches.
//   order    S3: the Session refuses, before running, a warm kernel whose prev is
//            not in the array, a rerun of another app, a rerun after a reset.
//   recover  S4: with the order check off, a warm kernel with a wrong prev: the
//            runlist fails and the Session opens a new hw_context; the buffers
//            made before still work. A rerun of the wrong app: wrong data, no
//            error, and the next cold kernels too (shim tasks that never finish;
//            what the order check prevents); Session::reopen recovers.
//   cap      the failure cap (order check off, --max-failures, default 2): a
//            success between two failures starts the count again; 2 failures
//            in a row stop the Session (no hw_context, status stopped) until
//            Session::reopen. Hangs the NPU 3 times: never on Windows.
//   ctx      S5: another hw_context in this process (vmul.xclbin, cold(vmul)):
//            while it exists, warm and rerun launches run as cold kernels; after
//            it ran, the array state is unknown and the next one does too; the
//            same with another process in between. All right.
//   guard    S6: another process uses the NPU while this one walks (all kinds):
//            --child loop: one context that runs cold(vmul), rerun(vmul) in a
//            loop; --child churn: contexts that open, run cold(vmul) once and
//            close, one after the other (~3 ms each); --child churn-empty: the same
//            without the run; --pause US between two of them. Counts what the Session
//            reported and the results that were wrong although it said ok.
//   watch    S7: the guard with the context list (Linux; Windows built with
//            LAUNCH_NPU_WATCH_XRT): a walk alone (warm and rerun as asked), a
//            walk next to another process that runs cold(vmul) only (all as
//            cold), a walk after it is gone (as asked again; the first runlist
//            may run cold). That process starts and stops between our runlists
//            only. Prints whether its submission count moves. Needs the NPU to
//            itself at the start; --launches split over the three walks.
//   lifetime S8: how short another process's context can be in the context
//            list (the Session's max_gap must stay below it): a child opens and
//            closes --count contexts, with one cold(vmul) each, then with none,
//            while this process asks for the list as fast as it can. Runs
//            nothing itself. Needs the NPU to itself at the start. Fails if
//            more than 1% of the queries fail.
//   all      lengths, order, then random.
//   latency  P4: one-launch runs of each kind, timed one by one (--count per
//            kind): mean, p50, p90, p99, p99.9, max. --no-guard: without the
//            two driver queries per run.
//   churn    P7: --count times: Session (build, hw_context), 100 launches, close.
//   warm_first  a list that starts with warm(vadd, vmul): the Session takes the
//            array as vadd's (the hw_context's xclbin), so a rerun(vmul) first is
//            refused and warm then rerun gives vmul's values (PR #183 review).
//
// --transport (default ctrlpkt): how the kernels load (launch::Transport). The
// modes with other hw_contexts on the NPU (ctx, guard, watch, lifetime, churn,
// the children), --robust, --policy allow and those without the order check
// (recover, cap) default to txn: with ctrlpkt a kernel whose packets lose their
// route waits on a Sync that never ends, and that has not been tested. With
// --transport ctrlpkt they also need --hang-risk (Linux only).
//
// Exit 0 when every check a mode expects to pass passes.
//
//===----------------------------------------------------------------------===//

#include "designs/design.hpp"
#include "sequence.hpp"

#include "launch/launch.hpp"
#include "launch/xrt/session.hpp"
#include "spawn.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace hosttest;
using Status = launch::RunReport::Status;

namespace {

constexpr int kApps = 3, kInputs = 4; // input buffers per argument to pick from
constexpr size_t kGuard = 1024;       // words after each output that must stay NaN
const char *kAppNames[kApps] = {"vadd", "vmul", "vminus"};
constexpr int kNone = -1;

/// The value app `a` computes from x (A) and y (B), and its bound (design.hpp).
float compute(int a, float x, float y) { return a == 0 ? x + y : a == 1 ? x * y : x - y; }
float bound(int a, float x, float y) {
  const float r = compute(a, x, y);
  return a == 1 ? fp32_product_bound(x, y, r) : fp32_sum_bound(x, y, r);
}

/// One launch of a runlist, to check afterwards.
struct Launch {
  int kernel;
  int app;  ///< the program that runs (kNone for reset)
  int a, b; ///< input buffers
  std::string what;
};

struct Stats {
  std::map<std::string, long> kinds;
  std::map<std::string, long> reports; ///< Session status -> runs
  long launches = 0, runlists = 0, checked = 0, bad = 0, overwrites = 0;
  long substituted = 0, reopened = 0, flagged_wrong = 0; ///< wrong, in a report that said so
  long dependent = 0; ///< warm and rerun launches asked for (run as asked: dependent - substituted)
  long seen = 0;      ///< runs where the Session saw another context (or a failed query)
  long cold_runs = 0; ///< runs with a launch run as cold
};

struct Options {
  bool robust = false; ///< walk: cold and reset only
  bool quiet = false;
  launch::Transport transport = launch::Transport::ctrlpkt;
  launch::SessionOptions session;
};

/// The list: reset; per app cold, rerun, warm to each app; vadd's second sequence.
struct Kernels {
  launch::ConfigList list;
  int cold[kApps]{}, rerun[kApps]{}, warm[kApps][kApps]{}, cold_rev = 0, rerun_rev = 0;

  Kernels(const std::string &package, launch::Transport transport) : list{package, transport, {launch::reset()}} {
    auto add = [&](launch::Config c) {
      list.configs.push_back(c);
      return (int)list.configs.size() - 1;
    };
    for (int a = 0; a < kApps; ++a) {
      cold[a] = add(launch::cold(kAppNames[a]));
      rerun[a] = add(launch::rerun(kAppNames[a]));
      for (int t = 0; t < kApps; ++t)
        warm[a][t] = add(launch::warm(kAppNames[a], kAppNames[t]));
    }
    launch::Config k = launch::cold("vadd", "cold_vadd_rev");
    k.sequence = "vadd_rev";
    cold_rev = add(k);
    k = launch::rerun("vadd", "rerun_vadd_rev");
    k.sequence = "vadd_rev";
    rerun_rev = add(k);
  }
};

class Stress {
public:
  Stress(const std::string &package, unsigned seed, Options opts = {})
      : package_(package), rng_(seed), opts_(opts), k_(package, opts.transport), session_(k_.list, sequences(), opts.session) {
    if (!opts_.quiet)
      std::printf("%zu kernels built in %.1f ms; sharing guard %s%s%s\n", session_.result().kernels.size(),
                  session_.result().build_ms, session_.watching() ? "on" : "off",
                  session_.watching() ? "" : ": ", session_.watching() ? "" : session_.watch_why().c_str());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
    const MakeBuffer mk = [this](size_t bytes, unsigned arg) { return session_.buffer(bytes, arg); };
    for (unsigned arg = 0; arg < 2; ++arg)
      for (int i = 0; i < kInputs; ++i) {
        std::vector<float> v(kElems);
        for (float &x : v)
          x = dis(rng_);
        xrt::bo &bo = inputs_[arg].emplace_back(fp32_buffer(mk, arg, kElems, v));
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }
    sync_state();
  }

  /// Output buffer `i`, made on first use with NaN in every word (guard words included).
  xrt::bo &out_bo(int i) {
    while ((int)outputs_.size() <= i) {
      xrt::bo &o = outputs_.emplace_back(session_.buffer((kElems + kGuard) * sizeof(float), 2));
      fp32_fill_nan(o, kElems + kGuard);
      o.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    return outputs_[i];
  }

  struct Check {
    int bad = 0;       ///< values wrong
    int unwritten = 0; ///< of them: still NaN (the launch did not write them)
    int over = 0;      ///< guard words written
  };

  /// Output `out` against launch `l`.
  Check check_out(const Launch &l, int out) {
    xrt::bo &o = out_bo(out);
    o.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const float *x = inputs_[0][l.a].map<float *>(), *y = inputs_[1][l.b].map<float *>(), *p = o.map<float *>();
    const size_t a_elems = l.app == 0 ? N : kElems;
    Check c;
    for (size_t e = 0; e < kElems; ++e) {
      const float xa = x[e % a_elems];
      if (!(std::fabs(p[e] - compute(l.app, xa, y[e])) <= bound(l.app, xa, y[e]))) {
        ++c.bad;
        c.unwritten += std::isnan(p[e]);
      }
    }
    for (size_t e = kElems; e < kElems + kGuard; ++e)
      c.over += !std::isnan(p[e]);
    return c;
  }

  /// Run launches with one Session::run and check them. Launches the Session
  /// reports "ok" must be right; in an other_context_ran report they may be
  /// wrong (counted apart); after a failure only the finished ones are checked.
  /// Returns the launches that were wrong where they had to be right.
  int run(const std::vector<Launch> &ls, bool expect_ok = true, bool quiet = false) {
    std::vector<launch::Launch> sl;
    for (size_t i = 0; i < ls.size(); ++i) {
      xrt::bo &o = out_bo((int)i);
      fp32_fill_nan(o, kElems + kGuard);
      o.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      const launch::Config &c = session_.list().configs[ls[i].kernel];
      stats_.dependent += c.kind() == launch::Config::Kind::warm || c.kind() == launch::Config::Kind::rerun;
      if (ls[i].app == kNone)
        sl.push_back({c.name, {}});
      else
        sl.push_back({c.name, {inputs_[0][ls[i].a], inputs_[1][ls[i].b], o}});
    }
    last_ = session_.run(sl);
    ++stats_.runlists;
    ++stats_.reports[launch::status_name(last_.status)];
    stats_.substituted += (long)last_.substituted;
    stats_.seen += last_.other_context_seen ? 1 : 0;
    stats_.cold_runs += last_.substituted ? 1 : 0;
    stats_.reopened += last_.reopened ? 1 : 0;
    sync_state();
    if (last_.status != Status::ok && !quiet)
      std::printf("    session: %s (%s)\n", launch::status_name(last_.status), last_.message.c_str());
    const size_t checked = last_.status == Status::failed                                         ? last_.done
                           : last_.status == Status::refused || last_.status == Status::stopped ? 0
                                                                                                  : ls.size();
    int failed = 0;
    for (size_t i = 0; i < ls.size(); ++i) {
      const Launch &l = ls[i];
      ++stats_.launches;
      ++stats_.kinds[l.what];
      if (l.app == kNone || i >= checked)
        continue;
      const Check c = check_out(l, (int)i);
      ++stats_.checked;
      if (!c.bad && !c.over)
        continue;
      if (last_.status == Status::other_context_ran) {
        ++stats_.flagged_wrong;
        continue;
      }
      ++failed;
      if (expect_ok) {
        stats_.bad += c.bad ? 1 : 0;
        stats_.overwrites += c.over ? 1 : 0;
      }
      if (!quiet && failed <= 3)
        std::printf("    launch %zu (%s, %s): %d wrong (%d not written), %d words written past the end\n", i,
                    l.what.c_str(), kAppNames[l.app], c.bad, c.unwritten, c.over);
    }
    return failed;
  }

  /// The walk's idea of the array: the Session's.
  void sync_state() {
    const std::optional<std::string> &s = session_.state();
    state_ = kNone;
    for (int a = 0; a < kApps; ++a)
      if (s && *s == kAppNames[a])
        state_ = a;
  }

  // --- the walk -------------------------------------------------------------------------

  Launch data(int k, int app, const std::string &what) {
    std::uniform_int_distribution<int> in(0, kInputs - 1);
    return {k, app, in(rng_), in(rng_), what};
  }

  /// Append one random legal launch to `ls` and update the array state.
  void step(std::vector<Launch> &ls) {
    std::uniform_int_distribution<int> pick_app(0, kApps - 1), coin(0, 1);
    auto cold = [&](int a) {
      // vadd has a second sequence: half its cold launches use it
      if (a == 0 && coin(rng_))
        ls.push_back(data(k_.cold_rev, 0, "cold, vadd_rev"));
      else
        ls.push_back(data(k_.cold[a], a, "cold"));
      state_ = a;
    };
    if (state_ == kNone)
      return cold(pick_app(rng_));
    // weights: rerun 4, warm to another app 4, warm to itself 1, cold 1, reset 1
    std::vector<std::pair<int, std::string>> menu = {{4, "rerun"}, {4, "warm"}, {1, "warm self"}, {1, "cold"},
                                                    {1, "reset"}};
    if (opts_.robust)
      menu = {{4, "cold"}, {1, "reset"}};
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
    if (what == "rerun") {
      if (state_ == 0 && coin(rng_))
        ls.push_back(data(k_.rerun_rev, 0, "rerun, vadd_rev"));
      else
        ls.push_back(data(k_.rerun[state_], state_, "rerun"));
    } else if (what == "warm" || what == "warm self") {
      int t = state_;
      if (what == "warm")
        t = (state_ + 1 + std::uniform_int_distribution<int>(0, kApps - 2)(rng_)) % kApps;
      ls.push_back(data(k_.warm[state_][t], t, what + " " + kAppNames[state_] + "->" + kAppNames[t]));
      state_ = t;
    } else if (what == "cold") {
      cold(pick_app(rng_));
    } else { // reset, then the walk must cold-start
      ls.push_back({0, kNone, 0, 0, "reset"});
      state_ = kNone;
    }
  }

  bool random_walk(long launches, int max_runlist) {
    if (!opts_.quiet)
      std::printf("S1 random walk%s: %ld launches, runlists of 1..%d\n", opts_.robust ? " (cold and reset only)" : "",
                  launches, max_runlist);
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
    return bad == 0;
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

  /// Bring the array to app `a` with a cold kernel.
  std::vector<Launch> start(int a) {
    state_ = a;
    return {data(k_.cold[a], a, "cold")};
  }

  /// Cold kernels of every app, one run each: the number that came out wrong.
  int colds() {
    int f = 0;
    for (int a : {2, 1, 0})
      f += run(start(a), false, true) ? 1 : 0;
    return f;
  }

  /// S3: illegal orders are refused before anything runs.
  bool order() {
    std::printf("S3 the order check\n");
    bool ok = true;
    auto refused = [&](const std::vector<Launch> &ls, const char *what) {
      bool threw = false;
      try {
        run(ls, false, true);
      } catch (const std::logic_error &e) {
        threw = true;
        std::printf("  %-44s refused: %s\n", what, e.what());
      }
      if (!threw)
        std::printf("  %-44s NOT refused\n", what);
      ok &= threw;
    };
    ok &= run(start(0)) == 0;
    refused({data(k_.warm[1][2], 2, "warm")}, "vadd in the array, warm(vmul, vminus):");
    refused({data(k_.rerun[1], 1, "rerun")}, "vadd in the array, rerun(vmul):");
    refused({data(k_.warm[0][1], 1, "warm"), data(k_.rerun[0], 0, "rerun")}, "warm(vadd, vmul), then rerun(vadd):");
    refused({{0, kNone, 0, 0, "reset"}, data(k_.rerun[0], 0, "rerun")}, "reset, then rerun(vadd):");
    const int after = run({data(k_.warm[0][1], 1, "warm"), data(k_.rerun[1], 1, "rerun")});
    std::printf("  the array is still vadd's: warm(vadd, vmul), rerun(vmul): %s\n", after ? "FAIL" : "PASS");
    return ok && after == 0;
  }

  /// S4 (mode recover: a Session whose order check is off): what a wrong order
  /// does, and the recovery.
  bool recover() {
    std::printf("S4 recovery (order check off)\n");
    bool ok = true;
    ok &= run(start(0)) == 0;
    const auto t0 = std::chrono::steady_clock::now();
    run({data(k_.warm[1][2], 2, "warm, wrong prev")}, false, true);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  vadd in the array, warm(vmul, vminus): %s after %.1f s, %s\n", launch::status_name(last_.status), s,
                last_.reopened ? "new hw_context" : "same hw_context");
    ok &= last_.status == Status::failed && last_.reopened;
    const int fresh = colds();
    std::vector<Launch> walk = start(0);
    walk.push_back(data(k_.warm[0][1], 1, "warm"));
    walk.push_back(data(k_.warm[1][2], 2, "warm"));
    walk.push_back(data(k_.rerun[2], 2, "rerun"));
    walk.push_back(data(k_.warm[2][0], 0, "warm"));
    const int legal = run(walk);
    std::printf("  then, buffers made before: cold vminus, vmul, vadd %s; a warm cycle with reruns %s\n",
                fresh ? "FAIL" : "PASS", legal ? "FAIL" : "PASS");
    ok &= fresh == 0 && legal == 0;
    // A rerun of the wrong app: no error, wrong data, and it stays.
    run(start(0));
    const int wrong = run({data(k_.rerun[2], 2, "rerun, wrong app")}, false, true);
    std::printf("  vadd in the array, rerun(vminus): %s, status %s\n", wrong ? "wrong data" : "right data",
                launch::status_name(last_.status));
    std::printf("    then cold vminus, vmul, vadd: %d of 3 wrong (shim tasks that never finish)\n", colds());
    session_.reopen();
    sync_state();
    const int after = colds();
    std::printf("    Session::reopen, cold vminus, vmul, vadd: %s\n", after ? "FAIL" : "PASS");
    return ok && after == 0;
  }

  /// Mode cap (order check off, max_failures 2): stop after two failures in a row.
  bool cap() {
    std::printf("the failure cap (order check off, max_failures %zu)\n", opts_.session.max_failures);
    auto hang = [&](const char *what) {
      run(start(0), false, true);
      run({data(k_.warm[1][2], 2, "warm, wrong prev")}, false, true);
      std::printf("  %-40s %s, %s\n", what, launch::status_name(last_.status),
                  last_.reopened ? "new hw_context" : session_.stopped() ? "stopped, no hw_context" : "same hw_context");
    };
    bool ok = run(start(0)) == 0;
    hang("failure 1:");
    ok &= last_.status == Status::failed && last_.reopened && !session_.stopped();
    const int between = colds();
    std::printf("  a success between: cold vminus, vmul, vadd %s\n", between ? "FAIL" : "PASS");
    ok &= between == 0;
    hang("failure 1 again (the count started over):");
    ok &= last_.status == Status::failed && last_.reopened && !session_.stopped();
    // Right after it, with no success between (hang() starts with a cold launch: not here).
    run({data(k_.warm[1][2], 2, "warm, wrong prev")}, false, true);
    std::printf("  %-40s %s, %s\n", "failure 2 in a row:", launch::status_name(last_.status),
                session_.stopped() ? "stopped, no hw_context" : "NOT stopped");
    ok &= last_.status == Status::failed && !last_.reopened && session_.stopped();
    run(start(0), false, true);
    std::printf("  then a cold launch: %s\n", launch::status_name(last_.status));
    ok &= last_.status == Status::stopped;
    session_.reopen();
    sync_state();
    const int after = colds();
    std::printf("  Session::reopen, cold vminus, vmul, vadd: %s\n", after ? "FAIL" : "PASS");
    return ok && after == 0;
  }

  /// Another hw_context in this process, on vmul.xclbin, that runs cold(vmul).
  struct Other {
    xrt::xclbin xb;
    xrt::hw_context ctx;
    xrt::kernel k;
    xrt::bo instr, a, b, c, packets;
    const launch::Kernel *kernel = nullptr;
    uint32_t words;
    Other(launch::Session &s, const std::string &package) {
      xb = xrt::xclbin(package + "/vmul/design.xclbin");
      s.device().register_xclbin(xb);
      ctx = xrt::hw_context(s.device(), xb.get_uuid());
      k = xrt::kernel(ctx, "MLIR_AIE");
      const auto &ks = s.result().kernels;
      size_t i = 0;
      while (ks[i].name != "cold_vmul")
        ++i;
      kernel = &ks[i];
      words = (uint32_t)ks[i].words.size();
      instr = xrt::bo(s.device(), words * 4, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
      std::memcpy(instr.map<void *>(), ks[i].words.data(), words * 4);
      instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      a = xrt::bo(s.device(), kElems * 4, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
      b = xrt::bo(s.device(), kElems * 4, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
      c = xrt::bo(s.device(), kElems * 4, XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));
      packets = launch::packet_buffer(s.device(), k, *kernel);
    }
    void run() {
      xrt::run r(k);
      r.set_arg(0, 3u);
      r.set_arg(1, instr);
      r.set_arg(2, words);
      r.set_arg(3, a);
      r.set_arg(4, b);
      r.set_arg(5, c);
      launch::set_packets(r, *kernel, packets);
      r.start();
      r.wait();
    }
  };

  /// A runlist that switches: warm to each app, reruns between.
  std::vector<Launch> switches() {
    std::vector<Launch> ls;
    int now = state_ == kNone ? 0 : state_;
    if (state_ == kNone)
      ls.push_back(data(k_.cold[0], 0, "cold"));
    for (int i = 0; i < 6; ++i) {
      const int t = (now + 1) % kApps;
      ls.push_back(data(k_.warm[now][t], t, "warm"));
      ls.push_back(data(k_.rerun[t], t, "rerun"));
      now = t;
    }
    state_ = now;
    return ls;
  }

  /// S5: another hw_context between and during our runs.
  bool context(const char *self) {
    std::printf("S5 another hw_context or process\n");
    bool ok = true;
    ok &= run(start(0)) == 0;
    {
      // Opened right after our last run (within max_gap): the Session sees it
      // after the run, not before. The run is then flagged, never silently wrong.
      Other first(session_, package_);
      const int f0 = run(switches(), true, true);
      std::printf("  another context opened just before our run: %s, %zu as cold, %s\n",
                  launch::status_name(last_.status), last_.substituted, f0 ? "FAIL" : "PASS");
      ok &= f0 == 0 && (last_.status == Status::other_context_ran || last_.substituted == 12);
    }
    {
      Other other(session_, package_);
      std::this_thread::sleep_for(std::chrono::milliseconds(5)); // more than max_gap
      const long s0 = stats_.substituted;
      const int f = run(switches());
      std::printf("  another context exists (idle): warm and rerun as cold: %ld of 12; %s\n",
                  stats_.substituted - s0, f ? "FAIL" : "PASS");
      ok &= f == 0 && stats_.substituted - s0 == 12;
      other.run();
      const int f2 = run(switches());
      std::printf("  it ran a launch between our runs: %s, %s\n", launch::status_name(last_.status),
                  f2 ? "FAIL" : "PASS");
      ok &= f2 == 0;
    }
    // It is gone: our array may be vmul.xclbin's... the Session does not know.
    const long s1 = stats_.substituted;
    const int f3 = run(switches());
    std::printf("  it closed: the Session ran %ld launches as cold (it existed at the last look, or the state is unknown); %s\n",
                stats_.substituted - s1, f3 ? "FAIL" : "PASS");
    ok &= f3 == 0;
    // Another process, between two runs; vmul in our array (not the xclbin's app).
    run(switches());
    run({data(k_.warm[state_][1], 1, "warm")});
    state_ = 1;
    const int status = ew::Child(self, {"--package", package_, "--mode", "child"}).wait();
    const long s2 = stats_.substituted;
    const int f4 = run(switches());
    std::printf("  a child process ran (exit %d): substituted %ld, %s\n", status, stats_.substituted - s2,
                f4 ? "FAIL" : "PASS");
    ok &= f4 == 0 && status == 0;
    return ok;
  }

  /// S6: another process loops on the NPU while this one walks.
  bool guard(const char *self, long launches) {
    std::printf("S6 the sharing guard: another process (%s) on the NPU, policy %s, max gap %lld us\n",
                child_.c_str(), policy_name(opts_.session.shared), (long long)opts_.session.max_gap.count());
    std::vector<std::string> args = {"--package", package_, "--mode", "child-" + child_};
    if (pause_us_)
      args.insert(args.end(), {"--pause", std::to_string(pause_us_)});
    if (life_us_)
      args.insert(args.end(), {"--life", std::to_string(life_us_)});
    ew::Child child(self, args);
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // let it open its context
    const Stats s0 = stats_;
    const bool ok = random_walk(launches, 16);
    child.stop();
    child.wait();
    std::printf("  runs: ");
    for (const auto &[k, v] : stats_.reports)
      std::printf("%s %ld  ", k.c_str(), v - (s0.reports.count(k) ? s0.reports.at(k) : 0));
    std::printf("\n  launches run as cold %ld; new hw_contexts %ld; wrong where the Session said ok %ld; wrong where it "
                "said other_context_ran %ld\n",
                stats_.substituted - s0.substituted, stats_.reopened - s0.reopened, stats_.bad - s0.bad,
                stats_.flagged_wrong - s0.flagged_wrong);
    return ok;
  }

  /// S7 (mode watch): the guard with the context list. Alone, warm and rerun run
  /// as asked; next to another process whose context runs cold launches only,
  /// every warm and rerun runs as cold; after it is gone, as asked again. The
  /// other process starts and stops between our runlists only, never inside one.
  bool watch(const char *self, long launches) {
    std::printf("S7 the sharing guard with the context list: %s%s\n", session_.watching() ? "on" : "off: ",
                session_.watch_why().c_str());
    if (!session_.watching())
      return false;
    const launch::xrt_support::NpuWatch list(opts_.session.accel);
    auto others = [&]() { // contexts of other processes -> submissions; nullopt: the query failed
      std::optional<launch::xrt_support::Snapshot> s = list.snapshot();
      if (s)
        std::erase_if(*s, [](const auto &e) { return e.first.pid == launch::xrt_support::detail::own_pid(); });
      return s;
    };
    auto submissions = [](const launch::xrt_support::Snapshot &s) {
      uint64_t n = 0;
      for (const auto &[k, v] : s)
        n += v;
      return n;
    };
    auto wait_for = [&](bool present) { // ms until another context is there (gone), -1 after 10 s
      const auto t0 = std::chrono::steady_clock::now();
      for (;;) {
        const auto s = others();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (s && s->empty() != present)
          return ms;
        if (ms > 10000)
          return -1.0;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    };
    struct Phase {
      long dependent, substituted, bad, runs, seen, cold_runs;
    };
    auto walk = [&](const char *name, long n) {
      const Stats s0 = stats_;
      const bool ok = random_walk(n, 16);
      const Phase p{stats_.dependent - s0.dependent,           stats_.substituted - s0.substituted,
                    stats_.bad + stats_.overwrites - s0.bad - s0.overwrites, stats_.runlists - s0.runlists,
                    stats_.seen - s0.seen,                     stats_.cold_runs - s0.cold_runs};
      std::printf("  %-14s warm/rerun asked %ld, run as asked %ld, as cold %ld; wrong %ld; new hw_contexts %ld; %s\n"
                  "  %-14s runs %ld: another context seen in %ld, a launch run as cold in %ld\n",
                  name, p.dependent, p.dependent - p.substituted, p.substituted, p.bad, stats_.reopened - s0.reopened,
                  ok ? "ok" : "FAIL", "", p.runs, p.seen, p.cold_runs);
      return std::pair{ok && p.bad == 0, p};
    };
    // Alone, a run may still start with its first warm or rerun as cold: after
    // max_gap without a look, the Session takes the array as unknown (a context
    // may have come and gone unseen). One per run, never more.
    auto alone_ok = [](const Phase &p, long seen) {
      return p.seen <= seen && p.substituted <= p.cold_runs + 16 * seen && p.dependent > p.substituted;
    };
    bool ok = true;

    const auto o0 = others();
    std::printf("  other contexts now: %s\n", !o0 ? "query failed" : std::to_string(o0->size()).c_str());
    if (!o0 || !o0->empty()) {
      std::printf("  the NPU is not ours alone: not run\n");
      return false;
    }
    const auto [ok1, alone] = walk("alone:", launches / 3);
    ok &= ok1 && alone_ok(alone, 0);

    ew::Child child(self, {"--package", package_, "--mode", "child-cold"});
    const double seen = wait_for(true);
    std::printf("  other process (cold launches only) seen after %.0f ms\n", seen);
    ok &= seen >= 0;
    if (seen >= 0) {
      const auto a = others();
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      const auto b = others();
      const bool moved = a && b && submissions(*b) > submissions(*a);
      std::printf("  its submissions over 200 ms: %llu -> %llu (%s)\n", a ? (unsigned long long)submissions(*a) : 0ull,
                  b ? (unsigned long long)submissions(*b) : 0ull,
                  moved ? "they move" : "they do NOT move: a context that runs is not seen by its count");
      const auto [ok2, shared] = walk("next to it:", launches / 3);
      ok &= ok2 && shared.substituted == shared.dependent;
    }
    child.stop();
    child.wait();
    const double gone = wait_for(false);
    std::printf("  it stopped; its context gone after %.0f ms\n", gone);
    ok &= gone >= 0;
    if (gone >= 0) {
      // The first run after it saw the change: all as cold; then as alone.
      const auto [ok3, after] = walk("alone again:", launches / 3);
      ok &= ok3 && alone_ok(after, 1);
    }
    std::printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok;
  }

  /// S8 (mode lifetime): how short another process's context can be, as the
  /// context list sees it. A child opens and closes `count` contexts (one
  /// cold(vmul) each, then none at all) while this process asks for the list
  /// as fast as it can. The Session's max_gap must stay below the shortest one:
  /// a context that comes and goes inside a gap is not seen. Runs nothing here.
  bool lifetime(const char *self, int count) {
    std::printf("S8 lifetime of another process's contexts in the list: guard %s%s\n",
                session_.watching() ? "on" : "off: ", session_.watch_why().c_str());
    if (!session_.watching())
      return false;
    using clock = std::chrono::steady_clock;
    auto ms = [](clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
    const launch::xrt_support::NpuWatch list(opts_.session.accel);
    auto others = [&]() {
      std::optional<launch::xrt_support::Snapshot> s = list.snapshot();
      if (s)
        std::erase_if(*s, [](const auto &e) { return e.first.pid == launch::xrt_support::detail::own_pid(); });
      return s;
    };
    const auto o0 = others();
    if (!o0 || !o0->empty()) {
      std::printf("  the NPU is not ours alone (%s): not run\n", o0 ? "other contexts" : "query failed");
      return false;
    }
    bool ok = true;
    for (const char *mode : {"child-churn", "child-churn-empty"}) {
      // One span per context: a key that was gone and is back is a new context
      // (Linux uses ids again after a close).
      std::vector<std::pair<clock::time_point, clock::time_point>> span;
      std::map<launch::xrt_support::ContextKey, size_t> open; // in the last list -> its span
      long polls = 0, failed = 0;
      const clock::time_point t0 = clock::now();
      ew::Child child(self, {"--package", package_, "--mode", mode, "--count", std::to_string(count)});
      for (bool alive = true; alive;) {
        for (int i = 0; i < 64; ++i) {
          const auto s = others();
          const clock::time_point t = clock::now();
          ++polls;
          if (!s) {
            if (++failed <= 3)
              std::printf("  query failed: %s\n", list.last_error().c_str());
            continue;
          }
          std::map<launch::xrt_support::ContextKey, size_t> now;
          for (const auto &[k, n] : *s) {
            auto it = open.find(k);
            if (it == open.end()) {
              now[k] = span.size();
              span.push_back({t, t});
            } else {
              now[k] = it->second;
              span[it->second].second = t;
            }
          }
          open.swap(now);
        }
        alive = child.running() && clock::now() - t0 < std::chrono::minutes(5);
      }
      const double total = ms(clock::now() - t0);
      const int status = child.wait();
      std::vector<double> seen;
      long once = 0;
      for (const auto &ts : span) {
        seen.push_back(ms(ts.second - ts.first));
        once += ts.second == ts.first;
      }
      std::sort(seen.begin(), seen.end());
      auto pct = [&](double p) { return seen[std::min(seen.size() - 1, (size_t)(p * (double)seen.size()))]; };
      std::printf("  %-18s exit %d; list asked %ld times (%.1f us each, %ld failed); contexts seen %zu of %d\n", mode,
                  status, polls, 1e3 * total / (double)std::max(polls, 1L), failed, span.size(), count);
      if (!seen.empty())
        std::printf("  %-18s seen for (first .. last sighting) min %.2f p1 %.2f p10 %.2f p50 %.2f ms; in one sighting "
                    "only %ld\n",
                    "", seen.front(), pct(0.01), pct(0.1), pct(0.5), once);
      // A failed query is a "shared" verdict to the Session (safe); a few while
      // contexts open and close are allowed (Windows: ~0.6% of XRT's queries throw).
      ok &= status == 0 && failed * 100 <= polls;
    }
    std::printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok;
  }

  static const char *policy_name(launch::SharedPolicy p) {
    return p == launch::SharedPolicy::cold_only ? "cold_only" : p == launch::SharedPolicy::refuse ? "refuse" : "allow";
  }

  /// P4: one-launch runs, each timed, per kernel kind; every output is checked.
  bool latency(int count) {
    std::printf("P4 latency of one-launch runs (%d per kind, sharing guard %s)\n", count,
                session_.watching() ? "on" : "off");
    struct Kind { std::string name; std::vector<Launch> cycle; };
    const std::vector<Kind> kinds = {
        {"rerun vadd", {data(k_.rerun[0], 0, "rerun")}},
        {"vminus: warm, rerun, warm back", {data(k_.warm[0][2], 2, "warm"), data(k_.rerun[2], 2, "rerun"),
                                            data(k_.warm[2][0], 0, "warm")}},
        {"warm vadd<->vmul", {data(k_.warm[0][1], 1, "warm"), data(k_.warm[1][0], 0, "warm")}},
        {"warm cycle of 3", {data(k_.warm[0][1], 1, "warm"), data(k_.warm[1][2], 2, "warm"),
                             data(k_.warm[2][0], 0, "warm")}},
        {"warm self (vadd->vadd)", {data(k_.warm[0][0], 0, "warm self")}},
        {"cold vadd", {data(k_.cold[0], 0, "cold")}},
        {"reset + cold vadd", {{0, kNone, 0, 0, "reset"}, data(k_.cold[0], 0, "cold")}},
    };
    bool ok = true;
    std::printf("  %-32s %8s %8s %8s %8s %8s %8s  (us per launch)\n", "kind", "mean", "p50", "p90", "p99", "p99.9",
                "max");
    for (const Kind &kind : kinds) {
      ok &= run(start(0)) == 0;
      std::vector<double> us;
      const int n = count / (int)kind.cycle.size() * (int)kind.cycle.size(); // whole cycles
      for (int i = -(int)kind.cycle.size() * 4; i < n; ++i) {                // 4 cycles untimed first
        const Launch &l = kind.cycle[(size_t)(i + (int)kind.cycle.size() * 4) % kind.cycle.size()];
        std::vector<launch::Launch> one;
        const launch::Config &c = session_.list().configs[l.kernel];
        if (l.app == kNone)
          one.push_back({c.name, {}});
        else
          one.push_back({c.name, {inputs_[0][l.a], inputs_[1][l.b], out_bo(0)}});
        const auto t0 = std::chrono::steady_clock::now();
        const launch::RunReport rep = session_.run(one);
        const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (rep.status != Status::ok)
          ok = false;
        if (i >= 0)
          us.push_back(t);
        if (l.app != kNone && i == n - 1) {
          const Check ch = check_out(l, 0);
          if (ch.bad) {
            std::printf("  %s: WRONG output\n", kind.name.c_str());
            ok = false;
          }
        }
      }
      sync_state();
      std::sort(us.begin(), us.end());
      auto pct = [&](double p) { return us[std::min(us.size() - 1, (size_t)(p * (double)us.size()))]; };
      double mean = 0;
      for (double v : us)
        mean += v / (double)us.size();
      std::printf("  %-32s %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f\n", kind.name.c_str(), mean, pct(0.5), pct(0.9),
                  pct(0.99), pct(0.999), us.back());
    }
    std::printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok;
  }

  /// The child of `context`: one cold(vmul) launch on its own Session.
  bool child() { return run({data(k_.cold[1], 1, "child cold")}) == 0; }

  /// The child of `guard`: cold(vmul), rerun(vmul) in a loop until killed.
  bool child_loop() {
    for (;;)
      run({data(k_.cold[1], 1, "child cold"), data(k_.rerun[1], 1, "child rerun")}, false, true);
  }

  /// The child of `watch`: cold(vmul) launches only (no hang if they meet ours), until killed.
  bool child_cold() {
    for (;;)
      run({data(k_.cold[1], 1, "child cold")}, false, true);
  }

  void report() const {
    std::printf("\n%ld launches in %ld runs, %ld outputs checked; unexpected: %ld wrong, %ld written past the end\n",
                stats_.launches, stats_.runlists, stats_.checked, stats_.bad, stats_.overwrites);
    std::printf("session: ");
    for (const auto &[k, v] : stats_.reports)
      std::printf("%s %ld  ", k.c_str(), v);
    std::printf("; launches run as cold %ld; new hw_contexts %ld\n", stats_.substituted, stats_.reopened);
    for (const auto &[k, v] : stats_.kinds)
      std::printf("  %-36s %8ld\n", k.c_str(), v);
  }

  launch::Session &session() { return session_; }
  void set_child(const std::string &c) { child_ = c; }
  void set_pause(long us) { pause_us_ = us; }
  void set_life(long us) { life_us_ = us; }

private:
  std::string package_;
  std::mt19937 rng_;
  Options opts_;
  Kernels k_;
  launch::Session session_;
  std::vector<xrt::bo> inputs_[2], outputs_;
  int state_ = kNone;
  std::string child_ = "loop";
  long pause_us_ = 0; // churn children: pause between contexts
  long life_us_ = 0;  // churn children: an empty context stays open this long
  launch::RunReport last_;
  Stats stats_;
};

/// The churn child of `guard` and `lifetime`: contexts on vmul.xclbin that open,
/// run cold(vmul) once (`run`; else nothing) and close, `count` times (0: until
/// killed), `pause_us` apart. No Session: it would keep one context open. Prints
/// how long they lived. `life_us`: an empty context stays open that long before it closes;
/// `no_context`: no hw_context at all, only a device handle and register_xclbin.
int child_churn(const std::string &package, long count, bool run, long pause_us, long life_us, bool no_context) {
  launch::ConfigList list{package, launch::Transport::txn, {launch::cold("vmul")}};
  const launch::Result r = launch::build(list, sequences());
  xrt::device dev(0u);
  xrt::xclbin xb(r.base_xclbin);
  dev.register_xclbin(xb);
  const uint32_t words = (uint32_t)r.kernels[0].words.size();
  using clock = std::chrono::steady_clock;
  auto ms = [](clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
  std::vector<double> whole, made; // open .. closed; made .. closing
  for (long i = 0; count <= 0 || i < count; ++i) {
    const clock::time_point t0 = clock::now();
    clock::time_point t1, t2;
    if (no_context) {
      xrt::device d(0u);
      t1 = clock::now();
      d.register_xclbin(xb);
      t2 = clock::now();
    } else {
      xrt::hw_context ctx(dev, xb.get_uuid());
      t1 = clock::now();
      if (life_us)
        std::this_thread::sleep_for(std::chrono::microseconds(life_us));
      if (run) {
        xrt::kernel k(ctx, "MLIR_AIE");
        xrt::bo instr(dev, words * 4, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
        std::memcpy(instr.map<void *>(), r.kernels[0].words.data(), words * 4);
        instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        xrt::bo a(dev, kElems * 4, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3)),
            b(dev, kElems * 4, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4)),
            c(dev, kElems * 4, XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));
        k(3u, instr, words, a, b, c).wait();
      }
      t2 = clock::now();
    }
    whole.push_back(ms(clock::now() - t0));
    made.push_back(ms(t2 - t1));
    if (pause_us)
      std::this_thread::sleep_for(std::chrono::microseconds(pause_us));
  }
  std::sort(whole.begin(), whole.end());
  std::sort(made.begin(), made.end());
  auto pct = [](const std::vector<double> &v, double p) { return v[std::min(v.size() - 1, (size_t)(p * (double)v.size()))]; };
  if (!whole.empty())
    std::printf("  child: %zu contexts (%s): open .. closed min %.2f p50 %.2f max %.2f ms; "
                "made .. closing min %.2f p50 %.2f ms\n",
                whole.size(), run ? "one cold(vmul) each" : "no run", whole.front(), pct(whole, 0.5), whole.back(),
                made.front(), pct(made, 0.5));
  std::fflush(stdout);
  return 0;
}

/// A list that starts with a warm kernel: {warm(vadd, vmul), rerun(vmul)}. The
/// hw_context is opened on vadd's xclbin (the first app the list names), so the
/// Session must take the array as vadd's: rerun(vmul) first is an order error,
/// warm(vadd, vmul) then rerun(vmul) gives vmul's values. (A Session that took
/// the first config's app, vmul, ran rerun(vmul) on vadd's program and said ok.)
bool warm_first(const std::string &package, launch::SessionOptions opts) {
  opts.max_gap = std::chrono::microseconds(0);
  launch::Session s(launch::ConfigList{package, launch::Transport::txn,
                                       {launch::warm("vadd", "vmul"), launch::rerun("vmul")}},
                    sequences(), opts);
  const MakeBuffer mk = [&](size_t bytes, unsigned arg) { return s.buffer(bytes, arg); };
  std::vector<float> x(kElems), y(kElems);
  for (size_t e = 0; e < kElems; ++e) {
    x[e] = 0.5f + (float)(e % 97);
    y[e] = 2.0f - (float)(e % 31) / 8;
  }
  xrt::bo a = fp32_buffer(mk, 0, kElems, x), b = fp32_buffer(mk, 1, kElems, y), c = fp32_buffer(mk, 2, kElems);
  a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bool refused = false;
  try {
    s.run({{"rerun_vmul", {a, b, c}}});
  } catch (const std::logic_error &e) {
    refused = true;
    std::printf("  rerun(vmul) first: refused (%s)\n", e.what());
  }
  fp32_fill_nan(c, kElems);
  c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  const launch::RunReport r = s.run({{"vadd_then_vmul", {a, b, c}}, {"rerun_vmul", {a, b, c}}});
  c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const float *p = c.map<float *>();
  size_t right = 0;
  for (size_t e = 0; e < kElems; ++e)
    right += std::fabs(p[e] - compute(1, x[e], y[e])) <= bound(1, x[e], y[e]);
  std::printf("  warm(vadd, vmul), rerun(vmul): session %s, %zu of %u values are vmul's\n",
              launch::status_name(r.status), right, kElems);
  return refused && r.status == Status::ok && right == kElems;
}

} // namespace

int main(int argc, char **argv) {
  std::string package, mode = "all";
  unsigned seed = 1;
  long launches = 20000;
  int max_runlist = 64, count = 2000;
  bool count_set = false; // child-churn: until killed unless --count
  std::string child = "loop";
  long max_failures = -1; // -1: the Session's default (cap mode: 2)
  long pause_us = 0;      // churn children: pause between contexts
  long life_us = 0;       // churn children: an empty context stays open this long
  bool hang_risk = false, transport_set = false;
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string opt = argv[i];
    if (opt == "--robust") {
      opts.robust = true;
      continue;
    }
    if (opt == "--no-guard") {
      opts.session.accel = "";
      opts.session.exclusive = true;
      continue;
    }
    if (opt == "--blind") {
      opts.session.accel = "";
      opts.session.exclusive = false;
      continue;
    }
    if (opt == "--exclusive") {
      opts.session.exclusive = true;
      continue;
    }
    if (opt == "--hang-risk") {
      hang_risk = true;
      continue;
    }
    if (i + 1 >= argc)
      break;
    const std::string v = argv[++i];
    if (opt == "--package") package = v;
    else if (opt == "--mode") mode = v;
    else if (opt == "--seed") seed = (unsigned)std::stoul(v);
    else if (opt == "--launches") launches = std::stol(v);
    else if (opt == "--max-runlist") max_runlist = std::stoi(v);
    else if (opt == "--count") count = std::stoi(v), count_set = true;
    else if (opt == "--max-gap") opts.session.max_gap = std::chrono::microseconds(std::stol(v));
    else if (opt == "--child") child = v;
    else if (opt == "--pause") pause_us = std::stol(v);
    else if (opt == "--life") life_us = std::stol(v);
    else if (opt == "--max-failures") max_failures = std::stol(v);
    else if (opt == "--transport") opts.transport = launch::transport_from_name(v), transport_set = true;
    else if (opt == "--flow") opts.session.flow = v == "elf" ? launch::Flow::elf : launch::Flow::xclbin;
    else if (opt == "--policy")
      opts.session.shared = v == "refuse"  ? launch::SharedPolicy::refuse
                            : v == "allow" ? launch::SharedPolicy::allow
                                           : launch::SharedPolicy::cold_only;
  }
  if (package.empty()) {
    std::cerr << "usage: stress.exe --package DIR [--mode MODE] [--seed S] [--launches L] [--max-runlist R]\n"
                 "                  [--count C] [--flow xclbin|elf] [--policy cold_only|refuse|allow] [--no-guard]\n"
                 "                  [--robust] [--max-gap US] [--child loop|churn|churn-empty|churn-register] [--pause US] [--life US]\n"
                 "                  [--exclusive] [--blind]\n"
                 "                  [--max-failures N] [--transport txn|ctrlpkt] [--hang-risk]\n";
    return 2;
  }
  // Other hw_contexts on the NPU, or no order check: txn unless asked.
  const bool is_child = mode.rfind("child", 0) == 0;
  const bool shared = is_child || mode == "ctx" || mode == "guard" || mode == "watch" || mode == "lifetime" ||
                      mode == "churn" || mode == "recover" || mode == "cap" || opts.robust ||
                      opts.session.shared == launch::SharedPolicy::allow;
  if (shared && !transport_set)
    opts.transport = launch::Transport::txn;
  if (shared && !is_child && opts.transport == launch::Transport::ctrlpkt && !hang_risk) {
    std::cerr << "--mode " << mode << " (or --robust, --policy allow) with --transport ctrlpkt may hang the NPU: "
                 "add --hang-risk (Linux only)\n";
    return 2;
  }
  if (max_failures >= 0)
    opts.session.max_failures = (size_t)max_failures;
  else if (mode == "cap")
    opts.session.max_failures = 2;
  try {
    if (mode == "churn") {
      // P7: a whole Stress (Session: build, hw_context; buffers; a short walk)
      // made and destroyed `count` times in this process.
      std::printf("P7 hw_context churn: %d times Session + 100 launches + destroy, transport %s\n", count,
                  launch::transport_name(opts.transport));
      int failed = 0;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < count; ++i) {
        Options quiet = opts;  // not `small`: a macro in Windows' rpcndr.h
        quiet.quiet = true;
        Stress s(package, seed + (unsigned)i, quiet);
        failed += s.random_walk(100, 16) ? 0 : 1;
      }
      const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("  %d of %d rounds failed, %.1f ms per round\n", failed, count, 1e3 * sec / count);
      std::printf("stress churn: %s\n", failed ? "FAIL" : "PASS");
      return failed ? 1 : 0;
    }
    if (mode == "child-churn" || mode == "child-churn-empty" || mode == "child-churn-register")
      return child_churn(package, count_set ? count : 0, mode == "child-churn", pause_us, life_us,
                         mode == "child-churn-register");
    if (mode == "warm_first") {
      const bool ok = warm_first(package, opts.session);
      std::printf("stress warm_first: %s\n", ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
    }
    if (mode == "recover" || mode == "cap")
      opts.session.check_order = false;
    if (mode == "cap") { // no guard: after a failure it would run the next wrong warm as cold
      opts.session.accel = "";
      opts.session.exclusive = true;
    }
    Stress s(package, seed, opts);
    s.set_child(child);
    s.set_pause(pause_us);
    s.set_life(life_us);
    if (mode == "child")
      return s.child() ? 0 : 1;
    if (mode == "child-loop")
      return s.child_loop() ? 0 : 1;
    if (mode == "child-cold")
      return s.child_cold() ? 0 : 1;
    std::printf("%u x %u cores, seed %u, transport %s\n", kCols, TILES, seed, launch::transport_name(opts.transport));
    bool ok = true;
    if (mode == "lengths" || mode == "all") ok &= s.lengths();
    if (mode == "order" || mode == "all") ok &= s.order();
    if (mode == "recover") ok &= s.recover();
    if (mode == "cap") ok &= s.cap();
    if (mode == "ctx") ok &= s.context(argv[0]);
    if (mode == "guard") ok &= s.guard(argv[0], launches);
    if (mode == "watch") ok &= s.watch(argv[0], launches);
    if (mode == "lifetime") ok &= s.lifetime(argv[0], count);
    if (mode == "latency") ok &= s.latency(count);
    if (mode == "random" || mode == "all") ok &= s.random_walk(launches, max_runlist);
    s.report();
    std::printf("stress %s: %s\n", mode.c_str(), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
