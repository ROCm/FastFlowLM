// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// session.cpp - y = (A + B) * C + D through launch::Session
//
// host.cpp runs the kernels itself; this runs them through the Session
// (launchlib/include/launch/xrt/session.hpp): its order check, sharing guard,
// recovery and resident lanes. Every result is checked.
//
//   session.exe --package DIR [--n N] [--iters N] [--pairs P] [--flow elf|xclbin]
//               [--resident on|off] [--other] [--seconds S] [--timeout MS] [--quiet MS]
//
//   expression   cold(vadd), warm(vadd, vmul), warm(vmul, vadd): full and pm
//   alternate    runs of 2P switches (full, then program only), timed
//   rerun        runs of 2P reruns of vadd, timed
//   --other      a second hw_context (vmul's xclbin) in this process: while it
//                is open, warm launches must run as cold; after it closes they
//                run warm again (with resident lanes: as their twins)
//   --seconds S  then the expression and alternate pm again and again for S
//                seconds (for several processes at once): counts each status;
//                a launch the Session reported ok must be right
//   --timeout MS SessionOptions::timeout (default 10000)
//   --quiet MS   SessionOptions::resident_quiet (default 100)
//
// --resident (default: the Session's, off) sets
// SessionOptions::resident_lanes. Exit 1 on a wrong result the Session
// reported ok, or a status that should not be.

#include "bf16.hpp"
#include "elementwise_sequence.hpp"
#include "flow.hpp"

#include "launch/launch.hpp"
#include "launch/xrt/session.hpp"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using ew::from_bf16;
using ew::to_bf16;

int main(int argc, char **argv) {
  std::string package;
  int n = 2 * ew::kBlockElements, iters = 50, pairs = 16;
  double seconds = 0;
  bool other = false;
  launch::SessionOptions so;
  for (int i = 1; i < argc; ++i) {
    const std::string opt = argv[i];
    if (opt == "--other") other = true;
    else if (i + 1 < argc && opt == "--package") package = argv[++i];
    else if (i + 1 < argc && opt == "--n") n = std::atoi(argv[++i]);
    else if (i + 1 < argc && opt == "--iters") iters = std::atoi(argv[++i]);
    else if (i + 1 < argc && opt == "--pairs") pairs = std::atoi(argv[++i]);
    else if (i + 1 < argc && opt == "--seconds") seconds = std::atof(argv[++i]);
    else if (i + 1 < argc && opt == "--quiet") so.resident_quiet = std::chrono::milliseconds(std::atoi(argv[++i]));
    else if (i + 1 < argc && opt == "--timeout") so.timeout = std::chrono::milliseconds(std::atoi(argv[++i]));
    else if (i + 1 < argc && opt == "--resident") {
      const std::string v = argv[++i];
      if (v != "on" && v != "off") {
        std::cerr << "--resident: on or off\n";
        return 2;
      }
      so.resident_lanes = v == "on";
    } else if (i + 1 < argc && opt == "--flow") {
      try {
        so.flow = ew::elf_flow(argv[++i]) ? launch::Flow::elf : launch::Flow::xclbin;
      } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 2;
      }
    } else {
      std::cerr << "unknown option '" << opt << "'\n";
      return 2;
    }
  }
  if (package.empty() || iters < 1 || pairs < 1) {
    std::cerr << "usage: session.exe --package DIR [--n N] [--iters N] [--pairs P] [--flow elf|xclbin]\n"
                 "                   [--resident on|off] [--other] [--seconds S] [--timeout MS] [--quiet MS]\n";
    return 2;
  }

  try {
    ew::check_length(n);
    const bool pm_ok = ew::blocks_per_core(n) % 2 == 0;

    launch::ConfigList list{package, launch::Transport::ctrlpkt, {}, {}};
    list.configs = {
        launch::cold("vadd", "cold_vadd"),
        launch::warm("vadd", "vmul", {}, "full_ab"),
        launch::warm("vmul", "vadd", {}, "full_ba"),
        launch::rerun("vadd", "rerun_vadd"),
    };
    if (pm_ok) {
      list.configs.push_back(launch::warm("vadd", "vmul", {.program_only = true}, "pm_ab"));
      list.configs.push_back(launch::warm("vmul", "vadd", {.program_only = true}, "pm_ba"));
    }
    auto stage = [n](launch::Rt &rt) { ew::stage(rt, n); };
    launch::Session s(list, {{"vadd", stage}, {"vmul", stage}}, so);
    std::printf("N=%d bf16 (%d x %d cores); %zu kernels built in %.1f ms; flow %s; resident lanes %s; guard %s\n", n,
                EW_COLS, EW_CORES, s.result().kernels.size(), s.result().build_ms,
                so.flow == launch::Flow::elf ? "elf" : "xclbin", so.resident_lanes ? "on" : "off",
                s.watching() ? "on" : ("off: " + s.watch_why()).c_str());

    const size_t bytes = (size_t)n * sizeof(uint16_t);
    std::vector<xrt::bo> bo;
    for (int i = 0; i < 7; ++i)
      bo.push_back(s.buffer(bytes, (unsigned)(i % 3)));
    xrt::bo &A = bo[0], &B = bo[1], &C = bo[2], &D = bo[3], &T1 = bo[4], &T2 = bo[5], &Y = bo[6];
    std::mt19937 gen(7);
    for (xrt::bo *in : {&A, &B, &C, &D}) {
      uint16_t *p = in->map<uint16_t *>();
      for (int i = 0; i < n; ++i)
        p[i] = to_bf16(-2.0f + 4.0f * (float)(gen() >> 8) / (float)(1u << 24));
      in->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    auto poison = [&] {
      for (xrt::bo *o : {&T1, &T2, &Y}) {
        std::memset(o->map<char *>(), 0xff, bytes);
        o->sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }
    };
    // Wrong elements of the three stages (host.cpp's check).
    auto wrong = [&] {
      for (xrt::bo *b : {&T1, &T2, &Y})
        b->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      struct Stage { xrt::bo &l, &r, &o; bool mul; };
      int bad = 0;
      for (const Stage &st : {Stage{A, B, T1, false}, Stage{T1, C, T2, true}, Stage{T2, D, Y, false}}) {
        const uint16_t *l = st.l.map<uint16_t *>(), *rh = st.r.map<uint16_t *>(), *o = st.o.map<uint16_t *>();
        for (int i = 0; i < n; ++i)
          bad += !ew::bf16_ok(l[i], rh[i], o[i], st.mul);
      }
      return bad;
    };
    auto L = [](const char *k, xrt::bo &l, xrt::bo &r, xrt::bo &o) {
      return launch::Launch{k, {l, r, o}};
    };

    int errs = 0;
    std::map<std::string, size_t> statuses;
    size_t resident = 0, substituted = 0, launches = 0, ok_wrong = 0;
    auto tally = [&](const launch::RunReport &r) {
      ++statuses[launch::status_name(r.status)];
      resident += r.resident;
      substituted += r.substituted;
      launches += r.ran.size();
    };
    auto report = [&](const char *what, const launch::RunReport &r, int bad) {
      std::printf("  %-26s %-17s resident %3zu cold %3zu  check %s\n", what, launch::status_name(r.status),
                  r.resident, r.substituted, bad ? "FAIL" : "PASS");
    };
    // Another process on the NPU (several session.exe at once) makes
    // other_context_ran: the Session saw it, nothing wrong was reported ok.
    auto acceptable = [](const launch::RunReport &r) {
      return r.status == launch::RunReport::Status::ok || r.status == launch::RunReport::Status::other_context_ran;
    };
    // The expression: always from cold(vadd), so it may follow anything.
    auto expression = [&](const char *ab, const char *ba) {
      poison();
      const launch::RunReport r = s.run({L("cold_vadd", A, B, T1), L(ab, T1, C, T2), L(ba, T2, D, Y)});
      tally(r);
      const int bad = r.status == launch::RunReport::Status::ok ? wrong() : 0;
      ok_wrong += bad ? 1 : 0;
      return std::pair{r, bad};
    };
    // 2P switches after the expression (the array holds vadd); T2 and Y again.
    auto alternate = [&](const char *ab, const char *ba) {
      std::vector<launch::Launch> ls;
      for (int i = 0; i < pairs; ++i) {
        ls.push_back(L(ab, T1, C, T2));
        ls.push_back(L(ba, T2, D, Y));
      }
      return ls;
    };

    // Twins only after SessionOptions::resident_quiet with no other context.
    std::this_thread::sleep_for(so.resident_quiet);
    std::printf("\nexpression and switches:\n");
    std::map<std::string, double> us;
    for (const auto &[name, ab, ba] : std::vector<std::tuple<std::string, const char *, const char *>>{
             {"full", "full_ab", "full_ba"}, {"pm", "pm_ab", "pm_ba"}}) {
      if (name == "pm" && !pm_ok)
        continue;
      // Again until ok (at most 3 times): the alternate needs its T1.
      for (int attempt = 0; attempt < 3; ++attempt) {
        auto [r, bad] = expression(ab, ba);
        report(("expression, " + name).c_str(), r, bad);
        errs += bad || !acceptable(r);
        if (r.status == launch::RunReport::Status::ok)
          break;
      }
      const std::vector<launch::Launch> ls = alternate(ab, ba);
      launch::RunReport a = s.run(ls); // untimed
      tally(a);
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        a = s.run(ls);
        tally(a);
        errs += !acceptable(a);
      }
      const std::chrono::duration<double, std::micro> dt = std::chrono::steady_clock::now() - t0;
      us["alternate, " + name] = dt.count() / iters / (double)ls.size();
      const int bad2 = wrong();
      report(("after alternate, " + name).c_str(), a, bad2);
      errs += bad2 != 0;
      ok_wrong += bad2 ? 1 : 0;
      if (so.resident_lanes && s.watching() && !a.other_context_seen && a.resident != ls.size()) {
        std::printf("    expected every switch as its resident twin (no other context)\n");
        ++errs;
      }
    }
    {
      std::vector<launch::Launch> ls(2 * (size_t)pairs, L("rerun_vadd", T1, C, T2));
      s.run(ls);
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i)
        tally(s.run(ls));
      const std::chrono::duration<double, std::micro> dt = std::chrono::steady_clock::now() - t0;
      us["rerun"] = dt.count() / iters / (double)ls.size();
    }
    std::printf("\nus per launch (through the Session):\n");
    for (const auto &[name, v] : us)
      std::printf("  %-28s %8.1f\n", name.c_str(), v);
    for (const char *kind : {"full", "pm"})
      if (us.count(std::string("alternate, ") + kind))
        std::printf("RESULT session_switch_%s %.2f\n", kind, us[std::string("alternate, ") + kind] - us["rerun"]);
    for (const auto &[name, v] : us) {
      std::string key = name;
      for (char &ch : key)
        if (!std::isalnum((unsigned char)ch))
          ch = '_';
      std::printf("RESULT session_%s %.2f\n", key.c_str(), v);
    }

    if (other) {
      // Another hw_context on the NPU: the guard sees it, so no warm launch
      // (twin or not) may run while it is open. After max_gap: within it the
      // Session asks only after the runlist (status other_context_ran).
      std::printf("\nnext to a second hw_context:\n");
      const auto gap = std::chrono::milliseconds(20);
      {
        xrt::xclbin xb(package + "/vmul/design.xclbin");
        s.device().register_xclbin(xb);
        xrt::hw_context ctx_b(s.device(), xb.get_uuid());
        std::this_thread::sleep_for(gap);
        auto [r, bad] = expression("full_ab", "full_ba");
        report("expression, other open", r, bad);
        if (!s.watching())
          std::printf("    (guard off: no check of the kernels it ran)\n");
        else if (r.resident || r.substituted != 2 || r.status != launch::RunReport::Status::ok || bad) {
          std::printf("    expected both switches as cold, status ok\n");
          ++errs;
        }
      }
      // The first runlist after it closed sees a change: cold. After resident_quiet: warm.
      std::this_thread::sleep_for(gap);
      auto [r0, bad0] = expression("full_ab", "full_ba");
      report("expression, other closed", r0, bad0);
      errs += bad0 || r0.status != launch::RunReport::Status::ok;
      std::this_thread::sleep_for(so.resident_quiet);
      auto [r, bad] = expression("full_ab", "full_ba");
      report("expression, then", r, bad);
      const size_t want = so.resident_lanes ? 2 : 0;
      if (s.watching() && (r.resident != want || r.substituted || r.status != launch::RunReport::Status::ok || bad)) {
        std::printf("    expected both switches warm (%zu as twins), status ok\n", want);
        ++errs;
      }
    }

    if (seconds > 0) {
      std::printf("\nloop for %.0f s:\n", seconds);
      const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
      size_t rounds = 0;
      const char *ab = pm_ok ? "pm_ab" : "full_ab", *ba = pm_ok ? "pm_ba" : "full_ba";
      while (std::chrono::steady_clock::now() < end) {
        auto [r, bad] = expression(ab, ba);
        if (r.status == launch::RunReport::Status::ok) {
          const launch::RunReport a = s.run(alternate(ab, ba));
          if (a.status == launch::RunReport::Status::failed)
            std::printf("  failed: %s\n", a.message.c_str());
          tally(a);
          if (a.status == launch::RunReport::Status::ok && wrong())
            ++ok_wrong;
        }
        if (r.status == launch::RunReport::Status::stopped)
          s.reopen();
        ++rounds;
      }
      std::printf("  %zu rounds\n", rounds);
    }

    std::printf("\nlaunches %zu: resident %zu, cold instead of warm %zu; reopens %zu\n", launches, resident,
                substituted, s.reopens());
    std::printf("statuses:");
    for (const auto &[st, c] : statuses)
      std::printf(" %s %zu", st.c_str(), c);
    std::printf("\nRESULT ok_but_wrong %zu\n", ok_wrong);
    const bool pass = errs == 0 && ok_wrong == 0;
    std::printf("SESSION %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
