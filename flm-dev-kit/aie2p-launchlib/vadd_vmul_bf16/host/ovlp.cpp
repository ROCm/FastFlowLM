// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// ovlp.cpp - an overlay app back from a guest with its op set up (APP@STATE)
//
//   ovlp.exe --package DIR [--n N] [--iters N] [--pairs P] [--transport txn|ctrlpkt] [--build-only]
//
// DIR holds vadd, vmul and the overlay ovl with its state add (make ovlp:
// host/make_overlay.cpp, make_package.py --state). vmul is the guest; the add
// launch after it comes back three ways, in one hw_context:
//
//   kernel                     the add launch writes
//   warm(vmul, vadd)           vadd, the whole app (reference)
//   warm(vmul, ovl) + add      the overlay's base, then add's configuration and
//                              sequence as TXN writes joined to the kernel (an
//                              overlay engine without states, as Qwen4's before ovl@P)
//   warm(vmul, "ovl@add")      the overlay with add set up; the sequence only runs
//
// and "ovl@add" stays set up: rerun("ovl@add") runs add again with no switch.
//
// The configuration is not ovl's runtime sequence: with control packets, the
// library may leave a lane in a switch register the overlay leaves at 0, and add
// takes some of them (the kernel's check refuses such a sequence). The next
// warm kernel writes the lanes again (CtrlPktOptions::rewrite_lanes). ovl@add
// has add's routes in its plan, so its lanes go around them.
// Each way: y = (A + B) * C + D checked, then alternating guest / add launches
// timed per launch against reruns.

#include "bf16.hpp"
#include "elementwise_sequence.hpp"

#include "launch/launch.hpp"
#include "launch/xrt/packets.hpp"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

constexpr unsigned kOpcode = 3;

using ew::from_bf16;
using ew::to_bf16;

int main(int argc, char **argv) {
  std::string package;
  int n = 2 * ew::kBlockElements, iters = 50, pairs = 16;
  bool build_only = false;
  launch::Transport transport = launch::Transport::ctrlpkt;
  for (int i = 1; i < argc; ++i) {
    const std::string opt = argv[i];
    if (opt == "--build-only") build_only = true;
    else if (i + 1 < argc && opt == "--package") package = argv[++i];
    else if (i + 1 < argc && opt == "--n") n = std::atoi(argv[++i]);
    else if (i + 1 < argc && opt == "--iters") iters = std::atoi(argv[++i]);
    else if (i + 1 < argc && opt == "--pairs") pairs = std::atoi(argv[++i]);
    else if (i + 1 < argc && opt == "--transport") {
      try {
        transport = launch::transport_from_name(argv[++i]);
      } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 2;
      }
    }
  }
  if (package.empty() || iters < 1 || pairs < 1) {
    std::cerr << "usage: ovlp.exe --package DIR [--n N] [--iters N] [--pairs P] [--transport txn|ctrlpkt]"
                 " [--build-only]\n";
    return 2;
  }

  try {
    ew::check_length(n);

    // 1. The kernels. Every warm kernel follows one that left its prev in the
    // array; ovl + add's configuration is the array of vadd and of ovl@add.
    enum { kCold, kToVmul, kToVadd, kOvlToVmul, kToOvl, kOvlAddToVmul, kToOvlAdd, kRerunVadd, kRerunOvlAdd };
    launch::ConfigList list{package, transport, {}};
    list.configs = {
        launch::cold("vadd"),
        launch::warm("vadd", "vmul"),
        launch::warm("vmul", "vadd"),
        launch::warm("ovl", "vmul"),
        launch::warm("vmul", "ovl"),
        launch::warm("ovl@add", "vmul"),
        launch::warm("vmul", "ovl@add"),
        launch::rerun("vadd"),
        launch::rerun("ovl@add"),
    };

    // 2. The sequences; the plain overlay has none (below).
    auto stage = [n](launch::Rt &rt) { ew::stage(rt, n); };
    launch::Result r =
        launch::build(list, {{"vadd", stage}, {"vmul", stage}, {"ovl", [](launch::Rt &) {}}, {"ovl@add", stage}});
    std::printf("N=%d bf16 (%d x %d cores); %zu kernels (%s) built in %.1f ms:\n", n, EW_COLS, EW_CORES,
                r.kernels.size(), launch::transport_name(transport), r.build_ms);
    for (const launch::Kernel &k : r.kernels)
      std::printf("  %-22s %6zu ops %7zu B (packets %6zu B), ~%.1f us of ops\n", k.name.c_str(), k.stats.ops,
                  k.stats.bytes, k.packets.size() * 4, launch::estimate_us(k));
    // The plain overlay's launch: the kernel, add's configuration (the state's
    // writes), add's sequence (rerun(vadd)'s: ovl has vadd's shim channels).
    const launch::txn::Stream add = launch::detail::read_txn(package + "/ovl/state_add.txn");
    {
      launch::txn::Stream joined = launch::txn::parse(r.kernels[kToOvl].words);
      joined.ops.insert(joined.ops.end(), add.ops.begin(), add.ops.end());
      const launch::txn::Stream seq = launch::txn::parse(r.kernels[kRerunVadd].words);
      joined.ops.insert(joined.ops.end(), seq.ops.begin(), seq.ops.end());
      r.kernels[kToOvl].words = launch::txn::encode(joined);
      std::printf("  %-22s %6zu ops %7zu B: + add's configuration (%zu ops) and sequence\n", "vmul_then_ovl + add",
                  joined.ops.size(), r.kernels[kToOvl].words.size() * 4, add.ops.size());
    }
    // ovl@add is vadd: the same kernels both ways.
    auto same = [&](int a, int b) {
      return r.kernels[a].words == r.kernels[b].words && r.kernels[a].packets == r.kernels[b].packets;
    };
    std::printf("warm(vmul, ovl@add) %s warm(vmul, vadd); warm(ovl@add, vmul) %s warm(vadd, vmul)\n",
                same(kToOvlAdd, kToVadd) ? "==" : "!=", same(kOvlAddToVmul, kToVmul) ? "==" : "!=");
    for (int k : {kToOvl, kToOvlAdd}) {
      std::printf("breakdown of %s (part, what, ops, words, estimated us):\n", r.kernels[k].name.c_str());
      for (const launch::BreakdownRow &row : launch::breakdown(r.kernels[k]))
        std::printf("  BREAKDOWN %s %-9s %-13s %6zu %7zu %7.1f\n", r.kernels[k].name.c_str(), row.part.c_str(),
                    row.what.c_str(), row.ops, row.words, row.us);
    }
    if (build_only)
      return 0;

    // 3. XRT, as host.cpp.
    xrt::device device(0);
    xrt::xclbin xclbin(r.base_xclbin);
    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());
    xrt::kernel kernel(context, "MLIR_AIE");
    std::vector<xrt::bo> instr;
    for (const launch::Kernel &k : r.kernels) {
      xrt::bo &bo = instr.emplace_back(device, k.words.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
      std::memcpy(bo.map<void *>(), k.words.data(), k.words.size() * 4);
      bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    const std::vector<xrt::bo> packets = launch::packet_buffers(device, kernel, r);

    const size_t bytes = (size_t)n * sizeof(uint16_t);
    std::vector<xrt::bo> bo;
    for (int i = 0; i < 7; ++i)
      bo.emplace_back(device, bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3 + i % 3));
    xrt::bo &A = bo[0], &B = bo[1], &C = bo[2], &D = bo[3], &T1 = bo[4], &T2 = bo[5], &Y = bo[6];
    std::mt19937 gen(7);
    for (xrt::bo *in : {&A, &B, &C, &D}) {
      uint16_t *p = in->map<uint16_t *>();
      for (int i = 0; i < n; ++i)
        p[i] = to_bf16(-2.0f + 4.0f * (float)(gen() >> 8) / (float)(1u << 24));
      in->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    auto poison = [&](std::initializer_list<xrt::bo *> outs) {
      for (xrt::bo *o : outs) {
        std::memset(o->map<char *>(), 0xff, bytes);
        o->sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }
    };
    int errs = 0;
    auto check = [&](const std::string &what) {
      for (xrt::bo *b : {&T1, &T2, &Y})
        b->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      struct Stage { const char *name; xrt::bo &l, &r, &o; bool mul; };
      int bad = 0;
      for (const Stage &st : {Stage{"T1=A+B", A, B, T1, false}, Stage{"T2=T1*C", T1, C, T2, true},
                              Stage{"Y=T2+D", T2, D, Y, false}}) {
        const uint16_t *l = st.l.map<uint16_t *>(), *rh = st.r.map<uint16_t *>(), *o = st.o.map<uint16_t *>();
        for (int i = 0; i < n; ++i)
          if (!ew::bf16_ok(l[i], rh[i], o[i], st.mul) && ++bad <= 3) {
            const float a = from_bf16(l[i]), b = from_bf16(rh[i]);
            std::printf("    %s [%d]: got %g, want %g\n", st.name, i, from_bf16(o[i]), st.mul ? a * b : a + b);
          }
      }
      std::printf("  %-36s check %s\n", what.c_str(), bad ? "FAIL" : "PASS");
      errs += bad ? 1 : 0;
    };

    struct Launch { int kernel; xrt::bo *l, *r, *o; };
    std::vector<std::unique_ptr<xrt::run>> runs;
    auto make_run = [&](const Launch &l) -> xrt::run & {
      auto run = std::make_unique<xrt::run>(kernel);
      run->set_arg(0, kOpcode);
      run->set_arg(1, instr[l.kernel]);
      run->set_arg(2, (uint32_t)r.kernels[l.kernel].words.size());
      run->set_arg(3 + ew::kArgLhs, *l.l);
      run->set_arg(3 + ew::kArgRhs, *l.r);
      run->set_arg(3 + ew::kArgOut, *l.o);
      launch::set_packets(*run, r.kernels[l.kernel], packets[l.kernel]);
      runs.push_back(std::move(run));
      return *runs.back();
    };
    auto make_runlist = [&](const std::vector<Launch> &ls) {
      auto rl = std::make_unique<xrt::runlist>(context);
      for (const Launch &l : ls)
        rl->add(make_run(l));
      return rl;
    };
    auto execute = [](xrt::runlist &rl) {
      rl.execute();
      rl.wait();
    };
    auto time_us = [&](xrt::runlist &rl, int launches) {
      execute(rl);
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i)
        execute(rl);
      const std::chrono::duration<double, std::micro> dt = std::chrono::steady_clock::now() - t0;
      return dt.count() / iters / launches;
    };
    auto repeat = [&](const std::vector<Launch> &unit) {
      std::vector<Launch> all;
      for (int i = 0; i < pairs; ++i)
        all.insert(all.end(), unit.begin(), unit.end());
      return all;
    };

    // 4. Each way: the expression from a cold start, then (for ovl@add) the op
    // again without a switch, then guest / add alternating.
    struct Way { const char *name; int to_guest, back; };
    const Way ways[] = {{"vadd (whole app)", kToVmul, kToVadd},
                        {"ovl, the host writes add", kOvlToVmul, kToOvl},
                        {"ovl@add", kOvlAddToVmul, kToOvlAdd}};
    std::map<std::string, double> us;
    std::printf("\nin one hw_context, runlists of %d launches, %d iters:\n", 2 * pairs, iters);
    for (const Way &w : ways) {
      poison({&T1, &T2, &Y});
      execute(*make_runlist({{kCold, &A, &B, &T1}, {kToVmul, &T1, &C, &T2}, {w.back, &T2, &D, &Y}}));
      check(std::string("expression, ") + w.name);
      if (w.back == kToOvlAdd) {
        poison({&Y});
        execute(*make_runlist({{kRerunOvlAdd, &T2, &D, &Y}}));
        check("rerun(ovl@add), no switch");
      }
      auto rl = make_runlist(repeat({{w.to_guest, &T1, &C, &T2}, {w.back, &T2, &D, &Y}}));
      poison({&T2, &Y});
      us[std::string("alternate, ") + w.name] = time_us(*rl, 2 * pairs);
      check(std::string("after alternate, ") + w.name);
    }
    {
      auto rl = make_runlist(repeat({{kRerunVadd, &T2, &D, &Y}, {kRerunVadd, &T2, &D, &Y}}));
      us["rerun"] = time_us(*rl, 2 * pairs);
    }

    std::printf("\nus per launch (guest / add alternating; rerun: add only, no switch):\n");
    for (const Way &w : ways) {
      const double v = us[std::string("alternate, ") + w.name];
      std::printf("  %-28s %8.1f   switch cost %7.1f\n", w.name, v, v - us["rerun"]);
    }
    std::printf("  %-28s %8.1f\n", "rerun", us["rerun"]);
    const char *keys[] = {"vadd", "ovl", "ovl_add"};
    for (int i = 0; i < 3; ++i)
      std::printf("RESULT alternate_%s %.2f\nRESULT switch_%s %.2f\n", keys[i],
                  us[std::string("alternate, ") + ways[i].name], keys[i],
                  us[std::string("alternate, ") + ways[i].name] - us["rerun"]);
    std::printf("RESULT rerun %.2f\n", us["rerun"]);
    return errs ? 1 : 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
