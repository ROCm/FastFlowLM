//===- host.cpp - y = (A + B) * C + D with the launch library -----*- C++ -*-===//
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
// Three launches on the same cores of one hw_context, each on its own buffers:
//
//   kernel 0  cold(vadd)            T1 = A + B    resets the array, loads vadd
//   kernel 1  warm(vadd, vmul)      T2 = T1 * C   switches to vmul (reset, load)
//   kernel 2  warm(vmul, vadd)      Y  = T2 + D   switches back
//   kernel 3  warm(vadd, vmul, pm)  T2 = T1 * C   the program only
//   kernel 4  warm(vmul, vadd, pm)  Y  = T2 + D   the program only
//   kernel 5  rerun(vadd)                         the sequence only
//   kernel 6  rerun(vmul)                         (for the two-context baseline)
//
// Then the cost of a switch, per launch (runlists of 2P launches):
//   alternate full  1, 2, 1, 2, ...   every launch switches (whole load)
//   alternate pm    3, 4, 3, 4, ...   every launch switches (program only)
//   rerun           5, 5, ...         no switch
//   switch cost     alternate - rerun
// and the FLM way: vadd.xclbin and vmul.xclbin in two hw_contexts.
//
//   host.exe --package DIR [--n N] [--iters N] [--pairs P] [--flow elf|xclbin] [--skip-baseline]
//            [--build-only] [--transport txn|ctrlpkt]
//
// --build-only builds the kernels, prints what the switches write (BREAKDOWN
// lines: part, register class, ops, words) and stops before XRT.
//
// --flow (default elf): how a launch reaches the NPU (include/flow.hpp): an ELF
// per kernel (xrt::module, xrt::ext::kernel), or the xclbin's kernel with an
// instruction BO. --elf is --flow elf.
//
// --transport (default ctrlpkt): how the kernels load (launch::Transport). A
// ctrlpkt kernel also takes its control packets (launch/xrt/packets.hpp).
//
//===----------------------------------------------------------------------===//

#include "bf16.hpp"
#include "elementwise_sequence.hpp"
#include "flow.hpp"

#include "launch/launch.hpp"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

constexpr unsigned kOpcode = 3; // "run this instruction stream"

using ew::from_bf16;
using ew::to_bf16;

int main(int argc, char **argv) {
  std::string package;
  int n = 2 * ew::kBlockElements, iters = 50, pairs = 16;
  bool use_elf = true, baseline = true, build_only = false, preempt = false;
  launch::Transport transport = launch::Transport::ctrlpkt;
  launch::CtrlPktOptions ctrlpkt;
  for (int i = 1; i < argc; ++i) {
    const std::string opt = argv[i];
    if (opt == "--elf") use_elf = true;
    else if (i + 1 < argc && opt == "--flow") {
      try {
        use_elf = ew::elf_flow(argv[++i]);
      } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 2;
      }
    }
    else if (opt == "--preempt") preempt = true;
    else if (opt == "--build-only") build_only = true;
    else if (opt == "--keep-lanes") ctrlpkt.rewrite_lanes = false;
    else if (opt == "--skip-baseline") baseline = false;
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
    std::cerr << "usage: host.exe --package DIR [--n N] [--iters N] [--pairs P] [--flow elf|xclbin]\n"
                 "                [--skip-baseline] [--build-only] [--preempt] [--transport txn|ctrlpkt]\n";
    return 2;
  }

  try {
    ew::check_length(n);
    // The program-only switch needs every launch to leave the ping/pong DMA
    // chains at ping: an even number of blocks per core.
    const bool pm_ok = ew::blocks_per_core(n) % 2 == 0;

    // 1. The kernels.
    launch::ConfigList list{package, transport, {}, ctrlpkt};
    list.configs = {
        launch::cold("vadd"),
        launch::warm("vadd", "vmul"),
        launch::warm("vmul", "vadd"),
        launch::warm("vadd", "vmul", {.program_only = true}),
        launch::warm("vmul", "vadd", {.program_only = true}),
        launch::rerun("vadd"),
        launch::rerun("vmul"),
    };

    // 2. The runtime sequence of each app, then the kernels, built in memory.
    auto stage = [n, preempt](launch::Rt &rt) { ew::stage(rt, n, preempt); };
    const launch::Result r = launch::build(list, {{"vadd", stage}, {"vmul", stage}});
    std::printf("N=%d bf16 (%d x %d cores, %d blocks of %d per core); %zu kernels (%s) built in %.1f ms:\n", n,
                EW_COLS, EW_CORES, ew::blocks_per_core(n), EW_N, r.kernels.size(), launch::transport_name(transport),
                r.build_ms);
    for (const launch::Kernel &k : r.kernels)
      std::printf("  %-20s %6zu ops %7zu B (program %zu B, packets %zu B), ~%.1f us of ops\n", k.name.c_str(),
                  k.stats.ops, k.stats.bytes, k.stats.program_bytes, k.packets.size() * 4, launch::estimate_us(k));
    // What a full and a program-only switch write, part by part.
    for (int k : {1, 3}) {
      std::printf("breakdown of %s (part, what, ops, words, estimated us):\n", r.kernels[k].name.c_str());
      for (const launch::BreakdownRow &row : launch::breakdown(r.kernels[k]))
        std::printf("  BREAKDOWN %s %-9s %-13s %6zu %7zu %7.1f\n", r.kernels[k].name.c_str(), row.part.c_str(),
                    row.what.c_str(), row.ops, row.words, row.us);
    }
    if (build_only)
      return 0;

    // 3. XRT: one hw_context on the base xclbin; a kernel is an ELF or an
    // instruction BO (--flow).
    xrt::device device(0);
    xrt::xclbin xclbin(r.base_xclbin);
    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());
    xrt::kernel kernel(context, "MLIR_AIE");
    const ew::FlowKernels kernels(device, context, kernel, r, use_elf);
    std::printf("flow %s\n", ew::flow_name(use_elf));

    // Buffers: the four inputs, the two temporaries and the result.
    const size_t bytes = (size_t)n * sizeof(uint16_t);
    std::vector<xrt::bo> bo;
    for (int i = 0; i < 7; ++i)
      bo.emplace_back(device, bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3 + i % 3));
    xrt::bo &A = bo[0], &B = bo[1], &C = bo[2], &D = bo[3], &T1 = bo[4], &T2 = bo[5], &Y = bo[6];

    // Inputs in [-2, 2); outputs poisoned, so a stage that never ran fails.
    // Straight from mt19937 (std::uniform_real_distribution differs between
    // standard libraries), so every platform runs the same numbers.
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

    // Each stage against what the part gave the next one, one ulp of slack for
    // the narrowing to bf16, +0 = -0. A wrong program is a different number
    // (T1 + C for T1 * C), not one ulp off.
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
      // FNV-1a of the result, to compare the bits across platforms.
      uint64_t h = 1469598103934665603ull;
      const unsigned char *y = Y.map<unsigned char *>();
      for (size_t i = 0; i < bytes; ++i)
        h = (h ^ y[i]) * 1099511628211ull;
      std::printf("  %-28s check %s  Y fnv %016llx\n", what.c_str(), bad ? "FAIL" : "PASS", (unsigned long long)h);
      errs += bad ? 1 : 0;
    };

    // 4. One xrt::run per launch; a runlist keeps references, so runs never move.
    struct Launch { int kernel; xrt::bo *l, *r, *o; };
    std::vector<std::unique_ptr<xrt::run>> runs;
    auto make_run = [&](const Launch &l) -> xrt::run & {
      std::unique_ptr<xrt::run> run = kernels.run(l.kernel);
      run->set_arg(3 + ew::kArgLhs, *l.l);
      run->set_arg(3 + ew::kArgRhs, *l.r);
      run->set_arg(3 + ew::kArgOut, *l.o);
      runs.push_back(std::move(run));
      return *runs.back();
    };
    auto make_runlist = [&](const std::vector<Launch> &ls) {
      auto rl = std::make_unique<xrt::runlist>(context);
      for (const Launch &l : ls)
        rl->add(make_run(l));
      return rl;
    };
    auto repeat = [&](const std::vector<Launch> &unit) {
      std::vector<Launch> all;
      for (int i = 0; i < pairs; ++i)
        all.insert(all.end(), unit.begin(), unit.end());
      return all;
    };
    // Mean time of one launch of `rl` (n launches), after one untimed execution.
    auto time_us = [&](xrt::runlist &rl, int launches) {
      rl.execute();
      rl.wait();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        rl.execute();
        rl.wait();
      }
      const std::chrono::duration<double, std::micro> dt = std::chrono::steady_clock::now() - t0;
      return dt.count() / iters / launches;
    };

    // The order of the runlists keeps every warm kernel after its prev: each
    // one starts on vadd's state and ends on it.
    std::map<std::string, double> us;
    std::printf("\nin one hw_context, runlists of %d launches, %d iters:\n", 2 * pairs, iters);
    auto expression = [&](int k1, int k2, const std::string &name) {
      poison();
      auto rl = make_runlist({{0, &A, &B, &T1}, {k1, &T1, &C, &T2}, {k2, &T2, &D, &Y}});
      rl->execute();
      rl->wait();
      check("expression, " + name);
      us["expression, " + name] = time_us(*rl, 3);
    };
    auto alternate = [&](int k1, int k2, const std::string &name) {
      auto rl = make_runlist(repeat({{k1, &T1, &C, &T2}, {k2, &T2, &D, &Y}}));
      us["alternate, " + name] = time_us(*rl, 2 * pairs);
      check("after alternate, " + name); // T2 and Y again, from the T1 of the expression
    };
    expression(1, 2, "full");
    alternate(1, 2, "full");
    if (pm_ok) {
      expression(3, 4, "pm");
      alternate(3, 4, "pm");
    } else {
      std::printf("  (program-only switch skipped: %d blocks per core is odd)\n", ew::blocks_per_core(n));
    }
    {
      // The same buffers as alternate; vadd both times, so T2 is wrong after
      // it: the expression after it computes everything again.
      auto rl = make_runlist(repeat({{5, &T1, &C, &T2}, {5, &T2, &D, &Y}}));
      us["rerun"] = time_us(*rl, 2 * pairs);
    }
    expression(1, 2, "full, after rerun");

    std::printf("\nus per launch:\n");
    for (const auto &[name, v] : us)
      std::printf("  %-28s %8.1f\n", name.c_str(), v);
    std::printf("switch cost = alternate - rerun, us per launch:\n");
    for (const char *kind : {"full", "pm"}) {
      if (!us.count(std::string("alternate, ") + kind))
        continue;
      const double cost = us[std::string("alternate, ") + kind] - us["rerun"];
      std::printf("  %-28s %8.1f\n", (std::string("SWITCH COST, ") + kind).c_str(), cost);
      std::printf("RESULT switch_%s %.2f\n", kind, cost);
    }
    for (const auto &[name, v] : us) {
      std::string key = name;
      for (char &ch : key)
        if (!std::isalnum((unsigned char)ch))
          ch = '_';
      std::printf("RESULT %s %.2f\n", key.c_str(), v);
    }

    if (baseline) {
      // The FLM way: vmul.xclbin in a second hw_context. Runs of two contexts
      // cannot share a runlist, so one launch at a time, for both. The cores of
      // a freshly loaded xclbin already loop, so a rerun kernel runs on it.
      xrt::xclbin xb(package + "/vmul/design.xclbin");
      device.register_xclbin(xb);
      xrt::hw_context ctx_b(device, xb.get_uuid());
      xrt::kernel kernel_b(ctx_b, "MLIR_AIE");
      xrt::bo instr_b(device, r.kernels[6].words.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel_b.group_id(1));
      std::memcpy(instr_b.map<void *>(), r.kernels[6].words.data(), r.kernels[6].words.size() * 4);
      instr_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      xrt::run run_b(kernel_b);
      run_b.set_arg(0, kOpcode);
      run_b.set_arg(1, instr_b);
      run_b.set_arg(2, (uint32_t)r.kernels[6].words.size());
      run_b.set_arg(3, T1);
      run_b.set_arg(4, C);
      run_b.set_arg(5, T2);
      xrt::run &run_a = make_run({5, &T2, &D, &Y});
      auto time_one = [&](const std::function<void(int)> &launch_i) {
        launch_i(0);
        launch_i(1);
        const int count = iters * 2 * pairs;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < count; ++i)
          launch_i(i);
        const std::chrono::duration<double, std::micro> dt = std::chrono::steady_clock::now() - t0;
        return dt.count() / count;
      };
      std::printf("\nbaseline, one launch at a time (start + wait):\n");
      const double one = time_one([&](int) { run_a.start(); run_a.wait(); });
      const double two = time_one([&](int i) {
        xrt::run &rr = (i % 2) ? run_b : run_a;
        rr.start();
        rr.wait();
      });
      std::printf("  %-28s %8.1f\n  %-28s %8.1f\n  %-28s %8.1f   two contexts - one context\n",
                  "one context, rerun", one, "two contexts, alternate", two, "CONTEXT SWITCH COST", two - one);
      std::printf("RESULT ctx_one %.2f\nRESULT ctx_two %.2f\nRESULT ctx_switch %.2f\n", one, two, two - one);
      // T2 = T1 * C (context B), then Y = T2 + D (context A): check them.
      run_b.start();
      run_b.wait();
      run_a.start();
      run_a.wait();
      check("two contexts");
    }
    return errs ? 1 : 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
