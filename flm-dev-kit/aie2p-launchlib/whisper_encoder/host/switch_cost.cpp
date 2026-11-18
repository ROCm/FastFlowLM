//===- switch_cost.cpp - Whisper encoder: warm switch vs hw_context switch -===//
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
//   switch_cost.exe --package build/pkg [--count N] [--transport txn|ctrlpkt]
//
// Step 1 of the Whisper trial: what does it cost to go from the encoder's
// matmul (encoder_mm.xclbin) to its attention (encoder_attn.xclbin) and back?
//
//   FLM today   two hw_contexts, one per xclbin; a launch in the other one makes
//               the firmware put that xclbin in the array (a context switch)
//   launchlib   one hw_context; warm(mm, attn) and warm(attn, mm) kernels
//
// No data moves: every kernel's runtime sequence is empty, so a launch costs
// its switch and the launch itself. Each launch is one start + wait, as FLM's
// npu_app does. Per launch (mean and p50, us):
//
//   rerun in one context            the floor of a launch (no switch)
//   warm, alternating               a warm switch each launch
//   two contexts, alternating       a context switch each launch
//
// --transport (default ctrlpkt): how the launchlib kernels load (launch::Transport).
//
//===----------------------------------------------------------------------===//

#include "launch/launch.hpp"
#include "launch/xrt/packets.hpp"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

/// An instruction BO for kernel `k`, in the memory group of `kern`.
xrt::bo instr_bo(xrt::device &dev, const xrt::kernel &kern, const launch::Kernel &k) {
  xrt::bo bo(dev, k.words.size() * 4, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
  std::memcpy(bo.map<void *>(), k.words.data(), k.words.size() * 4);
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return bo;
}

/// One run of an instruction stream with no buffers.
xrt::run make_run(const xrt::kernel &kern, const xrt::bo &instr, size_t words) {
  xrt::run r(kern);
  r.set_arg(0, 3u); // opcode: run this instruction stream
  r.set_arg(1, instr);
  r.set_arg(2, (uint32_t)words);
  return r;
}

/// Mean, p50, p90 and max of `count` calls of `step(i)` after 8 untimed ones, in
/// us. `after(i)`, if given, runs after each step, untimed.
void time_steps(const char *what, int count, const std::function<void(int)> &step,
                const std::function<void(int)> &after = {}) {
  for (int i = 0; i < 8; ++i) {
    step(i);
    if (after)
      after(i);
  }
  std::vector<double> us;
  for (int i = 0; i < count; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    step(i);
    us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    if (after)
      after(i);
  }
  double mean = 0;
  for (double v : us)
    mean += v / (double)us.size();
  std::sort(us.begin(), us.end());
  std::printf("  %-44s mean %8.1f  p50 %8.1f  p90 %8.1f  max %8.1f\n", what, mean, us[us.size() / 2],
              us[us.size() * 9 / 10], us.back());
}

} // namespace

int main(int argc, char **argv) {
  std::string package;
  int count = 400;
  launch::Transport transport = launch::Transport::ctrlpkt;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string opt = argv[i];
    if (opt == "--package") package = argv[i + 1];
    else if (opt == "--count") count = std::stoi(argv[i + 1]);
    else if (opt == "--transport") transport = launch::transport_from_name(argv[i + 1]);
  }
  if (package.empty() || count < 2) {
    std::cerr << "usage: switch_cost.exe --package DIR [--count N] [--transport txn|ctrlpkt]\n"
                 "       (N >= 2: each direction is timed N/2 times)\n";
    return 2;
  }
  try {
    // 1. The kernels, with empty runtime sequences.
    launch::ConfigList list{package, transport,
                            {launch::cold("mm"), launch::warm("mm", "attn"), launch::warm("attn", "mm"),
                             launch::rerun("mm"), launch::rerun("attn")}};
    const auto none = [](launch::Rt &) {};
    const launch::Result r = launch::build(list, {{"mm", none}, {"attn", none}});
    std::printf("%zu kernels (%s) built in %.1f ms:\n", r.kernels.size(), launch::transport_name(transport),
                r.build_ms);
    for (const launch::Kernel &k : r.kernels)
      std::printf("  %-14s %6zu ops %8zu B (program %7zu B, packets %7zu B)\n", k.name.c_str(), k.stats.ops,
                  k.stats.bytes, k.stats.program_bytes, k.packets.size() * 4);

    // 2. One context on mm.xclbin: reruns, and warm switches mm <-> attn.
    xrt::device dev(0);
    xrt::xclbin xb_mm(package + "/mm/design.xclbin"), xb_attn(package + "/attn/design.xclbin");
    dev.register_xclbin(xb_mm);
    dev.register_xclbin(xb_attn);
    std::printf("\nus per launch (one start + wait each, no data):\n");
    {
      xrt::hw_context ctx(dev, xb_mm.get_uuid());
      xrt::kernel k(ctx, "MLIR_AIE");
      std::vector<xrt::bo> instr;
      const std::vector<xrt::bo> packets = launch::packet_buffers(dev, k, r);
      std::vector<xrt::run> runs;
      for (size_t i = 0; i < r.kernels.size(); ++i) {
        instr.push_back(instr_bo(dev, k, r.kernels[i]));
        runs.push_back(make_run(k, instr.back(), r.kernels[i].words.size()));
        launch::set_packets(runs.back(), r.kernels[i], packets[i]);
      }
      auto launch_one = [&](size_t i) {
        runs[i].start();
        runs[i].wait();
      };
      launch_one(0); // cold(mm)
      time_steps("one context, rerun(mm)", count, [&](int) { launch_one(3); });
      // warm(mm, attn), warm(attn, mm), ...: the array is mm's at every even step
      // (time_steps runs an even number of untimed steps first).
      time_steps("one context, warm mm <-> attn (alternating)", count, [&](int i) { launch_one(i % 2 ? 2 : 1); });
      if (count % 2)
        launch_one(2); // an odd count ends with attn in the array: back to mm, untimed
      // Each direction alone: the other switch runs untimed before it.
      time_steps("  of them mm -> attn", count / 2, [&](int) { launch_one(1); }, [&](int) { launch_one(2); });
      launch_one(1); // the array is attn's before each timed attn -> mm
      time_steps("  of them attn -> mm", count / 2, [&](int) { launch_one(2); }, [&](int) { launch_one(1); });
      launch_one(2); // back to mm
    }

    // 3. FLM today: one context per xclbin, alternating.
    {
      xrt::hw_context ctx_mm(dev, xb_mm.get_uuid()), ctx_attn(dev, xb_attn.get_uuid());
      xrt::kernel k_mm(ctx_mm, "MLIR_AIE"), k_attn(ctx_attn, "MLIR_AIE");
      // Each context runs the rerun kernel of its own app (the xclbin's own state).
      xrt::bo i_mm = instr_bo(dev, k_mm, r.kernels[3]), i_attn = instr_bo(dev, k_attn, r.kernels[4]);
      xrt::run r_mm = make_run(k_mm, i_mm, r.kernels[3].words.size());
      xrt::run r_attn = make_run(k_attn, i_attn, r.kernels[4].words.size());
      time_steps("two contexts, the mm one only", count, [&](int) {
        r_mm.start();
        r_mm.wait();
      });
      time_steps("two contexts, alternating (FLM today)", count, [&](int i) {
        xrt::run &run = i % 2 ? r_attn : r_mm;
        run.start();
        run.wait();
      });
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
