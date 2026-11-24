//===- op_cost.cpp - what one TXN op costs the command processor ----*- C++ -*-===//
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
// Measures the cost of each kind of op the kernels write, on the NPU: a stream
// of N ops and one of 2N ops of one kind, each run ITERS times (start, wait);
// the cost of one op is the difference of the medians over N (the command's
// own cost cancels). Block writes: several sizes, then a straight line
// (cost = per op + per word * words). The numbers feed launch::CostModel
// (launch.hpp).
//
// The array is reset first (the library's reset kernel: cores held, channels
// reset), so the writes reach registers no one uses: core-tile locks, mem-tile
// data memory, core channel Ctrl, core program memory (the cores are held).
// A reset kernel runs again at the end; the next app needs a cold kernel.
//
//   op_cost.exe --package DIR --app NAME [--n N] [--iters I]
//
// Prints "COST <kind> <ns per op>" and "COST blockwrite_<target> <ns per op> <ns per word>".
//
//===----------------------------------------------------------------------===//

#include "launch/launch.hpp"

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

using launch::Tile;
namespace hw = launch::hw;
namespace txn = launch::txn;

int main(int argc, char **argv) {
  std::string package, app;
  int n = 2000, iters = 30;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string opt = argv[i];
    if (opt == "--package") package = argv[i + 1];
    else if (opt == "--app") app = argv[i + 1];
    else if (opt == "--n") n = std::atoi(argv[i + 1]);
    else if (opt == "--iters") iters = std::atoi(argv[i + 1]);
  }
  if (package.empty() || app.empty() || n < 1 || iters < 3) {
    std::cerr << "usage: op_cost.exe --package DIR --app NAME [--n N] [--iters I]\n";
    return 2;
  }
  try {
    launch::ConfigList list{package, launch::Transport::txn, {launch::reset("reset")}};
    list.configs.push_back(launch::cold(app)); // (names the app: the reset kernel's header and xclbin)
    const launch::Result r = launch::build(list, {{app, [](launch::Rt &) {}}});
    const txn::Header header = txn::parse(r.kernels[0].words).header;

    xrt::device device(0);
    xrt::xclbin xclbin(r.base_xclbin);
    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());
    xrt::kernel kernel(context, "MLIR_AIE");

    // Median us of one run of `words` (start, wait), after 2 untimed runs.
    auto time_us = [&](const std::vector<uint32_t> &words) {
      xrt::bo bo(device, words.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
      std::memcpy(bo.map<void *>(), words.data(), words.size() * 4);
      bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      xrt::run run(kernel);
      run.set_arg(0, 3u);
      run.set_arg(1, bo);
      run.set_arg(2, (uint32_t)words.size());
      std::vector<double> t;
      for (int i = 0; i < iters + 2; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        run.start();
        run.wait();
        if (i >= 2)
          t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
      }
      std::sort(t.begin(), t.end());
      return t[t.size() / 2];
    };
    auto stream = [&](const std::vector<txn::Op> &ops) {
      txn::Stream s;
      s.header = header;
      s.ops = ops;
      return txn::encode(s);
    };
    const std::vector<uint32_t> reset = r.kernels[0].words;
    time_us(reset);

    // ns per op of `make(i)`, i = 0 .. count-1, from count and 2 * count ops.
    auto per_op = [&](int count, const std::function<txn::Op(int)> &make) {
      std::vector<txn::Op> once, twice;
      for (int i = 0; i < 2 * count; ++i) {
        if (i < count)
          once.push_back(make(i));
        twice.push_back(make(i));
      }
      return 1000.0 * (time_us(stream(twice)) - time_us(stream(once))) / count;
    };
    const int lock_count = hw::locks(hw::Kind::core).count;
    auto core_tile = [](int i) { return Tile{i % hw::kCols, hw::kFirstCoreRow + (i / hw::kCols) % 4}; };
    auto lock_addr = [&](int i) {
      return core_tile(i).addr(hw::locks(hw::Kind::core).at((uint32_t)(i / 32) % lock_count));
    };
    auto mem_word = [](int i) { return Tile{i % hw::kCols, 1}.addr(0x40000 + 4 * (uint32_t)((i / hw::kCols) % 0x8000)); };
    auto channel_ctrl = [&](int i) {
      return core_tile(i).addr(hw::channels(hw::Kind::core, (i / 32) % 2 ? hw::Dir::mm2s : hw::Dir::s2mm).at(
          (uint32_t)(i / 64) % 2));
    };

    std::printf("COST write32_lock %.1f\n", per_op(n, [&](int i) { return txn::Write32{lock_addr(i), 0}; }));
    std::printf("COST write32_memdm %.1f\n", per_op(n, [&](int i) { return txn::Write32{mem_word(i), (uint32_t)i}; }));
    std::printf("COST write32_channel %.1f\n", per_op(n, [&](int i) { return txn::Write32{channel_ctrl(i), 0}; }));
    std::printf("COST maskwrite32_lock %.1f\n",
                per_op(n, [&](int i) { return txn::MaskWrite32{lock_addr(i), 0, 0x3f}; }));

    // Block writes: per op + per word, a least-squares line over the sizes.
    for (const char *target : {"memdm", "pm"}) {
      std::vector<std::pair<double, double>> pts; // (words, ns per op)
      for (uint32_t words : {16u, 64u, 256u, 1024u}) {
        const int count = std::max(32, 32 * n / (int)words); // (~64K words per size at the default n)
        const double ns = per_op(count, [&](int i) {
          const uint32_t addr = std::string(target) == "pm"
                                    ? core_tile(i).addr(hw::kPm + 4 * ((uint32_t)(i / 32) * words % (hw::kPmBytes / 4 - words)))
                                    : Tile{i % hw::kCols, 1}.addr(0x40000 + 4 * ((uint32_t)(i / 8) * words % 0x8000));
          return txn::BlockWrite{addr, std::vector<uint32_t>(words, (uint32_t)i)};
        });
        std::printf("COST blockwrite_%s_%u %.1f\n", target, words, ns);
        pts.push_back({(double)words, ns});
      }
      double sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (auto [x, y] : pts)
        sx += x, sy += y, sxx += x * x, sxy += x * y;
      const double k = pts.size(), slope = (k * sxy - sx * sy) / (k * sxx - sx * sx), icpt = (sy - slope * sx) / k;
      std::printf("COST blockwrite_%s %.1f %.3f\n", target, icpt, slope);
    }
    time_us(reset);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
