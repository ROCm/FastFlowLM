//===- stream_probe.cpp - run dumped instruction streams as ELFs ----------===//
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
//   stream_probe.exe XCLBIN [--other XCLBIN2] STREAM.txn [STREAM.txn ...]
//   (environment PROBE_WAIT_MS: how long a run may take, default 5000; PROBE_SLEEP_MS: a pause after
//   each run, for status snapshots of the idle array; PROBE_FILL=1: buffers 1 and 2 get fixed random
//   data, buffer 0 a fixed pattern before each run, and the line ends with a hash of buffer 0 after
//   it: two healthy runs of one stream on the same inputs give the same hash; an argument seed:N
//   refills buffers 1 and 2 with sequence N, seed:1 being the first; PROBE_DUMP=DIR: buffer 0 after
//   run n goes to DIR/n.bin)
//
// One hw_context on XCLBIN (and with --other a second one on XCLBIN2: a STREAM
// written other:PATH runs there, as FLM runs two xclbins today); each STREAM (a TXN binary, e.g. one dumped by the
// engine with FLM_WHISPER_SWITCH_DUMP) runs once, in order, as an ELF (txn::to_elf) the
// way FLM's npu_app runs it, with three 8 MB buffers as arguments 3, 4, 5.
// Prints the state of each run.
//
//===----------------------------------------------------------------------===//

#include "launch/txn/elf.hpp"
#include "launch/txn/txn.hpp"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: stream_probe.exe XCLBIN STREAM.txn [STREAM.txn ...]\n";
    return 2;
  }
  xrt::device dev(0);
  xrt::xclbin xb{std::string(argv[1])};
  dev.register_xclbin(xb);
  xrt::hw_context ctx(dev, xb.get_uuid());
  int first = 2;
  xrt::hw_context other;
  if (argc > 3 && std::string(argv[2]) == "--other") {
    xrt::xclbin xb2{std::string(argv[3])};
    dev.register_xclbin(xb2);
    other = xrt::hw_context(dev, xb2.get_uuid());
    first = 4;
  }
  std::vector<xrt::bo> bufs;
  for (int i = 0; i < 3; i++)
    bufs.emplace_back(dev, (size_t)8 << 20, XRT_BO_FLAGS_HOST_ONLY, 0);
  const bool fill = std::getenv("PROBE_FILL") != nullptr;
  auto fill_inputs = [&](uint32_t seed) {
    for (int i = 1; i < 3; i++) {
      uint32_t *p = bufs[i].map<uint32_t *>(), x = 12345u * (uint32_t)i * seed;
      for (size_t k = 0; k < ((size_t)8 << 20) / 4; k++) {
        x = x * 1664525u + 1013904223u;          // a fixed random sequence
        p[k] = (x & 0x3fff3fffu) | 0x3c003c00u;  // two bf16 in [1, 2): finite, no overflow
      }
      bufs[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
  };
  if (fill)
    fill_inputs(1);
  // FNV-1a of buffer 0 (after a run)
  auto hash0 = [&]() {
    bufs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const uint8_t *p = bufs[0].map<uint8_t *>();
    uint64_t h = 1469598103934665603ull;
    for (size_t k = 0; k < ((size_t)8 << 20); k++)
      h = (h ^ p[k]) * 1099511628211ull;
    return h;
  };
  for (int a = first; a < argc; a++) {
    std::string path = argv[a];
    if (path.rfind("seed:", 0) == 0) {
      fill_inputs((uint32_t)std::atoi(path.c_str() + 5));
      std::printf("   inputs: %s\n", path.c_str());
      continue;
    }
    const bool in_other = path.rfind("other:", 0) == 0;
    if (in_other)
      path = path.substr(6);
    std::ifstream in(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), words.size() * 4);
    const std::vector<char> elf = launch::txn::to_elf(words);
    xrt::module mod{xrt::elf(elf.data(), elf.size())};
    xrt::ext::kernel k(in_other ? other : ctx, mod, "MLIR_AIE");
    const auto t0 = std::chrono::steady_clock::now();
    if (fill) {
      std::memset(bufs[0].map<void *>(), 0x5a, (size_t)8 << 20);
      bufs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    try {
      xrt::run r = k(3u, 0, 0, bufs[0], bufs[1], bufs[2]);
      static const int wait_ms = std::getenv("PROBE_WAIT_MS") ? std::atoi(std::getenv("PROBE_WAIT_MS")) : 5000;
      const auto state = r.wait(std::chrono::milliseconds(wait_ms));
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("%2d %-56s %6zu words: state %d after %.1f ms (4 = completed)", a - first + 1, argv[a], words.size(),
                  (int)state, ms);
      if (fill)
        std::printf("  buf0 %016llx", (unsigned long long)hash0());
      if (const char *dir = std::getenv("PROBE_DUMP")) {
        std::ofstream out(std::string(dir) + "/" + std::to_string(a - first + 1) + ".bin", std::ios::binary);
        out.write(bufs[0].map<const char *>(), (size_t)8 << 20);
      }
      std::printf("\n");
    } catch (const std::exception &e) {
      std::printf("%2d %-56s FAILED: %s\n", a - first + 1, argv[a], e.what());
      return 1;
    }
    std::fflush(stdout);
    if (const char *sleep_ms = std::getenv("PROBE_SLEEP_MS"))
      std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(sleep_ms)));
  }
  return 0;
}
