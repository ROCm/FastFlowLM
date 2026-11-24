// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// ctx_probe: what XRT's own list of hardware contexts (the query behind
// `xrt-smi examine --report aie-partitions`) reports, and what one call costs,
// with nothing of ours on the NPU, with our context open, and with a second
// process holding one too. On Linux it also prints NpuWatch's list next to it.
//
// It opens contexts and launches nothing, so it cannot hang the NPU.
//
//   ctx_probe --xclbin precompiled/c8/vadd/design.xclbin [--calls 1000]
//   ctx_probe --hold <xclbin> <ms>     the second process (started by the probe)
//
// The query (xrt_core::query::aie_partition_info) is XRT-internal: it needs the
// XRT source headers of the exact XRT the program runs against.
#include "core/common/device.h"
#include "core/common/query_requests.h"

#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>

#include "spawn.hpp"

#if defined(__linux__)
#include "launch/xrt/npu_watch.hpp"
#include <unistd.h>
#else
#include <windows.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

using Partitions = xrt_core::query::aie_partition_info::result_type;

int own_pid() {
#if defined(_WIN32)
  return (int)GetCurrentProcessId();
#else
  return (int)::getpid();
#endif
}

Partitions query(const xrt::device &dev) {
  return xrt_core::device_query<xrt_core::query::aie_partition_info>(dev.get_handle());
}

int others(const Partitions &p) {
  return (int)std::count_if(p.begin(), p.end(), [](const auto &e) { return e.pid != own_pid(); });
}

void print(const char *what, const Partitions &p) {
  std::printf("  XRT list %s: %zu context(s)\n", what, p.size());
  for (const auto &e : p)
    std::printf("    pid %d%s  name '%s'  id '%s'  cols %llu+%llu  submitted %llu  completed %llu  suspended %d  errors %llu\n",
                e.pid, e.pid == own_pid() ? " (us)" : "", e.process_name.c_str(), e.metadata.id.c_str(),
                (unsigned long long)e.start_col, (unsigned long long)e.num_cols,
                (unsigned long long)e.command_submissions, (unsigned long long)e.command_completions, (int)e.is_suspended,
                (unsigned long long)e.errors);
}

// Microseconds per call: min, median, max over `calls` calls.
template <class F> void time_calls(const char *what, int calls, F &&f) {
  std::vector<double> us;
  us.reserve(calls);
  for (int i = 0; i < calls; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(us.begin(), us.end());
  std::printf("  %s: %d calls, us per call min %.1f  median %.1f  max %.1f\n", what, calls, us.front(),
              us[us.size() / 2], us.back());
}

void state(const char *name, const xrt::device &dev, int calls) {
  std::printf("%s\n", name);
  print("", query(dev));
  time_calls("XRT query", calls, [&] { (void)query(dev); });
#if defined(__linux__)
  launch::xrt_support::NpuWatch watch;
  if (const auto s = watch.snapshot()) {
    std::printf("  NpuWatch: %zu context(s):", s->size());
    for (const auto &[k, n] : *s)
      std::printf(" pid %lld id %u submitted %llu;", (long long)k.pid, k.id, (unsigned long long)n);
    std::printf("\n");
    time_calls("NpuWatch", calls, [&] { (void)watch.snapshot(); });
  } else {
    std::printf("  NpuWatch: not available\n");
  }
#endif
}

xrt::hw_context open_context(xrt::device &dev, const std::string &path) {
  const xrt::xclbin xclbin(path);
  dev.register_xclbin(xclbin);
  return xrt::hw_context(dev, xclbin.get_uuid());
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--hold") {
      xrt::device dev(0);
      xrt::hw_context ctx = open_context(dev, argv[2]);
      std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(argv[3])));
      return 0;
    }
    std::string xclbin;
    int calls = 1000;
    for (int i = 1; i + 1 < argc; i += 2) {
      const std::string a = argv[i];
      if (a == "--xclbin")
        xclbin = argv[i + 1];
      else if (a == "--calls")
        calls = std::atoi(argv[i + 1]);
    }
    if (xclbin.empty()) {
      std::fprintf(stderr, "usage: ctx_probe --xclbin <design.xclbin> [--calls N]\n");
      return 2;
    }

    xrt::device dev(0);
    std::printf("our pid %d\n", own_pid());
    state("1. nothing of ours on the NPU", dev, calls);
    {
      xrt::hw_context ctx = open_context(dev, xclbin);
      state("2. our context open", dev, calls);

      const int before = others(query(dev));
      ew::Child child(argv[0], {"--hold", xclbin, "20000"});
      const auto t0 = std::chrono::steady_clock::now();
      bool seen = false;
      while (!seen && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) {
        seen = others(query(dev)) > before;
        if (!seen)
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("second process's context %s in the XRT list after %.0f ms\n", seen ? "seen" : "NOT seen", ms);
      state("3. our context and a second process's", dev, calls);
      child.stop();
      child.wait();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::printf("4. both closed\n");
    print("", query(dev));
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ctx_probe: %s\n", e.what());
    return 1;
  }
}
