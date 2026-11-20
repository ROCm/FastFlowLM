// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// list_probe: XRT's context list with public XRT headers only, as
// NpuWatch reads it (npu_watch.hpp, LAUNCH_NPU_WATCH_XRT=1): the query keys by the
// RTTI name of their request object, the list decoded at each element size that
// divides its length, then NpuWatch's own snapshot. Opens a context (optional),
// launches nothing.
//
//   list_probe [-v] [<xclbin>]     -v: every key's request type
#include <any>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <typeinfo>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#define LAUNCH_NPU_WATCH_XRT 1
#include "launch/xrt/npu_watch.hpp"
#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

namespace mirror = launch::xrt_support::detail::xrt_mirror;

int main(int argc, char **argv) {
  const bool all = argc > 1 && std::strcmp(argv[1], "-v") == 0;
  const char *xclbin = argc > 1 + all ? argv[1 + all] : nullptr;
  xrt::device dev(0u);
  std::printf("host: %s\n", dev.get_info<xrt::info::device::host>().c_str());
  auto h = dev.get_handle();
  const auto *d = reinterpret_cast<const mirror::device *>(h.get());
  std::printf("device type: %s\n", typeid(*d).name());
  std::optional<xrt::hw_context> ctx;
  if (xclbin) {
    xrt::xclbin x{std::string(xclbin)};
    dev.register_xclbin(x);
    ctx.emplace(dev, x.get_uuid());
    std::printf("opened one context\n");
  }
  int found = -1;
  for (int k = 0; k < 400; ++k) {
    try {
      const mirror::request &r = d->lookup_query((mirror::device::key_type)k);
      const char *n = typeid(r).name();
      const bool hit = std::strstr(n, "aie_partition_info") != nullptr;
      if (all || hit)
        std::printf("key %3d: %s\n", k, n);
      if (hit)
        found = k;
    } catch (const std::exception &e) {
      if (all)
        std::printf("key %3d: throws %s\n", k, e.what());
    } catch (...) {
      if (all)
        std::printf("key %3d: throws\n", k);
    }
  }
  if (found < 0) {
    std::printf("no key found\n");
    return 1;
  }
  const mirror::request &r = d->lookup_query((mirror::device::key_type)found);
  const std::any a = r.get(h.get());
  std::printf("any type: %s\n", a.type().name());
  const auto &v = std::any_cast<const std::vector<xrt_core::query::aie_partition_info::data> &>(a);
  std::printf("bytes %zu; own pid %d\n", v.size(), (int)getpid());
  for (size_t s : launch::xrt_support::detail::kStrides) {
    std::vector<launch::xrt_support::detail::Entry> es;
    if (v.size() % s)
      continue;
    const bool ok = v.empty() || launch::xrt_support::detail::decode(&v[0].byte, v.size(), s, &es);
    std::printf("stride %zu: %zu entries, %s\n", s, v.size() / s, ok ? "all contexts" : "NOT contexts");
    for (const auto &e : es)
      std::printf("  pid %lld id %u submissions %llu\n", (long long)e.pid, e.id, (unsigned long long)e.submissions);
  }
  const launch::xrt_support::NpuWatch watch;
  std::printf("NpuWatch: %s%s\n", watch.available() ? "available" : "not available: ", watch.why().c_str());
  int status = watch.available() ? 0 : 1;
  if (const auto s = watch.snapshot()) {
    bool own = false;
    for (const auto &[k, n] : *s) {
      std::printf("  pid %lld id %u submissions %llu\n", (long long)k.pid, k.id, (unsigned long long)n);
      own |= k.pid == (int64_t)getpid();
    }
    if (ctx && !own) {
      std::printf("NpuWatch: our own context is not in the list\n");
      status = 1;
    }
  } else if (watch.available()) {
    std::printf("NpuWatch: the query failed: %s\n", watch.last_error().c_str());
    status = 1;
  }
  return status;
}
