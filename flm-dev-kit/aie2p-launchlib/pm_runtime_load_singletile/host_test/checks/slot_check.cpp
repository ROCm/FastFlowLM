// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// slot_check: the vtable offsets of device::lookup_query and
// request::get(device) in XRT's own headers (core/common) and in NpuWatch's
// stand-ins (npu_watch.hpp, detail::xrt_mirror) must agree. Needs the XRT source
// tree at the XRT to check; touches no device.
#include "core/common/device.h"
#include "core/common/query.h"
#define LAUNCH_NPU_WATCH_XRT 1
#include "launch/xrt/npu_watch.hpp"
namespace list_mirror = launch::xrt_support::detail::xrt_mirror;
#include <cstdint>
#include <cstdio>
#include <cstring>

// lookup_query is private: an explicit instantiation may name it, and hands
// its address out through a friend.
struct lookup_query_tag {};
auto lookup_query_address(lookup_query_tag);
template <auto M> struct expose {
  friend auto lookup_query_address(lookup_query_tag) { return M; }
};
template struct expose<&xrt_core::device::lookup_query>;

// The byte offset in the vtable that a pointer to a virtual member function calls.
template <typename P> long vtable_offset(P pmf) {
#if defined(_MSC_VER)
  // A pointer to the vcall thunk: mov rax,[rcx]; then jmp [rax+N] or (with
  // control-flow guard) mov rax,[rax+N]; an incremental-link stub (jmp rel32) first.
  const unsigned char *p;
  std::memcpy(&p, &pmf, sizeof p);
  if (p[0] == 0xE9) {
    int32_t rel;
    std::memcpy(&rel, p + 1, 4);
    p += 5 + rel;
  }
  if (p[0] != 0x48 || p[1] != 0x8B || p[2] != 0x01)
    return -1;
  p += 3;
  const bool jmp = p[0] == 0xFF, mov = p[0] == 0x48 && p[1] == 0x8B;
  const unsigned char modrm = jmp ? p[1] : mov ? p[2] : 0;
  const unsigned char *disp = jmp ? p + 2 : p + 3;
  if (modrm == 0x20 || modrm == 0x00)
    return 0;
  if (modrm == 0x60 || modrm == 0x40)
    return (signed char)disp[0];
  if (modrm == 0xA0 || modrm == 0x80) {
    int32_t d;
    std::memcpy(&d, disp, 4);
    return d;
  }
  return -1;
#else
  struct { uintptr_t ptr; ptrdiff_t adj; } r; // Itanium: 1 + the offset for a virtual
  static_assert(sizeof r == sizeof pmf);
  std::memcpy(&r, &pmf, sizeof r);
  return (r.ptr & 1) ? (long)(r.ptr - 1) : -1;
#endif
}

int main() {
  using xrt_core::device;
  using xrt_core::query::request;
  const long lq = vtable_offset(lookup_query_address(lookup_query_tag{}));
  const long lq_m = vtable_offset(&list_mirror::device::lookup_query);
  const long get = vtable_offset(static_cast<std::any (request::*)(const device *) const>(&request::get));
  const long get_m =
      vtable_offset(static_cast<std::any (list_mirror::request::*)(const void *) const>(&list_mirror::request::get));
  std::printf("device::lookup_query: XRT %ld, mirror %ld\n", lq, lq_m);
  std::printf("request::get(device): XRT %ld, mirror %ld\n", get, get_m);
  const bool ok = lq >= 0 && get >= 0 && lq == lq_m && get == get_m;
  std::printf("slot_check: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
