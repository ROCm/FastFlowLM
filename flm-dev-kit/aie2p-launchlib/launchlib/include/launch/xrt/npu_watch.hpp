//===- npu_watch.hpp - the hw_contexts on the NPU, from the driver -*- C++ -*-===//
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
// `NpuWatch` lists every hw_context on the NPU, of every process, with the
// number of commands each one has submitted: the amdxdna driver's
// DRM_IOCTL_AMDXDNA_GET_ARRAY (DRM_AMDXDNA_HW_CONTEXT_ALL). Session
// (session.hpp) uses it to see whether another context may use the array:
//
//   Snapshot s = watch.snapshot(mine);   // every context but `mine`
//   ... a runlist ...
//   watch.snapshot(mine) != s            // one came, went, or submitted a command
//
// Measured (amdxdna-dkms 7.0.0-rc1, npu5): the list has the contexts of other
// processes too; context ids are per process and are used again after a close;
// the per-context `preemptions` and `suspensions` counters stay 0 when another
// context takes the columns, so the submission counts are what tells.
//
// One gap: a context that opens, runs and closes between two snapshots is not
// seen. Without /dev/accel, or on another OS without the XRT source below,
// `available()` is false. A query that fails gives no snapshot (std::nullopt),
// never an empty one: an empty snapshot means "no other context", and the
// caller must not read a failure that way.
//
// With LAUNCH_NPU_WATCH_XRT=1 (Windows; Linux too, by choice) the list
// comes from XRT's own query, xrt_core::query::aie_partition_info, what
// `xrt-smi examine --report aie-partitions` shows. That query is XRT-internal
// and its ABI moves between XRT builds; this file needs public XRT headers only
// and finds at run time what moves:
//   - the query's key (78, 80, 81, 82 or 84 across XRT 2.17 .. master, not by
//     version): the one whose request object's RTTI name has
//     "aie_partition_info";
//   - the element size (200, 248 or 264 bytes from XRT 2.19 on): the one that
//     divides the list's length and gives plausible entries, kept once known;
//     the fields read (start_col, num_cols, pid, command_submissions, the id
//     string) are at the same offsets in all three.
// What it takes as fixed: the vtable slots of device::lookup_query and
// request::get(device) (detail::xrt_mirror; the same from XRT 2.19 to master,
// so older XRTs are refused) and the std::string/std::vector layouts of a
// release build (MSVC with _ITERATOR_DEBUG_LEVEL 0, or libstdc++). It follows no
// pointer it reads from an entry. Measured (Windows 11, NPU driver
// 32.0.20102.3930, XRT 2.21.0; Linux, XRT 2.21.75): every context of every
// process, with its process id and context id (unique on the machine on
// Windows); ~160-190 us a query on Windows, of which ~0.6% throw while another
// process opens and closes contexts (retried here, twice).
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if !defined(LAUNCH_NPU_WATCH_XRT)
#define LAUNCH_NPU_WATCH_XRT 0
#endif

#if LAUNCH_NPU_WATCH_XRT
#include <any>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <typeinfo>
#include <xrt/xrt_device.h>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif
// The name of XRT's element type (core/common/query_requests.h), so that
// std::any_cast takes XRT's vector (type_info compares names); one byte, so the
// vector's size() is its length in bytes. A file that includes XRT's
// query_requests.h must not include this one with the XRT source on.
namespace xrt_core::query {
struct aie_partition_info {
  struct data {
    unsigned char byte;
  };
};
} // namespace xrt_core::query
#elif defined(__linux__)
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace launch::xrt_support {

/// A hw_context: the process that made it and its id in that process.
struct ContextKey {
  int64_t pid;
  uint32_t id;
  auto operator<=>(const ContextKey &) const = default;
};

/// Every context but the excluded ones -> commands it has submitted.
using Snapshot = std::map<ContextKey, uint64_t>;

namespace detail {

/// One context of the list, whichever source it came from.
struct Entry {
  int64_t pid;
  uint32_t id;
  uint64_t submissions;
};

inline int64_t own_pid() {
#if defined(_WIN32) && LAUNCH_NPU_WATCH_XRT
  return (int64_t)::_getpid();
#elif defined(__linux__) || LAUNCH_NPU_WATCH_XRT
  return (int64_t)::getpid();
#else
  return -1;
#endif
}

#if !LAUNCH_NPU_WATCH_XRT
// The uapi layout (include/uapi/drm/amdxdna_accel.h), so that no kernel header
// is needed to build.
struct GetArray {
  uint32_t param;        // DRM_AMDXDNA_HW_CONTEXT_ALL = 0
  uint32_t element_size; // in: the most we take; out: the driver's
  uint32_t num_element;  // in: the most we take; out: how many
  uint32_t pad;
  uint64_t buffer;
};
// The start of struct amdxdna_drm_hwctx_entry (the rest is not read).
struct HwctxEntryHead {
  uint32_t context_id, start_col, num_col, hwctx_id;
  int64_t pid;
  uint64_t command_submissions, command_completions;
};
constexpr unsigned long kIoctlGetArray =
    (3ul << 30) | (sizeof(GetArray) << 16) | ('d' << 8) | (0x40 + 10); // DRM_IOWR(DRM_COMMAND_BASE + 10, ...)
constexpr uint32_t kMaxElementSize = 512, kMaxElements = 256;
#else
/// The string value of `"key": "..."` in XRT's host info (its first one: XRT's own).
inline std::string json_value(const std::string &json, const std::string &key) {
  const size_t k = json.find("\"" + key + "\"");
  const size_t a = k == std::string::npos ? k : json.find('"', json.find(':', k) + 1);
  const size_t b = a == std::string::npos ? a : json.find('"', a + 1);
  return b == std::string::npos ? std::string() : json.substr(a + 1, b - a - 1);
}

/// Stand-ins for XRT's xrt_core::device and query::request (core/common): only
/// their vtable slots up to the two calls made. Checked against XRT's headers by
/// pm_runtime_load_singletile's slot_check (gcc and MSVC).
namespace xrt_mirror {
struct request { // query.h: the same overloads in the same order
  enum class modifier { subdev, entry };
  virtual ~request() {}
  virtual std::any get(const void *device) const = 0;
  virtual std::any get(const void *, const std::any &) const = 0;
  virtual std::any get(const void *, const std::any &, const std::any &) const = 0;
  virtual std::any get(const void *, modifier, const std::string &) const = 0;
  virtual std::any get(const void *, const std::any &, const std::any &, const std::any &) const = 0;
};
// device.h: ishim's 48 virtuals (ishim.h, XRT 2.19 .. master; no destructor),
// ~device, 4 more, then lookup_query.
#define LAUNCH_XRT_MIRROR_V(n) virtual void v##n() = 0;
struct device {
  LAUNCH_XRT_MIRROR_V(0) LAUNCH_XRT_MIRROR_V(1) LAUNCH_XRT_MIRROR_V(2) LAUNCH_XRT_MIRROR_V(3)
  LAUNCH_XRT_MIRROR_V(4) LAUNCH_XRT_MIRROR_V(5) LAUNCH_XRT_MIRROR_V(6) LAUNCH_XRT_MIRROR_V(7)
  LAUNCH_XRT_MIRROR_V(8) LAUNCH_XRT_MIRROR_V(9) LAUNCH_XRT_MIRROR_V(10) LAUNCH_XRT_MIRROR_V(11)
  LAUNCH_XRT_MIRROR_V(12) LAUNCH_XRT_MIRROR_V(13) LAUNCH_XRT_MIRROR_V(14) LAUNCH_XRT_MIRROR_V(15)
  LAUNCH_XRT_MIRROR_V(16) LAUNCH_XRT_MIRROR_V(17) LAUNCH_XRT_MIRROR_V(18) LAUNCH_XRT_MIRROR_V(19)
  LAUNCH_XRT_MIRROR_V(20) LAUNCH_XRT_MIRROR_V(21) LAUNCH_XRT_MIRROR_V(22) LAUNCH_XRT_MIRROR_V(23)
  LAUNCH_XRT_MIRROR_V(24) LAUNCH_XRT_MIRROR_V(25) LAUNCH_XRT_MIRROR_V(26) LAUNCH_XRT_MIRROR_V(27)
  LAUNCH_XRT_MIRROR_V(28) LAUNCH_XRT_MIRROR_V(29) LAUNCH_XRT_MIRROR_V(30) LAUNCH_XRT_MIRROR_V(31)
  LAUNCH_XRT_MIRROR_V(32) LAUNCH_XRT_MIRROR_V(33) LAUNCH_XRT_MIRROR_V(34) LAUNCH_XRT_MIRROR_V(35)
  LAUNCH_XRT_MIRROR_V(36) LAUNCH_XRT_MIRROR_V(37) LAUNCH_XRT_MIRROR_V(38) LAUNCH_XRT_MIRROR_V(39)
  LAUNCH_XRT_MIRROR_V(40) LAUNCH_XRT_MIRROR_V(41) LAUNCH_XRT_MIRROR_V(42) LAUNCH_XRT_MIRROR_V(43)
  LAUNCH_XRT_MIRROR_V(44) LAUNCH_XRT_MIRROR_V(45) LAUNCH_XRT_MIRROR_V(46) LAUNCH_XRT_MIRROR_V(47)
  virtual ~device() {}
  LAUNCH_XRT_MIRROR_V(d0) LAUNCH_XRT_MIRROR_V(d1) LAUNCH_XRT_MIRROR_V(d2) LAUNCH_XRT_MIRROR_V(d3)
  enum class key_type {};
  virtual const request &lookup_query(key_type) const = 0;
};
#undef LAUNCH_XRT_MIRROR_V
} // namespace xrt_mirror

/// "2.21.0" -> 221; 0 when it does not parse.
inline int xrt_version(const std::string &v) {
  const size_t dot = v.find('.');
  if (dot == std::string::npos || dot == 0)
    return 0;
  return std::atoi(v.c_str()) * 100 + std::atoi(v.c_str() + dot + 1);
}

/// The element sizes of aie_partition_info::data from XRT 2.19 on; the offsets
/// below are the same in each (metadata = two std::string first).
constexpr size_t kStrides[] = {200, 248, 264};
constexpr size_t kStartCol = 64, kNumCols = 72, kPid = 80, kSubmissions = 96, kMaxCols = 64;

/// The context id: metadata.id, a std::string at the start of the entry, read
/// without following its pointer (short strings only: XRT's ids are decimal).
inline bool read_id(const unsigned char *e, uint32_t &id) {
  size_t size = 0;
  const char *s = nullptr;
#if defined(_MSC_VER)
  size_t cap; // { union { char buf[16]; char *ptr; }; size_t size, capacity; }
  std::memcpy(&size, e + 16, sizeof size);
  std::memcpy(&cap, e + 24, sizeof cap);
  if (cap != 15)
    return false;
  s = (const char *)e;
#else
  uintptr_t ptr; // libstdc++: { char *ptr; size_t size; char buf[16]; }
  std::memcpy(&ptr, e, sizeof ptr);
  std::memcpy(&size, e + 8, sizeof size);
  if (ptr != (uintptr_t)(e + 16))
    return false;
  s = (const char *)e + 16;
#endif
  if (size == 0 || size > 10)
    return false;
  uint64_t v = 0;
  for (size_t i = 0; i < size; ++i) {
    if (s[i] < '0' || s[i] > '9')
      return false;
    v = v * 10 + (uint64_t)(s[i] - '0');
  }
  if (v > 0xffffffffu)
    return false;
  id = (uint32_t)v;
  return true;
}

/// The list as `stride`-byte entries; false when one of them is not a context.
inline bool decode(const unsigned char *p, size_t bytes, size_t stride, std::vector<Entry> *out) {
  if (bytes % stride)
    return false;
  std::vector<Entry> v;
  for (size_t i = 0; i < bytes / stride; ++i) {
    const unsigned char *e = p + i * stride;
    uint64_t start, cols, subs;
    int32_t pid;
    uint32_t id;
    std::memcpy(&start, e + kStartCol, 8);
    std::memcpy(&cols, e + kNumCols, 8);
    std::memcpy(&pid, e + kPid, 4);
    std::memcpy(&subs, e + kSubmissions, 8);
    if (cols == 0 || cols > kMaxCols || start >= kMaxCols || start + cols > kMaxCols || pid <= 0 || !read_id(e, id))
      return false;
    v.push_back({(int64_t)pid, id, subs});
  }
  if (out)
    out->insert(out->end(), v.begin(), v.end());
  return true;
}
#endif

} // namespace detail

class NpuWatch {
public:
  /// `path`: the driver (Linux ioctl source); empty: no watch.
  explicit NpuWatch(const std::string &path = "/dev/accel/accel0") {
    if (path.empty()) {
      why_ = "no driver path given";
      return;
    }
#if LAUNCH_NPU_WATCH_XRT
#if defined(_MSC_VER) && _ITERATOR_DEBUG_LEVEL != 0
    why_ = "a debug build (_ITERATOR_DEBUG_LEVEL != 0) does not read XRT's std::vector";
    return;
#elif !defined(_MSC_VER) && !defined(__GLIBCXX__)
    why_ = "XRT's context list is read with MSVC's or libstdc++'s layouts only";
    return;
#else
    try {
      dev_.emplace(0u);
      const std::string version = detail::json_value(dev_->get_info<xrt::info::device::host>(), "version");
      if (detail::xrt_version(version) < 219) {
        why_ = "XRT " + version + ": the context list needs XRT 2.19 or newer";
        dev_.reset();
        return;
      }
      handle_ = dev_->get_handle();
      const auto *d = reinterpret_cast<const detail::xrt_mirror::device *>(handle_.get());
      // The key whose request is aie_partition_info's: the known ones first.
      auto is_list = [&](int k) {
        try {
          const char *name = typeid(d->lookup_query((detail::xrt_mirror::device::key_type)k)).name();
          return std::strstr(name, "aie_partition_info") != nullptr;
        } catch (...) { // no such key
          return false;
        }
      };
      for (int k : {80, 82, 84, 81, 78})
        if (key_ < 0 && is_list(k))
          key_ = k;
      for (int k = 0; key_ < 0 && k < 400; ++k)
        key_ = is_list(k) ? k : -1;
      if (key_ < 0) {
        why_ = "XRT " + version + ": no aie_partition_info query";
        close();
        return;
      }
      ok_ = true;
    } catch (const std::exception &e) {
      why_ = std::string("XRT: ") + e.what();
      close();
      return;
    }
#endif
#elif defined(__linux__)
    fd_ = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    ok_ = fd_ >= 0;
    if (!ok_)
      why_ = "cannot open " + path;
#else
    why_ = "no context list on this OS (build with LAUNCH_NPU_WATCH_XRT=1)";
#endif
    if (ok_ && !list(nullptr)) {
      ok_ = false;
      if (why_.empty())
        why_ = "the context list query failed" + (error_.empty() ? std::string() : ": " + error_);
      close();
    }
  }
  ~NpuWatch() { close(); }
  NpuWatch(const NpuWatch &) = delete;
  NpuWatch &operator=(const NpuWatch &) = delete;

  /// False when the driver cannot be asked (no device, old driver, an XRT older
  /// than 2.19, or no source on this OS); why() says which.
  bool available() const { return ok_; }
  const std::string &why() const { return why_; }
  /// Why the last query that failed did (XRT source only).
  const std::string &last_error() const { return error_; }

  /// Every context on the NPU now but those in `exclude`; std::nullopt when the
  /// driver cannot be asked or the query fails.
  std::optional<Snapshot> snapshot(const std::vector<ContextKey> &exclude = {}) const {
    Snapshot s;
    std::vector<detail::Entry> all;
    if (!list(&all))
      return std::nullopt;
    for (const detail::Entry &e : all) {
      const ContextKey k{e.pid, e.id};
      bool skip = false;
      for (const ContextKey &x : exclude)
        skip |= x == k;
      if (!skip)
        s[k] = e.submissions;
    }
    return s;
  }

  /// The contexts of this process now.
  std::vector<ContextKey> own() const {
    std::vector<ContextKey> out;
    if (const std::optional<Snapshot> s = snapshot())
      for (const auto &[k, n] : *s)
        if (k.pid == detail::own_pid())
          out.push_back(k);
    return out;
  }

private:
  bool list(std::vector<detail::Entry> *out) const {
    if (!ok_)
      return false;
#if LAUNCH_NPU_WATCH_XRT
    for (int attempt = 0; attempt < 3; ++attempt) {
      try {
        const auto *d = reinterpret_cast<const detail::xrt_mirror::device *>(handle_.get());
        const std::any a = d->lookup_query((detail::xrt_mirror::device::key_type)key_).get(handle_.get());
        const auto &v = std::any_cast<const std::vector<xrt_core::query::aie_partition_info::data> &>(a);
        if (v.empty())
          return true;
        const unsigned char *p = &v[0].byte;
        if (!stride_) { // the element size, once: the one size that gives contexts
          size_t n = 0;
          for (size_t s : detail::kStrides)
            if (detail::decode(p, v.size(), s, nullptr))
              stride_ = s, ++n;
          if (n != 1) {
            stride_ = 0;
            error_ = "a list of " + std::to_string(v.size()) + " bytes: " +
                     (n ? "more than one element size fits" : "no element size fits");
            return false;
          }
        }
        if (detail::decode(p, v.size(), stride_, out))
          return true;
        error_ = "an entry of the list is not a context";
        return false;
      } catch (const std::exception &e) {
        error_ = e.what();
      } catch (...) {
        error_ = "an unknown exception";
      }
    }
    return false;
#elif defined(__linux__)
    std::vector<unsigned char> &buf = buf_; // 128 KB, made once
    if (buf.empty())
      buf.resize((size_t)detail::kMaxElementSize * detail::kMaxElements);
    detail::GetArray a{0, detail::kMaxElementSize, detail::kMaxElements, 0, (uint64_t)(uintptr_t)buf.data()};
    if (::ioctl(fd_, detail::kIoctlGetArray, &a) != 0 || a.element_size < sizeof(detail::HwctxEntryHead))
      return false;
    if (out)
      for (uint32_t i = 0; i < a.num_element; ++i) {
        detail::HwctxEntryHead e;
        __builtin_memcpy(&e, buf.data() + (size_t)i * a.element_size, sizeof e);
        out->push_back({e.pid, e.context_id, e.command_submissions});
      }
    return true;
#else
    (void)out;
    return false;
#endif
  }

  void close() {
#if LAUNCH_NPU_WATCH_XRT
    handle_.reset();
    dev_.reset();
#elif defined(__linux__)
    if (fd_ >= 0)
      ::close(fd_);
    fd_ = -1;
#endif
  }

  bool ok_ = false;
  std::string why_;
  mutable std::string error_;
#if LAUNCH_NPU_WATCH_XRT
  std::optional<xrt::device> dev_;
  std::shared_ptr<xrt_core::device> handle_;
  int key_ = -1;               ///< aie_partition_info's query key
  mutable size_t stride_ = 0;  ///< its element size; 0: not known yet
#elif defined(__linux__)
  int fd_ = -1;
  mutable std::vector<unsigned char> buf_;
#endif
};

} // namespace launch::xrt_support
