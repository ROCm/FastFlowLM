// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hrx_cpp.hpp
/// \brief Minimal C++ `namespace hrx` providing the device/buffer/kernel/run
///        API that FastFlowLM uses, implemented directly on top of libhrx
///        (hrx_runtime.h).
///
/// NPU control code goes straight from npu_sequence::dump() into an HRX XADX
/// "direct executable"; there is no separate assembler step.
///
/// Coherence model: this amdxdna device is a single HOST_ONLY heap (cached
/// host DRAM the NPU can snoop). Buffers are allocated HOST_VISIBLE |
/// HOST_CACHED | DEVICE_VISIBLE (0x1A), mapped once (persistent). The heap
/// is not HOST_COHERENT, so FastFlowLM's sync_to_device()/sync_from_device()
/// map to hrx_buffer_flush_range()/hrx_buffer_invalidate_range(). Dispatch is
/// hrx_stream_dispatch() + hrx_stream_synchronize().
#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <algorithm>
#include <utility>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "hrx_amdxdna.h"
#include "hrx_runtime.h"
#include "npu_utils/runtime_wait.hpp"

// ---- ert_cmd_state: command states that FLM's npu_utils returns/maps.
#ifndef FLM_ERT_CMD_STATE_DEFINED
#define FLM_ERT_CMD_STATE_DEFINED
enum ert_cmd_state {
    ERT_CMD_STATE_NEW = 1,
    ERT_CMD_STATE_QUEUED = 2,
    ERT_CMD_STATE_RUNNING = 3,
    ERT_CMD_STATE_COMPLETED = 4,
    ERT_CMD_STATE_ERROR = 5,
    ERT_CMD_STATE_ABORT = 6,
    ERT_CMD_STATE_SUBMITTED = 7,
    ERT_CMD_STATE_TIMEOUT = 8,
    ERT_CMD_STATE_NORESPONSE = 9,
    ERT_CMD_STATE_SKERROR = 10,
    ERT_CMD_STATE_SKCRASHED = 11,
    ERT_CMD_STATE_MAX = 12,
};
#endif

namespace hrx {

// ---------------------------------------------------------------------------
// Process-wide HRX runtime (one device + one stream), lazily initialized.
// ---------------------------------------------------------------------------
class Runtime {
public:
    static Runtime& get() {
        static Runtime r;
        return r;
    }
    hrx_device_t dev = nullptr;
    hrx_stream_t stream = nullptr;
    bool ok = false;

    void ensure() {
        if (ok) return;
        // Assign device/stream only after both succeed. A failed stream_create
        // used to leave `dev` non-null with `ok == false`, and `if (dev) return`
        // then wedged every later call.
        hrx_device_t d = nullptr;
        hrx_stream_t s = nullptr;
        hrx_status_t init_status = hrx_gpu_initialize(0);
        const bool initialized =
            hrx_status_is_ok(init_status) ||
            hrx_status_code(init_status) == HRX_STATUS_ALREADY_EXISTS;
        hrx_status_ignore(init_status);
        if (initialized &&
            hrx_status_is_ok(hrx_gpu_device_get(0, &d)) &&
            hrx_status_is_ok(hrx_stream_create(d, 0, &s))) {
            dev = d;
            stream = s;
            ok = true;
            if (std::getenv("HRX_DEBUG"))
                std::fprintf(stderr, "[hrx] device+stream initialized (Runtime@%p)\n",
                             (void*)this);
        } else {
            std::fprintf(stderr, "[hrx] device init FAILED\n");
            ok = false;
            dev = nullptr;
            stream = nullptr;
        }
    }

private:
    Runtime() = default;
};

inline Runtime& rt() {
    Runtime& r = Runtime::get();
    r.ensure();
    return r;
}

// Report (do not swallow) an HRX error. Returns true if status was an error.
// FLM dispatch silently ignored synchronize/dispatch failures, which turns a
// failed ERT_CMD_CHAIN (e.g. a missing host patch table) into silent no-op
// dispatches -> garbage output at full speed. Always surface these.
inline bool hrx_report(hrx_status_t s, const char* where) {
    if (hrx_status_is_ok(s)) {
        flm::npu::ert_failure_detail().clear();
        return false;
    }
    char* m = nullptr;
    size_t mn = 0;
    hrx_status_to_string(s, &m, &mn);
    const char* text = m ? m : "?";
    std::fprintf(stderr, "[hrx][ERROR] %s: %s\n", where, text);
    flm::npu::ert_failure_detail() = text;
    hrx_status_free_message(m);
    hrx_status_ignore(s);
    return true;
}

inline hrx_amdxdna_xclbin_identity_t compute_xclbin_identity(
    const std::vector<uint8_t>& bytes) {
    uint64_t hash_1 = 1469598103934665603ULL;
    uint64_t hash_2 = 0xcbf29ce484222325ULL ^ bytes.size();
    for (uint8_t byte : bytes) {
        hash_1 ^= byte;
        hash_1 *= 1099511628211ULL;
        hash_2 += byte;
        hash_2 += hash_2 << 10;
        hash_2 ^= hash_2 >> 6;
    }
    hash_2 += hash_2 << 3;
    hash_2 ^= hash_2 >> 11;
    hash_2 += hash_2 << 15;
    return {hash_1, hash_2, static_cast<uint64_t>(bytes.size())};
}

// Build one executable for an immutable control revision. npu_app owns the
// returned handle directly and releases it when that revision is replaced.
// The optional precomputed xclbin identity lets HRX's device-owned bounded
// cache reuse equivalent native executables without rehashing the image.
inline hrx_executable_t build_executable(
    const std::vector<uint8_t>& xclbin_bytes,
    const hrx_amdxdna_xclbin_identity_t& xclbin_identity,
    const uint32_t* cc, size_t n,
    uint32_t* ord_out) {
    hrx_executable_t exe = nullptr;
    uint32_t ord = 0;
    const hrx_string_view_t entry_point_name = {
        "MLIR_AIE", std::strlen("MLIR_AIE")};
    const hrx_const_byte_span_t transaction = {
        reinterpret_cast<const uint8_t*>(cc), n * sizeof(uint32_t)};
    hrx_const_byte_span_t xclbin = {xclbin_bytes.data(),
                                   xclbin_bytes.size()};
    hrx_amdxdna_executable_run_t run =
        hrx_amdxdna_executable_run_default();
    run.transaction = transaction;
    hrx_amdxdna_executable_entry_point_t entry_point =
        hrx_amdxdna_executable_entry_point_default();
    entry_point.name = entry_point_name;
    entry_point.runs = &run;
    entry_point.run_count = 1;
    hrx_amdxdna_executable_create_params_t params =
        hrx_amdxdna_executable_create_params_default();
    params.abi_version = HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_2;
    params.xclbins = &xclbin;
    params.xclbin_count = 1;
    params.entry_points = &entry_point;
    params.entry_point_count = 1;
    params.xclbin_identities = &xclbin_identity;
    params.xclbin_identity_count = 1;
    hrx_status_t create_status =
        hrx_amdxdna_executable_create(rt().dev, &params, &exe);
    if (hrx_report(create_status, "hrx_amdxdna_executable_create")) {
        exe = nullptr;
    }
    if (exe && hrx_report(hrx_executable_lookup_export_by_name(
                              exe, "MLIR_AIE", &ord),
                          "hrx_executable_lookup_export_by_name")) {
        hrx_executable_release(exe);
        exe = nullptr;
    }
    if (ord_out) *ord_out = ord;
    return exe;
}

// Owns one executable generation. Acquired handles carry an independent
// reference so in-flight run values remain valid after the owner advances to a
// newer control revision.
class executable_owner {
public:
    executable_owner() = default;
    ~executable_owner() { reset(); }

    executable_owner(const executable_owner&) = delete;
    executable_owner& operator=(const executable_owner&) = delete;

    executable_owner(executable_owner&& other) noexcept {
        move_from(std::move(other));
    }

    executable_owner& operator=(executable_owner&& other) noexcept {
        if (this != &other) {
            reset();
            move_from(std::move(other));
        }
        return *this;
    }

    bool matches(uint64_t revision) const {
        return executable_ && revision_ == revision;
    }

    void replace(hrx_executable_t executable, uint64_t revision,
                 uint32_t ordinal) {
        reset();
        executable_ = executable;
        revision_ = revision;
        ordinal_ = ordinal;
    }

    hrx_executable_t acquire(uint32_t* ordinal) const {
        if (!executable_) return nullptr;
        hrx_executable_retain(executable_);
        if (ordinal) *ordinal = ordinal_;
        return executable_;
    }

    void reset() noexcept {
        if (executable_) hrx_executable_release(executable_);
        executable_ = nullptr;
        revision_ = UINT64_MAX;
        ordinal_ = 0;
    }

private:
    void move_from(executable_owner&& other) noexcept {
        executable_ = other.executable_;
        revision_ = other.revision_;
        ordinal_ = other.ordinal_;
        other.executable_ = nullptr;
        other.revision_ = UINT64_MAX;
        other.ordinal_ = 0;
    }

    hrx_executable_t executable_ = nullptr;
    uint64_t revision_ = UINT64_MAX;
    uint32_t ordinal_ = 0;
};

// ---------------------------------------------------------------------------
// uuid / xclbin / device / hw_context
// ---------------------------------------------------------------------------
class uuid {
public:
    unsigned char m_uuid[16] = {0};
};

class xclbin {
public:
    std::shared_ptr<std::vector<uint8_t>> bytes_ =
        std::make_shared<std::vector<uint8_t>>();

    xclbin() = default;
    explicit xclbin(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("hrx::xclbin: cannot open " + path);
        std::fseek(f, 0, SEEK_END);
        long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (n > 0) {
            bytes_->resize(static_cast<size_t>(n));
            size_t rd = std::fread(bytes_->data(), 1, bytes_->size(), f);
            (void)rd;
        }
        std::fclose(f);
    }

    // FLM searches kernels for one whose name starts with "MLIR_AIE"; the HRX
    // dispatch path always uses the "MLIR_AIE" export, so a single placeholder
    // kernel is sufficient (matches the proven interposer behavior).
    class kernel {
    public:
        std::string name = "MLIR_AIE";
        std::string get_name() const { return name; }
    };
    std::vector<kernel> get_kernels() const { return {kernel{}}; }
    uuid get_uuid() const { return uuid{}; }
    const std::vector<uint8_t>& bytes() const { return *bytes_; }
    std::shared_ptr<std::vector<uint8_t>> bytes_shared() const { return bytes_; }
};

namespace info {
// Argument to device::get_info<>(). FLM only queries the human-readable
// device name for diagnostics.
enum class device { name, architecture };
}

class device {
public:
    device() = default;
    explicit device(unsigned int /*index*/) { rt(); }
    uuid register_xclbin(const xclbin& /*xc*/) { return uuid{}; }
    void reset() {}

    // Returns a human-readable device identity string for diagnostics: the
    // IREE HAL device name (e.g. "amdxdna") via hrx_device_get_property().
    template <info::device P>
    std::string get_info() const {
        char buf[128] = {0};
        hrx_device_get_property(
            rt().dev,
            P == info::device::architecture ? HRX_DEVICE_PROPERTY_ARCHITECTURE
                                            : HRX_DEVICE_PROPERTY_NAME,
            buf, sizeof(buf));
        return std::string(buf);
    }
};

class hw_context {
public:
    std::shared_ptr<std::vector<uint8_t>> xclbin_bytes_;
    hrx_amdxdna_xclbin_identity_t xclbin_identity_ = {};

    hw_context() = default;
    hw_context(const device& /*dev*/, const xclbin& xc)
        : xclbin_bytes_(xc.bytes_shared()) {
        if (xclbin_bytes_ && !xclbin_bytes_->empty()) {
            xclbin_identity_ = compute_xclbin_identity(*xclbin_bytes_);
        }
    }
    // The (device, uuid) form is accepted for source compatibility; it carries
    // no xclbin bytes, so prefer the (device, xclbin) form.
    hw_context(const device& /*dev*/, const uuid& /*id*/) {}

    const std::vector<uint8_t>& xclbin_bytes() const {
        static const std::vector<uint8_t> empty;
        return xclbin_bytes_ ? *xclbin_bytes_ : empty;
    }
    const hrx_amdxdna_xclbin_identity_t& xclbin_identity() const {
        return xclbin_identity_;
    }
};

// ---------------------------------------------------------------------------
// Per-buffer host<->device coherence state (dirty tracking).
//
// This path uses ONE persistent host-mapped, device-visible buffer per BO;
// coherence is kept purely with clflush-style range ops. The correctness
// comes entirely from GATING those ops per buffer:
//
//   host_dirty : the host wrote bytes that are not yet on the device. Must be
//                flushed (h2d) before any dispatch reads the buffer, and only
//                then.  Cleared by the flush.
//   dev_dirty  : a dispatch wrote bytes that are not yet visible to the host.
//                Must be invalidated (d2h) before the host reads, and only
//                then.  Cleared by the invalidate.
//
// Why gating matters (this is the multi-turn bug):
//   * Unconditional flush-before-dispatch pushes a STALE host copy over a
//     device-resident buffer (e.g. kv_caches the previous layer just wrote) ->
//     corrupts the KV cache.
//   * Unconditional invalidate-on-read DROPS the host's own not-yet-flushed
//     writes -> the device never sees them on the next turn.
// This path avoids both by only acting when the matching dirty bit is set; on a
// single turn the bits happen to line up either way, but turn>1 reuses buffers
// whose bits diverge, which is why only later turns broke.
struct BufCoh {
    bool host_dirty = true;   // freshly-allocated content must reach the device once
    bool dev_dirty = false;
    // What host_dirty covers. A whole-buffer flush() (every existing caller) marks the whole buffer (host_all), so
    // they behave exactly as before. Only flush_range() records exact byte intervals (sorted, disjoint, merged when they
    // touch), so a buffer whose host writes were confined to a few rows is flushed for those rows, not whole, and bytes the
    // host did not write are never pushed (a stale host cache line must not overwrite device results).
    bool host_all = true;
    std::vector<std::pair<size_t, size_t>> host_ranges;   // [lo, hi) pairs, used while !host_all
};
// Add [lo, hi) to a sorted, disjoint interval list, merging overlapping or adjacent ones.
inline void coh_add_range(std::vector<std::pair<size_t, size_t>>& v, size_t lo, size_t hi) {
    std::vector<std::pair<size_t, size_t>> out;
    out.reserve(v.size() + 1);
    bool placed = false;
    for (const auto& r : v) {
        if (r.second < lo) {
            out.push_back(r);
        } else if (hi < r.first) {
            if (!placed) { out.emplace_back(lo, hi); placed = true; }
            out.push_back(r);
        } else {
            lo = (std::min)(lo, r.first);
            hi = (std::max)(hi, r.second);
        }
    }
    if (!placed) out.emplace_back(lo, hi);
    v.swap(out);
}
// Remove [lo, hi) from a sorted, disjoint interval list.
inline void coh_remove_range(std::vector<std::pair<size_t, size_t>>& v, size_t lo, size_t hi) {
    std::vector<std::pair<size_t, size_t>> out;
    out.reserve(v.size() + 1);
    for (const auto& r : v) {
        if (r.second <= lo || r.first >= hi) { out.push_back(r); continue; }
        if (r.first < lo) out.emplace_back(r.first, lo);
        if (r.second > hi) out.emplace_back(hi, r.second);
    }
    v.swap(out);
}
inline std::mutex& buf_coh_mu() {
    static std::mutex m;
    return m;
}
inline std::unordered_map<hrx_buffer_t, BufCoh>& buf_coh() {
    static std::unordered_map<hrx_buffer_t, BufCoh> m;
    return m;
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------
class bo {
public:
    hrx_buffer_t hbuf_ = nullptr;
    void* mapped_ = nullptr;
    size_t size_ = 0;
    bool owns_ = false;

    bo() = default;
    virtual ~bo() {
        if (owns_ && hbuf_) {
            {
                std::lock_guard<std::mutex> lk(buf_coh_mu());
                buf_coh().erase(hbuf_);
            }
            hrx_buffer_release(hbuf_);
        }
        hbuf_ = nullptr;
        mapped_ = nullptr;
    }
    bo(const bo&) = delete;
    bo& operator=(const bo&) = delete;

    template <typename T>
    T map() {
        return reinterpret_cast<T>(mapped_);
    }
    size_t size() const { return size_; }
    hrx_buffer_t handle() const { return hbuf_; }

    // sync_to_device(): the host has new data. Record it as host_dirty and defer
    // the actual h2d flush to the dispatch that consumes the buffer (matching the
    // shim). This is NOT an unconditional flush: a buffer the device owns is never
    // marked host_dirty here, so it can never be clobbered.
    void flush() {
        if (!hbuf_) return;
        std::lock_guard<std::mutex> lk(buf_coh_mu());
        BufCoh& st = buf_coh()[hbuf_];
        st.host_dirty = true;
        st.host_all = true;
        st.host_ranges.clear();
        st.dev_dirty = false;  // host is now the source of truth
    }
    // sync_to_device() for bytes [offset, offset + len) only: the host wrote just those bytes. They are recorded as
    // host_dirty and flushed by the dispatch that consumes the buffer, like flush(), but only that range goes. The rest
    // of the buffer keeps its state: dev_dirty is NOT cleared (rows outside the range may still hold device results the
    // host has not read yet), which is the one difference from flush().
    void flush_range(size_t offset, size_t len) {
        if (!hbuf_ || len == 0) return;
        std::lock_guard<std::mutex> lk(buf_coh_mu());
        BufCoh& st = buf_coh()[hbuf_];
        if (!st.host_dirty) {
            st.host_all = false;
            st.host_ranges.clear();
        }
        st.host_dirty = true;
        if (!st.host_all) coh_add_range(st.host_ranges, offset, offset + len);   // else: the whole buffer is already marked
    }
    // sync_from_device(): the host wants to read. Only invalidate if a dispatch
    // actually produced new device data (dev_dirty); otherwise this would drop
    // the host's own writes.
    void invalidate() {
        if (!hbuf_) return;
        std::lock_guard<std::mutex> lk(buf_coh_mu());
        BufCoh& st = buf_coh()[hbuf_];
        if (st.dev_dirty) {
            hrx_report(hrx_buffer_invalidate_range(hbuf_, 0, size_),
                       "hrx_buffer_invalidate_range");
            st.dev_dirty = false;
        }
    }
    // sync_from_device() for bytes [offset, offset + len) only. dev_dirty stays set: the other rows of the buffer
    // may still hold device results that have not been read.
    void invalidate_range(size_t offset, size_t len) {
        if (!hbuf_ || len == 0) return;
        std::lock_guard<std::mutex> lk(buf_coh_mu());
        BufCoh& st = buf_coh()[hbuf_];
        if (st.dev_dirty) {
            if (offset >= size_) return;
            if (offset + len > size_) len = size_ - offset;
            hrx_report(hrx_buffer_invalidate_range(hbuf_, offset, len),
                       "hrx_buffer_invalidate_range");
        }
    }
};

namespace ext {
class bo : public hrx::bo {
public:
    bo(const device& /*dev*/, size_t sz) {
        Runtime& r = rt();
        size_ = sz;
        owns_ = true;
        if (!r.ok) throw std::runtime_error("hrx::ext::bo: HRX device unavailable");
        // HOST_VISIBLE | HOST_CACHED | DEVICE_VISIBLE (0x1A). This device has
        // one HOST_ONLY heap; HOST_LOCAL / HOST_COHERENT / DEVICE_LOCAL are
        // rejected. Coherence is flush after host writes and invalidate after
        // device writes. Persistent map via hrx_buffer_map_with_mode.
        hrx_status_t s = hrx_buffer_allocate(
            r.stream, sz,
            HRX_MEMORY_TYPE_HOST_VISIBLE |
                HRX_MEMORY_TYPE_HOST_CACHED |
                HRX_MEMORY_TYPE_DEVICE_VISIBLE,
            HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
            &hbuf_);
        if (!hrx_status_is_ok(s) || !hbuf_) {
            hrx_status_ignore(s);
            throw std::runtime_error("hrx::ext::bo: hrx_buffer_allocate failed");
        }
        void* p = nullptr;
        s = hrx_buffer_map_with_mode(hbuf_, HRX_MAPPING_MODE_PERSISTENT,
                                     HRX_MAP_READ | HRX_MAP_WRITE, 0, sz, &p);
        if (!hrx_status_is_ok(s) || !p) {
            hrx_status_ignore(s);
            hrx_buffer_release(hbuf_);
            hbuf_ = nullptr;
            throw std::runtime_error("hrx::ext::bo: map_persistent failed");
        }
        mapped_ = p;
        std::memset(p, 0, sz);
        // The zeroed contents are host-side only until the first dispatch; mark
        // host_dirty so the gated h2d flushes them to the device exactly once.
        {
            std::lock_guard<std::mutex> lk(buf_coh_mu());
            buf_coh()[hbuf_] = BufCoh{};
        }
    }
};
}  // namespace ext

// ---------------------------------------------------------------------------
// run / runlist
// ---------------------------------------------------------------------------
// Gated host->device flush for a dispatch's bindings: flush only the buffers the
// host has dirtied (host_dirty), exactly once, right before the device reads
// them. Device-owned buffers (host_dirty == false) are left untouched so they
// are never clobbered. Mirrors the shim's "h2d only dirty inputs" phase.
inline void hrx_h2d_bindings(const std::vector<hrx_buffer_ref_t>& binds) {
    std::lock_guard<std::mutex> lk(buf_coh_mu());
    auto& coh = buf_coh();
    for (const auto& b : binds) {
        if (!b.buffer) continue;
        BufCoh& st = coh[b.buffer];
        if (st.host_dirty) {
            if (st.host_all) {
                // whole-buffer mark (every flush() caller): the binding's range, as before
                hrx_report(hrx_buffer_flush_range(b.buffer, b.offset, b.length),
                           "hrx_buffer_flush_range");
                // A 0-offset binding spans the whole buffer (length == size), so
                // the whole dirty buffer has now been flushed and can be cleared.
                // A non-zero offset (Shared-Executable Collapse sub-range binding)
                // flushes only [offset, offset+length); the bytes outside that
                // range are still host-dirty, so keep the flag set and let later
                // bindings flush their own ranges (re-flushing a clean range is
                // idempotent). This prevents clearing dirty bytes we never flushed.
                if (b.offset == 0) st.host_dirty = false;
            } else {
                // ranged marks (flush_range()): only the dirty bytes inside this binding go, and they are no longer dirty;
                // dirty bytes outside the binding stay marked for the dispatch that binds them
                const size_t b0 = b.offset, b1 = b.offset + b.length;
                const auto dirty = st.host_ranges;
                for (const auto& r : dirty) {
                    const size_t lo = (std::max)(b0, r.first), hi = (std::min)(b1, r.second);
                    if (lo >= hi) continue;
                    hrx_report(hrx_buffer_flush_range(b.buffer, lo, hi - lo),
                               "hrx_buffer_flush_range");
                    coh_remove_range(st.host_ranges, lo, hi);
                }
                if (st.host_ranges.empty()) {
                    st.host_dirty = false;
                    st.host_all = true;
                }
            }
        }
    }
}

// After a dispatch completes, the device holds the freshest copy of every bound
// buffer: clear host_dirty and mark dev_dirty so the next host read invalidates
// (lazily, via sync_from_device). Mirrors the shim's post-dispatch readback
// bookkeeping. We over-approximate by treating every binding as a potential
// output; that is safe because an extra dev_dirty only triggers a redundant
// invalidate that re-reads identical bytes.
inline void hrx_mark_dispatched(const std::vector<hrx_buffer_ref_t>& binds) {
    std::lock_guard<std::mutex> lk(buf_coh_mu());
    auto& coh = buf_coh();
    for (const auto& b : binds) {
        if (!b.buffer) continue;
        BufCoh& st = coh[b.buffer];
        st.host_dirty = false;
        st.dev_dirty = true;
    }
}

class run {
public:
    hrx_executable_t exe_ = nullptr;
    uint32_t ord_ = 0;
    std::vector<hrx_buffer_ref_t> binds_;

    run() = default;
    explicit run(hrx_executable_t exe, uint32_t ord) : exe_(exe), ord_(ord) {}
    ~run() {
        if (exe_) hrx_executable_release(exe_);
    }

    run(const run& other)
        : exe_(other.exe_), ord_(other.ord_), binds_(other.binds_) {
        if (exe_) hrx_executable_retain(exe_);
    }

    run& operator=(const run& other) {
        if (this == &other) return *this;
        if (other.exe_) hrx_executable_retain(other.exe_);
        if (exe_) hrx_executable_release(exe_);
        exe_ = other.exe_;
        ord_ = other.ord_;
        binds_ = other.binds_;
        return *this;
    }

    run(run&& other) noexcept
        : exe_(other.exe_), ord_(other.ord_), binds_(std::move(other.binds_)) {
        other.exe_ = nullptr;
    }

    run& operator=(run&& other) noexcept {
        if (this == &other) return *this;
        if (exe_) hrx_executable_release(exe_);
        exe_ = other.exe_;
        ord_ = other.ord_;
        binds_ = std::move(other.binds_);
        other.exe_ = nullptr;
        return *this;
    }

    void add_binding(hrx_buffer_t b, size_t size) {
        binds_.push_back({b, 0, size});
    }

    // Shared-Executable Collapse: bind a buffer argument at a non-zero byte
    // offset. Dispatch resolves the arg address as
    // device_address(buffer) + offset + arg_offset(TXN), so supplying the
    // per-expert weight/bias base here keeps the TXN expert-independent and lets
    // all experts at a given padded length reuse one executable generation.
    void add_binding(hrx_buffer_t b, size_t offset, size_t size) {
        binds_.push_back({b, offset, size});
    }

    void record() {
        if (!exe_) {
            std::fprintf(stderr, "[hrx][ERROR] run::record with null executable\n");
            return;
        }
        // Zero bindings is valid: RTP/control-only runs such as
        // set_layer_rtp.create_run() and gemma4e layer_pre_load.create_run().
        // Opcode scalars live in the TXN stream, not as BO bindings.
        hrx_h2d_bindings(binds_);
        hrx_dispatch_config_t cfg = {{1, 1, 1}, {1, 1, 1}, 0};
        hrx_status_t s = hrx_stream_dispatch(rt().stream, exe_, ord_, &cfg,
                                             nullptr, 0, binds_.data(),
                                             binds_.size(), HRX_DISPATCH_FLAG_NONE);
        hrx_report(s, "run::record hrx_stream_dispatch");
    }

    void start() {
        record();
        hrx_status_t s = hrx_stream_flush(rt().stream);
        hrx_report(s, "run::start hrx_stream_flush");
    }

    ert_cmd_state wait() {
        hrx_status_t s = hrx_stream_wait(rt().stream);
        bool err = hrx_report(s, "run::wait hrx_stream_wait");
        hrx_mark_dispatched(binds_);
        return err ? ERT_CMD_STATE_ERROR : ERT_CMD_STATE_COMPLETED;
    }
};

class runlist {
public:
    std::vector<run> runs_;

    runlist() = default;
    explicit runlist(const hw_context& /*ctx*/) {}

    void add(const run& r) { runs_.push_back(r); }
    void add(run&& r) { runs_.push_back(std::move(r)); }
    void reset() { runs_.clear(); }

    void execute() {
        for (auto& r : runs_) r.record();
        hrx_status_t s = hrx_stream_flush(rt().stream);
        hrx_report(s, "runlist::execute hrx_stream_flush");
    }
    ert_cmd_state wait() {
        hrx_status_t s = hrx_stream_wait(rt().stream);
        bool err = hrx_report(s, "runlist::wait hrx_stream_wait");
        for (auto& r : runs_) hrx_mark_dispatched(r.binds_);
        return err ? ERT_CMD_STATE_ERROR : ERT_CMD_STATE_COMPLETED;
    }
};

}  // namespace hrx
