// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
// Reconfigure the NPU array from inside an instruction stream, instead of switching hardware contexts.
// (Adapted from the Gemma4e vision single-context reconfiguration.)
//
// The MiniCPM-V language prefill alternates between two overlays: mm.xclbin (projections) and attn.xclbin (full attention, 6 layers).
// Each overlay is its own xclbin = its own hw_context, and every switch between contexts costs about 2.4 ms (the firmware reloads the PDI);
// the same switch as an in-stream reconfiguration costs 0.7 ms (measured).
// For the array, an xclbin's PDI is a CDO: a list of register writes (mask-write, write, block-write of core programs / DMA descriptors).
// The same writes can be issued as an NPU transaction (what npu_sequence / aiebu produce) inside ONE hw_context, at a fraction of the cost.
//
//   Chain::build(attn_xclbin, mm_xclbin, ...) -> transactions that make the array behave as if attn_xclbin had been loaded while mm_xclbin
//   was the previous overlay. Run them (as their own tiny dispatches) in the hw_context of ANY xclbin, then run the normal sequence.
//
// No new artifacts: the CDO is located inside the xclbins that are already shipped.
//
// A CDO replay is not the same as the firmware's PDI load, which starts from a reset array. Three things have to be done first:
//   1. DMA channels: the PDI load starts the memory-tile channels; replaying the CDO would queue the same task a second time
//      (TASK_QUEUE_SIZE 1 instead of 0). Pulse the reset bit of every channel of every tile.
//   2. Locks: a CDO only writes the locks its own design initialises, so a lock the previous overlay left non-zero stays non-zero
//      (e.g. a memory-tile lock of vision_mm, which corrupted the matmul result). Write 0 to every lock of every tile.
//   3. Stale registers: registers the previous overlay's CDO wrote and this one's does not (stream-switch routes!) stay as they were.
//      Only the stream-switch registers are zeroed (see txn_chunks).
//      Stale vision_mm routes made vision_attn deadlock. They reset to 0 (checked for the stream switch master/slave config).
//      Core program memory is left alone: stale code is never executed.
//   4. Pending lock acquires (the cause of the in-stream attention hang): a core or DMA channel that is blocked on a lock when its
//      design ends keeps that acquire pending inside the lock module. Neither a core reset nor a DMA channel reset cancels it, a lock value of
//      0 does not grant it, and the next design's first release of that lock is consumed by the stale acquire, so an object never reaches its
//      real consumer. Only the PDI load's array reset clears it, and writing the value 63 to a lock grants any pending acquire. So the
//      prelude holds every core in reset, holds the DMA channels in reset, writes every lock 63 and then 0, and only then releases the
//      channels (cores are re-enabled by the new design's CDO).
// Found with the driver's AIE status dump (DRM_AMDXDNA_QUERY_AIE_STATUS) as a state oracle: after a PDI load and after this reconfiguration
// the array must look the same.
//
// One transaction is limited to ~512 KB of instructions (a 537 KB one is rejected by the firmware), so a reconfiguration is cut at op
// boundaries into transactions of at most 400 KB, run in order.
#include <algorithm>
#include <atomic>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "npu_utils/runtime_wait.hpp"

// Only validated on Linux with the XRT backend (amdxdna driver, firmware transaction processor). The HRX backend (its own executable path)
// and Windows (different driver/firmware) keep the two-context path until the reconfiguration is validated there.
#if defined(FLM_USE_HRX) || defined(_WIN32)
#define FLM_MINICPM_RECONFIG_SUPPORTED 0
#else
#define FLM_MINICPM_RECONFIG_SUPPORTED 1
#endif

namespace minicpm_reconfig {

// FLM_MINICPM_SINGLE_CTX: 1 = the full-attention layers run in the mm context between two in-stream reconfigurations (mm -> attn -> mm);
// 0 = two contexts (old path). Default on: it still needs the load-time self-test, the per-call exclusivity check and the
// foreign-context guard, and falls back to the two-context path whenever one of them says no.
inline bool single_context() {
#if FLM_MINICPM_RECONFIG_SUPPORTED
    static const bool v = [] { const char* e = std::getenv("FLM_MINICPM_SINGLE_CTX"); return e ? std::atoi(e) != 0 : true; }();
    return v;
#else
    return false;
#endif
}

// One kill switch for the process: set by a failed self-test, a failed round trip, or an unusable NPU; never cleared.
inline std::atomic<bool>& disabled_flag() { static std::atomic<bool> d{false}; return d; }
inline bool disabled() { return disabled_flag().load(); }
inline void disable(const std::string& why) {
    if (!disabled_flag().exchange(true)) header_print_r("WARNING", "minicpm_v: single-context reconfiguration disabled: " << why << " (two-context path from here on)");
}

inline void validate_cdo(const std::vector<uint32_t>& cdo);   // defined below

// CDO v2 header: [0]=4 (words after this one) [1]='CDO\0' [2]=0x200 [3]=payload words [4]=checksum, then commands:
// command word = (payload_words << 16) | id; payload_words == 255 means the real length is in the next word.
inline std::vector<uint32_t> find_cdo(const std::string& xclbin_path) {
    std::ifstream f(xclbin_path, std::ios::binary);
    if (!f) throw std::runtime_error("minicpm_reconfig: cannot open " + xclbin_path);
    std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (size_t i = 4; i + 20 <= d.size(); i += 4) {
        uint32_t w[5];
        std::memcpy(w, d.data() + i - 4, 20);
        if (w[0] == 4 && w[1] == 0x004F4443u && w[2] == 0x200u && i - 4 + (5 + (size_t)w[3]) * 4 <= d.size()) {
            std::vector<uint32_t> cdo(5 + (size_t)w[3]);
            std::memcpy(cdo.data(), d.data() + i - 4, cdo.size() * 4);
            validate_cdo(cdo);
            return cdo;
        }
    }
    throw std::runtime_error("minicpm_reconfig: no CDO found in " + xclbin_path);
}

// The CDO comes out of a file: check the command stream before anyone trusts it (extended-length words, payload bounds, opcode minimum sizes),
// so a corrupt artifact is a load error and not an out-of-bounds read.
inline void validate_cdo(const std::vector<uint32_t>& cdo) {
    auto bad = [](const char* why) { throw std::runtime_error(std::string("minicpm_reconfig: corrupt CDO: ") + why); };
    if (cdo.size() < 5 || cdo[0] != 4 || cdo.size() != 5 + (size_t)cdo[3]) bad("header");
    size_t i = 1 + cdo[0];
    const size_t end = cdo.size();
    while (i < end) {
        const uint32_t h = cdo[i++], id = h & 0xFFFF;
        size_t len = h >> 16;
        if (len == 0xFF) { if (i >= end) bad("missing extended length"); len = cdo[i++]; }
        if (len > end - i) bad("payload past the end");
        size_t min = 0;
        switch (id) {
            case 0x103: min = 2; break;     // write32 addr, value
            case 0x102: min = 3; break;     // mask write32 addr, mask, value
            case 0x105: min = 2; break;     // dma write: 64-bit address
            case 0x111: break;              // NOP
            default: bad("unsupported command");
        }
        if (len < min) bad("payload shorter than the command needs");
        i += len;
    }
}

// Every register address a CDO writes (block-write words expanded), sorted and unique.
inline std::vector<uint32_t> written_registers(const std::vector<uint32_t>& cdo) {
    std::vector<uint32_t> r;
    size_t i = 1 + cdo[0];
    const size_t end = i + cdo[3];
    while (i < end) {
        uint32_t h = cdo[i++], len = h >> 16;
        const uint32_t id = h & 0xFFFF;
        if (len == 0xFF) len = cdo[i++];
        const uint32_t* p = &cdo[i];
        if (id == 0x102 || id == 0x103) r.push_back(p[0]);
        else if (id == 0x105) for (uint32_t k = 0; k + 2 < len; k++) r.push_back(p[1] + 4 * k);
        i += len;
    }
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
    return r;
}

// NPU2 array: 8 columns x (shim row 0, memory tile row 1, core tile rows 2..5); register address = col << 25 | row << 20 | offset (AIE2P)
constexpr unsigned kCols = 8;
constexpr unsigned kFirstCoreRow = 2, kEndCoreRow = 6;
constexpr uint32_t kMemTileDmaCtrl = 0xA0600, kMemTileLocks = 0xC0000;                              // 6 S2MM + 6 MM2S channels (8 B apart), 64 locks (16 B apart)
constexpr uint32_t kCoreDmaCtrl[4] = {0x1DE00, 0x1DE08, 0x1DE10, 0x1DE18}, kCoreLocks = 0x1F000;   // 16 locks
constexpr uint32_t kShimDmaCtrl[4] = {0x1D200, 0x1D208, 0x1D210, 0x1D218}, kShimLocks = 0x14000;   // 16 locks

inline uint32_t tile_addr(unsigned col, unsigned row, uint32_t off) { return ((uint32_t)col << 25) | ((uint32_t)row << 20) | off; }

// Builds the transactions that reconfigure the array to the design of `cdo`, coming from the design of `previous_cdo`.
// `quiesce` = false is the pre-fix prelude (zero the locks only), kept for the fault-injection test build (it reproduces the hang).
inline std::vector<std::vector<uint32_t>> txn_chunks(const std::vector<uint32_t>& cdo, const std::vector<uint32_t>& previous_cdo,
                                                     bool quiesce = true, size_t max_bytes = 400000) {
    std::vector<std::vector<uint32_t>> chunks;
    // transaction header for NPU2: version 0.1, device gen 4, 6 rows, 8 columns, 1 memory tile row; words 2/3 = op count / size in bytes
    auto fresh = [] { return std::vector<uint32_t>{0u | (1u << 8) | (4u << 16) | (6u << 24), (uint32_t)kCols | (1u << 8), 0u, 0u}; };
    std::vector<uint32_t> t = fresh();
    uint32_t ops = 0;
    auto close = [&] { if (ops) { t[2] = ops; t[3] = (uint32_t)(t.size() * 4); chunks.push_back(std::move(t)); } t = fresh(); ops = 0; };
    auto room = [&](size_t words) { if (ops && (t.size() + words) * 4 > max_bytes) close(); };
    auto write32 = [&](uint32_t addr, uint32_t val) { room(6); t.insert(t.end(), {0u, 0u, addr, 0u, val, 6u * 4}); ops++; };
    auto maskwrite32 = [&](uint32_t addr, uint32_t val, uint32_t mask) { room(7); t.insert(t.end(), {3u, 0u, addr, 0u, val, mask, 7u * 4}); ops++; };

    // 0. halt every core (Core_Control = reset asserted, disabled) before anything else is touched: a running core of the previous overlay
    //    would otherwise keep acquiring/releasing the locks that are zeroed below. The new overlay's CDO enables its own cores at its end.
    for (unsigned col = 0; col < kCols; col++)
        for (unsigned row = kFirstCoreRow; row < kEndCoreRow; row++) write32(tile_addr(col, row, 0x32000), 2u);
    auto all_dma = [&](uint32_t v) {
        for (unsigned col = 0; col < kCols; col++) {
            for (unsigned i = 0; i < 6; i++) {
                maskwrite32(tile_addr(col, 1, kMemTileDmaCtrl + 8 * i), v, 2u);
                maskwrite32(tile_addr(col, 1, kMemTileDmaCtrl + 0x30 + 8 * i), v, 2u);
            }
            for (unsigned row = kFirstCoreRow; row < kEndCoreRow; row++)
                for (uint32_t off : kCoreDmaCtrl) maskwrite32(tile_addr(col, row, off), v, 2u);
            for (uint32_t off : kShimDmaCtrl) maskwrite32(tile_addr(col, 0, off), v, 2u);
        }
    };
    auto all_locks = [&](uint32_t v) {
        for (unsigned col = 0; col < kCols; col++) {
            for (unsigned i = 0; i < 64; i++) write32(tile_addr(col, 1, kMemTileLocks + 0x10 * i), v);
            for (unsigned row = kFirstCoreRow; row < kEndCoreRow; row++)
                for (unsigned i = 0; i < 16; i++) write32(tile_addr(col, row, kCoreLocks + 0x10 * i), v);
            for (unsigned i = 0; i < 16; i++) write32(tile_addr(col, 0, kShimLocks + 0x10 * i), v);
        }
    };
    if (!quiesce) {
        // pre-fix prelude (fault injection only): pulse the DMA resets, zero every lock
        all_dma(2u); all_dma(0u);
        all_locks(0u);
    } else {
        // cores are already held in reset (step 0). DMA channels are held in reset while every lock is written 63 (grants a pending
        // acquire) and then 0; only then are the channels released.
        all_dma(2u);
        all_locks(63u);
        all_locks(0u);
        all_dma(0u);
    }
    // 3. registers only the previous overlay wrote -> 0
    {
        const std::vector<uint32_t> before = written_registers(previous_cdo), now = written_registers(cdo);
        std::vector<uint32_t> stale;
        std::set_difference(before.begin(), before.end(), now.begin(), now.end(), std::back_inserter(stale));
        for (uint32_t addr : stale) {
            const uint32_t row = (addr >> 20) & 0x1F, off = addr & 0xFFFFF;
            if (row >= kFirstCoreRow && off >= 0x20000 && off < 0x24000) continue;      // core program memory
            // Only the stream switch (master/slave config, slot config) needs it: for every other register the status oracle matched a
            // PDI load better WITHOUT zeroing (DMA BDs, core data memory, ...), so those are left to the new overlay's own writes.
            if (!(off >= 0x3F000 && off < 0x3F400) && !(row == 1 && off >= 0xB0000 && off < 0xB0400)) continue;
            write32(addr, 0u);
        }
    }
    // the CDO itself: 0x102 mask write32 (addr, mask, value), 0x103 write32 (addr, value), 0x105 DMA write (64-bit addr hi/lo, data), 0x111 NOP
    size_t i = 1 + cdo[0];
    const size_t end = i + cdo[3];
    while (i < end) {
        uint32_t h = cdo[i++], len = h >> 16;
        const uint32_t id = h & 0xFFFF;
        if (len == 0xFF) len = cdo[i++];
        const uint32_t* p = &cdo[i];
        if (id == 0x103) {
            write32(p[0], p[1]);
        } else if (id == 0x102) {
            maskwrite32(p[0], p[2], p[1]);
        } else if (id == 0x105) {
            if (p[0] != 0) throw std::runtime_error("minicpm_reconfig: 64-bit CDO address");
            const uint32_t addr = p[1], n = len - 2;
            room(4 + n);
            t.insert(t.end(), {1u, ((addr >> 25) & 0xFFu) | (((addr >> 20) & 0x1Fu) << 8), addr, (4u + n) * 4});
            t.insert(t.end(), p + 2, p + 2 + n);
            ops++;
        } else if (id != 0x111) {
            throw std::runtime_error("minicpm_reconfig: unsupported CDO command 0x" + std::to_string(id));
        }
        i += len;
    }
    close();
    return chunks;
}

#if FLM_MINICPM_RECONFIG_SUPPORTED
using RunType = xrt::run;   // npu_utils.hpp (XRT backend) is included before this header's users instantiate Chain
#else
using RunType = int;
#endif

#if FLM_MINICPM_RECONFIG_SUPPORTED
#if defined(__has_include)
#if __has_include("xrt/detail/version-git.h")
#include "xrt/detail/version-git.h"   // XRT_DRIVER_VERSION, only for the log line in versions(); not every XRT install ships it
#define FLM_MINICPM_HAVE_XRT_VERSION 1
#endif
#endif
// Is this process the only one with hw_contexts on the NPU? (amdxdna DRM_AMDXDNA_QUERY_HW_CONTEXTS lists the contexts of every process.)
// The single-context path keeps a reconfigured array state between submissions; when another process creates/destroys contexts or submits
// work meanwhile, the firmware reloads the context's PDI and that state is lost (a hang or a wrong result). The two-context path does not
// depend on any state between submissions, so it is used whenever another process is on the NPU, or when this cannot be determined.
// Residual risk: a process that appears during the ~3 ms of one layer's attention window (the round trip is ONE submission, so the firmware
// cannot interleave another process's command inside it; the failure mode left is a context created in the gap between two layers' checks,
// which the next check sees).
inline int accel_fd() {
    static const int fd = ::open("/dev/accel/accel0", O_RDWR | O_CLOEXEC);
    return fd;
}
// the two driver queries below mirror struct amdxdna_drm_get_info / amdxdna_drm_query_hwctx / amdxdna_drm_query_firmware_version
struct GetInfo { uint32_t param; uint32_t buffer_size; uint64_t buffer; };
// DRM_IOCTL_AMDXDNA_GET_INFO = DRM_IOWR(DRM_COMMAND_BASE (0x40) + DRM_AMDXDNA_GET_INFO (7), struct amdxdna_drm_get_info)
inline bool get_info(GetInfo& gi) {
    const int fd = accel_fd();
    if (fd < 0) return false;
    const unsigned long req = (3UL << 30) | ((unsigned long)sizeof gi << 16) | ((unsigned long)'d' << 8) | (0x40 + 7);
    return ::ioctl(fd, req, &gi) == 0;
}
inline bool npu_exclusive() {
    struct HwCtx { uint32_t id, start_col, num_col, pad; int64_t pid; uint64_t submissions, completions, migrations, preemptions, errors; };   // struct amdxdna_drm_query_hwctx
    HwCtx ctx[64];
    std::memset(ctx, 0, sizeof ctx);
    GetInfo gi{5 /* DRM_AMDXDNA_QUERY_HW_CONTEXTS */, (uint32_t)sizeof ctx, (uint64_t)(uintptr_t)ctx};
    if (!get_info(gi)) return false;
    const unsigned filled = (sizeof ctx - gi.buffer_size) / sizeof(HwCtx);   // the driver lowers buffer_size by the bytes it wrote
    const int64_t me = (int64_t)::getpid();
    for (unsigned i = 0; i < filled && i < 64; i++)
        if (ctx[i].pid != me) return false;
    return true;
}

/// A fingerprint of every hw_context that does not belong to this process: (pid, context id, submissions, completions). It changes iff another process
/// created or destroyed a context or had a command run since the previous fingerprint. Used as a witness around work that keeps state on the array
/// between commands (the decode layer chain keeps the residual stream in the array for the 4 runs of a layer group): the firmware reloads this
/// design's PDI whenever another process's command ran in between, which wipes that state, and the group then silently computes garbage.
struct ForeignSig { uint64_t h = 0; bool ok = false; };
inline ForeignSig foreign_sig() {
    struct HwCtx { uint32_t id, start_col, num_col, pad; int64_t pid; uint64_t submissions, completions, migrations, preemptions, errors; };   // struct amdxdna_drm_query_hwctx
    HwCtx ctx[64];
    ForeignSig sig;
    GetInfo gi{5 /* DRM_AMDXDNA_QUERY_HW_CONTEXTS */, (uint32_t)sizeof ctx, (uint64_t)(uintptr_t)ctx};
    if (!get_info(gi)) return sig;
    const unsigned filled = (sizeof ctx - gi.buffer_size) / sizeof(HwCtx);
    const int64_t me = (int64_t)::getpid();
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    for (unsigned i = 0; i < filled && i < 64; i++) {
        if (ctx[i].pid == me) continue;
        mix((uint64_t)ctx[i].pid); mix(ctx[i].id); mix(ctx[i].submissions); mix(ctx[i].completions);
    }
    sig.h = h;
    sig.ok = true;
    return sig;
}
/// FLM_MINICPM_NPU_GUARD=0 turns the witness off (default on)
inline bool guard_enabled() {
    static const bool on = [] { const char* e = std::getenv("FLM_MINICPM_NPU_GUARD"); return !(e && std::atoi(e) == 0); }();
    return on;
}
/// the stateless decode design asks for the witness only when FLM_MINICPM_NPU_GUARD=1 is set explicitly
inline bool guard_forced() {
    static const bool on = [] { const char* e = std::getenv("FLM_MINICPM_NPU_GUARD"); return e && std::atoi(e) == 1; }();
    return on && guard_enabled();
}

/// "fw 1.1.2.64, XRT 2.25.00": recorded in the log whenever the feature is probed, since the mechanism depends on firmware state
inline std::string versions() {
    struct Fw { uint32_t major, minor, patch, build; } fw{};
    GetInfo gi{8 /* DRM_AMDXDNA_QUERY_FIRMWARE_VERSION */, (uint32_t)sizeof fw, (uint64_t)(uintptr_t)&fw};
    std::string r = "firmware ";
    r += get_info(gi) ? std::to_string(fw.major) + "." + std::to_string(fw.minor) + "." + std::to_string(fw.patch) + "." + std::to_string(fw.build) : std::string("unknown");
    r += ", XRT headers ";
#ifdef FLM_MINICPM_HAVE_XRT_VERSION
    std::string x = XRT_DRIVER_VERSION;
    while (!x.empty() && (x.back() == ',' || x.back() == ' ')) x.pop_back();
#else
    std::string x = "unknown";
#endif
    return r + x;
}

// The transactions of one reconfiguration as apps in one hw_context. begin() starts the first one (so the CPU can work meanwhile);
// finish() waits for it and runs the rest in order. The array is reconfigured when finish() returns.
template <class App, class Run>
struct Chain {
    std::vector<std::unique_ptr<App>> apps;
    std::vector<Run> runs;
    template <class MakeApp>
    void build(const std::string& to_xclbin, const std::string& from_xclbin, MakeApp make_app, bool quiesce = true) {
        for (auto& words : txn_chunks(find_cdo(to_xclbin), find_cdo(from_xclbin), quiesce)) {
            apps.push_back(std::make_unique<App>(make_app()));
            apps.back()->seq()->from_vector(words);
        }
    }
    void begin() { runs.clear(); runs.push_back(apps.at(0)->create_run()); runs[0].start(); }
    void finish() {
        flm::npu::wait_for_completion(runs.at(0), "minicpm reconfig");
        for (size_t k = 1; k < apps.size(); k++) { auto r = apps[k]->create_run(); r.start(); flm::npu::wait_for_completion(r, "minicpm reconfig"); }
        runs.clear();
    }
    // Adds the transactions to a runlist (in order). The caller adds the run that needs this configuration right after them and executes the
    // runlist ONCE: the firmware scheduler can then not run another process's command between the reconfiguration and its consumer (if it did,
    // returning to this hw_context would reload the vision_mm PDI and the reconfiguration would be lost). release() after the runlist's wait().
    template <class Runlist>
    void add_to(Runlist& rl) {
        runs.clear();
        for (auto& a : apps) { runs.push_back(a->create_run()); rl.add(runs.back()); }
    }
    void release() { runs.clear(); }
};

#else
// never used: single_context() is false on this platform
template <class App, class Run>
struct Chain {
    template <class MakeApp> void build(const std::string&, const std::string&, MakeApp, bool = true) {}
    void begin() {}
    void finish() {}
    template <class Runlist> void add_to(Runlist&) {}
    void release() {}
};
inline bool npu_exclusive() { return false; }
struct ForeignSig { uint64_t h = 0; bool ok = false; };
inline ForeignSig foreign_sig() { return {}; }
inline bool guard_enabled() { return false; }
inline bool guard_forced() { return false; }
inline std::string versions() { return "n/a"; }
#endif

}  // namespace minicpm_reconfig
