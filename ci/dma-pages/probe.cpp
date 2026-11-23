// Host side of the DMA page-translation probe (FastFlowLM#1080).
//   probe <xclbin> <insts.bin> <mode> <R> <P> <in_bytes> <out_bytes> <per_ch_bytes> <nch> <backing> [iters]
// backing: bo (xrt::bo, driver memory) | user4k (user pointer, small pages) | user2m (user pointer,
// huge pages: MADV_HUGEPAGE on Linux, MEM_LARGE_PAGES on Windows when the account may use them).
// Prints one CSV line: mode,R,P(pages per channel),backing,median_us,min_us,GBps,Mrows_s,check
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_ext.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

static const size_t PAGE = 4096;

struct Host {
  void *p = nullptr;
  size_t n = 0;
  bool large = false;
};

#ifdef _WIN32
static bool enable_lock_memory() {
  HANDLE tok;
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) return false;
  TOKEN_PRIVILEGES tp{};
  tp.PrivilegeCount = 1;
  tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
  bool ok = LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid) &&
            AdjustTokenPrivileges(tok, FALSE, &tp, sizeof tp, nullptr, nullptr) &&
            GetLastError() == ERROR_SUCCESS;
  CloseHandle(tok);
  return ok;
}

static Host host_alloc(size_t n, bool huge) {
  Host h;
  h.n = (n + (2u << 20) - 1) & ~size_t((2u << 20) - 1);
  if (huge && enable_lock_memory())
    h.p = VirtualAlloc(nullptr, h.n, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
  h.large = h.p != nullptr;
  if (!h.p) h.p = VirtualAlloc(nullptr, h.n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  return h;
}
static std::string page_info(const void *) { return "-"; }
#else
static Host host_alloc(size_t n, bool huge) {
  Host h;
  const size_t A = 2u << 20;
  h.n = (n + A - 1) & ~(A - 1);
  void *raw = mmap(nullptr, h.n + A, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED) return h;
  uintptr_t a = (reinterpret_cast<uintptr_t>(raw) + A - 1) & ~(A - 1);
  h.p = reinterpret_cast<void *>(a);
  madvise(h.p, h.n, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
  std::memset(h.p, 0, h.n);
  h.large = huge;
  return h;
}

// Page facts of the mapping holding p, from /proc/self/smaps.
static std::string page_info(const void *p) {
  std::ifstream f("/proc/self/smaps");
  std::string line, out;
  uintptr_t a = reinterpret_cast<uintptr_t>(p);
  bool in = false;
  while (std::getline(f, line)) {
    unsigned long lo, hi;
    if (std::sscanf(line.c_str(), "%lx-%lx ", &lo, &hi) == 2 && line.find(':') != 5) {
      if (in) break;
      in = a >= lo && a < hi;
      continue;
    }
    if (!in) continue;
    for (const char *k : {"KernelPageSize", "AnonHugePages", "ShmemPmdMapped", "FilePmdMapped"})
      if (line.rfind(k, 0) == 0) {
        std::string v = line.substr(line.find(':') + 1);
        v.erase(0, v.find_first_not_of(' '));
        out += std::string(k) + "=" + v + " ";
      }
  }
  return out.empty() ? "?" : out;
}
#endif

int main(int argc, char **argv) {
  if (argc < 11) {
    std::fprintf(stderr, "usage: probe xclbin insts mode R P in_bytes out_bytes per_ch nch backing [iters]\n");
    return 2;
  }
  std::string xcl = argv[1], ins = argv[2], mode = argv[3];
  size_t R = std::strtoull(argv[4], nullptr, 0), P = std::strtoull(argv[5], nullptr, 0);
  size_t in_b = std::strtoull(argv[6], nullptr, 0), out_b = std::strtoull(argv[7], nullptr, 0);
  size_t per_ch = std::strtoull(argv[8], nullptr, 0), nch = std::strtoull(argv[9], nullptr, 0);
  std::string backing = argv[10];
  int iters = argc > 11 ? std::atoi(argv[11]) : 20;
  // x modes (pages.py design_x): rows `pitch` bytes apart, P = rows per channel
  size_t pitch = argc > 12 ? std::strtoull(argv[12], nullptr, 0) : 0;
  if (pitch) mode += "-k" + std::to_string(pitch / 1024);

  std::vector<uint32_t> instr;
  {
    std::ifstream g(ins, std::ios::binary | std::ios::ate);
    size_t sz = g.tellg();
    g.seekg(0);
    instr.resize(sz / 4);
    g.read(reinterpret_cast<char *>(instr.data()), sz);
  }

  xrt::device dev(0);
  xrt::xclbin xb(xcl);
  dev.register_xclbin(xb);
  xrt::hw_context ctx(dev, xb.get_uuid());
  xrt::kernel k(ctx, "MLIR_AIE");
  xrt::bo bi(dev, instr.size() * 4, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
  std::memcpy(bi.map<void *>(), instr.data(), instr.size() * 4);
  bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  Host ha, hc;
  xrt::bo ba, bc;
  bool large = false;
  if (backing == "bo") {
    ba = xrt::bo(dev, in_b, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
    bc = xrt::bo(dev, out_b, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
  } else {
    bool huge = backing == "user2m";
    ha = host_alloc(in_b, huge);
    hc = host_alloc(out_b, huge);
    if (!ha.p || !hc.p) { std::fprintf(stderr, "host alloc failed\n"); return 1; }
    large = ha.large && hc.large;
    if (getenv("PROBE_CORE_BO")) {
      ba = xrt::bo(dev, ha.p, ha.n, k.group_id(3));
      bc = xrt::bo(dev, hc.p, hc.n, k.group_id(4));
    } else {
      ba = xrt::ext::bo(dev, ha.p, ha.n);
      bc = xrt::ext::bo(dev, hc.p, hc.n);
    }
  }
  uint8_t *A = ha.p ? static_cast<uint8_t *>(ha.p) : ba.map<uint8_t *>();
  uint8_t *C = hc.p ? static_cast<uint8_t *>(hc.p) : bc.map<uint8_t *>();
  for (size_t i = 0; i < in_b; i += 4) {
    uint32_t v = uint32_t(i / 4) * 2654435761u;
    std::memcpy(A + i, &v, 4);
  }
  std::memset(C, 0, out_b);
  ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  std::string pinfo = page_info(A);

  std::vector<double> us;
  for (int i = 0; i < iters + 2; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    auto run = k(3u, bi, unsigned(instr.size()), ba, bc);
    auto st = run.wait();
    auto t1 = std::chrono::steady_clock::now();
    if (st != ERT_CMD_STATE_COMPLETED) { std::fprintf(stderr, "run state %d\n", int(st)); return 1; }
    if (i >= 2) us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // check channel 0 (see pages.py for the layouts): stream element j = slot q, page p
  size_t bad = 0;
  const size_t Q = PAGE / R, P1 = std::min<size_t>(P, 512), G = P / P1;
  if (pitch) {
    const bool sc = mode[1] == 's';  // xscatter: C scattered, xgather: A scattered
    for (size_t j = 0; j < P; ++j)
      bad += std::memcmp(sc ? C + j * pitch : C + j * R, sc ? A + j * R : A + j * pitch, R) != 0;
  } else if (mode == "stream") {
    bad = std::memcmp(A, C, P * PAGE) != 0;
  } else {
    for (size_t q = 0; q < Q; ++q)
      for (size_t g = 0; g < G; ++g)
        for (size_t p1 = 0; p1 < P1; ++p1) {
          size_t j = (q * G + g) * P1 + p1, page = g * P1 + p1;
          const uint8_t *src = mode == "scatter" ? A + j * R : A + page * PAGE + q * R;
          const uint8_t *dst = mode == "scatter" ? C + page * PAGE + q * R : C + j * R;
          bad += std::memcmp(dst, src, R) != 0;
        }
  }
  if (bad && getenv("PROBE_DUMP") && mode == "scatter" && !pitch) {
    // which input row j landed at (page, slot)
    for (size_t page = 0; page < std::min<size_t>(P, 4); ++page) {
      std::fprintf(stderr, "page %zu:", page);
      for (size_t q = 0; q < Q; ++q) {
        uint32_t v;
        std::memcpy(&v, C + page * PAGE + q * R, 4);
        std::fprintf(stderr, " %u", unsigned(v * 0x0E8B2F51u) * 4 / unsigned(R));
      }
      std::fprintf(stderr, "\n");
    }
  }
  std::sort(us.begin(), us.end());
  double med = us[us.size() / 2], mn = us.front();
  double bytes = double(per_ch) * nch;
  double rows = bytes / R;
  std::printf("%s,%zu,%zu,%s%s,%.1f,%.1f,%.2f,%.1f,%s,%s\n", mode.c_str(), R, P, backing.c_str(),
              backing == "user2m" && !large ? "(no-large)" : "", med, mn, bytes / med / 1e3,
              rows / med, bad ? "BAD" : "ok", pinfo.c_str());
  return bad ? 1 : 0;
}
