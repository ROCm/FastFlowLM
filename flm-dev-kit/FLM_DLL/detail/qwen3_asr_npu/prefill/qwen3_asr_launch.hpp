// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
#pragma once
/// \file qwen3_asr_launch.hpp
/// \brief aie2p-launchlib control-packet warm switches for the prompt prefill: the attention kernel runs in the hardware
///        context of prefill.xclbin, so the array switches app inside one context (about 0.1 ms) instead of between two
///        contexts (~2.5 ms each way, per layer).
///
///        The switches are built offline by aie2p-launchlib (flm-dev-kit/FLM_Xclbin/r2t2_switches) from the kernels' own
///        configuration: per kernel K its instruction stream and control packets; XRT argument 3 + packet_arg takes the
///        packet buffer. They ship as ONE file, xclbins/<model>/r2t2.switches:
///            "R2SW1\n", then per kernel a line "K <packet_arg> <txn bytes> <pkt bytes>\n", a line "END\n", then the
///            blobs in that order (K.txn, K.pkt, next kernel, ...).
///        Kernels used: cold_mm, mm_then_attn, attn_then_mm.
///
///        Contract: the array is mm's whenever the prefill is not inside an attention call; a call switches to attn, runs,
///        and switches back, all in one submission (window), which starts from cold_mm whatever ran on the array before.
///        Needs the xclbins the switches were built with (prefill.xclbin is the package's base); nothing checks that, as
///        for every dev-kit artifact. QWEN3_ASR_NO_SWITCH=1 turns it off (attention then uses attn.xclbin's own context).
///        A copy of d3's launcher (detail/d3_npu/d3_launch.hpp) under its own namespace: engines share no inline headers.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if !defined(FLM_USE_HRX)
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <aiebu/aiebu.h>
#endif

namespace qwen3_asr_launch {

/// \brief the array to this process for one window. Warm switches rely on the array staying as the last launch left it;
///        another process on the switches (a second r2t2 server, a d3 server: the same lock) must not run its window in
///        between. flock on a file in the temp directory, a named mutex on Windows; a lock that cannot be made leaves the
///        window unlocked.
class array_lock {
#if defined(_WIN32)
    HANDLE h = nullptr;
#else
    int fd = -1;
#endif
public:
    explicit array_lock(bool on) {
        if (!on) return;
#if defined(_WIN32)
        h = CreateMutexA(nullptr, FALSE, "Global\\flm_d3_array");
        if (h && WaitForSingleObject(h, INFINITE) == WAIT_FAILED) { CloseHandle(h); h = nullptr; }
#else
        const char* tmp = std::getenv("TMPDIR");
        const std::string path = std::string(tmp && tmp[0] ? tmp : "/tmp") + "/flm_d3_array.lock";
        const mode_t old = umask(0);   // shared between users
        fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
        umask(old);
        if (fd >= 0 && flock(fd, LOCK_EX) != 0) { close(fd); fd = -1; }
#endif
    }
    ~array_lock() {
#if defined(_WIN32)
        if (h) { ReleaseMutex(h); CloseHandle(h); }
#else
        if (fd >= 0) close(fd);   // closing drops the lock
#endif
    }
    array_lock(const array_lock&) = delete;
    array_lock& operator=(const array_lock&) = delete;
};

#if !defined(FLM_USE_HRX)
struct switches {
    struct Sw {
        std::unique_ptr<xrt::elf> elf;
        std::unique_ptr<xrt::module> mod;
        std::unique_ptr<xrt::ext::kernel> kernel;
        xrt::bo pkt;
        int arg = 0;
    };
    std::map<std::string, Sw> sw;

    /// \brief one switch: aiebu ELF of the instruction stream, an ext kernel in the mm context, the packet BO
    void _add(const std::string& name, int arg, const char* txn, size_t txn_size, const char* pkt, size_t pkt_size,
              xrt::device* dev, xrt::hw_context* ctx, const std::string& kernel_name) {
        if (name != "cold_mm" && name != "mm_then_attn" && name != "attn_then_mm") return;
        void* elf_buf = nullptr;
        const int elf_size = aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction, txn, txn_size,
                                                     nullptr, 0, &elf_buf, nullptr, 0, "", "", nullptr, 0);
        if (elf_size <= 0 || !elf_buf) throw std::runtime_error("aiebu failed on " + name);
        Sw s;
        s.elf = std::make_unique<xrt::elf>((char*)elf_buf, (size_t)elf_size);
        free(elf_buf);
        s.mod = std::make_unique<xrt::module>(*s.elf);
        s.kernel = std::make_unique<xrt::ext::kernel>(*ctx, *s.mod, kernel_name);
        s.arg = 3 + arg;
        s.pkt = xrt::bo(*dev, std::max<size_t>(pkt_size, 4), XRT_BO_FLAGS_HOST_ONLY, s.kernel->group_id(s.arg));
        std::memcpy(s.pkt.map<void*>(), pkt, pkt_size);
        s.pkt.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        sw.emplace(name, std::move(s));
    }

    /// \brief load the packed switches file (see the top) for the mm app manager's context (throws on any failure)
    void load_file(const std::string& path, xrt::device* dev, xrt::hw_context* ctx, const std::string& kernel_name) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot read " + path);
        const std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const std::string where = path + ": ";
        size_t pos = 0;
        auto next_line = [&]() {
            const char* nl = (const char*)std::memchr(buf.data() + pos, '\n', buf.size() - pos);
            if (!nl) throw std::runtime_error(where + "truncated header");
            std::string l(buf.data() + pos, nl - (buf.data() + pos));
            pos = (size_t)(nl - buf.data()) + 1;
            return l;
        };
        if (next_line() != "R2SW1") throw std::runtime_error(where + "not an R2SW1 file");
        struct Ent { std::string name; int arg; size_t txn, pkt; };
        std::vector<Ent> ents;
        for (std::string l = next_line(); l != "END"; l = next_line()) {
            std::istringstream is(l);
            Ent e{};
            if (!(is >> e.name >> e.arg >> e.txn >> e.pkt) || e.arg < 0) throw std::runtime_error(where + "bad entry '" + l + "'");
            ents.push_back(e);
        }
        for (const Ent& e : ents) {
            if (e.txn > buf.size() - pos || e.pkt > buf.size() - pos - e.txn) throw std::runtime_error(where + "truncated data");
            _add(e.name, e.arg, buf.data() + pos, e.txn, buf.data() + pos + e.txn, e.pkt, dev, ctx, kernel_name);
            pos += e.txn + e.pkt;
        }
        for (const char* n : {"cold_mm", "mm_then_attn", "attn_then_mm"})
            if (!sw.count(n)) throw std::runtime_error(where + "missing switch " + n);
    }

    /// \brief one switch as a run, not started
    xrt::run make_run(const std::string& name) {
        Sw& s = sw.at(name);
        xrt::run r(*s.kernel);
        r.set_arg(0, 3);
        r.set_arg(1, 0);
        r.set_arg(2, 0);
        r.set_arg(s.arg, s.pkt);
        return r;
    }

    /// \brief the array's turn as ONE submission: cold_mm, the switch to attention, its runs, the switch back (a runlist
    ///        goes to the device as one chain, so no other context takes the array between two of them)
    template <class Mgr>
    void window(Mgr* mgr, std::vector<xrt::run>& guest) {
        array_lock lock(true);
        xrt::runlist rl = mgr->create_runlist();
        rl.add(make_run("cold_mm"));
        rl.add(make_run("mm_then_attn"));
        for (xrt::run& g : guest) rl.add(g);
        rl.add(make_run("attn_then_mm"));
        rl.execute();
        rl.wait();
    }
};

/// \brief the switches of `path` in mgr's hardware context, or nullptr (QWEN3_ASR_NO_SWITCH=1, no file, or a load failure,
///        logged once by the caller through `why`)
template <class Mgr>
std::unique_ptr<switches> load(Mgr* mgr, const std::string& path, std::string& why) {
    const char* off = std::getenv("QWEN3_ASR_NO_SWITCH");
    if (off && off[0] == '1') { why = "QWEN3_ASR_NO_SWITCH"; return nullptr; }
    {
        std::ifstream probe(path, std::ios::binary);
        if (!probe) { why = "no " + path; return nullptr; }
    }
    try {
        auto s = std::make_unique<switches>();
        s->load_file(path, mgr->get_device(), mgr->get_context(), mgr->get_kernel_name());
        return s;
    } catch (const std::exception& e) {
        why = e.what();
        return nullptr;
    }
}
#else
struct switches {};   // HRX: no switches, attention uses attn.xclbin's own context
template <class Mgr>
std::unique_ptr<switches> load(Mgr*, const std::string&, std::string& why) {
    why = "not on HRX";
    return nullptr;
}
#endif

}  // namespace qwen3_asr_launch
