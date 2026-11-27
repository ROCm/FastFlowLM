// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
#pragma once
/// @file qwen3_5vl_launch.hpp
/// @brief aie2p-launchlib control-packet warm switches for the Qwen3.5 prefill: conv1d, the gated delta net and
///        gated attention run in the hardware context of the prompt matmul xclbin (mm_rtm2_glu), so the array
///        switches app inside one context (about 0.1 ms) instead of between two contexts (about 2.6 ms each, 88 per
///        prompt on the 4B).
///
///        The switches are built offline by aie2p-launchlib (flm-dev-kit/FLM_Xclbin/Qwen3_5/prefill_switches) from
///        the kernels' own configuration: per switch K its instruction stream (K.txn) and control packets (K.pkt);
///        XRT argument 3 + packet_arg takes the packet buffer. They ship as ONE file,
///        xclbins/<model>/qwen3_5.switches:
///            "Q35SW1\n", then per switch a line "K <packet_arg> <txn bytes> <pkt bytes>\n", a line
///            "END\n", then the blobs in that order (K.txn, K.pkt, next switch, ...).
///        A directory of the loose files (K.txn, K.pkt, switches.txt) loads too (rig).
///
///        Contract: the array is mm's whenever the engine is not inside a window; a window switches to one guest
///        or more, runs them, and switches back to mm, all in one submission. Needs the xclbins the switches were
///        built with (the mm xclbin is the package's base); nothing checks that, as for every dev-kit artifact.
///        On by default when the model's xclbin folder has qwen3_5.switches and the prefill matmul is
///        mm_rtm2_glu; QWEN35_NO_SWITCH=1 turns it off, QWEN35_SWITCH_DIR=<dir> loads a directory instead.
///        Without it conv.xclbin, GateDeltaNet_prefill.xclbin and attn.xclbin keep their own contexts.
///
///        The vision encoder (qwen3_5vl_image.cpp) has a second instance, switches cold_mm, mm_then_attn and
///        attn_then_mm for vision_mm.xclbin and vision_attn.xclbin (FLM_Xclbin/Qwen3_5/vision_switches), file
///        qwen3_5_vision.switches, environment QWEN35_VISION_NO_SWITCH / QWEN35_VISION_SWITCH_DIR.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
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

/// @brief the array to this process for a prefill on the switches (the same lock as the d3 and Inkling2 engines',
///        so they exclude each other). Warm switches rely on the array staying as the last launch left it, and
///        another process's context (a second FLM server on the same NPU) takes it at any command boundary: the
///        device hangs for every client until the commands time out. So a pass takes a system-wide lock (flock on
///        a file in the temp directory, a named mutex on Windows) from its first layer to its last; every window
///        starts from cold_mm. A lock that cannot be made leaves the pass unlocked.
class qwen3_5vl_array_lock {
#if defined(_WIN32)
    HANDLE h = nullptr;
#else
    int fd = -1;
#endif
public:
    explicit qwen3_5vl_array_lock(bool on) {
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
    ~qwen3_5vl_array_lock() {
#if defined(_WIN32)
        if (h) { ReleaseMutex(h); CloseHandle(h); }
#else
        if (fd >= 0) close(fd);   // closing drops the lock
#endif
    }
    qwen3_5vl_array_lock(const qwen3_5vl_array_lock&) = delete;
    qwen3_5vl_array_lock& operator=(const qwen3_5vl_array_lock&) = delete;
};

struct qwen3_5vl_launch {
    bool ready = false;
    /// @brief the switches to load: the cold one (mm's own state), and to and from each guest
    std::vector<std::string> names = {"cold_mm", "mm_then_conv", "conv_then_gdn", "gdn_then_mm",
                                      "mm_then_attn", "attn_then_mm"};
    /// @brief prefix of the instance's environment switches: <env>_NO_SWITCH, <env>_SWITCH_DIR
    std::string env = "QWEN35";

    qwen3_5vl_launch() = default;
    qwen3_5vl_launch(std::vector<std::string> n, std::string e) : names(std::move(n)), env(std::move(e)) {}

#if !defined(FLM_USE_HRX)
    struct Sw {
        std::unique_ptr<xrt::elf> elf;
        std::unique_ptr<xrt::module> mod;
        std::unique_ptr<xrt::ext::kernel> kernel;
        xrt::bo pkt;
        int arg = 0;
    };
    std::map<std::string, Sw> sw;

    /// @brief one hop of a window: the switch into a guest, then the guest's runs
    struct hop {
        const char* to;
        std::vector<xrt::run> runs;
    };

    static std::vector<char> _read(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("qwen3_5vl_launch: cannot read " + path);
        return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }

    /// @brief one switch: aiebu ELF of the instruction stream, an ext kernel in the mm context, the packet BO
    void _add(const std::string& name, int arg, const char* txn, size_t txn_size, const char* pkt, size_t pkt_size,
              xrt::device* dev, xrt::hw_context* ctx, const std::string& kernel_name) {
        if (std::find(names.begin(), names.end(), name) == names.end()) return;
        void* elf_buf = nullptr;
        const int elf_size = aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction, txn, txn_size,
                                                     nullptr, 0, &elf_buf, nullptr, 0, "", "", nullptr, 0);
        if (elf_size <= 0 || !elf_buf) throw std::runtime_error("qwen3_5vl_launch: aiebu failed on " + name);
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

    void _finish(const std::string& from) {
        for (const std::string& n : names)
            if (!sw.count(n)) throw std::runtime_error("qwen3_5vl_launch: missing switch " + n + " in " + from);
        ready = true;
    }

    /// @brief load the switches of a directory (K.txn, K.pkt, switches.txt) for the mm app manager's context
    ///        (throws on any failure)
    void load(const std::string& dir, xrt::device* dev, xrt::hw_context* ctx, const std::string& kernel_name) {
        std::ifstream meta(dir + "/switches.txt");
        if (!meta) throw std::runtime_error("qwen3_5vl_launch: no switches.txt in " + dir);
        for (std::string line; std::getline(meta, line);) {
            std::istringstream is(line);
            std::string name, key;
            int arg = -1;
            is >> name >> key >> arg;
            if (name.empty() || key != "packet_arg" || arg < 0) continue;
            const std::vector<char> txn = _read(dir + "/" + name + ".txn");
            const std::vector<char> pkt = _read(dir + "/" + name + ".pkt");
            _add(name, arg, txn.data(), txn.size(), pkt.data(), pkt.size(), dev, ctx, kernel_name);
        }
        _finish(dir);
    }

    /// @brief load the packed switches file (see the top) for the mm app manager's context (throws on any failure)
    void load_file(const std::string& path, xrt::device* dev, xrt::hw_context* ctx, const std::string& kernel_name) {
        const std::vector<char> buf = _read(path);
        const std::string where = path + ": ";
        size_t pos = 0;
        auto next_line = [&]() {
            const char* nl = (const char*)std::memchr(buf.data() + pos, '\n', buf.size() - pos);
            if (!nl) throw std::runtime_error("qwen3_5vl_launch: " + where + "truncated header");
            std::string l(buf.data() + pos, nl - (buf.data() + pos));
            pos = (size_t)(nl - buf.data()) + 1;
            return l;
        };
        if (next_line() != "Q35SW1") throw std::runtime_error("qwen3_5vl_launch: " + where + "not a Q35SW1 file");
        struct Ent { std::string name; int arg; size_t txn, pkt; };
        std::vector<Ent> ents;
        for (std::string l = next_line(); l != "END"; l = next_line()) {
            std::istringstream is(l);
            Ent e{};
            if (!(is >> e.name >> e.arg >> e.txn >> e.pkt) || e.arg < 0)
                throw std::runtime_error("qwen3_5vl_launch: " + where + "bad entry '" + l + "'");
            ents.push_back(e);
        }
        for (const Ent& e : ents) {
            if (e.txn > buf.size() - pos || e.pkt > buf.size() - pos - e.txn)
                throw std::runtime_error("qwen3_5vl_launch: " + where + "truncated data");
            _add(e.name, e.arg, buf.data() + pos, e.txn, buf.data() + pos + e.txn, e.pkt, dev, ctx, kernel_name);
            pos += e.txn + e.pkt;
        }
        _finish(path);
    }

    /// @brief one switch as a run, not started
    xrt::run make_run(const std::string& name) {
        Sw& s = sw.at(name);
        xrt::run r(*s.kernel);
        r.set_arg(0, 3);
        r.set_arg(1, 0);
        r.set_arg(2, 0);
        r.set_arg(s.arg, s.pkt);
        return r;
    }

    /// @brief the array's turn as ONE submission: cold_mm, then per hop the switch into its guest and the guest's
    ///        runs, then the switch back to mm. Warm switches rely on the array staying as the last launch left
    ///        it; a runlist goes to the device as one chain. cold_mm first, every time: the window then starts from
    ///        mm's own state whatever ran on the array since the last one (it sets the lanes the control packets
    ///        rely on).
    template <class Mgr>
    void window(Mgr* mgr, std::vector<hop>& hops, const char* back) {
        xrt::runlist rl = mgr->create_runlist();
        rl.add(make_run(names[0]));
        for (hop& h : hops) {
            rl.add(make_run(h.to));
            for (xrt::run& r : h.runs) rl.add(r);
        }
        rl.add(make_run(back));
        rl.execute();
        rl.wait();
    }
#endif

    /// @brief load the switches for the mm app manager's context: <env>_SWITCH_DIR (a directory, rig), else
    ///        `file`, unless <env>_NO_SWITCH=1. A failure is logged and leaves every kernel in its own context.
    template <class Mgr>
    bool ensure(Mgr* gemm, const std::string& file) {
#if !defined(FLM_USE_HRX)
        const char* off = std::getenv((env + "_NO_SWITCH").c_str());
        if (!gemm || (off && off[0] == '1')) return false;
        const char* sw_dir = std::getenv((env + "_SWITCH_DIR").c_str());
        const bool dir = sw_dir && sw_dir[0];
        if (!dir && (file.empty() || !std::ifstream(file).good())) return false;
        try {
            if (dir) load(sw_dir, gemm->get_device(), gemm->get_context(), gemm->get_kernel_name());
            else load_file(file, gemm->get_device(), gemm->get_context(), gemm->get_kernel_name());
        } catch (const std::exception& e) {
            sw.clear();
            ready = false;
            header_print("qwen3.5", env << " control-packet switches off: " << e.what());
        }
        return ready;
#else
        (void)gemm; (void)file;
        return false;
#endif
    }
};
