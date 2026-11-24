// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
#pragma once
/// @file inkling2_launch.hpp
/// @brief aie2p-launchlib control-packet warm switches for the Inkling2 prefill: attention runs in the hardware
///        context of the prompt matmul xclbin, so the array switches app inside one context (about 0.12 ms) instead
///        of between two contexts (2.5-5.5 ms per layer at 2k tokens).
///
///        The switches are built offline by aie2p-launchlib (flm-dev-kit/FLM_Xclbin/Inkling2/switches) from the
///        kernels' own configuration: per kernel K its instruction stream (K.txn) and control packets (K.pkt);
///        XRT argument 3 + packet_arg takes the packet buffer. They ship as ONE file,
///        xclbins/<model>/inkling2.switches:
///            "INK2SW1\n", then per kernel a line "K <packet_arg> <txn bytes> <pkt bytes>\n", a line
///            "END\n", then the blobs in that order (K.txn, K.pkt, next kernel, ...).
///        A directory of the loose files (K.txn, K.pkt, switches.txt) loads too (rig).
///
///        Contract: the array is mm's whenever the engine is not inside an attention call; a call switches to
///        attn, runs, and switches back, all in one submission (window). Needs the xclbins the switches were
///        built with (the mm xclbin is the package's base); nothing checks that, as for every dev-kit artifact.
///        On by default when the model's xclbin folder has inkling2.switches; INKLING2_NO_SWITCH=1 turns it off,
///        INKLING2_SWITCH_DIR=<dir> loads a directory instead. Without it attention uses prefill_attn.xclbin's
///        own context.

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

/// @brief the array to this process for a prefill on the switches (the same lock as the d3 engine's, so the two
///        exclude each other). Warm switches rely on the array staying as the last
///        launch left it, and another process's context (a second FLM server on the same NPU) takes it at any command
///        boundary: the device hangs for every client until the commands time out. So a pass takes a system-wide
///        lock (flock on a file in the temp directory, a named mutex on Windows) from its first layer to its last;
///        the next pass starts every window from cold_mm. A lock that cannot be made leaves the pass unlocked.
class inkling2_array_lock {
#if defined(_WIN32)
    HANDLE h = nullptr;
#else
    int fd = -1;
#endif
public:
    explicit inkling2_array_lock(bool on) {
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
    ~inkling2_array_lock() {
#if defined(_WIN32)
        if (h) { ReleaseMutex(h); CloseHandle(h); }
#else
        if (fd >= 0) close(fd);   // closing drops the lock
#endif
    }
    inkling2_array_lock(const inkling2_array_lock&) = delete;
    inkling2_array_lock& operator=(const inkling2_array_lock&) = delete;
};

struct inkling2_launch {
    bool ready = false;
    bool tried = false;      //!< load() was attempted (the context does it once)
    std::string file;        //!< <xclbin folder>/inkling2.switches
    /// @brief the switches to load: the cold one (the base app's own state), to the guest, back (the decode's:
    ///        cold_dec, dec_then_lmh, lmh_then_dec)
    std::vector<std::string> names = {"cold_mm", "mm_then_attn", "attn_then_mm"};

    /// @brief the switches are, or may still turn out to be, in use (load happens at the first pass)
    bool possible() const {
        if (tried) return ready;
        const char* off = std::getenv("INKLING2_NO_SWITCH");
        const char* dir = std::getenv("INKLING2_SWITCH_DIR");
        return !(off && off[0] == '1') && (!file.empty() || (dir && dir[0]));
    }

#if !defined(FLM_USE_HRX)
    struct Sw {
        std::unique_ptr<xrt::elf> elf;
        std::unique_ptr<xrt::module> mod;
        std::unique_ptr<xrt::ext::kernel> kernel;
        xrt::bo pkt;
        int arg = 0;
    };
    std::map<std::string, Sw> sw;
    long long launches = 0;

    static std::vector<char> _read(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("inkling2_launch: cannot read " + path);
        return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }

    /// @brief one switch: aiebu ELF of the instruction stream, an ext kernel in the mm context, the packet BO
    void _add(const std::string& name, int arg, const char* txn, size_t txn_size, const char* pkt, size_t pkt_size,
              xrt::device* dev, xrt::hw_context* ctx, const std::string& kernel_name) {
        if (std::find(names.begin(), names.end(), name) == names.end()) return;
        void* elf_buf = nullptr;
        const int elf_size = aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction, txn, txn_size,
                                                     nullptr, 0, &elf_buf, nullptr, 0, "", "", nullptr, 0);
        if (elf_size <= 0 || !elf_buf) throw std::runtime_error("inkling2_launch: aiebu failed on " + name);
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
            if (!sw.count(n)) throw std::runtime_error("inkling2_launch: missing switch " + n + " in " + from);
        ready = true;
    }

    /// @brief load the switches of a directory (K.txn, K.pkt, switches.txt) for the mm app manager's context
    ///        (throws on any failure)
    void load(const std::string& dir, xrt::device* dev, xrt::hw_context* ctx, const std::string& kernel_name) {
        std::ifstream meta(dir + "/switches.txt");
        if (!meta) throw std::runtime_error("inkling2_launch: no switches.txt in " + dir);
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
            if (!nl) throw std::runtime_error("inkling2_launch: " + where + "truncated header");
            std::string l(buf.data() + pos, nl - (buf.data() + pos));
            pos = (size_t)(nl - buf.data()) + 1;
            return l;
        };
        if (next_line() != "INK2SW1") throw std::runtime_error("inkling2_launch: " + where + "not an INK2SW1 file");
        struct Ent { std::string name; int arg; size_t txn, pkt; };
        std::vector<Ent> ents;
        for (std::string l = next_line(); l != "END"; l = next_line()) {
            std::istringstream is(l);
            Ent e{};
            if (!(is >> e.name >> e.arg >> e.txn >> e.pkt) || e.arg < 0)
                throw std::runtime_error("inkling2_launch: " + where + "bad entry '" + l + "'");
            ents.push_back(e);
        }
        for (const Ent& e : ents) {
            if (e.txn > buf.size() - pos || e.pkt > buf.size() - pos - e.txn)
                throw std::runtime_error("inkling2_launch: " + where + "truncated data");
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

    /// @brief the array's turn as ONE submission: cold_mm, the switch to the guest, the guest's runs, the switch back.
    ///        Warm switches rely on the array staying as the last launch left it; with the launches apart, another
    ///        process's context can take the array between two of them (it comes back as the xclbin's own state,
    ///        the next launch times out). A runlist goes to the device as one chain.
    template <class Mgr>
    void window(Mgr* mgr, const char* to, const char* back, std::vector<xrt::run>& guest) {
        // cold_mm first, every time: the window then starts from mm's own state whatever ran on the array since
        // the last one (it sets the lanes the control packets rely on)
        {
            xrt::runlist rl = mgr->create_runlist();
            rl.add(make_run(names[0]));
            rl.add(make_run(to));
            for (xrt::run& g : guest) rl.add(g);
            rl.add(make_run(back));
            rl.execute();
            rl.wait();
        }
        launches += 3;
    }
#endif
};
