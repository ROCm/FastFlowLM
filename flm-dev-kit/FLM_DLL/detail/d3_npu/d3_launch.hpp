// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
#pragma once
/// @file d3_launch.hpp
/// @brief aie2p-launchlib control-packet warm switches for d3: attention (and later conv1d) run
///        in the hardware context of the mm xclbin, so the array switches app inside one context
///        (about 0.12 ms) instead of between two contexts (2.2-2.8 ms).
///
///        The switches are built offline by aie2p-launchlib (flm-dev-kit/FLM_Xclbin/d3_switches) from
///        the kernels' own configuration: per kernel K its instruction stream (K.txn) and control
///        packets (K.pkt); XRT argument 3 + packet_arg takes the packet buffer. They ship as ONE file,
///        xclbins/<model>/d3.switches:
///            "D3SW1\n", then per kernel a line "K <packet_arg> <txn bytes> <pkt bytes>\n", a line
///            "END\n", then the blobs in that order (K.txn, K.pkt, next kernel, ...).
///        A directory of the loose files (K.txn, K.pkt, switches.txt) loads too (rig). Each switch is
///        its own run, so the mm sequences are the ones d3 always used.
///
///        Contract: the array is mm's whenever d3 is not inside an attention or conv1d call; a call
///        switches to attn (conv), runs, and switches back (to_attn / to_mm, to_conv / conv_to_mm).
///        Needs the xclbins the switches were built with (the mm xclbin is the package's base);
///        nothing checks that, as for every dev-kit artifact.
///        On by default when the model's xclbin folder has d3.switches (and the mm xclbin is
///        mm_rtm2_glu); D3_NO_SWITCH=1 turns it off, D3_SWITCH_DIR=<dir> loads a directory instead.
///        Without it attention uses attn.xclbin's own context and conv1d stays on the host.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

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

struct d3_launch {
    bool ready = false;
    bool has_conv = false;   //!< the mm <-> conv1d switches are loaded too
    bool tried = false;      //!< load() was attempted (the context does it once)
    std::string file;        //!< <xclbin folder>/d3.switches, set by the context when its mm xclbin is the one they were built with

#if !defined(FLM_USE_HRX)
    struct Sw {
        std::unique_ptr<xrt::elf> elf;
        std::unique_ptr<xrt::module> mod;
        std::unique_ptr<xrt::ext::kernel> kernel;
        xrt::bo pkt;
        int arg = 0;
    };
    std::map<std::string, Sw> sw;
    bool cold_done = false;
    long long launches = 0;

    static std::vector<char> _read(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("d3_launch: cannot read " + path);
        return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }

    /// @brief one switch: aiebu ELF of the instruction stream, an ext kernel in the mm context, the packet BO
    void _add(const std::string& name, int arg, const char* txn, size_t txn_size, const char* pkt, size_t pkt_size,
              xrt::device* dev, xrt::hw_context* ctx, const std::string& kernel_name) {
        if (name != "cold_mm" && name != "mm_then_attn" && name != "attn_then_mm" && name != "mm_then_conv" &&
            name != "conv_then_mm")
            return;
        void* elf_buf = nullptr;
        const int elf_size = aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction, txn, txn_size,
                                                     nullptr, 0, &elf_buf, nullptr, 0, "", "", nullptr, 0);
        if (elf_size <= 0 || !elf_buf) throw std::runtime_error("d3_launch: aiebu failed on " + name);
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
        for (const char* n : {"cold_mm", "mm_then_attn", "attn_then_mm"})
            if (!sw.count(n)) throw std::runtime_error(std::string("d3_launch: missing switch ") + n + " in " + from);
        has_conv = sw.count("mm_then_conv") && sw.count("conv_then_mm");
        ready = true;
    }

    /// @brief load the switches of a directory (K.txn, K.pkt, switches.txt) for the mm app manager's context
    ///        (throws on any failure)
    void load(const std::string& dir, xrt::device* dev, xrt::hw_context* ctx, const std::string& kernel_name) {
        std::ifstream meta(dir + "/switches.txt");
        if (!meta) throw std::runtime_error("d3_launch: no switches.txt in " + dir);
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
            if (!nl) throw std::runtime_error("d3_launch: " + where + "truncated header");
            std::string l(buf.data() + pos, nl - (buf.data() + pos));
            pos = (size_t)(nl - buf.data()) + 1;
            return l;
        };
        if (next_line() != "D3SW1") throw std::runtime_error("d3_launch: " + where + "not a D3SW1 file");
        struct Ent { std::string name; int arg; size_t txn, pkt; };
        std::vector<Ent> ents;
        for (std::string l = next_line(); l != "END"; l = next_line()) {
            std::istringstream is(l);
            Ent e{};
            if (!(is >> e.name >> e.arg >> e.txn >> e.pkt) || e.arg < 0)
                throw std::runtime_error("d3_launch: " + where + "bad entry '" + l + "'");
            ents.push_back(e);
        }
        for (const Ent& e : ents) {
            if (e.txn > buf.size() - pos || e.pkt > buf.size() - pos - e.txn)
                throw std::runtime_error("d3_launch: " + where + "truncated data");
            _add(e.name, e.arg, buf.data() + pos, e.txn, buf.data() + pos + e.txn, e.pkt, dev, ctx, kernel_name);
            pos += e.txn + e.pkt;
        }
        _finish(path);
    }

    void run(const std::string& name) {
        Sw& s = sw.at(name);
        xrt::run r(*s.kernel);
        r.set_arg(0, 3);
        r.set_arg(1, 0);
        r.set_arg(2, 0);
        r.set_arg(s.arg, s.pkt);
        r.start();
        r.wait();
        launches++;
    }

    /// @brief the array: mm's -> attn's. The first call also runs cold_mm, which sets the lanes
    ///        control packets rely on.
    void to_attn() { _cold(); run("mm_then_attn"); }
    void to_mm() { run("attn_then_mm"); }
    /// @brief the array: mm's -> conv1d's and back (has_conv)
    void to_conv() { _cold(); run("mm_then_conv"); }
    void conv_to_mm() { run("conv_then_mm"); }

    void _cold() {
        if (!cold_done) { run("cold_mm"); cold_done = true; }
    }
#endif
};
