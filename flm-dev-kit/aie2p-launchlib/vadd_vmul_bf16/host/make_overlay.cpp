// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// make_overlay.cpp - vadd as an overlay app and its op "add" as a state (APP@STATE)
//
//   make_overlay.exe PKG OUT.txn
//
// An overlay design (the FLM overlays, Qwen4's gdn_overlay) loads its programs
// once; the host writes an op's data movement (BDs, routes, chain starts) before
// the op runs. This tool splits vadd's init.txn that way:
//
//   PKG/ovl/   vadd without its data movement in the mem tiles and cores: the
//              programs, the locks, the shims' routes and mux (the shims keep
//              their routes: the task tokens of a Sync go through them)
//   OUT.txn    that data movement: the configuration of op "add"
//
// make_package.py --state ovl@add=OUT.txn then makes "ovl@add": the overlay
// with add set up, which a warm kernel can switch to (host/ovlp.cpp).

#include "launch/formats/package.hpp"
#include "launch/txn/txn.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

using namespace launch;
namespace fs = std::filesystem;

static void write_txn(const fs::path &p, const txn::Header &h, const std::vector<txn::Op> &ops) {
  txn::Stream s;
  s.header = h;
  s.ops = ops;
  const std::vector<uint32_t> w = txn::encode(s);
  std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char *>(w.data()), (std::streamsize)(w.size() * 4));
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: make_overlay.exe PKG OUT.txn\n");
    return 2;
  }
  try {
    const fs::path pkg = argv[1], vadd = pkg / "vadd", ovl = pkg / "ovl";
    const txn::Stream init = detail::read_txn((vadd / "init.txn").string());
    std::vector<txn::Op> base, op;
    size_t moved_words = 0;
    for (const txn::Op &o : init.ops) {
      const uint32_t a = txn::addr_of(o);
      bool data_movement = false;
      // A channel reset mask stays in the base (the library drops it from both).
      const auto *m = std::get_if<txn::MaskWrite32>(&o);
      if (a && !(m && m->mask == hw::kDmaReset) && hw::locate(a).kind != hw::Kind::shim) {
        const hw::Cls c = hw::classify(a);
        data_movement = c == hw::Cls::bd || c == hw::Cls::channel || c == hw::Cls::route;
      }
      (data_movement ? op : base).push_back(o);
      if (data_movement)
        moved_words += txn::op_words(o);
    }
    fs::remove_all(ovl);
    fs::create_directories(ovl);
    for (const char *f : {"design.xclbin", "manifest.txt", "insts.bin"})
      if (fs::exists(vadd / f))
        fs::copy_file(vadd / f, ovl / f);
    write_txn(ovl / "init.txn", init.header, base);
    write_txn(argv[2], init.header, op);
    std::printf("make_overlay: %s/ovl: %zu of vadd's %zu init ops; op add (%s): %zu ops, %zu words\n",
                pkg.string().c_str(), base.size(), init.ops.size(), argv[2], op.size(), moved_words);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "make_overlay: %s\n", e.what());
    return 1;
  }
}
