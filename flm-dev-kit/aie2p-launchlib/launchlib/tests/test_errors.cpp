//===- test_errors.cpp - every mistake gives a clear error ----------*- C++ -*-===//
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
//   test_errors PKG APP
//
// Corner cases of the input. Each one must throw, and the message must name
// what is wrong (the configuration, the file, the op). A copy of PKG/APP is
// changed in a temporary folder for the package cases.
//
//===----------------------------------------------------------------------===//

#include "check.hpp"

#include "launch/launch.hpp"

#include <filesystem>
#include <fstream>
#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

using namespace launch;
namespace fs = std::filesystem;

/// The message of what `f` throws ("" if it does not throw).
template <class F> std::string message(F f) {
  try {
    f();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "";
}

/// `f` throws, with a message that has every one of `words`.
template <class F> bool says(F f, std::initializer_list<const char *> words) {
  const std::string m = message(f);
  bool ok = !m.empty();
  for (const char *w : words)
    ok &= m.find(w) != std::string::npos;
  if (!ok)
    std::fprintf(stderr, "  message: '%s'\n", m.c_str());
  return ok;
}

static void write_words(const fs::path &p, const std::vector<uint32_t> &w) {
  std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char *>(w.data()), (std::streamsize)(w.size() * 4));
}

int main(int argc, char **argv) {
  // The runtime sequence (no package needed beyond a name).
  Package none;
  none.name = "none";
  std::vector<txn::Op> ops;
  Rt rt(ops, none);
  CHECK(says([&] { rt.npu_dma_memcpy_nd({.metadata = {0, hw::Dir::mm2s, 0}, .bd_id = 16, .mem = Arg{0, 4}}); },
             {"BD id 16"}));
  CHECK(says([&] { rt.npu_dma_memcpy_nd({.metadata = {0, hw::Dir::mm2s, 2}, .bd_id = 0, .mem = Arg{0, 4}}); },
             {"not in the array"}));
  CHECK(says([&] { rt.npu_dma_memcpy_nd({.metadata = {0, hw::Dir::mm2s, 0}, .bd_id = 0, .mem = Arg{0, 0}}); },
             {"elem_bytes"}));
  CHECK(says([&] {
          rt.npu_dma_memcpy_nd({.metadata = {0, hw::Dir::mm2s, 0}, .bd_id = 0, .mem = Arg{0, 2},
                                .sizes = {1, 1, 1, 3}});
        },
             {"32-bit words"}));
  CHECK(says([&] {
          rt.npu_dma_memcpy_nd({.metadata = {0, hw::Dir::mm2s, 0}, .bd_id = 0, .mem = Arg{0, 4},
                                .sizes = {1, 1, 0, 8}});
        },
             {"positive"}));
  CHECK(says([&] {
          rt.npu_dma_memcpy_nd({.metadata = {0, hw::Dir::mm2s, 0}, .bd_id = 0, .mem = Arg{0, 4},
                                .sizes = {1, 1, 1, 8}, .burst_length = 100});
        },
             {"burst length 100"}));
  CHECK(says([&] { rt.npu_rtp_write("rtp", 0, 1); }, {"no buffer 'rtp'"}));
  CHECK(says([&] { rt.set_lock_value("l", 1); }, {"no lock 'l'"}));
  CHECK(says([&] { rt.alloc("lhs9"); }, {"no shim_dma_allocation 'lhs9'"}));

  // The list.
  CHECK(says([] { validate(ConfigList{}); }, {"no configuration"}));
  CHECK(says([] { validate(ConfigList{"p", Transport::txn, {rerun("a"), rerun("a")}}); }, {"rerun_a", "second"}));
  CHECK(says([] { validate(ConfigList{"p", Transport::txn, {warm("", "a")}}); }, {"needs prev"}));
  CHECK(says([] { config_from_json(R"({"configs":[{"kind":"hot","app":"a"}]})"); }, {"unknown kind 'hot'"}));
  CHECK(says([] { config_from_json(R"({"configs":[{"kind":"rerun","app":"a","prev":"b"}]})"); },
             {"rerun", "'prev'"}));
  CHECK(says([] { config_from_json("{"); }, {"parse"}));
  CHECK(says([] { config_from_json(R"({"configs":[]})"); }, {"no configuration"}));

  if (argc < 3)
    return done("test_errors");
  const std::string pkg = argv[1], app = argv[2];
  SequenceMap seqs = {{app, [](Rt &r) { r.npu_write32(hw::address(0, 0, 0x1D000), 0); }}};

  // build(): a missing app, a missing sequence, a sequence that throws.
  CHECK(says([&] { build(ConfigList{pkg, Transport::txn, {cold("nothing")}}, seqs); }, {"cold_nothing", "nothing"}));
  CHECK(says([&] { build(ConfigList{pkg, Transport::txn, {cold(app)}}, {}); }, {"no runtime sequence"}));
  SequenceMap throwing = {{app, [](Rt &) { throw std::runtime_error("boom"); }}};
  CHECK(says([&] { build(ConfigList{pkg, Transport::txn, {rerun(app)}}, throwing); }, {"rerun_", "boom"}));
  SequenceMap bad_bd = {{app, [](Rt &r) {
                           r.npu_dma_memcpy_nd({.metadata = {0, hw::Dir::mm2s, 0}, .bd_id = 20, .mem = Arg{0, 4}});
                         }}};
  CHECK(says([&] { build(ConfigList{pkg, Transport::txn, {rerun(app)}}, bad_bd); }, {"rerun_", "BD id 20"}));

  // Broken packages: a copy of PKG/APP, changed one way each time.
  const fs::path tmp = fs::temp_directory_path() / ("launchlib_errors_" + std::to_string(::getpid()));
  auto fresh = [&] {
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    fs::copy(fs::path(pkg) / app, tmp / app, fs::copy_options::recursive);
    return txn::parse(txn::words_of(read_bytes((tmp / app / "init.txn").string())));
  };
  auto load = [&] { load_package(tmp.string(), app); };
  // missing files
  fresh();
  fs::remove(tmp / app / "design.xclbin");
  CHECK(says(load, {"design.xclbin"}));
  fresh();
  fs::remove(tmp / app / "init.txn");
  CHECK(says(load, {"init.txn"}));
  // a truncated init.txn
  {
    txn::Stream s = fresh();
    std::vector<uint32_t> w = txn::encode(s);
    w.resize(w.size() - 3);
    write_words(tmp / app / "init.txn", w);
    CHECK(says(load, {"init.txn", "TXN"}));
  }
  // another device (npu1: generation 3)
  {
    txn::Stream s = fresh();
    s.header.dev_gen = 3;
    write_words(tmp / app / "init.txn", txn::encode(s));
    CHECK(says(load, {"init.txn", "generation 3"}));
  }
  // a write to a register without a class (Core_Status)
  {
    txn::Stream s = fresh();
    s.ops.push_back(txn::Write32{hw::address(0, 2, 0x32004), 1});
    write_words(tmp / app / "init.txn", txn::encode(s));
    CHECK(says(load, {"init.txn", "op ", "0x32004", "no register class"}));
  }
  // a wait in the init sequence
  {
    txn::Stream s = fresh();
    s.ops.push_back(txn::Sync{0, 0, 0, 0});
    write_words(tmp / app / "init.txn", txn::encode(s));
    CHECK(says(load, {"init.txn", "only writes"}));
  }
  // a block write that runs from the mem-tile BDs into the DMA channels
  {
    txn::Stream s = fresh();
    s.ops.push_back(txn::BlockWrite{hw::address(0, 1, 0xA05F8), std::vector<uint32_t>(4, 0)});
    write_words(tmp / app / "init.txn", txn::encode(s));
    CHECK(says(load, {"init.txn", "crosses a register class border"}));
  }
  // a block write that runs from the core locks into no class at all
  {
    txn::Stream s = fresh();
    s.ops.push_back(txn::BlockWrite{hw::address(0, 2, 0x1F0F0), std::vector<uint32_t>(8, 0)});
    write_words(tmp / app / "init.txn", txn::encode(s));
    CHECK(says(load, {"init.txn", "0x1f10c", "no register class"}));
  }
  // a channel mask write that is not the reset bit
  {
    txn::Stream s = fresh();
    s.ops.push_back(txn::MaskWrite32{hw::address(0, 2, 0x1DE00), 8, 8});
    write_words(tmp / app / "init.txn", txn::encode(s));
    CHECK(says(load, {"init.txn", "mask 0x8"}));
  }
  // a broken manifest line
  {
    fresh();
    std::ofstream(tmp / app / "manifest.txt", std::ios::app) << "lock only_a_name\n";
    CHECK(says(load, {"manifest.txt", "cannot read", "only_a_name"}));
    fresh();
    std::ofstream(tmp / app / "manifest.txt", std::ios::app) << "alloc x sideways 0 0\n";
    CHECK(says(load, {"manifest.txt", "sideways"}));
  }
  // a core with no program
  {
    txn::Stream s = fresh();
    std::erase_if(s.ops, [](const txn::Op &op) {
      return !std::holds_alternative<txn::Sync>(op) && hw::classify(txn::addr_of(op)) == hw::Cls::program;
    });
    write_words(tmp / app / "init.txn", txn::encode(s));
    CHECK(says(load, {"init.txn", "no program"}));
  }
  // the untouched copy loads
  fresh();
  CHECK(!throws(load));
  fs::remove_all(tmp);
  return done("test_errors");
}
