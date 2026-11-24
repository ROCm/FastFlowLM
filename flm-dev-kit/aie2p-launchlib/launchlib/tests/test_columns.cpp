//===- test_columns.cpp - APP[A-B]: the part of an app in some columns -*- C++ -*-===//
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
//   test_columns PKG APP
//
// "APP[A-B]" (formats/package.hpp) of PKG/APP: it keeps the tiles of columns
// A..B and nothing else; warm kernels in and out of it write only those
// columns, with both transports; a bad range and a census acquire across the
// border stop the load with a message.
//
//===----------------------------------------------------------------------===//

#include "check.hpp"

#include "launch/launch.hpp"

#include <filesystem>
#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

using namespace launch;
namespace fs = std::filesystem;

template <class F> bool says(F f, std::initializer_list<const char *> words) {
  std::string m;
  try {
    f();
  } catch (const std::exception &e) {
    m = e.what();
  }
  bool ok = !m.empty();
  for (const char *w : words)
    ok &= m.find(w) != std::string::npos;
  if (!ok)
    std::fprintf(stderr, "  message: '%s'\n", m.c_str());
  return ok;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: test_columns PKG APP\n");
    return 2;
  }
  const std::string pkg = argv[1], app = argv[2];
  const Package plain = load_package(pkg, app);
  int last = 0;
  for (const Tile &c : plain.cores())
    last = std::max(last, c.col);
  if (last == 0) {
    std::printf("test_columns: SKIP ('%s' has cores in one column)\n", app.c_str());
    return 0;
  }
  const int hi = last / 2;
  const std::string part = app + "[0-" + std::to_string(hi) + "]";

  // The load: the tiles of 0..hi, as the app has them.
  const Package p = load_package(pkg, part);
  CHECK(p.name == part && p.xclbin == plain.xclbin);
  CHECK(!p.tiles.empty());
  for (const auto &[t, init] : p.tiles) {
    CHECK(t.col <= hi);
    CHECK(init.program == plain.tiles.at(t).program && init.bds == plain.tiles.at(t).bds &&
          init.routes == plain.tiles.at(t).routes && init.locks == plain.tiles.at(t).locks);
  }
  for (const auto &[t, init] : plain.tiles)
    CHECK(t.col > hi || p.tiles.count(t));
  for (uint32_t a : p.waits)
    CHECK(hw::locate(a).col <= hi);

  // The warm kernels in and out of it write nothing outside 0..hi.
  for (Transport tr : {Transport::txn, Transport::ctrlpkt}) {
    const auto none = [](Rt &) {};
    ConfigList list{pkg, tr, {cold(app), warm(part, part, {}, "in"), warm(part, part, {}, "out")}};
    Result r;
    try {
      r = build(list, {{app, none}, {part, none}});
    } catch (const std::exception &e) {
      std::fprintf(stderr, "  %s: %s\n", transport_name(tr), e.what());
      CHECK(false);
      continue;
    }
    for (size_t k = 1; k < r.kernels.size(); ++k)
      for (const auto &[a, v] : detail::values_of(txn::parse(r.kernels[k].words).ops))
        if (hw::locate(a).col > hi) {
          std::fprintf(stderr, "  %s %s: writes %#010x (column %d)\n", transport_name(tr), r.kernels[k].name.c_str(),
                       a, hw::locate(a).col);
          CHECK(false);
          break;
        }
    // and less than the whole app's warm kernel
    ConfigList whole{pkg, tr, {cold(app), warm(app, app, {}, "whole")}};
    const Result w = build(whole, {{app, none}});
    CHECK(r.kernels[1].stats.bytes + r.kernels[1].packets.size() * 4 <
          w.kernels[1].stats.bytes + w.kernels[1].packets.size() * 4);
  }

  // Names.
  CHECK(says([&] { load_package(pkg, app + "[1-0]"); }, {"APP[A-B]"}));
  CHECK(says([&] { load_package(pkg, app + "[0-8]"); }, {"APP[A-B]"}));
  CHECK(says([&] { load_package(pkg, app + "[0]"); }, {"APP[A-B]"}));
  CHECK(says([&] { load_package(pkg, app + "[x-1]"); }, {"APP[A-B]"}));

  // A core that waits on a lock across the border.
  const fs::path tmp = fs::temp_directory_path() / ("launchlib_columns_" + std::to_string(::getpid()));
  fs::remove_all(tmp);
  fs::create_directories(tmp);
  fs::copy(fs::path(pkg) / app, tmp / app, fs::copy_options::recursive);
  {
    std::ofstream m(tmp / app / "manifest.txt", std::ios::app);
    m << "census objdump\nacquire " << hi << " 2 " << hi + 1 << " 2 0\n";
  }
  CHECK(says([&] { load_package(tmp.string(), part); }, {"across the border"}));
  CHECK(load_package(tmp.string(), app + "[0-" + std::to_string(hi + 1) + "]").cores().size() > 0);
  fs::remove_all(tmp);
  return done("test_columns");
}
