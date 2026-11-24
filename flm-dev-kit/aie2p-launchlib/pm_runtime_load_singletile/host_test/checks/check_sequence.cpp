//===- check_sequence.cpp - each C++ sequence equals aiecc's ------*- C++ -*-===//
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
//   check_sequence PACKAGE_DIR DESIGN...
//
// For each design, the C++ runtime sequence of sequence.hpp, built with
// launch::Rt, must encode to the package's insts.bin (what aiecc made from the
// Python runtime_sequence) word for word. No NPU. Exit 1 on a difference; the
// first different op of each side is printed (tools/txn_dump.py shows all).
//
//===----------------------------------------------------------------------===//

#include "sequence.hpp"

#include <cstdio>

using namespace launch;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: check_sequence PACKAGE_DIR DESIGN...\n");
    return 2;
  }
  int bad = 0;
  try {
    for (int a = 2; a < argc; ++a) {
      const std::string name = argv[a];
      const Package pkg = load_package(argv[1], name);
      const std::vector<uint32_t> want = txn::words_of(detail::read_file(pkg.dir + "/insts.bin"));
      const txn::Stream aiecc = txn::parse(want);
      txn::Stream mine;
      mine.header = aiecc.header;
      Rt rt(mine.ops, pkg);
      hosttest::sequences().at(name)(rt);
      const std::vector<uint32_t> got = txn::encode(mine);
      const bool same = got == want;
      std::printf("  %-8s %3zu ops, %5zu words: %s\n", name.c_str(), mine.ops.size(), got.size(),
                  same ? "equal to aiecc" : "DIFFERENT");
      if (!same) {
        ++bad;
        for (size_t k = 0; k < std::min(mine.ops.size(), aiecc.ops.size()); ++k)
          if (!(mine.ops[k] == aiecc.ops[k])) {
            std::printf("    first different op: %zu (of %zu, aiecc %zu)\n", k, mine.ops.size(), aiecc.ops.size());
            break;
          }
      }
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "check_sequence: error: %s\n", e.what());
    return 1;
  }
  return bad ? 1 : 0;
}
