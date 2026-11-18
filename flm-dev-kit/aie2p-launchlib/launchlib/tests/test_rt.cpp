//===- test_rt.cpp - the C++ runtime sequence equals aiecc's --------*- C++ -*-===//
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
//   test_rt PKG APP [APP ...]
//
// For each app of an elementwise package (vadd_vmul_bf16: per column c the
// shim channels lhs<c>, rhs<c>, out<c>), the C++ copy of the IRON
// runtime_sequence, built with Rt, must encode to the package's insts.bin
// (aiecc's) word for word. The length per column is read from aiecc's first BD;
// everything else is the C++ encoding. Without arguments: Rt's own checks.
//
//===----------------------------------------------------------------------===//

#include "check.hpp"

#include "launch/rt/rt.hpp"

using namespace launch;

/// The elementwise sequence of vadd_vmul_bf16 (its iron/*_aie.py, i32 words):
/// per column three BDs (lhs, rhs, out with a token), then the pushes in the
/// same order (dma_start_task), then one wait per column.
static void elementwise(Rt &rt, int cols, uint32_t col_words) {
  std::vector<std::pair<ShimChannel, uint32_t>> pushes;
  std::vector<ShimChannel> outs;
  for (int c = 0; c < cols; ++c) {
    const uint32_t off = (uint32_t)c * col_words;
    const char *names[3] = {"lhs", "rhs", "out"};
    for (unsigned i = 0; i < 3; ++i) {
      const ShimChannel ch = rt.alloc(names[i] + std::to_string(c));
      rt.npu_dma_memcpy_nd({.metadata = ch, .bd_id = i, .mem = Arg{i, 4}, .offsets = {0, 0, 0, off},
                            .sizes = {1, 1, 1, col_words}, .strides = {0, 0, 0, 1}, .issue_token = i == 2,
                            .push = false});
      pushes.push_back({ch, i});
    }
    outs.push_back(rt.alloc("out" + std::to_string(c)));
  }
  for (const auto &[ch, bd] : pushes)
    rt.npu_push_bd(ch, bd, bd == 2);
  for (const ShimChannel &ch : outs)
    rt.npu_dma_wait(ch);
}

int main(int argc, char **argv) {
  // Rt's checks, without a package.
  Package none;
  none.name = "none";
  std::vector<txn::Op> ops;
  Rt rt(ops, none);
  CHECK(throws([&] { rt.npu_dma_memcpy_nd({.metadata = {0, hw::Dir::mm2s, 0}, .bd_id = 16, .mem = Arg{0, 4}}); }));
  CHECK(throws([&] { rt.npu_dma_wait({8, hw::Dir::s2mm, 0}); }));
  CHECK(throws([&] { rt.npu_rtp_write("rtp", 0, 1); }));
  CHECK(throws([&] { rt.alloc("lhs0"); }));
  // A linear 512 B read on MM2S 1 of column 2, BD 3, with an offset of 64 elements of 2 B.
  ops.clear();
  rt.npu_dma_memcpy_nd({.metadata = {2, hw::Dir::mm2s, 1}, .bd_id = 3, .mem = Arg{1, 2}, .offsets = {0, 0, 0, 64},
                        .sizes = {1, 1, 1, 256}, .strides = {0, 0, 0, 1}});
  CHECK(ops.size() == 3);
  const auto &bd = std::get<txn::BlockWrite>(ops[0]);
  CHECK(bd.addr == hw::address(2, 0, 0x1D060));
  CHECK((bd.words == std::vector<uint32_t>{128, 128, 0, 0, 0xC0000000, 0x02000000, 0, 0x02000000}));
  CHECK((std::get<txn::AddressPatch>(ops[1]) == txn::AddressPatch{hw::address(2, 0, 0x1D064), 1, 128}));
  CHECK((std::get<txn::Write32>(ops[2]) == txn::Write32{hw::address(2, 0, 0x1D21C), 3}));
  CHECK(rt.args() == std::set<unsigned>{1});

  // aiecc's sequences.
  if (argc >= 3) {
    for (int i = 2; i < argc; ++i) {
      const Package p = load_package(argv[1], argv[i]);
      const std::vector<uint32_t> want = txn::words_of(read_bytes(p.dir + "/insts.bin"));
      const txn::Stream aiecc = txn::parse(want);
      const uint32_t col_words = std::get<txn::BlockWrite>(aiecc.ops.at(0)).words.at(0);
      int cols = 0;
      while (p.allocs.count("lhs" + std::to_string(cols)))
        ++cols;
      txn::Stream mine;
      mine.header = aiecc.header;
      Rt r(mine.ops, p);
      elementwise(r, cols, col_words);
      const std::vector<uint32_t> got = txn::encode(mine);
      CHECK(got == want);
      if (got != want)
        for (size_t k = 0; k < std::min(got.size(), want.size()); ++k)
          if (got[k] != want[k]) {
            std::fprintf(stderr, "  %s: first difference at word %zu: %#x, aiecc %#x\n", argv[i], k, got[k],
                         want[k]);
            break;
          }
      std::printf("  %s: %d columns, %u words per column: %s\n", argv[i], cols, col_words,
                  got == want ? "equal to aiecc" : "DIFFERENT");
    }
  }
  return done("test_rt");
}
