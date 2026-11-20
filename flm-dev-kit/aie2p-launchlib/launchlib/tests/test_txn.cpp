//===- test_txn.cpp - TXN encode/parse and register classes ------*- C++ -*-===//
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
//   test_txn [FILE.bin ...]
//
// Without files: a stream built in C++ survives encode and parse, and the
// register classes of known addresses. With files (aiecc's insts.bin or
// configure binaries, e.g. a package's init.txn): each one parses, encodes back
// to the same bytes, and every write of it has a register class.
//
//===----------------------------------------------------------------------===//

#include "check.hpp"

#include "launch/txn/txn.hpp"

using namespace launch;

int main(int argc, char **argv) {
  // A stream of every op kind survives the round trip.
  txn::Stream s;
  s.ops = {txn::Write32{hw::address(1, 2, 0x1F010), 2},
           txn::MaskWrite32{hw::address(0, 2, hw::kCoreControl), 2, 2},
           txn::MaskPoll{hw::address(0, 2, 0x1D000), 1, 0xFFFFFFFF},
           txn::BlockWrite{hw::address(0, 2, hw::kPm), {1, 2, 3}},
           txn::AddressPatch{hw::address(0, 0, 0x1D004), 1, 64},
           txn::Sync{3, 0, 0, 1},
           txn::Preempt{1}};
  const std::vector<uint32_t> w = txn::encode(s);
  CHECK(w[2] == 7);
  CHECK(w[3] == w.size() * 4);
  CHECK(w[0] == (6u << 24 | 4u << 16 | 1u << 8));
  CHECK(txn::parse(w) == s);
  std::vector<uint32_t> bad = w;
  bad[3] += 4;
  CHECK(throws([&] { txn::parse(bad); }));

  // Register classes (offsets from aie-rt xaie2pgbl_params.h).
  CHECK(hw::classify(hw::address(0, 2, 0x20000)) == hw::Cls::program);
  CHECK(hw::classify(hw::address(0, 2, 0x23FFC)) == hw::Cls::program);
  CHECK(hw::classify(hw::address(0, 2, 0x00400)) == hw::Cls::data);
  CHECK(hw::classify(hw::address(0, 1, 0x7FFFC)) == hw::Cls::data);
  CHECK(hw::classify(hw::address(0, 2, 0x1F0F0)) == hw::Cls::lock);
  CHECK(hw::classify(hw::address(0, 1, 0xC03F0)) == hw::Cls::lock);
  CHECK(hw::classify(hw::address(0, 0, 0x14050)) == hw::Cls::lock);
  CHECK(hw::classify(hw::address(0, 2, 0x1D1E0)) == hw::Cls::bd);
  CHECK(hw::classify(hw::address(0, 1, 0xA05E0)) == hw::Cls::bd);
  CHECK(hw::classify(hw::address(0, 2, 0x1DE18)) == hw::Cls::channel);
  CHECK(hw::classify(hw::address(0, 2, 0x1DE1C)) == hw::Cls::channel);
  CHECK(hw::classify(hw::address(0, 1, 0xA065C)) == hw::Cls::channel);
  CHECK(hw::classify(hw::address(0, 0, 0x1D204)) == hw::Cls::channel);
  CHECK(hw::classify(hw::address(0, 2, 0x3F000)) == hw::Cls::route);
  CHECK(hw::classify(hw::address(7, 1, 0xB0000)) == hw::Cls::route);
  CHECK(hw::classify(hw::address(0, 0, 0x1F004)) == hw::Cls::mux);
  CHECK(hw::classify(hw::address(0, 2, 0x32000)) == hw::Cls::core_control);
  CHECK(throws([] { hw::classify(hw::address(0, 2, 0x32004)); })); // Core_Status: not a class
  CHECK(throws([] { hw::classify(hw::address(8, 2, 0)); }));       // column 8 is outside npu2

  // aiecc's binaries.
  for (int i = 1; i < argc; ++i) {
    const std::vector<uint32_t> words = txn::words_of(read_bytes(argv[i]));
    txn::Stream f;
    try {
      f = txn::parse(words);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "%s: %s\n", argv[i], e.what());
      ++g_failures;
      continue;
    }
    CHECK(txn::encode(f) == words);
    int unclassified = 0;
    for (const txn::Op &op : f.ops) {
      if (std::holds_alternative<txn::Sync>(op) || std::holds_alternative<txn::Preempt>(op))
        continue;
      try {
        hw::classify(txn::addr_of(op));
      } catch (const std::exception &e) {
        if (++unclassified <= 3)
          std::fprintf(stderr, "%s: %s\n", argv[i], e.what());
      }
    }
    CHECK(unclassified == 0);
    std::printf("  %s: %zu ops, round trip %s\n", argv[i], f.ops.size(),
                txn::encode(f) == words ? "ok" : "DIFFERS");
  }
  return done("test_txn");
}
