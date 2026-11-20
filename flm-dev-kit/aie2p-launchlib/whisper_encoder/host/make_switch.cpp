//===- make_switch.cpp - the two warm switches of the Whisper encoder -----===//
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
//   make_switch.exe --package build/pkg --out build/switch [--kind warm|cold] [--flush]
//
// Writes OUT/to_attn.txn (warm(mm, attn)) and OUT/to_mm.txn (warm(attn, mm)):
// the reset and load parts only (empty runtime sequences). --kind cold writes
// cold(attn) and cold(mm) under the same names (a whole-array reset first: for
// finding out what a warm switch misses). --flush (experiment) puts a stream
// flush in front of each switch: every core and mem-tile S2MM channel is reset,
// given a lock-free BD into a 16 KB scratch area and started, while the old
// routes still stand, so that words left in the stream switches drain into it;
// then the switch resets the channels again. --zero-dm (experiment): zeros in
// the data memory of every core (and with --zero-mem also every mem tile) right
// before the load part. The Whisper engine
// puts them in front of its own instructions when FLM_WHISPER_SWITCH_DIR=OUT
// (FLM_DLL/detail/whisper_npu/whisper_encoder.hpp, prepend_switch).
//
//===----------------------------------------------------------------------===//

#include "launch/launch.hpp"
#include "launch/hw/aie2p.hpp"
#include "launch/txn/txn.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace hw = launch::hw;
namespace txn = launch::txn;

/// The flush (see the header): ops for every core and mem-tile S2MM channel of the 8 columns.
std::vector<txn::Op> flush_ops() {
  std::vector<txn::Op> ops;
  constexpr uint32_t kWords = 4096; // 16 KB of scratch at data-memory address 0
  for (int col = 0; col < hw::kCols; ++col)
    for (int row = 1; row < hw::kRows; ++row) {
      const hw::Kind kind = hw::kind_of_row(row);
      const hw::Bank ch = hw::channels(kind, hw::Dir::s2mm);
      for (uint32_t c = 0; c < ch.count; ++c) {
        const uint32_t ctrl = hw::address(col, row, ch.at(c));
        ops.push_back(txn::MaskWrite32{ctrl, hw::kDmaReset, hw::kDmaReset});
        ops.push_back(txn::MaskWrite32{ctrl, 0, hw::kDmaReset});
        // a BD the channel may use: core 15 - c; mem tile: even channels 0..23, odd 24..47
        const uint32_t bd = kind == hw::Kind::core ? 15 - c : (c % 2 ? 42 : 18) + c / 2;
        const uint32_t bd_addr = hw::address(col, row, hw::bds(kind).at(bd));
        if (kind == hw::Kind::core)
          ops.push_back(txn::BlockWrite{bd_addr, {kWords, 0, 0, 0, 0, 1u << 25}});            // length, valid
        else
          ops.push_back(txn::BlockWrite{bd_addr, {kWords, 0, 0, 0, 0, 0, 0, 1u << 31}});      // length, valid
        ops.push_back(txn::Write32{ctrl + hw::kQueueFromCtrl, bd});
      }
    }
  // time for buffered words to arrive: the same write again
  for (int i = 0; i < 400; ++i)
    ops.push_back(ops.back());
  return ops;
}

int main(int argc, char **argv) {
  std::string package, out, kind = "warm";
  bool flush = false, zero_dm = false, zero_mem = false;
  for (int i = 1; i < argc; i++) {
    flush |= std::string(argv[i]) == "--flush";
    zero_dm |= std::string(argv[i]) == "--zero-dm";
    zero_mem |= std::string(argv[i]) == "--zero-mem";
  }
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string opt = argv[i];
    if (opt == "--flush" || opt == "--zero-dm" || opt == "--zero-mem") {
      --i;
      continue;
    }
    if (opt == "--package") package = argv[i + 1];
    else if (opt == "--out") out = argv[i + 1];
    else if (opt == "--kind") kind = argv[i + 1];
  }
  if (package.empty() || out.empty()) {
    std::cerr << "usage: make_switch.exe --package DIR --out DIR\n";
    return 2;
  }
  try {
    launch::ConfigList list{package, launch::Transport::txn, {}};
    if (kind == "cold")
      list.configs = {launch::cold("attn", "to_attn"), launch::cold("mm", "to_mm")};
    else
      list.configs = {launch::warm("mm", "attn", {}, "to_attn"), launch::warm("attn", "mm", {}, "to_mm")};
    const auto none = [](launch::Rt &) {};
    const launch::Result r = launch::build(list, {{"mm", none}, {"attn", none}});
    std::filesystem::create_directories(out);
    for (const launch::Kernel &k : r.kernels) {
      const std::string path = out + "/" + k.name + ".txn";
      std::vector<uint32_t> words = k.words;
      if (zero_dm || zero_mem) {
        txn::Stream st = txn::parse(words);
        size_t load = 0;   // the op where the load part starts
        for (const auto &part : k.parts)
          if (part.name == "load")
            load = part.first;
        std::vector<txn::Op> zeros;
        const std::vector<uint32_t> chunk(1024, 0); // 4 KB per write
        for (int col = 0; col < hw::kCols; ++col)
          for (int row = 1; row < hw::kRows; ++row) {
            const bool core = hw::kind_of_row(row) == hw::Kind::core;
            if ((core && !zero_dm) || (!core && !zero_mem))
              continue;
            const uint32_t bytes = core ? hw::kCoreDmBytes : hw::kMemDmBytes;
            for (uint32_t off = 0; off < bytes; off += 4096)
              zeros.push_back(txn::BlockWrite{hw::address(col, row, off), chunk});
          }
        st.ops.insert(st.ops.begin() + (long)load, zeros.begin(), zeros.end());
        words = txn::encode(st);
      }
      if (flush) {
        txn::Stream st = txn::parse(words);
        std::vector<txn::Op> ops = flush_ops();
        ops.insert(ops.end(), st.ops.begin(), st.ops.end());
        st.ops = std::move(ops);
        words = txn::encode(st);
      }
      std::ofstream(path, std::ios::binary)
          .write(reinterpret_cast<const char *>(words.data()), (std::streamsize)(words.size() * 4));
      std::printf("%s: %zu ops, %zu B (program %zu B)\n", path.c_str(), k.stats.ops, k.stats.bytes,
                  k.stats.program_bytes);
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
