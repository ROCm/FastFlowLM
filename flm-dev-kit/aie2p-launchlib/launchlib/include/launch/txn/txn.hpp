//===- txn.hpp - AIE2/AIE2P instruction streams (TXN binaries) ---*- C++ -*-===//
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
// An instruction stream is what the NPU's command processor runs for one
// kernel: 4 header words, then the ops in order. The format is the one of
// mlir-aie (lib/Targets/AIETargetNPU.cpp, AIETranslateNpuToBinary):
//
//   header   word 0  rows << 24 | device generation << 16 | minor << 8 | major
//            word 1  mem-tile rows << 8 | columns
//            word 2  op count
//            word 3  bytes of the whole stream
//   Write32       6 words  opcode 0, 0, address, 0, value, 24
//   MaskWrite32   7 words  opcode 3, 0, address, 0, value, mask, 28
//   MaskPoll      8 words  opcode 4, 0, address, 0, value, mask, 32, 0: wait until
//                          (register & mask) == value (XAIE_IO_MASKPOLL)
//   BlockWrite    4+n      opcode 1, col | row << 8 (or 0), address, bytes, n words
//   Sync          4 words  opcode 0x80, 16, dir | row << 8 | col << 16,
//                          rows << 8 | cols << 16 | channel << 24
//   AddressPatch 12 words  opcode 0x81, 48, 0, 0, 0, 0 (action), address, 0,
//                          kernel argument, 0, byte offset, 0
//
// The order of the ops is the order the command processor writes them: the
// writes of one op land before the next op starts. There are no pages, jobs or
// barriers as on AIE4 (one command processor, not six uCs).
//
// `Stream` is the IR: a header and a list of `Op`. `encode` writes the words,
// `parse` reads them back; parse(encode(s)) == s and encode(parse(w)) == w for
// every stream aiecc writes (tests/test_txn.cpp).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/hw/aie2p.hpp"

#include <cstdint>
#include <format>
#include <stdexcept>
#include <variant>
#include <vector>

namespace launch::txn {

enum Opcode : uint32_t {
  kWrite = 0,
  kBlockWrite = 1,
  kMaskWrite = 3,
  kMaskPoll = 4,
  kPreempt = 6,
  kSync = 0x80,       // XAIE_IO_CUSTOM_OP_TCT
  kAddressPatch = 0x81, // XAIE_IO_CUSTOM_OP_DDR_PATCH
};

struct Write32 {
  uint32_t addr, value;
  bool operator==(const Write32 &) const = default;
};
struct MaskWrite32 {
  uint32_t addr, value, mask;
  bool operator==(const MaskWrite32 &) const = default;
};
/// Wait until (register at `addr` & mask) == value.
struct MaskPoll {
  uint32_t addr, value, mask;
  bool operator==(const MaskPoll &) const = default;
};
struct BlockWrite {
  uint32_t addr;
  std::vector<uint32_t> words;
  uint32_t tile_word = 0; ///< word 1: col | row << 8 when aiecc set it, else 0 (kept for the round trip)
  bool operator==(const BlockWrite &) const = default;
};
/// Wait for the task-complete token of a shim DMA channel.
struct Sync {
  uint32_t col, row, dir, channel, cols = 1, rows = 1;
  bool operator==(const Sync &) const = default;
};
/// At dispatch, the firmware writes the address of kernel argument `arg` plus
/// `offset` bytes into the 64-bit BD address field at `addr`.
struct AddressPatch {
  uint32_t addr, arg, offset;
  bool operator==(const AddressPatch &) const = default;
};
struct Preempt {
  uint32_t level;
  bool operator==(const Preempt &) const = default;
};

using Op = std::variant<Write32, MaskWrite32, MaskPoll, BlockWrite, Sync, AddressPatch, Preempt>;

struct Header {
  uint8_t major = 0, minor = 1, dev_gen = hw::kTxnDevGen, rows = hw::kRows;
  uint8_t cols = hw::kCols, mem_rows = hw::kMemRows;
  bool operator==(const Header &) const = default;
};

struct Stream {
  Header header;
  std::vector<Op> ops;
  bool operator==(const Stream &) const = default;
};

template <class... F> struct Overload : F... {
  using F::operator()...;
};
template <class... F> Overload(F...) -> Overload<F...>;

/// Words of one op.
inline size_t op_words(const Op &op) {
  return std::visit(Overload{
                        [](const Write32 &) -> size_t { return 6; },
                        [](const MaskWrite32 &) -> size_t { return 7; },
                        [](const MaskPoll &) -> size_t { return 8; },
                        [](const BlockWrite &b) -> size_t { return 4 + b.words.size(); },
                        [](const Sync &) -> size_t { return 4; },
                        [](const AddressPatch &) -> size_t { return 12; },
                        [](const Preempt &) -> size_t { return 1; },
                    },
                    op);
}

inline void append(std::vector<uint32_t> &w, const Op &op) {
  std::visit(Overload{
                 [&](const Write32 &o) { w.insert(w.end(), {kWrite, 0, o.addr, 0, o.value, 24}); },
                 [&](const MaskWrite32 &o) { w.insert(w.end(), {kMaskWrite, 0, o.addr, 0, o.value, o.mask, 28}); },
                 [&](const MaskPoll &o) { w.insert(w.end(), {kMaskPoll, 0, o.addr, 0, o.value, o.mask, 32, 0}); },
                 [&](const BlockWrite &o) {
                   w.insert(w.end(), {kBlockWrite, o.tile_word, o.addr, (uint32_t)(4 * (4 + o.words.size()))});
                   w.insert(w.end(), o.words.begin(), o.words.end());
                 },
                 [&](const Sync &o) {
                   w.insert(w.end(), {kSync, 16, (o.dir & 0xff) | (o.row & 0xff) << 8 | (o.col & 0xff) << 16,
                                      (o.rows & 0xff) << 8 | (o.cols & 0xff) << 16 | (o.channel & 0xff) << 24});
                 },
                 [&](const AddressPatch &o) {
                   w.insert(w.end(), {kAddressPatch, 48, 0, 0, 0, 0, o.addr, 0, o.arg, 0, o.offset, 0});
                 },
                 [&](const Preempt &o) { w.push_back(kPreempt | o.level << 8); },
             },
             op);
}

inline std::vector<uint32_t> encode(const Stream &s) {
  std::vector<uint32_t> w;
  const Header &h = s.header;
  w.push_back((uint32_t)h.rows << 24 | (uint32_t)h.dev_gen << 16 | (uint32_t)h.minor << 8 | h.major);
  w.push_back((uint32_t)h.mem_rows << 8 | h.cols);
  w.push_back((uint32_t)s.ops.size());
  w.push_back(0); // the size, below
  for (const Op &op : s.ops)
    append(w, op);
  w[3] = (uint32_t)(w.size() * 4);
  return w;
}

/// Read a stream. Throws std::runtime_error with the op and word index of the
/// first word it does not understand.
inline Stream parse(const std::vector<uint32_t> &w) {
  auto fail = [](const std::string &m) { throw std::runtime_error("TXN: " + m); };
  if (w.size() < 4)
    fail("shorter than its header");
  Stream s;
  s.header = Header{(uint8_t)(w[0] & 0xff), (uint8_t)(w[0] >> 8), (uint8_t)(w[0] >> 16), (uint8_t)(w[0] >> 24),
                    (uint8_t)(w[1] & 0xff), (uint8_t)(w[1] >> 8)};
  if (w[3] != w.size() * 4)
    fail(std::format("header size {} B, stream {} B", w[3], w.size() * 4));
  size_t i = 4;
  auto need = [&](size_t n, uint32_t op) {
    if (i + n > w.size())
      fail(std::format("op {} at word {} runs past the end", op, i));
  };
  for (uint32_t n = 0; n < w[2]; ++n) {
    need(1, n);
    const uint32_t opcode = w[i] & 0xff;
    switch (opcode) {
    case kWrite:
      need(6, n);
      s.ops.push_back(Write32{w[i + 2], w[i + 4]});
      i += 6;
      break;
    case kMaskWrite:
      need(7, n);
      s.ops.push_back(MaskWrite32{w[i + 2], w[i + 4], w[i + 5]});
      i += 7;
      break;
    case kMaskPoll:
      need(8, n);
      s.ops.push_back(MaskPoll{w[i + 2], w[i + 4], w[i + 5]});
      i += 8;
      break;
    case kBlockWrite: {
      need(4, n);
      const size_t words = w[i + 3] / 4;
      if (words < 4)
        fail(std::format("op {}: block write of {} B", n, w[i + 3]));
      need(words, n);
      s.ops.push_back(BlockWrite{w[i + 2], {w.begin() + i + 4, w.begin() + i + words}, w[i + 1]});
      i += words;
      break;
    }
    case kSync:
      need(4, n);
      s.ops.push_back(Sync{(w[i + 2] >> 16) & 0xff, (w[i + 2] >> 8) & 0xff, w[i + 2] & 0xff, w[i + 3] >> 24,
                           (w[i + 3] >> 16) & 0xff, (w[i + 3] >> 8) & 0xff});
      i += 4;
      break;
    case kAddressPatch:
      need(12, n);
      s.ops.push_back(AddressPatch{w[i + 6], w[i + 8], w[i + 10]});
      i += 12;
      break;
    case kPreempt:
      s.ops.push_back(Preempt{w[i] >> 8});
      i += 1;
      break;
    default:
      fail(std::format("op {} at word {}: opcode {:#x} is not supported", n, i, opcode));
    }
  }
  if (i != w.size())
    fail(std::format("{} words after the last op", w.size() - i));
  return s;
}

/// The words of a binary file (as aie-translate writes them).
inline std::vector<uint32_t> words_of(const std::vector<char> &bytes) {
  if (bytes.size() % 4)
    throw std::runtime_error(std::format("TXN: {} bytes is not a whole number of words", bytes.size()));
  std::vector<uint32_t> w(bytes.size() / 4);
  for (size_t i = 0; i < w.size(); ++i)
    w[i] = (uint32_t)(uint8_t)bytes[4 * i] | (uint32_t)(uint8_t)bytes[4 * i + 1] << 8 |
           (uint32_t)(uint8_t)bytes[4 * i + 2] << 16 | (uint32_t)(uint8_t)bytes[4 * i + 3] << 24;
  return w;
}

/// The register address an op writes or polls (0 for Sync and Preempt).
inline uint32_t addr_of(const Op &op) {
  return std::visit(Overload{
                        [](const Write32 &o) { return o.addr; },
                        [](const MaskWrite32 &o) { return o.addr; },
                        [](const MaskPoll &o) { return o.addr; },
                        [](const BlockWrite &o) { return o.addr; },
                        [](const AddressPatch &o) { return o.addr; },
                        [](const Sync &) { return 0u; },
                        [](const Preempt &) { return 0u; },
                    },
                    op);
}

/// The array addresses an op writes (none for polls, patches, Sync, Preempt).
inline std::vector<uint32_t> written(const Op &op) {
  std::vector<uint32_t> out;
  if (const auto *b = std::get_if<BlockWrite>(&op))
    for (size_t i = 0; i < b->words.size(); ++i)
      out.push_back(b->addr + 4 * (uint32_t)i);
  else if (std::holds_alternative<Write32>(op) || std::holds_alternative<MaskWrite32>(op))
    out.push_back(addr_of(op));
  return out;
}

} // namespace launch::txn
