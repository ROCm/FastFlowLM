//===- ctrlpkt_packets.hpp - AIE2P control packets: encode, decode, check -*- C++ -*-===//
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
// A control packet writes 1 to 4 consecutive words of one tile. A shim DMA
// sends them down a lane of packet routes (ctrlpkt_lane.hpp) to the tile's
// TILE_CTRL port, which writes them. The encoding is ryzenai-corelib's
// (core/npu/direct/AIE2P.cpp `Packets`, measured there on npu5):
//
//   word 0    stream header: bits 4:0 the packet id (the lane routes on it),
//             bit 31 odd parity
//   word 1    control header: bits 19:0 the tile offset, 21:20 beats - 1,
//             23:22 operation (0: write), 28:24 return id (0), bit 31 odd parity
//   then      `beats` data words
//
// A write of n words is cut into packets of at most 4 words that do not cross a
// 16-byte line (beats = min(left, 4, 4 - offset % 16 / 4)).
//
// What a packet may write (corelib `Allowed`): a packet into a DMA channel's
// Ctrl can stall the column, one into a running core's stack can hang it. So:
//
//   core      PM, data memory, BDs; single words: Core_Control, the lock values,
//             the Start_Queue of each S2MM and MM2S channel
//   mem tile  BDs; single words: the lock values, the Start_Queues;
//             data memory only with CtrlPktOptions::memtile_data (corelib does
//             not write it by packet)
//   shim      nothing (the shim's own registers go by TXN writes)
//
// `decode` reads a stream of packets back, checking every field; the validator
// (ctrlpkt_check.hpp) replays a kernel's packets with it.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/hw/aie2p.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <vector>

namespace launch::ctrlpkt {

inline constexpr uint32_t odd_parity(uint32_t w) {
  w &= 0x7FFFFFFF;
  return w | (std::popcount(w) % 2 ? 0u : 1u << 31);
}

inline constexpr uint32_t kOffsetBits = 0xFFFFF;
inline constexpr unsigned kBeatsShift = 20;
inline constexpr size_t kMaxBeats = 4;

/// `words` into consecutive words of a tile from byte offset `offset`.
struct Write {
  uint32_t offset;
  std::vector<uint32_t> words;
  bool operator==(const Write &) const = default;
};

/// Whether a packet may write `words` words from `offset` in a tile of `kind`.
inline bool allowed(hw::Kind kind, uint32_t offset, size_t words, bool memtile_data) {
  const uint64_t end = offset + 4 * (uint64_t)words;
  auto in = [&](uint64_t from, uint64_t to) { return offset >= from && end <= to; };
  auto one_of = [&](const hw::Bank &b) { return words == 1 && b.holds(offset) && (offset - b.base) % b.step == 0; };
  auto queue = [&](hw::Dir d) {
    const hw::Bank c = hw::channels(kind, d);
    return one_of(hw::Bank{c.base + hw::kQueueFromCtrl, c.count, c.step});
  };
  if (offset % 4 || words == 0)
    return false;
  const hw::Bank bds = hw::bds(kind);
  switch (kind) {
  case hw::Kind::core:
    return in(hw::kPm, hw::kPm + hw::kPmBytes) || in(0, hw::kCoreDmBytes) || in(bds.base, bds.end()) ||
           (words == 1 && offset == hw::kCoreControl) || one_of(hw::locks(kind)) || queue(hw::Dir::s2mm) ||
           queue(hw::Dir::mm2s);
  case hw::Kind::mem:
    return in(bds.base, bds.end()) || (memtile_data && in(0, hw::kMemDmBytes)) || one_of(hw::locks(kind)) ||
           queue(hw::Dir::s2mm) || queue(hw::Dir::mm2s);
  case hw::Kind::shim: return false;
  }
  return false;
}

/// The packets of `writes` for packet id `id` into a tile of `kind`. Throws
/// std::invalid_argument for a write a packet may not do.
inline std::vector<uint32_t> encode(uint32_t id, hw::Kind kind, const std::vector<Write> &writes,
                                    bool memtile_data = false) {
  if (id > 0x1F)
    throw std::invalid_argument(std::format("packet id {} past 5 bits", id));
  std::vector<uint32_t> out;
  for (const Write &w : writes) {
    if (w.words.empty())
      throw std::invalid_argument(std::format("a control write of no words at {:#x}", w.offset));
    for (size_t at = 0; at < w.words.size();) {
      const uint32_t offset = w.offset + 4 * (uint32_t)at;
      const size_t beats = std::min({w.words.size() - at, kMaxBeats, (size_t)(4 - offset % 16 / 4)});
      if (offset > kOffsetBits || !allowed(kind, offset, beats, memtile_data))
        throw std::invalid_argument(std::format("a control packet may not write {} words at {:#x} of a {}", beats,
                                                offset, hw::kind_name(kind)));
      out.push_back(odd_parity(id));
      out.push_back(odd_parity((uint32_t)(beats - 1) << kBeatsShift | offset));
      out.insert(out.end(), w.words.begin() + (ptrdiff_t)at, w.words.begin() + (ptrdiff_t)(at + beats));
      at += beats;
    }
  }
  return out;
}

/// One packet read back.
struct Packet {
  uint32_t id, offset;
  std::vector<uint32_t> words;
  bool operator==(const Packet &) const = default;
};

/// Read `n` words of packets from `w` at `first`. Throws std::runtime_error for
/// a bad parity, a field that is not 0, a line crossing or a cut packet.
inline std::vector<Packet> decode(const uint32_t *w, size_t n) {
  std::vector<Packet> out;
  auto fail = [](size_t at, const std::string &m) {
    throw std::runtime_error(std::format("control packets: word {}: {}", at, m));
  };
  for (size_t i = 0; i < n;) {
    if (i + 2 > n)
      fail(i, "a packet cut after its stream header");
    const uint32_t h = w[i], c = w[i + 1];
    if (odd_parity(h) != h)
      fail(i, std::format("stream header {:#010x}: bad parity", h));
    if (h & 0x7FFFFFE0)
      fail(i, std::format("stream header {:#010x}: more than an id", h));
    if (odd_parity(c) != c)
      fail(i + 1, std::format("control header {:#010x}: bad parity", c));
    if (c & 0x7FC00000)
      fail(i + 1, std::format("control header {:#010x}: not a plain write", c));
    const uint32_t offset = c & kOffsetBits, beats = (c >> kBeatsShift & 3) + 1;
    if (offset % 4 || offset % 16 / 4 + beats > 4)
      fail(i + 1, std::format("{} beats at {:#x}: crosses a 16-byte line", beats, offset));
    if (i + 2 + beats > n)
      fail(i, "a packet cut in its data");
    out.push_back({h & 0x1F, offset, {w + i + 2, w + i + 2 + beats}});
    i += 2 + beats;
  }
  return out;
}
inline std::vector<Packet> decode(const std::vector<uint32_t> &w) { return decode(w.data(), w.size()); }

/// The 8 words of a shim BD that sends `words` words (no header of its own:
/// the packets carry theirs) and then BD `next` (-1: the chain ends). The
/// address (words 1 and 2) is the AddressPatch's; corelib `ShimBdWords`.
inline std::array<uint32_t, 8> shim_bd_words(uint32_t words, int next) {
  if (next > 15)
    throw std::invalid_argument(std::format("next BD {}: a shim has 16", next));
  return {words, 0, 0, 0, 3u << 30, 2u << 24, 0,
          1u << 25 | (next >= 0 ? (uint32_t)next << 27 | 1u << 26 : 0)};
}

} // namespace launch::ctrlpkt
