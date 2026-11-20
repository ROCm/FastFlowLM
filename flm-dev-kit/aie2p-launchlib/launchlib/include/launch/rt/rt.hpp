//===- rt.hpp - the runtime sequence of an app, in C++ ------------*- C++ -*-===//
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
// `Rt` has the calls of the IRON runtime_sequence (aiex.py) with the same names.
// The host program writes one function per app (`SequenceFn`) and gives them
// to `build()` in a `SequenceMap`. The library has no runtime sequence.
//
//   Rt call                     TXN ops (as mlir-aie AIEDmaToNpu.cpp lowers them)
//   npu_dma_memcpy_nd(m)        BlockWrite of the 8-word shim BD, AddressPatch of
//                               its address word (kernel argument + offset), then
//                               the push (unless m.push is false)
//   npu_push_bd(ch, bd, token)  [MaskWrite32 of the shim's controller id into the
//                               channel Ctrl, if token] Write32 of the Start_Queue
//   npu_dma_wait(ch)            Sync on the channel (its task-complete token)
//   npu_write32, npu_maskwrite32  Write32, MaskWrite32 (array addresses)
//   npu_rtp_write(buf, i, v)    Write32 to word i of a named buffer (manifest)
//   set_lock_value(lock, v)     Write32 to a named lock (manifest)
//
// A shim channel is given as ShimChannel{col, dir, channel} or by the name of
// its IRON object FIFO: `rt.alloc("lhs0")` (the alloc lines of the manifest).
//
// The encoding is checked against aiecc: tests/test_rt.cpp builds the sequence
// of an app and compares it with the package's insts.bin word for word.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/formats/package.hpp"
#include "launch/txn/txn.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace launch {

/// A kernel argument of the sequence (0 = the first buffer of the
/// runtime_sequence) and the bytes of one of its elements. No default: the
/// element size sets every length, stride and offset.
struct Arg {
  unsigned index;
  unsigned elem_bytes;
};

/// One npu_dma_memcpy_nd. Offsets, sizes and strides are in elements, the
/// outermost dimension first (as aiex.py).
struct Memcpy {
  ShimChannel metadata;
  uint32_t bd_id;
  Arg mem;
  std::array<int64_t, 4> offsets{0, 0, 0, 0};
  std::array<int64_t, 4> sizes{1, 1, 1, 1};
  std::array<int64_t, 4> strides{0, 0, 0, 1};
  bool issue_token = false;
  uint32_t burst_length = 0; ///< bytes; 0 = the largest (512)
  bool push = true;          ///< false: write and patch the BD only (npu_push_bd later)
};

class Rt;
using SequenceFn = std::function<void(Rt &)>;
using SequenceMap = std::map<std::string, SequenceFn>;

namespace detail {

/// The 8 words of a shim BD (mlir-aie AIEDmaToNpu.cpp DmaToNpuPattern,
/// AIEXDialect.cpp getHardwareStridesWraps; npu2 burst encodings).
struct ShimBd {
  std::array<uint32_t, 8> words{};
  uint32_t repeat = 0;
  uint32_t offset_bytes = 0;
};

inline uint32_t burst_encoding(uint32_t bytes) {
  switch (bytes) {
  case 0:
  case 512: return 3;
  case 64: return 0;
  case 128: return 1;
  case 256: return 2;
  }
  throw std::invalid_argument(std::format("burst length {} B: npu2 has 64, 128, 256, 512", bytes));
}

inline ShimBd shim_bd(const Memcpy &m) {
  const int64_t bits = 8 * (int64_t)m.mem.elem_bytes, gran = 32;
  if (m.mem.elem_bytes == 0)
    throw std::invalid_argument("Arg: elem_bytes must not be 0");
  // Innermost dimension first, as mlir-aie.
  std::array<int64_t, 4> in_sizes, in_strides, offsets;
  for (int i = 0; i < 4; ++i) {
    in_sizes[i] = m.sizes[3 - i];
    in_strides[i] = m.strides[3 - i];
    offsets[i] = m.offsets[3 - i];
  }
  for (int i = 0; i < 4; ++i)
    if (in_sizes[i] <= 0)
      throw std::invalid_argument(std::format("size {} must be positive", 3 - i));
  if (in_sizes[0] * bits % gran)
    throw std::invalid_argument("the innermost size must be a whole number of 32-bit words");

  std::array<int64_t, 4> hw_sizes{}, hw_strides{};
  hw_sizes[0] = in_sizes[0] * bits / gran;
  hw_strides[0] = (in_strides[0] * bits < gran || bits > gran) ? 0 : in_strides[0] * bits / gran - 1;
  hw_sizes[1] = in_sizes[1];
  if (in_sizes[1] > 1)
    hw_strides[1] = in_strides[1] * bits / gran - 1;
  hw_sizes[2] = in_sizes[2];
  if (in_sizes[2] > 1)
    hw_strides[2] = in_strides[2] * bits / gran - 1;
  if (in_sizes[3] > 1) {
    hw_sizes[3] = in_sizes[3] - 1;
    if (in_strides[3] > 0)
      hw_strides[3] = in_strides[3] * bits / gran - 1;
  }

  const bool linear = in_sizes[1] == 1 && in_sizes[2] == 1 && in_strides[0] == 1 && in_strides[1] == 0 &&
                      in_strides[2] == 0;
  uint32_t d0_size = 0, d0_stride = 0, d1_size = 0, d1_stride = 0, d2_stride = 0;
  if (!linear) {
    d0_size = (uint32_t)hw_sizes[0];
    d0_stride = (uint32_t)hw_strides[0];
    d1_size = (uint32_t)hw_sizes[1];
    d1_stride = (uint32_t)hw_strides[1];
    d2_stride = (uint32_t)hw_strides[2];
  }
  uint32_t iter_size = 0, iter_stride = 0;
  if (in_sizes[3] > 1 && in_strides[3] > 0) {
    iter_size = (uint32_t)hw_sizes[3];
    iter_stride = (uint32_t)hw_strides[3];
  }
  // The BD fields are masked below: a value that does not fit would wrap.
  // (As mlir-aie, the outer size is both the iteration size and the repeat count.)
  auto fits = [](int64_t v, unsigned field_bits, const char *what) {
    if (v < 0 || v >= (int64_t)1 << field_bits)
      throw std::invalid_argument(std::format("{} {} does not fit the shim BD's {}-bit field", what, v, field_bits));
  };
  fits(d0_size, 10, "dimension 0 size");
  fits(d1_size, 10, "dimension 1 size");
  fits(d0_stride, 20, "dimension 0 stride");
  fits(d1_stride, 20, "dimension 1 stride");
  fits(d2_stride, 20, "dimension 2 stride");
  fits(iter_size, 6, "iteration size (outer size - 1)");
  fits(iter_stride, 20, "iteration stride");
  fits(hw_sizes[3], 8, "repeat count (outer size - 1)");

  ShimBd bd;
  uint64_t length = (uint64_t)(in_sizes[0] * bits / gran);
  for (int i = 1; i < 3; ++i)
    length *= (uint64_t)in_sizes[i];
  if (length > 0xFFFFFFFFull)
    throw std::invalid_argument(std::format("transfer of {} words does not fit the BD's length", length));
  int64_t off = 0;
  for (int i = 0; i < 4; ++i)
    off += offsets[i] * in_strides[i] * (int64_t)m.mem.elem_bytes;
  bd.offset_bytes = (uint32_t)off;
  bd.words[0] = (uint32_t)length;
  // The address: the AddressPatch writes buffer + offset at dispatch. The
  // offset is here too, as aiecc writes it for IRON's task API
  // (shim_dma_single_bd_task); its npu_dma_memcpy_nd path writes 0. The patch
  // overwrites it either way.
  bd.words[1] = bd.offset_bytes;
  bd.words[2] = 0; // no packet header
  bd.words[3] = (d0_size & 0x3ff) << 20 | (d0_stride & 0xfffff);
  bd.words[4] = burst_encoding(m.burst_length) << 30 | (d1_size & 0x3ff) << 20 | (d1_stride & 0xfffff);
  bd.words[5] = 2u << 24 | (d2_stride & 0xfffff); // AxCache = 2: upsizing in the NoC
  bd.words[6] = (iter_size & 0x3f) << 20 | (iter_stride & 0xfffff);
  bd.words[7] = 1u << 25; // valid BD
  bd.repeat = (uint32_t)hw_sizes[3];
  return bd;
}

} // namespace detail

/// The runtime sequence of one kernel: appends TXN ops to `out`.
class Rt {
public:
  Rt(std::vector<txn::Op> &out, const Package &app) : out_(out), app_(app) {}

  ShimChannel alloc(const std::string &name) const { return app_.alloc(name); }

  void npu_dma_memcpy_nd(const Memcpy &m) {
    check_shim(m.metadata);
    if (m.bd_id >= hw::bds(hw::Kind::shim).count)
      throw std::invalid_argument(std::format("BD id {}: a shim has 16 BDs (0..15)", m.bd_id));
    const detail::ShimBd bd = detail::shim_bd(m);
    const uint32_t bd_addr = hw::address(m.metadata.col, 0, hw::bds(hw::Kind::shim).at(m.bd_id));
    out_.push_back(txn::BlockWrite{bd_addr, {bd.words.begin(), bd.words.end()}});
    out_.push_back(txn::AddressPatch{bd_addr + 4, m.mem.index, bd.offset_bytes});
    args_.insert(m.mem.index);
    repeat_[{m.metadata, m.bd_id}] = bd.repeat;
    if (m.push)
      npu_push_bd(m.metadata, m.bd_id, m.issue_token);
  }

  /// Start a BD written by npu_dma_memcpy_nd (its repeat count comes with it).
  void npu_push_bd(ShimChannel ch, uint32_t bd_id, bool issue_token) {
    check_shim(ch);
    const uint32_t ctrl = hw::address(ch.col, 0, hw::channels(hw::Kind::shim, ch.dir).at((uint32_t)ch.channel));
    if (issue_token) {
      // The token goes to the controller id in the channel's Ctrl; without one
      // it would go to whatever id the Ctrl holds, not to the one a sync waits on.
      auto it = app_.controller_id.find(Tile{ch.col, 0});
      if (it == app_.controller_id.end())
        throw std::invalid_argument(std::format(
            "issue_token on shim ({}, 0): the package has no controller id for that tile", ch.col));
      out_.push_back(txn::MaskWrite32{ctrl, it->second << hw::kTokenControllerShift, hw::kTokenControllerMask});
    }
    auto r = repeat_.find({ch, bd_id});
    const uint32_t repeat = r == repeat_.end() ? 0 : r->second;
    out_.push_back(txn::Write32{ctrl + hw::kQueueFromCtrl,
                                (bd_id & 0xF) | (repeat & 0xFF) << 16 | (issue_token ? hw::kQueueToken : 0)});
  }

  void npu_dma_wait(ShimChannel ch) {
    check_shim(ch);
    out_.push_back(txn::Sync{(uint32_t)ch.col, 0, (uint32_t)ch.dir, (uint32_t)ch.channel});
  }

  /// A point where the firmware may preempt this context (FLM puts one, level 0,
  /// at the start of each sequence on Windows).
  void npu_preempt(uint32_t level) { out_.push_back(txn::Preempt{level}); }

  void npu_write32(uint32_t addr, uint32_t value) { out_.push_back(txn::Write32{addr, value}); }
  void npu_maskwrite32(uint32_t addr, uint32_t value, uint32_t mask) {
    out_.push_back(txn::MaskWrite32{addr, value, mask});
  }

  /// Word `index` of a named buffer (an RTP), as aiex npu_rtp_write.
  void npu_rtp_write(const std::string &buffer, uint32_t index, uint32_t value) {
    const Buffer &b = app_.buffer(buffer);
    if (4 * index >= b.bytes)
      throw std::invalid_argument(std::format("npu_rtp_write: word {} is past the end of '{}' ({} B)", index,
                                              buffer, b.bytes));
    npu_write32(b.tile.addr(b.address + 4 * index), value);
  }

  void set_lock_value(const std::string &lock, uint32_t value) {
    const Lock &k = app_.lock(lock);
    npu_write32(k.tile.addr(hw::locks(k.tile.kind()).at(k.id)), value);
  }

  /// The kernel arguments the sequence uses.
  const std::set<unsigned> &args() const { return args_; }

private:
  static void check_shim(ShimChannel ch) {
    if (ch.col < 0 || ch.col >= hw::kCols || ch.channel < 0 ||
        (uint32_t)ch.channel >= hw::channels(hw::Kind::shim, ch.dir).count)
      throw std::invalid_argument(std::format("shim channel (col {}, {}, {}) is not in the array", ch.col,
                                              ch.dir == hw::Dir::mm2s ? "mm2s" : "s2mm", ch.channel));
  }

  std::vector<txn::Op> &out_;
  const Package &app_;
  std::set<unsigned> args_;
  std::map<std::pair<ShimChannel, uint32_t>, uint32_t> repeat_;
};

} // namespace launch
