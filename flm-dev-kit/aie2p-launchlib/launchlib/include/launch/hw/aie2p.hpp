//===- aie2p.hpp - the AIE2P (npu2) facts the library uses -------*- C++ -*-===//
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
// The array and its registers, as tile offsets. Sources (aie-rt
// driver/src/global/xaie2pgbl_params.h unless noted):
//
//   address       col << 25 | row << 20 | offset          mlir-aie AIE2TargetModel
//   rows          0 shim (NOC), 1 mem tile, 2..5 cores     npu2: 8 columns
//   core PM       0x20000, 16 KB                          XAIE2PGBL_CORE_MODULE_PROGRAM_MEMORY
//   core DM       0x0, 64 KB; mem tile DM 0x0, 512 KB
//   Core_Control  0x32000: bit 0 enable, bit 1 reset      XAIE2PGBL_CORE_MODULE_CORE_CONTROL
//   locks         core 0x1F000 (16), mem 0xC0000 (64),    *_LOCK0_VALUE, 0x10 apart
//                 shim 0x14000 (16); values 0..63
//   DMA BDs       core 0x1D000 (16), mem 0xA0000 (48),    *_DMA_BD0_0, 0x20 apart
//                 shim 0x1D000 (16)
//   DMA channels  Ctrl, then Start_Queue 4 bytes later:   *_DMA_{S2MM,MM2S}_0_CTRL
//                 core S2MM 0x1DE00, MM2S 0x1DE10 (2 each)
//                 mem  S2MM 0xA0600, MM2S 0xA0630 (6 each)
//                 shim S2MM 0x1D200, MM2S 0x1D210 (2 each)
//                 Ctrl bit 1 resets the channel; bits 12:8 the controller id of
//                 its task-complete token (mlir-aie AIEDmaToNpu.cpp)
//   shim mux      0x1F000, demux 0x1F004                  XAIE2PGBL_NOC_MODULE_{MUX,DEMUX}_CONFIG
//   switches      hw_gen.hpp (tools/gen_hw.py)
//
// `classify(address)` gives the class of a register write. The package loader
// sorts the init writes of an app by class (formats/package.hpp); a write
// without a class stops the build.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/hw/hw_gen.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>

namespace launch::hw {

inline constexpr int kCols = 8;
inline constexpr int kRows = 6;     ///< shim, mem tile, 4 core rows
inline constexpr int kMemRows = 1;
inline constexpr int kFirstCoreRow = 2;
inline constexpr unsigned kColShift = 25, kRowShift = 20;
inline constexpr uint32_t kOffsetMask = 0xFFFFF;

/// The TXN header of this device (mlir-aie AIETranslateNpuToBinary): version
/// 0.1, device generation 4 (NPU2: Strix, Krackan).
inline constexpr uint8_t kTxnDevGen = 4;

enum class Kind { shim, mem, core };

inline constexpr Kind kind_of_row(int row) {
  return row == 0 ? Kind::shim : row <= kMemRows ? Kind::mem : Kind::core;
}
inline const char *kind_name(Kind k) {
  return k == Kind::shim ? "shim" : k == Kind::mem ? "mem tile" : "core";
}

inline constexpr uint32_t address(int col, int row, uint32_t offset) {
  return ((uint32_t)col << kColShift) | ((uint32_t)row << kRowShift) | (offset & kOffsetMask);
}

// --- tile offsets -------------------------------------------------------------

inline constexpr uint32_t kPm = 0x20000, kPmBytes = 0x4000;
inline constexpr uint32_t kCoreDmBytes = 0x10000, kMemDmBytes = 0x80000;
inline constexpr uint32_t kCoreControl = 0x32000;
inline constexpr uint32_t kCoreEnable = 0x1, kCoreReset = 0x2;
inline constexpr uint32_t kShimMux = 0x1F000, kShimDemux = 0x1F004;

struct Bank {
  uint32_t base;  ///< offset of entry 0
  uint32_t count; ///< entries
  uint32_t step;  ///< bytes between entries
  constexpr uint32_t at(uint32_t i) const { return base + i * step; }
  constexpr uint32_t end() const { return base + count * step; }
  constexpr bool holds(uint32_t off) const { return off >= base && off < end(); }
};

inline constexpr Bank locks(Kind k) {
  return k == Kind::core ? Bank{0x1F000, 16, 0x10} : k == Kind::mem ? Bank{0xC0000, 64, 0x10} : Bank{0x14000, 16, 0x10};
}
inline constexpr uint32_t kLockMax = 63; ///< a lock value has 6 bits
inline constexpr Bank bds(Kind k) {
  return k == Kind::core ? Bank{0x1D000, 16, 0x20} : k == Kind::mem ? Bank{0xA0000, 48, 0x20} : Bank{0x1D000, 16, 0x20};
}

enum class Dir { s2mm = 0, mm2s = 1 }; ///< the npu.sync encoding of mlir-aie

/// The channels of one direction: their Ctrl registers, 8 bytes apart.
inline constexpr Bank channels(Kind k, Dir d) {
  const bool in = d == Dir::s2mm;
  if (k == Kind::core)
    return in ? Bank{0x1DE00, 2, 8} : Bank{0x1DE10, 2, 8};
  if (k == Kind::mem)
    return in ? Bank{0xA0600, 6, 8} : Bank{0xA0630, 6, 8};
  return in ? Bank{0x1D200, 2, 8} : Bank{0x1D210, 2, 8};
}
inline constexpr uint32_t kQueueFromCtrl = 4;
inline constexpr uint32_t kDmaReset = 0x2;          ///< Ctrl bit 1
inline constexpr uint32_t kTokenControllerMask = 0x1F00, kTokenControllerShift = 8;
inline constexpr uint32_t kQueueToken = 0x80000000; ///< Start_Queue bit 31

template <size_t N> inline bool in_list(const std::array<uint32_t, N> &a, uint32_t off) {
  return std::binary_search(a.begin(), a.end(), off);
}
inline bool is_switch(Kind k, uint32_t off) {
  return k == Kind::core ? in_list(kSwitchRegs_core, off)
         : k == Kind::mem ? in_list(kSwitchRegs_mem, off)
                          : in_list(kSwitchRegs_shim, off);
}

// --- classes of a register write ---------------------------------------------

/// What an init write sets. The order of the enum is not the write order; the
/// pipeline gives the order (kernel/pipeline.hpp).
enum class Cls {
  program,      ///< core PM
  data,         ///< core or mem-tile data memory (initial values of buffers)
  lock,         ///< lock values
  bd,           ///< DMA buffer descriptors
  channel,      ///< DMA channel Ctrl and Start_Queue (chain starts, resets)
  route,        ///< stream switch
  mux,          ///< shim mux / demux
  core_control, ///< Core_Control (enable, reset)
};
inline const char *cls_name(Cls c) {
  switch (c) {
  case Cls::program: return "program";
  case Cls::data: return "data";
  case Cls::lock: return "lock";
  case Cls::bd: return "bd";
  case Cls::channel: return "channel";
  case Cls::route: return "route";
  case Cls::mux: return "mux";
  case Cls::core_control: return "core_control";
  }
  return "?";
}

struct Where {
  int col, row;
  Kind kind;
  uint32_t offset;
};

inline Where locate(uint32_t addr) {
  const int col = (int)(addr >> kColShift), row = (int)((addr >> kRowShift) & 0x1F);
  if (col >= kCols || row >= kRows)
    throw std::out_of_range(std::format("address {:#010x}: tile ({}, {}) is not in the array", addr, col, row));
  return {col, row, kind_of_row(row), addr & kOffsetMask};
}

/// The class of the register (or memory word) at `addr`. Throws for an
/// address the library does not know.
inline Cls classify(uint32_t addr) {
  const Where w = locate(addr);
  const uint32_t off = w.offset;
  if (is_switch(w.kind, off))
    return Cls::route;
  if (locks(w.kind).holds(off))
    return Cls::lock;
  if (bds(w.kind).holds(off))
    return Cls::bd;
  for (Dir d : {Dir::s2mm, Dir::mm2s}) {
    const Bank b = channels(w.kind, d);
    // Ctrl at base + 8k, Start_Queue 4 bytes after it.
    if (off >= b.base && off < b.end() && ((off - b.base) % b.step == 0 || (off - b.base) % b.step == kQueueFromCtrl))
      return Cls::channel;
  }
  switch (w.kind) {
  case Kind::core:
    if (off >= kPm && off < kPm + kPmBytes)
      return Cls::program;
    if (off < kCoreDmBytes)
      return Cls::data;
    if (off == kCoreControl)
      return Cls::core_control;
    break;
  case Kind::mem:
    if (off < kMemDmBytes)
      return Cls::data;
    break;
  case Kind::shim:
    if (off == kShimMux || off == kShimDemux)
      return Cls::mux;
    break;
  }
  throw std::invalid_argument(std::format("address {:#010x} ({} ({}, {}) offset {:#07x}): no register class",
                                          addr, kind_name(w.kind), w.col, w.row, off));
}

} // namespace launch::hw
