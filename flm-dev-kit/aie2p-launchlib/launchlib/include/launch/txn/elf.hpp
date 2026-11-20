//===- elf.hpp - one instruction stream as an ELF for XRT ---------*- C++ -*-===//
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
// The ELF flow of FLM npu_utils: the host makes xrt::elf -> xrt::module ->
// xrt::ext::kernel(hw_context, module, name) on the hw_context of an xclbin,
// from an ELF of the kernel's instruction stream. `to_elf` writes that ELF, as
// aiebu does for a TXN binary, without aiebu:
//
//   .ctrltext   the stream without its AddressPatch ops; in each BD that one
//               patched, the address bits are zero (word 1 but its low 2 bits,
//               the low 16 bits of word 2)
//   .rela.dyn   one relocation per AddressPatch: type 5 (shim_dma_48) at the
//               BD's word 0, symbol = the XRT argument (sequence argument + 3)
//               in decimal, addend = the patch's byte offset
//   .dynsym, .dynstr, .dynamic (only with relocations), .shstrtab
//
// At each run XRT adds the address of the argument's buffer plus the addend to
// the 48-bit address in words 1 and 2 of the BD (XRT elf_patcher.cpp,
// patch_shim48): what the command processor writes for an AddressPatch in the
// xclbin flow. The firmware op is gone, and the stream is 12 words shorter per
// patch.
//
// The BD of an AddressPatch must come from one BlockWrite of the stream (its
// words 0 to 2, as aiecc and launch::Rt write a shim BD), with no other write
// to those words in between. Anything else throws.
//
// aiebu's ELF differs only in what XRT does not need (tests/test_elf.cpp):
//   * it keeps the AddressPatch ops in .ctrltext (its relocation offsets are 48
//     bytes larger per op before them);
//   * it adds a DYNAMIC program header and a .note.xrt.UID section (XRT gives a
//     partial ELF without it an empty UUID; it shows it in error messages);
//   * its sections are at other offsets.
// The ELF layout follows ryzenai-corelib's PartialElf (core/npu/direct/AIE2PElf.cpp).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/txn/txn.hpp"

#include <cstring>
#include <format>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace launch::txn {

/// XRT patches the BD whose word 0 is at byte `byte` of .ctrltext with the
/// address of XRT argument `arg` plus `addend` bytes.
struct Relocation {
  uint32_t byte, arg, addend;
  bool operator==(const Relocation &) const = default;
};

/// A stream as the ELF holds it.
struct ElfText {
  std::vector<uint32_t> words; ///< the stream without its AddressPatch ops
  std::vector<Relocation> relocations;
};

/// The AddressPatch ops of `words` as relocations. Throws std::runtime_error
/// when a patch has no BD of its own (see the top of this file).
inline ElfText elf_text(const std::vector<uint32_t> &words) {
  const Stream s = parse(words);
  ElfText t;
  t.words = {words[0], words[1], 0, 0};
  std::vector<size_t> at(s.ops.size()); // the word of each op in t.words
  uint32_t ops = 0;
  for (size_t i = 0; i < s.ops.size(); ++i) {
    const auto *p = std::get_if<AddressPatch>(&s.ops[i]);
    if (!p) {
      at[i] = t.words.size();
      append(t.words, s.ops[i]);
      ++ops;
      continue;
    }
    auto fail = [&](const std::string &why) {
      throw std::runtime_error(std::format("ELF: the address patch of op {} (address {:#x}) {}", i, p->addr, why));
    };
    if (p->addr % 4 || p->addr < 4)
      fail("is not word 1 of a BD");
    // Words 0 to 2 of the BD: [lo, hi). The last op before the patch that
    // writes any of them must be one BlockWrite of all three.
    const uint32_t lo = p->addr - 4, hi = p->addr + 8;
    size_t bd = 0; // the word of BD word 0 in t.words
    for (size_t j = i; j-- > 0 && !bd;) {
      const Op &op = s.ops[j];
      if (const auto *b = std::get_if<BlockWrite>(&op)) {
        const uint32_t end = b->addr + 4 * (uint32_t)b->words.size();
        if (end <= lo || b->addr >= hi)
          continue;
        if (b->addr > lo || end < hi || (lo - b->addr) % 4)
          fail(std::format("has a BD that the block write of op {} writes only in part", j));
        bd = at[j] + 4 + (lo - b->addr) / 4;
      } else if (const auto *o = std::get_if<AddressPatch>(&op)) {
        if (o->addr == p->addr)
          fail(std::format("patches the BD of op {} again", j));
      } else if (std::holds_alternative<Write32>(op) || std::holds_alternative<MaskWrite32>(op)) {
        if (addr_of(op) >= lo && addr_of(op) < hi)
          fail(std::format("has a BD that op {} writes word by word", j));
      }
    }
    if (!bd)
      fail("has no BD written before it");
    t.words[bd + 1] &= 3u; // as aiebu; XRT adds the old address bits to the new address
    t.words[bd + 2] &= 0xFFFF0000u;
    t.relocations.push_back({(uint32_t)(4 * bd), p->arg + 3, p->offset});
  }
  t.words[2] = ops;
  t.words[3] = (uint32_t)(4 * t.words.size());
  return t;
}

namespace detail {

inline void put(std::vector<char> &out, size_t at, const void *data, size_t bytes) {
  if (out.size() < at + bytes)
    out.resize(at + bytes);
  std::memcpy(out.data() + at, data, bytes);
}
inline void put_words(std::vector<char> &out, size_t at, std::initializer_list<uint32_t> w) {
  put(out, at, w.begin(), 4 * w.size());
}
inline size_t align(size_t at, size_t to) { return (at + to - 1) / to * to; }

} // namespace detail

/// The ELF of a stream in ELF form (little-endian host).
inline std::vector<char> elf_of(const ElfText &t) {
  using detail::align, detail::put, detail::put_words;
  // XRT's module_elf_aie2p: OS/ABI 0x45, ABI version 2; its relocation types
  // are the patch schemes (5: shim_dma_48). XRT does not read the machine: 1
  // as aiebu (corelib writes 269).
  constexpr uint8_t kOsAbi = 0x45, kAbiVersion = 2;
  constexpr uint16_t kMachine = 1;
  constexpr uint32_t kShimDma48 = 5, kHeader = 52, kProgramHeader = 32, kSectionHeader = 40, kTextAt = 0x80;
  enum : uint32_t { kNull, kShstrtab, kText, kDynstr, kDynsym, kRela, kDynamic, kSections };
  static constexpr char kNames[] = "\0.shstrtab\0.ctrltext\0.dynstr\0.dynsym\0.rela.dyn\0.dynamic";
  constexpr uint32_t name_at[kSections] = {0, 1, 11, 21, 29, 37, 47};

  const size_t text = 4 * t.words.size();
  // One symbol per argument, in the order of first use; its name is the
  // argument in decimal.
  std::map<uint32_t, uint32_t> symbol, name; // argument -> symbol index, -> name in .dynstr
  std::string dynstr(1, '\0');
  for (const Relocation &r : t.relocations) {
    if (r.byte % 4 || r.byte + 12 > text)
      throw std::logic_error(std::format("ELF: a relocation at byte {} of a {} B stream", r.byte, text));
    if (symbol.try_emplace(r.arg, (uint32_t)symbol.size() + 1).second) {
      name[r.arg] = (uint32_t)dynstr.size();
      dynstr += std::to_string(r.arg) + '\0';
    }
  }

  std::vector<char> out;
  put(out, kTextAt, t.words.data(), text);
  size_t at = kTextAt + text;
  const size_t shstrtab_at = at;
  put(out, at, kNames, sizeof kNames);
  at += sizeof kNames;
  // Without relocations only .shstrtab and .ctrltext, as aiebu.
  const uint32_t count = t.relocations.empty() ? (uint32_t)kDynstr : (uint32_t)kSections;
  const size_t dynstr_at = at;
  size_t dynsym_at = 0, dynsym = 0, rela_at = 0, dynamic_at = 0;
  uint32_t rela = 0;
  if (count == kSections) {
    put(out, at, dynstr.data(), dynstr.size());
    at += dynstr.size();
    // .dynsym: 16 B entries, the first one null.
    dynsym_at = align(at, 8);
    dynsym = 16 * (symbol.size() + 1);
    out.resize(dynsym_at + dynsym);
    for (const auto &[arg, index] : symbol) {
      const size_t entry = dynsym_at + 16 * index;
      put_words(out, entry, {name[arg], 0, 0}); // name, value, size
      const uint8_t info = 0x11, other = 0;     // global object
      const uint16_t section = kText;
      put(out, entry + 12, &info, 1);
      put(out, entry + 13, &other, 1);
      put(out, entry + 14, &section, 2);
    }
    // .rela.dyn: 12 B entries (offset, symbol << 8 | type, addend).
    rela_at = align(dynsym_at + dynsym, 8);
    at = rela_at;
    for (const Relocation &r : t.relocations) {
      put_words(out, at, {r.byte, symbol.at(r.arg) << 8 | kShimDma48, r.addend});
      at += 12;
    }
    rela = (uint32_t)(at - rela_at);
    // .dynamic: DT_RELA (the section index, as aiebu writes it), DT_RELASZ.
    dynamic_at = align(at, 8);
    put_words(out, dynamic_at, {7, kRela, 8, rela});
    at = dynamic_at + 16;
  }

  const size_t sections_at = align(at, 4);
  struct Section {
    uint32_t type, flags, offset, size, link, info, align, entsize;
  };
  const Section sections[kSections] = {
      {0, 0, 0, 0, 0, 0, 0, 0},
      {3, 0, (uint32_t)shstrtab_at, sizeof kNames, 0, 0, 1, 0},
      {1, 6, kTextAt, (uint32_t)text, 0, 0, 16, 0},
      {3, 0, (uint32_t)dynstr_at, (uint32_t)dynstr.size(), 0, 0, 1, 0},
      {11, 0, (uint32_t)dynsym_at, (uint32_t)dynsym, kDynstr, 1, 8, 16},
      {4, 0, (uint32_t)rela_at, rela, kDynsym, 0, 8, 12},
      {6, 0, (uint32_t)dynamic_at, 16, kDynstr, 0, 8, 8},
  };
  for (uint32_t i = 0; i < count; ++i) {
    const Section &s = sections[i];
    put_words(out, sections_at + kSectionHeader * i,
              {name_at[i], s.type, s.flags, 0, s.offset, s.size, s.link, s.info, s.align, s.entsize});
  }

  // ELF header: ELF32, little-endian, executable; then the program headers
  // PHDR and LOAD (.ctrltext, read + execute).
  const uint8_t ident[16] = {0x7F, 'E', 'L', 'F', 1, 1, 1, kOsAbi, kAbiVersion};
  put(out, 0, ident, sizeof ident);
  const uint16_t type = 2, machine = kMachine;
  put(out, 16, &type, 2);
  put(out, 18, &machine, 2);
  put_words(out, 20, {1, 0, kHeader, (uint32_t)sections_at, 0}); // version, entry, phoff, shoff, flags
  const uint16_t halves[] = {kHeader, kProgramHeader, 2, kSectionHeader, (uint16_t)count, kShstrtab};
  put(out, 40, halves, sizeof halves);
  put_words(out, kHeader, {6, kHeader, 0, 0, 2 * kProgramHeader, 2 * kProgramHeader, 4, 0});
  put_words(out, kHeader + kProgramHeader, {1, kTextAt, 0, 0, (uint32_t)text, (uint32_t)text, 5, 16});
  return out;
}

/// The ELF of an instruction stream, for xrt::elf.
inline std::vector<char> to_elf(const std::vector<uint32_t> &words) { return elf_of(elf_text(words)); }

} // namespace launch::txn
