//===- test_elf.cpp - txn::to_elf against aiebu -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
//   test_elf [PKG APP...] [FILE...]
//
// Without a package: the ELF form of small streams (patched BDs, relocations)
// and the patches it refuses. With one: for the package's binaries (init.txn,
// insts.bin), the kernels of build() (reset, cold and rerun of each app, warm
// between the first two, with a sequence of patched shim BDs) and each FILE (a
// TXN binary), to_elf's ELF and aiebu's (to_elf_aiebu) hold the same:
//   .ctrltext    aiebu's without its AddressPatch ops, byte for byte
//   relocations  the same (offset, symbol name, type, addend); aiebu's offsets
//                less 48 B per AddressPatch op before them
//   .dynamic     the same entries
//   ELF header   the same class, OS/ABI, ABI version, type and machine
// Needs aiebu (libaiebu.a of XRT; aiebu_static.lib on Windows), no NPU.
//
//===----------------------------------------------------------------------===//

#include "check.hpp"

#include "launch/launch.hpp"
#include "launch/txn/elf.hpp"
#include "launch/txn/elf_aiebu.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <set>
#ifdef _WIN32
#include <io.h> // dup, dup2, close
#define NULL_DEVICE "NUL"
#else
#include <unistd.h>
#define NULL_DEVICE "/dev/null"
#endif

using namespace launch;

/// The sections of an ELF32 (little-endian) by name.
struct Elf {
  struct Section {
    uint32_t type, offset, size;
  };
  std::vector<char> bytes;
  std::map<std::string, Section> sections;

  explicit Elf(std::vector<char> b) : bytes(std::move(b)) {
    const uint32_t shoff = u32(32), shstrndx = u16(50);
    const uint32_t names = u32(shoff + 40 * shstrndx + 16);
    for (uint32_t i = 0; i < u16(48); ++i) {
      const size_t h = shoff + 40 * i;
      sections[str(names + u32(h))] = Section{u32(h + 4), u32(h + 16), u32(h + 20)};
    }
  }
  uint32_t u32(size_t at) const {
    uint32_t v;
    std::memcpy(&v, bytes.data() + at, 4);
    return v;
  }
  uint16_t u16(size_t at) const {
    uint16_t v;
    std::memcpy(&v, bytes.data() + at, 2);
    return v;
  }
  std::string str(size_t at) const { return bytes.data() + at; }
  const Section &at(const std::string &name) const {
    auto it = sections.find(name);
    if (it == sections.end())
      throw std::runtime_error("no section " + name);
    return it->second;
  }
  std::vector<uint32_t> words(const std::string &name) const {
    const Section &s = at(name);
    return txn::words_of({bytes.begin() + s.offset, bytes.begin() + s.offset + s.size});
  }

  struct Rel {
    uint32_t offset;
    std::string symbol;
    uint32_t type, addend;
    bool operator==(const Rel &) const = default;
  };
  bool has(const std::string &name) const { return sections.count(name) > 0; }
  std::vector<Rel> relocations() const {
    if (!has(".rela.dyn"))
      return {};
    const Section &rela = at(".rela.dyn"), &sym = at(".dynsym"), &str_ = at(".dynstr");
    std::vector<Rel> r;
    for (uint32_t i = 0; i < rela.size; i += 12) {
      const uint32_t info = u32(rela.offset + i + 4);
      r.push_back({u32(rela.offset + i), str(str_.offset + u32(sym.offset + 16 * (info >> 8))), info & 0xff,
                   u32(rela.offset + i + 8)});
    }
    return r;
  }
  /// The .dynamic entries before DT_NULL.
  std::vector<uint32_t> dynamic() const {
    if (!has(".dynamic"))
      return {};
    std::vector<uint32_t> d = words(".dynamic");
    for (size_t i = 0; i + 1 < d.size(); i += 2)
      if (d[i] == 0)
        d.resize(i);
    return d;
  }
};

/// aiebu prints a summary of every stream on stdout.
static std::vector<char> aiebu_quiet(const std::vector<uint32_t> &words) {
  std::fflush(stdout);
  const int saved = dup(1);
  FILE *null = std::fopen(NULL_DEVICE, "w");
  dup2(fileno(null), 1);
  std::vector<char> elf;
  try {
    elf = txn::to_elf_aiebu(words);
  } catch (...) {
    dup2(saved, 1);
    throw;
  }
  std::fflush(stdout);
  dup2(saved, 1);
  close(saved);
  std::fclose(null);
  return elf;
}

static double us_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

/// One stream: our ELF against aiebu's (see the top).
static void compare(const std::string &what, const std::vector<uint32_t> &words) {
  auto t0 = std::chrono::steady_clock::now();
  const Elf ours(txn::to_elf(words));
  const double us_ours = us_since(t0);
  t0 = std::chrono::steady_clock::now();
  const Elf theirs(aiebu_quiet(words));
  const double us_aiebu = us_since(t0);

  // aiebu's .ctrltext without its patch ops; the byte of each patch op.
  const txn::Stream s = txn::parse(theirs.words(".ctrltext"));
  txn::Stream no_patch{s.header, {}};
  std::vector<uint32_t> patch_at;
  size_t w = 4;
  for (const txn::Op &op : s.ops) {
    if (std::holds_alternative<txn::AddressPatch>(op))
      patch_at.push_back((uint32_t)(4 * w));
    else
      no_patch.ops.push_back(op);
    w += txn::op_words(op);
  }
  const std::vector<uint32_t> text = ours.words(".ctrltext");
  const bool text_ok = text == txn::encode(no_patch);
  CHECK(text_ok);

  std::vector<Elf::Rel> want = theirs.relocations();
  for (Elf::Rel &r : want)
    r.offset -= 48 * (uint32_t)std::count_if(patch_at.begin(), patch_at.end(), [&](uint32_t p) { return p < r.offset; });
  const std::vector<Elf::Rel> got = ours.relocations();
  const bool rel_ok = got == want && got.size() == patch_at.size();
  CHECK(rel_ok);
  CHECK(ours.dynamic() == theirs.dynamic());
  // The ELF header but where the sections are and how many.
  CHECK(std::memcmp(ours.bytes.data(), theirs.bytes.data(), 32) == 0); // ident ... phoff
  CHECK(ours.u32(36) == theirs.u32(36));                               // flags
  CHECK(std::memcmp(ours.bytes.data() + 40, theirs.bytes.data() + 40, 4) == 0 && ours.u16(46) == theirs.u16(46) &&
        ours.u16(50) == theirs.u16(50));
  for (const char *name : {".dynsym", ".dynstr", ".rela.dyn", ".dynamic"})
    CHECK(ours.has(name) == theirs.has(name));
  // Every BD word that differs between the input and aiebu's text is a word
  // that a relocation patches: aiebu changed nothing else.
  const std::vector<uint32_t> aiebu_text = theirs.words(".ctrltext");
  std::set<uint32_t> patched;
  for (const Elf::Rel &r : theirs.relocations())
    patched.insert({r.offset / 4 + 1, r.offset / 4 + 2});
  size_t moved = 0;
  CHECK(aiebu_text.size() == words.size());
  for (size_t i = 0; i < words.size() && i < aiebu_text.size(); ++i)
    if (words[i] != aiebu_text[i]) {
      ++moved;
      CHECK(patched.count((uint32_t)i));
    }
  std::printf("  %-34s %6zu ops %8zu B -> %8zu B, %3zu relocations (%zu BD words cleared); to_elf %7.1f us, aiebu %8.1f "
              "us; %s\n",
              what.c_str(), s.ops.size(), 4 * words.size(), 4 * text.size(), got.size(), moved, us_ours, us_aiebu,
              text_ok && rel_ok ? "same" : "DIFFERS");
}

/// A shim BD (8 words) at shim (col, 0), BD `id`.
static uint32_t shim_bd(int col, uint32_t id) { return hw::address(col, 0, hw::bds(hw::Kind::shim).at(id)); }

int main(int argc, char **argv) {
  // A patched BD: its address bits go to zero (word 1 keeps its low 2 bits), the
  // patch to a relocation.
  const uint32_t bd = shim_bd(1, 3);
  txn::Stream s;
  s.ops = {txn::Write32{hw::address(0, 2, 0x1F010), 2},
           txn::BlockWrite{bd, {0x400, 0x43, 0xABCD0012, 3, 4, 5, 6, 7}},
           txn::AddressPatch{bd + 4, 1, 0x40},
           txn::Sync{1, 0, 0, 0}};
  const txn::ElfText t = txn::elf_text(txn::encode(s));
  txn::Stream want{s.header, {s.ops[0], txn::BlockWrite{bd, {0x400, 3, 0xABCD0000, 3, 4, 5, 6, 7}}, s.ops[3]}};
  CHECK(t.words == txn::encode(want));
  CHECK((t.relocations == std::vector<txn::Relocation>{{4 * (4 + 6 + 4), 4, 0x40}}));
  const Elf e(txn::elf_of(t));
  CHECK(e.words(".ctrltext") == t.words);
  CHECK((e.relocations() == std::vector<Elf::Rel>{{4 * (4 + 6 + 4), "4", 5, 0x40}}));
  CHECK((e.dynamic() == std::vector<uint32_t>{7, 5, 8, 12}));

  // Two BDs of one block write, patched against two arguments (one of them twice).
  const uint32_t bd0 = shim_bd(0, 0);
  std::vector<uint32_t> two(16, 9);
  txn::Stream s2;
  s2.ops = {txn::BlockWrite{bd0, two}, txn::AddressPatch{bd0 + 4, 0, 0}, txn::AddressPatch{bd0 + 36, 2, 8},
            txn::BlockWrite{shim_bd(0, 2), std::vector<uint32_t>(8, 1)}, txn::AddressPatch{shim_bd(0, 2) + 4, 0, 4}};
  const txn::ElfText t2 = txn::elf_text(txn::encode(s2));
  CHECK((t2.relocations == std::vector<txn::Relocation>{{32, 3, 0}, {64, 5, 8}, {4 * (4 + 20 + 4), 3, 4}}));
  CHECK(t2.words[2] == 2 && t2.words[3] == 4 * t2.words.size());
  CHECK(t2.words[9] == 1 && t2.words[10] == 0 && t2.words[11] == 9 && t2.words[17] == 1 && t2.words[18] == 0);
  const Elf e2(txn::elf_of(t2));
  CHECK(e2.relocations().size() == 3 && e2.relocations()[2].symbol == "3");

  // What it refuses.
  auto refuses = [](std::vector<txn::Op> ops) {
    txn::Stream x;
    x.ops = std::move(ops);
    return throws([&] { txn::elf_text(txn::encode(x)); });
  };
  CHECK(refuses({txn::AddressPatch{bd + 4, 0, 0}}));                                       // no BD
  CHECK(refuses({txn::BlockWrite{bd, {0, 0}}, txn::AddressPatch{bd + 4, 0, 0}}));           // words 0, 1 only
  CHECK(refuses({txn::BlockWrite{bd + 4, {0, 0, 0}}, txn::AddressPatch{bd + 4, 0, 0}}));    // no word 0
  CHECK(refuses({txn::Write32{bd, 0}, txn::Write32{bd + 4, 0}, txn::Write32{bd + 8, 0},
                 txn::AddressPatch{bd + 4, 0, 0}}));                                        // word by word
  CHECK(refuses({txn::BlockWrite{bd, std::vector<uint32_t>(8, 0)}, txn::Write32{bd + 8, 0},
                 txn::AddressPatch{bd + 4, 0, 0}}));                                        // a write in between
  CHECK(refuses({txn::BlockWrite{bd, std::vector<uint32_t>(8, 0)}, txn::AddressPatch{bd + 4, 0, 0},
                 txn::AddressPatch{bd + 4, 1, 0}}));                                        // patched twice
  CHECK(!refuses({txn::BlockWrite{bd, std::vector<uint32_t>(8, 0)}, txn::Write32{bd + 12, 0},
                  txn::AddressPatch{bd + 4, 0, 0}}));                                       // word 3: fine
  // Against aiebu.
  compare("unit: one BD", txn::encode(s));
  compare("unit: three BDs", txn::encode(s2));

  std::vector<std::string> apps;
  for (int i = 1; i < argc; ++i)
    if (std::filesystem::is_regular_file(argv[i]))
      compare(std::filesystem::path(argv[i]).filename().string(), txn::words_of(read_bytes(argv[i])));
    else
      apps.push_back(argv[i]);
  if (apps.size() < 2)
    return done("test_elf");
  const std::string pkg = apps[0];
  apps.erase(apps.begin());
  for (const std::string &app : apps)
    for (const char *f : {"init.txn", "insts.bin"})
      if (std::filesystem::exists(pkg + "/" + app + "/" + f))
        compare(app + "/" + f, txn::words_of(read_bytes(pkg + "/" + app + "/" + f)));

  // Kernels of build(): a sequence of 12 patched shim BDs (4 columns, both
  // directions), three arguments, offsets.
  SequenceMap seqs;
  for (const std::string &app : apps)
    seqs[app] = [](Rt &rt) {
      for (int col = 0; col < 4; ++col)
        for (uint32_t k = 0; k < 3; ++k)
          rt.npu_dma_memcpy_nd({.metadata = {col, k == 2 ? hw::Dir::s2mm : hw::Dir::mm2s, (int)(k & 1)},
                                .bd_id = k,
                                .mem = Arg{k, 4},
                                .offsets = {0, 0, 0, 256 * col},
                                .sizes = {1, 1, 1, 256},
                                .strides = {0, 0, 0, 1}});
    };
  ConfigList list{pkg, Transport::txn, {reset()}};
  for (const std::string &app : apps)
    list.configs.push_back(cold(app));
  if (apps.size() > 1) {
    list.configs.push_back(warm(apps[0], apps[1]));
    list.configs.push_back(warm(apps[1], apps[0]));
  }
  for (const std::string &app : apps)
    list.configs.push_back(rerun(app));
  for (const Kernel &k : build(list, seqs).kernels)
    compare(k.name, k.words);
  return done("test_elf");
}
