//===- elf_aiebu.hpp - one instruction stream as an aiebu ELF -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// What FLM npu_utils does: aiebu turns a TXN binary into an ELF. txn/elf.hpp
// writes an equivalent ELF without aiebu (tests/test_elf.cpp compares the two); this
// one stays for comparison and as a fallback (SessionOptions::elf_writer).
// Only this header needs aiebu: include it and link libaiebu.a (XRT) only if
// you use it.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <aiebu/aiebu.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace launch::txn {

inline std::vector<char> to_elf_aiebu(const std::vector<uint32_t> &words) {
  void *buf = nullptr;
  const int size = aiebu_assembler_get_elf(aiebu_assembler_buffer_type_blob_instr_transaction,
                                           reinterpret_cast<const char *>(words.data()),
                                           words.size() * sizeof(uint32_t), nullptr, 0, &buf, nullptr, 0, "", "",
                                           nullptr, 0);
  if (size <= 0 || !buf)
    throw std::runtime_error("aiebu: no ELF for the instruction stream (error " + std::to_string(size) + ")");
  std::vector<char> elf(static_cast<char *>(buf), static_cast<char *>(buf) + size);
  std::free(buf);
  return elf;
}

} // namespace launch::txn
