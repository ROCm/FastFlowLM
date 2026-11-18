//===- elf_probe.cpp - do the encoder's switch kernels run as ELFs? -------===//
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
//   elf_probe.exe --package build/pkg [--flow xclbin|elf]
//
// cold(mm), then warm(mm, attn), warm(attn, mm), cold(attn), cold(mm), each in
// its own run through launch::Session (empty runtime sequences). FLM runs every
// instruction stream as an ELF (aiebu's); switch_cost ran them in the xclbin flow.
//
//===----------------------------------------------------------------------===//

#include "launch/xrt/session.hpp"

#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  std::string package, flow = "elf";
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string opt = argv[i];
    if (opt == "--package") package = argv[i + 1];
    else if (opt == "--flow") flow = argv[i + 1];
  }
  if (package.empty()) {
    std::cerr << "usage: elf_probe.exe --package DIR [--flow xclbin|elf]\n";
    return 2;
  }
  launch::ConfigList list{package, launch::Transport::txn,
                          {launch::cold("mm"), launch::warm("mm", "attn"), launch::warm("attn", "mm"),
                           launch::cold("attn")}};
  const auto none = [](launch::Rt &) {};
  launch::SessionOptions o;
  o.flow = flow == "elf" ? launch::Flow::elf : launch::Flow::xclbin;
  o.timeout = std::chrono::milliseconds(5000);
  launch::Session s(list, {{"mm", none}, {"attn", none}}, o);
  for (const char *k : {"cold_mm", "mm_then_attn", "attn_then_mm", "cold_attn", "cold_mm"}) {
    const launch::RunReport r = s.run({{k, {}}});
    std::printf("%-6s %-14s %s %s\n", flow.c_str(), k, launch::status_name(r.status), r.message.c_str());
  }
  return 0;
}
