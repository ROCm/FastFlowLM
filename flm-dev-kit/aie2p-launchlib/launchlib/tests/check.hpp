//===- check.hpp - the few test helpers the unit tests share ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// Portions of this file consist of AI-generated content.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

inline int g_failures = 0;

#define CHECK(cond)                                                                                  \
  do {                                                                                               \
    if (!(cond)) {                                                                                   \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);                  \
      ++g_failures;                                                                                  \
    }                                                                                                \
  } while (0)

/// True if `f` throws (any exception).
template <class F> bool throws(F f) {
  try {
    f();
  } catch (const std::exception &) {
    return true;
  }
  return false;
}

inline std::vector<char> read_bytes(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

inline int done(const char *name) {
  std::printf("%s: %s\n", name, g_failures ? "FAIL" : "PASS");
  return g_failures ? 1 : 0;
}
