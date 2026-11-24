//===- designs.hpp - the validation of each design ---------------*- C++ -*-===//
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
// One class per design (the sizes of its sequence in sequence.hpp), and
// `make_design`: app name -> design. A new design adds a class and a line.
//
//   vadd    C[i] = A[i % N] + B[i]   A: N fp32, broadcast to every core
//   vmul    C = A * B                 A, B, C: kElems fp32
//   vminus  C = A - B                 A, B, C: kElems fp32
//
//===----------------------------------------------------------------------===//

#pragma once

#include "designs/design.hpp"
#include "sequence.hpp"

#include <memory>
#include <stdexcept>

namespace hosttest {

class Vadd final : public Fp32Design {
public:
  explicit Vadd(const MakeBuffer &mk) {
    make(mk, N, kElems, [](float a, float b) { return a + b; }, fp32_sum_bound);
  }
};

class Vmul final : public Fp32Design {
public:
  explicit Vmul(const MakeBuffer &mk) {
    make(mk, kElems, kElems, [](float a, float b) { return a * b; }, fp32_product_bound);
  }
};

class Vminus final : public Fp32Design {
public:
  explicit Vminus(const MakeBuffer &mk) {
    make(mk, kElems, kElems, [](float a, float b) { return a - b; }, fp32_sum_bound);
  }
};

/// The arguments and the check of one kernel that runs `app`. Each kernel gets its own.
inline std::unique_ptr<Design> make_design(const std::string &app, const MakeBuffer &mk) {
  if (app == "vadd") return std::make_unique<Vadd>(mk);
  if (app == "vmul") return std::make_unique<Vmul>(mk);
  if (app == "vminus") return std::make_unique<Vminus>(mk);
  throw std::runtime_error("the host test has no design for app '" + app + "' (add one in designs/designs.hpp)");
}

} // namespace hosttest
