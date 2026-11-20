//===- design.hpp - what the host test knows of one design -----*- C++ -*-===//
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
// A `Design` owns the kernel arguments of one kernel: it makes the buffers,
// fills the inputs, sets the arguments of a run and checks the output. Each
// design has its own file in this folder (vadd.hpp, vmul.hpp, vminus.hpp);
// `designs.hpp` maps the app name to it.
//
// The sequence's argument i is XRT argument 3 + i (launch::Session sets 0..2).
// A buffer is made by the Session, in the memory group of its argument.
//
// A design only touches the host copy of its buffers. The test does all the
// syncs, in `row` (test.cpp): inputs() and outputs() to the device before the
// launches, outputs() from the device after them.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace hosttest {

/// Makes a buffer of `bytes` for sequence argument `arg` (launch::Session::buffer).
using MakeBuffer = std::function<xrt::bo(size_t bytes, unsigned arg)>;

class Design {
public:
  virtual ~Design() = default;
  /// The sequence's arguments (XRT arguments 3, 4, ...).
  virtual std::vector<xrt::bo> args() const = 0;
  /// The buffers that the kernel reads.
  virtual std::vector<xrt::bo *> inputs() = 0;
  /// The buffers that the kernel writes.
  virtual std::vector<xrt::bo *> outputs() = 0;
  /// Host copy: make the outputs invalid (NaN), so that a stale result fails the check.
  virtual void clear_outputs() = 0;
  /// Host copy: compare the outputs with the golden result. `who` names the kernel.
  virtual bool check(const std::string &who) = 0;
};

//===----------------------------------------------------------------------===//
// fp32 helpers: all three designs are C = f(A, B), three buffers
//===----------------------------------------------------------------------===//

/// Two inputs of `n` values in [-100, 100). Seed 42: the same data in every run.
inline void fp32_random_inputs(size_t n, std::vector<float> &a, std::vector<float> &b) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
  a.resize(n);
  b.resize(n);
  for (size_t i = 0; i < n; i++) {
    a[i] = dis(gen);
    b[i] = dis(gen);
  }
}

/// A buffer of `n` fp32 for sequence argument `arg`, with `v` in its host copy (not synced).
inline xrt::bo fp32_buffer(const MakeBuffer &make, unsigned arg, size_t n, const std::vector<float> &v = {}) {
  xrt::bo bo = make(n * sizeof(float), arg);
  if (!v.empty())
    std::memcpy(bo.map<float *>(), v.data(), v.size() * sizeof(float));
  return bo;
}

/// Fill the host copy of a buffer of `n` fp32 with NaN (not synced).
inline void fp32_fill_nan(xrt::bo &bo, size_t n) {
  float *p = bo.map<float *>();
  for (size_t i = 0; i < n; i++)
    p[i] = NAN;
}

/// One unit in the last place of |x|.
inline float fp32_ulp(float x) { return std::nextafter(std::fabs(x), INFINITY) - std::fabs(x); }

/// The AIE2P fp32 results are not always IEEE round to nearest (measured on
/// these designs: 1.2% of the sums and differences, 0.3% of the products differ;
/// no directed rounding mode fits). A sum or difference is within 1 ulp of the
/// largest of the inputs and the result (the adder drops the bits it shifts out
/// of the smaller input: the error is largest when the result is much smaller
/// than the inputs), a product within 1 ulp of the result. The golden result
/// carries that bound per value; a wrong program (+ for *) or a stale output
/// (NaN) is far outside it.
inline float fp32_sum_bound(float a, float b, float result) {
  return fp32_ulp(std::max({std::fabs(a), std::fabs(b), std::fabs(result)}));
}
inline float fp32_product_bound(float, float, float result) { return fp32_ulp(result); }

/// Compare the host copy of `bo` with `want`, each within its `bound`. Print at most 3 mismatches.
inline bool fp32_compare(xrt::bo &bo, const std::vector<float> &want, const std::vector<float> &bound,
                         const std::string &who) {
  const float *got = bo.map<float *>();
  size_t errs = 0;
  for (size_t i = 0; i < want.size(); i++)
    if (!(std::fabs(got[i] - want[i]) <= bound[i]) && ++errs <= 3) // false for NaN
      printf("  mismatch %s [%zu]: got %.9g expected %.9g (+- %.3g)\n", who.c_str(), i, got[i], want[i], bound[i]);
  if (errs)
    printf("  %s: %zu of %zu values wrong\n", who.c_str(), errs, want.size());
  return errs == 0;
}

/// C = f(A, B) on kElems fp32; `a_elems` of A (vadd: N, shared by every core).
class Fp32Design : public Design {
public:
  std::vector<xrt::bo> args() const override { return {a_, b_, c_}; }
  std::vector<xrt::bo *> inputs() override { return {&a_, &b_}; }
  std::vector<xrt::bo *> outputs() override { return {&c_}; }
  void clear_outputs() override { fp32_fill_nan(c_, golden_.size()); }
  bool check(const std::string &who) override { return fp32_compare(c_, golden_, bound_, who); }

protected:
  /// Make the buffers: A has `a_elems` values, B and C `elems`; golden[i] = f(a[i % a_elems], b[i]),
  /// within bound(a[i % a_elems], b[i], golden[i]).
  template <class F, class Bound>
  void make(const MakeBuffer &mk, size_t a_elems, size_t elems, F f, Bound bound) {
    std::vector<float> a, b;
    fp32_random_inputs(elems, a, b);
    a.resize(a_elems);
    golden_.resize(elems);
    bound_.resize(elems);
    for (size_t i = 0; i < elems; i++) {
      golden_[i] = f(a[i % a_elems], b[i]);
      bound_[i] = bound(a[i % a_elems], b[i], golden_[i]);
    }
    a_ = fp32_buffer(mk, 0, a_elems, a);
    b_ = fp32_buffer(mk, 1, elems, b);
    c_ = fp32_buffer(mk, 2, elems);
  }

private:
  std::vector<float> golden_, bound_;
  xrt::bo a_, b_, c_;
};

} // namespace hosttest
