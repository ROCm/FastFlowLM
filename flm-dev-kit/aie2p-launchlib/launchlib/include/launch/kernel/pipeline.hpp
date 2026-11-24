//===- pipeline.hpp - the parts of one launch kernel --------------*- C++ -*-===//
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
// A kernel is made of the same parts, always in the same order. The kind of the
// kernel only says which parts it has: its `Recipe`, one row of this table.
//
//   kind                  reset        load  program  sequence
//   reset                 whole_array
//   cold                  whole_array  x              x
//   warm                  used_by      x              x
//   warm, program_only                       x        x
//   rerun                                             x
//
//   reset     reset.hpp `reset_targets`, `write_reset`: stop the cores and the DMA channels, set
//             the switches (the same for every transport)
//   load      the transport's `load`: the whole app, program included
//   program   the transport's `load_program`: the program only
//   sequence  the host's runtime sequence (rt/rt.hpp)
//
// The sequence is built first and appended last: the transport sees its ops,
// its shim BDs and its kernel arguments (KernelContext::sequence) before it
// writes the load.
//
// The cores loop forever (as the AIE4 designs): a kernel ends with them waiting
// for the next block, and the next kernel either runs the sequence again
// (rerun) or stops them (reset part, or load_program).
//
// A new kind is a new row of `recipe`. A new transport is a new
// `TransportStrategy` (transport/strategy.hpp).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/config.hpp"
#include "launch/kernel/context.hpp"
#include "launch/rt/rt.hpp"
#include "launch/transport/strategy.hpp"

#include <set>
#include <stdexcept>
#include <vector>

namespace launch::detail {

struct Recipe {
  ResetScope reset;
  bool load;     ///< the whole app
  bool program;  ///< the program only
  bool sequence; ///< the runtime sequence
};

inline Recipe recipe(const Config &c) {
  using K = Config::Kind;
  using S = ResetScope;
  switch (c.kind()) {
  //                               reset          load   program sequence
  case K::reset: return            {S::whole_array, false, false, false};
  case K::cold: return             {S::whole_array, true,  false, true};
  case K::warm:
    if (c.program_only()) return   {S::none,        false, true,  true};
    return                         {S::used_by,     true,  false, true};
  case K::rerun: return            {S::none,        false, false, true};
  }
  throw std::logic_error("recipe: unknown kind");
}

/// Where a part of a kernel starts in its op list (for reports: launch::breakdown).
struct PartStart {
  std::string name; ///< reset, load, program, sequence
  size_t first;     ///< index of its first op
};

/// The kernel arguments of the sequence buffers (XRT arguments 3 + i of the
/// MLIR_AIE kernel: bo0 .. bo4).
inline constexpr int kKernelArgs = 5;

struct KernelBuild {
  std::vector<txn::Op> ops;
  std::set<unsigned> args; ///< the kernel arguments the runtime sequence uses
  std::vector<PartStart> parts;
  std::vector<uint32_t> packets; ///< control packets of the transport (empty: none)
  int packet_arg = -1;           ///< their kernel argument
};

/// Builds the ops of each kernel (Template Method); the transport is the Strategy.
class KernelPipeline {
public:
  KernelPipeline(const Apps &apps, TransportStrategy &transport, const SequenceMap &seqs)
      : apps_(apps), transport_(transport), seqs_(seqs) {}

  KernelBuild build_kernel(const Config &c) const {
    KernelBuild out;
    KernelContext k(out.ops, apps_, c);
    const Recipe r = recipe(c);
    if (r.program) {
      const std::string diff = floorplan_difference(*k.prev, *k.run);
      if (!diff.empty())
        k.fail("program_only, but '" + c.prev() + "' and '" + c.app() + "' differ in more than the program: " +
               diff);
    }
    auto part = [&](const char *name) { out.parts.push_back({name, out.ops.size()}); };
    std::vector<txn::Op> sequence;
    if (r.sequence) {
      out.args = run_sequence(k, sequence);
      k.sequence = &sequence;
      k.sequence_args = out.args;
      for (int a = kKernelArgs - 1; a >= 0 && k.free_arg < 0; --a)
        if (!out.args.count((unsigned)a))
          k.free_arg = a;
    }
    if (r.reset != ResetScope::none) {
      part("reset");
      ResetTargets t = reset_targets(r.reset, k, k.prev ? &transport_.final_routes(*k.prev) : nullptr,
                                     k.run ? &transport_.load_routes(*k.run) : nullptr);
      transport_.adjust_reset(k, t);
      write_reset(k, t);
    }
    if (r.load) {
      part("load");
      transport_.load(k);
    }
    if (r.program) {
      part("program");
      transport_.load_program(k);
    }
    if (r.sequence) {
      part("sequence");
      out.ops.insert(out.ops.end(), sequence.begin(), sequence.end());
    }
    out.packets = std::move(k.packets);
    out.packet_arg = k.packet_arg;
    return out;
  }

private:
  std::set<unsigned> run_sequence(const KernelContext &k, std::vector<txn::Op> &ops) const {
    const std::string key = k.c.sequence_key();
    auto seq = seqs_.find(key);
    if (seq == seqs_.end())
      k.fail("no runtime sequence '" + key + "'");
    Rt rt(ops, *k.run);
    try {
      seq->second(rt);
    } catch (const std::exception &e) {
      k.fail("runtime sequence '" + key + "': " + e.what());
    }
    return rt.args();
  }

  const Apps &apps_;
  TransportStrategy &transport_;
  const SequenceMap &seqs_;
};

} // namespace launch::detail
