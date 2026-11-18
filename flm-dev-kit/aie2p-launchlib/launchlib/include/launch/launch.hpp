//===- launch.hpp - build the launch kernels of a list ------------*- C++ -*-===//
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
// The one call of the library:
//
//     launch::Result r = launch::build(list, sequences);
//     // list:      a launch::ConfigList (config.hpp), from JSON or filled in C++
//     // sequences: a launch::SequenceMap (rt/rt.hpp): app name -> its runtime
//     //            sequence. The library has none; they are the host program's.
//
// `build()` reads the package of every app of the list (formats/package.hpp)
// and makes one instruction stream (TXN binary) for each entry of the list.
// Nothing is written to a file and no tool is run.
//
// To run the kernels (AIE2P needs an xclbin for the hw_context; AIE4 does not):
//
//   xclbin flow   xrt::hw_context on r.base_xclbin (the xclbin of the first app
//                 of the list; any one works, because reset and cold kernels
//                 reset the whole array), kernel "MLIR_AIE", and per kernel
//                 run(3, instruction BO of r.kernels[i].words, its size, the
//                 sequence's buffers from argument 3 on)
//   ELF flow      txn/elf.hpp turns r.kernels[i].words into an ELF for
//                 xrt::elf / xrt::module / xrt::ext::kernel (FLM npu_utils)
//
// A reset or cold kernel resets the whole array, so it can follow any kernel. A
// warm kernel resets only what `prev` and its app use, so the host launches it
// right after a kernel of `prev`. A rerun kernel runs only the sequence of the
// app in the array. The design:
//
//   kernel/pipeline.hpp   `KernelPipeline::build_kernel`: the parts, in a fixed
//                         order; `recipe` (a table) says which parts each kind has
//   kernel/reset.hpp      the reset part; `ResetScope` says which registers
//   transport/            Strategy: how the state reaches the tiles
//     txn_strategy.hpp    command-processor writes in the kernel's own stream
//     ctrlpkt_strategy.hpp control packets the shim DMAs send, one lane per
//                         column (ctrlpkt_lane.hpp), and TXN writes for the rest
//   txn/txn.hpp           the instruction stream: IR, encode, parse
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/config.hpp"
#include "launch/formats/package.hpp"
#include "launch/kernel/pipeline.hpp"
#include "launch/rt/rt.hpp"
#include "launch/transport/ctrlpkt_check.hpp"
#include "launch/transport/ctrlpkt_strategy.hpp"
#include "launch/transport/txn_strategy.hpp"
#include "launch/txn/txn.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace launch {

/// The ops of one kernel (for reports and tests).
struct KernelStats {
  size_t ops = 0, bytes = 0;            ///< ops, and bytes of the whole stream
  size_t write32 = 0, maskwrite32 = 0, blockwrite = 0, block_bytes = 0;
  size_t program_bytes = 0;             ///< block-write bytes into core PM
  size_t patches = 0, syncs = 0, polls = 0;
};

struct Kernel {
  std::string name;
  std::vector<uint32_t> words; ///< the instruction stream (TXN binary)
  KernelStats stats;
  std::vector<detail::PartStart> parts; ///< where each part starts (op index)
  /// Transport::ctrlpkt: the control packets the kernel's shim BDs send, bound
  /// as sequence argument `packet_arg` (XRT argument 3 + packet_arg). Empty and
  /// -1 when the kernel sends none.
  std::vector<uint32_t> packets;
  int packet_arg = -1;
};

/// What the command processor takes per op, in ns: measured on npu5 by
/// tools/op_cost.cpp (two runs; ±20%). A block write costs `blockwrite` plus
/// `word` (data memory, BDs, routes) or `pm_word` (core program memory) per
/// word. A wait (Sync) takes as long as the DMA it waits for: not modelled.
/// Checked against measured switches of vadd_vmul_bf16 8 x 4 (the kernel less
/// its runtime sequence): the full warm switch 212 us estimated, ~237 measured;
/// program only 42, ~44.
struct CostModel {
  double write32 = 60, maskwrite32 = 110, blockwrite = 60, word = 4.0, pm_word = 5.0, patch = 60, poll = 110;

  double ns(const txn::Op &op) const {
    return std::visit(txn::Overload{
                          [&](const txn::Write32 &) { return write32; },
                          [&](const txn::MaskWrite32 &) { return maskwrite32; },
                          [&](const txn::MaskPoll &) { return poll; },
                          [&](const txn::BlockWrite &b) {
                            const bool pm = hw::classify(b.addr) == hw::Cls::program;
                            return blockwrite + (pm ? pm_word : word) * (double)b.words.size();
                          },
                          [&](const txn::AddressPatch &) { return patch; },
                          [&](const txn::Sync &) { return 0.0; },
                          [&](const txn::Preempt &) { return 0.0; },
                      },
                      op);
  }
};

/// The estimated time of a kernel's ops but its waits, in us (CostModel).
inline double estimate_us(const Kernel &k, const CostModel &m = {}) {
  double ns = 0;
  for (const txn::Op &op : txn::parse(k.words).ops)
    ns += m.ns(op);
  return ns / 1000;
}

/// One row of `breakdown`: what a part of a kernel writes into one register class.
struct BreakdownRow {
  std::string part;  ///< reset, load, program, sequence
  std::string what;  ///< a register class (hw::cls_name), or "wait", "patch"
  size_t ops = 0;    ///< TXN ops
  size_t words = 0;  ///< 32-bit registers or memory words written
  double us = 0;     ///< estimated time (CostModel)
};

/// What each part of a kernel writes, by register class, in op order of first
/// appearance (for reports: which registers a switch writes, how many, and
/// about how long they take).
inline std::vector<BreakdownRow> breakdown(const Kernel &k, const CostModel &m = {}) {
  const txn::Stream s = txn::parse(k.words);
  std::vector<BreakdownRow> rows;
  for (size_t p = 0; p < k.parts.size(); ++p) {
    const size_t first = k.parts[p].first, last = p + 1 < k.parts.size() ? k.parts[p + 1].first : s.ops.size();
    for (size_t i = first; i < last; ++i) {
      const txn::Op &op = s.ops[i];
      std::string what;
      size_t words = 0;
      if (std::holds_alternative<txn::Sync>(op))
        what = "wait";
      else if (std::holds_alternative<txn::AddressPatch>(op))
        what = "patch";
      else if (std::holds_alternative<txn::Preempt>(op))
        what = "preempt";
      else if (std::holds_alternative<txn::MaskPoll>(op))
        what = "poll";
      else {
        what = hw::cls_name(hw::classify(txn::addr_of(op)));
        const auto *b = std::get_if<txn::BlockWrite>(&op);
        words = b ? b->words.size() : 1;
      }
      auto it = std::find_if(rows.begin(), rows.end(),
                             [&](const BreakdownRow &r) { return r.part == k.parts[p].name && r.what == what; });
      if (it == rows.end()) {
        rows.push_back({k.parts[p].name, what, 0, 0});
        it = rows.end() - 1;
      }
      ++it->ops;
      it->words += words;
      it->us += m.ns(op) / 1000;
    }
  }
  return rows;
}

struct Result {
  std::vector<Kernel> kernels; ///< in the order of the list
  std::string base_xclbin;     ///< an xclbin for the hw_context (the first app's)
  std::string base_app;        ///< that first app: the first one the list names, a warm's prev included
  unsigned arg_count = 0;      ///< sequence buffers: the largest argument used + 1
  double build_ms = 0;
};

namespace detail {

inline const std::string kNoSequence = "\x01none"; ///< a sequence key of no ops (the check's cold kernels)

inline std::unique_ptr<TransportStrategy> make_transport(Transport t, const CtrlPktOptions &o = {}) {
  switch (t) {
  case Transport::txn: return std::make_unique<TxnStrategy>();
  case Transport::ctrlpkt: return std::make_unique<CtrlPktStrategy>(o);
  }
  throw std::invalid_argument("unknown transport");
}

inline KernelStats count_ops(const std::vector<txn::Op> &ops, size_t words) {
  KernelStats s;
  s.ops = ops.size();
  s.bytes = 4 * words;
  for (const txn::Op &op : ops)
    std::visit(txn::Overload{
                   [&](const txn::Write32 &) { ++s.write32; },
                   [&](const txn::MaskWrite32 &) { ++s.maskwrite32; },
                   [&](const txn::MaskPoll &) { ++s.polls; },
                   [&](const txn::BlockWrite &b) {
                     ++s.blockwrite;
                     s.block_bytes += 4 * b.words.size();
                     if (hw::classify(b.addr) == hw::Cls::program)
                       s.program_bytes += 4 * b.words.size();
                   },
                   [&](const txn::AddressPatch &) { ++s.patches; },
                   [&](const txn::Sync &) { ++s.syncs; },
                   [&](const txn::Preempt &) {},
               },
               op);
  return s;
}

} // namespace detail

/// Build the kernels of a list with the runtime sequences of `seqs`. Throws
/// std::invalid_argument for a wrong list and std::runtime_error for a wrong
/// package or sequence; the error names the entry or the file.
inline Result build(const ConfigList &list, const SequenceMap &seqs) {
  using namespace detail;
  const auto t0 = std::chrono::steady_clock::now();
  validate(list);

  // Read the package of every app of the list one time.
  Apps apps;
  std::string first_app;
  for (const Config &c : list.configs)
    for (const std::string &app : c.apps()) {
      if (first_app.empty())
        first_app = app;
      if (apps.count(app))
        continue;
      try {
        apps.emplace(app, load_package(list.package_dir, app));
      } catch (const std::exception &e) {
        throw std::runtime_error("configuration '" + c.name + "': " + e.what());
      }
    }

  const std::unique_ptr<TransportStrategy> transport = make_transport(list.transport, list.ctrlpkt);
  const KernelPipeline pipeline(apps, *transport, seqs);

  Result res;
  if (!first_app.empty()) {
    res.base_xclbin = apps.at(first_app).xclbin;
    res.base_app = first_app;
  }
  // Transport::ctrlpkt: the same kernels with TXN writes, for the check
  // (ctrlpkt_check.hpp). A warm kernel starts from each transport's array after
  // cold(prev) (without a sequence).
  auto *cps = dynamic_cast<CtrlPktStrategy *>(transport.get());
  std::unique_ptr<TransportStrategy> txn_transport;
  SequenceMap check_seqs;
  std::optional<KernelPipeline> txn_pipeline, cp_pipeline;
  std::map<std::string, std::pair<ctrlpkt::Replay, sw::Registers>> after_cold;
  if (cps && list.ctrlpkt.check) {
    txn_transport = make_transport(Transport::txn);
    check_seqs = seqs;
    check_seqs[kNoSequence] = [](Rt &) {};
    txn_pipeline.emplace(apps, *txn_transport, check_seqs);
    cp_pipeline.emplace(apps, *transport, check_seqs);
  }
  auto check = [&](const Config &c, const KernelBuild &b, const KernelBuild &t,
                   const std::pair<ctrlpkt::Replay, sw::Registers> *start, ctrlpkt::Replay *end) {
    const Package *run = c.app().empty() ? nullptr : &apps.at(c.app());
    const std::vector<std::string> errors = ctrlpkt::check_kernel(
        {run, run ? &cps->plan_of(*run) : nullptr, list.ctrlpkt.memtile_data, b.ops, b.packets, b.packet_arg, t.ops,
         start ? &start->first : nullptr, start ? &start->second : nullptr},
        end);
    if (!errors.empty()) {
      std::string m = "configuration '" + c.name + "': the control-packet kernel fails its check:";
      for (const std::string &e : errors)
        m += "\n  " + e;
      throw std::logic_error(m);
    }
  };
  auto start_of = [&](const std::string &app) -> const std::pair<ctrlpkt::Replay, sw::Registers> & {
    auto it = after_cold.find(app);
    if (it != after_cold.end())
      return it->second;
    Config c = cold(app);
    c.name = "check: cold(" + app + ")";
    c.sequence = kNoSequence;
    const KernelBuild b = cp_pipeline->build_kernel(c), t = txn_pipeline->build_kernel(c);
    std::pair<ctrlpkt::Replay, sw::Registers> st;
    check(c, b, t, nullptr, &st.first);
    for (const txn::Op &op : t.ops)
      ctrlpkt::check_detail::apply(st.second, op);
    return after_cold.emplace(app, std::move(st)).first->second;
  };

  for (const Config &c : list.configs) {
    KernelBuild b = pipeline.build_kernel(c);
    if (txn_pipeline && c.kind() != Config::Kind::rerun) {
      const KernelBuild t = txn_pipeline->build_kernel(c);
      check(c, b, t, c.kind() == Config::Kind::warm ? &start_of(c.prev()) : nullptr, nullptr);
    }
    if (!b.args.empty())
      res.arg_count = std::max(res.arg_count, *b.args.rbegin() + 1);
    txn::Stream s;
    if (!first_app.empty())
      s.header = apps.at(first_app).header;
    s.ops = std::move(b.ops);
    Kernel k{c.name, txn::encode(s), {}, std::move(b.parts), std::move(b.packets), b.packet_arg};
    k.stats = count_ops(s.ops, k.words.size());
    res.kernels.push_back(std::move(k));
  }
  res.build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return res;
}

} // namespace launch
