//===- session.hpp - run the launch kernels on the NPU, safely -----*- C++ -*-===//
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
// `Session` builds the kernels of a list, opens one hw_context and runs
// launches in runlists. The rest of the library has no XRT; this header does
// (and so needs -lxrt_coreutil).
//
//   launch::Session s(list, sequences);          // build + hw_context
//   xrt::bo a = s.buffer(bytes, 0), ...;         // sequence argument 0, ...
//   launch::RunReport r = s.run({{"cold_vadd", {a, b, c}}, {"vadd_then_vmul", {a, b, d}}});
//   if (r.status != launch::RunReport::Status::ok) ... r.message; run them again
//
// What it adds to running the kernels yourself:
//
//   order check    It knows which app is in the array (the xclbin's app when the
//                  context opens) and refuses, before anything runs, a warm kernel
//                  whose prev is not that app or a rerun of another app
//                  (std::logic_error). Between apps of different floorplans such a
//                  kernel hangs, and the context fails every runlist after it.
//   sharing guard  Another hw_context on the NPU (this process or another) puts its
//                  own xclbin in the array between our launches, without a sign
//                  (and XRT's access_mode::exclusive does not keep it out: measured).
//                  Before a runlist the Session asks the driver (npu_watch.hpp)
//                  whether another context exists; if so, by `SharedPolicy`:
//                    cold_only  each warm or rerun launch runs the cold kernel of
//                               its app and sequence instead (always right; a cold
//                               kernel is ~1.5x a warm one). The default.
//                    refuse     nothing runs; status refused
//                    allow      the launches run as asked (right only if the
//                               other context does not run meanwhile)
//                  After the runlist it asks again: if another context came, went
//                  or submitted a command meanwhile, and a launch depended on the
//                  array, the status is other_context_ran (run them again) and the
//                  Session opens a new hw_context (below). If no launch depended on
//                  it, the results stand and the array state becomes unknown: the
//                  next warm or rerun launch runs as the cold kernel. The same when
//                  more than `max_gap` (2 ms; 1 ms on Windows) passed since the last
//                  question: a context that opens, runs one launch and closes stays
//                  in the list for at least ~5.8 ms on Linux and ~2.2 ms on Windows
//                  (measured, stress S8), so it cannot pass unseen by both
//                  questions. Not covered: a context that lives and dies inside one
//                  of our runlists, and one that runs nothing (it can stay under
//                  0.1 ms on Linux, 0.7 ms on Windows). The second is not harmless:
//                  on Linux a process that opens and closes empty contexts (stress
//                  S6, --child churn-empty, pauses of 1 to 20 ms, policy cold_only)
//                  gave results that were wrong, in whole or in part, in 28 to 61
//                  of 1,000 launches the Session reported ok (0 without it), and
//                  made runlists fail. On Windows (one run, 300 launches, 20 ms
//                  pauses) none was wrong, but one runlist timed out and launches
//                  took ~100x longer. Do not share the NPU with such a process.
//                  Cost: a question takes ~1 us with no context on the NPU, ~35-65
//                  us once any exists (ours too: the driver asks the firmware);
//                  ~160-190 us on Windows (XRT's query; LAUNCH_NPU_WATCH_XRT=1).
//                  Within max_gap of the last question the Session asks only after
//                  the runlist; a context that opens in that gap is then seen after
//                  the runlist (status other_context_ran) instead of before it.
//                  Blind (Windows built without LAUNCH_NPU_WATCH_XRT=1, or no driver
//                  at `accel`): the Session cannot ask, so it takes the NPU as
//                  shared, by the same policy (cold_only: every warm and rerun
//                  launch runs as cold), unless `exclusive` says no other process
//                  uses the NPU. On Windows a warm or rerun launch after another
//                  app's context hangs the NPU; repeated, Windows resets it again
//                  and again, then stops with bugcheck 0x119.
//   recovery       A runlist that throws or does not finish in `timeout`: status
//                  failed, and the Session opens a new hw_context (~40 ms). That is
//                  the only way back from a hang or from shim DMA tasks that never
//                  finish (the AIE2P shim DMA has no channel reset; kernel/reset.hpp).
//                  Buffers made before stay valid. After `max_failures` failed
//                  runlists in a row (default: 2 on Windows, where each one is an NPU
//                  reset; no limit elsewhere) it opens none: the hw_context is
//                  closed, and every run is status stopped until reopen().
//   ELF chunks     In the ELF flow each run holds its own copy of its instruction
//                  stream in the process's 64 MB device heap (4 KB pages; a 32-core
//                  cold kernel takes 116 KB: 510 runs fit), and at most ~2,040 runs
//                  exist at once whatever their size (2,041 runs of a 4 KB rerun
//                  kernel fit, 8.4 MB; measured, amdxdna 7.0.0-rc1). The Session runs
//                  a long list of launches as runlists of at most `elf_budget` bytes
//                  and `elf_max_runs` runs, and uses the runs again (a run cannot be
//                  in one runlist twice, but it can be in the next one). The xclbin
//                  flow has neither limit (runlists of 5,000 launches, measured).
//
// The cold kernels for cold_only and for an unknown state are added to the list
// by the Session when it has none (named "cold_<app>@<sequence>").
//
// Transport::ctrlpkt (transport/ctrlpkt_strategy.hpp). Each kernel with packets
// gets its own buffer of them, bound to its packet argument (XRT argument 3 +
// Kernel::packet_arg, one the sequence does not use). A warm kernel needs the
// lanes a control-packet cold kernel leaves in the switches, and the xclbin
// the hw_context opens on has none: after open and reopen the array state is
// unknown, so the first warm or rerun launch runs as the cold kernel. A lane
// another context took away hangs the kernel, so check_order = false and
// SharedPolicy::allow are refused (std::invalid_argument).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/launch.hpp"
#include "launch/txn/elf.hpp"
#include "launch/xrt/packets.hpp"
#include "launch/xrt/npu_watch.hpp"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace launch {

enum class Flow { xclbin, elf };
enum class SharedPolicy { cold_only, refuse, allow };

struct SessionOptions {
  Flow flow = Flow::xclbin;
  SharedPolicy shared = SharedPolicy::cold_only;
  bool check_order = true;                 ///< false: run any order as asked (tests of what happens)
  size_t elf_budget = 24u << 20;           ///< ELF flow: instruction bytes per runlist (heap: 64 MB)
  size_t elf_max_runs = 1024;              ///< ELF flow: runs per runlist (~2,040 fit in a process)
  /// ELF flow: the ELF of a kernel's stream (txn::to_elf_aiebu of txn/elf_aiebu.hpp: aiebu's)
  std::function<std::vector<char>(const std::vector<uint32_t> &)> elf_writer = txn::to_elf;
  std::chrono::milliseconds timeout{10000}; ///< per runlist
  /// More than this since the Session last asked the driver: the array state is
  /// unknown (another context that runs a launch stays in the list for at least
  /// ~5.8 ms on Linux, ~2.2 ms on Windows: measured, stress S8). 0: no limit.
#if defined(_WIN32)
  std::chrono::microseconds max_gap{1000};
#else
  std::chrono::microseconds max_gap{2000};
#endif
  std::string accel = "/dev/accel/accel0"; ///< the driver, for the sharing guard
  /// The caller vouches that no other process uses the NPU: without the guard
  /// (Windows, or `accel` empty) warm and rerun launches run as asked.
  bool exclusive = false;
  /// Failed runlists in a row after which the Session stops (0: no limit).
#if defined(_WIN32)
  size_t max_failures = 2;
#else
  size_t max_failures = 0;
#endif
};

/// One launch: a kernel of the list (its name) and the sequence's buffers.
struct Launch {
  std::string kernel;
  std::vector<xrt::bo> args; ///< sequence argument i = XRT argument 3 + i
};

struct RunReport {
  enum class Status {
    ok,                ///< every launch ran; its results stand
    other_context_ran, ///< another context ran meanwhile; launches that depended on the array may be wrong
    failed,            ///< a runlist threw or did not finish; launches after `done` did not run
    refused,           ///< SharedPolicy::refuse and another context exists; nothing ran
    stopped,           ///< max_failures runlists failed in a row: no hw_context; nothing ran (reopen())
  };
  Status status = Status::ok;
  std::vector<std::string> ran; ///< the kernel each launch ran
  size_t substituted = 0;       ///< warm or rerun launches that ran as the cold kernel
  size_t done = 0;              ///< launches whose runlist finished
  size_t runlists = 0;
  bool other_context_seen = false; ///< another context existed or ran
  bool reopened = false;           ///< the Session opened a new hw_context
  std::string message;
};

inline const char *status_name(RunReport::Status s) {
  switch (s) {
  case RunReport::Status::ok: return "ok";
  case RunReport::Status::other_context_ran: return "other_context_ran";
  case RunReport::Status::failed: return "failed";
  case RunReport::Status::refused: return "refused";
  case RunReport::Status::stopped: return "stopped";
  }
  return "?";
}

class Session {
public:
  Session(ConfigList list, const SequenceMap &seqs, SessionOptions opts = {})
      : opts_(std::move(opts)), watch_(opts_.accel) {
    if (list.transport == Transport::ctrlpkt && (!opts_.check_order || opts_.shared == SharedPolicy::allow))
      throw std::invalid_argument("Transport::ctrlpkt needs check_order and a SharedPolicy other than allow (or use Transport::txn)");
    add_cold_kernels(list);
    list_ = std::move(list);
    result_ = build(list_, seqs);
    for (size_t i = 0; i < list_.configs.size(); ++i)
      index_[list_.configs[i].name] = i;
    // The array a new hw_context holds: the app of the xclbin it is opened on
    // (a warm config's prev when the list starts with one, not its app).
    base_app_ = result_.base_app;
    open();
  }

  /// Run `launches` in this order (one runlist, or several in the ELF flow).
  RunReport run(const std::vector<Launch> &launches) {
    RunReport rep;
    if (launches.empty())
      return rep;
    if (stopped_) {
      rep.status = RunReport::Status::stopped;
      rep.message = "stopped after " + std::to_string(failures_) + " failed runlists in a row (reopen() to go on)";
      return rep;
    }
    // A driver query costs ~35-55 us once any context exists (measured): when the
    // last one is younger than max_gap, it stands for "before" too. A context
    // that comes in that gap is still seen by the query after the runlist.
    const auto now = std::chrono::steady_clock::now();
    const bool fresh = opts_.max_gap.count() && now - last_time_ <= opts_.max_gap && last_;
    const std::optional<xrt_support::Snapshot> before = fresh ? last_ : others();
    // A failed query (no snapshot) counts as "another context may be there".
    const bool changed = !before || !last_ || *before != *last_;
    if (changed)
      state_.reset(); // another context came, went or ran since our last runlist
    else if (watch_.available() && opts_.max_gap.count() && !fresh)
      state_.reset(); // long enough for a whole context to come and go unseen
    rep.other_context_seen = !before || !before->empty() || changed;
    // Blind and not exclusive: another context may be there.
    const bool shared = rep.other_context_seen || (!watch_.available() && !opts_.exclusive);

    // Plan: the kernel of each launch, and the array state after it.
    std::vector<size_t> plan;
    bool depends = false; // a launch depends on what is in the array
    std::optional<std::string> state = state_;
    for (const Launch &l : launches) {
      size_t k = index_of(l.kernel);
      const Config &c = list_.configs[k];
      const bool dependent = c.kind() == Config::Kind::warm || c.kind() == Config::Kind::rerun;
      if (dependent && shared && opts_.shared == SharedPolicy::refuse) {
        rep.status = RunReport::Status::refused;
        rep.message = "another hw_context is on the NPU, and '" + c.name + "' depends on the array";
        last_ = before;
        last_time_ = now;
        return rep;
      }
      // The order first, also when the launch then runs as cold: a wrong order is
      // the caller's bug, refused alike whether the NPU is shared or not.
      if (dependent && opts_.check_order && state) {
        const std::string need = c.kind() == Config::Kind::warm ? c.prev() : c.app();
        if (*state != need)
          throw std::logic_error("launch '" + c.name + "' needs '" + need + "' in the array, but it has " +
                                 (state->empty() ? std::string("nothing (after a reset)") : "'" + *state + "'"));
      }
      if (dependent && ((shared && opts_.shared == SharedPolicy::cold_only) || (!state && opts_.check_order))) {
        k = cold_of_.at(k);
        ++rep.substituted;
      }
      depends |= list_.configs[k].kind() == Config::Kind::warm || list_.configs[k].kind() == Config::Kind::rerun;
      plan.push_back(k);
      rep.ran.push_back(list_.configs[k].name);
      state = list_.configs[k].kind() == Config::Kind::reset ? std::optional<std::string>("")
                                                              : std::optional<std::string>(list_.configs[k].app());
    }

    // Run, in chunks (ELF flow) or one runlist.
    size_t first = 0;
    while (first < plan.size()) {
      const size_t last = chunk_end(plan, first);
      try {
        run_chunk(plan, launches, first, last);
      } catch (const std::exception &e) {
        rep.status = RunReport::Status::failed;
        rep.message = e.what();
        if (opts_.max_failures && ++failures_ >= opts_.max_failures) {
          close(); // another context would hang again: stop here
          stopped_ = true;
          state_.reset();
          rep.message += "; stopped after " + std::to_string(failures_) + " failed runlists in a row";
          return rep;
        }
        restart();
        rep.reopened = true;
        last_ = others();
        return rep;
      }
      failures_ = 0;
      ++rep.runlists;
      rep.done = last;
      first = last;
    }

    const std::optional<xrt_support::Snapshot> after = others();
    last_ = after;
    last_time_ = std::chrono::steady_clock::now();
    if (!after || !before || *after != *before) {
      rep.other_context_seen = true;
      if (depends) {
        rep.status = RunReport::Status::other_context_ran;
        rep.message = "another hw_context came, went or ran during the runlist";
        restart();
        rep.reopened = true;
        last_ = others();
        return rep;
      }
      state_.reset();
      return rep;
    }
    state_ = state;
    return rep;
  }

  /// A buffer for sequence argument `arg` (in its memory group).
  xrt::bo buffer(size_t bytes, unsigned arg) {
    return xrt::bo(device_, bytes, XRT_BO_FLAGS_HOST_ONLY, kernel_.group_id(3 + (int)arg));
  }

  /// Close the hw_context and open a new one: the array is the xclbin's again.
  /// Also the way on after status stopped.
  void reopen() {
    failures_ = 0;
    stopped_ = false;
    restart();
  }

  /// The app in the array: empty after a reset, nullopt when not known.
  const std::optional<std::string> &state() const { return state_; }
  const Result &result() const { return result_; }
  const ConfigList &list() const { return list_; }
  bool watching() const { return watch_.available(); }
  /// Why the other contexts cannot be seen (empty when `watching()`).
  const std::string &watch_why() const { return watch_.why(); }
  size_t reopens() const { return reopens_; }
  bool stopped() const { return stopped_; }
  xrt::device &device() { return device_; }
  xrt::hw_context &context() { return context_; }

private:
  /// A cold kernel of the same app and sequence for each warm and rerun kernel.
  void add_cold_kernels(ConfigList &list) {
    std::map<std::pair<std::string, std::string>, std::string> cold; // (app, sequence) -> name
    for (const Config &c : list.configs)
      if (c.kind() == Config::Kind::cold)
        cold.emplace(std::pair{c.app(), c.sequence_key()}, c.name);
    const size_t n = list.configs.size();
    for (size_t i = 0; i < n; ++i) {
      const Config c = list.configs[i];
      if (c.kind() != Config::Kind::warm && c.kind() != Config::Kind::rerun)
        continue;
      const std::pair key{c.app(), c.sequence_key()};
      if (!cold.count(key)) {
        Config k = launch::cold(c.app(), "cold_" + c.app() + "@" + c.sequence_key());
        k.sequence = c.sequence;
        list.configs.push_back(k);
        cold[key] = k.name;
      }
      cold_name_[c.name] = cold[key];
    }
  }

  size_t index_of(const std::string &name) const {
    auto it = index_.find(name);
    if (it == index_.end())
      throw std::invalid_argument("no kernel '" + name + "' in the list");
    return it->second;
  }

  /// The other contexts now: an empty snapshot without the driver (nothing can
  /// be seen; `watching()` says so), std::nullopt when a query fails.
  std::optional<xrt_support::Snapshot> others() const {
    if (!watch_.available())
      return xrt_support::Snapshot{};
    return watch_.snapshot(mine_);
  }

  void close() {
    pool_.clear();
    pool_bytes_ = 0;
    kernels_.clear();
    modules_.clear();
    instr_.clear();
    packets_.clear();
    kernel_ = xrt::kernel();
    context_ = xrt::hw_context();
  }

  void restart() {
    close();
    open();
    ++reopens_;
  }

  void open() {
    const std::vector<xrt_support::ContextKey> before = watch_.own();
    xclbin_ = xrt::xclbin(result_.base_xclbin);
    device_.register_xclbin(xclbin_);
    context_ = xrt::hw_context(device_, xclbin_.get_uuid());
    kernel_ = xrt::kernel(context_, "MLIR_AIE");
    mine_.clear();
    for (const xrt_support::ContextKey &k : watch_.own())
      if (std::find(before.begin(), before.end(), k) == before.end())
        mine_.push_back(k);
    if (cold_of_.empty())
      for (const auto &[name, cold] : cold_name_)
        cold_of_[index_.at(name)] = index_.at(cold);
    run_bytes_.clear();
    for (const Kernel &k : result_.kernels) {
      if (opts_.flow == Flow::elf) {
        const std::vector<char> elf = opts_.elf_writer(k.words);
        modules_.emplace_back(xrt::elf(elf.data(), elf.size()));
        kernels_.push_back(xrt::ext::kernel(context_, modules_.back(), "MLIR_AIE"));
        run_bytes_.push_back((elf.size() + 4095) / 4096 * 4096);
      } else {
        xrt::bo &bo = instr_.emplace_back(device_, k.words.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel_.group_id(1));
        std::memcpy(bo.map<void *>(), k.words.data(), k.words.size() * 4);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }
      packets_.push_back(packet_buffer(device_, kernel_, k));
    }
    // The xclbin's app is in the array, but no control-packet lanes.
    state_ = list_.transport == Transport::ctrlpkt ? std::nullopt : std::optional<std::string>(base_app_);
    last_ = others();
    last_time_ = std::chrono::steady_clock::now();
  }

  /// The end of the runlist that starts at `first`: all of them (xclbin flow),
  /// or as many as fit in elf_budget and elf_max_runs (at least one).
  size_t chunk_end(const std::vector<size_t> &plan, size_t first) const {
    if (opts_.flow != Flow::elf)
      return plan.size();
    size_t bytes = 0, i = first;
    while (i < plan.size() && (i == first || (bytes + run_bytes_[plan[i]] <= opts_.elf_budget &&
                                              i - first < opts_.elf_max_runs)))
      bytes += run_bytes_[plan[i++]];
    return i;
  }

  using PoolKey = std::pair<size_t, std::vector<uint64_t>>; // kernel, buffer addresses

  static PoolKey key_of(size_t k, const Launch &l) {
    PoolKey key{k, {}};
    for (const xrt::bo &b : l.args)
      key.second.push_back(b.address());
    return key;
  }

  std::unique_ptr<xrt::run> make_run(size_t k, const Launch &l) {
    const xrt::kernel &kern = opts_.flow == Flow::elf ? kernels_[k] : kernel_;
    auto r = std::make_unique<xrt::run>(kern);
    r->set_arg(0, 3u); // opcode: run this instruction stream
    if (opts_.flow == Flow::elf) {
      r->set_arg(1, 0);
      r->set_arg(2, 0);
    } else {
      r->set_arg(1, instr_[k]);
      r->set_arg(2, (uint32_t)result_.kernels[k].words.size());
    }
    for (size_t i = 0; i < l.args.size(); ++i)
      r->set_arg((int)(3 + i), l.args[i]);
    set_packets(*r, result_.kernels[k], packets_[k]);
    return r;
  }

  void run_chunk(const std::vector<size_t> &plan, const std::vector<Launch> &ls, size_t first, size_t last) {
    // Runs of the pool that this chunk does not use go first (ELF: they hold heap).
    std::map<PoolKey, size_t> need;
    for (size_t i = first; i < last; ++i)
      ++need[key_of(plan[i], ls[i])];
    for (auto it = pool_.begin(); it != pool_.end();) {
      auto n = need.find(it->first);
      const size_t keep = n == need.end() ? 0 : n->second;
      while (it->second.size() > keep) {
        pool_bytes_ -= bytes_of(it->first.first);
        it->second.pop_back();
      }
      it = it->second.empty() ? pool_.erase(it) : std::next(it);
    }
    std::vector<std::pair<PoolKey, std::unique_ptr<xrt::run>>> runs;
    for (size_t i = first; i < last; ++i) {
      PoolKey key = key_of(plan[i], ls[i]);
      auto it = pool_.find(key);
      if (it != pool_.end() && !it->second.empty()) {
        runs.emplace_back(key, std::move(it->second.back()));
        it->second.pop_back();
        pool_bytes_ -= bytes_of(plan[i]);
      } else {
        runs.emplace_back(key, make_run(plan[i], ls[i]));
      }
    }
    {
      xrt::runlist rl(context_);
      for (auto &[key, r] : runs)
        rl.add(*r);
      rl.execute();
      if (rl.wait(opts_.timeout) == std::cv_status::timeout)
        throw std::runtime_error("a runlist did not finish in " + std::to_string(opts_.timeout.count()) + " ms");
    }
    for (auto &[key, r] : runs) {
      pool_bytes_ += bytes_of(key.first);
      pool_[key].push_back(std::move(r));
    }
  }

  size_t bytes_of(size_t k) const { return opts_.flow == Flow::elf ? run_bytes_[k] : 0; }

  SessionOptions opts_;
  xrt_support::NpuWatch watch_;
  ConfigList list_;
  Result result_;
  std::map<std::string, size_t> index_;
  std::map<std::string, std::string> cold_name_; ///< warm or rerun kernel -> its cold kernel
  std::map<size_t, size_t> cold_of_;
  std::string base_app_;

  xrt::device device_{0u};
  xrt::xclbin xclbin_;
  xrt::hw_context context_;
  xrt::kernel kernel_;
  std::vector<xrt::bo> instr_;       ///< xclbin flow, per kernel
  std::vector<xrt::module> modules_; ///< ELF flow, per kernel
  std::vector<xrt::kernel> kernels_; ///< ELF flow, per kernel
  std::vector<size_t> run_bytes_;    ///< ELF flow: heap bytes of one run, per kernel
  std::vector<xrt::bo> packets_;     ///< per kernel: its control packets (none: an empty bo)
  std::map<PoolKey, std::vector<std::unique_ptr<xrt::run>>> pool_;
  size_t pool_bytes_ = 0;

  std::vector<xrt_support::ContextKey> mine_;
  std::optional<xrt_support::Snapshot> last_;
  std::chrono::steady_clock::time_point last_time_;
  std::optional<std::string> state_;
  size_t reopens_ = 0;
  size_t failures_ = 0; ///< failed runlists in a row
  bool stopped_ = false;
};

} // namespace launch
