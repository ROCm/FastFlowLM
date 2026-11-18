//===- config.hpp - the list of kernels to build ------------------*- C++ -*-===//
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
// One configuration gives one kernel. The kinds (kernel/pipeline.hpp has the
// parts of each):
//
//   reset()                   resets the whole array; loads nothing
//   cold(app)                 resets the whole array, loads app, runs its sequence
//   warm(prev, app)           resets what prev and app use, loads app (with its
//                             program), runs its sequence. The host must launch
//                             it right after a kernel that left prev in the array;
//                             the kernel does not check that.
//   warm(prev, app, {.program_only = true})
//                             only the program: per core reset, program, start.
//                             build() checks that prev and app differ in their
//                             programs only. The launches of prev must also leave
//                             every ping/pong DMA chain at ping (an even number of
//                             blocks per launch); the library cannot see that.
//   rerun(app)                only the runtime sequence: app is still in the array
//
// There is no chain_start and no `next` as on AIE4: AIE2P has no PM ping-pong;
// a warm kernel writes its own program.
//
// A configuration may name the runtime sequence (`sequence`, default: the app),
// so one app can have several sequences (other shapes or buffers).
//
// JSON form (config_from_json / config_to_json):
//   { "package_dir": "build/pkg", "transport": "ctrlpkt",
//     "configs":[ {"kind": "cold", "app": "vadd"},
//                  {"kind": "warm", "prev": "vadd", "app": "vmul", "program_only": true},
//                  {"kind": "rerun", "app": "vadd", "name": "again", "sequence": "vadd_big"} ] }
// "transport": "ctrlpkt" (the default when the key is missing) or "txn"; with
// ctrlpkt an optional
//   "ctrlpkt": {"memtile_data": false, "starts_by_packet": true, "check": true, "rewrite_lanes": true}
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nlohmann/json.hpp>

#include <set>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace launch {

/// How the state of an app reaches the tiles (transport/). ctrlpkt is the
/// default (faster on every design wider than one column); txn where another
/// hw_context shares the NPU (not tested with ctrlpkt) or on 1 x 1 designs.
enum class Transport {
  txn,     ///< command-processor writes in the kernel's own instruction stream
  ctrlpkt, ///< control packets a shim DMA sends down a lane of packet routes (transport/ctrlpkt_*.hpp)
};

inline const char *transport_name(Transport t) { return t == Transport::ctrlpkt ? "ctrlpkt" : "txn"; }
inline Transport transport_from_name(const std::string &t) {
  if (t == "txn")
    return Transport::txn;
  if (t == "ctrlpkt")
    return Transport::ctrlpkt;
  throw std::invalid_argument("transport '" + t + "': 'txn' or 'ctrlpkt'");
}

/// Choices of the control-packet transport (ignored by txn).
struct CtrlPktOptions {
  bool memtile_data = false;   ///< memory-tile data by packet too (default: by TXN writes)
  bool starts_by_packet = true; ///< DMA starts and core starts by packet when the lane allows it
  bool check = true;            ///< build() replays every kernel against its TXN twin (ctrlpkt_check.hpp)
  /// Warm and program-only kernels write every register of their lanes (and
  /// the lane's mux bit), not only those prev left at other values: a lane
  /// another hw_context's switch took away then cannot hang the kernel.
  bool rewrite_lanes = true;
};

struct WarmOptions {
  bool program_only = false;
};

struct Config {
  struct Reset {};
  struct Cold {
    std::string app;
  };
  struct Warm {
    std::string prev, app;
    WarmOptions options;
  };
  struct Rerun {
    std::string app;
  };
  using Spec = std::variant<Reset, Cold, Warm, Rerun>;
  enum class Kind { reset, cold, warm, rerun };

  Spec spec;
  std::string name;
  std::string sequence; ///< key of the SequenceMap; empty = the app

  Kind kind() const { return (Kind)spec.index(); }
  /// The app the kernel runs (empty for reset).
  std::string app() const {
    return std::visit([](const auto &s) -> std::string {
      if constexpr (requires { s.app; })
        return s.app;
      else
        return "";
    }, spec);
  }
  std::string prev() const {
    const Warm *w = std::get_if<Warm>(&spec);
    return w ? w->prev : "";
  }
  bool program_only() const {
    const Warm *w = std::get_if<Warm>(&spec);
    return w && w->options.program_only;
  }
  /// The apps whose packages the kernel reads.
  std::vector<std::string> apps() const {
    std::vector<std::string> a;
    if (!prev().empty())
      a.push_back(prev());
    if (!app().empty())
      a.push_back(app());
    return a;
  }
  std::string sequence_key() const { return sequence.empty() ? app() : sequence; }
};

inline const char *kind_name(Config::Kind k) {
  switch (k) {
  case Config::Kind::reset: return "reset";
  case Config::Kind::cold: return "cold";
  case Config::Kind::warm: return "warm";
  case Config::Kind::rerun: return "rerun";
  }
  return "?";
}

struct ConfigList {
  std::string package_dir = "build/pkg";
  Transport transport = Transport::ctrlpkt;
  std::vector<Config> configs;
  CtrlPktOptions ctrlpkt;
};

namespace detail {

inline std::string default_name(const Config &c) {
  switch (c.kind()) {
  case Config::Kind::reset: return "reset";
  case Config::Kind::cold: return "cold_" + c.app();
  case Config::Kind::warm: return c.prev() + "_then_" + c.app() + (c.program_only() ? "_pm" : "");
  case Config::Kind::rerun: return "rerun_" + c.app();
  }
  return "";
}

inline Config named(Config::Spec spec, const std::string &name, const std::string &sequence = "") {
  Config c{std::move(spec), name, sequence};
  if (c.name.empty())
    c.name = default_name(c);
  return c;
}

} // namespace detail

inline Config reset(const std::string &name = "") { return detail::named(Config::Reset{}, name); }
inline Config cold(const std::string &app, const std::string &name = "") {
  return detail::named(Config::Cold{app}, name);
}
inline Config warm(const std::string &prev, const std::string &app, WarmOptions options = {},
                   const std::string &name = "") {
  return detail::named(Config::Warm{prev, app, options}, name);
}
inline Config rerun(const std::string &app, const std::string &name = "") {
  return detail::named(Config::Rerun{app}, name);
}

/// cold(apps[0]), then a warm kernel from each app to the next, and from the
/// last back to the first: launch the cold one once, then the warm ones in turn.
inline std::vector<Config> cycle(const std::vector<std::string> &apps, WarmOptions options = {}) {
  if (apps.size() < 2)
    throw std::invalid_argument("cycle: needs two apps or more");
  std::vector<Config> out{cold(apps[0])};
  for (size_t i = 0; i < apps.size(); ++i)
    out.push_back(warm(apps[i], apps[(i + 1) % apps.size()], options));
  return out;
}

/// Throws std::invalid_argument naming the configuration.
inline void validate(const ConfigList &list) {
  if (list.configs.empty())
    throw std::invalid_argument("the list has no configuration");
  std::set<std::string> names;
  for (const Config &c : list.configs) {
    auto fail = [&](const std::string &m) {
      throw std::invalid_argument("configuration '" + c.name + "': " + m);
    };
    if (c.name.empty())
      fail("no name");
    if (!names.insert(c.name).second)
      fail("a second configuration has this name");
    if (c.kind() != Config::Kind::reset && c.app().empty())
      fail("no app");
    if (c.kind() == Config::Kind::warm && c.prev().empty())
      fail("a warm kernel needs prev");
    if (c.kind() == Config::Kind::reset && !c.sequence.empty())
      fail("a reset kernel has no sequence");
  }
}

inline ConfigList config_from_json(const std::string &text) {
  using json = nlohmann::json;
  const json root = json::parse(text);
  ConfigList out;
  if (root.contains("package_dir"))
    out.package_dir = root["package_dir"].get<std::string>();
  if (root.contains("transport"))
    out.transport = transport_from_name(root["transport"].get<std::string>());
  if (root.contains("ctrlpkt")) {
    const json &o = root["ctrlpkt"];
    for (const auto &[key, value] : o.items())
      if (key != "memtile_data" && key != "starts_by_packet" && key != "check" && key != "rewrite_lanes")
        throw std::invalid_argument("ctrlpkt has no '" + key + "'");
    out.ctrlpkt.memtile_data = o.value("memtile_data", false);
    out.ctrlpkt.starts_by_packet = o.value("starts_by_packet", true);
    out.ctrlpkt.check = o.value("check", true);
    out.ctrlpkt.rewrite_lanes = o.value("rewrite_lanes", true);
  }
  for (const json &e : root.at("configs")) {
    const std::string kind = e.at("kind").get<std::string>();
    if (kind != "reset" && kind != "cold" && kind != "warm" && kind != "rerun")
      throw std::invalid_argument("unknown kind '" + kind + "' (reset, cold, warm, rerun)");
    auto str = [&](const char *key) { return e.contains(key) ? e[key].get<std::string>() : std::string(); };
    // A key of another kind is an error, as in C++ (the types do not have it).
    std::set<std::string> allowed = {"kind", "name", "sequence"};
    if (kind == "cold" || kind == "rerun")
      allowed.insert("app");
    if (kind == "warm")
      allowed.insert({"app", "prev", "program_only"});
    for (const auto &[key, value] : e.items())
      if (!allowed.count(key))
        throw std::invalid_argument("a " + kind + " kernel has no '" + key + "'");
    Config::Spec spec;
    if (kind == "reset")
      spec = Config::Reset{};
    else if (kind == "cold")
      spec = Config::Cold{str("app")};
    else if (kind == "warm")
      spec = Config::Warm{str("prev"), str("app"), {e.value("program_only", false)}};
    else if (kind == "rerun")
      spec = Config::Rerun{str("app")};
    else
      throw std::invalid_argument("unknown kind '" + kind + "'");
    out.configs.push_back(detail::named(spec, str("name"), str("sequence")));
  }
  validate(out);
  return out;
}

inline std::string config_to_json(const ConfigList &list) {
  using json = nlohmann::json;
  json root = {{"package_dir", list.package_dir}, {"transport", transport_name(list.transport)},
               {"configs", json::array()}};
  if (list.transport == Transport::ctrlpkt)
    root["ctrlpkt"] = {{"memtile_data", list.ctrlpkt.memtile_data},
                       {"starts_by_packet", list.ctrlpkt.starts_by_packet},
                       {"check", list.ctrlpkt.check},
                       {"rewrite_lanes", list.ctrlpkt.rewrite_lanes}};
  for (const Config &c : list.configs) {
    json e = {{"kind", kind_name(c.kind())}, {"name", c.name}};
    if (!c.app().empty())
      e["app"] = c.app();
    if (!c.prev().empty())
      e["prev"] = c.prev();
    if (c.program_only())
      e["program_only"] = true;
    if (!c.sequence.empty())
      e["sequence"] = c.sequence;
    root["configs"].push_back(e);
  }
  return root.dump(2);
}

} // namespace launch
