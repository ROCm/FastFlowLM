//===- lane_report.cpp - the control-packet lanes of packages -------------===//
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
// Prints, per app of each package folder, the lane the control-packet
// transport plans in each column (transport/ctrlpkt_lane.hpp): channel, ports,
// cost, residents and toggles, and the columns that stay on TXN writes.
//
//   build/lane_report <package_dir>...     (every app folder in each)
//
//===----------------------------------------------------------------------===//

#include "launch/formats/package.hpp"
#include "launch/transport/ctrlpkt_lane.hpp"

#include <filesystem>
#include <iostream>

using namespace launch;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: lane_report <package_dir>...\n";
    return 2;
  }
  int bad = 0;
  for (int i = 1; i < argc; ++i) {
    std::vector<std::string> apps;
    for (const auto &e : std::filesystem::directory_iterator(argv[i]))
      if (e.is_directory() && std::filesystem::exists(e.path() / "design.xclbin"))
        apps.push_back(e.path().filename().string());
    std::sort(apps.begin(), apps.end());
    for (const std::string &app : apps) {
      const Package p = load_package(argv[i], app);
      const ctrlpkt::Plan plan = ctrlpkt::plan(p);
      int residents = 0, toggles = 0;
      for (const auto &[t, m] : plan.final_map)
        for (const auto &[off, v] : m)
          if (!p.tiles.count(t) || !p.tiles.at(t).routes.count(off))
            ++residents;
      for (const auto &[t, m] : plan.toggles)
        toggles += (int)m.size();
      std::cout << std::format("{}/{}: {} lanes, {} TXN columns, {} residents, {} toggles{}\n", argv[i], app,
                               plan.lanes.size(), plan.txn_cols.size(), residents, toggles,
                               plan.shim_only_toggles ? "" : " (some above row 0: starts by TXN)");
      for (const ctrlpkt::Lane &l : plan.lanes) {
        std::string north;
        for (int j : l.north)
          north += std::format(" {}", j);
        std::cout << std::format("  col {}: MM2S {} NORTH {} up{} top {} arb", l.col, l.ch, l.k,
                                 north.empty() ? "" : north, l.top);
        for (const auto &[row, a] : l.arb)
          std::cout << std::format(" {}", a);
        std::cout << std::format(" cost {}{}\n", l.cost, l.mux_resident ? " (mux resident)" : "");
      }
      for (const auto &[col, why] : plan.txn_cols) {
        std::cout << std::format("  col {}: TXN ({})\n", col, why);
        ++bad;
      }
      for (const auto &[t, m] : plan.toggles)
        for (const auto &[off, v] : m)
          std::cout << std::format("  toggle {} {:#x}: app {:#010x} lane {:#010x}\n", t.describe(), off, v,
                                   plan.load_map.at(t).at(off));
    }
  }
  return bad ? 1 : 0;
}
