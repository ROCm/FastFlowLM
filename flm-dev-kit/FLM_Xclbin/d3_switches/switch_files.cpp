// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// switch_files.cpp : write the aie2p-launchlib switches of a package for the d3 engine (no NPU, no XRT).
//   switch_files --package DIR --out DIR --apps mm,attn[,conv]
// Writes OUT/d3.switches: per kernel its instruction stream (the engine turns it into an ELF with aiebu) and
// control packets (XRT argument 3 + packet_arg). Kernels: cold_<first app>, and <a>_then_<b> for every ordered pair.
#include "launch/launch.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string &s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  for (std::string t; std::getline(ss, t, ',');) v.push_back(t);
  return v;
}

int main(int argc, char **argv) {
  std::string package, out, apps_arg = "mm,attn";
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string o = argv[i];
    if (o == "--package") package = argv[i + 1];
    else if (o == "--out") out = argv[i + 1];
    else if (o == "--apps") apps_arg = argv[i + 1];
  }
  try {
    const auto apps = split(apps_arg);
    std::vector<launch::Config> cfg{launch::cold(apps[0])};
    for (auto &a : apps)
      for (auto &b : apps)
        if (a != b) cfg.push_back(launch::warm(a, b));
    launch::ConfigList list{package, launch::Transport::ctrlpkt, cfg};
    const auto none = [](launch::Rt &) {};
    launch::SequenceMap seqs;
    for (auto &a : apps) seqs[a] = none;
    const launch::Result r = launch::build(list, seqs);
    // d3.switches: "D3SW1\n", per kernel "name packet_arg txn_bytes pkt_bytes\n", "END\n", then the blobs in
    // that order (txn, pkt, ...); read by FLM_DLL/detail/d3_npu/d3_launch.hpp (load_file). packet_arg: XRT
    // argument 3 + packet_arg takes the packet buffer.
    std::string head = "D3SW1\n";
    for (const launch::Kernel &k : r.kernels) {
      head += k.name + " " + std::to_string(k.packet_arg) + " " + std::to_string(k.words.size() * 4) + " " +
              std::to_string(k.packets.size() * 4) + "\n";
      std::printf("%-18s txn %6zu B packets %6zu B packet_arg %d\n", k.name.c_str(), k.words.size() * 4,
                  k.packets.size() * 4, k.packet_arg);
    }
    head += "END\n";
    std::ofstream f(out + "/d3.switches", std::ios::binary);
    f.write(head.data(), head.size());
    for (const launch::Kernel &k : r.kernels) {
      f.write((const char *)k.words.data(), k.words.size() * 4);
      f.write((const char *)k.packets.data(), k.packets.size() * 4);
    }
    f.close();
    if (!f) throw std::runtime_error("cannot write " + out + "/d3.switches");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
