// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// switch_files.cpp : write the aie2p-launchlib switches of the SmolVLA vision package (no NPU, no XRT).
//   switch_files --package DIR --out DIR
// Writes OUT/vision.switches with four kernels, to_attn32 / to_mm8 (instruction streams that load the other
// design themselves) and cp_to_attn32 / cp_to_mm8 (the same switches as control packets: the instruction
// stream only starts the packet DMA, the packets go in the run that carries it).
#include "launch/launch.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  std::string package, out;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string o = argv[i];
    if (o == "--package") package = argv[i + 1];
    else if (o == "--out") out = argv[i + 1];
  }
  if (package.empty() || out.empty()) {
    std::cerr << "usage: switch_files --package DIR --out DIR\n";
    return 2;
  }
  try {
    const auto none = [](launch::Rt &) {};
    const launch::SequenceMap seqs{{"mm8", none}, {"attn32", none}};
    std::vector<launch::Kernel> kernels;
    for (const bool cp : {false, true}) {
      const std::string pre = cp ? "cp_" : "";
      launch::ConfigList list{package, cp ? launch::Transport::ctrlpkt : launch::Transport::txn,
                              {launch::warm("mm8", "attn32", {}, pre + "to_attn32"),
                               launch::warm("attn32", "mm8", {}, pre + "to_mm8")}};
      for (const launch::Kernel &k : launch::build(list, seqs).kernels) kernels.push_back(k);
    }
    // vision.switches: "SVSW1\n", per kernel "name packet_arg txn_bytes pkt_bytes\n", "END\n", then the blobs in
    // that order (txn, pkt, ...); read by FLM_Pybind/SmolVLA/src/SmolVLA/vlm/vision/seq_gen.hpp
    // (vision_switches). packet_arg: XRT argument 3 + packet_arg takes the packet buffer.
    std::string head = "SVSW1\n";
    for (const launch::Kernel &k : kernels) {
      head += k.name + " " + std::to_string(k.packet_arg) + " " + std::to_string(k.words.size() * 4) + " " +
              std::to_string(k.packets.size() * 4) + "\n";
      std::printf("%-14s txn %6zu B packets %6zu B packet_arg %d\n", k.name.c_str(), k.words.size() * 4,
                  k.packets.size() * 4, k.packet_arg);
    }
    head += "END\n";
    std::ofstream f(out + "/vision.switches", std::ios::binary);
    f.write(head.data(), head.size());
    for (const launch::Kernel &k : kernels) {
      f.write((const char *)k.words.data(), k.words.size() * 4);
      f.write((const char *)k.packets.data(), k.packets.size() * 4);
    }
    f.close();
    if (!f) throw std::runtime_error("cannot write " + out + "/vision.switches");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
