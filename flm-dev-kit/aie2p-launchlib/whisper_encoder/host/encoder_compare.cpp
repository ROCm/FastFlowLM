//===- encoder_compare.cpp - run the Whisper engine's encoder, dump, time --===//
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
//   encoder_compare.exe --model DIR --xroot DIR --out FILE [--windows N] [--seed S]
//
// Step 2 of the Whisper trial. Loads FLM's Whisper engine (libwhisper_npu.so
// of this tree) as FastFlowLM would, encodes the same 30 s window N times and
// writes the encoder's output of the last one to FILE: the cross-attention K/V
// it hands to the decoder (bf16, every decoder layer). Run it once without and
// once with FLM_WHISPER_SWITCH_DIR, then compare the files (cmp): the kernels
// and the data are the same, only the switches differ.
//
// The window is a fixed random log-mel spectrogram (seed S): enough to compare
// two paths bit for bit. Prints the ms of each encode; the first one also fills
// the dequant cache.
//
//   --model DIR   the model folder (config.json, model.q4nx)
//   --xroot DIR   holds xclbins/<model folder name>/*.xclbin (tools/build_xroot.sh)
//
//===----------------------------------------------------------------------===//

#include "lm_config.hpp"
#include "models/whisper/whisper_npu.hpp"
#include "npu_utils/flm_runtime.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

static double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0); // nothing lost when a hang ends in a kill
  std::string model, xroot, out;
  int windows = 3;
  unsigned seed = 1;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string opt = argv[i];
    if (opt == "--model") model = argv[i + 1];
    else if (opt == "--xroot") xroot = argv[i + 1];
    else if (opt == "--out") out = argv[i + 1];
    else if (opt == "--windows") windows = std::stoi(argv[i + 1]);
    else if (opt == "--seed") seed = (unsigned)std::stoul(argv[i + 1]);
  }
  if (model.empty() || xroot.empty() || out.empty() || windows < 1) {
    std::fprintf(stderr, "usage: encoder_compare.exe --model DIR --xroot DIR --out FILE [--windows N] [--seed S]\n");
    return 2;
  }
  const char *sw = std::getenv("FLM_WHISPER_SWITCH_DIR");
  std::printf("switch: %s\n", sw && *sw ? sw : "off (one hw_context per xclbin, as FLM)");

  // LM_Config::from_pretrained lives in FastFlowLM, not in FLM_DLL: fill the fields it would set
  // (as FLM_DLL/detail/d3_npu/tools/d3_engine_run.cpp).
  Whisper_Config cfg;
  cfg.model_path = model;
  cfg.model_name = std::filesystem::path(model).filename().string();
  cfg.exec_path = xroot;
  std::ifstream cf(std::filesystem::path(model) / "config.json");
  cfg._json_config = nlohmann::json::parse(cf);
  // What FastFlowLM's Whisper_Config::from_pretrained does (FastFlowLM/src/include/lm_config.hpp):
  // canonical key names, and the vocabulary rounded up to a multiple of 32 (the model file's embedding).
  nlohmann::json &jc = cfg._json_config;
  jc["hidden_size"] = jc.value("d_model", 0);
  jc["intermediate_size"] = jc.value("decoder_ffn_dim", 0);
  jc["vocab_size"] = (jc.value("vocab_size", 0) + 31) / 32 * 32;
  jc["is_vlm"] = false;
  jc["is_audio"] = false;
  cfg.flm_version = cfg._json_config.value("flm_version", std::string("0.0.0"));

  flm_rt::device device(0);
  npu_xclbin_manager npu(npu_device::device_npu2, &device, false);
  auto t = std::chrono::steady_clock::now();
  Q4NX q4nx(model);
  whisper_npu engine(cfg, &npu);
  engine.load_weights(q4nx);
  std::printf("load %.0f ms\n", ms_since(t));

  // One 30 s window of log-mel features: 128 bins x 3000 frames, values like Whisper's ([-1, 1.5]).
  constexpr size_t kMel = 128 * 3000;
  buffer<bf16> mel(kMel);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dis(-1.0f, 1.5f);
  for (size_t i = 0; i < kMel; i++)
    mel[i] = bf16(dis(rng));

  std::vector<bf16> first;
  for (int w = 0; w < windows; w++) {
    t = std::chrono::steady_clock::now();
    buffer<bf16> kv = engine.encode_audio(mel);
    std::printf("encode %d: %.1f ms\n", w, ms_since(t));
    if (w == 0)
      first.assign(kv.data(), kv.data() + kv.size());
    if (w == windows - 1) {
      std::ofstream(out, std::ios::binary).write(reinterpret_cast<const char *>(kv.data()), kv.size() * sizeof(bf16));
      const bool same = first.size() == kv.size() && std::memcmp(first.data(), kv.data(), kv.size() * sizeof(bf16)) == 0;
      std::printf("wrote %s (%zu bf16); last window %s the first\n", out.c_str(), kv.size(),
                  same ? "equals" : "DIFFERS FROM");
    }
  }
  return 0;
}
