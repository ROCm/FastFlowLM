// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file run_vision_e2e.cpp
/// \brief The engine end to end on an IMAGE prompt, against HF: the reference's
///        own pixel_values and prompt ids in, greedy tokens out.
///
///   run_vision_e2e <model_dir> <vision_ref.safetensors> <exec_path> <model_name> [--gen N] [--reps R]
///
/// `vision_ref.safetensors` is models_simple/capture_minicpm_v_vision.py's: it
/// carries the tower's input (`pixel_values`, `target_sizes`), the chat-templated
/// prompt with its placeholders expanded (`e2e.input_ids`), HF's last-row logits
/// and HF fp32's greedy continuation. So this scores the engine -- tower, splice,
/// canvas m-rope, language model -- without the app's preprocessing in the loop.
/// The app's preprocessing is a separate check.
///
/// xclbins are read from <exec_path>/xclbins/<model_name>/, which must hold the
/// text tower's six (Qwen3.5-0.8B's) and the vision tower's three (Qwen3.8's).
///
/// Prints the first divergence from HF's tokens, the last-row logits cosine,
/// and with FLM_MINICPM_V_VISION_PROFILE=1 the vision / prefill split.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "models/minicpm_v_4_7/minicpm_v_4_7_npu.hpp"
#include "image/vision_common.hpp"

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

static int argmax(buffer<bf16>& logits) {
    int best = 0;
    for (size_t i = 1; i < logits.size(); i++)
        if (static_cast<float>(logits[i]) > static_cast<float>(logits[best])) best = (int)i;
    return best;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0] << " <model_dir> <vision_ref.safetensors> <exec_path> <model_name>"
                  << " [--gen N] [--reps R]\n";
        return 2;
    }
    const std::string model_dir = argv[1], ref_path = argv[2];
    int gen = 48, reps = 1;
    for (int i = 5; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--gen") gen = std::atoi(argv[i + 1]);
        else if (a == "--reps") reps = std::atoi(argv[i + 1]);
    }

    LM_Config config;
    config.model_path = model_dir;
    config.exec_path = argv[3];
    config.model_name = argv[4];
    {
        std::ifstream cf(model_dir + "/config.json");
        config._json_config = nlohmann::json::parse(cf);
    }
    // What the app's LM_Config::_normalize_multi_modal() does; this harness
    // reads config.json raw.
    if (config._json_config.value("vision_model_weight", std::string()).empty())
        config._json_config["vision_model_weight"] = "vision_weight.q4nx";
    config._json_config["is_vlm"] = true;

    mcv::ref::safetensors R(ref_path);
    const std::vector<int32_t> ts = R.i32("target_sizes");
    const std::vector<float> px = R.f32("pixel_values");
    const std::vector<int32_t> ids32 = R.i32("e2e.input_ids");
    const std::vector<int32_t> want = R.i32("e2e.generated");
    const std::vector<float> want_logits = R.f32("e2e.logits_last");

    minicpm_v_image_payload_t payload;
    for (size_t i = 0; i + 1 < ts.size(); i += 2) {
        minicpm_v_image_t im{};
        im.grid_h = ts[i];
        im.grid_w = ts[i + 1];
        im.height_resized = im.grid_h * 14;
        im.width_resized = im.grid_w * 14;
        payload.images.push_back(im);
    }
    payload.num_images = (unsigned)payload.images.size();
    payload._data__processed.resize(px.size());
    mcv::simd::f32_to_bf16(payload._data__processed.data(), px.data(), px.size());

    flm_rt::device device(0u);
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);
    auto t0 = std::chrono::steady_clock::now();
    minicpm_v_4_7_npu engine(config, &npu, 4096);
    {
        Q4NX q4nx(model_dir);
        engine.load_weights(q4nx);
    }
    std::cout << "load: " << ms_since(t0) << " ms" << std::endl;

    std::vector<int> ids(ids32.begin(), ids32.end());
    for (int rep = 0; rep < reps; rep++) {
        engine.clear_context();
        auto tp = std::chrono::steady_clock::now();
        buffer<bf16> logits = engine.prefill(ids, &payload);
        const double t_prefill = ms_since(tp);

        std::vector<float> got(want_logits.size());
        for (size_t i = 0; i < got.size(); i++) got[i] = static_cast<float>(logits[i]);
        const mcv::ref::err_stats s = mcv::ref::error_matrix(got.data(), want_logits.data(), got.size());

        std::vector<int> out;
        int tok = argmax(logits);
        for (int i = 0; i < gen; i++) {
            out.push_back(tok);
            if (tok == 248046 || tok == 248044) break;
            buffer<bf16> l = engine.forward(tok);
            tok = argmax(l);
        }
        int first_diff = -1;
        for (size_t i = 0; i < out.size() && i < want.size(); i++)
            if (out[i] != (int)want[i]) { first_diff = (int)i; break; }
        std::printf("rep %d: prefill (vision + %zu tokens) %.0f ms; logits_last cos %.5f relL2 %.4f; "
                    "argmax %d vs HF %d\n", rep, ids.size(), t_prefill, s.cosine, s.rel_l2, out[0], (int)want[0]);
        std::printf("   greedy: %zu tokens, %s", out.size(),
                    first_diff < 0 ? "IDENTICAL to HF fp32 over the compared span\n" : "");
        if (first_diff >= 0) std::printf("first divergence at token %d (engine %d, HF %d)\n", first_diff,
                                         out[(size_t)first_diff], (int)want[(size_t)first_diff]);
        std::printf("   engine ids:");
        for (int t : out) std::printf(" %d", t);
        std::printf("\n");
    }
    return 0;
}
