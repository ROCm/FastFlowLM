// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file bench_verify_lm_head.cpp
/// \brief Phase-level timing for module 3: where does the verify phase go?
///
/// Build:
///   g++-13 -O3 -march=native -fopenmp -std=c++20
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/bench_verify_lm_head.cpp ../q4nx/safe_tensors.cpp
///       -L../../build/lib -lqwen3_8mtp_npu -lq4_npu_eXpress -lxrt_coreutil
///       -Wl,-rpath,<abs>/FLM_DLL/build/lib -o /tmp/bench_verify_lm_head
/// Run:
///   ./bench_verify_lm_head <model_dir> <exec_path>
///
/// ---------------------------------------------------------------------------
/// This was an A/B, and is not one any more
/// ---------------------------------------------------------------------------
/// It existed to answer "does routing module 3's projection through
/// `lm_head_8.xclbin` make the verify phase faster?", switched by
/// `FLM_Q38_LM_HEAD_8=0`. That switch is gone: `lm_head_8.xclbin` is required
/// and `verify_cpu` throws without it, so there is no second arm left to run.
/// The answer it reached is recorded rather than re-measurable -- N11 put the
/// projection at 29 ms on the device against 61 ms on the host, and that ~32 ms
/// never rose above the phase counter's own spread.
///
/// What survives is the driver: this is the only tool that puts the ENGINE
/// through 20 real speculation cycles over a real prompt and prints
/// `speculation_timing()`'s per-phase counters. Use it to see where the phase
/// budget sits, not to compare two builds of it -- a speculation cycle is
/// 3.5 .. 3.9 s and repeats within 9% of itself, which is +-320 ms, so
/// anything smaller than that is noise here.
///
/// ---------------------------------------------------------------------------
/// The prompt has to be real
/// ---------------------------------------------------------------------------
/// A made-up id list decodes to a degenerate repeat, which drafts at an
/// acceptance rate no real text reaches, and the cycle count then says nothing
/// about production. This drives the checkpoint's own prefill window and lets
/// the context genuinely grow.
#include "models/qwen3_8mtp/qwen3_8mtp_npu.hpp"
#include "common/cpu_ref_common.hpp"
#include "lm_config.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";

    LM_Config config;
    config.model_path = model_dir;
    config.model_name = "Qwen3.8-27B-NPU2";
    config.exec_path  = argc > 2 ? argv[2] : ".";

    std::printf("== module 3 phase timing -- the projection is on the DEVICE ==\n");

    {
        std::ifstream f(model_dir + "/config.json");
        if (!f) { std::fprintf(stderr, "cannot read config.json\n"); return 1; }
        f >> config._json_config;
    }

    xrt::device        npu_dev(0);
    npu_xclbin_manager npu_mgr(npu_device::device_npu2, &npu_dev, false);
    qwen3_8mtp_npu     engine(config, &npu_mgr, 4096);

    // Q4NX takes the model directory; load_weights mmaps model.q4nx itself.
    Q4NX q4nx(model_dir);
    engine.load_weights(q4nx);

    if (!engine.has_mtp_head()) {
        std::fprintf(stderr, "no mtp head -- nothing to bench\n");
        return 1;
    }

    // The prefill window the checkpoint was dumped over: real text, real
    // length, so the draft acceptance rate below is the production one.
    cpu_ref::safetensors ref(model_dir + "/ref.safetensors");
    std::vector<int>     pre_ids;
    {
        const std::vector<int32_t> ids = ref.i32("prefill.input_ids");
        pre_ids.assign(ids.begin(), ids.end());
    }

    engine.clear_context();
    buffer<bf16> y = engine.prefill(pre_ids, nullptr);
    int last = 0;
    for (uint32_t i = 1; i < y.size(); i++)
        if ((float)y[i] > (float)y[last]) last = (int)i;

    // Enough cycles that the verify counter is a mean over many dispatches
    // rather than one. Each is a real grow: the context is longer every time.
    const int CYCLES = 20;
    int       produced = 0, cycles_run = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < CYCLES; c++) {
        const std::vector<int> acc = engine.speculate(last, 7);
        if (acc.empty()) { std::printf("  cycle %d declined\n", c); break; }
        produced += (int)acc.size();
        cycles_run++;
        last = acc.back();
    }
    const double wall_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0).count();

    std::printf("\n%s\n", engine.speculation_timing().c_str());
    std::printf("  %d cycles, %d tokens, context now %d\n", cycles_run, produced,
                engine.get_current_context_length());
    std::printf("  wall %.0f ms total, %.1f ms/cycle\n", wall_ms,
                cycles_run ? wall_ms / cycles_run : 0.0);
    std::printf("\n  The verify line is the batched k+1 pass, replay excluded.\n"
                "  The cycle and wall figures are context only -- at 9%% run-to-run\n"
                "  spread they cannot resolve anything under ~320 ms.\n");
    return 0;
}
