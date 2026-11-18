// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file tools/profile_prefill.cpp
/// \brief Drive module 1 at a chosen window length and print its per-stage
///        wall clock. Build the .so with `make PREFILL_PROFILE=1` first --
///        without it every tick compiles to `do {} while (0)` and this prints
///        nothing but the totals.
///
/// Build (one line; -lq4_npu_eXpress is required, as for verify_qwen3_8mtp):
///   g++-13 -O2 -march=native -fopenmp -std=c++20
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/profile_prefill.cpp
///       -L../../build/lib -lqwen3_8mtp_npu -lq4_npu_eXpress
///       -L/opt/xilinx/xrt/lib -lxrt_coreutil
///       -Wl,-rpath,$PWD/../../build/lib -o /tmp/profile_prefill
///
/// Run:
///   /tmp/profile_prefill [model_dir] [exec_path] [L] [rounds]
///
/// ---------------------------------------------------------------------------
/// Why this exists instead of `verify_qwen3_8mtp`
///
/// That driver's window is pinned to the reference dump at **L = 221**, which
/// is below `dequant_mm_op::ROWS` (512). Every projection therefore runs as a
/// single partly-padded tile and the host staging loops never see a full one,
/// so the shape of the profile at 221 is not the shape at a real prompt. It
/// also runs exactly one window, so `_chunked_insert`'s second round -- where
/// the attention core reads a non-zero `L_begin` and the delta-net resumes a
/// carried state -- is never profiled at all.
///
/// The ids here are **synthetic**, and that is deliberate rather than a
/// shortcut. Token identity changes nothing about the work: every layer runs
/// the same matmuls over the same shapes whatever the ids decode to. What the
/// ids would change is the *output*, and this driver scores nothing -- it is a
/// stopwatch. Correctness lives in verify_qwen3_8mtp and the verify_n* suite.
/// The one property that does have to be real is the window length, and that
/// is the argument.
///
/// The window is pure text: no payload, so the engine emits ctx..ctx+L-1 on
/// all three m-rope channels itself. A multimodal window would only change
/// which positions the RoPE table holds, not how long any stage takes.
#include "models/qwen3_8mtp/qwen3_8mtp_npu.hpp"
#include "lm_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <chrono>

int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string exec_path =
        argc > 2 ? argv[2] : "/path/to/FastFlowLM/src";
    const int L      = argc > 3 ? std::atoi(argv[3]) : 1024;
    const int rounds = argc > 4 ? std::atoi(argv[4]) : 2;

    std::printf("== module 1 profile: L = %d, %d round(s) ==\n", L, rounds);
    std::printf("   model %s\n   xclbins under %s\n\n",
                model_dir.c_str(), exec_path.c_str());

    LM_Config config;
    config.model_path = model_dir;
    config.model_name = "Qwen3.8-27B-NPU2";
    config.exec_path  = exec_path;
    {
        std::ifstream f(model_dir + "/config.json");
        if (!f) { std::fprintf(stderr, "cannot read config.json\n"); return 1; }
        f >> config._json_config;
    }

    xrt::device        npu_dev(0);
    npu_xclbin_manager npu_mgr(npu_device::device_npu2, &npu_dev, false);

    // MAX_L has to cover every round's cumulative context, not one window.
    const int max_l = (L * rounds < 4096) ? 4096 : L * rounds + 512;
    qwen3_8mtp_npu engine(config, &npu_mgr, max_l);
    {
        Q4NX q4nx(model_dir);
        engine.load_weights(q4nx);
    }
    std::printf("weights loaded\n\n");

    // Low ids so the embedding lookup cannot fault on a vocab edge, and no
    // special tokens: an id list is a length here, not a prompt.
    std::vector<int> ids((size_t)L);
    for (int i = 0; i < L; i++) ids[(size_t)i] = 1000 + (i % 4096);

    for (int r = 0; r < rounds; r++) {
        std::printf("\n======== round %d (context before: %d) ========\n",
                    r, r * L);
        const auto t0 = std::chrono::high_resolution_clock::now();
        engine.prefill(ids, nullptr);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("round %d wall: %.1f ms  (%.3f ms/token)\n",
                    r, ms, ms / (double)L);
    }
    return 0;
}
