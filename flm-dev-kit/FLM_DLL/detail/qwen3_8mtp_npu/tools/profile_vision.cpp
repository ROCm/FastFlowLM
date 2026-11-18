// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file tools/profile_vision.cpp
/// \brief The vision tower at its real operating point, timed, and nothing
///        else — the A/B harness for module 4's host half.
///
/// Build -- **at the shipping flags**, not at -O2. The engine's own are
/// `-O3 -march=native -ffast-math` plus the avx512 set; at -O2 this tower
/// measures 2368 ms against the shipping build's 1823, a 30% error that is
/// larger than anything being A/B'd (memory: host-vs-device-ab-needs-shipping-
/// flags). Add `-DQWEN3_8MTP_VISION_PROFILE` for the per-stage table.
///   g++-13 -O3 -march=native -ffast-math -fopenmp -std=c++20
///       -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/profile_vision.cpp image/image_cpu.cpp image/vision_ops.cpp
///       image/vision_glue.cpp image/vision_rope.cpp image/mrope.cpp
///       image/npu/vision_attn_sequence.cpp image/npu/vision_mm_sequence.cpp
///       ../q4nx/q4_npu_eXpress.cpp ../q4nx/safe_tensors.cpp
///       -L../../../../src/lib/xrt -L/opt/xilinx/xrt/lib
///       -lxrt_coreutil -laiebu -luuid -lpthread -o /tmp/profile_vision
/// Run:
///   /tmp/profile_vision [model_dir] [rounds] [L] [vision_mm_build_dir] [attn.xclbin]
///
/// Why this exists rather than a timing arm inside a full check suite: such
/// an arm is the tail of an eight-minute suite that allocates 959 MB of weight BOs several
/// times over, and it fixes the backend ladder. This runs one backend, the
/// shipping one, as many times as asked, so an A/B of the tower's *host* half
/// is one flag apart and takes a minute.
///
/// It scores nothing. Correctness is a separate check (the tower against the
/// HF dump, the matmuls against the host op). The only thing
/// here that has to be real is `L`, and 1200 is the operating point — the
/// reference dump's 256 is 4.7x small and one round of the merger
/// (memory: vision-tower-operating-point-is-l1200).
///
/// The rows are the dump's own patch rows tiled out with a little jitter, the
/// same construction check 10 uses. Patch VALUES do not change which matmuls
/// run over which shapes, and the tower has no data-dependent control flow.
#include "image/image_cpu.hpp"
#include "image/npu/vision_op_factory.hpp"
#include "image/vision_model_def.hpp"
#include "image/vision_ops.hpp"
#include "image/vision_profile.hpp"
#include "image/vision_weights.hpp"

#include "common/cpu_ref_common.hpp"
#include "npu_utils/npu_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

double ms_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

/// The dump's rows, repeated to `want` with N(0, 0.02) jitter so no two rows
/// are bit-identical.
std::vector<float> tile_rows(const std::vector<float>& in, int rows, int cols,
                             int want, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> jitter(0.0f, 0.02f);
    std::vector<float> out((size_t)want * cols);
    for (int r = 0; r < want; r++) {
        const int src = r % rows;
        for (int c = 0; c < cols; c++)
            out[(size_t)r * cols + c] =
                in[(size_t)src * cols + c] + jitter(rng);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 3;
    const int L_want = argc > 3 ? std::atoi(argv[3]) : 1200;

    const std::string mm_root =
        argc > 4 ? std::string(argv[4]) + "/"
                 : "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/vision_mm/"
                   "build/";
    const std::string xclbin_a   = mm_root + "VISION_MM_A/xclbins/mm.xclbin";
    const std::string xclbin_b   = mm_root + "VISION_MM_B/xclbins/mm.xclbin";
    const std::string xclbin_attn =
        argc > 5 ? argv[5]
                 : "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/"
                   "vision_attn/build/xclbins/attn.xclbin";

    LM_Config config;
    config.model_path = model_dir;
    config.model_name = "Qwen3.8-27B-NPU2";
    config.exec_path  = ".";
    {
        std::ifstream f(model_dir + "/config.json");
        if (!f) { std::fprintf(stderr, "cannot read config.json\n"); return 1; }
        f >> config._json_config;
    }

    qwen3_8mtp_vision_desc desc;
    desc.build(config);
    if (!desc.present) {
        std::fprintf(stderr, "config.json has no vision_config\n");
        return 1;
    }

    // Constructing `weights` corrects D_ff from the config's 4304 to the
    // shipped 4608, so nothing may size a buffer off desc before this.
    qwen38::vision::weights w(desc);

    cpu_ref::safetensors ref(model_dir + "/vision_ref.safetensors");
    const std::vector<int32_t> thw = ref.i32("image_grid_thw");
    const int L_dump = desc.patch_count(thw[0], thw[1], thw[2]);
    const int PD     = (int)desc.patch_dim();
    const int OH     = (int)desc.out_hidden_size;
    const int MU     = (int)desc.merge_unit();

    // A grid whose patch count is L_want. 30x40 = 1200 is panda.png at
    // smart_resize factor 32; anything else is expressed as a 1-row grid.
    qwen38::image_grid g;
    if (L_want == 1200) { g.t = 1; g.h = 30; g.w = 40; }
    else {
        const int m = (int)desc.spatial_merge_size;
        if (L_want % (m * m)) {
            std::fprintf(stderr, "L must be a multiple of %d\n", m * m);
            return 1;
        }
        g.t = 1; g.h = m; g.w = L_want / m;
    }
    const int L    = desc.patch_count(g.t, g.h, g.w);
    const int rows = L / MU;

    std::printf("== vision tower, L=%d (%dx%dx%d), %d merged tokens ==\n",
                L, g.t, g.h, g.w, rows);
    std::printf("   model %s\n   D %u  D_ff %u  patch_dim %d  heads %u\n",
                model_dir.c_str(), desc.D, desc.D_ff, PD, desc.num_heads);
    std::printf("   host threading: %s\n",
                std::getenv("OMP_WAIT_POLICY")
                    ? std::getenv("OMP_WAIT_POLICY") : "default (spin)");

    const std::vector<float> pv = ref.f32("pixel_values");
    std::vector<float> px = tile_rows(pv, L_dump, PD, L, 11);

    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* mgr_a    = mgr.register_xclbin(xclbin_a);
    npu_app_manager* mgr_b    = mgr.register_xclbin(xclbin_b);
    npu_app_manager* attn_mgr = mgr.register_xclbin(xclbin_attn);
    const q38npu::vision_mm_rtp_addresses rtp;

    auto f = std::make_unique<q38npu::vision_op_factory>(attn_mgr);
    f->enable_mm(mgr_a, mgr_b, rtp);
    const std::string backend = f->name();

    auto t_load = std::chrono::steady_clock::now();
    qwen38::image_cpu tower(desc, std::move(f));
    const double load_ms = ms_since(t_load);
    std::printf("   backend \"%s\", load %.0f ms\n\n", backend.c_str(), load_ms);

    std::vector<float> out((size_t)rows * OH, 0.0f);
    tower.encode(px.data(), g, out.data());           // warm: BOs grow here

    double best = 1e30, sum = 0;
    for (int i = 0; i < rounds; i++) {
        qwen38::vision::prof::reset();
        auto t0 = std::chrono::steady_clock::now();
        tower.encode(px.data(), g, out.data());
        const double t = ms_since(t0);
        best = std::min(best, t);
        sum += t;
        std::printf("   round %d  encode %8.1f ms  (%.2f ms/merged token)\n",
                    i, t, t / (double)rows);
        // Built without VISION_PROFILE this is an empty function on zeroed
        // counters, so the harness times the shipping code either way; the
        // table appears only under `make VISION_PROFILE=1`'s -D.
        qwen38::vision::prof::report(t);
    }
    std::printf("\n   best %.1f ms   mean %.1f ms\n", best, sum / rounds);
    return 0;
}
