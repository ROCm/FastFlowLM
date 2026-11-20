// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file host.cpp
/// \brief Drive the Qwen3.8-27B gated-delta-net conv1d prefill on the NPU and
///        diff it against the fp32 CPU reference in cpu_ref.hpp.
///
/// Input is random (fixed seed) rather than a captured safetensors dump: this
/// stage is a depthwise conv plus a per-head norm, both pointwise in the token
/// axis, so real activations exercise nothing a Gaussian does not -- and a
/// seeded generator lets the sweep run at any L without a matching capture.
///
/// DDR layout, [L][CONV_DIM] bf16 token-major:
///
///     row 0 .. CONV_K-2   conv history (zero: start of sequence)
///     row CONV_K-1 ..     the L_wanted real tokens
///
/// Output token t is produced from input rows t .. t+CONV_K-1, so o[0]
/// corresponds to the first real token and the output is L_wanted rows.
#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include "utils/debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "lm_config.hpp"
#include <math.h>
#include "metrices.hpp"
#include "cpu_ref.hpp"
#include "seq_gen.hpp"

namespace po = boost::program_options;

int main(int argc, char **argv) {
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    desc.add_options()("Seed,s", po::value<unsigned int>()->default_value(32), "Random seed");
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(1120), "Sequence length");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(100), "Iterations");
    arg_utils::parse_options(argc, argv, desc, vm);

    const int NT = vm["Iteration"].as<int>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();
    const int L_wanted = vm["L_wanted"].as<int>();

    constexpr int D      = cpu_ref::q38::CONV_DIM;   // 10240
    constexpr int CONV_K = cpu_ref::q38::CONV_K;     // 4
    constexpr int DQ     = cpu_ref::q38::DQ;         // 2048

    // The 3 leading history rows are what make output row t read input rows
    // t..t+3; they stay zero here because this is the start of a sequence.
    const int L_in = L_wanted + CONV_K - 1;

    const size_t X_VOLUME = static_cast<size_t>(L_in) * D;
    const size_t W_VOLUME = static_cast<size_t>(CONV_K) * D;
    const size_t O_VOLUME = static_cast<size_t>(L_wanted) * D;

    header_print("info", "L = " << L_wanted << ", D = " << D
                  << ", conv_k = " << CONV_K << ", seed = " << SEED);

    // ---- NPU ------------------------------------------------------------
    xrt::device npu_device = xrt::device(0);
    std::string xclbin_name = "build/xclbins/conv.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* app_mgr = xclbin_mgr.register_xclbin(xclbin_name);
    npu_app app_0 = app_mgr->create_app();

    buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(O_VOLUME);
    buffer<bf16> x_0 = app_0.create_bo_buffer<bf16>(X_VOLUME);
    buffer<bf16> w_0 = app_0.create_bo_buffer<bf16>(W_VOLUME);

    // ---- random input, generated straight into the device buffers so the
    // CPU reference reads exactly the bf16 bits the NPU gets ---------------
    std::mt19937 rng(SEED);
    x_0.memset((bf16)0.0f);   // leaves the CONV_K-1 history rows zeroed
    cpu_ref::fill_normal(x_0.data() + static_cast<size_t>(CONV_K - 1) * D,
                         static_cast<size_t>(L_wanted) * D, 1.0f, rng);
    // Conv taps are small in a trained model; 0.5 keeps the pre-SiLU sum in a
    // range where bf16 is not the dominant error term.
    cpu_ref::fill_normal(w_0.data(), W_VOLUME, 0.5f, rng);

    x_0.sync_to_device();
    w_0.sync_to_device();
    o_0.memset((bf16)(-1.0f));
    o_0.sync_to_device();

    // ---- CPU reference ---------------------------------------------------
    header_print("info", "Running CPU reference.");
    time_utils::time_point cpu_start = time_utils::now();
    std::vector<float> y_ref_f32(O_VOLUME);
    cpu_ref::conv1d_silu_l2norm(x_0.data(), w_0.data(), L_wanted, D, y_ref_f32.data());
    time_utils::time_point cpu_stop = time_utils::now();
    header_print("info", "CPU reference took "
                  << time_utils::duration_us(cpu_start, cpu_stop).first / 1000.0 << " ms");

    buffer<bf16> y_ref(O_VOLUME);
    for (size_t i = 0; i < O_VOLUME; i++) y_ref.data()[i] = (bf16)y_ref_f32[i];

    // ---- run -------------------------------------------------------------
    // L_buffer's address on a core; from build/mlir/conv.mlir after
    // --aie-assign-buffer-addresses (all 32 cores get the same address).
    constexpr uint32_t L_BUFFER_ADDRESS = 49664;
    gen_seq(app_0.seq(), L_in, D, L_wanted, DQ, L_BUFFER_ADDRESS, 0);

    time_utils::time_with_unit npu_time = {0.0, "us"};
    header_print("info", "Running kernel with bare call (direct based).");
    for (int i = 0; i < NT; i++) {
        time_utils::time_point start = time_utils::now();
        app_0(o_0, x_0, w_0);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= NT;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    o_0.sync_from_device();

    // ---- compare ---------------------------------------------------------
    header_print("info", "CPU reference (first rows/cols)");
    utils::print_matrix(y_ref, D);
    header_print("info", "NPU output (first rows/cols)");
    utils::print_matrix(o_0, D);

    error_metrics metrics = get_error_metrics(o_0, y_ref);
    print_error_metrics(metrics, "Overall (q | k | v)");

    // Per-region, because the three have different failure modes: q and k go
    // through l2norm (and q through the extra 1/sqrt(d)), v does not.  A
    // mis-assigned norm mode shows up here and nowhere else.
    struct region { const char* name; int base; int width; };
    const region regions[3] = {
        {"q (l2norm * 1/sqrt(d))", 0,      DQ},
        {"k (l2norm)",             DQ,     DQ},
        {"v (conv + silu only)",   2 * DQ, D - 2 * DQ},
    };
    for (const region& r : regions) {
        buffer<bf16> got(static_cast<size_t>(L_wanted) * r.width);
        buffer<bf16> want(static_cast<size_t>(L_wanted) * r.width);
        for (int t = 0; t < L_wanted; t++) {
            memcpy(got.data()  + static_cast<size_t>(t) * r.width,
                   o_0.data()  + static_cast<size_t>(t) * D + r.base,
                   r.width * sizeof(bf16));
            memcpy(want.data() + static_cast<size_t>(t) * r.width,
                   y_ref.data()+ static_cast<size_t>(t) * D + r.base,
                   r.width * sizeof(bf16));
        }
        error_metrics m = get_error_metrics(got, want);
        print_error_metrics(m, r.name);
    }

    header_print("info", "NPU time: " << npu_time.first << " us");
    return 0;
}
