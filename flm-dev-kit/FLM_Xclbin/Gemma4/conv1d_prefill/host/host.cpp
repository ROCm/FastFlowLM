// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include "utils/debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "lm_config.hpp"
#include <math.h>
#include "metrices.hpp"
#include "seq_gen.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include "tensor_utils/safe_tensors.hpp"

namespace po = boost::program_options;
int round_up_to_multiple(int x, int multiple){
    if (multiple == 0) {
        return x; // Cannot divide by zero
    }

    return ((x + multiple - 1) / multiple) * multiple;
}
int main(int argc, char **argv) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("Seed,s", po::value<unsigned int>()->default_value(32), "Random seed");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(2048), "L_wanted");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "L_begin");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(100), "Iteration");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();

    // load reference data
    std::string reference_path = "../conv.safetensors";
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
    buffer<bf16> x_in;
    references->load_weights(x_in, "conv_in");

    buffer<bf16> w_buffer;
    references->load_weights(w_buffer, "conv_weight");

    buffer<bf16> conv_out;
    references->load_weights(conv_out, "conv_out");

    const int L_wanted = vm["L_wanted"].as<int>();
    const int L_begin = vm["L_begin"].as<int>();

    int L_end = L_begin + L_wanted;
    const size_t CONV_PAD_SIZE = 4;        
    const size_t L = L_end + CONV_PAD_SIZE;  // For simulation L is the actual, +2 from cache/padding
    const size_t L_PAD_REQUIREMENT =1;
    const size_t CONV_KERNEL_SIZE = 5;
    const size_t L_padded =  round_up_to_multiple(L, L_PAD_REQUIREMENT);
    const size_t  L_out = L_end;
    int D = 1024;
    std::cout << "L_padded: " << L_padded << ", L_out: " << L_out << std::endl;

    const int X_VOLUME = L_padded * D; // q, k, v, alpha, beta
    const int W_VOLUME = 5 * D;
    const int O_VOLUME = L_out * D;

    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/conv.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* attn_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    npu_app app_0 = attn_app_mgr->create_app();

    buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(O_VOLUME);
    buffer<bf16> x_0 = app_0.create_bo_buffer<bf16>(X_VOLUME);
    buffer<bf16> w_0 = app_0.create_bo_buffer<bf16>(W_VOLUME);

    x_0.memset((bf16)0.0f);
    memcpy(x_0.data() + 4 * D, x_in.data(), L_wanted * D * sizeof(bf16));
    utils::print_matrix(x_0, D);
    memcpy(w_0.data(), w_buffer.data(), W_VOLUME * sizeof(bf16));
    header_print("info", "weight");
    utils::print_matrix(w_0, D);
    x_0.sync_to_device();
    w_0.sync_to_device();
    o_0.memset((bf16)(-1.0f));
    o_0.sync_to_device();

    bf16 max_value = (bf16)5.0f;
    bf16 min_value = (bf16)-5.0f;
    time_utils::time_with_unit npu_time = {0.0, "us"};
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    gen_seq(app_0.seq(), L_out, min_value, max_value, 0, 0, D);

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
    buffer<bf16> o_wanted(L_wanted * D);
    o_wanted.copy_from(o_0.data(), L_wanted * D);
    buffer<bf16> aux_out(L_wanted * D);
    aux_out.copy_from(conv_out.data(), L_wanted * D);
    
    header_print("info", "Verifying the result with reference data.");
    utils::print_matrix(o_wanted, D);
    header_print("info", "Reference output");
    utils::print_matrix(aux_out, D);

    error_metrics metrics = get_error_metrics(o_wanted, aux_out);
    print_error_metrics(metrics);

    header_print("info", "Finished running kernel");
    header_print("info", "Running kernel with bare call (run based).");
    header_print("info", "NPU time: " << npu_time.first << " us");
    return 0;
}
