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
void exp_func_2(int dim, buffer<bf16>& x, buffer<float>& o) {
    for (int i = 0; i < dim; i++) {
        o[i] = exp2(x[i]);
    }
}

void exp_func(int dim, buffer<bf16>& x, buffer<float>& o) {
    for (int i = 0; i < dim; i++) {
        o[i] = exp(x[i]);
    }
}

void Gelu(int dim, buffer<bf16>& x, buffer<float>& o) {
    for (int i = 0; i < dim; i++) {
        o[i] = x[i] * (0.5 * (1 + tanh(sqrt(2 / M_PI) * (float(x[i]) + 0.044715 * pow(x[i], 3)))));
    }
}

void Silu(int dim, buffer<bf16>& x, buffer<float>& o) {
    for (int i = 0; i < dim; i++) {
        float xi = static_cast<float>(x[i]);
        o[i] = xi / (1.0f + std::exp(-xi));
    }
}

void Sigmoid(int dim, buffer<bf16>& x, buffer<float>& o) {
    for (int i = 0; i < dim; i++) {
        float xi = static_cast<float>(x[i]);  // promote to float for math
        float yi = 1.0f / (1.0f + std::exp(-xi));
        o[i] = static_cast<bf16>(yi);
    }
}

void softplus(int dim, buffer<bf16>& x, buffer<float>& o) {
    for (int i = 0; i < dim; i++) {
        float xi = static_cast<float>(x[i]);  // promote to float for math
        float yi = log(1.0f + std::exp(xi));
        o[i] = static_cast<bf16>(yi);
    }
}


void Tanh(int dim, buffer<bf16>& x, buffer<float>& o) {
    for (int i = 0; i < dim; i++) {
        float xi = static_cast<float>(x[i]);  // promote to float for math
        o[i] = std::tanh(xi);
    }
}

void Inv(int dim, buffer<bf16>& x, buffer<float>& o) {
    for (int i = 0; i < dim; i++) {
        o[i] = 1.0f / x[i];
    }
}

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
    desc.add_options()("Iteration,i", po::value<int>()->default_value(1), "Iteration");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();

    int D = 4096;
    const int X_VOLUME = 1 * D; 
    const int O_VOLUME = 1 * D;
    buffer<bf16> x_in(X_VOLUME);
    buffer<float> o_out(O_VOLUME);

    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/non_linear.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* attn_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    npu_app app_0 = attn_app_mgr->create_app();

    buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(O_VOLUME);
    buffer<bf16> x_0 = app_0.create_bo_buffer<bf16>(X_VOLUME);

    float min_x = -5.0f;
    float max_x = 5;
    for (int i = 0; i < X_VOLUME; i++){
        x_in[i] = min_x + i * ((max_x - min_x) / (X_VOLUME - 1));
        x_0[i] = min_x + i * ((max_x - min_x) / (X_VOLUME - 1));
    }
    x_0.sync_to_device();
    o_0.memset((bf16)(-1.0f));
    o_0.sync_to_device();

    int rtp_activate_func_impl_id = 1; // 0 is gelu, 1 is silu, 2 is sigmoid, 3 is exp, 4 is softplus, 5 is inv_aie, 6 is inv_lut
    header_print("info", "Calculate reference!");
    if (rtp_activate_func_impl_id == 0){
        Gelu(D, x_in, o_out);
    }
    else if (rtp_activate_func_impl_id == 1){
        Silu(D, x_in, o_out);
    }
    else if (rtp_activate_func_impl_id == 2){
        Sigmoid(D, x_in, o_out);
    }
    else if (rtp_activate_func_impl_id == 3){
        exp_func(D, x_in, o_out);
    }
    else if (rtp_activate_func_impl_id == 4){
        softplus(D, x_in, o_out);
    }
    else if (rtp_activate_func_impl_id == 5){
        Inv(D, x_in, o_out);
    }
    else if (rtp_activate_func_impl_id == 6){
        Tanh(D, x_in, o_out);
    }
    else{
        Inv(D, x_in, o_out);
    }


    time_utils::time_with_unit npu_time = {0.0, "us"};
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    // 0 is using gelu
    // 1 is using silu
    // 2 is using sigmoid
    // 3 is using exp
    // 4 is using softplus
    // 5 is using inv_aie,
    // 6 is using inv_lut, inv_lut only support positive input
    gen_seq(app_0.seq(), D, rtp_activate_func_impl_id);

    for (int i = 0; i < NT; i++) {
        time_utils::time_point start = time_utils::now();
        app_0(o_0, x_0);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= NT;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    o_0.sync_from_device();

    header_print("info", "reference output");
    utils::print_matrix(o_out, D);
    header_print("info", "O out");
    utils::print_matrix(o_0, D);

    for (int i = 0; i < O_VOLUME; i++) {
        if (std::isnan(o_out[i])) {
            std::cerr << "NaN found in o_out at index " << i << " which is x value of " << x_in[i] << std::endl;
        }
    }
    // Save x, o, o_ref into a CSV file for plotting with maximum precision
    std::ofstream csv_file("results.csv");
    csv_file << "x,o,o_ref\n";
    csv_file << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (int i = 0; i < X_VOLUME; i++) {
        float x_f = static_cast<float>(x_in[i]);
        float o_f = static_cast<float>(o_0[i]);
        float o_ref_f = static_cast<float>(o_out[i]);
        csv_file << x_f << "," << o_f << "," << o_ref_f << "\n";
    }
    csv_file.close();

    header_print("info", "Finished running kernel");
    header_print("info", "Running kernel with bare call (run based).");
    header_print("info", "NPU time: " << npu_time.first << " us");
    return 0;
}