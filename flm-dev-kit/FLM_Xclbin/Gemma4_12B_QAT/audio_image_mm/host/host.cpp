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
    desc.add_options()("Row,r", po::value<int>()->default_value(0), "Row of B to use as input x");
    desc.add_options()("Trace,t", po::value<int>()->default_value(131072), "Trace size");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
    desc.add_options()("EnableTrace,e", po::value<bool>()->default_value(false), "Enable trace");

    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int ROW = vm["Row"].as<int>();
    const int TRACE_SIZE = vm["Trace"].as<int>();
    const bool enable_trace = vm["EnableTrace"].as<bool>();
    const bool VERIFY = vm["Verify"].as<bool>();
    bool pass = true;

    std::string reference_path = "/path/to/models/Gemma4-12B-IT-NPU2/gemma4_audio_image.safetensors";
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);

    buffer<bf16> x_in;
    buffer<bf16> w_in;
    buffer<bf16> y_ref;
    
    buffer<bf16> x_bias_in;
    buffer<bf16> w_bias_in;
    buffer<bf16> bias_in;
    buffer<bf16> y_bias_ref;

    buffer<bf16> x_image_in;
    buffer<bf16> w_image_in;
    buffer<bf16> y_image_ref;

    references->load_weights(x_in, "audio.pre_projection_norm");
    references->load_weights(w_in, "audio.w.embedding_projection");
    references->load_weights(y_ref, "audio.features_valid");

    references->load_weights(x_bias_in, "image.patch_ln1");
    references->load_weights(w_bias_in, "image.w.patch_dense.weight");
    references->load_weights(bias_in, "image.w.patch_dense.bias");
    references->load_weights(y_bias_ref, "image.patch_dense");

    references->load_weights(x_image_in, "image.pre_projection_norm");
    references->load_weights(w_image_in, "image.w.embedding_projection");
    references->load_weights(y_image_ref, "image.features");

    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/mm.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* mm_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    {
        // audio
        const size_t SEQ = 2755;  // real sequence length in the reference
        const size_t m = 64;
        const size_t k = 640;
        const size_t n = 64;
        int M = round_up_to_multiple(SEQ, 256);    // padded rows for the kernel tile
        int K = 640;
        int N = 3840;

        const int Y_VOLUME = M * N;
        const int X_VOLUME = M * K;
        const int W_VOLUME = N * K;

        npu_app app_0 = mm_app_mgr->create_app();

        buffer<bf16> y_0 = app_0.create_bo_buffer<bf16>(Y_VOLUME);
        buffer<bf16> x_0 = app_0.create_bo_buffer<bf16>(X_VOLUME);
        buffer<bf16> w_0 = app_0.create_bo_buffer<bf16>(W_VOLUME);

        x_0.memset((bf16)(0.0f));  // zero-pad rows [SEQ, M)
        memcpy(x_0.data(), x_in.data(), SEQ * K * sizeof(bf16));
        memcpy(w_0.data(), w_in.data(), N * K * sizeof(bf16));

        y_0.memset((bf16)(-1.0f));
        w_0.sync_to_device();
        x_0.sync_to_device();
        y_0.sync_to_device();

        time_utils::time_with_unit npu_time = {0.0, "us"};
        const size_t read_elems = size_t(M) * K * N / n / 4 + size_t(K) * N * M / m / 4;
        const size_t write_elems = size_t(M) * N;
        if (VERIFY){
            header_print("info", "Running runtime test.");
            header_print("info", "Running kernel with bare call (direct based).");
            int Iterations = 1;
            gen_mm_seq(app_0.seq(), M, K, N, m, k, n, 0);
            for (int i = 0; i < Iterations; i++) {
                time_utils::time_point start = time_utils::now();
                app_0(y_0, x_0, w_0);
                time_utils::time_point stop = time_utils::now();
                npu_time.first += time_utils::duration_us(start, stop).first;
            }
            npu_time.first /= Iterations;
            MSG_BONDLINE(60);
            MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
            MSG_BONDLINE(60);

            y_0.sync_from_device(); 

            buffer<bf16> y_wanted(SEQ * N);
            memcpy(y_wanted.data(), y_0.data(), SEQ * N * sizeof(bf16));

            error_metrics metrics = get_error_metrics(y_wanted, y_ref);
            print_error_metrics(metrics);

            header_print("info", "Finished running kernel");
            header_print("info", "Running kernel with bare call (run based).");
            utils::print_npu_profile(npu_time, 2.0 * float(M) * float(K) * float(N));
            const double rd_gbps = double(read_elems) * 2 / npu_time.first * 1e6 / 1e9;
            const double wr_gbps = double(write_elems) * 2 / npu_time.first * 1e6 / 1e9;
            header_print("info", "Data size: " << (read_elems + write_elems) * 2 << " B (rd "
                                            << read_elems * 2 << " B, wr " << write_elems * 2 << " B)");
            header_print("info", "Bandwidth: " + std::to_string(rd_gbps + wr_gbps) + " GB/s (rd "
                                            + std::to_string(rd_gbps) + ", wr " + std::to_string(wr_gbps) + ")");
        }
    }

    {
        // image bias
        const size_t SEQ_bias = 280;  // real sequence length in the reference
        const size_t m_bias = 64;
        const size_t k_bias = 768;
        const size_t n_bias = 64;
        int M_bias = round_up_to_multiple(SEQ_bias, 256);    // padded rows for the kernel tile
        int K_bias = 6912;
        int N_bias = 3840;

        const int Y_VOLUME = M_bias * N_bias;
        const int X_VOLUME = M_bias * K_bias;
        const int W_VOLUME = N_bias * K_bias + N_bias;

        npu_app app_1 = mm_app_mgr->create_app();

        buffer<bf16> y_1 = app_1.create_bo_buffer<bf16>(Y_VOLUME);
        buffer<bf16> x_1 = app_1.create_bo_buffer<bf16>(X_VOLUME);
        buffer<bf16> w_1 = app_1.create_bo_buffer<bf16>(W_VOLUME);

        x_1.memset((bf16)(0.0f));  // zero-pad rows [SEQ, M)
        memcpy(x_1.data(), x_bias_in.data(), SEQ_bias * K_bias * sizeof(bf16));

        memcpy(w_1.data(), bias_in.data(), N_bias * sizeof(bf16));
        memcpy(w_1.data() + N_bias, w_bias_in.data(), N_bias * K_bias * sizeof(bf16));

        y_1.memset((bf16)(-1.0f));
        w_1.sync_to_device();
        x_1.sync_to_device();
        y_1.sync_to_device();

        time_utils::time_with_unit npu_time = {0.0, "us"};
        const size_t read_elems = size_t(M_bias) * K_bias * N_bias / n_bias / 4 + size_t(K_bias) * N_bias * M_bias / m_bias / 4;
        const size_t write_elems = size_t(M_bias) * N_bias;
        if (VERIFY){
            header_print("info", "Running runtime test.");
            header_print("info", "Running kernel with bare call (direct based).");
            int Iterations = 1;
            gen_mm_bias_seq(app_1.seq(), M_bias, K_bias, N_bias, m_bias, k_bias, n_bias, 0, 1);
            for (int i = 0; i < Iterations; i++) {
                time_utils::time_point start = time_utils::now();
                app_1(y_1, x_1, w_1);
                time_utils::time_point stop = time_utils::now();
                npu_time.first += time_utils::duration_us(start, stop).first;
            }
            npu_time.first /= Iterations;
            MSG_BONDLINE(60);
            MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
            MSG_BONDLINE(60);

            y_1.sync_from_device(); 

            buffer<bf16> y_wanted(SEQ_bias * N_bias);
            memcpy(y_wanted.data(), y_1.data(), SEQ_bias * N_bias * sizeof(bf16));

            error_metrics metrics = get_error_metrics(y_wanted, y_bias_ref);
            print_error_metrics(metrics);

            header_print("info", "Finished running kernel");
            header_print("info", "Running kernel with bare call (run based).");
            utils::print_npu_profile(npu_time, 2.0 * float(M_bias) * float(K_bias) * float(N_bias));
            const double rd_gbps = double(read_elems) * 2 / npu_time.first * 1e6 / 1e9;
            const double wr_gbps = double(write_elems) * 2 / npu_time.first * 1e6 / 1e9;
            header_print("info", "Data size: " << (read_elems + write_elems) * 2 << " B (rd "
                                            << read_elems * 2 << " B, wr " << write_elems * 2 << " B)");
            header_print("info", "Bandwidth: " + std::to_string(rd_gbps + wr_gbps) + " GB/s (rd "
                                            + std::to_string(rd_gbps) + ", wr " + std::to_string(wr_gbps) + ")");
        }
    }

    {
        // image
        const size_t SEQ_image = 280;  // real sequence length in the reference
        const size_t m_image = 64;
        const size_t k_image = 768;
        const size_t n_image = 64;
        int M_image = round_up_to_multiple(SEQ_image, 256);    // padded rows for the kernel tile
        int K_image = 3840;
        int N_image = 3840;

        const int Y_VOLUME = M_image * N_image;
        const int X_VOLUME = M_image * K_image;
        const int W_VOLUME = N_image * K_image;

        npu_app app_2 = mm_app_mgr->create_app();

        buffer<bf16> y_2 = app_2.create_bo_buffer<bf16>(Y_VOLUME);
        buffer<bf16> x_2 = app_2.create_bo_buffer<bf16>(X_VOLUME);
        buffer<bf16> w_2 = app_2.create_bo_buffer<bf16>(W_VOLUME);

        x_2.memset((bf16)(0.0f));  // zero-pad rows [SEQ, M)
        memcpy(x_2.data(), x_image_in.data(), SEQ_image * K_image * sizeof(bf16));

        memcpy(w_2.data(), w_image_in.data(), N_image * K_image * sizeof(bf16));

        y_2.memset((bf16)(-1.0f));
        w_2.sync_to_device();
        x_2.sync_to_device();
        y_2.sync_to_device();

        time_utils::time_with_unit npu_time = {0.0, "us"};
        const size_t read_elems = size_t(M_image) * K_image * N_image / n_image / 4 + size_t(K_image) * N_image * M_image / m_image / 4;
        const size_t write_elems = size_t(M_image) * N_image;
        if (VERIFY){
            header_print("info", "Running runtime test.");
            header_print("info", "Running kernel with bare call (direct based).");
            int Iterations = 1;
            gen_mm_bias_seq(app_2.seq(), M_image, K_image, N_image, m_image, k_image, n_image, 0, 0);
            for (int i = 0; i < Iterations; i++) {
                time_utils::time_point start = time_utils::now();
                app_2(y_2, x_2, w_2);
                time_utils::time_point stop = time_utils::now();
                npu_time.first += time_utils::duration_us(start, stop).first;
            }
            npu_time.first /= Iterations;
            MSG_BONDLINE(60);
            MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
            MSG_BONDLINE(60);

            y_2.sync_from_device();

            buffer<bf16> y_wanted(SEQ_image * N_image);
            memcpy(y_wanted.data(), y_2.data(), SEQ_image * N_image * sizeof(bf16));

            error_metrics metrics = get_error_metrics(y_wanted, y_image_ref);
            print_error_metrics(metrics);

            header_print("info", "Finished running kernel");
            header_print("info", "Running kernel with bare call (run based).");
            utils::print_npu_profile(npu_time, 2.0 * float(M_image) * float(K_image) * float(N_image));
            const double rd_gbps = double(read_elems) * 2 / npu_time.first * 1e6 / 1e9;
            const double wr_gbps = double(write_elems) * 2 / npu_time.first * 1e6 / 1e9;
            header_print("info", "Data size: " << (read_elems + write_elems) * 2 << " B (rd "
                                            << read_elems * 2 << " B, wr " << write_elems * 2 << " B)");
            header_print("info", "Bandwidth: " + std::to_string(rd_gbps + wr_gbps) + " GB/s (rd "
                                            + std::to_string(rd_gbps) + ", wr " + std::to_string(wr_gbps) + ")");
        }
    }
    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }
    return 0;
}