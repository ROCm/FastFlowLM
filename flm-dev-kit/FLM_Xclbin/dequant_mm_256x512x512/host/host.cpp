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
#include "embedding.hpp"
#include "util.hpp"
#include "weights_desc.hpp"

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
    arg_utils::parse_options(argc, argv, desc, vm);

    // load reference data and weights
    std::string weight_path = "/path/to/models/Qwen3.5-4B-NPU2/";
    std::string q41_reference_path = weight_path + "/gated_attention_4B.safetensors";
    std::unique_ptr<SafeTensors> q41_references = std::make_unique<SafeTensors>(q41_reference_path);
    std::string q80_reference_path = weight_path + "/gated_deltanet_4B.safetensors";
    std::unique_ptr<SafeTensors> q80_references = std::make_unique<SafeTensors>(q80_reference_path);
    Q4NX weights(weight_path);

 //////////////////////////// model parameters size  ////////////////////////////
    int hidden_size = 2560;
    int num_attention_heads = 16;
    int head_dim = 256;
    int num_key_value_heads = 4;
    int intermediate_size = 3584;
    int linear_head_dim = 128;
    int linear_num_key_heads = 16;
    int linear_num_value_heads = 32;

    int vocab_size = 248320;
    int D = hidden_size;
    int DQ = head_dim * num_attention_heads;
    int DK = head_dim * num_key_value_heads;
    int DV = head_dim * num_key_value_heads;
    int Linear_DQ = linear_head_dim * linear_num_key_heads;
    int Linear_DK = linear_head_dim * linear_num_key_heads;
    int Linear_DV = linear_head_dim * linear_num_value_heads;
    std::cout << "D: " << D << ", DQ: " << DQ << ", DK: " << DK << ", DV: " << DV << std::endl;
    std::cout << "D: " << D << ", Linear_DQ: " << Linear_DQ << ", Linear_DK: " << Linear_DK << ", Linear_DV: " << Linear_DV << std::endl;
    
    uint32_t L_in = 1024;
    std::cout << "[DEBUG] L_in: " << L_in << std::endl;
    uint32_t L_begin = 0;
    uint32_t L_end = L_begin + L_in;
    uint32_t L_chunk = 256;
    uint32_t L_begin_chunked = (L_begin / L_chunk) * L_chunk;
    int L_offset = L_begin - L_begin_chunked;
    std::cout << "[DEBUG] L_offset: " << L_offset << std::endl;
    uint32_t L_padded = round_up_to_multiple(L_in, L_chunk);
    std::cout << "[DEBUG] L_padded: " << L_padded << std::endl;
    int L_effective = L_end - L_begin;
    std::cout << "[DEBUG] L_effective: " << L_effective << std::endl;
    //////////////////////////// model parameters size  ////////////////////////////
    //////////////////////////// setup npu xclbins  ////////////////////////////
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    // dequantization xclbin
    std::string xclbin_name = "build/xclbins/dequant_mm.xclbin";
    npu_app_manager* dequant_mm = xclbin_mgr.register_xclbin(xclbin_name);

    // test q41
    npu_app q_proj = dequant_mm->create_app();
    gen_dequant_mm(q_proj.seq(), L_in, DQ, D, 0, 0);

    buffer<u8> quant_q_weights = q_proj.create_bo_buffer<u8>(q4_k_size((size_t)D * DQ));
    buffer<bf16> buffer_x = q_proj.create_bo_buffer<bf16>(L_padded * D);
    buffer<bf16> buffer_q = q_proj.create_bo_buffer<bf16>((size_t)L_padded * DQ);

    //////////////////////////// setup npu xclbins  ////////////////////////////
    //////////////////////////// Start  ////////////////////////////
    // test q41
    // load the q_proj quantized weight from disk and reorder it into the bo buffer
    int layer_idx = 3;
    {
        buffer<u8> w_q;
        char w_name[128];
        snprintf(w_name, sizeof(w_name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
        weights.load_weights(w_q, w_name);
        // q_proj holds q and the attention gate merged (out dim = 2*DQ); the q part
        // is the first half. Reorder only that half into the q4_1 bo buffer.
        buffer<u8> w_q_q(w_q.data(), w_q.size() / 2);
        assert(w_q_q.size() == quant_q_weights.size());
        reorder_cpy(quant_q_weights.data(), w_q_q, D, 4);
    }
    quant_q_weights.sync_to_device();
    header_print("DEBUG", "Start Dequant qkv weights");
    buffer<bf16> x;
    q41_references->load_weights(x, "Input_layernorm");
    buffer_x.copy_from(x.data() + L_offset * D, L_effective * D);
    buffer_x.sync_to_device();

    buffer_q.memset(bf16(-1.0f));
    buffer_q.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    int Iterations = 1000;
    for (int i = 0; i < Iterations; i++) {
        time_utils::time_point start = time_utils::now();
        q_proj(buffer_q, buffer_x, quant_q_weights);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    buffer_q.sync_from_device();
    
    npu_time.first /= Iterations;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    buffer<bf16> q_ref;
    buffer<bf16> q(L_effective * DQ);
    buffer<bf16> q_r(L_effective * DQ);
    q41_references->load_weights(q_ref, "q_proj");
    q.copy_from(buffer_q.data() + L_offset * DQ, L_effective * DQ);
    q_r.copy_from(q_ref.data() + L_offset * DQ, L_effective * DQ);

    header_print("DEBUG", "Output of dequantization + q projection");
    utils::print_matrix(q, DQ);
    header_print("DEBUG", "Reference q_proj output");
    utils::print_matrix(q_r, DQ);
    error_metrics metrics = get_error_metrics(buffer_q, q_r);
    print_error_metrics(metrics);

    return 0;
}