// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

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

#include <cassert>
#include <stdfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <bit>
#include <cstdint>
#include <vector>
#include <thread>
#include <cinttypes>

// #include "debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "vm_args.hpp"
#include "utils/utils.hpp"

#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_queue.h"

namespace po = boost::program_options;
#include "metrices.hpp"
#include <immintrin.h>
#include <algorithm>
#include "GateDeltaNetPrefillRuntimeSequence.hpp"
#include "tensor_utils/safe_tensors.hpp"

int round_up(int x, int multiple){
    if (multiple == 0) {
        return x; // Cannot divide by zero
    }

    return ((x + multiple - 1) / multiple) * multiple;
}

int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // // Program arguments parsing
    DeltaNetPrefillArgs args;
    if (!parse_matmul_args(argc, (char**)argv, args)){
        return 1;
    }

    // print out all the arguments in here 
    std::cout << "Parsed arguments:" << std::endl;
    std::cout << "Architecture: " << args.arch << std::endl;
    std::cout << "Trace size: " << args.trace_size << std::endl;
    std::cout << "Sequence length (L): " << args.L << std::endl;
    std::cout << "Dimension of Query/Key (DQ): " << args.DQ << std::endl;
    std::cout << "Dimension of Key/Value (DK): " << args.DK << std::endl;
    std::cout << "Dimension of Value (DV): " << args.DV << std::endl;
    std::cout << "Number of row tiles: " << args.NUM_OF_ROW << std::endl;
    std::cout << "Number of column tiles: " << args.NUM_OF_COLUMN << std::endl;
    std::cout << "L chunk size for MT " << args.MT_L_chunk_size << std::endl;
    std::cout << "L chunk size for each CT: " << args.CT_L_chunk_size << std::endl;
    std::cout << "Number of attention heads: " << args.NUM_HEAD << std::endl;
    std::cout << "Number of head groups that share the same Q/K: " << args.QK_HEAD_GROUP << std::endl;
    std::cout << "Whether the qkv buffer is in Lx (  DQ*NUM_Q_K_HEAD + DV*NUM_Q_K_HEAD+ DV*NUM_HEAD) row major (1) or in  LxDQ*NUM_Q_K_HEAD + L*DK*NUM_Q_K_HEAD + L*DV*NUM_HEAD row major (0, default): " << args.QKV_buffer_L_head_row_major << std::endl;

    // load reference data
    std::string reference_path = "/path/to/FastFlowLM/FLM_Xclbin/Qwen3_6/gated_deltanet_35BA3B.safetensors";
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);

    std::string xclbin_name = "build/xclbins/GateDeltaNet_prefill.xclbin";

    xrt::device npu_device = xrt::device(0); // NPU device should always be 0
    
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* deltanet_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    npu_app app_0 = deltanet_app_mgr->create_app();

    const int num_head = args.NUM_HEAD;
    const int L_size = args.L;
    const int DQ = args.DQ;
    const int DK = args.DK;
    const int DV = args.DV;
    assert(num_head % args.QK_HEAD_GROUP == 0);
    assert(DQ == DK && "DQ and DK must be equal for DeltaNet query-state projection");
    const int num_qk_head = num_head/ args.QK_HEAD_GROUP;

    std::vector<float> alpha((size_t)L_size*num_head); // lx num_head, in row major order
    std::vector<float> beta( (size_t)L_size*num_head); // lx num_head, in row major order

    std::vector<std::vector<float>> S_after_update(num_head, std::vector<float>((size_t)DV * DK));   // num_head x (DVxDK) in row major order
    std::vector<std::vector<float>> S_before_update(num_head, std::vector<float>((size_t)DV * DK)); // num_head x (DVxDK) in row major order       
    std::vector<std::vector<bf16>> out(num_head, std::vector<bf16>((size_t)L_size*DV));  // num_head/num_qk_head x (lxdim), in row-major order           

    for(int h = 0; h < num_head; h++){
        std::fill(S_before_update[h].begin(), S_before_update[h].end(), 0.0f);
    }

    // --- s: [num_head, DV, DK] row-major per head, the golden post-update state reference ---
    buffer<float> s_in((size_t)num_head * DV * DK);
    references->load_weights(s_in, "s");
    for(int h = 0; h < num_head; h++){
        memcpy(S_after_update[h].data(), s_in.data() + h * DV * DK, (size_t)DV * DK * sizeof(float));
    }

    // --- alpha / beta: [L, num_head] row-major, same layout as alpha_list ---
    {
        buffer<float> tmp((size_t)L_size * num_head);
        references->load_weights(tmp.as_bytes(), "alpha");
        memcpy(alpha.data(), tmp.data(), (size_t)L_size * num_head * sizeof(float));
        references->load_weights(tmp.as_bytes(), "beta");
        memcpy(beta.data(), tmp.data(), (size_t)L_size * num_head * sizeof(float));
    }

    // --- q/k/v: q_scale_and_k_norm_v is [L, qkv_row_stride] with each row = [q heads | k heads | v heads] ---
    const size_t row_stride = (size_t)num_qk_head * DQ + (size_t)num_qk_head * DK + (size_t)num_head * DV;
    buffer<bf16> qkv_in(L_size * row_stride);
    references->load_weights(qkv_in, "q_scale_and_k_norm_v");

    // Load the golden output reference into out_list.
    // deltanet_out is [L, num_head, DV] (L x num_head*DV interleaved); out_list[h] is [L, DV] per head.
    {
        const size_t o_row_stride = (size_t)num_head * DV;
        buffer<bf16> deltanet_out((size_t)L_size * num_head * DV);
        references->load_weights(deltanet_out.as_bytes(), "deltanet_out");
        for(int h = 0; h < num_head; h++){
            for(int l = 0; l < L_size; l++){
                memcpy(out[h].data() + (size_t)l * DV,
                       deltanet_out.data() + (size_t)l * o_row_stride + (size_t)h * DV,
                       DV * sizeof(bf16));
            }
        }
    }

    const int L_padded = round_up(L_size, args.MT_L_chunk_size);
    std::cout <<"L_size: " << L_size << ", L_padded: " << L_padded << std::endl;

    buffer<float> S_buffer = app_0.create_bo_buffer<float>((size_t)num_head * DV * DK); 
    buffer<bf16> qkv_buffer = app_0.create_bo_buffer<bf16>(L_padded * row_stride);
    buffer<float> alpha_beta_buffer = app_0.create_bo_buffer<float>((size_t)2 * L_padded * num_head);
    buffer<bf16> O_buffer = app_0.create_bo_buffer<bf16>((size_t)L_padded* num_head * DV); 

    
    bool IS_QKV_buffer_L_head_row_major = args.QKV_buffer_L_head_row_major==1? true : false;
    generate_gate_delta_net_prefill_sequence(
        *app_0.seq(),
        args.NUM_OF_COLUMN, args.NUM_OF_ROW,
        args.QK_HEAD_GROUP, args.NUM_HEAD,
        args.MT_L_chunk_size, L_size, L_padded,
        DQ, DK, DV,
        IS_QKV_buffer_L_head_row_major,
        0,0,0

    );
    
    std::vector<std::vector<float>> S_before_update_col_major(num_head, std::vector<float>((size_t)DV * DK)); // num_head x (DVxDK) in column major order
    std::vector<std::vector<float>> S_after_update_col_major(num_head, std::vector<float>((size_t)DV *DK));
    for(int h = 0; h< num_head; h++){
        for(int i = 0; i < DV; i++){
            for(int j = 0; j < DK; j++){
                S_before_update_col_major[h][j * DV + i] = S_before_update[h][i * DK + j];
                S_after_update_col_major[h][j * DV + i] = S_after_update[h][i * DK + j];
            }
        }
    }

    // now, copy S_before_update_col_major_list to S_buffer
    for(int h = 0; h < num_head; h++){
        size_t offset = (size_t)h * DV * DK;
        memcpy(S_buffer.data() + offset, S_before_update_col_major[h].data(), (size_t)DV * DK * sizeof(float));
    }

    float *alpha_beta_buffer_ptr =(float*)(alpha_beta_buffer.data());
    // now, copy alpha, beta 
    for(int h = 0; h < num_head; h++){
        for(int l_idx = 0; l_idx < L_size; l_idx++){
            size_t alpha_offset = h + l_idx * num_head;
            size_t beta_offset = alpha_offset + (size_t)L_padded * num_head;
            alpha_beta_buffer_ptr[alpha_offset] = alpha[h + l_idx* num_head];
            alpha_beta_buffer_ptr[beta_offset] = beta[h + l_idx* num_head];
        }
    }

    // now, copy q k v buffer: each row of qkv_in (length row_stride) into the
    // padded qkv_buffer row (stride 8192 bf16 elements).
    for (int l_idx = 0; l_idx < L_size; l_idx++){
        size_t offset = (size_t)l_idx * row_stride;
        memcpy(qkv_buffer.data() + offset, qkv_in.data() + (size_t)l_idx * row_stride, row_stride * sizeof(bf16));
    }

    int Iterations = 1;
    time_utils::time_with_unit npu_time = {0.0, "us"};    

    S_buffer.sync_to_device();
    qkv_buffer.sync_to_device();
    alpha_beta_buffer.sync_to_device();
    O_buffer.sync_to_device();

    auto start = std::chrono::high_resolution_clock::now();
    app_0(S_buffer, qkv_buffer,alpha_beta_buffer, O_buffer);
    auto end = std::chrono::high_resolution_clock::now();
    npu_time.first +=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    O_buffer.sync_from_device();
    S_buffer.sync_from_device();

    auto average_time_us = npu_time.first/Iterations;   

    // prin the time
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BOX_LINE(60, "Average NPU time per iteration: " << average_time_us << " us");
    MSG_BONDLINE(60);    
    
    int number_to_print = 32;
    int num_head_to_debug = 0;
    int l_idx_to_debug = 0;

    buffer<bf16> o_wanted((size_t)L_size * DV);
    buffer<bf16> aux_out((size_t)L_size * DV);
    buffer<float> s((size_t)DV * DK);
    buffer<float> s_ref((size_t)DV * DK);
    // O_buffer is L x (num_head*DV) row-major, so each head's output is strided.
    const size_t o_row_stride = (size_t)num_head * DV;
    std::vector<bf16> o_head((size_t)L_size * DV);
    for(int head_idx = 0; head_idx < 32; head_idx ++){

        std::cout << "printing head " << head_idx << " S matrix after update: " << std::endl;
        // gather this head's L x DV output from the interleaved O_buffer
        for(int l = 0; l < L_size; l++){
            memcpy(o_head.data() + l * DV,
                   O_buffer.data() + l * o_row_stride + (size_t)head_idx * DV,
                   DV * sizeof(bf16));
        }
        o_wanted.copy_from(o_head.data(), (size_t)L_size * DV);
        aux_out.copy_from(out[head_idx].data(), (size_t)L_size * DV);

        error_metrics metrics = get_error_metrics(o_wanted, aux_out);
        print_error_metrics(metrics);

        // compare S: device S_buffer comes back row-major per head (index i*DK + j),
        // matching the golden row-major reference.
        s.copy_from(S_buffer.data() + (size_t)DV * DK * head_idx, (size_t)DV * DK);
        s_ref.copy_from(S_after_update[head_idx].data(), (size_t)DV * DK);

        metrics = get_error_metrics(s, s_ref);
        print_error_metrics(metrics);
    }
    return 0;
}

