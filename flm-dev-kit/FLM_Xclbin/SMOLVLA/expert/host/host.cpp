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

void ref_attn_chunked(const bf16* q, int q_stride,
                      const bf16* kv, int chunk_stride, int kv_row,
                      bf16* out, int Lq, int d,
                      const int* chunk_nvalid, const int* chunk_causal, int nchunks) {
    const float scale = 1.0f / std::sqrt((float)d);
    std::vector<float> acc(d);
    for (int i = 0; i < Lq; i++) {
        float m = -INFINITY, l = 0.0f;
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int c = 0; c < nchunks; c++) {
            const bf16* Kc = kv + (size_t)c * chunk_stride;
            const bf16* Vc = kv + (size_t)c * chunk_stride + d;
            int jmax = chunk_nvalid[c];
            if (chunk_causal[c]) jmax = std::min(jmax, i + 1);   // key j <= query i
            for (int j = 0; j < jmax; j++) {
                float s = 0.0f;
                for (int e = 0; e < d; e++)
                    s += (float)q[i * q_stride + e] * (float)Kc[j * kv_row + e];
                s *= scale;
                float m_new = std::max(m, s);
                float corr = std::exp(m - m_new);   // rescale prior accumulation
                l = l * corr;
                for (int e = 0; e < d; e++) acc[e] *= corr;
                float pj = std::exp(s - m_new);
                l += pj;
                for (int e = 0; e < d; e++) acc[e] += pj * (float)Vc[j * kv_row + e];
                m = m_new;
            }
        }
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (int e = 0; e < d; e++) out[i * d + e] = (bf16)(acc[e] * inv_l);
    }
}

// int main(int argc, char **argv) {
//     // Fix the seed to ensure reproducibility in CI.
//     srand(0);
//     // Program arguments parsing
//     po::options_description desc("Allowed options");
//     po::variables_map vm;
//     arg_utils::add_default_options(desc);

//     // Add custom options
//     desc.add_options()("Seed,s", po::value<unsigned int>()->default_value(32), "Random seed");
//     desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
//     desc.add_options()("L_wanted,l", po::value<int>()->default_value(50), "L_wanted");
//     desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "L_begin");
//     desc.add_options()("Iteration,i", po::value<int>()->default_value(500), "Iteration");
//     arg_utils::parse_options(argc, argv, desc, vm);

//     // User logic
//     const int NT = vm["Iteration"].as<int>();
//     const bool VERIFY = vm["Verify"].as<bool>();
//     const unsigned int SEED = vm["Seed"].as<unsigned int>();

//     int num_heads = 15;
//     int num_kv_heads = 5;
//     int d = 64;
//     int DQ = d * num_heads;
//     int DK = d * num_kv_heads;

//     buffer<bf16> q_cache;
//     buffer<bf16> k_cache;
//     buffer<bf16> v_cache;
//     buffer<bf16> aux_out_whole;

//     std::string reference_path = "/path/to/models/SmolVLA/expert_reference.safetensors";
//     std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
//     std::string reference_path1 = "/path/to/models/SmolVLA/ref.safetensors";
//     std::unique_ptr<SafeTensors> references1 = std::make_unique<SafeTensors>(reference_path1);

//     buffer<bf16> q_in;
//     buffer<bf16> k_in;
//     buffer<bf16> v_in;
//     buffer<bf16> realk_in;
//     buffer<bf16> realv_in;
//     buffer<bf16> y_ref0;
//     buffer<bf16> y_ref1;

//     const bool IS_SELF = true;
//     if (IS_SELF == false){
//         references->load_weights(q_in, "layer_1.q_rope");
//         references->load_weights(k_in, "cross_kv_cache.layer_1.key_states");
//         references->load_weights(v_in, "cross_kv_cache.layer_1.value_states");
//         references->load_weights(y_ref0, "layer_1.attention_out");
//     }
//     else{
//         references1->load_weights(q_in, "expert.layer.0.q_rope");
//         references1->load_weights(k_in, "vlm.layer.0.k_rope");
//         references1->load_weights(v_in, "vlm.layer.0.v_proj");
//         references->load_weights(y_ref0, "layer_0.attention_out");
//         references1->load_weights(realk_in, "expert.layer.0.k_rope");
//         references1->load_weights(realv_in, "expert.layer.0.v_proj");
//         references1->load_weights(y_ref1, "expert.layer.0.attn_output");
//     }

//     const int L_wanted = vm["L_wanted"].as<int>();
//     int L_padded = 64;

//     const int Q_VOLUME = L_padded * DQ;
//     const int O_VOLUME = L_padded * DQ;

//     bool pass = true;
//     // NPU instance
//     xrt::device npu_device = xrt::device(0); // NPU device should always be 0

//     std::string xclbin_name = "build/xclbins/expert.xclbin";
//     npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
//     npu_app_manager* attn_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

//     header_print("info", "Matrix size " << L_padded << "x" << DQ << "x" << L_padded);

//     npu_app app_0 = attn_app_mgr->create_app();

//     buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(O_VOLUME);
//     buffer<bf16> q_0 = app_0.create_bo_buffer<bf16>(Q_VOLUME);

//     // move q_cache: q_in is [1, 50, 15, 64] (seq x heads x d); extract one head into q_0 [1, 50, 64]
//     const int NUM_Q_HEADS = 15;
//     const int HEAD = 0;  // which of the 15 heads to run
//     q_0.memset((bf16)(0.0f));  // zero-pad rows [L_wanted, L_padded)
//     for (int l = 0; l < L_wanted; l++){
//         memcpy(q_0.data() + l * DQ,
//                q_in.data() + (l * NUM_Q_HEADS + HEAD) * d,
//                DQ * sizeof(bf16));
//     }
//     q_0.sync_to_device();
    
//     const int KV_HEAD = 0;
//     const int NUM_KV_HEADS_SRC = 5;
//     const int PREFIX_LEN = 177;
//     const int N_IMG = 128;
//     const int N_LANG = 48;                      
//     const int CHUNK_KEYS = 64;                 
//     const int CHUNK_STRIDE = CHUNK_KEYS * DK;
//     const int NCHUNKS = IS_SELF ? 4 : 3;

//     bytes pad_mask_bytes;
//     references->load_weights(pad_mask_bytes, "prefix_pad_masks");
//     const uint8_t* pad_mask = pad_mask_bytes.data();
//     int total_valid = 0;
//     for (int i = 0; i < PREFIX_LEN; i++) if (pad_mask[i]) total_valid++;
//     const int eff_text = total_valid - N_IMG - 1;   // exclude N_IMG image + 1 state
//     header_print("info", "eff_text = " << eff_text << " (valid prefix = " << (N_IMG + eff_text + 1) << ")");

//     buffer<bf16> k_0 = app_0.create_bo_buffer<bf16>(NCHUNKS * CHUNK_STRIDE);
//     buffer<bf16> v_0 = app_0.create_bo_buffer<bf16>(NCHUNKS * CHUNK_STRIDE);
//     k_0.memset((bf16)(0.0f));
//     v_0.memset((bf16)(0.0f));

//     for (int l = 0; l < (N_IMG + eff_text + 1); l++){
//         memcpy(k_0.data() + l * DK,
//                k_in.data() + l * DK,
//                DK * sizeof(bf16));
//         memcpy(v_0.data() + l * DK,
//                v_in.data() + l * DK,
//                DK * sizeof(bf16));
//     }

//     for (int l = 0; l < L_wanted; l++){
//         memcpy(k_0.data() + l * DK + 3 * 64 * DK,
//                realk_in.data() + l * DK,
//                DK * sizeof(bf16));
//         memcpy(v_0.data() + l * DK + 3 * 64 * DK,
//                realv_in.data() + l * DK,
//                DK * sizeof(bf16));
//     }
    
//     k_0.sync_to_device();
//     v_0.sync_to_device();

//     o_0.memset((bf16)(-1.0f));
//     o_0.sync_to_device();

//     time_utils::time_with_unit npu_time = {0.0, "us"};
//     header_print("info", "Running runtime test.");
//     header_print("info", "Running kernel with bare call (direct based).");
//     gen_attn_seq(app_0.seq(), d, num_heads, num_kv_heads,
//                  /*number_of_blocks=*/3,
//                  /*text_chunk_id=*/2,
//                  /*effective_number=*/eff_text + 1,
//                  /*is_self=*/IS_SELF ? 1 : 0);

//     for (int i = 0; i < NT; i++) {
//         time_utils::time_point start = time_utils::now();
//         app_0(o_0, q_0, k_0, v_0);
//         time_utils::time_point stop = time_utils::now();
//         npu_time.first += time_utils::duration_us(start, stop).first;
//     }
//     npu_time.first /= NT;
//     MSG_BONDLINE(60);
//     MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
//     MSG_BONDLINE(60);

//     o_0.sync_from_device();

//     // o_0 holds the attention output [L_padded, d]; slice the valid [L_wanted, d] rows.
//     buffer<bf16> o_wanted(L_wanted * DQ);
//     o_wanted.memset((bf16)(0.0f));
//     for (int i = 0; i < L_wanted; i++){
//         memcpy(o_wanted.data() + i * DQ,
//                o_0.data() + i * DQ,
//                DQ * sizeof(bf16));
//     }

//     const int AO_ROW = 1024;
//     buffer<bf16> ref0(L_wanted * DQ);
//     ref0.memset((bf16)(0.0f));
//     for (int l = 0; l < L_wanted; l++){
//         memcpy(ref0.data() + l * DQ,
//                y_ref0.data() + l * AO_ROW + HEAD * d,
//                DQ * sizeof(bf16));
//     }
//     buffer<bf16> ref1(L_wanted * DQ);
//     ref1.memset((bf16)(0.0f));
//     for (int l = 0; l < L_wanted; l++){
//         memcpy(ref1.data() + l * DQ,
//                y_ref1.data() + l * DQ,
//                DQ * sizeof(bf16));
//     }

//     // tensor_2d<bf16> tensor_o(o_wanted, 8, 0);
//     // tensor_2d<bf16> tensor_ref(ref, 8, 0);
//     // buffer<bf16> c_sin(400);
//     // for (int i = 0; i < 400; i++){
//     //     error_metrics metrics = get_error_metrics(tensor_o[i], tensor_ref[i]);
//     //     c_sin[i] = metrics.CosineSimilarity;
//     // }
//     // header_print("info", "Cosine similarity");
//     // utils::print_matrix(c_sin, 8, 50, 8);

//     error_metrics metrics = get_error_metrics(o_wanted, ref1);
//     print_error_metrics(metrics);
//     header_print("info", "Finished running kernel");
//     header_print("info", "Running kernel with bare call (run based).");
//     header_print("info", "NPU time: " << npu_time.first << " us");

//     if (pass){
//         header_print("info", "PASSED ");
//     } else {
//         header_print("info", "FAILED!");
//     }
//     return 0;
// }

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
    const size_t SEQ = 50;  // real sequence length in the reference
    const size_t m = 16;
    const size_t k = 256;
    const size_t n = 64;

    int bias = 0;
    bool pass = true;

    std::string reference_path = "/path/to/models/SmolVLA/expert_reference.safetensors";
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
    std::string weight_path = "/path/to/models/SmolVLA/expert_weights_pad256.safetensors";
    std::unique_ptr<SafeTensors> weights = std::make_unique<SafeTensors>(weight_path);

    buffer<bf16> x_in;
    buffer<bf16> w_in;
    buffer<bf16> b_in;
    buffer<bf16> y_ref;
    int M;
    int K;
    int N;

    if (bias == 1){
        references->load_weights(x_in, "suffix.action_time");
        weights->load_weights(w_in, "action_time_mlp_out.weight");
        weights->load_weights(b_in, "action_time_mlp_out.bias");
        references->load_weights(y_ref, "suffix.embs");
        M = 64;    // padded rows for the kernel tile
        K = 768;
        N = 768;
    }
    else{

        references->load_weights(x_in, "layer_0.input_layernorm");
        weights->load_weights(w_in, "layer_0.qkv_proj.weight");
        references->load_weights(y_ref, "layer_0.qkv_proj");
        M = 64;    // padded rows for the kernel tile
        K = 768;
        N = 1792;
    }

    const int Y_VOLUME = M * N;
    const int X_VOLUME = M * K;
    int bias_offset = (bias == 1)? N : 0;
    const int W_VOLUME = N * K + bias_offset;

    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/expert.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* mm_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    header_print("info", "Matrix size " << M << "x" << K << "x" << N);

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
    unsigned long data_size = M * K * N / n / 4 + K * N * M / m / 4;
    if (VERIFY){
        header_print("info", "Running runtime test.");
        header_print("info", "Running kernel with bare call (direct based).");
        int Iterations = 1000;
        gen_mm_seq(app_0.seq(), M, K, N, m, k, n, 0, 0);
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
        header_print("info", "NPU result:");
        utils::print_matrix(y_wanted, N);
        header_print("info", "Ref result:");
        utils::print_matrix(y_ref, N);

        error_metrics metrics = get_error_metrics(y_wanted, y_ref);
        print_error_metrics(metrics);

        header_print("info", "Finished running kernel");
        header_print("info", "Running kernel with bare call (run based).");
        utils::print_npu_profile(npu_time, 2.0 * float(M) * float(K) * float(N));
        header_print("info", "Data size: " << std::to_string(data_size));
        header_print("info", "Bandwidth: " + std::to_string(data_size * 2/ npu_time.first * 1e6 / 1e9) + " GB/s");
    }

    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }
    return 0;
}