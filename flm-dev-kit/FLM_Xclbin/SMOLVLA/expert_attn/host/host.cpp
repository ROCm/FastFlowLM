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
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(50), "L_wanted");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "L_begin");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(500), "Iteration");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();

    int num_heads = 15;
    int num_kv_heads = 5;
    int d = 64;
    int DQ = d * num_heads;
    int DK = d * num_kv_heads;

    buffer<bf16> q_cache;
    buffer<bf16> k_cache;
    buffer<bf16> v_cache;
    buffer<bf16> aux_out_whole;

    std::string reference_path = "/path/to/models/SmolVLA/expert_reference.safetensors";
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
    std::string reference_path1 = "/path/to/models/SmolVLA/ref.safetensors";
    std::unique_ptr<SafeTensors> references1 = std::make_unique<SafeTensors>(reference_path1);

    buffer<bf16> q_in;
    buffer<bf16> k_in;
    buffer<bf16> v_in;
    buffer<bf16> realk_in;
    buffer<bf16> realv_in;
    buffer<bf16> y_ref0;
    buffer<bf16> y_ref1;

    const bool IS_SELF = true;
    if (IS_SELF == false){
        references->load_weights(q_in, "layer_1.q_rope");
        references->load_weights(k_in, "cross_kv_cache.layer_1.key_states");
        references->load_weights(v_in, "cross_kv_cache.layer_1.value_states");
        references->load_weights(y_ref0, "layer_1.attention_out");
    }
    else{
        references->load_weights(q_in, "layer_0.q_rope");
        references1->load_weights(k_in, "vlm.layer.0.k_rope");
        references1->load_weights(v_in, "vlm.layer.0.v_proj");
        references->load_weights(y_ref0, "layer_0.attention_out");
        references1->load_weights(realk_in, "expert.layer.0.k_rope");
        references1->load_weights(realv_in, "expert.layer.0.v_proj");
        references1->load_weights(y_ref1, "expert.layer.0.attn_output");
    }

    const int L_wanted = vm["L_wanted"].as<int>();
    int L_padded = 64;

    const int Q_VOLUME = L_padded * DQ;
    const int O_VOLUME = L_padded * DQ;

    bool pass = true;
    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/attn.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* attn_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    header_print("info", "Matrix size " << L_padded << "x" << DQ << "x" << L_padded);

    npu_app app_0 = attn_app_mgr->create_app();

    buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(O_VOLUME);
    buffer<bf16> q_0 = app_0.create_bo_buffer<bf16>(Q_VOLUME);

    // move q_cache: q_in is [1, 50, 15, 64] (seq x heads x d); extract one head into q_0 [1, 50, 64]
    const int NUM_Q_HEADS = 15;
    const int HEAD = 0;  // which of the 15 heads to run
    q_0.memset((bf16)(0.0f));  // zero-pad rows [L_wanted, L_padded)
    for (int l = 0; l < L_wanted; l++){
        memcpy(q_0.data() + l * DQ,
               q_in.data() + (l * NUM_Q_HEADS + HEAD) * d,
               DQ * sizeof(bf16));
    }
    q_0.sync_to_device();
    
    const int KV_HEAD = 0;
    const int NUM_KV_HEADS_SRC = 5;
    const int PREFIX_LEN = 177;
    const int N_IMG = 128;
    const int N_LANG = 48;                      
    const int CHUNK_KEYS = 64;                 
    const int CHUNK_STRIDE = CHUNK_KEYS * DK;
    const int NCHUNKS = IS_SELF ? 4 : 3;

    bytes pad_mask_bytes;
    references->load_weights(pad_mask_bytes, "prefix_pad_masks");
    const uint8_t* pad_mask = pad_mask_bytes.data();
    int total_valid = 0;
    for (int i = 0; i < PREFIX_LEN; i++) if (pad_mask[i]) total_valid++;
    const int eff_text = total_valid - N_IMG - 1;   // exclude N_IMG image + 1 state
    header_print("info", "eff_text = " << eff_text << " (valid prefix = " << (N_IMG + eff_text + 1) << ")");

    buffer<bf16> k_0 = app_0.create_bo_buffer<bf16>(NCHUNKS * CHUNK_STRIDE);
    buffer<bf16> v_0 = app_0.create_bo_buffer<bf16>(NCHUNKS * CHUNK_STRIDE);
    k_0.memset((bf16)(0.0f));
    v_0.memset((bf16)(0.0f));

    for (int l = 0; l < (N_IMG + eff_text + 1); l++){
        memcpy(k_0.data() + l * DK,
               k_in.data() + l * DK,
               DK * sizeof(bf16));
        memcpy(v_0.data() + l * DK,
               v_in.data() + l * DK,
               DK * sizeof(bf16));
    }

    for (int l = 0; l < L_wanted; l++){
        memcpy(k_0.data() + l * DK + 3 * 64 * DK,
               realk_in.data() + l * DK,
               DK * sizeof(bf16));
        memcpy(v_0.data() + l * DK + 3 * 64 * DK,
               realv_in.data() + l * DK,
               DK * sizeof(bf16));
    }
    
    k_0.sync_to_device();
    v_0.sync_to_device();

    o_0.memset((bf16)(-1.0f));
    o_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    gen_attn_seq(app_0.seq(), d, num_heads, num_kv_heads,
                 /*number_of_blocks=*/3,
                 /*text_chunk_id=*/2,
                 /*effective_number=*/eff_text + 1,
                 /*is_self=*/IS_SELF ? 1 : 0);

    for (int i = 0; i < NT; i++) {
        time_utils::time_point start = time_utils::now();
        app_0(o_0, q_0, k_0, v_0);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= NT;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    o_0.sync_from_device();

    // o_0 holds the attention output [L_padded, d]; slice the valid [L_wanted, d] rows.
    buffer<bf16> o_wanted((size_t)L_wanted * DQ);
    o_wanted.memset((bf16)(0.0f));
    for (int i = 0; i < L_wanted; i++){
        memcpy(o_wanted.data() + i * DQ,
               o_0.data() + i * DQ,
               DQ * sizeof(bf16));
    }

    const int AO_ROW = 1024;
    buffer<bf16> ref0((size_t)L_wanted * DQ);
    ref0.memset((bf16)(0.0f));
    for (int l = 0; l < L_wanted; l++){
        memcpy(ref0.data() + l * DQ,
               y_ref0.data() + l * AO_ROW + HEAD * d,
               DQ * sizeof(bf16));
    }
    buffer<bf16> ref1((size_t)L_wanted * DQ);
    ref1.memset((bf16)(0.0f));
    for (int l = 0; l < L_wanted; l++){
        memcpy(ref1.data() + l * DQ,
               y_ref1.data() + l * DQ,
               DQ * sizeof(bf16));
    }

    // tensor_2d<bf16> tensor_o(o_wanted, 8, 0);
    // tensor_2d<bf16> tensor_ref(ref, 8, 0);
    // buffer<bf16> c_sin(400);
    // for (int i = 0; i < 400; i++){
    //     error_metrics metrics = get_error_metrics(tensor_o[i], tensor_ref[i]);
    //     c_sin[i] = metrics.CosineSimilarity;
    // }
    // header_print("info", "Cosine similarity");
    // utils::print_matrix(c_sin, 8, 50, 8);

    error_metrics metrics = get_error_metrics(o_wanted, ref1);
    print_error_metrics(metrics);
    header_print("info", "Finished running kernel");
    header_print("info", "Running kernel with bare call (run based).");
    header_print("info", "NPU time: " << npu_time.first << " us");

    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }
    return 0;
}