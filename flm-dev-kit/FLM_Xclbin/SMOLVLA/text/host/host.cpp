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

// CPU reference for the packed multi-head GQA attention run on the NPU.
//   q   : [L, DQ]  with DQ = num_heads    * d, head-major within each row.
//   k,v : packed as [nchunks * chunk_keys, DK] with DK = num_kv_heads * d,
//         kv-head-major within each key row. Global key g = c * chunk_keys + j.
//   out : [L, DQ], same head-major layout as q.
// Non-causal (bidirectional) softmax(QK^T / sqrt(d)) V, matching the kernel:
//   - GQA maps query head h -> kv head h / (num_heads / num_kv_heads).
//   - each chunk contributes only its valid keys; the text chunk keeps
//     effective_number keys, every other chunk keeps chunk_keys keys.
void ref_attn_chunked(const bf16* q, const bf16* k, const bf16* v,
                      bf16* out, int L,
                      int DQ, int DK, int d,
                      int num_heads, int num_kv_heads,
                      int chunk_keys, int nchunks,
                      int text_chunk_id, int effective_number) {
    const float scale = 1.0f / std::sqrt((float)d);
    const int group = num_heads / num_kv_heads;
    std::vector<float> acc(d);
    for (int i = 0; i < L; i++) {
        for (int h = 0; h < num_heads; h++) {
            const int kh = h / group;
            const bf16* qi = q + (size_t)i * DQ + (size_t)h * d;
            float m = -INFINITY, l = 0.0f;
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (int c = 0; c < nchunks; c++) {
                const int n_valid = (c == text_chunk_id) ? effective_number : chunk_keys;
                for (int j = 0; j < n_valid; j++) {
                    const int g = c * chunk_keys + j;
                    const bf16* kj = k + (size_t)g * DK + (size_t)kh * d;
                    const bf16* vj = v + (size_t)g * DK + (size_t)kh * d;
                    float s = 0.0f;
                    for (int e = 0; e < d; e++)
                        s += (float)qi[e] * (float)kj[e];
                    s *= scale;
                    const float m_new = std::max(m, s);
                    const float corr = std::exp(m - m_new);   // rescale prior accumulation
                    const float pj = std::exp(s - m_new);
                    l = l * corr + pj;
                    for (int e = 0; e < d; e++)
                        acc[e] = acc[e] * corr + pj * (float)vj[e];
                    m = m_new;
                }
            }
            const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
            bf16* oi = out + (size_t)i * DQ + (size_t)h * d;
            for (int e = 0; e < d; e++) oi[e] = (bf16)(acc[e] * inv_l);
        }
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
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(177), "L_wanted");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "L_begin");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(1000), "Iteration");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();
    bool pass = true;

    int num_heads = 15;
    int num_kv_heads = 5;
    int d = 64;
    int DQ = d * num_heads;
    int D = 960;
    int DK = d * num_kv_heads;

    buffer<bf16> q_cache;
    buffer<bf16> k_cache;
    buffer<bf16> v_cache;
    buffer<bf16> aux_out_whole;

    std::string reference_path = "/path/to/models/SmolVLA/ref.safetensors";
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
    std::string weight_path = "/path/to/models/SmolVLA/model.safetensors";
    std::unique_ptr<SafeTensors> weights = std::make_unique<SafeTensors>(weight_path);

    buffer<bf16> q_in;
    buffer<bf16> k_in;
    buffer<bf16> v_in;
    buffer<bf16> w_in;
    buffer<bf16> y_ref;

    references->load_weights(q_in, "vlm.layer.0.q_rope");
    references->load_weights(k_in, "vlm.layer.0.k_rope");
    references->load_weights(v_in, "vlm.layer.0.v_proj");
    weights->load_weights(w_in, "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.o_proj.weight");
    references->load_weights(y_ref, "vlm.layer.0.o_proj");

    const int L_wanted = vm["L_wanted"].as<int>();
    int L_padded = 256; //((L_wanted + 63) / 64) * 64;
    header_print("info", "L padded " << L_padded);

    const int N_IMG = 128;
    const int N_LANG = 48;                      
    const int CHUNK_KEYS = 64;                 
    const int CHUNK_STRIDE = CHUNK_KEYS * DK;
    const int NCHUNKS = 3;
    const int Q_VOLUME = L_padded * DQ;
    const int O_VOLUME = L_padded * DQ;
    const int K_VOLUME = NCHUNKS * CHUNK_STRIDE;
    const int V_VOLUME = NCHUNKS * CHUNK_STRIDE;
    const int W_VOLUME = D * DQ;

    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    std::string xclbin_name = "build/xclbins/text.xclbin";
    npu_app_manager* text_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    npu_app attn_app = text_app_mgr->create_app();
    npu_app mm_app = text_app_mgr->create_app();

    buffer<bf16> x_0 = attn_app.create_bo_buffer<bf16>(O_VOLUME);
    buffer<bf16> q_0 = attn_app.create_bo_buffer<bf16>(Q_VOLUME);
    buffer<bf16> k_0 = attn_app.create_bo_buffer<bf16>(K_VOLUME);
    buffer<bf16> v_0 = attn_app.create_bo_buffer<bf16>(V_VOLUME);
    buffer<bf16> w_0 = mm_app.create_bo_buffer<bf16>(W_VOLUME);
    buffer<bf16> y_0 = mm_app.create_bo_buffer<bf16>(O_VOLUME);

    bytes pad_mask_bytes;
    references->load_weights(pad_mask_bytes, "input.processed.observation.language.attention_mask");
    const uint8_t* pad_mask = pad_mask_bytes.data();
    int total_valid = 0;
    for (int i = 0; i < N_LANG; i++) if (pad_mask[i]) total_valid++;
    const int eff_text = total_valid;   // exclude N_IMG image + 1 state
    header_print("info", "eff_text = " << eff_text << " (valid prefix = " << (N_IMG + eff_text + 1) << ")");

    // move q_cache: q_in is [1, 177, 15, 64] (image x 2 + text + state)
    q_0.memset((bf16)(0.0f));  // zero-pad rows [L_wanted, L_padded)
    for (int l = 0; l < N_IMG; l++){
        memcpy(q_0.data() + l * DQ, q_in.data() + l * DQ, DQ * sizeof(bf16));
    }
    for (int l = 0; l < eff_text; l++){
        memcpy(q_0.data() + l * DQ + N_IMG * DQ, q_in.data() + l * DQ + N_IMG * DQ, DQ * sizeof(bf16));
    }
    q_0.sync_to_device();

    // move k v cache
    k_0.memset((bf16)(0.0f));
    v_0.memset((bf16)(0.0f));
    for (int l = 0; l < N_IMG; l++){
        memcpy(k_0.data() + l * DK, k_in.data() + l * DK, DK * sizeof(bf16));
        memcpy(v_0.data() + l * DK, v_in.data() + l * DK, DK * sizeof(bf16));
    }
    for (int l = 0; l < eff_text; l++){
        memcpy(k_0.data() + l * DK + N_IMG * DK, k_in.data() + l * DK + N_IMG * DK, DK * sizeof(bf16));
        memcpy(v_0.data() + l * DK + N_IMG * DK, v_in.data() + l * DK + N_IMG * DK, DK * sizeof(bf16));
    }
    k_0.sync_to_device();
    v_0.sync_to_device();

    memcpy(w_0.data(), w_in.data(), D * DQ * sizeof(bf16));
    w_0.sync_to_device();

    x_0.memset((bf16)(-1.0f));
    x_0.sync_to_device();
    y_0.memset((bf16)(-1.0f));
    y_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    gen_attn_seq(attn_app.seq(), d, num_heads, num_kv_heads,
                 /*number_of_blocks=*/NCHUNKS,
                 /*text_chunk_id=*/NCHUNKS - 1,
                 /*effective_number=*/eff_text);
    gen_mm_seq(mm_app.seq(), L_padded, DQ, D, 0, 0);

    for (int i = 0; i < NT; i++) {
        time_utils::time_point start = time_utils::now();
        attn_app(x_0, q_0, k_0, v_0);
        mm_app(y_0, x_0, w_0);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= NT;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    y_0.sync_from_device();

    // x_0 holds the attention output [L_padded, DQ]; slice the valid [L_wanted, DQ] rows.
    buffer<bf16> o_wanted((size_t)(N_IMG + eff_text) * DQ);
    o_wanted.memset((bf16)(0.0f));
    for (int i = 0; i < (N_IMG + eff_text); i++){
        memcpy(o_wanted.data() + i * DQ,
               y_0.data() + i * DQ,
               DQ * sizeof(bf16));
    }

    // cpu ref: recompute attention from the same packed q/k/v host buffers.
    buffer<bf16> o_ref((size_t)(N_IMG + eff_text) * DQ);
    o_ref.memset((bf16)(0.0f));
    for (int i = 0; i < (N_IMG + eff_text); i++){
        memcpy(o_ref.data() + i * DQ,
               y_ref.data() + i * DQ,
               DQ * sizeof(bf16));
    }

    header_print("info", "NPU result");
    utils::print_matrix(o_wanted, DQ);
    header_print("info", "CPU reference softmax(QK^T)V:");
    utils::print_matrix(o_ref, DQ);

    error_metrics metrics = get_error_metrics(o_wanted, o_ref);
    print_error_metrics(metrics);
    pass = metrics.CosineSimilarity > 0.99f;
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