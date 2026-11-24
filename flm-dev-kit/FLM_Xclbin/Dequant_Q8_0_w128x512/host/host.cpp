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
#include "utils/vm_args.hpp"
#include "utils/utils.hpp"
#include "lm_config.hpp"

#include "tensor_utils/q4_npu_eXpress.hpp"
#include "dequant.hpp"

namespace po = boost::program_options;

inline void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 2)
{
    // memcpy(dst, src.data(), src.size());
    // return;
    const int a_block_size = 32 * 256 * 5 / 8;
    const int blocks_per_row = col / 256;

    const int rows = src.size() / a_block_size / blocks_per_row;

    u8 *dst_ptr = dst;
    std::vector<u8 *> src_ptr(vertical_blocks);
    for (int i = 0; i < vertical_blocks; i++)
    {
        src_ptr[i] = src.data() + i * a_block_size * blocks_per_row;
    }
    for (int r = 0; r < rows; r += vertical_blocks)
    {
        for (int c = 0; c < blocks_per_row; c++)
        {
            for (int i = 0; i < vertical_blocks; i++)
            {
                memcpy(dst_ptr, src_ptr[i], a_block_size);
                dst_ptr += a_block_size;
                src_ptr[i] += a_block_size;
            }
        }
        for (int i = 0; i < vertical_blocks; i++)
        {
            src_ptr[i] += (vertical_blocks - 1) * a_block_size * blocks_per_row;
            if (src_ptr[i] + a_block_size * blocks_per_row > src.end())
            {
                src_ptr[i] = src.data(); // useless padding
            }
        }
    }
}
inline void reorder_cpy_q80(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 2)
{
    // memcpy(dst, src.data(), src.size());
    // return;
    const int a_block_size = 32 * 128 * 10 / 8;
    const int blocks_per_row = col / 128;

    const int rows = src.size() / a_block_size / blocks_per_row;

    u8 *dst_ptr = dst;
    std::vector<u8 *> src_ptr(vertical_blocks);
    for (int i = 0; i < vertical_blocks; i++)
    {
        src_ptr[i] = src.data() + i * a_block_size * blocks_per_row;
    }
    for (int r = 0; r < rows; r += vertical_blocks)
    {
        for (int c = 0; c < blocks_per_row; c++)
        {
            for (int i = 0; i < vertical_blocks; i++)
            {
                memcpy(dst_ptr, src_ptr[i], a_block_size);
                dst_ptr += a_block_size;
                src_ptr[i] += a_block_size;
            }
        }
        for (int i = 0; i < vertical_blocks; i++)
        {
            src_ptr[i] += (vertical_blocks - 1) * a_block_size * blocks_per_row;
            if (src_ptr[i] + a_block_size * blocks_per_row > src.end())
            {
                src_ptr[i] = src.data(); // useless padding
            }
        }
    }
}
float get_rmse(buffer<bf16>& y, buffer<bf16>& y_ref);

int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options

    arg_utils::parse_options(argc, const_cast<char **>(argv), desc, vm);
    std::string model_path = "/path/to/models/Qwen3.5-4B-NPU2";

    header_print("info", "Model path: " + model_path);
    LM_Config lm_config;
    lm_config.from_pretrained(model_path);

    int D = lm_config.hidden_size;
    int DH = lm_config.head_dim;
    int DQ = lm_config.num_attention_heads * lm_config.head_dim;
    int DK = lm_config.num_key_value_heads * lm_config.head_dim;
    int DV = lm_config.num_key_value_heads * lm_config.head_dim;
    // overide the layer xclbin

    lm_config.layer_xclbin_name = "build/xclbins/dequant.xclbin";
    header_print("info", "Layer xclbin path: " + lm_config.layer_xclbin_name);
    header_print("info", "Hidden size (D): " + std::to_string(D));


    std::unique_ptr<Q4NX> q4nx_model = std::make_unique<Q4NX>(model_path);

    buffer<bf16> embed;
    Dequant dequant_seq_gen(lm_config);
    
    header_print("info", "Testing dequant...");
    header_print("info", "Initializing NPU manager...");
    
    xrt::device npu_inst = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_inst, false);
    npu_app_manager* layer_app_mgr = xclbin_mgr.register_xclbin(lm_config.layer_xclbin_name);

    npu_app dequant = layer_app_mgr->create_app();

    {
        buffer<u8> wq_0;
        q4nx_model->load_weights(wq_0, "model.layers.0.linear_attn.ssm_out_proj.weight");

        buffer<u8> proj_weight = dequant.create_bo_buffer<u8>(wq_0.size());
        buffer<bf16> dequanted_w = dequant.create_bo_buffer<bf16>(2560 * 4096);

        reorder_cpy_q80(proj_weight.data(), wq_0, 4096, 2);
        proj_weight.sync_to_device();

        dequant_seq_gen.generate_dequant_q80_packed_in_q4nx_seq(dequant.seq(), 4096, 2560, 0, 0);

        dequant(dequanted_w, proj_weight);
        dequanted_w.sync_from_device();

        utils::print_matrix(dequanted_w, 4096);
    }

    std::cout << std::endl;
    {
        buffer<u8> wq_0;
        q4nx_model->load_weights(wq_0, "model.layers.0.linear_attn.qkv_proj.weight");

        buffer<u8> proj_weight = dequant.create_bo_buffer<u8>(wq_0.size());
        buffer<bf16> dequanted_w = dequant.create_bo_buffer<bf16>(2560 * 8192);

        reorder_cpy(proj_weight.data(), wq_0, 2560, 2);
        proj_weight.sync_to_device();

        dequant_seq_gen.generate_dequant_q4_1_seq(dequant.seq(), 2560, 8192, 0, 0);

        dequant(dequanted_w, proj_weight);
        dequanted_w.sync_from_device();

        utils::print_matrix(dequanted_w, 2560);
    }

    // // User logic
    // LM_Config config;
    // config.from_pretrained(model_path);

    // Dequant dequant_seq(config);

    // Q4NX weights(model_path);

    // const int D_in = 3072;
    // const int D_out = 512;

    // const int W_VOLUME = D_in * D_out;
    // const int WQ_VOLUME = D_in * D_out * 5 / 8;

    // bool pass = true;
    // // NPU instance
    // npu_manager npu_instance(npu_device::device_npu2);

    // std::string xclbin_name = "build/xclbins/dequant.xclbin";
    // npu_app_desc accel_desc_0;
    // accel_desc_0.xclbin_name = xclbin_name;
    // accel_desc_0.app_name = "dequant_q4k_bf16_compiled";

    // header_print("info", "Matrix size " << D_in << "x" << D_out);

    // npu_app app_0 = npu_instance.create_app(accel_desc_0);

    // dequant_seq.generate_seq(app_0.instr_seq, D_in, D_out, 0);


    // buffer<bf16> w_0 = app_0.create_bo_buffer<bf16>(W_VOLUME, 3);
    // buffer<bf16> w_ref = buffer<bf16>(W_VOLUME);
    // buffer<u8> wq_0 = app_0.create_bo_buffer<u8>(WQ_VOLUME, 4);
    // buffer<bf16> w_ref_T(W_VOLUME);
    // buffer<bf16> x_ref(2048 * D_in);
    // for (int i = 0; i < 2048; i++){
    //     for (int j = 0; j < D_in; j++){
    //         x_ref[i * D_in + j] = utils::getRand(-1.0, 1.0);
    //     }
    // }
   
    // weights.load_weights(wq_0, "model.layers.0.self_attn.v_proj.weight");


    // buffer<bf16> w_ref_reordered(w_ref.size());
    // bf16* p = w_ref.data();
    
    // // transpose every 8x8
    // bf16 p_temp[8][8];
    // for (int i = 0; i < D_out / 8; i++){
    //     for (int j = 0; j < D_in / 8; j++){
    //         for (int k = 0; k < 8; k++){
    //             for (int l = 0; l < 8; l++){
    //                 p_temp[k][l] = p[(i * 8 + k) * D_in + (j * 8 + l)];
    //             }
    //         }
            
    //         // write transpose_ptemp to original p
    //         for (int k = 0; k < 8; k++){
    //             for (int l = 0; l < 8; l++){
    //                 p[(i * 8 + k) * D_in + (j * 8 + l)] = p_temp[l][k];
    //             }
    //         }
    //     }
    // }

    
    // // reorder
    // p = w_ref_reordered.data();
    // for (int r = 0; r < D_out / 64; r++){
    //     for(int c = 0; c < D_in / 512; c++){
    //         for (int i = 0; i < 64; i++){
    //             for (int j = 0; j < 512; j++){
    //                 *p++ = w_ref[(r * 64 + i) * D_in + (c * 512 + j)];
    //             }
    //         }
    //     }
    // }
    // header_print("info", "w_ref_reordered!");

    // wq_0.sync_to_device();

    // time_utils::time_with_unit npu_time = {0.0, "us"};
    // unsigned long data_size_rd = D_in * D_out * 5 / 8;
    // unsigned long data_size_wr = D_in * D_out * 2;
    // if (VERIFY){
    //     header_print("info", "Running runtime test.");
    //     header_print("info", "Running kernel with bare call (direct based).");
    //     auto run_list = npu_instance.create_runlist(app_0);
    //     for (int i = 0; i < ITERATIONS; i++){
    //         auto run = app_0.create_run(w_0, wq_0);
    //         run_list.add(run);
    //     }

    //     time_utils::time_point start = time_utils::now();
    //     run_list.execute();
    //     run_list.wait();
    //     time_utils::time_point stop = time_utils::now();
    //     npu_time.first += time_utils::duration_us(start, stop).first;
    //     npu_time.first /= ITERATIONS;
    //     MSG_BONDLINE(60);
    //     MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    //     MSG_BONDLINE(60);

    //     w_0.sync_from_device(); 
    //     // header_print("info", "w_ref:");
    //     // utils::print_matrix(w_ref_reordered, D_in, 16, 16);
    //     header_print("info", "NPU result:");
    //     utils::print_matrix(w_0, D_in, 256, 32);

    //     // if (utils::compare_vectors(w_0, w_ref_reordered, 8, 1, 99) > 0){
    //     //     pass = false;
    //     // }

    //     // float rmse = get_rmse(w_0, w_ref_reordered);
    //     // write_bin(w_0, "w_reordered.bin");
    //     // header_print("info", "RMSE: " << rmse);

    //     header_print("info", "Finished running kernel");
    //     header_print("info", "Running kernel with bare call (run based).");
    //     utils::print_npu_profile(npu_time, 2.0 * float(D_in) * float(D_out));
    //     header_print("info", "U_mem^rd: " + std::to_string(data_size_rd / npu_time.first * 1e6 / 1e9) + " GB/s");
    //     header_print("info", "U_mem^wr: " + std::to_string(data_size_wr / npu_time.first * 1e6 / 1e9) + " GB/s");
    // }

    // header_print("info", "trace");
   
    // if (pass){
    //     header_print("info", "PASSED ");
    // } else {
    //     header_print("info", "FAILED!");
    // }
    // return 0;
}

// void linear(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& w, int M, int K, int N){
//     assert((M * K) == x.size());
//     assert((K * N) == w.size());
//     assert((M * N) == y.size());
//     for (int row = 0; row < M; row++){
//         for (int col = 0; col < N; col++){
//             float sum = 0;
//             for (int k = 0; k < K; k++){
//                 sum = sum + (x[row * K + k] * w[k * N + col]);
//             }
//             y[row * N + col] = (bf16)sum;
//         }
//     }
// }

// float get_rmse(buffer<bf16>& y, buffer<bf16>& y_ref){
//     float rmse = 0;
//     float ref_sum = 0;
//     for (int i = 0; i < y.size(); i++){
//         ref_sum += y_ref[i] * y_ref[i];
//         rmse += (y[i] - y_ref[i]) * (y[i] - y_ref[i]);
//     }
//     return sqrt(rmse / y.size()) / sqrt(ref_sum / y.size());
// }
