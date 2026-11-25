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
#include <random>
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
#include "tensor_2d.hpp"
#include "cpu_q4_k.hpp"
#include "util.hpp"
#include "weights_desc.hpp"

namespace po = boost::program_options;

int round_up_to_multiple(int x, int multiple){
    if (multiple == 0) {
        return x; // Cannot divide by zero
    }

    return ((x + multiple - 1) / multiple) * multiple;
}

/// \brief Put a weight matrix on the grid aie::mmul actually multiplies on.
///
/// mm_aie's `aie::mmul<8,8,8,bf16,bf16,accauto>` lowers to the bfp16 path on
/// npu2: every block of 8 consecutive k at a fixed n is given a shared exponent
/// and an 8-bit mantissa, truncated toward -infinity.  Multiplying by this
/// instead of by w_deq removes the kernel's largest remaining error term, so the
/// two references below bracket it.
static void bfp16_floor_k(bf16* v, size_t rows, size_t k) {
    for (size_t r = 0; r < rows; r++) {
        bf16* row = v + r * k;
        for (size_t g = 0; g < k; g += 8) {
            float mx = 0.0f;
            for (int e = 0; e < 8; e++) mx = std::max(mx, std::fabs((float)row[g + e]));
            if (mx == 0.0f) continue;
            const float st = std::exp2(std::floor(std::log2(mx)) + 1 - 7);
            for (int e = 0; e < 8; e++) row[g + e] = bf16(std::floor((float)row[g + e] / st) * st);
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
    arg_utils::parse_options(argc, argv, desc, vm);

    // This test is self-contained: the weights and activations are generated
    // here, quantized to the FLM q4_k block by cpu_q4_k.hpp, and the golden
    // output is the fp32 product of the *dequantized* weights, so the
    // quantization error itself cancels and what is left is the kernel's
    // arithmetic.  cpu_q4_k.hpp was checked against hardware with a one-hot x
    // (y[l] then reads back a weight row): it reproduces the dequantized
    // weights bit-for-bit, 100% of 4.19M entries, once bfp16_floor_k is applied.

 //////////////////////////// problem size  ////////////////////////////
    // Same shape as the q_proj of Qwen3.5-4B.  Constraints from
    // gen_dequant_mm_512: Din % 512 == 0, Dout % (32 * 8) == 0,
    // L % (128 * 4) == 0.
    int hidden_size = 2560;
    int num_attention_heads = 16;
    int head_dim = 256;

    int D = hidden_size;                       // Din
    int DQ = head_dim * num_attention_heads;   // Dout
    std::cout << "D (Din): " << D << ", DQ (Dout): " << DQ << std::endl;

    uint32_t L_in = 1024;
    std::cout << "[DEBUG] L_in: " << L_in << std::endl;
    uint32_t L_chunk = 256;
    uint32_t L_padded = round_up_to_multiple(L_in, L_chunk);
    std::cout << "[DEBUG] L_padded: " << L_padded << std::endl;
    //////////////////////////// problem size  ////////////////////////////
    //////////////////////////// setup npu xclbins  ////////////////////////////
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    // dequantization xclbin
    std::string xclbin_name = "build/xclbins/dequant_mm.xclbin";
    npu_app_manager* dequant_mm = xclbin_mgr.register_xclbin(xclbin_name);

    npu_app q_proj = dequant_mm->create_app();
    // (L, Din, Dout, weight_offset, out_offset, rtp_act) -- see gen_dequant_mm_512
    gen_dequant_mm_512(q_proj.seq(), L_in, D, DQ, 0, 0, 0);

    buffer<u8> quant_q_weights = q_proj.create_bo_buffer<u8>(q4_k_size((size_t)D * DQ));
    buffer<bf16> buffer_x = q_proj.create_bo_buffer<bf16>(L_padded * D);
    buffer<bf16> buffer_q = q_proj.create_bo_buffer<bf16>((size_t)L_padded * DQ);

    //////////////////////////// setup npu xclbins  ////////////////////////////
    //////////////////////////// Start  ////////////////////////////
    // Generate the weights and quantize them to the q4nx q4_k block on the CPU.
    header_print("DEBUG", "Generating and quantizing q4_k weights on CPU");
    buffer<bf16> w((size_t)DQ * D);      // reference weights, out x in
    buffer<bf16> w_deq((size_t)DQ * D);  // what the dequant core reconstructs
    buffer<u8> w_packed(q4_k_size((size_t)D * DQ));
    {
        std::mt19937 rng(0);
        // Roughly the spread of a real projection weight; the exact scale does
        // not matter, only that groups of 32 have a non-degenerate range.
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (size_t i = 0; i < w.size(); i++) w[i] = bf16(dist(rng));
    }
    q4_k_quantize_pack(w_packed.data(), w_deq.data(), w.data(), DQ, D);

    buffer<bf16> w_bfp((size_t)DQ * D);  // ... and what the mmul then multiplies
    memcpy(w_bfp.data(), w_deq.data(), w_deq.size() * sizeof(bf16));
    bfp16_floor_k(w_bfp.data(), DQ, D);

    // Interleave adjacent 32-row blocks the way the row-pair cores consume them.
    assert(w_packed.size() == quant_q_weights.size());
    reorder_cpy(quant_q_weights.data(), w_packed, D, 2);
    quant_q_weights.sync_to_device();

    // Activations.
    buffer<bf16> x((size_t)L_in * D);
    {
        std::mt19937 rng(1);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < x.size(); i++) x[i] = bf16(dist(rng));
    }
    buffer_x.memset(bf16(0.0f));
    buffer_x.copy_from(x.data(), x.size());
    buffer_x.sync_to_device();

    buffer_q.memset(bf16(-1.0f));
    buffer_q.sync_to_device();

    header_print("DEBUG", "Computing CPU reference");
    buffer<bf16> q_ref((size_t)L_in * DQ);
    buffer<bf16> q_ref_bfp((size_t)L_in * DQ);
    time_utils::time_point cpu_start = time_utils::now();
    cpu_matmul_nt(q_ref.data(), x.data(), w_deq.data(), L_in, DQ, D);
    cpu_matmul_nt(q_ref_bfp.data(), x.data(), w_bfp.data(), L_in, DQ, D);
    time_utils::time_point cpu_stop = time_utils::now();
    MSG_BOX_LINE(60, "CPU reference: " << time_utils::duration_us(cpu_start, cpu_stop).first / 1000.0 << " ms");

    header_print("DEBUG", "Start Dequant + q projection on NPU");
    time_utils::time_with_unit npu_time = {0.0, "us"};
    int Iterations = 500;
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

    buffer<bf16> q((size_t)L_in * DQ);
    q.copy_from(buffer_q.data(), q.size());

    header_print("DEBUG", "Output of dequantization + q projection");
    utils::print_matrix(q, DQ);
    header_print("DEBUG", "CPU reference");
    utils::print_matrix(q_ref, DQ);

    // vs the dequantized weights: the kernel's total arithmetic error.
    header_print("DEBUG", "vs fp32 matmul of the dequantized weights");
    print_error_metrics(get_error_metrics(q, q_ref));
    // vs the bfp16-floored weights: what is left once the mmul's operand
    // truncation is accounted for (x is still truncated, and the accumulation
    // order differs, so this is small but not zero).
    header_print("DEBUG", "vs fp32 matmul of the bfp16-floored weights");
    print_error_metrics(get_error_metrics(q, q_ref_bfp));

    return 0;
}
