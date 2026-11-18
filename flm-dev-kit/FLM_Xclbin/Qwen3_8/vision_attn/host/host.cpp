// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

// Qwen3-VL vision encoder attention on the NPU.
//
// 16 heads, head_dim 72, no GQA, full bidirectional mask, variable L.
// There are no vision reference tensors in the Qwen3.8 ref.safetensors (it
// carries only the text prefill/verify/mtp taps), so the golden is generated
// here in fp32 by cpu_ref::vision_attention.

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
#include "cpu_ref_vision_attn.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"

namespace po = boost::program_options;

// Relative L2 of one head's slice. A global cosine over all 16 heads hides a
// single bad head, so the gate is per head.
static float rel_l2_head(
    const buffer<bf16>& got, const std::vector<float>& ref,
    int L, int D, int h, int dh
){
    double num = 0.0, den = 0.0;
    for (int t = 0; t < L; t++){
        for (int d = 0; d < dh; d++){
            const size_t idx = (size_t)t * D + (size_t)h * dh + d;
            const double a = (float)got[idx];
            const double b = ref[idx];
            num += (a - b) * (a - b);
            den += b * b;
        }
    }
    return (float)std::sqrt(num / (den > 0.0 ? den : 1.0));
}

int main(int argc, char **argv) {
    srand(0);
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("../../models"), "Model file");
    desc.add_options()("Seed,s", po::value<unsigned int>()->default_value(32), "Random seed");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(2240), "Sequence length");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(20), "Iteration");
    desc.add_options()("Rtp,r", po::value<int>()->default_value(-1), "L RTP byte address (-1 = read build/rtp_addr.txt)");
    // 0 = random q/k/v. 1 = q == 0, which makes every score 0 and the softmax
    //     exactly uniform, so O[t,h] is the mean of V over the L valid rows and
    //     every row of O is identical. That isolates the V path, the streaming
    //     reduction and the tail mask from the QK matmul entirely: if the mask
    //     leaks the padded rows the result is scaled by L_valid/L_padded.
    // 2 = every row of V identical. The attention weights sum to 1 whatever the
    //     scores are, so O[t] must equal that one vector exactly, for every t
    //     and every head. Any deviation is pure arithmetic, and a constant
    //     offset here means the weights do NOT sum to 1 - i.e. the denominator
    //     l disagrees with the numerator, which no end-to-end cosine isolates.
    desc.add_options()("Mode", po::value<int>()->default_value(0), "0=random, 1=uniform(q=0), 2=constant V, 3=V=ones");
    arg_utils::parse_options(argc, argv, desc, vm);

    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();

    // Qwen3-VL vision_config: hidden_size 1152, num_heads 16 => head_dim 72.
    const int DH = 72;
    const int num_of_heads = 16;
    const int D = DH * num_of_heads; // 1152
    const float scale = 1.0f / std::sqrt((float)DH);

    const int L_wanted = vm["L_wanted"].as<int>();
    // the array walks K in whole 512-row chunks; the remainder is masked
    const int L_padded = ((L_wanted + 511) / 512) * 512;

    int l_rtp_address = vm["Rtp"].as<int>();
    if (l_rtp_address < 0){
        std::ifstream f("build/rtp_addr.txt");
        if (!f.good()){
            header_print("info", "FAILED: build/rtp_addr.txt missing. RTP addresses are "
                                 "per-build; run tools/extract_rtp_addr.sh after make bitstream.");
            return 1;
        }
        f >> l_rtp_address;
    }
    header_print("info", "L wanted " << L_wanted << " padded to " << L_padded
                          << ", heads " << num_of_heads << ", head_dim " << DH
                          << ", scale " << scale
                          << ", L rtp @ " << l_rtp_address);

    const int Q_VOLUME = L_padded * D;

    xrt::device npu_device = xrt::device(0);
    std::string xclbin_name = "build/xclbins/attn.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* attn_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);
    npu_app app_0 = attn_app_mgr->create_app();

    buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(Q_VOLUME);
    buffer<bf16> q_0 = app_0.create_bo_buffer<bf16>(Q_VOLUME);
    buffer<bf16> k_0 = app_0.create_bo_buffer<bf16>(Q_VOLUME);
    buffer<bf16> v_0 = app_0.create_bo_buffer<bf16>(Q_VOLUME);

    // Correlated q/k. Independent N(0,1) rows would put essentially all of the
    // softmax mass on one key, which passes regardless of how keys are
    // addressed; effective_key_count below reports what this actually achieves.
    std::mt19937 rng(SEED);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> qf((size_t)L_wanted * D), kf((size_t)L_wanted * D), vf((size_t)L_wanted * D);
    std::vector<float> base((size_t)D);
    for (int d = 0; d < D; d++) base[d] = nd(rng);
    const int MODE = vm["Mode"].as<int>();
    for (int t = 0; t < L_wanted; t++){
        for (int d = 0; d < D; d++){
            const float shared = 0.70f * base[d];
            qf[(size_t)t * D + d] = (MODE == 1) ? 0.0f : shared + 0.70f * nd(rng);
            kf[(size_t)t * D + d] = shared + 0.70f * nd(rng);
            vf[(size_t)t * D + d] = nd(rng);
        }
    }
    if (MODE == 2){
        for (int t = 1; t < L_wanted; t++)
            for (int d = 0; d < D; d++) vf[(size_t)t * D + d] = vf[d];
    }
    // 3 = V is all ones. The weights sum to 1, so every output element must be
    //     exactly 1.0. Whatever the device prints instead IS the error, with no
    //     reference model in the way and no cancellation to inflate it.
    if (MODE == 3){
        for (size_t i = 0; i < vf.size(); i++) vf[i] = 1.0f;
    }
    if (MODE == 1) header_print("info", "Uniform mode: q = 0, so O[t] = mean(V) for every t.");
    if (MODE == 2) header_print("info", "Constant-V mode: O[t] must equal V[0] exactly (weights sum to 1).");

    // round-trip through bf16 so the reference sees exactly the device's inputs
    q_0.memset((bf16)0.0f); k_0.memset((bf16)0.0f); v_0.memset((bf16)0.0f);
    for (size_t i = 0; i < (size_t)L_wanted * D; i++){
        q_0[i] = (bf16)qf[i]; k_0[i] = (bf16)kf[i]; v_0[i] = (bf16)vf[i];
        qf[i] = (float)q_0[i]; kf[i] = (float)k_0[i]; vf[i] = (float)v_0[i];
    }
    q_0.sync_to_device(); k_0.sync_to_device(); v_0.sync_to_device();

    o_0.memset((bf16)(-1.0f));
    o_0.sync_to_device();

    const double eff = cpu_ref::effective_key_count(qf, kf, L_wanted, num_of_heads, DH, scale);
    header_print("info", "Effective key count (head 0, query 0): " << eff << " of " << L_wanted);

    time_utils::time_with_unit npu_time = {0.0, "us"};
    gen_attn_seq(app_0.seq(), L_padded, L_wanted, DH, num_of_heads, l_rtp_address);

    for (int i = 0; i < NT; i++) {
        time_utils::time_point start = time_utils::now();
        ert_cmd_state st = app_0(o_0, q_0, k_0, v_0);
        time_utils::time_point stop = time_utils::now();
        if (st != ERT_CMD_STATE_COMPLETED){
            header_print("info", "FAILED: dispatch " << i << " returned state " << st);
            return 1;
        }
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= NT;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    o_0.sync_from_device();

    // Bring-up dump: q/k/v exactly as fed (post bf16 round trip) and the full
    // padded output, so layout hypotheses can be tested offline without a
    // rebuild. tools/analyze_dump.py consumes these.
    {
        auto dumpf = [](const char* path, const float* p, size_t n){
            std::ofstream f(path, std::ios::binary);
            f.write((const char*)p, n * sizeof(float));
        };
        dumpf("build/dump_q.bin", qf.data(), qf.size());
        dumpf("build/dump_k.bin", kf.data(), kf.size());
        dumpf("build/dump_v.bin", vf.data(), vf.size());
        std::vector<float> of((size_t)L_padded * D);
        for (size_t i = 0; i < of.size(); i++) of[i] = (float)o_0[i];
        dumpf("build/dump_o.bin", of.data(), of.size());
        std::ofstream m("build/dump_meta.txt");
        m << L_wanted << " " << L_padded << " " << num_of_heads << " " << DH << "\n";
        header_print("info", "Dumped q/k/v/o to build/dump_*.bin");
    }

    if (!VERIFY){
        header_print("info", "Verify disabled.");
        return 0;
    }

    header_print("info", "Computing fp32 CPU golden...");
    std::vector<float> ref;
    cpu_ref::vision_attention(ref, qf, kf, vf, L_wanted, num_of_heads, DH, scale);

    // Control: every head scored against the next head's k/v. A run that has
    // its head addressing wrong tends to agree with this instead.
    std::vector<int> rot((size_t)num_of_heads);
    for (int h = 0; h < num_of_heads; h++) rot[h] = (h + 1) % num_of_heads;
    std::vector<float> imposter;
    cpu_ref::vision_attention(imposter, qf, kf, vf, L_wanted, num_of_heads, DH, scale, rot.data());

    // Structural dump. In uniform mode every output row must be the same
    // vector (the mean of V), so comparing row 0 against row 1 separates a
    // broken reduction from a broken q/softmax.
    {
        auto dump = [&](const char* tag, int t, int h){
            std::cout << "   " << tag << " t=" << t << " h=" << h << ":";
            for (int d = 0; d < 8; d++)
                std::cout << " " << std::fixed << std::setprecision(4)
                          << (float)o_0[(size_t)t * D + (size_t)h * DH + d];
            std::cout << std::endl;
        };
        auto dumpr = [&](const char* tag, int t, int h){
            std::cout << "   " << tag << " t=" << t << " h=" << h << ":";
            for (int d = 0; d < 8; d++)
                std::cout << " " << std::fixed << std::setprecision(4)
                          << ref[(size_t)t * D + (size_t)h * DH + d];
            std::cout << std::endl;
        };
        header_print("info", "Structural dump (first 8 of head 0 / head 1):");
        dump("npu", 0, 0); dumpr("ref", 0, 0);
        dump("npu", 1, 0); dumpr("ref", 1, 0);
        dump("npu", 0, 1); dumpr("ref", 0, 1);
    }

    bool pass = true;
    float worst = 0.0f; int worst_h = -1;
    header_print("info", "Per-head relative L2 (reference / rotated-head control):");
    for (int h = 0; h < num_of_heads; h++){
        const float r = rel_l2_head(o_0, ref, L_wanted, D, h, DH);
        const float c = rel_l2_head(o_0, imposter, L_wanted, D, h, DH);
        std::cout << "   head " << std::setw(2) << h
                  << "  relL2 " << std::scientific << std::setprecision(3) << r
                  << "   control " << c << std::endl;
        if (r > worst){ worst = r; worst_h = h; }
        // The reference is exact fp32 while the kernel runs bfp16 mmuls
        // (-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16, 7 mantissa bits per
        // 8-element block). That floor alone is ~0.03-0.04 here and grows with
        // L, so the bar is set above it; tools/bfp16_floor.py is what separates
        // "at the floor" from "wrong", scoring the device against a
        // same-arithmetic model instead of against exact fp32.
        if (!(r < 0.06f)) pass = false;
        // The control only discriminates when the heads actually differ. In the
        // synthetic V modes every head shares the same V, so the rotated
        // reference is identical to the real one and this check is vacuous -
        // applying it there would fail a bit-exact result.
        if (MODE == 0 && !(c > 10.0f * r)) pass = false;
    }
    header_print("info", "Worst head " << worst_h << " relative L2 " << worst);

    // whole-tensor metrics over the valid rows
    buffer<bf16> got_v((size_t)L_wanted * D), ref_v((size_t)L_wanted * D);
    for (size_t i = 0; i < (size_t)L_wanted * D; i++){ got_v[i] = o_0[i]; ref_v[i] = (bf16)ref[i]; }
    error_metrics metrics = get_error_metrics(got_v, ref_v);
    print_error_metrics(metrics);

    // The padded tail must be untouched by the masking, and the valid rows must
    // not be zero (which a fully-masked run would produce).
    double sum_abs = 0.0;
    for (size_t i = 0; i < (size_t)L_wanted * D; i++) sum_abs += std::fabs((float)o_0[i]);
    if (sum_abs == 0.0){ header_print("info", "FAILED: output is all zero"); pass = false; }

    if (pass){
        header_print("info", "PASSED");
    } else {
        header_print("info", "FAILED!");
    }
    return pass ? 0 : 1;
}
