// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __TEST_PLE8_HPP__
#define __TEST_PLE8_HPP__
// PLE stage test (plan S6.8): runs the `ple8` xclbin (gate tile + up tile + memtile stage + norm tile, iron/ple8_aie.py) on 8 token rows with
// the real Bolt weights of one layer.
//
//   host -> x_in.bin = [h (8 x 2560) | pli (8 x 256) | w_post_pli (2560) | layer_scalar, pad (32)]  bf16, from tools/ple8_ref.py gen
//   device -> y.bin  = [8][2560] bf16 = (h + rmsnorm(W_up . (gelu(W_gate . h) * pli)) * w_post_pli) * layer_scalar
//
// Scoring is done by tools/ple8_ref.py check.  The weights come from the layer's proj BO exactly as the decode design streams them
// (W_gate at w_inp_gate_offset, W_up at w_inp_up_offset, both [row block][k][32 rows] bf16).
#include "decoding_layer.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <vector>
#include <chrono>

struct ple8_opts {
    int layer = 0;
    std::string dir = "ple8_data";
    int repeat = 3;
    int timeout_ms = 20000;
    std::string stage = "full";   // gate | up | full: what the xclbin routes to the shim (debug builds, iron/ple8_aie.py PLE8_STAGE)
};

/// \return 0 = the run itself went fine (repeats bit-identical, no timeout); scores are computed by the python script
inline int test_ple8(model_list::model_t model, npu_app_manager* mgr, LM_Config& lm_config, Q4NX& q4nx, const ple8_opts& o) {
    decoding_layer::dispatch_timeout_ms = o.timeout_ms;
    decoding_layer layer(model, lm_config, mgr, o.layer, 64);
    layer.load_weights(q4nx);
    const int T = 8, D = layer.D, P = layer.hidden_size_per_layer_input;
    if (P != 256 || D != 2560) { std::cerr << "ple8: expects D 2560 / PLI 256" << std::endl; return 2; }
    const size_t h_off = 0, pli_off = (size_t)T * D, wc_off = pli_off + (size_t)T * P;
    const bool st_gate = o.stage == "gate", st_up = o.stage == "up", st_full = o.stage == "full";
    const size_t x_elems = wc_off + D + 32, y_elems = st_gate ? (size_t)T * P : (size_t)T * D;
    const size_t x_total = (x_elems + 8191) / 8192 * 8192, y_total = (y_elems + 8191) / 8192 * 8192;

    std::vector<uint16_t> xin(x_elems);
    {
        std::ifstream f(o.dir + "/x_in.bin", std::ios::binary);
        if (!f) { std::cerr << "ple8: cannot read " << o.dir << "/x_in.bin" << std::endl; return 2; }
        f.read(reinterpret_cast<char*>(xin.data()), x_elems * 2);
        if ((size_t)f.gcount() != x_elems * 2) { std::cerr << "ple8: short x_in.bin" << std::endl; return 2; }
    }
    buffer<bf16> x_bo = layer.layer_app.create_bo_buffer<bf16>(x_total);
    buffer<bf16> y_bo = layer.layer_app.create_bo_buffer<bf16>(y_total);
    x_bo.memset((bf16)0.0f);
    y_bo.memset((bf16)0.0f);
    std::memcpy(x_bo.data(), xin.data(), x_elems * 2);
    x_bo.sync_to_device();
    y_bo.sync_to_device();

    // ---- instruction stream (all arguments in bf16 elements) ----
    npu_sequence* seq = layer.layer_app.seq();
    const int x_arg = layer.x_arg_id, w_arg = layer.proj_arg_id, y_arg = layer.rms_arg_id;
    seq->clear_cmds();
    // out rows <- N (5,3) via IT5 S2MM0
    seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg, S2MM, IT5, bd_10, it_channel_0,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)y_elems}, {0, 0, 0, 1}, -1, 0, true);
    // W_gate -> G.S2MM1, W_up -> U.S2MM1
    seq->npu_dma_memcpy_nd(sizeof(bf16), w_arg, MM2S, IT5, bd_14, it_channel_0,
        {0, 0, 0, (uint32_t)layer.w_inp_gate_offset}, {1, 1, 1, (uint32_t)(P * D)}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
    if (!st_gate)
        seq->npu_dma_memcpy_nd(sizeof(bf16), w_arg, MM2S, IT5, bd_15, it_channel_1,
            {0, 0, 0, (uint32_t)layer.w_inp_up_offset}, {1, 1, 1, (uint32_t)(P * D)}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
    // G.S2MM0: pli rows, then [rb 8][kb 10][t 8][256] slices of h (h is re-sent for every row block)
    seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT4, bd_0, it_channel_0,
        {0, 0, 0, (uint32_t)pli_off}, {1, 1, 1, (uint32_t)(T * P)}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
    seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT4, bd_1, it_channel_0,
        {0, 0, 0, (uint32_t)h_off}, {(uint32_t)(P / 32), (uint32_t)(D / 256), (uint32_t)T, 256}, {0, 256, (uint32_t)D, 1}, -1, 0, true, aggressive_cache);
    // N.S2MM1: [w_post_pli | layer_scalar, pad], then the 8 h rows
    if (st_full) {
        seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT4, bd_2, it_channel_1,
            {0, 0, 0, (uint32_t)wc_off}, {1, 1, 1, (uint32_t)(D + 32)}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
        seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT4, bd_3, it_channel_1,
            {0, 0, 0, (uint32_t)h_off}, {1, 1, 1, (uint32_t)(T * D)}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
    }
    seq->npu_dma_wait(IT4, MM2S, it_channel_0);
    seq->npu_dma_wait(IT4, MM2S, it_channel_0);
    if (st_full) {
        seq->npu_dma_wait(IT4, MM2S, it_channel_1);
        seq->npu_dma_wait(IT4, MM2S, it_channel_1);
    }
    seq->npu_dma_wait(IT5, MM2S, it_channel_0);
    if (!st_gate) seq->npu_dma_wait(IT5, MM2S, it_channel_1);
    seq->npu_dma_wait(IT5, S2MM, it_channel_0);
    seq->cmds2seq();

    // ---- run ----
    std::vector<double> us;
    std::vector<uint16_t> first(y_elems);
    int bad = 0;
    for (int r = 0; r < o.repeat; r++) {
        y_bo.memset((bf16)0.0f);
        y_bo.sync_to_device();
        auto t0 = std::chrono::steady_clock::now();
        ert_cmd_state st = ERT_CMD_STATE_NEW;
        try {
            xrt::run run = layer.layer_app.create_run(x_bo, layer.all_proj_weight_buffer, y_bo, layer.rope_rms_buffer, layer.kv_cache);
            run.start();
            st = run.wait(std::chrono::milliseconds(o.timeout_ms));
        } catch (const std::exception& e) {
            std::cerr << "ple8: dispatch " << r << " failed: " << e.what() << std::endl;
            return 3;
        }
        auto t1 = std::chrono::steady_clock::now();
        if (st != ERT_CMD_STATE_COMPLETED) { std::cerr << "ple8: dispatch " << r << " ended in state " << (int)st << std::endl; return 3; }
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        y_bo.sync_from_device();
        const uint16_t* y = reinterpret_cast<const uint16_t*>(y_bo.data());
        if (r == 0) {
            std::memcpy(first.data(), y, y_elems * 2);
            std::ofstream f(o.dir + "/y_" + o.stage + ".bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(y), y_elems * 2);
        } else if (std::memcmp(first.data(), y, y_elems * 2) != 0) {
            bad++;
        }
    }
    std::vector<double> s = us;
    std::sort(s.begin(), s.end());
    double mean = 0; for (double v : us) mean += v; mean /= us.size();
    std::cout << "ple8: layer " << o.layer << " dispatches " << o.repeat << " latency us min " << s.front() << " median " << s[s.size() / 2]
              << " mean " << mean << " max " << s.back() << " ; repeats not bit-identical to the first: " << bad << std::endl;
    return bad == 0 ? 0 : 4;
}
#endif
