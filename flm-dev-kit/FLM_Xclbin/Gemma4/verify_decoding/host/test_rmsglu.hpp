// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __TEST_RMSGLU_HPP__
#define __TEST_RMSGLU_HPP__
// rms/GLU stage test (plan S6.6 / S6.7): runs the `rmsglu` xclbin = projection engine + 8-row norm tile with the resident residual
// + 8-row GLU, with the shim standing in for RoPE / attention / PLE, through ONE WHOLE LAYER with the real Bolt weights:
//
//   x_rms [8][D] --> norm phase 1 --> qkv proj --> (y_qkv.bin, captured)
//   x_o   [8][DQ] (stand-in for the attention output) --> o proj --> norm phase 2 --> up|gate --> GLU --> down --> norm phase 3 --> y_out.bin [8][D]
//
// Scoring is done by tools/rmsglu_ref.py check.  Whole-layer dispatches repeat validly (every dispatch runs all four projection phases).
#include "decoding_layer.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <vector>
#include <chrono>

struct rg_opts {
    int layer = 5;
    std::string dir = "rmsglu_data";
    int repeat = 3;
    int timeout_ms = 20000;
    bool taps = false;   // xclbin built with RMSGLU_TAPS=1: also capture the raw o / down / up|gate rounds on IT4
};

namespace rg {
inline std::vector<uint16_t> read_bf16(const std::string& fn, size_t n) {
    std::ifstream f(fn, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + fn);
    std::vector<uint16_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * 2);
    if ((size_t)f.gcount() != n * 2) throw std::runtime_error("short read " + fn);
    return v;
}
}

inline int test_rmsglu(model_list::model_t model, npu_app_manager* mgr, LM_Config& lm_config, Q4NX& q4nx, const rg_opts& o) {
    decoding_layer::dispatch_timeout_ms = o.timeout_ms;
    decoding_layer layer(model, lm_config, mgr, o.layer, 64);
    layer.load_weights(q4nx);
    constexpr int L = 8;
    const int D = layer.D, I = layer.intermediate_size;
    const bool swa = layer.is_swa_layer, skip = layer.shall_skip_kv;
    const int DQ = layer.DQ;
    const int N_qkv = skip ? DQ : DQ + layer.DK + layer.DV;
    header_print("info", "rmsglu: layer " + std::to_string(o.layer) + (swa ? " swa" : " global") + (skip ? " skip" : " non-skip"));

    // BO layouts
    const size_t x_rms_off = 0, x_o_off = (size_t)L * D;
    size_t x_total = x_o_off + (size_t)L * DQ;
    x_total = (x_total + 8191) / 8192 * 8192;
    const size_t y_qkv_off = 0, y_out_off = (size_t)L * N_qkv;
    const size_t tap_o_off = y_out_off + (size_t)L * D, tap_dn_off = tap_o_off + (size_t)L * D, tap_ug_off = tap_dn_off + (size_t)L * D;
    size_t y_total = o.taps ? tap_ug_off + (size_t)L * 2 * I : y_out_off + (size_t)L * D;
    y_total = (y_total + 8191) / 8192 * 8192;
    buffer<bf16> x_bo = layer.layer_app.create_bo_buffer<bf16>(x_total);
    buffer<bf16> y_bo = layer.layer_app.create_bo_buffer<bf16>(y_total);
    x_bo.memset((bf16)0.0f);
    y_bo.memset((bf16)0.0f);
    {
        auto v = rg::read_bf16(o.dir + "/x_rms.bin", (size_t)L * D);
        std::memcpy(x_bo.data() + x_rms_off, v.data(), v.size() * 2);
        auto w = rg::read_bf16(o.dir + "/x_o.bin", (size_t)L * DQ);
        std::memcpy(x_bo.data() + x_o_off, w.data(), w.size() * 2);
    }
    x_bo.sync_to_device();
    y_bo.sync_to_device();

    // packet ids of iron/rmsglu_tiles.py
    constexpr int PKT_SHIM_TO_RMS = 26, PKT_O_GLOBAL = 17, PKT_O_SWA = 18;
    npu_sequence* seq = layer.layer_app.seq();
    constexpr size_t CT_lock_address_base = 0x000001F000;
    const int x_arg = layer.x_arg_id, w_arg = layer.proj_arg_id, rms_arg = layer.rms_arg_id, y_arg = layer.rope_rms_arg_id;
    seq->clear_cmds();
    for (int i = 0; i < 16; i++) {
        seq->rtp_write(layer.proj_tiles[i], layer.proj_swa_address, swa ? 1 : 0);
        seq->rtp_write(layer.proj_tiles[i], layer.proj_skip_address, skip ? 1 : 0);
        seq->rtp_write(layer.proj_tiles[i], CT_lock_address_base + 16 * 6, 1);   // rtp_available_lock
    }
    // captures: qkv y rounds [round][b][t][32] -> [t][round * 512 + b * 32 + r] (ch0), the layer output rows (ch1)
    const uint32_t qkv_rounds = N_qkv / 512;
    seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg, S2MM, IT3, bd_10, it_channel_0,
        {0, 0, 0, (uint32_t)y_qkv_off}, {qkv_rounds, 16, (uint32_t)L, 32}, {512, 32, (uint32_t)N_qkv, 1}, -1, 0, true);
    seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg, S2MM, IT3, bd_11, it_channel_1,
        {0, 0, 0, (uint32_t)y_out_off}, {1, 1, 1, (uint32_t)(L * D)}, {0, 0, 0, 1}, -1, 0, true);
    if (o.taps) {   // raw rounds [round][b][t][32] of the o and down projections (rows stage) and of the up|gate projection (GLU stage)
        seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg, S2MM, IT4, bd_10, it_channel_0,
            {0, 0, 0, (uint32_t)tap_o_off}, {1, 1, 1, (uint32_t)(L * D)}, {0, 0, 0, 1}, -1, 0, true);
        seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg, S2MM, IT4, bd_12, it_channel_0,
            {0, 0, 0, (uint32_t)tap_dn_off}, {1, 1, 1, (uint32_t)(L * D)}, {0, 0, 0, 1}, -1, 0, true);
        seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg, S2MM, IT4, bd_11, it_channel_1,
            {0, 0, 0, (uint32_t)tap_ug_off}, {1, 1, 1, (uint32_t)(L * 2 * I)}, {0, 0, 0, 1}, -1, 0, true);
    }
    // inputs: the layer input rows to the norm tile, the attention-output stand-in to the o landing, the four norm weight vectors
    seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT3, bd_0, it_channel_0,
        {0, 0, 0, (uint32_t)x_rms_off}, {1, 1, 1, (uint32_t)(L * D)}, {0, 0, 0, 1}, PKT_SHIM_TO_RMS, 0, true, aggressive_cache);
    seq->npu_dma_memcpy_nd(sizeof(bf16), rms_arg, MM2S, IT3, bd_2, it_channel_1,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)(4 * D)}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
    // weights, in the layer's phase order.  The attention-output stand-in is sent only after the qkv projection has been captured: the
    // replay channels of all phases merge into one x FIFO and nothing orders them against each other (in the real layer the o input
    // exists only after attention, so the data dependency does it).
    layer._move_weights(seq, N_qkv, D, layer.w_qkv_offset);
    seq->npu_dma_wait(IT3, S2MM, it_channel_0);
    seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT3, bd_1, it_channel_0,
        {0, 0, 0, (uint32_t)x_o_off}, {1, 1, 1, (uint32_t)(L * DQ)}, {0, 0, 0, 1}, swa ? PKT_O_SWA : PKT_O_GLOBAL, 0, true, aggressive_cache);
    layer._move_weights(seq, D, DQ, layer.w_o_offset);
    layer._move_weights(seq, 2 * I, D, layer.w_up_offset);
    layer._move_weights(seq, D, I, layer.w_down_offset);
    if (o.taps) {
        seq->npu_dma_wait(IT4, S2MM, it_channel_0);
        seq->npu_dma_wait(IT4, S2MM, it_channel_1);
    }
    seq->npu_dma_wait(IT3, S2MM, it_channel_1);   // the layer output is last
    seq->npu_dma_wait(IT3, MM2S, it_channel_0);
    seq->npu_dma_wait(IT3, MM2S, it_channel_1);
    seq->cmds2seq();

    std::vector<double> us;
    for (int r = 0; r < o.repeat; r++) {
        y_bo.memset((bf16)0.0f);
        y_bo.sync_to_device();
        auto t0 = std::chrono::steady_clock::now();
        ert_cmd_state st = ERT_CMD_STATE_NEW;
        try {
            xrt::run run = layer.layer_app.create_run(x_bo, layer.all_proj_weight_buffer, layer.rms_weights_buffer, y_bo, layer.kv_cache);
            run.start();
            st = run.wait(std::chrono::milliseconds(o.timeout_ms));
        } catch (const std::exception& e) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            header_print("ERROR", std::string("rmsglu: dispatch threw after ") + std::to_string((int)ms) + " ms: " + e.what());
            return 3;
        }
        double d = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (st != ERT_CMD_STATE_COMPLETED) {
            header_print("ERROR", "rmsglu: dispatch did not complete after " + std::to_string((int)(d / 1000)) + " ms, state " + std::to_string((int)st));
            return 3;
        }
        us.push_back(d);
        y_bo.sync_from_device();
        if (r == 0) {   // the first dispatch is the one scored; later ones must reproduce it exactly
            std::ofstream f(o.dir + "/y_qkv.bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(y_bo.data() + y_qkv_off), (size_t)L * N_qkv * 2);
            std::ofstream g(o.dir + "/y_out.bin", std::ios::binary);
            g.write(reinterpret_cast<const char*>(y_bo.data() + y_out_off), (size_t)L * D * 2);
            if (o.taps) {
                std::ofstream a(o.dir + "/tap_o.raw", std::ios::binary);
                a.write(reinterpret_cast<const char*>(y_bo.data() + tap_o_off), (size_t)L * D * 2);
                std::ofstream b(o.dir + "/tap_down.raw", std::ios::binary);
                b.write(reinterpret_cast<const char*>(y_bo.data() + tap_dn_off), (size_t)L * D * 2);
                std::ofstream c(o.dir + "/tap_ug.raw", std::ios::binary);
                c.write(reinterpret_cast<const char*>(y_bo.data() + tap_ug_off), (size_t)L * 2 * I * 2);
            }
        } else {
            std::ifstream g(o.dir + "/y_out.bin", std::ios::binary);
            std::vector<uint16_t> first((size_t)L * D);
            g.read(reinterpret_cast<char*>(first.data()), first.size() * 2);
            if (std::memcmp(first.data(), y_bo.data() + y_out_off, first.size() * 2) != 0) {
                header_print("ERROR", "rmsglu: dispatch " + std::to_string(r) + " differs from dispatch 0");
                return 4;
            }
        }
    }
    std::sort(us.begin(), us.end());
    std::string line;
    for (double v : us) line += " " + std::to_string((int)v);
    header_print("info", "rmsglu dispatch us (sorted):" + line);
    std::cout << "rmsglu layer " << o.layer << " min_us " << (int)us.front() << " median_us " << (int)us[us.size() / 2] << " repeats " << o.repeat << std::endl;
    return 0;
}

#endif
