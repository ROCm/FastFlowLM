// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __TEST_PROJ8_HPP__
#define __TEST_PROJ8_HPP__
// Projection-only test (plan S6.3/S6.4): runs the `projonly` xclbin (16 Q4 projection cores + x replay tile + shim feed, no rms/rope/
// attention/glu/PLE) through the four projection phases of one layer with the real Bolt weights.
//
//   host -> x_<phase>.bin ([t][K] bf16, from tools/proj8_ref.py gen) -> device -> y_<phase>.bin ([t][N] bf16)
//
// The y packets come back round by round as [round][b = 16 row blocks][t][32 rows]; the shim S2MM BD de-permutes them on the way
// to DDR so that the file is token-major.  Scoring is done by tools/proj8_ref.py check (numpy reference on the dequantised weights).
// `rows` = 1 drives the single-token core (L_TOKENS=1 build), 8 the 8-token core.
#include "decoding_layer.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <chrono>

struct proj8_opts {
    int layer = 5;
    int rows = 8;
    std::string dir = "proj8_data";
    std::string phases = "qkv,o,up,down";
    int repeat = 3;
    int timeout_ms = 20000;
    bool stage = false;   // projstage design: y comes token-major from the MT4 stage via IT4 instead of as raw packets via IT3
};

namespace p8 {
inline std::vector<uint16_t> read_bf16(const std::string& fn, size_t n) {
    std::ifstream f(fn, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + fn);
    std::vector<uint16_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * 2);
    if ((size_t)f.gcount() != n * 2) throw std::runtime_error("short read " + fn);
    return v;
}
struct phase_t {
    std::string name;
    int K, N;            // reduce width, output rows
    size_t w_offset;     // bytes/2 offset of the weights in the proj BO (the host's _move_weights unit)
    int pkt;             // landing packet id on the replay tile
    size_t x_off, y_off; // element offsets in the x / y BOs
};
}

/// \return 0 = pass of the run itself (scores are computed by the python script), non-zero = failure
inline int test_proj8(model_list::model_t model, npu_app_manager* mgr, LM_Config& lm_config, Q4NX& q4nx, const proj8_opts& o) {
    decoding_layer::dispatch_timeout_ms = o.timeout_ms;
    decoding_layer layer(model, lm_config, mgr, o.layer, 64);
    layer.load_weights(q4nx);
    const int L = o.rows;
    const int D = layer.D, I = layer.intermediate_size;
    const bool swa = layer.is_swa_layer, skip = layer.shall_skip_kv;
    const int DQ = layer.DQ;
    header_print("info", "proj8: layer " + std::to_string(o.layer) + (swa ? " swa" : " global") + (skip ? " skip" : " non-skip") + ", rows " + std::to_string(L));

    std::vector<p8::phase_t> ph;
    ph.push_back({"qkv", D, skip ? DQ : DQ + layer.DK + layer.DV, layer.w_qkv_offset, 16, 0, 0});
    ph.push_back({"o", DQ, D, layer.w_o_offset, swa ? 18 : 17, 0, 0});
    ph.push_back({"up", D, 2 * I, layer.w_up_offset, 16, 0, 0});
    ph.push_back({"down", I, D, layer.w_down_offset, 19, 0, 0});
    size_t x_total = 0, y_total = 0;
    for (auto& p : ph) { p.x_off = x_total; p.y_off = y_total; x_total += (size_t)L * p.K; y_total += (size_t)L * p.N; }
    x_total = (x_total + 8191) / 8192 * 8192; y_total = (y_total + 8191) / 8192 * 8192;

    std::vector<bool> run_phase(ph.size(), false);
    {
        std::stringstream ss(o.phases); std::string t;
        while (std::getline(ss, t, ',')) for (size_t i = 0; i < ph.size(); i++) if (ph[i].name == t) run_phase[i] = true;
    }

    buffer<bf16> x_bo = layer.layer_app.create_bo_buffer<bf16>(x_total);
    buffer<bf16> y_bo = layer.layer_app.create_bo_buffer<bf16>(y_total);
    x_bo.memset((bf16)0.0f);
    y_bo.memset((bf16)0.0f);
    for (size_t i = 0; i < ph.size(); i++) {
        if (!run_phase[i]) continue;
        auto v = p8::read_bf16(o.dir + "/x_" + ph[i].name + ".bin", (size_t)L * ph[i].K);
        std::memcpy(x_bo.data() + ph[i].x_off, v.data(), v.size() * 2);
    }
    x_bo.sync_to_device();
    y_bo.sync_to_device();

    // ---- instruction stream ----
    npu_sequence* seq = layer.layer_app.seq();
    constexpr size_t CT_lock_address_base = 0x000001F000;
    const int x_arg = layer.x_arg_id, w_arg = layer.proj_arg_id, y_arg = layer.rms_arg_id;
    seq->clear_cmds();
    for (int i = 0; i < 16; i++) {
        seq->rtp_write(layer.proj_tiles[i], layer.proj_swa_address, swa ? 1 : 0);
        seq->rtp_write(layer.proj_tiles[i], layer.proj_skip_address, skip ? 1 : 0);
        seq->rtp_write(layer.proj_tiles[i], CT_lock_address_base + 16 * 6, 1);   // rtp_available_lock
    }
    for (size_t i = 0; i < ph.size(); i++) {
        if (!run_phase[i]) continue;
        const auto& p = ph[i];
        const uint32_t rounds = p.N / 512;
        if (!o.stage) {
            // y: [round][b][t][32] -> [t][round * 512 + b * 32 + r]
            if (L == 1) {
                seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg, S2MM, IT3, bd_10, it_channel_0,
                    {0, 0, 0, (uint32_t)p.y_off}, {1, 1, 1, (uint32_t)p.N}, {0, 0, 0, 1}, -1, 0, true);
            } else {
                seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg, S2MM, IT3, bd_10, it_channel_0,
                    {0, 0, 0, (uint32_t)p.y_off}, {rounds, 16, (uint32_t)L, 32}, {512, 32, (uint32_t)p.N, 1}, -1, 0, true);
            }
        } else {
            // the MT4 stage already made each round token-major [t][512]: [round][t][512] -> [t][round * 512 + c]
            if (L == 1) {
                seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg, S2MM, IT4, bd_10, it_channel_0,
                    {0, 0, 0, (uint32_t)p.y_off}, {1, 1, 1, (uint32_t)p.N}, {0, 0, 0, 1}, -1, 0, true);
            } else {
                seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg, S2MM, IT4, bd_10, it_channel_0,
                    {0, 0, 0, (uint32_t)p.y_off}, {rounds, 1, (uint32_t)L, 512}, {512, 0, (uint32_t)p.N, 1}, -1, 0, true);
            }
        }
        // x: [t][K] as one packet to the phase's landing channel
        seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT3, bd_0, it_channel_0,
            {0, 0, 0, (uint32_t)p.x_off}, {1, 1, 1, (uint32_t)(L * p.K)}, {0, 0, 0, 1}, p.pkt, 0, true, aggressive_cache);
        layer._move_weights(seq, p.N, p.K, p.w_offset);
        seq->npu_dma_wait(o.stage ? IT4 : IT3, S2MM, it_channel_0);
        seq->npu_dma_wait(IT3, MM2S, it_channel_0);
    }
    seq->cmds2seq();

    // ---- run ----
    std::vector<double> us;
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
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            header_print("ERROR", std::string("proj8: dispatch threw after ") + std::to_string((int)ms) + " ms: " + e.what());
            return 3;
        }
        double d = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (st != ERT_CMD_STATE_COMPLETED) {
            header_print("ERROR", "proj8: dispatch did not complete after " + std::to_string((int)(d / 1000)) + " ms, state " + std::to_string((int)st));
            return 3;
        }
        us.push_back(d);
    }
    y_bo.sync_from_device();
    for (size_t i = 0; i < ph.size(); i++) {
        if (!run_phase[i]) continue;
        std::ofstream f(o.dir + "/y_" + ph[i].name + ".bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(y_bo.data() + ph[i].y_off), (size_t)L * ph[i].N * 2);
    }
    std::sort(us.begin(), us.end());
    std::string line;
    for (double v : us) line += " " + std::to_string((int)v);
    header_print("info", "proj8 dispatch us (sorted):" + line + "  min " + std::to_string((int)us.front()));
    std::cout << "proj8 layer " << o.layer << " rows " << L << " phases " << o.phases << " min_us " << (int)us.front()
              << " median_us " << (int)us[us.size() / 2] << std::endl;
    return 0;
}

#endif
