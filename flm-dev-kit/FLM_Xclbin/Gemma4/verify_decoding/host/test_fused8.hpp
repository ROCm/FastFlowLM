// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __TEST_FUSED8_HPP__
#define __TEST_FUSED8_HPP__
// fused8: host driver of the whole 8-row verify layer (plan S6.11/S6.12, iron/fused8_aie.py) and the verify_layer_hook of the harness.
//
// Per dispatch (one layer, 8 tokens at positions cursor..cursor+7):
//   x BO  [8][D] layer input rows | [8*DHg + 2 DHg] global RoPE table | [8*DHs + 2 DHs] SWA RoPE table | [8][256] pli | [D + 32] w_post_pli, scalar
//   rms   the layer's 4 norm vectors (decoding_layer::rms_weights_buffer, unchanged)
//   proj  the layer's Q4 / bf16 weights (decoding_layer::all_proj_weight_buffer, unchanged DDR layout)
//   y BO  [8][D] layer output rows (stage full: IT5.S2MM0, nople: IT3.S2MM0), stage attn: [8][DQ] attention output rows
//   kv    global: the decoding_layer cache (K plane | V plane, row = position); SWA: a LINEAR window buffer built from the engine ring for this
//         dispatch (row = position - S, S = floor16(max(0, cursor - 511))), the appended rows are copied back to the ring slots afterwards
//         (notes/attn8_L8_results.md: the mirrored ring of plan D6 was replaced by a linear window buffer with host compaction)
// The host computes the per layer input (pli) of the 8 tokens (tools/ple8_ref.py pli_row, the roundings of kernels/proj_layer_embedding.cc).
#include "decoding_layer.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <vector>
#include <chrono>

namespace f8 {

inline npu_xclbin_manager* g_mgr = nullptr;
inline std::string g_xclbin;                    // build/fused8_<stage>_L8/xclbins/fused8.xclbin
inline std::string g_stage = "full";            // attn | nople | full
inline int g_timeout_ms = 20000;
inline double g_last_us = 0;                    // wall time of the last dispatch
inline bool g_debug = false;

constexpr int PKT_SHIM_TO_RMS = 26, PKT_RMS_WEIGHTS = 11, PKT_ROPE_TABLE = 9, PKT_SWA_ROPE_TABLE = 10, PKT_PLI = 27, PKT_PLE_CONST = 28;
constexpr int PKT_O_GLOBAL = 17, PKT_O_SWA = 18;

inline std::map<std::string, uint32_t> read_rtp_file(const std::string& fn) {
    std::map<std::string, uint32_t> m;
    std::ifstream f(fn);
    std::string name; uint32_t addr;
    while (f >> name >> addr) m[name] = addr;
    return m;
}

struct fused_app {
    npu_app app;
    std::map<std::string, uint32_t> rtp;
};

inline fused_app& app_for(decoding_layer& layer) {
    static std::unique_ptr<fused_app> fa;
    static npu_app_manager* mgr = nullptr;
    if (!fa) {
        if (!g_mgr) throw std::runtime_error("fused8: no xclbin manager");
        mgr = g_mgr->register_xclbin(g_xclbin);
        fa.reset(new fused_app{mgr->create_app(), {}});
        const std::string dir = g_xclbin.substr(0, g_xclbin.find_last_of('/'));
        fa->rtp = read_rtp_file(dir + "/fused8_rtp_addresses.txt");
    }
    return *fa;
}

inline float bf2f(bf16 v) { return (float)v; }

/// pli of one token for one layer (tools/ple8_ref.py pli_row; same roundings as the decode PLI tile)
inline void pli_row(decoding_layer& layer, int tok, bf16* out /*[256]*/) {
    const int P = layer.hidden_size_per_layer_input, D = layer.D;
    // memo for the hammer test (same tokens every dispatch).  Keyed by the LAYER INDEX, not the object: the harness destroys and
    // re-creates decoding_layer objects, and a new layer can land at the address of an old one.
    static std::map<std::pair<int, int>, std::vector<bf16>> memo;
    auto mk = std::make_pair(layer.layer_idx, tok);
    if (auto it = memo.find(mk); it != memo.end()) { std::memcpy(out, it->second.data(), P * sizeof(bf16)); return; }
    struct store_t { std::map<std::pair<int, int>, std::vector<bf16>>& m; std::pair<int, int> k; bf16* o; int n;
                     ~store_t() { m[k].assign(o, o + n); } } store{memo, mk, out, P};
    buffer<bf16> x0 = layer.embedding(tok);
    buffer<bf16> pl = layer.pli_embed->forward(tok);
    const bf16* x0pl = pl.data() + (size_t)layer.layer_idx * P;
    // W_plmp [row block][k][32] bf16 inside the layer's proj BO
    const bf16* Wp = reinterpret_cast<const bf16*>(layer.all_proj_weight_buffer.data()) + layer.w_per_layer_input_proj_offset;
    std::vector<float> t(P);
    for (int rb = 0; rb < P / 32; rb++) {
        float acc[32] = {};   // the 32 rows of a block together (contiguous weights); each row still sums k in order, as before
        for (int k = 0; k < D; k++) {
            const float xk = bf2f(x0.data()[k]);
            const bf16* w = Wp + ((size_t)rb * D + k) * 32;
            for (int r = 0; r < 32; r++) acc[r] += bf2f(w[r]) * xk;
        }
        for (int r = 0; r < 32; r++) t[rb * 32 + r] = bf2f((bf16)acc[r]);
    }
    const bf16* wpn = layer.rope_rms_buffer.data() + 3 * (size_t)layer.DH + P;   // pli norm weights
    float ms = 0.f;
    for (int i = 0; i < P; i++) { t[i] = bf2f((bf16)(t[i] * 0.01976423537605237f)); ms += t[i] * t[i]; }
    ms /= P;
    const float rs = 1.0f / std::sqrt(ms + 1e-6f);
    for (int i = 0; i < P; i++) {
        float v = bf2f((bf16)(t[i] * rs * bf2f(wpn[i])));
        v = bf2f((bf16)(v + bf2f(x0pl[i])));
        out[i] = (bf16)(v * 0.7071067811865476f);
    }
}

struct kvlin_t { buffer<bf16> bo; size_t plane_rows = 0; };

/// one dispatch of the fused layer.  Returns 0 on success.
inline int run(decoding_layer& layer, int cursor, const std::vector<buffer<bf16>>& x_rows, const std::vector<int>& ids,
               std::vector<buffer<bf16>>& out_rows) {
    constexpr int L = 8, L_CHUNK = 16, W = 512;
    const bool swa = layer.is_swa_layer, skip = layer.shall_skip_kv;
    const int D = layer.D, DQ = layer.DQ, DH = layer.DH, I = layer.intermediate_size, P = layer.hidden_size_per_layer_input;
    const size_t DKv = (size_t)layer.DK;                      // both kv heads of one row
    const int NQ = 8, NKV = 2;
    const int DHg = 512, DHs = 256;
    auto& fa = app_for(layer);
    auto need = [&](const std::string& n) { auto it = fa.rtp.find(n); if (it == fa.rtp.end()) throw std::runtime_error("fused8 rtp file has no " + n); return it->second; };
    const bool st_attn = g_stage == "attn", st_nople = g_stage == "nople", st_full = g_stage == "full";

    // ---- BO layouts ----
    const size_t tabg_n = (size_t)L * DHg + 2 * DHg, tabs_n = (size_t)L * DHs + 2 * DHs;
    const size_t x_off = 0, tg_off = (size_t)L * D, ts_off = tg_off + tabg_n, pli_off = ts_off + tabs_n, wc_off = pli_off + (size_t)L * P;
    const size_t x_total = (wc_off + D + 32 + 8191) / 8192 * 8192;
    static buffer<bf16> x_bo, y_bo; static size_t x_bo_n = 0, y_bo_n = 0;
    const size_t y_elems = st_attn ? (size_t)L * DQ : (size_t)L * D;
    const size_t y_total = (y_elems + 8191) / 8192 * 8192;
    if (x_bo_n != x_total) { x_bo = layer.layer_app.create_bo_buffer<bf16>(x_total); x_bo_n = x_total; }
    if (y_bo_n != y_total) { y_bo = layer.layer_app.create_bo_buffer<bf16>(y_total); y_bo_n = y_total; }
    x_bo.memset((bf16)0.0f);
    y_bo.memset((bf16)0.0f);
    bf16* xp = x_bo.data();
    for (int t = 0; t < L; t++) std::memcpy(xp + x_off + (size_t)t * D, x_rows[t].data(), D * sizeof(bf16));
    // RoPE tables of the layer's kind: [(cos | sin) x 8 | q_norm | k_norm]
    {
        const size_t off = swa ? ts_off : tg_off;
        for (int t = 0; t < L; t++) {
            layer._set_rope_weights(cursor + t);
            std::memcpy(xp + off + (size_t)t * DH, layer.rope_rms_buffer.data(), DH * sizeof(bf16));
        }
        std::memcpy(xp + off + (size_t)L * DH, layer.rope_rms_buffer.data() + DH, 2 * DH * sizeof(bf16));   // q_norm, k_norm
        layer._set_rope_weights(cursor);
    }
    // pli rows and the PLE constants
    if (st_full) {
        for (int t = 0; t < L; t++) pli_row(layer, ids[t], xp + pli_off + (size_t)t * P);
        const bf16* rr = layer.rope_rms_buffer.data() + 3 * (size_t)DH;
        std::memcpy(xp + wc_off, rr + 2 * P, D * sizeof(bf16));                    // w_post_pli
        xp[wc_off + D] = rr[2 * P + D];                                            // layer_scalar
    }
    x_bo.sync_to_device();
    y_bo.sync_to_device();

    // ---- KV cache as the kernels see it ----
    int S = 0, base = 0, first_row = 0, n_rows = cursor + L;
    size_t plane_rows = 0, v_off = 0;
    static buffer<bf16> kvlin; static size_t kvlin_n = 0;
    bf16* kvp = nullptr;
    int app_row0 = cursor;                                                       // buffer row of the first appended token
    if (!swa) {
        v_off = layer.kv_cache_size / 2;
        kvp = layer.kv_cache.data();
    } else {
        S = cursor - (W - 1) > 0 ? cursor - (W - 1) : 0;
        S = S / L_CHUNK * L_CHUNK;
        base = S; first_row = 0; n_rows = cursor + L - S;
        plane_rows = (size_t)((n_rows + L_CHUNK - 1) / L_CHUNK * L_CHUNK) + 64;
        v_off = plane_rows * DKv;
        if (kvlin_n != 2 * plane_rows * DKv) { kvlin = layer.layer_app.create_bo_buffer<bf16>(2 * plane_rows * DKv); kvlin_n = 2 * plane_rows * DKv; }
        kvlin.memset((bf16)0.0f);
        layer.kv_cache.sync_from_device();
        // ring -> linear: position p lives in ring slot p % W (valid for the last W positions written)
        for (int p = S; p < cursor + (skip ? L : 0); p++) {
            for (int pl = 0; pl < 2; pl++)
                std::memcpy(kvlin.data() + pl * v_off + (size_t)(p - base) * DKv,
                            layer.kv_cache.data() + pl * (layer.kv_cache_size / 2) + (size_t)(p % W) * DKv, DKv * sizeof(bf16));
        }
        kvlin.sync_to_device();
        kvp = kvlin.data();
        app_row0 = cursor - base;
    }
    const int L_max = n_rows;
    const int chunks = (n_rows + L_CHUNK - 1) / L_CHUNK;
    // attnx attention (iron/attnx.py, FUSED8_ATTN=x builds): K / V streamed head-major ONCE, rows rounded up to G (32 global, 16 SWA);
    // the rows past the valid range must be finite (zero), the cache must hold the rounded range.
    const bool attnx = fa.rtp.count("RTP_attnx_score_3_4") != 0;
    const int ax_G = swa ? 16 : 32, ax_kps = swa ? 16 : 8;
    const int ax_rows = (n_rows + ax_G - 1) / ax_G * ax_G;
    const int ax_nsub = ax_rows / ax_kps;
    if (attnx && !swa && (size_t)ax_rows > (size_t)layer.MAX_L) throw std::runtime_error("fused8/attnx: replay rows exceed the cache (MAX_L)");

    // ---- instruction stream ----
    npu_sequence* seq = fa.app.seq();
    constexpr size_t CT_lock_address_base = 0x000001F000;
    seq->clear_cmds();
    const uint32_t a_swa = need("RTP_PROJ_IS_SWA_BUFFER_2_0"), a_skip = need("RTP_PROJ_SKIP_KV_BUFFER_2_0");
    for (int i = 0; i < 16; i++) {
        seq->rtp_write(layer.proj_tiles[i], a_swa, swa ? 1 : 0);
        seq->rtp_write(layer.proj_tiles[i], a_skip, skip ? 1 : 0);
        seq->rtp_write(layer.proj_tiles[i], CT_lock_address_base + 16 * 6, 1);   // rtp_available_lock
    }
    seq->rtp_write(swa ? layer.swa_rope_ct : layer.rope_ct, need(swa ? "RTP_SWA_RoPE_SKIP_KV_5_3" : "RTP_RoPE_SKIP_KV_4_3"), skip ? 1 : 0);
    if (attnx) {
        const int hi0 = swa ? cursor - S : cursor;
        const int u0 = swa ? cursor + 1 - W - S : -(1 << 30);
        const npu_tiles score_tiles[2] = {CT14, CT34};
        const char* score_names[2] = {"RTP_attnx_score_3_4", "RTP_attnx_score_5_4"};
        for (int sc = 0; sc < 2; sc++) {
            const uint32_t a = need(score_names[sc]);
            const uint32_t w[5] = {(uint32_t)ax_nsub, (uint32_t)hi0, (uint32_t)u0, (uint32_t)(swa ? 1 : 0), (uint32_t)(4 * sc)};
            for (int i = 0; i < 5; i++) seq->rtp_write(score_tiles[sc], a + 4 * i, w[i]);
        }
        const npu_tiles pv_tiles[4] = {CT02, CT12, CT22, CT32};
        for (int j = 0; j < 4; j++) {
            const uint32_t a = need("RTP_attnx_pv_" + std::to_string(2 + j) + "_2");
            const uint32_t w[3] = {(uint32_t)ax_nsub, (uint32_t)(swa ? 1 : 0), (uint32_t)j};
            for (int i = 0; i < 3; i++) seq->rtp_write(pv_tiles[j], a + 4 * i, w[i]);
        }
    } else {
        const npu_tiles qk_tile = swa ? layer.swa_attn_qk_tile : layer.attn_qk_tile;
        const npu_tiles kv_tile = swa ? layer.swa_attn_kv_tile : layer.attn_kv_tile;
        const uint32_t qk_addr = need(swa ? "RTP_L_swa_attn_qk_core_4_2" : "RTP_L_attn_qk_core_2_2");
        seq->rtp_write(qk_tile, qk_addr, L_max);
        if (swa) {   // swa qk: L[1] = hi0 = ctx - S, L[2] = u0 = ctx + 1 - W - S (kernels/swa_attn_qk.cc)
            seq->rtp_write(qk_tile, qk_addr + 4, (uint32_t)(cursor - S));
            seq->rtp_write(qk_tile, qk_addr + 8, (uint32_t)(cursor + 1 - W - S));
        }
        seq->rtp_write(kv_tile, need(swa ? "RTP_L_swa_attn_kv_core_5_2" : "RTP_L_attn_kv_core_3_2"), L_max);
    }
    const int x_arg = layer.x_arg_id, w_arg = layer.proj_arg_id, rms_arg = layer.rms_arg_id, y_arg = layer.rope_rms_arg_id, kv_arg = layer.kv_cache_arg_id;
    // Task-completion token ledger.  Every shim BD submitted with a token (all S2MM, and MM2S with issue_token) must be consumed by exactly
    // one npu_dma_wait on the same (tile, direction, channel) inside this dispatch.  A token left over is not dropped: it stays in the
    // shim's task-completion queue across dispatches, and after ~255 dispatches the queue is full and the array stops answering
    // (notes/hang_rootcause.md).  The ledger asserts the balance every time the stream is generated.
    std::map<int, int> tokens;
    auto tkey = [](npu_tiles t, dma_direction d, npu_it_channel c) { return ((int)t << 8) | ((int)d << 4) | (int)c; };
    auto issued = [&](npu_tiles t, dma_direction d, npu_it_channel c) { tokens[tkey(t, d, c)]++; };
    auto waited = [&](npu_tiles t, dma_direction d, npu_it_channel c) { seq->npu_dma_wait(t, d, c); tokens[tkey(t, d, c)]--; };
    // inputs of the norm tile and the RoPE tile
    seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT3, bd_0, it_channel_0,
        {0, 0, 0, (uint32_t)x_off}, {1, 1, 1, (uint32_t)(L * D)}, {0, 0, 0, 1}, PKT_SHIM_TO_RMS, 0, true, aggressive_cache);
    issued(IT3, MM2S, it_channel_0);
    seq->npu_dma_memcpy_nd(sizeof(bf16), rms_arg, MM2S, IT3, bd_1, it_channel_1,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)(4 * D)}, {0, 0, 0, 1}, PKT_RMS_WEIGHTS, 0, true, aggressive_cache);
    issued(IT3, MM2S, it_channel_1);
    if (!swa) {
        seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT3, bd_2, it_channel_0,
            {0, 0, 0, (uint32_t)tg_off}, {1, 1, 1, (uint32_t)tabg_n}, {0, 0, 0, 1}, PKT_ROPE_TABLE, 0, true, no_cache);
        issued(IT3, MM2S, it_channel_0);
    } else {
        seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT3, bd_3, it_channel_1,
            {0, 0, 0, (uint32_t)ts_off}, {1, 1, 1, (uint32_t)tabs_n}, {0, 0, 0, 1}, PKT_SWA_ROPE_TABLE, 0, true, no_cache);
        issued(IT3, MM2S, it_channel_1);
    }
    // outputs
    if (st_full) {
        seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg, S2MM, IT5, bd_10, it_channel_0,
            {0, 0, 0, 0}, {1, 1, 1, (uint32_t)(L * D)}, {0, 0, 0, 1}, -1, 0, true);
        issued(IT5, S2MM, it_channel_0);
    } else {
        seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg, S2MM, IT3, bd_10, it_channel_0,
            {0, 0, 0, 0}, {1, 1, 1, (uint32_t)y_elems}, {0, 0, 0, 1}, -1, 0, true);
        issued(IT3, S2MM, it_channel_0);
    }
    if (st_full) {
        // PLE inputs (memtile MT5 feed) and the two small weight matrices (to G and U)
        seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT4, bd_0, it_channel_0,
            {0, 0, 0, (uint32_t)pli_off}, {1, 1, 1, (uint32_t)(L * P)}, {0, 0, 0, 1}, PKT_PLI, 0, true, no_cache);
        issued(IT4, MM2S, it_channel_0);
        seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT4, bd_1, it_channel_1,
            {0, 0, 0, (uint32_t)wc_off}, {1, 1, 1, (uint32_t)(D + 32)}, {0, 0, 0, 1}, PKT_PLE_CONST, 0, true, no_cache);
        issued(IT4, MM2S, it_channel_1);
        seq->npu_dma_memcpy_nd(sizeof(bf16), w_arg, MM2S, IT5, bd_14, it_channel_0,
            {0, 0, 0, (uint32_t)layer.w_inp_gate_offset}, {1, 1, 1, (uint32_t)(P * D)}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
        issued(IT5, MM2S, it_channel_0);
        seq->npu_dma_memcpy_nd(sizeof(bf16), w_arg, MM2S, IT5, bd_15, it_channel_1,
            {0, 0, 0, (uint32_t)layer.w_inp_up_offset}, {1, 1, 1, (uint32_t)(P * D)}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
        issued(IT5, MM2S, it_channel_1);
    }
    // qkv weights, then the KV append (needs the qkv projection and RoPE to run: the weights are streaming), then the replay
    layer._move_weights(seq, skip ? DQ : DQ + layer.DK + layer.DV, D, layer.w_qkv_offset);
    const npu_it_channel kvch = swa ? it_channel_1 : it_channel_0;
    if (!skip) {
        for (int pl = 0; pl < 2; pl++) {
            seq->npu_dma_memcpy_nd(sizeof(bf16), kv_arg, S2MM, IT2, pl == 0 ? bd_0 : bd_1, kvch,
                {0, 0, 0, (uint32_t)((size_t)pl * v_off + (size_t)app_row0 * DKv)}, {1, (uint32_t)NKV, (uint32_t)L, (uint32_t)DH}, {0, (uint32_t)DH, (uint32_t)DKv, 1}, -1, 0, true);
            issued(IT2, S2MM, kvch);
            waited(IT2, S2MM, kvch);
        }
    }
    for (int pl = 0; pl < 2; pl++) {
        if (attnx)   // K (ch0) / V (ch1) once, head-major: repeat h (stride DH) x [rows / G][G][DH], circuits (no packet header)
            seq->npu_dma_memcpy_nd(sizeof(bf16), kv_arg, MM2S, IT2, pl == 0 ? bd_8 : bd_9, pl == 0 ? it_channel_0 : it_channel_1,
                {0, 0, 0, (uint32_t)((size_t)pl * v_off + (size_t)first_row * DKv)},
                {2, (uint32_t)(ax_rows / ax_G), (uint32_t)ax_G, (uint32_t)DH}, {(uint32_t)DH, (uint32_t)(ax_G * DKv), (uint32_t)DKv, 1},
                -1, 0, true, no_cache);
        else
            seq->npu_dma_memcpy_nd(sizeof(bf16), kv_arg, MM2S, IT2, pl == 0 ? bd_8 : bd_9, pl == 0 ? it_channel_0 : it_channel_1,
                {0, 0, 0, (uint32_t)((size_t)pl * v_off + (size_t)first_row * DKv)},
                {(uint32_t)L, (uint32_t)chunks, (uint32_t)L_CHUNK, (uint32_t)DKv}, {0, (uint32_t)(L_CHUNK * DKv), (uint32_t)DKv, 1},
                swa ? 13 : 12, 0, true, no_cache);
        issued(IT2, MM2S, pl == 0 ? it_channel_0 : it_channel_1);
    }
    if (!st_attn) {
        layer._move_weights(seq, D, DQ, layer.w_o_offset);
        layer._move_weights(seq, 2 * I, D, layer.w_up_offset);
        layer._move_weights(seq, D, I, layer.w_down_offset);
    }
    // waits: the output, the KV replay, the shim senders -- one wait per issued token (the ledger below checks it)
    for (auto& kv : tokens) {
        const npu_tiles t = (npu_tiles)(kv.first >> 8); const dma_direction d = (dma_direction)((kv.first >> 4) & 0xF); const npu_it_channel c = (npu_it_channel)(kv.first & 0xF);
        while (kv.second > 0) waited(t, d, c);
    }
    for (auto& kv : tokens) if (kv.second != 0) throw std::runtime_error("fused8: task-completion tokens unbalanced on shim key " + std::to_string(kv.first) + ": " + std::to_string(kv.second));
    seq->cmds2seq();

    // ---- dispatch ----
    auto t0 = std::chrono::steady_clock::now();
    ert_cmd_state st = ERT_CMD_STATE_NEW;
    try {
        xrt::run run = fa.app.create_run(x_bo, layer.all_proj_weight_buffer, layer.rms_weights_buffer, y_bo, swa ? kvlin : layer.kv_cache);
        run.start();
        st = run.wait(std::chrono::milliseconds(g_timeout_ms));
    } catch (const std::exception& e) {
        std::cerr << "ERROR fused8: dispatch threw: " << e.what() << std::endl;
        return 3;
    }
    g_last_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    if (st != ERT_CMD_STATE_COMPLETED) { std::cerr << "ERROR fused8: dispatch state " << (int)st << " after " << (int)(g_last_us / 1000) << " ms" << std::endl; return 3; }

    if (std::getenv("FUSED8_PERF")) std::printf("fused8 perf layer %d (%s%s) cursor %d: dispatch %.0f us\n", layer.layer_idx, swa ? "swa" : "glob", skip ? "-skip" : "", cursor, g_last_us);
    y_bo.sync_from_device();
    out_rows.resize(L);
    for (int t = 0; t < L; t++) {
        out_rows[t] = buffer<bf16>(st_attn ? DQ : D);
        std::memcpy(out_rows[t].data(), y_bo.data() + (size_t)t * (st_attn ? DQ : D), (size_t)(st_attn ? DQ : D) * sizeof(bf16));
    }
    if (g_debug) {   // repeat check: report which rows differ from the previous dispatch of the same (layer, cursor)
        static std::map<std::pair<int,int>, std::vector<uint16_t>> prev; static std::map<std::pair<int,int>, int> cnt;
        const int n = st_attn ? DQ : D;
        std::vector<uint16_t> cur((size_t)L * n);
        for (int t = 0; t < L; t++) std::memcpy(cur.data() + (size_t)t * n, out_rows[t].data(), (size_t)n * 2);
        auto key = std::make_pair(layer.layer_idx, cursor);
        auto it = prev.find(key);
        if (it != prev.end()) {
            int rows_diff = 0; size_t first = (size_t)-1, nd = 0;
            for (int t = 0; t < L; t++) { bool d = false; for (int j = 0; j < n; j++) if (cur[(size_t)t * n + j] != it->second[(size_t)t * n + j]) { d = true; nd++; if (first == (size_t)-1) first = (size_t)t * n + j; } rows_diff += d; }
            cnt[key]++;
            if (nd) std::printf("fused8 REPEAT DIFF layer %d cursor %d dispatch #%d vs previous: %zu elems in %d rows, first at row %zu col %zu\n", layer.layer_idx, cursor, cnt[key], nd, rows_diff, first / n, first % n);
        }
        prev[key] = cur;
        if (const char* dd = std::getenv("FUSED8_DUMP")) {
            static int seq_no = 0;
            std::ofstream f(std::string(dd) + "/out_l" + std::to_string(layer.layer_idx) + "_c" + std::to_string(cursor) + "_" + std::to_string(seq_no++) + ".bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(cur.data()), cur.size() * 2);
        }
    }
    if (g_debug) {
        const int n = st_attn ? DQ : D;
        for (int t = 0; t < L; t++) {
            double ss = 0; int nan = 0, zero = 0;
            for (int j = 0; j < n; j++) { float v = bf2f(out_rows[t].data()[j]); if (std::isnan(v) || std::isinf(v)) nan++; else { ss += (double)v * v; if (v == 0.f) zero++; } }
            std::printf("fused8 dbg row %d: rms %.4f nan %d zero %d (of %d)\n", t, std::sqrt(ss / n), nan, zero, n);
        }
    }
    if (g_debug && !skip) {   // appended K/V rows: first dispatch vs the previous one of the same (layer, cursor)
        static std::map<std::pair<int,int>, std::vector<uint16_t>> kvprev;
        if (swa) kvlin.sync_from_device(); else layer.kv_cache.sync_from_device();
        std::vector<uint16_t> kc;
        for (int pl = 0; pl < 2; pl++)
            for (int t = 0; t < L; t++) {
                const bf16* r = (swa ? kvlin.data() : layer.kv_cache.data()) + (size_t)pl * v_off + (size_t)(app_row0 + t) * DKv;
                kc.insert(kc.end(), (const uint16_t*)r, (const uint16_t*)r + DKv);
            }
        auto key = std::make_pair(layer.layer_idx, cursor);
        auto it = kvprev.find(key);
        if (it != kvprev.end()) {
            size_t nd = 0; for (size_t i = 0; i < kc.size(); i++) nd += kc[i] != it->second[i];
            if (nd) std::printf("fused8 KV REPEAT DIFF layer %d cursor %d: %zu of %zu elems\n", layer.layer_idx, cursor, nd, kc.size());
        }
        kvprev[key] = kc;
    }
    if (!skip) {
        if (swa) {
            // linear -> ring: the 8 appended rows go to the engine ring slots
            kvlin.sync_from_device();
            for (int t = 0; t < L; t++)
                for (int pl = 0; pl < 2; pl++)
                    std::memcpy(layer.kv_cache.data() + pl * (layer.kv_cache_size / 2) + (size_t)((cursor + t) % W) * DKv,
                                kvlin.data() + pl * v_off + (size_t)(app_row0 + t) * DKv, DKv * sizeof(bf16));
            layer.kv_cache.sync_to_device();
        } else {
            layer.kv_cache.sync_from_device();
        }
    }
    return 0;
}


// ------------------------------------------------------------------------------------------------------------------------------------
// chain8: all layers with real weights.  For each sequence: fill c0 positions with sequential decode (the oracle path), snapshot the KV caches,
// run 8 more tokens (a) as 8 sequential decode passes through every layer, (b) as ONE fused8 dispatch per layer; compare the final hidden rows
// (cosine) and the argmax of the logits (final norm, tied lm_head) of every row.  Skip layers share their owner layer's cache buffer.
struct chain_opts { std::vector<int> cursors = {0, 24, 100}; int seqs = 3; int maxl = 1536; int layers = 42; };

inline int test_chain8(model_list::model_t model, npu_app_manager* mgr, LM_Config& cfg, Q4NX& q4nx, const chain_opts& o) {
    using clock = std::chrono::steady_clock;
    const int NL = o.layers;
    std::vector<std::unique_ptr<decoding_layer>> Ls;
    for (int i = 0; i < NL; i++) { Ls.push_back(std::make_unique<decoding_layer>(model, cfg, mgr, i, o.maxl)); Ls.back()->load_weights(q4nx); }
    const int D = Ls[0]->D;
    // skip layers share the owner's cache buffer (shallow buffer copy = same BO)
    for (int i = 0; i < NL; i++) if (Ls[i]->shall_skip_kv) Ls[i]->kv_cache = Ls[Ls[i]->final_kv_cache_layer]->kv_cache;
    // final norm + tied lm_head on the host
    buffer<bf16> wfin; q4nx.load_weights(wfin, "model.norm.weight");
    const int V = 262144;
    std::vector<float> E((size_t)V * D);
    for (int v = 0; v < V; v++) { buffer<bf16> r = Ls[0]->embedding(v); for (int j = 0; j < D; j++) E[(size_t)v * D + j] = (float)r.data()[j]; }
    auto logits_argmax = [&](const bf16* h, double& margin) {
        std::vector<float> x(D); double ms = 0; for (int j = 0; j < D; j++) { float f = (float)h[j]; ms += (double)f * f; x[j] = f; }
        const float rs = 1.0f / std::sqrt((float)(ms / D) + 1e-6f);
        for (int j = 0; j < D; j++) x[j] = x[j] * rs * (float)wfin.data()[j];
        int best = 0; float b1 = -1e30f, b2 = -1e30f;
        for (int v = 0; v < V; v++) { float a = 0; const float* e = &E[(size_t)v * D]; for (int j = 0; j < D; j++) a += e[j] * x[j]; if (a > b1) { b2 = b1; b1 = a; best = v; } else if (a > b2) b2 = a; }
        margin = b1 - b2; return best;
    };
    auto ids_for = [&](int n, uint64_t seed) { std::vector<int> ids(n); uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1; for (int i = 0; i < n; i++) { s = s * 6364136223846793005ull + 1442695040888963407ull; ids[i] = 1000 + (int)((s >> 33) % 200000); } return ids; };

    int noise_argmax_ok = 0, noise_rows = 0; double noise_min_cos = 1; int total_rows = 0, argmax_ok = 0, bad_cos = 0; double min_cos = 1; double t_fused_sum = 0, t_oracle_sum = 0; int nseq_done = 0;
    std::vector<double> fused_layer_us[4];   // per layer kind: 0 swa, 1 glob, 2 swa-skip, 3 glob-skip
    for (int sq = 0; sq < o.seqs; sq++) {
        const int c0 = o.cursors[sq % o.cursors.size()];
        std::vector<int> ids = ids_for(c0 + 8, 1234 + sq);
        // reset the caches (rows past the context must be zero) and the context lengths
        for (int i = 0; i < NL; i++) {
            if (!Ls[i]->shall_skip_kv) { Ls[i]->kv_cache.memset((bf16)0.0f); Ls[i]->kv_cache.sync_to_device(); }
            Ls[i]->set_context_length(0);
        }
        const bool trace = std::getenv("CHAIN_TRACE") != nullptr;
        std::vector<std::vector<buffer<bf16>>> layer_ref;   // [layer][token] (first oracle pass only)
        bool store_layers = false;
        auto decode_token = [&](int t, std::vector<buffer<bf16>>* final_out, double* us) {
            buffer<bf16> x = Ls[0]->embedding(ids[t]);
            buffer<bf16> cur(D); std::memcpy(cur.data(), x.data(), D * sizeof(bf16));
            for (int i = 0; i < NL; i++) {
                buffer<bf16> out = (*Ls[i])(cur, ids[t]);
                if (us) *us += Ls[i]->last_dispatch_us;
                std::memcpy(cur.data(), out.data(), D * sizeof(bf16));
                if (store_layers) { if ((int)layer_ref.size() <= i) layer_ref.resize(i + 1); buffer<bf16> r(D); std::memcpy(r.data(), cur.data(), D * sizeof(bf16)); layer_ref[i].push_back(std::move(r)); }
            }
            if (final_out) { buffer<bf16> r(D); std::memcpy(r.data(), cur.data(), D * sizeof(bf16)); final_out->push_back(std::move(r)); }
        };
        int seeded = 0;
        if (std::getenv("CHAIN_SEED_FUSED")) {   // long contexts: seed with the fused design, 8 positions per pass (decode seeding is ~1 s per token here)
            for (; seeded + 8 <= c0; seeded += 8) {
                std::vector<buffer<bf16>> cur; for (int t = 0; t < 8; t++) { buffer<bf16> x = Ls[0]->embedding(ids[seeded + t]); buffer<bf16> r(D); std::memcpy(r.data(), x.data(), D * sizeof(bf16)); cur.push_back(std::move(r)); }
                std::vector<int> s8(ids.begin() + seeded, ids.begin() + seeded + 8);
                for (int i = 0; i < NL; i++) {
                    std::vector<buffer<bf16>> out;
                    if (run(*Ls[i], seeded, cur, s8, out) != 0) { std::printf("chain8: fused seeding failed at layer %d pos %d\n", i, seeded); return 3; }
                    cur = std::move(out);
                }
                for (int i = 0; i < NL; i++) Ls[i]->set_context_length(seeded + 8);
                if ((seeded / 8) % 250 == 249) { std::printf("chain8 seeding: %d / %d positions\n", seeded + 8, c0); std::fflush(stdout); }
            }
        }
        for (int t = seeded; t < c0; t++) decode_token(t, nullptr, nullptr);
        // snapshot every owner cache
        std::vector<std::vector<uint8_t>> snap(NL);
        for (int i = 0; i < NL; i++) if (!Ls[i]->shall_skip_kv) { Ls[i]->kv_cache.sync_from_device(); snap[i].assign((uint8_t*)Ls[i]->kv_cache.data(), (uint8_t*)Ls[i]->kv_cache.data() + Ls[i]->kv_cache.size() * sizeof(bf16)); }
        auto restore = [&]() { for (int i = 0; i < NL; i++) { if (!Ls[i]->shall_skip_kv) { std::memcpy(Ls[i]->kv_cache.data(), snap[i].data(), snap[i].size()); Ls[i]->kv_cache.sync_to_device(); } Ls[i]->set_context_length(c0); } };
        // (a) oracle: 8 sequential tokens through all layers
        std::vector<buffer<bf16>> ref; double us_o = 0;
        store_layers = trace;
        for (int t = 0; t < 8; t++) decode_token(c0 + t, &ref, &us_o);
        store_layers = false;
        restore();
        // (b) fused: one dispatch per layer for the 8 rows
        std::vector<buffer<bf16>> rows; for (int t = 0; t < 8; t++) { buffer<bf16> x = Ls[0]->embedding(ids[c0 + t]); buffer<bf16> r(D); std::memcpy(r.data(), x.data(), D * sizeof(bf16)); rows.push_back(std::move(r)); }
        std::vector<int> id8(ids.begin() + c0, ids.begin() + c0 + 8);
        double us_f = 0;
        for (int i = 0; i < NL; i++) {
            std::vector<buffer<bf16>> out;
            int rc = run(*Ls[i], c0, rows, id8, out);
            if (rc != 0) { std::printf("chain8: layer %d dispatch failed rc=%d\n", i, rc); return 3; }
            us_f += g_last_us;
            if (trace) {
                double mn = 1; int mrow = 0;
                for (int t = 0; t < 8; t++) {
                    double dot = 0, na = 0, nb = 0;
                    for (int j = 0; j < D; j++) { double a = (float)out[t].data()[j], b = (float)layer_ref[i][t].data()[j]; dot += a * b; na += a * a; nb += b * b; }
                    const double cs = dot / std::sqrt(na * nb + 1e-30); if (cs < mn) { mn = cs; mrow = t; }
                }
                std::printf("  trace seq %d layer %2d (%s%s): min cos %.5f at row %d\n", sq, i, Ls[i]->is_swa_layer ? "swa" : "glob", Ls[i]->shall_skip_kv ? "-skip" : "", mn, mrow);
            }
            const int kind = (Ls[i]->is_swa_layer ? 0 : 1) + (Ls[i]->shall_skip_kv ? 2 : 0);
            fused_layer_us[Ls[i]->is_swa_layer ? (Ls[i]->shall_skip_kv ? 2 : 0) : (Ls[i]->shall_skip_kv ? 3 : 1)].push_back(g_last_us); (void)kind;
            rows = std::move(out);
        }
        // noise floor of the oracle itself: a second sequential-decode pass from the same state
        restore();
        std::vector<buffer<bf16>> ref2; double us_o2 = 0;
        for (int t = 0; t < 8; t++) decode_token(c0 + t, &ref2, &us_o2);
        int ok_noise = 0; double mc_noise = 1;
        for (int t = 0; t < 8; t++) {
            double dot = 0, na = 0, nb = 0;
            for (int j = 0; j < D; j++) { double a = (float)ref2[t].data()[j], b = (float)ref[t].data()[j]; dot += a * b; na += a * a; nb += b * b; }
            mc_noise = std::min(mc_noise, dot / std::sqrt(na * nb + 1e-30));
            double m1, m2; ok_noise += logits_argmax(ref2[t].data(), m1) == logits_argmax(ref[t].data(), m2);
        }
        noise_argmax_ok += ok_noise; noise_rows += 8; noise_min_cos = std::min(noise_min_cos, mc_noise);
        // compare
        int ok_here = 0; double mc = 1;
        for (int t = 0; t < 8; t++) {
            double dot = 0, na = 0, nb = 0;
            for (int j = 0; j < D; j++) { double a = (float)rows[t].data()[j], b = (float)ref[t].data()[j]; dot += a * b; na += a * a; nb += b * b; }
            const double cs = dot / std::sqrt(na * nb + 1e-30); mc = std::min(mc, cs);
            double m1, m2; int a1 = logits_argmax(rows[t].data(), m1), a2 = logits_argmax(ref[t].data(), m2);
            total_rows++; if (a1 == a2) { argmax_ok++; ok_here++; }
            else std::printf("  seq %d row %d ARGMAX MISMATCH fused %d (margin %.3f) vs oracle %d (margin %.3f), hidden cos %.5f\n", sq, t, a1, m1, a2, m2, cs);
            if (cs < 0.995) std::printf("  seq %d row %d LOW hidden cos %.5f (rms fused %.2f oracle %.2f)\n", sq, t, cs, std::sqrt(na / D), std::sqrt(nb / D));
            if (cs < 0.99) bad_cos++;
        }
        min_cos = std::min(min_cos, mc);
        t_fused_sum += us_f; t_oracle_sum += us_o; nseq_done++;
        std::printf("chain8 seq %d c0 %3d: argmax %d/8, min hidden cos %.5f | fused 42 layers %.1f ms (dispatch only), oracle 8x42 decode %.1f ms (%.1f ms/token)\n",
                    sq, c0, ok_here, mc, us_f / 1000, us_o / 1000, us_o / 8000);
        std::fflush(stdout);
        restore();   // leave the state clean
    }
    const char* kn[4] = {"swa", "glob", "swa-skip", "glob-skip"};
    for (int k = 0; k < 4; k++) if (!fused_layer_us[k].empty()) {
        std::sort(fused_layer_us[k].begin(), fused_layer_us[k].end());
        std::printf("chain8 fused8 per-layer dispatch us (%s): median %.0f min %.0f n %zu\n", kn[k], fused_layer_us[k][fused_layer_us[k].size() / 2], fused_layer_us[k][0], fused_layer_us[k].size());
    }
    std::printf("chain8 NOISE FLOOR (decode vs a second decode pass from the same state): argmax %d/%d, min hidden cos %.5f\n", noise_argmax_ok, noise_rows, noise_min_cos);
    std::printf("chain8 SUMMARY: argmax %d/%d rows, min cos %.5f, rows with cos<0.99: %d; mean fused chain %.1f ms vs oracle 8 tokens %.1f ms (x%.2f faster)\n",
                argmax_ok, total_rows, min_cos, bad_cos, t_fused_sum / nseq_done / 1000, t_oracle_sum / nseq_done / 1000, t_oracle_sum / t_fused_sum);
    return (argmax_ok >= total_rows * 7 / 8 && bad_cos == 0) ? 0 : 1;
}



// ------------------------------------------------------------------------------------------------------------------------------------
// hammer8: many consecutive fused8 dispatches in ONE hardware context (the dispatch-count hang of notes/hang_rootcause.md).
// Round robin over (layer, cursor) pairs with fixed inputs; every output must equal the first output of its pair bit for bit.
struct hammer_opts { std::vector<int> layers = {0, 5, 40, 35}; std::vector<int> cursors = {64}; long count = 1000; int maxl = 4608; int report = 1000; };

inline int test_hammer8(model_list::model_t model, npu_app_manager* mgr, LM_Config& cfg, Q4NX& q4nx, const hammer_opts& o) {
    std::map<int, std::unique_ptr<decoding_layer>> Ls;
    for (int l : o.layers) { Ls[l] = std::make_unique<decoding_layer>(model, cfg, mgr, l, o.maxl); Ls[l]->load_weights(q4nx); }
    const int D = Ls.begin()->second->D;
    std::vector<int> ids(8); for (int t = 0; t < 8; t++) ids[t] = 2000 + 7919 * t;
    std::vector<buffer<bf16>> rows;
    for (int t = 0; t < 8; t++) { buffer<bf16> e = Ls.begin()->second->embedding(ids[t]); buffer<bf16> r(D); std::memcpy(r.data(), e.data(), D * sizeof(bf16)); rows.push_back(std::move(r)); }
    std::map<std::pair<int,int>, std::vector<uint16_t>> first, prev;
    // every dispatch starts from the same KV state (dispatches append rows, and the SWA ring slots of one cursor are read by another)
    std::map<int, std::vector<uint8_t>> snap;
    for (auto& kv : Ls) { kv.second->kv_cache.sync_from_device(); snap[kv.first].assign((uint8_t*)kv.second->kv_cache.data(), (uint8_t*)kv.second->kv_cache.data() + kv.second->kv_cache.size() * sizeof(bf16)); }
    auto t0 = std::chrono::steady_clock::now();
    long n = 0, diffs = 0, first_after_load_diffs = 0;
    std::map<std::pair<int,int>, std::vector<uint16_t>> seen;
    std::map<std::pair<int,int>, std::vector<double>> lat;   // dispatch latency per pair (min/median printed at the end)
    auto continue_report = [&](long i) {
        if (o.report > 0 && (i + 1) % o.report == 0) {
            std::printf("hammer8 %ld dispatches ok (%.1f s, last %.0f us), diffs %ld\n", i + 1,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), g_last_us, diffs);
            std::fflush(stdout);
        }
    };
    const size_t npairs = o.layers.size() * o.cursors.size();
    for (n = 0; n < o.count; n++) {
        const size_t k = (size_t)n % npairs;
        const int l = o.layers[k / o.cursors.size()], c = o.cursors[k % o.cursors.size()];
        decoding_layer& L = *Ls[l];
        std::memcpy(L.kv_cache.data(), snap[l].data(), snap[l].size()); L.kv_cache.sync_to_device();
        L.set_context_length(c);
        std::vector<buffer<bf16>> out;
        int rc = run(L, c, rows, ids, out);
        if (rc != 0) {
            std::printf("hammer8 FAIL: dispatch #%ld (0-based, layer %d cursor %d) rc=%d after %.1f s\n", n, l, c, rc,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            std::fflush(stdout);
            return 3;
        }
        std::vector<uint16_t> cur((size_t)8 * D);
        for (int t = 0; t < 8; t++) std::memcpy(cur.data() + (size_t)t * D, out[t].data(), (size_t)D * 2);
        auto key = std::make_pair(l, c);
        lat[key].push_back(g_last_us);
        // reference = the SECOND dispatch of each pair: the first dispatch after the xclbin load differs at rounding level for SWA
        // non-skip layers with > 32 chunks (an attention-kernel state initialised at load, see hang_rootcause.md); it is counted apart
        if (!seen.count(key)) { seen[key] = cur; prev[key] = cur; continue_report(n); continue; }
        if (!first.count(key)) { if (cur != seen[key]) first_after_load_diffs++; first[key] = cur; prev[key] = cur; continue_report(n); continue; }
        auto it = first.find(key);
        if (cur != it->second) {
            diffs++;
            size_t nd = 0; double dot = 0, na = 0, nb = 0, mx = 0;
            for (size_t i = 0; i < cur.size(); i++) {
                nd += cur[i] != it->second[i];
                uint32_t ua = (uint32_t)cur[i] << 16, ub = (uint32_t)it->second[i] << 16; float fa, fb; std::memcpy(&fa, &ua, 4); std::memcpy(&fb, &ub, 4);
                dot += (double)fa * fb; na += (double)fa * fa; nb += (double)fb * fb; mx = std::max(mx, (double)std::fabs(fa - fb));
            }
            std::printf("hammer8   cos(first, this) %.6f, max abs diff %.4g\n", dot / std::sqrt(na * nb + 1e-30), mx);
            const bool same_prev = prev.count(key) && prev[key] == cur;
            if (diffs <= 12) std::printf("hammer8 DIFF dispatch #%ld layer %d cursor %d: %zu of %zu elems differ from the first, %s the previous one\n", n, l, c, nd, cur.size(), same_prev ? "EQUAL to" : "differs from");
        }
        prev[key] = cur;
        continue_report(n);
    }
    for (auto& kv : lat) {
        auto v = kv.second; std::sort(v.begin(), v.end());
        std::printf("hammer8 latency layer %d cursor %d: min %.0f us  p10 %.0f  median %.0f  (n=%zu)\n", kv.first.first, kv.first.second,
                    v.front(), v[v.size() / 10], v[v.size() / 2], v.size());
    }
    std::printf("hammer8 DONE: %ld dispatches in one hardware context, %zu (layer,cursor) pairs, diffs vs the reference %ld, "
                "first-after-load diffs %ld, %.1f s\n", n, npairs, diffs, first_after_load_diffs,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return diffs == 0 ? 0 : 1;
}

} // namespace f8

#endif
