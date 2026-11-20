// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// attnx stage host (iron/attnx_aie.py, kernels/attnx_*.cc): rope + q park + the once-per-verify K/V attention.
//
// One attnx_case per (layer, cursor): it owns its BOs and reference inputs (tools/attn8_ref.py gen --stage attn) and generates its own
// instruction stream; several cases run in ONE hardware context (sequence / soak modes), which is what exposes on-chip state that leaks
// from one dispatch into the next.
//
// K / V are streamed head-major ONCE: [head][rows][DH] in sub-chunks of 4096 elements (global 8 keys, SWA 16 keys); the replayed row count is
// rounded up to G (32 global, 16 SWA) rows, the rows past the valid range must be finite (zero: the engine contract).
// Every token-issuing shim BD is balanced by exactly one wait (token ledger, notes/hang_rootcause.md).
#pragma once
#include <map>
#include <memory>

struct attnx_case {
    decoding_layer& layer;
    std::string dir;
    int ctx, poison_mask;
    std::map<std::string, uint32_t> rtp;
    // geometry
    bool swa, skip;
    int DH, L, n_sub = 0, n_rows = 0, first_row = 0, base = 0, S = 0, hi0 = 0, u0 = 0, G = 0, app_row0 = 0;
    size_t DKv, DQ, qkv_el, table_el, plane_rows, v_off;
    std::vector<uint16_t> qkv, tab, kctx, vctx;
    buffer<bf16> in_bo, o_bo, kv_bo;
    std::vector<uint16_t> first_o, first_kv;
    std::vector<double> us;
    int dispatches = 0, diffs = 0;

    attnx_case(decoding_layer& layer_, const std::string& dir_, int ctx_, int poison_mask_, const std::map<std::string, uint32_t>& rtp_)
        : layer(layer_), dir(dir_), ctx(ctx_), poison_mask(poison_mask_), rtp(rtp_) {
        L = L_TOKENS;
        swa = layer.is_swa_layer; skip = layer.shall_skip_kv;
        DH = layer.DH;
        const int NQ = 8, NKV = 2, W = 512;
        DKv = (size_t)NKV * DH;
        DQ = (size_t)NQ * DH;
        const size_t chunk = (size_t)L * DH;
        qkv_el = chunk * (NQ + (skip ? 0 : 2 * NKV));
        table_el = (size_t)L * DH + 2 * DH;
        int n_valid;
        if (!swa) { n_valid = ctx + L; }
        else {
            S = ctx - (W - 1) > 0 ? ctx - (W - 1) : 0;
            S = S / 16 * 16;
            const int lead = std::min(S, 37);
            base = S - lead; first_row = lead; n_valid = ctx + L - S;
        }
        G = swa ? 16 : 32;
        const int kps = swa ? 16 : 8;
        n_rows = (n_valid + G - 1) / G * G;
        n_sub = n_rows / kps;
        hi0 = swa ? ctx - S : ctx;
        u0 = swa ? ctx + 1 - W - S : -(1 << 30);
        plane_rows = (size_t)first_row + n_rows + 64;
        v_off = plane_rows * DKv;
        app_row0 = ctx - base;

        qkv = read_bf16(dir + "/qkv_in.bin", qkv_el);
        tab = read_bf16(dir + "/table.bin", table_el);
        const int ctx_rows = ctx + (skip ? L : 0);
        if (ctx_rows) {
            kctx = read_bf16(dir + "/kctx.bin", (size_t)ctx_rows * DKv);
            vctx = read_bf16(dir + "/vctx.bin", (size_t)ctx_rows * DKv);
        }
        in_bo = layer.layer_app.create_bo_buffer<bf16>(((qkv_el + table_el) + 8191) / 8192 * 8192);
        in_bo.memset((bf16)0.0f);
        std::memcpy(in_bo.data(), qkv.data(), qkv_el * 2);
        std::memcpy(in_bo.data() + qkv_el, tab.data(), table_el * 2);
        in_bo.sync_to_device();
        o_bo = layer.layer_app.create_bo_buffer<bf16>(((size_t)L * DQ + 8191) / 8192 * 8192);
        kv_bo = layer.layer_app.create_bo_buffer<bf16>(2 * plane_rows * DKv);
    }

    uint32_t need(const std::string& n) const {
        auto it = rtp.find(n);
        if (it == rtp.end()) throw std::runtime_error("rtp file has no " + n);
        return it->second;
    }

    void fill_kv() {
        kv_bo.memset((bf16)0.0f);
        if (poison_mask) {
            // K rows past the new rows -> NaN (masked keys: tolerated); V rows from the next 16-row block of the replay (SWA sub-chunks hold the
            // even / odd keys of 16 rows, so a V value only shares a bfp16 block with keys of its own 16-row group) -> 1e4.
            uint16_t* p = (uint16_t*)kv_bo.data();
            const size_t k_from = (size_t)(app_row0 + L);
            const size_t v_from = (size_t)first_row + ((size_t)(app_row0 + L - first_row) + 15) / 16 * 16;
            for (size_t r = 0; r < plane_rows; r++)
                for (size_t j = 0; j < DKv; j++) {
                    if ((poison_mask & 1) && r >= k_from) p[r * DKv + j] = 0x7FC0;
                    if ((poison_mask & 2) && r >= v_from) p[v_off + r * DKv + j] = 0x461C;
                    if ((poison_mask & 8) && r >= k_from) p[v_off + r * DKv + j] = 0x7FC0;
                }
        }
        const int ctx_rows = ctx + (skip ? L : 0);
        if (ctx_rows - base > 0) {
            std::memcpy(kv_bo.data(), kctx.data() + (size_t)base * DKv, (size_t)(ctx_rows - base) * DKv * 2);
            std::memcpy(kv_bo.data() + v_off, vctx.data() + (size_t)base * DKv, (size_t)(ctx_rows - base) * DKv * 2);
        }
        kv_bo.sync_to_device();
    }

    void gen_seq() {
        npu_sequence* seq = layer.layer_app.seq();
        seq->clear_cmds();
        std::map<int, int> tokens;
        auto tkey = [](npu_tiles t, dma_direction d, npu_it_channel c) { return ((int)t << 8) | ((int)d << 4) | (int)c; };
        auto issued = [&](npu_tiles t, dma_direction d, npu_it_channel c) { tokens[tkey(t, d, c)]++; };
        auto waited = [&](npu_tiles t, dma_direction d, npu_it_channel c) { seq->npu_dma_wait(t, d, c); tokens[tkey(t, d, c)]--; };

        seq->rtp_write(swa ? CT33 : CT23, need(swa ? "RTP_SWA_RoPE_SKIP_KV_5_3" : "RTP_RoPE_SKIP_KV_4_3"), skip ? 1 : 0);
        const npu_tiles score_tiles[2] = {CT14, CT34};
        const char* score_names[2] = {"RTP_attnx_score_3_4", "RTP_attnx_score_5_4"};
        for (int s = 0; s < 2; s++) {
            const uint32_t a = need(score_names[s]);
            const uint32_t w[5] = {(uint32_t)n_sub, (uint32_t)hi0, (uint32_t)u0, (uint32_t)(swa ? 1 : 0), (uint32_t)(4 * s)};
            for (int i = 0; i < 5; i++) seq->rtp_write(score_tiles[s], a + 4 * i, w[i]);
        }
        const npu_tiles pv_tiles[4] = {CT02, CT12, CT22, CT32};
        for (int j = 0; j < 4; j++) {
            const uint32_t a = need("RTP_attnx_pv_" + std::to_string(2 + j) + "_2");
            const uint32_t w[3] = {(uint32_t)n_sub, (uint32_t)(swa ? 1 : 0), (uint32_t)j};
            for (int i = 0; i < 3; i++) seq->rtp_write(pv_tiles[j], a + 4 * i, w[i]);
        }
        const npu_tiles feed = swa ? IT4 : IT3;
        seq->npu_dma_memcpy_nd(sizeof(bf16), layer.x_arg_id, MM2S, feed, bd_1, it_channel_1,
            {0, 0, 0, (uint32_t)qkv_el}, {1, 1, 1, (uint32_t)table_el}, {0, 0, 0, 1}, PKT_TABLE, 0, true, no_cache);
        issued(feed, MM2S, it_channel_1);
        seq->npu_dma_memcpy_nd(sizeof(bf16), layer.x_arg_id, MM2S, feed, bd_0, it_channel_0,
            {0, 0, 0, 0}, {1, 1, 1, (uint32_t)qkv_el}, {0, 0, 0, 1}, PKT_QKV, 0, true, no_cache);
        issued(feed, MM2S, it_channel_0);
        // o rows back on IT3 (global) / IT4 (SWA): [t][NQ*DH]
        seq->npu_dma_memcpy_nd(sizeof(bf16), layer.rms_arg_id, S2MM, feed, bd_10, it_channel_0,
            {0, 0, 0, 0}, {1, 1, 1, (uint32_t)(L * DQ)}, {0, 0, 0, 1}, -1, 0, true);
        issued(feed, S2MM, it_channel_0);
        const npu_it_channel kvch = swa ? it_channel_1 : it_channel_0;
        if (!skip) {
            for (int pl = 0; pl < 2; pl++) {
                seq->npu_dma_memcpy_nd(sizeof(bf16), layer.kv_cache_arg_id, S2MM, IT2, pl == 0 ? bd_0 : bd_1, kvch,
                    {0, 0, 0, (uint32_t)(pl * v_off + (size_t)app_row0 * DKv)}, {1, 2, (uint32_t)L, (uint32_t)DH}, {0, (uint32_t)DH, (uint32_t)DKv, 1}, -1, 0, true);
                issued(IT2, S2MM, kvch);
                waited(IT2, S2MM, kvch);   // the new rows must be in the cache before the replay reads them
            }
        }
        // K (ch0) and V (ch1) once, head-major: repeat h (stride DH) x [n_rows / G][G][DH]
        for (int pl = 0; pl < 2; pl++) {
            const npu_it_channel ch = pl == 0 ? it_channel_0 : it_channel_1;
            seq->npu_dma_memcpy_nd(sizeof(bf16), layer.kv_cache_arg_id, MM2S, IT2, pl == 0 ? bd_8 : bd_9, ch,
                {0, 0, 0, (uint32_t)(pl * v_off + (size_t)first_row * DKv)},
                {2, (uint32_t)(n_rows / G), (uint32_t)G, (uint32_t)DH}, {(uint32_t)DH, (uint32_t)(G * DKv), (uint32_t)DKv, 1},
                -1, 0, true, no_cache);
            issued(IT2, MM2S, ch);
        }
        for (auto& kv : tokens) {
            const npu_tiles t = (npu_tiles)(kv.first >> 8); const dma_direction d = (dma_direction)((kv.first >> 4) & 0xF); const npu_it_channel c = (npu_it_channel)(kv.first & 0xF);
            while (kv.second > 0) waited(t, d, c);
        }
        for (auto& kv : tokens) if (kv.second != 0) throw std::runtime_error("attnx: task-completion tokens unbalanced on shim key " + std::to_string(kv.first));
        seq->cmds2seq();
    }

    /// one dispatch; returns 0 ok, 3 hang / error, 4 output differs from this case's first dispatch
    int dispatch(int timeout_ms) {
        o_bo.memset((bf16)0.0f); o_bo.sync_to_device();
        fill_kv();
        auto t0 = std::chrono::steady_clock::now();
        ert_cmd_state st = ERT_CMD_STATE_NEW;
        try {
            xrt::run run = layer.layer_app.create_run(in_bo, layer.all_proj_weight_buffer, o_bo, layer.rope_rms_buffer, kv_bo);
            run.start();
            st = run.wait(std::chrono::milliseconds(timeout_ms));
        } catch (const std::exception& e) {
            std::cerr << "ERROR attnx: dispatch threw (ctx " << ctx << "): " << e.what() << std::endl;
            return 3;
        }
        const double d = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (st != ERT_CMD_STATE_COMPLETED) { std::cerr << "ERROR attnx: dispatch state " << (int)st << " (ctx " << ctx << ")" << std::endl; return 3; }
        us.push_back(d);
        o_bo.sync_from_device(); kv_bo.sync_from_device();
        std::vector<uint16_t> o((uint16_t*)o_bo.data(), (uint16_t*)o_bo.data() + L * DQ);
        std::vector<uint16_t> kvn;
        if (!skip) {
            for (int pl = 0; pl < 2; pl++)
                for (int t = 0; t < L; t++) {
                    const uint16_t* row = (const uint16_t*)kv_bo.data() + pl * v_off + (size_t)(app_row0 + t) * DKv;
                    kvn.insert(kvn.end(), row, row + DKv);
                }
        }
        dispatches++;
        if (dispatches == 1) {
            first_o = o; first_kv = kvn;
            std::ofstream f(dir + "/o_out.bin", std::ios::binary); f.write((const char*)o.data(), o.size() * 2);
            if (!skip) { std::ofstream g(dir + "/kv_new.bin", std::ios::binary); g.write((const char*)kvn.data(), kvn.size() * 2); }
            return 0;
        }
        if (o != first_o || kvn != first_kv) { diffs++; return 4; }
        return 0;
    }

    void report() const {
        std::vector<double> s = us;
        std::sort(s.begin(), s.end());
        std::cout << "attn8 attnx layer " << (swa ? "swa" : "global") << (skip ? "-skip" : "") << " ctx " << ctx << " n_sub " << n_sub
                  << " dispatches " << dispatches << " diffs " << diffs;
        if (!s.empty()) std::cout << " min_us " << (int)s.front() << " median_us " << (int)s[s.size() / 2];
        std::cout << std::endl;
    }
};

/// --ctxs a,b,c: the cases run one after the other in ONE hardware context (repeat dispatches each); --soak N: N more dispatches round robin
static int run_attnx(decoding_layer& layer, const std::string& dir_base, const std::string& xclbin_dir, const std::vector<int>& ctxs, bool multi,
                     int poison_mask, int repeat, int soak, int timeout_ms) {
    auto rtp = read_rtp_file(xclbin_dir + "/attnx_rtp_addresses.txt");
    std::vector<std::unique_ptr<attnx_case>> cases;
    for (int c : ctxs) {
        const std::string d = multi ? dir_base + "/layer" + std::to_string(layer.layer_idx) + "_ctx" + std::to_string(c) : dir_base;
        cases.push_back(std::make_unique<attnx_case>(layer, d, c, poison_mask, rtp));
    }
    int rc = 0;
    for (auto& c : cases) {
        c->gen_seq();
        for (int r = 0; r < repeat; r++) {
            const int e = c->dispatch(timeout_ms);
            if (e == 3) return 3;
            if (e == 4 && rc == 0) { std::cerr << "ERROR attnx: ctx " << c->ctx << " dispatch " << r << " differs from its first dispatch" << std::endl; rc = 4; }
        }
    }
    for (int i = 0; i < soak; i++) {
        auto& c = cases[i % cases.size()];
        if (cases.size() > 1 || i == 0) c->gen_seq();
        const int e = c->dispatch(timeout_ms);
        if (e == 3) { std::cerr << "ERROR attnx: soak stopped at dispatch " << i << std::endl; return 3; }
        if (e == 4 && rc == 0) { std::cerr << "ERROR attnx: soak dispatch " << i << " (ctx " << c->ctx << ") differs" << std::endl; rc = 4; }
    }
    for (auto& c : cases) c->report();
    return rc;
}
