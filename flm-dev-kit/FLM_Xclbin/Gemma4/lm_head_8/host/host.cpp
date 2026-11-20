// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// lm_head_8 host test: 8 hidden rows -> [8][vocab] logits on the NPU, compared with the shipping single-row lm_head xclbin
// (the "plain path" the wrapper samples from) and dumped for an independent numpy check (check_lm8.py).
//   ./run.exe -m GEMMA4_E4B [--xclbin8 build/GEMMA4_E4B/xclbins/lm_head_8.xclbin] [--xclbin1 <single-row lm_head.xclbin>]
//             [--model_dir <Bolt NPU2 package>] [--repeat N] [--dump dir] [--skip1]
#include "typedef.hpp"
#include "buffer.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/debug_utils.hpp"
#include "utils/vm_args.hpp"
#include "utils/utils.hpp"
#include "metrices.hpp"
#include "tensor_2d.hpp"
#include "lm_config.hpp"
#include "embedding.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

static constexpr int LT = 8;
static float g_noise = 1.0f;
static std::vector<bf16> g_raw; static bool g_keep_raw = false;
static constexpr size_t M_PER_ROUND = 32 * 32;
static constexpr int COLS = 8;
static constexpr int M_PER_COL = 32 * 4;

void reorder_cpy(u8* dst, buffer<u8>& src, const int col, const int vertical_blocks);

// instruction stream of the L=1 lm_head (copy of the engine's gen_lm_head_seq)
static void gen_seq_l1(npu_sequence* seq, int VOCAB, int D, float final_scale, uint32_t final_scale_address) {
    static npu_tiles ITs[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    const int rounds = VOCAB / M_PER_ROUND;
    const size_t W_PER_COL = (M_PER_COL * (size_t)D * 5 / 8 / 2);
    const size_t W_PER_ROUND = W_PER_COL * COLS;
    uint32_t scale_int_view = *((uint32_t*)(&final_scale));
    seq->clear_cmds();
    for (int row = 0; row < 4; row++)
        for (int col = 0; col < 8; col++)
            seq->rtp_write(get_tile(row + 2, col), final_scale_address, scale_int_view);
    seq->npu_dma_memcpy_nd(sizeof(bf16), 2, MM2S, ITs[0], npu_bd_id(bd_0), it_channel_0,
                           {0, 0, 0, 0}, {1, 1, 1, (uint32_t)D * 2}, {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
    for (int r = 0; r < rounds; r++) {
        int bd_offset = (r % 2) * 8;
        npu_bd_id bd_y = npu_bd_id(bd_1 + bd_offset);
        npu_bd_id bd_w = npu_bd_id(bd_2 + bd_offset);
        for (size_t col = 0; col < COLS; col++) {
            size_t w_col_offset = r * W_PER_ROUND + col * W_PER_COL;
            uint32_t y_offset = r * M_PER_ROUND + col * M_PER_COL;
            seq->npu_dma_memcpy_nd(sizeof(bf16), 0, S2MM, ITs[col], bd_y, it_channel_0,
                                   {0, 0, 0, y_offset}, {1, 1, 1, (uint32_t)M_PER_COL}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
            seq->npu_dma_memcpy_nd(sizeof(bf16), 1, MM2S, ITs[col], bd_w, it_channel_1,
                                   {0, 0, 0, (uint32_t)w_col_offset}, {1, 1, 1, (uint32_t)W_PER_COL}, {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
            if (r > 0) seq->npu_dma_wait(ITs[col], S2MM, it_channel_0);
        }
    }
    for (size_t col = 0; col < COLS; col++) seq->npu_dma_wait(ITs[col], S2MM, it_channel_0);
    seq->cmds2seq();
}

// instruction stream of the 8-row lm_head: x BO = [8 rows interleaved per 8 columns | norm weight] (D*9), y per (round,col): [core][t][32]
static void gen_seq_l8(npu_sequence* seq, int VOCAB, int D, float final_scale, uint32_t final_scale_address) {
    static npu_tiles ITs[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    const int rounds = VOCAB / M_PER_ROUND;
    const size_t W_PER_COL = (M_PER_COL * (size_t)D * 5 / 8 / 2);
    const size_t W_PER_ROUND = W_PER_COL * COLS;
    const uint32_t Y_PER_COL = M_PER_COL * LT;           // 1024 bf16
    uint32_t scale_int_view = *((uint32_t*)(&final_scale));
    seq->clear_cmds();
    for (int row = 0; row < 4; row++)
        for (int col = 0; col < 8; col++)
            seq->rtp_write(get_tile(row + 2, col), final_scale_address, scale_int_view);
    seq->npu_dma_memcpy_nd(sizeof(bf16), 2, MM2S, ITs[0], npu_bd_id(bd_0), it_channel_0,
                           {0, 0, 0, 0}, {1, 1, 1, (uint32_t)D * (LT + 1)}, {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
    for (int r = 0; r < rounds; r++) {
        int bd_offset = (r % 2) * 8;
        npu_bd_id bd_y = npu_bd_id(bd_1 + bd_offset);
        npu_bd_id bd_w = npu_bd_id(bd_2 + bd_offset);
        for (size_t col = 0; col < COLS; col++) {
            size_t w_col_offset = r * W_PER_ROUND + col * W_PER_COL;
            uint32_t y_offset = (r * COLS + col) * Y_PER_COL;
            seq->npu_dma_memcpy_nd(sizeof(bf16), 0, S2MM, ITs[col], bd_y, it_channel_0,
                                   {0, 0, 0, y_offset}, {1, 1, 1, Y_PER_COL}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
            seq->npu_dma_memcpy_nd(sizeof(bf16), 1, MM2S, ITs[col], bd_w, it_channel_1,
                                   {0, 0, 0, (uint32_t)w_col_offset}, {1, 1, 1, (uint32_t)W_PER_COL}, {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
            if (r > 0) seq->npu_dma_wait(ITs[col], S2MM, it_channel_0);
        }
    }
    for (size_t col = 0; col < COLS; col++) seq->npu_dma_wait(ITs[col], S2MM, it_channel_0);
    seq->cmds2seq();
}

static uint32_t read_rtp_address(const std::string& xclbin, uint32_t fallback) {
    std::string dir = xclbin.substr(0, xclbin.find_last_of('/') + 1);
    std::ifstream f(dir + "rtp_addresses.txt");
    std::string name; uint32_t addr = 0;
    if (f && (f >> name >> addr)) return addr;
    return fallback;
}

static void pack_x(bf16* x_bo, const std::vector<std::vector<bf16>>& rows, const bf16* nw, int D) {
    for (int t = 0; t < LT; t++)
        for (int i = 0; i < D; i++) {
            int blk = i / 256, kt = (i % 256) / 8, j = i % 8;
            x_bo[(size_t)(blk * 32 + kt) * 64 + t * 8 + j] = rows[t][i];
        }
    memcpy(x_bo + (size_t)LT * D, nw, D * sizeof(bf16));
}

static void unpack_y(const bf16* raw, std::vector<bf16>& logits, size_t V) {   // raw [round][col][core][t][32] -> logits[t][V]
    const size_t rounds = V / M_PER_ROUND;
    for (size_t r = 0; r < rounds; r++)
        for (int c = 0; c < COLS; c++)
            for (int core = 0; core < 4; core++)
                for (int t = 0; t < LT; t++)
                    memcpy(&logits[(size_t)t * V + r * M_PER_ROUND + c * M_PER_COL + core * 32],
                           raw + ((r * COLS + c) * 4 + core) * (LT * 32) + t * 32, 32 * sizeof(bf16));
}

struct stat_t { double cos, rel_l2, max_abs; int argmax_a, argmax_b; };
static stat_t compare(const bf16* a, const bf16* b, size_t V) {
    double dot = 0, na = 0, nb = 0, d2 = 0, mx = 0; int ia = 0, ib = 0; float ma = -1e30f, mb = -1e30f;
    for (size_t i = 0; i < V; i++) {
        double x = (float)a[i], y = (float)b[i];
        dot += x * y; na += x * x; nb += y * y; d2 += (x - y) * (x - y); mx = std::max(mx, std::fabs(x - y));
        if ((float)a[i] > ma) { ma = (float)a[i]; ia = (int)i; }
        if ((float)b[i] > mb) { mb = (float)b[i]; ib = (int)i; }
    }
    return {dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30), std::sqrt(d2 / (nb + 1e-30)), mx, ia, ib};
}

int main(int argc, char** argv) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("GEMMA4_E4B"), "model type");
    desc.add_options()("model_dir", arg_utils::po::value<std::string>()->default_value("../verify_decoding/q4nx_files/Gemma4-E4B-IT-NPU2"), "Bolt NPU2 package");
    desc.add_options()("xclbin8", arg_utils::po::value<std::string>()->default_value("./build/GEMMA4_E4B/xclbins/lm_head_8.xclbin"), "8-row lm_head xclbin");
    desc.add_options()("xclbin1", arg_utils::po::value<std::string>()->default_value("../lm_head_npu_bin/build/GEMMA4_E4B/xclbins/lm_head.xclbin"), "single-row lm_head xclbin");
    desc.add_options()("repeat", arg_utils::po::value<int>()->default_value(5), "timing repeats");
    desc.add_options()("dump", arg_utils::po::value<std::string>()->default_value(""), "dump dir for check_lm8.py");
    desc.add_options()("skip1", "do not run the single-row xclbin");
    desc.add_options()("noise", arg_utils::po::value<float>()->default_value(1.0f), "noise amplitude of the test rows (larger = less saturated logits)");
    desc.add_options()("seed", arg_utils::po::value<int>()->default_value(1), "token seed");
    arg_utils::parse_options(argc, argv, desc, vm);

    const std::string model_dir = vm["model_dir"].as<std::string>();
    g_noise = vm["noise"].as<float>();
    const int repeat = vm["repeat"].as<int>();
    const bool skip1 = vm.count("skip1") > 0;
    LM_Config lm_config; lm_config.from_pretrained(model_dir);
    const int D = lm_config.hidden_size;
    const size_t V = 262144;                       // vocab (already a multiple of 1024)
    header_print("info", "D=" << D << " V=" << V);

    Q4NX q4nx(model_dir);
    embedding_module emb(V, D);
    emb.init_weights(q4nx, "model.embed_tokens");
    buffer<bf16> nw; q4nx.load_weights(nw, "model.norm.weight");

    // 3 sets of 8 hidden rows: embedding rows scaled by sqrt(D) (final hiddens are larger than embeddings, argmax is scale free)
    // mean embedding direction (estimated on 4096 rows): embedding rows share a large common component that would push every tied logit
    // towards +softcap; hidden states are modelled without it
    std::vector<float> emb_mean(D, 0.0f);
    {
        srand(7);
        for (int n = 0; n < 4096; n++) { buffer<bf16> e(D); emb.forward(rand() % V, e); for (int i = 0; i < D; i++) emb_mean[i] += (float)e[i] / 4096.0f; }
    }
    auto make_rows = [&](int seed) {
        // realistic-ish hidden rows: a random mix of 6 embedding rows plus noise (a single embedding row would saturate the softcap
        // because the lm_head is tied to the embedding), scaled like a final hidden state
        std::vector<std::vector<bf16>> rows(LT, std::vector<bf16>(D));
        srand(seed);
        for (int t = 0; t < LT; t++) {
            std::vector<float> acc(D, 0.0f);
            for (int j = 0; j < 6; j++) {
                int id = rand() % V; float a = (rand() % 2001 - 1000) / 1000.0f;
                buffer<bf16> e(D); emb.forward(id, e);
                for (int i = 0; i < D; i++) acc[i] += a * ((float)e[i] - emb_mean[i]);
            }
            for (int i = 0; i < D; i++) { float n = ((rand() % 2001) - 1000) / 1000.0f; rows[t][i] = (bf16)((acc[i] + g_noise * n) * (6.0f + t)); }
        }
        return rows;
    };
    auto rowsA = make_rows(vm["seed"].as<int>()), rowsB = make_rows(vm["seed"].as<int>() + 100);

    xrt::device npu_inst = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_inst, false);

    buffer<u8> lm_raw; q4nx.load_weights(lm_raw, "lm_head.weight");
    const size_t w_bytes = (size_t)D * V * 5 / 8;
    header_print("info", "lm_head weight bytes " << lm_raw.size() << " expected " << w_bytes);

    // ---- 8-row xclbin
    const std::string x8 = vm["xclbin8"].as<std::string>();
    npu_app_manager* mgr8 = xclbin_mgr.register_xclbin(x8);
    npu_app app8 = mgr8->create_app();
    buffer<u8> w8 = app8.create_bo_buffer<u8>(w_bytes);
    buffer<bf16> x8bo = app8.create_bo_buffer<bf16>((size_t)D * (LT + 1));
    buffer<bf16> y8bo = app8.create_bo_buffer<bf16>(V * LT);
    reorder_cpy(w8.data(), lm_raw, D, 4);
    w8.sync_to_device();
    const uint32_t rtp8 = read_rtp_address(x8, 61440);
    header_print("info", "8-row final-scale RTP address " << rtp8);
    gen_seq_l8(app8.seq(), (int)V, D, 30.0f, rtp8);
    mgr8->get_ready(); mgr8->ready();

    auto run8 = [&](const std::vector<std::vector<bf16>>& rows, std::vector<bf16>& logits, double* ms) {
        pack_x(x8bo.data(), rows, nw.data(), D);
        x8bo.sync_to_device();
        auto t0 = std::chrono::steady_clock::now();
        app8(y8bo, w8, x8bo);
        auto t1 = std::chrono::steady_clock::now();
        y8bo.sync_from_device();
        if (ms) *ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (g_keep_raw) g_raw.assign(y8bo.data(), y8bo.data() + V * LT);
        unpack_y(y8bo.data(), logits, V);
    };

    // ---- 1-row xclbin (reference path)
    std::vector<std::vector<bf16>> ref1A(LT, std::vector<bf16>(V)), ref1B(LT, std::vector<bf16>(V));
    double ms1 = 0, ms1_min = 1e30;
    if (!skip1) {
        const std::string x1 = vm["xclbin1"].as<std::string>();
        npu_app_manager* mgr1 = xclbin_mgr.register_xclbin(x1);
        npu_app app1 = mgr1->create_app();
        buffer<u8> w1 = app1.create_bo_buffer<u8>(w_bytes);
        buffer<bf16> x1bo = app1.create_bo_buffer<bf16>((size_t)D * 2);
        buffer<bf16> y1bo = app1.create_bo_buffer<bf16>(V);
        reorder_cpy(w1.data(), lm_raw, D, 4);
        w1.sync_to_device();
        gen_seq_l1(app1.seq(), (int)V, D, 30.0f, 10240);
        mgr1->get_ready(); mgr1->ready();
        memcpy(x1bo.data() + D, nw.data(), D * sizeof(bf16));
        int n = 0;
        for (auto* set : {&rowsA, &rowsB}) {
            auto& dst = (set == &rowsA) ? ref1A : ref1B;
            for (int t = 0; t < LT; t++) {
                memcpy(x1bo.data(), (*set)[t].data(), D * sizeof(bf16));
                x1bo.sync_to_device();
                auto t0 = std::chrono::steady_clock::now();
                app1(y1bo, w1, x1bo);
                auto t1 = std::chrono::steady_clock::now();
                y1bo.sync_from_device();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                ms1 += ms; ms1_min = std::min(ms1_min, ms); n++;
                memcpy(dst[t].data(), y1bo.data(), V * sizeof(bf16));
            }
        }
        ms1 /= n;
        header_print("info", "single-row dispatch: mean " << ms1 << " ms, min " << ms1_min << " ms");
    }

    // ---- 8-row: first dispatch (A), second (B), third (A again) must reproduce the first bit for bit
    std::vector<bf16> LA(V * LT), LB(V * LT), LA2(V * LT);
    double msA = 0, msB = 0, msA2 = 0;
    g_keep_raw = true;
    run8(rowsA, LA, &msA);
    g_keep_raw = false;
    run8(rowsB, LB, &msB);
    run8(rowsA, LA2, &msA2);
    bool repeat_ok = memcmp(LA.data(), LA2.data(), V * LT * sizeof(bf16)) == 0;
    header_print("info", "dispatch 1 " << msA << " ms, dispatch 2 " << msB << " ms, dispatch 3 " << msA2 << " ms; dispatch3 == dispatch1 bit-exact: " << (repeat_ok ? "YES" : "NO"));

    int fails = repeat_ok ? 0 : 1;
    if (!skip1) {
        for (int pass = 0; pass < 2; pass++) {
            auto& L = pass == 0 ? LA : LB; auto& R = pass == 0 ? ref1A : ref1B;
            int am_ok = 0; double min_cos = 1, max_rel = 0;
            for (int t = 0; t < LT; t++) {
                stat_t s = compare(&L[(size_t)t * V], R[t].data(), V);
                // argmax: equal, or a tie in the reference (bf16 logits)
                // equal, or indistinguishable in the reference: at the softcap many logits sit at +-30 where one bf16 ulp is 0.125, so allow 2 ulps
                bool ok = s.argmax_a == s.argmax_b || std::fabs((float)R[t][s.argmax_a] - (float)R[t][s.argmax_b]) <= 0.25f;
                am_ok += ok; min_cos = std::min(min_cos, s.cos); max_rel = std::max(max_rel, s.rel_l2);
                std::cout << (pass ? "B" : "A") << " row " << t << ": argmax8=" << s.argmax_a << " argmax1=" << s.argmax_b << (ok ? " OK" : " MISMATCH")
                          << " cos=" << s.cos << " relL2=" << s.rel_l2 << " maxabs=" << s.max_abs << std::endl;
            }
            std::cout << (pass ? "B" : "A") << ": argmax match " << am_ok << "/8, min cos " << min_cos << ", max relL2 " << max_rel << std::endl;
            if (am_ok != LT || min_cos < 0.9995) fails++;   // both paths are ~0.9997 from the exact value on their own (check_lm8.py)
        }
    }

    // ---- timing
    std::vector<double> t8;
    int repeat_bad = 0;
    for (int i = 0; i < repeat; i++) {
        double ms; run8((i & 1) ? rowsB : rowsA, LA2, &ms); t8.push_back(ms);
        if (memcmp(LA2.data(), (i & 1) ? LB.data() : LA.data(), V * LT * sizeof(bf16)) != 0) repeat_bad++;   // rotating A/B: every dispatch must reproduce its first result bit for bit
    }
    header_print("info", "timing loop: " << repeat << " dispatches alternating A/B, bit-exact vs first results: " << (repeat - repeat_bad) << "/" << repeat);
    if (repeat_bad) fails++;
    std::sort(t8.begin(), t8.end());
    if (!t8.empty()) header_print("info", "8-row dispatch (host wall, incl. x pack+sync): min " << t8.front() << " ms, median " << t8[t8.size() / 2] << " ms over " << repeat);
    if (!skip1) header_print("info", "ratio 8-row/1-row (median/mean): " << t8[t8.size() / 2] / ms1 << "   vs 8x single: " << t8[t8.size() / 2] / (8 * ms1));

    std::string dump = vm["dump"].as<std::string>();
    if (!dump.empty()) {
        auto wr = [&](const std::string& n, const void* p, size_t b) { std::ofstream f(dump + "/" + n, std::ios::binary); f.write((const char*)p, b); };
        std::vector<bf16> hA((size_t)LT * D);
        for (int t = 0; t < LT; t++) memcpy(&hA[(size_t)t * D], rowsA[t].data(), D * sizeof(bf16));
        wr("hidden_A.bin", hA.data(), hA.size() * 2); wr("nw.bin", nw.data(), D * 2); wr("logits8_A.bin", LA.data(), LA.size() * 2); wr("y8raw_A.bin", g_raw.data(), g_raw.size() * 2); wr("x8_A.bin", x8bo.data(), (size_t)D * (LT + 1) * 2);
        if (!skip1) { std::vector<bf16> r1((size_t)LT * V); for (int t = 0; t < LT; t++) memcpy(&r1[(size_t)t * V], ref1A[t].data(), V * 2); wr("logits1_A.bin", r1.data(), r1.size() * 2); }
    }
    std::cout << (fails == 0 ? "RESULT: PASS" : "RESULT: FAIL") << std::endl;
    return fails == 0 ? 0 : 1;
}

void reorder_cpy(u8* dst, buffer<u8>& src, const int col, const int vertical_blocks) {
    const int a_block_size = 32 * 256 * 5 / 8;
    const int blocks_per_row = col / 256;
    const int rows = src.size() / a_block_size / blocks_per_row;
    u8* dst_ptr = dst;
    std::vector<u8*> src_ptr(vertical_blocks);
    for (int i = 0; i < vertical_blocks; i++) src_ptr[i] = src.data() + (size_t)i * a_block_size * blocks_per_row;
    for (int r = 0; r < rows; r += vertical_blocks) {
        for (int c = 0; c < blocks_per_row; c++)
            for (int i = 0; i < vertical_blocks; i++) { memcpy(dst_ptr, src_ptr[i], a_block_size); dst_ptr += a_block_size; src_ptr[i] += a_block_size; }
        for (int i = 0; i < vertical_blocks; i++) {
            src_ptr[i] += (size_t)(vertical_blocks - 1) * a_block_size * blocks_per_row;
            if (src_ptr[i] + (size_t)a_block_size * blocks_per_row > src.end()) src_ptr[i] = src.data();
        }
    }
}
