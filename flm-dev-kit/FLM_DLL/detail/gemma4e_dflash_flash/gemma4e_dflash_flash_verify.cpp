// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Fused 8-row verify for gemma4e_dflash_flash (README.md, "Fused verify").
//
// One verify cycle replaces 8 sequential decode passes by ONE pass of the 8-row verify xclbin (FLM_Xclbin/Gemma4/verify_decoding, DESIGN=fused8)
// per layer plus ONE dispatch of lm_head_8:
//   * x BO per layer i (43 of them):  [8][D] layer input rows | global RoPE table | SWA RoPE table | [8][PLI_D] pli | [D + 32] w_post_pli, layer_scalar.
//     Layer i writes its output rows to the FIRST 8*D elements of the BO of layer i+1 (kernel arg 3 = that BO), so the 42 runs chain on the device
//     and the five tap layers (5, 11, 17, 23, 35) are simply the rows of BOs 6, 12, 18, 24, 36: per-layer x slots (D7) without any DMA offset games.
//   * weights: the decode BOs (proj_weights[i], rms_weights[i]) are passed unchanged; K/V: the engine's caches (global: row = position; sliding:
//     the linear window buffer of gemma4e_dflash_flash_swa.hpp, replay range [S - base, ctx + 8 - base)).
//   * the host computes per cycle: the 8 embedding rows, 8 RoPE tables per layer kind, 8 x 42 pli rows (host contract of FLM_Xclbin/Gemma4/verify_decoding/notes/ple8_L8_results.md).
// Interface and layouts: FLM_Xclbin/Gemma4/verify_decoding/notes/fused8_results.md.
#include "gemma4e_dflash_flash_rtp.hpp"
#include <map>
#include "gemma4e_dflash_flash_spec_state.hpp"


#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <immintrin.h>
#include <omp.h>
#include <sstream>
#include "gemma4e_cpu_functions.hpp"

namespace {
constexpr int kL = 8;                  // rows of one verify block
constexpr int kChunk = 16;             // attention replay chunk
constexpr int kWindow = 512;           // sliding window
constexpr int kNKV = 2;                // kv heads of both layer kinds
constexpr int kLmCols = 8, kLmMPerCol = 128, kLmMPerRound = 1024;

inline float bf2f(bf16 v) { return bf16_to_f32(v); }

std::map<std::string, uint32_t> read_rtp_file(const std::string& fn) {
    std::map<std::string, uint32_t> m;
    std::ifstream f(fn);
    std::string name;
    uint32_t addr;
    while (f >> name >> addr) m[name] = addr;
    return m;
}
}  // namespace

struct gemma4e_dflash_flash::Impl::Spec::Verify {
    bool ready = false;
    // the verify xclbin lives in its own manager (own hardware context) so that the context can be torn down and rebuilt
    // (optional safety switch FLM_DFLASH_VERIFY_RELOAD=N; the dispatch-count hang it was added for was an unbalanced task-completion
    // token in the verify instruction stream, fixed: FLM_Xclbin/Gemma4/verify_decoding/notes/hang_rootcause.md)
    std::unique_ptr<flm_rt::device> vdev;
    std::unique_ptr<npu_xclbin_manager> vmgr;
    std::string xclbin_path;
    long long runs_since_load = 0;
    std::future<void> prep_future;           // verify_prepare running concurrently with the host draft
    bool prepared = false;
    int prepared_ctx = -1;
    npu_app_manager* mgr = nullptr;
    npu_app app[4];                          // indexed by gemma4e_layer_type_t
    std::map<std::string, uint32_t> rtp;
    bool attnx = false;                 // the verify design reads K/V once per verify (RTP_attnx_* present)
    flm_rt::runlist run;
    std::vector<buffer<bf16>> xbo;           // num_hidden_layers + 1 regions
    size_t tg_off = 0, ts_off = 0, pli_off = 0, wc_off = 0, region = 0;
    int DHg = 0, DHs = 0;
    size_t tabg_n = 0, tabs_n = 0;

    npu_app_manager* lm8_mgr = nullptr;
    npu_app lm8;
    buffer<bf16> lm8_x, lm8_y;
    uint32_t lm8_scale_addr = 61440;
    bool lm8_ready = false;

    std::vector<bf16> x0, pl, pli_rows;      // scratch: [8][D], [8][layers*PLI_D], [layers][8][PLI_D]

    void open_context() {
        vmgr.reset();
        vmgr = std::make_unique<npu_xclbin_manager>(npu_device::device_npu2, vdev.get(), false);
        mgr = vmgr->register_xclbin(xclbin_path);
        for (int t = 0; t < 4; t++) app[t] = mgr->create_app();
        run = mgr->create_runlist();
        runs_since_load = 0;
    }
};

bool gemma4e_dflash_flash::Impl::verify_fused_ready() const {
    return this->spec != nullptr && this->spec->verify != nullptr && this->spec->verify->ready;
}

// instruction stream of the 8-row lm_head (reference: FLM_Xclbin/Gemma4/lm_head_8/host/host.cpp gen_seq_l8)
static void gen_lm8_seq(npu_sequence* seq, int VOCAB, int D, float final_scale, uint32_t final_scale_address) {
    static const npu_tiles ITs[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr size_t M_PER_ROUND = 32 * 32;
    constexpr int COLS = 8;
    constexpr int M_PER_COL = 32 * 4;
    const int rounds = VOCAB / (int)M_PER_ROUND;
    const size_t W_PER_COL = (M_PER_COL * (size_t)D * 5 / 8 / 2);
    const size_t W_PER_ROUND = W_PER_COL * COLS;
    const uint32_t Y_PER_COL = M_PER_COL * kL;
    uint32_t scale_int_view;
    std::memcpy(&scale_int_view, &final_scale, 4);
    seq->clear_cmds();
    for (int row = 0; row < 4; row++)
        for (int col = 0; col < 8; col++)
            seq->rtp_write(get_tile(row + 2, col), final_scale_address, scale_int_view);
    seq->npu_dma_memcpy_nd(sizeof(bf16), 2, MM2S, ITs[0], npu_bd_id(bd_0), it_channel_0,
                           {0, 0, 0, 0}, {1, 1, 1, (uint32_t)D * (kL + 1)}, {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
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

bool gemma4e_dflash_flash::Impl::verify_load() {
    Spec& S = *this->spec;
    const char* mode = std::getenv("FLM_DFLASH_VERIFY");
    const std::string m = mode ? mode : "";
    if (m == "seq") {
        header_print("FLM", "gemma4e_dflash_flash: FLM_DFLASH_VERIFY=seq, sequential verify");
        return false;
    }
    S.verify_check = (m == "check");
    if (this->npu->is_preemption_enabled()) {
        header_print("FLM", "gemma4e_dflash_flash: preemption enabled, sequential verify");
        return false;
    }
    if (S.verify != nullptr) return S.verify->ready;
    const std::string dir = utils::path_join(config.exec_path, "xclbins", config.model_name);
    // FLM_DFLASH_ATTN=8 selects the older verify design (attention replays K/V once per token) for A/B runs
    const char* attn_env = std::getenv("FLM_DFLASH_ATTN");
    const std::string vname = (attn_env != nullptr && std::string(attn_env) == "8") ? "verify_attn8" : "verify";
    const std::string vx = utils::path_join(dir, vname + ".xclbin"), lx = utils::path_join(dir, "lm_head_8.xclbin");
    {
        std::ifstream a(vx), b(lx);
        if (!a.good() || !b.good()) {
            header_print("FLM", "gemma4e_dflash_flash: verify.xclbin / lm_head_8.xclbin not found in " + dir + ", sequential verify");
            return false;
        }
    }
    try {
        auto V = std::make_unique<Spec::Verify>();
        // a local build may carry its own address list next to the xclbin; the shipped verify.xclbin uses the compiled-in one
        V->rtp = read_rtp_file(utils::path_join(dir, vname + "_rtp_addresses.txt"));
        if (V->rtp.empty() && vname == "verify") {
            for (const auto& kv : gemma4e_dflash_flash_rtp::verify_attnx) V->rtp[kv.first] = kv.second;
        }
        if (V->rtp.empty()) throw std::runtime_error(vname + "_rtp_addresses.txt missing or empty");
        V->attnx = V->rtp.count("RTP_attnx_score_3_4") != 0;
        header_print("FLM", "gemma4e_dflash_flash: verify design " + vname + (V->attnx ? " (K/V read once per verify)" : " (K/V replayed per token)"));
        {
            auto r8 = read_rtp_file(utils::path_join(dir, "lm_head_8_rtp_addresses.txt"));
            if (r8.empty()) {
                for (const auto& kv : gemma4e_dflash_flash_rtp::lm_head_8) r8[kv.first] = kv.second;
            }
            for (auto& kv : r8) { V->lm8_scale_addr = kv.second; break; }
        }
        V->vdev = std::make_unique<flm_rt::device>(0);
        V->xclbin_path = vx;
        V->open_context();
        V->DHg = (int)this->DH;
        V->DHs = (int)this->SWA_DH;
        V->tabg_n = (size_t)kL * V->DHg + 2 * V->DHg;
        V->tabs_n = (size_t)kL * V->DHs + 2 * V->DHs;
        V->tg_off = (size_t)kL * D;
        V->ts_off = V->tg_off + V->tabg_n;
        V->pli_off = V->ts_off + V->tabs_n;
        V->wc_off = V->pli_off + (size_t)kL * PLI_D;
        V->region = (V->wc_off + D + 32 + 8191) / 8192 * 8192;
        V->xbo.resize(num_hidden_layers + 1);
        for (int i = 0; i <= num_hidden_layers; i++) {
            V->xbo[i] = V->mgr->create_bo_buffer<bf16>(V->region);
            V->xbo[i].memset((bf16)0.0f);
        }
        // static per layer content: q_norm | k_norm after the RoPE rows, w_post_pli and layer_scalar
        for (int i = 0; i < num_hidden_layers; i++) {
            const gemma4e_layer_type_t type = layer_types[i];
            const bool swa = is_swa_layer(type);
            const int DHl = swa ? SWA_DH : DH;
            bf16* xp = V->xbo[i].data();
            bf16* tab = xp + (swa ? V->ts_off : V->tg_off) + (size_t)kL * DHl;
            const bf16* rr = this->rope_rms_weights[i].data();
            std::memcpy(tab, rr + desc.get_q_norm_offset(type), (size_t)DHl * sizeof(bf16));
            std::memcpy(tab + DHl, rr + desc.get_k_norm_offset(type), (size_t)DHl * sizeof(bf16));
            std::memcpy(xp + V->wc_off, rr + desc.get_post_pli_norm_offset(type), (size_t)D * sizeof(bf16));
            xp[V->wc_off + D] = rr[desc.get_layer_scale_offset(type)];
        }
        // lm_head_8
        V->lm8_mgr = this->npu->register_xclbin(lx);
        V->lm8 = V->lm8_mgr->create_app();
        V->lm8_x = V->lm8_mgr->create_bo_buffer<bf16>((size_t)D * (kL + 1));
        V->lm8_y = V->lm8_mgr->create_bo_buffer<bf16>((size_t)this->vocab_size_padded * kL);
        gen_lm8_seq(V->lm8.seq(), (int)this->vocab_size_padded, (int)D, this->final_logit_softcapping > 0 ? (float)this->final_logit_softcapping : 30.0f, V->lm8_scale_addr);
        V->x0.resize((size_t)kL * D);
        V->pl.resize((size_t)kL * num_hidden_layers * PLI_D);
        V->pli_rows.resize((size_t)num_hidden_layers * kL * PLI_D);
        V->ready = true;
        S.verify = V.release();
    } catch (const std::exception& e) {
        header_print("WARNING", std::string("gemma4e_dflash_flash: fused verify unavailable (") + e.what() + "), sequential verify");
        return false;
    }
    header_print("FLM", std::string("gemma4e_dflash_flash: fused 8-row verify enabled") + (S.verify_check ? " (check mode: also runs the sequential path)" : ""));
    return true;
}

// called from spec_destroy (gemma4e_dflash_flash_spec.cpp)
void gemma4e_dflash_flash::Impl::verify_destroy() {
    if (this->spec != nullptr && this->spec->verify != nullptr && this->spec->verify->prep_future.valid()) {
        try { this->spec->verify->prep_future.get(); } catch (...) {}
    }
    if (this->spec != nullptr) { delete this->spec->verify; this->spec->verify = nullptr; }
}

// ---------------------------------------------------------------------------------------------------------------------
// instruction stream of one verify layer (reference: verify_decoding/host/test_fused8.hpp run(), stage "full")
// ---------------------------------------------------------------------------------------------------------------------
void gemma4e_dflash_flash::Impl::verify_gen_seq(npu_sequence* seq, gemma4e_layer_type_t type, int ctx) {
    Spec::Verify& V = *this->spec->verify;
    gemma4e_dflash_flash_sequence& sq = *this->sequence;
    constexpr size_t CT_lock_address_base = 0x000001F000;
    const bool swa = is_swa_layer(type), skip = is_skip_layer(type);
    const int DHl = swa ? SWA_DH : DH;
    const size_t DKv = swa ? SWA_DK : DK;
    const int P = PLI_D;
    auto need = [&](const std::string& n) {
        auto it = V.rtp.find(n);
        if (it == V.rtp.end()) throw std::runtime_error("verify rtp file has no " + n);
        return it->second;
    };
    // window geometry
    int S = 0, first_row = 0, app_row0 = ctx, n_rows = ctx + kL;
    size_t v_off = (size_t)desc.get_kv_cache_size(type, MAX_L) / 2;
    if (swa) {
        S = std::max(0, ctx - (kWindow - 1)) / kChunk * kChunk;
        n_rows = ctx + kL - S;
        first_row = S - this->swa_base;
        app_row0 = ctx - this->swa_base;
        v_off = (size_t)kSwaBufferRows * DKv;
        assert(first_row >= 0 && app_row0 + kL <= kSwaBufferRows);
    }
    const int L_max = n_rows;
    const int chunks = (n_rows + kChunk - 1) / kChunk;
    // attnx (verify design with FUSED8_ATTN=x): K / V streamed head-major ONCE per verify, rows rounded up to G (32 global, 16 SWA),
    // ax_kps keys per sub-chunk. The rows between the valid range and the rounded end must be finite: they are past the context,
    // and the engine keeps every row past the context zero (commit zeroes rejected rows).
    const int ax_G = swa ? 16 : 32, ax_kps = swa ? 16 : 8;
    const int ax_rows = (n_rows + ax_G - 1) / ax_G * ax_G;
    const int ax_nsub = ax_rows / ax_kps;
    if (V.attnx) {
        const size_t cap = swa ? (size_t)kSwaBufferRows : (size_t)MAX_L;
        if ((size_t)(first_row + ax_rows) > cap)
            throw std::runtime_error("gemma4e_dflash_flash verify: attention replay rows exceed the KV buffer");
    }
    const int x_arg = 0, w_arg = 1, rms_arg = 2, y_arg = 3, kv_arg = 4;

    seq->clear_cmds();
    const uint32_t a_swa = need("RTP_PROJ_IS_SWA_BUFFER_2_0"), a_skip = need("RTP_PROJ_SKIP_KV_BUFFER_2_0");
    for (int i = 0; i < 16; i++) {
        seq->rtp_write(sq.proj_tiles[i], a_swa, swa ? 1 : 0);
        seq->rtp_write(sq.proj_tiles[i], a_skip, skip ? 1 : 0);
        seq->rtp_write(sq.proj_tiles[i], CT_lock_address_base + 16 * 6, 1);   // rtp_available_lock
    }
    seq->rtp_write(swa ? sq.swa_rope_ct : sq.rope_ct, need(swa ? "RTP_SWA_RoPE_SKIP_KV_5_3" : "RTP_RoPE_SKIP_KV_4_3"), skip ? 1 : 0);
    if (V.attnx) {
        const int hi0 = swa ? ctx - S : ctx;
        const int u0 = swa ? ctx + 1 - kWindow - S : -(1 << 30);
        const npu_tiles score_tiles[2] = {CT14, CT34};
        const char* score_names[2] = {"RTP_attnx_score_3_4", "RTP_attnx_score_5_4"};
        for (int sc = 0; sc < 2; sc++) {   // [n_sub, hi0, u0, is_swa, first query head]
            const uint32_t a = need(score_names[sc]);
            const uint32_t w[5] = {(uint32_t)ax_nsub, (uint32_t)hi0, (uint32_t)u0, (uint32_t)(swa ? 1 : 0), (uint32_t)(4 * sc)};
            for (int i = 0; i < 5; i++) seq->rtp_write(score_tiles[sc], a + 4 * i, w[i]);
        }
        const npu_tiles pv_tiles[4] = {CT02, CT12, CT22, CT32};
        for (int j = 0; j < 4; j++) {      // [n_sub, is_swa, j]
            const uint32_t a = need("RTP_attnx_pv_" + std::to_string(2 + j) + "_2");
            const uint32_t w[3] = {(uint32_t)ax_nsub, (uint32_t)(swa ? 1 : 0), (uint32_t)j};
            for (int i = 0; i < 3; i++) seq->rtp_write(pv_tiles[j], a + 4 * i, w[i]);
        }
    } else {
        const npu_tiles qk_tile = swa ? sq.swa_attn_qk_tile : sq.attn_qk_tile;
        const npu_tiles kv_tile = swa ? sq.swa_attn_kv_tile : sq.attn_kv_tile;
        const uint32_t qk_addr = need(swa ? "RTP_L_swa_attn_qk_core_4_2" : "RTP_L_attn_qk_core_2_2");
        seq->rtp_write(qk_tile, qk_addr, (uint32_t)L_max);
        if (swa) {   // swa qk: L[1] = hi0 = ctx - S, L[2] = u0 = ctx + 1 - W - S (kernels/swa_attn_qk.cc)
            seq->rtp_write(qk_tile, qk_addr + 4, (uint32_t)(ctx - S));
            seq->rtp_write(qk_tile, qk_addr + 8, (uint32_t)(ctx + 1 - kWindow - S));
        }
        seq->rtp_write(kv_tile, need(swa ? "RTP_L_swa_attn_kv_core_5_2" : "RTP_L_attn_kv_core_3_2"), (uint32_t)L_max);
    }
    constexpr int PKT_SHIM_TO_RMS = 26, PKT_RMS_WEIGHTS = 11, PKT_ROPE_TABLE = 9, PKT_SWA_ROPE_TABLE = 10, PKT_PLI = 27, PKT_PLE_CONST = 28;
    // Task-completion token ledger: every shim BD submitted with a token (all S2MM, and MM2S with issue_token) must be consumed by exactly one
    // npu_dma_wait on the same (tile, direction, channel) inside this dispatch.  A token left over stays in the shim's task-completion queue
    // across dispatches; when the queue is full (~255 dispatches) the array stops answering (FLM_Xclbin/Gemma4/verify_decoding/notes/hang_rootcause.md).
    std::map<int, int> tokens;
    auto tkey = [](npu_tiles t, dma_direction d, npu_it_channel c) { return ((int)t << 8) | ((int)d << 4) | (int)c; };
    auto issued = [&](npu_tiles t, dma_direction d, npu_it_channel c) { tokens[tkey(t, d, c)]++; };
    auto waited = [&](npu_tiles t, dma_direction d, npu_it_channel c) { seq->npu_dma_wait(t, d, c); tokens[tkey(t, d, c)]--; };
    // inputs of the norm tile and the RoPE tile
    seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT3, bd_0, it_channel_0,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)(kL * D)}, {0, 0, 0, 1}, PKT_SHIM_TO_RMS, 0, true, aggressive_cache);
    issued(IT3, MM2S, it_channel_0);
    seq->npu_dma_memcpy_nd(sizeof(bf16), rms_arg, MM2S, IT3, bd_1, it_channel_1,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)(4 * D)}, {0, 0, 0, 1}, PKT_RMS_WEIGHTS, 0, true, aggressive_cache);
    issued(IT3, MM2S, it_channel_1);
    if (!swa) {
        seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT3, bd_2, it_channel_0,
            {0, 0, 0, (uint32_t)V.tg_off}, {1, 1, 1, (uint32_t)V.tabg_n}, {0, 0, 0, 1}, PKT_ROPE_TABLE, 0, true, no_cache);
        issued(IT3, MM2S, it_channel_0);
    } else {
        seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT3, bd_3, it_channel_1,
            {0, 0, 0, (uint32_t)V.ts_off}, {1, 1, 1, (uint32_t)V.tabs_n}, {0, 0, 0, 1}, PKT_SWA_ROPE_TABLE, 0, true, no_cache);
        issued(IT3, MM2S, it_channel_1);
    }
    // output rows: into the first 8*D elements of the next layer's BO (arg 3)
    seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg, S2MM, IT5, bd_10, it_channel_0,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)(kL * D)}, {0, 0, 0, 1}, -1, 0, true);
    issued(IT5, S2MM, it_channel_0);
    // PLE inputs (pli rows, constants) and the two small weight matrices (to G and U)
    seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT4, bd_0, it_channel_0,
        {0, 0, 0, (uint32_t)V.pli_off}, {1, 1, 1, (uint32_t)(kL * P)}, {0, 0, 0, 1}, PKT_PLI, 0, true, no_cache);
    issued(IT4, MM2S, it_channel_0);
    seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg, MM2S, IT4, bd_1, it_channel_1,
        {0, 0, 0, (uint32_t)V.wc_off}, {1, 1, 1, (uint32_t)(D + 32)}, {0, 0, 0, 1}, PKT_PLE_CONST, 0, true, no_cache);
    issued(IT4, MM2S, it_channel_1);
    gemma4e_layer_weight_def& W = sq.layer_weights(type);
    seq->npu_dma_memcpy_nd(sizeof(bf16), w_arg, MM2S, IT5, bd_14, it_channel_0,
        {0, 0, 0, gemma4e_dflash_flash_sequence::weight_elem_offset(W.pli_gate_proj)}, {1, 1, 1, (uint32_t)(P * D)}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
    issued(IT5, MM2S, it_channel_0);
    seq->npu_dma_memcpy_nd(sizeof(bf16), w_arg, MM2S, IT5, bd_15, it_channel_1,
        {0, 0, 0, gemma4e_dflash_flash_sequence::weight_elem_offset(W.pli_up_proj)}, {1, 1, 1, (uint32_t)(P * D)}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
    issued(IT5, MM2S, it_channel_1);
    // qkv weights, then the KV append (needs qkv projection and RoPE to run: the weights are streaming), then the replay
    sq._move_weights(seq, W.attn_qkv);
    const npu_it_channel kvch = swa ? it_channel_1 : it_channel_0;
    if (!skip) {
        for (int pl = 0; pl < 2; pl++) {
            seq->npu_dma_memcpy_nd(sizeof(bf16), kv_arg, S2MM, IT2, pl == 0 ? bd_0 : bd_1, kvch,
                {0, 0, 0, (uint32_t)((size_t)pl * v_off + (size_t)app_row0 * DKv)}, {1, (uint32_t)kNKV, (uint32_t)kL, (uint32_t)DHl},
                {0, (uint32_t)DHl, (uint32_t)DKv, 1}, -1, 0, true);
            issued(IT2, S2MM, kvch);
            waited(IT2, S2MM, kvch);
        }
    }
    for (int pl = 0; pl < 2; pl++) {
        if (V.attnx)   // K (ch0) / V (ch1) once, head-major: repeat h (stride DH) x [rows / G][G][DH], circuits (no packet header)
            seq->npu_dma_memcpy_nd(sizeof(bf16), kv_arg, MM2S, IT2, pl == 0 ? bd_8 : bd_9, pl == 0 ? it_channel_0 : it_channel_1,
                {0, 0, 0, (uint32_t)((size_t)pl * v_off + (size_t)first_row * DKv)},
                {2, (uint32_t)(ax_rows / ax_G), (uint32_t)ax_G, (uint32_t)DHl}, {(uint32_t)DHl, (uint32_t)(ax_G * DKv), (uint32_t)DKv, 1},
                -1, 0, true, no_cache);
        else
            seq->npu_dma_memcpy_nd(sizeof(bf16), kv_arg, MM2S, IT2, pl == 0 ? bd_8 : bd_9, pl == 0 ? it_channel_0 : it_channel_1,
                {0, 0, 0, (uint32_t)((size_t)pl * v_off + (size_t)first_row * DKv)},
                {(uint32_t)kL, (uint32_t)chunks, (uint32_t)kChunk, (uint32_t)DKv}, {0, (uint32_t)(kChunk * DKv), (uint32_t)DKv, 1},
                swa ? 13 : 12, 0, true, no_cache);
        issued(IT2, MM2S, pl == 0 ? it_channel_0 : it_channel_1);
    }
    sq._move_weights(seq, W.attn_output);
    sq._move_weights(seq, W.ffn_up_gate);
    sq._move_weights(seq, W.ffn_down);
    // waits: one per issued token (the output, the KV replay, the shim senders; IT3 MM2S carries two BDs on one channel: x and the RoPE table)
    for (auto& kv : tokens) {
        const npu_tiles t = (npu_tiles)(kv.first >> 8);
        const dma_direction d = (dma_direction)((kv.first >> 4) & 0xF);
        const npu_it_channel c = (npu_it_channel)(kv.first & 0xF);
        while (kv.second > 0) waited(t, d, c);
    }
    for (auto& kv : tokens)
        if (kv.second != 0) throw std::runtime_error("gemma4e_dflash_flash verify: unbalanced task-completion tokens on shim key " + std::to_string(kv.first));
    seq->cmds2seq();
}

// ---------------------------------------------------------------------------------------------------------------------
// host side of the PLE input: pli(token, layer) = bf16 chain of the decode kernel (tools/ple8_ref.py pli_row)
//   t = bf16(W_plmp . x0); t = bf16(t * 0.01976423537605237); t = bf16(t * rsqrt(mean(t^2) + 1e-6) * w_pn); t = bf16(t + x0_pl); pli = bf16(t * 0.7071067811865476)
// ---------------------------------------------------------------------------------------------------------------------
static inline __m512 bf16lo_to_ps(__m256i v) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16)); }

static void pli_layer_rows(const bf16* Wp /*[8 rb][D][32]*/, const float* x0f /*[8][D]*/, const bf16* x0pl_rows /*[8][layers*P] base + layer*P*/, size_t x0pl_stride,
                           const bf16* wpn, int D, int P, bf16* out /*[8][P]*/) {
    float tf[kL][256];
    const int RB = P / 32;
    for (int rb = 0; rb < RB; rb++) {
        __m512 acc[kL][2];
        for (int t = 0; t < kL; t++) { acc[t][0] = _mm512_setzero_ps(); acc[t][1] = _mm512_setzero_ps(); }
        const uint16_t* w = reinterpret_cast<const uint16_t*>(Wp) + (size_t)rb * D * 32;
        for (int k = 0; k < D; k++) {
            const __m512 wl = bf16lo_to_ps(_mm256_loadu_si256((const __m256i*)(w + (size_t)k * 32)));
            const __m512 wh = bf16lo_to_ps(_mm256_loadu_si256((const __m256i*)(w + (size_t)k * 32 + 16)));
            for (int t = 0; t < kL; t++) {
                const __m512 xv = _mm512_set1_ps(x0f[(size_t)t * D + k]);
                acc[t][0] = _mm512_fmadd_ps(wl, xv, acc[t][0]);
                acc[t][1] = _mm512_fmadd_ps(wh, xv, acc[t][1]);
            }
        }
        for (int t = 0; t < kL; t++) {
            _mm512_storeu_ps(&tf[t][rb * 32], acc[t][0]);
            _mm512_storeu_ps(&tf[t][rb * 32 + 16], acc[t][1]);
        }
    }
    for (int t = 0; t < kL; t++) {
        float* tt = tf[t];
        float ms = 0.f;
        for (int i = 0; i < P; i++) {
            const float a = bf2f((bf16)tt[i]);
            tt[i] = bf2f((bf16)(a * 0.01976423537605237f));
            ms += tt[i] * tt[i];
        }
        ms /= (float)P;
        const float rs = 1.0f / std::sqrt(ms + 1e-6f);
        const bf16* x0pl = x0pl_rows + (size_t)t * x0pl_stride;
        for (int i = 0; i < P; i++) {
            float v = bf2f((bf16)(tt[i] * rs * bf2f(wpn[i])));
            v = bf2f((bf16)(v + bf2f(x0pl[i])));
            out[(size_t)t * P + i] = (bf16)(v * 0.7071067811865476f);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// lm_head_8 helpers
// ---------------------------------------------------------------------------------------------------------------------
static void lm8_pack(bf16* x8, const bf16* rows /*[8][D]*/, const bf16* nw, int D) {
    for (int t = 0; t < kL; t++) {
        const bf16* r = rows + (size_t)t * D;
        for (int blk = 0; blk < D / 256; blk++)
            for (int kt = 0; kt < 32; kt++)
                std::memcpy(x8 + (size_t)(blk * 32 + kt) * 64 + t * 8, r + blk * 256 + kt * 8, 8 * sizeof(bf16));
    }
    std::memcpy(x8 + (size_t)kL * D, nw, (size_t)D * sizeof(bf16));
}

// raw [round][col][core][t][32] -> logits[t][V]
static void lm8_unpack(const bf16* raw, bf16* logits, size_t V, int rows) {
    const size_t rounds = V / kLmMPerRound;
#pragma omp parallel for schedule(static) num_threads(8)
    for (long r = 0; r < (long)rounds; r++)
        for (int cc = 0; cc < 32; cc++)
            for (int t = 0; t < rows; t++)
                std::memcpy(logits + (size_t)t * V + (size_t)r * kLmMPerRound + cc * 32, raw + ((size_t)r * 32 + cc) * 256 + t * 32, 32 * sizeof(bf16));
}

// ids_out[t] = argmax of row t; with topk_out != nullptr also the K best ids of every row, best first ([7][K], K <= 16)
bool gemma4e_dflash_flash::Impl::verify_draft_head(const bf16* hidden, const bf16* norm_w, int* ids_out, int* topk_out, int K) {
    Spec& S = *this->spec;
    if (!verify_fused_ready() || !S.verify->lm8_mgr) return false;
    Spec::Verify& V = *S.verify;
    std::vector<bf16> rows((size_t)kL * D, bf16(0.0f));
    std::memcpy(rows.data(), hidden, (size_t)(kBlock - 1) * D * sizeof(bf16));
    lm8_pack(V.lm8_x.data(), rows.data(), norm_w, (int)D);
    V.lm8_x.sync_to_device();
    V.lm8(V.lm8_y, this->lm_head_weights, V.lm8_x);
    V.lm8_y.sync_from_device();
    const bf16* y = V.lm8_y.data();
    const size_t rounds = this->vocab_size_padded / kLmMPerRound;
#pragma omp parallel for schedule(static, 1) num_threads(kBlock - 1)
    for (int t = 0; t < kBlock - 1; t++) {
        if (topk_out != nullptr && K > 0) {
            // top-K by insertion into a small sorted list (almost every value fails the threshold test)
            float tv[16]; int ti[16];
            const int kk = std::min(K, 16);
            for (int j = 0; j < kk; j++) { tv[j] = -1e30f; ti[j] = 0; }
            for (size_t r = 0; r < rounds; r++)
                for (int cc = 0; cc < 32; cc++) {
                    const bf16* p = y + ((r * 32 + cc) * 256) + t * 32;
                    for (int n = 0; n < 32; n++) {
                        const float v = bf2f(p[n]);
                        if (v <= tv[kk - 1]) continue;
                        const int id = (int)(r * 1024 + cc * 32 + n);
                        if (id >= (int)this->vocab_size) continue;
                        int j = kk - 1;
                        while (j > 0 && tv[j - 1] < v) { tv[j] = tv[j - 1]; ti[j] = ti[j - 1]; j--; }
                        tv[j] = v; ti[j] = id;
                    }
                }
            for (int j = 0; j < kk; j++) topk_out[t * K + j] = ti[j];
            ids_out[t] = ti[0];
            continue;
        }
        int best = 0;
        float bv = bf2f(y[t * 32]);
        for (size_t r = 0; r < rounds; r++)
            for (int cc = 0; cc < 32; cc++) {
                const bf16* p = y + ((r * 32 + cc) * 256) + t * 32;
                for (int n = 0; n < 32; n++) {
                    const float v = bf2f(p[n]);
                    if (v > bv) { bv = v; best = (int)(r * 1024 + cc * 32 + n); }
                }
            }
        ids_out[t] = best;
    }
    return true;
}

// compaction, instruction streams and the 42-run chain of the next fused verify at context length ctx0
void gemma4e_dflash_flash::Impl::verify_prepare(int ctx0) {
    Spec& S = *this->spec;
    Spec::Verify& V = *S.verify;
    if (V.prepared && V.prepared_ctx == ctx0) return;
    const auto t_work = clk::now();
    // room for the 8 rows BEFORE the batch (a compaction inside it would drop positions a rollback needs; gemma4e_dflash_flash_swa.hpp)
    if (swa_needs_compaction(ctx0 + kL, this->swa_base)) {
        this->swa_compact(ctx0);
        this->decode_seq_stale = true;        // the decode streams bake the base in: forward() regenerates them lazily
    }
    {   // optional safety switch, off by default: FLM_DFLASH_VERIFY_RELOAD=N rebuilds the verify hardware context (same BOs) every N
        // dispatches.  It was the workaround for the dispatch-count hang (one task-completion token per dispatch left in the shim queue,
        // fixed in verify_gen_seq by the token ledger; FLM_Xclbin/Gemma4/verify_decoding/notes/hang_rootcause.md).
        static const long long reload_every = std::getenv("FLM_DFLASH_VERIFY_RELOAD") ? std::atoll(std::getenv("FLM_DFLASH_VERIFY_RELOAD")) : 0;
        static const int per_cycle = std::getenv("FLM_DFLASH_VERIFY_NLAYERS") ? std::atoi(std::getenv("FLM_DFLASH_VERIFY_NLAYERS")) : (int)num_hidden_layers;
        if (reload_every > 0 && V.runs_since_load + per_cycle > reload_every) {
            auto tr = clk::now();
            V.open_context();
            static const bool dbg_r = std::getenv("FLM_DFLASH_DEBUG") != nullptr;
            if (dbg_r) std::fprintf(stderr, "[verify] hardware context rebuilt in %lld us\n", us_since(tr));
        }
    }
    this->sequence->swa_base = this->swa_base;
    static const gemma4e_layer_type_t kinds[4] = {e_gemma4e_swa_layer, e_gemma4e_global_layer, e_gemma4e_swa_layer_skip, e_gemma4e_global_layer_skip};
    for (auto k : kinds) this->verify_gen_seq(V.app[(int)k].seq(), k, ctx0);
    static const bool step_mode = std::getenv("FLM_DFLASH_VERIFY_STEP") != nullptr;
    if (!step_mode) {
        V.run.reset();
        for (int i = 0; i < num_hidden_layers; i++) {
            const gemma4e_layer_type_t type = layer_types[i];
            buffer<bf16>& kv = (type == e_gemma4e_global_layer_skip) ? this->kv_caches[last_global_kv_cache_layer_idx]
                             : (type == e_gemma4e_swa_layer_skip)    ? this->kv_caches[last_swa_kv_cache_layer_idx]
                                                                     : this->kv_caches[i];
            V.run.add(V.app[(int)type].create_run(V.xbo[i], this->proj_weights[i], this->rms_weights[i], V.xbo[i + 1], kv));
        }
    }
    V.prepared_ctx = ctx0;
    V.prepared = true;
    S.t_v_prep_work += us_since(t_work);     // CPU time of the preparation (overlapped: see dflash_draft)
}

void gemma4e_dflash_flash::Impl::verify_prepare_async(int ctx0) {
    Spec::Verify& V = *this->spec->verify;
    if (V.prep_future.valid()) V.prep_future.get();
    V.prepared = false;
    V.prep_future = std::async(std::launch::async, [this, ctx0] { this->verify_prepare(ctx0); });
}

// ---------------------------------------------------------------------------------------------------------------------
// one fused verify cycle
// ---------------------------------------------------------------------------------------------------------------------
const bf16* gemma4e_dflash_flash::Impl::verify_fused(int last_token, const int* drafts, int n_drafts) {
    Spec& S = *this->spec;
    if (!verify_fused_ready()) return nullptr;
    Spec::Verify& V = *S.verify;
    const int ctx0 = this->current_context_length;
    if (n_drafts < 0 || n_drafts > kBlock - 1 || ctx0 + kBlock > this->MAX_L) return nullptr;
    auto t0 = clk::now();
    if (this->is_preload_launched) { this->is_preload_launched = false; flm::npu::wait_for_completion(this->pre_load_run, "pre_load_run"); }
    // room for the 8 rows BEFORE the batch is made by verify_prepare (never inside a batch; gemma4e_dflash_flash_swa.hpp)
    auto t_prep = clk::now();
    if (V.prep_future.valid()) V.prep_future.get();
    if (!(V.prepared && V.prepared_ctx == ctx0)) this->verify_prepare(ctx0);
    V.prepared = false;
    S.t_v_seq += us_since(t_prep);      // what is left of the preparation after the draft overlapped it
    this->dirty_rows = std::max(this->dirty_rows, (size_t)(ctx0 + kBlock));

    int tok[kBlock];
    tok[0] = last_token;
    for (int i = 1; i <= n_drafts; i++) tok[i] = drafts[i - 1];
    for (int i = n_drafts + 1; i < kBlock; i++) tok[i] = tok[n_drafts];    // padding rows: ignored, their K/V rows are zeroed again on commit

    // ---- inputs ----
    // embedding rows, x0 (float) and the per layer slices of the pli embedding
    std::vector<float> x0f((size_t)kL * D);
    const size_t plstride = (size_t)num_hidden_layers * PLI_D;
    for (int t = 0; t < kL; t++) {
        buffer<bf16> e = this->embedding->forward(tok[t]);
        std::memcpy(V.xbo[0].data() + (size_t)t * D, e.data(), (size_t)D * sizeof(bf16));
        std::memcpy(V.x0.data() + (size_t)t * D, e.data(), (size_t)D * sizeof(bf16));
        for (int k = 0; k < D; k++) x0f[(size_t)t * D + k] = bf2f(e.data()[k]);
        buffer<bf16> p = this->pli_embedding->forward(tok[t]);
        std::memcpy(V.pl.data() + (size_t)t * plstride, p.data(), plstride * sizeof(bf16));
    }
    // RoPE tables of the two layer kinds: row t = [cos(DH/2) | sin(DH/2)] of position ctx0 + t
    std::vector<bf16> tabg((size_t)kL * V.DHg), tabs((size_t)kL * V.DHs);
    for (int t = 0; t < kL; t++) {
        const int pos = ctx0 + t;
        for (int j = 0; j < V.DHg / 2; j++) {
            tabg[(size_t)t * V.DHg + j] = (bf16)cos(gemma4e_cpu_func::inv_freq_global[j] * pos);
            tabg[(size_t)t * V.DHg + V.DHg / 2 + j] = (bf16)sin(gemma4e_cpu_func::inv_freq_global[j] * pos);
        }
        for (int j = 0; j < V.DHs / 2; j++) {
            tabs[(size_t)t * V.DHs + j] = (bf16)cos(gemma4e_cpu_func::inv_freq_swa[j] * pos);
            tabs[(size_t)t * V.DHs + V.DHs / 2 + j] = (bf16)sin(gemma4e_cpu_func::inv_freq_swa[j] * pos);
        }
    }
    // pli rows of every layer (parallel over layers)
#pragma omp parallel for schedule(dynamic, 1) num_threads(8)
    for (int i = 0; i < num_hidden_layers; i++) {
        const gemma4e_layer_type_t type = layer_types[i];
        const bf16* Wp = reinterpret_cast<const bf16*>(this->proj_weights[i].data() + desc.weight_desc(type).pli_down_proj.offset);
        const bf16* wpn = this->rope_rms_weights[i].data() + desc.get_pli_norm_offset(type);
        pli_layer_rows(Wp, x0f.data(), V.pl.data() + (size_t)i * PLI_D, plstride, wpn, (int)D, (int)PLI_D, V.pli_rows.data() + (size_t)i * kL * PLI_D);
    }
    for (int i = 0; i < num_hidden_layers; i++) {
        const bool swa = is_swa_layer(layer_types[i]);
        bf16* xp = V.xbo[i].data();
        if (swa) std::memcpy(xp + V.ts_off, tabs.data(), (size_t)kL * V.DHs * sizeof(bf16));
        else     std::memcpy(xp + V.tg_off, tabg.data(), (size_t)kL * V.DHg * sizeof(bf16));
        std::memcpy(xp + V.pli_off, V.pli_rows.data() + (size_t)i * kL * PLI_D, (size_t)kL * PLI_D * sizeof(bf16));
        V.xbo[i].sync_to_device();
    }
    S.t_v_prep += us_since(t0);

    // ---- instruction streams and the 42-run chain: built by verify_prepare (concurrently with the host draft when it ran first) ----
    // FLM_DFLASH_VERIFY_STEP=1 (debug): one run per layer with a host wait and a timeout instead of one 42-run runlist, to find a hanging layer
    static const bool step_mode = std::getenv("FLM_DFLASH_VERIFY_STEP") != nullptr;
    auto kv_of = [&](int i) -> buffer<bf16>& {
        const gemma4e_layer_type_t type = layer_types[i];
        return (type == e_gemma4e_global_layer_skip) ? this->kv_caches[last_global_kv_cache_layer_idx]
             : (type == e_gemma4e_swa_layer_skip)    ? this->kv_caches[last_swa_kv_cache_layer_idx]
                                                     : this->kv_caches[i];
    };

    static const bool dbg = std::getenv("FLM_DFLASH_DEBUG") != nullptr;
    if (dbg) std::fprintf(stderr, "[verify] ctx0 %d base %d: executing 42-layer runlist\n", ctx0, this->swa_base);
    t0 = clk::now();
    if (step_mode) {
        static const int n_layers_dbg = std::getenv("FLM_DFLASH_VERIFY_NLAYERS") ? std::atoi(std::getenv("FLM_DFLASH_VERIFY_NLAYERS")) : (int)num_hidden_layers;   // debug: results are garbage
        for (int i = 0; i < n_layers_dbg; i++) {
            V.runs_since_load++;
            auto r = V.app[(int)layer_types[i]].create_run(V.xbo[i], this->proj_weights[i], this->rms_weights[i], V.xbo[i + 1], kv_of(i));
            ert_cmd_state st = ERT_CMD_STATE_NEW;
            try {
                r.start();
#if defined(FLM_USE_HRX)
                st = r.wait();
#else
                st = r.wait(std::chrono::milliseconds(3000));
#endif
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[verify] layer %d (type %d) threw: %s (ctx0 %d)\n", i, (int)layer_types[i], e.what(), ctx0);
                throw;
            }
            if (std::getenv("FLM_DFLASH_STATS")) {   // per layer output statistics (finite check / rms)
                V.xbo[i + 1].sync_from_device();
                for (int t = 0; t < kL; t++) {
                    double ss = 0; int bad = 0;
                    for (int j = 0; j < (int)D; j++) { const float v = bf2f(V.xbo[i + 1].data()[(size_t)t * D + j]); if (!std::isfinite(v)) bad++; else ss += (double)v * v; }
                    std::fprintf(stderr, "[verify] ctx0 %d layer %d row %d rms %.3f nonfinite %d\n", ctx0, i, t, std::sqrt(ss / D), bad);
                }
            }
            if (st != ERT_CMD_STATE_COMPLETED) {
                std::fprintf(stderr, "[verify] layer %d (type %d) did not complete: state %d, ctx0 %d\n", i, (int)layer_types[i], (int)st, ctx0);
                throw std::runtime_error("verify layer timeout");
            }
        }
    } else {
        V.run.execute();
        flm::npu::wait_for_completion(V.run, "gemma4e dflash-flash verify runlist");
    }
    if (dbg) std::fprintf(stderr, "[verify] runlist done in %lld us\n", us_since(t0));
    S.t_v_run += us_since(t0);
    V.runs_since_load += (step_mode ? 0 : num_hidden_layers);

    // ---- taps ----
    t0 = clk::now();
    for (int t = 0; t < kTaps; t++) V.xbo[kTapLayers[t] + 1].sync_from_device();
    for (int i = 0; i <= n_drafts; i++) {
        const long long p = (long long)ctx0 + i;
        if (S.tap_hi != p) { S.tap_lo = S.tap_hi = p; }
        for (int t = 0; t < kTaps; t++)
            std::memcpy(S.tap_ring.data() + ((size_t)(p % kRing) * kTaps + t) * D, V.xbo[kTapLayers[t] + 1].data() + (size_t)i * D, (size_t)D * sizeof(bf16));
        S.tap_hi = p + 1;
        if (S.tap_hi - S.tap_lo > kRing) S.tap_lo = S.tap_hi - kRing;
        S.ids.push_back(tok[i]);
    }
    S.t_v_taps += us_since(t0);
    this->spec_async_inject(ctx0, n_drafts + 1);     // draft-context injection of these rows runs during the lm_head_8 below

    // ---- lm_head_8 on the 8 final rows ----
    t0 = clk::now();
    if (dbg) std::fprintf(stderr, "[verify] lm_head_8\n");
    V.xbo[num_hidden_layers].sync_from_device();
    lm8_pack(V.lm8_x.data(), V.xbo[num_hidden_layers].data(), S.target_norm.data(), (int)D);
    V.lm8_x.sync_to_device();
    V.lm8(V.lm8_y, this->lm_head_weights, V.lm8_x);
    V.lm8_y.sync_from_device();
    lm8_unpack(V.lm8_y.data(), S.logit_rows.data(), (size_t)this->vocab_size_padded, kL);
    S.t_v_lm += us_since(t0);

    S.verify_rows = kBlock;
    S.fused_cycles++;
    return S.logit_rows.data();
}

