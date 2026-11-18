// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Stand-alone host for the 8-token attention stage designs (plan S6.5 second half / S6.9 / S6.10).
//   attn8.exe --stage rope|attn --layer N --ctx C --dir D --xclbin build/.../attn8rope.xclbin [--repeat R] [--timeout ms]
// Data files come from tools/attn8_ref.py gen; results are written next to them and scored by tools/attn8_ref.py check.
// Built separately from run.exe (host/*.cpp is one executable per folder): see README.md, "Host tests".
#include "typedef.hpp"
#include "buffer.hpp"
#include "decoding_layer.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/utils.hpp"
#include "models.h"
#include "lm_config.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

static std::vector<uint16_t> read_bf16(const std::string& fn, size_t n) {
    std::ifstream f(fn, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + fn);
    std::vector<uint16_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * 2);
    if ((size_t)f.gcount() != n * 2) throw std::runtime_error("short read " + fn + " (" + std::to_string(f.gcount() / 2) + " of " + std::to_string(n) + ")");
    return v;
}

static std::map<std::string, uint32_t> read_rtp_file(const std::string& fn) {
    std::map<std::string, uint32_t> m;
    std::ifstream f(fn);
    std::string name; uint32_t addr;
    while (f >> name >> addr) m[name] = addr;
    return m;
}

constexpr int PKT_QKV = 1;
constexpr int PKT_TABLE = 2;

/// stage "rope": the shim feeds qkv rows + table to the rope tile of the layer's kind and captures q and k/v
static int run_rope_stage(decoding_layer& layer, const std::string& dir, const std::string& xclbin_dir, int repeat, int timeout_ms) {
    const int L = L_TOKENS;
    const bool swa = layer.is_swa_layer, skip = layer.shall_skip_kv;
    const int DH = layer.DH;
    const int NQ = 8, NKV = 2;
    const size_t chunk = (size_t)L * DH;
    const size_t qkv_el = chunk * (NQ + (skip ? 0 : 2 * NKV));
    const size_t table_el = (size_t)L * DH + 2 * DH;
    auto rtp = read_rtp_file(xclbin_dir + "/attn8rope_rtp_addresses.txt");
    const std::string rtp_name = swa ? "RTP_SWA_RoPE_SKIP_KV_5_3" : "RTP_RoPE_SKIP_KV_4_3";
    if (!rtp.count(rtp_name)) throw std::runtime_error("rtp file has no " + rtp_name);

    auto qkv = read_bf16(dir + "/qkv_in.bin", qkv_el);
    auto tab = read_bf16(dir + "/table.bin", table_el);
    buffer<bf16> in_bo = layer.layer_app.create_bo_buffer<bf16>(((qkv_el + table_el) + 8191) / 8192 * 8192);
    in_bo.memset((bf16)0.0f);
    std::memcpy(in_bo.data(), qkv.data(), qkv_el * 2);
    std::memcpy(in_bo.data() + qkv_el, tab.data(), table_el * 2);
    in_bo.sync_to_device();
    const size_t q_el = chunk * NQ, kv_el = chunk * 2 * NKV;
    buffer<bf16> q_bo = layer.layer_app.create_bo_buffer<bf16>((q_el + 8191) / 8192 * 8192);
    buffer<bf16> kv_bo = layer.layer_app.create_bo_buffer<bf16>((kv_el + 8191) / 8192 * 8192);
    q_bo.memset((bf16)0.0f); kv_bo.memset((bf16)0.0f);
    q_bo.sync_to_device(); kv_bo.sync_to_device();

    npu_sequence* seq = layer.layer_app.seq();
    seq->clear_cmds();
    const npu_tiles rope_tile = swa ? CT33 : CT23;
    seq->rtp_write(rope_tile, rtp[rtp_name], skip ? 1 : 0);
    const npu_tiles feed = swa ? IT4 : IT3;
    // table (ch1), qkv rows (ch0): packet streams into the rope tile
    seq->npu_dma_memcpy_nd(sizeof(bf16), layer.x_arg_id, MM2S, feed, bd_1, it_channel_1,
        {0, 0, 0, (uint32_t)qkv_el}, {1, 1, 1, (uint32_t)table_el}, {0, 0, 0, 1}, PKT_TABLE, 0, true, no_cache);
    seq->npu_dma_memcpy_nd(sizeof(bf16), layer.x_arg_id, MM2S, feed, bd_0, it_channel_0,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)qkv_el}, {0, 0, 0, 1}, PKT_QKV, 0, true, no_cache);
    // q back: [head][t][DH]
    seq->npu_dma_memcpy_nd(sizeof(bf16), layer.rms_arg_id, S2MM, feed, bd_10, it_channel_0,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)q_el}, {0, 0, 0, 1}, -1, 0, true);
    if (!skip) {
        const npu_it_channel kvch = swa ? it_channel_1 : it_channel_0;
        // k (both heads) then v (both heads), in the order the rope tile emits them
        seq->npu_dma_memcpy_nd(sizeof(bf16), layer.kv_cache_arg_id, S2MM, IT2, bd_0, kvch,
            {0, 0, 0, 0}, {1, 1, 1, (uint32_t)(chunk * NKV)}, {0, 0, 0, 1}, -1, 0, true);
        seq->npu_dma_wait(IT2, S2MM, kvch);
        seq->npu_dma_memcpy_nd(sizeof(bf16), layer.kv_cache_arg_id, S2MM, IT2, bd_1, kvch,
            {0, 0, 0, (uint32_t)(chunk * NKV)}, {1, 1, 1, (uint32_t)(chunk * NKV)}, {0, 0, 0, 1}, -1, 0, true);
        seq->npu_dma_wait(IT2, S2MM, kvch);
    }
    seq->npu_dma_wait(feed, S2MM, it_channel_0);
    seq->npu_dma_wait(feed, MM2S, it_channel_0);
    seq->npu_dma_wait(feed, MM2S, it_channel_1);
    seq->cmds2seq();

    std::vector<double> us;
    std::vector<uint16_t> first_q, first_kv;
    for (int r = 0; r < repeat; r++) {
        q_bo.memset((bf16)0.0f); kv_bo.memset((bf16)0.0f);
        q_bo.sync_to_device(); kv_bo.sync_to_device();
        auto t0 = std::chrono::steady_clock::now();
        ert_cmd_state st = ERT_CMD_STATE_NEW;
        try {
            xrt::run run = layer.layer_app.create_run(in_bo, layer.all_proj_weight_buffer, q_bo, layer.rope_rms_buffer, kv_bo);
            run.start();
            st = run.wait(std::chrono::milliseconds(timeout_ms));
        } catch (const std::exception& e) {
            std::cerr << "ERROR attn8 rope: dispatch threw: " << e.what() << std::endl;
            return 3;
        }
        double d = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (st != ERT_CMD_STATE_COMPLETED) { std::cerr << "ERROR attn8 rope: dispatch state " << (int)st << std::endl; return 3; }
        us.push_back(d);
        q_bo.sync_from_device(); kv_bo.sync_from_device();
        std::vector<uint16_t> q((uint16_t*)q_bo.data(), (uint16_t*)q_bo.data() + q_el), kv((uint16_t*)kv_bo.data(), (uint16_t*)kv_bo.data() + kv_el);
        if (r == 0) { first_q = q; first_kv = kv; }
        else if (q != first_q || (!skip && kv != first_kv)) { std::cerr << "ERROR attn8 rope: repeat " << r << " differs from repeat 0" << std::endl; return 4; }
    }
    { std::ofstream f(dir + "/q_out.bin", std::ios::binary); f.write((const char*)first_q.data(), first_q.size() * 2); }
    if (!skip) { std::ofstream f(dir + "/kv_out.bin", std::ios::binary); f.write((const char*)first_kv.data(), first_kv.size() * 2); }
    std::sort(us.begin(), us.end());
    std::cout << "attn8 rope layer " << (swa ? "swa" : "global") << (skip ? "-skip" : "") << " dispatches " << us.size() << " min_us " << (int)us.front()
              << " median_us " << (int)us[us.size() / 2] << std::endl;
    return 0;
}


/// stage "attn": rope + q park + attention.  KV cache layout of this test (the engine contract for the 8-row design):
///   global : two planes [MAXL][2 heads x DH] (K plane then V plane), row = absolute position
///   swa    : (added with S6.10) mirrored ring
static int run_attn_stage(decoding_layer& layer, const std::string& dir, const std::string& xclbin_dir, int ctx, int poison_mask, int repeat, int timeout_ms) {
    const bool poison = poison_mask != 0;
    const int L = L_TOKENS;
    const bool swa = layer.is_swa_layer, skip = layer.shall_skip_kv;
    const int DH = layer.DH, NQ = 8, NKV = 2;
    const size_t DKv = (size_t)NKV * DH;
    const size_t chunk = (size_t)L * DH;
    const size_t qkv_el = chunk * (NQ + (skip ? 0 : 2 * NKV));
    const size_t table_el = (size_t)L * DH + 2 * DH;
    const size_t DQ = (size_t)NQ * DH;
    const int L_CHUNK = 16;
    auto rtp = read_rtp_file(xclbin_dir + "/attn8attn_rtp_addresses.txt");
    auto need = [&](const std::string& n) { if (!rtp.count(n)) throw std::runtime_error("rtp file has no " + n); return rtp[n]; };

    // ---- cache rows seen by the kernels ----
    // global: row = absolute position.  swa: LINEAR window buffer, row = position - base (host compaction keeps base current, see
    // notes/attn8_L8_results.md); the test leaves `lead` rows before the window start so the replay offset is not zero.
    const int W = 512;
    int base = 0, first_row = 0, n_rows;       // replayed range [first_row, first_row + n_rows) in buffer rows
    int S = 0;
    if (!swa) { n_rows = ctx + L; }
    else {
        S = ctx - (W - 1) > 0 ? ctx - (W - 1) : 0;
        S = S / 16 * 16;
        const int lead = std::min(S, 37);
        base = S - lead; first_row = lead; n_rows = ctx + L - S;
    }
    const size_t plane_rows = (size_t)(first_row + (n_rows + L_CHUNK - 1) / L_CHUNK * L_CHUNK) + 64;
    const int L_max = n_rows;
    const int chunks = (n_rows + L_CHUNK - 1) / L_CHUNK;
    const size_t v_off = plane_rows * DKv;
    const int app_row0 = ctx - base;           // buffer row of the first appended token

    auto qkv = read_bf16(dir + "/qkv_in.bin", qkv_el);
    auto tab = read_bf16(dir + "/table.bin", table_el);
    const int ctx_rows = ctx + (skip ? L : 0);
    auto kctx = ctx_rows ? read_bf16(dir + "/kctx.bin", (size_t)ctx_rows * DKv) : std::vector<uint16_t>();
    auto vctx = ctx_rows ? read_bf16(dir + "/vctx.bin", (size_t)ctx_rows * DKv) : std::vector<uint16_t>();

    buffer<bf16> in_bo = layer.layer_app.create_bo_buffer<bf16>(((qkv_el + table_el) + 8191) / 8192 * 8192);
    in_bo.memset((bf16)0.0f);
    std::memcpy(in_bo.data(), qkv.data(), qkv_el * 2);
    std::memcpy(in_bo.data() + qkv_el, tab.data(), table_el * 2);
    in_bo.sync_to_device();
    buffer<bf16> o_bo = layer.layer_app.create_bo_buffer<bf16>(((size_t)L * DQ + 8191) / 8192 * 8192);
    buffer<bf16> kv_bo = layer.layer_app.create_bo_buffer<bf16>(2 * plane_rows * DKv);
    auto fill_kv = [&]() {
        kv_bo.memset((bf16)0.0f);
        if (poison) {
            // Contract of the kernels (found with this test): padding rows of the replayed range must be finite; the V mmul runs through
            // bfp16 blocks of 8 keys, so a large V value in the same 8-row block as a valid key destroys it, and NaN in V is never
            // tolerated (0 x NaN).  The engine guarantees it (rows past the committed context are zero).  So: K rows >= ctx+8 -> NaN,
            // V rows from the next 8-row block boundary on -> 1e4; bit 2 additionally poisons V with NaN (expected to FAIL, documents the limit).
            uint16_t* p = (uint16_t*)kv_bo.data();
            const size_t k_from = (size_t)(app_row0 + L), v_from = (size_t)((app_row0 + L + 7) / 8 * 8);
            for (size_t r = 0; r < plane_rows; r++)
                for (size_t j = 0; j < DKv; j++) {
                    if ((poison_mask & 1) && r >= k_from) p[r * DKv + j] = 0x7FC0;
                    if ((poison_mask & 2) && r >= v_from) p[v_off + r * DKv + j] = 0x461C;
                    if ((poison_mask & 8) && r >= k_from) p[v_off + r * DKv + j] = 0x7FC0;
                }
        }
        std::memcpy(kv_bo.data(), kctx.data() + (size_t)base * DKv, (size_t)(ctx_rows - base) * DKv * 2);
        std::memcpy(kv_bo.data() + v_off, vctx.data() + (size_t)base * DKv, (size_t)(ctx_rows - base) * DKv * 2);
        kv_bo.sync_to_device();
    };

    npu_sequence* seq = layer.layer_app.seq();
    seq->clear_cmds();
    const npu_tiles rope_tile = swa ? CT33 : CT23;
    seq->rtp_write(rope_tile, need(swa ? "RTP_SWA_RoPE_SKIP_KV_5_3" : "RTP_RoPE_SKIP_KV_4_3"), skip ? 1 : 0);
    const npu_tiles qk_tile = swa ? layer.swa_attn_qk_tile : layer.attn_qk_tile;
    const npu_tiles kv_tile = swa ? layer.swa_attn_kv_tile : layer.attn_kv_tile;
    {
        const uint32_t qk_addr = need(swa ? "RTP_L_swa_attn_qk_core_4_2" : "RTP_L_attn_qk_core_2_2");
        seq->rtp_write(qk_tile, qk_addr, L_max);
        if (swa) {   // swa qk: L[1] = hi0 = ctx - S, L[2] = u0 = ctx + 1 - W - S (kernels/swa_attn_qk.cc)
            seq->rtp_write(qk_tile, qk_addr + 4, (uint32_t)(ctx - S));
            seq->rtp_write(qk_tile, qk_addr + 8, (uint32_t)(ctx + 1 - W - S));
        }
    }
    seq->rtp_write(kv_tile, need(swa ? "RTP_L_swa_attn_kv_core_5_2" : "RTP_L_attn_kv_core_3_2"), L_max);
    const npu_tiles feed = swa ? IT4 : IT3;
    seq->npu_dma_memcpy_nd(sizeof(bf16), layer.x_arg_id, MM2S, feed, bd_1, it_channel_1,
        {0, 0, 0, (uint32_t)qkv_el}, {1, 1, 1, (uint32_t)table_el}, {0, 0, 0, 1}, PKT_TABLE, 0, true, no_cache);
    seq->npu_dma_memcpy_nd(sizeof(bf16), layer.x_arg_id, MM2S, feed, bd_0, it_channel_0,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)qkv_el}, {0, 0, 0, 1}, PKT_QKV, 0, true, no_cache);
    // o back: [t][NQ*DH]
    seq->npu_dma_memcpy_nd(sizeof(bf16), layer.rms_arg_id, S2MM, feed, bd_10, it_channel_0,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)(L * DQ)}, {0, 0, 0, 1}, -1, 0, true);
    const npu_it_channel kvch = swa ? it_channel_1 : it_channel_0;
    if (!skip) {
        // append the 8 new K rows then V rows BEFORE any replay: stream [head][t][DH] -> row (ctx+t), column head*DH + d
        for (int pl = 0; pl < 2; pl++) {
            seq->npu_dma_memcpy_nd(sizeof(bf16), layer.kv_cache_arg_id, S2MM, IT2, pl == 0 ? bd_0 : bd_1, kvch,
                {0, 0, 0, (uint32_t)(pl * v_off + (size_t)app_row0 * DKv)}, {1, (uint32_t)NKV, (uint32_t)L, (uint32_t)DH}, {0, (uint32_t)DH, (uint32_t)DKv, 1}, -1, 0, true);
            seq->npu_dma_wait(IT2, S2MM, kvch);
        }
    }
    // replay: the context once per token (repeat count L carried by the queue), uniform chunk count
    for (int pl = 0; pl < 2; pl++) {
        seq->npu_dma_memcpy_nd(sizeof(bf16), layer.kv_cache_arg_id, MM2S, IT2, pl == 0 ? bd_8 : bd_9, pl == 0 ? it_channel_0 : it_channel_1,
            {0, 0, 0, (uint32_t)(pl * v_off + (size_t)first_row * DKv)},
            {(uint32_t)L, (uint32_t)chunks, (uint32_t)L_CHUNK, (uint32_t)DKv}, {0, (uint32_t)(L_CHUNK * DKv), (uint32_t)DKv, 1},
            swa ? 13 : 12, 0, true, no_cache);
    }
    seq->npu_dma_wait(feed, S2MM, it_channel_0);
    seq->npu_dma_wait(IT2, MM2S, it_channel_0);
    seq->npu_dma_wait(IT2, MM2S, it_channel_1);
    seq->npu_dma_wait(feed, MM2S, it_channel_0);
    seq->npu_dma_wait(feed, MM2S, it_channel_1);
    seq->cmds2seq();

    std::vector<double> us;
    std::vector<uint16_t> first_o, first_kv;
    for (int r = 0; r < repeat; r++) {
        o_bo.memset((bf16)0.0f); o_bo.sync_to_device();
        fill_kv();
        auto t0 = std::chrono::steady_clock::now();
        ert_cmd_state st = ERT_CMD_STATE_NEW;
        try {
            xrt::run run = layer.layer_app.create_run(in_bo, layer.all_proj_weight_buffer, o_bo, layer.rope_rms_buffer, kv_bo);
            run.start();
            st = run.wait(std::chrono::milliseconds(timeout_ms));
        } catch (const std::exception& e) {
            std::cerr << "ERROR attn8 attn: dispatch threw: " << e.what() << std::endl;
            return 3;
        }
        double d = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (st != ERT_CMD_STATE_COMPLETED) { std::cerr << "ERROR attn8 attn: dispatch state " << (int)st << std::endl; return 3; }
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
        if (r == 0) { first_o = o; first_kv = kvn; }
        else if (o != first_o || kvn != first_kv) { std::cerr << "ERROR attn8 attn: repeat " << r << " differs from repeat 0" << std::endl; return 4; }
    }
    { std::ofstream f(dir + "/o_out.bin", std::ios::binary); f.write((const char*)first_o.data(), first_o.size() * 2); }
    if (!skip) { std::ofstream f(dir + "/kv_new.bin", std::ios::binary); f.write((const char*)first_kv.data(), first_kv.size() * 2); }
    std::sort(us.begin(), us.end());
    std::cout << "attn8 attn layer " << (swa ? "swa" : "global") << (skip ? "-skip" : "") << " ctx " << ctx << " dispatches " << us.size() << " min_us " << (int)us.front()
              << " median_us " << (int)us[us.size() / 2] << std::endl;
    return 0;
}

#include "attnx_host.hpp"

int main(int argc, char** argv) {
    po::options_description desc("attn8");
    desc.add_options()("stage", po::value<std::string>()->default_value("rope"), "rope | attn")
        ("layer", po::value<int>()->default_value(5), "layer index (picks swa/global and skip/non-skip)")
        ("dir", po::value<std::string>()->required(), "data directory")
        ("xclbin", po::value<std::string>()->required(), "stage xclbin")
        ("model_dir", po::value<std::string>()->default_value("q4nx_files/Gemma4-E4B-IT-NPU2"), "model dir")
        ("ctx", po::value<int>()->default_value(0), "attn: context length (cursor)")
        ("poison", po::value<int>()->default_value(0)->implicit_value(3), "attn: fill the cache rows the kernels must not use with NaN (bit 1 = K plane, bit 2 = V plane)")
        ("repeat", po::value<int>()->default_value(1), "dispatches (outputs must be bit-identical)")
        ("timeout", po::value<int>()->default_value(8000), "dispatch timeout in ms")
        ("ctxs", po::value<std::string>()->default_value(""), "attnx: comma list of cursors run one after the other in ONE hardware context (data in <dir>/layer<N>_ctx<C>)")
        ("soak", po::value<int>()->default_value(0), "attnx: extra dispatches round robin over the cursors (outputs compared to each cursor's first)");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    model_list::model_t model = model_list::get_model_type("GEMMA4_E4B");
    LM_Config lm_config;
    lm_config.from_pretrained(vm["model_dir"].as<std::string>());
    lm_config.layer_xclbin_name = vm["xclbin"].as<std::string>();
    decoding_layer::dispatch_timeout_ms = vm["timeout"].as<int>();
    xrt::device npu_inst = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_inst, false);
    npu_app_manager* mgr = xclbin_mgr.register_xclbin(lm_config.layer_xclbin_name);
    const std::string x = lm_config.layer_xclbin_name;
    const std::string xdir = x.substr(0, x.find_last_of('/'));

    decoding_layer layer(model, lm_config, mgr, vm["layer"].as<int>(), 64);
    std::cout << "attn8: layer " << vm["layer"].as<int>() << (layer.is_swa_layer ? " swa" : " global") << (layer.shall_skip_kv ? " skip" : " non-skip")
              << " DH " << layer.DH << " L_TOKENS " << L_TOKENS << std::endl;
    const std::string stage = vm["stage"].as<std::string>();
    if (stage == "rope") return run_rope_stage(layer, vm["dir"].as<std::string>(), xdir, vm["repeat"].as<int>(), vm["timeout"].as<int>());
    if (stage == "attnx") {
        std::vector<int> ctxs;
        const std::string cl = vm["ctxs"].as<std::string>();
        if (cl.empty()) ctxs.push_back(vm["ctx"].as<int>());
        else { std::stringstream ss(cl); std::string t; while (std::getline(ss, t, ',')) ctxs.push_back(std::stoi(t)); }
        return run_attnx(layer, vm["dir"].as<std::string>(), xdir, ctxs, !cl.empty(), vm["poison"].as<int>(), vm["repeat"].as<int>(), vm["soak"].as<int>(), vm["timeout"].as<int>());
    }
    if (stage == "attn") return run_attn_stage(layer, vm["dir"].as<std::string>(), xdir, vm["ctx"].as<int>(), vm["poison"].as<int>(), vm["repeat"].as<int>(), vm["timeout"].as<int>());
    std::cerr << "stage not implemented: " << stage << std::endl;
    return 2;
}
