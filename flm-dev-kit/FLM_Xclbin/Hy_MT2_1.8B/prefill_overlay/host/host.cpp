// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Test of the Hy-MT2 prefill overlay: dequant_mm and attention in ONE xclbin.
//
//   1. correctness through every switch direction, against the engine's own
//      host references (FLM_DLL/detail/hunyuan_npu/cpu_functions.hpp):
//        dequant_mm  cpu_func::_dequant_mm on the same reordered q4_k bytes
//        attention   cpu_func::_attention on the four-way split KV cache
//   2. speed of each op and of a switch
// See README.md.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "npu_utils/npu_utils.hpp"
#include "utils/vm_args.hpp"
#include "metrices.hpp"
#include "hunyuan_model_def.hpp"
#include "utils/avx512_util.hpp"   // cpu_functions.hpp needs it first, as in the engine
#include "cpu_functions.hpp"
#include "hunyuan_overlay.hpp"
#include "hunyuan_bfp16.hpp"

using clk = std::chrono::steady_clock;

static double us_since(clk::time_point t) {
    return std::chrono::duration<double, std::micro>(clk::now() - t).count();
}

/// mean and fastest of `iters` calls; the fastest is the one to compare when other
/// processes load the CPU (a busy host delays the wake-up after a run by up to ~1 ms)
template <typename F>
static std::string time_runs(int iters, F&& run) {
    double total = 0.0, best = 1e30;
    for (int it = 0; it < iters; it++) {
        auto t = clk::now();
        run();
        const double us = us_since(t);
        total += us;
        best = std::min(best, us);
    }
    std::ostringstream o;
    o << std::fixed << std::setprecision(1) << total / iters << " us (min " << best << ")";
    return o.str();
}

static bool check(const std::string& label, buffer<bf16>& got, buffer<bf16>& ref, size_t n, float min_cos) {
    buffer<bf16> g(n), r(n);
    memcpy(g.data(), got.data(), n * sizeof(bf16));
    memcpy(r.data(), ref.data(), n * sizeof(bf16));
    error_metrics m = get_error_metrics(g, r);
    const bool ok = m.CosineSimilarity > min_cos;
    std::cout << "[check] " << std::left << std::setw(46) << label << std::right
              << " cos " << std::fixed << std::setprecision(6) << m.CosineSimilarity
              << "  rel L2 " << std::setprecision(4) << m.RelativeL2 << (ok ? "  OK" : "  FAIL") << std::endl;
    return ok;
}

constexpr size_t Q4K_BYTES = 4736;   // one q4_k chunk, see cpu_functions.hpp
constexpr size_t Q4K_S = 4608, Q4K_M = 4672;   // byte offsets of bf16 S[32], M[32]

/// random q4_k weights for a Dout x Din matrix in the checkpoint's block order
/// (32 x 256 chunks: uint8 s[8][32], m[8][32], nibbles [256][16 B], bf16 S[32], M[32] (negated))
static buffer<u8> random_q4k(uint32_t Dout, uint32_t Din, std::mt19937& rng) {
    const size_t blocks = (size_t)(Dout / 32) * (Din / 256);
    buffer<u8> w(blocks * Q4K_BYTES);
    std::uniform_int_distribution<int> byte(0, 255), six_bits(0, 63);
    std::normal_distribution<float> S_dist(0.0f, 1.3e-4f), M_dist(0.0f, 1.0e-3f);
    for (size_t b = 0; b < blocks; b++) {
        u8* chunk = w.data() + b * Q4K_BYTES;
        for (int i = 0; i < 512; i++) chunk[i] = (u8)six_bits(rng);
        for (int i = 512; i < (int)Q4K_S; i++) chunk[i] = (u8)byte(rng);
        bf16* S = (bf16*)(chunk + Q4K_S);
        bf16* M = (bf16*)(chunk + Q4K_M);
        for (int r = 0; r < 32; r++) {
            S[r] = bf16(std::fabs(S_dist(rng)) + 1e-6f);
            M[r] = bf16(-std::fabs(M_dist(rng)));
        }
    }
    return w;
}

static uint64_t fnv1a(const void* p, size_t n) {
    const u8* b = (const u8*)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

static void print_hash(const std::string& label, const void* p, size_t bytes) {
    std::cout << "[hash] " << std::left << std::setw(50) << label << std::right << " "
              << std::hex << std::setw(16) << std::setfill('0') << fnv1a(p, bytes)
              << std::dec << std::setfill(' ') << std::endl;
}

/// rows that stress the bf16 -> bfp16 conversion; row type = row % 8
static void fill_probe_rows(buffer<bf16>& x, uint32_t rows, uint32_t Din, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::uniform_int_distribution<int> coin(0, 1);
    uint16_t* bits = (uint16_t*)x.data();
    auto put = [&](size_t i, float v) { x[i] = bf16(v); };
    for (uint32_t l = 0; l < rows; l++) {
        for (uint32_t k = 0; k < Din; k++) {
            const size_t i = (size_t)l * Din + k;
            const float sign = coin(rng) ? -1.0f : 1.0f;
            switch (l % 8) {
            case 0: put(i, nd(rng)); break;
            case 1: put(i, sign * std::ldexp(1.0f + u01(rng), (int)(u01(rng) * 40) - 30)); break;
            case 2: put(i, k % 3 == 0 ? sign * std::ldexp(2.0f - 1.0f / 128, (int)(u01(rng) * 10) - 5) : nd(rng)); break;
            case 3: if (coin(rng)) bits[i] = coin(rng) ? 0x8000 : 0x0000; else put(i, nd(rng)); break;
            case 4:
                if (coin(rng)) bits[i] = (uint16_t)((coin(rng) ? 0x8000 : 0) | (1 + (int)(u01(rng) * 126)));
                else put(i, sign * std::ldexp(1.0f + u01(rng), -126 + (int)(u01(rng) * 6)));
                break;
            case 5: put(i, k % 8 == 0 ? sign * 100.0f : sign * std::ldexp(1.0f + u01(rng), -(int)(u01(rng) * 30) - 1)); break;
            case 6: put(i, (k / 8) % 2 ? -1.5f : 0.75f * sign); break;
            default: put(i, 1000.0f * nd(rng)); break;
            }
        }
    }
}

/// value of element (row, k) of host-made bfp16 tiles [rows / 16][Din / 8][2][72 B]
static double bfp16_value(const u8* tiles, uint32_t Din, uint32_t row, uint32_t k) {
    const u8* block = tiles + (size_t)(row / 16) * Din * 18 + (size_t)(k / 8) * 144 + (row % 16 / 8) * 72 + (row % 8) * 9;
    return std::ldexp((double)(int8_t)block[1 + k % 8], (int)block[0] - 133);
}

static float silu(float v) { return v / (1.0f + std::exp(-v)); }

struct dq_case {
    std::string name;
    uint32_t Din, Dout;
    int act_left, act_right;
};

int main(int argc, char** argv) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("xclbin", arg_utils::po::value<std::string>()->default_value("build/xclbins/prefill_overlay.xclbin"), "overlay xclbin");
    desc.add_options()("L", arg_utils::po::value<int>()->default_value(512), "dq rows per run (multiple of 128, or 32 / 64); attention runs L rounded up to 128");
    desc.add_options()("iters,i", arg_utils::po::value<int>()->default_value(20), "timing iterations");
    desc.add_options()("gap-ms", arg_utils::po::value<int>()->default_value(-1), "idle-gap test only: wait this long between runs, then check");
    desc.add_options()("gap-repeat", arg_utils::po::value<int>()->default_value(1), "with --gap-ms: this many gapped dq runs, output cleared before each, failures counted");
    desc.add_options()("streak", arg_utils::po::value<int>()->default_value(0), "streak test only: this many rounds of all dq shapes back to back, no switch, then check");
    desc.add_options()("identity", "bf16 -> bfp16 probe only: y = x W^T with W = I, compared bit for bit with the host-made bfp16 blocks");
    desc.add_options()("identity-dump", arg_utils::po::value<std::string>()->default_value(""), "with --identity: write x and y (raw bf16) to <prefix>_x.bin, <prefix>_y<e>.bin");
    desc.add_options()("help,h", "Show this help");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    const uint32_t L = vm["L"].as<int>();
    const uint32_t La = (L + 127) / 128 * 128;   // attention rows; also the rows every buffer has
    const int iters = vm["iters"].as<int>();
    const int D = 2048, DH = 128, Heads = 16, KV_Heads = 4, DQ = Heads * DH, DK = KV_Heads * DH, I = 6144;

    flm_rt::device device(0);
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);
    npu_app_manager* mgr = npu.register_xclbin(vm["xclbin"].as<std::string>());
    npu_app dq_app = mgr->create_app();
    npu_app attn_app = mgr->create_app();
    int configured = -1;
    int bd_owner = -1;
    auto ran = [&](int op) {
        bd_owner = hunyuan_overlay::bd_owner_after(op, configured, bd_owner);
        configured = op;
    };
    bool all_ok = true;
    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    hunyuan_desc hdesc;

    // ---------------- identity probe: what does the GEMM make of x? ----------------
    // With W = s I (q4_k S = 2^e, group scales 1, mins 0, nibble 1 on the diagonal) every output is one
    // product: y[l][n] = (x as bfp16)[l][n] * s, exact in fp32 and in bf16 (<= 8 significant
    // bits). The kernel reads x as the host's bfp16 tiles (hunyuan_bfp16.hpp), so this checks
    // the x path end to end (DMA, tile order, the kernel's A side). With the bf16-x kernel
    // (before WP2) the same y showed the blocks the emulated mmul made itself. Values below the
    // fp32 normal range (the array flushes them) are not observable and only counted.
    if (vm.count("identity")) {
        const uint32_t N = 2048;
        buffer<bf16> xi = npu.create_bo_buffer<bf16>((size_t)La * N);
        buffer<bf16> yi = npu.create_bo_buffer<bf16>((size_t)La * N);
        buffer<u8> wi = npu.create_bo_buffer<u8>((size_t)(N / 32) * (N / 256) * Q4K_BYTES);
        buffer<u8> tiles = npu.create_bo_buffer<u8>(hunyuan_bfp16::x_tiles_bytes(La, N));
        xi.memset(bf16(0.0f));
        fill_probe_rows(xi, L, N, rng);
        xi.sync_to_device();
        for (uint32_t g = 0; g < La / 16; g++)
            hunyuan_bfp16::_bfp16_rne_tile_group(tiles.data() + (size_t)g * N * 18, xi.data() + (size_t)g * 16 * N, N);
        tiles.sync_to_device();
        for (int lg2s : {0, 100}) {
            buffer<u8> plain((size_t)(N / 32) * (N / 256) * Q4K_BYTES);
            memset(plain.data(), 0, plain.size());
            for (uint32_t b = 0; b < N / 32; b++) {
                for (uint32_t cb = 0; cb < N / 256; cb++) {
                    u8* chunk = plain.data() + ((size_t)b * (N / 256) + cb) * Q4K_BYTES;
                    for (int i = 0; i < 256; i++) chunk[i] = 1;
                    for (int r = 0; r < 32; r++) ((bf16*)(chunk + Q4K_S))[r] = bf16(std::ldexp(1.0f, lg2s));
                    if (cb != b / 8) continue;
                    for (int r = 0; r < 32; r++) {
                        const int c = (b % 8) * 32 + r;
                        chunk[512 + c * 16 + r / 2] |= (r % 2) ? 0x10 : 0x01;
                    }
                }
            }
            hdesc.reorder_cpy(wi.data(), plain, N, flm_q4k, 2);
            wi.sync_to_device();
            hunyuan_overlay::gen_dq_run(dq_app.seq(), configured, L, N, N, 0, 0, 0, bd_owner);
            ran(hunyuan_overlay::dq_op_for(L));
            dq_app(yi, tiles, wi);
            yi.sync_from_device();
            const std::string dump = vm["identity-dump"].as<std::string>();
            if (!dump.empty()) {
                FILE* f = fopen((dump + "_x.bin").c_str(), "wb");
                fwrite(xi.data(), sizeof(bf16), (size_t)L * N, f);
                fclose(f);
                f = fopen((dump + "_y" + std::to_string(lg2s) + ".bin").c_str(), "wb");
                fwrite(yi.data(), sizeof(bf16), (size_t)L * N, f);
                fclose(f);
            }
            size_t same[8] = {}, differ[8] = {}, unseen[8] = {};
            int shown = 0;
            for (uint32_t l = 0; l < L; l++) {
                for (uint32_t n = 0; n < N; n++) {
                    const double want = std::ldexp(bfp16_value(tiles.data(), N, l, n), lg2s);
                    if (want != 0.0 && (std::fabs(want) < 0x1p-126 || std::fabs(want) > 0x1p127)) { unseen[l % 8]++; continue; }
                    const uint16_t got_bits = *(const uint16_t*)&yi[(size_t)l * N + n];
                    const bf16 want_bf = bf16((float)want);
                    const uint16_t want_bits = *(const uint16_t*)&want_bf;
                    // +0 / -0 both mean zero here
                    if (got_bits == want_bits || ((got_bits & 0x7FFF) == 0 && want == 0.0)) { same[l % 8]++; continue; }
                    differ[l % 8]++;
                    if (shown++ < 10)
                        std::cout << "  differ row " << l << " col " << n << ": x " << (float)xi[(size_t)l * N + n]
                                  << " npu " << (float)yi[(size_t)l * N + n] << " host bfp16 " << want << std::endl;
                }
            }
            size_t total_differ = 0;
            std::cout << "[identity] s = 2^" << lg2s << ", L " << L << " (row type: same / differ / not observable)" << std::endl;
            for (int t = 0; t < 8; t++) {
                std::cout << "    type " << t << ": " << same[t] << " / " << differ[t] << " / " << unseen[t] << std::endl;
                total_differ += differ[t];
            }
            std::cout << "[identity] s = 2^" << lg2s << (total_differ == 0 ? ": host blocks == kernel blocks  OK" : ": DIFFER  FAIL") << std::endl;
            all_ok &= total_differ == 0;
        }
        std::cout << std::endl << (all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << std::endl;
        return all_ok ? 0 : 1;
    }

    // ---------------- dequant_mm cases (Hy-MT2 projection shapes) ----------------
    const std::vector<dq_case> cases = {
        {"qkv  2048 -> 3072", (uint32_t)D, (uint32_t)(DQ + 2 * DK), 0, 0},
        {"kv   2048 -> 1024", (uint32_t)D, (uint32_t)(2 * DK), 0, 0},
        {"o    2048 -> 2048", (uint32_t)DQ, (uint32_t)D, 0, 0},
        {"up|gate 2048 -> 12288, silu right", (uint32_t)D, (uint32_t)(2 * I), 0, 1},
        {"down 6144 -> 2048", (uint32_t)I, (uint32_t)D, 0, 0},
    };
    // every matrix sits at a nonzero offset in one shared buffer, as in the
    // engine's per-layer proj_weights
    // x: bf16 for the CPU reference, xb: the same rows as the bfp16 tiles the kernel reads
    struct dq_data { size_t w_off; buffer<bf16> x, y, y_ref, y128; buffer<u8> xb; };
    size_t w_total = Q4K_BYTES;
    for (const dq_case& c : cases) w_total += (size_t)(c.Dout / 32) * (c.Din / 256) * Q4K_BYTES;
    buffer<u8> w_all = npu.create_bo_buffer<u8>(w_total);
    size_t w_next = Q4K_BYTES;
    std::vector<dq_data> data;
    for (const dq_case& c : cases) {
        dq_data d;
        buffer<u8> plain = random_q4k(c.Dout, c.Din, rng);
        d.w_off = w_next;
        w_next += plain.size();
        hdesc.reorder_cpy(w_all.data() + d.w_off, plain, c.Din, flm_q4k, 2);
        d.x = npu.create_bo_buffer<bf16>((size_t)La * c.Din);
        for (size_t i = 0; i < d.x.size(); i++) d.x[i] = bf16(nd(rng));
        d.xb = npu.create_bo_buffer<u8>(hunyuan_bfp16::x_tiles_bytes(La, c.Din));
        for (uint32_t g = 0; g < La / 16; g++)
            hunyuan_bfp16::_bfp16_rne_tile_group(d.xb.data() + hunyuan_bfp16::x_tiles_bytes(16 * g, c.Din),
                                                 d.x.data() + (size_t)g * 16 * c.Din, c.Din);
        d.xb.sync_to_device();
        d.y = npu.create_bo_buffer<bf16>((size_t)La * c.Dout);
        d.y128 = npu.create_bo_buffer<bf16>((size_t)La * c.Dout);
        d.y_ref = buffer<bf16>((size_t)L * c.Dout);
        cpu_func::_dequant_mm(d.y_ref.data(), w_all.data() + d.w_off, d.x.data(), L, c.Din, c.Dout);
        // the right half's 512-wide tiles (the odd ones) carry the on-chip silu
        for (uint32_t l = 0; l < L; l++)
            for (uint32_t n = 0; n < c.Dout; n++) {
                const bool right = (n / 512) % 2 == 1;
                const int act = right ? c.act_right : c.act_left;
                if (act) d.y_ref[(size_t)l * c.Dout + n] = bf16(silu((float)d.y_ref[(size_t)l * c.Dout + n]));
            }
        data.push_back(std::move(d));
    }
    w_all.sync_to_device();
    auto run_dq_rows = [&](size_t i, uint32_t rows, buffer<bf16>& y) {
        const dq_case& c = cases[i];
        hunyuan_overlay::gen_dq_run(dq_app.seq(), configured, rows, c.Din, c.Dout, data[i].w_off, c.act_left, c.act_right, bd_owner);
        ran(hunyuan_overlay::dq_op_for(rows));
        dq_app(y, data[i].xb, w_all);
        y.sync_from_device();
    };
    auto run_dq = [&](size_t i) { run_dq_rows(i, L, data[i].y); };

    // ---------------- attention (one layer's worth) ----------------
    const uint32_t L_begin = 256;                          // second case: 256 cached tokens first
    const uint32_t kv_rows = ((L_begin + La + 127) / 128 + 2) * 128;
    buffer<bf16> q = npu.create_bo_buffer<bf16>((size_t)La * DQ);
    buffer<bf16> o = npu.create_bo_buffer<bf16>((size_t)La * DQ);
    buffer<bf16> cache = npu.create_bo_buffer<bf16>((size_t)kv_rows * (DK + DK));
    for (size_t i = 0; i < q.size(); i++) q[i] = bf16(nd(rng));
    cache.memset(bf16(0.0f));
    const size_t quadrant = cache.size() / 4;             // [K kv 0,1][K kv 2,3][V kv 0,1][V kv 2,3]
    for (uint32_t row = 0; row < L_begin + La; row++)
        for (int qd = 0; qd < 4; qd++)
            for (int e = 0; e < DK / 2; e++) cache[qd * quadrant + (size_t)row * (DK / 2) + e] = bf16(nd(rng));
    q.sync_to_device();
    cache.sync_to_device();
    auto attn_ref = [&](uint32_t begin) {
        buffer<bf16> out((size_t)La * DQ);
        for (int half = 0; half < 2; half++) {
            const size_t q_off = (size_t)half * (Heads / 2) * DH;
            cpu_func::_attention(out.data() + q_off, q.data() + q_off,
                                 cache.data() + half * quadrant, cache.data() + (2 + half) * quadrant,
                                 La, begin, Heads / 2, KV_Heads / 2, DH, DK / 2, DK / 2, 1.0f / std::sqrt((float)DH), DQ);
        }
        return out;
    };
    buffer<bf16> o_ref0 = attn_ref(0), o_ref1 = attn_ref(L_begin);
    auto run_attn = [&](uint32_t begin) {
        hunyuan_overlay::gen_attn_run(attn_app.seq(), configured, begin, La, kv_rows, DH, Heads, KV_Heads, bd_owner);
        ran(hunyuan_overlay::OP_ATTN);
        attn_app(o, q, cache);
        o.sync_from_device();
    };

    // ---------------- 1. correctness, every switch direction ----------------
    std::cout << "L = " << L << ", Peano/chess build as in the Makefile" << std::endl;
    for (size_t i = 0; i < cases.size(); i++) {
        run_dq(i);
        all_ok &= check("dq   " + cases[i].name + (i == 0 ? " (first, full switch)" : ""), data[i].y, data[i].y_ref, data[i].y_ref.size(), 0.999f);
    }
    run_attn(0);
    all_ok &= check("attn L_begin 0 (switch from dq)", o, o_ref0, o.size(), 0.995f);
    run_attn(L_begin);
    all_ok &= check("attn L_begin 256 (no switch)", o, o_ref1, o.size(), 0.995f);
    run_dq(3);
    all_ok &= check("dq   " + cases[3].name + " (switch from attn)", data[3].y, data[3].y_ref, data[3].y_ref.size(), 0.999f);
    run_attn(0);
    all_ok &= check("attn L_begin 0 (switch from dq again)", o, o_ref0, o.size(), 0.995f);

    // a small-row plan must give the 128-row plan's bytes in its L rows (same math per row)
    if (L < La) {
        for (size_t i = 0; i < cases.size(); i++) {
            run_dq_rows(i, La, data[i].y128);
            run_dq(i);
            const size_t n = (size_t)L * cases[i].Dout;
            const bool same = memcmp(data[i].y.data(), data[i].y128.data(), n * sizeof(bf16)) == 0;
            std::cout << "[bits]  dq   " << std::left << std::setw(36) << cases[i].name << std::right
                      << " L " << L << " rows vs the 128-row plan: " << (same ? "identical  OK" : "DIFFER  FAIL") << std::endl;
            all_ok &= same;
        }
    }

    // output hashes: same hash in two builds = bit-identical output (inputs are seeded)
    for (size_t i = 0; i < cases.size(); i++) {
        run_dq(i);
        print_hash("dq   " + cases[i].name + " L " + std::to_string(L), data[i].y.data(), (size_t)L * cases[i].Dout * sizeof(bf16));
    }
    run_attn(0);
    print_hash("attn L_begin 0, rows " + std::to_string(La), o.data(), o.size() * sizeof(bf16));
    run_attn(L_begin);
    print_hash("attn L_begin 256, rows " + std::to_string(La), o.data(), o.size() * sizeof(bf16));

    // ---------------- idle gap: is the host-written overlay state kept while the NPU idles? ----------------
    const int gap_ms = vm["gap-ms"].as<int>();
    if (gap_ms >= 0) {
        const int repeat = std::max(1, vm["gap-repeat"].as<int>());
        std::cout << std::endl << "idle gap " << gap_ms << " ms before each of " << repeat
                  << " dq runs, output cleared first" << std::endl;
        int fails = 0;
        double slowest_ms = 0.0;
        for (int r = 0; r < repeat; r++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
            const size_t i = (size_t)r % cases.size();
            data[i].y.memset(bf16(0.0f));
            data[i].y.sync_to_device();
            const auto t0 = std::chrono::steady_clock::now();
            run_dq(i);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            slowest_ms = std::max(slowest_ms, ms);
            const bool ok = check("dq   " + cases[i].name + " after gap", data[i].y, data[i].y_ref, data[i].y_ref.size(), 0.999f);
            std::cout << "        run " << r << ": " << std::setprecision(1) << ms << " ms" << std::endl;
            if (!ok) {
                fails++;
                configured = -1;   // as the engine does after a failed run: full switch next
                bd_owner = -1;
            }
        }
        std::cout << "gap summary: " << fails << " of " << repeat << " runs failed, slowest "
                  << std::setprecision(1) << slowest_ms << " ms" << std::endl;
        all_ok &= fails == 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
        run_attn(0);
        all_ok &= check("attn after gap (switch from dq)", o, o_ref0, o.size(), 0.995f);
        std::cout << std::endl << (all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << std::endl;
        return all_ok ? 0 : 1;
    }

    // ---------------- streak: many dq runs without a switch (as with attention on the host) ----------------
    const int streak = vm["streak"].as<int>();
    if (streak > 0) {
        std::cout << std::endl << "streak of " << streak << " x " << cases.size() << " dq runs, no switch" << std::endl;
        for (int r = 0; r < streak; r++) {
            for (size_t i = 0; i < cases.size(); i++) run_dq(i);
            if ((r + 1) % 50 == 0) std::cout << "  " << (r + 1) * cases.size() << " runs done" << std::endl;
        }
        for (size_t i = 0; i < cases.size(); i++)
            all_ok &= check("dq   " + cases[i].name + " after streak", data[i].y, data[i].y_ref, data[i].y_ref.size(), 0.999f);
        std::cout << std::endl << (all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << std::endl;
        return all_ok ? 0 : 1;
    }

    // ---------------- 2. speed ----------------
    std::cout << std::endl;
    for (size_t i = 0; i < cases.size(); i++) {
        run_dq(i);
        hunyuan_overlay::gen_dq_run(dq_app.seq(), configured, L, cases[i].Din, cases[i].Dout, data[i].w_off, cases[i].act_left, cases[i].act_right, bd_owner);
        const std::string times = time_runs(iters, [&] { dq_app(data[i].y, data[i].xb, w_all); });
        std::cout << "[time] dq   " << std::left << std::setw(36) << cases[i].name << std::right << times << std::endl;
    }
    run_attn(0);
    hunyuan_overlay::gen_attn_run(attn_app.seq(), configured, 0, La, kv_rows, DH, Heads, KV_Heads, bd_owner);
    {
        const std::string times = time_runs(iters, [&] { attn_app(o, q, cache); });
        std::cout << "[time] attn L_begin 0                            " << times << std::endl;
    }
    // a switch = the same run with the switch in front: alternate the two ops
    {
        using namespace hunyuan_overlay;
        const int dq_op = dq_op_for(L);
        gen_dq_run(dq_app.seq(), OP_ATTN, L, cases[2].Din, cases[2].Dout, data[2].w_off, 0, 0, dq_op);    // attn -> dq (o)
        const size_t to_dq = last_switch_writes;
        gen_attn_run(attn_app.seq(), dq_op, 0, La, kv_rows, DH, Heads, KV_Heads, dq_op);                   // dq -> attn
        const size_t to_attn = last_switch_writes;
        const std::string times = time_runs(iters, [&] {
            dq_app(data[2].y, data[2].xb, w_all);
            attn_app(o, q, cache);
        });
        std::cout << "[time] o-proj + attn, switching every run        " << times
                  << "  (switch writes: to dq " << to_dq << ", to attn " << to_attn << ")" << std::endl;
    }
    std::cout << std::endl << (all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << std::endl;
    return all_ok ? 0 : 1;
}
