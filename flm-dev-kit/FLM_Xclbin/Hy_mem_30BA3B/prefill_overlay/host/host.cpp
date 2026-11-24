// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Test of the prefill overlay: dequant_mm and attention in ONE xclbin.
//
//   1. correctness, through switch / no-switch runs in both directions:
//      dequant_mm  bit for bit against the standalone dequant_mm_512x512x512
//                  (it adds over K in the same order, so the output must be
//                  identical), and against the CPU reference
//      attention   against the CPU reference (attn_ref.hpp)
//   2. Hy-Memory's projection shapes that only the overlay's dequant_mm takes
//      (Din % 256, L % 256): against the CPU reference
//   3. speed: dequant_mm overlay vs standalone, attention, and one overlay switch

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "metrices.hpp"
#include "cpu_q4_k.hpp"          // dequant_mm_512x512x512/host
#include "util.hpp"              // reorder_cpy
#include "seq_gen.hpp"           // gen_dequant_mm_512 (standalone)
#include "attn_ref.hpp"          // attn_cpu_ref
#include "overlay_host.hpp"

namespace po = boost::program_options;
using clk = std::chrono::steady_clock;

template <typename F>
static double time_us(F&& f, int iters) {
    for (int i = 0; i < 2; i++) f();
    clk::time_point t0 = clk::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count() / iters;
}

/// @brief cosine against the CPU reference, and (if given) how far another NPU run
///        disagrees bit for bit
/// @note The standalone dequant_mm_512x512x512 keeps the hardware default rounding
///       (floor) while this overlay's kernel sets conv_even, so the two no longer
///       agree bit for bit and `exact` is reported, not required: floor biases every
///       dequantized weight about half an ulp low, which is invisible in one matmul
///       but compounds over the prefill's 48 layers (see kernel/dq/dequant_mm_e.cc).
///       The layout of the relaid reader was proven against the standalone while both
///       still rounded the same way.
static bool compare(const char* label, buffer<bf16>& got, buffer<bf16>* exact, buffer<bf16>& cpu, size_t n,
                    float min_cos) {
    size_t diff = 0;
    if (exact) {
        for (size_t i = 0; i < n; i++) diff += (std::bit_cast<uint16_t>(got[i]) != std::bit_cast<uint16_t>((*exact)[i]));
    }
    buffer<bf16> g(n), c(n);
    g.copy_from(got.data(), n);
    c.copy_from(cpu.data(), n);
    error_metrics m = get_error_metrics(g, c);
    const bool ok = m.CosineSimilarity > min_cos;
    std::cout << "[check] " << std::left << std::setw(38) << label;
    if (exact) std::cout << " differs from standalone at " << diff << " of " << n << ",";
    std::cout << " cos vs CPU " << std::setprecision(6) << m.CosineSimilarity << (ok ? "  OK" : "  FAIL") << std::endl;
    return ok;
}

/// @brief w on the grid the mmul really multiplies on: aie::mmul<8,8,8,bf16> is
///        emulated with bfp16 (shared exponent per 8 k, 8-bit mantissa, floor),
///        so this reference leaves only the kernel's own arithmetic as error
///        (from dequant_mm_512x512x512/host/host.cpp)
static void bfp16_floor_k(bf16* v, size_t rows, size_t k) {
    for (size_t r = 0; r < rows; r++) {
        bf16* row = v + r * k;
        for (size_t g = 0; g < k; g += 8) {
            float mx = 0.0f;
            for (int e = 0; e < 8; e++) mx = std::max(mx, std::fabs((float)row[g + e]));
            if (mx == 0.0f) continue;
            const float st = std::exp2(std::floor(std::log2(mx)) + 1 - 7);
            for (int e = 0; e < 8; e++) row[g + e] = bf16(std::floor((float)row[g + e] / st) * st);
        }
    }
}

/// @brief every 32 x 256 block into the int8-mmul kernel's order, in place
/// @note Mirrors q4k_relayout_for_mmul in
///       FLM_DLL/detail/hy_mem_npu/hy_mem_model_def.cpp, which is what really
///       lays out Hy-Memory's MoE weights. The overlay's dequant_mm reads this
///       order (kernel/dq/q4_core.h), the standalone dequant_mm_512x512x512
///       still reads the plain one -- so feeding the same weights to both, in
///       their own orders, must give the same y bit for bit.
static void relayout_for_mmul(u8* data, size_t bytes) {
    constexpr size_t BLOCK = 4736, SCALES = 0, QS = 512;
    assert(bytes % BLOCK == 0);
    for (size_t b = 0; b < bytes / BLOCK; b++) {
        u8* blk = data + b * BLOCK;
        std::array<u8, BLOCK> old;
        memcpy(old.data(), blk, BLOCK);
        for (int k = 0; k < 16; k++)                 // 16 columns per k step
            for (int t = 0; t < 2; t++)              // rows 16t .. 16t+15
                for (int c = 0; c < 16; c++)
                    memcpy(blk + QS + (k * 2 + t) * 128 + c * 8,
                           old.data() + QS + (k * 16 + c) * 16 + t * 8, 8);
        for (int h = 0; h < 2; h++)
            for (int t = 0; t < 2; t++)
                for (int a = 0; a < 4; a++)
                    memcpy(blk + SCALES + ((h * 2 + t) * 4 + a) * 16,
                           old.data() + SCALES + (4 * h + a) * 32 + 16 * t, 16);
    }
}

/// @brief the overlay's weight copy: decode order (pairs), then relaid
static void reorder_cpy_relaid(buffer<u8>& dst, buffer<u8>& src, uint32_t Din) {
    reorder_cpy(dst.data(), src, Din, 2);
    relayout_for_mmul(dst.data(), src.size());
}

/// @brief random q4_k weights (pairs layout, as the decode buffers keep them),
///        random x, and the CPU reference y = x W^T on the bfp16 grid
struct dq_case {
    uint32_t L, Din, Dout;
    buffer<u8> w_packed;
    buffer<bf16> x, y_cpu;

    dq_case(uint32_t L, uint32_t Din, uint32_t Dout, uint32_t seed) : L(L), Din(Din), Dout(Dout) {
        buffer<bf16> w((size_t)Dout * Din), w_deq((size_t)Dout * Din);
        w_packed = buffer<u8>(q4_k_size((size_t)Din * Dout));
        std::mt19937 rng(seed);
        std::normal_distribution<float> wd(0.0f, 0.02f), xd(0.0f, 1.0f);
        for (size_t i = 0; i < w.size(); i++) w[i] = bf16(wd(rng));
        q4_k_quantize_pack(w_packed.data(), w_deq.data(), w.data(), Dout, Din);
        bfp16_floor_k(w_deq.data(), Dout, Din);
        x = buffer<bf16>((size_t)L * Din);
        for (size_t i = 0; i < x.size(); i++) x[i] = bf16(xd(rng));
        y_cpu = buffer<bf16>((size_t)L * Dout);
        cpu_matmul_nt(y_cpu.data(), x.data(), w_deq.data(), L, Dout, Din);
    }
};

int main(int argc, char** argv) {
    po::options_description desc("Allowed options");
    desc.add_options()("iters,i", po::value<int>()->default_value(20), "timing iterations")
        ("shape", po::value<std::string>()->default_value(""), "only run one dequant_mm shape \"L,Din,Dout\" from a fresh xclbin");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    const int ITERS = vm["iters"].as<int>();

    xrt::device device(0);
    npu_xclbin_manager mgr(device_npu2, &device, false);
    npu_app_manager* dq_mgr = mgr.register_xclbin("../../dequant_mm_512x512x512/build/xclbins/dequant_mm.xclbin");
    npu_app_manager* ovl_mgr = mgr.register_xclbin("build/xclbins/overlay.xclbin");

    // --shape L,Din,Dout: one dequant_mm run on the fresh overlay, against the CPU
    if (!vm["shape"].as<std::string>().empty()) {
        uint32_t sL, sDin, sDout;
        if (sscanf(vm["shape"].as<std::string>().c_str(), "%u,%u,%u", &sL, &sDin, &sDout) != 3) return 2;
        dq_case c(sL, sDin, sDout, 7);
        int fresh = -1;
        npu_app app = ovl_mgr->create_app();
        overlay::gen_dq_run(app.seq(), fresh, sL, sDin, sDout, 0, 0, 0);
        buffer<bf16> y = app.create_bo_buffer<bf16>((size_t)sL * sDout);
        buffer<bf16> xb = app.create_bo_buffer<bf16>((size_t)sL * sDin);
        buffer<u8> wb = app.create_bo_buffer<u8>(c.w_packed.size());
        reorder_cpy_relaid(wb, c.w_packed, sDin);
        xb.copy_from(c.x.data(), c.x.size());
        y.memset((bf16)-1.0f);
        wb.sync_to_device();
        xb.sync_to_device();
        y.sync_to_device();
        ert_cmd_state st = app(y, xb, wb);
        y.sync_from_device();
        std::cout << "[shape] L " << sL << ", " << sDin << " -> " << sDout << ": state " << app.cmd_state_map[st] << std::endl;
        const bool ok = st == ERT_CMD_STATE_COMPLETED && compare("shape", y, nullptr, c.y_cpu, (size_t)sL * sDout, 0.999f);
        // speed of this shape, NPU only (the array stays configured for dq): best of 10
        double best_us = 1e30;
        for (int i = 0; i < 10; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            app(y, xb, wb);
            best_us = std::min(best_us, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        std::cout << "[shape] best of 10: " << best_us << " us" << std::endl;
        return ok ? 0 : 1;
    }

    // ---- dequant_mm: a shape the standalone 512 x 512 x 512 one also takes -------
    dq_case dq(512, 2048, 1024, 0);
    const uint32_t L = dq.L, Din = dq.Din, Dout = dq.Dout;

    // ---- attention data (Hy: 32 q / 4 kv heads, dh 128, decode KV layout) --------
    const int AL = 512, MAX_L = 1024, HEADS = 32, KV_HEADS = 4, DH = 128, PER_COL = 2;
    const int DQ = HEADS * DH, DK = KV_HEADS * DH;
    const int KV_CACHE_SIZE = MAX_L * 2 * DK;
    buffer<bf16> q_host((size_t)AL * DQ), k_host((size_t)AL * DK), v_host((size_t)AL * DK);
    {
        std::mt19937 rng(32);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < q_host.size(); i++) q_host[i] = (bf16)dist(rng);
        for (size_t i = 0; i < k_host.size(); i++) k_host[i] = (bf16)dist(rng);
        for (size_t i = 0; i < v_host.size(); i++) v_host[i] = (bf16)dist(rng);
    }
    buffer<bf16> kv_host(KV_CACHE_SIZE);
    for (int i = 0; i < KV_CACHE_SIZE; i++) kv_host[i] = (bf16)((i & 1) ? 64.0f : -64.0f);   // finite junk past L
    for (int j = 0; j < AL; j++) {
        for (int h = 0; h < KV_HEADS; h++) {
            const size_t row = (size_t)j * PER_COL * DH + (size_t)(h % PER_COL) * DH;
            const size_t kb = (size_t)(h / PER_COL) * MAX_L * PER_COL * DH;           // [K col0][K col1]
            const size_t vb = (size_t)(2 + h / PER_COL) * MAX_L * PER_COL * DH;       // [V col0][V col1]
            memcpy(kv_host.data() + kb + row, k_host.data() + (size_t)j * DK + h * DH, DH * sizeof(bf16));
            memcpy(kv_host.data() + vb + row, v_host.data() + (size_t)j * DK + h * DH, DH * sizeof(bf16));
        }
    }
    buffer<bf16> o_cpu((size_t)AL * DQ);
    {
        attn_ref_config cfg;
        cfg.dh = DH;
        cfg.n_q_heads = HEADS;
        cfg.n_kv_heads = KV_HEADS;
        cfg.window_size = MAX_L;
        cfg.scale = 1.0f / std::sqrt((float)DH);
        attn_cpu_ref(o_cpu, q_host, k_host, v_host, AL, 0, cfg);
    }

    // ---- standalone dequant_mm: the exact reference --------------------------------
    npu_app dq_std = dq_mgr->create_app();
    gen_dequant_mm_512(dq_std.seq(), L, Din, Dout, 0, 0, 0);
    buffer<bf16> s_y = dq_std.create_bo_buffer<bf16>((size_t)L * Dout);
    buffer<bf16> s_x = dq_std.create_bo_buffer<bf16>((size_t)L * Din);
    buffer<u8> s_w = dq_std.create_bo_buffer<u8>(dq.w_packed.size());
    reorder_cpy(s_w.data(), dq.w_packed, Din, 2);
    s_x.copy_from(dq.x.data(), dq.x.size());
    s_x.sync_to_device();
    s_w.sync_to_device();
    dq_std(s_y, s_x, s_w);
    s_y.sync_from_device();

    // ---- overlay apps: (switch +) op. `configured` follows the run order below ----
    int cfg = -1;
    npu_app dq_full = ovl_mgr->create_app();
    overlay::gen_dq_run(dq_full.seq(), cfg, L, Din, Dout, 0, 0, 0);      // full switch to dq (fresh xclbin)
    npu_app dq_ns = ovl_mgr->create_app();
    overlay::gen_dq_run(dq_ns.seq(), cfg, L, Din, Dout, 0, 0, 0);        // dq again
    npu_app at_sw = ovl_mgr->create_app();
    overlay::gen_attn_run(at_sw.seq(), cfg, 0, AL, MAX_L, 512, 16, DH, HEADS, KV_HEADS, PER_COL, KV_CACHE_SIZE);
    npu_app at_ns = ovl_mgr->create_app();
    overlay::gen_attn_run(at_ns.seq(), cfg, 0, AL, MAX_L, 512, 16, DH, HEADS, KV_HEADS, PER_COL, KV_CACHE_SIZE);
    npu_app dq_sw = ovl_mgr->create_app();
    overlay::gen_dq_run(dq_sw.seq(), cfg, L, Din, Dout, 0, 0, 0);        // attn -> dq

    buffer<bf16> o_y = dq_full.create_bo_buffer<bf16>((size_t)L * Dout);
    buffer<bf16> o_x = dq_full.create_bo_buffer<bf16>((size_t)L * Din);
    buffer<u8> o_w = dq_full.create_bo_buffer<u8>(dq.w_packed.size());
    reorder_cpy_relaid(o_w, dq.w_packed, Din);
    o_x.copy_from(dq.x.data(), dq.x.size());
    o_x.sync_to_device();
    o_w.sync_to_device();
    buffer<bf16> o_o = at_sw.create_bo_buffer<bf16>((size_t)AL * DQ);
    buffer<bf16> o_q = at_sw.create_bo_buffer<bf16>((size_t)AL * DQ);
    buffer<bf16> o_kv = at_sw.create_bo_buffer<bf16>(KV_CACHE_SIZE);
    o_q.copy_from(q_host.data(), q_host.size());
    o_kv.copy_from(kv_host.data(), kv_host.size());
    o_q.sync_to_device();
    o_kv.sync_to_device();

    std::cout << "[info] register writes per switch: full (fresh) to dq " << overlay::switch_writes(overlay::OP_DQ, -1)
              << ", attn -> dq " << overlay::switch_writes(overlay::OP_DQ, overlay::OP_ATTN)
              << ", dq -> attn " << overlay::switch_writes(overlay::OP_ATTN, overlay::OP_DQ) << std::endl;

    auto run_dq = [&](npu_app& app, const char* label) {
        o_y.memset((bf16)-1.0f);
        o_y.sync_to_device();
        ert_cmd_state st = app(o_y, o_x, o_w);
        o_y.sync_from_device();
        if (st != ERT_CMD_STATE_COMPLETED) std::cout << "[check] " << label << ": state " << app.cmd_state_map[st] << std::endl;
        return st == ERT_CMD_STATE_COMPLETED && compare(label, o_y, &s_y, dq.y_cpu, (size_t)L * Dout, 0.999f);
    };
    auto run_at = [&](npu_app& app, const char* label) {
        o_o.memset((bf16)-1.0f);
        o_o.sync_to_device();
        ert_cmd_state st = app(o_o, o_q, o_kv);
        o_o.sync_from_device();
        if (st != ERT_CMD_STATE_COMPLETED) std::cout << "[check] " << label << ": state " << app.cmd_state_map[st] << std::endl;
        return st == ERT_CMD_STATE_COMPLETED && compare(label, o_o, nullptr, o_cpu, (size_t)AL * DQ, 0.998f);
    };

    // ---- 1. correctness through the switches ---------------------------------------
    bool pass = true;
    pass &= run_dq(dq_full, "dq   (full switch, fresh)");
    pass &= run_dq(dq_ns, "dq   (no switch)");
    pass &= run_at(at_sw, "attn (switch from dq)");
    pass &= run_at(at_ns, "attn (no switch)");
    pass &= run_dq(dq_sw, "dq   (switch from attn)");
    pass &= run_dq(dq_ns, "dq   (no switch)");
    pass &= run_at(at_sw, "attn (switch from dq)");
    pass &= run_dq(dq_full, "dq   (full switch from attn)");
    pass &= run_at(at_sw, "attn (switch from dq)");
    // The array is set up for attention now. `cfg` followed the order the apps
    // above were BUILT in (it ended on dq_sw), not the order they ran in, so put
    // it right before building more: a dq run built without its switch hangs.
    cfg = overlay::OP_ATTN;

    // ---- 2. Hy-Memory shapes only the overlay takes (Din % 256, L % 256) -----------
    struct hy_shape { const char* label; uint32_t L, Din, Dout; };
    const hy_shape shapes[] = {
        {"dq   Hy expert down 768 -> 2048", 256, 768, 2048},
        {"dq   Hy dense down 6912 -> 2048", 256, 6912, 2048},
        {"dq   Hy expert up|gate 2048 -> 1536", 256, 2048, 1536},
        {"dq   Hy k|v|q 2048 -> 5120", 768, 2048, 5120},
    };
    for (const hy_shape& s : shapes) {
        dq_case c(s.L, s.Din, s.Dout, 7);
        npu_app app = ovl_mgr->create_app();
        overlay::gen_dq_run(app.seq(), cfg, s.L, s.Din, s.Dout, 0, 0, 0);
        buffer<bf16> y = app.create_bo_buffer<bf16>((size_t)s.L * s.Dout);
        buffer<bf16> xb = app.create_bo_buffer<bf16>((size_t)s.L * s.Din);
        buffer<u8> wb = app.create_bo_buffer<u8>(c.w_packed.size());
        reorder_cpy_relaid(wb, c.w_packed, s.Din);
        xb.copy_from(c.x.data(), c.x.size());
        y.memset((bf16)-1.0f);
        wb.sync_to_device();
        xb.sync_to_device();
        y.sync_to_device();
        ert_cmd_state st = app(y, xb, wb);
        y.sync_from_device();
        if (st != ERT_CMD_STATE_COMPLETED) std::cout << "[check] " << s.label << ": state " << app.cmd_state_map[st] << std::endl;
        pass &= st == ERT_CMD_STATE_COMPLETED && compare(s.label, y, nullptr, c.y_cpu, (size_t)s.L * s.Dout, 0.999f);
    }
    // configured: dq
    std::cout << "[check] " << (pass ? "PASSED" : "FAILED") << std::endl;
    if (!pass) return 1;

    // ---- 3. speed -----------------------------------------------------------------
    std::cout << std::fixed << std::setprecision(1);
    const double t_dq_std = time_us([&] { dq_std(s_y, s_x, s_w); }, ITERS);
    // The standalone xclbin just ran: the NPU reloads the overlay on its next
    // run and everything the host wrote into it (routes, BDs, locks) is gone.
    // So the first overlay run after another xclbin must be a full switch.
    dq_full(o_y, o_x, o_w);                                     // configured: dq
    const double t_dq_ns = time_us([&] { dq_ns(o_y, o_x, o_w); }, ITERS);
    at_sw(o_o, o_q, o_kv);                                      // configured: attn
    const double t_at_ns = time_us([&] { at_ns(o_o, o_q, o_kv); }, ITERS);
    const double t_ovl_pair = time_us([&] { dq_sw(o_y, o_x, o_w); at_sw(o_o, o_q, o_kv); }, ITERS);

    std::cout << "[speed] dq   L=" << L << " " << Din << "->" << Dout << ": standalone 512x512x512 " << t_dq_std
              << " us, overlay (M 256, K 256) " << t_dq_ns << " us" << std::endl;
    std::cout << "[speed] attn L=" << AL << ": overlay " << t_at_ns << " us" << std::endl;
    std::cout << "[speed] dq + attn alternating: overlay " << t_ovl_pair << " us => one overlay switch ~ "
              << (t_ovl_pair - t_dq_ns - t_at_ns) / 2 << " us" << std::endl;
    return 0;
}
