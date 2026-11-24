// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdfloat>
#include "utils/debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "lm_config.hpp"
#include <math.h>
#include "metrices.hpp"
#include "seq_gen.hpp"
#include "seq_gen_old.hpp"
#include "tensor_2d.hpp"
#include "cpu_q4_k.hpp"
#include "util.hpp"
#include "weights_desc.hpp"

namespace po = boost::program_options;

/// \brief Put a weight matrix on the grid aie::mmul actually multiplies on.
///
/// mm_aie's `aie::mmul<8,8,8,bf16,bf16,accauto>` lowers to the bfp16 path on
/// npu2: every block of 8 consecutive k at a fixed n is given a shared exponent
/// and an 8-bit mantissa, truncated toward -infinity.  Multiplying by this
/// instead of by w_deq removes the kernel's largest remaining error term, so the
/// two references below bracket it.
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

struct mm_shape {
    std::string name;
    int din;
    int dout;
    int act;
    int calls_per_prefill;
};

// Every dequant_mm call of one MiniCPM-OCR prefill (18 linear + 6 attention layers), as the engine makes them
// (minicpm_ocr_prefill.hpp). act 1 = SiLU fused on the array (the linear layers' z / gate projection).
static const std::vector<mm_shape> minicpm_ocr_shapes = {
    {"linear qkv",       1024, 6144, 0, 18},
    {"linear z (silu)",  1024, 2048, 1, 18},
    {"o (linear+attn)",  2048, 1024, 0, 24},
    {"attn q / g",       1024, 2048, 0, 12},
    {"attn k / v",       1024,  512, 0, 12},
    {"mlp up|gate",      1024, 7168, 0, 24},
    {"mlp down",         3584, 1024, 0, 24},
};

static uint64_t fnv1a(const void* data, size_t bytes) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// Rows the CPU checks: both ends of every 512-row round boundary region plus random rows.
static std::vector<int> pick_check_rows(int L, int count) {
    std::vector<int> rows;
    for (int r : {0, 1, 127, 128, 511, 512, L - 1}) {
        if (r < L) rows.push_back(r);
    }
    std::mt19937 rng(2);
    std::uniform_int_distribution<int> dist(0, L - 1);
    while ((int)rows.size() < count) rows.push_back(dist(rng));
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

// x (rows x Din, bf16) -> bfp16 tiles [16-row group][Din / 8][2 row blocks][72 B]; a tile is 8 rows x 8 k, each row
// one block: shared exponent byte (largest exponent), then 8 two's complement mantissas, truncated (mlir-aie
// programming_examples/ml/block_datatypes/helper.h floatToBfp16, the AIE2P conversion).
static void x_to_bfp16_tiles(uint8_t* out, const bf16* x, int rows, int Din) {
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < rows / 16; g++) {
        for (int kb = 0; kb < Din / 8; kb++) {
            for (int z = 0; z < 2; z++) {
                uint8_t* tile = out + (((size_t)g * (Din / 8) + kb) * 2 + z) * 72;
                for (int r = 0; r < 8; r++) {
                    const bf16* v = x + (size_t)(g * 16 + z * 8 + r) * Din + kb * 8;
                    uint32_t bits[8];
                    uint32_t max_exp = 0;
                    for (int c = 0; c < 8; c++) {
                        float f = (float)v[c];
                        memcpy(&bits[c], &f, 4);
                        max_exp = std::max(max_exp, (bits[c] >> 23) & 0xFF);
                    }
                    uint8_t* block = tile + r * 9;
                    block[0] = (uint8_t)max_exp;
                    for (int c = 0; c < 8; c++) {
                        const uint32_t sign = bits[c] & 0x80000000u;
                        const uint32_t exp = (bits[c] >> 23) & 0xFF;
                        uint32_t mantissa = bits[c] & 0x007FFFFFu;
                        if (exp) mantissa |= 0x00800000u;
                        mantissa = sign ? ~mantissa + 1 : mantissa;
                        uint8_t m8 = (uint8_t)(mantissa >> 17);
                        if (max_exp - exp >= 32) {
                            m8 = sign ? 0xFF : 0x00;
                        } else {
                            m8 = (uint8_t)((int8_t)m8 >> (max_exp - exp));
                        }
                        block[1 + c] = m8;
                    }
                }
            }
        }
    }
}

template <typename Call>
static double time_calls(Call call, int iters, const char* label) {
    call();
    double total_us = 0.0;
    std::ostringstream call_times;
    for (int i = 0; i < iters; i++) {
        time_utils::time_point start = time_utils::now();
        call();
        time_utils::time_point stop = time_utils::now();
        double us = time_utils::duration_us(start, stop).first;
        total_us += us;
        call_times << " " << (int)us;
    }
    header_print("shape", label << " call times (us):" << call_times.str());
    return total_us / iters;
}

// bfp16 tiles [16-row group][Din / 8][2 row blocks][72 B] -> floats: a value is its mantissa * 2^(shared exp - 133)
static void bfp16_tiles_to_float(float* out, const uint8_t* tiles, int rows, int Din) {
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < rows / 16; g++) {
        for (int kb = 0; kb < Din / 8; kb++) {
            for (int z = 0; z < 2; z++) {
                const uint8_t* tile = tiles + (((size_t)g * (Din / 8) + kb) * 2 + z) * 72;
                for (int r = 0; r < 8; r++) {
                    const uint8_t* block = tile + r * 9;
                    const float scale = std::ldexp(1.0f, (int)block[0] - 133);
                    for (int c = 0; c < 8; c++) {
                        out[(size_t)(g * 16 + z * 8 + r) * Din + kb * 8 + c] = (float)(int8_t)block[1 + c] * scale;
                    }
                }
            }
        }
    }
}

// GLU slots (gen_dequant_mm_glu) -> the row tile layout, a pure byte move
static void glu_slots_to_tiles(uint8_t* tiles, const uint8_t* slots, int rows, int IS) {
    const int slots_per_rb = IS / glu_slot_features;
    #pragma omp parallel for schedule(static)
    for (int rb = 0; rb < rows / 128; rb++) {
        for (int fs = 0; fs < slots_per_rb; fs++) {
            const uint8_t* slot = slots + ((size_t)rb * slots_per_rb + fs) * glu_slot_bytes;
            for (int g = 0; g < 8; g++) {
                for (int kb = 0; kb < 4; kb++) {
                    uint8_t* dst = tiles + (((size_t)(rb * 8 + g) * (IS / 8) + fs * 4 + kb) * 2) * 72;
                    memcpy(dst, slot + (g * 4 + kb) * 144, 144);
                }
            }
        }
    }
}

// the engine's host GLU (cpu_func::_exp_glu) in scalar form: up / gate in 512-wide slices, fp32, round to nearest even
static void host_glu(bf16* hid, const bf16* ug, int rows, int IS) {
    #pragma omp parallel for schedule(static)
    for (int l = 0; l < rows; l++) {
        for (int f = 0; f < IS; f++) {
            const float up = (float)ug[(size_t)l * IS * 2 + (f / 512) * 1024 + f % 512];
            const float gate = (float)ug[(size_t)l * IS * 2 + (f / 512) * 1024 + 512 + f % 512];
            hid[(size_t)l * IS + f] = bf16(up * (gate / (1.0f + std::exp(-gate))));
        }
    }
}

struct diff_stats {
    double rel_l2;
    double cos;
    double max_abs;
};

static diff_stats compare_floats(const float* got, const float* want, size_t n) {
    double d2 = 0.0, w2 = 0.0, g2 = 0.0, gw = 0.0, mx = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double g = got[i], w = want[i];
        d2 += (g - w) * (g - w);
        w2 += w * w;
        g2 += g * g;
        gw += g * w;
        mx = std::max(mx, std::fabs(g - w));
    }
    return {std::sqrt(d2 / w2), gw / std::sqrt(g2 * w2), mx};
}

// Part B check at 1024 -> 2 x IS (up/gate) -> GLU -> IS -> 1024 (down):
//   A: up/gate run (act 0), host GLU, host bfp16 tiles (the engine's mm_bfp16 path)
//   B: up/gate run with the GLU epilogue (act 2) into slots
// hid of both as the down projection sees it (decoded tiles) vs the host GLU's bf16 hid, and the down projection
// reading B's slots vs reading the same bytes moved to row tiles (must be bit-identical).
static void run_glu(npu_app& app, int L, int iters) {
    const int D = 1024;
    const int IS = 3584;
    header_print("glu", "L " << L << ", D " << D << ", IS " << IS);

    buffer<u8> w_ug = app.create_bo_buffer<u8>(q4_k_size((size_t)D * IS * 2));
    buffer<u8> w_down = app.create_bo_buffer<u8>(q4_k_size((size_t)IS * D));
    {
        buffer<bf16> w((size_t)IS * 2 * D);
        buffer<bf16> w_deq((size_t)IS * 2 * D);
        buffer<u8> w_packed(q4_k_size((size_t)D * IS * 2));
        std::mt19937 rng(3);
        std::normal_distribution<float> dist(0.0f, 0.03f);
        for (size_t i = 0; i < w.size(); i++) w[i] = bf16(dist(rng));
        q4_k_quantize_pack(w_packed.data(), w_deq.data(), w.data(), IS * 2, D);
        reorder_cpy(w_ug.data(), w_packed, D, 2);
        w_ug.sync_to_device();
    }
    {
        buffer<bf16> w((size_t)D * IS);
        buffer<bf16> w_deq((size_t)D * IS);
        buffer<u8> w_packed(q4_k_size((size_t)IS * D));
        std::mt19937 rng(4);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (size_t i = 0; i < w.size(); i++) w[i] = bf16(dist(rng));
        q4_k_quantize_pack(w_packed.data(), w_deq.data(), w.data(), D, IS);
        reorder_cpy(w_down.data(), w_packed, IS, 2);
        w_down.sync_to_device();
    }
    buffer<bf16> x((size_t)L * D);
    {
        std::mt19937 rng(5);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < x.size(); i++) x[i] = bf16(dist(rng));
    }
    buffer<u8> xb = app.create_bo_buffer<u8>((size_t)L * D * 9 / 8);
    x_to_bfp16_tiles(xb.data(), x.data(), L, D);
    xb.sync_to_device();

    // A
    buffer<bf16> ug = app.create_bo_buffer<bf16>((size_t)L * IS * 2);
    gen_dequant_mm_512(app.seq(), L, D, IS * 2, 0, 0, 0);
    const double a_us = time_calls([&]() { app(ug, xb, w_ug); }, iters, "up/gate act 0");
    ug.sync_from_device();
    buffer<bf16> hid_a((size_t)L * IS);
    host_glu(hid_a.data(), ug.data(), L, IS);
    buffer<u8> hb_a = app.create_bo_buffer<u8>((size_t)L * IS * 9 / 8);
    x_to_bfp16_tiles(hb_a.data(), hid_a.data(), L, IS);
    hb_a.sync_to_device();

    // B
    buffer<u8> slots = app.create_bo_buffer<u8>(glu_slots_bytes(L, IS));
    memset(slots.data(), 0xA5, slots.size());
    slots.sync_to_device();
    gen_dequant_mm_glu(app.seq(), L, D, IS, 0, 0);
    const double b_us = time_calls([&]() { app(slots, xb, w_ug); }, iters, "up/gate act 2 (GLU)");
    slots.sync_from_device();
    buffer<u8> hb_b((size_t)L * IS * 9 / 8);
    glu_slots_to_tiles(hb_b.data(), slots.data(), L, IS);

    std::vector<float> f_hid((size_t)L * IS), f_a((size_t)L * IS), f_b((size_t)L * IS);
    for (size_t i = 0; i < f_hid.size(); i++) f_hid[i] = (float)hid_a[i];
    bfp16_tiles_to_float(f_a.data(), hb_a.data(), L, IS);
    bfp16_tiles_to_float(f_b.data(), hb_b.data(), L, IS);
    const diff_stats a_vs_hid = compare_floats(f_a.data(), f_hid.data(), f_hid.size());
    const diff_stats b_vs_hid = compare_floats(f_b.data(), f_hid.data(), f_hid.size());
    const diff_stats b_vs_a = compare_floats(f_b.data(), f_a.data(), f_hid.size());
    std::cout << std::scientific << std::setprecision(3)
              << "GLU L " << L << ": hid vs the host GLU (bf16): A tiles rel L2 " << a_vs_hid.rel_l2 << " (cos " << std::fixed << std::setprecision(7) << a_vs_hid.cos << ")"
              << std::scientific << std::setprecision(3) << ", B tiles rel L2 " << b_vs_hid.rel_l2 << " (cos " << std::fixed << std::setprecision(7) << b_vs_hid.cos
              << std::scientific << std::setprecision(3) << ", max abs " << b_vs_hid.max_abs << "); B vs A tiles rel L2 " << b_vs_a.rel_l2 << std::endl;
    std::cout << std::fixed << std::setprecision(0) << "GLU L " << L << ": up/gate call A " << a_us << " us, B " << b_us << " us" << std::endl;

    // down from B's slots vs from the same bytes as row tiles
    buffer<u8> hb_rows = app.create_bo_buffer<u8>((size_t)L * IS * 9 / 8);
    memcpy(hb_rows.data(), hb_b.data(), hb_b.size());
    hb_rows.sync_to_device();
    buffer<bf16> y_rows = app.create_bo_buffer<bf16>((size_t)L * D);
    buffer<bf16> y_slots = app.create_bo_buffer<bf16>((size_t)L * D);
    gen_dequant_mm_512(app.seq(), L, IS, D, 0, 0, 0);
    const double d_rows_us = time_calls([&]() { app(y_rows, hb_rows, w_down); }, iters, "down from row tiles");
    gen_dequant_mm_512(app.seq(), L, IS, D, 0, 0, 0, true);
    const double d_slots_us = time_calls([&]() { app(y_slots, slots, w_down); }, iters, "down from GLU slots");
    y_rows.sync_from_device();
    y_slots.sync_from_device();
    const bool same = memcmp(y_rows.data(), y_slots.data(), y_rows.size() * sizeof(bf16)) == 0;
    std::cout << std::fixed << std::setprecision(0) << "GLU L " << L << ": down from slots vs row tiles: " << (same ? "IDENTICAL" : "DIFFERENT")
              << ", " << d_slots_us << " us vs " << d_rows_us << " us" << std::endl;

    // what reaches the residual: down of A's tiles vs down of B's slots
    buffer<bf16> y_a = app.create_bo_buffer<bf16>((size_t)L * D);
    gen_dequant_mm_512(app.seq(), L, IS, D, 0, 0, 0);
    app(y_a, hb_a, w_down);
    y_a.sync_from_device();
    std::vector<float> f_ya((size_t)L * D), f_yb((size_t)L * D);
    for (size_t i = 0; i < f_ya.size(); i++) {
        f_ya[i] = (float)y_a[i];
        f_yb[i] = (float)y_slots[i];
    }
    const diff_stats down_b_vs_a = compare_floats(f_yb.data(), f_ya.data(), f_ya.size());
    std::cout << std::scientific << std::setprecision(3) << "GLU L " << L << ": down output B vs A rel L2 " << down_b_vs_a.rel_l2
              << " (cos " << std::fixed << std::setprecision(7) << down_b_vs_a.cos << ")" << std::endl;
}

struct shape_result {
    double us_per_call;
    uint64_t hash;
    double old_us_per_call = 0.0;
    uint64_t old_hash = 0;
    size_t old_mismatches = 0;
};

static shape_result run_shape(npu_app& app, npu_app* old_app, const mm_shape& s, int L, int iters, int check_rows) {
    const int D = s.din;
    const int DQ = s.dout;
    header_print("shape", s.name << ": L " << L << ", Din " << D << ", Dout " << DQ << ", act " << s.act);

    gen_dequant_mm_512(app.seq(), L, D, DQ, 0, 0, s.act);

    buffer<u8> quant_w = app.create_bo_buffer<u8>(q4_k_size((size_t)D * DQ));
    buffer<bf16> buffer_x = app.create_bo_buffer<bf16>((size_t)L * D);
    buffer<u8> buffer_xb = app.create_bo_buffer<u8>((size_t)L * D * 9 / 8);
    buffer<bf16> buffer_y = app.create_bo_buffer<bf16>((size_t)L * DQ);

    buffer<bf16> w((size_t)DQ * D);
    buffer<bf16> w_deq((size_t)DQ * D);
    buffer<u8> w_packed(q4_k_size((size_t)D * DQ));
    {
        std::mt19937 rng(0);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (size_t i = 0; i < w.size(); i++) w[i] = bf16(dist(rng));
    }
    q4_k_quantize_pack(w_packed.data(), w_deq.data(), w.data(), DQ, D);
    buffer<bf16> w_bfp((size_t)DQ * D);
    memcpy(w_bfp.data(), w_deq.data(), w_deq.size() * sizeof(bf16));
    bfp16_floor_k(w_bfp.data(), DQ, D);

    assert(w_packed.size() == quant_w.size());
    reorder_cpy(quant_w.data(), w_packed, D, 2);
    quant_w.sync_to_device();

    {
        std::mt19937 rng(1);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < buffer_x.size(); i++) buffer_x[i] = bf16(dist(rng));
    }
    x_to_bfp16_tiles(buffer_xb.data(), buffer_x.data(), L, D);
    buffer_xb.sync_to_device();
    buffer_y.memset(bf16(-1.0f));
    buffer_y.sync_to_device();

    shape_result res;
    res.us_per_call = time_calls([&]() { app(buffer_y, buffer_xb, quant_w); }, iters, "new");
    buffer_y.sync_from_device();
    res.hash = fnv1a(buffer_y.data(), buffer_y.size() * sizeof(bf16));

    // the old mm.xclbin on the same weights and the same x (as bf16 rows): must give the same bytes
    if (old_app != nullptr) {
        gen_dequant_mm_512_old(old_app->seq(), L, D, DQ, 0, 0, s.act);
        buffer<bf16> buffer_y_old = old_app->create_bo_buffer<bf16>((size_t)L * DQ);
        buffer_x.sync_to_device();
        buffer_y_old.memset(bf16(-2.0f));
        buffer_y_old.sync_to_device();
        res.old_us_per_call = time_calls([&]() { (*old_app)(buffer_y_old, buffer_x, quant_w); }, iters, "old");
        buffer_y_old.sync_from_device();
        res.old_hash = fnv1a(buffer_y_old.data(), buffer_y_old.size() * sizeof(bf16));
        const uint16_t* a = (const uint16_t*)buffer_y.data();
        const uint16_t* b = (const uint16_t*)buffer_y_old.data();
        for (size_t i = 0; i < buffer_y.size(); i++) {
            if (a[i] != b[i]) {
                if (res.old_mismatches < 5) {
                    std::cout << "MISMATCH row " << i / DQ << " col " << i % DQ << ": new " << (float)buffer_y[i]
                              << " old " << (float)buffer_y_old[i] << std::endl;
                }
                res.old_mismatches++;
            }
        }
        std::cout << "COMPARE " << s.name << " L " << L << " Din " << D << " Dout " << DQ << " act " << s.act
                  << ": " << (res.old_mismatches == 0 ? "IDENTICAL" : "DIFFERENT") << " (" << res.old_mismatches
                  << " of " << buffer_y.size() << " values differ), new " << (int)res.us_per_call << " us, old "
                  << (int)res.old_us_per_call << " us" << std::endl;
    }

    std::vector<int> rows = pick_check_rows(L, check_rows);
    const int R = rows.size();
    buffer<bf16> x_rows((size_t)R * D);
    buffer<bf16> y_rows((size_t)R * DQ);
    for (int r = 0; r < R; r++) {
        memcpy(x_rows.data() + (size_t)r * D, buffer_x.data() + (size_t)rows[r] * D, D * sizeof(bf16));
        memcpy(y_rows.data() + (size_t)r * DQ, buffer_y.data() + (size_t)rows[r] * DQ, DQ * sizeof(bf16));
    }
    buffer<bf16> y_ref((size_t)R * DQ);
    buffer<bf16> y_ref_bfp((size_t)R * DQ);
    cpu_matmul_nt(y_ref.data(), x_rows.data(), w_deq.data(), R, DQ, D);
    cpu_matmul_nt(y_ref_bfp.data(), x_rows.data(), w_bfp.data(), R, DQ, D);
    if (s.act == 1) {
        for (size_t i = 0; i < y_ref.size(); i++) {
            const float a = (float)y_ref[i];
            const float b = (float)y_ref_bfp[i];
            y_ref[i] = bf16(a / (1.0f + std::exp(-a)));
            y_ref_bfp[i] = bf16(b / (1.0f + std::exp(-b)));
        }
    }

    print_error_metrics(get_error_metrics(y_rows, y_ref), "vs dequantized weights (" + std::to_string(R) + " rows)");
    print_error_metrics(get_error_metrics(y_rows, y_ref_bfp), "vs bfp16-floored weights");
    return res;
}

int main(int argc, char **argv) {
    srand(0);
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("len,L", po::value<int>()->default_value(10752), "rows (prompt length, a multiple of 512)");
    desc.add_options()("shape,s", po::value<std::string>()->default_value("all"), "\"all\" (the MiniCPM-OCR shapes) or Din,Dout[,act]");
    desc.add_options()("iters,i", po::value<int>()->default_value(3), "timed calls per shape (after 1 warmup)");
    desc.add_options()("check,c", po::value<int>()->default_value(64), "rows checked against the CPU");
    desc.add_options()("xclbin,x", po::value<std::string>()->default_value("build/final/xclbins/dequant_mm.xclbin"), "xclbin");
    desc.add_options()("old", po::value<std::string>()->default_value(""), "old mm.xclbin: also run it on the same data and compare bit for bit");
    desc.add_options()("glu", po::bool_switch()->default_value(false), "only the GLU epilogue check (needs a GLU build)");
    arg_utils::parse_options(argc, argv, desc, vm);

    const int L = vm["len"].as<int>();
    const int iters = vm["iters"].as<int>();
    const int check_rows = vm["check"].as<int>();
    const std::string shape_arg = vm["shape"].as<std::string>();
    assert(L % 512 == 0);

    std::vector<mm_shape> shapes = minicpm_ocr_shapes;
    if (shape_arg != "all") {
        int din = 0;
        int dout = 0;
        int act = 0;
        char comma = 0;
        std::istringstream in(shape_arg);
        in >> din >> comma >> dout;
        if (in >> comma) {
            in >> act;
        }
        shapes = {{"custom", din, dout, act, 1}};
    }

    xrt::device npu_device = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* dequant_mm = xclbin_mgr.register_xclbin(vm["xclbin"].as<std::string>());
    npu_app app = dequant_mm->create_app();
    const std::string old_xclbin = vm["old"].as<std::string>();
    npu_app old_app;
    if (!old_xclbin.empty()) {
        old_app = xclbin_mgr.register_xclbin(old_xclbin)->create_app();
    }

    if (vm["glu"].as<bool>()) {
        run_glu(app, L, iters);
        return 0;
    }

    std::vector<shape_result> results;
    for (const mm_shape& s : shapes) {
        results.push_back(run_shape(app, old_xclbin.empty() ? nullptr : &old_app, s, L, iters, check_rows));
    }

    MSG_BONDLINE(100);
    std::cout << std::left << std::setw(18) << "shape" << std::setw(14) << "Din x Dout" << std::right
              << std::setw(12) << "us/call" << std::setw(10) << "TMAC/s" << std::setw(8) << "calls"
              << std::setw(14) << "ms/prefill" << "  output hash" << std::endl;
    double prefill_ms = 0.0;
    for (size_t i = 0; i < shapes.size(); i++) {
        const mm_shape& s = shapes[i];
        const shape_result& r = results[i];
        double macs = (double)L * s.din * s.dout;
        double tmacs = macs / (r.us_per_call * 1e-6) / 1e12;
        double ms = r.us_per_call * s.calls_per_prefill / 1000.0;
        prefill_ms += ms;
        std::ostringstream dims;
        dims << s.din << "x" << s.dout;
        std::cout << std::left << std::setw(18) << s.name << std::setw(14) << dims.str() << std::right
                  << std::setw(12) << std::fixed << std::setprecision(0) << r.us_per_call
                  << std::setw(10) << std::setprecision(2) << tmacs << std::setw(8) << s.calls_per_prefill
                  << std::setw(14) << std::setprecision(1) << ms
                  << "  " << std::hex << std::setw(16) << std::setfill('0') << r.hash << std::dec
                  << std::setfill(' ') << std::endl;
    }
    std::cout << "mm total per prefill at L " << L << ": " << std::fixed << std::setprecision(1) << prefill_ms << " ms" << std::endl;
    if (!old_xclbin.empty()) {
        double old_ms = 0.0;
        size_t different = 0;
        for (size_t i = 0; i < shapes.size(); i++) {
            old_ms += results[i].old_us_per_call * shapes[i].calls_per_prefill / 1000.0;
            different += results[i].old_mismatches != 0;
        }
        std::cout << "old mm total per prefill at L " << L << ": " << std::fixed << std::setprecision(1) << old_ms
                  << " ms; shapes bit-identical to the old xclbin: " << shapes.size() - different << " / " << shapes.size() << std::endl;
    }
    return 0;
}
