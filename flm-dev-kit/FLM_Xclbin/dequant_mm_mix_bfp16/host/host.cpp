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
#include <immintrin.h>
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
#include <math.h>
#include "metrices.hpp"
#include "seq_gen.hpp"
#include "cpu_mix.hpp"

namespace po = boost::program_options;

/// \brief Put a matrix on the grid aie::mmul actually multiplies on: every block of 8 consecutive k at a fixed row is given
/// a shared exponent and an 8-bit mantissa, truncated toward -infinity.
static void bfp16_floor_k(bf16* v, size_t rows, size_t k) {
#pragma omp parallel for schedule(static)
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

/// One block of 8 values as the NPU stores it: a shared exponent byte, then 8 two's complement mantissa bytes.  The core's
/// conversion truncates, so this does too (same arithmetic as the converter in mlir-aie's block datatype helper).
static void bfp16_block8(const float* v, uint8_t* out9) {
    unsigned max_exp = 0;
    for (int e = 0; e < 8; e++) {
        unsigned bits; memcpy(&bits, v + e, 4);
        max_exp = std::max(max_exp, (bits >> 23) & 0xFFu);
    }
    out9[0] = (uint8_t)max_exp;
    for (int e = 0; e < 8; e++) {
        unsigned bits; memcpy(&bits, v + e, 4);
        const unsigned sign = bits & 0x80000000u;
        const unsigned exp = (bits >> 23) & 0xFFu;
        unsigned mant = bits & 0x007FFFFFu;
        if (exp) mant |= 0x00800000u;
        mant = sign ? ~mant + 1 : mant;
        int8_t q = (int8_t)(mant >> (23 - 7 + 1));
        q = (max_exp - exp >= 32) ? (int8_t)(sign ? 0xFF : 0x00) : (int8_t)(q >> (max_exp - exp));
        out9[1 + e] = (uint8_t)q;
    }
}

/// X as gen_dequant_mm_mix streams it: per pass mp and core row r one block of
///     [512 k round][256 k slice j (2)][16-row group i (8 / nb)][128 k half v (2)][k tile (16)][8-row tile z (2)]
/// tiles of 8 rows x 8 k (72 B), the rows of group i being mp (512 / nb) + r 16 (8 / nb) + 16 i ..  Rows past L are zero.
static void pack_x_mix(uint8_t* out, const bf16* x, uint32_t L, uint32_t L_padded, uint32_t Din, uint32_t nb) {
    const uint32_t ms = 8 / nb;
    const uint32_t m_pass = mix_pass_rows(nb);
    const uint32_t k_rounds = Din / 512;
    const size_t round_bytes = (size_t)4 * ms * MIX_X_PACKET_BYTES;
    const size_t block_bytes = k_rounds * round_bytes;
    const uint32_t blocks = L_padded / m_pass * MIX_CORE_ROWS;
#pragma omp parallel for collapse(2) schedule(static)
    for (uint32_t b = 0; b < blocks; b++) {
        for (uint32_t kr = 0; kr < k_rounds; kr++) {
            const uint32_t mp = b / MIX_CORE_ROWS, r = b % MIX_CORE_ROWS;
            uint8_t* p = out + b * block_bytes + kr * round_bytes;
            for (uint32_t j = 0; j < 2; j++) for (uint32_t i = 0; i < ms; i++) for (uint32_t v = 0; v < 2; v++)
            for (uint32_t kt = 0; kt < 16; kt++) for (uint32_t z = 0; z < 2; z++) {
                const size_t r0 = (size_t)mp * m_pass + r * 16 * ms + i * 16 + z * 8;
                const size_t k0 = (size_t)kr * 512 + j * 256 + v * 128 + kt * 8;
                for (int row = 0; row < 8; row++) {
                    float vals[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    if (r0 + row < L) for (int e = 0; e < 8; e++) vals[e] = (float)x[(r0 + row) * Din + k0 + e];
                    bfp16_block8(vals, p); p += 9;
                }
            }
        }
    }
}

/// Two rows of bfp16_block8 at once: lanes 0..7 are 8 k of row a, 8..15 of row b (bf16, zero for a row past L).
static inline void bfp16_block8_x2(__m256i ab, uint8_t* out_a, uint8_t* out_b) {
    const __m512i bits = _mm512_slli_epi32(_mm512_cvtepu16_epi32(ab), 16);
    const __m512i exp = _mm512_and_si512(_mm512_srli_epi32(bits, 23), _mm512_set1_epi32(0xFF));
    // the largest exponent of each 8 lanes, in every lane of them
    __m512i mx = exp;
    mx = _mm512_max_epu32(mx, _mm512_permutexvar_epi32(_mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1), mx));
    mx = _mm512_max_epu32(mx, _mm512_permutexvar_epi32(_mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2), mx));
    mx = _mm512_max_epu32(mx, _mm512_permutexvar_epi32(_mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4), mx));
    __m512i mant = _mm512_and_si512(bits, _mm512_set1_epi32(0x007FFFFF));
    mant = _mm512_mask_or_epi32(mant, _mm512_test_epi32_mask(exp, exp), mant, _mm512_set1_epi32(0x00800000));
    mant = _mm512_mask_sub_epi32(mant, _mm512_test_epi32_mask(bits, _mm512_set1_epi32((int)0x80000000)), _mm512_setzero_si512(), mant);
    // (int8)(mant >> 17) >> (max - exp), as one arithmetic shift (32 or more gives the sign, as the scalar code)
    const __m512i q = _mm512_srav_epi32(mant, _mm512_add_epi32(_mm512_sub_epi32(mx, exp), _mm512_set1_epi32(17)));
    alignas(16) uint8_t b[16];
    _mm_store_si128((__m128i*)b, _mm512_cvtepi32_epi8(q));
    out_a[0] = (uint8_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(mx));
    out_b[0] = (uint8_t)_mm_cvtsi128_si32(_mm512_extracti32x4_epi32(mx, 2));
    memcpy(out_a + 1, b, 8);
    memcpy(out_b + 1, b + 8, 8);
}

/// pack_x_mix with AVX-512, the one the engine uses
static void pack_x_mix_fast(uint8_t* out, const bf16* x, uint32_t L, uint32_t L_padded, uint32_t Din, uint32_t nb) {
    const uint32_t ms = 8 / nb;
    const uint32_t m_pass = mix_pass_rows(nb);
    const uint32_t k_rounds = Din / 512;
    const size_t round_bytes = (size_t)4 * ms * MIX_X_PACKET_BYTES;
    const size_t block_bytes = k_rounds * round_bytes;
    const uint32_t blocks = L_padded / m_pass * MIX_CORE_ROWS;
#pragma omp parallel for collapse(2) schedule(static)
    for (uint32_t b = 0; b < blocks; b++) {
        for (uint32_t kr = 0; kr < k_rounds; kr++) {
            const uint32_t mp = b / MIX_CORE_ROWS, r = b % MIX_CORE_ROWS;
            uint8_t* p = out + b * block_bytes + kr * round_bytes;
            for (uint32_t j = 0; j < 2; j++) for (uint32_t i = 0; i < ms; i++) for (uint32_t v = 0; v < 2; v++)
            for (uint32_t kt = 0; kt < 16; kt++) for (uint32_t z = 0; z < 2; z++) {
                const size_t r0 = (size_t)mp * m_pass + r * 16 * ms + i * 16 + z * 8;
                const size_t k0 = (size_t)kr * 512 + j * 256 + v * 128 + kt * 8;
                for (int row = 0; row < 8; row += 2, p += 18) {
                    const __m128i a = r0 + row < L ? _mm_loadu_si128((const __m128i*)(x + (r0 + row) * Din + k0)) : _mm_setzero_si128();
                    const __m128i c = r0 + row + 1 < L ? _mm_loadu_si128((const __m128i*)(x + (r0 + row + 1) * Din + k0)) : _mm_setzero_si128();
                    bfp16_block8_x2(_mm256_inserti128_si256(_mm256_castsi128_si256(a), c, 1), p, p + 9);
                }
            }
        }
    }
}

static std::vector<uint32_t> parse_list(const char* s, std::vector<uint32_t> fallback) {
    if (!s) return fallback;
    std::vector<uint32_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back((uint32_t)std::stoul(item));
    return out;
}

/// "q4:1024,q8:1024,q4:2048": the bands of one call, in output order
static std::vector<std::pair<int, uint32_t>> parse_bands(const char* s) {
    std::vector<std::pair<int, uint32_t>> out;
    std::stringstream ss(s ? s : "q4:1024,q8:1024,q4:2048");
    std::string item;
    while (std::getline(ss, item, ',')) {
        const size_t colon = item.find(':');
        const std::string f = item.substr(0, colon);
        out.push_back({f == "q8" ? MIX_FMT_Q8_0 : MIX_FMT_Q4_1, (uint32_t)std::stoul(item.substr(colon + 1))});
    }
    return out;
}

int main(int argc, char **argv) {
    srand(0);
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);
    arg_utils::parse_options(argc, argv, desc, vm);

    // Self-contained: random weights are quantized to the mixed decode layout (Q4_1 / Q8_0 blocks in 10240 B units) on the
    // CPU, and the golden output is the fp32 product of what the cores multiply: the dequantized weights and X, both on the
    // bfp16 grid.  Shapes (env): KT_DIN, KT_BANDS (fmt:rows,...), KT_L (comma list), KT_WARMUP, KT_ITERS.
    const uint32_t Din = getenv("KT_DIN") ? atoi(getenv("KT_DIN")) : 2048;
    const auto band_spec = parse_bands(getenv("KT_BANDS"));
    const std::vector<uint32_t> Ls = parse_list(getenv("KT_L"), {28, 64, 128, 255, 512, 904});
    const int Warmup = getenv("KT_WARMUP") ? atoi(getenv("KT_WARMUP")) : 10;
    const int Iterations = getenv("KT_ITERS") ? atoi(getenv("KT_ITERS")) : 30;

    std::vector<mix_band> bands;
    uint32_t Dout = 0;
    size_t w_bytes = 0;
    for (const auto& [fmt, rows] : band_spec) {
        bands.push_back({w_bytes, rows, fmt});
        w_bytes += (size_t)rows / MIX_SLAB_ROWS * mix_slab_bytes(fmt, Din);
        Dout += rows;
    }
    std::cout << "Din " << Din << ", Dout " << Dout << ", weight bytes " << w_bytes << std::endl;

    xrt::device npu_device = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* dequant_mm = xclbin_mgr.register_xclbin("build/xclbins/dequant_mm.xclbin");

    //////////////////////////// weights ////////////////////////////
    header_print("DEBUG", "Quantizing the weights on the CPU");
    buffer<bf16> w_bfp((size_t)Dout * Din);   // what the cores multiply
    std::vector<u8> w_units(w_bytes);
    {
        std::mt19937 rng(0);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        uint32_t row0 = 0;
        for (const mix_band& b : bands) {
            buffer<bf16> w((size_t)b.rows * Din);
            for (size_t i = 0; i < w.size(); i++) w[i] = bf16(dist(rng));
            const size_t block = b.fmt == MIX_FMT_Q4_1 ? MIX_Q41_BLOCK : MIX_Q80_BLOCK;
            std::vector<u8> blocks((size_t)b.rows / 32 * Din / 256 * block);
            mix_quantize(blocks.data(), w_bfp.data() + (size_t)row0 * Din, w.data(), b.rows, Din, b.fmt);
            mix_reorder_units(w_units.data() + b.offset, blocks.data(), b.rows, Din, b.fmt);
            row0 += b.rows;
        }
        bfp16_floor_k(w_bfp.data(), Dout, Din);
    }

    bool all_ok = true;
    for (const uint32_t L : Ls) {
        const uint32_t nb = mix_nb(L, Dout);
        const uint32_t L_padded = mix_rows_padded(L, Dout);
        std::cout << "==== L " << L << " (nb " << nb << ", rows padded " << L_padded << ")" << std::endl;

        npu_app mm = dequant_mm->create_app();
        gen_dequant_mm_mix(mm.seq(), L, Din, bands, 0, 0);
        buffer<u8> buffer_w = mm.create_bo_buffer<u8>(w_bytes);
        buffer<u8> buffer_x = mm.create_bo_buffer<u8>((size_t)L_padded * Din * 9 / 8);
        buffer<bf16> buffer_y = mm.create_bo_buffer<bf16>((size_t)L_padded * Dout);
        memcpy(buffer_w.data(), w_units.data(), w_bytes);
        buffer_w.sync_to_device();

        buffer<bf16> x((size_t)L * Din);
        {
            std::mt19937 rng(1 + L);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (size_t i = 0; i < x.size(); i++) x[i] = bf16(dist(rng));
        }
        // the scalar packer is the reference of the fast one; X goes to the NPU from the fast one
        std::vector<uint8_t> x_ref_packed(buffer_x.size());
        pack_x_mix(x_ref_packed.data(), x.data(), L, L_padded, Din, nb);
        pack_x_mix_fast(buffer_x.data(), x.data(), L, L_padded, Din, nb);
        time_utils::time_point pack_start = time_utils::now();
        for (int i = 0; i < 5; i++) pack_x_mix_fast(buffer_x.data(), x.data(), L, L_padded, Din, nb);
        time_utils::time_point pack_stop = time_utils::now();
        const bool pack_same = memcmp(x_ref_packed.data(), buffer_x.data(), buffer_x.size()) == 0;
        std::cout << "  fast X packer " << (pack_same ? "matches" : "DIFFERS FROM") << " the scalar one" << std::endl;
        all_ok &= pack_same;
        buffer_x.sync_to_device();
        buffer_y.memset(bf16(-1.0f));
        buffer_y.sync_to_device();

        // a strided subset of rows (every pass and core row is hit), all of them when L is small
        const uint32_t stride = (L + 255) / 256;
        std::vector<uint32_t> rows;
        for (uint32_t l = 0; l < L; l += stride) rows.push_back(l);
        if (rows.back() != L - 1) rows.push_back(L - 1);
        buffer<bf16> x_ref((size_t)rows.size() * Din);
        for (size_t i = 0; i < rows.size(); i++) memcpy(x_ref.data() + i * Din, x.data() + (size_t)rows[i] * Din, Din * 2);
        bfp16_floor_k(x_ref.data(), rows.size(), Din);
        buffer<bf16> y_ref((size_t)rows.size() * Dout);
        cpu_matmul_nt(y_ref.data(), x_ref.data(), w_bfp.data(), rows.size(), Dout, Din);

        // one call first, on its own: a new sequence must finish before it is timed
        mm(buffer_y, buffer_x, buffer_w);
        for (int i = 0; i < Warmup; i++) mm(buffer_y, buffer_x, buffer_w);
        std::vector<double> call_us;
        for (int i = 0; i < Iterations; i++) {
            time_utils::time_point start = time_utils::now();
            mm(buffer_y, buffer_x, buffer_w);
            time_utils::time_point stop = time_utils::now();
            call_us.push_back(time_utils::duration_us(start, stop).first);
        }
        std::sort(call_us.begin(), call_us.end());
        const double med_us = call_us[call_us.size() / 2];
        buffer_y.sync_from_device();

        buffer<bf16> y((size_t)rows.size() * Dout);
        for (size_t i = 0; i < rows.size(); i++) memcpy(y.data() + i * Dout, buffer_y.data() + (size_t)rows[i] * Dout, Dout * 2);
        error_metrics m = get_error_metrics(y, y_ref);
        print_error_metrics(m);
        {
            // the outputs far off the reference (a handful of ulps is rounding): where they are says what is wrong
            size_t bad = 0;
            std::map<uint32_t, size_t> bad_cols;
            for (size_t i = 0; i < rows.size(); i++)
                for (uint32_t n = 0; n < Dout; n++) {
                    const float a = (float)y[i * Dout + n], b = (float)y_ref[i * Dout + n];
                    if (std::fabs(a - b) > 0.05f * std::fabs(b) + 0.05f) {
                        if (bad < 8) std::cout << "  off: row " << rows[i] << " col " << n << " npu " << a << " ref " << b << std::endl;
                        bad++;
                        bad_cols[n / 64]++;
                    }
                }
            std::cout << "  off: " << bad << " of " << y.size() << " in " << bad_cols.size() << " 64-col slabs";
            int shown = 0;
            for (const auto& [s, c] : bad_cols) if (shown++ < 12) std::cout << " " << s << ":" << c;
            std::cout << std::endl;
            all_ok &= bad == 0;
        }
        // per band, so a format or a band offset that is wrong shows on its own
        uint32_t row0 = 0;
        for (const mix_band& b : bands) {
            buffer<bf16> yb((size_t)rows.size() * b.rows), rb((size_t)rows.size() * b.rows);
            for (size_t i = 0; i < rows.size(); i++) {
                memcpy(yb.data() + i * b.rows, y.data() + i * Dout + row0, b.rows * 2);
                memcpy(rb.data() + i * b.rows, y_ref.data() + i * Dout + row0, b.rows * 2);
            }
            error_metrics mb = get_error_metrics(yb, rb);
            std::cout << "  band " << (b.fmt == MIX_FMT_Q4_1 ? "q4" : "q8") << ":" << b.rows << " cos " << mb.CosineSimilarity
                      << " relL2 " << mb.RelativeL2 << std::endl;
            // the bf16 truncation of the output alone is ~0.004 (the fp32 cosine of get_error_metrics loses digits over
            // millions of outputs, so it is not the gate)
            all_ok &= mb.RelativeL2 < 0.006f;
            row0 += b.rows;
        }
        all_ok &= m.RelativeL2 < 0.006f;

        const double flops = 2.0 * L * (double)Din * Dout;
        std::cout << "RESULT L=" << L << " nb=" << nb << " Din=" << Din << " Dout=" << Dout << " us=" << med_us
                  << " min_us=" << call_us.front() << " TFLOPS=" << flops / (med_us * 1e-6) / 1e12
                  << " weight_GBps=" << w_bytes / (med_us * 1e-6) / 1e9
                  << " pack_x_ms=" << time_utils::duration_us(pack_start, pack_stop).first / 5000.0
                  << " cos=" << m.CosineSimilarity << std::endl;
    }
    std::cout << (all_ok ? "PASSED" : "FAILED") << std::endl;
    return all_ok ? 0 : 1;
}
