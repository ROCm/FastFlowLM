// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
#pragma once
/// @file lm_head_q4k.hpp
/// @brief Re-quantizes the tied q8_0 lm_head into the FLM q4_k format at load time.
///
/// Both formats tile the weight into 32 x 256 blocks (QXNX_ROW_BLOCK_SIZE x
/// QXNX_COL_BLOCK_SIZE), column major, so the conversion is block by block and keeps
/// the order reorder_cpy_lm_head expects.
///   q8_0 (8704 B): bf16 scales[g * 32 + r], int8 qs[c * 32 + r]; w = s q
///   q4_k (4736 B): uint8 s'[g * 32 + r], uint8 m'[g * 32 + r], uint4 qs (one column
///                  = 16 bytes, even row in the low nibble), bf16 S[r], bf16 M[r];
///                  w = S s' q + M m'  (M stored negated, the kernel adds the min path)
/// One block row holds exactly one 256-wide super-block, so S / M are one per row.
/// Fit: per group of 32 a weighted least-squares (scale, min) search over the uint4
/// grid (llama.cpp make_qkx2_quants), then the (BF16, uint8) re-fit of
/// FLM_Q4NX_Converter/quant.md with a 3-candidate search, then q re-rounded against
/// the fitted values.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace qwen3_5vl_lm_head_q4k {

constexpr int ROWS = 32, COLS = 256, GROUP = 32, NG = COLS / GROUP;
constexpr size_t Q80_BLOCK = 2 * ROWS * NG + ROWS * COLS;                 // 8704
constexpr size_t Q4K_BLOCK = 2 * ROWS * NG + ROWS * COLS / 2 + 4 * ROWS;  // 4736

static inline float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static inline uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
/// smallest bf16 >= f, f >= 0
static inline uint16_t f2bf_up(float f) { uint32_t u; std::memcpy(&u, &f, 4); uint16_t h = (uint16_t)(u >> 16); if (bf2f(h) < f) h++; return h; }

/// x ~= scale * q - umin, q in [0, 15], scale >= 0, umin >= 0
static inline void fit_group(const float* x, float& scale, float& umin)
{
    float mn = x[0], mx = x[0], sx2 = 0;
    for (int i = 0; i < GROUP; i++) { mn = std::min(mn, x[i]); mx = std::max(mx, x[i]); sx2 += x[i] * x[i]; }
    if (mn > 0) mn = 0;
    if (mx == mn) { scale = 0; umin = -mn; return; }
    const float av = std::sqrt(sx2 / GROUP);
    float w[GROUP];
    for (int i = 0; i < GROUP; i++) w[i] = av + std::fabs(x[i]);
    auto err_of = [&](float s, float m) {
        float e = 0;
        for (int i = 0; i < GROUP; i++) {
            float q = s > 0 ? std::clamp(std::nearbyint((x[i] - m) / s), 0.f, 15.f) : 0.f;
            float d = s * q + m - x[i]; e += w[i] * d * d;
        }
        return e;
    };
    float best_s = (mx - mn) / 15, best_m = mn, best_e = err_of(best_s, best_m);
    for (int is = 0; is <= 20; is++) {
        const float iscale = (-1.f + 0.1f * is + 15.f) / (mx - mn);
        float sw = 0, sxv = 0, sl = 0, sl2 = 0, sxl = 0;
        for (int i = 0; i < GROUP; i++) {
            float l = std::clamp(std::nearbyint(iscale * (x[i] - mn)), 0.f, 15.f);
            sw += w[i]; sxv += w[i] * x[i]; sl += w[i] * l; sl2 += w[i] * l * l; sxl += w[i] * l * x[i];
        }
        const float D = sw * sl2 - sl * sl;
        if (D <= 0) continue;
        float s = (sw * sxl - sxv * sl) / D, m = (sl2 * sxv - sl * sxl) / D;
        if (m > 0) { m = 0; s = sl2 > 0 ? sxl / sl2 : 0; }
        if (s <= 0) continue;
        const float e = err_of(s, m);
        if (e < best_e) { best_e = e; best_s = s; best_m = m; }
    }
    scale = best_s; umin = -best_m;
}

/// t[0..7] >= 0 -> bf16 P and uint8 p[j] minimizing sum (P p_j - t_j)^2 over 3 candidates
static inline void refit(const float* t, uint16_t& P, uint8_t* p)
{
    float tmax = 0;
    for (int j = 0; j < NG; j++) tmax = std::max(tmax, t[j]);
    if (tmax == 0) { P = 0; std::memset(p, 0, NG); return; }
    const uint16_t h0 = f2bf_up(tmax / 255);
    float best = INFINITY;
    for (uint16_t h = h0; h < h0 + 3; h++) {
        const float Pf = bf2f(h);
        uint8_t c[NG]; float e = 0;
        for (int j = 0; j < NG; j++) {
            c[j] = (uint8_t)std::min(255.f, std::nearbyint(t[j] / Pf));
            float d = Pf * c[j] - t[j]; e += d * d;
        }
        if (e < best) { best = e; P = h; std::memcpy(p, c, NG); }
    }
}

/// one 32 x 256 q8_0 block -> one q4_k block
static inline void q80_to_q4k(const uint8_t* src, uint8_t* dst)
{
    const uint16_t* s8 = (const uint16_t*)src;
    const int8_t* q8 = (const int8_t*)(src + 2 * ROWS * NG);
    uint8_t* sc = dst;
    uint8_t* mi = dst + ROWS * NG;
    uint8_t* qs = dst + 2 * ROWS * NG;
    uint16_t* S = (uint16_t*)(qs + ROWS * COLS / 2);
    uint16_t* M = S + ROWS;
    std::memset(qs, 0, ROWS * COLS / 2);
    for (int r = 0; r < ROWS; r++) {
        float x[COLS], t[NG], u[NG];
        for (int c = 0; c < COLS; c++) x[c] = bf2f(s8[(c / GROUP) * ROWS + r]) * q8[c * ROWS + r];
        for (int g = 0; g < NG; g++) fit_group(x + g * GROUP, t[g], u[g]);
        uint8_t sp[NG], mp[NG];
        refit(t, S[r], sp);
        refit(u, M[r], mp);
        const float Sf = bf2f(S[r]), Mf = bf2f(M[r]);
        M[r] = f2bf(-Mf);
        for (int g = 0; g < NG; g++) {
            sc[g * ROWS + r] = sp[g]; mi[g * ROWS + r] = mp[g];
            const float d = Sf * sp[g], dm = Mf * mp[g];
            for (int i = 0; i < GROUP; i++) {
                const int c = g * GROUP + i;
                const int q = d > 0 ? (int)std::clamp(std::nearbyint((x[c] + dm) / d), 0.f, 15.f) : 0;
                qs[c * ROWS / 2 + r / 2] |= (uint8_t)(q << ((r & 1) * 4));
            }
        }
    }
}

} // namespace qwen3_5vl_lm_head_q4k
