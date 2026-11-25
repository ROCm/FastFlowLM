// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "qwen3_asr_audio_gemm_npu.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "qwen3_asr_audio_gemm_seq.hpp"
#include "qwen3_asr_audio_simd.hpp"
#include "../omp_threads.hpp"

namespace seqg = qwen3_asr_audio_gemm_seq;
namespace simd = qwen3_asr_audio_simd;

namespace {

constexpr float LO_SCALE = 256.f;   // the rest of A is scaled up so the bias the cores add to it costs no precision

inline uint16_t f32_to_bf16(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
inline float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// two bfp16 blocks from 16 floats (rounded to bf16 first), the same integer steps as seqg::encode_block, one block per 8 lanes.
/// rest (optional): (bf16(a) - its bfp16 value) * LO_SCALE
inline void encode2(uint8_t* o0, uint8_t* o1, const float* a, float* rest) {
    const __m512i one = _mm512_set1_epi32(1);
    __m512i u = _mm512_castps_si512(_mm512_loadu_ps(a));
    u = _mm512_srli_epi32(_mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), _mm512_and_si512(_mm512_srli_epi32(u, 16), one))), 16);
    const __m512i ex = _mm512_and_si512(_mm512_srli_epi32(u, 7), _mm512_set1_epi32(0xFF));
    const __m512i x1 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    const __m512i x2 = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
    const __m512i x4 = _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
    __m512i mx = _mm512_max_epi32(ex, _mm512_permutexvar_epi32(x1, ex));
    mx = _mm512_max_epi32(mx, _mm512_permutexvar_epi32(x2, mx));
    mx = _mm512_max_epi32(mx, _mm512_permutexvar_epi32(x4, mx));
    __m512i mag = _mm512_slli_epi32(_mm512_and_si512(u, _mm512_set1_epi32(0x7F)), 16);
    mag = _mm512_mask_or_epi32(mag, _mm512_test_epi32_mask(ex, ex), mag, _mm512_set1_epi32(0x800000));
    const __m512i s = _mm512_mask_sub_epi32(mag, _mm512_test_epi32_mask(u, _mm512_set1_epi32(0x8000)), _mm512_setzero_si512(), mag);
    const __m512i sh = _mm512_min_epi32(_mm512_sub_epi32(_mm512_add_epi32(mx, _mm512_set1_epi32(17)), ex), _mm512_set1_epi32(31));
    __m512i q = _mm512_srav_epi32(s, sh);
    const __m512i rem = _mm512_sub_epi32(s, _mm512_sllv_epi32(q, sh));
    const __m512i half = _mm512_sllv_epi32(one, _mm512_sub_epi32(sh, one));
    const __mmask16 up = _mm512_cmpgt_epi32_mask(rem, half) | (_mm512_cmpeq_epi32_mask(rem, half) & _mm512_test_epi32_mask(q, one));
    q = _mm512_min_epi32(_mm512_mask_add_epi32(q, up, q, one), _mm512_set1_epi32(127));
    alignas(16) uint8_t qb[16], eb[16];
    _mm_store_si128((__m128i*)qb, _mm512_cvtepi32_epi8(q));
    _mm_store_si128((__m128i*)eb, _mm512_cvtepi32_epi8(mx));
    o0[0] = eb[0];
    std::memcpy(o0 + 1, qb, 8);
    o1[0] = eb[8];
    std::memcpy(o1 + 1, qb + 8, 8);
    if (rest) {   // block value = q * 2^(mx - 133); blocks under 2^-120 decode to 0 (their rest is then the whole value)
        const __m512 scale = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_max_epi32(_mm512_sub_epi32(mx, _mm512_set1_epi32(6)), _mm512_setzero_si512()), 23));
        const __m512 val = _mm512_mul_ps(_mm512_cvtepi32_ps(q), scale);
        const __m512 a_bf = _mm512_castsi512_ps(_mm512_slli_epi32(u, 16));
        _mm512_storeu_ps(rest, _mm512_mul_ps(_mm512_sub_ps(a_bf, val), _mm512_set1_ps(LO_SCALE)));
    }
}

/// K padded to the kernel's chunk: 128, or 512 when the weights stream (more than KC_SINGLE chunks)
int pad_k(int K) {
    const int k = (K + 127) / 128 * 128;
    return k / 128 > (int)seqg::KC_SINGLE ? (K + 511) / 512 * 512 : k;
}

/// one block of 8 (k = 8 k8 .. ; values past K, or of a missing row, are 0) the slow way: hi, and lo (optional) as in encode2
void encode_tail(uint8_t* hi, uint8_t* lo, const float* a, int k8, int K) {
    uint16_t v[8] = {}, w[8] = {};
    for (int i = 0; i < 8; i++)
        if (a && 8 * k8 + i < K) v[i] = f32_to_bf16(a[8 * k8 + i]);
    seqg::encode_block(hi, v);
    if (!lo) return;
    for (int i = 0; i < 8; i++) w[i] = f32_to_bf16((bf16_to_f32(v[i]) - std::ldexp((float)(int8_t)hi[1 + i], (int)hi[0] - 133)) * LO_SCALE);
    seqg::encode_block(lo, w);
}

/// rows [r0, r0 + rows) of A -> x: block (row / 16, kc), then [16 k tiles][2 row halves][8 rows][9 B]; the last block's rows past
/// rows are 0. KC: chunks of 128 k. lo (optional): the same for the rest of A, scaled by LO_SCALE. tmp: 16 K floats per thread for A.
/// A task packs a 16-row block: a block's bytes are written by one thread only, a whole kc at a time.
void pack_x(uint8_t* hi, uint8_t* lo, const qwen3_asr_audio_gemm::rows_fn& A, int K, int r0, int rows, int KC, float* tmp) {
    constexpr size_t BLOCK = seqg::X_BLOCK_WORDS * 4;
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads()) schedule(dynamic)
    for (int rb = 0; rb < (rows + 15) / 16; rb++) {
        float* t = tmp + (size_t)omp_get_thread_num() * 16 * K;
        const float* a[16];
        const bool full = rb * 16 + 16 <= rows;
        for (int i = 0; i < 16; i++) a[i] = rb * 16 + i < rows ? A(r0 + rb * 16 + i, t + (size_t)i * K) : nullptr;
        alignas(64) uint8_t bh[BLOCK], bl[BLOCK];
        for (int kc = 0; kc < KC; kc++) {
            if (!full) {   // a zero block is all zero bytes
                std::memset(bh, 0, BLOCK);
                std::memset(bl, 0, BLOCK);
            }
            for (int i = 0; i < 16; i++) {
                if (!a[i]) continue;
                uint8_t* h = bh + (i / 8) * 72 + (i % 8) * 9;
                uint8_t* l = bl + (i / 8) * 72 + (i % 8) * 9;
                for (int kt = 0; kt < 16; kt += 2) {
                    const int k8 = kc * 16 + kt;
                    if (8 * k8 + 16 <= K) {
                        float rest[16] = {};
                        encode2(h + kt * 144, h + (kt + 1) * 144, a[i] + 8 * k8, lo ? rest : nullptr);
                        if (lo) encode2(l + kt * 144, l + (kt + 1) * 144, rest, nullptr);
                    } else {
                        encode_tail(h + kt * 144, lo ? l + kt * 144 : nullptr, a[i], k8, K);
                        encode_tail(h + (kt + 1) * 144, lo ? l + (kt + 1) * 144 : nullptr, a[i], k8 + 1, K);
                    }
                }
            }
            std::memcpy(hi + ((size_t)rb * KC + kc) * BLOCK, bh, BLOCK);
            if (lo) std::memcpy(lo + ((size_t)rb * KC + kc) * BLOCK, bl, BLOCK);
        }
    }
}

/// W [N][K] -> chunk (nb, column, kc, half) = [4 jp][16 k tiles][2][8 columns][9 B], then the half's 64 bf16 biases
void pack_w(uint8_t* dst, const float* W, const float* bias, int N, int K, int n_pad, int k_pad) {
    const int KC = k_pad / 128;
    const size_t half_bytes = seqg::W_HALF_WORDS * 4;
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
    for (int n = 0; n < n_pad; n++) {
        const int nb = n / 1024, col = (n / 128) % 8, h = (n / 64) % 2, jp = (n / 16) % 4, j = (n / 8) % 2, c8 = n % 8;
        const uint16_t b = (n < N && bias) ? f32_to_bf16(bias[n]) : 0;
        for (int kc = 0; kc < KC; kc++) {
            uint8_t* chunk = dst + ((((size_t)nb * 8 + col) * KC + kc) * 2 + h) * half_bytes;
            for (int kt = 0; kt < 16; kt++) {
                uint16_t v[8] = {};
                if (n < N)
                    for (int i = 0; i < 8; i++) {
                        const int k = kc * 128 + kt * 8 + i;
                        if (k < K) v[i] = f32_to_bf16(W[(size_t)n * K + k]);
                    }
                seqg::encode_block(chunk + ((jp * 16 + kt) * 2 + j) * 72 + c8 * 9, v);
            }
            std::memcpy(chunk + 9216 + 2 * (n % 64), &b, 2);
        }
    }
}

}  // namespace

qwen3_asr_audio_gemm_npu::qwen3_asr_audio_gemm_npu(npu_app_manager* mgr) : mgr_(mgr), io_app_(mgr->create_app()) {}

int qwen3_asr_audio_gemm_npu::add_weight(const float* W, int N, int K, const float* bias, bool exact_a, bool exact_w) {
    if (N % 16) throw std::runtime_error("qwen3_asr_audio_gemm_npu: N must be a multiple of 16");
    auto w = std::make_unique<weight_t>();
    w->N = N;
    w->K = K;
    w->n_pad = ((exact_w ? 2 : 1) * N + 1023) / 1024 * 1024;
    w->k_pad = pad_k(K);
    w->split = exact_a;
    w->split_w = exact_w;
    w->bias.assign(N, 0.f);
    if (bias)
        for (int n = 0; n < N; n++) w->bias[n] = bf16_to_f32(f32_to_bf16(bias[n]));
    std::vector<float> w2, b2;
    if (exact_w) {   // [W; (W - its bfp16 value) * LO_SCALE], the rest without bias
        w2.assign((size_t)2 * N * K, 0.f);
        std::memcpy(w2.data(), W, sizeof(float) * N * K);
        b2.assign(2 * N, 0.f);
        if (bias) std::memcpy(b2.data(), bias, sizeof(float) * N);
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
        for (int n = 0; n < N; n++)
            for (int k0 = 0; k0 < K; k0 += 8) {
                uint16_t v[8] = {};
                uint8_t e[9];
                for (int i = 0; i < 8 && k0 + i < K; i++) v[i] = f32_to_bf16(W[(size_t)n * K + k0 + i]);
                seqg::encode_block(e, v);
                for (int i = 0; i < 8 && k0 + i < K; i++)
                    w2[(size_t)(N + n) * K + k0 + i] = (bf16_to_f32(v[i]) - std::ldexp((float)(int8_t)e[1 + i], (int)e[0] - 133)) * LO_SCALE;
            }
        W = w2.data();
        bias = b2.data();
        N *= 2;
    }
    w->w = io_app_.create_bo_buffer<u8>(seqg::w_bytes(w->n_pad, w->k_pad));
    pack_w(w->w.data(), W, bias, N, K, w->n_pad, w->k_pad);
    w->w.sync_to_device();
    weights_.push_back(std::move(w));
    return (int)weights_.size() - 1;
}

npu_app& qwen3_asr_audio_gemm_npu::app_for(uint32_t rounds, uint32_t kc, uint32_t nb) {
    const shape_t s{rounds, kc, nb};
    auto it = apps_.find(s);
    if (it != apps_.end()) return *it->second;
    auto app = std::make_unique<npu_app>(mgr_->create_app());
    seqg::call_t call{rounds, kc, nb, nb * seqg::BLOCK_COLS, /*act=*/0};
    seqg::gen_gemm(app->seq(), call);
    return *apps_.emplace(s, std::move(app)).first->second;
}

void qwen3_asr_audio_gemm_npu::grow(size_t x_bytes, size_t y_elems, int k) {
    if (x_.size() < x_bytes) x_ = io_app_.create_bo_buffer<u8>(x_bytes);
    if (y_.size() < y_elems) y_ = io_app_.create_bo_buffer<u16>(y_elems);
    const size_t t = (size_t)qwen3_asr_cpu::asr_threads() * 16 * k;
    if (tmp_.size() < t) tmp_.resize(t);
}

void qwen3_asr_audio_gemm_npu::run(int weight, const rows_fn& A, int M, float* C, bool gelu) {
    std::lock_guard<std::mutex> lock(mu_);
    weight_t& w = *weights_.at(weight);
    grow(0, 0, w.K);
    run_packed(w, M, C, gelu, [&](uint8_t* hi, uint8_t* lo, int r0, int rows, int KC) { pack_x(hi, lo, A, w.K, r0, rows, KC, tmp_.data()); });
}

void qwen3_asr_audio_gemm_npu::run_conv(int weight, const float* in, int count, int H, int W, int cin, int K, float* C, bool gelu) {
    if (cin % 16 || K != weights_.at(weight)->K || K < 9 * cin) return qwen3_asr_audio_gemm::run_conv(weight, in, count, H, W, cin, K, C, gelu);
    std::lock_guard<std::mutex> lock(mu_);
    weight_t& w = *weights_.at(weight);
    constexpr size_t BLOCK = seqg::X_BLOCK_WORDS * 4;
    const int cb = cin / 8, Wo = (W + 1) / 2, per = (H + 1) / 2 * Wo;   // blocks per pixel, output pixels per image
    const size_t npix = (size_t)count * H * W, pb = (size_t)cb * 9;     // bytes of a pixel's blocks
    if (px_hi_.size() < npix * pb + pb) px_hi_.resize(npix * pb + pb);   // + one zero pixel (the padding)
    if (w.split && px_lo_.size() < npix * pb + pb) px_lo_.resize(npix * pb + pb);
    uint8_t* lo_px = w.split ? px_lo_.data() : nullptr;
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
    for (size_t p = 0; p <= npix; p++) {
        uint8_t* h = px_hi_.data() + p * pb;
        uint8_t* l = lo_px ? lo_px + p * pb : nullptr;
        if (p == npix) {
            std::memset(h, 0, pb);
            if (l) std::memset(l, 0, pb);
            continue;
        }
        for (int b = 0; b < cb; b += 2) {
            float rest[16] = {};
            encode2(h + b * 9, h + (b + 1) * 9, in + p * cin + b * 8, l ? rest : nullptr);
            if (l) encode2(l + b * 9, l + (b + 1) * 9, rest, nullptr);
        }
    }
    // a row's 9 taps (kh, kw) are blocks [tap cb, tap cb + cb); blocks from 9 cb on are zero
    run_packed(w, count * per, C, gelu, [&](uint8_t* hi, uint8_t* lo, int r0, int rows, int KC) {
        const int k8_end = 9 * cb, kc_tail = k8_end / 16;
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads()) schedule(dynamic)
        for (int rb = 0; rb < (rows + 15) / 16; rb++) {
            const bool full = rb * 16 + 16 <= rows;
            for (int s = 0; s < (lo ? 2 : 1); s++) {
                uint8_t* dst = (s ? lo : hi) + (size_t)rb * KC * BLOCK;
                const uint8_t* src = s ? lo_px : px_hi_.data();
                const int kc0 = full ? kc_tail : 0;
                std::memset(dst + (size_t)kc0 * BLOCK, 0, (size_t)(KC - kc0) * BLOCK);
                for (int i = 0; i < 16 && rb * 16 + i < rows; i++) {
                    const int r = r0 + rb * 16 + i, img = r / per, ho = r % per / Wo, wo = r % per % Wo;
                    uint8_t* d = dst + (i / 8) * 72 + (i % 8) * 9;
                    for (int tap = 0; tap < 9; tap++) {
                        const int y = 2 * ho - 1 + tap / 3, x = 2 * wo - 1 + tap % 3;
                        const size_t pix = (y < 0 || y >= H || x < 0 || x >= W) ? npix : ((size_t)img * H + y) * W + x;
                        const uint8_t* sp = src + pix * pb;
                        for (int b = 0; b < cb; b++) {
                            const int k8 = tap * cb + b;
                            std::memcpy(d + (size_t)(k8 / 16) * BLOCK + (k8 % 16) * 144, sp + b * 9, 9);
                        }
                    }
                }
            }
        }
    });
}

void qwen3_asr_audio_gemm_npu::run_packed(weight_t& w, int M, float* C, bool gelu, const pack_fn& pack) {
    const uint32_t KC = w.k_pad / 128, NB = w.n_pad / 1024;
    const bool stream = KC > seqg::KC_SINGLE;
    // split: A's rest goes as more rows of the same call (right after A's rows, from a 16-row boundary), so the weights are
    // read once for both
    const int parts = w.split ? 2 : 1;
    const int cap = (int)(stream ? std::min(seqg::MAX_ROUNDS, seqg::MAX_QUEUE_REPEAT / KC) : seqg::MAX_ROUNDS) * 256 / parts;
    const int calls = (M + cap - 1) / cap;
    const int per = ((M + calls - 1) / calls + 15) / 16 * 16;   // rows of A per call, spread evenly
    static const bool prof = std::getenv("QWEN3_ASR_GEMM_PROF") != nullptr;
    using clk = std::chrono::steady_clock;
    for (int m0 = 0; m0 < M; m0 += per) {
        const int rows = std::min(per, M - m0), rows16 = (rows + 15) / 16 * 16, R = (parts * rows16 + 255) / 256;
        const size_t xb = seqg::x_bytes(rows16, w.k_pad), xr = seqg::x_bytes(R * 256, w.k_pad), ye = (size_t)rows16 * w.n_pad;
        grow(xr, (size_t)R * 256 * w.n_pad, w.K);
        const auto t0 = clk::now();
        pack(x_.data(), w.split ? x_.data() + xb : nullptr, m0, rows, KC);
        std::memset(x_.data() + parts * xb, 0, xr - parts * xb);
        x_.sync_to_device_bytes(xr);
        const auto t1 = clk::now();
        app_for((uint32_t)R, KC, NB)(y_, x_, w.w);
        const auto t2 = clk::now();
        y_.sync_from_device_bytes(parts * ye * sizeof(u16));
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
        for (int r = 0; r < rows; r++) {
            const uint16_t* yh = y_.data() + (size_t)r * w.n_pad;
            const uint16_t* yl = yh + ye;
            float* out = C + (size_t)(m0 + r) * w.N;
            for (int n = 0; n < w.N; n += 16) {
                __m512 v = simd::load_bf16(yh + n);
                // the rests, scaled by LO_SCALE: A times W's (columns N..), A's times W (more rows, plus the bias)
                __m512 rest = _mm512_setzero_ps();
                if (w.split_w) rest = simd::load_bf16(yh + w.N + n);
                if (w.split) rest = _mm512_add_ps(rest, _mm512_sub_ps(simd::load_bf16(yl + n), _mm512_loadu_ps(w.bias.data() + n)));
                if (w.split || w.split_w) v = simd::round_bf16(_mm512_fmadd_ps(rest, _mm512_set1_ps(1.f / LO_SCALE), v));
                if (gelu) v = simd::round_bf16(simd::gelu_erf(v));
                _mm512_storeu_ps(out + n, v);
            }
        }
        if (prof) {
            const auto t3 = clk::now();
            auto ms = [](clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
            std::fprintf(stderr, "[gemm] M %d K %d N %d R %d x%d: pack %.2f npu %.2f unpack %.2f ms\n", M, w.K, w.N, R, parts, ms(t1 - t0),
                         ms(t2 - t1), ms(t3 - t2));
        }
    }
}
