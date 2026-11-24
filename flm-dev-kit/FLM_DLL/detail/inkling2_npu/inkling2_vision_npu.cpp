// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_vision_npu.cpp
/// \brief see inkling2_vision_npu.hpp
#include "inkling2_vision_npu.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "inkling2_vision_seq.hpp"
#include "npu_sequences/mmRuntimeSequence.hpp"

namespace inkling2_vision {

namespace {

constexpr int MM_M = 128, MM_K = 512, MM_N = 64;
constexpr int MM_ROWS = 4, MM_COLS = 8;
constexpr uint32_t RTP_ADDRESS = 4096, RTP_SYNC_LOCK = 10;
constexpr int M_PAD = MM_ROWS * MM_M;            // the kernel runs whole 4 x 128 row blocks
constexpr int PATCH_KP = 1536;                    // PATCH_K padded to the tile (OUT_DIM is 36 whole N tiles already)

inline int up(int v, int m) { return (v + m - 1) / m * m; }

inline float bf2f(u16 v) {
    const uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
inline u16 f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (u16)(u >> 16);
}

// ---------------------------------------------------------------------------------------------------------------------
// bfp16 pre-round. The mmul floors both operands onto a grid with one exponent per 8 consecutive values; rounding
// onto the same grid first (to nearest, clamped to the block's floored max) makes the floor a no-op.
// ---------------------------------------------------------------------------------------------------------------------

/// \brief one group of 8 bf16 values, in place
void preround_group(u16* blk) {
    uint16_t mxb = 0;
    for (int i = 0; i < 8; i++) mxb = std::max<uint16_t>(mxb, blk[i] & 0x7FFFu);
    const uint32_t be = mxb >> 7;
    if (be < 7u || be == 0xFFu) return;
    const uint32_t sb = (be - 6u) << 23, ib = (260u - be) << 23;
    float step, inv;
    memcpy(&step, &sb, 4);
    memcpy(&inv, &ib, 4);
    const float lim = std::truncf(bf2f(mxb) * inv) * step;
    for (int i = 0; i < 8; i++) {
        float q = std::nearbyintf(bf2f(blk[i]) * inv) * step;
        if (q > lim) q = lim;
        if (q < -lim - step) q = -lim - step;
        blk[i] = f2bf(q);
    }
}

/// 16 fp32 -> bf16 bit patterns, one per 32-bit lane (round to nearest even)
inline __m512i bf16_bits(__m512 v) {
    const __m512i u = _mm512_castps_si512(v);
    const __m512i trunc = _mm512_srli_epi32(u, 16);
    const __m512i lsb = _mm512_and_si512(trunc, _mm512_set1_epi32(1));
    __m512i r = _mm512_srli_epi32(_mm512_add_epi32(_mm512_add_epi32(u, _mm512_set1_epi32(0x7FFF)), lsb), 16);
    const __mmask16 zero_or_denorm = _mm512_fpclass_ps_mask(v, 0x02 | 0x04 | 0x20);
    return _mm512_mask_and_epi32(r, zero_or_denorm, trunc, _mm512_set1_epi32(0x8000));
}

/// two groups of 8 (bf16 patterns one per 32-bit lane), pre-rounded
inline __m512i preround16(__m512i b32) {
    const __m512i I1 = _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    const __m512i I2 = _mm512_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
    const __m512i I4 = _mm512_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);
    __m512i mx = _mm512_and_si512(b32, _mm512_set1_epi32(0x7FFF));
    mx = _mm512_max_epu32(mx, _mm512_permutexvar_epi32(I1, mx));
    mx = _mm512_max_epu32(mx, _mm512_permutexvar_epi32(I2, mx));
    mx = _mm512_max_epu32(mx, _mm512_permutexvar_epi32(I4, mx));
    const __m512i be = _mm512_srli_epi32(mx, 7);
    const __mmask16 skip = _mm512_cmplt_epu32_mask(be, _mm512_set1_epi32(7)) | _mm512_cmpeq_epu32_mask(be, _mm512_set1_epi32(0xFF));
    const __m512 step = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_sub_epi32(be, _mm512_set1_epi32(6)), 23));
    const __m512 inv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(260), be), 23));
    const __m512 fmax = _mm512_castsi512_ps(_mm512_slli_epi32(mx, 16));
    const __m512 lim = _mm512_mul_ps(_mm512_roundscale_ps(_mm512_mul_ps(fmax, inv), 0x0B), step);
    const __m512 v = _mm512_castsi512_ps(_mm512_slli_epi32(b32, 16));
    __m512 q = _mm512_mul_ps(_mm512_roundscale_ps(_mm512_mul_ps(v, inv), 0x08), step);
    q = _mm512_min_ps(q, lim);
    q = _mm512_max_ps(q, _mm512_sub_ps(_mm512_sub_ps(_mm512_setzero_ps(), lim), step));
    return _mm512_mask_blend_epi32(skip, bf16_bits(q), b32);
}

constexpr size_t CHUNK = 1u << 16;   // a multiple of 16: a chunk never splits a group

/// \brief bf16 range [0, n) (n a multiple of 8) onto the grid, in place
void preround_inplace(u16* w, size_t n) {
    const int64_t nchunk = (int64_t)((n + CHUNK - 1) / CHUNK);
#pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < nchunk; c++) {
        u16* p = w + (size_t)c * CHUNK;
        const size_t len = std::min(n - (size_t)c * CHUNK, CHUNK);
        size_t i = 0;
        for (; i + 16 <= len; i += 16) {
            const __m512i b = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(p + i)));
            _mm256_storeu_si256((__m256i*)(p + i), _mm512_cvtepi32_epi16(preround16(b)));
        }
        for (; i + 8 <= len; i += 8) preround_group(p + i);
    }
}

/// \brief dst = bf16(src) [n], optionally pre-rounded; n a multiple of 8
void narrow(u16* dst, const float* src, size_t n, bool round) {
    const int64_t nchunk = (int64_t)((n + CHUNK - 1) / CHUNK);
#pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < nchunk; c++) {
        const size_t i0 = (size_t)c * CHUNK, len = std::min(n - i0, CHUNK);
        size_t i = 0;
        for (; i + 16 <= len; i += 16) {
            __m512i b = bf16_bits(_mm512_loadu_ps(src + i0 + i));
            if (round) b = preround16(b);
            _mm256_storeu_si256((__m256i*)(dst + i0 + i), _mm512_cvtepi32_epi16(b));
        }
        for (; i < len; i++) dst[i0 + i] = f2bf(src[i0 + i]);
        if (round && (len % 16)) preround_group(dst + i0 + (len - len % 16));
    }
}

/// \brief dst[r][0, cols) = float(src[r][0, cols)) for rows of stride src_stride
void widen(float* dst, int dst_stride, const u16* src, int src_stride, int rows, int cols) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const u16* s = src + (size_t)r * src_stride;
        float* d = dst + (size_t)r * dst_stride;
        int i = 0;
        for (; i + 16 <= cols; i += 16)
            _mm512_storeu_ps(d + i, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(s + i))), 16)));
        for (; i < cols; i++) d[i] = bf2f(s[i]);
    }
}

/// \brief the bf16 rows of fc1 / proj_in after the kernel: gelu (erf) in place, then pre-rounded for the next product
void gelu_inplace(u16* a, size_t n, bool round) {
    const int64_t nchunk = (int64_t)((n + CHUNK - 1) / CHUNK);
#pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < nchunk; c++) {
        u16* p = a + (size_t)c * CHUNK;
        const size_t len = std::min(n - (size_t)c * CHUNK, CHUNK);
        for (size_t i = 0; i < len; i++) {
            const float x = bf2f(p[i]);
            p[i] = f2bf(0.5f * x * (1.f + std::erf(x * 0.70710678118654752f)));
        }
    }
    if (round) preround_inplace(a, n);
}

/// \brief rotate q and k (pairs (2i, 2i + 1) of every head) of the first L rows of a [.][3 * DIM] bf16 buffer
void rope_inplace(u16* qkv, int L, const float* cs, const float* sn) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < L; t++) {
        u16* row = qkv + (size_t)t * 3 * DIM;
        const float* c = cs + (size_t)t * (HDIM / 2);
        const float* s = sn + (size_t)t * (HDIM / 2);
        for (int part = 0; part < 2; part++)
            for (int hd = 0; hd < NHEAD; hd++) {
                u16* v = row + part * DIM + hd * HDIM;
                for (int i = 0; i < HDIM / 2; i++) {
                    const float a = bf2f(v[2 * i]), b = bf2f(v[2 * i + 1]);
                    v[2 * i] = f2bf(a * c[i] - b * s[i]);
                    v[2 * i + 1] = f2bf(b * c[i] + a * s[i]);
                }
            }
    }
}

using clk = std::chrono::steady_clock;
inline double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

// ---------------------------------------------------------------------------------------------------------------------

/// \brief one matrix product: its app (one per distinct shape, the sequence regenerated when M moves) and its resident weights
struct mm_op {
    npu_app app;
    buffer<bf16> w;       // [bias (Np)] [tiles (Np * Kp)]
    int Np = 0, Kp = 0, M = -1, mode = 0;
};

class npu_accel final : public accel {
public:
    npu_accel(npu_xclbin_manager* npu, const std::string& mm_xclbin, const std::string& attn_xclbin, const npu_options& o)
        : opt(o), mm(npu->register_xclbin(mm_xclbin)), at(npu->register_xclbin(attn_xclbin)) {
        if (!mm || !at) throw std::runtime_error("inkling2 vision: cannot register the vision xclbins");
        attn_app = at->create_app();
    }

    void bind(const mat& patch, const std::vector<layer_mats>& layers, const mat& proj_in, const mat& proj_out) override {
        make(op_patch, patch, 0);
        L_ops.resize(layers.size());
        for (size_t i = 0; i < layers.size(); i++) {
            make(L_ops[i].qkv, layers[i].qkv, 0);
            make(L_ops[i].out, layers[i].out, 0);
            make(L_ops[i].fc1, layers[i].fc1, opt.fused_gelu ? 1 : 0);
            make(L_ops[i].fc2, layers[i].fc2, 0);
        }
        make(op_in, proj_in, opt.fused_gelu ? 1 : 0);
        make(op_out, proj_out, 0);
    }

    void embed(const float* patches, int T, float* y) override {
        auto t0 = clk::now();
        const int Mp = up(T, M_PAD);
        stage(Mp);
        u16* x = xin();
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            narrow(x + (size_t)t * PATCH_KP, patches + (size_t)t * PATCH_K, PATCH_K, opt.preround);
            memset(x + (size_t)t * PATCH_KP + PATCH_K, 0, (PATCH_KP - PATCH_K) * sizeof(u16));
        }
        run(op_patch, Mp, b_x, b_y, t_embed, t0);
        widen(y, DIM, ptr(b_y), DIM, T, DIM);
    }

    void attention(int layer, const float* h, int L, const float* cos, const float* sin, float* delta) override {
        auto t0 = clk::now();
        const int Mp = up(L, M_PAD);
        stage(Mp);
        layer_ops& ly = L_ops[layer];
        narrow(xin(), h, (size_t)L * DIM, opt.preround);
        run(ly.qkv, Mp, b_x, b_qkv, t_mm, t0);
        auto t1 = clk::now();
        b_qkv.sync_from_device();
        rope_inplace(ptr(b_qkv), L, cos, sin);
        b_qkv.sync_to_device();
        t_host += ms_since(t1);
        t1 = clk::now();
        if (attn_L != L) {
            attn_app.seq()->clear_cmds();
            gen_attention_sequence(attn_app.seq(), L, NHEAD, HDIM);
            attn_L = L;
        }
        attn_app(b_ao, b_qkv);
        t_attn += ms_since(t1);
        t1 = clk::now();
        b_ao.sync_from_device();
        if (opt.preround) preround_inplace(ptr(b_ao), (size_t)L * DIM);
        b_ao.sync_to_device();
        t_host += ms_since(t1);
        run(ly.out, Mp, b_ao, b_y, t_mm, clk::now());
        t1 = clk::now();
        widen(delta, DIM, ptr(b_y), DIM, L, DIM);
        t_host += ms_since(t1);
    }

    void mlp(int layer, const float* h, int L, float* delta) override {
        const int Mp = up(L, M_PAD);
        stage(Mp);
        layer_ops& ly = L_ops[layer];
        auto t1 = clk::now();
        narrow(xin(), h, (size_t)L * DIM, opt.preround);
        t_host += ms_since(t1);
        run(ly.fc1, Mp, b_x, b_ff, t_mm, clk::now());
        t1 = clk::now();
        b_ff.sync_from_device();
        if (opt.fused_gelu) {
            if (opt.preround) preround_inplace(ptr(b_ff), (size_t)L * FFN);
        } else {
            gelu_inplace(ptr(b_ff), (size_t)L * FFN, opt.preround);
        }
        b_ff.sync_to_device();
        t_host += ms_since(t1);
        run(ly.fc2, Mp, b_ff, b_y, t_mm, clk::now());
        t1 = clk::now();
        widen(delta, DIM, ptr(b_y), DIM, L, DIM);
        t_host += ms_since(t1);
    }

    void merge(const float* rows, int N, float* out) override {
        const int Mp = up(N, M_PAD);
        stage(Mp);
        auto t1 = clk::now();
        narrow(ptr(b_ff), rows, (size_t)N * MERGED, opt.preround);
        t_host += ms_since(t1);
        run(op_in, Mp, b_ff, b_ff2, t_mm, clk::now());
        t1 = clk::now();
        b_ff2.sync_from_device();
        if (opt.fused_gelu) {
            if (opt.preround) preround_inplace(ptr(b_ff2), (size_t)N * MERGED);
        } else {
            gelu_inplace(ptr(b_ff2), (size_t)N * MERGED, opt.preround);
        }
        b_ff2.sync_to_device();
        t_host += ms_since(t1);
        run(op_out, Mp, b_ff2, b_y, t_mm, clk::now());
        t1 = clk::now();
        widen(out, OUT_DIM, ptr(b_y), OUT_DIM, N, OUT_DIM);
        t_host += ms_since(t1);
        if (opt.profile) {
            fprintf(stderr, "[vision npu] mm %.0f ms (dispatch + sync)  attention %.0f ms  host %.0f ms  embed %.0f ms\n", t_mm, t_attn, t_host, t_embed);
        }
        t_mm = t_attn = t_host = t_embed = 0;
    }

private:
    struct layer_ops {
        mm_op qkv, out, fc1, fc2;
    };

    static u16* ptr(buffer<bf16>& b) { return reinterpret_cast<u16*>(b.data()); }
    u16* xin() { return ptr(b_x); }

    /// \brief make the app and the resident weight buffer of one matrix (padded with zeros to whole tiles)
    void make(mm_op& op, const mat& m, int mode) {
        op.Np = up(m.N, MM_N);
        op.Kp = up(m.K, MM_K);
        op.mode = mode;
        op.app = mm->create_app();
        op.w = mm->create_bo_buffer<bf16>((size_t)op.Np * (1 + op.Kp));
        u16* w = ptr(op.w);
        for (int i = 0; i < op.Np; i++) w[i] = (m.bias && i < m.N) ? f2bf(m.bias[i]) : (u16)0;
        u16* tiles = w + op.Np;
        memcpy(tiles, m.tiles, (size_t)op.Np * op.Kp * sizeof(u16));
        if (opt.preround) preround_inplace(tiles, (size_t)op.Np * op.Kp);
        op.w.sync_to_device();
    }

    /// \brief size the staging buffers for Mp rows (grow only)
    void stage(int Mp) {
        if (Mp <= staged) return;
        auto mk = [&](buffer<bf16>& b, size_t n) {
            b = mm->create_bo_buffer<bf16>(n);
            b.memset((bf16)0);
            b.sync_to_device();
        };
        mk(b_x, (size_t)Mp * PATCH_KP);
        mk(b_qkv, (size_t)Mp * 3 * DIM);
        mk(b_ao, (size_t)Mp * DIM);
        mk(b_ff, (size_t)Mp * FFN);
        mk(b_ff2, (size_t)Mp * FFN);
        mk(b_y, (size_t)Mp * FFN);
        staged = Mp;
    }

    /// \brief y = x W^T + bias for Mp rows (x already in the device buffer or flushed here)
    void run(mm_op& op, int Mp, buffer<bf16>& x, buffer<bf16>& y, double& acc, clk::time_point t0) {
        if (op.M != Mp) {
            op.app.seq()->clear_cmds();
            generate_mm_sequence<bf16, bf16>(*op.app.seq(), Mp, op.Kp, op.Np, MM_M, MM_K, MM_N, 8, 8, 8, RTP_ADDRESS, RTP_SYNC_LOCK,
                                             MM_ROWS, MM_COLS, 0, 0, 0, false, true, true, true, op.mode, false, 0);
            op.M = Mp;
        }
        x.sync_to_device();
        op.app(x, op.w, y);
        y.sync_from_device();
        acc += ms_since(t0);
    }

    npu_options opt;
    npu_app_manager* mm;
    npu_app_manager* at;
    npu_app attn_app;
    int attn_L = -1, staged = 0;
    mm_op op_patch, op_in, op_out;
    std::vector<layer_ops> L_ops;
    buffer<bf16> b_x, b_qkv, b_ao, b_ff, b_ff2, b_y;
    double t_mm = 0, t_attn = 0, t_host = 0, t_embed = 0;
};

}  // namespace

std::unique_ptr<accel> make_npu_accel(npu_xclbin_manager* npu, const std::string& mm_xclbin, const std::string& attn_xclbin,
                                      const npu_options& opt) {
    return std::make_unique<npu_accel>(npu, mm_xclbin, attn_xclbin, opt);
}

}  // namespace inkling2_vision
