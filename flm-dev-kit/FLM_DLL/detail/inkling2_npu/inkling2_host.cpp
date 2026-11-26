// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_host.cpp
/// \brief Inkling-2 on the host (see inkling2_host.hpp)
#include "inkling2_host.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <immintrin.h>
#if !defined(_WIN32)
#include <sys/mman.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "inkling2_q4nx_file.hpp"
#include "nlohmann/json.hpp"
#include "tensor_utils/mapped_file.hpp"

namespace inkling2_host {

std::function<void(int, const char*, const float*, int)> decode_trace;

// =============================================================================
// small helpers
// =============================================================================
namespace {

inline float bf2f(u16 v) {
    const uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/// \brief fp32 -> bf16, round to nearest even (values here are finite)
inline u16 f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (u16)(u >> 16);
}

/// \brief n % 16 == 0
inline float dot(const float* a, const float* b, int n) {
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    return _mm512_reduce_add_ps(acc);
}

inline float dot_bf16(const u16* a, const float* b, int n) {
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) acc = _mm512_fmadd_ps(load_bf16x16(a + i), _mm512_loadu_ps(b + i), acc);
    return _mm512_reduce_add_ps(acc);
}

/// \brief y = x / rms(x) [* g]; y may alias x
inline void rms_norm(float* y, const float* x, const float* g, int n) {
    const float inv = 1.0f / std::sqrt(dot(x, x, n) / n + EPS);
    if (g) {
        for (int i = 0; i < n; i++) y[i] = x[i] * inv * g[i];
    } else {
        for (int i = 0; i < n; i++) y[i] = x[i] * inv;
    }
}

/// \brief scale x so that its L2 norm is at most thr (the norm gets +1e-6)
inline void l2_clip(float* x, int n, float thr) {
    const float norm = std::sqrt(dot(x, x, n)) + 1e-6f;
    const float f = std::min(std::max(thr / norm, 0.0f), 1.0f);
    for (int i = 0; i < n; i++) x[i] *= f;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/// \brief one thread per physical core (logical / 2) unless OMP_NUM_THREADS is set: SMT siblings compete for the
///        same load ports, 32 threads decode ~25% slower than 16 on a 16-core part. Set for the calling thread only
///        (nthreads-var is per thread), not for other engines.
/// \note  libgomp spins between parallel regions only while it manages no more threads than there are logical
///        CPUs; past that its waiters sleep on a futex after a short spin. The decode wants the long spin (a token's
///        hundreds of short regions would pay a wake-up each: ~80 instead of ~59 ms per token), a prompt the short
///        one (spinning waiters take the cores the NPU glue and the other half compute on: 2k tokens ~830 instead
///        of ~1100 tokens/s, 137 tokens ~155 instead of ~185). A pool lives as long as its thread and never
///        shrinks, so the engine loads on a thread that exits, and a prompt starts the helper with a pool of every
///        logical CPU (wide), which pushes libgomp past the limit for the prompt only: the helper exits after it.
///        MSVC's runtime (vcomp) has no such limit and spins whatever the count, so on Windows a wide pool would only
///        put 1.5 threads on every core (the helper's plus the caller's half): the helper stays at one per core.
void use_core_threads(bool wide = false) {
#ifdef _WIN32
    wide = false;
#endif
#ifdef _OPENMP
    static const int env = [] { return std::getenv("OMP_NUM_THREADS") ? omp_get_max_threads() : 0; }();
    const int all = std::max(1, (int)std::thread::hardware_concurrency());
    omp_set_num_threads(env ? env : wide ? all : std::max(1, all / 2));
#endif
}

int n_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/// \brief the activations' allocator: no zero fill (every caller writes the whole buffer), and blocks of 2 MB and
///        more on huge pages where the kernel offers them (THP madvise): a 2k-token activation is ~4600 page faults
///        on first touch in 4 KB pages, 9 in 2 MB ones (the first prompt allocates ~40 of them)
template <typename T>
struct act_alloc {
    using value_type = T;
    static constexpr size_t HUGE_PAGE = (size_t)2 << 20;
    act_alloc() = default;
    template <typename U>
    act_alloc(const act_alloc<U>&) {}
    T* allocate(size_t n) {
        const size_t bytes = n * sizeof(T);
#if !defined(_WIN32)
        if (bytes >= HUGE_PAGE) {
            const size_t size = (bytes + HUGE_PAGE - 1) / HUGE_PAGE * HUGE_PAGE;
            void* p = std::aligned_alloc(HUGE_PAGE, size);
            if (!p) throw std::bad_alloc();
            madvise(p, size, MADV_HUGEPAGE);
            return static_cast<T*>(p);
        }
#endif
        return static_cast<T*>(::operator new(bytes));
    }
    void deallocate(T* p, size_t n) {
#if !defined(_WIN32)
        if (n * sizeof(T) >= HUGE_PAGE) return std::free(p);
#else
        (void)n;
#endif
        ::operator delete(p);
    }
    template <typename U>
    void construct(U* p) noexcept {
        ::new ((void*)p) U;   // default-initialized: no zero fill
    }
    template <typename U, typename... A>
    void construct(U* p, A&&... a) {
        ::new ((void*)p) U(std::forward<A>(a)...);
    }
    template <typename U>
    bool operator==(const act_alloc<U>&) const { return true; }
    template <typename U>
    bool operator!=(const act_alloc<U>&) const { return false; }
};
using act_vec = std::vector<float, act_alloc<float>>;
using act = std::shared_ptr<act_vec>;
// activations come from a pool: a fresh multi-MB vector page-faults on every layer. Contents are undefined.
std::mutex& act_mutex() {
    static std::mutex m;
    return m;
}
std::multimap<size_t, act_vec*>& act_pool() {
    static std::multimap<size_t, act_vec*> p;
    return p;
}
act make_act(size_t n) {
    act_vec* v = nullptr;
    {
        std::lock_guard<std::mutex> lock(act_mutex());
        auto it = act_pool().find(n);
        if (it != act_pool().end()) {
            v = it->second;
            act_pool().erase(it);
        }
    }
    if (!v) v = new act_vec(n);
    return act(v, [](act_vec* p) {
        std::lock_guard<std::mutex> lock(act_mutex());
        act_pool().emplace(p->size(), p);
    });
}

/// INKLING2_PROF=1: time per part of the model, printed at exit
enum prof_part { P_MIX, P_CTRL, P_NORM, P_PROJ, P_CONV, P_ATTN, P_OPROJ, P_ROUTER, P_EXPERT, P_LMHEAD, P_X_SINK, P_X_CLIP, P_X_KVW, P_X_GATE, P_X_RES, P_X_ATTNPU, P_X_MSGMIX, P_X_FFNMIX, P_X_FFNCTRL, P_NPART };
struct profiler {
    const bool on = std::getenv("INKLING2_PROF") != nullptr;
    double sec[P_NPART] = {};
    double hsec[2][P_NPART] = {};   // DIAG
    ~profiler() {
        if (!on) return;
        static const char* names[P_NPART] = {"mix", "loop control", "norms+clips", "qkv/rel/gate/sink proj", "convs", "attention",
                                             "o proj", "router", "experts", "lm_head", "x sink+convproj host", "x clips", "x kv write", "x gate sigmoid", "x residual", "x attn npu", "x msg mix", "x ffn mix", "x ffn ctrl"};
        double total = 0;
        for (double s : sec) total += s;
        for (int i = 0; i < P_NPART; i++)
            std::fprintf(stderr, "inkling2 prof %-24s %8.3f s %5.1f%%\n", names[i], sec[i], 100 * sec[i] / std::max(total, 1e-9));
        // DIAG: the two halves of a split prompt layer, per part (each on its own thread)
        for (int h = 0; h < 2; h++)
            for (int i = 0; i < P_NPART; i++)
                if (hsec[h][i] > 0) std::fprintf(stderr, "inkling2 prof half%d %-24s %8.3f s\n", h, names[i], hsec[h][i]);
    }
} prof;
/// \brief add the time since `t` to `part`, restart `t`
void prof_lap(std::chrono::steady_clock::time_point& t, prof_part part) {
    if (!prof.on) return;
    const auto now = std::chrono::steady_clock::now();
    prof.sec[part] += std::chrono::duration<double>(now - t).count();
    t = now;
}
/// DIAG: INKLING2_HALF_TRACE=<layer>: the steps of that layer's halves (steady clock ns)
int half_trace() {
    static const int v = [] { const char* e = std::getenv("INKLING2_HALF_TRACE"); return e ? std::atoi(e) : -1; }();
    return v;
}
long long trace_ns(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}
/// DIAG: the same for one half of a split prompt layer
void prof_lap_half(std::chrono::steady_clock::time_point& t, prof_part part, int h) {
    if (!prof.on) return;
    const auto now = std::chrono::steady_clock::now();
    prof.hsec[h][part] += std::chrono::duration<double>(now - t).count();
    t = now;
}

}  // namespace

// =============================================================================
// quantized matmul: y[T][N] = x[T][K] W^T
// =============================================================================
// A 32-row block of W is decoded into a column-major fp32 tile [K][32] (one
// column = the 32 rows' values for one input channel), then each token is an
// outer-product walk over the columns: no horizontal sums, and the tile is
// read once per 8 tokens.
namespace {

/// \brief one q4nx block (32 rows x 256 columns) into tile[(col0 + c) * 32 + r]
void decode_block(float* tile, const u8* blk, qfmt fmt) {
    __m512 sc[2], mn[2];
    if (fmt == qfmt::q4k) {
        // s8[8][32], m8[8][32], q[256][16], bf16 S[32], bf16 M[32] (stored negated)
        const u8* s8 = blk;
        const u8* m8 = blk + 256;
        const u8* qs = blk + 512;
        const u16* S = reinterpret_cast<const u16*>(blk + 4608);
        const u16* M = reinterpret_cast<const u16*>(blk + 4672);
        const __m512 S0 = load_bf16x16(S), S1 = load_bf16x16(S + 16);
        const __m512 M0 = load_bf16x16(M), M1 = load_bf16x16(M + 16);
        for (int g = 0; g < 8; g++) {
            const __m128i* sp = reinterpret_cast<const __m128i*>(s8 + g * 32);
            const __m128i* mp = reinterpret_cast<const __m128i*>(m8 + g * 32);
            sc[0] = _mm512_mul_ps(S0, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(sp))));
            sc[1] = _mm512_mul_ps(S1, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(sp + 1))));
            mn[0] = _mm512_mul_ps(M0, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(mp))));
            mn[1] = _mm512_mul_ps(M1, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(mp + 1))));
            for (int c = g * 32; c < g * 32 + 32; c++) {
                const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + c * 16));
                const __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
                const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
                // byte k of a column: rows 2k (low nibble), 2k + 1 (high)
                const __m512 q0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi)));
                const __m512 q1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(lo, hi)));
                _mm512_storeu_ps(tile + c * 32, _mm512_fmadd_ps(q0, sc[0], mn[0]));
                _mm512_storeu_ps(tile + c * 32 + 16, _mm512_fmadd_ps(q1, sc[1], mn[1]));
            }
        }
    } else if (fmt == qfmt::q41) {
        // bf16 s[8][32], bf16 m[8][32], q[256][16]
        const u16* s = reinterpret_cast<const u16*>(blk);
        const u16* m = reinterpret_cast<const u16*>(blk + 512);
        const u8* qs = blk + 1024;
        for (int g = 0; g < 8; g++) {
            sc[0] = load_bf16x16(s + g * 32);
            sc[1] = load_bf16x16(s + g * 32 + 16);
            mn[0] = load_bf16x16(m + g * 32);
            mn[1] = load_bf16x16(m + g * 32 + 16);
            for (int c = g * 32; c < g * 32 + 32; c++) {
                const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + c * 16));
                const __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
                const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
                const __m512 q0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi)));
                const __m512 q1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(lo, hi)));
                _mm512_storeu_ps(tile + c * 32, _mm512_fmadd_ps(q0, sc[0], mn[0]));
                _mm512_storeu_ps(tile + c * 32 + 16, _mm512_fmadd_ps(q1, sc[1], mn[1]));
            }
        }
    } else {
        // bf16 d[8][32], int8 q[256][32]
        const u16* d = reinterpret_cast<const u16*>(blk);
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 512);
        for (int g = 0; g < 8; g++) {
            sc[0] = load_bf16x16(d + g * 32);
            sc[1] = load_bf16x16(d + g * 32 + 16);
            for (int c = g * 32; c < g * 32 + 32; c++) {
                const __m128i* p = reinterpret_cast<const __m128i*>(qs + c * 32);
                const __m512 q0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p)));
                const __m512 q1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p + 1)));
                _mm512_storeu_ps(tile + c * 32, _mm512_mul_ps(q0, sc[0]));
                _mm512_storeu_ps(tile + c * 32 + 16, _mm512_mul_ps(q1, sc[1]));
            }
        }
    }
}

/// \brief one Q4_K block in q4k_to_npu order into tile[(col0 + c) * 32 + r], the same values as decode_block
void decode_block_npu(float* tile, const u8* blk) {
    alignas(64) u8 orig[Q4K_BYTES];
    q4k_from_npu(orig, blk);
    decode_block(tile, orig, qfmt::q4k);
}

/// \brief y[t][rb rows] = x[t] . W for NT tokens straight from a row block of q4k_to_npu blocks: per 8-row quarter
///        hp (row 16 H + 2 r + par) the nibbles [r][k] of 8 columns are 4 vectors of 2 rows x 8 columns, dequantized in
///        registers and multiplied by x[8 kt .. 8 kt + 8] twice; no reshuffle, no fp32 tile
constexpr int NPU_DIRECT_T = 8;   // up to this many tokens qmm reads NPU-order blocks directly

template <int NT>
void mm_npu_rows(float* y, int ldy, const qmat& w, int rb, const float* x, int ldx) {
    const __m512i dup[4] = {_mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
                            _mm512_set_epi32(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2),
                            _mm512_set_epi32(5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4),
                            _mm512_set_epi32(7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6)};
    const __m128i low4 = _mm_set1_epi8(0x0F);
    for (int hp = 0; hp < 4; hp++) {
        __m512 acc[NT][4];
        for (int t = 0; t < NT; t++)
            for (int z = 0; z < 4; z++) acc[t][z] = _mm512_setzero_ps();
        for (int cb = 0; cb < w.col_blocks(); cb++) {
            const u8* blk = w.block(rb, cb);
            const __m512 S = _mm512_castps256_ps512(_mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 4608 + 16 * hp))), 16)));
            const __m512 M = _mm512_castps256_ps512(_mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 4672 + 16 * hp))), 16)));
            const u8* q = blk + 512 + 2048 * (hp >> 1) + 1024 * (hp & 1);
            const float* xc = x + (size_t)cb * QCOLS;
            for (int g = 0; g < 8; g++) {
                const __m512 s8 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(blk + 64 * hp + 8 * g))));
                const __m512 m8 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(blk + 256 + 64 * hp + 8 * g))));
                const __m512 scr = _mm512_mul_ps(S, s8), mnr = _mm512_mul_ps(M, m8);   // lanes 0-7: rows r
                __m512 sc[4], mn[4];
                for (int z = 0; z < 4; z++) {
                    sc[z] = _mm512_permutexvar_ps(dup[z], scr);
                    mn[z] = _mm512_permutexvar_ps(dup[z], mnr);
                }
                for (int kt = 4 * g; kt < 4 * g + 4; kt++) {
                    __m512 wv[4];
                    for (int hf = 0; hf < 2; hf++) {
                        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 32 * kt + 16 * hf));
                        const __m128i lo = _mm_and_si128(b, low4), hi = _mm_and_si128(_mm_srli_epi16(b, 4), low4);
                        // nibble i = r * 8 + k, low nibble first
                        wv[2 * hf] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi))), sc[2 * hf], mn[2 * hf]);
                        wv[2 * hf + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(lo, hi))), sc[2 * hf + 1], mn[2 * hf + 1]);
                    }
                    for (int t = 0; t < NT; t++) {
                        const __m512 xv = _mm512_broadcast_f32x8(_mm256_loadu_ps(xc + (size_t)t * ldx + 8 * kt));
                        for (int z = 0; z < 4; z++) acc[t][z] = _mm512_fmadd_ps(wv[z], xv, acc[t][z]);
                    }
                }
            }
        }
        for (int t = 0; t < NT; t++)
            for (int z = 0; z < 4; z++) {
                // the 256-bit halves are rows 2 z and 2 z + 1
                float* yr = y + (size_t)t * ldy + (size_t)rb * QROWS + 16 * (hp >> 1) + (hp & 1);
                yr[4 * z] = _mm512_mask_reduce_add_ps(0x00FF, acc[t][z]);
                yr[4 * z + 2] = _mm512_mask_reduce_add_ps(0xFF00, acc[t][z]);
            }
    }
}

/// \brief the grouped Q4_K row block rb of w for T tokens, NT at a time
void mm_npu(float* y, int ldy, const qmat& w, int rb, const float* x, int ldx, int T) {
    int t = 0;
    for (; t + 4 <= T; t += 4) mm_npu_rows<4>(y + (size_t)t * ldy, ldy, w, rb, x + (size_t)t * ldx, ldx);
    switch (T - t) {
        case 3: mm_npu_rows<3>(y + (size_t)t * ldy, ldy, w, rb, x + (size_t)t * ldx, ldx); break;
        case 2: mm_npu_rows<2>(y + (size_t)t * ldy, ldy, w, rb, x + (size_t)t * ldx, ldx); break;
        case 1: mm_npu_rows<1>(y + (size_t)t * ldy, ldy, w, rb, x + (size_t)t * ldx, ldx); break;
        default: break;
    }
}

/// \brief y[i][0 .. 31] = x[i] . tile, NT tokens
template <int NT>
inline void mm_rows(float* y, int ldy, const float* tile, int K, const float* x, int ldx) {
    __m512 acc[NT][2];
    for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = _mm512_setzero_ps();
    for (int c = 0; c < K; c++) {
        const __m512 w0 = _mm512_loadu_ps(tile + c * 32);
        const __m512 w1 = _mm512_loadu_ps(tile + c * 32 + 16);
        for (int i = 0; i < NT; i++) {
            const __m512 xb = _mm512_set1_ps(x[(size_t)i * ldx + c]);
            acc[i][0] = _mm512_fmadd_ps(xb, w0, acc[i][0]);
            acc[i][1] = _mm512_fmadd_ps(xb, w1, acc[i][1]);
        }
    }
    for (int i = 0; i < NT; i++) {
        _mm512_storeu_ps(y + (size_t)i * ldy, acc[i][0]);
        _mm512_storeu_ps(y + (size_t)i * ldy + 16, acc[i][1]);
    }
}

void mm_tile(float* y, int ldy, const float* tile, int K, const float* x, int ldx, int T) {
    int t = 0;
    for (; t + 8 <= T; t += 8) mm_rows<8>(y + (size_t)t * ldy, ldy, tile, K, x + (size_t)t * ldx, ldx);
    float* yt = y + (size_t)t * ldy;
    const float* xt = x + (size_t)t * ldx;
    switch (T - t) {
        case 7: mm_rows<7>(yt, ldy, tile, K, xt, ldx); break;
        case 6: mm_rows<6>(yt, ldy, tile, K, xt, ldx); break;
        case 5: mm_rows<5>(yt, ldy, tile, K, xt, ldx); break;
        case 4: mm_rows<4>(yt, ldy, tile, K, xt, ldx); break;
        case 3: mm_rows<3>(yt, ldy, tile, K, xt, ldx); break;
        case 2: mm_rows<2>(yt, ldy, tile, K, xt, ldx); break;
        case 1: mm_rows<1>(yt, ldy, tile, K, xt, ldx); break;
        default: break;
    }
}

/// \brief one matmul of a batch: y[T][w.N] (row stride ldy) = x[T][w.K] (row stride ldx) W^T
struct mm_job {
    qmat w;
    const float* x;
    int ldx;
    int T;
    float* y;
    int ldy;
};

/// \brief all jobs in one parallel region, one work item per 32-row block
void qmm(const std::vector<mm_job>& jobs) {
    std::vector<std::pair<int, int>> work;
    int kmax = 0;   // the widest job that needs a dequantized tile
    // NPU-order blocks, few tokens (decode, short prompts): straight from the blocks, no tile
    auto direct = [](const mm_job& jb) { return jb.w.grouped && jb.w.fmt == qfmt::q4k && jb.T <= NPU_DIRECT_T; };
    for (int j = 0; j < (int)jobs.size(); j++) {
        if (jobs[j].T <= 0) continue;
        if (!direct(jobs[j])) kmax = std::max(kmax, jobs[j].w.K);
        for (int rb = 0; rb < jobs[j].w.row_blocks(); rb++) work.emplace_back(j, rb);
    }
    if (work.empty()) return;
    #pragma omp parallel
    {
        // each thread's tile lives as long as the thread: a fresh one per call (up to 384 KB, zero-filled, a
        // decode step makes hundreds of calls) costs page faults and a heap lock on Windows
        static thread_local std::vector<float> tile;
        if (tile.size() < (size_t)kmax * QROWS) tile.resize((size_t)kmax * QROWS);
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < (int)work.size(); i++) {
            const mm_job& jb = jobs[work[i].first];
            const int rb = work[i].second;
            if (direct(jb)) {
                mm_npu(jb.y, jb.ldy, jb.w, rb, jb.x, jb.ldx, jb.T);
                continue;
            }
            for (int cb = 0; cb < jb.w.col_blocks(); cb++)
                if (jb.w.grouped && jb.w.fmt == qfmt::q4k)
                    decode_block_npu(tile.data() + (size_t)cb * QCOLS * QROWS, jb.w.block(rb, cb));
                else
                    decode_block(tile.data() + (size_t)cb * QCOLS * QROWS, jb.w.block(rb, cb), jb.w.fmt);
            mm_tile(jb.y + (size_t)rb * QROWS, jb.ldy, tile.data(), jb.w.K, jb.x, jb.ldx, jb.T);
        }
    }
}

inline __m512 load16(const float* p) { return _mm512_loadu_ps(p); }
inline __m512 load16(const u16* p) { return load_bf16x16(p); }

/// \brief y[t * ldy + n] = x[t * ldx ..] . w[n * ldw ..] over K (a multiple of 16) for t < T, n < N; tiles of
///        6 tokens x 4 rows (24 accumulators + 4 weight vectors + 1 token vector of the 32 registers), so each
///        weight row is read once per 6 tokens; the router at 2k tokens: 0.61 vs 0.70 ms for 4 x 4
template <typename W>
void gemm_nt(float* y, int ldy, const float* x, int ldx, const W* w, int ldw, int T, int N, int K) {
    constexpr int TB = 6, NB = 4;
    const int tb = (T + TB - 1) / TB, nb = (N + NB - 1) / NB;
    // one flat loop, not collapse(2): MSVC's OpenMP 2.0 ignores collapse and would split only the tb tiles
    #pragma omp parallel for schedule(static)
    for (int64_t ib = 0; ib < (int64_t)tb * nb; ib++) {
        const int t0 = (int)(ib / nb) * TB, n0 = (int)(ib % nb) * NB;
        const float* xp[TB];
        const W* wp[NB];
        for (int i = 0; i < TB; i++) xp[i] = x + (size_t)std::min(t0 + i, T - 1) * ldx;
        for (int j = 0; j < NB; j++) wp[j] = w + (size_t)std::min(n0 + j, N - 1) * ldw;
        __m512 acc[TB][NB];
        for (int i = 0; i < TB; i++)
            for (int j = 0; j < NB; j++) acc[i][j] = _mm512_setzero_ps();
        for (int k = 0; k < K; k += 16) {
            __m512 wv[NB];
            for (int j = 0; j < NB; j++) wv[j] = load16(wp[j] + k);
            for (int i = 0; i < TB; i++) {
                const __m512 xv = _mm512_loadu_ps(xp[i] + k);
                for (int j = 0; j < NB; j++) acc[i][j] = _mm512_fmadd_ps(xv, wv[j], acc[i][j]);
            }
        }
        for (int i = 0; i < TB && t0 + i < T; i++)
            for (int j = 0; j < NB && n0 + j < N; j++) y[(size_t)(t0 + i) * ldy + n0 + j] = _mm512_reduce_add_ps(acc[i][j]);
    }
}

/// \brief y[T][N] = x[T][K] W^T for a bf16 or fp32 [N][K] matrix (the small projections)
template <typename W>
void dense_mm(float* y, const W* w, int N, int K, const float* x, int T) {
    if (T >= 16 && K % 16 == 0) return gemm_nt(y, N, x, K, w, K, T, N, K);
    // flat (gemm_nt): over t alone, a decode step (T = 1) would run on one thread under MSVC
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * N; i++) {
        const int t = (int)(i / N), n = (int)(i % N);
        if constexpr (std::is_same_v<W, u16>)
            y[i] = dot_bf16(w + (size_t)n * K, x + (size_t)t * K, K);
        else
            y[i] = dot(w + (size_t)n * K, x + (size_t)t * K, K);
    }
}

}  // namespace

// =============================================================================
// the sequence state
// =============================================================================
struct model::state {
    int max_len = 0;
    int pos = 0;
    // K and V per slot, bf16 [rows][KVD]: global slots max_len rows, SWA slots a ring of SWA_RING
    std::vector<std::vector<u16>> k, v;
    // the last NCONV - 1 convolution inputs per slot, [NSLOT][NCONV - 1][CONV_CH], oldest first
    std::vector<float> conv;

    int rows(int slot) const { return is_global(slot_layer(slot)) ? max_len : SWA_RING; }
    size_t row(int slot, int p) const { return (size_t)(is_global(slot_layer(slot)) ? p : p % SWA_RING) * KVD; }
    float* conv_state(int slot) { return conv.data() + (size_t)slot * (NCONV - 1) * CONV_CH; }
    mm_offload* offload = nullptr;   // the prompt matmuls elsewhere (not owned)
};

// =============================================================================
// the layer pieces
// =============================================================================
namespace {

/// \brief depth attention: softmax mixes of the RMS-normalized states {blocks, cur}
/// \param query [nq][D]
/// \param t0    first token (row) of the states to mix
/// \return [nq][T][D]
act mix(const std::vector<act>& blocks, const act& cur, const float* query, int nq, int T, int t0 = 0) {
    std::vector<const float*> states;
    for (const auto& b : blocks) states.push_back(b->data());
    states.push_back(cur->data());
    const int S = (int)states.size();
    if (S > 64) throw std::runtime_error("inkling2: more than 64 residual states");
    act out = make_act((size_t)nq * T * D);
    const float scale = 1.0f / std::sqrt((float)D);
    // one pass per token: rms(state) and query . state from the raw states, then the weighted sum of the
    // raw states with weight w / rms (no normalized copies)
    #pragma omp parallel for schedule(static) if (T >= 16)
    for (int t = 0; t < T; t++) {
        float inv[64], w[64];
        const float* x[64];
        for (int i = 0; i < S; i++) {
            x[i] = states[i] + (size_t)(t0 + t) * D;
            inv[i] = 1.0f / std::sqrt(dot(x[i], x[i], D) / D + EPS);
        }
        for (int q = 0; q < nq; q++) {
            const float* qv = query + (size_t)q * D;
            float mx = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < S; i++) {
                w[i] = dot(qv, x[i], D) * inv[i] * scale;
                mx = std::max(mx, w[i]);
            }
            float sum = 0;
            for (int i = 0; i < S; i++) sum += (w[i] = std::exp(w[i] - mx));
            for (int i = 0; i < S; i++) w[i] *= inv[i] / sum;
            float* o = out->data() + ((size_t)q * T + t) * D;
            for (int c = 0; c < D; c += 16) {
                __m512 acc = _mm512_mul_ps(_mm512_set1_ps(w[0]), _mm512_loadu_ps(x[0] + c));
                for (int i = 1; i < S; i++) acc = _mm512_fmadd_ps(_mm512_set1_ps(w[i]), _mm512_loadu_ps(x[i] + c), acc);
                _mm512_storeu_ps(o + c, acc);
            }
        }
    }
    return out;
}

/// \brief the loop controller's weights in bf16 (exact: the checkpoint stores them in bf16), once per controller
struct ctrl_bf16 {
    std::vector<u16> down;     // [RANK][2 D]
    std::vector<u16> f, a;     // forget, gain as [RANK / 2][D][2]: rank pairs side by side per channel
};

const ctrl_bf16& ctrl_weights(const loop_ctrl& c) {
    static std::mutex mu;
    static std::map<const float*, ctrl_bf16> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(c.down);
    if (it == cache.end()) {
        ctrl_bf16 w;
        w.down.resize((size_t)RANK * 2 * D);
        for (size_t i = 0; i < w.down.size(); i++) w.down[i] = f2bf(c.down[i]);
        for (auto [src, dst] : {std::make_pair(c.forget, &w.f), std::make_pair(c.gain, &w.a)}) {
            dst->resize((size_t)RANK * D);
            for (int r = 0; r < RANK; r++)
                for (int i = 0; i < D; i++) (*dst)[((size_t)(r / 2) * D + i) * 2 + r % 2] = f2bf(src[(size_t)i * RANK + r]);
        }
        it = cache.emplace(c.down, std::move(w)).first;
    }
    return it->second;
}

/// \brief 16 fp32 as bf16, round to nearest even
inline __m256i to_bf16x16(__m512 v) { return (__m256i)_mm512_cvtneps_pbh(v); }

/// \brief y[t * ldy + n] = x[t * ldx ..] . w[n * ldw ..] over K (a multiple of 32), bf16 x and w, fp32 sums
///        (vdpbf16ps); tiles of 4 tokens x 4 rows as gemm_nt
void gemm_nt_bf16(float* y, int ldy, const u16* x, int ldx, const u16* w, int ldw, int T, int N, int K) {
    constexpr int TB = 4, NB = 4;
    const int tb = (T + TB - 1) / TB, nb = (N + NB - 1) / NB;
    #pragma omp parallel for schedule(static) if (T >= 16)   // flat, as gemm_nt
    for (int64_t ib = 0; ib < (int64_t)tb * nb; ib++) {
        const int t0 = (int)(ib / nb) * TB, n0 = (int)(ib % nb) * NB;
        const u16* xp[TB];
        const u16* wp[NB];
        for (int i = 0; i < TB; i++) xp[i] = x + (size_t)std::min(t0 + i, T - 1) * ldx;
        for (int j = 0; j < NB; j++) wp[j] = w + (size_t)std::min(n0 + j, N - 1) * ldw;
        __m512 acc[TB][NB];
        for (int i = 0; i < TB; i++)
            for (int j = 0; j < NB; j++) acc[i][j] = _mm512_setzero_ps();
        for (int k = 0; k < K; k += 32) {
            __m512bh xv[TB], wv[NB];
            for (int i = 0; i < TB; i++) xv[i] = (__m512bh)_mm512_loadu_si512(xp[i] + k);
            for (int j = 0; j < NB; j++) wv[j] = (__m512bh)_mm512_loadu_si512(wp[j] + k);
            for (int i = 0; i < TB; i++)
                for (int j = 0; j < NB; j++) acc[i][j] = _mm512_dpbf16_ps(acc[i][j], xv[i], wv[j]);
        }
        for (int i = 0; i < TB && t0 + i < T; i++)
            for (int j = 0; j < NB && n0 + j < N; j++) y[(size_t)(t0 + i) * ldy + n0 + j] = _mm512_reduce_add_ps(acc[i][j]);
    }
}

/// \brief loop controller: cur - tanh(forget g) * cur + tanh(gain g) * msg, g = down [rms(cur); rms(msg)]
///        (bf16 products, fp32 sums)
act control(const float* cur, const float* msg, const loop_ctrl& c, int T) {
    act out = make_act((size_t)T * D);
    const ctrl_bf16& W = ctrl_weights(c);
    // tiles of TB tokens, each start to end while its rows stay in the core's cache: [ix x; im m] in bf16, g = that
    // . down^T [TB][RANK], then the up projections and the output (whole-prompt passes moved ~2.4x the bytes)
    constexpr int TB = 8;
    const int tb = (T + TB - 1) / TB;
    #pragma omp parallel for schedule(static) if (T >= 16)
    for (int bt = 0; bt < tb; bt++) {
        const int t0 = bt * TB, nt = std::min(TB, T - t0);
        thread_local std::vector<u16> xm_tl(TB * 2 * D), gb_tl(TB * RANK);   // each worker its own
        thread_local std::vector<float> g_tl(TB * RANK);
        u16* xm = xm_tl.data();
        u16* gb = gb_tl.data();
        float* g = g_tl.data();
        for (int k = 0; k < nt; k++) {
            const float* x = cur + (size_t)(t0 + k) * D;
            const float* m = msg + (size_t)(t0 + k) * D;
            const __m512 ix = _mm512_set1_ps(1.0f / std::sqrt(dot(x, x, D) / D + EPS));
            const __m512 im = _mm512_set1_ps(1.0f / std::sqrt(dot(m, m, D) / D + EPS));
            u16* row = xm + (size_t)k * 2 * D;
            for (int i = 0; i < D; i += 16) {
                _mm256_storeu_si256((__m256i*)(row + i), to_bf16x16(_mm512_mul_ps(ix, _mm512_loadu_ps(x + i))));
                _mm256_storeu_si256((__m256i*)(row + D + i), to_bf16x16(_mm512_mul_ps(im, _mm512_loadu_ps(m + i))));
            }
        }
        gemm_nt_bf16(g, RANK, xm, 2 * D, W.down.data(), 2 * D, nt, RANK, 2 * D);   // nt < 16: serial
        for (int i = 0; i < nt * RANK; i++) gb[i] = f2bf(g[i]);
        // 16 columns at a time: each up-projection row pair is read once per tile, one vdpbf16ps per rank pair
        const uint32_t* gp = reinterpret_cast<const uint32_t*>(gb);   // rank pairs
        for (int i = 0; i < D; i += 16) {
            __m512 f[TB], a[TB];
            for (int k = 0; k < TB; k++) f[k] = a[k] = _mm512_setzero_ps();
            auto tile = [&](auto n) {
                for (int r = 0; r < RANK / 2; r++) {
                    const __m512bh fv = (__m512bh)_mm512_loadu_si512(W.f.data() + ((size_t)r * D + i) * 2);
                    const __m512bh av = (__m512bh)_mm512_loadu_si512(W.a.data() + ((size_t)r * D + i) * 2);
                    for (int k = 0; k < (int)n; k++) {
                        const __m512bh gr = (__m512bh)_mm512_set1_epi32((int)gp[(size_t)k * RANK / 2 + r]);
                        f[k] = _mm512_dpbf16_ps(f[k], gr, fv);
                        a[k] = _mm512_dpbf16_ps(a[k], gr, av);
                    }
                }
            };
            if (nt == TB) tile(std::integral_constant<int, TB>());
            else tile(nt);
            for (int k = 0; k < nt; k++) {
                const size_t o = (size_t)(t0 + k) * D + i;
                const __m512 xv = _mm512_loadu_ps(cur + o);
                // x - f x + a m
                _mm512_storeu_ps(out->data() + o, _mm512_fmadd_ps(tanh16(a[k]), _mm512_loadu_ps(msg + o), _mm512_fnmadd_ps(tanh16(f[k]), xv, xv)));
            }
        }
    }
    return out;
}

/// \brief causal short convolution with a residual, in place on x [T][C]
/// \param w     [C][NCONV], tap NCONV - 1 = this token
/// \param dyn   [T][NKV * NCONV] input-dependent taps per head of DH channels (lag 0 = this token), or null
/// \param state [NCONV - 1][CONV_CH] at this tensor's channel offset; updated
/// \param src   the input as bf16 rows instead of x, or null
void convolve(float* x, int T, int C, const float* w, const float* dyn, float* state, const bf16_rows* src = nullptr) {
    static_assert(NCONV == 4, "convolve keeps the last 3 inputs in registers");
    // the taps as [NCONV][C], once per matrix
    static std::mutex mu;
    static std::map<const float*, std::vector<float>> taps;
    const float* wt;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = taps.find(w);
        if (it == taps.end()) {
            std::vector<float> t((size_t)NCONV * C);
            for (int c = 0; c < C; c++)
                for (int k = 0; k < NCONV; k++) t[(size_t)k * C + c] = w[c * NCONV + k];
            it = taps.emplace(w, std::move(t)).first;
        }
        wt = it->second.data();
    }
    const __m512 inv = _mm512_set1_ps(1.0f / std::sqrt((float)((dyn ? 2 : 1) * NCONV + 1)));
    // in place, 16 channels per task in token order: the 3 inputs before t ride along in registers
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; c += 16) {
        __m512 wk[NCONV];
        for (int k = 0; k < NCONV; k++) wk[k] = _mm512_loadu_ps(wt + (size_t)k * C + c);
        __m512 p0 = _mm512_loadu_ps(state + c), p1 = _mm512_loadu_ps(state + CONV_CH + c),
               p2 = _mm512_loadu_ps(state + 2 * CONV_CH + c);
        const float* d = dyn ? dyn + (c / DH) * NCONV : nullptr;
        for (int t = 0; t < T; t++) {
            float* xt = x + (size_t)t * C + c;
            const __m512 p3 = src ? load_bf16x16(src->row(t) + c) : _mm512_loadu_ps(xt);
            __m512 acc = _mm512_mul_ps(wk[0], p0);
            acc = _mm512_fmadd_ps(wk[1], p1, acc);
            acc = _mm512_fmadd_ps(wk[2], p2, acc);
            acc = _mm512_fmadd_ps(wk[3], p3, acc);
            if (d) {
                const float* dt = d + (size_t)t * NKV * NCONV;   // lag 0 = this token
                acc = _mm512_fmadd_ps(_mm512_set1_ps(dt[0]), p3, acc);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(dt[1]), p2, acc);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(dt[2]), p1, acc);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(dt[3]), p0, acc);
            }
            _mm512_storeu_ps(xt, _mm512_mul_ps(_mm512_add_ps(p3, acc), inv));
            p0 = p1;
            p1 = p2;
            p2 = p3;
        }
        _mm512_storeu_ps(state + c, p0);
        _mm512_storeu_ps(state + CONV_CH + c, p1);
        _mm512_storeu_ps(state + 2 * CONV_CH + c, p2);
    }
}

/// \brief attention of T queries (positions p0 ..) over one slot's cache, which already holds them
/// \param O     [T][QD] out
/// \param Q     [T][QD] clipped (and log-scaled on global layers)
/// \param R     [T][NH * NREL] clipped (and log-scaled)
/// \param sinks [T][NH] sink logits
/// \param window 0 = all earlier keys, else keys with p_q - p_k < window
/// \param ring  0 = row p is key p, else row p % ring
void attention(float* O, const float* Q, const float* R, const float* sinks, const float* rel_bias, const u16* Kc,
               const u16* Vc, int ring, int window, int p0, int T) {
    constexpr int BQ = 16;          // tokens per task
    constexpr int G = NH / NKV;     // query heads per KV head
    constexpr int KB = 64;          // keys per inner block
    const float scale = 1.0f / std::sqrt((float)DH);
    const int nqb = (T + BQ - 1) / BQ;
    // split the keys when there are too few query tasks to fill the machine (decode)
    const int span = window ? std::min(p0 + T, window + T) : p0 + T;
    int nsplit = std::max(1, (4 * n_threads()) / (nqb * NKV));
    nsplit = std::min(nsplit, std::max(1, span / 1024));
    const int ntask = nqb * NKV * nsplit;
    constexpr int NQ = BQ * G;
    // partial results per task: m, l, acc[DH] per query, each written before it is read. This buffer and the
    // workers' below live as long as their threads (fresh ones on every call cost page faults on Windows).
    static thread_local std::vector<float> part_tl;
    if (part_tl.size() < (size_t)ntask * NQ * (DH + 2)) part_tl.resize((size_t)ntask * NQ * (DH + 2));
    float* const part = part_tl.data();

    #pragma omp parallel
    {
        static thread_local std::vector<float> tab((size_t)NQ * (REL_EXT + 1));
        static thread_local std::vector<float> kf((size_t)KB * DH), vf((size_t)KB * DH), s(KB);
        #pragma omp for schedule(dynamic)
        for (int task = 0; task < ntask; task++) {
            const int split = task % nsplit;
            const int g = (task / nsplit) % NKV;
            const int qb = task / (nsplit * NKV);
            const int t0 = qb * BQ, t1 = std::min(T, t0 + BQ);
            const int nq = (t1 - t0) * G;
            // key range of the whole task, then this split's share
            const int pmin = p0 + t0, pmax = p0 + t1 - 1;
            const int lo_all = window ? std::max(0, pmin - window + 1) : 0;
            const int n_all = pmax - lo_all + 1;
            const int per = (n_all + nsplit - 1) / nsplit;
            const int lo = lo_all + split * per, hi = std::min(pmax, lo + per - 1);
            float* pr = part + (size_t)task * NQ * (DH + 2);
            // relative bias table per query: tab[d] = rel_bias[d] . r (distance >= REL_EXT: 0)
            for (int qi = 0; qi < nq; qi++) {
                const int t = t0 + qi / G, h = g * G + qi % G;
                const float* r = R + (size_t)t * NH * NREL + h * NREL;
                float* tb = tab.data() + (size_t)qi * (REL_EXT + 1);
                for (int d = 0; d < REL_EXT; d++) {
                    float a = 0;
                    for (int k = 0; k < NREL; k++) a += rel_bias[d * NREL + k] * r[k];
                    tb[d] = a;
                }
                tb[REL_EXT] = 0;
                float* st = pr + (size_t)qi * (DH + 2);
                // the sink joins the softmax as one more logit with a zero value
                st[0] = split == 0 ? sinks[(size_t)t * NH + h] : -std::numeric_limits<float>::infinity();
                st[1] = split == 0 ? 1.0f : 0.0f;
                std::fill(st + 2, st + 2 + DH, 0.0f);
            }
            for (int k0 = lo; k0 <= hi; k0 += KB) {
                const int nk = std::min(KB, hi - k0 + 1);
                for (int j = 0; j < nk; j++) {
                    const int p = k0 + j;
                    const size_t row = (size_t)(ring ? p % ring : p) * KVD + g * DH;
                    for (int c = 0; c < DH; c += 16) {
                        _mm512_storeu_ps(kf.data() + j * DH + c, load_bf16x16(Kc + row + c));
                        _mm512_storeu_ps(vf.data() + j * DH + c, load_bf16x16(Vc + row + c));
                    }
                }
                for (int qi = 0; qi < nq; qi++) {
                    const int t = t0 + qi / G, h = g * G + qi % G;
                    const int pq = p0 + t;
                    const int jlo = std::max(0, (window ? pq - window + 1 : 0) - k0);
                    const int jhi = std::min(nk - 1, pq - k0);
                    if (jlo > jhi) continue;
                    const float* q = Q + (size_t)t * QD + h * DH;
                    const float* tb = tab.data() + (size_t)qi * (REL_EXT + 1);
                    float mx = -std::numeric_limits<float>::infinity();
                    for (int j = jlo; j <= jhi; j++) {
                        const int dist = pq - (k0 + j);
                        s[j] = dot(q, kf.data() + j * DH, DH) * scale + tb[std::min(dist, REL_EXT)];
                        mx = std::max(mx, s[j]);
                    }
                    float* st = pr + (size_t)qi * (DH + 2);
                    const float m_new = std::max(st[0], mx);
                    const float corr = std::exp(st[0] - m_new);
                    float l = st[1] * corr;
                    __m512 acc[DH / 16];
                    for (int c = 0; c < DH / 16; c++)
                        acc[c] = _mm512_mul_ps(_mm512_loadu_ps(st + 2 + c * 16), _mm512_set1_ps(corr));
                    for (int j = jlo; j <= jhi; j++) {
                        const float pj = std::exp(s[j] - m_new);
                        l += pj;
                        const __m512 pv = _mm512_set1_ps(pj);
                        for (int c = 0; c < DH / 16; c++)
                            acc[c] = _mm512_fmadd_ps(pv, _mm512_loadu_ps(vf.data() + j * DH + c * 16), acc[c]);
                    }
                    for (int c = 0; c < DH / 16; c++) _mm512_storeu_ps(st + 2 + c * 16, acc[c]);
                    st[0] = m_new;
                    st[1] = l;
                }
            }
        }
    }
    // merge the splits
    #pragma omp parallel for schedule(static)   // flat, as gemm_nt
    for (int i = 0; i < nqb * NKV; i++) {
        const int qb = i / NKV, g = i % NKV;
        const int t0 = qb * BQ, t1 = std::min(T, t0 + BQ);
        for (int qi = 0; qi < (t1 - t0) * G; qi++) {
            const int t = t0 + qi / G, h = g * G + qi % G;
            float m = -std::numeric_limits<float>::infinity();
            for (int sp = 0; sp < nsplit; sp++)
                m = std::max(m, part[((size_t)((qb * NKV + g) * nsplit + sp) * NQ + qi) * (DH + 2)]);
            float l = 0;
            float o[DH] = {};
            for (int sp = 0; sp < nsplit; sp++) {
                const float* st = part + ((size_t)((qb * NKV + g) * nsplit + sp) * NQ + qi) * (DH + 2);
                if (st[1] == 0) continue;
                const float f = std::exp(st[0] - m);
                l += st[1] * f;
                for (int c = 0; c < DH; c++) o[c] += st[2 + c] * f;
            }
            float* out = O + (size_t)t * QD + h * DH;
            for (int c = 0; c < DH; c++) out[c] = o[c] / l;
        }
    }
}

thread_local int trace_slot = -1;   // the slot decode_trace reports from moe()

/// \brief routed experts: out[T][D] = sum over the top 8 of w * down(situ(gate_up x))
void moe(float* out, const float* x, int T, const layer_weights& L, int il, mm_offload* offload) {
    // router (fp32), sigmoid, top 8 on prob + bias, weights = normalized probs * ROUTE_SCALE
    auto tp = std::chrono::steady_clock::now();
    std::vector<float> logits((size_t)T * NE);
    dense_mm(logits.data(), L.router, NE, D, x, T);
    std::vector<int> sel((size_t)T * TOPK);
    std::vector<float> wsel((size_t)T * TOPK);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        float p[NE], key[NE];
        int idx[NE];
        for (int e = 0; e < NE; e++) {
            p[e] = sigmoid(logits[(size_t)t * NE + e]);
            key[e] = p[e] + L.expert_bias[e];
            idx[e] = e;
        }
        std::partial_sort(idx, idx + TOPK, idx + NE, [&](int a, int b) { return key[a] > key[b]; });
        float sum = 0;
        for (int k = 0; k < TOPK; k++) sum += p[idx[k]];
        for (int k = 0; k < TOPK; k++) {
            sel[(size_t)t * TOPK + k] = idx[k];
            wsel[(size_t)t * TOPK + k] = p[idx[k]] / (sum + 1e-20f) * ROUTE_SCALE;
        }
    }
    if (decode_trace && T == 1) {
        float sw[2 * TOPK];
        for (int k = 0; k < TOPK; k++) { sw[k] = (float)sel[k]; sw[TOPK + k] = wsel[k]; }
        decode_trace(trace_slot, "sel", sw, 2 * TOPK);
    }
    // group the (token, k) pairs by expert
    std::vector<int> count(NE, 0), start(NE + 1, 0);
    for (int i = 0; i < T * TOPK; i++) count[sel[i]]++;
    for (int e = 0; e < NE; e++) start[e + 1] = start[e] + count[e];
    std::vector<int> fill(start.begin(), start.end() - 1);
    std::vector<int> slot_of((size_t)T * TOPK);   // (t, k) -> row in the grouped buffers
    std::vector<int> token_of((size_t)T * TOPK);
    for (int i = 0; i < T * TOPK; i++) {
        const int r = fill[sel[i]]++;
        slot_of[i] = r;
        token_of[r] = i / TOPK;
    }
    prof_lap(tp, P_ROUTER);
    if (offload && offload->experts(il, x, T, token_of.data(), count.data(), start.data(), slot_of.data(), wsel.data(), out)) {
        prof_lap(tp, P_EXPERT);
        return;
    }
    {
        const int M = T * TOPK;
        // reused across calls: a fresh 75 MB vector page-faults on every layer (A's pad columns stay 0)
        static thread_local std::vector<float> X_buf, GU_buf, A_buf, Z_buf;
        std::vector<float>& X = X_buf;
        std::vector<float>& GU = GU_buf;
        std::vector<float>& A = A_buf;
        std::vector<float>& Z = Z_buf;
        if (X.size() < (size_t)M * D) X.resize((size_t)M * D);
        if (GU.size() < (size_t)M * 2 * FF) GU.resize((size_t)M * 2 * FF);
        if (A.size() < (size_t)M * FF_PAD) A.resize((size_t)M * FF_PAD, 0.0f);
        if (Z.size() < (size_t)M * D) Z.resize((size_t)M * D);
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < M; r++) memcpy(X.data() + (size_t)r * D, x + (size_t)token_of[r] * D, D * sizeof(float));
        const int rb_per_expert = D / QROWS;   // gate_up: 2 FF = D rows, down: D rows
        std::vector<mm_job> jobs;
        for (int e = 0; e < NE; e++)
            if (count[e])
                jobs.push_back({L.gate_up.rows(e * rb_per_expert, 2 * FF), X.data() + (size_t)start[e] * D, D, count[e],
                                GU.data() + (size_t)start[e] * 2 * FF, 2 * FF});
        qmm(jobs);
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < M; r++) {
            const float* gu = GU.data() + (size_t)r * 2 * FF;
            float* a = A.data() + (size_t)r * FF_PAD;
            for (int i = 0; i < FF; i += 16) _mm512_storeu_ps(a + i, situ16(_mm512_loadu_ps(gu + i), _mm512_loadu_ps(gu + FF + i)));
        }
        jobs.clear();
        for (int e = 0; e < NE; e++)
            if (count[e])
                jobs.push_back({L.down.rows(e * rb_per_expert, D), A.data() + (size_t)start[e] * FF_PAD, FF_PAD, count[e],
                                Z.data() + (size_t)start[e] * D, D});
        qmm(jobs);
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            float* o = out + (size_t)t * D;
            std::fill(o, o + D, 0.0f);
            for (int k = 0; k < TOPK; k++) {
                const float wk = wsel[(size_t)t * TOPK + k];
                const float* z = Z.data() + (size_t)slot_of[(size_t)t * TOPK + k] * D;
                for (int c = 0; c < D; c++) o[c] += wk * z[c];
            }
        }
    }
    prof_lap(tp, P_EXPERT);
}

/// \brief one persistent thread with its own OpenMP pool: runs the first half of a split prompt layer
class helper_thread {
public:
    helper_thread() : th([this] { loop(); }) {}
    ~helper_thread() {
        {
            std::lock_guard<std::mutex> lock(mu);
            stop = true;
        }
        cv.notify_one();
        th.join();
    }
    std::future<void> submit(std::function<void()> f) {
        std::packaged_task<void()> task(std::move(f));
        std::future<void> done = task.get_future();
        {
            std::lock_guard<std::mutex> lock(mu);
            q.push_back(std::move(task));
        }
        cv.notify_one();
        return done;
    }

private:
    void loop() {
        for (;;) {
            std::packaged_task<void()> task;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] { return stop || !q.empty(); });
                if (q.empty()) return;
                task = std::move(q.front());
                q.pop_front();
            }
            task();
        }
    }
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::packaged_task<void()>> q;
    bool stop = false;
    std::thread th;
};

/// \brief the calling (compute) thread's helper, started on first use; release_helper() ends it and its OpenMP pool
std::unique_ptr<helper_thread>& helper_slot() {
    thread_local std::unique_ptr<helper_thread> h;
    return h;
}

helper_thread& helper() {
    std::unique_ptr<helper_thread>& h = helper_slot();
    if (!h) h = std::make_unique<helper_thread>();
    return *h;
}

void release_helper() { helper_slot().reset(); }

/// \brief prompt steps of at least this many tokens split the attention side of a layer in two halves
///        (INKLING2_SPLIT=0: off): the host works on one half while the NPU runs the other's attention
int split_min() {
    static const int v = [] {
        const char* e = std::getenv("INKLING2_SPLIT");
        return e && std::atoi(e) == 0 ? 1 << 30 : 256;
    }();
    return v;
}

/// \brief one layer application (cache slot `slot`), T tokens at positions p0 ..
/// \param ip   the layer's place in its pass (blocks start at ip % BLOCK == 0)
/// \param msgs [4][T][D] loop messages (q, k, v, ffn), or null
act layer(const layer_weights& L, int il, int ip, int slot, model::state& s, int p0, int T, act cur,
          std::vector<act>& blocks, const float* msgs) {
    const bool global = is_global(il);
    const size_t TD = (size_t)T * D;
    const act inpSA = cur;
    const bool tr = decode_trace && T == 1;
    if (tr) decode_trace(slot, "in", cur->data(), D);
    const bool split = s.offload && T >= split_min();
    auto t_all = std::chrono::steady_clock::now();
    // the first layer of a block starts a new residual stream; the old one joins the blocks (the FFN mix sees it)
    const bool starts = ip % BLOCK == 0;
    std::vector<act> fblocks = blocks;
    if (starts) fblocks.push_back(inpSA);
    act y = make_act(TD);
    const act res = starts ? y : make_act(TD);   // the residual stream after attention
    act xf = make_act(TD);                         // the FFN input (mixed, controlled, normed)
    float* cs = s.conv_state(slot);
    u16* Kc = s.k[slot].data();
    u16* Vc = s.v[slot].data();
    const int ring = global ? 0 : SWA_RING;

    // a later half waits for the earlier one where state passes between them: the K/V conv state, the cache rows
    // its attention reads, the attention conv state; and it starts once the earlier half's projections are on their
    // way to the NPU, so each host part has every core and host and NPU work alternate
    struct order {
        std::promise<void> p[4];
        std::shared_future<void> f[4];
        bool set[4] = {};
        order() {
            for (int i = 0; i < 4; i++) f[i] = p[i].get_future().share();
        }
        void done(int i) {
            set[i] = true;
            p[i].set_value();
        }
        void fail(std::exception_ptr e) {
            for (int i = 0; i < 4; i++)
                if (!set[i]) p[i].set_exception(e);
        }
    };
    enum { O_CONV, O_KV, O_ACONV, O_IN };

    // the attention side of tokens [t0, t0 + n), then the FFN input of those tokens
    auto half = [&](int t0, int n, order* mine, order* prev) {
        auto tp = std::chrono::steady_clock::now();
        auto lap = [&](prof_part part) {
            const auto t_start = tp;
            if (!split) prof_lap(tp, part);
            else prof_lap_half(tp, part, mine ? 0 : 1);
            if ((il == half_trace() || half_trace() == -2) && T >= 256)   // DIAG (-2: every layer)
                std::fprintf(stderr, "HT %d %d %d %lld %lld\n", il, mine ? 0 : 1, (int)part, trace_ns(t_start),
                             trace_ns(std::chrono::steady_clock::now()));
        };
        auto wait = [&](int i) {
            if (prev) prev->f[i].get();
        };
        auto signal = [&](int i) {
            if (mine) mine->done(i);
        };
        const size_t ND = (size_t)n * D;
        const int q0 = p0 + t0;   // position of the first token
        const float* cur_rows = cur->data() + (size_t)t0 * D;
        wait(O_IN);

        // attention inputs: SWA layers mix one, global layers one each for Q, K, V
        const float* in[3] = {cur_rows, cur_rows, cur_rows};
        act mixed;
        if (L.attn_query) {
            mixed = mix(blocks, cur, L.attn_query, global ? 3 : 1, n, t0);
            for (int j = 0; j < 3; j++) in[j] = mixed->data() + (global ? j : 0) * ND;
        }
        lap(P_MIX);
        act ctl[3];
        if (msgs) {
            for (int j = 0; j < 3; j++) {
                ctl[j] = control(in[j], msgs + j * TD + (size_t)t0 * D, L.ctrl[j], n);
                in[j] = ctl[j]->data();
            }
        }
        lap(P_CTRL);
        // the input norms
        const bool three = !global ? (msgs != nullptr) : true;
        act nrm = make_act((three ? 3 : 1) * ND);
        for (int j = 0; j < (three ? 3 : 1); j++) {
            const float* g = global ? L.qkv_norm[j] : L.attn_norm;
            float* dst = nrm->data() + j * ND;
            const float* src = in[j];
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < n; t++) rms_norm(dst + (size_t)t * D, src + (size_t)t * D, g, D);
        }
        const float* xq = nrm->data();
        const float* xk = three ? xq + ND : xq;
        const float* xv = three ? xq + 2 * ND : xq;
        if (tr) decode_trace(slot, "nrm", xq, (three ? 3 : 1) * D);
        lap(P_NORM);

        // projections
        // pooled: a fresh multi-MB vector page-faults on every layer; each is written whole before it is read
        const act Qa = make_act((size_t)n * QD), Ka = make_act((size_t)n * KVD), Va = make_act((size_t)n * KVD),
                  Ra = make_act((size_t)n * NH * NREL), Ga = make_act((size_t)n * QD), Sa = make_act((size_t)n * NH);
        act_vec &Q = *Qa, &K = *Ka, &V = *Va, &R = *Ra, &Gt = *Ga, &sinks = *Sa;
        // from the NPU: Q K V R G stay bf16 where it wrote them, read by the convolutions, the clips and the gate
        bf16_rows pv[5];
        bool npu_in;
        {
            signal(O_IN);
            npu_in = s.offload && s.offload->attn_in(il, xq, xk, xv, n, Q.data(), K.data(), V.data(), R.data(), Gt.data(), pv);
        }
        if (!npu_in)
            qmm({{L.q, xq, D, n, Q.data(), QD},
                 {L.k, xk, D, n, K.data(), KVD},
                 {L.v, xv, D, n, V.data(), KVD},
                 {L.rel, xq, D, n, R.data(), NH * NREL},
                 {L.gate, xq, D, n, Gt.data(), QD}});
        lap(P_PROJ);
        dense_mm(sinks.data(), L.sink_proj, NH, D, xq, n);
        std::vector<float> dk, dv;
        if (global) {
            dk.resize((size_t)n * NKV * NCONV);
            dv.resize((size_t)n * NKV * NCONV);
            dense_mm(dk.data(), L.k_conv_proj, NKV * NCONV, D, xk, n);
            dense_mm(dv.data(), L.v_conv_proj, NKV * NCONV, D, xv, n);
        }
        lap(P_X_SINK);
        if (tr && !npu_in) { decode_trace(slot, "q", Q.data(), QD); decode_trace(slot, "k", K.data(), KVD); }
        wait(O_CONV);
        convolve(K.data(), n, KVD, L.k_conv, global ? dk.data() : nullptr, cs + CONV_K, npu_in ? &pv[1] : nullptr);
        convolve(V.data(), n, KVD, L.v_conv, global ? dv.data() : nullptr, cs + CONV_V, npu_in ? &pv[2] : nullptr);
        signal(O_CONV);
        lap(P_CONV);

        // clip / norm and log scaling
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < n; t++) {
            const int p = q0 + t;
            const float tau = global ? 1.0f + LOG_SCALE_ALPHA * std::log(std::max(1.0f, (float)(p + 1) / LOG_SCALE_FLOOR)) : 1.0f;
            if (npu_in) {
                bf16_to_f32(Q.data() + (size_t)t * QD, pv[0].row(t), QD);
                bf16_to_f32(R.data() + (size_t)t * NH * NREL, pv[3].row(t), NH * NREL);
            }
            for (int h = 0; h < NH; h++) {
                float* q = Q.data() + (size_t)t * QD + h * DH;
                for (int c = 0; c < DH; c++) q[c] *= L.q_clip[h * DH + c];
                l2_clip(q, DH, QK_CLIP);
                float* r = R.data() + (size_t)t * NH * NREL + h * NREL;
                for (int c = 0; c < NREL; c++) r[c] *= L.rel_clip[h * NREL + c];
                l2_clip(r, NREL, REL_CLIP);
                if (global) {
                    for (int c = 0; c < DH; c++) q[c] *= tau;
                    for (int c = 0; c < NREL; c++) r[c] *= tau;
                }
            }
            for (int h = 0; h < NKV; h++) {
                l2_clip(K.data() + (size_t)t * KVD + h * DH, DH, QK_CLIP);
                float* v = V.data() + (size_t)t * KVD + h * DH;
                rms_norm(v, v, nullptr, DH);
            }
        }
        lap(P_X_CLIP);

        const act Oa = make_act((size_t)n * QD);
        act_vec& O = *Oa;
        // the NPU attends the tokens from the cache and their K, V; the cache takes them once the attention has
        // read it (a SWA ring keeps the last SWA_RING), before the device runs, so the next half packs meanwhile;
        // else K and V into the cache, then attention, SWA layers in sub-steps so the ring holds every key in use
        wait(O_KV);
        const std::function<void()> kv_write = [&] {
            #pragma omp parallel for schedule(static)
            for (int t = global ? 0 : std::max(0, n - SWA_RING); t < n; t++) {
                const size_t row = s.row(slot, q0 + t);
                for (int c = 0; c < KVD; c++) {
                    Kc[row + c] = f2bf(K[(size_t)t * KVD + c]);
                    Vc[row + c] = f2bf(V[(size_t)t * KVD + c]);
                }
            }
            signal(O_KV);
        };
        const bool npu_attention = s.offload && s.offload->attention(il, q0, n, Q.data(), R.data(), sinks.data(), L.rel_bias,
                                                                     Kc, Vc, ring, global ? 0 : SWA, K.data(), V.data(),
                                                                     Gt.data(), npu_in ? &pv[4] : nullptr, O.data(), kv_write);
        if (npu_attention) {
            lap(P_X_ATTNPU);
        } else {
            for (int u0 = 0, sub = global ? n : SUB; u0 < n; u0 += sub) {
                const int m = std::min(sub, n - u0);
                #pragma omp parallel for schedule(static)
                for (int t = u0; t < u0 + m; t++) {
                    const size_t row = s.row(slot, q0 + t);
                    for (int c = 0; c < KVD; c++) {
                        Kc[row + c] = f2bf(K[(size_t)t * KVD + c]);
                        Vc[row + c] = f2bf(V[(size_t)t * KVD + c]);
                    }
                }
                attention(O.data() + (size_t)u0 * QD, Q.data() + (size_t)u0 * QD, R.data() + (size_t)u0 * NH * NREL,
                          sinks.data() + (size_t)u0 * NH, L.rel_bias, Kc, Vc, ring, global ? 0 : SWA, q0 + u0, m);
            }
            signal(O_KV);
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < n; t++) {
                float* g = Gt.data() + (size_t)t * QD;
                if (npu_in) bf16_to_f32(g, pv[4].row(t), QD);
                for (int c = 0; c < QD; c++) O[(size_t)t * QD + c] *= sigmoid(g[c]);
            }
        }
        lap(P_X_GATE);
        if (tr) decode_trace(slot, "o", O.data(), QD);
        float* yr = y->data() + (size_t)t0 * D;
        bf16_rows yv;
        bool npu_out;
        {
            npu_out = s.offload && s.offload->attn_out(il, O.data(), n, yr, &yv);
        }
        if (!npu_out) qmm({{L.o, O.data(), QD, n, yr, D}});
        lap(P_OPROJ);
        if (tr && !npu_out) decode_trace(slot, "y", yr, D);
        wait(O_ACONV);
        convolve(yr, n, D, L.attn_conv, nullptr, cs + CONV_ATTN, npu_out ? &yv : nullptr);
        signal(O_ACONV);
        lap(P_CONV);
        if (!starts) {
            float* r = res->data() + (size_t)t0 * D;
            const float* a = inpSA->data() + (size_t)t0 * D;
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < (int64_t)ND; i++) r[i] = a[i] + yr[i];
        }
        lap(P_X_RES);
        if (tr) decode_trace(slot, "res", res->data(), D);

        // the FFN input of these tokens
        act x = mix(fblocks, res, L.ffn_query, 1, n, t0);
        lap(P_X_FFNMIX);
        if (msgs) x = control(x->data(), msgs + 3 * TD + (size_t)t0 * D, L.ctrl[3], n);
        lap(P_X_FFNCTRL);
        float* xr = xf->data() + (size_t)t0 * D;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < n; t++) rms_norm(xr + (size_t)t * D, x->data() + (size_t)t * D, L.ffn_norm, D);
        if (tr) decode_trace(slot, "xf", xr, D);
        lap(P_NORM);
    };

    if (!split) half(0, T, nullptr, nullptr);
    else {
        // halves of whole 128-query attention rounds
        const int t1 = std::clamp((T / 2 + 64) / 128 * 128, 128, T - 128);
        order first;
        // DIAG: INKLING2_HALF_THREADS=<n>: the OpenMP team of each half
        static const int half_threads = [] { const char* e = std::getenv("INKLING2_HALF_THREADS"); return e ? std::atoi(e) : 0; }();
        const int caller_threads = n_threads();
        auto run_first = [&] {
            s.offload->set_lane(0);
            try {
                half(0, t1, &first, nullptr);
            } catch (...) {
                first.fail(std::current_exception());
                throw;
            }
        };
        auto run_second = [&] {
            s.offload->set_lane(1);
            half(t1, T - t1, nullptr, &first);
        };
        // DIAG: INKLING2_SWAP_HALVES=1: the caller runs the first half, the helper the second
        static const bool swap_halves = [] { const char* e = std::getenv("INKLING2_SWAP_HALVES"); return e && e[0] == '1'; }();
        std::future<void> a = helper().submit([&] {
            use_core_threads(true);
            if (half_threads > 0) omp_set_num_threads(half_threads);
            if (swap_halves) run_second();
            else run_first();
        });
        if (half_threads > 0) omp_set_num_threads(half_threads);
        std::exception_ptr err;
        try {
            if (swap_halves) run_first();
            else run_second();
        } catch (...) {
            err = std::current_exception();
        }
        a.get();
        omp_set_num_threads(caller_threads);
        s.offload->set_lane(0);
        if (err) std::rethrow_exception(err);
        prof_lap(t_all, P_ATTN);
    }
    blocks = std::move(fblocks);
    auto tp = std::chrono::steady_clock::now();

    // MoE
    act f = make_act(TD);
    trace_slot = slot;
    moe(f->data(), xf->data(), T, L, il, s.offload);
    if (tr) decode_trace(slot, "moe", f->data(), D);
    tp = std::chrono::steady_clock::now();
    convolve(f->data(), T, D, L.ffn_conv, nullptr, cs + CONV_FFN);
    prof_lap(tp, P_CONV);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)TD; i++) (*f)[i] += (*res)[i];
    prof_lap(tp, P_X_RES);
    if (tr) decode_trace(slot, "out", f->data(), D);
    return f;
}

}  // namespace

void control_norm(const float* x, const float* m, const loop_ctrl& c, const float* g, float* out) {
    const act y = control(x, m, c, 1);
    rms_norm(out, y->data(), g, D);
}

void lm_head(const weights& w, const float* x, float* logits) {
    std::vector<float> full(VOCAB_PAD);
    qmm({{w.lm_head, x, D, 1, full.data(), VOCAB_PAD}});
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < VOCAB; i++) logits[i] = SOFTCAP * std::tanh(full[i] / SOFTCAP);
}

// =============================================================================
// model
// =============================================================================
model::model() : file_(new q4nx_file), w_(new weights), s_(new state) {}
model::~model() = default;

const weights& model::w() const { return *w_; }
weights& model::w_mut() { return *w_; }
void model::set_offload(mm_offload* o) { s_->offload = o; }
int model::position() const { return s_->pos; }
int model::max_length() const { return s_->max_len; }

void model::load(const std::string& path) {
    q4nx_file& f = *file_;
    f.open(path);
    weights& w = *w_;
    w.embed = f.bf16("model.embed_tokens.weight", {VOCAB_PAD, D});
    w.embed_norm = f.f32("model.embed_norm.weight", {D});
    w.final_query = f.f32("model.final_input.query", {D});
    w.norm = f.f32("model.norm.weight", {D});
    w.lm_head = f.q("lm_head.weight", qfmt::q80, VOCAB_PAD, D);
    for (int il = 0; il < NL; il++) {
        layer_weights& L = w.layer[il];
        const std::string p = "model.layer." + std::to_string(il) + ".";
        const std::string a = p + "self_attn.";
        L.q = f.q(a + "q_proj.weight", qfmt::q4k, QD, D);
        L.k = f.q(a + "k_proj.weight", qfmt::q4k, KVD, D);
        L.v = f.q(a + "v_proj.weight", qfmt::q4k, KVD, D);
        L.o = f.q(a + "o_proj.weight", qfmt::q4k, D, QD);
        L.gate = f.q(a + "gate_proj.weight", qfmt::q4k, QD, D);
        L.rel = f.q(a + "rel_proj.weight", qfmt::q4k, NH * NREL, D);
        L.sink_proj = f.bf16(a + "sink_proj.weight", {NH, D});
        L.q_clip = f.f32(a + "q_clip.weight", {NH, DH});
        L.rel_clip = f.f32(a + "rel_clip.weight", {NH, NREL});
        L.rel_bias = f.f32(a + "rel_bias.weight", {REL_EXT, NREL});
        L.k_conv = f.f32(a + "k_conv.weight", {KVD, NCONV});
        L.v_conv = f.f32(a + "v_conv.weight", {KVD, NCONV});
        L.attn_conv = f.f32(p + "post_attention_convolution.weight", {D, NCONV});
        L.ffn_conv = f.f32(p + "post_mlp_convolution.weight", {D, NCONV});
        if (is_global(il)) {
            L.k_conv_proj = f.bf16(a + "k_conv_proj.weight", {NKV * NCONV, D});
            L.v_conv_proj = f.bf16(a + "v_conv_proj.weight", {NKV * NCONV, D});
            L.qkv_norm[0] = f.f32(p + "q_input_layernorm.weight", {D});
            L.qkv_norm[1] = f.f32(p + "k_input_layernorm.weight", {D});
            L.qkv_norm[2] = f.f32(p + "v_input_layernorm.weight", {D});
            if (il > 0) L.attn_query = f.f32(p + "attention_input.query", {3, D});
        } else {
            L.attn_norm = f.f32(p + "input_layernorm.weight", {D});
            if (il > 0) L.attn_query = f.f32(p + "attention_input.query", {D});
        }
        L.ffn_query = f.f32(p + "mlp_input.query", {D});
        L.ffn_norm = f.f32(p + "post_attention_layernorm.weight", {D});
        L.router = f.f32(p + "mlp.router.weight", {NE, D});
        L.expert_bias = f.f32(p + "mlp.expert_bias", {NE});
        L.gate_up = f.q(p + "mlp.experts.gate_up_proj.weight", qfmt::q4k, NE * 2 * FF, D);
        L.down = f.q(p + "mlp.experts.down_proj.weight", qfmt::q4k, NE * D, FF_PAD);
        if (il >= CORE0 && il < CORE1) {
            L.msg_query = f.f32(p + "loop.message_query", {4, D});
            const char* names[4] = {"q_gate", "k_gate", "v_gate", "ffn_gate"};
            for (int c = 0; c < 4; c++) {
                const std::string g = p + "loop." + names[c] + ".";
                L.ctrl[c].down = f.f32(g + "down", {RANK, 2 * D});
                L.ctrl[c].forget = f.f32(g + "forget", {D, RANK});
                L.ctrl[c].gain = f.f32(g + "gain", {D, RANK});
                ctrl_weights(L.ctrl[c]);   // in bf16 now: the first prompt would convert all 48 on one thread (~40 ms)
            }
        }
    }
}

void model::set_max_length(int max_len) {
    state& s = *s_;
    const bool keep = !s.k.empty() && s.pos <= max_len;
    s.max_len = max_len;
    if (s.k.empty()) {
        s.k.assign(NSLOT, {});
        s.v.assign(NSLOT, {});
        s.conv.assign((size_t)NSLOT * (NCONV - 1) * CONV_CH, 0.0f);
    }
    // SWA rings keep their size; global slots grow or shrink, the used rows stay
    for (int slot = 0; slot < NSLOT; slot++) {
        s.k[slot].resize((size_t)s.rows(slot) * KVD, 0);
        s.v[slot].resize((size_t)s.rows(slot) * KVD, 0);
        s.k[slot].shrink_to_fit();
        s.v[slot].shrink_to_fit();
    }
    if (!keep) reset();
}

void model::reset() {
    s_->pos = 0;
    std::fill(s_->conv.begin(), s_->conv.end(), 0.0f);
}

void model::set_position(int pos) { s_->pos = pos; }

size_t model::fixed_bytes() {
    return (size_t)NSLOT * (NCONV - 1) * CONV_CH * sizeof(float) + (size_t)(NSLOT - NGLOBAL) * (SWA - 1) * 2 * KVD * sizeof(u16);
}

// the fixed part: conv state, then per SWA slot (in slot order) SWA - 1 K rows and SWA - 1 V rows,
// oldest first, the rows before position 0 zero
void model::save_fixed(u8* dst) const {
    const state& s = *s_;
    memcpy(dst, s.conv.data(), s.conv.size() * sizeof(float));
    u16* p = reinterpret_cast<u16*>(dst + s.conv.size() * sizeof(float));
    for (int slot = 0; slot < NSLOT; slot++) {
        if (is_global(slot_layer(slot))) continue;
        for (int kv = 0; kv < 2; kv++) {
            const u16* src = (kv ? s.v : s.k)[slot].data();
            for (int i = 0; i < SWA - 1; i++, p += KVD) {
                const int q = s.pos - (SWA - 1) + i;
                if (q < 0) std::fill(p, p + KVD, 0);
                else memcpy(p, src + s.row(slot, q), KVD * sizeof(u16));
            }
        }
    }
}

void model::load_fixed(const u8* src, int pos) {
    state& s = *s_;
    s.pos = pos;
    memcpy(s.conv.data(), src, s.conv.size() * sizeof(float));
    const u16* p = reinterpret_cast<const u16*>(src + s.conv.size() * sizeof(float));
    for (int slot = 0; slot < NSLOT; slot++) {
        if (is_global(slot_layer(slot))) continue;
        for (int kv = 0; kv < 2; kv++) {
            u16* dst = (kv ? s.v : s.k)[slot].data();
            for (int i = 0; i < SWA - 1; i++, p += KVD) {
                const int q = pos - (SWA - 1) + i;
                if (q >= 0) memcpy(dst + s.row(slot, q), p, KVD * sizeof(u16));
            }
        }
    }
}

namespace {
/// \brief cache slot of the j-th global slot (slot order)
int global_slot(int j) {
    for (int slot = 0, n = 0; slot < NSLOT; slot++)
        if (is_global(slot_layer(slot)) && n++ == j) return slot;
    throw std::out_of_range("inkling2: no global slot " + std::to_string(j));
}
}  // namespace

void model::read_rows(int j, int from, int n, u16* k, u16* v) const {
    const int slot = global_slot(j);
    memcpy(k, s_->k[slot].data() + (size_t)from * KVD, (size_t)n * KVD * sizeof(u16));
    memcpy(v, s_->v[slot].data() + (size_t)from * KVD, (size_t)n * KVD * sizeof(u16));
}

void model::write_rows(int j, int from, int n, const u16* k, const u16* v) {
    const int slot = global_slot(j);
    memcpy(s_->k[slot].data() + (size_t)from * KVD, k, (size_t)n * KVD * sizeof(u16));
    memcpy(s_->v[slot].data() + (size_t)from * KVD, v, (size_t)n * KVD * sizeof(u16));
}

void model::slot_conv(int slot, float* dst) const {
    memcpy(dst, s_->conv.data() + (size_t)slot * (NCONV - 1) * CONV_CH, (size_t)(NCONV - 1) * CONV_CH * sizeof(float));
}

void model::slot_row(int slot, int pos, bool v, u16* dst) const {
    memcpy(dst, (v ? s_->v : s_->k)[slot].data() + s_->row(slot, pos), KVD * sizeof(u16));
}

void model::set_slot_row(int slot, int pos, bool v, const u16* src) {
    memcpy((v ? s_->v : s_->k)[slot].data() + s_->row(slot, pos), src, KVD * sizeof(u16));
}

void model::set_slot_conv(int slot, const float* src) {
    memcpy(s_->conv.data() + (size_t)slot * (NCONV - 1) * CONV_CH, src, (size_t)(NCONV - 1) * CONV_CH * sizeof(float));
}

int model::chunk() {
    static const int c = [] {
        const char* e = std::getenv("INKLING2_CHUNK");
        return e ? std::max(1, std::atoi(e)) : CHUNK;
    }();
    return c;
}

void model::run(const int* ids, int n, float* logits, const u16* image_rows, int image_count, int image_id) {
    if (s_->pos + n > s_->max_len)
        throw std::runtime_error("inkling2: " + std::to_string(s_->pos + n) + " tokens exceed the context length " +
                                 std::to_string(s_->max_len));
    if (image_rows) {
        const int have = (int)std::count(ids, ids + n, image_id);
        if (have != image_count)
            throw std::runtime_error("inkling2: the prompt has " + std::to_string(have) + " image placeholder(s) for " +
                                     std::to_string(image_count) + " image row(s)");
    }
    const int chunk = model::chunk();
    // a prompt: the helper's full pool keeps libgomp's waiters to the short spin, it ends with the prompt
    // (use_core_threads)
    if (n >= 64)
        helper().submit([] {
            use_core_threads(true);
            #pragma omp parallel
            {
            }
        });
    for (int i = 0; i < n; i += chunk) {
        const int T = std::min(chunk, n - i);
        step(ids + i, T, i + T == n ? logits : nullptr, image_rows, image_id);
        if (image_rows) image_rows += (size_t)std::count(ids + i, ids + i + T, image_id) * D;
    }
    release_helper();
}

void model::step(const int* ids, int T, float* logits, const u16* image_rows, int image_id) {
    use_core_threads();
    const weights& w = *w_;
    state& s = *s_;
    const int p0 = s.pos;

    act cur = make_act((size_t)T * D);
    // the image rows enter as they are; the k-th placeholder of the step takes the k-th row
    std::vector<int> image_at;
    if (image_rows) {
        for (int t = 0; t < T; t++)
            if (ids[t] == image_id) image_at.push_back(t);
    }
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        float e[D];
        const u16* row = w.embed + (size_t)ids[t] * D;
        for (int c = 0; c < D; c++) e[c] = bf2f(row[c]);
        rms_norm(cur->data() + (size_t)t * D, e, w.embed_norm, D);
    }
    for (size_t k = 0; k < image_at.size(); k++) {
        float* dst = cur->data() + (size_t)image_at[k] * D;
        const u16* src = image_rows + k * D;
        for (int c = 0; c < D; c++) dst[c] = bf2f(src[c]);
    }

    std::vector<act> blocks;
    for (int il = 0; il < CORE0; il++) cur = layer(w.layer[il], il, il, il, s, p0, T, cur, blocks, nullptr);

    // every pass starts from the prelude output; passes 2.. get messages from the previous pass's
    // core blocks (after its first) and its output
    const act prelude = cur;
    const std::vector<act> prelude_blocks = blocks;
    for (int pass = 0; pass < NPASS; pass++) {
        std::vector<act> bank;
        act bank_out;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
        }
        act next_msgs;   // the next layer's messages, mixed while the NPU runs this layer's experts
        for (int ip = CORE0; ip < CORE1; ip++) {
            const layer_weights& L = w.layer[ip];
            act msgs;
            auto tp = std::chrono::steady_clock::now();
            if (pass > 0) msgs = next_msgs ? next_msgs : mix(bank, bank_out, L.msg_query, 4, T);
            next_msgs = nullptr;
            prof_lap(tp, P_X_MSGMIX);
            if (pass > 0 && ip + 1 < CORE1 && s.offload)
                s.offload->idle = [&, ip] { next_msgs = mix(bank, bank_out, w.layer[ip + 1].msg_query, 4, T); };
            cur = layer(L, ip, ip, layer_slot(ip, pass), s, p0, T, cur, blocks, msgs ? msgs->data() : nullptr);
            if (s.offload) s.offload->idle = nullptr;   // not run (the host took the experts): mixed at the next layer
        }
    }
    for (int il = CORE1; il < NL; il++) cur = layer(w.layer[il], il, il, il, s, p0, T, cur, blocks, nullptr);
    s.pos += T;

    if (!logits) return;
    auto tp = std::chrono::steady_clock::now();
    // output: the last token only
    std::vector<act> last_blocks;
    for (const auto& b : blocks) {
        act a = make_act(D);
        memcpy(a->data(), b->data() + (size_t)(T - 1) * D, D * sizeof(float));
        last_blocks.push_back(a);
    }
    act last = make_act(D);
    memcpy(last->data(), cur->data() + (size_t)(T - 1) * D, D * sizeof(float));
    act x = mix(last_blocks, last, w.final_query, 1, 1);
    rms_norm(x->data(), x->data(), w.norm, D);
    if (decode_trace && T == 1) decode_trace(NSLOT, "final", x->data(), D);
    lm_head(w, x->data(), logits);
    prof_lap(tp, P_LMHEAD);
}

}  // namespace inkling2_host
