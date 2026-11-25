// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "qwen3_asr_prefill_npu.hpp"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "../audio/qwen3_asr_audio_simd.hpp"
#include "../omp_threads.hpp"

namespace seqg = qwen3_asr_prefill_seq;
namespace simd = qwen3_asr_audio_simd;

namespace {

/// two bfp16 blocks (o0: lanes 0..7, o1: lanes 8..15), each 8 values sharing the largest exponent of the 8, rounded to
/// nearest (the same integer steps as the audio encoder's encode2)
inline void encode2(__m512 f, uint8_t* o0, uint8_t* o1) {
    const __m512i one = _mm512_set1_epi32(1);
    __m512i u = _mm512_castps_si512(f);
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
}

/// rows [0, rows) of x [rows][Din] -> X as the cores take it, for rows_padded rows (the rest 0) and nb slabs per pass: per pass
/// and core row a block of [512 k round][256 k slice 2][16-row group 8 / nb][128 k half 2][k tile 16][8-row tile 2] of 8 rows
/// x 9 B (seqg::gen_dequant_mm_mix)
void pack_x(uint8_t* out, const float* x, int rows, uint32_t rows_padded, uint32_t Din, uint32_t nb) {
    const uint32_t ms = 8 / nb;
    const uint32_t m_pass = seqg::mix_pass_rows(nb);
    const uint32_t k_rounds = Din / 512;
    const size_t round_bytes = (size_t)4 * ms * seqg::MIX_X_PACKET_BYTES;
    const int blocks = (int)(rows_padded / m_pass * seqg::MIX_CORE_ROWS);
#pragma omp parallel for collapse(2) schedule(static) num_threads(qwen3_asr_cpu::asr_threads())
    for (int b = 0; b < blocks; b++) {
        for (int kr = 0; kr < (int)k_rounds; kr++) {
            const uint32_t mp = b / seqg::MIX_CORE_ROWS, r = b % seqg::MIX_CORE_ROWS;
            uint8_t* p = out + ((size_t)b * k_rounds + kr) * round_bytes;
            for (uint32_t j = 0; j < 2; j++)
                for (uint32_t i = 0; i < ms; i++) {
                    const int r0 = (int)(mp * m_pass + r * 16 * ms + i * 16);
                    if (r0 >= rows) {   // a group past the rows is all zero bytes
                        std::memset(p, 0, 2 * 16 * 2 * 72);
                        p += 2 * 16 * 2 * 72;
                        continue;
                    }
                    for (uint32_t v = 0; v < 2; v++)
                        for (uint32_t kt = 0; kt < 16; kt++)
                            for (int z = 0; z < 2; z++) {
                                const size_t k0 = (size_t)kr * 512 + j * 256 + v * 128 + kt * 8;
                                for (int row = r0 + z * 8; row < r0 + z * 8 + 8; row += 2, p += 18) {
                                    const __m256 a = row < rows ? _mm256_loadu_ps(x + (size_t)row * Din + k0) : _mm256_setzero_ps();
                                    const __m256 c = row + 1 < rows ? _mm256_loadu_ps(x + (size_t)(row + 1) * Din + k0) : _mm256_setzero_ps();
                                    encode2(_mm512_insertf32x8(_mm512_castps256_ps512(a), c, 1), p, p + 9);
                                }
                            }
                }
        }
    }
}

/// byte offset in pack_x's layout of the 8 values at k (a multiple of 8) of row `row`
inline size_t x_offset(int row, uint32_t k, uint32_t Din, uint32_t nb) {
    const uint32_t ms = 8 / nb, m_pass = seqg::mix_pass_rows(nb), k_rounds = Din / 512;
    const uint32_t mp = row / m_pass, rr = row % m_pass, b = mp * seqg::MIX_CORE_ROWS + rr / (16 * ms);
    const uint32_t kr = k / 512, j = k / 256 % 2, v = k / 128 % 2, kt = k / 8 % 16;
    return ((size_t)b * k_rounds + kr) * 4 * ms * seqg::MIX_X_PACKET_BYTES + (j * ms + rr / 16 % ms) * 2 * 16 * 2 * 72 +
           ((v * 16 + kt) * 2 + rr / 8 % 2) * 72 + rr % 8 / 2 * 18 + rr % 2 * 9;
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

qwen3_asr_prefill_npu::qwen3_asr_prefill_npu(npu_app_manager* mgr, const layout& l, std::vector<buffer<u8>>& weights, npu_app_manager* attn_mgr)
    : mgr_(mgr), l_(l), weights_(weights), io_app_(mgr->create_app()), attn_mgr_(attn_mgr) {
    if (attn_mgr_) {
        mha_type_t ty = mha_d128_q2;
        mha_ = std::make_unique<MHA>(ty, l.DK / 128);
        attn_app_ = std::make_unique<npu_app>(attn_mgr_->create_app());
        const char* min_env = std::getenv("QWEN3_ASR_NPU_ATTENTION_MIN");
        const int64_t rows = min_env ? std::max(0, std::atoi(min_env)) : 1024;
        attn_min_work_ = rows * (rows + 1);
    }
    using seqg::MIX_FMT_Q4_1;
    using seqg::MIX_FMT_Q8_0;
    if (l.D % 512 || l.DQ % 512 || l.I % 512) throw std::runtime_error("qwen3_asr_prefill_npu: dimensions must be multiples of 512");
    calls_[KVQ] = {(uint32_t)l.D, (uint32_t)(2 * l.DK + l.DQ),
                   {{l.k, (uint32_t)l.DK, MIX_FMT_Q4_1}, {l.v, (uint32_t)l.DK, MIX_FMT_Q8_0}, {l.q, (uint32_t)l.DQ, MIX_FMT_Q4_1}}};
    calls_[O] = {(uint32_t)l.DQ, (uint32_t)l.D, {{l.o, (uint32_t)l.D, MIX_FMT_Q4_1}}};
    calls_[UP_GATE] = {(uint32_t)l.D, (uint32_t)(2 * l.I), {{l.up, (uint32_t)(2 * l.I), MIX_FMT_Q4_1}}};   // up and gate, 512 rows each in turn
    calls_[DOWN] = {(uint32_t)l.I, (uint32_t)l.D, {{l.down, (uint32_t)l.D, MIX_FMT_Q8_0}}};
    prof_ = std::getenv("QWEN3_ASR_PROF") != nullptr;
}

npu_app& qwen3_asr_prefill_npu::app_for(call_id c, uint32_t rows_padded) {
    const auto key = std::make_pair((int)c, rows_padded);
    auto it = apps_.find(key);
    if (it != apps_.end()) return *it->second;
    auto app = std::make_unique<npu_app>(mgr_->create_app());
    // generated for the padded rows: they pick the same nb as the rows themselves, so every prompt of a size shares it
    seqg::gen_dequant_mm_mix(app->seq(), rows_padded, calls_[c].din, calls_[c].bands, 0, 0);
    return *apps_.emplace(key, std::move(app)).first->second;
}

void qwen3_asr_prefill_npu::run(int layer, const std::vector<job>& jobs) {
    const size_t n = jobs.size();
    // every run is created (and its app set up) before the first starts; the two slots fit the largest job
    std::vector<uint32_t> rp(n);
    size_t x_bytes = 0, y_elems = 0;
    for (size_t j = 0; j < n; j++) {
        const call_t& call = calls_[jobs[j].c];
        rp[j] = seqg::mix_rows_padded(jobs[j].rows, call.dout);
        if (!jobs[j].packed) x_bytes = std::max(x_bytes, (size_t)rp[j] * call.din / 8 * 9);
        y_elems = std::max(y_elems, (size_t)rp[j] * call.dout);
    }
    for (int s = 0; s < 2; s++) {
        if (x_[s].size() < x_bytes) x_[s] = io_app_.create_bo_buffer<u8>(x_bytes);
        if (y_[s].size() < y_elems) y_[s] = io_app_.create_bo_buffer<u16>(y_elems);
    }
    std::vector<flm_rt::run> runs;
    runs.reserve(n);
    for (size_t j = 0; j < n; j++)
        runs.push_back(app_for(jobs[j].c, rp[j]).create_run(y_[j % 2], jobs[j].packed ? *jobs[j].packed : x_[j % 2], weights_.at(layer)));

    const auto pack = [&](size_t j) {
        const job& J = jobs[j];
        const call_t& call = calls_[J.c];
        const auto t = std::chrono::steady_clock::now();
        if (J.packed) J.packed->sync_to_device_bytes((size_t)rp[j] * call.din / 8 * 9);
        else {
            pack_x(x_[j % 2].data(), J.x, J.rows, rp[j], call.din, seqg::mix_nb(J.rows, call.dout));
            x_[j % 2].sync_to_device_bytes((size_t)rp[j] * call.din / 8 * 9);
        }
        if (prof_) prof_ms_[J.c][0] += ms_since(t);
    };
    const auto unpack = [&](size_t j) {
        const job& J = jobs[j];
        const auto t = std::chrono::steady_clock::now();
        J.unpack(y_[j % 2].data(), J.r0, J.rows);
        if (prof_) prof_ms_[J.c][2] += ms_since(t);
    };
    std::chrono::steady_clock::time_point started;
    for (size_t j = 0; j <= n; j++) {
        const bool early = j < n && !(j > 0 && jobs[j].needs_prev);
        if (early) pack(j);   // while the NPU runs job j - 1
        if (j > 0) {
            const job& J = jobs[j - 1];
            flm::npu::wait_for_completion(runs[j - 1], "qwen3_asr_prefill_npu");
            if (prof_) {
                prof_ms_[J.c][1] += ms_since(started);
                prof_flop_[J.c] += 2.0 * rp[j - 1] * calls_[J.c].din * calls_[J.c].dout;
            }
            y_[(j - 1) % 2].sync_from_device_bytes((size_t)J.rows * calls_[J.c].dout * sizeof(u16));
        }
        if (j < n && !early) {
            unpack(j - 1);
            pack(j);
        }
        if (j < n) {
            started = std::chrono::steady_clock::now();
            runs[j].start();
        }
        if (j > 0 && (early || j == n)) unpack(j - 1);   // while the NPU runs job j
    }
}

void qwen3_asr_prefill_npu::qkv(int layer, const float* x, int n, float* q, float* k, float* v) {
    std::lock_guard<std::mutex> lock(mu_);
    const int DK = l_.DK, DQ = l_.DQ, dout = (int)calls_[KVQ].dout;
    const auto unpack = [=](const uint16_t* Y, int r0, int rows) {
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
        for (int r = 0; r < rows; r++) {
            const uint16_t* y = Y + (size_t)r * dout;   // [K | V | Q]
            const size_t row = (size_t)(r0 + r);
            for (int c = 0; c < DK; c += 16) {
                _mm512_storeu_ps(k + row * DK + c, simd::load_bf16(y + c));
                _mm512_storeu_ps(v + row * DK + c, simd::load_bf16(y + DK + c));
            }
            for (int c = 0; c < DQ; c += 16) _mm512_storeu_ps(q + row * DQ + c, simd::load_bf16(y + 2 * DK + c));
        }
    };
    std::vector<job> jobs;
    for (int r0 = 0; r0 < n; r0 += step(n)) jobs.push_back({KVQ, r0, std::min(step(n), n - r0), x + (size_t)r0 * l_.D, unpack});
    run(layer, jobs);
}

void qwen3_asr_prefill_npu::o(int layer, const float* att, int n, float* out) {
    std::lock_guard<std::mutex> lock(mu_);
    const int D = l_.D;
    const auto unpack = [=](const uint16_t* y, int r0, int rows) {
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < D; c += 16) _mm512_storeu_ps(out + (size_t)(r0 + r) * D + c, simd::load_bf16(y + (size_t)r * D + c));
    };
    std::vector<job> jobs;
    for (int r0 = 0; r0 < n; r0 += step(n)) jobs.push_back({O, r0, std::min(step(n), n - r0), att + (size_t)r0 * l_.DQ, unpack});
    run(layer, jobs);
}

void qwen3_asr_prefill_npu::mlp(int layer, const float* x, int n, float* out) {
    std::lock_guard<std::mutex> lock(mu_);
    const int D = l_.D, I = l_.I;
    // down's X, packed by the up_gate unpack: two buffers in turn (down s runs before up_gate s + 2 is unpacked)
    const size_t xd_bytes = (size_t)seqg::mix_rows_padded(step(n), D) * I / 8 * 9;
    for (int s = 0; s < 2; s++)
        if (xd_[s].size() < xd_bytes) xd_[s] = io_app_.create_bo_buffer<u8>(xd_bytes);
    // fused row c is up row (c / 1024) 512 + c % 1024 when c % 1024 < 512, else that gate row (512 earlier); silu and the
    // product are rounded to bf16 as on the CPU path, then packed as pack_x would (padding rows 0); a task is 16 rows (one
    // row group: whole 72-byte tiles) by 512 k
    const auto silu_up = [=](uint8_t* xd, const uint16_t* Y, int rows) {
        const uint32_t nb = seqg::mix_nb(rows, D), ms = 8 / nb;
        const int groups = (int)seqg::mix_rows_padded(rows, D) / 16;
#pragma omp parallel for collapse(2) schedule(dynamic, 2) num_threads(qwen3_asr_cpu::asr_threads())
        for (int kr = 0; kr < I / 512; kr++)
            for (int g = 0; g < groups; g++) {
                uint8_t* base = xd + x_offset(g * 16, kr * 512, I, nb);
                if (g * 16 >= rows) {   // a group past the rows: its two 256 k slices
                    std::memset(base, 0, 2 * 16 * 2 * 72);
                    std::memset(base + ms * 2 * 16 * 2 * 72, 0, 2 * 16 * 2 * 72);
                    continue;
                }
                for (int rl = 0; rl < 16; rl++) {
                    const int r = g * 16 + rl;
                    uint8_t* p = base + rl / 8 * 72 + rl % 8 / 2 * 18 + rl % 2 * 9;
                    if (r >= rows) {   // padding: zero blocks
                        for (int t = 0; t < 64; t++) std::memset(p + t / 32 * ms * 2 * 16 * 2 * 72 + t % 32 * 144, 0, 9);
                        continue;
                    }
                    const uint16_t* y = Y + (size_t)r * 2 * I + kr * 1024;
                    for (int c = 0; c < 512; c += 16, p += 2 * 144) {
                        if (c == 256) p += (ms - 1) * 2 * 16 * 2 * 72;   // the second 256 k slice follows this row's ms groups
                        const __m512 u = simd::load_bf16(y + c), gt = simd::load_bf16(y + c + 512);
                        const __m512 s = simd::round_bf16(_mm512_div_ps(gt, _mm512_add_ps(_mm512_set1_ps(1.f), simd::exp(_mm512_sub_ps(_mm512_setzero_ps(), gt)))));
                        encode2(simd::round_bf16(_mm512_mul_ps(s, u)), p, p + 144);
                    }
                }
            }
    };
    const auto unpack = [=](const uint16_t* y, int r0, int rows) {
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < D; c += 16) _mm512_storeu_ps(out + (size_t)(r0 + r) * D + c, simd::load_bf16(y + (size_t)r * D + c));
    };
    // up_gate 0, up_gate 1, down 0, up_gate 2, down 1, ..., down last: a down job's input is written by its up_gate job's
    // unpack, which is two jobs earlier, or (one step) the job just before
    std::vector<job> jobs;
    const int steps = (n + step(n) - 1) / step(n);
    const auto ug = [&](int s) {
        const int r0 = s * step(n);
        uint8_t* xd = xd_[s % 2].data();
        jobs.push_back({UP_GATE, r0, std::min(step(n), n - r0), x + (size_t)r0 * D, [=](const uint16_t* y, int, int rows) { silu_up(xd, y, rows); }});
    };
    const auto dn = [&](int s) {
        const int r0 = s * step(n);
        const bool prev = jobs.back().c == UP_GATE && jobs.back().r0 == r0;
        jobs.push_back({DOWN, r0, std::min(step(n), n - r0), nullptr, unpack, prev, &xd_[s % 2]});
    };
    ug(0);
    for (int s = 1; s < steps; s++) {
        ug(s);
        dn(s - 1);
    }
    dn(steps - 1);
    run(layer, jobs);
    if (prof_ && layer + 1 == (int)weights_.size()) report(n);
}

bool qwen3_asr_prefill_npu::attention(int /*layer*/, const float* q, int pos0, int n, const uint16_t* k, const uint16_t* v, float* att) {
    if (!attn_mgr_ || (int64_t)n * (2 * (int64_t)pos0 + n + 1) < attn_min_work_) return false;
    std::lock_guard<std::mutex> lock(mu_);
    const int DQ = l_.DQ, DK = l_.DK, half = DK / 2;
    const int chunk = mha_->get_chunk_size(), L = pos0 + n;
    const int L0 = pos0 / chunk * chunk, L1 = (L + chunk - 1) / chunk * chunk, rows = L1 - L0, off = pos0 - L0;
    auto t = std::chrono::steady_clock::now();
    if ((uint32_t)L0 != attn_l0_ || (uint32_t)L1 != attn_l1_) {
        mha_->generate_mha_sequence(attn_app_->seq(), L0, L1, L1, false);   // MAX_L = L1: the bands are L1 rows apart
        attn_l0_ = L0;
        attn_l1_ = L1;
    }
    if (attn_q_.size() < (size_t)rows * DQ) {
        attn_q_ = attn_app_->create_bo_buffer<u16>((size_t)rows * DQ);
        attn_y_ = attn_app_->create_bo_buffer<u16>((size_t)rows * DQ);
    }
    if (attn_kv_.size() < (size_t)L1 * 2 * DK) attn_kv_ = attn_app_->create_bo_buffer<u16>((size_t)L1 * 2 * DK);
    // q rows (bf16 values: the top halves), 0 outside the prompt; cache rows split into the bands, 0 past the last row (the
    // dropped rows' keys: finite, so their zero weights stay zero)
    uint16_t* Q = attn_q_.data();
    uint16_t* KV = attn_kv_.data();
    const size_t band = (size_t)L1 * half;
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
    for (int r = 0; r < rows; r++) {
        uint16_t* d = Q + (size_t)r * DQ;
        if (r < off || r >= off + n) {
            std::memset(d, 0, (size_t)DQ * sizeof(u16));
            continue;
        }
        const float* s = q + (size_t)(r - off) * DQ;
        for (int c = 0; c < DQ; c += 16)
            _mm256_storeu_si256((__m256i*)(d + c), _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(_mm512_loadu_ps(s + c)), 16)));
    }
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
    for (int j = 0; j < L1; j++) {
        uint16_t* d[4] = {KV + (size_t)j * half, KV + band + (size_t)j * half, KV + 2 * band + (size_t)j * half, KV + 3 * band + (size_t)j * half};
        if (j >= L) {
            for (int b = 0; b < 4; b++) std::memset(d[b], 0, (size_t)half * sizeof(u16));
            continue;
        }
        std::memcpy(d[0], k + (size_t)j * DK, (size_t)half * sizeof(u16));
        std::memcpy(d[1], k + (size_t)j * DK + half, (size_t)half * sizeof(u16));
        std::memcpy(d[2], v + (size_t)j * DK, (size_t)half * sizeof(u16));
        std::memcpy(d[3], v + (size_t)j * DK + half, (size_t)half * sizeof(u16));
    }
    attn_q_.sync_to_device_bytes((size_t)rows * DQ * sizeof(u16));
    attn_kv_.sync_to_device_bytes((size_t)L1 * 2 * DK * sizeof(u16));
    if (prof_) {
        prof_ms_[CALLS][0] += ms_since(t);
        t = std::chrono::steady_clock::now();
    }
    (*attn_app_)(attn_y_, attn_q_, attn_kv_);
    if (prof_) {
        prof_ms_[CALLS][1] += ms_since(t);
        // QK and PV over each real row's causal length: 2 x 2 x DQ x (pos0 + i + 1) for i in [0, n)
        prof_flop_[CALLS] += 4.0 * DQ * ((double)n * pos0 + (double)n * (n + 1) / 2);
        t = std::chrono::steady_clock::now();
    }
    attn_y_.sync_from_device_bytes((size_t)(off + n) * DQ * sizeof(u16));
    const uint16_t* Y = attn_y_.data() + (size_t)off * DQ;
#pragma omp parallel for num_threads(qwen3_asr_cpu::asr_threads())
    for (int r = 0; r < n; r++)
        for (int c = 0; c < DQ; c += 16) _mm512_storeu_ps(att + (size_t)r * DQ + c, simd::load_bf16(Y + (size_t)r * DQ + c));
    if (prof_) prof_ms_[CALLS][2] += ms_since(t);
    return true;
}

void qwen3_asr_prefill_npu::report(int n) {
    static const char* names[CALLS + 1] = {"kvq", "o", "up_gate", "down", "attn"};
    double npu = 0, flop = 0;
    for (int c = 0; c <= CALLS; c++) {
        if (c == CALLS && prof_ms_[c][1] == 0) break;   // the CPU did the attention
        std::fprintf(stderr, "[prefill npu] n=%d %-7s pack %.1f npu %.1f unpack %.1f ms, %.2f TOPS\n", n, names[c], prof_ms_[c][0], prof_ms_[c][1],
                     prof_ms_[c][2], prof_ms_[c][1] > 0 ? prof_flop_[c] / prof_ms_[c][1] * 1e-9 : 0.0);
        npu += prof_ms_[c][1];
        flop += prof_flop_[c];
        prof_ms_[c][0] = prof_ms_[c][1] = prof_ms_[c][2] = 0;
        prof_flop_[c] = 0;
    }
    std::fprintf(stderr, "[prefill npu] n=%d all layers: npu %.1f ms, %.2f TOPS (padded rows)\n", n, npu, npu > 0 ? flop / npu * 1e-9 : 0.0);
}
