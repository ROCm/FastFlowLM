// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_lm_head.cpp
/// \brief lm_head on the NPU (see inkling2_lm_head.hpp). The transfers follow the Qwen3.5 engine's gen_lm_head_seq:
///        the cores count on exactly these sizes.

#include "inkling2_lm_head.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#if !defined(_WIN32)
#include <sys/mman.h>
#endif

#if !defined(FLM_USE_HRX)   // see inkling2_fused.hpp

using namespace inkling2_host;

namespace {
constexpr int COLS = 8, M_PER_COL = 128;
constexpr npu_tiles IT[COLS] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
// IT0's MM2S 0 carries x and a weight stream: packet ids as in the xclbin (X_PKT_ID / W_PKT_ID)
constexpr int X_PKT_ID = 0, W_PKT_ID = 1, NO_PKT = -1;
constexpr int LOOKAHEAD = 3;   // rounds of shim BDs in flight; the y waits lag the y issues by this much
constexpr size_t COL_BYTES = (size_t)M_PER_COL / QROWS * (D / QCOLS) * Q80_BYTES;   // a column's rows of a round

// logits[VOCAB] = SOFTCAP tanh(y / SOFTCAP) (y has whole vectors past VOCAB: ROWS)
void softcap(const uint16_t* y, float* logits) {
    const __m512 inv = _mm512_set1_ps(1.0f / SOFTCAP), cap = _mm512_set1_ps(SOFTCAP);
    int i = 0;
    for (; i + 16 <= VOCAB; i += 16) _mm512_storeu_ps(logits + i, _mm512_mul_ps(cap, tanh16(_mm512_mul_ps(load_bf16x16(y + i), inv))));
    if (i < VOCAB)
        _mm512_mask_storeu_ps(logits + i, (__mmask16)((1u << (VOCAB - i)) - 1),
                              _mm512_mul_ps(cap, tanh16(_mm512_mul_ps(load_bf16x16(y + i), inv))));
}
static_assert(inkling2_lm_head::ROWS >= VOCAB + 16);
}  // namespace

void inkling2_lm_head::finish(float* logits) {
    const auto t0 = std::chrono::steady_clock::now();
    this->y.sync_from_device();
    softcap(this->y.data(), logits);
    this->t_fin += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

inkling2_lm_head::inkling2_lm_head(npu_xclbin_manager* npu, const std::string& xclbin, const weights& w) {
    this->mgr = npu->register_xclbin(xclbin);
    this->init(npu, w);
    this->run_ = this->app.create_run(this->y, this->wb, this->x);
}

inkling2_lm_head::inkling2_lm_head(npu_xclbin_manager* npu, npu_app_manager* mgr, const weights& w, buffer<uint8_t>* xsrc,
                                   size_t x_off)
    : xsrc(xsrc), x_off(x_off) {
    if (x_off % 4 || x_off + (size_t)4 * D > xsrc->size()) throw std::runtime_error("inkling2 lm_head: x outside its buffer");
    this->mgr = mgr;
    this->init(npu, w);
}

void inkling2_lm_head::init(npu_xclbin_manager* npu, const weights& w) {
    const qmat& q = w.lm_head;
    if (q.fmt != qfmt::q80 || q.K != D || q.N != VOCAB_PAD) throw std::runtime_error("inkling2 lm_head: need Q8_0 [VOCAB_PAD] x D");
    this->app = this->mgr->create_app();
    const int rb_all = ROWS / QROWS, rb_real = q.row_blocks(), cbs = q.col_blocks();
    const size_t wbytes = (size_t)rb_all * cbs * Q80_BYTES;
#if !defined(_WIN32)   // 2 MB pages: Linux only
    if (getenv("INKLING2_LMH_HUGE") && atoi(getenv("INKLING2_LMH_HUGE"))) {
        constexpr size_t HP = (size_t)2 << 20;
        this->huge.n = (wbytes + HP - 1) / HP * HP;
        void* p = mmap(nullptr, this->huge.n + HP, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) throw std::runtime_error("inkling2 lm_head: mmap failed");
        // keep the 2 MB-aligned span, give back the rest
        const uintptr_t a = ((uintptr_t)p + HP - 1) / HP * HP;
        if (a > (uintptr_t)p) munmap(p, a - (uintptr_t)p);
        if ((uintptr_t)p + HP > a) munmap((void*)(a + this->huge.n), (uintptr_t)p + HP - a);
        this->huge.p = (void*)a;
        madvise(this->huge.p, this->huge.n, MADV_HUGEPAGE);
        memset(this->huge.p, 0, this->huge.n);
        this->huge_bo = std::make_unique<xrt::ext::bo>(*this->mgr->get_device(), this->huge.p, this->huge.n);
        this->wb = buffer<uint8_t>(*this->huge_bo);
    } else
#endif
    {
        this->wb = npu->create_bo_buffer<uint8_t>(wbytes);
    }
    this->y = npu->create_bo_buffer<uint16_t>(ROWS);
    this->x = npu->create_bo_buffer<uint16_t>(2 * D);
    // per 4 row blocks (a column's 128 rows): stream a (row blocks 0-1), then stream b (2-3), each column block by
    // column block with its two row blocks together
    uint8_t* d = this->wb.data();
    for (int r = 0; r < rb_all; r += 4)
        for (int half = 0; half < 2; half++)
            for (int c = 0; c < cbs; c++)
                for (int i = 0; i < 2; i++, d += Q80_BYTES) {
                    const int rb = r + 2 * half + i;
                    if (rb < rb_real) memcpy(d, q.block(rb, c), Q80_BYTES);
                    else memset(d, 0, Q80_BYTES);
                }
    memset(this->x.data(), 0, 2 * D * sizeof(uint16_t));
    this->wb.sync_to_device();
    this->x.sync_to_device();
    if (const char* e = getenv("INKLING2_LMH_RPB")) this->rpb = std::max(1, atoi(e));
    this->gen_seq();
}

void inkling2_lm_head::gen_seq() {
    constexpr int y_arg = 0, w_arg = 1, x_arg = 2;
    const int rounds = ROWS / M_PER_ROUND;
    const uint32_t W_PER_COL = COL_BYTES / 2, W_PER_ROUND = W_PER_COL * COLS, W_PER_STREAM = W_PER_COL / 2;   // bf16 units
    npu_sequence* seq = this->app.seq();
    seq->clear_cmds();
    seq->npu_dma_memcpy_nd(2, x_arg, MM2S, IT[0], npu_bd_id(bd_0), it_channel_0, {0, 0, 0, (uint32_t)(this->x_off / 2)}, {1, 1, 1, (uint32_t)D * 2},
                           {0, 0, 0, 1}, X_PKT_ID, 0, false, aggressive_cache);
    // a step: rounds [r0, r0 + n) with one descriptor per stream; a weight stream's round is W_PER_STREAM bf16 as
    // 51 rows of 1536 (a shim dimension wraps at 1024 words)
    constexpr uint32_t W_IN = 1536;
    static_assert(COL_BYTES / 4 % W_IN == 0);
    const int steps = (rounds + this->rpb - 1) / this->rpb;
    for (int st = 0; st < steps; st++) {
        const int r = st * this->rpb, n = std::min(this->rpb, rounds - r);
        const int bo = (st % (LOOKAHEAD + 1)) * 3;
        const npu_bd_id bd_y = npu_bd_id(bd_1 + bo), bd_wa = npu_bd_id(bd_2 + bo), bd_wb = npu_bd_id(bd_3 + bo);
        for (int col = 0; col < COLS; col++) {
            const uint32_t wo = (uint32_t)r * W_PER_ROUND + col * W_PER_COL, yo = r * M_PER_ROUND + col * M_PER_COL;
            if (n == 1) {
                seq->npu_dma_memcpy_nd(2, y_arg, S2MM, IT[col], bd_y, it_channel_0, {0, 0, 0, yo}, {1, 1, 1, (uint32_t)M_PER_COL},
                                       {0, 0, 0, 1}, NO_PKT, 0, true, aggressive_cache);
                seq->npu_dma_memcpy_nd(2, w_arg, MM2S, IT[col], bd_wa, it_channel_1, {0, 0, 0, wo}, {1, 1, 1, W_PER_STREAM},
                                       {0, 0, 0, 1}, NO_PKT, 0, false, aggressive_cache);
                seq->npu_dma_memcpy_nd(2, w_arg, MM2S, IT[col], bd_wb, it_channel_0, {0, 0, 0, wo + W_PER_STREAM},
                                       {1, 1, 1, W_PER_STREAM}, {0, 0, 0, 1}, col == 0 ? W_PKT_ID : NO_PKT, 0, false,
                                       aggressive_cache);
            } else {
                const uint32_t nn = (uint32_t)n, wr = W_PER_STREAM / W_IN;
                seq->npu_dma_memcpy_nd(2, y_arg, S2MM, IT[col], bd_y, it_channel_0, {0, 0, 0, yo}, {1, 1, nn, (uint32_t)M_PER_COL},
                                       {0, 0, (uint32_t)M_PER_ROUND, 1}, NO_PKT, 0, true, aggressive_cache);
                seq->npu_dma_memcpy_nd(2, w_arg, MM2S, IT[col], bd_wa, it_channel_1, {0, 0, 0, wo}, {1, nn, wr, W_IN},
                                       {0, W_PER_ROUND, W_IN, 1}, NO_PKT, 0, false, aggressive_cache);
                seq->npu_dma_memcpy_nd(2, w_arg, MM2S, IT[col], bd_wb, it_channel_0, {0, 0, 0, wo + W_PER_STREAM},
                                       {1, nn, wr, W_IN}, {0, W_PER_ROUND, W_IN, 1}, col == 0 ? W_PKT_ID : NO_PKT, 0, false,
                                       aggressive_cache);
            }
            if (st > LOOKAHEAD - 1) seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
        }
    }
    for (int col = 0; col < COLS; col++)
        for (int i = 0; i < std::min(LOOKAHEAD, steps); i++) seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
    seq->cmds2seq();
}

inkling2_lm_head::huge_mem::~huge_mem() {
#if !defined(_WIN32)
    if (this->p) munmap(this->p, this->n);
#endif
}

bool inkling2_lm_head::run(const uint16_t* xv, float* logits) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    memcpy(this->x.data(), xv, D * sizeof(uint16_t));
    this->x.sync_to_device();
    const auto ts = clk::now();
    this->run_.start();
    ert_cmd_state st = ERT_CMD_STATE_TIMEOUT;
    try { st = this->run_.wait(std::chrono::milliseconds(5000)); } catch (const std::exception& e) { std::fprintf(stderr, "inkling2 lm_head: wait: %s\n", e.what()); }
    if (st != ERT_CMD_STATE_COMPLETED) {
        // a stalled array: keep the context past the driver's job timeout before anything tears it down
        if (FILE* f = getenv("INKLING2_SUSPECT_FILE") ? fopen(getenv("INKLING2_SUSPECT_FILE"), "w") : nullptr) {
            std::fprintf(f, "inkling2 lm_head run state %d\n", (int)st);
            fclose(f);
        }
        std::fprintf(stderr, "inkling2 lm_head: RUN DID NOT COMPLETE (state %d): holding the context 90 s\n", (int)st);
        std::this_thread::sleep_for(std::chrono::seconds(90));
        return false;
    }
    const auto td = clk::now();
    this->y.sync_from_device();
    const auto t1 = clk::now();
    this->t_dev += std::chrono::duration<double, std::milli>(td - ts).count();
    this->t_sync += std::chrono::duration<double, std::milli>(t1 - td).count();
    softcap(this->y.data(), logits);
    const auto t2 = clk::now();
    this->t_run += std::chrono::duration<double, std::milli>(t1 - t0).count();
    this->t_post += std::chrono::duration<double, std::milli>(t2 - t1).count();
    this->n_run++;
    return true;
}

#endif   // !FLM_USE_HRX
