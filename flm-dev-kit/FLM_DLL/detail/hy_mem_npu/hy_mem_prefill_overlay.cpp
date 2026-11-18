// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hy_mem_prefill_overlay.cpp
/// \brief The prefill overlay: switching the array between its ops, and the ops' runs
/// \note gen_switch / gen_dispatch / gen_attn_seq are the ones the overlay's own test
///       checks bit for bit against the standalone xclbins
///       (FLM_Xclbin/Hy_mem_30BA3B/prefill_overlay/host/overlay_host.hpp).
#include <cstdint>
#include <cstdlib>
#include <set>
#include <tuple>
#include <vector>
#include "hy_mem_prefill_overlay.hpp"

// Each op's configuration (routes, BDs, lock values, chain starts), generated from
// the overlay build by FLM_Xclbin/Hy_mem_30BA3B/prefill_overlay/tools/extract_txn2.py
// (its host/op_configs.hpp). Copy it again whenever the overlay is rebuilt.
namespace hy_overlay_cfg {
#include "hy_mem_overlay_configs.hpp"
}

namespace {
using namespace hy_overlay_cfg;

// must match the overlay's iron/layout.py
constexpr uint32_t RTP_ADDR = 61696;                   // every core's RTP block
constexpr int RTP_OP = 0, RTP_CALLS = 1, RTP_SWITCHED = 2;
constexpr int RTP_DQ_K_DIV_K = 4, RTP_DQ_ACT = 5;
constexpr int RTP_AT_L_BEGIN = 8, RTP_AT_L_CHUNKS = 9, RTP_AT_WINDOW = 10;
constexpr uint32_t OP_LOCK_REG = 0x1F000 + 16 * 15;    // core lock 15 value register
constexpr uint32_t DMA_CTRL_RESET = 0x2;               // bit 1 of a DMA channel control register
constexpr int ROWS = 4, ATTN_COLS = 4, DQ_COLS = 8;
constexpr int ATTN_ROWS_PER_ROUND = 128;               // q rows of one attention round (lc 16 x 8)

void write_reg(npu_sequence* seq, uint32_t row, uint32_t col, uint32_t offset, uint32_t value) {
    seq->rtp_write(get_tile(row, col), offset, value);   // a plain register write
}

const op_config& config_of(int op) {
    return op == hy_mem_prefill_overlay::OP_DQ ? op_dq : op_attn;
}

using reg_key = std::tuple<uint32_t, uint32_t, uint32_t>;

/// \brief (col, row, offset) of every register of one kind that either op writes
template <typename Get>
std::vector<reg_key> union_of(Get get) {
    std::set<reg_key> all;
    for (const op_config* cfg : {&op_dq, &op_attn})
        for (const reg_write& w : get(*cfg)) all.insert({w.col, w.row, w.offset});
    return {all.begin(), all.end()};
}

/// \brief the value `regs` gives (col, row, offset), 0 if it doesn't write it
uint32_t value_in(const std::vector<reg_write>& regs, const reg_key& key) {
    for (const reg_write& w : regs)
        if (w.col == std::get<0>(key) && w.row == std::get<1>(key) && w.offset == std::get<2>(key)) return w.value;
    return 0;
}

size_t write_bd(npu_sequence* seq, const reg_block& b) {
    for (size_t i = 0; i < b.words.size(); i++) write_reg(seq, b.row, b.col, b.offset + 4 * (uint32_t)i, b.words[i]);
    return b.words.size();
}

/// \brief configure the array for op `to`, coming from op `from` (OP_NONE: unknown,
///        so everything is written): stop the DMA channels, then write the op's
///        stream switches, BDs and lock values, then start its chains
void gen_switch(npu_sequence* seq, int to, int from) {
    static const std::vector<reg_key> sw_regs = union_of([](const op_config& c) -> const auto& { return c.sw; });
    static const std::vector<reg_key> lock_regs = union_of([](const op_config& c) -> const auto& { return c.locks; });
    const op_config& cfg = config_of(to);
    const bool full = from == hy_mem_prefill_overlay::OP_NONE;

    // 1. reset the channels the old op runs on: its chains loop forever on a lock
    auto reset = [&](uint32_t row, uint32_t col, uint32_t offset) {
        seq->npu_maskwrite(get_tile(row, col), offset, DMA_CTRL_RESET, DMA_CTRL_RESET);
        seq->npu_maskwrite(get_tile(row, col), offset, 0, DMA_CTRL_RESET);
    };
    if (full) {
        for (const reg_write& ch : all_channels) reset(ch.row, ch.col, ch.offset);
    } else {
        for (const chain_start& s : config_of(from).starts) reset(s.row, s.col, s.ctrl_offset);
    }
    // 2. routes: the op's value, 0 (port off) where only the other op routes;
    //    after a known op only the registers whose value differs
    for (const reg_key& key : sw_regs) {
        const uint32_t v = value_in(cfg.sw, key);
        if (!full && v == value_in(config_of(from).sw, key)) continue;
        write_reg(seq, std::get<1>(key), std::get<0>(key), std::get<2>(key), v);
    }
    for (const mask_write& m : cfg.mux) seq->npu_maskwrite(get_tile(m.row, m.col), m.offset, m.value, m.mask);
    // 3. BDs only one op uses stay loaded once written (full switch); the few
    //    both ops use are this op's again after every switch
    if (full) {
        for (const op_config* c : {&op_dq, &op_attn})
            for (const reg_block& b : c->bds)
                if (!b.shared) write_bd(seq, b);
    }
    for (const reg_block& b : cfg.bds)
        if (b.shared) write_bd(seq, b);
    // 4. lock values, after the reset (a reset chain may hold a lock count);
    //    not the op lock: every core is waiting on it
    for (const reg_key& key : lock_regs) {
        if (std::get<1>(key) >= 2 && std::get<2>(key) == OP_LOCK_REG) continue;
        write_reg(seq, std::get<1>(key), std::get<0>(key), std::get<2>(key), value_in(cfg.locks, key));
    }
    // 5. start the op's chains
    for (const chain_start& s : cfg.starts) {
        write_reg(seq, s.row, s.col, s.queue_offset, s.queue_value);
        write_reg(seq, s.row, s.col, s.ctrl_offset, s.ctrl_value);
    }
}

/// \brief each core's RTP block for one run of `op`, then let the cores go
void gen_dispatch(npu_sequence* seq, int op, int calls, bool switched, int cols,
                  std::initializer_list<std::pair<int, uint32_t>> extra) {
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < cols; c++) {
            for (const auto& [idx, value] : extra) write_reg(seq, r + 2, c, RTP_ADDR + 4 * idx, value);
            write_reg(seq, r + 2, c, RTP_ADDR + 4 * RTP_OP, op);
            write_reg(seq, r + 2, c, RTP_ADDR + 4 * RTP_CALLS, calls);
            write_reg(seq, r + 2, c, RTP_ADDR + 4 * RTP_SWITCHED, switched ? 1 : 0);
        }
    }
    // after all RTPs: a core reads its block as soon as its lock is set
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < cols; c++) write_reg(seq, r + 2, c, OP_LOCK_REG, 1);
}

/// \brief one attention run: buffers arg 0 = o, 1 = q, 2 = the layer's KV cache
///        ([K col0][K col1][V col0][V col1], MAX_L x 2 heads x DH each).
///        gen_seq_1_4 of FLM_Xclbin/Hy_mem_30BA3B/attention_prefill with the
///        overlay's switch + RTP blocks in front.
void gen_attn_seq(npu_sequence* seq, int from, uint32_t L_begin, uint32_t L_wanted, int MAX_L, int DH, int Heads,
                  int KV_Heads, int heads_per_col) {
    npu_tiles IT[2][2] = {{IT0, IT1}, {IT2, IT3}};
    const int lc = 16;
    const int num_cu = 2;
    const int q_per_kv = Heads / KV_Heads;
    const int Q_chunks = L_wanted / (lc * 8);
    const int window = MAX_L;                       // plain causal
    const size_t KV_CACHE_SIZE = (size_t)MAX_L * 2 * KV_Heads * DH;
    assert(L_wanted % (lc * 8) == 0 && q_per_kv % num_cu == 0);

    seq->clear_cmds();
    const bool switched = from != hy_mem_prefill_overlay::OP_ATTN;
    if (switched) gen_switch(seq, hy_mem_prefill_overlay::OP_ATTN, from);
    // one attn() call per head round
    gen_dispatch(seq, hy_mem_prefill_overlay::OP_ATTN, Heads / num_cu, switched, ATTN_COLS,
                 {{RTP_AT_L_BEGIN, L_begin}, {RTP_AT_L_CHUNKS, (uint32_t)Q_chunks}, {RTP_AT_WINDOW, (uint32_t)window}});

    const int data_per_round = lc * 8;
    for (int head = 0; head < Heads / num_cu; head++) {
        for (int round = 0; round < Q_chunks; round++) {
            const int Lq_current = L_begin + round * data_per_round;
            const int kv_begin = std::max(Lq_current - window, 0);
            const int kv_end = (Lq_current + 2 * data_per_round - 1) / data_per_round * data_per_round;
            const int kv_length = kv_end - kv_begin;
            const int bd_offset = (round % 2) * 8;
            for (int cu = 0; cu < num_cu; cu++) {
                const int head_offset = head * num_cu + cu;
                for (int col = 0; col < 2; col++) {
                    const size_t y_offset = (size_t)head_offset * DH + (size_t)(round * data_per_round + col * lc * 4) * DH * Heads;
                    seq->npu_dma_memcpy_nd(2, 0, S2MM, IT[cu][col], (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset}, {1, 1, (uint32_t)4 * lc, (uint32_t)DH},
                        {0, 0, (uint32_t)DH * Heads, 1}, -1, 0, true);
                }
                const size_t q_offset = (size_t)head_offset * DH + (size_t)round * data_per_round * DH * Heads;
                seq->npu_dma_memcpy_nd(2, 1, MM2S, IT[cu][0], (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset}, {1, 1, (uint32_t)lc * 4, (uint32_t)DH},
                    {0, 0, (uint32_t)DH * Heads, 1}, -1, 0, false);
                seq->npu_dma_memcpy_nd(2, 1, MM2S, IT[cu][0], (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)}, {1, 1, (uint32_t)lc * 4, (uint32_t)DH},
                    {0, 0, (uint32_t)DH * Heads, 1}, -1, 0, false);
            }
            // both CUs' q heads share this KV head (GQA); its column block and half of the row
            const int kv_head = head * num_cu / q_per_kv;
            const size_t k_offset = (size_t)(kv_head / heads_per_col) * MAX_L * DH * heads_per_col
                                    + (size_t)(kv_head % heads_per_col) * DH + (size_t)kv_begin * DH * heads_per_col;
            const size_t v_offset = k_offset + KV_CACHE_SIZE / 2;
            const std::vector<uint32_t> kv_size = {1, (uint32_t)kv_length / 128, 128, (uint32_t)DH};
            const std::vector<uint32_t> kv_stride = {0, (uint32_t)(128 * DH * heads_per_col), (uint32_t)(DH * heads_per_col), 1};
            seq->npu_dma_memcpy_nd(2, 2, MM2S, IT[1][1], (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset}, kv_size, kv_stride, -1, 0, false);
            seq->npu_dma_memcpy_nd(2, 2, MM2S, IT[1][1], (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset}, kv_size, kv_stride, -1, 0, false);
            if (round > 0) {
                for (int cu = 0; cu < num_cu; cu++)
                    for (int col = 0; col < 2; col++) seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
            }
        }
        for (int cu = 0; cu < num_cu; cu++)
            for (int col = 0; col < 2; col++) seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}

/// \brief one dequant_mm run: y[L][Dout] = x[L][Din] W^T, W Q4_K in the overlay's
///        block order. Buffers: arg 0 = y, 1 = x, 2 = the weight buffer.
/// \note gen_dq_run of FLM_Xclbin/Hy_mem_30BA3B/prefill_overlay/host/overlay_host.hpp
///       with the overlay's switch + RTP blocks in front. Core column c of 8 takes
///       64 output rows; the 4 cores of a column split the 256 tokens of a round.
/// \param channel_bytes 0: the rows run straight through the buffer (`pairs`, what
///        the attention / dense / shared weights use). Otherwise a routed expert's
///        `expert_channels` view: its row blocks alternate between two streams this
///        far apart, 4 row blocks at a time, so 64 output rows come from one stream
///        and the stream is chosen by the array column (see w_chunk below).
void gen_dq_seq(npu_sequence* seq, int from, uint32_t L, uint32_t Din, uint32_t Dout,
                size_t w_offset, int rtp_act, size_t channel_bytes) {
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    static constexpr uint32_t desired_m = 64;    // kernel/dq/common.hpp NPU_M_TILE
    static constexpr uint32_t desired_k = 256;   // NPU_K_TILE: one Q4_K block
    static constexpr uint32_t desired_n = 64;
    const uint32_t columns = DQ_COLS;
    const int K_div_k = Din / desired_k;
    const int N_div_n = Dout / desired_n / columns;
    assert(Din % desired_k == 0 && Dout % (desired_n * columns) == 0);

    const int M_div_m = L / (desired_m * 4);
    assert(L % (desired_m * 4) == 0);

    seq->clear_cmds();
    const bool switched = from != hy_mem_prefill_overlay::OP_DQ;
    if (switched) gen_switch(seq, hy_mem_prefill_overlay::OP_DQ, from);
    // every core computes one 128 x 64 tile per round
    gen_dispatch(seq, hy_mem_prefill_overlay::OP_DQ, M_div_m * N_div_n, switched, DQ_COLS,
                 {{RTP_DQ_K_DIV_K, (uint32_t)K_div_k}, {RTP_DQ_ACT, (uint32_t)rtp_act}});

    auto w_bytes = [](size_t elems) { return get_quantization_byte_size(elems, flm_q4k); };
    // bytes of the 64 output rows one array column takes in a round
    const size_t chunk_bytes = w_bytes((size_t)desired_n * Din);

    /// where array column `c` of round `j` finds its 64 rows.
    /// Both layouts keep a column's 64 rows contiguous (2 row blocks x every column
    /// block); only their starts differ. For expert_channels the row blocks go
    /// [ch0 ch0 ch1 ch1] in fours, so chunk g = j * columns + c lands in channel
    /// g % 2 (= c % 2, columns being even) as that channel's g / 2'th chunk.
    auto w_chunk = [&](int j, uint32_t c) {
        const size_t g = (size_t)j * columns + c;
        if (channel_bytes == 0) return w_offset + chunk_bytes * g;
        return w_offset + (c % 2) * channel_bytes + chunk_bytes * (g / 2);
    };

    for (int i = 0; i < M_div_m; i++) {
        for (int j = 0; j < N_div_n; j++) {
            const int round = i * N_div_n + j;
            const int bd_offset = (round % 2) * 8;
            const uint32_t y_offset = i * desired_m * 4 * Dout + j * desired_n * columns;
            const uint32_t x_offset = i * desired_m * 4 * Din;
            const npu_bd_id y_bd_0 = npu_bd_id(0 + bd_offset), y_bd_1 = npu_bd_id(3 + bd_offset);
            const npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            const npu_bd_id w_bd_0 = npu_bd_id(2 + bd_offset), w_bd_1 = npu_bd_id(4 + bd_offset);
            for (uint32_t col = 0; col < 4; col++) {
                seq->npu_dma_memcpy_nd(sizeof(char), 2, MM2S, IT[col], w_bd_0, it_channel_1,
                    {0, 0, 0, (uint32_t)w_chunk(j, col)}, {1, 1, 1, (uint32_t)chunk_bytes},
                    {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
                seq->npu_dma_memcpy_nd(sizeof(char), 2, MM2S, IT[col + 4], w_bd_1, it_channel_1,
                    {0, 0, 0, (uint32_t)w_chunk(j, col + 4)}, {1, 1, 1, (uint32_t)chunk_bytes},
                    {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
                seq->npu_dma_memcpy_nd(2, 0, S2MM, IT[col], y_bd_0, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * desired_m * Dout)},
                    {1, 1, desired_m, desired_n * 4}, {0, 0, Dout, 1}, -1, 0, true);
                seq->npu_dma_memcpy_nd(2, 0, S2MM, IT[col + 4], y_bd_1, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * desired_m * Dout + 4 * desired_n)},
                    {1, 1, desired_m, desired_n * 4}, {0, 0, Dout, 1}, -1, 0, true);
                seq->npu_dma_memcpy_nd(2, 1, MM2S, IT[2 * col], x_bd, it_channel_0,
                    {0, 0, 0, x_offset + col * desired_m * Din},
                    {1, (uint32_t)K_div_k, desired_m, desired_k}, {0, desired_k, Din, 1}, -1, 0, false);
            }
            if (round > 0) {
                for (int r = 0; r < 4; r++) {
                    seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                    seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int r = 0; r < 4; r++) {
        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
        seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}

int round_up(int x, int m) { return (x + m - 1) / m * m; }
}  // namespace

hy_mem_prefill_overlay::hy_mem_prefill_overlay(npu_xclbin_manager* npu, const std::string& xclbin, const hy_mem_desc& desc)
    : desc(desc) {
    mgr = npu->register_xclbin(xclbin);
    buf_owner = mgr->create_app();
    q_buf = buf_owner.create_bo_buffer<bf16>((size_t)MAX_ROWS * desc.DQ);
    o_buf = buf_owner.create_bo_buffer<bf16>((size_t)MAX_ROWS * desc.DQ);
    q_buf.memset((bf16)0.0f);
}

void hy_mem_prefill_overlay::prepare_attention(int L, int pos0, uint32_t MAX_L) {
    // padded q rows are zeros; their output rows are dropped
    const int L_pad = round_up(L, ATTN_ROWS_PER_ROUND);
    if (L_pad > MAX_ROWS || pos0 + L > (int)MAX_L) {
        throw std::runtime_error("hy_mem_prefill_overlay: chunk of " + std::to_string(L) + " rows at " + std::to_string(pos0)
                                 + " does not fit (" + std::to_string(MAX_ROWS) + " rows, KV cache " + std::to_string(MAX_L) + ")");
    }
    // The last round reads KV rows up to the 128 boundary past its own 128 q rows,
    // at most this far past the cache: the last block's zero tail must cover it.
    const int last_q = pos0 + L_pad - ATTN_ROWS_PER_ROUND;
    const int kv_end = (last_q + 2 * ATTN_ROWS_PER_ROUND - 1) / ATTN_ROWS_PER_ROUND * ATTN_ROWS_PER_ROUND;
    assert(kv_end <= (int)(MAX_L + hy_mem_desc::KV_TAIL_ROWS));
    (void)kv_end;
    attn_L = L;
    attn_L_pad = L_pad;
    attn_pos0 = pos0;
    attn_MAX_L = MAX_L;
    attn_runs.clear();
    // q rows past L (padding) only have to be finite
    std::fill(q_buf.data() + (size_t)L * desc.DQ, q_buf.data() + (size_t)attn_L_pad * desc.DQ, (bf16)0.0f);
}

void hy_mem_prefill_overlay::attention(bf16* out, const bf16* q, int q_stride, buffer<bf16>& kv_cache) {
    const size_t DQ = desc.DQ;
    for (int l = 0; l < attn_L; l++) memcpy(q_buf.data() + l * DQ, q + (size_t)l * q_stride, DQ * sizeof(bf16));
    q_buf.sync_to_device((size_t)attn_L_pad * DQ);   // the buffer holds MAX_ROWS, this chunk is smaller

    // the switch (if any) is part of the run, so there is one run per op the array may come from
    auto it = attn_runs.find(configured);
    if (it == attn_runs.end()) {
        npu_app app = mgr->create_app();
        gen_attn_seq(app.seq(), configured, attn_pos0, attn_L_pad, attn_MAX_L, desc.DH, desc.num_heads,
                     desc.num_kv_heads, desc.num_kv_heads / hy_mem_desc::ATTN_COLS);
        it = attn_runs.emplace(configured, std::move(app)).first;
    }
    it->second(o_buf, q_buf, kv_cache);
    configured = OP_ATTN;

    o_buf.sync_from_device((size_t)attn_L_pad * DQ);
    memcpy(out, o_buf.data(), (size_t)attn_L * DQ * sizeof(bf16));
}

void hy_mem_prefill_overlay::_ensure_mm_buffers(size_t x_elems, size_t y_elems) {
    // A reallocation only changes which buffer a run is called with, never the
    // run itself (the buffers are call arguments), so dq_runs stays valid.
    if (mm_x.size() < x_elems) mm_x = buf_owner.create_bo_buffer<bf16>(x_elems);
    if (mm_y.size() < y_elems) mm_y = buf_owner.create_bo_buffer<bf16>(y_elems);
}

void hy_mem_prefill_overlay::dequant_mm(bf16* y, const bf16* x, int L, buffer<u8>& weights,
                                        size_t w_offset, int Din, int Dout, size_t channel_bytes) {
    if (L <= 0) return;
    if (!fits(Din, Dout)) {
        throw std::runtime_error("hy_mem_prefill_overlay: dequant_mm " + std::to_string(Din) + " -> "
                                 + std::to_string(Dout) + " is not a shape the overlay takes");
    }
    const int L_pad = round_up(L, DQ_ROWS_PER_ROUND);
    _ensure_mm_buffers((size_t)L_pad * Din, (size_t)L_pad * Dout);

    // the padding rows only have to be finite; their output rows are dropped
    memcpy(mm_x.data(), x, (size_t)L * Din * sizeof(bf16));
    std::fill(mm_x.data() + (size_t)L * Din, mm_x.data() + (size_t)L_pad * Din, (bf16)0.0f);
    mm_x.sync_to_device((size_t)L_pad * Din);     // not the whole allocation: it is
                                                  // sized for the largest call, and the
                                                  // routed experts are the smallest

    // the switch (if any) is part of the run, so the op the array comes from is part of the key
    const dq_key key{L_pad, Din, Dout, configured, w_offset, channel_bytes};
    auto it = dq_runs.find(key);
    if (it == dq_runs.end()) {
        if (dq_runs.size() >= MAX_DQ_RUNS) dq_runs.clear();
        npu_app app = mgr->create_app();
        gen_dq_seq(app.seq(), configured, (uint32_t)L_pad, (uint32_t)Din, (uint32_t)Dout, w_offset, 0, channel_bytes);
        it = dq_runs.emplace(key, std::move(app)).first;
    }
    it->second(mm_y, mm_x, weights);
    configured = OP_DQ;

    mm_y.sync_from_device((size_t)L_pad * Dout);
    memcpy(y, mm_y.data(), (size_t)L * Dout * sizeof(bf16));
}


