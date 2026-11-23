// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__VISION__SEQ__GEN__HPP__
#define __SMOLVLA__VISION__SEQ__GEN__HPP__
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "npu_utils/npu_utils.hpp"
#include "utils/xclbin_path.hpp"
#include "weight_desc.hpp"

namespace smolvla_vlm_vision_seq{
/// @brief build the NPU DMA sequence for bidirectional vision self-attention.
///        Attention is computed independently per image: each image's 1024
///        tokens form one block, and blocks are laid out contiguously in the
///        q/k/v buffers, so the sequence loops over images and offsets each by
///        one image's worth of elements (1024*768).
/// @param seq        sequence to populate (cleared and rebuilt)
/// @param L          tokens per image (1024)
/// @param DH         head dimension
/// @param Heads      number of attention heads
/// @param num_images number of images in the batch
/// @param need_set_rtp force re-writing the runtime params into the tiles
inline void gen_vision_attn_seq(
    npu_sequence* seq,
    const uint32_t L,
    const uint32_t DH,
    const int Heads,
    const int num_images,
    bool need_set_rtp = false
){

    static uint32_t last_L = 0;
    static int last_DH = DH;
    static int last_Heads = Heads;


    if (last_L != L) {
        need_set_rtp = true;
        last_L = L;
    }
    if (last_DH != DH) {
        need_set_rtp = true;
        last_DH = DH;
    }
    if (last_Heads != Heads) {
        need_set_rtp = true;
        last_Heads = Heads;
    }

    constexpr int lq = 64;
    npu_tiles IT[2][2] = {{IT0, IT1}, {IT6, IT7}};
    assert(L % (lq * 8) == 0);

    int l_begin_mha_address = 10496;

    seq->clear_cmds();
    if (need_set_rtp) {
        for (int row = 2; row < 6; row++){
            npu_tiles tile0 = get_tile(row, 0);
            npu_tiles tile1 = get_tile(row, 1);
            npu_tiles tile2 = get_tile(row, 6);
            npu_tiles tile3 = get_tile(row, 7);
            seq->rtp_write(tile0, l_begin_mha_address, L);
            seq->rtp_write(tile1, l_begin_mha_address, L);
            seq->rtp_write(tile2, l_begin_mha_address, L);
            seq->rtp_write(tile3, l_begin_mha_address, L);
        }
    }

    // each round, each CU works on 1 head and lc * 8 of l, total_cols / 2 is corresponding to the number of CUs
    const int data_per_round = lq * 8; 
    // zero padding the last round if necessary
    const int down_rounds = (L + data_per_round - 1) / data_per_round;
    for (int image_id = 0; image_id < num_images; image_id++) {
        uint32_t image_offset = 1024 * 768 * image_id;
        for (int head = 0; head < Heads / 2; head++){
            for (int round = 0; round < down_rounds; round++){
                int _round = image_id * Heads / 2 * down_rounds + head * down_rounds + round;
                int bd_offset = (_round % 2) * 8;
                // cu 0,
                for (int cu = 0; cu < 2; cu++){
                    int head_offset = head * 2 + cu;
                    // send q
                    size_t q_offset = head_offset * DH + round * data_per_round * DH * Heads + image_offset;
                    seq->npu_dma_memcpy_nd(
                        2, 1,
                        MM2S, IT[cu][0],
                        (npu_bd_id)(bd_offset + 1), it_channel_0,
                        {0, 0, 0, (uint32_t)q_offset},
                        {1, 1, (uint32_t)lq * 4, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, false
                    );
                    seq->npu_dma_memcpy_nd(
                        2, 1,
                        MM2S, IT[cu][0],
                        (npu_bd_id)(bd_offset + 2), it_channel_1,
                        {0, 0, 0, (uint32_t)(q_offset + (size_t)lq * 4 * DH * Heads)},
                        {1, 1, (uint32_t)lq * 4, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, false
                    );
                    // send k
                    size_t k_offset = head_offset * DH + image_offset;
                    seq->npu_dma_memcpy_nd(
                        2, 2,
                        MM2S, IT[cu][1],
                        (npu_bd_id)(bd_offset + 3), it_channel_0,
                        {0, 0, 0, (uint32_t)k_offset},
                        {1, (L / 512), 512, DH},
                        {0, (uint32_t)512 * DH * Heads, (uint32_t)DH * Heads, 1},
                        -1, 0, false
                    );
                    // send v
                    size_t v_offset = head_offset * DH + image_offset;
                    seq->npu_dma_memcpy_nd(
                        2, 3,
                        MM2S, IT[cu][1],
                        (npu_bd_id)(bd_offset + 4), it_channel_1,
                        {0, 0, 0, (uint32_t)v_offset},
                        {1, (L / 512), 512, DH},
                        {0, (uint32_t)512 * DH * Heads, (uint32_t)DH * Heads, 1},
                        -1, 0, false
                    );
                    for (int col = 0; col < 2; col++){    
                        // receive y
                        size_t y_offset = head_offset * DH + round * data_per_round * DH * Heads + col * lq * 4 * DH * Heads + image_offset; 
                        seq->npu_dma_memcpy_nd(
                            2, 0,
                            S2MM, IT[cu][col],
                            (npu_bd_id)(bd_offset + 0), it_channel_0,
                            {0, 0, 0, (uint32_t)y_offset},
                            {1, 1, (uint32_t)4 * lq, (uint32_t)(DH)},
                            {0, 0, (uint32_t)DH * Heads, 1},
                            -1, 0, true
                        );
                    }
                } // cu
                if (_round > 0){
                    for (int cu = 0; cu < 2; cu++){
                        for (int col = 0; col < 2; col++){
                            seq->npu_dma_wait(
                                IT[cu][col],
                                S2MM,
                                it_channel_0
                            );
                        } // col
                    } // cu
                }// round > 0
            }// round
        } // head
    }
    for (int cu = 0; cu < 2; cu++){
        for (int col = 0; col < 2; col++){
            seq->npu_dma_wait(
                IT[cu][col],
                S2MM,
                it_channel_0
            );
        } // col
    } // cu
    seq->cmds2seq();
}

/// force_rtp writes all three RTPs into this sequence and leaves the "already
/// set" state untouched: sequences are cached, so a later sequence's decision to
/// skip an RTP write assumes the order the earlier ones were generated in.
inline void gen_vision_mm_seq(npu_sequence* seq, const uint32_t M, const weight_desc_t& weight, int32_t rtp_bias, int32_t rtp_act, bool force_rtp = false, uint32_t row0 = 0){
    static uint32_t last_M = 0;
    static uint32_t last_K = 0;
    static uint32_t last_N = 0;
    static int32_t  last_rtp_bias = -1;
    static int32_t  last_rtp_act = -1;
    const uint32_t w_base = (uint32_t)weight.offset / sizeof(bf16);
    constexpr npu_tiles IT[4] = {IT2, IT3, IT4, IT5};
    constexpr int total_cols = 4;
    constexpr int total_rows = 4;
    const uint32_t K = weight.shape[0];
    const uint32_t N = weight.shape[1];
    constexpr int m = 128;
    constexpr int k = 256;
    constexpr int n = 64;
    assert(N % k == 0);
    seq->clear_cmds();

    bool need_reset_rtp = force_rtp;

    constexpr int rtp_address = 56832;
    uint32_t K_div_k = K / k;
    const int M_div_m = M / m / total_rows;
    const int N_div_n  = N / n / total_cols; 

    if (force_rtp || (last_M != M)){
        need_reset_rtp = true;
        if (!force_rtp) last_M = M;
        for (int row = 2; row < 6; row++){
            for (int col = 2; col < total_cols + 2; col++){
                npu_tiles tile = get_tile(row, col);
                seq->rtp_write(tile, rtp_address, M_div_m);
            }
        }
    }
    if (force_rtp || (last_K != K)){
        need_reset_rtp = true;
        for (int row = 2; row < 6; row++){
            for (int col = 2; col < total_cols + 2; col++){
                npu_tiles tile = get_tile(row, col);
                seq->rtp_write(tile, rtp_address + 4, K_div_k);
            }
        }
        if (!force_rtp) last_K = K;
    }
    if (force_rtp || (last_rtp_act != rtp_act)){
        need_reset_rtp = true;
        for (int row = 2; row < 6; row++){
            for (int col = 2; col < total_cols + 2; col++){
                npu_tiles tile = get_tile(row, col);
                seq->rtp_write(tile, rtp_address + 8, rtp_act);
            }
        }
        if (!force_rtp) last_rtp_act = rtp_act;
    }

    for (int j = 0; j < N_div_n; j++){
        if(rtp_bias == 1){
            // send b
            uint32_t b_offset = j * n * total_cols + w_base - N;
            for (int c = 0; c < total_cols; c++){
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[c],
                    npu_bd_id(3), it_channel_1,
                    {0, 0, 0, b_offset + c * n},
                    {1, 1, 1, (uint32_t)n * k},
                    {0, 0, 0, 1},
                    -1, 0, false
                );
            }
        }
        for (int i = 0; i < M_div_m; i++){
            int round = j * M_div_m + i;
            int bd_offset = (round % 2) * 8;
            // row0: first row of x / y this sequence works on (the matmul can be split along M)
            int y_offset = (row0 + i * m * total_rows) * N + j * n * total_cols;
            int x_offset = (row0 + i * m * total_rows) * K;
            uint32_t w_offset = j * n * K * total_cols + w_base;
            npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd = npu_bd_id(2 + bd_offset);
            // send w
            for (int c = 0; c < total_cols; c++){
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[c],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset + c * n * K},
                    {1, K_div_k, n, (uint32_t)k},
                    {0, k, K, 1},
                    -1, 0, false
                );
                // receive y
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[c],
                    y_bd, it_channel_0,
                    {0, 0, 0, y_offset + c * m * N},
                    {1, 1, m, n * 4},
                    {0, 0, N, 1},
                    -1, 0, true
                );
                // send x
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[c],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + c * m * K},
                    {1, (uint32_t)K_div_k, m, k},
                    {0, k, K, 1},
                    -1, 0, false
                );
            }
            if (round > 0){
                for (int r = 0; r < total_rows; r++){
                    seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int r = 0; r < total_rows; r++){
        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}

// ---------------------------------------------------------------------------
// The vision tower runs in one hw_context on the 8-column matmul xclbin (vision_mm8, 32 cores) and switches to
// the 32-core attention design (vision_attn32) and back for every layer's attention. The switches are the
// launchlib warm kernels (to_attn32 / to_mm8, vision.switches of SMOLVLA/vision_switches); each is prepended to
// the sequence that follows it, so the engine's npu_app flow is unchanged. A switch leaves the runtime
// parameters at the design's defaults: every sequence writes all of its.
// ---------------------------------------------------------------------------

/// true (default; SMOLVLA_VISION_L32=0: the 16-core design on vision.xclbin): vision on the mm8 + attn32 launchlib path
inline bool vision_l32() {
    static const bool v = [] { const char* e = std::getenv("SMOLVLA_VISION_L32"); return !e || atoi(e) != 0; }();
    return v;
}

/// true (default; SMOLVLA_VISION_CTRLPKT=0: TXN-only): the switches load by control packets. The streams
/// cp_to_* of vision.switches only start the packet DMA; the packets are one buffer at XRT argument
/// kVisionPacketArg of the runs that carry the switch.
inline bool vision_ctrlpkt() {
    static const bool v = [] { const char* e = std::getenv("SMOLVLA_VISION_CTRLPKT"); return !e || atoi(e) != 0; }();
    return v;
}
inline constexpr int kVisionPacketArg = 7; // 3 + packet_arg (4): no vision sequence uses more than 4 arguments

/// one switch of vision.switches (SMOLVLA/vision_switches)
struct VisionSwitch {
    std::vector<uint32_t> words;   // the TXN instruction stream
    std::vector<uint32_t> packets; // control packets (cp_* only)
};

/// the switches of vision.switches by name (to_attn32, to_mm8, cp_to_attn32, cp_to_mm8), loaded once. Format
/// (SMOLVLA/vision_switches/switch_files.cpp): "SVSW1\n", per kernel "name packet_arg txn_bytes pkt_bytes\n",
/// "END\n", then the blobs in that order.
inline const VisionSwitch& vision_switch(const std::string& name) {
    static const std::map<std::string, VisionSwitch> all = [] {
        const std::string path = utils::xclbin_path("vision.switches");
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("SmolVLA vision: cannot open " + path + " (SMOLVLA_VISION_L32=0 runs without it)");
        std::string line;
        std::getline(f, line);
        if (line != "SVSW1") throw std::runtime_error("SmolVLA vision: " + path + " is not a vision.switches file");
        struct Entry { std::string name; size_t txn, pkt; };
        std::vector<Entry> entries;
        while (std::getline(f, line) && line != "END") {
            std::istringstream ls(line);
            Entry e; int packet_arg;
            ls >> e.name >> packet_arg >> e.txn >> e.pkt;
            if (!ls) throw std::runtime_error("SmolVLA vision: malformed header in " + path);
            entries.push_back(e);
        }
        std::map<std::string, VisionSwitch> m;
        for (const Entry& e : entries) {
            VisionSwitch sw;
            sw.words.resize(e.txn / sizeof(uint32_t));
            sw.packets.resize(e.pkt / sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(sw.words.data()), (std::streamsize)e.txn);
            f.read(reinterpret_cast<char*>(sw.packets.data()), (std::streamsize)e.pkt);
            if (!f) throw std::runtime_error("SmolVLA vision: " + path + " is truncated");
            m.emplace(e.name, std::move(sw));
        }
        return m;
    }();
    auto it = all.find(name);
    if (it == all.end()) throw std::runtime_error("SmolVLA vision: vision.switches has no switch " + name);
    return it->second;
}

/// seq = the ops of `pre` followed by the ops of seq (both TXN streams with the 4-word header)
inline void prepend_txn(npu_sequence* seq, const std::vector<uint32_t>& pre) {
    auto d = seq->dump();
    std::vector<uint32_t> w;
    w.reserve(pre.size() + d.second);
    w.insert(w.end(), pre.begin(), pre.begin() + 4);
    w.insert(w.end(), pre.begin() + 4, pre.end());
    w.insert(w.end(), d.first + 4, d.first + d.second);
    w[2] = pre[2] + d.first[2];
    w[3] = (uint32_t)(w.size() * sizeof(uint32_t));
    seq->from_vector(w);
}

/// 8-column matmul y[M,N] = x[M,K] * w[N,K]^T + bias (bias stored right before w), as gen_vision_mm_seq.
/// 32 cores: core row r works on rows [r*128, +128) of a 512-row block, core column c on 48 of the 384 columns
/// of an N strip (M % 512 == 0, K % 256 == 0, N % 384 == 0). Writes all three RTPs. row0: first row of x / y.
inline void gen_vision_mm8_seq(npu_sequence* seq, const uint32_t M, const weight_desc_t& weight, int32_t rtp_act,
                               uint32_t row0 = 0, const std::vector<uint32_t>* prefix = nullptr) {
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr int total_cols = 8;
    constexpr int total_rows = 4;
    constexpr uint32_t m = 128, k = 256, n = 48;
    constexpr int rtp_address = 44480;
    const uint32_t w_base = (uint32_t)weight.offset / sizeof(bf16);
    const uint32_t K = weight.shape[0];
    const uint32_t N = weight.shape[1];
    assert(M % (m * total_rows) == 0 && K % k == 0 && N % (n * total_cols) == 0);
    seq->clear_cmds();
    const uint32_t K_div_k = K / k;
    const int M_div_m = M / m / total_rows;
    const int N_div_n = N / n / total_cols;
    for (int row = 2; row < 6; row++) {
        for (int col = 0; col < total_cols; col++) {
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, rtp_address, M_div_m);
            seq->rtp_write(tile, rtp_address + 4, K_div_k);
            seq->rtp_write(tile, rtp_address + 8, rtp_act);
        }
    }
    for (int j = 0; j < N_div_n; j++) {
        const uint32_t b_offset = j * n * total_cols + w_base - N;
        for (int c = 0; c < total_cols; c++) {
            seq->npu_dma_memcpy_nd(2, 2, MM2S, IT[c], npu_bd_id(3), it_channel_1,
                                   {0, 0, 0, b_offset + c * n}, {1, 1, 1, n * k}, {0, 0, 0, 1}, -1, 0, false);
        }
        for (int i = 0; i < M_div_m; i++) {
            const int round = j * M_div_m + i;
            const int bd_offset = (round % 2) * 8;
            const uint32_t row_base = row0 + i * m * total_rows;
            const uint32_t w_offset = j * n * K * total_cols + w_base;
            for (int c = 0; c < total_cols; c++) {
                const uint32_t r = c % 4, g = c / 4;
                seq->npu_dma_memcpy_nd(2, 2, MM2S, IT[c], npu_bd_id(2 + bd_offset), it_channel_1,
                                       {0, 0, 0, w_offset + c * n * K}, {1, K_div_k, n, k}, {0, k, K, 1}, -1, 0, false);
                seq->npu_dma_memcpy_nd(2, 0, S2MM, IT[c], npu_bd_id(0 + bd_offset), it_channel_0,
                                       {0, 0, 0, (row_base + r * m) * N + j * n * total_cols + g * n * 4},
                                       {1, 1, m, n * 4}, {0, 0, N, 1}, -1, 0, true);
                if (c < 4) {  // x of row block r: one read, multicast to the memtiles r and r + 4
                    seq->npu_dma_memcpy_nd(2, 1, MM2S, IT[c], npu_bd_id(1 + bd_offset), it_channel_0,
                                           {0, 0, 0, (row_base + r * m) * K}, {1, K_div_k, m, k}, {0, k, K, 1}, -1, 0, false);
                }
            }
            if (round > 0) {
                for (int c = 0; c < total_cols; c++) seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
            }
        }
    }
    for (int c = 0; c < total_cols; c++) seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
    seq->cmds2seq();
    if (prefix) prepend_txn(seq, *prefix);
}

/// 32-core vision attention (vision_attn32): 4 compute units of 8 cores, one head per CU per round, tensors
/// [image][L][Heads*DH]; L per image must be a multiple of 512. Args: y, q, k, v.
inline void gen_vision_attn32_seq(npu_sequence* seq, const uint32_t L, const int DH, const int Heads,
                                  const int num_images, const std::vector<uint32_t>* prefix = nullptr) {
    constexpr int lq = 64;
    constexpr int NCU = 4;
    const npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};
    constexpr int l_begin_mha_address = 10496;
    assert(L % 512 == 0);

    seq->clear_cmds();
    for (int row = 2; row < 6; row++)
        for (int col = 0; col < 8; col++)
            seq->rtp_write(get_tile(row, col), l_begin_mha_address, L);

    const int data_per_round = lq * 8;
    const int down_rounds = (L + data_per_round - 1) / data_per_round;
    const int groups = Heads / NCU;
    int _round = 0;
    for (int image_id = 0; image_id < num_images; image_id++) {
        const size_t image_offset = (size_t)L * DH * Heads * image_id;
        for (int g = 0; g < groups; g++) {
            for (int round = 0; round < down_rounds; round++, _round++) {
                const int bd_offset = (_round % 2) * 8;
                for (int cu = 0; cu < NCU; cu++) {
                    const int head_offset = g * NCU + cu;
                    const size_t q_offset = (size_t)head_offset * DH + (size_t)round * data_per_round * DH * Heads + image_offset;
                    seq->npu_dma_memcpy_nd(2, 1, MM2S, IT[cu][0], (npu_bd_id)(bd_offset + 1), it_channel_0,
                        {0, 0, 0, (uint32_t)q_offset}, {1, 1, (uint32_t)lq * 4, (uint32_t)DH},
                        {0, 0, (uint32_t)DH * Heads, 1}, -1, 0, false);
                    seq->npu_dma_memcpy_nd(2, 1, MM2S, IT[cu][0], (npu_bd_id)(bd_offset + 2), it_channel_1,
                        {0, 0, 0, (uint32_t)(q_offset + (size_t)lq * 4 * DH * Heads)}, {1, 1, (uint32_t)lq * 4, (uint32_t)DH},
                        {0, 0, (uint32_t)DH * Heads, 1}, -1, 0, false);
                    const size_t kv_offset = (size_t)head_offset * DH + image_offset;
                    seq->npu_dma_memcpy_nd(2, 2, MM2S, IT[cu][1], (npu_bd_id)(bd_offset + 3), it_channel_0,
                        {0, 0, 0, (uint32_t)kv_offset}, {1, (L / 512), 512, (uint32_t)DH},
                        {0, (uint32_t)512 * DH * Heads, (uint32_t)DH * Heads, 1}, -1, 0, false);
                    seq->npu_dma_memcpy_nd(2, 3, MM2S, IT[cu][1], (npu_bd_id)(bd_offset + 4), it_channel_1,
                        {0, 0, 0, (uint32_t)kv_offset}, {1, (L / 512), 512, (uint32_t)DH},
                        {0, (uint32_t)512 * DH * Heads, (uint32_t)DH * Heads, 1}, -1, 0, false);
                    for (int col = 0; col < 2; col++) {
                        const size_t y_offset = (size_t)head_offset * DH + (size_t)round * data_per_round * DH * Heads
                                                + (size_t)col * lq * 4 * DH * Heads + image_offset;
                        seq->npu_dma_memcpy_nd(2, 0, S2MM, IT[cu][col], (npu_bd_id)(bd_offset + 0), it_channel_0,
                            {0, 0, 0, (uint32_t)y_offset}, {1, 1, (uint32_t)4 * lq, (uint32_t)DH},
                            {0, 0, (uint32_t)DH * Heads, 1}, -1, 0, true);
                    }
                }
                if (_round > 0) {
                    for (int cu = 0; cu < NCU; cu++)
                        for (int col = 0; col < 2; col++)
                            seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int cu = 0; cu < NCU; cu++)
        for (int col = 0; col < 2; col++)
            seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
    seq->cmds2seq();
    if (prefix) prepend_txn(seq, *prefix);
}

}
#endif