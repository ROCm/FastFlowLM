// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hy_mem_model_def.cpp
/// \brief weight loaders for Hy-Memory-30B-A3B
/// \note Ported from FLM_Xclbin/Hy_mem_30BA3B/{dense,moe}_decoding/host/decoding_layer.cpp
///       and lm_head_npu_bin/host/host.cpp. A Q4_K tensor in model.q4nx is row blocks
///       of 32 rows, each row block's 256-column blocks back to back, so "rows a..b"
///       of a tensor is one contiguous byte range.
#include <omp.h>
#include <array>
#include "hy_mem_model_def.hpp"

namespace {

constexpr size_t Q4K_BLOCK = hy_mem_desc::Q4K_BLOCK_BYTES;

/// Lays a quantized weight out in the order the proj engine reads it: rows of
/// blocks are taken vertical_blocks at a time and interleaved block by block,
/// so one shim BD streams the blocks of vertical_blocks proj cores together.
void reorder_cpy(u8* dst, const u8* src, size_t src_bytes, int col, size_t a_block_size, int vertical_blocks = 2) {
    const int blocks_per_row = col / QXNX_COL_BLOCK_SIZE;
    const size_t row_bytes = a_block_size * blocks_per_row;    // one 32-row block row
    const int rows = (int)(src_bytes / row_bytes);
    const u8* src_end = src + src_bytes;

    u8* dst_ptr = dst;
    std::vector<const u8*> src_ptr(vertical_blocks);
    for (int i = 0; i < vertical_blocks; i++)
        src_ptr[i] = src + i * row_bytes;

    for (int r = 0; r < rows; r += vertical_blocks) {
        for (int c = 0; c < blocks_per_row; c++) {
            for (int i = 0; i < vertical_blocks; i++) {
                memcpy(dst_ptr, src_ptr[i], a_block_size);
                dst_ptr += a_block_size;
                src_ptr[i] += a_block_size;
            }
        }
        for (int i = 0; i < vertical_blocks; i++) {
            src_ptr[i] += (vertical_blocks - 1) * row_bytes;
            if (src_ptr[i] + row_bytes > src_end)
                src_ptr[i] = src;   // useless padding
        }
    }
}

/// Q4_K blocks (32 rows x 256 columns, 4736 B) from the q4nx order into the order
/// of moe_decoding's int8-mmul kernel (_Q4K_x_int8 in kernels/q4_core.h), in place:
///   qs     [256 columns][16 B]  -> [k step 16][half t 2][16 columns][8 B]
///          (byte b of a column holds rows 2b, 2b + 1; half t = rows 16t .. 16t+15)
///   scales [group 8][32 rows]   -> [h 2][t 2][a 4][16 rows], group = 4h + a
///   mins, S, M unchanged.
/// One B matrix of the mmul (16 columns x 16 rows) is then 128 contiguous bytes.
/// The dense overlay reads the blocks as stored, so this is for MoE layers only.
void q4k_relayout_for_mmul(u8* data, size_t bytes) {
    constexpr size_t SCALES = 0, QS = 512;
    assert(bytes % Q4K_BLOCK == 0);
    const int64_t n_blocks = (int64_t)(bytes / Q4K_BLOCK);
    // every block is independent: split them over the host cores
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < n_blocks; b++) {
        u8* blk = data + b * Q4K_BLOCK;
        std::array<u8, Q4K_BLOCK> old;
        memcpy(old.data(), blk, Q4K_BLOCK);
        for (int k = 0; k < 16; k++)                     // 16 columns per k step
            for (int t = 0; t < 2; t++)                  // rows 16t .. 16t+15
                for (int c = 0; c < 16; c++)
                    memcpy(blk + QS + (k * 2 + t) * 128 + c * 8,
                           old.data() + QS + (k * 16 + c) * 16 + t * 8, 8);
        for (int h = 0; h < 2; h++)
            for (int t = 0; t < 2; t++)
                for (int a = 0; a < 4; a++)
                    memcpy(blk + SCALES + ((h * 2 + t) * 4 + a) * 16,
                           old.data() + SCALES + (4 * h + a) * 32 + 16 * t, 16);
    }
}

/// bytes of `rows` rows of a Q4_K matrix with `in_dim` columns
size_t q4k_row_bytes(size_t rows, size_t in_dim) {
    return rows / QXNX_ROW_BLOCK_SIZE * (in_dim / QXNX_COL_BLOCK_SIZE) * Q4K_BLOCK;
}

/// [up rows 0..n | gate rows 0..n | up rows n..2n | gate rows n..2n | ...], n = rows_per_piece,
/// so every proj round holds matching up and gate rows for the GLU
void interleave_up_gate(u8* out, const u8* up, const u8* gate, size_t rows, size_t rows_per_piece, size_t in_dim) {
    const size_t piece = q4k_row_bytes(rows_per_piece, in_dim);
    for (size_t r = 0; r < rows; r += rows_per_piece) {
        memcpy(out, up, piece);    out += piece; up += piece;
        memcpy(out, gate, piece);  out += piece; gate += piece;
    }
}

/// load one tensor and check its size against the descriptor
buffer<u8> load_checked(Q4NX& q4nx, weight_desc_t& desc, int layer_idx) {
    buffer<u8> w;
    q4nx.load_weights(w, desc.format_name(layer_idx));
    if (w.size() != desc.get_size()) {
        throw std::runtime_error("hy_mem_npu: " + desc.format_name(layer_idx) + " is " + std::to_string(w.size()) +
                                 " bytes, expected " + std::to_string(desc.get_size()));
    }
    return w;
}

/// k, v, q back to back (the proj engine runs them as one matrix), then o
template <typename LayerDef>
void load_attention(LayerDef& L, int layer_idx, Q4NX& q4nx, u8* proj, uint32_t D, uint32_t DQ) {
    assert(L.attn_v.offset == L.attn_k.offset + L.attn_k.get_size());
    assert(L.attn_q.offset == L.attn_v.offset + L.attn_v.get_size());
    for (weight_desc_t* w : {&L.attn_k, &L.attn_v, &L.attn_q}) {
        buffer<u8> src = load_checked(q4nx, *w, layer_idx);
        reorder_cpy(proj + w->offset, src.data(), src.size(), D, Q4K_BLOCK);
        w->load();
    }
    buffer<u8> w_o = load_checked(q4nx, L.attn_output, layer_idx);
    reorder_cpy(proj + L.attn_output.offset, w_o.data(), w_o.size(), DQ, Q4K_BLOCK);
    L.attn_output.load();
}

/// q_norm / k_norm go behind the cos/sin table: [cos | sin | q_norm | k_norm]
template <typename LayerDef>
void load_qk_norms(LayerDef& L, int layer_idx, Q4NX& q4nx, bf16* rope_rms, uint32_t RoPE_DIM, uint32_t DH) {
    buffer<bf16> q_norm, k_norm;
    q4nx.load_weights(q_norm, L.attn_q_norm.format_name(layer_idx));
    q4nx.load_weights(k_norm, L.attn_k_norm.format_name(layer_idx));
    memcpy(rope_rms + RoPE_DIM, q_norm.data(), DH * sizeof(bf16));
    memcpy(rope_rms + RoPE_DIM + DH, k_norm.data(), DH * sizeof(bf16));
    L.attn_q_norm.load();
    L.attn_k_norm.load();
}

template <typename LayerDef>
void check_ready(LayerDef& L, int layer_idx) {
    for (weight_desc_t* w : L.all()) {
        if (!w->ready()) {
            throw std::runtime_error("hy_mem_npu: weight not loaded: " + w->format_name(layer_idx));
        }
    }
}

}  // namespace

void hy_mem_desc::load_dense_layer(int layer_idx, Q4NX& q4nx, buffer<u8>& proj, buffer<bf16>& rms, buffer<bf16>& rope_rms) {
    assert(layer_kind(layer_idx) == hy_mem_layer_kind::dense);
    assert(proj.size() >= dense_proj_bytes);
    u8* p = proj.data();

    // up/gate: 2 * 6912 / 512 = 27 rounds of [up 256 | gate 256]
    {
        assert(dense.ffn_gate.offset == dense.ffn_up.offset + dense.ffn_up.get_size());
        buffer<u8> w_up = load_checked(q4nx, dense.ffn_up, layer_idx);
        buffer<u8> w_gate = load_checked(q4nx, dense.ffn_gate, layer_idx);
        buffer<u8> both(w_up.size() + w_gate.size());
        interleave_up_gate(both.data(), w_up.data(), w_gate.data(), dense_width, DENSE_ROWS_PER_ROUND, D);
        reorder_cpy(p + dense.ffn_up.offset, both.data(), both.size(), D, Q4K_BLOCK);
        dense.ffn_up.load();
        dense.ffn_gate.load();
    }
    // down: plain row order
    {
        buffer<u8> w_down = load_checked(q4nx, dense.ffn_down, layer_idx);
        reorder_cpy(p + dense.ffn_down.offset, w_down.data(), w_down.size(), dense_width, Q4K_BLOCK);
        dense.ffn_down.load();
    }
    load_attention(dense, layer_idx, q4nx, p, D, DQ);

    // [input_layernorm | post_attention_layernorm]
    for (weight_desc_t* w : dense.rms_all()) {
        buffer<bf16> src;
        q4nx.load_weights(src, w->format_name(layer_idx));
        memcpy((u8*)rms.data() + w->offset, src.data(), w->get_size());
        w->load();
    }
    load_qk_norms(dense, layer_idx, q4nx, rope_rms.data(), RoPE_DIM, DH);
    check_ready(dense, layer_idx);
    // no q4k_relayout_for_mmul: this overlay's Q4_K kernel reads the blocks as stored
}

void hy_mem_desc::load_moe_layer(int layer_idx, Q4NX& q4nx, buffer<u8>& proj, buffer<bf16>& rms, buffer<bf16>& rope_rms,
                                 std::vector<float>& router_bias) {
    assert(layer_kind(layer_idx) == hy_mem_layer_kind::moe);
    assert(proj.size() >= moe_proj_bytes);
    u8* p = proj.data();

    // ---- routed experts ----
    // One expert = [channel 0 stream | channel 1 stream], expert e at
    // e * (its gate/up + down bytes) from the up region's start (the router uses
    // the same offsets). Each channel stream is that channel's gate/up blocks then
    // its down blocks, in the order the proj cores read them, so the router starts
    // one linear BD per channel per expert.
    //   gate/up rows: the 4 cores of a column take 4 row blocks per round; with
    //     64-row pieces cores 0, 1 make up and cores 2, 3 make gate, so one round
    //     of a column is [up 64 | gate 64]
    //   down rows: in order
    //   channels: reorder_cpy(..., 4) puts the 4 blocks of a group next to each
    //     other; channel 0 feeds cores 0, 1 (the first 2), channel 1 cores 2, 3
    {
        weight_desc_t& d_up = moe.ffn_up_exps;
        weight_desc_t& d_gate = moe.ffn_gate_exps;
        weight_desc_t& d_down = moe.ffn_down_exps;
        assert(d_gate.offset == d_up.offset + d_up.get_size());
        assert(d_down.offset == d_gate.offset + d_gate.get_size());
        buffer<u8> w_up = load_checked(q4nx, d_up, layer_idx);
        buffer<u8> w_gate = load_checked(q4nx, d_gate, layer_idx);
        buffer<u8> w_down = load_checked(q4nx, d_down, layer_idx);

        const size_t up_per_expert = d_up.get_size() / num_experts;
        const size_t down_per_expert = d_down.get_size() / num_experts;
        assert(up_per_expert == q4k_row_bytes(expert_width, D));
        const size_t gate_up_bytes = 2 * up_per_expert;
        const size_t expert_bytes = gate_up_bytes + down_per_expert;
        const size_t channel_bytes = expert_bytes / 2;
        const size_t group = 4 * Q4K_BLOCK;          // the 4 blocks of one group

        // ch0 takes the first 2 blocks of every group of `src`, ch1 the last 2
        auto split_channels = [&](u8*& ch0, u8*& ch1, const u8* src, size_t bytes) {
            assert(bytes % group == 0);
            for (size_t g = 0; g < bytes; g += group) {
                memcpy(ch0, src + g, group / 2);              ch0 += group / 2;
                memcpy(ch1, src + g + group / 2, group / 2);  ch1 += group / 2;
            }
        };

        // every expert writes its own byte range: run them on all host cores
        #pragma omp parallel
        {
            std::vector<u8> both(gate_up_bytes), gate_up(gate_up_bytes), down(down_per_expert);
            #pragma omp for schedule(static)
            for (int e = 0; e < (int)num_experts; e++) {
                interleave_up_gate(both.data(), w_up.data() + e * up_per_expert, w_gate.data() + e * up_per_expert,
                                   expert_width, EXPERT_ROWS_PER_ROUND, D);
                reorder_cpy(gate_up.data(), both.data(), gate_up_bytes, D, Q4K_BLOCK, 4);
                reorder_cpy(down.data(), w_down.data() + e * down_per_expert, down_per_expert, expert_width, Q4K_BLOCK, 4);

                u8* ch0 = p + d_up.offset + e * expert_bytes;
                u8* ch1 = ch0 + channel_bytes;
                split_channels(ch0, ch1, gate_up.data(), gate_up_bytes);
                split_channels(ch0, ch1, down.data(), down_per_expert);
                assert(ch0 == p + d_up.offset + e * expert_bytes + channel_bytes);
            }
        }
        d_up.load();
        d_gate.load();
        d_down.load();
    }
    // ---- shared expert: up/gate one proj round = [up 256 | gate 256], down in row order ----
    {
        assert(moe.ffn_gate_shexp.offset == moe.ffn_up_shexp.offset + moe.ffn_up_shexp.get_size());
        buffer<u8> w_up = load_checked(q4nx, moe.ffn_up_shexp, layer_idx);
        buffer<u8> w_gate = load_checked(q4nx, moe.ffn_gate_shexp, layer_idx);
        buffer<u8> both(w_up.size() + w_gate.size());
        interleave_up_gate(both.data(), w_up.data(), w_gate.data(), expert_width, SHARED_ROWS_PER_ROUND, D);
        reorder_cpy(p + moe.ffn_up_shexp.offset, both.data(), both.size(), D, Q4K_BLOCK);
        moe.ffn_up_shexp.load();
        moe.ffn_gate_shexp.load();

        buffer<u8> w_down = load_checked(q4nx, moe.ffn_down_shexp, layer_idx);
        reorder_cpy(p + moe.ffn_down_shexp.offset, w_down.data(), w_down.size(), expert_width, Q4K_BLOCK);
        moe.ffn_down_shexp.load();
    }
    load_attention(moe, layer_idx, q4nx, p, D, DQ);

    // every proj weight block into the int8-mmul kernel's order (Q4K_INT8_MMUL 1)
    q4k_relayout_for_mmul(p, moe_proj_bytes);

    // ---- rms buffer: the two norms, then the router weight ----
    for (weight_desc_t* w : {&moe.input_layer_norm, &moe.post_attntion_norm}) {
        buffer<bf16> src;
        q4nx.load_weights(src, w->format_name(layer_idx));
        memcpy((u8*)rms.data() + w->offset, src.data(), w->get_size());
        w->load();
    }
    // Router weight, stored in the exact order the router reads it, so its DMA is
    // one straight 512 KB read. model.q4nx holds it as [D][E]; the router takes
    // 32-expert x 256-input blocks, expert-block major, each block input-major
    // with the 32 experts adjacent.
    {
        buffer<bf16> w_router;
        q4nx.load_weights(w_router, moe.ffn_gate_inp.format_name(layer_idx));
        assert(w_router.size() == (size_t)D * num_experts);
        bf16* dst = (bf16*)((u8*)rms.data() + moe.ffn_gate_inp.offset);
        for (uint32_t eb = 0; eb < num_experts; eb += ROUTER_ROW_BLOCK) {
            for (uint32_t db = 0; db < D; db += ROUTER_COL_BLOCK) {
                for (uint32_t d = db; d < db + ROUTER_COL_BLOCK; d++) {
                    memcpy(dst, w_router.data() + (size_t)d * num_experts + eb, ROUTER_ROW_BLOCK * sizeof(bf16));
                    dst += ROUTER_ROW_BLOCK;
                }
            }
        }
        moe.ffn_gate_inp.load();
    }
    // router expert bias: fp32, only picks the experts (see router_math.hpp)
    {
        buffer<u8> bias_bytes;
        q4nx.load_weights(bias_bytes, "model.layer." + std::to_string(layer_idx) + ".moe_expert_bias");
        if (bias_bytes.size() != num_experts * sizeof(float)) {
            throw std::runtime_error("hy_mem_npu: moe_expert_bias of layer " + std::to_string(layer_idx) + " has the wrong size");
        }
        router_bias.resize(num_experts);
        memcpy(router_bias.data(), bias_bytes.data(), bias_bytes.size());
    }
    load_qk_norms(moe, layer_idx, q4nx, rope_rms.data(), RoPE_DIM, DH);
    check_ready(moe, layer_idx);
}

void hy_mem_desc::make_prefill_proj(buffer<u8>& dst, const buffer<u8>& proj) {
    assert(dst.size() == proj.size());
    memcpy(dst.data(), proj.data(), proj.size());
    q4k_relayout_for_mmul(dst.data(), dst.size());
}

/// lm_head.weight, Q8_0 blocks [scales bf16 (8 groups x 32 rows) | q int8 (256 cols x 32 rows)].
/// A column's weights arrive on two shim channels, one per pair of cores, so each
/// column-round has to be two contiguous halves in DDR rather than one run with all
/// four cores' blocks interleaved: rows r+0/r+1 of the group for the stream that
/// feeds cores 0 and 1, rows r+2/r+3 for the stream that feeds cores 2 and 3. Within
/// a half the order is still k-major, one block per core, because that is exactly one
/// memtile slot. See lm_head_memtile in lm_head_npu_bin/iron/q4k_mm.py.
void hy_mem_desc::load_lm_head(Q4NX& q4nx, buffer<u8>& dst) {
    constexpr int vertical_blocks = 4;
    constexpr int rows_per_stream = vertical_blocks / 2;
    buffer<u8> src;
    q4nx.load_weights(src, "lm_head.weight");
    const int blocks_per_row = D / QXNX_COL_BLOCK_SIZE;
    const int rows = (int)(src.size() / Q80_BLOCK_BYTES / blocks_per_row);
    assert(dst.size() >= (size_t)vocab_size_padded / 32 * blocks_per_row * Q80_BLOCK_BYTES);

    u8* dst_ptr = dst.data();
    for (int r = 0; r < rows; r += vertical_blocks) {
        for (int half = 0; half < 2; half++) {
            for (int c = 0; c < blocks_per_row; c++) {
                for (int i = 0; i < rows_per_stream; i++) {
                    const int row = r + half * rows_per_stream + i;
                    const u8* src_ptr = (row < rows)
                        ? src.data() + (size_t)(row * blocks_per_row + c) * Q80_BLOCK_BYTES
                        : src.data();   // useless padding
                    memcpy(dst_ptr, src_ptr, Q80_BLOCK_BYTES);
                    dst_ptr += Q80_BLOCK_BYTES;
                }
            }
        }
    }
}
