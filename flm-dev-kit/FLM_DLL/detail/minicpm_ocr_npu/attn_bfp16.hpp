// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file attn_bfp16.hpp
/// \brief Host side of attn_bfp16.xclbin (FLM_Xclbin/minicpm_ocr/gated_attention_prefill): q, K and V^T as bfp16
///        tiles, and each attention layer's bfp16 copy of its KV cache. See README.md "Prefill attention on
///        bfp16 tiles".
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <immintrin.h>
#include "minicpm_ocr_model_def.hpp"
#include "minicpm_ocr_prefill_cpu.hpp"
#include "bfp16_pack.hpp"

namespace minicpm_ocr_attn_bfp16 {

constexpr int group_rows = 16;
constexpr int key_block = 128;

inline size_t row_bytes(int DH) { return (size_t)DH * 9 / 8; }

// tile_group / vt_tile_group: bfp16_pack.hpp (shared with decode_kv_bfp16.hpp and the decode kernel test)
using minicpm_ocr_bfp16::tile_group;
using minicpm_ocr_bfp16::vt_tile_group;

// q [L_padded][heads * DH] -> q tiles [head][L_padded / 16][DH / 8][2][72 B]
inline void q_tiles(buffer<u8>& qb, const bf16* q, int L_padded, int heads, int DH) {
    const int DQ = heads * DH;
    const int groups = L_padded / group_rows;
    #pragma omp parallel for num_threads(minicpm_ocr_cpu::prefill_threads()) schedule(static)
    for (int i = 0; i < heads * groups; i++) {
        const int h = i / groups;
        const int group = i % groups;
        tile_group(qb.data() + ((size_t)h * L_padded + group * group_rows) * row_bytes(DH),
                   q + (size_t)group * group_rows * DQ + h * DH, DH, DQ);
    }
    qb.sync_to_device();
}

// Each attention layer's K tiles, then V^T tiles: [K head 0][K head 1][V^T head 0][V^T head 1], `rows` rows each,
// rows [0, valid) already converted from the bf16 cache.
struct kv_copy {
    buffer<u8> tiles;
    int rows = 0;
    int valid = 0;
};

// What one call converts, from kv_store::prepare() to its two halves (kv_store::convert).
struct kv_plan {
    kv_copy* copy = nullptr;
    int L_begin = 0;
    int L = 0;
    int first = 0;
    int last = 0;
    int cache_rows = 0;
    int group_begin = 0;
    int groups = 0;
};

enum class kv_half { k, v };

struct kv_store {
    std::vector<kv_copy> layers;
    int rows = 0;

    // the host rewrote the bf16 caches from row from_row on (clear, restore, write_state): convert those rows again
    void cache_changed(int from_row) {
        for (auto& c : layers) {
            c.valid = std::min(c.valid, from_row);
        }
    }

    static int needed_rows(int L_begin, int L_padded) {
        return (L_begin + L_padded + key_block - 1) / key_block * key_block;
    }

    // the rows every copy has after a call with L_padded query rows from L_begin: up to the last 128-key block the
    // kernel reads, grown by at least half when it grows
    int rows_for(int L_begin, int L_padded, int cache_rows) const {
        const int needed = needed_rows(L_begin, L_padded);
        if (needed <= rows) {
            return rows;
        }
        const int grown = (std::max(needed, rows * 3 / 2) + 4095) / 4096 * 4096;
        return std::max(needed, std::min(grown, (cache_rows + key_block - 1) / key_block * key_block));
    }

    // layer layer_idx's copy at `rows` rows: a new BO when it is smaller (all its rows are converted again)
    kv_copy& allocate(npu_xclbin_manager* npu, const minicpm_ocr_desc* desc, int layer_idx) {
        if ((int)layers.size() <= layer_idx) {
            layers.resize(layer_idx + 1);
        }
        kv_copy& c = layers[layer_idx];
        if (c.rows != rows) {
            c.tiles = npu->create_bo_buffer<u8>((size_t)2 * desc->num_kv_heads * rows * row_bytes(desc->DH));
            c.rows = rows;
            c.valid = 0;
        }
        return c;
    }

    // whether reserve() would allocate anything for these layers
    bool short_for(const std::vector<int>& layer_ids, int L_begin, int L_padded, int cache_rows) const {
        const int r = rows_for(L_begin, L_padded, cache_rows);
        for (int id : layer_ids) {
            if (id >= (int)layers.size() || layers[id].rows != r) {
                return true;
            }
        }
        return false;
    }

    // the copies of these layers sized for a call ahead of time (the same rows and BOs prepare() would make)
    void reserve(npu_xclbin_manager* npu, const minicpm_ocr_desc* desc, const std::vector<int>& layer_ids,
                 int L_begin, int L_padded, int cache_rows) {
        rows = rows_for(L_begin, L_padded, cache_rows);
        for (int id : layer_ids) {
            allocate(npu, desc, id);
        }
    }

    // Size layer layer_idx's copy for L_padded query rows from L_begin and work out which rows this call converts.
    // Nothing is read from the cache yet.
    kv_plan prepare(npu_xclbin_manager* npu, const minicpm_ocr_desc* desc, int layer_idx, buffer<bf16>& cache,
                    int L_begin, int L, int L_padded) {
        const int needed = needed_rows(L_begin, L_padded);
        const int cache_rows = (int)(cache.size() / 2 / desc->DK);
        rows = rows_for(L_begin, L_padded, cache_rows);
        kv_copy& c = allocate(npu, desc, layer_idx);

        kv_plan p;
        p.copy = &c;
        p.L_begin = L_begin;
        p.L = L;
        p.first = std::min(c.valid, L_begin) / group_rows * group_rows;
        p.last = std::min(needed, cache_rows);
        p.cache_rows = cache_rows;
        p.group_begin = p.first / group_rows;
        p.groups = needed / group_rows - p.group_begin;
        return p;
    }

    // One half (K or V) of the copy, once the host wrote that half's new rows [L_begin, L_begin + L) into the bf16
    // cache. Its other rows (rows decode added since the last array call, and the rows past L_begin + L the kernel
    // also reads) are pulled from the device first: the host never pulls them, and the kernel used to read them
    // from the device.
    void convert(const kv_plan& p, const minicpm_ocr_desc* desc, buffer<bf16>& cache, kv_half half) {
        const int DH = desc->DH;
        const int DK = desc->DK;
        const int kv_heads = desc->num_kv_heads;
        const size_t rb = row_bytes(DH);
        kv_copy& c = *p.copy;
        const size_t base = half == kv_half::k ? 0 : cache.size() / 2;
        auto pull = [&](int from, int to) {
            if (to <= from) return;
            minicpm_ocr_cpu::sync_range_from_device(cache, base + (size_t)from * DK, (size_t)(to - from) * DK);
        };
        pull(p.first, std::min(p.L_begin, p.last));
        pull(std::max(p.first, p.L_begin + p.L), p.last);

        const bf16* src = cache.data() + base;
        const int part0 = half == kv_half::k ? 0 : kv_heads;
        const int groups = p.groups;
        #pragma omp parallel for num_threads(minicpm_ocr_cpu::prefill_threads()) schedule(static)
        for (int i = 0; i < kv_heads * groups; i++) {
            const int h = i / groups;
            const int g = p.group_begin + i % groups;
            uint8_t* out = c.tiles.data() + ((size_t)(part0 + h) * c.rows + (size_t)g * group_rows) * rb;
            if ((g + 1) * group_rows > p.cache_rows) {
                memset(out, 0, group_rows * rb);
            }
            else if (half == kv_half::k) {
                tile_group(out, src + (size_t)g * group_rows * DK + h * DH, DH, DK);
            }
            else {
                vt_tile_group(out, src + (size_t)g * group_rows * DK + h * DH, DH, DK);
            }
        }
        for (int part = part0; part < part0 + kv_heads; part++) {
            const size_t offset = ((size_t)part * c.rows + (size_t)p.group_begin * group_rows) * rb;
            minicpm_ocr_cpu::sync_range_to_device(c.tiles, offset, (size_t)groups * group_rows * rb);
        }
    }

    // both halves converted: rows [0, L_begin + L) of the copy match the bf16 cache
    kv_copy& finish(const kv_plan& p) {
        p.copy->valid = p.L_begin + p.L;
        return *p.copy;
    }

    // Bring layer layer_idx's copy up to date with its bf16 cache in one go, after fill_kv_cache wrote both halves.
    kv_copy& update(npu_xclbin_manager* npu, const minicpm_ocr_desc* desc, int layer_idx, buffer<bf16>& cache,
                    int L_begin, int L, int L_padded) {
        const kv_plan p = prepare(npu, desc, layer_idx, cache, L_begin, L, L_padded);
        convert(p, desc, cache, kv_half::k);
        convert(p, desc, cache, kv_half::v);
        return finish(p);
    }
};

} // namespace minicpm_ocr_attn_bfp16
