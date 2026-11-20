// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file decode_kv_bfp16.hpp
/// \brief layer_with_lm_head_kvb.xclbin (FLM_Xclbin/minicpm_ocr/decoding, KV_BFP16=1): each attention layer's bfp16
///        copy of its finished 16-key KV chunks, which the decode attention reads through arg 3. See README.md
///        "Decode attention on bfp16 K and V".
#pragma once

#include <algorithm>
#include <cstring>
#include <vector>
#include "buffer.hpp"
#include "npu_utils/npu_utils.hpp"
#include "minicpm_ocr_prefill_cpu.hpp"
#include "bfp16_pack.hpp"

namespace minicpm_ocr_decode_kv {

constexpr int chunk_rows = minicpm_ocr_bfp16::decode_chunk_rows;
// arg 3 starts with what rope_rms_weights holds (RoPE cos / sin, q and k norm weights); the units follow
constexpr size_t head_area_bytes = 4096;
// the copies grow in steps of 256 chunks (4,096 tokens)
constexpr int grow_chunks = 256;

// arg 3 of an attention layer: [head area][K unit x cap][V unit x cap]; unit = one finished chunk of both KV heads
struct store {
    int kv_heads = 0;
    int DH = 0;
    size_t unit = 0;
    int cap = 0;
    std::vector<int> layer_ids;
    std::vector<buffer<u8>> bos;    // by layer index, attention layers only
    std::vector<int> valid;         // by layer index: chunks [0, valid) match the bf16 cache

    size_t k_base() const { return head_area_bytes; }
    size_t v_base() const { return head_area_bytes + (size_t)cap * unit; }
    size_t bytes() const { return v_base() + (size_t)cap * unit; }

    void init(npu_xclbin_manager* npu, int num_layers, const std::vector<int>& attn_layers, int kv_heads_, int DH_) {
        kv_heads = kv_heads_;
        DH = DH_;
        unit = minicpm_ocr_bfp16::decode_unit_bytes(kv_heads, DH);
        layer_ids = attn_layers;
        bos.resize(num_layers);
        valid.assign(num_layers, 0);
        cap = grow_chunks;
        for (int id : layer_ids) {
            bos[id] = npu->create_bo_buffer<u8>(bytes());
            memset(bos[id].data(), 0, bytes());
            bos[id].sync_to_device();
        }
    }

    // the host rewrote the bf16 caches from row from_row on (clear, restore, write_state): convert those chunks again
    void cache_changed(int from_row) {
        for (int& v : valid) {
            v = std::min(v, from_row / chunk_rows);
        }
    }

    // rope_rms_weights[layer] changed (every token: the RoPE position): copy it to the head area
    void set_head(int layer, const buffer<bf16>& rope) {
        const size_t n = rope.size() * sizeof(bf16);
        memcpy(bos[layer].data(), rope.data(), n);
        minicpm_ocr_cpu::sync_range_to_device(bos[layer], 0, (n + 63) / 64 * 64);
    }

    // Chunks [0, n) of every attention layer current with the bf16 caches. Returns true when the copies had to grow:
    // new BOs, so the caller makes its runs again. Rows the array wrote (decode) are pulled from the device first.
    bool ensure(npu_xclbin_manager* npu, int n, std::vector<buffer<bf16>>& caches, size_t V_offset,
                std::vector<buffer<bf16>>& rope) {
        bool grown = false;
        if (n > cap) {
            const int new_cap = (n + grow_chunks - 1) / grow_chunks * grow_chunks;
            const size_t old_k = k_base();
            const size_t old_v = v_base();
            cap = new_cap;
            for (int id : layer_ids) {
                buffer<u8> bo = npu->create_bo_buffer<u8>(bytes());
                memset(bo.data(), 0, bytes());
                memcpy(bo.data() + k_base(), bos[id].data() + old_k, (size_t)valid[id] * unit);
                memcpy(bo.data() + v_base(), bos[id].data() + old_v, (size_t)valid[id] * unit);
                bos[id] = std::move(bo);
                set_head(id, rope[id]);
                bos[id].sync_to_device();
            }
            grown = true;
        }

        struct item { int layer; int chunk; };
        std::vector<item> work;
        const size_t row = (size_t)kv_heads * DH;
        for (int id : layer_ids) {
            if (valid[id] >= n) {
                continue;
            }
            const size_t first = (size_t)valid[id] * chunk_rows * row;
            const size_t count = (size_t)(n - valid[id]) * chunk_rows * row;
            minicpm_ocr_cpu::sync_range_from_device(caches[id], first, count);
            minicpm_ocr_cpu::sync_range_from_device(caches[id], V_offset + first, count);
            for (int g = valid[id]; g < n; g++) {
                work.push_back({id, g});
            }
        }
        if (work.empty()) {
            return grown;
        }
        const int work_count = (int)work.size();
        #pragma omp parallel for num_threads(work_count > 8 ? minicpm_ocr_cpu::prefill_threads() : 1) schedule(static)
        for (int i = 0; i < work_count; i++) {
            const int id = work[i].layer;
            const int g = work[i].chunk;
            const size_t first = (size_t)g * chunk_rows * row;
            uint8_t* base = bos[id].data();
            minicpm_ocr_bfp16::pack_decode_chunk(base + k_base() + (size_t)g * unit, base + v_base() + (size_t)g * unit,
                                                 caches[id].data() + first, caches[id].data() + V_offset + first, kv_heads, DH);
        }
        for (int id : layer_ids) {
            if (valid[id] >= n) {
                continue;
            }
            const size_t from = (size_t)valid[id] * unit;
            const size_t count = (size_t)(n - valid[id]) * unit;
            minicpm_ocr_cpu::sync_range_to_device(bos[id], k_base() + from, count);
            minicpm_ocr_cpu::sync_range_to_device(bos[id], v_base() + from, count);
            valid[id] = n;
        }
        return grown;
    }
};

} // namespace minicpm_ocr_decode_kv
