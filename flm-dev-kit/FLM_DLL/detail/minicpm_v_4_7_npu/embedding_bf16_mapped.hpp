// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __EMBEDDING_BF16_MAPPED_HPP__
#define __EMBEDDING_BF16_MAPPED_HPP__
#include "typedef.hpp"
#include "buffer.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_utils/mapped_file.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

/// bf16 [vocab, d_model] token table, read one row per token. Same interface as Embedding,
/// but the table is mapped read-only from model.q4nx instead of copied into a vocab x d_model
/// buffer (~0.5 GiB on MiniCPM-V 1B). owned is the fallback when the range cannot be mapped
/// (or when FLM_MINICPM_EMBED_MAP=0, for A/B runs).
class EmbeddingBF16Mapped{
private:
    size_t vocab_size;
    size_t d_model;
    mapped_file mapped;
    buffer<bf16> owned;
    const uint8_t* base = nullptr; // row r at base + r * d_model * 2, possibly unaligned in the file
    vdtype y;
    bool prefetch_on = true; // FLM_MINICPM_EMBED_PREFETCH=0 turns the hint off (cold-cache A/B only)

    void copy_row(int x, bf16* dst) const {
        if (x < 0 || (size_t)x >= vocab_size){
            std::memset(dst, 0, d_model * sizeof(bf16));
            return;
        }
        std::memcpy(dst, base + (size_t)x * d_model * sizeof(bf16), d_model * sizeof(bf16));
    }

public:
    EmbeddingBF16Mapped(size_t vocab_size, size_t d_model) : vocab_size(vocab_size), d_model(d_model) {
        this->y.resize(d_model);
    }

    /// \param weight_name the tensor without ".weight" (as Embedding::init_weights)
    void init_weights(Q4NX& q4nx, const std::string& q4nx_path, const std::string& weight_name){
        const std::string name = weight_name + ".weight";
        tensor_metadata meta = q4nx.get_tensor_metadata(name);
        const size_t nbytes = meta.offsets[1] - meta.offsets[0];
        const char* off = std::getenv("FLM_MINICPM_EMBED_MAP");
        const bool want_map = !(off && off[0] == '0');
        const char* pf = std::getenv("FLM_MINICPM_EMBED_PREFETCH");
        this->prefetch_on = !(pf && pf[0] == '0');
        if (want_map && nbytes >= vocab_size * d_model * sizeof(bf16)
            && this->mapped.open(q4nx_path, meta.offsets[0], nbytes)){
            this->base = this->mapped.data();
        }
        else{
            this->owned = buffer<bf16>(vocab_size * d_model);
            q4nx.load_weights(this->owned, name);
            this->base = reinterpret_cast<const uint8_t*>(this->owned.begin());
        }
    }

    /// \brief Hint that these token rows are about to be looked up (see embedding_q8_0::prefetch).
    void prefetch(const int* ids, size_t n){
        if (!this->mapped || !this->prefetch_on) return;
        std::vector<size_t> rows;
        rows.reserve(n);
        for (size_t i = 0; i < n; i++){
            if (ids[i] >= 0 && (size_t)ids[i] < vocab_size) rows.push_back((size_t)ids[i]);
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        for (size_t r : rows) this->mapped.will_need(r * d_model * sizeof(bf16), d_model * sizeof(bf16));
    }

    vdtype forward(int x){
        copy_row(x, this->y.begin());
        return this->y;
    }

    vdtype forward(int x, vdtype& out){
        copy_row(x, out.begin());
        return out;
    }
};

#endif
