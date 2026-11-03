/// \file kv_rows_state.hpp
/// \brief saved state of a full-attention-only engine: K/V rows per layer, copied out of / back into the live cache
/// \note Engines whose every layer is full attention (llama, qwen3, phi4, qwen2, nanbeige, ...) keep one K row and one
///       V row per position per layer, in a layer buffer whose layout depends on MAX_L. A saved state holds plain
///       rows (`hybrid_engine_state` with no recurrent part), so it does not depend on MAX_L and can be written back
///       after the live cache has grown. Two live layouts exist:
///       - striped: K and V are each split in two halves of DK/2 / DV/2 elements stored in four planes
///         (K03 at 0, K47, V03, V47 at engine-given element offsets), row stride DK/2
///       - split:   K rows from the start of the buffer, V rows from `v_offset` (the middle), row stride DK / DV
///       The engine builds a kv_row_layout from its own offsets and calls the functions below from its
///       read_state / write_state / write_state_kv; nothing here knows about slots.
///       An engine that mixes attention layers with other kinds (lfm2: convolution layers) passes `attn`, one flag per layer;
///       the layers not flagged are left alone here (no rows saved, copied or zeroed) and the engine keeps their state itself
///       in `hybrid_engine_state::state`.
#pragma once
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "utils/hybrid_state.hpp"

struct kv_row_layout {
    size_t DK = 0;       ///< elements of one K row (all kv heads)
    size_t DV = 0;       ///< elements of one V row
    bool striped = true; ///< four-plane layout (see above) instead of K rows then V rows
    size_t k47 = 0, v03 = 0, v47 = 0; ///< striped: element offsets of the planes (K03 is at 0)
    size_t v_offset = 0; ///< split: element offset of the first V row

    /// \brief striped: elements of one row in one plane. The four planes split the layer buffer evenly, MAX_L rows each,
    ///        so a row is (DK + DV) / 4 = DK / 2 elements (not head_dim * 4, which is the same only with 8 KV heads)
    size_t stride() const { return DK / 2; }

    /// \brief whether the layout is one this file can address
    bool valid() const { return DK > 0 && DV > 0 && (!striped || (DK == DV && DK % 2 == 0)); }
};

/// \brief an empty state for `layers` full-attention layers; `row_bytes` of a saved row = layers * (DK + DV) * 2
/// \param fixed_bytes bytes of the per-layer state the engine saves besides the rows (0 for pure attention)
inline std::unique_ptr<hybrid_engine_state> kv_rows_new_state(size_t layers, const kv_row_layout& l, size_t fixed_bytes = 0) {
    if (!l.valid()) return nullptr;
    return std::make_unique<hybrid_engine_state>(fixed_bytes, layers * (l.DK + l.DV) * sizeof(bf16));
}

/// \brief whether layer `i` keeps K/V rows (`attn` null: all layers do)
inline bool kv_rows_layer(const std::vector<bool>* attn, size_t i) { return !attn || (*attn)[i]; }

/// \brief copy `rows` rows starting at `from` between a layer buffer `kv` and dense row arrays `k` (rows * DK) / `v` (rows * DV)
/// \param to_state true: kv -> k, v ; false: k, v -> kv
inline void kv_rows_copy(bf16* kv, const kv_row_layout& l, size_t from, size_t rows, bf16* k, bf16* v, bool to_state) {
    auto move = [to_state](bf16* live, bf16* dense, size_t n) {
        if (to_state) std::memcpy(dense, live, n * sizeof(bf16));
        else std::memcpy(live, dense, n * sizeof(bf16));
    };
    for (size_t r = 0; r < rows; r++) {
        const size_t row = from + r;
        if (l.striped) {
            const size_t at = row * l.stride(), half = l.stride();
            move(kv + at, k + r * l.DK, half);
            move(kv + l.k47 + at, k + r * l.DK + half, half);
            move(kv + l.v03 + at, v + r * l.DV, half);
            move(kv + l.v47 + at, v + r * l.DV + half, half);
        } else {
            move(kv + row * l.DK, k + r * l.DK, l.DK);
            move(kv + l.v_offset + row * l.DV, v + r * l.DV, l.DV);
        }
    }
}

/// \brief zero the rows [from, to) of every layer of the live cache
/// \note The kernels do not keep rows past the context length out of attention completely: what an earlier conversation left there
///       changes the answer slightly. A state written back, and a cleared context, must look like a fresh engine, so the engine
///       calls this with `from` = the length it now holds and `to` = the highest length it has reached since the rows were last zero.
///       Rows are contiguous within a plane, so each plane is one memset.
template <class Caches>
void kv_rows_zero(Caches& kv_caches, const kv_row_layout& l, size_t from, size_t to, const std::vector<bool>* attn = nullptr) {
    if (!l.valid() || to <= from) return;
    const size_t rows = to - from;
    for (size_t i = 0; i < kv_caches.size(); i++) {
        if (!kv_rows_layer(attn, i)) continue;
        kv_caches[i].sync_from_device();
        bf16* kv = kv_caches[i].data();
        if (l.striped) {
            const size_t stride = l.stride();
            for (size_t plane : {size_t(0), l.k47, l.v03, l.v47}) {
                std::memset(kv + plane + from * stride, 0, rows * stride * sizeof(bf16));
            }
        } else {
            std::memset(kv + from * l.DK, 0, rows * l.DK * sizeof(bf16));
            std::memset(kv + l.v_offset + from * l.DV, 0, rows * l.DV * sizeof(bf16));
        }
        kv_caches[i].sync_to_device();
    }
}

/// \brief copy the live rows `kv_from` .. `length` of every layer into `st`; returns the length, or -1
template <class Caches>
int kv_rows_read(hybrid_engine_state& st, Caches& kv_caches, const kv_row_layout& l, int length, int kv_from, int general_token_idx = -1, const std::vector<bool>* attn = nullptr) {
    if (!l.valid()) return -1;
    kv_from = std::clamp(kv_from, 0, length);
    const size_t rows = static_cast<size_t>(length - kv_from), layers = kv_caches.size();
    st.state.assign(layers, buffer<bf16>());
    st.kv_k.resize(layers);
    st.kv_v.resize(layers);
    for (size_t i = 0; i < layers; i++) {
        if (rows == 0 || !kv_rows_layer(attn, i)) { // a reused state must not keep rows of its previous contents
            st.kv_k[i] = buffer<bf16>();
            st.kv_v[i] = buffer<bf16>();
            continue;
        }
        kv_caches[i].sync_from_device();
        if (st.kv_k[i].size() != rows * l.DK) st.kv_k[i] = buffer<bf16>(rows * l.DK);
        if (st.kv_v[i].size() != rows * l.DV) st.kv_v[i] = buffer<bf16>(rows * l.DV);
        kv_rows_copy(kv_caches[i].data(), l, kv_from, rows, st.kv_k[i].data(), st.kv_v[i].data(), true);
    }
    st.filled = true;
    st.context_length = length;
    st.kv_from = kv_from;
    st.general_token_idx = general_token_idx >= 0 ? general_token_idx : length; // the rope position runs on its own once pictures are in the context
    return length;
}

/// \brief whether `st` holds what a write needs for this cache (a state of another engine would not)
template <class Caches>
bool kv_rows_fits(const hybrid_engine_state& st, const Caches& kv_caches, const kv_row_layout& l, size_t max_length, const std::vector<bool>* attn = nullptr) {
    if (!l.valid() || !st.filled || st.state.size() != kv_caches.size() || st.kv_k.size() != kv_caches.size() ||
        st.kv_v.size() != kv_caches.size() || st.context_length < st.kv_from || static_cast<size_t>(st.context_length) > max_length) {
        return false;
    }
    const size_t rows = static_cast<size_t>(st.context_length - st.kv_from);
    for (size_t i = 0; i < kv_caches.size(); i++) {
        if (rows > 0 && kv_rows_layer(attn, i) && (st.kv_k[i].size() != rows * l.DK || st.kv_v[i].size() != rows * l.DV)) return false;
    }
    return true;
}

/// \brief put the rows `st` holds back into the live cache; returns the state's context length, or -1
/// \note the caller sets the context length afterwards (write_state) or leaves it (write_state_kv)
template <class Caches>
int kv_rows_write(const hybrid_engine_state& st, Caches& kv_caches, const kv_row_layout& l, size_t max_length, const std::vector<bool>* attn = nullptr) {
    if (!kv_rows_fits(st, kv_caches, l, max_length, attn)) return -1;
    const size_t rows = static_cast<size_t>(st.context_length - st.kv_from);
    if (rows == 0) return st.context_length;
    for (size_t i = 0; i < kv_caches.size(); i++) {
        if (!kv_rows_layer(attn, i)) continue;
        kv_caches[i].sync_from_device();
        kv_rows_copy(kv_caches[i].data(), l, st.kv_from, rows, const_cast<bf16*>(st.kv_k[i].data()), const_cast<bf16*>(st.kv_v[i].data()), false);
        kv_caches[i].sync_to_device();
    }
    return st.context_length;
}
