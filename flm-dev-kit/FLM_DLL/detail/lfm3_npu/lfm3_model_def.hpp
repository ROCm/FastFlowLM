// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __LFM3_MODEL_DEF_HPP__
#define __LFM3_MODEL_DEF_HPP__
/// \file lfm3_model_def.hpp
/// \brief LFM3 dimensions read from config.json, the layer map, the RoPE tables and the weight loader.
///
/// LFM3-1.7B interleaves three block kinds: gated short convolution (width 7), sliding-window attention (window 256)
/// and full attention. Every block is followed by an FFN, and both the block and the FFN are wrapped in a sandwich norm
/// (norm before, norm after, then the residual add). Nothing here is hard-coded to one checkpoint: sizes, the layer
/// map, RoPE bases and the window all come from config.json.
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "modules/embedding.hpp"
#include "lfm3_cpu.hpp"

enum class lfm3_block_kind { conv, sliding_attention, full_attention };

struct lfm3_layer_weights {
    lfm3_block_kind kind = lfm3_block_kind::conv;
    bool is_attn() const { return kind != lfm3_block_kind::conv; }

    /// All projection weights of the layer in the device layout (one buffer: a device BO when the NPU stages are on).
    /// Layout (same as the LFM2 engine, so the same dequant kernels read it):
    ///   [ slot 0: 3D rows | slot 1: D rows | up/gate band | down ]
    /// an attention layer keeps q|k|v at the front of slot 0 and o_proj in slot 1; a conv layer keeps in_proj in slot 0
    /// and out_proj in slot 1.
    buffer<u8> proj;

    // sandwich norms, all [D] bf16
    buffer<bf16> operator_norm, post_operator_norm, ffn_norm, post_ffn_norm;
    // views into `proj`
    lfm3_cpu::qmat w1, w3, w2;             ///< SwiGLU: w1 = gate, w3 = up, w2 = down
    lfm3_cpu::qmat conv_in, conv_out;      ///< in_proj [3D x D], out_proj [D x D]
    lfm3_cpu::qmat q_proj, k_proj, v_proj, o_proj;
    // small tensors kept as plain bf16
    buffer<bf16> conv_taps;                 ///< [D][K], tap k multiplies the input k - (K - 1) steps back
    buffer<bf16> q_norm, k_norm;            ///< [DH]
    buffer<bf16> gate_proj;                 ///< [num_heads][D], BF16 (16 rows is below one Q4 block)
};

struct lfm3_desc {
    uint32_t num_hidden_layers = 0;
    uint32_t D = 0, DH = 0, num_heads = 0, num_kv_heads = 0, DQ = 0, DK = 0, DV = 0;
    uint32_t intermediate_size = 0, vocab_size = 0;
    uint32_t conv_kernel = 0;               ///< conv_L_cache (7)
    uint32_t sliding_window = 0;            ///< 256
    float eps = 1e-5f;
    std::vector<lfm3_block_kind> kinds;

    /// inverse frequencies per attention kind, DH/2 entries each. Partial rotary keeps the frequency progression of the
    /// full head dimension for the rotated angles and uses zero for the rest, which leaves those dims unrotated.
    std::vector<float> inv_freq_full, inv_freq_sliding;

    // ---- layout of a layer's proj buffer (bytes) ----
    static constexpr int UP_GATE_SLICE_ROWS = 512;      ///< up and gate are interleaved in slices of this many output rows
    static size_t qbytes(size_t rows, size_t cols) {
        return rows / lfm3_cpu::QB_ROWS * (cols / lfm3_cpu::QB_COLS) * lfm3_cpu::QB_BYTES;
    }
    size_t slot0_off() const { return 0; }
    size_t slot1_off() const { return qbytes(3 * D, D); }
    size_t upgate_off() const { return slot1_off() + qbytes(D, D); }
    size_t up_gate_slice_bytes() const { return qbytes(UP_GATE_SLICE_ROWS, D); }
    size_t down_off() const { return upgate_off() + 2 * qbytes(intermediate_size, D); }
    size_t proj_bytes() const { return down_off() + qbytes(D, intermediate_size); }
    size_t q_off() const { return slot0_off(); }
    size_t k_off() const { return q_off() + qbytes(DQ, D); }
    size_t v_off() const { return k_off() + qbytes(DK, D); }

    // ---- KV cache bands (elements): K rows of the first half of the kv heads, K second half, V first half, V second half
    size_t k_band_off(uint32_t band, uint32_t max_len) const { return (size_t)band * max_len * (DK / 2); }
    size_t v_band_off(uint32_t band, uint32_t max_len) const { return (size_t)max_len * DK + (size_t)band * max_len * (DK / 2); }
    /// element index of (kv head h, position j, element d) inside a K (is_v = false) or V cache
    size_t kv_index(bool is_v, uint32_t h, uint32_t j, uint32_t d, uint32_t max_len) const {
        const uint32_t per_band = num_kv_heads / 2;
        const uint32_t band = h / per_band;
        const size_t base = is_v ? v_band_off(band, max_len) : k_band_off(band, max_len);
        return base + (size_t)j * (per_band * DH) + (size_t)(h % per_band) * DH + d;
    }

    inline bool is_attn_layer(uint32_t i) const { return kinds[i] != lfm3_block_kind::conv; }
    inline uint32_t num_attn_layers() const {
        uint32_t n = 0;
        for (auto k : kinds) n += (k != lfm3_block_kind::conv);
        return n;
    }
    inline uint32_t num_conv_layers() const { return num_hidden_layers - num_attn_layers(); }
    inline size_t kv_cache_elems(uint32_t max_len) const { return (size_t)max_len * DK; }

    static std::vector<float> make_inv_freq(const nlohmann::json& rp, uint32_t DH) {
        const double base = rp.value("rope_theta", 10000.0);
        const double factor = rp.value("partial_rotary_factor", 1.0);
        const std::string type = rp.value("rope_type", std::string("default"));
        if (type != "default") throw std::runtime_error("lfm3: unsupported rope_type " + type);
        const int angles = (int)(factor * DH) / 2;
        std::vector<float> inv(DH / 2, 0.0f);
        for (int i = 0; i < angles; i++) {
            // computed like the reference: 1 / base^(2i / DH) in fp32
            inv[i] = (float)(1.0 / std::pow(base, (double)(2 * i) / (double)DH));
        }
        return inv;
    }

    void build(const LM_Config& config) {
        num_hidden_layers = config.get("num_hidden_layers");
        D = config.get("hidden_size");
        DH = config.get("head_dim");
        num_heads = config.get("num_attention_heads");
        num_kv_heads = config.get("num_key_value_heads");
        DQ = DH * num_heads;
        DK = DH * num_kv_heads;
        DV = DH * num_kv_heads;
        intermediate_size = config.get("intermediate_size");
        vocab_size = config.get("vocab_size");
        conv_kernel = config.get("conv_L_cache");
        sliding_window = config.get("sliding_window");
        eps = config.get<float>("norm_eps", 1e-5f);

        const auto& lt = config._json_config["layer_types"];
        if (lt.size() != num_hidden_layers) throw std::runtime_error("lfm3: layer_types length != num_hidden_layers");
        for (const auto& t : lt) {
            const std::string s = t.get<std::string>();
            if (s == "conv") kinds.push_back(lfm3_block_kind::conv);
            else if (s == "sliding_attention") kinds.push_back(lfm3_block_kind::sliding_attention);
            else if (s == "full_attention") kinds.push_back(lfm3_block_kind::full_attention);
            else throw std::runtime_error("lfm3: unknown layer type " + s);
        }
        const auto& rp = config._json_config["rope_parameters"];
        inv_freq_full = make_inv_freq(rp["full_attention"], DH);
        inv_freq_sliding = make_inv_freq(rp["sliding_attention"], DH);

        if (D % lfm3_cpu::QB_COLS || intermediate_size % lfm3_cpu::QB_COLS || DQ % lfm3_cpu::QB_ROWS)
            throw std::runtime_error("lfm3: dimensions are not multiples of the q4nx block");
        if (DH % 2) throw std::runtime_error("lfm3: head_dim must be even");

        DEBUG_BLOCK(1,
            std::cout << "================== lfm3_desc =================" << std::endl;
            std::cout << "layers " << num_hidden_layers << " (conv " << num_conv_layers() << ", attn " << num_attn_layers() << ")"
                      << " D " << D << " DH " << DH << " heads " << num_heads << "/" << num_kv_heads
                      << " I " << intermediate_size << " V " << vocab_size << " conv_k " << conv_kernel
                      << " window " << sliding_window << " eps " << eps << std::endl;
            std::cout << "==============================================" << std::endl;
        )
    }

    /// cos/sin of one position for one attention kind, rounded to bf16 like the reference (cos/sin are cast to the
    /// activation dtype before use)
    void rope_cos_sin(lfm3_block_kind kind, int pos, float* cosv, float* sinv) const {
        const std::vector<float>& inv = (kind == lfm3_block_kind::full_attention) ? inv_freq_full : inv_freq_sliding;
        for (uint32_t i = 0; i < DH / 2; i++) {
            const float a = (float)pos * inv[i];
            cosv[i] = lfm3_cpu::bf2f(lfm3_cpu::f2bf(std::cos(a)));
            sinv[i] = lfm3_cpu::bf2f(lfm3_cpu::f2bf(std::sin(a)));
        }
    }
};

/// \brief Copies a quantized weight into the device layout: the blocks of `vblocks` consecutive row blocks are interleaved
///        per column block (what the dequant kernels read). `src` holds the blocks in q4nx order.
inline void lfm3_reorder_cpy(uint8_t* dst, const bytes& src, int cols, int vblocks = 2) {
    using namespace lfm3_cpu;
    const int cbs = cols / QB_COLS;
    const size_t rbs = src.size() / QB_BYTES / cbs;
    if (rbs % vblocks) throw std::runtime_error("lfm3: row blocks not a multiple of the interleave");
    const uint8_t* s = reinterpret_cast<const uint8_t*>(src.data());
    size_t o = 0;
    for (size_t r = 0; r < rbs; r += vblocks)
        for (int c = 0; c < cbs; c++)
            for (int i = 0; i < vblocks; i++, o += QB_BYTES)
                std::memcpy(dst + o, s + ((r + i) * cbs + c) * QB_BYTES, QB_BYTES);
}

/// \brief Loads one quantized tensor from the q4nx file (checking its size) and writes it into `dst` in the device layout.
inline void lfm3_load_into(Q4NX& q4nx, uint8_t* dst, size_t dst_capacity, const std::string& name, int rows, int cols) {
    bytes tmp;
    q4nx.load_weights(tmp, name);
    const size_t want = lfm3_desc::qbytes(rows, cols);
    if (tmp.size() != want || want > dst_capacity)
        throw std::runtime_error("lfm3: tensor " + name + " has " + std::to_string(tmp.size()) + " bytes, expected " + std::to_string(want) +
                                 " for " + std::to_string(rows) + " x " + std::to_string(cols));
    lfm3_reorder_cpy(dst, tmp, cols);
}

/// \brief Loads the up and gate matrices as one band: slices of 512 output rows, up first and gate second, each slice interleaved.
inline void lfm3_load_up_gate(Q4NX& q4nx, const lfm3_desc& d, uint8_t* dst, const std::string& up_name, const std::string& gate_name) {
    bytes up, gate;
    q4nx.load_weights(up, up_name);
    q4nx.load_weights(gate, gate_name);
    const size_t want = lfm3_desc::qbytes(d.intermediate_size, d.D);
    if (up.size() != want || gate.size() != want) throw std::runtime_error("lfm3: up/gate size mismatch for " + up_name);
    const int slices = d.intermediate_size / lfm3_desc::UP_GATE_SLICE_ROWS;
    const size_t sb = d.up_gate_slice_bytes();
    for (int i = 0; i < slices; i++) {
        bytes us(sb), gs(sb);
        std::memcpy(us.data(), reinterpret_cast<const uint8_t*>(up.data()) + (size_t)i * sb, sb);
        std::memcpy(gs.data(), reinterpret_cast<const uint8_t*>(gate.data()) + (size_t)i * sb, sb);
        lfm3_reorder_cpy(dst + (size_t)(2 * i) * sb, us, d.D);
        lfm3_reorder_cpy(dst + (size_t)(2 * i + 1) * sb, gs, d.D);
    }
}

/// \brief view of a matrix at `off` bytes into a layer's proj buffer
inline lfm3_cpu::qmat lfm3_view(const buffer<u8>& proj, size_t off, int rows, int cols, bool is_signed = true) {
    lfm3_cpu::qmat m;
    m.base = reinterpret_cast<const uint8_t*>(proj.data()) + off;
    m.rows = rows;
    m.cols = cols;
    m.is_signed = is_signed;
    return m;
}

inline void lfm3_load_bf16(Q4NX& q4nx, buffer<bf16>& b, const std::string& name, size_t expected) {
    q4nx.load_weights(b, name);
    if (b.size() != expected) {
        throw std::runtime_error("lfm3: tensor " + name + " has " + std::to_string(b.size()) + " elements, expected " + std::to_string(expected));
    }
}

#endif
