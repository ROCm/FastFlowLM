// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file prefill/npu/delta_weights.hpp
/// \brief Device-side weight layout for one gated-delta-net layer, and the
///        staging that fills it.
#ifndef __QWEN3_8MTP_NPU_DELTA_WEIGHTS_HPP__
#define __QWEN3_8MTP_NPU_DELTA_WEIGHTS_HPP__

#include "common/cpu_ref_common.hpp"           // cpu_ref::safetensors
#include "common/cpu_ref_delta_net_layer.hpp"  // cpu_ref::delta_net_geometry
#include "verify/npu/q4nx.hpp"                 // q4k_bytes, reorder_cpy

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace q38npu {

// ==========================================================================
// geometry
// ==========================================================================
/// Byte offsets into one delta-net layer's projection BO.
struct delta_layout {
    // --- from the model ----------------------------------------------------
    int D = 0;
    int intermediate_size = 0;
    int conv_dim = 0;     ///< qkv_proj's row count: dq + dk + dv
    int DV = 0;           ///< z's row count and ssm_out's reduction width
    int DQ = 0, DK = 0;   ///< the qkv row's first two sub-tensors
    int v_heads_padded = 0;  ///< v_heads rounded up to the 32-row quant block

    // --- BYTE offsets, in qwen3_5vl_npu's linear order ----------------------
    size_t w_upgate_offset = 0;   ///< always 0; named for symmetry
    size_t w_down_offset   = 0;
    size_t w_qkv_offset    = 0;
    size_t w_z_offset      = 0;
    size_t w_out_offset    = 0;
    /// The verify bitstream's a/b tail. Past `proj_bytes` worth of q4_k, so a
    /// prefill dispatch never sees it -- see the header comment.
    size_t w_ab_bias_offset = 0;
    size_t w_ab_offset      = 0;
    size_t bo_bytes         = 0;  ///< what to allocate: proj_bytes + the tail
    size_t proj_bytes       = 0;  ///< the five q4_k projections alone

    /// One q8_0 32x256 block, which is what every BD on the a/b tile's weight
    /// channel transfers and therefore the exact size of the bias block.
    static constexpr size_t AB_W_BLOCK_BYTES = q80_bytes(32 * 256);

    // --- the same offsets in bf16 words -------------------------------------
    // `dequant_mm_op` takes bytes; a *sequence generator* takes words, because
    // it builds its descriptors with an element size of sizeof(bf16) and
    // npu_dma_memcpy_nd folds that to u32. These accessors are the single place
    // the conversion happens -- handing a byte offset straight to a descriptor
    // addresses twice as far into the BO and streams whatever is past the end,
    // which the cores dequantise into NaN rather than rejecting.
    size_t w_upgate_words()  const { return w_upgate_offset  / sizeof(uint16_t); }
    size_t w_down_words()    const { return w_down_offset    / sizeof(uint16_t); }
    size_t w_qkv_words()     const { return w_qkv_offset     / sizeof(uint16_t); }
    size_t w_z_words()       const { return w_z_offset       / sizeof(uint16_t); }
    size_t w_out_words()     const { return w_out_offset     / sizeof(uint16_t); }
    size_t w_ab_bias_words() const { return w_ab_bias_offset / sizeof(uint16_t); }
    size_t w_ab_words()      const { return w_ab_offset      / sizeof(uint16_t); }

    /// alpha|beta's fused output width: both padded to the quant block, f32.
    int AB_OUT() const { return 2 * v_heads_padded; }

    /// qkv|z's fused output width, and the one thing that depends on z
    /// immediately following qkv.
    int QKVZ_OUT() const { return conv_dim + DV; }

    static delta_layout build(const cpu_ref::delta_net_geometry& g) {
        delta_layout t;
        t.D                 = g.model_dim;
        t.intermediate_size = g.intermediate_size;
        t.conv_dim          = g.conv_dim();
        t.DV                = g.dv();
        t.DQ                = g.dq();
        t.DK                = g.dk();
        t.v_heads_padded    = (g.v_heads + 31) / 32 * 32;

        const size_t Dz = static_cast<size_t>(t.D);
        const size_t FF = static_cast<size_t>(t.intermediate_size);

        t.w_upgate_offset = 0;
        t.w_down_offset   = t.w_upgate_offset + q4k_bytes(2 * FF * Dz);
        t.w_qkv_offset    = t.w_down_offset   + q4k_bytes(Dz * FF);
        t.w_z_offset      = t.w_qkv_offset    + q4k_bytes((size_t)t.conv_dim * Dz);
        t.w_out_offset    = t.w_z_offset      + q4k_bytes((size_t)t.DV * Dz);
        t.proj_bytes      = t.w_out_offset    + q4k_bytes(Dz * (size_t)t.DV);

        // The verify bitstream's a/b tail, appended so every offset above is
        // the one the prefill dispatches already use.
        t.w_ab_bias_offset = t.proj_bytes;
        t.w_ab_offset      = t.w_ab_bias_offset + AB_W_BLOCK_BYTES;
        t.bo_bytes         = t.w_ab_offset +
                             2 * q80_bytes((size_t)t.v_heads_padded * Dz);

        // Every offset the DMA sees must be an even bf16 word count:
        // npu_dma_memcpy_nd folds bf16 into 32-bit words, halving the offset
        // and every stride, so an odd one silently truncates.
        if (t.w_down_offset % 4 || t.w_qkv_offset % 4 || t.w_z_offset % 4 ||
            t.w_out_offset % 4 || t.w_ab_bias_offset % 4 || t.w_ab_offset % 4)
            throw std::runtime_error("delta_layout: a projection offset is not "
                                     "an even bf16 word -- the DMA folds it to u32");

        // The weave grain, which both the staging below and delta_proj's
        // un-weaving assume.
        if (t.intermediate_size % 512 != 0)
            throw std::runtime_error("delta_layout: a projection round is 512 "
                                     "output rows; up|gate is woven at that grain");
        return t;
    }
};

// ==========================================================================
// staging
// ==========================================================================
inline void stage_delta_ab_weights(uint8_t* dst, const cpu_ref::safetensors& f,
                                   const delta_layout& t, int layer);

/// Stage one delta-net layer's five q4_k projections into its BO.
///
/// Reads straight out of the mmap'd .q4nx through `reorder_cpy`, so no heap
/// copy of the 211 MB is ever made -- the only cost is the permutation itself.
/// Tensor names match `cpu_ref::delta_net_weights::load`, so the reference and
/// the device layout cannot disagree about which tensor is which.
inline void stage_delta_proj_weights(uint8_t* dst, const cpu_ref::safetensors& f,
                                     const delta_layout& t, int layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    const std::string l = p + "linear_attn.";
    const size_t Dz = static_cast<size_t>(t.D);

    {
        // mlp.up_proj and mlp.gate_proj, **woven 512 output rows at a time
        // rather than concatenated**, exactly as qwen3_5vl_model_def.hpp:402-428
        // weaves them (its UP_GATE_OUTDIM_REORDEER is this 512). The GLU tile
        // takes one engine round at a time and needs a round to be all up or
        // all gate, so round 2p is up and round 2p+1 is gate, both covering
        // intermediate columns [p*512, (p+1)*512).
        //
        // 512 rows is an even number of the engine's 32-row blocks, so this is
        // byte for byte what one reorder_cpy over the woven matrix would
        // produce -- the weave and the row-block pairing do not interfere.
        const std::string un = p + "mlp.up_proj.weight";
        const std::string gn = p + "mlp.gate_proj.weight";
        const size_t blocks = 1024 / 2 / 32 * (Dz / 256);
        const size_t chunk  = blocks * W_BLOCK_BYTES;      // 512 rows of q4_k
        const size_t total  = q4k_bytes((size_t)t.intermediate_size * Dz);
        const size_t phases = total / chunk;
        if (phases * chunk != total)
            throw std::runtime_error("up|gate does not divide into whole phases");
        if (f.nbytes(un) != total || f.nbytes(gn) != total)
            throw std::runtime_error("up/gate are not [intermediate_size][D] q4_k");

        const uint8_t* up   = f.raw(un);
        const uint8_t* gate = f.raw(gn);
        uint8_t* w = dst + t.w_upgate_offset;
        for (size_t i = 0; i < phases; i++) {
            reorder_cpy(w, up   + i * chunk, chunk, t.D); w += chunk;
            reorder_cpy(w, gate + i * chunk, chunk, t.D); w += chunk;
        }
    }
    {
        // mlp.down_proj, D x intermediate_size. **col is intermediate_size**:
        // it reduces over the MLP hidden, not over D.
        const std::string dn = p + "mlp.down_proj.weight";
        if (f.nbytes(dn) != q4k_bytes(Dz * (size_t)t.intermediate_size))
            throw std::runtime_error(dn + " is not [D][intermediate_size] q4_k");
        reorder_cpy(dst + t.w_down_offset, f.raw(dn), f.nbytes(dn),
                    t.intermediate_size);
    }
    {
        // linear_attn.qkv_proj, one [conv_dim][D] tensor in the checkpoint --
        // unlike the attention layer, where q and the gate share q_proj and k
        // and v are separate tensors. So this is a single reorder and the
        // q|k|v split is purely a column offset on readback.
        const std::string qn = l + "qkv_proj.weight";
        if (f.nbytes(qn) != q4k_bytes((size_t)t.conv_dim * Dz))
            throw std::runtime_error(qn + " is not [conv_dim][D] q4_k");
        reorder_cpy(dst + t.w_qkv_offset, f.raw(qn), f.nbytes(qn), t.D);
    }
    {
        // The output gate z. Named self_attn.gate_proj even in a linear layer,
        // which is the checkpoint's naming and not a mistake here;
        // qwen3_5vl_model_def.hpp:241 calls the same tensor attn_gate.
        const std::string zn = p + "self_attn.gate_proj.weight";
        if (f.nbytes(zn) != q4k_bytes((size_t)t.DV * Dz))
            throw std::runtime_error(zn + " is not [dv][D] q4_k");
        if (t.w_z_offset != t.w_qkv_offset + q4k_bytes((size_t)t.conv_dim * Dz))
            throw std::runtime_error("z does not immediately follow qkv");
        reorder_cpy(dst + t.w_z_offset, f.raw(zn), f.nbytes(zn), t.D);
    }
    {
        // linear_attn.ssm_out_proj, D x dv. **col is dv**: like down_proj it
        // reduces over something other than D -- here the gated delta row.
        const std::string on = l + "ssm_out_proj.weight";
        if (f.nbytes(on) != q4k_bytes(Dz * (size_t)t.DV))
            throw std::runtime_error(on + " is not [D][dv] q4_k");
        reorder_cpy(dst + t.w_out_offset, f.raw(on), f.nbytes(on), t.DV);
    }
    stage_delta_ab_weights(dst, f, t, layer);
}

/// Stage one layer's a/b tail: the activation bias block, then alpha|beta q8_0.
///
/// Split out of the above because only the verify bitstream reads it; the
/// prefill phase pays the bytes but never dispatches against them.
inline void stage_delta_ab_weights(uint8_t* dst, const cpu_ref::safetensors& f,
                                   const delta_layout& t, int layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    const std::string l = p + "linear_attn.";
    const size_t Dz = static_cast<size_t>(t.D);
    const size_t PH = static_cast<size_t>(t.v_heads_padded);

    {
        // The a/b activation bias, laid out as the a/b kernel's activate_a()
        // reads it: A in f32 lanes [0, v_heads_padded), ssm_dt.bias in
        // [v_heads_padded, 2 * v_heads_padded), the rest zero.
        //
        // ssm_a and ssm_dt.bias only have v_heads entries, so the pad lanes
        // stay zero and their alpha comes out exp(0 * softplus(0)) = 1 rather
        // than a garbage exponential. Zero the whole block first: past the
        // 2 * padded floats nothing is read, but it crosses DDR either way.
        const std::vector<float> a  = f.f32(l + "ssm_a");
        const std::vector<float> dt = f.f32(l + "ssm_dt.bias");
        const size_t v_heads = a.size();
        if (v_heads > PH || dt.size() < v_heads)
            throw std::runtime_error("ssm_a / ssm_dt.bias do not match v_heads");

        uint8_t* blk = dst + t.w_ab_bias_offset;
        std::memset(blk, 0, delta_layout::AB_W_BLOCK_BYTES);
        float* bias = reinterpret_cast<float*>(blk);
        std::memcpy(bias,      a.data(),  v_heads * sizeof(float));
        std::memcpy(bias + PH, dt.data(), v_heads * sizeof(float));
    }
    {
        // alpha and beta, q8_0, [v_heads_padded][D] each. A plain memcpy and
        // **not** reorder_cpy: the a/b kernel walks its blocks row-block major
        // (all of one row block's column blocks, then the next), which is
        // already the .q4nx file order. The two-block interleave reorder_cpy
        // produces comes from the *projection* memtile handing half a fill to
        // each of two row blocks, and the a/b tile has no such split.
        const std::string an = l + "ssm_alpha_proj.weight";
        const std::string bn = l + "ssm_beta_proj.weight";
        const size_t half = q80_bytes(PH * Dz);
        if (f.nbytes(an) != half || f.nbytes(bn) != half)
            throw std::runtime_error("alpha/beta are not [v_heads_padded][D] q8_0");
        std::memcpy(dst + t.w_ab_offset,        f.raw(an), half);
        std::memcpy(dst + t.w_ab_offset + half, f.raw(bn), half);
    }
}

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_DELTA_WEIGHTS_HPP__
