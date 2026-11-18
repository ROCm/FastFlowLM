// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hy_mem_prefill.hpp
/// \brief Batched prefill of Hy-Memory-30B-A3B: a chunk of prompt tokens through all
///        48 layers at once, writing the decode KV cache as it goes
/// \note Per layer (models_simple/simple/hy_mem_simple.py has the math):
///         x -> rms_norm -> dequant_mm(k|v|q) -> qk_norm_rope -> store_kv
///           -> attention -> dequant_mm(o) -> +x
///           -> rms_norm -> dense:  dequant_mm(up|gate) -> silu_glu -> dequant_mm(down) -> +x
///                          MoE:    router_logits -> route
///                                  shared expert as dense (768 wide)
///                                  per used expert: its tokens only, same three steps,
///                                  times the routing weight, summed in fp32 -> +x
///       The heavy steps are dequant_mm and attention (hy_mem_prefill_ops.hpp); the
///       switches below say where each runs. The NPU versions run on the prefill
///       overlay (hy_mem_prefill_overlay.hpp), reading the same decode weight
///       buffers and KV cache.
///
///       The other prefill, PREFILL_WITH_MV (Makefile PREFILL_WITH_MV=1), feeds the
///       prompt through the decode overlays one token at a time instead.
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "hy_mem_model_def.hpp"
#include "hy_mem_prefill_ops.hpp"
#include "hy_mem_cpu_functions.hpp"
#include "hy_mem_prefill_overlay.hpp"

// Where the two heavy prefill stages run (1 = the prefill overlay, 0 = the host).
// A chunk is zero-padded to a whole number of rounds: 128 rows for the attention,
// 256 for dequant_mm. At run time HY_MEM_ATTN_HOST=1 / HY_MEM_MM_HOST=1 put either
// back on the host (for comparisons), and HY_MEM_MM_NPU_MIN (1024) is the shortest
// chunk whose routed experts are worth the overlay rather than the host.
#ifndef HY_MEM_MM_ON_NPU
#define HY_MEM_MM_ON_NPU 1
#endif
#ifndef HY_MEM_ATTN_ON_NPU
#define HY_MEM_ATTN_ON_NPU 1
#endif

/// \brief the decode buffers of one layer, which the prefill reads (weights) and
///        writes (KV cache) in place
struct hy_mem_layer_buffers {
    buffer<u8>* proj;
    buffer<bf16>* rms;
    buffer<bf16>* rope_rms;
    buffer<bf16>* kv_cache;
    const std::vector<float>* router_bias;   // MoE layers only
    /// the projections in the prefill overlay's block order: `proj` itself for a
    /// MoE layer (already relaid for the int8-mmul decode kernel), a private
    /// relaid copy for a dense one (hy_mem_desc::make_prefill_proj)
    buffer<u8>* proj_prefill;
};

class hy_mem_prefill_context {
public:
    explicit hy_mem_prefill_context(hy_mem_desc& desc) : desc(desc) {}

    /// \brief the prefill overlay the NPU stages run on (nullptr: all on the host)
    void set_overlay(hy_mem_prefill_overlay* ovl) { overlay = ovl; }

    /// \brief L tokens at positions pos0 .. pos0 + L - 1 through every layer
    /// \param x in: the embeddings [L][D]; out: the last layer's output [L][D]
    /// \param layers the decode buffers, one entry per layer
    /// \param MAX_L the KV cache length the caches were allocated with
    void forward(bf16* x, int L, int pos0, std::vector<hy_mem_layer_buffers>& layers, uint32_t MAX_L);

    // ---- where each weight is in a layer's proj buffer (see hy_mem_model_def.cpp) ----
    hy_mem_cpu::q4k_matrix kvq(int layer, const buffer<u8>& proj) const;
    hy_mem_cpu::q4k_matrix o(int layer, const buffer<u8>& proj) const;
    hy_mem_cpu::q4k_matrix up_gate(int layer, const buffer<u8>& proj) const;   // dense or shared expert
    hy_mem_cpu::q4k_matrix down(int layer, const buffer<u8>& proj) const;      // dense or shared expert
    hy_mem_cpu::q4k_matrix expert_up_gate(int expert, const buffer<u8>& proj) const;
    hy_mem_cpu::q4k_matrix expert_down(int expert, const buffer<u8>& proj) const;

    /// \brief HY_CHECK_LAYOUT=1: every block the views above point at, compared byte
    ///        for byte (relayout undone) with the tensors in model.q4nx
    /// \return the number of blocks that differ (0 = the views are right)
    size_t check_layout(int layer, Q4NX& q4nx, const buffer<u8>& proj) const;

private:
    hy_mem_desc& desc;
    hy_mem_prefill_overlay* overlay = nullptr;
    bool attn_on_npu = false;   // this forward()'s attention runs on the overlay
    bool mm_on_npu = false;     // ... and so do its dequant_mm calls
    bool experts_on_npu = false;// ... including the routed experts (long chunks only)

    /// \brief y[L][w.N] = x[L][w.K] W^T, on the overlay when it can take it
    /// \note The overlay reads the weights from `buf.proj_prefill` at the same byte
    ///       offset the host view has in `buf.proj`: the two buffers differ only
    ///       inside each Q4_K block, never in where the blocks are.
    void _mm(bf16* y, const hy_mem_cpu::q4k_matrix& w, const bf16* x, int L, hy_mem_layer_buffers& buf);

    /// \brief where a weight view starts in the layer's projection buffer
    size_t _w_offset(const hy_mem_cpu::q4k_matrix& w, const hy_mem_layer_buffers& buf) const;


    // scratch, grown to the largest chunk seen
    int capacity = 0;
    std::vector<bf16> xn, kvq_out, attn_out, proj_out, fused, hid;
    std::vector<float> logits, moe_sum;
    void _reserve(int L);

    void _attention_block(int layer, bf16* x, int L, int pos0, hy_mem_layer_buffers& buf, uint32_t MAX_L);
    void _dense_mlp(int layer, bf16* x, int L, hy_mem_layer_buffers& buf);
    void _moe(int layer, bf16* x, int L, hy_mem_layer_buffers& buf);

    // HY_TIME_HOST=1: time per stage, summed over a forward() and printed at its end
    enum stage_t { S_NORM, S_MM_ATTN, S_ROPE_KV, S_ATTN, S_MM_MLP, S_ROUTER, S_MM_EXPERT, S_GLU_ADD, S_SYNC, S_COUNT };
    double stage_us[S_COUNT] = {};
};
