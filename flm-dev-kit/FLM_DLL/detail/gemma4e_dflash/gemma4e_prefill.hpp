// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "flm_override.hpp"
#ifndef __GEMMA4E_PREFILL_HPP__
#define __GEMMA4E_PREFILL_HPP__
#include <filesystem>
#include <memory>
#include "modules/gemm.hpp"
#include "tensor_2d.hpp"
#include "gemma4e_npu_def.hpp"
#include "gemma4e_dflash_sequence.hpp"
#include "gemma4e_cpu_functions.hpp"
#include "gemma4e_image.hpp"
#include "mmRuntimeSequence.hpp"
#include "utils/error_measure.hpp"
#include "gemma4e_bfp_gemm.hpp"
#include "gemma4e_prof.hpp"

/// @brief Row geometry of one prefill call.
///
/// The attention kernel consumes whole chunks, so the batch is padded in front
/// (L_offset rows that belong to already-cached tokens) and at the back (up to
/// L_padded rows), and every gemm runs over the full L_padded rows. The per
/// layer input path runs over a coarser 512-row grid, hence L_padded_512.
struct gemma4e_prefill_shape {
    int L_in = 0;             ///< tokens handed to this call
    int L_begin = 0;          ///< context length before this call
    int L_end = 0;            ///< context length after this call
    int L_effective = 0;      ///< rows that carry a real token, == L_in
    int L_offset = 0;         ///< rows of chunk padding in front of the first new token
    int L_begin_chunked = 0;  ///< chunk-aligned start fed to the attention kernel
    int L_end_chunked = 0;    ///< chunk-aligned end fed to the attention kernel
    int L_padded = 0;         ///< rows every gemm runs over
    int L_padded_512 = 0;     ///< L_padded rounded up for the per-layer-input gemms
    int sliding_l_begin = 0;  ///< first position the sliding window still covers
    // linear sliding-window decode buffer (gemma4e_dflash_swa.hpp): row of position p is p - base
    int swa_base_old = 0;     ///< base before this call (where the rows already in the decode buffer are)
    int swa_base_new = 0;     ///< base after this call (>= old; the window rows are moved when it grows)
    int swa_zero_hi = 0;      ///< rows [L_end - base_new, swa_zero_hi) of the decode buffer hold stale data and are zeroed
};

/// @brief Buffers shared by every block of a prefill layer.
///
/// Allocated once per batch geometry and reused across prefill calls; only the
/// residual and the per-layer-input embeddings need clearing, since every other
/// buffer is fully overwritten by the kernel that reads it.
struct gemma4e_common_buffers {
    buffer<bf16> residual_buffer;      ///< host: the running residual stream
    buffer<bf16> hidden_state_buffer;  ///< device: block input and block output
    buffer<bf16> q_buffer;
    buffer<bf16> k_buffer;
    buffer<bf16> v_buffer;
    buffer<bf16> attn_out_buffer;
    buffer<bf16> gate_buffer;
    buffer<bf16> up_buffer;
    buffer<bf16> hid_buffer;

    // per layer input path
    buffer<bf16> pli_embed_buffer;
    buffer<bf16> pli_down_buffer;
    buffer<bf16> pli_gate_buffer;
    buffer<bf16> pli_hid_buffer;

    tensor_2d<bf16> tensor_residual;
    tensor_2d<bf16> tensor_pli_embed;

    /// rows (of the 512-row grid) the buffers were allocated for; they only grow, and are rounded up so that prompts of nearby
    /// lengths share one allocation instead of reallocating about half a gigabyte each time the length class changes
    int allocated_rows = 0;
    static constexpr int ALLOC_GRANULE_ROWS = 1024;

    gemma4e_common_buffers() {}

    /// @brief (Re)allocates every buffer for a batch geometry.
    void allocate(npu_xclbin_manager* npu, gemma4e_desc* desc, const gemma4e_prefill_shape& s) {
        const size_t D = desc->D;
        const size_t I = desc->INTERMEDIATE_SIZE;
        const size_t PLI = (size_t)desc->PLI_D * desc->num_hidden_layers;
        residual_buffer     = buffer<bf16>((size_t)s.L_padded_512 * D);
        hidden_state_buffer = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * D);
        // the projections write whole passes of 512 rows (the block-float kernel's granularity), so these span L_padded_512 rows
        q_buffer            = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * desc->DQ);
        k_buffer            = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * desc->DK);
        v_buffer            = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * desc->DV);
        attn_out_buffer     = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * desc->DQ);
        // the widest mlp a layer can ask for, so skip layers share these buffers
        const size_t I_widest = desc->enable_double_wide_mlp ? I * 2 : I;
        gate_buffer         = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * I_widest);
        up_buffer           = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * I_widest);
        hid_buffer          = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * I_widest);
        pli_embed_buffer    = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * PLI);
        pli_down_buffer     = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * PLI);
        pli_gate_buffer     = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * desc->PLI_D);
        pli_hid_buffer      = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * desc->PLI_D);
    }

    /// @brief Clears the padding rows and re-points the row views at this batch.
    void reset(gemma4e_desc* desc, const gemma4e_prefill_shape& s) {
        // rows outside [L_offset, L_offset + L_in) are padding; they still take
        // part in every gemm, so leave no stale values in them.
        // (only this batch's rows: the buffers are kept at the largest size seen, see allocated_rows)
        memset(residual_buffer.data(), 0, (size_t)s.L_padded_512 * desc->D * sizeof(bf16));
        memset(pli_embed_buffer.data(), 0, (size_t)s.L_padded_512 * desc->PLI_D * desc->num_hidden_layers * sizeof(bf16));
        tensor_residual.assign(residual_buffer, desc->D, s.L_offset);
        tensor_pli_embed.assign(pli_embed_buffer, desc->PLI_D * desc->num_hidden_layers, s.L_offset);
    }
};

/// @brief Dequantizes one layer's projections into the bf16 buffers the prefill
///        gemms consume.
///
/// One sequence per (layer kind, weight): which bytes each one reads is entirely
/// the descriptor's business, so changing the quantization type needs no edit
/// here. The five matrices of a layer are launched back to back.
struct gemma4e_dequant_prefill_context {
    /// index into the per-layer-kind app table
    enum matrix_t { QKV = 0, O = 1, GATE = 2, UP = 3, DOWN = 4, NUM_MATRICES = 5 };

    gemma4e_desc* desc;
    npu_app apps[4][NUM_MATRICES];  ///< [gemma4e_layer_type_t][matrix_t]

    buffer<bf16> qkv_weights;
    buffer<bf16> o_weights;
    buffer<bf16> gate_weights;
    buffer<bf16> up_weights;
    buffer<bf16> down_weights;

    gemma4e_dequant_prefill_context(
        gemma4e_desc* desc,
        gemma4e_dflash_sequence* seq_gen,
        npu_app_manager* dequant_app_manager
    ) : desc(desc)
    {
        const uint32_t D = desc->D;
        const uint32_t I = desc->get_max_intermediate_size();
        this->qkv_weights  = dequant_app_manager->create_bo_buffer<bf16>((size_t)D * (desc->DQ + desc->DK + desc->DV));
        this->o_weights    = dequant_app_manager->create_bo_buffer<bf16>((size_t)desc->DQ * D);
        this->gate_weights = dequant_app_manager->create_bo_buffer<bf16>((size_t)D * I);
        this->up_weights   = dequant_app_manager->create_bo_buffer<bf16>((size_t)D * I);
        this->down_weights = dequant_app_manager->create_bo_buffer<bf16>((size_t)I * D);

        for (int t = 0; t < 4; t++) {
            gemma4e_layer_type_t type = static_cast<gemma4e_layer_type_t>(t);
            gemma4e_layer_weight_def& W = desc->weight_desc(type);
            for (int m = 0; m < NUM_MATRICES; m++) {
                apps[t][m] = dequant_app_manager->create_app();
            }
            // up and gate share one interleaved band and are told apart by the mode
            seq_gen->generate_dequant_seq(apps[t][QKV].seq(),  W.attn_qkv,    gemma4e_dflash_sequence::NORMAL_DEQUANT);
            seq_gen->generate_dequant_seq(apps[t][O].seq(),    W.attn_output, gemma4e_dflash_sequence::NORMAL_DEQUANT);
            seq_gen->generate_dequant_seq(apps[t][GATE].seq(), W.ffn_gate,    gemma4e_dflash_sequence::GATE_MATRIX);
            seq_gen->generate_dequant_seq(apps[t][UP].seq(),   W.ffn_up,      gemma4e_dflash_sequence::UP_MATRIX);
            seq_gen->generate_dequant_seq(apps[t][DOWN].seq(), W.ffn_down,    gemma4e_dflash_sequence::NORMAL_DEQUANT);
        }
    }

    /// @brief Dequantizes every projection of one layer, in place.
    void run(gemma4e_layer_type_t type, buffer<u8>& proj_weights) {
        npu_app* a = apps[int(type)];
        FLM_OVERRIDE(dequant_qkv,  a[QKV](this->qkv_weights, proj_weights),   this->desc, type);
        FLM_OVERRIDE(dequant_o,    a[O](this->o_weights, proj_weights),       this->desc, type);
        FLM_OVERRIDE(dequant_up,   a[UP](this->up_weights, proj_weights),     this->desc, type);
        FLM_OVERRIDE(dequant_gate, a[GATE](this->gate_weights, proj_weights), this->desc, type);
        FLM_OVERRIDE(dequant_down, a[DOWN](this->down_weights, proj_weights),  this->desc, type);
    }
};

/// @brief Attention half of one prefill layer: q/k/v projections, rope, kv cache
///        fill, attention, output projection.
///
/// Sliding and global layers run different kernels over differently shaped
/// caches, so both sets of apps live here and forward() dispatches on the layer
/// kind. Skip layers reuse the previous layer's cache and project q only.
struct gemma4e_attn_block_prefill_context {
    gemma4e_desc* desc;
    Gemm* gemm_seq_gen;
    gemma4e_dflash_sequence* seq_gen;
    gemma4e_common_buffers* bufs;

    int L_padded_old = -1;
    int L_begin_chunked_old = -1;
    int L_end_chunked_old = -1;
    uint32_t MAX_L = 0;

    npu_app q_swa_proj, k_swa_proj, v_swa_proj, o_swa_proj;
    npu_app q_global_proj, k_global_proj, v_global_proj, o_global_proj;
    npu_app mha_engine;
    npu_app swa_engine;

    /// caches the attention kernels read: contiguous, and rebuilt every layer
    buffer<bf16> kv_cache_sliding_prefill;
    /// @brief rows of the two prefill scratch caches that an earlier call wrote (see zero_scratch)
    size_t global_scratch_rows = 0;
    size_t sliding_scratch_rows = 0;
    /// @brief The scratch caches are rebuilt from the live caches on every call, but what an earlier call left in the rows
    ///        outside what it writes is not harmless to the kernels: start every prefill from zeros, so a prefill continued
    ///        live and one that follows a restored state see the same scratch.
    void zero_scratch() {
        if (global_scratch_rows > 0) {
            memset(kv_cache_global_prefill.data(), 0, global_scratch_rows * desc->DK * sizeof(bf16));
            memset(kv_cache_global_prefill.data() + (size_t)MAX_L * desc->DK, 0, global_scratch_rows * desc->DV * sizeof(bf16));
            kv_cache_global_prefill.sync_to_device();
            global_scratch_rows = 0;
        }
        if (sliding_scratch_rows > 0) {
            memset(kv_cache_sliding_prefill.data(), 0, sliding_scratch_rows * desc->SWA_DK * sizeof(bf16));
            memset(kv_cache_sliding_prefill.data() + (size_t)MAX_L * desc->SWA_DK, 0, sliding_scratch_rows * desc->SWA_DV * sizeof(bf16));
            kv_cache_sliding_prefill.sync_to_device();
            sliding_scratch_rows = 0;
        }
    }
    buffer<bf16> kv_cache_global_prefill;

    npu_app_manager* mha_app_manager;
    npu_app_manager* swa_app_manager;

    gemma4e_attn_block_prefill_context(
        gemma4e_desc* desc,
        Gemm* gemm_seq_gen,
        gemma4e_dflash_sequence* seq_gen,
        gemma4e_common_buffers* bufs,
        npu_app_manager* gemm_app_manager,
        npu_app_manager* mha_app_manager,
        npu_app_manager* swa_app_manager
    ) : desc(desc), gemm_seq_gen(gemm_seq_gen), seq_gen(seq_gen), bufs(bufs),
        mha_app_manager(mha_app_manager), swa_app_manager(swa_app_manager)
    {
        this->q_global_proj = gemm_app_manager->create_app();
        this->k_global_proj = gemm_app_manager->create_app();
        this->v_global_proj = gemm_app_manager->create_app();
        this->o_global_proj = gemm_app_manager->create_app();
        this->q_swa_proj = gemm_app_manager->create_app();
        this->k_swa_proj = gemm_app_manager->create_app();
        this->v_swa_proj = gemm_app_manager->create_app();
        this->o_swa_proj = gemm_app_manager->create_app();
        this->mha_engine = mha_app_manager->create_app();
        this->swa_engine = swa_app_manager->create_app();
    }

    /// @brief Reads and writes the prefill-side caches of another attention block instead of owning a pair.
    ///
    /// The tail pass runs its own sequences (a different row geometry) over the very caches the full pass has just
    /// filled, so it has to see the same two buffers; copies of a buffer share the allocation.
    void borrow_caches_from(const gemma4e_attn_block_prefill_context& owner) {
        this->MAX_L = owner.MAX_L;
        this->kv_cache_sliding_prefill = owner.kv_cache_sliding_prefill;
        this->kv_cache_global_prefill  = owner.kv_cache_global_prefill;
        // the cached sequences address these buffers, so force a regeneration
        this->L_padded_old = -1;
        this->L_begin_chunked_old = -1;
        this->L_end_chunked_old = -1;
    }

    /// @brief Sizes the prefill-side caches, which span the whole context.
    void set_max_length(uint32_t MAX_L) {
        this->MAX_L = MAX_L;
        this->kv_cache_sliding_prefill = swa_app_manager->create_bo_buffer<bf16>((size_t)MAX_L * (desc->SWA_DK + desc->SWA_DV));
        this->kv_cache_global_prefill  = mha_app_manager->create_bo_buffer<bf16>((size_t)MAX_L * (desc->DK + desc->DV));
        this->kv_cache_sliding_prefill.memset((bf16)0);
        this->kv_cache_global_prefill.memset((bf16)0);
        this->kv_cache_sliding_prefill.sync_to_device();
        this->kv_cache_global_prefill.sync_to_device();
        this->global_scratch_rows = 0; // zeroed just now
        this->sliding_scratch_rows = 0;
        // the cached sequences address these buffers, so force a regeneration
        this->L_padded_old = -1;
        this->L_begin_chunked_old = -1;
        this->L_end_chunked_old = -1;
    }

    /// @brief Regenerates the projection and attention sequences, if the geometry moved.
    /// @param with_projections  false when the projections run on the block-float kernel instead: the bf16 gemm sequences are then
    ///                          not needed for this geometry
    void setup(const gemma4e_prefill_shape& s, bool with_projections = true) {
        const uint32_t D = desc->D;
        if (with_projections && L_padded_old != s.L_padded) {
            gemm_seq_gen->generate_seq(this->q_swa_proj.seq(), s.L_padded, D, desc->SWA_DQ, 0,
                false, Gemm::NO_Activation, 0);
            gemm_seq_gen->generate_seq(this->k_swa_proj.seq(), s.L_padded, D, desc->SWA_DK, D * desc->SWA_DQ,
                false, Gemm::NO_Activation, 0);
            gemm_seq_gen->generate_seq(this->v_swa_proj.seq(), s.L_padded, D, desc->SWA_DV, D * (desc->SWA_DQ + desc->SWA_DK),
                false, Gemm::NO_Activation, 0);
            gemm_seq_gen->generate_seq(this->o_swa_proj.seq(), s.L_padded, desc->SWA_DQ, D, 0,
                false, Gemm::NO_Activation, 0);

            gemm_seq_gen->generate_seq(this->q_global_proj.seq(), s.L_padded, D, desc->DQ, 0,
                false, Gemm::NO_Activation, 0);
            gemm_seq_gen->generate_seq(this->k_global_proj.seq(), s.L_padded, D, desc->DK, D * desc->DQ,
                false, Gemm::NO_Activation, 0);
            gemm_seq_gen->generate_seq(this->v_global_proj.seq(), s.L_padded, D, desc->DV, D * (desc->DQ + desc->DK),
                false, Gemm::NO_Activation, 0);
            gemm_seq_gen->generate_seq(this->o_global_proj.seq(), s.L_padded, desc->DQ, D, 0,
                false, Gemm::NO_Activation, 0);
            L_padded_old = s.L_padded;
        }
        if (L_begin_chunked_old != s.L_begin_chunked || L_end_chunked_old != s.L_end_chunked) {
            seq_gen->gen_mha_engine_seq(this->mha_engine.seq(), s.L_begin_chunked, s.L_end_chunked);
            seq_gen->gen_swa_engine_seq(this->swa_engine.seq(), s.L_begin_chunked, s.L_end_chunked);
            L_begin_chunked_old = s.L_begin_chunked;
            L_end_chunked_old = s.L_end_chunked;
        }
    }

    /// @brief Runs the attention block of one layer, hidden_state_buffer in and out.
    /// @param bfp           when set, q/k/v/o run on the block-float kernel straight from `proj_weights` (the dequantized
    ///                      qkv_weights / o_weights are then not used); when null they run on the bf16 gemms
    /// @param proj_weights  the layer's packed projection weights, read by the block-float kernel
    void forward(
        const gemma4e_prefill_shape& s,
        gemma4e_layer_type_t type,
        buffer<bf16>& qkv_weights,
        buffer<bf16>& o_weights,
        buffer<bf16>& kv_cache,
        buffer<bf16>& q_norm,
        buffer<bf16>& k_norm,
        SafeTensors* reference,
        gemma4e_bfp::gemma4e_bfp_block* bfp = nullptr,
        buffer<u8>* proj_weights = nullptr
    ) {
        if (is_swa_layer(type)) {
            _forward_swa(s, type, qkv_weights, o_weights, kv_cache, q_norm, k_norm, reference, bfp, proj_weights);
        }
        else {
            _forward_global(s, type, qkv_weights, o_weights, kv_cache, q_norm, k_norm, reference, bfp, proj_weights);
        }
    }

    /// @brief Rebuilds the global prefill cache: everything up to L_begin from the
    ///        decode cache, then this batch's new rows, written to both caches.
    void sync_kv_cache(
        buffer<bf16>& buffer_k,
        buffer<bf16>& buffer_v,
        buffer<bf16>& prefill_cache,
        buffer<bf16>& decoding_cache,
        int L_offset,
        int L_begin,
        int L_effective,
        int DK, int DV
    );

    /// @brief Rebuilds the sliding prefill cache, unrolling the decode ring buffer
    ///        into the linear order the swa kernel expects.
    void sync_sliding_kv_cache(
        buffer<bf16>& buffer_k,
        buffer<bf16>& buffer_v,
        buffer<bf16>& prefill_cache,
        buffer<bf16>& decoding_cache,
        int L_offset,
        int L_begin,
        int L_effective,
        int sliding_l_begin,
        int L_end_chunked,
        int DK, int DV,
        int swa_base_old, int swa_base_new, int swa_zero_hi
    );

private:
    void _forward_swa(const gemma4e_prefill_shape& s, gemma4e_layer_type_t type,
                      buffer<bf16>& qkv_weights, buffer<bf16>& o_weights, buffer<bf16>& kv_cache,
                      buffer<bf16>& q_norm, buffer<bf16>& k_norm, SafeTensors* reference,
                      gemma4e_bfp::gemma4e_bfp_block* bfp, buffer<u8>* proj_weights);
    void _forward_global(const gemma4e_prefill_shape& s, gemma4e_layer_type_t type,
                         buffer<bf16>& qkv_weights, buffer<bf16>& o_weights, buffer<bf16>& kv_cache,
                         buffer<bf16>& q_norm, buffer<bf16>& k_norm, SafeTensors* reference,
                         gemma4e_bfp::gemma4e_bfp_block* bfp, buffer<u8>* proj_weights);
};

/// @brief Feed-forward half of one prefill layer.
///
/// Skip layers run a double-wide mlp when the model enables it, which is a
/// different sequence over the same buffers, so both widths are kept ready.
struct gemma4e_mlp_prefill_context {
    gemma4e_desc* desc;
    Gemm* gemm_seq_gen;
    gemma4e_common_buffers* bufs;

    int L_padded_old = -1;

    npu_app gate_proj, up_proj, down_proj;
    npu_app gate_skip_proj, up_skip_proj, down_skip_proj;

    gemma4e_mlp_prefill_context(
        gemma4e_desc* desc,
        Gemm* gemm_seq_gen,
        gemma4e_common_buffers* bufs,
        npu_app_manager* gemm_app_manager
    ) : desc(desc), gemm_seq_gen(gemm_seq_gen), bufs(bufs)
    {
        this->gate_proj = gemm_app_manager->create_app();
        this->up_proj = gemm_app_manager->create_app();
        this->down_proj = gemm_app_manager->create_app();
        this->gate_skip_proj = gemm_app_manager->create_app();
        this->up_skip_proj = gemm_app_manager->create_app();
        this->down_skip_proj = gemm_app_manager->create_app();
    }

    void setup(const gemma4e_prefill_shape& s) {
        if (L_padded_old == s.L_padded) {
            return;
        }
        const uint32_t D = desc->D;
        const uint32_t I = desc->INTERMEDIATE_SIZE;
        gemm_seq_gen->generate_seq(this->gate_proj.seq(), s.L_padded, D, I, 0,
            false, Gemm::GeLU, 0);
        gemm_seq_gen->generate_seq(this->up_proj.seq(), s.L_padded, D, I, 0,
            false, Gemm::NO_Activation, 0);
        gemm_seq_gen->generate_seq(this->down_proj.seq(), s.L_padded, I, D, 0,
            false, Gemm::NO_Activation, 0);

        gemm_seq_gen->generate_seq(this->gate_skip_proj.seq(), s.L_padded, D, I * 2, 0,
            false, Gemm::GeLU, 0);
        gemm_seq_gen->generate_seq(this->up_skip_proj.seq(), s.L_padded, D, I * 2, 0,
            false, Gemm::NO_Activation, 0);
        gemm_seq_gen->generate_seq(this->down_skip_proj.seq(), s.L_padded, I * 2, D, 0,
            false, Gemm::NO_Activation, 0);
        L_padded_old = s.L_padded;
    }

    /// @brief Runs the mlp of one layer, hidden_state_buffer in and out.
    /// @param bfp           when set, gate/up/down run on the block-float kernel straight from `proj_weights`
    void forward(
        const gemma4e_prefill_shape& s,
        bool double_wide,
        buffer<bf16>& gate_weights,
        buffer<bf16>& up_weights,
        buffer<bf16>& down_weights,
        SafeTensors* reference,
        gemma4e_layer_type_t type = e_gemma4e_swa_layer,
        gemma4e_bfp::gemma4e_bfp_block* bfp = nullptr,
        buffer<u8>* proj_weights = nullptr
    );
};

/// @brief The per-layer-input path: a model-wide down projection run once per
///        batch, and a gate/up pair run at the end of every layer.
struct gemma4e_pli_prefill_context {
    gemma4e_desc* desc;
    gemma4e_common_buffers* bufs;
    Gemma4e_ImageEncoder* image_encoder;  ///< owns the mm tiling the pli gemms reuse

    int L_padded_512_old = -1;

    npu_app pli_down_proj, pli_gate_proj, pli_up_proj;

    gemma4e_pli_prefill_context(
        gemma4e_desc* desc,
        gemma4e_common_buffers* bufs,
        Gemma4e_ImageEncoder* image_encoder
    ) : desc(desc), bufs(bufs), image_encoder(image_encoder)
    {
        this->pli_down_proj = image_encoder->proj->create_app();
        this->pli_gate_proj = image_encoder->proj->create_app();
        this->pli_up_proj   = image_encoder->proj->create_app();
    }

    void setup(const gemma4e_prefill_shape& s);

    /// @brief Another block over the same buffers and weights that keeps its own sequences, for a second row geometry.
    std::unique_ptr<gemma4e_pli_prefill_context> make_sibling() const {
        return std::make_unique<gemma4e_pli_prefill_context>(desc, bufs, image_encoder);
    }

    /// @brief Projects the token embeddings down into the per-layer embeddings and
    ///        folds them into the per-layer-input stream. Runs once per batch.
    void pre_pass(const gemma4e_prefill_shape& s, buffer<bf16>& pli_down_weights, buffer<bf16>& pli_input_norm);

    /// @brief Gates this layer's per-layer embedding and adds it back to the
    ///        residual, leaving the result in hidden_state_buffer.
    void layer_pass(const gemma4e_prefill_shape& s, int layer_idx,
                    buffer<bf16>& pli_gate_up_weights, buffer<bf16>& pli_final_norm);
};

/// @brief One prefill pass over the model, one layer at a time.
///
/// Owns the prefill xclbins, the sequence generators and every scratch buffer
/// the prefill path needs; the model only hands it weights and a kv cache. The
/// generated sequences are cached on the batch geometry, so a prefill that
/// repeats a shape regenerates nothing.
struct gemma4e_prefill_context {
    gemma4e_desc* desc;
    npu_xclbin_manager* npu;
    gemma4e_common_buffers bufs;

    std::unique_ptr<Gemm> gemm_seq_gen;
    std::unique_ptr<gemma4e_dequant_prefill_context> dequant_block;
    std::unique_ptr<gemma4e_attn_block_prefill_context> attn_block;
    std::unique_ptr<gemma4e_mlp_prefill_context> mlp;
    std::unique_ptr<gemma4e_pli_prefill_context> pli;

    /// The "tail" block set: same buffers and caches, sequences for a geometry of one attention chunk. Once the layers
    /// that own a kv cache are done, only the last token still matters (the shared layers fill no cache and the logits
    /// read one row), so those layers run over the final chunk of rows alone instead of the whole batch.
    std::unique_ptr<gemma4e_attn_block_prefill_context> attn_block_tail;
    std::unique_ptr<gemma4e_mlp_prefill_context> mlp_tail;
    std::unique_ptr<gemma4e_pli_prefill_context> pli_tail;
    bool tail_active = false;

    /// The projections of the caching layers on the block-float kernel (null when that xclbin is not shipped or switched off with
    /// FLM_GEMMA4E_NO_BFP): they then skip the dequant and run straight from the packed weights.
    std::unique_ptr<gemma4e_bfp::gemma4e_bfp_block> bfp;

    npu_app_manager* gemm_app_manager;
    npu_app_manager* dequant_app_manager;
    npu_app_manager* mha_app_manager;
    npu_app_manager* swa_app_manager;

    /// @brief the attention kernel consumes whole chunks of this many rows
    static constexpr int L_chunk = 16 * 8;
    /// @brief the attention kernel refuses batches shorter than this
    static constexpr int L_MIN = 256;

    /// @note  The shared buffers are sized from both L_padded_512 (the residual-side
    ///        tensors) and L_padded (the q/k/v/gate/up scratch), and the two can move
    ///        independently, so reallocate when either changes.
    int L_padded_512_old = -1;
    int L_padded_old = -1;

    gemma4e_prefill_context(
        npu_xclbin_manager* npu,
        gemma4e_desc* desc,
        LM_Config& config,
        gemma4e_dflash_sequence* seq_gen,
        std::unique_ptr<gemma4e_pli_prefill_context> pli,
        uint32_t MAX_L
    ) : desc(desc), npu(npu)
    {
        this->gemm_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "mm.xclbin"));
        this->dequant_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "dequant.xclbin"));
        this->mha_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "attn.xclbin"));
        this->swa_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "swa.xclbin"));

        this->gemm_seq_gen = std::make_unique<Gemm>(config);

        const std::string bfp_xclbin = utils::path_join(config.exec_path, "xclbins", config.model_name, "dequant_mm_bfp.xclbin");
        if (!std::getenv("FLM_GEMMA4E_NO_BFP") && std::filesystem::exists(bfp_xclbin)) {
            this->bfp = std::make_unique<gemma4e_bfp::gemma4e_bfp_block>(desc, npu, npu->register_xclbin(bfp_xclbin));
        }

        this->attn_block = std::make_unique<gemma4e_attn_block_prefill_context>(
            desc, gemm_seq_gen.get(), seq_gen, &bufs,
            gemm_app_manager, mha_app_manager, swa_app_manager
        );
        this->mlp = std::make_unique<gemma4e_mlp_prefill_context>(
            desc, gemm_seq_gen.get(), &bufs, gemm_app_manager
        );
        this->dequant_block = std::make_unique<gemma4e_dequant_prefill_context>(
            desc, seq_gen, dequant_app_manager
        );
        // the per layer input apps must already exist by now: they are created on
        // the image encoder's manager, which has to happen before layer.xclbin is
        // registered, so the model builds this block and hands it over.
        this->pli = std::move(pli);
        this->pli->bufs = &bufs;

        this->attn_block_tail = std::make_unique<gemma4e_attn_block_prefill_context>(
            desc, gemm_seq_gen.get(), seq_gen, &bufs,
            gemm_app_manager, mha_app_manager, swa_app_manager
        );
        this->mlp_tail = std::make_unique<gemma4e_mlp_prefill_context>(
            desc, gemm_seq_gen.get(), &bufs, gemm_app_manager
        );
        this->pli_tail = this->pli->make_sibling();

        this->set_max_length(MAX_L);
    }

    void set_max_length(uint32_t MAX_L) {
        this->attn_block->set_max_length(MAX_L);
        this->attn_block_tail->borrow_caches_from(*this->attn_block);
    }

    void zero_scratch() { this->attn_block->zero_scratch(); }

    /// @brief Computes the row geometry of a prefill call and readies every block.
    /// @param L_in     number of tokens to prefill
    /// @param L_begin  context length already in the kv cache
    gemma4e_prefill_shape setup(int L_in, int L_begin, int swa_base_old = 0, int swa_base_new = 0, int swa_zero_hi = 0) {
        gemma4e_prefill_shape s;
        s.swa_base_old = swa_base_old;
        s.swa_base_new = swa_base_new;
        s.swa_zero_hi = swa_zero_hi;
        s.L_in = L_in;
        s.L_effective = L_in;
        s.L_begin = L_begin;
        s.L_end = L_begin + L_in;
        s.L_begin_chunked = (L_begin / L_chunk) * L_chunk;
        s.L_offset = L_begin - s.L_begin_chunked;
        s.L_end_chunked = s.L_begin_chunked + (L_in + s.L_offset + L_MIN - 1) / L_MIN * L_MIN;
        s.L_padded = s.L_end_chunked - s.L_begin_chunked;
        s.L_padded_512 = ((s.L_padded + 511) / 512) * 512;
        s.sliding_l_begin = (s.L_begin_chunked - (int)desc->SLIDING_LENGTH) > 0
                          ? (s.L_begin_chunked - (int)desc->SLIDING_LENGTH) : 0;

        if (s.L_padded_512 > bufs.allocated_rows) {
            gemma4e_prefill_shape grown = s;
            const int granule = gemma4e_common_buffers::ALLOC_GRANULE_ROWS;
            grown.L_padded_512 = (s.L_padded_512 + granule - 1) / granule * granule;
            grown.L_padded     = grown.L_padded_512;
            bufs.allocate(npu, desc, grown);
            bufs.allocated_rows = grown.L_padded_512;
        }
        bufs.reset(desc, s);
        this->tail_active = false;

        // with the block-float kernel the caching layers need no bf16 gemm sequences for this geometry (the shared layers' tail pass
        // has its own block set and sets itself up in begin_tail)
        const bool generic_gemms = !this->bfp;
        PROF_BEGIN();
        this->attn_block->setup(s, generic_gemms);
        PROF_TICK("setup.attn");
        if (generic_gemms) this->mlp->setup(s);
        this->pli->setup(s);
        PROF_TICK("setup.pli");
        if (this->bfp) this->bfp->setup(s.L_padded_512);
        PROF_TICK("setup.bfp");
        return s;
    }

    /// @brief Whether the shared layers can run over the last chunk of rows alone for this batch.
    ///
    /// The chunk must start at or after the first new token (earlier rows are not in the batch), and the batch must be
    /// longer than one chunk or there is nothing to save.
    bool tail_applies(const gemma4e_prefill_shape& s) const {
        if (s.L_padded <= L_MIN) {
            return false;
        }
        return tail_begin(s) >= s.L_begin;
    }

    /// @brief Switches the remaining layers to the last chunk of rows; returns that chunk's geometry.
    ///
    /// The final rows of the residual stream and of the per-layer inputs move to the front of their buffers (the tail
    /// sequences address row 0 on), the rest is cleared as padding, and `forward` then runs the tail block set. The
    /// last real token ends up in row `L_effective - 1` of the returned geometry.
    gemma4e_prefill_shape begin_tail(const gemma4e_prefill_shape& s) {
        const int first = tail_begin(s);
        gemma4e_prefill_shape t;
        t.L_in = t.L_effective = s.L_end - first;
        t.L_begin = first;
        t.L_end = s.L_end;
        t.L_offset = 0;
        t.L_begin_chunked = first;
        t.L_end_chunked = first + L_MIN;
        t.L_padded = L_MIN;
        t.L_padded_512 = 512;
        t.sliding_l_begin = (first - (int)desc->SLIDING_LENGTH) > 0 ? (first - (int)desc->SLIDING_LENGTH) : 0;

        // where the tail's first row sits in the buffers of the full batch
        const size_t src_row = (size_t)(s.L_offset + first - s.L_begin);
        const size_t D = desc->D;
        const size_t PLI_ROW = (size_t)desc->PLI_D * desc->num_hidden_layers;
        move_rows_to_front(bufs.residual_buffer, D, src_row, t.L_effective, t.L_padded_512);
        move_rows_to_front(bufs.pli_embed_buffer, PLI_ROW, src_row, t.L_effective, t.L_padded_512);
        bufs.tensor_residual.assign(bufs.residual_buffer, desc->D, 0);
        bufs.tensor_pli_embed.assign(bufs.pli_embed_buffer, desc->PLI_D * desc->num_hidden_layers, 0);

        attn_block_tail->setup(t);
        mlp_tail->setup(t);
        pli_tail->setup(t);
        tail_active = true;
        return t;
    }

    /// @brief Row `i` of the residual stream, where the token embeddings are written.
    buffer<bf16>& residual_row(int i) { return bufs.tensor_residual[i]; }
    /// @brief Row `i` of the per-layer-input embeddings.
    buffer<bf16>& pli_embed_row(int i) { return bufs.tensor_pli_embed[i]; }

    /// @brief Runs one decoder layer over the whole batch, in place on the residual.
    void forward(
        int layer_idx,
        gemma4e_layer_type_t type,
        const gemma4e_prefill_shape& s,
        buffer<u8>& proj_weights,
        buffer<bf16>& rms_weights,
        buffer<bf16>& rope_rms_weights,
        buffer<bf16>& pli_gate_up_weights,
        buffer<bf16>& kv_cache,
        float layer_scale,
        SafeTensors* reference
    );

private:
    /// first position of the tail: the earliest chunk-aligned row from which the batch's end is within one chunk
    int tail_begin(const gemma4e_prefill_shape& s) const {
        const int first = s.L_end - L_MIN;
        return first <= 0 ? 0 : (first + L_chunk - 1) / L_chunk * L_chunk;
    }

    /// Moves `count` rows of `row_size` elements, starting at row `src_row`, to the start of `buf`, then clears the
    /// rows from `count` up to `total_rows` (padding rows must hold no stale values).
    static void move_rows_to_front(buffer<bf16>& buf, size_t row_size, size_t src_row, int count, int total_rows) {
        bf16* base = buf.data();
        memmove(base, base + src_row * row_size, (size_t)count * row_size * sizeof(bf16));
        memset(base + (size_t)count * row_size, 0, (size_t)(total_rows - count) * row_size * sizeof(bf16));
    }
};

#endif // __GEMMA4E_PREFILL_HPP__
