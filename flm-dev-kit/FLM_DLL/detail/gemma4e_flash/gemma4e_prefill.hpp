// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __GEMMA4E_PREFILL_HPP__
#define __GEMMA4E_PREFILL_HPP__
#include <memory>
#include "modules/gemm.hpp"
#include "tensor_2d.hpp"
#include "gemma4e_npu_def.hpp"
#include "gemma4e_flash_sequence.hpp"
#include "gemma4e_cpu_functions.hpp"
#include "gemma4e_image.hpp"
#include "mmRuntimeSequence.hpp"
#include "utils/error_measure.hpp"

// ---------------------------------------------------------------------------
// Prefill profiling.
// Enable by compiling with -DGEMMA4E_PREFILL_PROFILE (all code below compiles to
// nothing otherwise, so it is zero-cost when disabled).
//
//   PREFILL_PROF_INIT();               // once at the start of a function
//   ... work ...
//   PREFILL_PROF_TICK("qkv_proj");     // prints time since the previous tick
// ---------------------------------------------------------------------------
// #define GEMMA4E_PREFILL_PROFILE
#ifdef GEMMA4E_PREFILL_PROFILE
#define PREFILL_PROF_INIT() time_utils::time_point __prof_t = time_utils::now()
#define PREFILL_PROF_TICK(name)                                                \
    do {                                                                       \
        time_utils::time_point __prof_now = time_utils::now();                 \
        time_utils::time_with_unit __prof_d =                                  \
            time_utils::re_unit(time_utils::duration_us(__prof_t, __prof_now));\
        header_print("prefill-prof", name << ": " << __prof_d.first << " "     \
                                          << __prof_d.second);                 \
        __prof_t = __prof_now;                                                 \
    } while (0)
#else
#define PREFILL_PROF_INIT() do {} while (0)
#define PREFILL_PROF_TICK(name) do {} while (0)
#endif

/// @brief Row geometry of one prefill call.
///
/// The attention kernel consumes whole chunks, so the batch is padded in front
/// (L_offset rows that belong to already-cached tokens) and at the back (up to
/// L_padded rows), and every gemm runs over the full L_padded rows -- the per
/// layer input gemms included, since L_MIN already matches the fused mm's row
/// granularity.
struct gemma4e_prefill_shape {
    int L_in = 0;             ///< tokens handed to this call
    int L_begin = 0;          ///< context length before this call
    int L_end = 0;            ///< context length after this call
    int L_effective = 0;      ///< rows that carry a real token, == L_in
    int L_offset = 0;         ///< rows of chunk padding in front of the first new token
    int L_begin_chunked = 0;  ///< chunk-aligned start fed to the attention kernel
    int L_end_chunked = 0;    ///< chunk-aligned end fed to the attention kernel
    int L_padded = 0;         ///< rows every gemm runs over
    int sliding_l_begin = 0;  ///< first position the sliding window still covers
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
    buffer<bf16> upgate_buffer;  ///< the interleaved up|gate band the fused mlp writes
    buffer<bf16> hid_buffer;

    // per layer input path
    buffer<bf16> pli_embed_buffer;
    buffer<bf16> pli_down_buffer;
    buffer<bf16> pli_gate_buffer;
    buffer<bf16> pli_hid_buffer;

    tensor_2d<bf16> tensor_residual;
    tensor_2d<bf16> tensor_pli_embed;

    gemma4e_common_buffers() {}

    /// @brief (Re)allocates every buffer for a batch geometry.
    void allocate(npu_xclbin_manager* npu, gemma4e_desc* desc, const gemma4e_prefill_shape& s) {
        const size_t D = desc->D;
        // The widest mlp any layer kind can ask for -- which is 2 * INTERMEDIATE_SIZE
        // only when the checkpoint enables double-wide skip layers. Sizing off
        // INTERMEDIATE_SIZE * 2 unconditionally, as this used to, allocated twice the
        // needed bytes for the two largest buffers on the whole prefill path whenever
        // use_double_wide_mlp is false (E4B, for one).
        const size_t I = desc->get_max_intermediate_size();
        const size_t PLI = (size_t)desc->PLI_D * desc->num_hidden_layers;
        // Release the previous geometry's buffers before cutting the new ones.
        // create_bo_buffer returns by value, so `x = create_bo_buffer(...)` only drops
        // the old bo *after* the new one exists. The allocation order below already
        // drains the old set as it goes, so this does not lower the peak by itself --
        // it makes the invariant explicit, so that a future pre-armed run holding one
        // of these bos cannot silently reintroduce a full doubling the way it did on
        // the qwen3vl prefill path.
        tensor_residual.clear();
        tensor_pli_embed.clear();
        residual_buffer     = buffer<bf16>();
        hidden_state_buffer = buffer<bf16>();
        q_buffer            = buffer<bf16>();
        k_buffer            = buffer<bf16>();
        v_buffer            = buffer<bf16>();
        attn_out_buffer     = buffer<bf16>();
        upgate_buffer       = buffer<bf16>();
        hid_buffer          = buffer<bf16>();
        pli_embed_buffer    = buffer<bf16>();
        pli_down_buffer     = buffer<bf16>();
        pli_gate_buffer     = buffer<bf16>();
        pli_hid_buffer      = buffer<bf16>();

        residual_buffer     = buffer<bf16>((size_t)s.L_padded * D);
        hidden_state_buffer = npu->create_bo_buffer<bf16>((size_t)s.L_padded * D);
        q_buffer            = npu->create_bo_buffer<bf16>((size_t)s.L_padded * desc->DQ);
        k_buffer            = npu->create_bo_buffer<bf16>((size_t)s.L_padded * desc->DK);
        v_buffer            = npu->create_bo_buffer<bf16>((size_t)s.L_padded * desc->DV);
        attn_out_buffer     = npu->create_bo_buffer<bf16>((size_t)s.L_padded * desc->DQ);
        // sized for the widest layer kind, so every layer shares these buffers; up and
        // gate come back interleaved in one band, hence the factor 2 on upgate
        upgate_buffer       = npu->create_bo_buffer<bf16>((size_t)s.L_padded * I * 2);
        hid_buffer          = npu->create_bo_buffer<bf16>((size_t)s.L_padded * I);
        pli_embed_buffer    = npu->create_bo_buffer<bf16>((size_t)s.L_padded * PLI);
        pli_down_buffer     = npu->create_bo_buffer<bf16>((size_t)s.L_padded * PLI);
        pli_gate_buffer     = npu->create_bo_buffer<bf16>((size_t)s.L_padded * desc->PLI_D);
        // the only buffer a kernel does not fully overwrite: the gate multiply
        // fills just the L_effective real-token rows, so clear the padding once
        pli_hid_buffer      = npu->create_bo_buffer<bf16>((size_t)s.L_padded * desc->PLI_D);
        pli_hid_buffer.memset((bf16)0);
    }

    /// @brief Clears the padding rows and re-points the row views at this batch.
    void reset(gemma4e_desc* desc, const gemma4e_prefill_shape& s) {
        // rows outside [L_offset, L_offset + L_in) are padding; they still take
        // part in every gemm, so leave no stale values in them.
        memset(residual_buffer.data(), 0, residual_buffer.size() * sizeof(bf16));
        memset(pli_embed_buffer.data(), 0, pli_embed_buffer.size() * sizeof(bf16));
        tensor_residual.assign(residual_buffer, desc->D, s.L_offset);
        tensor_pli_embed.assign(pli_embed_buffer, desc->PLI_D * desc->num_hidden_layers, s.L_offset);
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
    gemma4e_flash_sequence* seq_gen;
    gemma4e_common_buffers* bufs;

    int L_padded_old = -1;
    int L_begin_chunked_old = -1;
    int L_end_chunked_old = -1;
    uint32_t MAX_L = 0;

    /// projections, one sequence per layer kind: the fused array reads the
    /// quantized weights straight out of the layer blob, and every kind puts
    /// them at a different offset, so the offset is baked into the sequence.
    /// The k/v entries of a skip layer are never generated or run.
    npu_app q_proj[e_gemma4e_total_layer_types];
    npu_app k_proj[e_gemma4e_total_layer_types];
    npu_app v_proj[e_gemma4e_total_layer_types];
    npu_app o_proj[e_gemma4e_total_layer_types];
    npu_app mha_engine;
    npu_app swa_engine;

    /// caches the attention kernels read: contiguous, and rebuilt every layer
    buffer<bf16> kv_cache_sliding_prefill;
    buffer<bf16> kv_cache_global_prefill;

    npu_app_manager* app_manager;

    gemma4e_attn_block_prefill_context(
        gemma4e_desc* desc,
        gemma4e_flash_sequence* seq_gen,
        gemma4e_common_buffers* bufs,
        npu_app_manager* app_manager
    ) : desc(desc), seq_gen(seq_gen), bufs(bufs), app_manager(app_manager)
    {
        for (int t = 0; t < e_gemma4e_total_layer_types; t++) {
            this->q_proj[t] = app_manager->create_app();
            this->k_proj[t] = app_manager->create_app();
            this->v_proj[t] = app_manager->create_app();
            this->o_proj[t] = app_manager->create_app();
        }
        this->mha_engine = app_manager->create_app();
        this->swa_engine = app_manager->create_app();
    }

    /// @brief Sizes the prefill-side caches, which span the whole context.
    void set_max_length(uint32_t MAX_L) {
        this->MAX_L = MAX_L;
        this->kv_cache_sliding_prefill = app_manager->create_bo_buffer<bf16>((size_t)MAX_L * (desc->SWA_DK + desc->SWA_DV));
        this->kv_cache_global_prefill  = app_manager->create_bo_buffer<bf16>((size_t)MAX_L * (desc->DK + desc->DV));
        // the swa kernel rounds its window start down to a whole kv block, so it
        // can reach rows below sliding_l_begin; they must read as zero.
        this->kv_cache_sliding_prefill.memset((bf16)0);
        this->kv_cache_global_prefill.memset((bf16)0);
        this->kv_cache_sliding_prefill.sync_to_device();
        this->kv_cache_global_prefill.sync_to_device();
        // the cached sequences address these buffers, so force a regeneration
        this->L_padded_old = -1;
        this->L_begin_chunked_old = -1;
        this->L_end_chunked_old = -1;
    }

    /// @brief Regenerates the projection and attention sequences, if the geometry moved.
    void setup(const gemma4e_prefill_shape& s) {
        const uint32_t D = desc->D;
        if (L_padded_old != s.L_padded) {
            for (int t = 0; t < e_gemma4e_total_layer_types; t++) {
                gemma4e_layer_type_t type = static_cast<gemma4e_layer_type_t>(t);
                gemma4e_layer_weight_def& W = desc->weight_desc(type);
                const uint32_t _DQ = desc->get_DQ(type);
                seq_gen->gen_dequant_mm_512(this->q_proj[t].seq(), s.L_padded, D, _DQ,
                    W.attn_q.offset, 0, /*rtp_act=*/0);
                if (W.has_kv) {
                    seq_gen->gen_dequant_mm_512(this->k_proj[t].seq(), s.L_padded, D, desc->get_DK(type),
                        W.attn_k.offset, 0, /*rtp_act=*/0);
                    seq_gen->gen_dequant_mm_512(this->v_proj[t].seq(), s.L_padded, D, desc->get_DV(type),
                        W.attn_v.offset, 0, /*rtp_act=*/0);
                }
                seq_gen->gen_dequant_mm_512(this->o_proj[t].seq(), s.L_padded, _DQ, D,
                    W.attn_output.offset, 0, /*rtp_act=*/0);
            }
            L_padded_old = s.L_padded;
        }
        if (L_begin_chunked_old != s.L_begin_chunked || L_end_chunked_old != s.L_end_chunked) {
            seq_gen->gen_fused_mha_seq(this->mha_engine.seq(), s.L_begin_chunked, s.L_end_chunked);
            seq_gen->gen_fused_swa_seq(this->swa_engine.seq(), s.L_begin_chunked, s.L_end_chunked);
            L_begin_chunked_old = s.L_begin_chunked;
            L_end_chunked_old = s.L_end_chunked;
        }
    }

    /// @brief Runs the attention block of one layer, hidden_state_buffer in and out.
    void forward(
        const gemma4e_prefill_shape& s,
        gemma4e_layer_type_t type,
        buffer<u8>& proj_weights,
        buffer<bf16>& kv_cache,
        buffer<bf16>& q_norm,
        buffer<bf16>& k_norm,
        SafeTensors* reference
    ) {
        if (is_swa_layer(type)) {
            _forward_swa(s, type, proj_weights, kv_cache, q_norm, k_norm, reference);
        }
        else {
            _forward_global(s, type, proj_weights, kv_cache, q_norm, k_norm, reference);
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
        int DK, int DV
    );

private:
    void _forward_swa(const gemma4e_prefill_shape& s, gemma4e_layer_type_t type,
                      buffer<u8>& proj_weights, buffer<bf16>& kv_cache,
                      buffer<bf16>& q_norm, buffer<bf16>& k_norm, SafeTensors* reference);
    void _forward_global(const gemma4e_prefill_shape& s, gemma4e_layer_type_t type,
                         buffer<u8>& proj_weights, buffer<bf16>& kv_cache,
                         buffer<bf16>& q_norm, buffer<bf16>& k_norm, SafeTensors* reference);
};

/// @brief Feed-forward half of one prefill layer.
///
/// Skip layers run a double-wide mlp when the model enables it, which is a
/// different sequence over the same buffers, so both widths are kept ready.
struct gemma4e_mlp_prefill_context {
    gemma4e_desc* desc;
    gemma4e_flash_sequence* seq_gen;
    gemma4e_common_buffers* bufs;

    int L_padded_old = -1;

    /// [layer kind][0 = up, 1 = gate] over the one interleaved up|gate band,
    /// plus the down projection; a skip layer is simply a wider band.
    npu_app upgate_proj[e_gemma4e_total_layer_types][2];
    npu_app down_proj[e_gemma4e_total_layer_types];

    /// @brief set by the owning prefill context
    struct gemma4e_pli_prefill_context* pli = nullptr;

    gemma4e_mlp_prefill_context(
        gemma4e_desc* desc,
        gemma4e_flash_sequence* seq_gen,
        gemma4e_common_buffers* bufs,
        npu_app_manager* app_manager
    ) : desc(desc), seq_gen(seq_gen), bufs(bufs)
    {
        for (int t = 0; t < e_gemma4e_total_layer_types; t++) {
            this->upgate_proj[t][0] = app_manager->create_app();
            this->upgate_proj[t][1] = app_manager->create_app();
            this->down_proj[t] = app_manager->create_app();
        }
    }

    void setup(const gemma4e_prefill_shape& s) {
        if (L_padded_old == s.L_padded) {
            return;
        }
        const uint32_t D = desc->D;
        for (int t = 0; t < e_gemma4e_total_layer_types; t++) {
            gemma4e_layer_type_t type = static_cast<gemma4e_layer_type_t>(t);
            gemma4e_layer_weight_def& W = desc->weight_desc(type);
            const uint32_t I = desc->get_intermediate_size(type);
            // n_stride = 2 walks every other 512-column slab of the band and
            // n_first picks which half of the up/gate interleave it lands on.
            // Both runs drive all six mm columns, so they serialize; they only
            // ever write disjoint slabs of upgate_buffer.
            seq_gen->gen_dequant_mm_512(this->upgate_proj[t][0].seq(), s.L_padded, D, I * 2,
                W.ffn_up_gate.offset, 0, /*rtp_act=*/0, /*n_stride=*/2, /*n_first=*/0);
            seq_gen->gen_dequant_mm_512(this->upgate_proj[t][1].seq(), s.L_padded, D, I * 2,
                W.ffn_up_gate.offset, 0, /*rtp_act=*/2, /*n_stride=*/2, /*n_first=*/1);
            seq_gen->gen_dequant_mm_512(this->down_proj[t].seq(), s.L_padded, I, D,
                W.ffn_down.offset, 0, /*rtp_act=*/0);
        }
        L_padded_old = s.L_padded;
    }

    /// @brief Runs the mlp of one layer, hidden_state_buffer in and out.
    void forward(
        const gemma4e_prefill_shape& s,
        gemma4e_layer_type_t type,
        buffer<u8>& proj_weights,
        SafeTensors* reference
    );
};

/// @brief The per-layer-input path: a model-wide down projection run once per
///        batch, and a gate/up pair run at the end of every layer.
struct gemma4e_pli_prefill_context {
    gemma4e_desc* desc;
    gemma4e_flash_sequence* seq_gen;
    gemma4e_common_buffers* bufs;

    int L_padded_old = -1;

    npu_app pli_down_proj, pli_gate_proj, pli_up_proj;

    /// @note These three gemms used to run on the image encoder's bf16 mm
    ///       overlay, which meant every layer crossed the overlay boundary
    ///       twice and paid a ~2.5 ms hardware context switch each way.  They
    ///       run on fused_prefill now, through its bf16 passthrough mode, so
    ///       there is no boundary left to cross and no preload to hide it with.
    gemma4e_pli_prefill_context(
        gemma4e_desc* desc,
        gemma4e_flash_sequence* seq_gen,
        gemma4e_common_buffers* bufs,
        npu_app_manager* app_manager
    ) : desc(desc), seq_gen(seq_gen), bufs(bufs)
    {
        this->pli_down_proj = app_manager->create_app();
        this->pli_gate_proj = app_manager->create_app();
        this->pli_up_proj   = app_manager->create_app();
    }

    void setup(const gemma4e_prefill_shape& s);

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

    std::unique_ptr<gemma4e_attn_block_prefill_context> attn_block;
    std::unique_ptr<gemma4e_mlp_prefill_context> mlp;
    std::unique_ptr<gemma4e_pli_prefill_context> pli;

    /// @brief the one prefill overlay: full attention on column 0, the fused
    ///        Q4_1 dequant+matmul on columns 1-6, sliding attention on column 7
    npu_app_manager* prefill_app_manager;

    /// @brief the attention kernel consumes whole chunks of this many rows
    static constexpr int L_chunk = 16 * 8;
    /// @brief row granularity every gemm and both attention columns round up to.
    ///        The fused mm's `i` step is 4 core rows of m = 32; the attention
    ///        columns step 4 * lq, which is 64 at most.  128 satisfies both.
    static constexpr int L_MIN = 128;

    int L_padded_old = -1;

    gemma4e_prefill_context(
        npu_xclbin_manager* npu,
        gemma4e_desc* desc,
        LM_Config& config,
        gemma4e_flash_sequence* seq_gen,
        uint32_t MAX_L
    ) : desc(desc), npu(npu)
    {
        // one overlay covers the whole prefill layer: the three regions are
        // independent apps on the same xclbin, so nothing is swapped mid-layer.
        this->prefill_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "fused_prefill.xclbin"));

        this->attn_block = std::make_unique<gemma4e_attn_block_prefill_context>(
            desc, seq_gen, &bufs, prefill_app_manager
        );
        this->mlp = std::make_unique<gemma4e_mlp_prefill_context>(
            desc, seq_gen, &bufs, prefill_app_manager
        );
        this->pli = std::make_unique<gemma4e_pli_prefill_context>(
            desc, seq_gen, &bufs, prefill_app_manager
        );
        this->mlp->pli = this->pli.get();

        this->attn_block->set_max_length(MAX_L);
    }

    void set_max_length(uint32_t MAX_L) { this->attn_block->set_max_length(MAX_L); }

    /// @brief Computes the row geometry of a prefill call and readies every block.
    /// @param L_in     number of tokens to prefill
    /// @param L_begin  context length already in the kv cache
    gemma4e_prefill_shape setup(int L_in, int L_begin) {
        gemma4e_prefill_shape s;
        s.L_in = L_in;
        s.L_effective = L_in;
        s.L_begin = L_begin;
        s.L_end = L_begin + L_in;
        s.L_begin_chunked = (L_begin / L_chunk) * L_chunk;
        s.L_offset = L_begin - s.L_begin_chunked;
        s.L_end_chunked = s.L_begin_chunked + (L_in + s.L_offset + L_MIN - 1) / L_MIN * L_MIN;
        s.L_padded = s.L_end_chunked - s.L_begin_chunked;
        s.sliding_l_begin = (s.L_begin_chunked - (int)desc->SLIDING_LENGTH) > 0
                          ? (s.L_begin_chunked - (int)desc->SLIDING_LENGTH) : 0;

        if (L_padded_old != s.L_padded) {
            bufs.allocate(npu, desc, s);
            L_padded_old = s.L_padded;
        }
        bufs.reset(desc, s);

        // fprintf(stderr, "[shape] L_in=%d L_begin=%d L_offset=%d L_padded=%d\n", s.L_in, s.L_begin, s.L_offset, s.L_padded);
        this->attn_block->setup(s);
        this->mlp->setup(s);
        this->pli->setup(s);
        return s;
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
};

#endif // __GEMMA4E_PREFILL_HPP__
