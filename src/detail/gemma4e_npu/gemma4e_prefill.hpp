#include "flm_override.hpp"
#ifndef __GEMMA4E_PREFILL_HPP__
#define __GEMMA4E_PREFILL_HPP__
#include <memory>
#include "modules/gemm.hpp"
#include "tensor_2d.hpp"
#include "gemma4e_npu_def.hpp"
#include "gemma4e_npu_sequence.hpp"
#include "gemma4e_cpu_functions.hpp"
#include "gemma4e_image.hpp"
#include "mmRuntimeSequence.hpp"
#include "utils/error_measure.hpp"

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

    gemma4e_common_buffers() {}

    /// @brief (Re)allocates every buffer for a batch geometry.
    void allocate(npu_xclbin_manager* npu, gemma4e_desc* desc, const gemma4e_prefill_shape& s) {
        const size_t D = desc->D;
        const size_t I = desc->INTERMEDIATE_SIZE;
        const size_t PLI = (size_t)desc->PLI_D * desc->num_hidden_layers;
        residual_buffer     = buffer<bf16>((size_t)s.L_padded_512 * D);
        hidden_state_buffer = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * D);
        q_buffer            = npu->create_bo_buffer<bf16>((size_t)s.L_padded * desc->DQ);
        k_buffer            = npu->create_bo_buffer<bf16>((size_t)s.L_padded * desc->DK);
        v_buffer            = npu->create_bo_buffer<bf16>((size_t)s.L_padded * desc->DV);
        attn_out_buffer     = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * desc->DQ);
        // the widest mlp a layer can ask for, so skip layers share these buffers
        gate_buffer         = npu->create_bo_buffer<bf16>((size_t)s.L_padded * I * 2);
        up_buffer           = npu->create_bo_buffer<bf16>((size_t)s.L_padded * I * 2);
        hid_buffer          = npu->create_bo_buffer<bf16>((size_t)s.L_padded * I * 2);
        pli_embed_buffer    = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * PLI);
        pli_down_buffer     = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * PLI);
        pli_gate_buffer     = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * desc->PLI_D);
        pli_hid_buffer      = npu->create_bo_buffer<bf16>((size_t)s.L_padded_512 * desc->PLI_D);
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
        gemma4e_npu_sequence* seq_gen,
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
            seq_gen->generate_dequant_seq(apps[t][QKV].seq(),  W.attn_qkv,    gemma4e_npu_sequence::NORMAL_DEQUANT);
            seq_gen->generate_dequant_seq(apps[t][O].seq(),    W.attn_output, gemma4e_npu_sequence::NORMAL_DEQUANT);
            seq_gen->generate_dequant_seq(apps[t][GATE].seq(), W.ffn_gate,    gemma4e_npu_sequence::GATE_MATRIX);
            seq_gen->generate_dequant_seq(apps[t][UP].seq(),   W.ffn_up,      gemma4e_npu_sequence::UP_MATRIX);
            seq_gen->generate_dequant_seq(apps[t][DOWN].seq(), W.ffn_down,    gemma4e_npu_sequence::NORMAL_DEQUANT);
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
    gemma4e_npu_sequence* seq_gen;
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
    buffer<bf16> kv_cache_global_prefill;

    npu_app_manager* mha_app_manager;
    npu_app_manager* swa_app_manager;

    gemma4e_attn_block_prefill_context(
        gemma4e_desc* desc,
        Gemm* gemm_seq_gen,
        gemma4e_npu_sequence* seq_gen,
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

    /// @brief Sizes the prefill-side caches, which span the whole context.
    void set_max_length(uint32_t MAX_L) {
        this->MAX_L = MAX_L;
        this->kv_cache_sliding_prefill = swa_app_manager->create_bo_buffer<bf16>((size_t)MAX_L * (desc->SWA_DK + desc->SWA_DV));
        this->kv_cache_global_prefill  = mha_app_manager->create_bo_buffer<bf16>((size_t)MAX_L * (desc->DK + desc->DV));
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
    void forward(
        const gemma4e_prefill_shape& s,
        gemma4e_layer_type_t type,
        buffer<bf16>& qkv_weights,
        buffer<bf16>& o_weights,
        buffer<bf16>& kv_cache,
        buffer<bf16>& q_norm,
        buffer<bf16>& k_norm,
        SafeTensors* reference
    ) {
        if (is_swa_layer(type)) {
            _forward_swa(s, type, qkv_weights, o_weights, kv_cache, q_norm, k_norm, reference);
        }
        else {
            _forward_global(s, type, qkv_weights, o_weights, kv_cache, q_norm, k_norm, reference);
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
                      buffer<bf16>& qkv_weights, buffer<bf16>& o_weights, buffer<bf16>& kv_cache,
                      buffer<bf16>& q_norm, buffer<bf16>& k_norm, SafeTensors* reference);
    void _forward_global(const gemma4e_prefill_shape& s, gemma4e_layer_type_t type,
                         buffer<bf16>& qkv_weights, buffer<bf16>& o_weights, buffer<bf16>& kv_cache,
                         buffer<bf16>& q_norm, buffer<bf16>& k_norm, SafeTensors* reference);
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
    void forward(
        const gemma4e_prefill_shape& s,
        bool double_wide,
        buffer<bf16>& gate_weights,
        buffer<bf16>& up_weights,
        buffer<bf16>& down_weights,
        SafeTensors* reference
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
        gemma4e_npu_sequence* seq_gen,
        std::unique_ptr<gemma4e_pli_prefill_context> pli,
        uint32_t MAX_L
    ) : desc(desc), npu(npu)
    {
        this->gemm_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "mm.xclbin"));
        this->dequant_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "dequant.xclbin"));
        this->mha_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "attn.xclbin"));
        this->swa_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "swa.xclbin"));

        this->gemm_seq_gen = std::make_unique<Gemm>(config);

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
        s.L_padded_512 = ((s.L_padded + 511) / 512) * 512;
        s.sliding_l_begin = (s.L_begin_chunked - (int)desc->SLIDING_LENGTH) > 0
                          ? (s.L_begin_chunked - (int)desc->SLIDING_LENGTH) : 0;

        if (L_padded_512_old != s.L_padded_512 || L_padded_old != s.L_padded) {
            bufs.allocate(npu, desc, s);
            L_padded_512_old = s.L_padded_512;
            L_padded_old     = s.L_padded;
        }
        bufs.reset(desc, s);

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
