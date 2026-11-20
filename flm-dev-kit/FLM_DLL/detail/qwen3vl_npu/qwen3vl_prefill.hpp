// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __QWEN3VL_PREFILL_HPP__
#define __QWEN3VL_PREFILL_HPP__
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"
#include "modules/mha.hpp"
#include "tensor_2d.hpp"
#include "qwen3vl_model_def.hpp"
#include "qwen3vl_cpu_functions.hpp"
#include <vector>

// ---------------------------------------------------------------------------
// Prefill profiling.
// Enable by compiling with -DQWEN3VL_PREFILL_PROFILE (all code below compiles to
// nothing otherwise, so it is zero-cost when disabled).
//
//   PREFILL_PROF_INIT();               // once at the start of a function
//   ... work ...
//   PREFILL_PROF_TICK("dequant");      // prints time since the previous tick
// ---------------------------------------------------------------------------
// #define QWEN3VL_PREFILL_PROFILE
#ifdef QWEN3VL_PREFILL_PROFILE
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
/// (L_offset rows belonging to already-cached tokens) and at the back (up to
/// L_padded rows), and every gemm runs over the full L_padded rows.
struct qwen3vl_prefill_shape {
    int L_in = 0;             ///< tokens handed to this call
    int L_begin = 0;          ///< context length before this call
    int L_offset = 0;         ///< rows of chunk padding in front of the first new token
    int L_padded = 0;         ///< rows every gemm runs over
    int L_begin_chunked = 0;  ///< chunk-aligned start fed to the mha kernel
    int L_end_chunked = 0;    ///< chunk-aligned end fed to the mha kernel
};

/// @brief Buffers shared by every block of a prefill layer.
struct qwen3vl_common_buffers {
    buffer<bf16> x;         ///< device: rms-normed activations, block input
    buffer<bf16> o;         ///< device: block output (attn o_proj, then mlp down_proj)
    buffer<bf16> residual;  ///< host: the running residual stream

    tensor_2d<bf16> x_2d;
    tensor_2d<bf16> residual_2d;

    qwen3vl_common_buffers() {}
};

/// @brief Dequantizes one layer's q4_1 weights into the bf16 buffers the prefill
///        gemms consume.
///
/// The five matrices are launched together and waited on separately so the
/// caller can overlap the host-side rms norm with the last one.
struct qwen3vl_dequant_prefill_context {
    qwen3vl_desc* desc;
    Dequant* seq_gen;
    bool use_runlist;

    npu_app dequant_qkv;
    npu_app dequant_o;
    npu_app dequant_gate;
    npu_app dequant_up;
    npu_app dequant_down;
    flm_rt::runlist dequant_all;
    flm_rt::run pending_run;

    buffer<bf16> qkv_weights;
    buffer<bf16> o_weights;
    buffer<bf16> gate_weights;
    buffer<bf16> up_weights;
    buffer<bf16> down_weights;

    qwen3vl_dequant_prefill_context(
        qwen3vl_desc* desc,
        Dequant* seq_gen,
        npu_app_manager* dequant_app_manager,
        bool use_runlist
    ) : desc(desc), seq_gen(seq_gen), use_runlist(use_runlist)
    {
        this->dequant_qkv = dequant_app_manager->create_app();
        this->dequant_o = dequant_app_manager->create_app();
        this->dequant_gate = dequant_app_manager->create_app();
        this->dequant_up = dequant_app_manager->create_app();
        this->dequant_down = dequant_app_manager->create_app();
        if (use_runlist) {
            this->dequant_all = dequant_app_manager->create_runlist();
        }

        const uint32_t D = desc->D;
        const uint32_t DQ = desc->DQ;
        const uint32_t QKV = desc->get_qkv_out_dim();
        const uint32_t I = desc->intermediate_size;

        this->qkv_weights  = this->dequant_qkv.create_bo_buffer<bf16>((size_t)D * QKV);
        this->o_weights    = this->dequant_o.create_bo_buffer<bf16>((size_t)DQ * D);
        this->gate_weights = this->dequant_gate.create_bo_buffer<bf16>((size_t)D * I);
        this->up_weights   = this->dequant_up.create_bo_buffer<bf16>((size_t)D * I);
        this->down_weights = this->dequant_down.create_bo_buffer<bf16>((size_t)I * D);

        // up and gate share one band and are told apart by the mode argument
        seq_gen->generate_dequant_q4_1_seq(this->dequant_qkv.seq(), D, QKV, desc->get_qkv_offset(), 0);
        seq_gen->generate_dequant_q4_1_seq(this->dequant_o.seq(), DQ, D, desc->get_attn_o_offset(), 0);
        seq_gen->generate_dequant_q4_1_seq(this->dequant_gate.seq(), D, I, desc->get_upgate_offset(), 2);
        seq_gen->generate_dequant_q4_1_seq(this->dequant_up.seq(), D, I, desc->get_upgate_offset(), 1);
        seq_gen->generate_dequant_q4_1_seq(this->dequant_down.seq(), I, D, desc->get_down_offset(), 0);
    }

    /// @brief Launches all five dequants; the last one is still in flight on return.
    void start(buffer<u8>& proj_weights) {
        auto qkv_run  = this->dequant_qkv.create_run(this->qkv_weights, proj_weights);
        auto o_run    = this->dequant_o.create_run(this->o_weights, proj_weights);
        auto gate_run = this->dequant_gate.create_run(this->gate_weights, proj_weights);
        auto up_run   = this->dequant_up.create_run(this->up_weights, proj_weights);
        auto down_run = this->dequant_down.create_run(this->down_weights, proj_weights);

        if (use_runlist) {
            this->dequant_all.reset();
            this->dequant_all.add(qkv_run);
            this->dequant_all.add(o_run);
            this->dequant_all.add(gate_run);
            this->dequant_all.add(up_run);
            this->dequant_all.add(down_run);
            this->dequant_all.execute();
        }
        else {
            // preemption is enabled: the runlist is unavailable, so the first
            // four are serialized and only the last one overlaps the caller.
            qkv_run.start();
            flm::npu::wait_for_completion(qkv_run, "qkv_run");
            o_run.start();
            flm::npu::wait_for_completion(o_run, "o_run");
            gate_run.start();
            flm::npu::wait_for_completion(gate_run, "gate_run");
            up_run.start();
            flm::npu::wait_for_completion(up_run, "up_run");
            down_run.start();
            this->pending_run = down_run;
        }
    }

    void wait() {
        if (use_runlist) {
            flm::npu::wait_for_completion(this->dequant_all, "dequant_all");
        }
        else {
            flm::npu::wait_for_completion(this->pending_run, "pending_run");
        }
    }
};

/// @brief Attention half of one prefill layer: q/k/v projections, per-head norm
///        plus rope, kv cache fill, mha, output projection.
struct qwen3vl_attn_block_prefill_context {
    qwen3vl_desc* desc;
    Gemm* gemm_seq_gen;
    MHA* mha_seq_gen;
    qwen3vl_common_buffers* common_bufs;

    int L_padded_old = -1;
    int L_begin_chunked_old = -1;
    int L_end_chunked_old = -1;
    uint32_t MAX_L = 0;

    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app o_proj;
    npu_app mha_engine;

    buffer<bf16> q;
    buffer<bf16> k;
    buffer<bf16> v;
    buffer<bf16> attn_out;

    tensor_2d<bf16> q_2d;
    tensor_2d<bf16> k_2d;

    /// @brief mrope cos/sin scratch, DH floats, rewritten for every row
    std::vector<float> rope_values;

    /// @brief where the four quarter-cache bands start inside a layer's kv cache
    size_t K03_offset = 0;
    size_t K47_offset = 0;
    size_t V03_offset = 0;
    size_t V47_offset = 0;

    qwen3vl_attn_block_prefill_context(
        qwen3vl_desc* desc,
        Gemm* gemm_seq_gen,
        MHA* mha_seq_gen,
        qwen3vl_common_buffers* common_bufs,
        npu_app_manager* gemm_app_manager,
        npu_app_manager* mha_app_manager
    ) : desc(desc), gemm_seq_gen(gemm_seq_gen), mha_seq_gen(mha_seq_gen), common_bufs(common_bufs)
    {
        this->q_proj = gemm_app_manager->create_app();
        this->k_proj = gemm_app_manager->create_app();
        this->v_proj = gemm_app_manager->create_app();
        this->o_proj = gemm_app_manager->create_app();
        this->mha_engine = mha_app_manager->create_app();
        this->rope_values.resize(desc->DH);
    }

    void set_kv_layout(size_t K03, size_t K47, size_t V03, size_t V47) {
        this->K03_offset = K03;
        this->K47_offset = K47;
        this->V03_offset = V03;
        this->V47_offset = V47;
        // the mha sequence bakes in the cache layout, so force a regeneration
        this->L_begin_chunked_old = -1;
    }

    void set_max_length(uint32_t MAX_L) {
        this->MAX_L = MAX_L;
        this->L_begin_chunked_old = -1;
    }

    /// @brief Regenerates the sequences and grows the scratch buffers for a shape.
    void setup(const qwen3vl_prefill_shape& s) {
        const uint32_t D = desc->D;
        const uint32_t DQ = desc->DQ;
        const uint32_t DK = desc->DK;
        const uint32_t DV = desc->DV;
        const bool has_bias = false;
        const uint32_t bias_offset = 0;

        if (L_padded_old != s.L_padded) {
            // element offsets into the dequantized band; o has its own buffer.
            // The band is laid out K | V | Q (see qwen3vl_desc::build).
            gemm_seq_gen->generate_seq(this->k_proj.seq(), s.L_padded, D, DK, 0,
                has_bias, Gemm::NO_Activation, bias_offset);
            gemm_seq_gen->generate_seq(this->v_proj.seq(), s.L_padded, D, DV, D * DK,
                has_bias, Gemm::NO_Activation, bias_offset);
            gemm_seq_gen->generate_seq(this->q_proj.seq(), s.L_padded, D, DQ, D * (DK + DV),
                has_bias, Gemm::NO_Activation, bias_offset);
            gemm_seq_gen->generate_seq(this->o_proj.seq(), s.L_padded, DQ, D, 0,
                has_bias, Gemm::NO_Activation, bias_offset);
            this->q = this->q_proj.create_bo_buffer<bf16>((size_t)s.L_padded * DQ);
            this->k = this->k_proj.create_bo_buffer<bf16>((size_t)s.L_padded * DK);
            this->v = this->v_proj.create_bo_buffer<bf16>((size_t)s.L_padded * DV);
            this->attn_out = this->mha_engine.create_bo_buffer<bf16>((size_t)s.L_padded * DQ);
        }
        if (L_begin_chunked_old != s.L_begin_chunked || L_end_chunked_old != s.L_end_chunked) {
            mha_seq_gen->generate_mha_sequence(this->mha_engine.seq(), s.L_begin_chunked, s.L_end_chunked, this->MAX_L, false);
        }
        this->q_2d.assign(this->q, DQ, s.L_offset);
        this->k_2d.assign(this->k, DK, s.L_offset);

        L_padded_old = s.L_padded;
        L_begin_chunked_old = s.L_begin_chunked;
        L_end_chunked_old = s.L_end_chunked;
    }

    /// @brief Runs the attention block; the result lands in common_bufs->o.
    /// @param input_idx  three mrope indices (t, h, w) per token, from the model.
    /// @param rope_rms_w rows of the rope/q_norm/k_norm buffer: [1] is q_norm, [2] is k_norm.
    buffer<bf16> forward(
        const qwen3vl_prefill_shape& s,
        const std::vector<int>& input_idx,
        buffer<bf16>& qkv_weights,
        buffer<bf16>& o_weights,
        tensor_2d<bf16>& rope_rms_w,
        buffer<bf16>& kv_cache
    ) {
        PREFILL_PROF_INIT();
        this->q_proj(this->q, common_bufs->x, qkv_weights);

        // the k gemm runs while the host rotates q
        auto run_k = this->k_proj.create_run(this->k, common_bufs->x, qkv_weights);
        run_k.start();
        this->q.sync_from_device();
        for (int ll = 0; ll < s.L_in; ll++) {
            qwen3vl_cpu_func::_rope_rms(this->q_2d[ll],
                input_idx[ll * 3 + 0], input_idx[ll * 3 + 1], input_idx[ll * 3 + 2],
                rope_rms_w[1], desc->DH, desc->inv_freq, this->rope_values, desc->rms_eps);
        }
        this->q.sync_to_device();
        flm::npu::wait_for_completion(run_k, "run_k");
        this->k.sync_from_device();

        // and the v gemm runs while the host rotates k
        auto run_v = this->v_proj.create_run(this->v, common_bufs->x, qkv_weights);
        run_v.start();
        for (int ll = 0; ll < s.L_in; ll++) {
            qwen3vl_cpu_func::_rope_rms(this->k_2d[ll],
                input_idx[ll * 3 + 0], input_idx[ll * 3 + 1], input_idx[ll * 3 + 2],
                rope_rms_w[2], desc->DH, desc->inv_freq, this->rope_values, desc->rms_eps);
        }
        this->k.sync_to_device();
        flm::npu::wait_for_completion(run_v, "run_v");
        this->q.sync_to_device();
        this->v.sync_from_device();
        PREFILL_PROF_TICK("attn:qkv_proj");

        this->fill_kv_cache(kv_cache, s);
        kv_cache.sync_to_device();
        PREFILL_PROF_TICK("attn:fill_kv_cache");

        this->mha_engine(this->attn_out, this->q, kv_cache);
        this->attn_out.sync_from_device();
        PREFILL_PROF_TICK("attn:mha");

        this->o_proj(common_bufs->o, this->attn_out, o_weights);
        common_bufs->o.sync_from_device();
        PREFILL_PROF_TICK("attn:o_proj");

        return common_bufs->o;
    }

    /// @brief Scatters the freshly projected k/v rows into the layer's kv cache.
    void fill_kv_cache(buffer<bf16>& kv_cache, const qwen3vl_prefill_shape& s) {
        const uint32_t DK = desc->DK;
        const uint32_t DV = desc->DV;
        // the cache splits each row into two half-rows, one per core pair
        tensor_2d<bf16> k_half(this->k, DK / 2, s.L_offset * 2);
        tensor_2d<bf16> v_half(this->v, DV / 2, s.L_offset * 2);

        kv_cache.sync_from_device();

        bf16* k_03_ptr = kv_cache.data() + this->K03_offset + (size_t)s.L_begin * DK / 2;
        bf16* k_47_ptr = kv_cache.data() + this->K47_offset + (size_t)s.L_begin * DK / 2;
        bf16* v_03_ptr = kv_cache.data() + this->V03_offset + (size_t)s.L_begin * DV / 2;
        bf16* v_47_ptr = kv_cache.data() + this->V47_offset + (size_t)s.L_begin * DV / 2;

        for (int l = 0; l < s.L_in; l++) {
            memcpy(k_03_ptr, k_half[2 * l].data(), DK / 2 * sizeof(bf16));
            memcpy(k_47_ptr, k_half[2 * l + 1].data(), DK / 2 * sizeof(bf16));
            memcpy(v_03_ptr, v_half[2 * l].data(), DV / 2 * sizeof(bf16));
            memcpy(v_47_ptr, v_half[2 * l + 1].data(), DV / 2 * sizeof(bf16));

            k_03_ptr += DK / 2;
            k_47_ptr += DK / 2;
            v_03_ptr += DV / 2;
            v_47_ptr += DV / 2;
        }
    }
};

/// @brief Feed-forward half of one prefill layer: SiLU-gate, up, glu, down.
struct qwen3vl_mlp_prefill_context {
    qwen3vl_desc* desc;
    Gemm* gemm_seq_gen;
    qwen3vl_common_buffers* common_bufs;

    int L_padded_old = -1;

    npu_app gate_proj;
    npu_app up_proj;
    npu_app down_proj;

    buffer<bf16> gate;
    buffer<bf16> up;
    buffer<bf16> hid;

    qwen3vl_mlp_prefill_context(
        qwen3vl_desc* desc,
        Gemm* gemm_seq_gen,
        qwen3vl_common_buffers* common_bufs,
        npu_app_manager* gemm_app_manager
    ) : desc(desc), gemm_seq_gen(gemm_seq_gen), common_bufs(common_bufs)
    {
        this->gate_proj = gemm_app_manager->create_app();
        this->up_proj = gemm_app_manager->create_app();
        this->down_proj = gemm_app_manager->create_app();
    }

    void setup(const qwen3vl_prefill_shape& s) {
        const uint32_t D = desc->D;
        const uint32_t I = desc->intermediate_size;
        const bool has_bias = false;
        const uint32_t bias_offset = 0;

        if (L_padded_old != s.L_padded) {
            // each matrix has its own dequantized buffer, hence weight offset 0
            gemm_seq_gen->generate_seq(this->gate_proj.seq(), s.L_padded, D, I, 0,
                has_bias, Gemm::SiLU, bias_offset);
            gemm_seq_gen->generate_seq(this->up_proj.seq(), s.L_padded, D, I, 0,
                has_bias, Gemm::NO_Activation, bias_offset);
            gemm_seq_gen->generate_seq(this->down_proj.seq(), s.L_padded, I, D, 0,
                has_bias, Gemm::NO_Activation, bias_offset);
            L_padded_old = s.L_padded;
            this->gate = this->gate_proj.create_bo_buffer<bf16>((size_t)s.L_padded * I);
            this->up = this->up_proj.create_bo_buffer<bf16>((size_t)s.L_padded * I);
            this->hid = this->down_proj.create_bo_buffer<bf16>((size_t)s.L_padded * I);
        }
    }

    /// @brief Runs the mlp block; the result lands in common_bufs->o.
    buffer<bf16> forward(buffer<bf16>& gate_weights, buffer<bf16>& up_weights, buffer<bf16>& down_weights) {
        PREFILL_PROF_INIT();
        this->gate_proj(this->gate, common_bufs->x, gate_weights);
        this->gate.sync_from_device();

        this->up_proj(this->up, common_bufs->x, up_weights);
        this->up.sync_from_device();
        PREFILL_PROF_TICK("mlp:gate+up_proj");

        qwen3vl_cpu_func::_elementwise_mul(this->hid, this->up, this->gate);
        this->hid.sync_to_device();

        // this round trip looks redundant but dropping it corrupts the weights
        // the down gemm reads; carried over verbatim from the pre-refactor code
        common_bufs->o.sync_from_device();
        common_bufs->o.sync_to_device();

        this->down_proj(common_bufs->o, this->hid, down_weights);
        common_bufs->o.sync_from_device();
        PREFILL_PROF_TICK("mlp:down_proj");

        return common_bufs->o;
    }
};

/// @brief One prefill pass over the model, one layer at a time.
///
/// Owns the xclbins, the sequence generators and every scratch buffer the
/// prefill path needs; the model only hands it weights and a kv cache.
struct qwen3vl_prefill_context {
    qwen3vl_desc* desc;
    qwen3vl_common_buffers common_bufs;

    std::unique_ptr<Gemm> gemm_seq_gen;
    std::unique_ptr<Dequant> dequant_seq_gen;
    std::unique_ptr<MHA> mha_seq_gen;

    std::unique_ptr<qwen3vl_dequant_prefill_context> dequant_block;
    std::unique_ptr<qwen3vl_attn_block_prefill_context> attn_block;
    std::unique_ptr<qwen3vl_mlp_prefill_context> mlp;

    // for xclbins
    npu_app_manager* gemm_app_manager;
    npu_app_manager* dequant_app_manager;
    npu_app_manager* mha_app_manager;

    /// @brief the mha kernel refuses batches shorter than this
    static constexpr int L_MIN = 256;

    int L_padded_old = -1;

    qwen3vl_prefill_context() {}
    qwen3vl_prefill_context(
        npu_xclbin_manager* npu,
        qwen3vl_desc* model_desc,
        LM_Config& config,
        uint32_t MAX_L
    ) : desc(model_desc) {
        this->gemm_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "mm.xclbin"));
        this->dequant_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "dequant.xclbin"));
        this->mha_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "attn.xclbin"));

        this->gemm_seq_gen = std::make_unique<Gemm>(config);
        this->dequant_seq_gen = std::make_unique<Dequant>(config);
        this->mha_seq_gen = std::make_unique<MHA>(desc->mha_type, qwen3vl_desc::mha_kv_heads);

        this->dequant_block = std::make_unique<qwen3vl_dequant_prefill_context>(
            desc,
            dequant_seq_gen.get(),
            dequant_app_manager,
            !npu->is_preemption_enabled()
        );

        this->attn_block = std::make_unique<qwen3vl_attn_block_prefill_context>(
            desc,
            gemm_seq_gen.get(),
            mha_seq_gen.get(),
            &common_bufs,
            gemm_app_manager,
            mha_app_manager
        );

        this->mlp = std::make_unique<qwen3vl_mlp_prefill_context>(
            desc,
            gemm_seq_gen.get(),
            &common_bufs,
            gemm_app_manager
        );

        this->attn_block->set_max_length(MAX_L);
    }

    /// @brief chunk size the mha kernel works in; the decode path needs it too.
    uint32_t get_chunk_size() { return this->mha_seq_gen->get_chunk_size(); }

    void set_max_length(uint32_t MAX_L) { this->attn_block->set_max_length(MAX_L); }
    void set_kv_layout(size_t K03, size_t K47, size_t V03, size_t V47) {
        this->attn_block->set_kv_layout(K03, K47, V03, V47);
    }

    /// @brief Computes the row geometry of a prefill call and readies every block.
    /// @param L_in     number of tokens to prefill
    /// @param L_begin  context length already in the kv cache
    qwen3vl_prefill_shape setup(int L_in, int L_begin) {
        qwen3vl_prefill_shape s;
        s.L_in = L_in;
        s.L_begin = L_begin;

        uint32_t L_chunk = this->mha_seq_gen->get_chunk_size();
        s.L_begin_chunked = (L_begin / L_chunk) * L_chunk;
        s.L_offset = L_begin - s.L_begin_chunked;
        s.L_end_chunked = s.L_begin_chunked + (L_in + s.L_offset + L_MIN - 1) / L_MIN * L_MIN;
        s.L_padded = s.L_end_chunked - s.L_begin_chunked;

        if (L_padded_old != s.L_padded) {
            common_bufs.x = this->gemm_app_manager->create_bo_buffer<bf16>((size_t)s.L_padded * desc->D);
            common_bufs.o = this->gemm_app_manager->create_bo_buffer<bf16>((size_t)s.L_padded * desc->D);
            common_bufs.residual = buffer<bf16>((size_t)s.L_padded * desc->D);
            L_padded_old = s.L_padded;
        }
        // rows outside [L_offset, L_offset + L_in) are padding; they still take
        // part in every gemm, so leave no stale values in them.
        common_bufs.residual.memset((bf16)0.0f);

        common_bufs.x_2d.assign(common_bufs.x, desc->D, s.L_offset);
        common_bufs.residual_2d.assign(common_bufs.residual, desc->D, s.L_offset);

        this->attn_block->setup(s);
        this->mlp->setup(s);

        return s;
    }

    /// @brief Row `i` of the residual stream, where the embeddings are written.
    buffer<bf16>& residual_row(int i) { return common_bufs.residual_2d[i]; }

    /// @brief Runs one decoder layer over the whole batch, in place on the residual.
    buffer<bf16> forward(
        int layer_idx,
        const qwen3vl_prefill_shape& s,
        const std::vector<int>& input_idx,
        buffer<u8>& proj_weights,
        buffer<bf16>& rms_weights,
        buffer<bf16>& rope_rms_weights,
        buffer<bf16>& kv_cache
    ) {
        PREFILL_PROF_INIT();
        tensor_2d<bf16> rms_w(rms_weights, desc->D);
        tensor_2d<bf16> rope_rms_w(rope_rms_weights, desc->DH);

        // step 1: dequantize this layer's weights, overlapping the input norm
        this->dequant_block->start(proj_weights);
        for (int i = 0; i < s.L_in; i++) {
            qwen3vl_cpu_func::_rms_norm(common_bufs.x_2d[i], common_bufs.residual_2d[i], rms_w[0], desc->rms_eps);
        }
        common_bufs.x.sync_to_device();
        this->dequant_block->wait();
        PREFILL_PROF_TICK("layer:dequant+input_rms_norm");

        // step 2: attention
        auto attn_out = this->attn_block->forward(
            s,
            input_idx,
            this->dequant_block->qkv_weights,
            this->dequant_block->o_weights,
            rope_rms_w,
            kv_cache
        );
        qwen3vl_cpu_func::_residual_add(common_bufs.residual, attn_out, common_bufs.residual);
        PREFILL_PROF_TICK("layer:attn_block_total");

        // step 3: post-attention norm
        for (int i = 0; i < s.L_in; i++) {
            qwen3vl_cpu_func::_rms_norm(common_bufs.x_2d[i], common_bufs.residual_2d[i], rms_w[1], desc->rms_eps);
        }
        common_bufs.x.sync_to_device();
        PREFILL_PROF_TICK("layer:post_attn_rms_norm");

        // step 4: mlp
        auto mlp_out = this->mlp->forward(
            this->dequant_block->gate_weights,
            this->dequant_block->up_weights,
            this->dequant_block->down_weights
        );
        qwen3vl_cpu_func::_residual_add(common_bufs.residual, mlp_out, common_bufs.residual);
        PREFILL_PROF_TICK("layer:mlp_block_total");

        return common_bufs.residual;
    }
};

#endif
