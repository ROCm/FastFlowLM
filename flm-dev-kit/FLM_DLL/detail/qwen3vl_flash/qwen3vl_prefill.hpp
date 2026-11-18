// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __QWEN3VL_PREFILL_HPP__
#define __QWEN3VL_PREFILL_HPP__
#include "modules/mha.hpp"
#include "npu_utils/flm_runtime.hpp"
#include "tensor_2d.hpp"
#include "qwen3vl_model_def.hpp"
#include "qwen3vl_flash_sequence.hpp"
#include "qwen3vl_cpu_functions.hpp"
#include <vector>
#include <iostream>

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

/// @brief Attention half of one prefill layer: q/k/v projections, per-head norm
///        plus rope, kv cache fill, mha, output projection.
struct qwen3vl_attn_block_prefill_context {
    qwen3vl_desc* desc;
    qwen3vl_flash_sequence* seq_gen;
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
    npu_app mha_preload;
    npu_app mha_engine;

    flm_rt::run preload_run;

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
        qwen3vl_flash_sequence* seq_gen,
        MHA* mha_seq_gen,
        qwen3vl_common_buffers* common_bufs,
        npu_app_manager* gemm_app_manager,
        npu_app_manager* mha_app_manager
    ) : desc(desc), seq_gen(seq_gen), mha_seq_gen(mha_seq_gen), common_bufs(common_bufs)
    {
        this->q_proj = gemm_app_manager->create_app();
        this->k_proj = gemm_app_manager->create_app();
        this->v_proj = gemm_app_manager->create_app();
        this->o_proj = gemm_app_manager->create_app();
        this->mha_preload = mha_app_manager->create_app();
        this->mha_engine = mha_app_manager->create_app();
        this->rope_values.resize(desc->DH);

        npu_sequence* seq = this->mha_preload.seq();

        seq->clear_cmds();
        seq->cmds2seq();

        this->preload_run = this->mha_preload.create_run();
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

        if (L_padded_old != s.L_padded) {
            // Release the previous set before cutting the new one. create_bo_buffer
            // returns by value, so `this->q = ...` only drops the old bo *after* the
            // new one exists: without this the peak is two full sets of scratch at
            // exactly the moment it is largest, and a longer prompt fails to allocate
            // on a machine whose npu carve-out has no headroom for the doubling.
            this->q_2d.clear();
            this->k_2d.clear();
            this->q = buffer<bf16>();
            this->k = buffer<bf16>();
            this->v = buffer<bf16>();
            this->attn_out = buffer<bf16>();
            // Byte offsets of each quantized matrix inside the layer's weight blob;
            // the array dequantizes straight into the gemm, so unlike the old path
            // there is no dequantized band to index and every projection reads the
            // same buffer at its own offset.
            seq_gen->gen_dequant_mm_512(this->k_proj.seq(), s.L_padded, D, DK,
                desc->get_attn_k_offset(), 0, /*rtp_act=*/0);
            seq_gen->gen_dequant_mm_512(this->v_proj.seq(), s.L_padded, D, DV,
                desc->get_attn_v_offset(), 0, /*rtp_act=*/0);
            seq_gen->gen_dequant_mm_512(this->q_proj.seq(), s.L_padded, D, DQ,
                desc->get_attn_q_offset(), 0, /*rtp_act=*/0);
            seq_gen->gen_dequant_mm_512(this->o_proj.seq(), s.L_padded, DQ, D,
                desc->get_attn_o_offset(), 0, /*rtp_act=*/0);
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
        buffer<u8>& proj_weights,
        tensor_2d<bf16>& rope_rms_w,
        buffer<bf16>& kv_cache
    ) {
        PREFILL_PROF_INIT();
        this->q_proj(this->q, common_bufs->x, proj_weights);

        // the k gemm runs while the host rotates q
        auto run_k = this->k_proj.create_run(this->k, common_bufs->x, proj_weights);
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
        auto run_v = this->v_proj.create_run(this->v, common_bufs->x, proj_weights);
        run_v.start();
        this->preload_run.start();
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
        PREFILL_PROF_TICK("attn:fill_kv_cache_cpu");
        kv_cache.sync_to_device();
        PREFILL_PROF_TICK("attn:fill_kv_cache_sync_to");
        flm::npu::wait_for_completion(this->preload_run, "preload_run");
        PREFILL_PROF_TICK("attn:preload_wait");

        this->mha_engine(this->attn_out, this->q, kv_cache);
        // this->attn_out.sync_from_device();
        PREFILL_PROF_TICK("attn:mha");

        this->o_proj(common_bufs->o, this->attn_out, proj_weights);
        PREFILL_PROF_TICK("attn:o_proj_dispatch");
        common_bufs->o.sync_from_device();
        PREFILL_PROF_TICK("attn:o_proj_sync");

        return common_bufs->o;
    }

    /// @brief Scatters the freshly projected k/v rows into the layer's kv cache.
    void fill_kv_cache(buffer<bf16>& kv_cache, const qwen3vl_prefill_shape& s) {
        const uint32_t DK = desc->DK;
        const uint32_t DV = desc->DV;
        // the cache splits each row into two half-rows, one per core pair
        tensor_2d<bf16> k_half(this->k, DK / 2, s.L_offset * 2);
        tensor_2d<bf16> v_half(this->v, DV / 2, s.L_offset * 2);

        {PREFILL_PROF_INIT();
        kv_cache.sync_from_device();
        PREFILL_PROF_TICK("attn:kv_sync_from");}

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
///
/// up and gate share one quantized band, interleaved in
/// qwen3vl_desc::UP_GATE_OUTDIM_REORDER-column slabs as
/// [up0 | gate0 | up1 | gate1 | ...] (see qwen3vl_desc::load_layer_weights).
/// One `j` step of the fused gemm covers exactly 512 output columns, so the two
/// projections are two strided runs over the same weight offset and the same
/// output buffer: even slabs (up, no activation) and odd slabs (gate, silu on
/// the array). The host is then left with just the multiply.
struct qwen3vl_mlp_prefill_context {
    qwen3vl_desc* desc;
    qwen3vl_flash_sequence* seq_gen;
    qwen3vl_common_buffers* common_bufs;
    npu_app_manager* gemm_app_manager;
    bool use_runlist;

    int L_padded = 0;

    npu_app gate_proj;
    npu_app up_proj;
    npu_app down_proj;

    /// @brief the model's per-layer weight buffers, borrowed; see set_weights()
    std::vector<buffer<u8>>* proj_weights_all = nullptr;
    /// @brief one pre-armed up+gate runlist per layer, built by setup()
    std::vector<flm_rt::runlist> upgate_lists;

    buffer<bf16> upgate;  ///< (L_padded, 2 * intermediate_size), interleaved slabs
    buffer<bf16> hid;     ///< (L_padded, intermediate_size), the glu result

    qwen3vl_mlp_prefill_context(
        qwen3vl_desc* desc,
        qwen3vl_flash_sequence* seq_gen,
        qwen3vl_common_buffers* common_bufs,
        npu_app_manager* gemm_app_manager,
        bool use_runlist
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs),
        gemm_app_manager(gemm_app_manager), use_runlist(use_runlist)
    {
        this->gate_proj = gemm_app_manager->create_app();
        this->up_proj = gemm_app_manager->create_app();
        this->down_proj = gemm_app_manager->create_app();
    }

    /// @brief Hands over the per-layer weight buffers so setup() can bind them
    ///        into the up/gate runlists. Call once, after they are allocated;
    ///        only the buffer handles are captured, so refilling them is fine.
    void set_weights(std::vector<buffer<u8>>& proj_weights) {
        this->proj_weights_all = &proj_weights;
        this->upgate_lists.clear();
    }

    void setup(const qwen3vl_prefill_shape& s) {
        const uint32_t D = desc->D;
        const uint32_t I = desc->intermediate_size;
        const uint32_t upgate_Dout = I * 2;

        if (this->L_padded != s.L_padded) {
            // n_stride = 2 walks every other 512-column slab; n_first picks which
            // half of the interleave. Both runs drive all eight columns, so they
            // serialize -- they only ever write disjoint slabs of `upgate`.
            seq_gen->gen_dequant_mm_512(this->up_proj.seq(), s.L_padded, D, upgate_Dout,
                desc->get_upgate_offset(), 0, /*rtp_act=*/0, /*n_stride=*/2, /*n_first=*/0);
            seq_gen->gen_dequant_mm_512(this->gate_proj.seq(), s.L_padded, D, upgate_Dout,
                desc->get_upgate_offset(), 0, /*rtp_act=*/1, /*n_stride=*/2, /*n_first=*/1);
            seq_gen->gen_dequant_mm_512(this->down_proj.seq(), s.L_padded, I, D,
                desc->get_down_offset(), 0, /*rtp_act=*/0);
            this->L_padded = s.L_padded;
            // Drop the old runs and the old buffers *before* allocating the new ones.
            // The runs hold a reference to the bos they were bound to, so clearing the
            // lists afterwards kept the previous upgate/hid -- the two largest buffers
            // on the whole prefill path -- alive across the new allocation.
            this->upgate_lists.clear();
            this->upgate = buffer<bf16>();
            this->hid = buffer<bf16>();
            this->upgate = this->up_proj.create_bo_buffer<bf16>((size_t)s.L_padded * upgate_Dout);
            this->hid = this->down_proj.create_bo_buffer<bf16>((size_t)s.L_padded * I);
        }

        // The up/gate pair is the same two dispatches for every layer and every
        // prefill call, so the runs and the runlist chaining them are built here
        // rather than in forward(): binding the buffers, elaborating the kernel
        // and allocating the command objects all move off the per-layer path,
        // which is left with one submission and one wait for the pair.
        if (this->use_runlist && this->proj_weights_all != nullptr && this->upgate_lists.empty()) {
            this->upgate_lists.reserve(this->proj_weights_all->size());
            for (buffer<u8>& w : *this->proj_weights_all) {
                flm_rt::runlist list = this->gemm_app_manager->create_runlist();
                list.add(this->up_proj.create_run(this->upgate, common_bufs->x, w));
                list.add(this->gate_proj.create_run(this->upgate, common_bufs->x, w));
                this->upgate_lists.push_back(std::move(list));
            }
        }
    }

    /// @brief Runs the mlp block; the result lands in common_bufs->o.
    buffer<bf16> forward(int layer_idx, buffer<u8>& proj_weights) {
        PREFILL_PROF_INIT();
        if (!this->upgate_lists.empty()) {
            // both dispatches are already bound; the pair goes out as one
            // submission, so the host never sits between them
            this->upgate_lists[layer_idx].execute();
            flm::npu::wait_for_completion(
                this->upgate_lists[layer_idx], "upgate_lists");
        }
        else {
            // preemption is enabled (no runlist), or the weights were never
            // handed over: fall back to two blocking dispatches
            this->up_proj(this->upgate, common_bufs->x, proj_weights);
            this->gate_proj(this->upgate, common_bufs->x, proj_weights);
        }
        this->upgate.sync_from_device();
        PREFILL_PROF_TICK("mlp:up+gate_proj");

        // the array already applied silu to the gate slabs
        static_assert(qwen3vl_desc::UP_GATE_OUTDIM_REORDER == 512,
            "one fused-gemm j step covers 512 output columns; the up/gate "
            "interleave period must match or the strided split lands mid-slab");
        // over all L_padded rows, not L_in: the real tokens start at row L_offset,
        // and the padding rows still feed the down gemm.
        qwen3vl_cpu_func::_glu_mul(this->hid.data(), this->upgate.data(),
            this->L_padded, desc->intermediate_size, qwen3vl_desc::UP_GATE_OUTDIM_REORDER);
        this->hid.sync_to_device();

        // this round trip looks redundant but dropping it corrupts the weights
        // the down gemm reads; carried over verbatim from the pre-refactor code
        common_bufs->o.sync_from_device();
        common_bufs->o.sync_to_device();

        this->down_proj(common_bufs->o, this->hid, proj_weights);
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

    /// MHA's constructor binds a reference, so the prefill-only variant of the
    /// model's mha_type has to outlive it.
    mha_type_t prefill_mha_type = mha_not_supported;
    std::unique_ptr<MHA> mha_seq_gen;

    std::unique_ptr<qwen3vl_attn_block_prefill_context> attn_block;
    std::unique_ptr<qwen3vl_mlp_prefill_context> mlp;

    // for xclbins
    npu_app_manager* gemm_app_manager;
    npu_app_manager* mha_app_manager;

    /// @brief Shortest batch every kernel on this path accepts. The mha kernel
    ///        works in chunks of `MHA::get_chunk_size()` = ct_chunk_size * 8 =
    ///        128, so it has never been the binding constraint -- an earlier
    ///        version of this comment claimed 256 and was wrong. The fused
    ///        dequant-matmul consumes four desired_m-row blocks per i step, and
    ///        the overlay this engine uses has desired_m = 32, so it needs 128
    ///        too. Its m = 64 predecessor forced 256 here, which cost a ~100
    ///        token prompt 156 rows of pure padding through every gemm.
    static constexpr int L_MIN = 128;

    int L_padded_old = -1;

    qwen3vl_prefill_context() {}
    qwen3vl_prefill_context(
        npu_xclbin_manager* npu,
        qwen3vl_desc* model_desc,
        LM_Config& config,
        qwen3vl_flash_sequence* seq_gen,
        uint32_t MAX_L
    ) : desc(model_desc) {
        // Both halves of prefill live in one overlay: columns 0-5 are the
        // 6-column dequant+mm array, columns 6-7 a single attention CU.  The two
        // used to be separate xclbins, and reconfiguring the array between a
        // projection and the attention that follows it cost ~5 ms per layer --
        // about a quarter of the 256x256-image prefill.  Registering one xclbin
        // for both app managers removes every swap.
        //
        // The overlay is built for GQA 1:4, and gen_dequant_mm_512 emits the
        // 6-column geometry unconditionally, so the d128_q2 checkpoints have no
        // working prefill path here until a matching overlay is built.
        if (desc->mha_type != mha_d128_q4) {
            throw std::runtime_error("qwen3vl prefill: fused_prefill.xclbin is GQA 1:4 only");
        }
        npu_app_manager* prefill_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "fused_prefill.xclbin"));
        this->gemm_app_manager = prefill_app_manager;
        this->mha_app_manager = prefill_app_manager;

        this->prefill_mha_type = mha_d128_q4_1cu;
        this->mha_seq_gen = std::make_unique<MHA>(this->prefill_mha_type, qwen3vl_desc::mha_kv_heads);

        this->attn_block = std::make_unique<qwen3vl_attn_block_prefill_context>(
            desc,
            seq_gen,
            mha_seq_gen.get(),
            &common_bufs,
            gemm_app_manager,
            mha_app_manager
        );

        this->mlp = std::make_unique<qwen3vl_mlp_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            gemm_app_manager,
            !npu->is_preemption_enabled()
        );

        this->attn_block->set_max_length(MAX_L);
    }

    /// The chunk size mha_d128_q4_1cu is built for: ct_chunk_size 16, times the
    /// 8 tokens a compute tile retires per round.  Only used as a fallback; the
    /// live value comes from MHA so the two cannot silently drift.
    static constexpr uint32_t L_CHUNK_FALLBACK = 128;

    /// @brief chunk size the mha kernel works in; the decode path needs it too.
    ///
    /// MHA returns 0 for an mha_type_t its switch statements do not list, which
    /// is what a libmha.so older than this engine looks like -- and 0 here used
    /// to reach `L_begin / L_chunk` in setup() as a divide-by-zero, killing the
    /// process with SIGFPE at the first prompt.  Fall back to the constant the
    /// overlay is built for and say so, so a lib mismatch degrades into a loud
    /// warning instead of a crash.
    uint32_t get_chunk_size() {
        const uint32_t chunk = this->mha_seq_gen->get_chunk_size();
        if (chunk != 0) return chunk;
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[qwen3vl] warning: MHA reported chunk size 0 for this attention type; "
                         "falling back to " << L_CHUNK_FALLBACK
                      << ".  libmha.so is probably older than libqwen3vl_flash.so." << std::endl;
        }
        return L_CHUNK_FALLBACK;
    }

    void set_max_length(uint32_t MAX_L) { this->attn_block->set_max_length(MAX_L); }
    void set_kv_layout(size_t K03, size_t K47, size_t V03, size_t V47) {
        this->attn_block->set_kv_layout(K03, K47, V03, V47);
    }
    /// @brief Lends the per-layer weights to the blocks that pre-arm runlists.
    void set_weights(std::vector<buffer<u8>>& proj_weights) {
        this->mlp->set_weights(proj_weights);
    }

    /// @brief Computes the row geometry of a prefill call and readies every block.
    /// @param L_in     number of tokens to prefill
    /// @param L_begin  context length already in the kv cache
    qwen3vl_prefill_shape setup(int L_in, int L_begin) {
        qwen3vl_prefill_shape s;
        s.L_in = L_in;
        s.L_begin = L_begin;

        uint32_t L_chunk = this->get_chunk_size();
        s.L_begin_chunked = (L_begin / L_chunk) * L_chunk;
        s.L_offset = L_begin - s.L_begin_chunked;
        s.L_end_chunked = s.L_begin_chunked + (L_in + s.L_offset + L_MIN - 1) / L_MIN * L_MIN;
        s.L_padded = s.L_end_chunked - s.L_begin_chunked;

        if (L_padded_old != s.L_padded) {
            // same rule as the blocks below: release, then allocate, so a regrow never
            // needs both sets resident at once. The mlp's up/gate runs are bound to
            // common_bufs.x, so they have to go first or they pin the old one; mlp->setup()
            // rebuilds them a few lines down.
            this->mlp->upgate_lists.clear();
            common_bufs.x_2d.clear();
            common_bufs.residual_2d.clear();
            common_bufs.x = buffer<bf16>();
            common_bufs.o = buffer<bf16>();
            common_bufs.residual = buffer<bf16>();
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

        // step 1: input norm (the array dequantizes weights inside each gemm now,
        // so there is no separate dequant pass to overlap it with)
        for (int i = 0; i < s.L_in; i++) {
            qwen3vl_cpu_func::_rms_norm(common_bufs.x_2d[i], common_bufs.residual_2d[i], rms_w[0], desc->rms_eps);
        }
        common_bufs.x.sync_to_device();
        PREFILL_PROF_TICK("layer:input_rms_norm");

        // step 2: attention
        auto attn_out = this->attn_block->forward(
            s,
            input_idx,
            proj_weights,
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
        auto mlp_out = this->mlp->forward(layer_idx, proj_weights);
        qwen3vl_cpu_func::_residual_add(common_bufs.residual, mlp_out, common_bufs.residual);
        PREFILL_PROF_TICK("layer:mlp_block_total");

        return common_bufs.residual;
    }
};

#endif
