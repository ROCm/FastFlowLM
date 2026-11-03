#ifndef __QWEN3_5VL_PREFILL_HPP__
#define __QWEN3_5VL_PREFILL_HPP__
#include "qwen3_5vl_npu_sequence.hpp"
#include "npu_sequences/GateDeltaNetPrefillRuntimeSequence.hpp"
#include "qwen3_5vl_image.hpp"
#include "modules/gemm.hpp"
#include "qwen3_5vl_model_def.hpp"
#include "utils/avx512_util.hpp"
#include "cpu_functions.hpp"

// ---------------------------------------------------------------------------
// Prefill profiling.
// Enable by compiling with -DQWEN3_5VL_PREFILL_PROFILE (all code below
// compiles to nothing otherwise, so it is zero-cost when disabled).
//
//   PREFILL_PROF_INIT();               // once at the start of a function
//   ... work ...
//   PREFILL_PROF_TICK("dequant");      // prints time since the previous tick
//   ... work ...
//   PREFILL_PROF_TICK("qkv_proj");
// ---------------------------------------------------------------------------
// For accumulating a stage that runs once per loop iteration and reporting the
// summed total once after the loop (avoids one print per iteration):
//
//   PREFILL_PROF_ACC_DECL(upgate);        // before the loop
//   for (...) {
//       PREFILL_PROF_LAP();               // reset cursor at top of iteration
//       ... work ...
//       PREFILL_PROF_ACC(upgate);         // add elapsed to the accumulator
//   }
//   PREFILL_PROF_ACC_PRINT("mlp_loop:upgate", upgate);
// #define QWEN3_5VL_PREFILL_PROFILE
#ifdef QWEN3_5VL_PREFILL_PROFILE
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
#define PREFILL_PROF_LAP() (__prof_t = time_utils::now())
#define PREFILL_PROF_ACC_DECL(var) float __prof_acc_##var = 0.0f
#define PREFILL_PROF_ACC(var)                                                  \
    do {                                                                       \
        time_utils::time_point __prof_now = time_utils::now();                 \
        __prof_acc_##var += time_utils::duration_us(__prof_t, __prof_now).first;\
        __prof_t = __prof_now;                                                 \
    } while (0)
#define PREFILL_PROF_ACC_PRINT(name, var)                                     \
    do {                                                                       \
        time_utils::time_with_unit __prof_d =                                  \
            time_utils::re_unit(time_utils::time_with_unit(__prof_acc_##var, "us"));\
        header_print("prefill-prof", name << " (sum): " << __prof_d.first      \
                                          << " " << __prof_d.second);          \
    } while (0)
#else
#define PREFILL_PROF_INIT() do {} while (0)
#define PREFILL_PROF_TICK(name) do {} while (0)
#define PREFILL_PROF_LAP() do {} while (0)
#define PREFILL_PROF_ACC_DECL(var)
#define PREFILL_PROF_ACC(var) do {} while (0)
#define PREFILL_PROF_ACC_PRINT(name, var) do {} while (0)
#endif

struct qwen3_5vl_common_buffers{
    buffer<bf16> x;
    buffer<bf16> gate;
    buffer<bf16> after_mix;

    qwen3_5vl_common_buffers(){}
};

struct qwen3_5vl_linear_block_prefill_context{
    int L_padded_old = -1;
    int L_old = -1;
    bool conv_v2 = false;
    qwen3_5vl_desc* desc;
    qwen3_5vl_npu_sequence* seq_gen;
    qwen3_5vl_common_buffers* common_bufs;

    npu_app linear_qkv_proj;
    npu_app linear_g_proj;
    npu_app linear_o_proj;
    npu_app conv1d;
    npu_app deltanet;

    buffer<bf16> qkv;
    buffer<bf16> conv_qkv;
    buffer<bf16> gate;
    buffer<float> alpha_beta;

    qwen3_5vl_linear_block_prefill_context(
        qwen3_5vl_desc* desc,
        qwen3_5vl_npu_sequence* seq_gen,
        qwen3_5vl_common_buffers* common_bufs,
        npu_app_manager* gemm_app_manager,
        npu_app_manager* conv1d_app_manager,
        npu_app_manager* gate_delta_net_manager,
        bool conv_v2
    ) : conv_v2(conv_v2), desc(desc), seq_gen(seq_gen), common_bufs(common_bufs)
    {
        this->linear_qkv_proj = gemm_app_manager->create_app();
        this->linear_g_proj = gemm_app_manager->create_app();
        this->linear_o_proj = gemm_app_manager->create_app();

        this->conv1d = conv1d_app_manager->create_app();
        this->deltanet = gate_delta_net_manager->create_app();
    }

    buffer<bf16> forward(int L, buffer<u8>& proj_weights, buffer<bf16>& ssm_weights, buffer<bf16>& cache, buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu) {
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        if (L_padded_old != L_padded){
            seq_gen->gen_dequant_mm_512(this->linear_qkv_proj.seq(), L_padded, desc->D, (desc->Linear_DQKV), desc->get_linear_qkv_offset(), (desc->conv_kernel_size - 1) * (desc->Linear_DQKV), 0);
            seq_gen->gen_dequant_mm_512(this->linear_g_proj.seq(), L_padded, desc->D, desc->Linear_DV, desc->get_linear_g_offset(), 0, 1);
            seq_gen->gen_dequant_mm_512(this->linear_o_proj.seq(), L_padded, desc->Linear_DV, desc->D, desc->get_linear_o_offset(), 0, 0);
            if (conv_v2) {
                seq_gen->gen_seq_conv1d_v2(
                    this->conv1d.seq(),
                    L_padded + (desc->conv_kernel_size - 1),  desc->Linear_DQKV , L_padded, 0
                );
            }
            else {
                seq_gen->gen_seq_conv1d(
                    this->conv1d.seq(),
                    L_padded + (desc->conv_kernel_size - 1),  desc->Linear_DQKV , L_padded, 0
                );
            }
            qkv  = this->linear_qkv_proj.create_bo_buffer<bf16>((L_padded + 3) * (desc->Linear_DQ + desc->Linear_DK + desc->Linear_DV));
            conv_qkv  = this->linear_qkv_proj.create_bo_buffer<bf16>(L_padded * (desc->Linear_DQ + desc->Linear_DK + desc->Linear_DV));
            gate = this->linear_qkv_proj.create_bo_buffer<bf16>(L_padded * desc->Linear_DV);
            alpha_beta = this->linear_qkv_proj.create_bo_buffer<float>(L_padded * desc->linear_num_v_heads * 2);
            memset(alpha_beta.data(), 0, alpha_beta.size() * sizeof(float));
        }
        if (L_old != L) {
            generate_gate_delta_net_prefill_sequence(
                *deltanet.seq(),
                8, 4,
                desc->linear_num_v_heads/desc->linear_num_qk_heads, desc->linear_num_v_heads,
                128, //TODO: configute it
                L, (L + 127) / 128 * 128,
                desc->Linear_DQ/desc->linear_num_qk_heads, desc->Linear_DK/desc->linear_num_qk_heads, desc->Linear_DV/desc->linear_num_v_heads,
                0,
                // because kvcache is used to store (conv_kernel_size-1) )* (Linear_DQ + Linear_DK + Linear_DV)
                ((desc->conv_kernel_size - 1) * (desc->Linear_DQKV) ) /2,
                (L_padded * desc->linear_num_v_heads)
            );
        }
        L_padded_old = L_padded;
        L_old = L;

        PREFILL_PROF_TICK("linear:seq_gen+alloc");

        auto qkv_run = linear_qkv_proj.create_run(qkv, common_bufs->x, proj_weights);
        qkv_run.start();

        float* alpha_ptr = alpha_beta.data();
        float* beta_ptr  = alpha_ptr + alpha_beta.size() / 2;
        cpu_func::_cpu_mm<bf16>(alpha_ptr, alpha_proj_cpu.data(), common_bufs->x.data(), L, desc->D, desc->linear_num_v_heads);
        cpu_func::_cpu_mm<bf16>(beta_ptr,  beta_proj_cpu.data(),  common_bufs->x.data(), L, desc->D, desc->linear_num_v_heads);
        // Activate over the full 128-chunk padding the deltanet consumes, not L.
        // The kernel reads ceil(L/128)*128 rows of alpha/beta, so the tail rows
        // sharing the last chunk with real tokens must be activated. Leaving them
        // raw (alpha == 0, an invalid decay) produces NaN in the chunked
        // delta-rule scan at the end of the sequence.
        int L_ssm_padded = (L + 127) / 128 * 128;
        cpu_func::_activate_a(
            alpha_ptr,
            (float*)(ssm_weights.data() + desc->get_linear_A_offset() / sizeof(bf16)),
            (float*)(ssm_weights.data() + desc->get_linear_dt_offset() / sizeof(bf16)),
            L_ssm_padded,
            desc->linear_num_v_heads
        );
        cpu_func::_activate_b(beta_ptr, L_ssm_padded, desc->linear_num_v_heads);

        alpha_beta.sync_to_device();

        qkv_run.wait();
        linear_g_proj(common_bufs->gate, common_bufs->x, proj_weights);
        PREFILL_PROF_TICK("linear:qkv+g_proj+alpha_beta");

        const size_t conv_state_bytes = (desc->conv_kernel_size - 1) * desc->Linear_DQKV * sizeof(bf16);
        qkv.sync_range_from_device(L * desc->Linear_DQKV * sizeof(bf16), conv_state_bytes);
        PREFILL_PROF_TICK("linear:qkv_sync_from");
        common_bufs->gate.sync_from_device();
        PREFILL_PROF_TICK("linear:gate_sync_from");

        // copy cache back
        cache.sync_from_device();
        memcpy(qkv.data(), cache.data(), (desc->conv_kernel_size - 1) * desc->Linear_DQKV * sizeof(bf16));
        memcpy(cache.data(), qkv.data() + L * desc->Linear_DQKV, (desc->conv_kernel_size - 1) * desc->Linear_DQKV * sizeof(bf16));
        cache.sync_to_device();
        PREFILL_PROF_TICK("linear:conv_cache_stitch");
        qkv.sync_range_to_device(0, conv_state_bytes);
        PREFILL_PROF_TICK("linear:qkv_sync_to");

        conv1d(conv_qkv, qkv, ssm_weights);
        PREFILL_PROF_TICK("linear:conv1d");

        deltanet(
            cache,
            conv_qkv,
            alpha_beta,
            common_bufs->after_mix
        );

        common_bufs->after_mix.sync_from_device();
        PREFILL_PROF_TICK("linear:deltanet");

        bf16* gate_rms_norm_weight_ptr = ssm_weights.data() + desc->get_linear_ssm_norm_offset() / sizeof(bf16);

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for(int ll = 0; ll< L; ll++){
            for(int head_idx = 0; head_idx < (int)desc->linear_num_v_heads; head_idx++){
                uint32_t offset = (ll*desc->Linear_DV) + head_idx * (desc->Linear_DV/desc->linear_num_v_heads);

                cpu_func::_gated_norm(
                    common_bufs->after_mix.data() + offset,
                    common_bufs->gate.data() + offset,
                    gate_rms_norm_weight_ptr,
                    desc->linear_v_head_dim, 1
                );
            }
        }

        common_bufs->after_mix.sync_to_device();
        PREFILL_PROF_TICK("linear:gated_norm");

        common_bufs->x.sync_to_device();
        linear_o_proj(common_bufs->x, common_bufs->after_mix, proj_weights);
        common_bufs->x.sync_from_device();
        PREFILL_PROF_TICK("linear:o_proj");

        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), L * desc->D);
        return valid;
    }
};

struct qwen3_5vl_attn_block_prefill_context{
    qwen3_5vl_desc* desc;
    qwen3_5vl_npu_sequence* seq_gen;
    qwen3_5vl_common_buffers* common_bufs;

    int L_padded_old = -1;
    int L_begin_old = -1;

    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app g_proj;
    npu_app o_proj;
    npu_app gate_proj;
    npu_app up_proj;
    npu_app down_proj;

    npu_app mha;

    buffer<bf16> q;
    buffer<bf16> k;
    buffer<bf16> v;
    buffer<bf16> g;
    buffer<bf16> g_sigmoid;

    qwen3_5vl_attn_block_prefill_context(
        qwen3_5vl_desc* desc,
        qwen3_5vl_npu_sequence* seq_gen,
        qwen3_5vl_common_buffers* common_bufs,
        npu_app_manager* gemm_app_manager,
        npu_app_manager* mha_app_manager
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs){
        this->q_proj = gemm_app_manager->create_app();
        this->k_proj = gemm_app_manager->create_app();
        this->v_proj = gemm_app_manager->create_app();
        this->g_proj = gemm_app_manager->create_app();
        this->o_proj = gemm_app_manager->create_app();

        this->mha = mha_app_manager->create_app();
    }

    buffer<bf16> forward(int L, buffer<u8>& proj_weights, buffer<bf16>& rope_rms_weights, buffer<bf16>& cache, std::vector<int>& input_idx, int L_begin) {
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        if (L_padded_old != L_padded){
            seq_gen->gen_dequant_mm_512(this->q_proj.seq(), L_padded, desc->D, desc->DQ, desc->get_attn_q_offset(), 0, 0);
            seq_gen->gen_dequant_mm_512(this->k_proj.seq(), L_padded, desc->D, desc->DK, desc->get_attn_k_offset(), 0, 0);
            seq_gen->gen_dequant_mm_512(this->v_proj.seq(), L_padded, desc->D, desc->DV, desc->get_attn_v_offset(), 0, 0);
            seq_gen->gen_dequant_mm_512(this->g_proj.seq(), L_padded, desc->D, desc->DQ, desc->get_attn_g_offset(), 0, 0);
            seq_gen->gen_dequant_mm_512(this->o_proj.seq(), L_padded, desc->DQ, desc->D, desc->get_attn_o_offset(), 0, 0);
            q  = this->q_proj.create_bo_buffer<bf16>(L_padded * desc->DQ);
            k  = this->q_proj.create_bo_buffer<bf16>(L_padded * desc->DK);
            v  = this->q_proj.create_bo_buffer<bf16>(L_padded * desc->DV);
            g  = this->q_proj.create_bo_buffer<bf16>(L_padded * desc->DQ);
            g_sigmoid  = buffer<bf16>(L_padded * desc->DQ);
        }
        if ((L_begin_old != L_begin) || (L_padded_old != L_padded)) {
            seq_gen->gen_mha_engine_seq(mha.seq(), L_begin, L_padded);
        }
        L_padded_old = L_padded;
        L_begin_old = L_begin;

        bf16* q_norm_weight = rope_rms_weights.data() + desc->RoPE_DIM;
        bf16* k_norm_weight = rope_rms_weights.data() + desc->RoPE_DIM + desc->DH;

        q_proj(q, common_bufs->x, proj_weights);

        q.sync_from_device();
        auto g_run = g_proj.create_run(g, common_bufs->x, proj_weights);
        g_run.start();
        cpu_func::_rope_rms_batch(q, desc->DQ, input_idx, L, desc->num_heads, q_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        q.sync_to_device();
        g_run.wait();
        g.sync_from_device();

        auto k_run = k_proj.create_run(k, common_bufs->x, proj_weights);
        k_run.start();
        cpu_func::_sigmoid(g_sigmoid.data(), g.data(), L * desc->DQ);
        g.sync_to_device();
        k_run.wait();
        k.sync_from_device();

        auto v_run = v_proj.create_run(v, common_bufs->x, proj_weights);
        v_run.start();
        cpu_func::_rope_rms_batch(k, desc->DK, input_idx, L, desc->num_kv_heads, k_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        k.sync_to_device();
        v_run.wait();
        v.sync_from_device();

        PREFILL_PROF_TICK("attn:qkvg_proj");

        fill_kv_cache(k, v, cache, L_begin, L);
        PREFILL_PROF_TICK("attn:fill_kv_cache");

        mha(common_bufs->after_mix, q, cache);

        common_bufs->after_mix.sync_from_device();
        PREFILL_PROF_TICK("attn:mha");

        cpu_func::_elementwise_mul(g.data(), g_sigmoid.data(), common_bufs->after_mix.data(), L * desc->DQ);
        g.sync_to_device();

        o_proj(common_bufs->x, g, proj_weights);

        common_bufs->x.sync_from_device();
        PREFILL_PROF_TICK("attn:o_proj");

        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), L * desc->D);
        // deltanet(cache, conv_qkv)
        return valid;
    }

    void fill_kv_cache(buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, buffer<bf16>& cache, int L_begin, int L){
        tensor_2d<bf16> k_cache(buffer_k, desc->DK);
        tensor_2d<bf16> v_cache(buffer_v, desc->DV);

        cache.sync_from_device();

        bf16* k_ptr = cache.data() + L_begin * desc->DK;
        bf16* v_ptr = cache.data() + L_begin * desc->DV + cache.size() / 2;

        for (int l = 0; l < L; l++){
            memcpy(k_ptr, k_cache[l].data(), desc->DK * sizeof(bf16));
            memcpy(v_ptr, v_cache[l].data(), desc->DV * sizeof(bf16));
            k_ptr += desc->DK;
            v_ptr += desc->DV;
        }
        cache.sync_to_device();
    }
};

struct qwen3_5vl_mlp_prefill_context{
    static constexpr int max_chunks = 2;

    qwen3_5vl_desc* desc;
    qwen3_5vl_npu_sequence* seq_gen;
    qwen3_5vl_common_buffers* common_bufs;

    int L_old = -1;
    int num_chunks = 1;
    int chunk_rows_begin[max_chunks];
    int chunk_rows_padded[max_chunks];

    npu_app upgate_proj[max_chunks];
    npu_app down_proj[max_chunks];

    buffer<bf16> upgate;
    buffer<bf16> hid;

    qwen3_5vl_mlp_prefill_context(
        qwen3_5vl_desc* desc,
        qwen3_5vl_npu_sequence* seq_gen,
        qwen3_5vl_common_buffers* common_bufs,
        npu_app_manager* gemm_app_manager
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs){
        for (int c = 0; c < max_chunks; c++){
            upgate_proj[c] = gemm_app_manager->create_app();
            down_proj[c] = gemm_app_manager->create_app();
        }
    }

    void setup_chunks(int L_padded) {
        const int rounds = L_padded / 512;
        num_chunks = std::min(max_chunks, rounds);
        int rounds_begin = 0;
        for (int c = 0; c < num_chunks; c++){
            int rounds_in_chunk = (rounds - rounds_begin) / (num_chunks - c);
            chunk_rows_begin[c] = rounds_begin * 512;
            chunk_rows_padded[c] = rounds_in_chunk * 512;
            rounds_begin += rounds_in_chunk;
        }

        const size_t D = desc->D;
        const size_t I = desc->intermediate_size;
        for (int c = 0; c < num_chunks; c++){
            const size_t row = chunk_rows_begin[c];
            seq_gen->gen_dequant_mm_512(upgate_proj[c].seq(), chunk_rows_padded[c], D, I * 2, desc->get_upgate_offset(), row * I * 2, 0, row * D);
            seq_gen->gen_dequant_mm_512(down_proj[c].seq(), chunk_rows_padded[c], I, D, desc->get_down_offset(), row * D, 0, row * I);
        }
    }

    void glu_chunk(int c, int L) {
        const size_t I = desc->intermediate_size;
        const int row_begin = chunk_rows_begin[c];
        const int rows = std::min(L, row_begin + chunk_rows_padded[c]) - row_begin;
        if (rows <= 0) {
            return;
        }
        upgate.sync_range_from_device(row_begin * I * 2 * sizeof(bf16), rows * I * 2 * sizeof(bf16));
        cpu_func::_exp_glu(hid.data() + row_begin * I, upgate.data() + row_begin * I * 2, rows, I, 512);
        hid.sync_range_to_device(row_begin * I * sizeof(bf16), rows * I * sizeof(bf16));
    }

    buffer<bf16> forward(int L, buffer<u8>& proj_weights, buffer<bf16>& rms_weights) {
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        if (L_old != L_padded){
            setup_chunks(L_padded);
            L_old = L_padded;
            upgate = this->upgate_proj[0].create_bo_buffer<bf16>(L_padded * desc->intermediate_size * 2);
            hid = this->upgate_proj[0].create_bo_buffer<bf16>(L_padded * desc->intermediate_size);
            memset(hid.data(), 0, hid.size() * sizeof(bf16));
            hid.sync_to_device();
        }
        PREFILL_PROF_TICK("mlp:seq_gen+alloc");

        this->upgate_proj[0](upgate, this->common_bufs->x, proj_weights);
        for (int c = 1; c < num_chunks; c++){
            auto next_run = this->upgate_proj[c].create_run(upgate, this->common_bufs->x, proj_weights);
            next_run.start();
            glu_chunk(c - 1, L);
            next_run.wait();
        }
        PREFILL_PROF_TICK("mlp:upgate_proj+glu");

        if (num_chunks == 1){
            glu_chunk(0, L);
            this->down_proj[0](common_bufs->x, hid, proj_weights);
        }
        else {
            auto first_down = this->down_proj[0].create_run(common_bufs->x, hid, proj_weights);
            first_down.start();
            glu_chunk(num_chunks - 1, L);
            first_down.wait();
            for (int c = 1; c < num_chunks; c++){
                this->down_proj[c](common_bufs->x, hid, proj_weights);
            }
        }
        PREFILL_PROF_TICK("mlp:down_proj+glu");

        common_bufs->x.sync_from_device();
        PREFILL_PROF_TICK("mlp:x_sync_from");
        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), L * desc->D);

        return valid;
    }
};

struct qwen3_5vl_prefill_context{
    qwen3_5vl_desc* desc;
    qwen3_5vl_npu_sequence* seq_gen;
    qwen3_5vl_common_buffers common_bufs;

    std::unique_ptr<qwen3_5vl_attn_block_prefill_context> attn_block;
    std::unique_ptr<qwen3_5vl_linear_block_prefill_context> linear_block;
    std::unique_ptr<qwen3_5vl_mlp_prefill_context> mlp;

    // for xclbins
    npu_app_manager* gemm_app_manager;
    npu_app_manager* mha_app_manager;
    npu_app_manager * conv1d_app_manager;
    npu_app_manager * gate_delta_net_manager;

    qwen3_5vl_prefill_context(){}
    qwen3_5vl_prefill_context(
        npu_xclbin_manager* npu,
        qwen3_5vl_desc* model_desc,
        qwen3_5vl_npu_sequence* seq_gen,
        LM_Config& config
    ) : desc(model_desc), seq_gen(seq_gen) {
        std::string mm_xclbin = utils::path_join(config.exec_path, "xclbins", config.model_name, "mm_v2.xclbin");
        if (!utils::check_file_exists(mm_xclbin)) {
            mm_xclbin = utils::path_join(config.exec_path, "xclbins", config.model_name, "mm.xclbin");
        }
        this->gemm_app_manager = npu->register_xclbin(mm_xclbin);
        std::string conv_xclbin = utils::path_join(config.exec_path, "xclbins", config.model_name, "conv_v2.xclbin");
        const bool conv_v2 = utils::check_file_exists(conv_xclbin);
        if (!conv_v2) {
            conv_xclbin = utils::path_join(config.exec_path, "xclbins", config.model_name, "conv.xclbin");
        }
        this->conv1d_app_manager = npu->register_xclbin(conv_xclbin);
        std::string gdn_xclbin = utils::path_join(config.exec_path, "xclbins", config.model_name, "GateDeltaNet_prefill_v2.xclbin");
        if (!utils::check_file_exists(gdn_xclbin)) {
            gdn_xclbin = utils::path_join(config.exec_path, "xclbins", config.model_name, "GateDeltaNet_prefill.xclbin");
        }
        this->gate_delta_net_manager = npu->register_xclbin(gdn_xclbin);
        std::string attn_xclbin = utils::path_join(config.exec_path, "xclbins", config.model_name, "attn_v2.xclbin");
        if (!utils::check_file_exists(attn_xclbin)) {
            attn_xclbin = utils::path_join(config.exec_path, "xclbins", config.model_name, "attn.xclbin");
        }
        this->mha_app_manager = npu->register_xclbin(attn_xclbin);

        // create buffers for storing dequantized weights
        // this is to ensure linear_qkv buffer has same size with attn qkvg
        size_t qkvg_size = std::max(desc->Linear_DQ + desc->Linear_DK + desc->Linear_DV, desc->DQ * 2 + desc->DK + desc->DV);
        // this is to ensure they have same out projection
        assert(desc->Linear_DV == desc->DQ);

        this->linear_block = std::make_unique<qwen3_5vl_linear_block_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            gemm_app_manager,
            conv1d_app_manager,
            gate_delta_net_manager,
            conv_v2
        );

        this->attn_block = std::make_unique<qwen3_5vl_attn_block_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            gemm_app_manager,
            mha_app_manager
        );

        this->mlp = std::make_unique<qwen3_5vl_mlp_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            gemm_app_manager
        );
    }

    buffer<bf16> forward(int layer_idx, buffer<bf16>& residual, int L, buffer<u8>& all_proj_weights, buffer<bf16>& rms_weights, buffer<bf16>& rope_weights, buffer<bf16>& kv_cache, std::vector<int>& input_idx, buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu, int L_begin = 0){
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        size_t x_size = L_padded * desc->D;

        if (x_size != common_bufs.x.size()) {
            common_bufs.x = this->gemm_app_manager->create_bo_buffer<bf16>(L_padded * desc->D);
            common_bufs.after_mix = this->gemm_app_manager->create_bo_buffer<bf16>(L_padded * desc->Linear_DV);
            common_bufs.gate = this->gemm_app_manager->create_bo_buffer<bf16>(L_padded * desc->Linear_DV);
        }
        memset(common_bufs.x.data() + (size_t)L * desc->D, 0, (size_t)(L_padded - L) * desc->D * sizeof(bf16));

        // for all layers , first step is store a residual and do input layer norm
        // this step do the rms norm and store it to x, the residual is kept
        cpu_func::_rms_norm(common_bufs.x, residual, rms_weights.data(), desc->D, L, 0, 0);
        common_bufs.x.sync_to_device();
        buffer<bf16> block_output;
        PREFILL_PROF_TICK("layer:input_rms_norm");
        if (desc->is_attn_layer(layer_idx)) {
            block_output = this->attn_block->forward(L, all_proj_weights, rope_weights, kv_cache, input_idx, L_begin);
            PREFILL_PROF_TICK("layer:attn_block_total");
        }
        else {
            block_output = this->linear_block->forward(L, all_proj_weights, rope_weights, kv_cache, alpha_proj_cpu, beta_proj_cpu);
            PREFILL_PROF_TICK("layer:linear_block_total");
        }

        assert(residual.size() == (size_t)L * desc->D);
        cpu_func::_residual_add_rms_norm(residual.data(), block_output.data(), common_bufs.x.data(), rms_weights.data() + desc->D, desc->D, L);

        common_bufs.x.sync_to_device();
        PREFILL_PROF_TICK("layer:post_attn_rms_norm");

        auto mlp_out = mlp->forward(L, all_proj_weights, rms_weights);
        PREFILL_PROF_TICK("layer:mlp_block_total");

        cpu_func::_residual_add(residual, mlp_out, residual, residual.size());
        PREFILL_PROF_TICK("layer:mlp_residual_add");

        return residual;
    }
};

#endif
