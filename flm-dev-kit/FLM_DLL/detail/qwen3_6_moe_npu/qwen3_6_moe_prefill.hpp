// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __QWEN3_6_MOE_PREFILL_HPP__
#define __QWEN3_6_MOE_PREFILL_HPP__
#include "qwen3_6_moe_npu_sequence.hpp"
#include "npu_sequences/GateDeltaNetPrefillRuntimeSequence.hpp"
#include "qwen3_6_moe_image.hpp"
#include "modules/gemm.hpp"
#include "qwen3_6_moe_model_def.hpp"
#include "utils/avx512_util.hpp"
#include "cpu_functions.hpp"

// ---------------------------------------------------------------------------
// Prefill profiling.
// Enable by compiling with -DQWEN3_6_MOE_PREFILL_PROFILE (all code below
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
//   PREFILL_PROF_ACC_PRINT("moe_loop:upgate", upgate);
// #define QWEN3_6_MOE_PREFILL_PROFILE
#ifdef QWEN3_6_MOE_PREFILL_PROFILE
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

struct qwen3_6_moe_common_buffers{
    buffer<bf16> x;
    buffer<bf16> gate;
    buffer<bf16> after_mix;

    qwen3_6_moe_common_buffers(){}
};

struct qwen3_6_moe_linear_block_prefill_context{
    int L_padded_old = -1;
    int L_old = -1;
    qwen3_6_moe_desc* desc;
    qwen3_6_moe_npu_sequence* seq_gen;
    qwen3_6_moe_common_buffers* common_bufs;

    npu_app linear_qkv_proj;
    npu_app linear_g_proj;
    npu_app linear_o_proj;
    npu_app conv1d;
    npu_app deltanet;

    buffer<bf16> qkv;
    buffer<bf16> conv_qkv;
    buffer<bf16> gate;
    buffer<float> alpha_beta;

    qwen3_6_moe_linear_block_prefill_context(
        qwen3_6_moe_desc* desc,
        qwen3_6_moe_npu_sequence* seq_gen,
        qwen3_6_moe_common_buffers* common_bufs,
        npu_app_manager* gemm_app_manager,
        npu_app_manager* conv1d_app_manager,
        npu_app_manager* gate_delta_net_manager
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs)
    {
        this->linear_qkv_proj = gemm_app_manager->create_app();
        this->linear_g_proj = gemm_app_manager->create_app();
        this->linear_o_proj = gemm_app_manager->create_app();

        this->conv1d = conv1d_app_manager->create_app();
        this->deltanet = gate_delta_net_manager->create_app();
    }

    buffer<bf16> forward(int L, buffer<u8>& proj_weights, buffer<bf16>& ssm_weights, buffer<bf16>& cache) {
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        if (L_padded_old != L_padded){
            seq_gen->gen_dequant_mm_512(this->linear_qkv_proj.seq(), L_padded, desc->D, (desc->Linear_DQ + desc->Linear_DK + desc->Linear_DV), desc->get_linear_qkv_offset(), (size_t)(desc->conv_kernel_size - 1) * (desc->Linear_DQKV), 0, flm_q4k);
            seq_gen->gen_dequant_mm_512(this->linear_g_proj.seq(), L_padded, desc->D, desc->Linear_DV, desc->get_linear_g_offset(), 0, 1, flm_q4k);
            seq_gen->gen_dequant_mm_512(this->linear_o_proj.seq(), L_padded, desc->Linear_DV, desc->D, desc->get_linear_o_offset(), 0, 0, flm_q4k);
            seq_gen->gen_seq_conv1d(
                this->conv1d.seq(),
                L_padded + (desc->conv_kernel_size - 1),  desc->Linear_DQKV , L_padded, 0
            );
            qkv  = this->linear_qkv_proj.create_bo_buffer<bf16>((size_t)(L_padded + 3) * (desc->Linear_DQ + desc->Linear_DK + desc->Linear_DV));
            conv_qkv  = this->linear_qkv_proj.create_bo_buffer<bf16>((size_t)L_padded * (desc->Linear_DQ + desc->Linear_DK + desc->Linear_DV));
            gate = this->linear_qkv_proj.create_bo_buffer<bf16>((size_t)L_padded * desc->Linear_DV);
            alpha_beta = this->linear_qkv_proj.create_bo_buffer<float>((size_t)L_padded * desc->linear_num_v_heads * 2);
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

        linear_qkv_proj(qkv, common_bufs->x, proj_weights);
        linear_g_proj(common_bufs->gate, common_bufs->x, proj_weights);

        qkv.sync_from_device();
        common_bufs->gate.sync_from_device();
        PREFILL_PROF_TICK("linear:qkv+g_proj");

        // copy cache back
        cache.sync_from_device();
        memcpy(qkv.data(), cache.data(), (size_t)(desc->conv_kernel_size - 1) * desc->Linear_DQKV * sizeof(bf16));
        memcpy(cache.data(), qkv.data() + L * desc->Linear_DQKV, (size_t)(desc->conv_kernel_size - 1) * desc->Linear_DQKV * sizeof(bf16));
        cache.sync_to_device();
        qkv.sync_to_device();

        conv1d(conv_qkv, qkv, ssm_weights);

        conv_qkv.sync_from_device();
        PREFILL_PROF_TICK("linear:conv1d");

        float* alpha_ptr = alpha_beta.data();
        float* beta_ptr  = alpha_ptr + alpha_beta.size() / 2;
        cpu_func::_cpu_mm_col<bf16>(alpha_ptr, ssm_weights.data() + desc->get_linear_alpha_offset() / sizeof(bf16), common_bufs->x.data(), L, desc->D, desc->linear_num_v_heads);
        cpu_func::_cpu_mm_col<bf16>(beta_ptr,  ssm_weights.data() + desc->get_linear_beta_offset() / sizeof(bf16), common_bufs->x.data(), L, desc->D, desc->linear_num_v_heads);
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
        PREFILL_PROF_TICK("linear:alpha_beta");

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
        linear_o_proj(common_bufs->x, common_bufs->after_mix, ssm_weights);
        common_bufs->x.sync_from_device();
        PREFILL_PROF_TICK("linear:o_proj");

        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);
        return valid;
    }
};

struct qwen3_6_moe_attn_block_prefill_context{
    qwen3_6_moe_desc* desc;
    qwen3_6_moe_npu_sequence* seq_gen;
    qwen3_6_moe_common_buffers* common_bufs;

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

    qwen3_6_moe_attn_block_prefill_context(
        qwen3_6_moe_desc* desc,
        qwen3_6_moe_npu_sequence* seq_gen,
        qwen3_6_moe_common_buffers* common_bufs,
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
            seq_gen->gen_dequant_mm_512(this->q_proj.seq(), L_padded, desc->D, desc->DQ, desc->get_attn_q_offset(), 0, 0, flm_q4k);
            seq_gen->gen_dequant_mm_512(this->k_proj.seq(), L_padded, desc->D, desc->DK, desc->get_attn_k_offset(), 0, 0, flm_q4k);
            seq_gen->gen_dequant_mm_512(this->v_proj.seq(), L_padded, desc->D, desc->DV, desc->get_attn_v_offset(), 0, 0, flm_q4k);
            seq_gen->gen_dequant_mm_512(this->g_proj.seq(), L_padded, desc->D, desc->DQ, desc->get_attn_g_offset(), 0, 0, flm_q4k);
            seq_gen->gen_dequant_mm_512(this->o_proj.seq(), L_padded, desc->DQ, desc->D, desc->get_attn_o_offset(), 0, 0, flm_q4k);
            q  = this->q_proj.create_bo_buffer<bf16>((size_t)L_padded * desc->DQ);
            k  = this->q_proj.create_bo_buffer<bf16>((size_t)L_padded * desc->DK);
            v  = this->q_proj.create_bo_buffer<bf16>((size_t)L_padded * desc->DV);
            g  = this->q_proj.create_bo_buffer<bf16>((size_t)L_padded * desc->DQ);
            g_sigmoid  = buffer<bf16>((size_t)L_padded * desc->DQ);
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
        flm::npu::wait_for_completion(g_run, "g_run");
        g.sync_from_device();

        auto k_run = k_proj.create_run(k, common_bufs->x, proj_weights);
        k_run.start();
        cpu_func::_sigmoid(g_sigmoid.data(), g.data(), L * desc->DQ);
        g.sync_to_device();
        flm::npu::wait_for_completion(k_run, "k_run");
        k.sync_from_device();

        auto v_run = v_proj.create_run(v, common_bufs->x, proj_weights);
        v_run.start();
        cpu_func::_rope_rms_batch(k, desc->DK, input_idx, L, desc->num_kv_heads, k_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        k.sync_to_device();
        flm::npu::wait_for_completion(v_run, "v_run");
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

        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);
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

struct qwen3_6_moe_expert_prefill_context{
    qwen3_6_moe_desc* desc;
    qwen3_6_moe_npu_sequence* seq_gen;
    qwen3_6_moe_common_buffers* common_bufs;

    int L_old = -1;

    // Regeneration-skip (per ping-pong slot): after the shared-executable
    // collapse the routed-expert MM TXN depends only on the padded token count
    // l_pad -- the per-expert weight base is supplied at dispatch as a bound_arg
    // byte offset, not baked into the sequence. Re-running gen_dequant_mm for an
    // app whose last l_pad is unchanged would rebuild a bit-identical TXN and bump
    // its sequence version, forcing a needless executable rebuild under #196.
    // Cache the last l_pad per app and skip when it matches. Slot-indexed scalars
    // (not a shared map) so the two gens in prep_expert's omp sections write
    // disjoint memory and never race.
    int _last_lp_up[2] = {-1, -1};
    int _last_lp_dn[2] = {-1, -1};

    // Regenerate a collapsed expert MM sequence only when l_pad changed for `app`.
    // weight_offset is baked as 0 (expert-independent); the per-expert weight base
    // is applied at create_run as a bound_arg byte offset.
    inline void _gen_expert_mm_if_needed(npu_app& app, int& last_lp, int l_pad,
                                         uint32_t Din, uint32_t Dout) {
        if (last_lp == l_pad) return;
        seq_gen->gen_dequant_mm(app.seq(), l_pad, Din, Dout, /*weight_offset=*/0, /*rtp_act=*/0, flm_q4k);
        last_lp = l_pad;
    }

    npu_app shexp_up_proj;
    npu_app shexp_gate_proj;
    npu_app shexp_down_proj;

    npu_app exp_upgate;
    npu_app exp_down;
    // Second ping-pong app set so expert i+1's host prep (seq-gen + fill_x) can be
    // built while expert i's dispatch is still in flight. Each app owns its own
    // ctrl_seq, so regenerating slot (i+1)&1 never disturbs the in-flight run on
    // slot i&1 (its TXN is already captured at create_run).
    npu_app exp_upgate2;
    npu_app exp_down2;

    flm_rt::runlist shexp_upgate;

    buffer<float> experts_score;
    buffer<float> shared_expert_score;
    buffer<float> expert_output;
    buffer<bf16> up;
    buffer<bf16> gate;
    buffer<bf16> down;
    buffer<bf16> cpu_x_buf;

    qwen3_6_moe_expert_prefill_context(
        qwen3_6_moe_desc* desc,
        qwen3_6_moe_npu_sequence* seq_gen,
        qwen3_6_moe_common_buffers* common_bufs,
        npu_app_manager* gemm_app_manager,
        npu_app_manager* dequant_mm_app_manager
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs){
        shexp_up_proj = gemm_app_manager->create_app();
        shexp_gate_proj = gemm_app_manager->create_app();
        shexp_down_proj = gemm_app_manager->create_app();

        exp_upgate = dequant_mm_app_manager->create_app();
        exp_down = dequant_mm_app_manager->create_app();
        exp_upgate2 = dequant_mm_app_manager->create_app();
        exp_down2 = dequant_mm_app_manager->create_app();

        shexp_upgate = gemm_app_manager->create_runlist();
    }

    buffer<bf16> forward(int layer_idx, int L, buffer<u8>& proj_weights, buffer<bf16>& rms_weights) {
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        if (L_old != L_padded){
            seq_gen->gen_dequant_mm_512(this->shexp_up_proj.seq(), L_padded, desc->D, desc->intermediate_size, desc->get_shexp_upgate_offset(), 0, 0, flm_q4k);
            seq_gen->gen_dequant_mm_512(this->shexp_gate_proj.seq(), L_padded, desc->D, desc->intermediate_size, desc->get_shexp_gate_offset(), 0, 1, flm_q4k);
            seq_gen->gen_dequant_mm_512(this->shexp_down_proj.seq(), L_padded, desc->intermediate_size, desc->D, desc->get_shexp_down_offset(), 0, 0, flm_q4k);
            L_old = L_padded;
            experts_score = buffer<float>((size_t)L_padded * desc->num_experts);
            shared_expert_score = buffer<float>(L_padded);
            expert_output = buffer<float>((size_t)L_padded * desc->D);
            up = this->shexp_up_proj.create_bo_buffer<bf16>((size_t)L_padded * desc->intermediate_size);
            gate = this->shexp_up_proj.create_bo_buffer<bf16>((size_t)L_padded * desc->intermediate_size);
            down = this->shexp_up_proj.create_bo_buffer<bf16>((size_t)L_padded * desc->D);
            cpu_x_buf = buffer<bf16>((size_t)L_padded * desc->D);
            memset(cpu_x_buf.data(), 0, cpu_x_buf.size() * sizeof(bf16));
            memset(expert_output.data(), 0, expert_output.size() * sizeof(float));
        }
        PREFILL_PROF_TICK("moe:seq_gen+alloc");
        auto up_run = this->shexp_up_proj.create_run(up, this->common_bufs->x, proj_weights);
        auto gate_run = this->shexp_gate_proj.create_run(gate, this->common_bufs->x, proj_weights);
        memcpy(cpu_x_buf.data(), common_bufs->x.data(), (size_t)L * desc->D * sizeof(bf16));

        shexp_upgate.reset();
        shexp_upgate.add(up_run);
        shexp_upgate.add(gate_run);

        shexp_upgate.execute();

        cpu_func::_cpu_mm_col<bf16>(experts_score.data(), (bf16*)(rms_weights.data() + desc->get_exp_score_offset() / sizeof(rms_weights[0])), cpu_x_buf.data(), L, desc->D, desc->num_experts);
        cpu_func::_cpu_mm<bf16>(shared_expert_score.data(), (bf16*)(rms_weights.data() + desc->get_shexp_score_offset() / sizeof(rms_weights[0])), cpu_x_buf.data(), L, desc->D, 1);
        cpu_func::_sigmoid(shared_expert_score.data(), shared_expert_score.data(), L);

        flm::npu::wait_for_completion(shexp_upgate, "shexp_upgate");
        up.sync_from_device();
        gate.sync_from_device();

        cpu_func::_elementwise_mul(up.data(), up.data(), gate.data(), L * desc->intermediate_size);

        up.sync_to_device();

        PREFILL_PROF_TICK("moe:scores");

        auto sh_down_run = this->shexp_down_proj.create_run(down, up, proj_weights);
        sh_down_run.start();
        auto routing_info = cpu_func::_moe_gate(experts_score.data(), L, desc->num_experts, desc->num_activate_experts);
        auto moe_tasks    = cpu_func::_pack_moe_tasks(routing_info, desc->num_experts);
        flm::npu::wait_for_completion(sh_down_run, "sh_down_run");

        down.sync_from_device();

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for(int i = 0; i < L; i++){
            cpu_func::_elementwise_mul(expert_output.data() + i * desc->D, down.data() + i * desc->D, shared_expert_score[i], desc->D);
        }
        PREFILL_PROF_TICK("moe:shared_expert");
        PREFILL_PROF_TICK("moe:gate+pack");

        // find the expert with maximum tokens, pads to multiple of 128;
        int max_num_tokens = 0;
        for (int e = 0; e < desc->num_experts; e++)
            max_num_tokens = std::max(max_num_tokens, routing_info.count(e));
        max_num_tokens = (max_num_tokens + 255) / 256 * 256;

        // Double-buffered I/O so the host can prep expert i+1 (and scatter expert
        // i-1) while expert i's dispatch is still executing on the NPU. Buffers are
        // sized to the layer's largest expert; padding rows carry stale/zero data
        // that maps to padding output rows which are never scattered, so the
        // pipelined result is bit-identical to the serial one.
        buffer<bf16> exp_x_temp[2];
        buffer<bf16> exp_upgate_buf[2];
        buffer<bf16> exp_hid_buf[2];
        buffer<bf16> exp_down_buf[2];
        for (int s = 0; s < 2; s++) {
            exp_x_temp[s]     = this->exp_upgate.create_bo_buffer<bf16>((size_t)max_num_tokens * desc->D);
            exp_upgate_buf[s] = this->exp_upgate.create_bo_buffer<bf16>((size_t)max_num_tokens * desc->intermediate_size * 2);
            exp_hid_buf[s]    = this->exp_upgate.create_bo_buffer<bf16>((size_t)max_num_tokens * desc->intermediate_size);
            exp_down_buf[s]   = this->exp_upgate.create_bo_buffer<bf16>((size_t)max_num_tokens * desc->D);
            memset(exp_x_temp[s].data(), 0, exp_x_temp[s].size() * sizeof(bf16));
            memset(exp_hid_buf[s].data(), 0, exp_hid_buf[s].size() * sizeof(bf16));
        }

        PREFILL_PROF_ACC_DECL(seqgen_fillx);
        PREFILL_PROF_ACC_DECL(upgate_mm);
        PREFILL_PROF_ACC_DECL(exp_glu);
        PREFILL_PROF_ACC_DECL(down_mm);
        PREFILL_PROF_ACC_DECL(scatter);

        const int n_tasks = (int)moe_tasks.size();

        // Host-only prep for one expert into ping-pong slot `slot`: regenerate the
        // two dequant-MM sequences (on that slot's own app set) and gather this
        // expert's tokens into exp_x_temp[slot]. Safe to run concurrently with the
        // other slot's in-flight dispatch: it touches only this slot's apps/buffers,
        // and the in-flight run already captured its TXN at create_run.
        auto prep_expert = [&](int ec, int slot) {
            cpu_func::moe_task& t = moe_tasks[ec];
            int l_pad = ((int)t.token_ids.size() + 255) / 256 * 256;
            npu_app& up_app = slot ? this->exp_upgate2 : this->exp_upgate;
            npu_app& dn_app = slot ? this->exp_down2   : this->exp_down;
            #pragma omp parallel sections num_threads(3)
            {
                #pragma omp section
                // Collapse: bake weight_offset 0 (expert-independent); skip when
                // this slot's last l_pad is unchanged (bit-identical TXN).
                this->_gen_expert_mm_if_needed(up_app, _last_lp_up[slot], l_pad, desc->D, 2 * desc->intermediate_size);
                #pragma omp section
                this->_gen_expert_mm_if_needed(dn_app, _last_lp_dn[slot], l_pad, desc->intermediate_size, desc->D);
                #pragma omp section
                {
                    bf16* xb = exp_x_temp[slot].data();
                    for (int tk = 0; tk < (int)t.token_ids.size(); tk++)
                        memcpy(xb + tk * desc->D, cpu_x_buf.data() + t.token_ids[tk] * desc->D, desc->D * sizeof(bf16));
                }
            }
        };

        // Scatter one expert's down output (held in slot `slot`) into expert_output.
        auto scatter_expert = [&](int ec, int slot) {
            cpu_func::moe_task& t = moe_tasks[ec];
            #pragma omp parallel for num_threads(max_prefill_thread) schedule(dynamic)
            for (int tk = 0; tk < (int)t.token_ids.size(); tk++) {
                int token_id = t.token_ids[tk];
                cpu_func::_elementwise_mac(expert_output.data() + token_id * desc->D, exp_down_buf[slot].data() + tk * desc->D, t.weights[tk], desc->D);
            }
        };

        if (n_tasks > 0) prep_expert(0, 0);   // prologue: prep expert 0 into slot 0
        for (int exp_count = 0; exp_count < n_tasks; exp_count++){
            PREFILL_PROF_LAP();
            const int s = exp_count & 1;
            cpu_func::moe_task& task = moe_tasks[exp_count];
            int l_this_expert = task.token_ids.size();
            npu_app& up_app = s ? this->exp_upgate2 : this->exp_upgate;
            npu_app& dn_app = s ? this->exp_down2   : this->exp_down;

            // --- upgate dispatch for expert exp_count (slot s) ---
            // Full H2D: the device kernel reads the whole max_num_tokens-padded
            // input and relies on the zero padding (ranging the H2D leaves device
            // padding uninitialized and perturbs valid output), so flush all of it.
            exp_x_temp[s].sync_to_device();
            // Collapse: the TXN is expert-independent; bind proj_weights at this
            // expert's upgate base offset so the shared executable reads the right
            // weights.
            auto up_run = up_app.create_run(exp_upgate_buf[s], exp_x_temp[s], bound_arg{&proj_weights, desc->get_upgate_exp_offset(task.expert_id)});
            up_run.start();

            // --- overlap window 1: NPU runs upgate(exp_count); host scatters the
            // PREVIOUS expert (down output lives in the other slot). The NEXT
            // expert's prep is deferred to the down window below: the per-expert
            // upgate GEMM is small, so loading it with scatter+prep overflowed the
            // GEMM and left the NPU idle; prep now overlaps the down GEMM instead,
            // which otherwise ran with zero host overlap. Both touch only the
            // opposite slot, so they never race the in-flight run.
            if (exp_count > 0) scatter_expert(exp_count - 1, (exp_count - 1) & 1);
            // This window overlaps the previous expert's scatter with upgate(i), so
            // its cost is bucketed as scatter (prep/seqgen+fill_x now runs in the
            // down window below and is bucketed there).
            PREFILL_PROF_ACC(scatter);

            // in-flight depth stays <= 1 (wedge-safe)
            flm::npu::wait_for_completion(
                up_run,
                "qwen3.6-moe layer " + std::to_string(layer_idx) +
                " expert " + std::to_string(task.expert_id) + " up");
            // _exp_glu reads only the l_this_expert valid rows, so invalidate just
            // those, not the whole max_num_tokens-sized allocation. The
            // un-invalidated tail is never read before the next device write, so the
            // result is bit-identical. Use the *_exact entry point: the plain
            // sync_from_device(elems) drops its range on the HRX coherence path
            // (whole-buffer invalidate), so the intended ranging only took effect on
            // XRT; _exact ranges on both backends.
            exp_upgate_buf[s].sync_range_from_device_exact(0, (size_t)l_this_expert * desc->intermediate_size * 2 * sizeof(bf16));
            PREFILL_PROF_ACC(upgate_mm);

            cpu_func::_exp_glu(exp_hid_buf[s].data(), exp_upgate_buf[s].data(), l_this_expert, desc->intermediate_size, 128);
            // Full H2D: the down kernel relies on the zero padding (see upgate H2D).
            exp_hid_buf[s].sync_to_device();
            PREFILL_PROF_ACC(exp_glu);

            // --- down dispatch for expert exp_count (slot s) ---
            // Collapse: bind proj_weights at this expert's down base offset.
            auto dn_run = dn_app.create_run(exp_down_buf[s], exp_hid_buf[s], bound_arg{&proj_weights, desc->get_down_exp_offset(task.expert_id)});
            dn_run.start();
            // --- overlap window 2: NPU runs down(exp_count); host preps the NEXT
            // expert into the opposite slot (never races the in-flight down run). ---
            if (exp_count + 1 < n_tasks) prep_expert(exp_count + 1, (exp_count + 1) & 1);
            // prep_expert is the seqgen + fill_x work for the NEXT expert, overlapped
            // with down(i); bucket it as seqgen+fill_x so that name still measures it.
            PREFILL_PROF_ACC(seqgen_fillx);
            flm::npu::wait_for_completion(
                dn_run,
                "qwen3.6-moe layer " + std::to_string(layer_idx) +
                " expert " + std::to_string(task.expert_id) + " down");
            // Ranged invalidate: scatter reads only the l_this_expert rows. *_exact
            // so the range really applies on the HRX path too (see upgate above).
            exp_down_buf[s].sync_range_from_device_exact(0, (size_t)l_this_expert * desc->D * sizeof(bf16));
            // down_mm now measures only the down wait + invalidate (prep moved to the
            // seqgen+fill_x bucket above).
            PREFILL_PROF_ACC(down_mm);
            // scatter for this expert is deferred to the next iteration's overlap
            // window (or the epilogue below for the final expert).
        }
        // epilogue: scatter the final expert, left pending by the loop above.
        if (n_tasks > 0) scatter_expert(n_tasks - 1, (n_tasks - 1) & 1);
        PREFILL_PROF_ACC(scatter);
#ifdef QWEN3_6_MOE_PREFILL_PROFILE
        header_print("prefill-prof", "moe_loop: num_experts_active=" << moe_tasks.size());
#endif
        PREFILL_PROF_ACC_PRINT("moe_loop:seqgen+fill_x", seqgen_fillx);
        PREFILL_PROF_ACC_PRINT("moe_loop:upgate_mm", upgate_mm);
        PREFILL_PROF_ACC_PRINT("moe_loop:exp_glu", exp_glu);
        PREFILL_PROF_ACC_PRINT("moe_loop:down_mm", down_mm);
        PREFILL_PROF_ACC_PRINT("moe_loop:scatter", scatter);

        cpu_func::_f32_to_bf16(common_bufs->x.data(), expert_output.data(), L * desc->D);
        common_bufs->x.sync_to_device();
        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);

        return valid;
    }
};

struct qwen3_6_moe_prefill_context{
    qwen3_6_moe_desc* desc;
    qwen3_6_moe_npu_sequence* seq_gen;
    qwen3_6_moe_common_buffers common_bufs;

    std::unique_ptr<qwen3_6_moe_attn_block_prefill_context> attn_block;
    std::unique_ptr<qwen3_6_moe_linear_block_prefill_context> linear_block;
    std::unique_ptr<qwen3_6_moe_expert_prefill_context> moe_mlp;

    // for xclbins
    npu_app_manager* gemm_app_manager;
    npu_app_manager* mha_app_manager;
    npu_app_manager * conv1d_app_manager;
    npu_app_manager * gate_delta_net_manager;
    npu_app_manager * dequant_mm_app_manager;

    qwen3_6_moe_prefill_context(){}
    qwen3_6_moe_prefill_context(
        npu_xclbin_manager* npu,
        qwen3_6_moe_desc* model_desc,
        qwen3_6_moe_npu_sequence* seq_gen,
        LM_Config& config
    ) : desc(model_desc), seq_gen(seq_gen) {
        this->gemm_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "mm.xclbin"));
        this->conv1d_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "conv.xclbin"));
        this->gate_delta_net_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "GateDeltaNet_prefill.xclbin"));
        this->mha_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "attn.xclbin"));
        this->dequant_mm_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "dequant_mm.xclbin"));

        // create buffers for storing dequantized weights
        // this is to ensure linear_qkv buffer has same size with attn qkvg
        size_t qkvg_size = std::max(desc->Linear_DQ + desc->Linear_DK + desc->Linear_DV, desc->DQ * 2 + desc->DK + desc->DV);
        // this is to ensure they have same out projection
        assert(desc->Linear_DV == desc->DQ);

        this->linear_block = std::make_unique<qwen3_6_moe_linear_block_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            gemm_app_manager,
            conv1d_app_manager,
            gate_delta_net_manager
        );

        this->attn_block = std::make_unique<qwen3_6_moe_attn_block_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            gemm_app_manager,
            mha_app_manager
        );

        this->moe_mlp = std::make_unique<qwen3_6_moe_expert_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            gemm_app_manager,
            dequant_mm_app_manager
        );
    }

    buffer<bf16> forward(int layer_idx, buffer<bf16>& residual, int L, buffer<u8>& all_proj_weights, buffer<bf16>& rms_weights, buffer<bf16>& rope_weights, buffer<bf16>& kv_cache, std::vector<int>& input_idx, int L_begin = 0){
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        size_t x_size = (size_t)L_padded * desc->D;

        if (x_size != common_bufs.x.size()) {
            common_bufs.x = this->gemm_app_manager->create_bo_buffer<bf16>((size_t)L_padded * desc->D);
            common_bufs.after_mix = this->gemm_app_manager->create_bo_buffer<bf16>((size_t)L_padded * desc->Linear_DV);
            common_bufs.gate = this->gemm_app_manager->create_bo_buffer<bf16>((size_t)L_padded * desc->Linear_DV);
        }
        memset(common_bufs.x.data(), 0, common_bufs.x.size() * sizeof(bf16));

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
            block_output = this->linear_block->forward(L, all_proj_weights, rope_weights, kv_cache);
            PREFILL_PROF_TICK("layer:linear_block_total");
        }

        cpu_func::_residual_add(residual, block_output, residual, residual.size());

        cpu_func::_rms_norm(common_bufs.x, residual, rms_weights.data() + desc->D, desc->D, L, 0, 0);

        common_bufs.x.sync_to_device();
        PREFILL_PROF_TICK("layer:post_attn_rms_norm");

        auto mlp_out = moe_mlp->forward(layer_idx, L, all_proj_weights, rms_weights);
        PREFILL_PROF_TICK("layer:moe_block_total");

        cpu_func::_residual_add(residual, mlp_out, residual, residual.size());

        return residual;
    }
};

#endif
