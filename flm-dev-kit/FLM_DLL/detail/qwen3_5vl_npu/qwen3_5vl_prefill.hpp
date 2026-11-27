// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __QWEN3_5VL_PREFILL_HPP__
#define __QWEN3_5VL_PREFILL_HPP__
#include "qwen3_5vl_npu_sequence.hpp"
#include "npu_sequences/GateDeltaNetPrefillRuntimeSequence.hpp"
#include "qwen3_5vl_image.hpp"
#include "modules/gemm.hpp"
#include "qwen3_5vl_model_def.hpp"
#include "utils/avx512_util.hpp"
#include "cpu_functions.hpp"
#include "qwen3_5vl_launch.hpp"

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

    // control-packet switches (qwen3_5vl_launch.hpp): conv1d and the delta net run as guests in the mm context,
    // one window per layer; null without them
    qwen3_5vl_launch* ll;
    npu_app_manager* gemm_mgr;
    // the switches into and out of the delta net's app (gdn or gdn0, see qwen3_5vl_prefill_context)
    std::string sw_to_gdn = "conv_then_gdn";
    std::string sw_from_gdn = "gdn_then_mm";

    // host stages overlapped with the projections, ranged qkv syncs; QWEN35_NO_OVERLAP=1 keeps the serial path
    bool overlap = std::getenv("QWEN35_NO_OVERLAP") == nullptr;
    // only the conv state rows of the cache cross in the overlapped path; QWEN35_NO_GEMM_FUSE=1 syncs all of it
    bool ranged_state = std::getenv("QWEN35_NO_GEMM_FUSE") == nullptr;

    qwen3_5vl_linear_block_prefill_context(
        qwen3_5vl_desc* desc,
        qwen3_5vl_npu_sequence* seq_gen,
        qwen3_5vl_common_buffers* common_bufs,
        npu_app_manager* gemm_app_manager,
        npu_app_manager* conv1d_app_manager,
        npu_app_manager* gate_delta_net_manager,
        qwen3_5vl_launch* ll = nullptr
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs), ll(ll), gemm_mgr(gemm_app_manager)
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
            seq_gen->gen_dequant_mm_512(this->linear_qkv_proj.seq(), L_padded, desc->D, (desc->Linear_DQKV), desc->get_linear_qkv_offset(), (size_t)(desc->conv_kernel_size - 1) * (desc->Linear_DQKV), 0);
            seq_gen->gen_dequant_mm_512(this->linear_g_proj.seq(), L_padded, desc->D, desc->Linear_DV, desc->get_linear_g_offset(), 0, 1);
            seq_gen->gen_dequant_mm_512(this->linear_o_proj.seq(), L_padded, desc->Linear_DV, desc->D, desc->get_linear_o_offset(), 0, 0);
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

        // alpha / beta depend on x only: done before conv1d, so conv1d and the delta net can run back to back
        auto host_alpha_beta = [&]() {
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
        };
        // conv state: qkv rows [0, k-1) take the previous chunk's last k-1 rows from cache, cache takes this chunk's
        const size_t conv_state_bytes = (size_t)(desc->conv_kernel_size - 1) * desc->Linear_DQKV * sizeof(bf16);
        const size_t qkv_tail_bytes = (size_t)L * desc->Linear_DQKV * sizeof(bf16);

        if (overlap) {
            // alpha / beta on the host while the array runs the qkv projection, the conv state swap while it runs the z
            // projection; only the conv state rows of qkv cross between host and device
            auto qkv_run = linear_qkv_proj.create_run(qkv, common_bufs->x, proj_weights);
            qkv_run.start();
            host_alpha_beta();
            flm::npu::wait_for_completion(qkv_run, "linear_qkv_proj");
            auto g_run = linear_g_proj.create_run(common_bufs->gate, common_bufs->x, proj_weights);
            g_run.start();
            qkv.sync_range_from_device_exact(qkv_tail_bytes, conv_state_bytes);
            // the conv state rows lead the cache; the delta net state behind them stays on the device
            if (ranged_state) cache.sync_range_from_device_exact(0, conv_state_bytes);
            else cache.sync_from_device();
            memcpy(qkv.data(), cache.data(), conv_state_bytes);
            memcpy(cache.data(), (u8*)qkv.data() + qkv_tail_bytes, conv_state_bytes);
            if (ranged_state) cache.sync_range_to_device_exact(0, conv_state_bytes);
            else cache.sync_to_device();
            qkv.sync_range_to_device_exact(0, conv_state_bytes);
            flm::npu::wait_for_completion(g_run, "linear_g_proj");
            common_bufs->gate.sync_from_device();
            PREFILL_PROF_TICK("linear:qkv+g_proj+alpha_beta");
        } else {
            linear_qkv_proj(qkv, common_bufs->x, proj_weights);
            linear_g_proj(common_bufs->gate, common_bufs->x, proj_weights);

            qkv.sync_from_device();
            common_bufs->gate.sync_from_device();
            PREFILL_PROF_TICK("linear:qkv+g_proj");

            // copy cache back
            cache.sync_from_device();
            memcpy(qkv.data(), cache.data(), conv_state_bytes);
            memcpy(cache.data(), (u8*)qkv.data() + qkv_tail_bytes, conv_state_bytes);
            cache.sync_to_device();
            qkv.sync_to_device();
            PREFILL_PROF_TICK("linear:conv_cache");

            host_alpha_beta();
            PREFILL_PROF_TICK("linear:alpha_beta");
        }

#if !defined(FLM_USE_HRX)
        if (ll) {
            // one submission: mm -> conv1d -> delta net -> mm
            std::vector<qwen3_5vl_launch::hop> hops(2);
            hops[0].to = "mm_then_conv";
            hops[0].runs.push_back(conv1d.create_run(conv_qkv, qkv, ssm_weights));
            hops[1].to = sw_to_gdn.c_str();
            hops[1].runs.push_back(deltanet.create_run(cache, conv_qkv, alpha_beta, common_bufs->after_mix));
            ll->window(gemm_mgr, hops, sw_from_gdn.c_str());
            common_bufs->after_mix.sync_from_device();
            PREFILL_PROF_TICK("linear:conv1d+deltanet");
        } else
#endif
        {
            conv1d(conv_qkv, qkv, ssm_weights);
            conv_qkv.sync_from_device();
            PREFILL_PROF_TICK("linear:conv1d");

            deltanet(
                cache,
                conv_qkv,
                alpha_beta,
                common_bufs->after_mix
            );

            common_bufs->after_mix.sync_from_device();
            PREFILL_PROF_TICK("linear:deltanet");
        }

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

        // x was flushed after the input norm and the host only read it since
        if (!overlap) common_bufs->x.sync_to_device();
        linear_o_proj(common_bufs->x, common_bufs->after_mix, proj_weights);
        common_bufs->x.sync_from_device();
        PREFILL_PROF_TICK("linear:o_proj");

        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);
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

    // q, k and v in one dispatch into qkv = [q | k | v] (L_padded rows each, q first so attention reads it at offset
    // 0), their weights being stored back to back; z alone while the host ropes q and k and fills the kv cache.
    // QWEN35_NO_GEMM_FUSE=1 keeps the four projections
    npu_app qkv_proj;
    buffer<bf16> qkv;
    bool fuse = std::getenv("QWEN35_NO_GEMM_FUSE") == nullptr;

    // control-packet switches (qwen3_5vl_launch.hpp): attention runs as a guest in the mm context; null without them
    qwen3_5vl_launch* ll;
    npu_app_manager* gemm_mgr;

    qwen3_5vl_attn_block_prefill_context(
        qwen3_5vl_desc* desc,
        qwen3_5vl_npu_sequence* seq_gen,
        qwen3_5vl_common_buffers* common_bufs,
        npu_app_manager* gemm_app_manager,
        npu_app_manager* mha_app_manager,
        qwen3_5vl_launch* ll = nullptr
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs), ll(ll), gemm_mgr(gemm_app_manager){
        this->q_proj = gemm_app_manager->create_app();
        this->k_proj = gemm_app_manager->create_app();
        this->v_proj = gemm_app_manager->create_app();
        this->g_proj = gemm_app_manager->create_app();
        this->o_proj = gemm_app_manager->create_app();
        this->qkv_proj = gemm_app_manager->create_app();

        this->mha = mha_app_manager->create_app();
    }

    // the fused dispatch reads q, k and v as one weight block
    bool qkv_weights_contiguous() {
        auto* w = (qwen3_5vl_attn_layer*)(desc->weight_desc[desc->attn_idx].get());
        return w->attn_k.offset == w->attn_q.offset + w->attn_q.get_size()
            && w->attn_v.offset == w->attn_k.offset + w->attn_k.get_size();
    }

    buffer<bf16> forward(int L, buffer<u8>& proj_weights, buffer<bf16>& rope_rms_weights, buffer<bf16>& cache, std::vector<int>& input_idx, int L_begin) {
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        if (L_padded_old != L_padded){
            fuse = fuse && qkv_weights_contiguous();
            if (fuse) {
                const size_t rows = (size_t)L_padded;
                seq_gen->gen_dequant_mm_512(this->qkv_proj.seq(), L_padded, desc->D,
                    {{desc->DQ, 0}, {desc->DK, rows * desc->DQ}, {desc->DV, rows * (desc->DQ + desc->DK)}},
                    desc->get_attn_q_offset(), 0);
                qkv = this->qkv_proj.create_bo_buffer<bf16>(rows * (desc->DQ + desc->DK + desc->DV));
            } else {
                seq_gen->gen_dequant_mm_512(this->q_proj.seq(), L_padded, desc->D, desc->DQ, desc->get_attn_q_offset(), 0, 0);
                seq_gen->gen_dequant_mm_512(this->k_proj.seq(), L_padded, desc->D, desc->DK, desc->get_attn_k_offset(), 0, 0);
                seq_gen->gen_dequant_mm_512(this->v_proj.seq(), L_padded, desc->D, desc->DV, desc->get_attn_v_offset(), 0, 0);
                q  = this->q_proj.create_bo_buffer<bf16>((size_t)L_padded * desc->DQ);
                k  = this->q_proj.create_bo_buffer<bf16>((size_t)L_padded * desc->DK);
                v  = this->q_proj.create_bo_buffer<bf16>((size_t)L_padded * desc->DV);
            }
            seq_gen->gen_dequant_mm_512(this->g_proj.seq(), L_padded, desc->D, desc->DQ, desc->get_attn_g_offset(), 0, 0);
            seq_gen->gen_dequant_mm_512(this->o_proj.seq(), L_padded, desc->DQ, desc->D, desc->get_attn_o_offset(), 0, 0);
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

        if (fuse) {
            const size_t rows = (size_t)L_padded;
            buffer<bf16> q_rows(qkv.data(), rows * desc->DQ);
            buffer<bf16> k_rows(qkv.data() + rows * desc->DQ, rows * desc->DK);
            buffer<bf16> v_rows(qkv.data() + rows * (desc->DQ + desc->DK), rows * desc->DV);
            qkv_proj(qkv, common_bufs->x, proj_weights);
            qkv.sync_from_device();

            auto g_run = g_proj.create_run(g, common_bufs->x, proj_weights);
            g_run.start();
            cpu_func::_rope_rms_batch(q_rows, desc->DQ, input_idx, L, desc->num_heads, q_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
            cpu_func::_rope_rms_batch(k_rows, desc->DK, input_idx, L, desc->num_kv_heads, k_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
            // q for attention; k was written on the host too, flushed before the array writes this bo again
            qkv.sync_to_device();
            fill_kv_cache(k_rows, v_rows, cache, L_begin, L, true);
            flm::npu::wait_for_completion(g_run, "g_run");
            g.sync_from_device();
            // the host only reads g here, so no flush; 16-aligned chunks give the same values as one call
            const int n = L * (int)desc->DQ;
            const int chunk = 16 * 4096;
            #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
            for (int c = 0; c < n; c += chunk)
                cpu_func::_sigmoid(g_sigmoid.data() + c, g.data() + c, std::min(chunk, n - c));
            PREFILL_PROF_TICK("attn:qkvg_proj");
            PREFILL_PROF_TICK("attn:fill_kv_cache");
        } else {
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
        }

        buffer<bf16>& q_bo = fuse ? qkv : q;
#if !defined(FLM_USE_HRX)
        if (ll) {
            // one submission: mm -> attention -> mm
            std::vector<qwen3_5vl_launch::hop> hops(1);
            hops[0].to = "mm_then_attn";
            hops[0].runs.push_back(mha.create_run(common_bufs->after_mix, q_bo, cache));
            ll->window(gemm_mgr, hops, "attn_then_mm");
        } else
#endif
        mha(common_bufs->after_mix, q_bo, cache);

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

    // ranged: only the rows written cross to the device (whole rows of whole cache lines; the host does not read the
    // cache, so it skips the invalidate)
    void fill_kv_cache(buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, buffer<bf16>& cache, int L_begin, int L, bool ranged = false){
        tensor_2d<bf16> k_cache(buffer_k, desc->DK);
        tensor_2d<bf16> v_cache(buffer_v, desc->DV);

        if (ranged) {
            bf16* k_ptr = cache.data() + (size_t)L_begin * desc->DK;
            bf16* v_ptr = cache.data() + (size_t)L_begin * desc->DV + cache.size() / 2;
            for (int l = 0; l < L; l++){
                memcpy(k_ptr + (size_t)l * desc->DK, k_cache[l].data(), desc->DK * sizeof(bf16));
                memcpy(v_ptr + (size_t)l * desc->DV, v_cache[l].data(), desc->DV * sizeof(bf16));
            }
            cache.sync_range_to_device_exact((size_t)L_begin * desc->DK * sizeof(bf16), (size_t)L * desc->DK * sizeof(bf16));
            cache.sync_range_to_device_exact((cache.size() / 2 + (size_t)L_begin * desc->DV) * sizeof(bf16), (size_t)L * desc->DV * sizeof(bf16));
            return;
        }

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
    qwen3_5vl_desc* desc;
    qwen3_5vl_npu_sequence* seq_gen;
    qwen3_5vl_common_buffers* common_bufs;

    int L_old = -1;
    /// up/gate runs the fused GLU (seq_gen->mm_glu and at least QWEN35_GLU_MIN_ROWS padded rows, default 0); set with the sequences
    bool glu_now = false;
    static int glu_min_rows() {
        static const int v = [] { const char* e = std::getenv("QWEN35_GLU_MIN_ROWS"); return e && *e ? std::max(0, std::atoi(e)) : 0; }();
        return v;
    }

    npu_app upgate_proj;
    npu_app down_proj;

    buffer<bf16> upgate;
    buffer<bf16> hid;

    qwen3_5vl_mlp_prefill_context(
        qwen3_5vl_desc* desc,
        qwen3_5vl_npu_sequence* seq_gen,
        qwen3_5vl_common_buffers* common_bufs,
        npu_app_manager* gemm_app_manager
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs){
        upgate_proj = gemm_app_manager->create_app();
        down_proj = gemm_app_manager->create_app();
    }

    buffer<bf16> forward(int L, buffer<u8>& proj_weights, buffer<bf16>& rms_weights) {
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        if (L_old != L_padded){
            glu_now = seq_gen->mm_glu && L_padded >= glu_min_rows();
            seq_gen->gen_dequant_mm_512(this->upgate_proj.seq(), L_padded, desc->D, desc->intermediate_size * 2, desc->get_upgate_offset(), 0, glu_now ? 3 : 0);
            seq_gen->gen_dequant_mm_512(this->down_proj.seq(), L_padded, desc->intermediate_size, desc->D, desc->get_down_offset(), 0, 0);
            L_old = L_padded;
            if (!glu_now) upgate = this->upgate_proj.create_bo_buffer<bf16>((size_t)L_padded * desc->intermediate_size * 2);
            hid = this->upgate_proj.create_bo_buffer<bf16>((size_t)L_padded * desc->intermediate_size);
            memset(hid.data(), 0, hid.size() * sizeof(bf16));
            // write the zeros back now: with the fused GLU the host never syncs hid again, and dirty cache lines left by
            // the memset would otherwise be evicted over the array's output at random later (first prefill after a resize)
            hid.sync_to_device();
        }
        PREFILL_PROF_TICK("mlp:seq_gen+alloc");

        if (glu_now) {
            // the array leaves silu(gate) * up in hid: no pull, no host GLU, no push
            this->upgate_proj(hid, this->common_bufs->x, proj_weights);
        } else {
            this->upgate_proj(upgate, this->common_bufs->x, proj_weights);
            upgate.sync_from_device();

            hid.sync_from_device();

            cpu_func::_exp_glu(hid.data(), upgate.data(), L, desc->intermediate_size, 512);

            hid.sync_to_device();
        }
        PREFILL_PROF_TICK("mlp:upgate+glu");

        this->down_proj(common_bufs->x, hid, proj_weights);

        PREFILL_PROF_TICK("mlp:down_proj");

        common_bufs->x.sync_from_device();
        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);

        return valid;
    }
};

struct qwen3_5vl_prefill_context{
    qwen3_5vl_desc* desc;
    qwen3_5vl_npu_sequence* seq_gen;
    qwen3_5vl_common_buffers common_bufs;

    std::unique_ptr<qwen3_5vl_attn_block_prefill_context> attn_block;
    bool pad_only_clear = std::getenv("QWEN35_NO_GEMM_FUSE") == nullptr;
    std::unique_ptr<qwen3_5vl_linear_block_prefill_context> linear_block;
    std::unique_ptr<qwen3_5vl_mlp_prefill_context> mlp;

    // for xclbins
    npu_app_manager* gemm_app_manager;
    npu_app_manager* mha_app_manager;
    npu_app_manager * conv1d_app_manager;
    npu_app_manager * gate_delta_net_manager;
    // conv1d / delta net / attention as guests in the mm context (qwen3_5vl_launch.hpp); ll.ready says so
    qwen3_5vl_launch ll;

    qwen3_5vl_prefill_context(){}
    qwen3_5vl_prefill_context(
        npu_xclbin_manager* npu,
        qwen3_5vl_desc* model_desc,
        qwen3_5vl_npu_sequence* seq_gen,
        LM_Config& config
    ) : desc(model_desc), seq_gen(seq_gen) {
        // mm_rtm2_glu.xclbin (FLM_Xclbin/dequant_mm_bfp16_rtm2_glu_512x512x512): same Q4_K weights and dispatch layout as mm.xclbin,
        // dequantized in the core with bfp16 activations and a native bfp16 multiply. Used when the model's xclbin folder has it;
        // QWEN35_NO_BFP_MM=1 keeps mm.xclbin.
        const std::string xdir = utils::path_join(config.exec_path, "xclbins", config.model_name);
        const std::string bfp_mm = utils::path_join(xdir, "mm_rtm2_glu.xclbin");
        const char* no_bfp = std::getenv("QWEN35_NO_BFP_MM");
        const bool use_bfp = std::filesystem::exists(bfp_mm) && !(no_bfp && no_bfp[0] == '1');
        this->gemm_app_manager = npu->register_xclbin(use_bfp ? bfp_mm : utils::path_join(xdir, "mm.xclbin"));
        // its fused GLU epilogue (rtp_act 3) computes silu(gate) * up in the array; QWEN35_NO_GLU=1 keeps the host GLU
        const char* no_glu = std::getenv("QWEN35_NO_GLU");
        seq_gen->mm_glu = use_bfp && desc->intermediate_size % 512 == 0 && !(no_glu && no_glu[0] == '1');
        // conv1d, the delta net and attention in mm_rtm2_glu's context when the folder has qwen3_5.switches (built
        // against it, see qwen3_5vl_launch.hpp); runlists need preemption off. Its delta net app gdn is the chunked
        // core (GDN_CHUNK_MODE=1); QWEN35_GDN_RECURRENT=1 takes gdn0, the per-token recurrence of
        // GateDeltaNet_prefill.xclbin (FLM_Xclbin/Qwen3_5/prefill_switches)
        const char* gdn_rec = std::getenv("QWEN35_GDN_RECURRENT");
        const std::string gdn_app = gdn_rec && gdn_rec[0] == '1' ? "gdn0" : "gdn";
        ll.names = {"cold_mm", "mm_then_conv", "conv_then_" + gdn_app, gdn_app + "_then_mm", "mm_then_attn", "attn_then_mm"};
        if (use_bfp && !npu->is_preemption_enabled() && ll.ensure(gemm_app_manager, utils::path_join(xdir, "qwen3_5.switches"))) {
            header_print("qwen3.5", "prefill conv1d, delta net and attention share the mm hardware context (control-packet switches)");
            this->conv1d_app_manager = this->gate_delta_net_manager = this->mha_app_manager = gemm_app_manager;
        } else {
            this->conv1d_app_manager = npu->register_xclbin(utils::path_join(xdir, "conv.xclbin"));
            this->gate_delta_net_manager = npu->register_xclbin(utils::path_join(xdir, "GateDeltaNet_prefill.xclbin"));
            this->mha_app_manager = npu->register_xclbin(utils::path_join(xdir, "attn.xclbin"));
        }
        qwen3_5vl_launch* sw = ll.ready ? &ll : nullptr;

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
            sw
        );
        this->linear_block->sw_to_gdn = ll.names[2];
        this->linear_block->sw_from_gdn = ll.names[3];

        this->attn_block = std::make_unique<qwen3_5vl_attn_block_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            gemm_app_manager,
            mha_app_manager,
            sw
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
        size_t x_size = (size_t)L_padded * desc->D;

        if (x_size != common_bufs.x.size()) {
            common_bufs.x = this->gemm_app_manager->create_bo_buffer<bf16>((size_t)L_padded * desc->D);
            common_bufs.after_mix = this->gemm_app_manager->create_bo_buffer<bf16>((size_t)L_padded * desc->Linear_DV);
            common_bufs.gate = this->gemm_app_manager->create_bo_buffer<bf16>((size_t)L_padded * desc->Linear_DV);
        }
        // the input norm writes rows [0, L) whole, so only the padding rows need clearing (QWEN35_NO_GEMM_FUSE=1: all)
        if (pad_only_clear) memset(common_bufs.x.data() + (size_t)L * desc->D, 0, (x_size - (size_t)L * desc->D) * sizeof(bf16));
        else memset(common_bufs.x.data(), 0, common_bufs.x.size() * sizeof(bf16));

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

        cpu_func::_residual_add(residual, block_output, residual, residual.size());

        cpu_func::_rms_norm(common_bufs.x, residual, rms_weights.data() + desc->D, desc->D, L, 0, 0);

        common_bufs.x.sync_to_device();
        PREFILL_PROF_TICK("layer:post_attn_rms_norm");

        auto mlp_out = mlp->forward(L, all_proj_weights, rms_weights);
        PREFILL_PROF_TICK("layer:mlp_block_total");

        cpu_func::_residual_add(residual, mlp_out, residual, residual.size());

        return residual;
    }
};

#endif
