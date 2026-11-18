// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __MINICPM_V_PREFILL_HPP__
#define __MINICPM_V_PREFILL_HPP__
#include "minicpm_v_4_7_npu_sequence.hpp"
#include "npu_sequences/GateDeltaNetPrefillRuntimeSequence.hpp"
#include "modules/gemm.hpp"
#include "minicpm_v_model_def.hpp"
#include "utils/avx512_util.hpp"
#include "cpu_functions.hpp"
#include "minicpm_v_prefill_cpu.hpp"
#include "minicpm_v_prefill_xref.hpp"
#include <unordered_map>
#include <map>
#include <tuple>
#include <memory>
#include <optional>
#include <sstream>
#include <fstream>
#include <filesystem>
#include "image/npu/vision_bo.hpp"
#include "minicpm_v_reconfig.hpp"

// ---------------------------------------------------------------------------
// Prefill profiling.
// Enable by compiling with -DMINICPM_V_PREFILL_PROFILE (all code below
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
// #define MINICPM_V_PREFILL_PROFILE
#ifdef MINICPM_V_PREFILL_PROFILE
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

struct minicpm_v_common_buffers{
    buffer<bf16> x;
    buffer<bf16> gate;
    buffer<bf16> after_mix;

    minicpm_v_common_buffers(){}
};

// ---------------------------------------------------------------------------
// Each block below splits into stages, one per array overlay. A stage is a pair:
// _<stage>_npu() runs it on the array and returns true, or returns false when its
// MINICPM_V_*_ON_NPU macro is 0 (or it declines at run time); the caller then runs
// _<stage>_host(), the minicpm_v_cpu kernel over the same buffers. Both leave the
// stage's output valid on host *and* device (minicpm_v_cpu::scratch_push/pull), so
// any mix of backends is coherent.
// ---------------------------------------------------------------------------

/// The z projection and alpha/beta overlap with the qkv projection / the host conv + scan only when the
/// array runs nothing but the projections (the stock Linux build) and nothing wants the intermediates.
#if MINICPM_V_MM_ON_NPU && !MINICPM_V_CONV_ON_NPU && !MINICPM_V_GDN_ON_NPU && !defined(ENABLE_DEBUG)
#define LINEAR_ASYNC_Z 1
#else
#define LINEAR_ASYNC_Z 0
#endif

/// The block-pipelined linear layer (minicpm_v_prefill_context::forward_linear_pipelined) needs the same
/// split as LINEAR_ASYNC_Z (projections on the array, conv + delta rule on the host) and no debug taps.
#if LINEAR_ASYNC_Z && !MINICPM_V_GDN_ON_NPU && !MINICPM_V_CONV_ON_NPU && DEBUG_LEVEL < 1
#define LINEAR_PIPELINE 1
#else
#define LINEAR_PIPELINE 0
#endif

struct minicpm_v_linear_block_prefill_context{
    int L_padded_old = -1;
    int seq_padded_old = -1;   //!< mm_seq_key the whole-L array sequences were generated for
    int L_old = -1;
#ifdef ENABLE_DEBUG
    //!< set by minicpm_v_prefill_context::forward() before every call, so the xref lines
    //!< can name the layer without threading it through every stage signature
    int layer_idx = -1;
    //!< non-null in a debug build; whether it scores anything is up to enabled(layer_idx)
    minicpm_v_prefill_xref* xref = nullptr;
#endif
    minicpm_v_desc* desc;
    minicpm_v_4_7_npu_sequence* seq_gen;
    minicpm_v_common_buffers* common_bufs;
    npu_xclbin_manager* npu;

#if MINICPM_V_MM_ON_NPU
    npu_app linear_qkv_proj;
    npu_app linear_g_proj;
    npu_app linear_o_proj;
#endif
#if MINICPM_V_CONV_ON_NPU
    npu_app conv1d;
#endif
#if MINICPM_V_GDN_ON_NPU
    npu_app deltanet;
#endif

    buffer<bf16> qkv;
    buffer<bf16> conv_qkv;
    buffer<float> alpha_beta;

    minicpm_v_linear_block_prefill_context(
        minicpm_v_desc* desc,
        minicpm_v_4_7_npu_sequence* seq_gen,
        minicpm_v_common_buffers* common_bufs,
        npu_xclbin_manager* npu,
        npu_app_manager* gemm_app_manager,
        npu_app_manager* conv1d_app_manager,
        npu_app_manager* gate_delta_net_manager
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs), npu(npu)
    {
#if MINICPM_V_MM_ON_NPU
        this->linear_qkv_proj = gemm_app_manager->create_app();
        this->linear_g_proj = gemm_app_manager->create_app();
        this->linear_o_proj = gemm_app_manager->create_app();
#endif
#if MINICPM_V_CONV_ON_NPU
        this->conv1d = conv1d_app_manager->create_app();
#endif
#if MINICPM_V_GDN_ON_NPU
        this->deltanet = gate_delta_net_manager->create_app();
#endif
    }

    minicpm_v_linear_layer& weights() {
        return static_cast<minicpm_v_linear_layer&>(*desc->weight_desc[desc->lin_idx]);
    }

    /// @brief grow the scratch and regenerate the array sequences for this L
    /// @param want_seq false when the block-pipelined path runs the projections (it has its own,
    ///        L independent sequences) so the whole-L ones are not generated for nothing
    void reserve(int L, int L_padded, bool want_seq = true) {
#if MINICPM_V_MM_ON_NPU || MINICPM_V_CONV_ON_NPU
        const int seq_key = (int)seq_gen->mm_seq_key(L_padded, L);   // L_padded, or also the rows a runtime-m mm covers
        if (want_seq && seq_padded_old != seq_key){
#if MINICPM_V_MM_ON_NPU
            seq_gen->gen_dequant_mm_512(this->linear_qkv_proj.seq(), L_padded, desc->D, (desc->Linear_DQKV), desc->get_linear_qkv_offset(), (size_t)(desc->conv_kernel_size - 1) * (desc->Linear_DQKV), 0, 0, L);
            seq_gen->gen_dequant_mm_512(this->linear_g_proj.seq(), L_padded, desc->D, desc->Linear_DV, desc->get_linear_g_offset(), 0, 1, 0, L);
            seq_gen->gen_dequant_mm_512(this->linear_o_proj.seq(), L_padded, desc->Linear_DV, desc->D, desc->get_linear_o_offset(), 0, 0, 0, L);
#endif
#if MINICPM_V_CONV_ON_NPU
            seq_gen->gen_seq_conv1d(
                this->conv1d.seq(),
                L_padded + (desc->conv_kernel_size - 1),  desc->Linear_DQKV , L_padded, 0
            );
#endif
            seq_padded_old = seq_key;
        }
#endif
        if (L_padded_old != L_padded){
            qkv  = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)(L_padded + desc->conv_kernel_size - 1) * desc->Linear_DQKV);
            conv_qkv  = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->Linear_DQKV);
            alpha_beta = minicpm_v_cpu::scratch_alloc<float>(npu, (size_t)L_padded * desc->linear_num_v_heads * 2);
            memset(alpha_beta.data(), 0, alpha_beta.size() * sizeof(float));
            minicpm_v_cpu::scratch_push(alpha_beta);
        }
#if MINICPM_V_GDN_ON_NPU
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
#endif
        L_padded_old = L_padded;
        L_old = L;
    }

    /// @brief in_proj_qkv into qkv rows [KS-1, KS-1+L), in_proj_z into common gate with
    ///        the silu applied (the array fuses it, rtp_act = 1)
    bool _in_proj_npu(int L, buffer<u8>& proj_weights) {
#if MINICPM_V_MM_ON_NPU
        linear_qkv_proj(qkv, common_bufs->x, proj_weights);
        linear_g_proj(common_bufs->gate, common_bufs->x, proj_weights);

        const size_t pad = desc->conv_kernel_size - 1;
        minicpm_v_cpu::scratch_pull_rows(qkv, pad, L, desc->Linear_DQKV);
        minicpm_v_cpu::scratch_pull_rows(common_bufs->gate, 0, L, desc->Linear_DV);
        return true;
#else
        (void)L; (void)proj_weights;
        return false;
#endif
    }

    void _in_proj_host(int L, buffer<u8>& proj_weights) {
        const int DQKV = desc->Linear_DQKV;
        const int pad = desc->conv_kernel_size - 1;
        minicpm_v_cpu::dequant_mm(qkv.data() + (size_t)pad * DQKV, DQKV,
                                  proj_weights.data() + desc->get_linear_qkv_offset(),
                                  common_bufs->x.data(), desc->D, L, desc->D, DQKV, minicpm_v_cpu::BODY_LAYOUT);
        minicpm_v_cpu::dequant_mm(common_bufs->gate.data(), desc->Linear_DV,
                                  proj_weights.data() + desc->get_linear_g_offset(),
                                  common_bufs->x.data(), desc->D, L, desc->D, desc->Linear_DV, minicpm_v_cpu::BODY_LAYOUT);
        MINICPM_V_DUMP("linear_z", common_bufs->gate.data(), L, desc->Linear_DV);
        minicpm_v_cpu::silu_inplace(common_bufs->gate.data(), (size_t)L * desc->Linear_DV);
        minicpm_v_cpu::scratch_push_rows(qkv, 0, (size_t)pad + L, DQKV);
        minicpm_v_cpu::scratch_push_rows(common_bufs->gate, 0, L, desc->Linear_DV);
    }

    /// @brief conv_qkv = causal conv1d + silu over [state | qkv], then L2 norm over the q
    ///        and k heads
    bool _conv_npu(int L, buffer<bf16>& ssm_weights) {
#if MINICPM_V_CONV_ON_NPU
        conv1d(conv_qkv, qkv, ssm_weights);

        minicpm_v_cpu::scratch_pull_rows(conv_qkv, 0, L, desc->Linear_DQKV);
        return true;
#else
        (void)L; (void)ssm_weights;
        return false;
#endif
    }

    void _conv_host(int L, buffer<bf16>& ssm_weights) {
        PREFILL_PROF_INIT();
        const int DQKV = desc->Linear_DQKV;
        const bf16* w = ssm_weights.data() + weights().ssm_conv1d.offset / sizeof(bf16);
#if DEBUG_LEVEL >= 1
        // the dump wants the conv output before the L2 norm
        minicpm_v_cpu::conv1d_silu(conv_qkv.data(), qkv.data(), w, L, DQKV, desc->conv_kernel_size);
        MINICPM_V_DUMP("linear_conv", conv_qkv.data(), L, DQKV);
        minicpm_v_cpu::l2norm_heads(conv_qkv.data(), DQKV, L,
                                    (desc->Linear_DQ + desc->Linear_DK) / desc->linear_qk_head_dim,
                                    desc->linear_qk_head_dim);
#else
        minicpm_v_cpu::conv1d_silu_l2norm(conv_qkv.data(), qkv.data(), w, L, DQKV, desc->conv_kernel_size,
                                          (desc->Linear_DQ + desc->Linear_DK) / desc->linear_qk_head_dim,
                                          desc->linear_qk_head_dim);
#endif
        PREFILL_PROF_TICK("conv:silu+l2norm");
#if MINICPM_V_GDN_ON_NPU
        // only the array's delta rule reads conv_qkv from the device
        minicpm_v_cpu::scratch_push_rows(conv_qkv, 0, L, DQKV);
#endif
    }

    /// @brief the gated delta rule over conv_qkv into common after_mix, updating the
    ///        recurrent state that sits behind the conv state in the cache slot
    bool _gdn_npu(int L, buffer<bf16>& cache) {
#if MINICPM_V_GDN_ON_NPU
        deltanet(
            cache,
            conv_qkv,
            alpha_beta,
            common_bufs->after_mix
        );

        minicpm_v_cpu::scratch_pull_rows(common_bufs->after_mix, 0, L, desc->Linear_DV);
        cache.sync_from_device();
        return true;
#else
        (void)L; (void)cache;
        return false;
#endif
    }

    void _gdn_host(int L, buffer<bf16>& cache) {
        PREFILL_PROF_INIT();
        float* S = reinterpret_cast<float*>(cache.data() + desc->get_s_offset());
        float* alpha_ptr = alpha_beta.data();
        float* beta_ptr  = alpha_ptr + alpha_beta.size() / 2;
        minicpm_v_cpu::gated_delta_net(
            common_bufs->after_mix.data(), S, conv_qkv.data(), desc->Linear_DQKV,
            0, desc->Linear_DQ, desc->Linear_DQ + desc->Linear_DK,
            alpha_ptr, beta_ptr, L,
            desc->linear_num_qk_heads, desc->linear_num_v_heads,
            desc->linear_qk_head_dim, desc->linear_v_head_dim
        );
        PREFILL_PROF_TICK("gdn:kernel");
        cache.sync_to_device();
        PREFILL_PROF_TICK("gdn:cache_push");
        // after_mix is pushed once, by forward(), after the gated norm has rewritten it
    }

    /// @brief out_proj of after_mix into common x
    bool _o_proj_npu(int L, buffer<u8>& proj_weights) {
#if MINICPM_V_MM_ON_NPU
        linear_o_proj(common_bufs->x, common_bufs->after_mix, proj_weights);
        minicpm_v_cpu::scratch_pull_rows(common_bufs->x, 0, L, desc->D);
        return true;
#else
        (void)L; (void)proj_weights;
        return false;
#endif
    }

    void _o_proj_host(int L, buffer<u8>& proj_weights) {
        minicpm_v_cpu::dequant_mm(common_bufs->x.data(), desc->D,
                                  proj_weights.data() + desc->get_linear_o_offset(),
                                  common_bufs->after_mix.data(), desc->Linear_DV, L, desc->Linear_DV, desc->D,
                                  minicpm_v_cpu::BODY_LAYOUT);
        minicpm_v_cpu::scratch_push_rows(common_bufs->x, 0, L, desc->D);
    }

    // -----------------------------------------------------------------------------------
    // Cross reference. Each block replays the stage above it on the host over the array's
    // own inputs and scores the two. ENABLE_DEBUG only; see minicpm_v_prefill_xref.hpp.
    // -----------------------------------------------------------------------------------
#ifdef ENABLE_DEBUG
    /// \brief score the two array projections of the input block
    /// \note  Called before the conv-state carry overwrites rows [0, pad) of `qkv`, though
    ///        it would not matter: only rows [pad, pad + rows) are scored.
    void _xref_in_proj(int L, buffer<u8>& proj_weights) {
#if MINICPM_V_MM_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        const int   DQKV = desc->Linear_DQKV;
        const int   pad  = desc->conv_kernel_size - 1;
        const bf16* got  = qkv.data() + (size_t)pad * DQKV;

        std::vector<bf16> want((size_t)rows * DQKV);
        minicpm_v_cpu::dequant_mm(want.data(), DQKV, proj_weights.data() + desc->get_linear_qkv_offset(),
                                  common_bufs->x.data(), desc->D, rows, desc->D, DQKV,
                                  minicpm_v_cpu::BODY_LAYOUT);
        {
            minicpm_v_prefill_xref::guard g(xref, "linear_qkv", layer_idx, got, DQKV, DQKV, rows);
            xref->check("linear_qkv", layer_idx, got, want.data(), DQKV, DQKV, rows);
        }

        // the array fuses the silu into the g projection (rtp_act = 1); the host does the
        // matmul and then silu_inplace, so the reference has to do both to be comparable
        const int DV = desc->Linear_DV;
        std::vector<bf16> want_g((size_t)rows * DV);
        minicpm_v_cpu::dequant_mm(want_g.data(), DV, proj_weights.data() + desc->get_linear_g_offset(),
                                  common_bufs->x.data(), desc->D, rows, desc->D, DV,
                                  minicpm_v_cpu::BODY_LAYOUT);
        minicpm_v_cpu::silu_inplace(want_g.data(), (size_t)rows * DV);
        {
            minicpm_v_prefill_xref::guard g(xref, "linear_z", layer_idx, common_bufs->gate.data(), DV, DV, rows);
            xref->check("linear_z", layer_idx, common_bufs->gate.data(), want_g.data(), DV, DV, rows);
        }
#else
        (void)L; (void)proj_weights;
#endif
    }

    /// \brief score the conv overlay: causal conv1d + silu + the q/k L2 norm
    /// \note  Not row independent -- a row reads the conv_kernel_size - 1 rows before it --
    ///        so the replay runs over the whole prompt and only the report is truncated.
    void _xref_conv(int L, buffer<bf16>& ssm_weights) {
#if MINICPM_V_CONV_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        const int   DQKV = desc->Linear_DQKV;
        const bf16* w    = ssm_weights.data() + weights().ssm_conv1d.offset / sizeof(bf16);

        std::vector<bf16> want((size_t)L * DQKV);
        minicpm_v_cpu::conv1d_silu(want.data(), qkv.data(), w, L, DQKV, desc->conv_kernel_size);
        minicpm_v_cpu::l2norm_heads(want.data(), DQKV, L,
                                    (desc->Linear_DQ + desc->Linear_DK) / desc->linear_qk_head_dim,
                                    desc->linear_qk_head_dim);
        minicpm_v_prefill_xref::guard g(xref, "conv_qkv", layer_idx, conv_qkv.data(), DQKV, DQKV, rows);
        xref->check("conv_qkv", layer_idx, conv_qkv.data(), want.data(), DQKV, DQKV, rows);
#else
        (void)L; (void)ssm_weights;
#endif
    }

    /// \brief score the gated delta rule overlay, both its output and the state it leaves
    /// \param S_before snapshot of the recurrent state taken before the array run; the host
    ///        kernel updates the state in place, so replaying it against the array's own
    ///        cache slot would double-apply the scan. This is exactly what xref::guard
    ///        exists to catch.
    void _xref_gdn(int L, buffer<bf16>& cache, std::vector<float>& S_before) {
#if MINICPM_V_GDN_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx) || S_before.empty()) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        const int DV = desc->Linear_DV;

        std::vector<bf16> want((size_t)L * DV);
        float* alpha_ptr = alpha_beta.data();
        float* beta_ptr  = alpha_ptr + alpha_beta.size() / 2;
        minicpm_v_cpu::gated_delta_net(
            want.data(), S_before.data(), conv_qkv.data(), desc->Linear_DQKV,
            0, desc->Linear_DQ, desc->Linear_DQ + desc->Linear_DK,
            alpha_ptr, beta_ptr, L,
            desc->linear_num_qk_heads, desc->linear_num_v_heads,
            desc->linear_qk_head_dim, desc->linear_v_head_dim);
        {
            minicpm_v_prefill_xref::guard g(xref, "gdn_out", layer_idx, common_bufs->after_mix.data(), DV, DV, rows);
            xref->check("gdn_out", layer_idx, common_bufs->after_mix.data(), want.data(), DV, DV, rows);
        }

        // the state is fp32 [Hv][DK][DV] behind the conv state, and is the half of this
        // stage a wrong answer at token L would not show up in until the next chunk
        const float* S_got = reinterpret_cast<const float*>(cache.data() + desc->get_s_offset());
        const int    S_rows = (int)desc->linear_num_v_heads * desc->linear_qk_head_dim;
        minicpm_v_prefill_xref::guard_t<float> g(xref, "gdn_state", layer_idx, S_got,
                                                 desc->linear_v_head_dim, desc->linear_v_head_dim, S_rows);
        xref->check("gdn_state", layer_idx, S_got, S_before.data(),
                    desc->linear_v_head_dim, desc->linear_v_head_dim, S_rows);
#else
        (void)L; (void)cache; (void)S_before;
#endif
    }

    /// \brief score the output projection of the linear block
    void _xref_o_proj(int L, buffer<u8>& proj_weights) {
#if MINICPM_V_MM_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        std::vector<bf16> want((size_t)rows * desc->D);
        minicpm_v_cpu::dequant_mm(want.data(), desc->D, proj_weights.data() + desc->get_linear_o_offset(),
                                  common_bufs->after_mix.data(), desc->Linear_DV, rows,
                                  desc->Linear_DV, desc->D, minicpm_v_cpu::BODY_LAYOUT);
        minicpm_v_prefill_xref::guard g(xref, "linear_o", layer_idx, common_bufs->x.data(), desc->D, desc->D, rows);
        xref->check("linear_o", layer_idx, common_bufs->x.data(), want.data(), desc->D, desc->D, rows);
#else
        (void)L; (void)proj_weights;
#endif
    }
#endif // ENABLE_DEBUG

    /// fp32 copies of a layer's alpha / beta projection rows (the host kernel's operands), made once
    std::unordered_map<const bf16*, std::vector<float>> ab_weights;

    /// alpha = exp(g) and beta = sigmoid(b) for the L rows of x (the input norm output)
    void _alpha_beta(int L, buffer<bf16>& ssm_weights, buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu, int row0 = 0) {
        const int Hv = desc->linear_num_v_heads;
        float* alpha_ptr = alpha_beta.data() + (size_t)row0 * Hv;
        float* beta_ptr  = alpha_beta.data() + alpha_beta.size() / 2 + (size_t)row0 * Hv;
#if !MINICPM_V_GDN_ON_NPU && DEBUG_LEVEL < 1
        // The host delta rule reads only the first L rows, so there is no padding to activate, and
        // the two projections and their activations are one pass (bitwise the unfused result).
        std::vector<float>& wf = ab_weights[alpha_proj_cpu.data()];
        if (wf.empty()) {
            wf.resize((size_t)2 * Hv * desc->D);
            for (size_t i = 0; i < (size_t)Hv * desc->D; i++) {
                wf[i] = static_cast<float>(alpha_proj_cpu.data()[i]);
                wf[(size_t)Hv * desc->D + i] = static_cast<float>(beta_proj_cpu.data()[i]);
            }
        }
        minicpm_v_cpu::alpha_beta_proj(alpha_ptr, beta_ptr, wf.data(), wf.data() + (size_t)Hv * desc->D,
                                       common_bufs->x.data() + (size_t)row0 * desc->D, L, desc->D, Hv,
                                       (const float*)(ssm_weights.data() + desc->get_linear_A_offset() / sizeof(bf16)),
                                       (const float*)(ssm_weights.data() + desc->get_linear_dt_offset() / sizeof(bf16)));
#else
        cpu_func::_cpu_mm<bf16>(alpha_ptr, alpha_proj_cpu.data(), common_bufs->x.data(), L, desc->D, desc->linear_num_v_heads);
        cpu_func::_cpu_mm<bf16>(beta_ptr,  beta_proj_cpu.data(),  common_bufs->x.data(), L, desc->D, desc->linear_num_v_heads);
        MINICPM_V_DUMP("linear_alpha", alpha_ptr, L, desc->linear_num_v_heads);
        MINICPM_V_DUMP("linear_beta", beta_ptr, L, desc->linear_num_v_heads);
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
#if DEBUG_LEVEL >= 1
        if (minicpm_v_cpu::dumper().on()) {
            // the reference dumps the log-space decay g; the kernels consume exp(g)
            std::vector<float> g((size_t)L * desc->linear_num_v_heads);
            for (size_t i = 0; i < g.size(); i++) g[i] = logf(alpha_ptr[i]);
            MINICPM_V_DUMP("linear_g", g.data(), L, desc->linear_num_v_heads);
        }
#endif

#if MINICPM_V_GDN_ON_NPU
        minicpm_v_cpu::scratch_push(alpha_beta);
#endif
#endif
    }

    buffer<bf16> forward(int L, buffer<u8>& proj_weights, buffer<bf16>& ssm_weights, buffer<bf16>& cache, buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu) {
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        const int DQKV = desc->Linear_DQKV;
        const int pad = desc->conv_kernel_size - 1;
        this->reserve(L, L_padded);

        PREFILL_PROF_TICK("linear:seq_gen+alloc");

#if LINEAR_ASYNC_Z
        // The three host-only inputs are independent of the array's work: alpha/beta need only x,
        // so they run while the qkv projection does, and the z projection (consumed by the gated
        // norm at the very end) runs while the host does the conv and the delta rule.
        auto qkv_run = linear_qkv_proj.create_run(qkv, common_bufs->x, proj_weights);
        qkv_run.start();
        _alpha_beta(L, ssm_weights, alpha_proj_cpu, beta_proj_cpu);
        PREFILL_PROF_TICK("linear:alpha_beta");
        flm::npu::wait_for_completion(qkv_run, "qkv_run");
        minicpm_v_cpu::scratch_pull_rows(qkv, pad, L, DQKV);
        auto g_run = linear_g_proj.create_run(common_bufs->gate, common_bufs->x, proj_weights);
        g_run.start();
        PREFILL_PROF_TICK("linear:qkv_proj");
#else
        if (!_in_proj_npu(L, proj_weights)) {
            _in_proj_host(L, proj_weights);
        }
        MINICPM_V_DUMP("linear_qkv", qkv.data() + (size_t)pad * DQKV, L, DQKV);
#ifdef ENABLE_DEBUG
        _xref_in_proj(L, proj_weights);
#endif
        PREFILL_PROF_TICK("linear:qkv+g_proj");
#endif

        // copy cache back
        PREFILL_PROF_TICK("linear:pre_carry");
        cache.sync_from_device();
        memcpy(qkv.data(), cache.data(), (size_t)pad * DQKV * sizeof(bf16));
        memcpy(cache.data(), qkv.data() + L * DQKV, (size_t)pad * DQKV * sizeof(bf16));
#if MINICPM_V_GDN_ON_NPU
        cache.sync_to_device();   // the array's delta rule reads the state from the device; the host one pushes it itself
#endif
#if MINICPM_V_CONV_ON_NPU
        minicpm_v_cpu::scratch_push_rows(qkv, 0, (size_t)pad + L, DQKV);
#endif

        if (!_conv_npu(L, ssm_weights)) {
            _conv_host(L, ssm_weights);
        }
#ifdef ENABLE_DEBUG
        _xref_conv(L, ssm_weights);
#endif
        PREFILL_PROF_TICK("linear:conv1d");

#if !LINEAR_ASYNC_Z
        _alpha_beta(L, ssm_weights, alpha_proj_cpu, beta_proj_cpu);
#endif
        PREFILL_PROF_TICK("linear:alpha_beta");

#ifdef ENABLE_DEBUG
        // snapshot the recurrent state before the array consumes it, so the host replay has
        // the same starting point (it updates the state in place)
        std::vector<float> S_before;
#if MINICPM_V_GDN_ON_NPU
        if (xref != nullptr && xref->enabled(layer_idx)) {
            cache.sync_from_device();
            const float* S = reinterpret_cast<const float*>(cache.data() + desc->get_s_offset());
            S_before.assign(S, S + (size_t)desc->linear_num_v_heads * desc->linear_qk_head_dim * desc->linear_v_head_dim);
        }
#endif
#endif

        if (!_gdn_npu(L, cache)) {
            _gdn_host(L, cache);
        }
#ifdef ENABLE_DEBUG
        _xref_gdn(L, cache, S_before);
#endif
#if LINEAR_ASYNC_Z
        flm::npu::wait_for_completion(g_run, "g_run");
        minicpm_v_cpu::scratch_pull_rows(common_bufs->gate, 0, L, desc->Linear_DV);
        PREFILL_PROF_TICK("linear:wait_z");
#endif
        MINICPM_V_DUMP("linear_core", common_bufs->after_mix.data(), L, desc->Linear_DV);
        MINICPM_V_DUMP3("linear_state", reinterpret_cast<const float*>(cache.data() + desc->get_s_offset()),
                        desc->linear_num_v_heads, desc->linear_qk_head_dim, desc->linear_v_head_dim);
        PREFILL_PROF_TICK("linear:deltanet");

        bf16* gate_rms_norm_weight_ptr = ssm_weights.data() + desc->get_linear_ssm_norm_offset() / sizeof(bf16);

        // host-only stage with nothing running beside it: the full team (the division per element is the cost)
        #pragma omp parallel for num_threads(minicpm_v_cpu::prefill_threads()) schedule(static)
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
        MINICPM_V_DUMP("linear_norm", common_bufs->after_mix.data(), L, desc->Linear_DV);

        minicpm_v_cpu::scratch_push_rows(common_bufs->after_mix, 0, L, desc->Linear_DV);
        PREFILL_PROF_TICK("linear:gated_norm");

        if (!_o_proj_npu(L, proj_weights)) {
            _o_proj_host(L, proj_weights);
        }
        MINICPM_V_DUMP("linear_out", common_bufs->x.data(), L, desc->D);
#ifdef ENABLE_DEBUG
        _xref_o_proj(L, proj_weights);
#endif
        PREFILL_PROF_TICK("linear:o_proj");

        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);
        return valid;
    }
};

#if MINICPM_V_ATTN_ON_NPU && MINICPM_V_MM_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
/// @brief the two in-stream reconfigurations (mm -> attn, attn -> mm), built once from the shipped xclbins and shared by every attention layer.
///        Opt-in (FLM_MINICPM_SINGLE_CTX=1). Three gates, each with the two-context path as the fallback:
///          built      the NPU runs without preemption and both CDOs parse (init)
///          validated  a known-answer round trip on the real overlays passed at load, on an NPU this process has to itself (self-test)
///          usable()   per call: validated, not disabled by an earlier failure, and every hw_context on the NPU still belongs to this process
struct minicpm_v_attn_reconfig {
    using chain_t = minicpm_reconfig::Chain<npu_app, xrt::run>;
    chain_t to_attn, to_mm;
    bool built = false;
    bool validated = false;
    std::atomic<long> trips{0};     //!< attention layers run through the round trip
    std::atomic<long> skipped{0};   //!< attention layers that took the two-context path because another process was on the NPU
    ~minicpm_v_attn_reconfig() {
        if (built) header_print_r("INFO", "minicpm_v: single-context attention: " << trips.load() << " layers by round trip, " << skipped.load() << " skipped (NPU shared)");
    }
#ifdef FLM_MINICPM_RECONFIG_FAULT_INJECT
    // test build only: the Nth round trip (FLM_MCPM_FAULT_AT=N, 1-based, counted after the self-test) starts with the pre-fix prelude, which
    // hangs the attention dispatch; used to prove the failure path (timeout, disable, recovery, two-context fallback).
    chain_t to_attn_unquiesced;
    std::atomic<long> calls{0};
    long fault_at = 0;
#endif
    /// \param mm_xclbin the mm overlay actually in use (the home hw_context; mm.xclbin or mm_rtm.xclbin), \param attn_xclbin the attention overlay
    void init(npu_xclbin_manager* npu, npu_app_manager* gemm_app_manager, const std::string& mm_xclbin, const std::string& attn_xclbin) {
        if (!minicpm_reconfig::single_context()) return;
        if (npu->is_preemption_enabled()) {
            header_print_r("WARNING", "minicpm_v: single-context reconfiguration needs the NPU without preemption, using two contexts");
            return;
        }
        try {
            auto mk = [&] { return gemm_app_manager->create_app(); };
            to_attn.build(attn_xclbin, mm_xclbin, mk);
            to_mm.build(mm_xclbin, attn_xclbin, mk);
#ifdef FLM_MINICPM_RECONFIG_FAULT_INJECT
            to_attn_unquiesced.build(attn_xclbin, mm_xclbin, mk, false);
            if (const char* f = std::getenv("FLM_MCPM_FAULT_AT")) fault_at = std::atol(f);
#endif
            built = true;
        } catch (const std::exception& e) {
            header_print_r("WARNING", "minicpm_v: attention single-context reconfiguration unavailable (" << e.what() << "), using two contexts");
        }
    }
    /// usable for this call (re-checked every time: another process can appear at any moment)
    bool usable() {
        if (!built || !validated || minicpm_reconfig::disabled()) return false;
        if (minicpm_reconfig::npu_exclusive()) return true;
        skipped++;
        return false;
    }
};
#endif

struct minicpm_v_attn_block_prefill_context{
    minicpm_v_desc* desc;
    minicpm_v_4_7_npu_sequence* seq_gen;
    minicpm_v_common_buffers* common_bufs;
    npu_xclbin_manager* npu;

    int L_padded_old = -1;
    int mm_seq_key_old = -1;   //!< mm_seq_key the projection sequences were generated for
    int L_begin_old = -1;
    int mha_seq_begin = -1;    //!< (L_begin, L_padded) the mha sequence was last generated for
    int mha_seq_padded = -1;
#ifdef ENABLE_DEBUG
    //!< set by minicpm_v_prefill_context::forward() before every call
    int layer_idx = -1;
    //!< non-null in a debug build; whether it scores anything is up to enabled(layer_idx)
    minicpm_v_prefill_xref* xref = nullptr;

    /// \brief score one projection of this block against the host kernel
    /// \note Called from inside _qkvg_proj_npu() for q and k, because both are rotated in
    ///       place immediately after the run and the raw projection is gone by the time
    ///       forward() gets control back.
    void _xref_proj(const char* name, int L, const bf16* got, size_t w_offset,
                    buffer<u8>& proj_weights, int N) {
#if MINICPM_V_MM_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        std::vector<bf16> want((size_t)rows * N);
        minicpm_v_cpu::dequant_mm(want.data(), N, proj_weights.data() + w_offset,
                                  common_bufs->x.data(), desc->D, rows, desc->D, N,
                                  minicpm_v_cpu::BODY_LAYOUT);
        minicpm_v_prefill_xref::guard gu(xref, name, layer_idx, got, N, N, rows);
        xref->check(name, layer_idx, got, want.data(), N, N, rows);
#else
        (void)name; (void)L; (void)got; (void)w_offset; (void)proj_weights; (void)N;
#endif
    }

    /// \brief score the attention overlay against the host kernel over the same cache
    /// \note Gated on enabled() *before* anything is read, because the reference is an
    ///       L x DQ tensor plus a full pass over the kv cache -- for a 4 k prompt that is
    ///       more work than the attention it is checking, every layer.
    void _xref_attend(int L, buffer<bf16>& cache, int L_begin) {
#if MINICPM_V_ATTN_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        const bf16* k_cache = cache.data();
        const bf16* v_cache = cache.data() + cache.size() / 2;
        std::vector<bf16> want((size_t)L * desc->DQ);
        minicpm_v_cpu::attention(want.data(), q.data(), k_cache, v_cache,
                                 L, L_begin + L, desc->num_heads, desc->num_kv_heads, desc->DH,
                                 L_begin, 0, desc->DK, desc->DV, 1.0f / sqrtf((float)desc->DH));
        minicpm_v_prefill_xref::guard gu(xref, "attention_out", layer_idx,
                                         common_bufs->after_mix.data(), desc->DQ, desc->DQ, rows);
        xref->check("attention_out", layer_idx, common_bufs->after_mix.data(), want.data(),
                    desc->DQ, desc->DQ, rows);
#else
        (void)L; (void)cache; (void)L_begin;
#endif
    }

    /// \brief score the output projection of the attention block
    void _xref_o_proj(int L, buffer<u8>& proj_weights) {
#if MINICPM_V_MM_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        std::vector<bf16> want((size_t)rows * desc->D);
        minicpm_v_cpu::dequant_mm(want.data(), desc->D, proj_weights.data() + desc->get_attn_o_offset(),
                                  g.data(), desc->DQ, rows, desc->DQ, desc->D,
                                  minicpm_v_cpu::BODY_LAYOUT);
        minicpm_v_prefill_xref::guard gu(xref, "attn_o", layer_idx, common_bufs->x.data(), desc->D, desc->D, rows);
        xref->check("attn_o", layer_idx, common_bufs->x.data(), want.data(), desc->D, desc->D, rows);
#else
        (void)L; (void)proj_weights;
#endif
    }
#endif // ENABLE_DEBUG

#if MINICPM_V_MM_ON_NPU
    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app g_proj;
    npu_app o_proj;
#endif
#if MINICPM_V_ATTN_ON_NPU
    npu_app mha;
#if MINICPM_V_ATTN_ON_NPU && MINICPM_V_MM_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
    // the same attention sequence for the mm hw_context: it runs between the two in-stream reconfigurations of `reconf`
    minicpm_v_attn_reconfig* reconf = nullptr;
    npu_app_manager* gemm_mgr = nullptr;
    npu_app_manager* mha_mgr = nullptr;
    npu_app mha_sc;
    bool mha_sc_valid = false;
    int sc_L_begin = -1, sc_L_padded = -1, sc_kv_rows = -1;
#endif
#endif
#if MINICPM_V_ATTN_BFP16
    /// \brief true when `mha` runs attn_bfp16.xclbin (q and the KV cache go to the array as bfp16 tiles)
    bool mha_bfp16 = false;
    /// \brief V^T tiles in the chunk-major layout of attn_bfp16c.xclbin (per 128 keys: 8 chunks of 4 dh blocks, see
    ///        FLM_Xclbin/Qwen3_5/gated_attention_prefill_chunked), not the per 16 keys layout of attn_bfp16.xclbin
    bool kv_chunked = false;
    /// Each attention layer keeps a bfp16 copy of its K (and V^T) rows, kv_bfp16_rows rows per kv head; rows
    /// [0, valid) are already converted from the bf16 cache.
    struct kv_bfp16_t {
        buffer<u8> tiles;
        int rows = 0;
        int valid = 0;
    };
    buffer<u8> qb;
    int qb_rows = 0;
    std::vector<kv_bfp16_t> kv_bfp16;
    int kv_bfp16_rows = 0;
    int mha_kv_rows_old = -1;
    int mha_seq_L_begin = -1, mha_seq_L_padded = -1;
#endif

    buffer<bf16> q;
    buffer<bf16> k;
    buffer<bf16> v;
    buffer<bf16> g;
    buffer<bf16> g_sigmoid;

    minicpm_v_attn_block_prefill_context(
        minicpm_v_desc* desc,
        minicpm_v_4_7_npu_sequence* seq_gen,
        minicpm_v_common_buffers* common_bufs,
        npu_xclbin_manager* npu,
        npu_app_manager* gemm_app_manager,
        npu_app_manager* mha_app_manager,
        void* attn_reconfig = nullptr,
        bool attn_bfp16 = false,
        bool attn_chunked = false
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs), npu(npu){
#if MINICPM_V_ATTN_ON_NPU && MINICPM_V_MM_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
        this->reconf = static_cast<minicpm_v_attn_reconfig*>(attn_reconfig);
        this->gemm_mgr = gemm_app_manager;
        this->mha_mgr = mha_app_manager;
        if (this->reconf != nullptr && this->reconf->built) { this->mha_sc = gemm_app_manager->create_app(); this->mha_sc_valid = true; }
#else
        (void)attn_reconfig;
#endif
#if MINICPM_V_ATTN_BFP16
        this->mha_bfp16 = attn_bfp16;
        this->kv_chunked = attn_chunked;
#else
        (void)attn_bfp16; (void)attn_chunked;
#endif
#if MINICPM_V_MM_ON_NPU
        this->q_proj = gemm_app_manager->create_app();
        this->k_proj = gemm_app_manager->create_app();
        this->v_proj = gemm_app_manager->create_app();
        this->g_proj = gemm_app_manager->create_app();
        this->o_proj = gemm_app_manager->create_app();
#endif
#if MINICPM_V_ATTN_ON_NPU
        this->mha = mha_app_manager->create_app();
#endif
    }

    /// @brief grow the scratch and regenerate the array sequences for this L / L_begin
    void reserve(int L_padded, int L_begin, int L) {
#if MINICPM_V_MM_ON_NPU
        const int seq_key = (int)seq_gen->mm_seq_key(L_padded, L);   // L_padded, or also the rows a runtime-m mm covers
        if (mm_seq_key_old != seq_key){
            seq_gen->gen_dequant_mm_512(this->q_proj.seq(), L_padded, desc->D, desc->DQ, desc->get_attn_q_offset(), 0, 0, 0, L);
            seq_gen->gen_dequant_mm_512(this->k_proj.seq(), L_padded, desc->D, desc->DK, desc->get_attn_k_offset(), 0, 0, 0, L);
            seq_gen->gen_dequant_mm_512(this->v_proj.seq(), L_padded, desc->D, desc->DV, desc->get_attn_v_offset(), 0, 0, 0, L);
            seq_gen->gen_dequant_mm_512(this->g_proj.seq(), L_padded, desc->D, desc->DQ, desc->get_attn_g_offset(), 0, 0, 0, L);
            seq_gen->gen_dequant_mm_512(this->o_proj.seq(), L_padded, desc->DQ, desc->D, desc->get_attn_o_offset(), 0, 0, 0, L);
            mm_seq_key_old = seq_key;
        }
#endif
        if (L_padded_old != L_padded){
            q  = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->DQ);
            k  = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->DK);
            v  = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->DV);
            g  = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->DQ);
            g_sigmoid  = buffer<bf16>((size_t)L_padded * desc->DQ);
        }
#if MINICPM_V_ATTN_ON_NPU
        if (!attn_bfp16_now() && ((mha_seq_begin != L_begin) || (mha_seq_padded != L_padded))) {
            seq_gen->gen_mha_engine_seq(mha.seq(), L_begin, L_padded);
            mha_seq_begin = L_begin;
            mha_seq_padded = L_padded;
        }
#endif
        L_padded_old = L_padded;
        L_begin_old = L_begin;
    }

    /// @brief q / gate / k / v projections, q_norm + k_norm + rope on q and k, sigmoid of
    ///        the gate into g_sigmoid. On the array the host stages overlap the next run.
    bool _qkvg_proj_npu(int L, buffer<u8>& proj_weights, bf16* q_norm_weight, bf16* k_norm_weight, std::vector<int>& input_idx) {
#if MINICPM_V_MM_ON_NPU
        q_proj(q, common_bufs->x, proj_weights);

        minicpm_v_cpu::scratch_pull_rows(q, 0, L, desc->DQ);
        MINICPM_V_DUMP("attn_q_raw", q.data(), L, desc->DQ);
#ifdef ENABLE_DEBUG
        //!< q and k are rotated in place below, so the raw projection has to be scored here
        _xref_proj("attn_q", L, q.data(), desc->get_attn_q_offset(), proj_weights, desc->DQ);
#endif
        auto g_run = g_proj.create_run(g, common_bufs->x, proj_weights);
        g_run.start();
        cpu_func::_rope_rms_batch(q, desc->DQ, input_idx, L, desc->num_heads, q_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        minicpm_v_cpu::scratch_push_rows(q, 0, L, desc->DQ);   // also on the host-attention path: flushes the rotated q before the array rewrites it
        flm::npu::wait_for_completion(g_run, "g_run");
        minicpm_v_cpu::scratch_pull_rows(g, 0, L, desc->DQ);
        MINICPM_V_DUMP("attn_gate_raw", g.data(), L, desc->DQ);
#ifdef ENABLE_DEBUG
        _xref_proj("attn_gate", L, g.data(), desc->get_attn_g_offset(), proj_weights, desc->DQ);
#endif

        auto k_run = k_proj.create_run(k, common_bufs->x, proj_weights);
        k_run.start();
        cpu_func::_sigmoid(g_sigmoid.data(), g.data(), L * desc->DQ);
        flm::npu::wait_for_completion(k_run, "k_run");
        minicpm_v_cpu::scratch_pull_rows(k, 0, L, desc->DK);
#ifdef ENABLE_DEBUG
        _xref_proj("attn_k", L, k.data(), desc->get_attn_k_offset(), proj_weights, desc->DK);
#endif

        auto v_run = v_proj.create_run(v, common_bufs->x, proj_weights);
        v_run.start();
        cpu_func::_rope_rms_batch(k, desc->DK, input_idx, L, desc->num_kv_heads, k_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        minicpm_v_cpu::scratch_push_rows(k, 0, L, desc->DK);
        flm::npu::wait_for_completion(v_run, "v_run");
        minicpm_v_cpu::scratch_pull_rows(v, 0, L, desc->DV);
#ifdef ENABLE_DEBUG
        _xref_proj("attn_v", L, v.data(), desc->get_attn_v_offset(), proj_weights, desc->DV);
#endif
        return true;
#else
        (void)L; (void)proj_weights; (void)q_norm_weight; (void)k_norm_weight; (void)input_idx;
        return false;
#endif
    }

    void _qkvg_proj_host(int L, buffer<u8>& proj_weights, bf16* q_norm_weight, bf16* k_norm_weight, std::vector<int>& input_idx) {
        const u8*   w = proj_weights.data();
        const bf16* x = common_bufs->x.data();
        const int   D = desc->D;
        minicpm_v_cpu::dequant_mm(q.data(), desc->DQ, w + desc->get_attn_q_offset(), x, D, L, D, desc->DQ, minicpm_v_cpu::BODY_LAYOUT);
        MINICPM_V_DUMP("attn_q_raw", q.data(), L, desc->DQ);
        minicpm_v_cpu::dequant_mm(g.data(), desc->DQ, w + desc->get_attn_g_offset(), x, D, L, D, desc->DQ, minicpm_v_cpu::BODY_LAYOUT);
        MINICPM_V_DUMP("attn_gate_raw", g.data(), L, desc->DQ);
        minicpm_v_cpu::dequant_mm(k.data(), desc->DK, w + desc->get_attn_k_offset(), x, D, L, D, desc->DK, minicpm_v_cpu::BODY_LAYOUT);
        minicpm_v_cpu::dequant_mm(v.data(), desc->DV, w + desc->get_attn_v_offset(), x, D, L, D, desc->DV, minicpm_v_cpu::BODY_LAYOUT);
        cpu_func::_rope_rms_batch(q, desc->DQ, input_idx, L, desc->num_heads, q_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        cpu_func::_rope_rms_batch(k, desc->DK, input_idx, L, desc->num_kv_heads, k_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        cpu_func::_sigmoid(g_sigmoid.data(), g.data(), L * desc->DQ);
        minicpm_v_cpu::scratch_push(q);
        minicpm_v_cpu::scratch_push(k);
        minicpm_v_cpu::scratch_push(v);
        minicpm_v_cpu::scratch_push(g);
    }

#if MINICPM_V_ATTN_ON_NPU && MINICPM_V_MM_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
    /// one submission: reconfigure the array to attn, attention, reconfigure back to mm. The firmware cannot run another process's command
    /// in between (that would reload the mm PDI and lose the attention configuration). Returns false when the round trip failed: the single-
    /// context path is then off for good and the caller runs this layer's attention through attn.xclbin's own hw_context, which is also the
    /// recovery (the firmware loads the attn PDI, so the next mm run reloads the mm PDI instead of trusting a half-reconfigured array).
    template <class MakeRun>
    bool _submit_single_ctx(MakeRun&& make_attn_run) {
        try {
            xrt::runlist rl = gemm_mgr->create_runlist();
#ifdef FLM_MINICPM_RECONFIG_FAULT_INJECT
            if (reconf->fault_at > 0 && ++reconf->calls == reconf->fault_at) reconf->to_attn_unquiesced.add_to(rl);
            else
#endif
            reconf->to_attn.add_to(rl);
            xrt::run attn_run = make_attn_run();
            rl.add(attn_run);
            reconf->to_mm.add_to(rl);
            rl.execute();
            // the driver ends a hung command by itself (ERT timeout -> command_error); this bound is a backstop for a wait that never returns
            if (rl.wait(std::chrono::milliseconds(30000)) == std::cv_status::timeout) throw std::runtime_error("round trip did not complete in 30 s");
            reconf->to_attn.release();
            reconf->to_mm.release();
            if (reconf->trips.fetch_add(1) == 0) header_print_r("INFO", "minicpm_v: attention runs through single-context reconfiguration (first round trip done)");
            return true;
        } catch (const std::exception& e) {
            reconf->to_attn.release();
            reconf->to_mm.release();
            minicpm_reconfig::disable(std::string("attention round trip failed: ") + e.what());
            return false;
        }
    }

    bool _attend_single_ctx(buffer<bf16>& cache) {
        if (sc_L_begin != L_begin_old || sc_L_padded != L_padded_old) {
            seq_gen->gen_mha_engine_seq(mha_sc.seq(), L_begin_old, L_padded_old);
            sc_L_begin = L_begin_old; sc_L_padded = L_padded_old;
        }
        return _submit_single_ctx([&] { return mha_sc.create_run(common_bufs->after_mix, q, cache); });
    }

#if MINICPM_V_ATTN_BFP16
    /// the same for the bfp16 attention kernels (attn_bfp16.xclbin / attn_bfp16c.xclbin): the arguments are q and the layer's K / V^T as
    /// bfp16 tiles, the sequence depends on (L_begin, L_padded, kv rows); q tiles and K / V^T tiles are already converted (forward())
    bool _attend_single_ctx_bfp16(int layer_idx) {
        if (sc_L_begin != L_begin_old || sc_L_padded != L_padded_old || sc_kv_rows != kv_bfp16_rows) {
            seq_gen->gen_mha_engine_seq_bfp16(mha_sc.seq(), L_begin_old, L_padded_old, kv_bfp16_rows);
            sc_L_begin = L_begin_old; sc_L_padded = L_padded_old; sc_kv_rows = kv_bfp16_rows;
        }
        return _submit_single_ctx([&] { return mha_sc.create_run(common_bufs->after_mix, qb, kv_bfp16[layer_idx].tiles); });
    }
#endif

    /// Known-answer test of the single-context path on the real overlays, run once at load (after the weights are in) on an NPU this process
    /// has to itself. Attention of 512 deterministic rows against the real K/V cache buffer, k projection with this layer's real weights:
    ///   reference attention through attn.xclbin's own hw_context and reference k projection (mm context)
    ///   then 4 round trips (one of them twice in a row), each followed by the k projection again, and a last attention through attn.xclbin
    /// Every output must equal its reference bit for bit (same overlays, same sequences, deterministic arithmetic). Any exception, timeout or
    /// mismatch disables the feature for the process. The cache rows the test wrote (the first 512) are reported in `kv_rows`, so the
    /// caller zeroes them again (clear_context does that for rows below the high-water mark).
    bool single_ctx_self_test(buffer<u8>& proj_weights, buffer<bf16>& cache, int& kv_rows) {
        if (reconf == nullptr || !reconf->built || minicpm_reconfig::disabled()) return false;
        const std::string ver = minicpm_reconfig::versions();
        if (!minicpm_reconfig::npu_exclusive()) {
            minicpm_reconfig::disable("another process has hw contexts on the NPU (" + ver + ")");
            return false;
        }
        constexpr int LT = 512;
        const int D = desc->D, DQ = desc->DQ, DK = desc->DK, DV = desc->DV;
        auto rnd = [](uint64_t& st) { st = st * 6364136223846793005ULL + 1442695040888963407ULL; return (float)((int)(st >> 40) & 0xFFFF) / 32768.0f - 1.0f; };
        try {
            uint64_t st = 0x9E3779B97F4A7C15ULL;
            buffer<bf16> tx = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)LT * D);
            buffer<bf16> tq = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)LT * DQ);
            buffer<bf16> o_ref = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)LT * DQ);
            buffer<bf16> o_sc = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)LT * DQ);
            buffer<bf16> k_ref = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)LT * DK);
            buffer<bf16> k_sc = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)LT * DK);
            for (size_t i = 0; i < tx.size(); i++) tx.data()[i] = (bf16)(0.5f * rnd(st));
            for (size_t i = 0; i < tq.size(); i++) tq.data()[i] = (bf16)rnd(st);
            for (int r = 0; r < LT; r++) {
                bf16* kp = cache.data() + (size_t)r * DK;
                bf16* vp = cache.data() + cache.size() / 2 + (size_t)r * DV;
                for (int j = 0; j < DK; j++) kp[j] = (bf16)rnd(st);
                for (int j = 0; j < DV; j++) vp[j] = (bf16)rnd(st);
            }
            minicpm_v_cpu::scratch_push(tx);
            minicpm_v_cpu::scratch_push(tq);
            mcv_npu::sync_to_device_range(cache, 0, (size_t)LT * DK);
            mcv_npu::sync_to_device_range(cache, cache.size() / 2, (size_t)LT * DV);
            kv_rows = std::max(kv_rows, LT);

            npu_app t_mm = gemm_mgr->create_app();
            npu_app t_mm_small = gemm_mgr->create_app();   // runtime-m mm (mm_rtm.xclbin): a reduced-row dispatch rewrites memtile registers; with the fixed mm it is the same dispatch
            npu_app t_attn = mha_mgr->create_app();
            npu_app t_attn_sc = gemm_mgr->create_app();
            seq_gen->gen_dequant_mm_512(t_mm.seq(), LT, D, DK, desc->get_attn_k_offset(), 0, 0);
            seq_gen->gen_dequant_mm_512(t_mm_small.seq(), LT, D, DK, desc->get_attn_k_offset(), 0, 0, 0, 64);
            bool bfp16_attn = false;
#if MINICPM_V_ATTN_BFP16
            bfp16_attn = mha_bfp16;
#endif
            if (!bfp16_attn) {
                seq_gen->gen_mha_engine_seq(t_attn.seq(), 0, LT);
                seq_gen->gen_mha_engine_seq(t_attn_sc.seq(), 0, LT);
            }
#if MINICPM_V_ATTN_BFP16
            else {
                // the bfp16 attention kernels take q and layer 0's K / V^T as bfp16 tiles made by the engine's own conversion; the test uses the
                // real members and the first attention layer's slot (marked stale again below, so the first request rebuilds it from the cleared cache)
                reserve(LT, 0, LT);
                std::memcpy(q.data(), tq.data(), (size_t)LT * DQ * sizeof(bf16));
                minicpm_v_cpu::scratch_push(q);   // q is written by the array (q projection): a host write must be flushed (notes/tricks 2026-11-05)
                convert_q(LT, LT);
                update_kv_bfp16(0, cache, 0, LT, LT);
                seq_gen->gen_mha_engine_seq_bfp16(mha.seq(), 0, LT, kv_bfp16_rows);
                seq_gen->gen_mha_engine_seq_bfp16(mha_sc.seq(), 0, LT, kv_bfp16_rows);
            }
#endif
            auto ref_attn = [&] {
#if MINICPM_V_ATTN_BFP16
                if (bfp16_attn) { mha(o_ref, qb, kv_bfp16[0].tiles); return; }
#endif
                t_attn(o_ref, tq, cache);
            };
            auto make_sc_run = [&]() -> xrt::run {
#if MINICPM_V_ATTN_BFP16
                if (bfp16_attn) return mha_sc.create_run(o_sc, qb, kv_bfp16[0].tiles);
#endif
                return t_attn_sc.create_run(o_sc, tq, cache);
            };

            auto grab = [&](buffer<bf16>& b, int rows, int ld) {
                minicpm_v_cpu::scratch_pull_rows(b, 0, rows, ld);
                return std::vector<bf16>(b.data(), b.data() + (size_t)rows * ld);
            };
            auto same = [](const std::vector<bf16>& a, const std::vector<bf16>& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(bf16)) == 0; };

            t_mm(k_ref, tx, proj_weights);
            const std::vector<bf16> want_k = grab(k_ref, LT, DK);
            std::memset(k_sc.data(), 0xFF, k_sc.size() * sizeof(bf16));
            minicpm_v_cpu::scratch_push(k_sc);
            t_mm_small(k_sc, tx, proj_weights);
            const std::vector<bf16> want_ks = grab(k_sc, LT, DK);
            ref_attn();
            const std::vector<bf16> want_o = grab(o_ref, LT, DQ);

            auto round_trip = [&] {
                xrt::runlist rl = gemm_mgr->create_runlist();
                reconf->to_attn.add_to(rl);
                xrt::run r = make_sc_run();
                rl.add(r);
                reconf->to_mm.add_to(rl);
                rl.execute();
                if (rl.wait(std::chrono::milliseconds(30000)) == std::cv_status::timeout) throw std::runtime_error("self-test round trip did not complete in 30 s");
                reconf->to_attn.release();
                reconf->to_mm.release();
            };
            const int trips[4] = {1, 2, 1, 1};
            time_utils::time_point t0 = time_utils::now();
            int n_trips = 0;
            for (int it = 0; it < 4; it++) {
                for (int j = 0; j < trips[it]; j++) {
                    std::memset(o_sc.data(), 0xFF, o_sc.size() * sizeof(bf16));
                    minicpm_v_cpu::scratch_push(o_sc);
                    round_trip();
                    n_trips++;
                    if (!same(grab(o_sc, LT, DQ), want_o)) throw std::runtime_error("attention output after round trip " + std::to_string(n_trips) + " differs from the two-context result");
                }
                std::memset(k_sc.data(), 0xFF, k_sc.size() * sizeof(bf16));
                minicpm_v_cpu::scratch_push(k_sc);
                t_mm(k_sc, tx, proj_weights);
                if (!same(grab(k_sc, LT, DK), want_k)) throw std::runtime_error("mm output after round trip " + std::to_string(n_trips) + " differs from the reference (array state not restored)");
                std::memset(k_sc.data(), 0xFF, k_sc.size() * sizeof(bf16));
                minicpm_v_cpu::scratch_push(k_sc);
                t_mm_small(k_sc, tx, proj_weights);
                if (!same(grab(k_sc, LT, DK), want_ks)) throw std::runtime_error("reduced-row mm output after round trip " + std::to_string(n_trips) + " differs from the reference");
            }
            const double ms = time_utils::duration_us(t0, time_utils::now()).first / 1000.0;
            std::memset(o_ref.data(), 0xFF, o_ref.size() * sizeof(bf16));
            minicpm_v_cpu::scratch_push(o_ref);
            ref_attn();
            if (!same(grab(o_ref, LT, DQ), want_o)) throw std::runtime_error("attention through the attention overlay's own context after the round trips differs from the reference");
#if MINICPM_V_ATTN_BFP16
            if (bfp16_attn) {
                // leave no trace of the test in the members the requests use
                kv_cache_changed(0);
                mha_seq_L_begin = mha_seq_L_padded = mha_kv_rows_old = -1;
                mha_seq_begin = mha_seq_padded = -1;
                sc_L_begin = sc_L_padded = sc_kv_rows = -1;
            }
#endif
            reconf->validated = true;
            header_print_r("INFO", "minicpm_v: single-context reconfiguration self-test passed (" << n_trips << " round trips + mm checks in " << ms << " ms; " << ver << ")");
            return true;
        } catch (const std::exception& e) {
            minicpm_reconfig::disable(std::string("self-test failed: ") + e.what() + " (" + ver + ")");
            // recovery: a dispatch in attn.xclbin's own context makes the firmware load the attn PDI, so the next mm run reloads the mm PDI
            try {
                buffer<bf16> to = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)LT * DQ);
#if MINICPM_V_ATTN_BFP16
                if (mha_bfp16 && !kv_bfp16.empty() && kv_bfp16[0].rows > 0 && qb_rows == LT) {
                    kv_cache_changed(0);
                    seq_gen->gen_mha_engine_seq_bfp16(mha.seq(), 0, LT, kv_bfp16_rows);
                    mha_seq_L_begin = mha_seq_L_padded = mha_kv_rows_old = -1;
                    sc_L_begin = sc_L_padded = sc_kv_rows = -1;
                    mha(to, qb, kv_bfp16[0].tiles);
                } else
#endif
                {
                    npu_app t_attn = mha_mgr->create_app();
                    buffer<bf16> tq = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)LT * DQ);
                    seq_gen->gen_mha_engine_seq(t_attn.seq(), 0, LT);
                    t_attn(to, tq, cache);
                }
            } catch (const std::exception& e2) {
                header_print_r("WARNING", "minicpm_v: recovery dispatch after the failed self-test failed too: " << e2.what());
            }
            return false;
        }
    }
#endif

#if MINICPM_V_ATTN_BFP16
    /// \brief whether the array attention of this chunk takes bfp16 tiles
    bool attn_bfp16_now() const { return mha_bfp16; }
#else
    bool attn_bfp16_now() const { return false; }
#endif

#if MINICPM_V_ATTN_BFP16
    /// the host changed the bf16 KV cache from row from_row on (restore, clear, write_state): convert those rows again
    void kv_cache_changed(int from_row) {
        for (auto& kvb : kv_bfp16) {
            kvb.valid = std::min(kvb.valid, from_row);
        }
    }

    /// q (already rotated) -> bfp16 tiles [head][L_padded / 16][...]
    void convert_q(int L, int L_padded) {
        const int DH = desc->DH;
        const int DQ = desc->DQ;
        if (qb_rows != L_padded) {
            qb = mha.create_bo_buffer<u8>((size_t)L_padded * DQ * 9 / 8);
            memset(qb.data(), 0, qb.size());   // rows past L: finite tiles whatever the previous chunk left
            qb.sync_to_device();
            qb_rows = L_padded;
        }
        const int groups = L_padded / 16;
        const int real_groups = (L + 15) / 16;
        const int heads = (int)desc->num_heads;
        // one flat loop over (head, group): MSVC's /openmp (OpenMP 2.0) has no collapse
        #pragma omp parallel for num_threads(minicpm_v_cpu::prefill_threads()) schedule(static)
        for (int i = 0; i < heads * real_groups; i++) {
            const int h = i / real_groups;
            const int group = i % real_groups;
            uint8_t* out = qb.data() + ((size_t)h * L_padded + group * 16) * DH * 9 / 8;
            if (group * 16 + 16 <= L) {
                cpu_func::_bfp16_tile_group(out, q.data() + (size_t)group * 16 * DQ + h * DH, DH, DQ);
            }
            else {
                // the last group is partly padding: its rows past L come from a zeroed scratch
                alignas(64) bf16 tmp[16 * 256];
                const int n = L - group * 16;
                memset(tmp, 0, sizeof(tmp));
                for (int r = 0; r < n; r++) {
                    memcpy(tmp + r * DH, q.data() + (size_t)(group * 16 + r) * DQ + h * DH, DH * sizeof(bf16));
                }
                cpu_func::_bfp16_tile_group(out, tmp, DH, DH);
            }
        }
        (void)groups;
        // the tiles of rows past L keep what the previous chunk left; only rows [0, real_groups * 16) changed
        for (int h = 0; h < heads; h++) {
            mcv_npu::sync_to_device_range(qb, ((size_t)h * L_padded) * DH * 9 / 8, (size_t)real_groups * 16 * DH * 9 / 8);
        }
    }

    /// bring layer layer_idx's bfp16 K / V^T copy up to date with its bf16 cache, up to the last key block the
    /// kernel reads. Rows past L_begin + L are zero tiles: they are masked, but the 8 keys of a V^T block share an
    /// exponent, so garbage there would cost its valid neighbours precision.
    void update_kv_bfp16(int layer_idx, buffer<bf16>& cache, int L_begin, int L, int L_padded) {
        const int DH = desc->DH;
        const int DK = desc->DK;
        const int kv_heads = desc->num_kv_heads;
        const size_t row_bytes = (size_t)DH * 9 / 8;
        const int needed = (L_begin + L_padded + 127) / 128 * 128;
        const int cache_rows = (int)(cache.size() / 2 / DK);
        if (needed > kv_bfp16_rows) {
            // grow by 1.5x in 4096-row steps, but not past the bf16 cache (rounded up to a 128-row key block)
            const int grown = (std::max(needed, kv_bfp16_rows * 3 / 2) + 4095) / 4096 * 4096;
            kv_bfp16_rows = std::max(needed, std::min(grown, (cache_rows + 127) / 128 * 128));
        }
        if ((int)kv_bfp16.size() <= layer_idx) {
            kv_bfp16.resize(layer_idx + 1);
        }
        kv_bfp16_t& kvb = kv_bfp16[layer_idx];
        if (kvb.rows != kv_bfp16_rows) {
            // grown (or first use): the converted rows move to the new buffer
            buffer<u8> bigger = mha.create_bo_buffer<u8>((size_t)2 * kv_heads * kv_bfp16_rows * row_bytes);
            int keep = (kvb.rows > 0) ? std::min(kvb.valid, kvb.rows) : 0;
            // chunk-major tiles interleave the 8 key groups of a 128-key block: move whole blocks
            if (kv_chunked) keep = std::min((keep + 127) / 128 * 128, kvb.rows);
            for (int part = 0; part < 2 * kv_heads && keep > 0; part++) {
                memcpy(bigger.data() + (size_t)part * kv_bfp16_rows * row_bytes, kvb.tiles.data() + (size_t)part * kvb.rows * row_bytes, (size_t)keep * row_bytes);
                mcv_npu::sync_to_device_range(bigger, (size_t)part * kv_bfp16_rows * row_bytes, (size_t)keep * row_bytes);
            }
            kvb.tiles = std::move(bigger);
            kvb.rows = kv_bfp16_rows;
            kvb.valid = (kvb.valid > 0) ? std::min(kvb.valid, keep) : 0;
        }

        const int group_begin = std::min(kvb.valid, L_begin) / 16;
        const int group_end = needed / 16;
        const int groups = group_end - group_begin;
        const int end_row = L_begin + L;   // rows from here on are zero tiles
        // rows [group_begin * 16, L_begin) were written by the array (decode) or by an earlier host write the cache
        // lines of which may be stale here: read them back before converting
        if (L_begin > group_begin * 16) {
            const size_t rows_back = (size_t)(L_begin - group_begin * 16);
            mcv_npu::sync_from_device_range(cache, (size_t)group_begin * 16 * DK, rows_back * DK);
            mcv_npu::sync_from_device_range(cache, cache.size() / 2 + (size_t)group_begin * 16 * DK, rows_back * DK);
        }
        const bf16* k_cache = cache.data();
        const bf16* v_cache = cache.data() + cache.size() / 2;
        // one flat loop over (kv head, group): MSVC's /openmp (OpenMP 2.0) has no collapse
        #pragma omp parallel for num_threads(minicpm_v_cpu::prefill_threads()) schedule(static)
        for (int i = 0; i < kv_heads * groups; i++) {
            const int h = i / groups;
            const int g = group_begin + i % groups;
            uint8_t* k_out = kvb.tiles.data() + ((size_t)h * kvb.rows + g * 16) * row_bytes;
            uint8_t* v_out = kvb.tiles.data() + ((size_t)(kv_heads + h) * kvb.rows + g * 16) * row_bytes;
            // V^T tiles of this key group: in place (per 16 keys layout) or through a scratch group that is scattered into the
            // chunk-major layout: block b = g / 8, key group kt = g % 8, chunk c, dh block pair dp -> 288 B at
            // b * 36864 + c * 4608 + dp * 2304 + kt * 288, from the group's dh blocks 4c + 2dp, 4c + 2dp + 1
            alignas(64) uint8_t vt_group[4608];
            uint8_t* vt_dst = v_out;
            if (kv_chunked) vt_dst = vt_group;
            auto scatter_v = [&]() {
                if (!kv_chunked) return;
                uint8_t* block = kvb.tiles.data() + ((size_t)(kv_heads + h) * kvb.rows + (size_t)(g / 8) * 128) * row_bytes;
                for (int c = 0; c < 8; c++)
                    for (int dp = 0; dp < 2; dp++)
                        memcpy(block + (size_t)c * 4608 + (size_t)dp * 2304 + (size_t)(g % 8) * 288, vt_group + (size_t)(4 * c + 2 * dp) * 144, 288);
            };
            if (g * 16 >= end_row || g * 16 + 16 > cache_rows) {
                memset(k_out, 0, 16 * row_bytes);
                memset(vt_dst, 0, 16 * row_bytes);
            }
            else if (g * 16 + 16 <= end_row) {
                cpu_func::_bfp16_tile_group(k_out, k_cache + (size_t)g * 16 * DK + h * DH, DH, DK);
                cpu_func::_bfp16_vt_tile_group(vt_dst, v_cache + (size_t)g * 16 * DK + h * DH, DH, DK);
            }
            else {
                alignas(64) bf16 tmp[16 * 256];
                const int n = end_row - g * 16;
                memset(tmp, 0, sizeof(tmp));
                for (int r = 0; r < n; r++) memcpy(tmp + r * DH, k_cache + (size_t)(g * 16 + r) * DK + h * DH, DH * sizeof(bf16));
                cpu_func::_bfp16_tile_group(k_out, tmp, DH, DH);
                memset(tmp, 0, sizeof(tmp));
                for (int r = 0; r < n; r++) memcpy(tmp + r * DH, v_cache + (size_t)(g * 16 + r) * DK + h * DH, DH * sizeof(bf16));
                cpu_func::_bfp16_vt_tile_group(vt_dst, tmp, DH, DH);
            }
            scatter_v();
        }
        // chunk-major V^T tiles: a key group's bytes are spread over its 128-key block, so flush whole blocks
        const int sync_begin = kv_chunked ? group_begin * 16 / 128 * 128 : group_begin * 16;
        const int sync_end = kv_chunked ? (group_end * 16 + 127) / 128 * 128 : group_end * 16;
        for (int part = 0; part < 2 * kv_heads; part++) {
            mcv_npu::sync_to_device_range(kvb.tiles, ((size_t)part * kvb.rows + sync_begin) * row_bytes, (size_t)(sync_end - sync_begin) * row_bytes);
        }
        kvb.valid = end_row;
    }

    /// the sequence of this chunk's attention (bfp16 kernel): depends on (L_begin, L_padded, kv_bfp16_rows)
    void gen_mha_seq_bfp16(int L_begin, int L_padded) {
        if (mha_seq_L_begin != L_begin || mha_seq_L_padded != L_padded || mha_kv_rows_old != kv_bfp16_rows) {
            seq_gen->gen_mha_engine_seq_bfp16(mha.seq(), L_begin, L_padded, kv_bfp16_rows);
            mha_seq_L_begin = L_begin;
            mha_seq_L_padded = L_padded;
            mha_kv_rows_old = kv_bfp16_rows;
        }
    }
#endif

    /// @brief causal attention of q against cache rows [0, L_begin + L) into common after_mix
    bool _attend_npu(int L, buffer<bf16>& cache, int layer_idx = 0) {
#if MINICPM_V_ATTN_ON_NPU
#if MINICPM_V_ATTN_BFP16
        if (mha_bfp16) {
#if MINICPM_V_MM_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
            if (!(mha_sc_valid && reconf->usable() && _attend_single_ctx_bfp16(layer_idx)))
#endif
                mha(common_bufs->after_mix, qb, kv_bfp16[layer_idx].tiles);
            minicpm_v_cpu::scratch_pull_rows(common_bufs->after_mix, 0, L, desc->DQ);
            return true;
        }
#endif
        (void)layer_idx;
#if MINICPM_V_MM_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
        if (!(mha_sc_valid && reconf->usable() && _attend_single_ctx(cache)))
#endif
            mha(common_bufs->after_mix, q, cache);

        minicpm_v_cpu::scratch_pull_rows(common_bufs->after_mix, 0, L, desc->DQ);
        return true;
#else
        (void)L; (void)cache;
        return false;
#endif
    }

    void _attend_host(int L, buffer<bf16>& cache, int L_begin) {
        // fill_kv_cache's layout: K rows from 0, V rows from the half-way point
        const bf16* k_cache = cache.data();
        const bf16* v_cache = cache.data() + cache.size() / 2;
        minicpm_v_cpu::attention(common_bufs->after_mix.data(), q.data(), k_cache, v_cache,
                                 L, L_begin + L, desc->num_heads, desc->num_kv_heads, desc->DH,
                                 L_begin, 0, desc->DK, desc->DV, 1.0f / sqrtf((float)desc->DH));
        // flush: after_mix is a device-written buffer (the array's mha output, the linear blocks' o_proj input),
        // so the dirty lines the host just wrote must not linger in the cache and later overwrite what the
        // array writes there; only the first L rows were written
        minicpm_v_cpu::scratch_push_rows(common_bufs->after_mix, 0, L, desc->DQ);
    }

    /// @brief o_proj of the gated attention (g) into common x
    bool _o_proj_npu(int L, buffer<u8>& proj_weights) {
#if MINICPM_V_MM_ON_NPU
        o_proj(common_bufs->x, g, proj_weights);

        minicpm_v_cpu::scratch_pull_rows(common_bufs->x, 0, L, desc->D);
        return true;
#else
        (void)L; (void)proj_weights;
        return false;
#endif
    }

    void _o_proj_host(int L, buffer<u8>& proj_weights) {
        minicpm_v_cpu::dequant_mm(common_bufs->x.data(), desc->D,
                                  proj_weights.data() + desc->get_attn_o_offset(),
                                  g.data(), desc->DQ, L, desc->DQ, desc->D, minicpm_v_cpu::BODY_LAYOUT);
        minicpm_v_cpu::scratch_push_rows(common_bufs->x, 0, L, desc->D);
    }

    buffer<bf16> forward(int L, buffer<u8>& proj_weights, buffer<bf16>& rope_rms_weights, buffer<bf16>& cache, std::vector<int>& input_idx, int L_begin, int attn_layer = 0) {
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        this->reserve(L_padded, L_begin, L);

        bf16* q_norm_weight = rope_rms_weights.data() + desc->RoPE_DIM;
        bf16* k_norm_weight = rope_rms_weights.data() + desc->RoPE_DIM + desc->DH;

        if (!_qkvg_proj_npu(L, proj_weights, q_norm_weight, k_norm_weight, input_idx)) {
            _qkvg_proj_host(L, proj_weights, q_norm_weight, k_norm_weight, input_idx);
        }
        MINICPM_V_DUMP("attn_q_rope", q.data(), L, desc->DQ);
        MINICPM_V_DUMP("attn_k_rope", k.data(), L, desc->DK);
        MINICPM_V_DUMP("attn_v", v.data(), L, desc->DV);

        PREFILL_PROF_TICK("attn:qkvg_proj");

        fill_kv_cache(k, v, cache, L_begin, L);
#if MINICPM_V_ATTN_BFP16
        if (attn_bfp16_now()) {
            convert_q(L, L_padded);
            update_kv_bfp16(attn_layer, cache, L_begin, L, L_padded);
            gen_mha_seq_bfp16(L_begin, L_padded);
        }
#endif
        PREFILL_PROF_TICK("attn:fill_kv_cache");

        if (!_attend_npu(L, cache, attn_layer)) {
            _attend_host(L, cache, L_begin);
        }
        MINICPM_V_DUMP("attn_out", common_bufs->after_mix.data(), L, desc->DQ);
#ifdef ENABLE_DEBUG
        _xref_attend(L, cache, L_begin);
#endif
        PREFILL_PROF_TICK("attn:mha");

        cpu_func::_elementwise_mul(g.data(), g_sigmoid.data(), common_bufs->after_mix.data(), L * desc->DQ);
        minicpm_v_cpu::scratch_push_rows(g, 0, L, desc->DQ);
        MINICPM_V_DUMP("attn_gated", g.data(), L, desc->DQ);

        if (!_o_proj_npu(L, proj_weights)) {
            _o_proj_host(L, proj_weights);
        }
        MINICPM_V_DUMP("attn_o_proj", common_bufs->x.data(), L, desc->D);
#ifdef ENABLE_DEBUG
        _xref_o_proj(L, proj_weights);
#endif
        PREFILL_PROF_TICK("attn:o_proj");

        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);
        return valid;
    }

    void fill_kv_cache(buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, buffer<bf16>& cache, int L_begin, int L){
        tensor_2d<bf16> k_cache(buffer_k, desc->DK);
        tensor_2d<bf16> v_cache(buffer_v, desc->DV);

#if defined(FLM_USE_HRX)
        cache.sync_from_device();
#endif
        // XRT: flush only the rows written (the buffer is 64 MB), so there is nothing to read
        // back first either: rows outside the range are not touched on either side

        bf16* k_ptr = cache.data() + L_begin * desc->DK;
        bf16* v_ptr = cache.data() + L_begin * desc->DV + cache.size() / 2;

        for (int l = 0; l < L; l++){
            memcpy(k_ptr, k_cache[l].data(), desc->DK * sizeof(bf16));
            memcpy(v_ptr, v_cache[l].data(), desc->DV * sizeof(bf16));
            k_ptr += desc->DK;
            v_ptr += desc->DV;
        }
#if defined(FLM_USE_HRX)
        cache.sync_to_device();
#else
        mcv_npu::sync_to_device_range(cache, (size_t)L_begin * desc->DK, (size_t)L * desc->DK);
        mcv_npu::sync_to_device_range(cache, cache.size() / 2 + (size_t)L_begin * desc->DV, (size_t)L * desc->DV);
#endif
    }
};

struct minicpm_v_mlp_prefill_context{
    minicpm_v_desc* desc;
    minicpm_v_4_7_npu_sequence* seq_gen;
    minicpm_v_common_buffers* common_bufs;
    npu_xclbin_manager* npu;

    int L_old = -1;
    int seq_padded_old = -1;   //!< mm_seq_key the whole-L array sequences were generated for
    /// Whether this call's up/gate projection is the fused GLU one (seq_gen->mm_glu and enough rows: the fused epilogue saves
    /// nothing on short prompts and differs numerically from the host GLU, so it is used from FLM_MINICPM_GLU_MIN_ROWS padded rows
    /// up, default 1024). A function of L_padded only, so the sequence key and the scratch sizes stay consistent.
    bool glu_now = false;
    static int glu_min_rows() {
        static const int v = [] { const char* e = std::getenv("FLM_MINICPM_GLU_MIN_ROWS"); return e && *e ? std::max(0, std::atoi(e)) : 1024; }();
        return v;
    }
#ifdef ENABLE_DEBUG
    //!< set by minicpm_v_prefill_context::forward() before every call
    int layer_idx = -1;
    //!< non-null in a debug build; whether it scores anything is up to enabled(layer_idx)
    minicpm_v_prefill_xref* xref = nullptr;
#endif

#if MINICPM_V_MM_ON_NPU
    npu_app upgate_proj;
    npu_app down_proj;
#endif

    /// @brief [L, 2 * IS]: interleaved 512-row up/gate slabs, raw, from the array run; or
    ///        de-interleaved [ up | silu(gate) ] rows from the host (dequant_mm_upgate)
    buffer<bf16> upgate;
    buffer<bf16> hid;

    minicpm_v_mlp_prefill_context(
        minicpm_v_desc* desc,
        minicpm_v_4_7_npu_sequence* seq_gen,
        minicpm_v_common_buffers* common_bufs,
        npu_xclbin_manager* npu,
        npu_app_manager* gemm_app_manager
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs), npu(npu){
#if MINICPM_V_MM_ON_NPU
        upgate_proj = gemm_app_manager->create_app();
        down_proj = gemm_app_manager->create_app();
#endif
    }

    /// @brief grow the scratch and regenerate the array sequences for this L
    /// @param want_seq false when the block-pipelined path runs the projections
    void reserve(int L_padded, bool want_seq = true, int L = 0) {
#if MINICPM_V_MM_ON_NPU
        glu_now = seq_gen->mm_glu && L_padded >= glu_min_rows();
        const int seq_key = (int)seq_gen->mm_seq_key(L_padded, L ? L : L_padded);
        if (want_seq && seq_padded_old != seq_key){
            // fused GLU (mm_rtm2_glu.xclbin): the array writes silu(gate) * up straight into hid, from the re-packed copy of the weights
            if (glu_now) seq_gen->gen_dequant_mm_512(this->upgate_proj.seq(), L_padded, desc->D, desc->intermediate_size * 2, desc->get_upgate_glu_offset(), 0, 3, 0, L);
            else seq_gen->gen_dequant_mm_512(this->upgate_proj.seq(), L_padded, desc->D, desc->intermediate_size * 2, desc->get_upgate_offset(), 0, 0, 0, L);
            seq_gen->gen_dequant_mm_512(this->down_proj.seq(), L_padded, desc->intermediate_size, desc->D, desc->get_down_offset(), 0, 0, 0, L);
            seq_padded_old = seq_key;
        }
#endif
        if (L_old != L_padded){
            L_old = L_padded;
            if (!glu_now) upgate = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->intermediate_size * 2);
            hid = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->intermediate_size);
            memset(hid.data(), 0, hid.size() * sizeof(bf16));
            minicpm_v_cpu::scratch_push(hid);
        }
    }

    /// @brief up/gate projection and the glu, into hid
    bool _upgate_npu(int L, buffer<u8>& proj_weights) {
#if MINICPM_V_MM_ON_NPU
        if (glu_now) {
            // the array leaves silu(gate) * up in hid (the host never touches it: no pull, no GLU, no push)
            this->upgate_proj(hid, this->common_bufs->x, proj_weights);
            return true;
        }
        this->upgate_proj(upgate, this->common_bufs->x, proj_weights);
        minicpm_v_cpu::scratch_pull_rows(upgate, 0, L, (size_t)desc->intermediate_size * 2);

        // hid is written by the host only (no read-back needed); the array reads its first L rows
        // nothing overlaps this stage, so it gets the host kernels' full team
        cpu_func::_exp_glu(hid.data(), upgate.data(), L, desc->intermediate_size, 512, minicpm_v_cpu::prefill_threads());

        minicpm_v_cpu::scratch_push_rows(hid, 0, L, desc->intermediate_size);
        return true;
#else
        (void)L; (void)proj_weights;
        return false;
#endif
    }

    void _upgate_host(int L, buffer<u8>& proj_weights) {
        const int IS = desc->intermediate_size;
        minicpm_v_cpu::dequant_mm_upgate(upgate.data(), proj_weights.data() + desc->get_upgate_offset(),
                                         common_bufs->x.data(), L, desc->D, IS, 512);
        MINICPM_V_DUMP("mlp_up", upgate.data(), L, IS, 2 * IS);
        MINICPM_V_DUMP("mlp_gate", upgate.data() + IS, L, IS, 2 * IS);
        minicpm_v_cpu::glu_mul(hid.data(), upgate.data(), L, IS);
        minicpm_v_cpu::scratch_push_rows(upgate, 0, L, (size_t)IS * 2);
        minicpm_v_cpu::scratch_push_rows(hid, 0, L, IS);
    }

    /// @brief down projection of hid into common x
    bool _down_npu(int L, buffer<u8>& proj_weights) {
#if MINICPM_V_MM_ON_NPU
        this->down_proj(common_bufs->x, hid, proj_weights);

        minicpm_v_cpu::scratch_pull_rows(common_bufs->x, 0, L, desc->D);
        return true;
#else
        (void)L; (void)proj_weights;
        return false;
#endif
    }

    void _down_host(int L, buffer<u8>& proj_weights) {
        minicpm_v_cpu::dequant_mm(common_bufs->x.data(), desc->D,
                                  proj_weights.data() + desc->get_down_offset(),
                                  hid.data(), desc->intermediate_size, L, desc->intermediate_size, desc->D,
                                  minicpm_v_cpu::BODY_LAYOUT);
        minicpm_v_cpu::scratch_push_rows(common_bufs->x, 0, L, desc->D);
    }

#ifdef ENABLE_DEBUG
    /// \brief score the up/gate projection and the GLU that follows it
    /// \note The two backends genuinely differ here, and this is the only check that sees
    ///       it: the array leaves `upgate` in the interleaved 512-row slab layout the
    ///       weights are stored in and cpu_func::_exp_glu walks that directly, while the
    ///       host path de-interleaves first (dequant_mm_upgate) and then glu_mul's. So
    ///       `mlp_upgate` scores the matmul alone and `mlp_glu` scores matmul + GLU.
    void _xref_upgate(int L, buffer<u8>& proj_weights) {
#if MINICPM_V_MM_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        const int IS = desc->intermediate_size;

        std::vector<bf16> want((size_t)rows * IS * 2);
        minicpm_v_cpu::dequant_mm(want.data(), IS * 2, proj_weights.data() + desc->get_upgate_offset(),
                                  common_bufs->x.data(), desc->D, rows, desc->D, IS * 2,
                                  minicpm_v_cpu::BODY_LAYOUT);
        {
            minicpm_v_prefill_xref::guard gu(xref, "mlp_upgate", layer_idx, upgate.data(), IS * 2, IS * 2, rows);
            xref->check("mlp_upgate", layer_idx, upgate.data(), want.data(), IS * 2, IS * 2, rows);
        }

        std::vector<bf16> want_ug((size_t)rows * IS * 2), want_hid((size_t)rows * IS);
        minicpm_v_cpu::dequant_mm_upgate(want_ug.data(), proj_weights.data() + desc->get_upgate_offset(),
                                         common_bufs->x.data(), rows, desc->D, IS, 512);
        minicpm_v_cpu::glu_mul(want_hid.data(), want_ug.data(), rows, IS);
        minicpm_v_prefill_xref::guard gu(xref, "mlp_glu", layer_idx, hid.data(), IS, IS, rows);
        xref->check("mlp_glu", layer_idx, hid.data(), want_hid.data(), IS, IS, rows);
#else
        (void)L; (void)proj_weights;
#endif
    }

    /// \brief score the down projection
    void _xref_down(int L, buffer<u8>& proj_weights) {
#if MINICPM_V_MM_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        std::vector<bf16> want((size_t)rows * desc->D);
        minicpm_v_cpu::dequant_mm(want.data(), desc->D, proj_weights.data() + desc->get_down_offset(),
                                  hid.data(), desc->intermediate_size, rows,
                                  desc->intermediate_size, desc->D, minicpm_v_cpu::BODY_LAYOUT);
        minicpm_v_prefill_xref::guard gu(xref, "mlp_down", layer_idx, common_bufs->x.data(), desc->D, desc->D, rows);
        xref->check("mlp_down", layer_idx, common_bufs->x.data(), want.data(), desc->D, desc->D, rows);
#else
        (void)L; (void)proj_weights;
#endif
    }
#endif // ENABLE_DEBUG

    buffer<bf16> forward(int L, buffer<u8>& proj_weights, buffer<bf16>& rms_weights) {
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        this->reserve(L_padded, true, L);
        PREFILL_PROF_TICK("mlp:seq_gen+alloc");

        if (!_upgate_npu(L, proj_weights)) {
            _upgate_host(L, proj_weights);
        }
#ifdef ENABLE_DEBUG
        _xref_upgate(L, proj_weights);
#endif

        if (!_down_npu(L, proj_weights)) {
            _down_host(L, proj_weights);
        }
        MINICPM_V_DUMP("mlp_down", common_bufs->x.data(), L, desc->D);
#ifdef ENABLE_DEBUG
        _xref_down(L, proj_weights);
#endif

        PREFILL_PROF_TICK("mlp:down_proj");

        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);

        return valid;
    }
};

struct minicpm_v_prefill_context{
    minicpm_v_desc* desc;
    minicpm_v_4_7_npu_sequence* seq_gen;
    npu_xclbin_manager* npu;
    minicpm_v_common_buffers common_bufs;

    std::unique_ptr<minicpm_v_attn_block_prefill_context> attn_block;
    std::unique_ptr<minicpm_v_linear_block_prefill_context> linear_block;
    std::unique_ptr<minicpm_v_mlp_prefill_context> mlp;

    // for xclbins; null for every overlay whose stage is on the host
    npu_app_manager* gemm_app_manager = nullptr;
    std::string mm_xclbin_path;   //!< the mm overlay in use (mm.xclbin or the runtime-m mm_rtm.xclbin)
    std::string mha_xclbin_path;  //!< the attention overlay in use (attn.xclbin, attn_bfp16.xclbin or attn_bfp16c.xclbin)
    npu_app_manager* mha_app_manager = nullptr;
    bool attn_bfp16 = false;   //!< mha_app_manager is attn_bfp16.xclbin or attn_bfp16c.xclbin
    bool attn_chunked = false; //!< ... the latter (chunk-major V^T tiles)
    npu_app_manager * conv1d_app_manager = nullptr;
    npu_app_manager * gate_delta_net_manager = nullptr;
#if MINICPM_V_ATTN_ON_NPU && MINICPM_V_MM_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
    std::unique_ptr<minicpm_v_attn_reconfig> attn_reconfig;
#endif

#ifdef ENABLE_DEBUG
    //!< the cpu/npu cross reference. Constructed in every debug build; whether it scores
    //!< anything is FLM_XREF_LAYERS / DEBUG_LEVEL. See minicpm_v_prefill_xref.hpp.
    std::unique_ptr<minicpm_v_prefill_xref> xref;
#endif

    /// the host rewrote rows [from_row, ...) of the attention layers' bf16 KV cache (restore, clear, write_state,
    /// a resize): the bfp16 copies the array attention reads must be rebuilt from there
    void kv_cache_changed(int from_row) {
#if MINICPM_V_ATTN_BFP16
        if (attn_block) attn_block->kv_cache_changed(from_row);
#else
        (void)from_row;
#endif
    }

    minicpm_v_prefill_context(){}
    minicpm_v_prefill_context(
        npu_xclbin_manager* npu,
        minicpm_v_desc* model_desc,
        minicpm_v_4_7_npu_sequence* seq_gen,
        LM_Config& config
    ) : desc(model_desc), seq_gen(seq_gen), npu(npu) {
#if MINICPM_V_MM_ON_NPU
        {
            // mm_rtm.xclbin (a first-generation runtime-m design, still accepted if a model folder has it): the same mm with the row count chosen per
            // dispatch (see minicpm_v_4_7_npu_sequence::gen_dequant_mm_rtm). Its sequences only work with that xclbin,
            // so the choice is made by which file is there; FLM_MINICPM_MM_RTM=0 forces the fixed 512-row mm.xclbin.
            const std::string xdir = utils::path_join(config.exec_path, "xclbins", config.model_name);
            const std::string rtm = utils::path_join(xdir, "mm_rtm.xclbin");
            // mm_rtm2.xclbin (the same overlay without the GLU epilogue, also accepted): the same overlay with a parallel memtile y join
            // (no serialised 4-core drain), the accumulator zero pass and the byte ping flags removed: bit-identical, 7-30% faster
            // kernel. Preferred over mm_rtm.xclbin when present (its re-arm sequences use other memtile lock values).
            const std::string rtm2 = utils::path_join(xdir, "mm_rtm2.xclbin");
            const char* env = std::getenv("FLM_MINICPM_MM_RTM");
            const bool want = !(env && std::atoi(env) == 0);
            // mm_rtm2_glu.xclbin = FLM_Xclbin/dequant_mm_bfp16_rtm2_glu_512x512x512: mm_rtm2 plus the fused GLU epilogue (rtp_act 3); plain
            // dispatches are bit-identical to mm_rtm2.xclbin, so it serves as the runtime-m overlay with or without GLU. GLU is used when the
            // engine re-packed the up/gate weights (desc->glu_fused, FLM_MINICPM_GLU_FUSED=0 switches that off)
            const std::string rtm2_glu = utils::path_join(xdir, "mm_rtm2_glu.xclbin");
            const bool have_glu_file = std::filesystem::exists(rtm2_glu);
            const bool use_glu = want && desc->glu_fused && have_glu_file;
            const bool use_rtm2 = use_glu || (want && (have_glu_file || std::filesystem::exists(rtm2)));
            const bool use_rtm = use_rtm2 || (want && std::filesystem::exists(rtm));
            this->seq_gen->mm_rtm = use_rtm;
            this->seq_gen->mm_yjoin_par = use_rtm2;
            this->seq_gen->mm_glu = use_glu;
            this->mm_xclbin_path = use_rtm2 ? ((use_glu || have_glu_file) ? rtm2_glu : rtm2)
                                            : (use_rtm ? rtm : utils::path_join(xdir, "mm.xclbin"));
            this->gemm_app_manager = npu->register_xclbin(this->mm_xclbin_path);
        }
#endif
#if MINICPM_V_CONV_ON_NPU
        this->conv1d_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "conv.xclbin"));
#endif
#if MINICPM_V_GDN_ON_NPU
        this->gate_delta_net_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "GateDeltaNet_prefill.xclbin"));
#endif
#if MINICPM_V_ATTN_ON_NPU
        {
            // the bfp16 attention kernel (attn_bfp16.xclbin) when the folder has it; FLM_MINICPM_ATTN_BFP16=0 keeps the
            // original attn.xclbin (q and the KV cache as bf16)
            const std::string dir = utils::path_join(config.exec_path, "xclbins", config.model_name);
            const char* env = std::getenv("FLM_MINICPM_ATTN_BFP16");
            const bool disabled = env != nullptr && std::atoi(env) == 0;
            // attn_bfp16c.xclbin (chunk-major V^T tiles, faster P.V) before attn_bfp16.xclbin (the 2B kernel, V^T per 16 keys)
            const bool have_c = MINICPM_V_ATTN_BFP16 && !disabled && std::filesystem::exists(utils::path_join(dir, "attn_bfp16c.xclbin"));
            const bool have_b = MINICPM_V_ATTN_BFP16 && !disabled && std::filesystem::exists(utils::path_join(dir, "attn_bfp16.xclbin"));
            this->attn_chunked = have_c;
            this->attn_bfp16 = have_c || have_b;
            this->mha_xclbin_path = utils::path_join(dir, have_c ? "attn_bfp16c.xclbin" : (have_b ? "attn_bfp16.xclbin" : "attn.xclbin"));
            this->mha_app_manager = npu->register_xclbin(this->mha_xclbin_path);
        }
#if MINICPM_V_MM_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
        // the single-context chains replay the overlays in use (mm.xclbin / mm_rtm.xclbin, attn.xclbin / attn_bfp16.xclbin / attn_bfp16c.xclbin)
        this->attn_reconfig = std::make_unique<minicpm_v_attn_reconfig>();
        this->attn_reconfig->init(npu, gemm_app_manager, mm_xclbin_path, mha_xclbin_path);
#endif
#endif

        // this is to ensure they have same out projection
        assert(desc->Linear_DV == desc->DQ);

        this->linear_block = std::make_unique<minicpm_v_linear_block_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            npu,
            gemm_app_manager,
            conv1d_app_manager,
            gate_delta_net_manager
        );

        this->attn_block = std::make_unique<minicpm_v_attn_block_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            npu,
            gemm_app_manager,
            mha_app_manager,
#if MINICPM_V_ATTN_ON_NPU && MINICPM_V_MM_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
            attn_reconfig.get(),
#else
            nullptr,
#endif
            attn_bfp16,
            attn_chunked
        );

        this->mlp = std::make_unique<minicpm_v_mlp_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            npu,
            gemm_app_manager
        );

#ifdef ENABLE_DEBUG
        // layer 0 is a linear (gated delta net) layer and attn_idx is the first full
        // attention one, so the default set covers both layer kinds plus the deepest layer
        this->xref = std::make_unique<minicpm_v_prefill_xref>((int)desc->attn_idx,
                                                             (int)desc->num_hidden_layers - 1);
        this->linear_block->xref = this->xref.get();
        this->attn_block->xref   = this->xref.get();
        this->mlp->xref          = this->xref.get();
#endif
    }


#if MINICPM_V_ATTN_ON_NPU && MINICPM_V_MM_ON_NPU && FLM_MINICPM_RECONFIG_SUPPORTED
    /// FLM_MINICPM_SINGLE_CTX=1: validate the single-context attention path once the weights are loaded (see the attn block's self test).
    /// `kv_rows` is raised to the number of K/V cache rows the test dirtied so the caller's clear_context() zeroes them.
    void validate_single_context(buffer<u8>& attn_proj_weights, buffer<bf16>& attn_kv_cache, int& kv_rows) {
        if (attn_reconfig && attn_reconfig->built && !attn_reconfig->validated) attn_block->single_ctx_self_test(attn_proj_weights, attn_kv_cache, kv_rows);
    }
#endif

#if LINEAR_PIPELINE
    // -----------------------------------------------------------------------------------------
    // Block-pipelined linear (gated delta net) layer + its MLP.
    //
    // The array does 512-row passes. For L > 512 the rows are cut into groups of whole 512-row
    // blocks; each projection of a group is one array run over just those rows (same kernel, the
    // sequence only carries a row base), so the host stages of group g (conv, delta rule, gated
    // norm, residual + norm, GLU) run while the array works on the other groups. Every host stage
    // is row independent except the conv (reads 3 rows of the previous group's qkv, already there)
    // and the delta rule (carries the state; groups run in order, so its arithmetic is the one-call
    // arithmetic). Results are bitwise those of the unpipelined layer.
    //
    // The stage order is a plain list (pipe_order()), so it can be tuned without touching the code:
    //   R(g) input norm + submit qkv, z      A(g) alpha/beta       C(g) wait qkv, conv, delta rule
    //   G(g) wait z, gated norm, submit o    P(g) wait o, residual, norm, submit up/gate
    //   U(g) wait up/gate, GLU, submit down  F(g) wait down, residual
    // -----------------------------------------------------------------------------------------
    using pipe_run_t = decltype(std::declval<npu_app&>().create_run(std::declval<buffer<bf16>&>(), std::declval<buffer<bf16>&>(), std::declval<buffer<u8>&>()));
    enum pipe_op { OP_QKV = 0, OP_Z, OP_O, OP_UP, OP_DN };
    std::map<std::tuple<int, int, int, int>, std::unique_ptr<npu_app>> pipe_apps;
    int pipe_group_blocks = -1;
    int pipe_csub = 2;   //!< pieces the delta rule of a group is cut into (gaps for the other stages)
    std::string pipe_order_cfg;

    /// the array app that runs `op` over blocks [b0, b0 + nb) (512 rows each, `n` of them real); L independent
    /// (a runtime-m mm also keys on the rows it covers), so cached for good
    npu_app& pipe_app(int op, int b0, int nb, int n) {
        const uint32_t valid = (uint32_t)std::min(n, nb * 512);
        const bool glu = mlp->glu_now;   // set by ml.reserve() of this call
        auto key = std::make_tuple(op, b0, nb, (seq_gen->mm_rtm ? (int)seq_gen->mm_pick_geometry(valid).m : 0) + (glu ? 4096 : 0));
        auto it = pipe_apps.find(key);
        if (it != pipe_apps.end()) return *it->second;
        auto app = std::make_unique<npu_app>(gemm_app_manager->create_app());
        const uint32_t rows = (uint32_t)nb * 512, base = (uint32_t)b0 * 512;
        const int pad = desc->conv_kernel_size - 1;
        switch (op) {
        case OP_QKV: seq_gen->gen_dequant_mm_512(app->seq(), rows, desc->D, desc->Linear_DQKV, desc->get_linear_qkv_offset(), (size_t)pad * desc->Linear_DQKV, 0, base, valid); break;
        case OP_Z:   seq_gen->gen_dequant_mm_512(app->seq(), rows, desc->D, desc->Linear_DV, desc->get_linear_g_offset(), 0, 1, base, valid); break;
        case OP_O:   seq_gen->gen_dequant_mm_512(app->seq(), rows, desc->Linear_DV, desc->D, desc->get_linear_o_offset(), 0, 0, base, valid); break;
        case OP_UP:
            if (glu) seq_gen->gen_dequant_mm_512(app->seq(), rows, desc->D, desc->intermediate_size * 2, desc->get_upgate_glu_offset(), 0, 3, base, valid);
            else seq_gen->gen_dequant_mm_512(app->seq(), rows, desc->D, desc->intermediate_size * 2, desc->get_upgate_offset(), 0, 0, base, valid);
            break;
        default:     seq_gen->gen_dequant_mm_512(app->seq(), rows, desc->intermediate_size, desc->D, desc->get_down_offset(), 0, 0, base, valid); break;
        }
        return *pipe_apps.emplace(key, std::move(app)).first->second;
    }

    /// FLM_MCPM_PIPE=0 turns the pipeline off; FLM_MCPM_GROUP = blocks per group (default 1)
    bool pipe_active(int L) {
        auto load = [&](const char* grp, const char* csub, const char* ord, const char* en) {
            pipe_group_blocks = (en && std::atoi(en) == 0) ? 0 : (grp ? std::max(1, std::atoi(grp)) : 1);
            pipe_csub = csub ? std::max(1, std::atoi(csub)) : 2;
            pipe_order_cfg = ord ? ord : "";
            pipe_order_cfg.erase(0, pipe_order_cfg.find_first_not_of(" \t"));
        };
        if (const char* f = std::getenv("FLM_MCPM_CFG")) {   // tuning aid: "<enable> <group> <csub> <order...>", re-read per layer
            std::ifstream in(f);
            std::string en, grp, csub, ord;
            if (in >> en >> grp >> csub) { std::getline(in, ord); load(grp.c_str(), csub.c_str(), ord.c_str(), en.c_str()); }
        }
        else if (pipe_group_blocks < 0) {
            load(std::getenv("FLM_MCPM_GROUP"), std::getenv("FLM_MCPM_CSUB"), std::getenv("FLM_MCPM_ORDER"), std::getenv("FLM_MCPM_PIPE"));
        }
        return pipe_group_blocks > 0 && L > 512;
    }

    /// The stage list for G groups. Default (variant 'a'): all input norms first, then per group the
    /// delta rule in csub pieces; the previous group's residual/norm/up-gate step goes into the gap before the
    /// last piece (so the array is fed while the host still scans) and its GLU after this group's gated norm.
    /// A written-out list ("R0 R1 ... F2") in FLM_MCPM_ORDER overrides it.
    std::vector<std::pair<char, int>> pipe_order(int G) {
        std::vector<std::pair<char, int>> v;
        const std::string cfg = pipe_order_cfg;
        if (cfg.empty() || cfg[0] == '@') {
            const char var = cfg.size() > 1 ? cfg[1] : 'a';
            const int k = pipe_csub;
            for (int g = 0; g < G; g++) v.emplace_back('R', g);
            for (int g = 0; g < G; g++) {
                v.emplace_back('A', g);
                for (int i = 0; i + 1 < k; i++) v.emplace_back('C', g);
                if (var == 'a' && g == 1) v.emplace_back('P', 0);
                if (var == 'b' && g >= 1) v.emplace_back('P', g - 1);
                v.emplace_back('C', g);
                v.emplace_back('G', g);
                if (g >= 1) {
                    v.emplace_back('U', g - 1);
                    if (var == 'a') v.emplace_back('P', g);
                }
            }
            if (var == 'b' || G == 1) v.emplace_back('P', G - 1);
            v.emplace_back('U', G - 1);
            for (int g = 0; g < G; g++) v.emplace_back('F', g);
            return v;
        }
        std::istringstream in(cfg);
        std::string t;
        while (in >> t) {
            const int g = std::atoi(t.c_str() + 1);
            if (g >= G) continue;   // an order written for more groups than this prompt has
            v.emplace_back(t[0], g);
        }
        // every stage of every group must appear (once; the delta rule csub times), or rows would be skipped
        std::map<std::pair<char, int>, int> cnt;
        for (auto& e : v) cnt[e]++;
        for (int g = 0; g < G; g++)
            for (char st : std::string("RAGPUF"))
                if (cnt[{st, g}] != 1) { pipe_order_cfg.clear(); return pipe_order(G); }
        for (int g = 0; g < G; g++)
            if (cnt[{'C', g}] != pipe_csub) { pipe_order_cfg.clear(); return pipe_order(G); }
        return v;
    }

    buffer<bf16> forward_linear_pipelined(int layer_idx, buffer<bf16>& residual, int L, buffer<u8>& W, buffer<bf16>& rms_w,
                                          buffer<bf16>& ssm_w, buffer<bf16>& cache, buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu) {
        PREFILL_PROF_INIT();
        auto& lb = *linear_block;
        auto& ml = *mlp;
        const std::string pipe_where = "minicpm-v layer " + std::to_string(layer_idx) + " linear ";
        const int L_padded = (L + 511) / 512 * 512;
        const int nblk = L_padded / 512;
        const int D = desc->D, DQKV = desc->Linear_DQKV, DV = desc->Linear_DV, IS = desc->intermediate_size;
        const int pad = desc->conv_kernel_size - 1;
        const int Hv = desc->linear_num_v_heads;
        lb.reserve(L, L_padded, false);
        ml.reserve(L_padded, false);

        struct group_t { int b0, nb, r0, n; };
        std::vector<group_t> grp;
        for (int b = 0; b < nblk; b += pipe_group_blocks) {
            const int nb = std::min(pipe_group_blocks, nblk - b);
            grp.push_back({b, nb, b * 512, std::min(L, (b + nb) * 512) - b * 512});
        }
        const int G = (int)grp.size();
        std::vector<std::optional<pipe_run_t>> rq(G), rz(G), ro(G), rup(G), rdn(G);
        std::vector<int> cdone(G, 0);
        auto submit = [&](std::optional<pipe_run_t>& slot, int op, int g, buffer<bf16>& out, buffer<bf16>& in) {
            slot.emplace(pipe_app(op, grp[g].b0, grp[g].nb, grp[g].n).create_run(out, in, W));
            slot->start();
        };
        auto finish = [&](std::optional<pipe_run_t>& slot, const char* what) {
            flm::npu::wait_for_completion(*slot, pipe_where + what);
            slot.reset();
        };

        float* S = reinterpret_cast<float*>(cache.data() + desc->get_s_offset());
        float* alpha_ptr = lb.alpha_beta.data();
        float* beta_ptr  = alpha_ptr + lb.alpha_beta.size() / 2;
        const bf16* conv_w = ssm_w.data() + lb.weights().ssm_conv1d.offset / sizeof(bf16);
        const bf16* gate_norm_w = ssm_w.data() + desc->get_linear_ssm_norm_offset() / sizeof(bf16);
        bf16* x = common_bufs.x.data();
        bf16* res = residual.data();

        {   // the conv state of the previous request enters as the first rows of qkv
            cache.sync_from_device();
            memcpy(lb.qkv.data(), cache.data(), (size_t)pad * DQKV * sizeof(bf16));
        }
        PREFILL_PROF_TICK("pipe:setup");

        for (auto [st, g] : pipe_order(G)) {
            const group_t& gr = grp[g];
            const int r0 = gr.r0, n = gr.n;
            switch (st) {
            case 'R':
                cpu_func::_rms_norm(x, res, rms_w.data(), D, n, r0, r0);
                minicpm_v_cpu::scratch_push_rows(common_bufs.x, r0, n, D);
                submit(rq[g], OP_QKV, g, lb.qkv, common_bufs.x);
                submit(rz[g], OP_Z, g, common_bufs.gate, common_bufs.x);
                break;
            case 'A':
                lb._alpha_beta(n, ssm_w, alpha_proj_cpu, beta_proj_cpu, r0);
                break;
            case 'C': {
                // piece k of csub of this group's rows; the first piece waits for the projection
                if (cdone[g] == 0) {
                    finish(rq[g], "qkv");
                    minicpm_v_cpu::scratch_pull_rows(lb.qkv, pad + r0, n, DQKV);
                    PREFILL_PROF_TICK("wC" << g);
                }
                if (cdone[g] >= n) break;
                const int per = (n + pipe_csub - 1) / pipe_csub;
                const int p0 = r0 + cdone[g], pn = std::min(per, n - cdone[g]);
                cdone[g] += pn;
                minicpm_v_cpu::conv1d_silu_l2norm(lb.conv_qkv.data() + (size_t)p0 * DQKV, lb.qkv.data() + (size_t)p0 * DQKV, conv_w, pn, DQKV,
                                                  desc->conv_kernel_size, (desc->Linear_DQ + desc->Linear_DK) / desc->linear_qk_head_dim,
                                                  desc->linear_qk_head_dim);
                minicpm_v_cpu::gated_delta_net(
                    common_bufs.after_mix.data() + (size_t)p0 * DV, S, lb.conv_qkv.data() + (size_t)p0 * DQKV, DQKV,
                    0, desc->Linear_DQ, desc->Linear_DQ + desc->Linear_DK,
                    alpha_ptr + (size_t)p0 * Hv, beta_ptr + (size_t)p0 * Hv, pn,
                    desc->linear_num_qk_heads, desc->linear_num_v_heads, desc->linear_qk_head_dim, desc->linear_v_head_dim);
                if (cdone[g] == n && g == G - 1) {   // the conv state leaves with the last rows; the state goes back to the device for decode
                    memcpy(cache.data(), lb.qkv.data() + (size_t)L * DQKV, (size_t)pad * DQKV * sizeof(bf16));
                    cache.sync_to_device();
                }
                break;
            }
            case 'G':
                assert(cdone[g] == n);
                finish(rz[g], "z");
                minicpm_v_cpu::scratch_pull_rows(common_bufs.gate, r0, n, DV);
                PREFILL_PROF_TICK("wG" << g);
                #pragma omp parallel for num_threads(minicpm_v_cpu::prefill_threads()) schedule(static)
                for (int ll = r0; ll < r0 + n; ll++) {
                    for (int h = 0; h < Hv; h++) {
                        const uint32_t off = (uint32_t)ll * DV + h * (DV / Hv);
                        cpu_func::_gated_norm(common_bufs.after_mix.data() + off, common_bufs.gate.data() + off, gate_norm_w, desc->linear_v_head_dim, 1);
                    }
                }
                minicpm_v_cpu::scratch_push_rows(common_bufs.after_mix, r0, n, DV);
                submit(ro[g], OP_O, g, common_bufs.x, common_bufs.after_mix);
                break;
            case 'P':
                finish(ro[g], "o");
                minicpm_v_cpu::scratch_pull_rows(common_bufs.x, r0, n, D);
                PREFILL_PROF_TICK("wP" << g);
                cpu_func::_residual_add(res + (size_t)r0 * D, x + (size_t)r0 * D, res + (size_t)r0 * D, (size_t)n * D);
                cpu_func::_rms_norm(x, res, rms_w.data() + D, D, n, r0, r0);
                minicpm_v_cpu::scratch_push_rows(common_bufs.x, r0, n, D);
                submit(rup[g], OP_UP, g, ml.glu_now ? ml.hid : ml.upgate, common_bufs.x);
                break;
            case 'U':
                finish(rup[g], "up");
                if (!ml.glu_now) {
                    minicpm_v_cpu::scratch_pull_rows(ml.upgate, r0, n, (size_t)IS * 2);
                    PREFILL_PROF_TICK("wU" << g);
                    cpu_func::_exp_glu(ml.hid.data() + (size_t)r0 * IS, ml.upgate.data() + (size_t)r0 * IS * 2, n, IS, 512, minicpm_v_cpu::prefill_threads());
                    minicpm_v_cpu::scratch_push_rows(ml.hid, r0, n, IS);
                }
                else PREFILL_PROF_TICK("wU" << g);   // the array wrote hid
                submit(rdn[g], OP_DN, g, common_bufs.x, ml.hid);
                break;
            case 'F':
                finish(rdn[g], "down");
                minicpm_v_cpu::scratch_pull_rows(common_bufs.x, r0, n, D);
                PREFILL_PROF_TICK("wF" << g);
                cpu_func::_residual_add(res + (size_t)r0 * D, x + (size_t)r0 * D, res + (size_t)r0 * D, (size_t)n * D);
                break;
            }
            PREFILL_PROF_TICK("s" << st << g);
        }
        return residual;
    }
#endif

    buffer<bf16> forward(int layer_idx, buffer<bf16>& residual, int L, buffer<u8>& all_proj_weights, buffer<bf16>& rms_weights, buffer<bf16>& rope_weights, buffer<bf16>& kv_cache, std::vector<int>& input_idx, buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu, int L_begin = 0){
        PREFILL_PROF_INIT();
        minicpm_v_cpu::dumper().layer = layer_idx;
#ifdef ENABLE_DEBUG
        this->linear_block->layer_idx = layer_idx;
        this->attn_block->layer_idx   = layer_idx;
        this->mlp->layer_idx          = layer_idx;
#endif
        int L_padded = (L + 511) / 512 * 512;
        size_t x_size = (size_t)L_padded * desc->D;

        if (x_size != common_bufs.x.size()) {
            common_bufs.x = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->D);
            common_bufs.after_mix = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->Linear_DV);
            common_bufs.gate = minicpm_v_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->Linear_DV);
            // The rows past L are padding for the array's 512-row passes; start them at zero once
            // (host and device). Every stage then syncs only its first L rows, so the padding is
            // never re-zeroed or re-flushed per layer; whatever a later, shorter request leaves
            // there is finite and, the projections being row independent, never read.
            memset(common_bufs.x.data(), 0, common_bufs.x.size() * sizeof(bf16));
            minicpm_v_cpu::scratch_push(common_bufs.x);
        }

#if LINEAR_PIPELINE
        if (!desc->is_attn_layer(layer_idx) && pipe_active(L)) {
            auto out = forward_linear_pipelined(layer_idx, residual, L, all_proj_weights, rms_weights, rope_weights, kv_cache, alpha_proj_cpu, beta_proj_cpu);
            minicpm_v_cpu::dumper().layer = -1;
            return out;
        }
#endif
        // for all layers , first step is store a residual and do input layer norm
        // this step do the rms norm and store it to x, the residual is kept
        cpu_func::_rms_norm(common_bufs.x, residual, rms_weights.data(), desc->D, L, 0, 0);
        minicpm_v_cpu::scratch_push_rows(common_bufs.x, 0, L, desc->D);
        MINICPM_V_DUMP("input_layernorm", common_bufs.x.data(), L, desc->D);
        buffer<bf16> block_output;
        PREFILL_PROF_TICK("layer:input_rms_norm");
        if (desc->is_attn_layer(layer_idx)) {
            block_output = this->attn_block->forward(L, all_proj_weights, rope_weights, kv_cache, input_idx, L_begin, layer_idx);
            PREFILL_PROF_TICK("layer:attn_block_total");
        }
        else {
            block_output = this->linear_block->forward(L, all_proj_weights, rope_weights, kv_cache, alpha_proj_cpu, beta_proj_cpu);
            PREFILL_PROF_TICK("layer:linear_block_total");
        }

        cpu_func::_residual_add(residual, block_output, residual, residual.size());

        cpu_func::_rms_norm(common_bufs.x, residual, rms_weights.data() + desc->D, desc->D, L, 0, 0);
        MINICPM_V_DUMP("post_attention_layernorm", common_bufs.x.data(), L, desc->D);

        minicpm_v_cpu::scratch_push_rows(common_bufs.x, 0, L, desc->D);
        PREFILL_PROF_TICK("layer:post_attn_rms_norm");

        auto mlp_out = mlp->forward(L, all_proj_weights, rms_weights);
        PREFILL_PROF_TICK("layer:mlp_block_total");

        cpu_func::_residual_add(residual, mlp_out, residual, residual.size());
        MINICPM_V_DUMP("residual_out", residual.data(), L, desc->D);
        minicpm_v_cpu::dumper().layer = -1;

#ifdef ENABLE_DEBUG
        // one pass/fail line after the last layer: at FLM_XREF_LAYERS=all the per tensor
        // report is hundreds of lines and the question being asked is a single bit
        if (layer_idx == (int)desc->num_hidden_layers - 1) { this->xref->summary(); }
#endif

        return residual;
    }
};

#endif
