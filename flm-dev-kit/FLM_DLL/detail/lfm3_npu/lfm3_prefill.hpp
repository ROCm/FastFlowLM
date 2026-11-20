// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __LFM3_PREFILL_HPP__
#define __LFM3_PREFILL_HPP__
/// \file lfm3_prefill.hpp
/// \brief NPU prefill resources of the LFM3 engine: xclbin apps, instruction generators and scratch buffers.
///
/// Pipeline per layer (the same shape as the LFM2 engine):
///   dequant the layer's weights to bf16 (dequant.xclbin)  ||  host: operator RMSNorm
///   mixer:  attention: q/k/v gemms (mm.xclbin) -> host norm + RoPE + KV rows -> attention (attn.xclbin, or host) -> gate -> o gemm
///           conv:      in_proj gemm -> host conv (width 7) -> out_proj gemm
///   host: post-operator norm + residual; FFN norm; up / gate(SiLU) gemms -> host gate*up -> down gemm; post-FFN norm + residual
/// The rows of a batch are padded to a multiple of 256 (and the front to the attention chunk of 128) because the kernels
/// work on whole tiles; padding rows are zero and never written back.
#include <memory>
#include <vector>
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"
#include "modules/mha.hpp"
#include "npu_utils/npu_utils.hpp"
#include "lfm3_model_def.hpp"

/// Row geometry of one prefill pass.
struct lfm3_prefill_shape {
    int L_in = 0;             ///< tokens in this pass
    int L_begin = 0;          ///< context length before this pass
    int L_offset = 0;         ///< rows of chunk padding in front of the first new token
    int L_padded = 0;         ///< rows every gemm runs over
    int L_begin_chunked = 0;  ///< chunk-aligned start fed to the attention kernel
    int L_end_chunked = 0;    ///< chunk-aligned end fed to the attention kernel
};

struct lfm3_prefill_ctx {
    /// the attention kernel refuses batches shorter than this and the gemms want whole tiles
    static constexpr int L_MIN = 256;
    /// tokens per pass; scratch buffers are sized for one pass
    static constexpr int PASS_TOKENS = 2048;

    lfm3_desc* desc = nullptr;
    uint32_t MAX_L = 0;
    bool use_runlist = true;

    npu_app_manager* gemm_mgr = nullptr;
    npu_app_manager* dq_mgr = nullptr;
    npu_app_manager* mha_mgr = nullptr;
    std::unique_ptr<Gemm> gemm;
    std::unique_ptr<Dequant> dq;
    std::unique_ptr<MHA> mha;

    // dequant: one app per matrix kind, bf16 weight buffers they fill
    npu_app dq_qkv, dq_conv_in, dq_o, dq_up, dq_gate, dq_down;
    flm_rt::runlist dq_all;
    flm_rt::run pending_run, up_run, gate_run, down_run;
    buffer<bf16> w_slot0, w_o, w_up, w_gate, w_down;

    // gemm apps
    npu_app q_proj, k_proj, v_proj, o_proj, conv_in_proj, up_proj, gate_proj, down_proj;
    // attention apps: one sequence for full attention, one for the sliding window
    npu_app mha_full, mha_swa;
    int full_key[3] = {-1, -1, -1}, swa_key[3] = {-1, -1, -1};   ///< (L_begin_chunked, L_end_chunked, MAX_L) of the generated sequences

    // scratch (device buffers unless noted)
    buffer<bf16> x, o, mixer_out, q, k, v, bcx, up, gate, hid;
    std::vector<uint16_t> residual;     ///< host: running residual stream [rows][D]
    int L_padded_old = -1;
    int cap_rows = 0;

    lfm3_prefill_ctx(npu_xclbin_manager* npu, lfm3_desc* d, LM_Config& config, uint32_t max_len)
        : desc(d), MAX_L(max_len) {
        const std::string dir = utils::path_join(config.exec_path, "xclbins", config.model_name);
        gemm_mgr = npu->register_xclbin(utils::path_join(dir, "mm.xclbin"));
        dq_mgr = npu->register_xclbin(utils::path_join(dir, "dequant.xclbin"));
        mha_mgr = npu->register_xclbin(utils::path_join(dir, "attn.xclbin"));
        use_runlist = !npu->is_preemption_enabled();

        gemm = std::make_unique<Gemm>(config);
        dq = std::make_unique<Dequant>(config);
        mha_type_t mt = mha_d128_q4;
        if (d->DH != 128 || d->num_heads != 4 * d->num_kv_heads)
            throw std::runtime_error("lfm3 prefill: the attention kernel needs head_dim 128 and 4 query heads per kv head");
        mha = std::make_unique<MHA>(mt, (int)d->num_kv_heads);

        dq_qkv = dq_mgr->create_app();
        dq_conv_in = dq_mgr->create_app();
        dq_o = dq_mgr->create_app();
        dq_up = dq_mgr->create_app();
        dq_gate = dq_mgr->create_app();
        dq_down = dq_mgr->create_app();
        if (use_runlist) dq_all = dq_mgr->create_runlist();

        const uint32_t D = d->D, DQ = d->DQ, I = d->intermediate_size, SLOT0 = 3 * d->D;
        w_slot0 = dq_qkv.create_bo_buffer<bf16>((size_t)D * SLOT0);
        w_o = dq_o.create_bo_buffer<bf16>((size_t)D * DQ);
        w_up = dq_up.create_bo_buffer<bf16>((size_t)D * I);
        w_gate = dq_gate.create_bo_buffer<bf16>((size_t)D * I);
        w_down = dq_down.create_bo_buffer<bf16>((size_t)I * D);
        // up and gate share one band and are told apart by the mode argument
        dq->generate_dequant_q4_1_seq(dq_qkv.seq(), D, DQ + d->DK + d->DV, (uint32_t)d->q_off(), 0);
        dq->generate_dequant_q4_1_seq(dq_conv_in.seq(), D, SLOT0, (uint32_t)d->slot0_off(), 0);
        dq->generate_dequant_q4_1_seq(dq_o.seq(), DQ, D, (uint32_t)d->slot1_off(), 0);
        dq->generate_dequant_q4_1_seq(dq_up.seq(), D, I, (uint32_t)d->upgate_off(), 1);
        dq->generate_dequant_q4_1_seq(dq_gate.seq(), D, I, (uint32_t)d->upgate_off(), 2);
        dq->generate_dequant_q4_1_seq(dq_down.seq(), I, D, (uint32_t)d->down_off(), 0);

        q_proj = gemm_mgr->create_app();
        k_proj = gemm_mgr->create_app();
        v_proj = gemm_mgr->create_app();
        o_proj = gemm_mgr->create_app();
        conv_in_proj = gemm_mgr->create_app();
        up_proj = gemm_mgr->create_app();
        gate_proj = gemm_mgr->create_app();
        down_proj = gemm_mgr->create_app();
        mha_full = mha_mgr->create_app();
        mha_swa = mha_mgr->create_app();

        cap_rows = PASS_TOKENS + L_MIN;
        x = gemm_mgr->create_bo_buffer<bf16>((size_t)cap_rows * D);
        o = gemm_mgr->create_bo_buffer<bf16>((size_t)cap_rows * D);
        mixer_out = mha_mgr->create_bo_buffer<bf16>((size_t)cap_rows * DQ);
        q = q_proj.create_bo_buffer<bf16>((size_t)cap_rows * DQ);
        k = k_proj.create_bo_buffer<bf16>((size_t)cap_rows * d->DK);
        v = v_proj.create_bo_buffer<bf16>((size_t)cap_rows * d->DV);
        bcx = conv_in_proj.create_bo_buffer<bf16>((size_t)cap_rows * SLOT0);
        up = up_proj.create_bo_buffer<bf16>((size_t)cap_rows * I);
        gate = gate_proj.create_bo_buffer<bf16>((size_t)cap_rows * I);
        hid = down_proj.create_bo_buffer<bf16>((size_t)cap_rows * I);
        // padding rows must be zero and stay zero: nothing ever writes them
        // the dequant buffers are written by the device: flush them once so no dirty host cache line is ever written back over
        // what the kernels produce (a stale line corrupted the o-projection weights from the second layer on)
        for (buffer<bf16>* b : {&x, &o, &mixer_out, &q, &k, &v, &bcx, &up, &gate, &hid, &w_slot0, &w_o, &w_up, &w_gate, &w_down}) {
            b->memset((bf16)0.0f);
            b->sync_to_device();
        }
        residual.assign((size_t)cap_rows * D, 0);
    }

    int chunk() { return mha->get_chunk_size(); }

    void set_max_length(uint32_t max_len) {
        MAX_L = max_len;
        full_key[0] = swa_key[0] = -1;       // the attention sequences bake in the cache layout
    }

    /// row geometry of a pass of L_in tokens after L_begin tokens of context
    lfm3_prefill_shape shape(int L_in, int L_begin) {
        lfm3_prefill_shape s;
        s.L_in = L_in;
        s.L_begin = L_begin;
        const int L_chunk = chunk();
        s.L_begin_chunked = (L_begin / L_chunk) * L_chunk;
        s.L_offset = L_begin - s.L_begin_chunked;
        s.L_end_chunked = s.L_begin_chunked + (L_in + s.L_offset + L_MIN - 1) / L_MIN * L_MIN;
        s.L_padded = s.L_end_chunked - s.L_begin_chunked;
        if (s.L_padded > cap_rows) throw std::runtime_error("lfm3 prefill: pass too long for the scratch buffers");
        return s;
    }

    /// regenerate the instruction sequences that depend on the row count
    void setup(const lfm3_prefill_shape& s) {
        const uint32_t D = desc->D, DQ = desc->DQ, DK = desc->DK, DV = desc->DV, I = desc->intermediate_size;
        if (L_padded_old != s.L_padded) {
            const bool no_bias = false;
            gemm->generate_seq(q_proj.seq(), s.L_padded, D, DQ, 0, no_bias, Gemm::NO_Activation, 0);
            gemm->generate_seq(k_proj.seq(), s.L_padded, D, DK, D * DQ, no_bias, Gemm::NO_Activation, 0);
            gemm->generate_seq(v_proj.seq(), s.L_padded, D, DV, D * (DQ + DK), no_bias, Gemm::NO_Activation, 0);
            gemm->generate_seq(o_proj.seq(), s.L_padded, DQ, D, 0, no_bias, Gemm::NO_Activation, 0);
            gemm->generate_seq(conv_in_proj.seq(), s.L_padded, D, 3 * D, 0, no_bias, Gemm::NO_Activation, 0);
            // each matrix has its own dequantized buffer, hence weight offset 0
            gemm->generate_seq(up_proj.seq(), s.L_padded, D, I, 0, no_bias, Gemm::NO_Activation, 0);
            gemm->generate_seq(gate_proj.seq(), s.L_padded, D, I, 0, no_bias, Gemm::SiLU, 0);
            gemm->generate_seq(down_proj.seq(), s.L_padded, I, D, 0, no_bias, Gemm::NO_Activation, 0);
            L_padded_old = s.L_padded;
        }
    }

    /// attention sequence for a pass; `window` is the sliding window or 0 for full attention
    npu_app& attention_app(const lfm3_prefill_shape& s, int window) {
        npu_app& app = window ? mha_swa : mha_full;
        int* key = window ? swa_key : full_key;
        if (key[0] != s.L_begin_chunked || key[1] != s.L_end_chunked || key[2] != (int)MAX_L) {
            mha->generate_mha_sequence(app.seq(), s.L_begin_chunked, s.L_end_chunked, MAX_L, window != 0, window ? window : -1);
            key[0] = s.L_begin_chunked; key[1] = s.L_end_chunked; key[2] = (int)MAX_L;
        }
        return app;
    }

    // ---- dequantization of one layer's weights, overlapped with the caller's host work ----
    void dequant_start(buffer<u8>& proj, bool is_attn) {
        npu_app& slot0_app = is_attn ? dq_qkv : dq_conv_in;
        auto slot0_run = slot0_app.create_run(w_slot0, proj);
        auto o_run = dq_o.create_run(w_o, proj);
        up_run = dq_up.create_run(w_up, proj);
        gate_run = dq_gate.create_run(w_gate, proj);
        down_run = dq_down.create_run(w_down, proj);
        if (use_runlist) {
            dq_all.reset();
            dq_all.add(slot0_run);
            dq_all.add(o_run);
            dq_all.add(up_run);
            dq_all.add(gate_run);
            dq_all.add(down_run);
            dq_all.execute();
        } else {
            // preemption: no runlists; slot 0 is serialized, the rest follow in dequant_wait()
            slot0_run.start();
            flm::npu::wait_for_completion(slot0_run, "slot0_run");
            o_run.start();
            pending_run = o_run;
        }
    }
    void dequant_wait() {
        if (use_runlist) {
            flm::npu::wait_for_completion(dq_all, "dq_all");
        } else {
            flm::npu::wait_for_completion(pending_run, "pending_run");
            up_run.start();
            flm::npu::wait_for_completion(up_run, "up_run");
            gate_run.start();
            flm::npu::wait_for_completion(gate_run, "gate_run");
            down_run.start();
            flm::npu::wait_for_completion(down_run, "down_run");
        }
    }
};

#endif
