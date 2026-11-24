// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__VISION__EXEC__HPP__
#define __SMOLVLA__VISION__EXEC__HPP__

#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "buffer.hpp"
#include "typedef.hpp"
#include "SmolVLA_config.hpp"
#include "SmolVLA_vision_def.hpp"
#include "npu_utils/npu_utils.hpp"
#include "cpu_functions.hpp"
#include "seq_gen.hpp"
#include "utils/prof_utils.hpp"
#include "utils/xclbin_path.hpp"

// ---------------------------------------------------------------------------
// SmolVLA vision-encoder (SigLIP) execution contexts.
//
// The vision tower turns a patchified image into a sequence of visual tokens:
//   1. a conv2d patch embedding + learned position embedding produces the
//      initial [num_patches, D] token grid,
//   2. a stack of identical pre-norm transformer layers (bidirectional
//      self-attention, no GQA, no rope; biased q/k/v/out; GELU fc1/fc2 MLP),
//   3. a final post_layernorm.
// The resulting visual tokens are handed to the connector (see the vlm exec).
//
// All activations flowing between blocks are buffer<bf16>. Each block owns its
// npu_app handles + scratch buffers and exposes a forward() that takes the
// (layer-normed) hidden state and returns the block output (pre residual-add).
// ---------------------------------------------------------------------------

/// @brief bidirectional self-attention block. Separate biased q/k/v/out
///        projections; no GQA (num_kv_heads == num_attn_heads) and no rope.
struct SmolVLA_vision_attn_block {
    static constexpr int L_per_image = smolvla_cfg::TOKENS_PER_IMAGE;
    static constexpr int MAX_L       = smolvla_cfg::MAX_VISION_TOKENS;

    SmolVLA_Vision_def* def;

    int last_L = -1;

    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app out_proj;
    npu_app mha;

    xrt::runlist rl;

    /// per-layer runs (the weights buffer is the only per-layer argument), built on the layer's first call for a
    /// given L and re-executed afterwards (SMOLVLA_VISION_RUNCACHE=0 rebuilds them on every call)
    struct layer_runs_t {
        xrt::runlist rl;
        xrt::run mha_run;
        xrt::run o_run;
    };
    std::unordered_map<const void*, layer_runs_t> layer_runs;
    static bool run_cache_on() {
        static const bool v = std::getenv("SMOLVLA_VISION_RUNCACHE") == nullptr ||
                              std::getenv("SMOLVLA_VISION_RUNCACHE")[0] != '0';
        return v;
    }
    npu_app_manager* mgr = nullptr;

    buffer<bf16> q;
    buffer<bf16> k;
    buffer<bf16> v;
    buffer<bf16> attn_out;
    buffer<bf16> o;
    buffer<uint32_t> pk_attn32; // control packets of the switch to attn32 / back to mm8 (vision_ctrlpkt())
    buffer<uint32_t> pk_mm8;

    SmolVLA_vision_attn_block(
        SmolVLA_Vision_def* def,
        npu_app_manager* vision_app_manager)
        : def(def), mgr(vision_app_manager){
        this->q_proj   = vision_app_manager->create_app();
        this->k_proj   = vision_app_manager->create_app();
        this->v_proj   = vision_app_manager->create_app();
        this->out_proj = vision_app_manager->create_app();
        this->mha      = vision_app_manager->create_app();

        this->rl = vision_app_manager->create_runlist();
        
        q  = this->q_proj.create_bo_buffer<bf16>(MAX_L * def->DQ);
        k  = this->k_proj.create_bo_buffer<bf16>(MAX_L * def->DK);
        v  = this->v_proj.create_bo_buffer<bf16>(MAX_L * def->DV);
        attn_out  = this->mha.create_bo_buffer<bf16>(MAX_L * def->DQ);
        o  = this->out_proj.create_bo_buffer<bf16>(MAX_L * def->D);
        if (smolvla_vlm_vision_seq::vision_l32() && smolvla_vlm_vision_seq::vision_ctrlpkt()) {
            // the control packets of the two switches: binding 4 of the run that carries the switch
            for (auto* pk : {&pk_attn32, &pk_mm8}) {
                const auto& w = smolvla_vlm_vision_seq::vision_switch(
                    pk == &pk_attn32 ? "cp_to_attn32" : "cp_to_mm8").packets;
                *pk = this->mha.create_bo_buffer<uint32_t>(w.size());
                memcpy(pk->data(), w.data(), w.size() * sizeof(uint32_t));
                pk->sync_to_device();
            }
        }
    }


    /// @brief run bidirectional self-attention over the visual-token sequence.
    ///        L spans all images in the batch (num_images * L_per_image); the
    ///        mha sequence attends within each image's block separately.
    /// @param patch   layer-normed hidden state, [L, D]
    /// @param weights this layer's packed weight buffer
    /// @param L       total number of visual tokens across the batch
    /// @return attention output projected back to model dim, [L, D]
    buffer<bf16> forward(buffer<bf16>& patch, buffer<bf16>& weights, int L) {
        // 1. (re)generate sequences / (re)alloc scratch as needed.

        if (last_L != L && smolvla_vlm_vision_seq::vision_l32()) {
            // mm8 projections; the attention sequence starts with the switch to attn32, the output projection
            // with the switch back (the order of the runs below is the order in the array)
            using namespace smolvla_vlm_vision_seq;
            gen_vision_mm8_seq(this->q_proj.seq(), L, def->layer_def.q_proj, 0);
            gen_vision_mm8_seq(this->k_proj.seq(), L, def->layer_def.k_proj, 0);
            gen_vision_mm8_seq(this->v_proj.seq(), L, def->layer_def.v_proj, 0);
            const std::string cp = vision_ctrlpkt() ? "cp_" : "";
            gen_vision_attn32_seq(this->mha.seq(), L_per_image, def->head_dim, def->num_attn_heads, L / L_per_image,
                                  &vision_switch(cp + "to_attn32").words);
            gen_vision_mm8_seq(this->out_proj.seq(), L, def->layer_def.out_proj, 0, 0,
                               &vision_switch(cp + "to_mm8").words);
        } else if (last_L != L) {
            smolvla_vlm_vision_seq::gen_vision_mm_seq(this->q_proj.seq(), L, def->layer_def.q_proj, 1, 0);
            smolvla_vlm_vision_seq::gen_vision_mm_seq(this->k_proj.seq(), L, def->layer_def.k_proj, 1, 0);
            smolvla_vlm_vision_seq::gen_vision_mm_seq(this->v_proj.seq(), L, def->layer_def.v_proj, 1, 0);
            smolvla_vlm_vision_seq::gen_vision_attn_seq(this->mha.seq(), L_per_image, def->head_dim, def->num_attn_heads, L / L_per_image);
            smolvla_vlm_vision_seq::gen_vision_mm_seq(this->out_proj.seq(), L, def->layer_def.out_proj, 1, 0);
        }
        if (last_L != L) layer_runs.clear();
        last_L = L;

        // 2. q/k/v projections run concurrently in one runlist, then mha, then
        //    the output projection. No GQA here: DQ == DK == DV == D.
        PROF_INIT();
        const bool cp = smolvla_vlm_vision_seq::vision_l32() && smolvla_vlm_vision_seq::vision_ctrlpkt();
        auto build = [&](layer_runs_t& lr) {
            lr.rl = mgr->create_runlist();
            lr.rl.add(q_proj.create_run(q, patch, weights));
            lr.rl.add(k_proj.create_run(k, patch, weights));
            lr.rl.add(v_proj.create_run(v, patch, weights));
            // the switch's packets are binding 4 (kVisionPacketArg - 3); the output projection has three buffers,
            // so binding 3 is a filler there
            lr.mha_run = cp ? mha.create_run(attn_out, q, k, v, pk_attn32) : mha.create_run(attn_out, q, k, v);
            lr.o_run = cp ? out_proj.create_run(o, attn_out, weights, weights, pk_mm8)
                          : out_proj.create_run(o, attn_out, weights);
        
        };
        layer_runs_t local;
        layer_runs_t* lrp = &local;
        if (run_cache_on()) {
            auto ins = layer_runs.try_emplace(weights.data());
            lrp = &ins.first->second;
            if (ins.second) build(*lrp);
        } else {
            build(local);
        }
        layer_runs_t& lr = *lrp;
        PROF_TICK("vision.attn:init");
        lr.rl.execute();
        lr.rl.wait();
        PROF_TICK("vision.attn:qkv");
        lr.mha_run.start();
        lr.mha_run.wait();
        PROF_TICK("vision.attn:mha");
        lr.o_run.start();
        lr.o_run.wait();
        o.sync_from_device();
        PROF_TICK("vision.attn:o");
        return o;
    }
};

/// @brief feed-forward block: fc1 -> GELU -> fc2 (SigLIP MLP, biased).
struct SmolVLA_vision_mlp_block {
    static constexpr int MAX_L = smolvla_cfg::MAX_VISION_TOKENS;

    SmolVLA_Vision_def* def;

    int last_L = -1;

    npu_app fc1_proj;
    npu_app fc2_proj;
    // The same matmuls split along M, one image (CHUNK rows) each, so the CPU GELU of one image
    // runs while the NPU computes the next (SMOLVLA_MLP_PIPE=0 turns this off).
    static constexpr int CHUNK = smolvla_cfg::TOKENS_PER_IMAGE;
    static constexpr int MAX_CHUNKS = MAX_L / CHUNK;
    npu_app fc1_c[MAX_CHUNKS];
    npu_app fc2_c[MAX_CHUNKS];

    buffer<bf16> hid;     // fc1 output / activated hidden, [L, intermediate]
    buffer<bf16> o;     // fc1 output / activated hidden, [L, intermediate]

    /// per-layer chunk runs (SMOLVLA_VISION_RUNCACHE=0 rebuilds them on every call)
    struct mlp_runs_t {
        std::vector<xrt::run> r1, r2;
    };
    std::unordered_map<const void*, mlp_runs_t> mlp_runs;

    SmolVLA_vision_mlp_block(
        SmolVLA_Vision_def* def,
        npu_app_manager* vision_app_manager)
        : def(def){
        this->fc1_proj   = vision_app_manager->create_app();
        this->fc2_proj   = vision_app_manager->create_app();
        for (int c = 0; c < MAX_CHUNKS; c++) {
            fc1_c[c] = vision_app_manager->create_app();
            fc2_c[c] = vision_app_manager->create_app();
        }

        hid  = this->fc1_proj.create_bo_buffer<bf16>(MAX_L * def->intermediate_size);
        o  = this->fc2_proj.create_bo_buffer<bf16>(MAX_L * def->D);

    }

    static bool pipelined() {
        static const bool v = std::getenv("SMOLVLA_MLP_PIPE") == nullptr;
        return v;
    }

    /// fc1 -> CPU GELU -> fc2 per image: NPU order fc1[0..n), fc2[0..n); the GELU of chunk c runs on the host
    /// while the NPU computes fc1[c+1] (the last one while fc2[0]).
    buffer<bf16> forward_pipelined(buffer<bf16>& input, buffer<bf16>& weights, int L) {
        const int nc = L / CHUNK;
        const int N1 = (int)def->intermediate_size;
        if (last_L != L && smolvla_vlm_vision_seq::vision_l32()) {
            for (int c = 0; c < nc; c++) {
                smolvla_vlm_vision_seq::gen_vision_mm8_seq(fc1_c[c].seq(), CHUNK, def->layer_def.fc1, 0, c * CHUNK);
                smolvla_vlm_vision_seq::gen_vision_mm8_seq(fc2_c[c].seq(), CHUNK, def->layer_def.fc2, 0, c * CHUNK);
            }
        } else if (last_L != L) {
            // chunk 0 forces the RTPs (M = CHUNK, K, act): the attention sequences that precede/follow in
            // every layer rewrite them for M = L, and the sequence generator's skip logic is not touched.
            for (int c = 0; c < nc; c++) {
                smolvla_vlm_vision_seq::gen_vision_mm_seq(fc1_c[c].seq(), CHUNK, def->layer_def.fc1, 1, 0, c == 0, c * CHUNK);
                smolvla_vlm_vision_seq::gen_vision_mm_seq(fc2_c[c].seq(), CHUNK, def->layer_def.fc2, 1, 0, c == 0, c * CHUNK);
            }
        }
        if (last_L != L) mlp_runs.clear();
        last_L = L;
        PROF_INIT();
        mlp_runs_t local_runs;
        mlp_runs_t* mr = &local_runs;
        bool fresh = true;
        if (SmolVLA_vision_attn_block::run_cache_on()) {
            auto ins = mlp_runs.try_emplace(weights.data());
            mr = &ins.first->second;
            fresh = ins.second;
        }
        if (fresh) {
            for (int c = 0; c < nc; c++) {
                mr->r1.push_back(fc1_c[c].create_run(hid, input, weights));
                mr->r2.push_back(fc2_c[c].create_run(o, hid, weights));
            }
        }
        auto& r1 = mr->r1;
        auto& r2 = mr->r2;
        auto gelu_chunk = [&](int c) {
            const size_t off = (size_t)c * CHUNK * N1;
            hid.bo().sync(XCL_BO_SYNC_BO_FROM_DEVICE, (size_t)CHUNK * N1 * sizeof(bf16), off * sizeof(bf16));
            std::atomic_thread_fence(std::memory_order_seq_cst);
            cpu_func::_gelu_tanh(hid.data() + off, hid.data() + off, CHUNK * N1);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            hid.bo().sync(XCL_BO_SYNC_BO_TO_DEVICE, (size_t)CHUNK * N1 * sizeof(bf16), off * sizeof(bf16));
            std::atomic_thread_fence(std::memory_order_seq_cst);
        };
        PROF_TICK("vision.mlp:init");
        r1[0].start();
        r1[0].wait();
        for (int c = 1; c < nc; c++) {
            r1[c].start();
            gelu_chunk(c - 1);
            r1[c].wait();
        }
        PROF_TICK("vision.mlp:fc1+gelu");
        r2[0].start();
        gelu_chunk(nc - 1);
        r2[0].wait();
        for (int c = 1; c < nc; c++) {
            r2[c].start();
            r2[c].wait();
        }
        o.sync_from_device();
        PROF_TICK("vision.mlp:fc2");
        return o;
    }

    /// @brief run the feed-forward network.
    /// @param input   layer-normed hidden state, [L, D]
    /// @param weights this layer's packed weight buffer (holds fc1/fc2)
    /// @param L       total number of visual tokens across the batch
    /// @return mlp output, [L, D]
    buffer<bf16> forward(buffer<bf16>& input, buffer<bf16>& weights, int L) {
        if (pipelined() && L % CHUNK == 0 && L / CHUNK > 1 && L / CHUNK <= MAX_CHUNKS) return forward_pipelined(input, weights, L);
        // 1. (re)generate sequences / (re)alloc scratch as needed.
        if (last_L != L && smolvla_vlm_vision_seq::vision_l32()) {
            smolvla_vlm_vision_seq::gen_vision_mm8_seq(this->fc1_proj.seq(), L, def->layer_def.fc1, 0);
            smolvla_vlm_vision_seq::gen_vision_mm8_seq(this->fc2_proj.seq(), L, def->layer_def.fc2, 0);
        } else if (last_L != L) {
            smolvla_vlm_vision_seq::gen_vision_mm_seq(this->fc1_proj.seq(), L, def->layer_def.fc1, 1, 0);
            smolvla_vlm_vision_seq::gen_vision_mm_seq(this->fc2_proj.seq(), L, def->layer_def.fc2, 1, 0);
        }
        last_L = L;
        // 2. hid = gelu(fc1(input) + bias), [L, intermediate]. fc1 runs with
        //    rtp_act = 0, so the tanh-approximate GELU (gelu_pytorch_tanh) is
        //    applied here on the CPU.
        // 3. out = fc2(hid) + bias, [L, D]
        PROF_INIT();
        fc1_proj(hid, input, weights);
        hid.sync_from_device();

        PROF_TICK("vision.mlp:fc1");
        cpu_func::_gelu_tanh(hid.data(), hid.data(), L * (int)def->intermediate_size);
        hid.sync_to_device();
        PROF_TICK("vision.mlp:fc1_act");
        fc2_proj(o, hid, weights);
        o.sync_from_device();
        PROF_TICK("vision.mlp:fc2");
        return o;
    }
};

/// @brief top-level driver: patch-embed images and run the full SigLIP stack
///        over the batch, producing the visual-token sequences fed to the
///        connector. Attention still runs per-image (see gen_vision_attn_seq).
struct SmolVLA_vision_exec {
    static constexpr int L_per_image = smolvla_cfg::TOKENS_PER_IMAGE;
    static constexpr int MAX_L       = smolvla_cfg::MAX_VISION_TOKENS;

    SmolVLA_Vision_def* def;

    std::unique_ptr<SmolVLA_vision_attn_block> attn;
    std::unique_ptr<SmolVLA_vision_mlp_block>  mlp;
    
    buffer<bf16> patch_embedding;

    // conv2d patch-embedding weights, cached once. kernel == stride makes the
    // patch embedding an im2col + GEMM, and the [D, K] weight layout the
    // checkpoint already stores (K = 3*16*16 = 768) is exactly what _cpu_mm
    // wants: y[L,D] = im2col[L,K] * w[D,K]^T. So this is just a bf16 copy of the
    // checkpoint weight, and the matmul itself reuses the tuned bf16 kernel.
    std::vector<bf16>  conv_w;    // [D * K], row-major over (o, i)
    std::vector<float> conv_bias; // [D]
    std::vector<bf16>  conv_col;  // [L * K] im2col scratch, one row per patch

    buffer<bf16> x;           // layer-normed hidden state for the batch, [MAX_L, D]
    // xclbin app managers
    npu_app_manager* fused_xclbin = nullptr;

    SmolVLA_vision_exec() {}
    SmolVLA_vision_exec(npu_xclbin_manager* npu, SmolVLA_Vision_def* def)
        : def(def) {
        // register xclbin
        this->fused_xclbin = npu->register_xclbin(utils::xclbin_path(
            smolvla_vlm_vision_seq::vision_l32() ? "vision_mm8.xclbin" : "vision.xclbin"));
        // construct the two blocks with the shared common_bufs.

        auto init_load_app = this->fused_xclbin->create_app();
        init_load_app.seq()->clear_cmds();
        init_load_app.seq()->cmds2seq();
        init_load_app();
        init_load_app();

        patch_embedding.resize(L_per_image * def->D);
        x = fused_xclbin->create_bo_buffer<bf16>(MAX_L * def->D);
        this->attn = std::make_unique<SmolVLA_vision_attn_block>(def, fused_xclbin);
        this->mlp = std::make_unique<SmolVLA_vision_mlp_block>(def, fused_xclbin);
    }

    /// @brief one-time setup that depends on loaded weights. Prepacks the conv
    ///        patch-embedding weight so the first patch_embed() timing does not
    ///        include the (image-independent) prepack cost. Call after weights
    ///        are loaded.
    void init(buffer<u8>& non_layer_weights) {
        buffer<bf16> conv2d_w = def->patch_embedding.locate_myself<bf16>(non_layer_weights);
        buffer<bf16> conv2d_bias = def->patch_embedding_bias.locate_myself<bf16>(non_layer_weights);
        _conv2d_prepack(conv2d_w, conv2d_bias);
        init_patch_npu(conv2d_w, conv2d_bias);
    }

    // --- patch embedding on the vision mm app -------------------------------
    // y[M, D] = im2col[M, 768] * w[D, 768]^T + bias, M = images * 1024 (always a
    // multiple of 512). Weight BO layout is [bias D][w D*K], as the layer weights.
    npu_app pe_proj;
    buffer<bf16> pe_x, pe_y, pe_w;
    weight_desc_t pe_desc;
    bool pe_npu = false;

    void init_patch_npu(buffer<bf16>& w, buffer<bf16>& b) {
        if (std::getenv("SMOLVLA_PATCH_CPU")) return;
        const int K = smolvla_cfg::PATCH_CIN * smolvla_cfg::PATCH_SIZE * smolvla_cfg::PATCH_SIZE;
        const int D = (int)def->D;
        pe_proj = fused_xclbin->create_app();
        pe_x = pe_proj.create_bo_buffer<bf16>((size_t)MAX_L * K);
        pe_y = pe_proj.create_bo_buffer<bf16>((size_t)MAX_L * D);
        pe_w = pe_proj.create_bo_buffer<bf16>((size_t)D + (size_t)D * K);
        memcpy(pe_w.data(), b.data(), (size_t)D * sizeof(bf16));
        memcpy(pe_w.data() + D, w.data(), (size_t)D * K * sizeof(bf16));
        pe_w.sync_to_device();
        pe_desc = weight_desc_t(flm_bf16, {(uint32_t)K, (uint32_t)D}, "patch_npu");
        pe_desc.offset = (size_t)D * sizeof(bf16);
        pe_npu = true;
    }

    /// @brief embed the whole batch into dst ([num_images*L, D]): im2col per
    ///        image into the BO, one NPU matmul, then the position embedding.
    bool patch_embed_batch(bf16* dst, std::vector<buffer<bf16>>& pixels, buffer<u8>& non_layer_weights) {
        if (!pe_npu) return false;
        const int IM = smolvla_cfg::IMAGE_SIZE, PK = smolvla_cfg::PATCH_SIZE;
        const int PS = smolvla_cfg::PATCH_GRID, CIN = smolvla_cfg::PATCH_CIN;
        const int K = CIN * PK * PK, L = smolvla_cfg::TOKENS_PER_IMAGE, D = (int)def->D;
        const int n = (int)pixels.size();
        PROF_INIT();
        #pragma omp parallel for num_threads(cpu_func::max_prefill_thread) schedule(static)
        for (int g = 0; g < n * L; g++) {
            const int img = g / L, p = g % L;
            const int ph = p / PS, pwi = p % PS;
            bf16* d = pe_x.data() + (size_t)g * K;
            int idx = 0;
            for (int c = 0; c < CIN; c++)
                for (int kh = 0; kh < PK; kh++) {
                    const bf16* prow = pixels[img].data() + ((size_t)c * IM + ph * PK + kh) * IM + (size_t)pwi * PK;
                    for (int kw = 0; kw < PK; kw++) d[idx++] = prow[kw];
                }
        }
        pe_x.sync_to_device();
        PROF_TICK("vision.patch_embed:im2col");
        if (smolvla_vlm_vision_seq::vision_l32())
            smolvla_vlm_vision_seq::gen_vision_mm8_seq(pe_proj.seq(), n * L, pe_desc, 0);
        else
            smolvla_vlm_vision_seq::gen_vision_mm_seq(pe_proj.seq(), n * L, pe_desc, 1, 0, true);
        pe_proj(pe_y, pe_x, pe_w);
        pe_y.sync_from_device();
        PROF_TICK("vision.patch_embed:mm");
        buffer<bf16> pos = def->position_embedding.locate_myself<bf16>(non_layer_weights);
        for (int i = 0; i < n; i++)
            cpu_func::_residual_add(dst + (size_t)i * L * D, pe_y.data() + (size_t)i * L * D, pos.data(), L, D);
        PROF_TICK("vision.patch_embed:pos_embed_add");
        return true;
    }

    // --- connector on the vision mm app -------------------------------------
    // y[M, N] = x[M, K] * w[N, K]^T with K = 12288, N = 960 (no bias). The mm
    // kernel wants M % 512 == 0 and N % 256 == 0, so rows and output columns
    // are zero padded (M -> 512, N -> 1024); a zero bias row precedes the weight.
    static constexpr int CONN_M = 512;
    // (the 8-column matmul takes N in multiples of 384: 1152)
    static inline const int CONN_N = smolvla_vlm_vision_seq::vision_l32() ? 1152 : 1024;
    npu_app conn_proj;
    buffer<bf16> conn_x, conn_y, conn_w;
    weight_desc_t conn_desc;
    int conn_n_valid = 0;
    bool conn_npu = false;

    /// @param w [n_valid, K] connector weight as stored in the checkpoint
    void init_connector(const bf16* w, int n_valid) {
        if (std::getenv("SMOLVLA_CONN_CPU")) return;
        constexpr int K = smolvla_cfg::CONNECTOR_IN;
        conn_n_valid = n_valid;
        conn_proj = fused_xclbin->create_app();
        conn_x = conn_proj.create_bo_buffer<bf16>((size_t)CONN_M * K);
        conn_y = conn_proj.create_bo_buffer<bf16>((size_t)CONN_M * CONN_N);
        conn_w = conn_proj.create_bo_buffer<bf16>((size_t)CONN_N + (size_t)CONN_N * K);
        memset(conn_x.data(), 0, conn_x.size() * sizeof(bf16));
        memset(conn_w.data(), 0, conn_w.size() * sizeof(bf16));
        memcpy(conn_w.data() + CONN_N, w, (size_t)n_valid * K * sizeof(bf16));
        conn_w.sync_to_device();
        conn_desc = weight_desc_t(flm_bf16, {(uint32_t)K, (uint32_t)CONN_N}, "connector_npu");
        conn_desc.offset = (size_t)CONN_N * sizeof(bf16);
        conn_npu = true;
    }

    /// @brief dst[M, dst_stride] = src[M, K] * w^T. Returns false when the NPU
    ///        path is off or M does not fit, so the caller runs the CPU matmul.
    bool connector(bf16* dst, int dst_stride, const bf16* src, int M) {
        constexpr int K = smolvla_cfg::CONNECTOR_IN;
        if (!conn_npu || M > CONN_M) return false;
        PROF_INIT();
        if (smolvla_vlm_vision_seq::vision_l32())
            smolvla_vlm_vision_seq::gen_vision_mm8_seq(conn_proj.seq(), CONN_M, conn_desc, 0);
        else
            smolvla_vlm_vision_seq::gen_vision_mm_seq(conn_proj.seq(), CONN_M, conn_desc, 1, 0, true);
        memcpy(conn_x.data(), src, (size_t)M * K * sizeof(bf16));
        conn_x.sync_to_device();
        conn_proj(conn_y, conn_x, conn_w);
        conn_y.sync_from_device();
        for (int r = 0; r < M; r++)
            memcpy(dst + (size_t)r * dst_stride, conn_y.data() + (size_t)r * CONN_N,
                   conn_n_valid * sizeof(bf16));
        PROF_TICK("vision.connector_npu");
        return true;
    }

    /// @brief run one pre-norm SigLIP encoder layer, updating residual in place.
    ///        residual = residual + attn(ln1(residual)); then
    ///        residual = residual + mlp(ln2(residual)).
    /// @param residual   running hidden state for the whole batch, updated in
    ///                    place, [num_images*L_per_image, D]
    /// @param weights    packed NPU projection weights for this layer
    /// @param norms      packed CPU-only layer-norm weights for this layer
    /// @param num_images number of images in the batch (L = num_images*L_per_image)
    /// @return reference to the updated residual buffer, [num_images*L_per_image, D]
    buffer<bf16> forward(buffer<bf16>& residual, buffer<bf16>& weights, buffer<bf16>& norms, int num_images) {
        const int L  = num_images * L_per_image;
        const int D  = (int)def->D;
        const int valid_size = L * D;
        constexpr float eps = smolvla_cfg::VISION_NORM_EPS;
        SmolVLA_vision_layer& ld = def->layer_def;

        auto N = [&](weight_desc_t& d) { return d.locate_myself<bf16>(norms); };

        // --- attention sub-block ---
        PROF_INIT();
        buffer<bf16> ln1_w = N(ld.layer_norm1), ln1_b = N(ld.layer_norm1_bias);
        cpu_func::_layer_norm_biased(x, residual, ln1_w, ln1_b, L, D, eps);
        x.sync_to_device();
        PROF_TICK("vision.layer:layer_norm1");
        auto attn_out = this->attn->forward(x, weights, L);
        cpu_func::_residual_add(residual, residual, attn_out, valid_size);
        PROF_TICK("vision.layer:attn+residual");

        // --- mlp sub-block ---
        buffer<bf16> ln2_w = N(ld.layer_norm2), ln2_b = N(ld.layer_norm2_bias);
        cpu_func::_layer_norm_biased(x, residual, ln2_w, ln2_b, L, D, eps);
        x.sync_to_device();
        PROF_TICK("vision.layer:layer_norm2");
        auto block_out = this->mlp->forward(x, weights, L);
        cpu_func::_residual_add(residual, residual, block_out, valid_size);
        PROF_TICK("vision.layer:mlp+residual");

        return residual;
    }

    /// @brief final post-layernorm over [L, D], in place.
    void post_layernorm(buffer<bf16>& hidden, buffer<u8>& non_layer_weights) {
        const int L = (int)(hidden.size() / def->D);
        buffer<bf16> g = def->post_layernorm.locate_myself<bf16>(non_layer_weights);
        buffer<bf16> b = def->post_layernorm_bias.locate_myself<bf16>(non_layer_weights);
        PROF_INIT();
        cpu_func::_layer_norm_biased(hidden, hidden, g, b, L, (int)def->D, smolvla_cfg::VISION_NORM_EPS);
        PROF_TICK("vision.post_layernorm");
    }

    /// @brief connector pixel-shuffle: [TOKENS_PER_IMAGE, D] ->
    ///        [VISION_TOKENS_PER_IMAGE, D*sf*sf]. Faithful index map from
    ///        transformers SmolVLM pixel_shuffle with height = width = PATCH_GRID.
    void pixel_shuffle(buffer<bf16>& dst, buffer<bf16>& hidden) {
        const int D  = (int)def->D;
        const int PS = smolvla_cfg::PATCH_GRID;         // patches per side
        const int sf = smolvla_cfg::SHUFFLE_FACTOR;
        const int out_dim = D * sf * sf;
        const int W2 = smolvla_cfg::SHUFFLED_GRID;      // PS / sf

        // Working the index map through by hand collapses it to
        //   dst[t][(q*sf + u)*D + e] = hidden[((sf*bb + q)*PS + sf*aa + u)*D + e]
        // for q,u in [0,sf) and e in [0,D). e is contiguous and spans the whole
        // channel dim, so each (q,u) is one D-element run: the shuffle is
        // sf*sf contiguous copies per output token, not out_dim scalar gathers
        // with two integer divisions apiece.
        PROF_INIT();
        const bf16* src_base = hidden.data();
        #pragma omp parallel for num_threads(cpu_func::max_prefill_thread) schedule(static)
        for (int t = 0; t < W2 * W2; t++) {
            const int bb = t / W2;
            const int aa = t % W2;
            bf16* sr = dst.data() + (size_t)t * out_dim;
            for (int q = 0; q < sf; q++) {
                const int src_row0 = (sf * bb + q) * PS + sf * aa;
                for (int u = 0; u < sf; u++) {
                    memcpy(sr + (size_t)(q * sf + u) * D,
                           src_base + (size_t)(src_row0 + u) * D,
                           (size_t)D * sizeof(bf16));
                }
            }
        }
        PROF_TICK("vision.pixel_shuffle");
    }

    /// @brief conv2d patch embedding: a PATCH_SIZE x PATCH_SIZE stride-PATCH_SIZE
    ///        kernel over a [PATCH_CIN, IMAGE_SIZE, IMAGE_SIZE] image reduces to
    ///        PATCH_GRID^2 = TOKENS_PER_IMAGE non-overlapping patches, each
    ///        projected to D channels. Since kernel==stride, this is an im2col +
    ///        matmul: gather each patch's PATCH_CIN*PATCH_SIZE^2 input values once,
    ///        then dot against all D output filters. Weight layout is
    ///        [D, PATCH_CIN, PATCH_SIZE, PATCH_SIZE], accumulated in float and
    ///        written into patch_embedding as [TOKENS_PER_IMAGE, D].
    /// @brief cache the conv weight/bias once, in the layout the GEMM wants.
    ///        The checkpoint already stores [D, PATCH_CIN, PATCH_SIZE,
    ///        PATCH_SIZE] == [D, K], which is _cpu_mm's weight layout as-is, so
    ///        no transpose is needed -- just a copy plus the bias widened to
    ///        fp32 (the matmul folds it in before rounding to bf16).
    void _conv2d_prepack(buffer<bf16>& w, buffer<bf16>& b) {
        const int PK = smolvla_cfg::PATCH_SIZE;
        const int D  = (int)def->D;
        const int K  = smolvla_cfg::PATCH_CIN * PK * PK;
        const int L  = smolvla_cfg::TOKENS_PER_IMAGE;
        if (!conv_w.empty()) return;        // prepack once, reuse across cameras
        conv_w.assign(w.data(), w.data() + (size_t)D * K);
        conv_bias.resize(D);
        for (int o = 0; o < D; o++) conv_bias[o] = float(b[o]);
        conv_col.assign((size_t)L * K, bf16(0.0f));
    }

    void _conv2d(buffer<bf16>& pixel_values, buffer<bf16>& w, buffer<bf16>& b){
        (void)w; (void)b;                   // weights cached in conv_w in init()
        const int IM = smolvla_cfg::IMAGE_SIZE;     // image size
        const int PK = smolvla_cfg::PATCH_SIZE;     // patch kernel / stride
        const int PS = smolvla_cfg::PATCH_GRID;     // patches per side
        const int CIN = smolvla_cfg::PATCH_CIN;     // input channels
        const int D  = (int)def->D;                 // output channels
        const int K  = CIN * PK * PK;               // im2col length
        const int L = smolvla_cfg::TOKENS_PER_IMAGE;    // total patches

        // kernel == stride, so every patch reads a disjoint window: gather each
        // patch's K inputs into one contiguous row and the whole embedding is a
        // single [L,K] x [D,K]^T GEMM. Handing that to _cpu_mm gets the blocked
        // VDPBF16PS kernel (which reuses each weight vector across 4 patches and
        // needs no fp32 im2col staging) instead of a bespoke fp32 broadcast loop.
        #pragma omp parallel for num_threads(cpu_func::max_prefill_thread) schedule(static)
        for (int p = 0; p < L; p++) {
            const int ph  = p / PS;
            const int pwi = p % PS;
            bf16* dst = conv_col.data() + (size_t)p * K;
            int idx = 0;
            for (int c = 0; c < CIN; c++) {
                for (int kh = 0; kh < PK; kh++) {
                    const int iy = ph * PK + kh;
                    const bf16* prow = pixel_values.data()
                                     + ((size_t)c * IM + iy) * IM + (size_t)pwi * PK;
                    for (int kw = 0; kw < PK; kw++) dst[idx++] = prow[kw];
                }
            }
        }

        cpu_func::_cpu_mm(patch_embedding.data(), conv_w.data(), conv_col.data(),
                          L, K, D, 0, conv_bias.data());
    }

    /// @brief add the learned position embedding in place (position_ids identity).
    void _elemwise_add(buffer<bf16>& pe){
        // in-place bf16 add, AVX512 + OpenMP (16 lanes/iter, converts via fp32).
        cpu_func::_residual_add(patch_embedding, patch_embedding, pe, patch_embedding.size());
    }

    /// @brief embed one image: conv2d patch embedding + learned position
    ///        embedding, writing the [TOKENS_PER_IMAGE, D] token grid into dst.
    /// @param dst        destination slice for this image's tokens, [TOKENS_PER_IMAGE, D]
    /// @param pixel_values raw pixel buffer for this image,
    ///                     [PATCH_CIN, IMAGE_SIZE, IMAGE_SIZE]
    /// @param non_layer_weights packed non-layer weights (conv + pos embedding)
    void patch_embed(buffer<bf16>& dst, buffer<bf16>& pixel_values, buffer<u8>& non_layer_weights) {
        // conv2d patch embedding, flattened to [TOKENS_PER_IMAGE, D].
        buffer<bf16> conv2d_w = def->patch_embedding.locate_myself<bf16>(non_layer_weights);
        buffer<bf16> conv2d_bias = def->patch_embedding_bias.locate_myself<bf16>(non_layer_weights);
        buffer<bf16> positional_embedding = def->position_embedding.locate_myself<bf16>(non_layer_weights);

        PROF_INIT();
        _conv2d(pixel_values, conv2d_w, conv2d_bias);
        PROF_TICK("vision.patch_embed:conv2d");
        _elemwise_add(positional_embedding);
        PROF_TICK("vision.patch_embed:pos_embed_add");

        memcpy(dst.data(), patch_embedding.data(), patch_embedding.size() * sizeof(bf16));
    }


};

#endif // __SMOLVLA__VISION__EXEC__HPP__
