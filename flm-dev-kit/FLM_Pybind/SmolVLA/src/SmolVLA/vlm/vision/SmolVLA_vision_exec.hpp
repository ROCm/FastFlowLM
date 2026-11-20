// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__VISION__EXEC__HPP__
#define __SMOLVLA__VISION__EXEC__HPP__

#include <memory>
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

    buffer<bf16> q;
    buffer<bf16> k;
    buffer<bf16> v;
    buffer<bf16> attn_out;
    buffer<bf16> o;

    SmolVLA_vision_attn_block(
        SmolVLA_Vision_def* def,
        npu_app_manager* vision_app_manager)
        : def(def){
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

        if (last_L != L) {
            smolvla_vlm_vision_seq::gen_vision_mm_seq(this->q_proj.seq(), L, def->layer_def.q_proj, 1, 0);
            smolvla_vlm_vision_seq::gen_vision_mm_seq(this->k_proj.seq(), L, def->layer_def.k_proj, 1, 0);
            smolvla_vlm_vision_seq::gen_vision_mm_seq(this->v_proj.seq(), L, def->layer_def.v_proj, 1, 0);
            smolvla_vlm_vision_seq::gen_vision_attn_seq(this->mha.seq(), L_per_image, def->head_dim, def->num_attn_heads, L / L_per_image);
            smolvla_vlm_vision_seq::gen_vision_mm_seq(this->out_proj.seq(), L, def->layer_def.out_proj, 1, 0);
        }
        last_L = L;

        // 2. q/k/v projections run concurrently in one runlist, then mha, then
        //    the output projection. No GQA here: DQ == DK == DV == D.
        PROF_INIT();
        rl.reset();
        rl.add(q_proj.create_run(q, patch, weights));
        rl.add(k_proj.create_run(k, patch, weights));
        rl.add(v_proj.create_run(v, patch, weights));
        auto mha_run = mha.create_run(attn_out, q, k, v);
        auto o_run = out_proj.create_run(o, attn_out, weights);
        
        PROF_TICK("vision.attn:init");
        rl.execute();
        rl.wait();
        PROF_TICK("vision.attn:qkv");
        mha_run.start();
        mha_run.wait();
        PROF_TICK("vision.attn:mha");
        o_run.start();
        o_run.wait();
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

    buffer<bf16> hid;     // fc1 output / activated hidden, [L, intermediate]
    buffer<bf16> o;     // fc1 output / activated hidden, [L, intermediate]

    SmolVLA_vision_mlp_block(
        SmolVLA_Vision_def* def,
        npu_app_manager* vision_app_manager)
        : def(def){
        this->fc1_proj   = vision_app_manager->create_app();
        this->fc2_proj   = vision_app_manager->create_app();

        hid  = this->fc1_proj.create_bo_buffer<bf16>(MAX_L * def->intermediate_size);
        o  = this->fc2_proj.create_bo_buffer<bf16>(MAX_L * def->D);

    }

    /// @brief run the feed-forward network.
    /// @param input   layer-normed hidden state, [L, D]
    /// @param weights this layer's packed weight buffer (holds fc1/fc2)
    /// @param L       total number of visual tokens across the batch
    /// @return mlp output, [L, D]
    buffer<bf16> forward(buffer<bf16>& input, buffer<bf16>& weights, int L) {
        // 1. (re)generate sequences / (re)alloc scratch as needed.
        if (last_L != L) {
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
        this->fused_xclbin = npu->register_xclbin(utils::xclbin_path("vision.xclbin"));
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
