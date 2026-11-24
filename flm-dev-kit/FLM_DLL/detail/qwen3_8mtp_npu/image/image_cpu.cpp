// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/image_cpu.cpp
#include "image/image_cpu.hpp"

#include "image/vision_glue.hpp"
#include "image/vision_par.hpp"
#include "image/vision_profile.hpp"
#include "image/vision_rope.hpp"

#include <cstring>
#include <stdexcept>

namespace qwen38 {

using vision::activation;

image_cpu::image_cpu(qwen3_8mtp_vision_desc& desc,
                     std::unique_ptr<vision::op_factory> factory)
    : desc_(desc), factory_(std::move(factory)) {
    if (!desc_.present)
        throw std::runtime_error("image_cpu built for a checkpoint with no "
                                 "vision_config");

    // Opening the checkpoint corrects desc_.D_ff from the shipped tensors, so
    // it must happen before any op is bound to a width.
    w_ = std::make_unique<vision::weights>(desc_);

    const int D  = (int)desc_.D;
    const int F  = (int)desc_.D_ff;
    const int PD = (int)desc_.patch_dim();
    const int MI = (int)desc_.merger_in();

    patch_mm_ = factory_->make_mm("patch_embed");
    patch_mm_->bind(w_->patch_w(), w_->patch_b(), PD, D, activation::none);

    block_ops_.resize(desc_.depth);
    for (uint32_t i = 0; i < desc_.depth; i++) {
        const vision::block_weights& bw = w_->block((int)i);
        block_ops& o = block_ops_[i];
        const std::string tag = "block" + std::to_string(i);
        o.qkv  = factory_->make_mm((tag + ".qkv").c_str());
        o.proj = factory_->make_mm((tag + ".proj").c_str());
        o.fc1  = factory_->make_mm((tag + ".fc1").c_str());
        o.fc2  = factory_->make_mm((tag + ".fc2").c_str());
        o.qkv ->bind(bw.qkv_w,  bw.qkv_b,  D, 3 * D, activation::none);
        o.proj->bind(bw.proj_w, bw.proj_b, D, D,     activation::none);
        // The activation is fused into fc1, which is what the vision_mm kernel
        // does with rtp_act. Doing it as a separate host pass would cost an
        // extra [L][4608] round trip per block for nothing.
        o.fc1 ->bind(bw.fc1_w,  bw.fc1_b,  D, F,     activation::gelu);
        o.fc2 ->bind(bw.fc2_w,  bw.fc2_b,  F, D,     activation::none);
    }

    merger_fc1_ = factory_->make_mm("merger.fc1");
    merger_fc2_ = factory_->make_mm("merger.fc2");
    // gelu_erf, NOT gelu: Qwen3_5VisionPatchMerger hardcodes nn.GELU() while
    // the block MLPs above take vision_config.hidden_act, which this checkpoint
    // sets to gelu_pytorch_tanh. See the note on activation::gelu_erf.
    merger_fc1_->bind(w_->merger_fc1_w(), w_->merger_fc1_b(), MI, MI,
                      activation::gelu_erf);
    merger_fc2_->bind(w_->merger_fc2_w(), w_->merger_fc2_b(), MI,
                      (int)desc_.out_hidden_size, activation::none);

    attn_ = factory_->make_attn("vision_attn");

    backend_ = std::string("mm=") + patch_mm_->backend() +
               " attn=" + attn_->backend() +
               (vision::cpu_ops_emulate_bf16() ? " bf16" : " fp32");
}

void image_cpu::_ensure_scratch(int L) {
    if (L <= scratch_L_) return;
    const size_t D  = desc_.D;
    const size_t F  = desc_.D_ff;
    const size_t MI = desc_.merger_in();
    const size_t MU = desc_.merge_unit();

    x_.assign((size_t)L * D, 0.0f);
    h_.assign((size_t)L * D, 0.0f);
    qkv_.assign((size_t)L * 3 * D, 0.0f);
    q_.assign((size_t)L * D, 0.0f);
    k_.assign((size_t)L * D, 0.0f);
    v_.assign((size_t)L * D, 0.0f);
    attn_o_.assign((size_t)L * D, 0.0f);
    proj_.assign((size_t)L * D, 0.0f);
    ff_.assign((size_t)L * F, 0.0f);
    pos_.assign((size_t)L * D, 0.0f);
    merged_.assign(((size_t)L / MU + 1) * MI, 0.0f);
    scratch_L_ = L;
}

void image_cpu::_block(int idx, int L, const vision::rope_tables& rope) {
    const int D  = (int)desc_.D;
    const int H  = (int)desc_.num_heads;
    const vision::block_weights& bw = w_->block(idx);
    block_ops& op = block_ops_[(size_t)idx];

    const std::string pre = "block_" + (idx < 10 ? "0" + std::to_string(idx)
                                                 : std::to_string(idx));

    // ---- attention sublayer ------------------------------------------------
    { VPROF(norm_us);
      vision::layer_norm(x_.data(), L, D, bw.norm1_w.data(), bw.norm1_b.data(),
                         desc_.ln_eps, h_.data()); }
    _emit((pre + ".norm1").c_str(), h_.data(), L, D);

    op.qkv->run(h_.data(), L, qkv_.data());

    // nn.Linear(D, 3D) viewed as (L, 3, heads, dh): q is the first D of each
    // row, k the next D, v the last, head-major inside each third. Attention
    // wants a row stride of D, not 3D, so split the thirds into planes --
    // splitting on the wrong axis is a permutation attention consumes happily.
    float* q = q_.data();
    float* k = k_.data();
    float* v = v_.data();
    { VPROF(split_us);
      // 3 x 1.38M floats moved per block. Memory bound, but it was on one core.
#pragma omp parallel for schedule(static) num_threads(vision::par_threads())
      for (int t = 0; t < L; t++) {
          const float* r = qkv_.data() + (size_t)t * 3 * D;
          std::memcpy(q + (size_t)t * D, r,         sizeof(float) * D);
          std::memcpy(k + (size_t)t * D, r + D,     sizeof(float) * D);
          std::memcpy(v + (size_t)t * D, r + 2 * D, sizeof(float) * D);
      } }

    { VPROF(rope_us);
      vision::apply_rope(q, L, H, rope);
      vision::apply_rope(k, L, H, rope); }
    _emit((pre + ".q_rope").c_str(), q, L, D);
    _emit((pre + ".k_rope").c_str(), k, L, D);

    attn_->run(q, k, v, L, attn_o_.data());
    _emit((pre + ".attn").c_str(), attn_o_.data(), L, D);

    op.proj->run(attn_o_.data(), L, proj_.data());
    // Emitted BEFORE the residual add, because that is where
    // Qwen3_5VisionAttention.forward returns and so where the reference dump's
    // `block_NN.attn_proj` is hooked. `.attn` above is the PRE-proj value and
    // has no counterpart in the dump; the two names must not be paired.
    _emit((pre + ".attn_proj").c_str(), proj_.data(), L, D);
    { VPROF(resid_us);
      vision::add_inplace(x_.data(), proj_.data(), (size_t)L * D); }
    _emit((pre + ".attn_residual").c_str(), x_.data(), L, D);

    // ---- MLP sublayer ------------------------------------------------------
    { VPROF(norm_us);
      vision::layer_norm(x_.data(), L, D, bw.norm2_w.data(), bw.norm2_b.data(),
                         desc_.ln_eps, h_.data()); }
    op.fc1->run(h_.data(), L, ff_.data());          // gelu fused
    op.fc2->run(ff_.data(), L, proj_.data());
    { VPROF(resid_us);
      vision::add_inplace(x_.data(), proj_.data(), (size_t)L * D); }
    _emit((pre + ".out").c_str(), x_.data(), L, D);
}

void image_cpu::encode(const float* pixel_values, image_grid g, float* out) {
    const int D  = (int)desc_.D;
    const int MU = (int)desc_.merge_unit();
    const int L  = desc_.patch_count(g.t, g.h, g.w);
    if (L <= 0)
        throw std::runtime_error("image_cpu::encode got an empty grid");
    if (L % MU)
        throw std::runtime_error("image grid produces " + std::to_string(L) +
                                 " patches, not a multiple of the merge unit");

    _ensure_scratch(L);

    // The attention op is sized for the largest image seen so far. Reconfiguring
    // on every call would rebuild a device sequence per image.
    if (L > attn_configured_L_) {
        attn_->configure(L, (int)desc_.num_heads, (int)desc_.DH, desc_.attn_scale);
        attn_configured_L_ = L;
    }

    // ---- patch embedding + learned position grid ---------------------------
    patch_mm_->run(pixel_values, L, x_.data());
    _emit("patch_embed", x_.data(), L, D);

    { VPROF(pos_us);
      vision::pos_embed_interpolate(w_->pos_embed(),
                                    (int)desc_.num_grid_per_side, D,
                                    g.t, g.h, g.w,
                                    (int)desc_.spatial_merge_size,
                                    pos_.data()); }
    _emit("pos_embed", pos_.data(), L, D);
    { VPROF(resid_us);
      vision::add_inplace(x_.data(), pos_.data(), (size_t)L * D); }
    _emit("embed", x_.data(), L, D);

    // ---- 27 blocks ---------------------------------------------------------
    const vision::rope_tables rope = VPROF_EXPR(pos_us,
        vision::build_rope(g.t, g.h, g.w, (int)desc_.spatial_merge_size,
                           (int)desc_.DH, desc_.rope_theta));
    for (uint32_t i = 0; i < desc_.depth; i++) _block((int)i, L, rope);

    // ---- merger ------------------------------------------------------------
    // The norm is over D per patch, BEFORE the 2x2 regroup: merger.norm.weight
    // is [1152], not [4608], which is how the checkpoint says
    // use_postshuffle_norm is false.
    { VPROF(norm_us);
      vision::layer_norm(x_.data(), L, D, w_->merger_norm_w().data(),
                         w_->merger_norm_b().data(), desc_.ln_eps,
                         h_.data()); }
    _emit("merger_norm", h_.data(), L, D);

    // The regroup itself is free: merge-block order already put patches
    // 4s..4s+3 consecutively, so [L][1152] reinterpreted as [L/4][4608] is the
    // merger's input with no copy. See the note in vision_glue.hpp.
    const int rows = L / MU;
    merger_fc1_->run(h_.data(), rows, merged_.data());   // gelu fused
    _emit("merger_fc1", merged_.data(), rows, (int)desc_.merger_in());
    merger_fc2_->run(merged_.data(), rows, out);
    _emit("image_embeds", out, rows, (int)desc_.out_hidden_size);
}

}  // namespace qwen38
