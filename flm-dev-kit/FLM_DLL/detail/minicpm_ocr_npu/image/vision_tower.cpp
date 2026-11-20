// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/vision_tower.cpp
#include "image/vision_tower.hpp"

#include "image/vision_glue.hpp"
#include "image/vision_par.hpp"
#include "image/vision_profile.hpp"

#include <algorithm>
#include <stdexcept>

namespace mco {

using vision::activation;

namespace {
std::string block_name(int idx) {
    return std::string("block_") + (idx < 10 ? "0" : "") + std::to_string(idx);
}
}  // namespace

vision_tower::vision_tower(minicpm_ocr_vision_desc& desc,
                           std::unique_ptr<vision::op_factory> factory)
    : desc_(desc), factory_(std::move(factory)) {
    if (!desc_.present)
        throw std::runtime_error("vision_tower built for a checkpoint with no vision_model_weight");

    // Opening the checkpoint corrects desc_.D_ff / vm_ff from the shipped
    // tensors, so it must happen before any op is bound to a width.
    w_ = std::make_unique<vision::weights>(desc_);

    const int D  = (int)desc_.D;
    const int F  = (int)desc_.D_ff;
    const int PD = (int)desc_.patch_dim();
    const int MI = (int)desc_.merger_in();

    // K = 588 is not a multiple of vision_mm's 384; the device op pads it.
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
        o.qkv ->bind(bw.qkv.w,  bw.qkv.b,  D, 3 * D, activation::none);
        o.proj->bind(bw.proj.w, bw.proj.b, D, D,     activation::none);
        // gelu_pytorch_tanh, fused: what vision_mm's rtp_act 2 computes.
        o.fc1 ->bind(bw.fc1.w,  bw.fc1.b,  D, F,     activation::gelu);
        o.fc2 ->bind(bw.fc2.w,  bw.fc2.b,  F, D,     activation::none);
    }

    if (desc_.use_vit_merger()) {
        const vision::vit_merger_weights& vw = w_->vit_merger();
        const int VI = (int)desc_.vm_in();
        const int VF = (int)desc_.vm_ff;
        vm_ops_.qkv  = factory_->make_mm("vit_merger.qkv");
        vm_ops_.proj = factory_->make_mm("vit_merger.proj");
        vm_ops_.fc1  = factory_->make_mm("vit_merger.fc1");
        vm_ops_.fc2  = factory_->make_mm("vit_merger.fc2");
        vm_ops_.qkv ->bind(vw.qkv.w,  vw.qkv.b,  D,  3 * D, activation::none);
        vm_ops_.proj->bind(vw.proj.w, vw.proj.b, D,  D,     activation::none);
        // tanh, NOT erf: the vit_merger hardcodes gelu_pytorch_tanh
        // (modeling_minicpmv4_6.py:311). Only the projector below is erf.
        vm_ops_.fc1 ->bind(vw.fc1.w,  vw.fc1.b,  VI, VF,    activation::gelu);
        vm_ops_.fc2 ->bind(vw.fc2.w,  vw.fc2.b,  VF, D,     activation::none);
    }

    merger_fc1_ = factory_->make_mm("merger.fc1");
    merger_fc2_ = factory_->make_mm("merger.fc2");
    // gelu_erf: MiniCPMV4_7DownsampleMLP is nn.GELU(). vision_mm has no erf, so
    // the device op binds `none` and keeps this epilogue on the host.
    merger_fc1_->bind(w_->merger_fc1().w, w_->merger_fc1().b, MI, MI, activation::gelu_erf);
    merger_fc2_->bind(w_->merger_fc2().w, w_->merger_fc2().b, MI,
                      (int)desc_.out_hidden_size, activation::none);

    attn_ = factory_->make_attn("vision_attn");

    backend_ = std::string("mm=") + patch_mm_->backend() + " attn=" + attn_->backend() +
               (vision::cpu_ops_emulate_bf16() ? " bf16" : " fp32");
}

void vision_tower::_ensure_scratch(int L) {
    if (L <= scratch_L_) return;
    const size_t D  = desc_.D;
    const size_t F  = std::max<size_t>(desc_.D_ff, (desc_.vm_ff + 3) / 4);
    const size_t MI = std::max(desc_.merger_in(), desc_.vm_in());
    const size_t Lq = ((size_t)L + 3) / 4;

    x_.assign((size_t)L * D, 0.0f);
    h_.assign((size_t)L * D, 0.0f);
    qkv_.assign((size_t)L * 3 * D, 0.0f);
    attn_o_.assign((size_t)L * D, 0.0f);
    proj_.assign((size_t)L * D, 0.0f);
    ff_.assign((size_t)L * F, 0.0f);
    merged_.assign(Lq * MI, 0.0f);
    merged_n_.assign(Lq * MI, 0.0f);
    mean_.assign(Lq * D, 0.0f);
    scratch_L_ = L;
}

void vision_tower::_prepare(const std::vector<view_grid>& views, int L) {
    // Every op's M for this call, then all the device sequences at once. The
    // ops are independent (one app each) and the per-op cost is CPU-side ELF
    // assembly plus a module/kernel build, so this is a plain parallel loop.
    // Cheap when nothing changed: prepare() is a compare per op.
    struct job { vision::mm_op* op; int M; };
    std::vector<job> jobs;
    const int Lq = desc_.use_vit_merger() ? L / 4 : L;
    jobs.push_back({patch_mm_.get(), L});
    for (uint32_t i = 0; i < desc_.depth; i++) {
        const int M = ((int32_t)i <= desc_.insert_layer_id) ? L : Lq;
        for (vision::mm_op* o : {block_ops_[i].qkv.get(), block_ops_[i].proj.get(),
                                 block_ops_[i].fc1.get(), block_ops_[i].fc2.get()})
            jobs.push_back({o, M});
    }
    if (desc_.use_vit_merger()) {
        jobs.push_back({vm_ops_.qkv.get(), L});
        jobs.push_back({vm_ops_.proj.get(), L});
        jobs.push_back({vm_ops_.fc1.get(), L / 4});
        jobs.push_back({vm_ops_.fc2.get(), L / 4});
    }
    jobs.push_back({merger_fc1_.get(), Lq / 4});
    jobs.push_back({merger_fc2_.get(), Lq / 4});

    std::vector<int> full, quarter;
    for (const view_grid& v : views) {
        full.push_back(v.h * v.w);
        quarter.push_back(v.h * v.w / 4);
    }
    attn_->prepare(full);
    if (desc_.use_vit_merger()) attn_->prepare(quarter);

#pragma omp parallel for schedule(dynamic, 1) num_threads(vision::par_threads())
    for (int j = 0; j < (int)jobs.size(); j++) jobs[(size_t)j].op->prepare(jobs[(size_t)j].M);
}

void vision_tower::_block(int idx, const std::vector<view_grid>& views, int L) {
    const int D = (int)desc_.D;
    const vision::block_weights& bw = w_->block(idx);
    block_ops& op = block_ops_[(size_t)idx];
    const std::string pre = block_name(idx);

    // ---- attention sublayer ------------------------------------------------
    { VPROF(norm_us);
      vision::layer_norm(x_.data(), L, D, bw.norm1_w.data(), bw.norm1_b.data(),
                         desc_.ln_eps, h_.data()); }
    _emit((pre + ".norm1").c_str(), h_.data(), L, D);

    op.qkv->run(h_.data(), L, qkv_.data());

    // Block-diagonal over VIEWS: each view attends only within itself. q, k and
    // v are read in place out of the fused [L][3D] projection.
    std::vector<int> lengths;
    for (const view_grid& v : views) lengths.push_back(v.h * v.w);
    attn_->run_views(qkv_.data(), qkv_.data() + D, qkv_.data() + 2 * D, 3 * D, lengths,
                     attn_o_.data());

    op.proj->run(attn_o_.data(), L, proj_.data());
    // After out_proj and before the residual: where MiniCPMV4_7VisionAttention
    // returns, so where the reference's `block_NN.attn_proj` is hooked.
    _emit((pre + ".attn_proj").c_str(), proj_.data(), L, D);
    { VPROF(resid_us); vision::add_inplace(x_.data(), proj_.data(), (size_t)L * D); }

    // ---- MLP sublayer ------------------------------------------------------
    { VPROF(norm_us);
      vision::layer_norm(x_.data(), L, D, bw.norm2_w.data(), bw.norm2_b.data(),
                         desc_.ln_eps, h_.data()); }
    op.fc1->run(h_.data(), L, ff_.data());          // gelu_tanh fused
    op.fc2->run(ff_.data(), L, proj_.data());
    { VPROF(resid_us); vision::add_inplace(x_.data(), proj_.data(), (size_t)L * D); }
    _emit((pre + ".out").c_str(), x_.data(), L, D);
}

void vision_tower::_vit_merger(std::vector<view_grid>& views, int L) {
    const int D  = (int)desc_.D;
    const int H  = (int)desc_.num_heads;
    const int VI = (int)desc_.vm_in();
    const int VF = (int)desc_.vm_ff;
    const int Lq = L / 4;
    const vision::vit_merger_weights& vw = w_->vit_merger();

    // ---- windowed attention + residual #1 ----------------------------------
    { VPROF(norm_us);
      vision::layer_norm(x_.data(), L, D, vw.norm1_w.data(), vw.norm1_b.data(),
                         desc_.ln_eps, h_.data()); }
    _emit("vm.norm1", h_.data(), L, D);

    vm_ops_.qkv->run(h_.data(), L, qkv_.data());
    {
        VPROF(wattn_us);
        const float* qkv = qkv_.data();
        size_t off = 0;
        for (const view_grid& v : views) {
            vision::window_attention(qkv + off * 3 * D, qkv + off * 3 * D + D,
                                     qkv + off * 3 * D + 2 * D, 3 * D, v.h, v.w, H,
                                     (int)desc_.DH, desc_.attn_scale, attn_o_.data() + off * D);
            off += (size_t)v.h * v.w;
        }
    }
    vm_ops_.proj->run(attn_o_.data(), L, proj_.data());
    _emit("vm.attn_proj", proj_.data(), L, D);
    { VPROF(resid_us); vision::add_inplace(x_.data(), proj_.data(), (size_t)L * D); }
    _emit("vm.attn_residual", x_.data(), L, D);

    // ---- 2x2 merge, MLP, + mean-of-4 (residual #2) -------------------------
    {
        VPROF(gather_us);
        size_t off = 0, moff = 0;
        for (const view_grid& v : views) {
            vision::regroup_2x2(x_.data() + off * D, v.h, v.w, D,
                                merged_.data() + moff * VI, mean_.data() + moff * D);
            off  += (size_t)v.h * v.w;
            moff += (size_t)(v.h / 2) * (v.w / 2);
        }
    }
    _emit("vm.merged", merged_.data(), Lq, VI);
    _emit("vm.mean", mean_.data(), Lq, D);
    { VPROF(norm_us);
      vision::layer_norm(merged_.data(), Lq, VI, vw.pre_norm_w.data(), vw.pre_norm_b.data(),
                         desc_.ln_eps, merged_n_.data()); }
    _emit("vm.pre_norm", merged_n_.data(), Lq, VI);

    vm_ops_.fc1->run(merged_n_.data(), Lq, ff_.data());     // gelu_tanh fused
    _emit("vm.fc1", ff_.data(), Lq, VF);
    vm_ops_.fc2->run(ff_.data(), Lq, x_.data());
    { VPROF(resid_us); vision::add_inplace(x_.data(), mean_.data(), (size_t)Lq * D); }
    _emit("vm.out", x_.data(), Lq, D);

    for (view_grid& v : views) { v.h /= 2; v.w /= 2; }
}

void vision_tower::encode(const float* pixel_values, const std::vector<view_grid>& views_in,
                          float* out) {
    const int D  = (int)desc_.D;
    const int MI = (int)desc_.merger_in();
    const int gm = desc_.grid_multiple();
    if (views_in.empty()) throw std::runtime_error("vision_tower::encode got no views");

    int L = 0, max_view = 0;
    for (const view_grid& v : views_in) {
        if (v.h <= 0 || v.w <= 0 || v.h % gm || v.w % gm)
            throw std::runtime_error("vision_tower::encode: view " + std::to_string(v.h) + "x" +
                                     std::to_string(v.w) + " is not a positive multiple of " +
                                     std::to_string(gm) + " patches");
        L += v.h * v.w;
        max_view = std::max(max_view, v.h * v.w);
    }
    _ensure_scratch(L);

    // Sized for the largest view seen so far: reconfiguring per call would
    // rebuild device state per image.
    if (max_view > attn_configured_L_) {
        attn_->configure(max_view, (int)desc_.num_heads, (int)desc_.DH, desc_.attn_scale);
        attn_configured_L_ = max_view;
    }

    _prepare(views_in, L);

    // ---- patch embedding + position table -----------------------------------
    patch_mm_->run(pixel_values, L, x_.data());
    _emit("patch_embed", x_.data(), L, D);
    {
        VPROF(pos_us);
        size_t off = 0;
        for (const view_grid& v : views_in) {
            vision::pos_embed_lookup(w_->pos_embed(), (int)desc_.num_patches_per_side, D,
                                     v.h, v.w, h_.data() + off * D);
            off += (size_t)v.h * v.w;
        }
    }
    _emit("pos_embed", h_.data(), L, D);
    { VPROF(resid_us); vision::add_inplace(x_.data(), h_.data(), (size_t)L * D); }
    _emit("embed", x_.data(), L, D);

    // ---- encoder, with the vit_merger after block insert_layer_id ------------
    std::vector<view_grid> views = views_in;
    for (uint32_t i = 0; i < desc_.depth; i++) {
        _block((int)i, views, L);
        if ((int32_t)i == desc_.insert_layer_id) {
            _vit_merger(views, L);
            L /= 4;
        }
    }

    // ---- post_layernorm + projector -------------------------------------------
    { VPROF(norm_us);
      vision::layer_norm(x_.data(), L, D, w_->post_ln_w().data(), w_->post_ln_b().data(),
                         desc_.ln_eps, h_.data()); }
    _emit("post_ln", h_.data(), L, D);

    const int rows = L / 4;
    {
        VPROF(gather_us);
        size_t off = 0, moff = 0;
        for (const view_grid& v : views) {
            vision::regroup_2x2(h_.data() + off * D, v.h, v.w, D, merged_.data() + moff * MI,
                                nullptr);
            off  += (size_t)v.h * v.w;
            moff += (size_t)(v.h / 2) * (v.w / 2);
        }
    }
    _emit("merger.merged", merged_.data(), rows, MI);
    // MiniCPMV4_7DownsampleMLP builds this LayerNorm with eps=1e-6 explicitly.
    { VPROF(norm_us);
      vision::layer_norm(merged_.data(), rows, MI, w_->pre_norm_w().data(),
                         w_->pre_norm_b().data(), 1e-6f, merged_n_.data()); }
    _emit("merger.pre_norm", merged_n_.data(), rows, MI);
    merger_fc1_->run(merged_n_.data(), rows, merged_.data());   // erf gelu
    _emit("merger_fc1", merged_.data(), rows, MI);
    merger_fc2_->run(merged_.data(), rows, out);
    _emit("image_embeds", out, rows, (int)desc_.out_hidden_size);
}

}  // namespace mco
