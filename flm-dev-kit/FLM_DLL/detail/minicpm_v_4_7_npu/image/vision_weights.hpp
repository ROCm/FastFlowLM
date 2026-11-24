// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/vision_weights.hpp
/// \brief Views into vision_weight.q4nx.
///
/// Ported from Qwen3.8's. The file is ordinary safetensors, every tensor is
/// BF16 (bit-exact to the mmproj GGUF, per S7a), and the matmul weights stay
/// raw pointers INTO THE MAPPING -- 1.1 GB the op seam takes as bf16 anyway.
/// Layernorm gains and biases are materialised as fp32 because the host glue
/// reads them element by element.
///
/// Tensor names follow Qwen3.8's where the role matches (S7a's name map). What
/// is MiniCPM's own: `patch_embed.proj.weight` is 4-D `(1152, 3, 14, 14)`, a
/// Conv2d with no temporal axis -- checked as such, not padded with a fake one
/// -- and `vit_merger.*` plus `merger.pre_norm` have no Qwen3.8 counterpart.
/// `merger.norm` is the tower's `post_layernorm` (GGUF `v.post_ln`).
#ifndef __MINICPM_V_VISION_WEIGHTS_HPP__
#define __MINICPM_V_VISION_WEIGHTS_HPP__
#pragma once

#include "image/vision_common.hpp"
#include "image/vision_model_def.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mcv {
namespace vision {

/// The four matmuls of one attention + MLP pair. Shared shape between the 27
/// encoder blocks and the vit_merger's attention, whose MLP is different.
struct linear {
    const uint16_t* w = nullptr;   ///< [N][K] bf16, nn.Linear order
    const uint16_t* b = nullptr;   ///< [N]
};

struct block_weights {
    std::vector<float> norm1_w, norm1_b;
    std::vector<float> norm2_w, norm2_b;
    linear qkv;    ///< [3D][D], rows [q | k | v], head-major within each third
    linear proj;   ///< [D][D]
    linear fc1;    ///< [D_ff][D]
    linear fc2;    ///< [D][D_ff]
};

struct vit_merger_weights {
    std::vector<float> norm1_w, norm1_b;       ///< [D], before the window attention
    linear qkv, proj;                           ///< as a block's
    std::vector<float> pre_norm_w, pre_norm_b; ///< [4D], after the 2x2 regroup
    linear fc1;                                 ///< [vm_ff][4D], gelu_tanh
    linear fc2;                                 ///< [D][vm_ff]
};

class weights {
public:
    /// \param desc filled by minicpm_v_vision_desc::build(); the two padded MLP
    ///        widths are corrected from the tensors here, so it is taken by
    ///        reference and must be read only after this constructor returns.
    explicit weights(minicpm_v_vision_desc& desc) : desc_(desc) {
        file_ = std::make_unique<ref::safetensors>(desc.weight_path);

        const auto& fc1 = _info("model.visual.blocks.0.mlp.linear_fc1.weight");
        uint32_t vm_rows = 0;
        if (desc.use_vit_merger())
            vm_rows = (uint32_t)_info("model.visual.vit_merger.mlp.linear_fc1.weight").shape.at(0);
        desc.adopt_tensor_widths((uint32_t)fc1.shape.at(0), vm_rows);

        _check_shapes();

        patch_w_   = _bf16("model.visual.patch_embed.proj.weight");
        patch_b_   = _bf16("model.visual.patch_embed.proj.bias");
        pos_embed_ = _bf16("model.visual.pos_embed.weight");

        blocks_.resize(desc.depth);
        for (uint32_t i = 0; i < desc.depth; i++) _load_block((int)i);

        if (desc.use_vit_merger()) {
            const std::string p = "model.visual.vit_merger.";
            vm_.norm1_w    = file_->f32(p + "norm1.weight");
            vm_.norm1_b    = file_->f32(p + "norm1.bias");
            vm_.qkv        = _lin(p + "attn.qkv");
            vm_.proj       = _lin(p + "attn.proj");
            vm_.pre_norm_w = file_->f32(p + "pre_norm.weight");
            vm_.pre_norm_b = file_->f32(p + "pre_norm.bias");
            vm_.fc1        = _lin(p + "mlp.linear_fc1");
            vm_.fc2        = _lin(p + "mlp.linear_fc2");
        }

        post_ln_w_  = file_->f32("model.visual.merger.norm.weight");
        post_ln_b_  = file_->f32("model.visual.merger.norm.bias");
        pre_norm_w_ = file_->f32("model.visual.merger.pre_norm.weight");
        pre_norm_b_ = file_->f32("model.visual.merger.pre_norm.bias");
        merger_fc1_ = _lin("model.visual.merger.linear_fc1");
        merger_fc2_ = _lin("model.visual.merger.linear_fc2");
    }

    const block_weights& block(int i) const { return blocks_[(size_t)i]; }
    const vit_merger_weights& vit_merger() const { return vm_; }

    /// [D][3*14*14] bf16: the Conv2d weight flattened as (c, kh, kw), which is
    /// the order `pixel_values` rows arrive in, so it is a plain [N][K] matmul.
    const uint16_t* patch_w() const { return patch_w_; }
    const uint16_t* patch_b() const { return patch_b_; }
    /// [70*70][D] bf16, the square learned table.
    const uint16_t* pos_embed() const { return pos_embed_; }

    const std::vector<float>& post_ln_w() const { return post_ln_w_; }
    const std::vector<float>& post_ln_b() const { return post_ln_b_; }
    const std::vector<float>& pre_norm_w() const { return pre_norm_w_; }
    const std::vector<float>& pre_norm_b() const { return pre_norm_b_; }
    const linear& merger_fc1() const { return merger_fc1_; }
    const linear& merger_fc2() const { return merger_fc2_; }

private:
    static std::string _bk(int i, const char* suffix) {
        return "model.visual.blocks." + std::to_string(i) + "." + suffix;
    }

    const ref::tensor_info& _info(const std::string& n) const {
        if (!file_->has(n))
            throw std::runtime_error("vision checkpoint is missing " + n);
        return file_->info(n);
    }

    const uint16_t* _bf16(const std::string& n) const {
        const ref::tensor_info& t = _info(n);
        if (t.dtype != "BF16")
            throw std::runtime_error(
                n + " is " + t.dtype + ", not BF16 -- this tower reads its "
                "weights as raw bf16 and would misread anything else");
        return reinterpret_cast<const uint16_t*>(file_->raw(n));
    }

    linear _lin(const std::string& stem) const {
        return linear{_bf16(stem + ".weight"), _bf16(stem + ".bias")};
    }

    void _load_block(int i) {
        block_weights& b = blocks_[(size_t)i];
        b.norm1_w = file_->f32(_bk(i, "norm1.weight"));
        b.norm1_b = file_->f32(_bk(i, "norm1.bias"));
        b.norm2_w = file_->f32(_bk(i, "norm2.weight"));
        b.norm2_b = file_->f32(_bk(i, "norm2.bias"));
        b.qkv  = _lin(_bk(i, "attn.qkv"));
        b.proj = _lin(_bk(i, "attn.proj"));
        b.fc1  = _lin(_bk(i, "mlp.linear_fc1"));
        b.fc2  = _lin(_bk(i, "mlp.linear_fc2"));
    }

    /// Every stride in the tower comes from `desc`; a shipped tensor that
    /// disagrees would be silently permuted output, not an error.
    void _check_shapes() const {
        const auto& d = desc_;
        auto want = [&](const std::string& n, std::vector<int64_t> s) {
            const ref::tensor_info& t = _info(n);
            bool ok = t.shape.size() == s.size();
            for (size_t i = 0; ok && i < s.size(); i++) ok = t.shape[i] == s[i];
            if (!ok) _fail(n, t, s);
        };
        const int64_t D  = d.D;
        const int64_t F  = d.D_ff;
        const int64_t P  = d.patch_size;

        // 4-D: Conv2d, no temporal axis (Qwen3.8's is 5-D).
        want("model.visual.patch_embed.proj.weight", {D, (int64_t)d.in_channels, P, P});
        want("model.visual.patch_embed.proj.bias", {D});
        want("model.visual.pos_embed.weight", {(int64_t)d.num_position_embeddings, D});

        for (uint32_t i = 0; i < d.depth; i++) {
            const int b = (int)i;
            want(_bk(b, "norm1.weight"), {D});
            want(_bk(b, "attn.qkv.weight"), {3 * D, D});
            want(_bk(b, "attn.proj.weight"), {D, D});
            want(_bk(b, "mlp.linear_fc1.weight"), {F, D});
            want(_bk(b, "mlp.linear_fc2.weight"), {D, F});
        }
        if (d.use_vit_merger()) {
            const std::string p = "model.visual.vit_merger.";
            want(p + "norm1.weight", {D});
            want(p + "attn.qkv.weight", {3 * D, D});
            want(p + "attn.proj.weight", {D, D});
            want(p + "pre_norm.weight", {(int64_t)d.vm_in()});
            want(p + "mlp.linear_fc1.weight", {(int64_t)d.vm_ff, (int64_t)d.vm_in()});
            want(p + "mlp.linear_fc2.weight", {D, (int64_t)d.vm_ff});
        }
        want("model.visual.merger.norm.weight", {D});
        want("model.visual.merger.pre_norm.weight", {(int64_t)d.merger_in()});
        want("model.visual.merger.linear_fc1.weight",
             {(int64_t)d.merger_in(), (int64_t)d.merger_in()});
        want("model.visual.merger.linear_fc2.weight",
             {(int64_t)d.out_hidden_size, (int64_t)d.merger_in()});
    }

    [[noreturn]] static void _fail(const std::string& n, const ref::tensor_info& t,
                                   const std::vector<int64_t>& s) {
        auto j = [](const std::vector<int64_t>& v) {
            std::string o = "[";
            for (size_t i = 0; i < v.size(); i++) o += (i ? ", " : "") + std::to_string(v[i]);
            return o + "]";
        };
        throw std::runtime_error(n + " has shape " + j(t.shape) +
                                 " but the config implies " + j(s));
    }

    minicpm_v_vision_desc&            desc_;
    std::unique_ptr<ref::safetensors> file_;
    std::vector<block_weights>        blocks_;
    vit_merger_weights                vm_;

    const uint16_t* patch_w_   = nullptr;
    const uint16_t* patch_b_   = nullptr;
    const uint16_t* pos_embed_ = nullptr;

    std::vector<float> post_ln_w_, post_ln_b_, pre_norm_w_, pre_norm_b_;
    linear merger_fc1_, merger_fc2_;
};

}  // namespace vision
}  // namespace mcv

#endif  // __MINICPM_V_VISION_WEIGHTS_HPP__
