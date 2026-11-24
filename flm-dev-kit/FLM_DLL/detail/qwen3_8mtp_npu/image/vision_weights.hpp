// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/vision_weights.hpp
/// \brief Views into vision_weight.q4nx.
///
/// Despite the extension the file is ordinary safetensors and every tensor in
/// it is BF16 -- the vision tower is not quantised. So this reuses the engine's
/// own cpu_ref::safetensors (mmap + a header scan) and never touches Q4NX,
/// which would want 3-D block tensors that are not there.
///
/// Matmul weights are kept as raw bf16 pointers INTO THE MAPPING: they are
/// 959 MB and the op seam takes bf16 anyway, so nothing is copied. The small
/// per-block vectors (layernorm gain/bias) are materialised as fp32 because
/// they are read element-by-element in the host glue.
#ifndef __QWEN3_8MTP_VISION_WEIGHTS_HPP__
#define __QWEN3_8MTP_VISION_WEIGHTS_HPP__
#pragma once

#include "common/cpu_ref_common.hpp"
#include "image/vision_model_def.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {
namespace vision {

/// One transformer block's tensors.
struct block_weights {
    std::vector<float> norm1_w, norm1_b;
    std::vector<float> norm2_w, norm2_b;

    const uint16_t* qkv_w  = nullptr;   ///< [3D][D]
    const uint16_t* qkv_b  = nullptr;   ///< [3D]
    const uint16_t* proj_w = nullptr;   ///< [D][D]
    const uint16_t* proj_b = nullptr;   ///< [D]
    const uint16_t* fc1_w  = nullptr;   ///< [D_ff][D]
    const uint16_t* fc1_b  = nullptr;   ///< [D_ff]
    const uint16_t* fc2_w  = nullptr;   ///< [D][D_ff]
    const uint16_t* fc2_b  = nullptr;   ///< [D]
};

/// \brief The mmap'd vision checkpoint plus resolved views.
class weights {
public:
    /// \param desc filled by qwen3_8mtp_vision_desc::build(); the MLP width is
    ///        corrected from the tensors here, so `desc` is taken by reference.
    weights(qwen3_8mtp_vision_desc& desc) : desc_(desc) {
        file_ = std::make_unique<cpu_ref::safetensors>(desc.weight_path);

        // The shipped fc1 is wider than vision_config says (4608 vs 4304); the
        // tensors are the authority on the stride, so adopt their width before
        // anything sizes a buffer off it.
        const auto& fc1 = _info("model.visual.blocks.0.mlp.linear_fc1.weight");
        if (fc1.shape.size() != 2)
            throw std::runtime_error("vision fc1.weight is not 2-D");
        desc.adopt_tensor_widths((uint32_t)fc1.shape[0]);

        _check_shapes();

        patch_w_   = _bf16("model.visual.patch_embed.proj.weight");
        patch_b_   = _bf16("model.visual.patch_embed.proj.bias");
        pos_embed_ = _bf16("model.visual.pos_embed.weight");

        merger_norm_w_ = file_->f32("model.visual.merger.norm.weight");
        merger_norm_b_ = file_->f32("model.visual.merger.norm.bias");
        merger_fc1_w_  = _bf16("model.visual.merger.linear_fc1.weight");
        merger_fc1_b_  = _bf16("model.visual.merger.linear_fc1.bias");
        merger_fc2_w_  = _bf16("model.visual.merger.linear_fc2.weight");
        merger_fc2_b_  = _bf16("model.visual.merger.linear_fc2.bias");

        blocks_.resize(desc.depth);
        for (uint32_t i = 0; i < desc.depth; i++) _load_block((int)i);
    }

    const block_weights& block(int i) const { return blocks_[(size_t)i]; }

    /// [D][patch_dim] bf16. The conv3d is a plain matmul once the input is
    /// flattened into patches, which is how pixel_values already arrives, so
    /// this shape is [out_channels][in_channels*Tp*Ps*Ps] with no reordering.
    const uint16_t* patch_w() const { return patch_w_; }
    const uint16_t* patch_b() const { return patch_b_; }
    /// [num_position_embeddings][D] bf16, the square learned grid.
    const uint16_t* pos_embed() const { return pos_embed_; }

    const std::vector<float>& merger_norm_w() const { return merger_norm_w_; }
    const std::vector<float>& merger_norm_b() const { return merger_norm_b_; }
    const uint16_t* merger_fc1_w() const { return merger_fc1_w_; }
    const uint16_t* merger_fc1_b() const { return merger_fc1_b_; }
    const uint16_t* merger_fc2_w() const { return merger_fc2_w_; }
    const uint16_t* merger_fc2_b() const { return merger_fc2_b_; }

    const cpu_ref::safetensors& file() const { return *file_; }

private:
    static std::string _block_key(int i, const char* suffix) {
        return "model.visual.blocks." + std::to_string(i) + "." + suffix;
    }

    const cpu_ref::tensor_info& _info(const std::string& n) const {
        if (!file_->has(n))
            throw std::runtime_error("vision checkpoint is missing " + n);
        return file_->info(n);
    }

    /// Raw bf16 view. Fails loudly on a non-BF16 tensor rather than
    /// reinterpreting fp32 bytes as half the elements.
    const uint16_t* _bf16(const std::string& n) const {
        const cpu_ref::tensor_info& t = _info(n);
        if (t.dtype != "BF16")
            throw std::runtime_error(
                n + " is " + t.dtype + ", not BF16 -- this folder reads the "
                "vision tower as raw bf16 and would misread anything else");
        return reinterpret_cast<const uint16_t*>(file_->raw(n));
    }

    void _load_block(int i) {
        block_weights& b = blocks_[(size_t)i];
        b.norm1_w = file_->f32(_block_key(i, "norm1.weight"));
        b.norm1_b = file_->f32(_block_key(i, "norm1.bias"));
        b.norm2_w = file_->f32(_block_key(i, "norm2.weight"));
        b.norm2_b = file_->f32(_block_key(i, "norm2.bias"));
        b.qkv_w  = _bf16(_block_key(i, "attn.qkv.weight"));
        b.qkv_b  = _bf16(_block_key(i, "attn.qkv.bias"));
        b.proj_w = _bf16(_block_key(i, "attn.proj.weight"));
        b.proj_b = _bf16(_block_key(i, "attn.proj.bias"));
        b.fc1_w  = _bf16(_block_key(i, "mlp.linear_fc1.weight"));
        b.fc1_b  = _bf16(_block_key(i, "mlp.linear_fc1.bias"));
        b.fc2_w  = _bf16(_block_key(i, "mlp.linear_fc2.weight"));
        b.fc2_b  = _bf16(_block_key(i, "mlp.linear_fc2.bias"));
    }

    /// Every stride in the tower comes from `desc`. If a shipped tensor
    /// disagrees the result is not an error, it is silently permuted output, so
    /// check each one against the shape the config implies.
    void _check_shapes() const {
        const auto& d = desc_;
        auto want = [&](const char* n, std::vector<int64_t> s) {
            const cpu_ref::tensor_info& t = _info(n);
            if (t.shape.size() != s.size()) _fail(n, t, s);
            for (size_t i = 0; i < s.size(); i++)
                if (t.shape[i] != s[i]) _fail(n, t, s);
        };
        const int64_t D  = d.D;
        const int64_t F  = d.D_ff;
        const int64_t PD = d.patch_dim();

        want("model.visual.patch_embed.proj.weight",
             {D, (int64_t)d.in_channels, (int64_t)d.temporal_patch_size,
              (int64_t)d.patch_size, (int64_t)d.patch_size});
        if ((int64_t)_info("model.visual.patch_embed.proj.weight").numel() != D * PD)
            throw std::runtime_error("vision patch_embed weight volume != D*patch_dim");
        want("model.visual.patch_embed.proj.bias", {D});
        want("model.visual.pos_embed.weight",
             {(int64_t)d.num_position_embeddings, D});

        for (uint32_t i = 0; i < d.depth; i++) {
            want(_block_key((int)i, "norm1.weight").c_str(), {D});
            want(_block_key((int)i, "attn.qkv.weight").c_str(), {3 * D, D});
            want(_block_key((int)i, "attn.proj.weight").c_str(), {D, D});
            want(_block_key((int)i, "mlp.linear_fc1.weight").c_str(), {F, D});
            want(_block_key((int)i, "mlp.linear_fc2.weight").c_str(), {D, F});
        }

        want("model.visual.merger.norm.weight", {D});
        want("model.visual.merger.linear_fc1.weight",
             {(int64_t)d.merger_in(), (int64_t)d.merger_in()});
        want("model.visual.merger.linear_fc2.weight",
             {(int64_t)d.out_hidden_size, (int64_t)d.merger_in()});
    }

    [[noreturn]] static void _fail(const char* n, const cpu_ref::tensor_info& t,
                                   const std::vector<int64_t>& s) {
        auto j = [](const std::vector<int64_t>& v) {
            std::string o = "[";
            for (size_t i = 0; i < v.size(); i++)
                o += (i ? ", " : "") + std::to_string(v[i]);
            return o + "]";
        };
        throw std::runtime_error(std::string(n) + " has shape " + j(t.shape) +
                                 " but the config implies " + j(s));
    }

    qwen3_8mtp_vision_desc&                  desc_;
    std::unique_ptr<cpu_ref::safetensors>    file_;
    std::vector<block_weights>               blocks_;

    const uint16_t* patch_w_   = nullptr;
    const uint16_t* patch_b_   = nullptr;
    const uint16_t* pos_embed_ = nullptr;

    std::vector<float> merger_norm_w_, merger_norm_b_;
    const uint16_t* merger_fc1_w_ = nullptr;
    const uint16_t* merger_fc1_b_ = nullptr;
    const uint16_t* merger_fc2_w_ = nullptr;
    const uint16_t* merger_fc2_b_ = nullptr;
};

}  // namespace vision
}  // namespace qwen38

#endif  // __QWEN3_8MTP_VISION_WEIGHTS_HPP__
