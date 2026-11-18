// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/lm_head.hpp
/// \brief Final norm + vocabulary projection.
///
/// Shared by the base model and the MTP head: the checkpoint has no
/// mtp.lm_head, so mtp.norm feeds this same 248320 x 5120 Q8_0 table
/// (the PyTorch reference projects the head's norm_out through model.lm_head).
#pragma once

#include "common/cpu_ref_common.hpp"
#include "common/cpu_simd.hpp"   // argmax_f32

#include <string>
#include <vector>

namespace cpu_ref {

class lm_head {
public:
    explicit lm_head(const safetensors& q4nx,
                     const std::string& name = "lm_head.weight")
        : w_(get_q4nx(q4nx, name)) {}

    int vocab() const { return w_.rows; }
    int dim()   const { return w_.cols; }

    /// Project L rows of [dim()] to logits [L][vocab()].
    /// The caller norms first -- this is the bare matmul, so the same object
    /// serves model.norm and mtp.norm.
    void forward(const float* x, int L, float* logits) const {
        q4nx_matmul(w_, x, L, logits);
    }

    /// rms_norm(x, w) then project. `keep` selects a suffix of the L rows:
    /// prefill wants 1 (last token), a verify pass wants k+1, because
    /// position i predicts the token at i+1 and acceptance cannot be read
    /// off the last row alone.
    void norm_and_forward(const float* x, int L, const float* norm_w,
                          int keep, float* logits,
                          float eps = q38::RMS_EPS) const {
        if (keep <= 0 || keep > L) throw std::runtime_error("lm_head: bad keep");
        const int D = w_.cols;
        const int off = L - keep;
        std::vector<float> xn((size_t)keep * D);
        for (int t = 0; t < keep; t++)
            rms_norm(x + (size_t)(off + t) * D, norm_w, D,
                     xn.data() + (size_t)t * D, eps);
        q4nx_matmul(w_, xn.data(), keep, logits);
    }

private:
    q4nx_tensor w_;
};

/// Index of the largest logit. Greedy sampling is what makes MTP acceptance a
/// plain integer compare, so the draft path must call this, never a sampler.
inline int argmax(const float* logits, int n) {
    return q38simd::argmax_f32(logits, n);
}

} // namespace cpu_ref
