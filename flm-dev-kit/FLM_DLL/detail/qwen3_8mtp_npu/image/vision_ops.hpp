// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/vision_ops.hpp
/// \brief The two operators the vision tower spends all its time in, behind an
///        abstract interface: matmul and attention.
///
/// This is the seam the whole folder exists to protect. Everything else in the
/// tower -- layernorm, the bilinear position-embedding resample, the 2-D rotary
/// application, the residual adds, the merger's regrouping -- is O(L*D) glue
/// that will stay on the host. The matmuls and the attention are what move to
/// the device, and the bitstreams for them ALREADY EXIST:
///
///   FLM_Xclbin/Qwen3_8/vision_mm     two designs, A (m=64,k=384,n=48) and
///                                    B (m=32,k=384,n=128); y = act(x W^T + b)
///                                    with act in {none, silu, gelu}
///   FLM_Xclbin/Qwen3_8/vision_attn   16 heads, head_dim 72, full bidirectional
///                                    mask, L padded to whole 512-row chunks
///
/// The interfaces below are shaped to those kernels on purpose:
///
///  - A matmul op is WEIGHT-RESIDENT. bind() is called once at load and run()
///    only streams activations. A per-call weight pointer would have forced the
///    NPU backend to re-upload 959 MB of tower weights every image, which is
///    the whole cost of the port.
///  - Activations cross the seam as fp32 host buffers, exactly like this
///    engine's existing device ops (prefill/npu/prefill_attn_op.hpp takes
///    `const float* q_rows` and writes `float* o_rows`, converting to bf16 at
///    the BO boundary). Keeping fp32 here means the CPU and NPU backends are
///    drop-in for each other with no dtype churn at the call sites.
///  - The activation is part of bind(), not run(), because on the device it is
///    an RTP the sequence bakes in. See rtp-act-is-per-round-not-per-dispatch:
///    a mid-sequence RTP rewrite is a protocol change, not a free parameter.
///  - Attention runs ONE IMAGE AT A TIME. The true mask is block-diagonal over
///    cu_seqlens -- full within an image, zero across images -- and a single
///    call over the concatenated sequence would silently let two images attend
///    to each other. With one image that bug is invisible, which is exactly how
///    it survives to production.
#ifndef __QWEN3_8MTP_VISION_OPS_HPP__
#define __QWEN3_8MTP_VISION_OPS_HPP__
#pragma once

#include "common/cpu_ref_common.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {
namespace vision {

/// Fused epilogue. The first three values are vision_mm's rtp_act; gelu_erf is
/// this folder's own and has no kernel behind it yet -- see below.
enum class activation : int32_t {
    none     = 0,
    silu     = 1,
    gelu     = 2,   ///< gelu_pytorch_tanh, what vision_config.hidden_act names
    gelu_erf = 3,   ///< exact erf gelu; THE MERGER'S, and only the merger's
};

/// gelu_pytorch_tanh: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))).
///
/// The constant is sqrt(2/PI) = 0.7978845608. FLM_DLL/include/vision/
/// vision_prefill_helper.hpp's inline gelu_tanh() writes sqrt(2.0f/3)
/// = 0.8164966 instead, which is a different function; do not reuse it.
inline float gelu_tanh(float x) {
    constexpr float kSqrt2OverPi = 0.79788456080286535588f;
    constexpr float kBeta        = 0.044715f;
    return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * (x + kBeta * x * x * x)));
}

/// Exact gelu: 0.5x(1 + erf(x/sqrt(2))).
///
/// THE TOWER USES BOTH, and they are not interchangeable. Qwen3_5VisionMLP
/// takes its activation from `vision_config.hidden_act`, which this checkpoint
/// sets to "gelu_pytorch_tanh"; Qwen3_5VisionPatchMerger hardcodes `nn.GELU()`,
/// which is the erf form. So the 27 block MLPs are tanh and the merger -- the
/// last thing before the language model sees the image -- is erf.
///
/// The two agree to ~1e-3 absolute, so getting it wrong costs a few thousandths
/// of cosine on the final rows and nothing that reads as a bug. It is written
/// down here because that is precisely why it would never be found later.
inline float gelu_erf(float x) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
}

inline float apply_activation(float x, activation act) {
    switch (act) {
        case activation::silu:     return cpu_ref::siluf_(x);
        case activation::gelu:     return gelu_tanh(x);
        case activation::gelu_erf: return gelu_erf(x);
        default:                   return x;
    }
}

inline const char* act_name(activation a) {
    switch (a) {
        case activation::silu:     return "silu";
        case activation::gelu:     return "gelu_tanh";
        case activation::gelu_erf: return "gelu_erf";
        default:                   return "none";
    }
}

// ---------------------------------------------------------------------------
// matmul
// ---------------------------------------------------------------------------

/// \brief y[M][N] = act(x[M][K] @ w[N][K]^T + bias[N]).
///
/// Weight layout is [N][K] row-major -- PyTorch's nn.Linear order, which is how
/// the tensors ship, so no transpose happens anywhere in this folder.
class mm_op {
public:
    virtual ~mm_op() = default;

    /// Bind a weight for the lifetime of the op. `w` is [N][K] raw bf16 and
    /// `bias` is [N] raw bf16 or null; both point into the mmap'd checkpoint
    /// and must outlive the op. An NPU backend uploads to a BO here.
    /// \note vision_mm's rtp_act has no gelu_erf. A device backend asked for it
    ///       must either add one or bind `none` and leave the epilogue on the
    ///       host -- it must NOT substitute gelu_tanh, which is what makes this
    ///       a separate enumerator instead of a comment.
    virtual void bind(const uint16_t* w, const uint16_t* bias,
                      int K, int N, activation act) = 0;

    /// Run M rows. `x` is [M][K] fp32, `y` is [M][N] fp32; they may not alias.
    virtual void run(const float* x, int M, float* y) = 0;

    virtual int K() const = 0;
    virtual int N() const = 0;
    virtual const char* backend() const = 0;
};

// ---------------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------------

/// \brief Full bidirectional multi-head attention over one image's patches.
///
/// q, k, v and o are all [L][heads*dh] fp32, head-major within a row (head h at
/// offset h*dh) -- the layout the fused qkv projection already produces once
/// its [Q|K|V] thirds are split.
class attn_op {
public:
    virtual ~attn_op() = default;

    /// Size internal scratch for up to this many rows and this geometry. An
    /// NPU backend allocates BOs and builds its sequence here.
    virtual void configure(int max_L, int heads, int dh, float scale) = 0;

    /// \param head_perm null for the reference; non-null scores head h of q
    ///        against k/v head head_perm[h]. A rotation gives the imposter that
    ///        a correct run must score badly against -- see
    ///        cross-head-control-catches-gqa-mapping. The CPU backend honours
    ///        it; a device backend may refuse it.
    virtual void run(const float* q, const float* k, const float* v,
                     int L, float* o, const int* head_perm = nullptr) = 0;

    virtual const char* backend() const = 0;
};

// ---------------------------------------------------------------------------
// backend selection
// ---------------------------------------------------------------------------

/// One place decides which backend the tower gets, exactly like the engine's
/// three existing modules. image/npu/vision_op_factory.hpp adds the device
/// implementation without touching image_cpu.cpp.
class op_factory {
public:
    virtual ~op_factory() = default;
    virtual std::unique_ptr<mm_op>   make_mm(const char* tag)   = 0;
    virtual std::unique_ptr<attn_op> make_attn(const char* tag) = 0;
    virtual const char* name() const = 0;
};

// ---------------------------------------------------------------------------
// CPU backends
// ---------------------------------------------------------------------------

/// Whether the CPU ops round their activations through bf16.
///
/// On by default, and this is not a performance choice. The device consumes
/// bf16 activations, so rounding here makes the bf16 error common-mode: when
/// Phase 2 scores the NPU matmul against this reference, what is left is the
/// device's own arithmetic and its addressing, and the pass bar can sit at
/// cosine > 0.999 instead of somewhere vague. See
/// score-dequant-mm-against-its-own-quantized-bytes.
///
/// FLM_Q38_VISION_FP32=1 turns it off, which is the control that separates a
/// precision floor from a bug: if a stage only passes in fp32, the problem is
/// range, not addressing.
bool cpu_ops_emulate_bf16();

/// fp32 accumulation over bf16 operands -- what both the AIE mmul and the
/// reference in FLM_Xclbin/Qwen3_8/vision_mm/host/cpu_ref_vision_mm.hpp do.
class cpu_mm_op final : public mm_op {
public:
    void bind(const uint16_t* w, const uint16_t* bias,
              int K, int N, activation act) override {
        if (!w) throw std::runtime_error("cpu_mm_op::bind got a null weight");
        w_ = w; bias_ = bias; K_ = K; N_ = N; act_ = act;
    }

    void run(const float* x, int M, float* y) override;

    int K() const override { return K_; }
    int N() const override { return N_; }
    const char* backend() const override { return "cpu"; }

private:
    const uint16_t* w_    = nullptr;
    const uint16_t* bias_ = nullptr;
    int K_ = 0, N_ = 0;
    activation act_ = activation::none;
};

/// Streaming softmax is not worth it here: L is at most a few thousand and the
/// scores for one (head, query) fit in L floats.
class cpu_attn_op final : public attn_op {
public:
    void configure(int max_L, int heads, int dh, float scale) override {
        heads_ = heads; dh_ = dh; scale_ = scale;
        scores_.resize((size_t)max_L);
        max_L_ = max_L;
    }

    void run(const float* q, const float* k, const float* v,
             int L, float* o, const int* head_perm = nullptr) override;

    const char* backend() const override { return "cpu"; }

private:
    int   heads_ = 0, dh_ = 0, max_L_ = 0;
    float scale_ = 0.0f;
    std::vector<float> scores_;
};

class cpu_op_factory final : public op_factory {
public:
    std::unique_ptr<mm_op> make_mm(const char*) override {
        return std::make_unique<cpu_mm_op>();
    }
    std::unique_ptr<attn_op> make_attn(const char*) override {
        return std::make_unique<cpu_attn_op>();
    }
    const char* name() const override { return "cpu"; }
};

/// \brief Effective number of keys carrying the softmax mass, exp(entropy).
///
/// A random q/k pair concentrates almost all mass on a handful of keys, which
/// makes attention score well no matter how the keys are addressed. Print this
/// alongside any attention comparison: it should be a large fraction of L, not
/// ~1. See random-kv-cache-cannot-test-attention.
double effective_key_count(const float* q, const float* k,
                           int L, int heads, int dh, float scale,
                           int head = 0, int query = 0);

}  // namespace vision
}  // namespace qwen38

#endif  // __QWEN3_8MTP_VISION_OPS_HPP__
