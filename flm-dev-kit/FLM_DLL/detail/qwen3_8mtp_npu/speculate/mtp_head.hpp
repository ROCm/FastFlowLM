// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file speculate/mtp_head.hpp
/// \brief The 1-layer MTP draft head: fusion + one gated full-attention
///        decoder layer + mtp.norm, over its own private KV cache.
///
/// mtp_use_dedicated_embeddings is false and there is no mtp.lm_head in the
/// checkpoint, so the head borrows the base model's embedding_table and
/// lm_head rather than owning a second copy of 2.37 GiB.
#pragma once

#include "common/cpu_ref_common.hpp"
#include "common/embedding.hpp"
#include "common/lm_head.hpp"
#include "common/rope.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpu_ref {

namespace mtp {
/// fc's input is cat(norm_embed, norm_hidden) -- two hidden vectors.
constexpr int FC_IN     = 2 * q38::MODEL_DIM;   // 10240
/// Draft depth k. verify_qwen3_8.py drafts 7 and verifies k+1 = 8.
constexpr int MTP_STEPS = 7;
} // namespace mtp

// ==========================================================================
/// Every tensor under model.mtp.*, plus the two the head shares with the base
/// model. Views into the mmap; holds no bulk data.
// ==========================================================================
struct mtp_weights {
    q4nx_tensor fc;         // [ 5120][10240]  cols 0:5120 embed, 5120:10240 hidden
    q4nx_tensor q;          // [12288][ 5120]  q rows then gate rows
    q4nx_tensor k;          // [ 1024][ 5120]
    q4nx_tensor v;          // [ 1024][ 5120]
    q4nx_tensor o;          // [ 5120][ 6144]
    q4nx_tensor mlp_gate;   // [17408][ 5120]
    q4nx_tensor mlp_up;     // [17408][ 5120]
    q4nx_tensor mlp_down;   // [ 5120][17408]

    std::vector<float> pre_fc_norm_embedding;  // [5120]
    std::vector<float> pre_fc_norm_hidden;     // [5120]
    std::vector<float> in_ln;                  // [5120]
    std::vector<float> post_ln;                // [5120]
    std::vector<float> out_norm;               // [5120]  mtp.norm
    std::vector<float> q_norm;                 // [256]
    std::vector<float> k_norm;                 // [256]

    void load(const safetensors& f);
};

// ==========================================================================
/// The draft head.
///
/// One step consumes a window of T tokens plus the hidden states that
/// produced them, and returns norm_out -- the rows the shared lm_head
/// projects. Its KV cache is private: drafting appends to this cache and
/// leaves all 64 base layers untouched, which is what makes the draft loop
/// snapshot-free (README section 5.2).
// ==========================================================================
struct mtp_taps {
    std::vector<float> inputs_embeds;   // [T][5120]
    std::vector<float> norm_embed;      // [T][5120]
    std::vector<float> norm_hidden;     // [T][5120]
    std::vector<float> fc_out;          // [T][5120]
    std::vector<float> layer_out;       // [T][5120]
    std::vector<float> norm_out;        // [T][5120]
};

class mtp_head {
public:
    /// \param q4nx  an open model.q4nx
    /// \param embed the base model's table -- mtp_use_dedicated_embeddings is
    ///              false, so the head reads model.embed_tokens like everyone
    ///              else. Borrowed, must outlive the head.
    /// \param max_len cache capacity; the head's cache runs one token behind
    ///              the base stack's but never ahead, so the same bound holds.
    mtp_head(const safetensors& q4nx, const embedding_table* embed, int max_len = 4096);

    void reset() { cache_len_ = 0; }
    int  cache_len() const { return cache_len_; }

    /// The head's cache, post-q/k-norm and post-RoPE for k, raw for v --
    /// q38npu::npu_mtp_layer::read_k / read_v layout.
    const std::vector<float>& cache_k() const { return cache_k_; }
    const std::vector<float>& cache_v() const { return cache_v_; }

    /// Install `n` rows as this head's cache.
    ///
    /// The counterpart of npu_mtp_layer::mirror_from_cpu, and its reason for
    /// existing is the same: a control that starts both implementations from
    /// one identical context, including a deliberately perturbed one that
    /// neither would produce on its own.
    ///
    /// Not part of the scored reference; step() never calls it.
    void seed_cache(const std::vector<float>& k, const std::vector<float>& v, int n) {
        if (n < 0 || n > max_len_)
            throw std::runtime_error("mtp_head::seed_cache out of range");
        const size_t want = (size_t)n * q38::DKV;
        if (k.size() < want || v.size() < want)
            throw std::runtime_error("mtp_head::seed_cache: short k or v");
        std::memcpy(cache_k_.data(), k.data(), want * sizeof(float));
        std::memcpy(cache_v_.data(), v.data(), want * sizeof(float));
        cache_len_ = n;
    }

    /// Crop the head's cache back to `len`.
    ///
    /// The head's cursor is NOT the base stack's: step 0 feeds
    /// [input_ids[1:], x_P] at positions 1..P, so its rows start at 1, not 0,
    /// and the two lengths differ by exactly one permanently. Aligning them
    /// makes the head condition on a token the base model never emitted --
    /// the symptom is acceptance decaying while the text stays correct
    /// (README section 5.3). Do not "fix" the off-by-one.
    void crop(int len) { cache_len_ = std::min(cache_len_, std::max(0, len)); }

    /// Grow the head's own cache, keeping what it holds. Same contract as
    /// attention_layer::set_max_len: the head trails the base stack, so it is
    /// resized alongside it rather than rebuilt.
    void set_max_length(int n) {
        if (n < cache_len_)
            throw std::runtime_error(
                "mtp_head::set_max_length(" + std::to_string(n) +
                ") is below the " + std::to_string(cache_len_) +
                " rows already cached");
        cache_k_.resize((size_t)n * q38::DKV, 0.0f);
        cache_v_.resize((size_t)n * q38::DKV, 0.0f);
        max_len_ = n;
    }

    /// One MTP step. `ids`/`pos_thw` are the window contents (T tokens,
    /// positions laid out [3][T] as t,h,w), `prev_hidden` is [T][5120] -- the
    /// hidden states whose next-token predictions these ids are.
    ///
    /// Appends T rows to the head's cache and fills `tp`. If cos_in/sin_in
    /// are null the m-rope table is generated here; passing dumped tables
    /// isolates a table mismatch from a math one.
    void step(const int32_t* ids, const int32_t* pos_thw, const float* prev_hidden,
              int T, mtp_taps& tp,
              const float* cos_in = nullptr, const float* sin_in = nullptr);

    /// mtp.norm's weight, for callers that project norm_out themselves
    /// through the shared lm_head.
    const std::vector<float>& out_norm() const { return w_.out_norm; }

    // ---- step(), in pieces ---------------------------------------------------
    /// The four public helpers `step()` is made of, exposed so that a backend
    /// can replace exactly one of them and keep the rest byte-identical.
    ///
    /// `q38npu`'s hybrid step 0 runs fuse -> project_qkv_rope -> (append to the
    /// NPU cache BO) -> attn.xclbin -> gate_out_mlp -> mtp.norm, i.e. every
    /// helper below except attn_core. That is the whole point of the split:
    /// speculate_cpu is the reference speculate_npu is scored against, so the
    /// two must share this code rather than mirror it.
    ///
    /// Calling them by hand does NOT maintain `cache_len_` -- only step() does.
    /// A caller driving them directly owns its own cursor.

    /// Embedding lookup, the two pre-fc RMSNorms, cat and `model.mtp.fc`.
    /// Fills `tp` through `fc_out`.
    void fuse(const int32_t* ids, const float* prev_hidden, int T, mtp_taps& tp);

    /// input_layernorm, q/k/v, per-head q/k RMSNorm, partial RoPE. No cache.
    /// `kk`/`vv` come back ready to append; `qh` is [L][DQ] head-major.
    void project_qkv_rope(const float* x, int L, const float* cos,
                          const float* sin, std::vector<float>& qg,
                          std::vector<float>& kk, std::vector<float>& vv,
                          std::vector<float>& qh);

    /// Causal GQA softmax over this head's cache. `base` is the cursor BEFORE
    /// the window was appended. The helper attn.xclbin replaces.
    void attn_core(const float* qh, int L, int base,
                   std::vector<float>& attn) const;

    /// Output gate, o_proj, residual, post_attention_layernorm, MLP, residual.
    /// `attn` is gated in place.
    void gate_out_mlp(const float* qg, float* attn, const float* x, int L,
                      float* y);

    // ---- step(), in finer pieces ---------------------------------------------
    /// The six helpers above, split once more along every matmul boundary, so
    /// that a backend can replace the *projections* and keep the elementwise
    /// work byte-identical. Same contract and same reason as the four above:
    /// the four are re-expressed in terms of these, so every existing caller
    /// gets the same code and the NPU path cannot drift from the reference.
    ///
    /// The seven matmuls they bracket are fc, q|z, k, v, o, up, gate and down;
    /// on the device those become five `dequant_mm` dispatches, because k|v|q|z
    /// is one fused matrix and up|gate another.

    /// `fuse` up to but not including `model.mtp.fc`: embedding lookup, the two
    /// pre-fc RMSNorms and the concatenation. Fills `tp` through `norm_hidden`
    /// and sizes `tp.fc_out` for the caller's matmul to write.
    /// \param fc_in [T][FC_IN] out, cat(norm_embed, norm_hidden)
    void fuse_pre(const int32_t* ids, const float* prev_hidden, int T,
                  mtp_taps& tp, std::vector<float>& fc_in);

    /// `input_layernorm` alone. \param xn [L][D] out
    void in_norm(const float* x, int L, std::vector<float>& xn);

    /// `project_qkv_rope` after its three matmuls: per-head q/k RMSNorm then
    /// partial RoPE. v is untouched by both, so it is not a parameter.
    /// \param qg [L][2*DQ] q then gate, read only
    /// \param kk [L][DK] normed and rotated IN PLACE
    /// \param qh [L][DQ] out, head-major
    void qkv_post_rope(const float* qg, float* kk, int L, const float* cos,
                       const float* sin, std::vector<float>& qh);

    /// The sigmoid output gate. \param attn [L][DQ] gated in place.
    void apply_gate(const float* qg, float* attn, int L) const;

    /// `post_attention_layernorm` alone. \param hn [L][D] out
    void post_norm(const float* h1, int L, std::vector<float>& hn);

    /// `silu(gate) * up`, elementwise. May alias `gate`.
    /// \param g [L][intermediate_size] out
    void swiglu(const float* gate, const float* up, int L, float* g) const;

    /// HF stores q_proj head-interleaved (row h*512+0..255 = head h query,
    /// +256..511 = head h gate); the converter de-interleaves the main
    /// model's layers into [all q | all gate]. Which the MTP head's copy got
    /// is a property of the checkpoint, so it stays selectable.
    bool q_interleaved = false;

private:
    /// fuse's counterpart: the cache append plus the three helpers above, in
    /// order. Private because it is the only one that moves `cache_len_`.
    void decoder_layer(const float* x, int L, const float* cos, const float* sin,
                       float* y);
    void mlp_residual(const float* h1, int L, float* y);

    /// Partial rotary: dims [0,64) only, rotate_half with halves 0..31/32..63.
    static void rope(float* v, const float* cos, const float* sin);

    /// Offsets of head h's query and gate rows under the active layout.
    void q_gate_off(int h, int& qo, int& go) const {
        if (q_interleaved) { qo = h * 2 * q38::HEAD_DIM; go = qo + q38::HEAD_DIM; }
        else               { qo = h * q38::HEAD_DIM;     go = q38::DQ + qo;       }
    }

    int                     max_len_, cache_len_ = 0;
    mtp_weights             w_;
    const embedding_table*  embed_ = nullptr;   // borrowed
    std::vector<float>      cache_k_, cache_v_; // [max_len][1024]
};

} // namespace cpu_ref
