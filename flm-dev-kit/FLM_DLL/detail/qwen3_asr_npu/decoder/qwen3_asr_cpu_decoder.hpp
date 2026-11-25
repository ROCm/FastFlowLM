// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file qwen3_asr_cpu_decoder.hpp
/// \brief CPU (AVX-512) reference of the Qwen3-ASR / Confucius4-R2T2 text decoder, reading the Q8_0 q4nx package.
///
/// This is the "full CPU path first" of the engine (FLM_DLL/ADDING_A_NEW_MODEL.md): every semantic question
/// (norm form, rope, rounding points) is settled here against the BF16 PyTorch reference before any NPU work.
/// It also runs the shipped engine's prompt prefill, handing the matrix products (and the attention of long prompts) to
/// the NPU through `projections` (prefill/qwen3_asr_prefill_npu.hpp), and all of it with QWEN3_ASR_CPU_PREFILL=1.
///
/// Weight format: model.q4nx from FLM_Q4NX_Converter/configs/qwen3_asr.json: Q8_0 projections in 32x256 blocks
/// (512 B of bf16 scales [8 groups][32 rows], then 8192 B of int8 stored column major [256 cols][32 rows]).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace qwen3_asr_cpu {

struct decoder_config {
    int layers = 28;
    int hidden = 2048;
    int heads = 16;
    int kv_heads = 8;
    int head_dim = 128;
    int inter = 6144;
    int vocab = 151936;
    float rms_eps = 1e-6f;
    float rope_theta = 1000000.0f;
};

/// The matrix products (and optionally the attention) of a decoder layer, done elsewhere for a prompt (the NPU prefill). Inputs
/// are row-major floats holding bf16 values; results go, a block of rows at a time, to a sink of the decoder, which keeps the
/// norms, RoPE, the KV cache, the residual stream and the logits.
class projections {
public:
    /// rows [r0, r0 + rows) of a product's result, bf16 [rows][outputs]; may be called while the next rows are computed, so
    /// it must not write input rows it has not been given yet
    using rows_sink = std::function<void(const uint16_t* y, int r0, int rows)>;
    virtual ~projections() = default;
    /// prompts of fewer positions stay on the CPU
    virtual int min_rows() const = 0;
    /// [k | v | q] [n][2 kv_heads * head_dim + heads * head_dim] of the normed input x [n][hidden]
    virtual void qkv(int layer, const float* x, int n, const rows_sink& out) = 0;
    /// o_proj(att [n][heads * head_dim]), [n][hidden]
    virtual void o(int layer, const float* att, int n, const rows_sink& out) = 0;
    /// down(silu(gate x) * up x) of the normed input x [n][hidden], [n][hidden]; out may write rows of x it was given
    virtual void mlp(int layer, const float* x, int n, const rows_sink& out) = 0;
    /// causal attention of the n new positions pos0 .. over the cache rows [0, pos0 + n): q [n][heads * head_dim] after norm and
    /// RoPE, k and v the layer's cache [pos0 + n][kv_heads * head_dim] (bf16), att [n][heads * head_dim]; false: the decoder does it
    virtual bool attention(int /*layer*/, const float* /*q*/, int /*pos0*/, int /*n*/, const uint16_t* /*k*/, const uint16_t* /*v*/, float* /*att*/) {
        return false;
    }
};

class decoder {
public:
    explicit decoder(const std::string& q4nx_path, decoder_config cfg = decoder_config());
    ~decoder();

    const decoder_config& cfg() const;

    /// forward() hands the projections of n >= p->min_rows() positions to p (nullptr: all on the CPU); p must outlive the
    /// decoder or be reset first
    void set_projections(projections* p);

    /// bf16-valued row of the (BF16) embedding table
    void embed(int token, float* out) const;
    /// the row itself (bf16, `hidden` values, in the mapped file)
    const uint16_t* embed_row(int token) const;

    /// drop the KV cache
    void reset();
    /// drop the KV cache and give its memory (and the attention scratch) back
    void release();
    /// the longest context (rows) since the last release()
    int peak_context_length() const;
    int context_length() const;

    /// K/V cache exchange with the NPU engine (rows of kv_heads*head_dim bf16 values, K after norm + rope).
    /// export_kv: rows [from, from+rows) of one layer; import_kv: replace the layer's cache by `rows` rows (call
    /// set_context_length(rows) once all layers are in).
    void export_kv(int layer, int from, int rows, uint16_t* k, uint16_t* v) const;
    void import_kv(int layer, int rows, const uint16_t* k, const uint16_t* v);
    void set_context_length(int rows);

    /// Run `n` new positions (rows of `embeds`, hidden floats each, bf16-valued) after the current context and
    /// return the logits of the LAST one, rounded to bf16 like the reference's bf16 lm_head output.
    void forward(const float* embeds, int n, std::vector<float>& last_logits);

private:
    struct impl;
    std::unique_ptr<impl> p_;
};

}  // namespace qwen3_asr_cpu
