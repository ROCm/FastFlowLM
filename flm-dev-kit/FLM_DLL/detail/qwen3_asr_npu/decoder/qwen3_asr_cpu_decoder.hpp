// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file qwen3_asr_cpu_decoder.hpp
/// \brief CPU (AVX-512) reference of the Qwen3-ASR / Confucius4-R2T2 text decoder, reading the Q8_0 q4nx package.
///
/// This is the "full CPU path first" of the engine (FLM_DLL/ADDING_A_NEW_MODEL.md): every semantic question
/// (norm form, rope, rounding points) is settled here against the BF16 PyTorch reference before any NPU work.
/// It is also the planned prefill of the shipped engine (short prompts: CPU beats the NPU prefill below ~300-400
/// tokens) and the fallback when the NPU is busy.
///
/// Weight format: model.q4nx from FLM_Q4NX_Converter/configs/qwen3_asr.json: Q8_0 projections in 32x256 blocks
/// (512 B of bf16 scales [8 groups][32 rows], then 8192 B of int8 stored column major [256 cols][32 rows]).
#pragma once
#include <cstdint>
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

/// The matrix products of a decoder layer, done elsewhere for a prompt (the NPU prefill). Inputs and outputs are row-major
/// floats holding bf16 values; the decoder keeps the norms, RoPE, attention, the KV cache and the logits.
class projections {
public:
    virtual ~projections() = default;
    /// prompts of fewer positions stay on the CPU
    virtual int min_rows() const = 0;
    /// q [n][heads * head_dim], k and v [n][kv_heads * head_dim] of the normed input x [n][hidden]
    virtual void qkv(int layer, const float* x, int n, float* q, float* k, float* v) = 0;
    /// out [n][hidden] = o_proj(att [n][heads * head_dim])
    virtual void o(int layer, const float* att, int n, float* out) = 0;
    /// out [n][hidden] = down(silu(gate x) * up x) of the normed input x [n][hidden]
    virtual void mlp(int layer, const float* x, int n, float* out) = 0;
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
