// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "typedef.hpp"

// -----------------------------------------------------------------------------
// CPU reference for the Qwen3.8-27B prefill attention core.
//
// This is *only* the attention: scores -> causal mask -> softmax -> weighted V.
// The sigmoid output gate that follows in the full layer is deliberately not
// modelled here, because the NPU design does not implement it either -- it
// produces the raw attention_output.
// -----------------------------------------------------------------------------

namespace cpu_ref {

struct attn_geometry {
    int head_dim;
    int n_q_heads;
    int n_kv_heads;
    float scale;

    int dq()         const { return head_dim * n_q_heads; }
    int dkv()        const { return head_dim * n_kv_heads; }
    int gqa_group()  const { return n_q_heads / n_kv_heads; }
};

// q   : [L_q ][DQ ]  holding tokens [L_begin, L_begin + L_q)
// k,v : [L_kv][DKV]  holding tokens [0, L_kv)
// out : [L_q ][DQ ]  fp32
//
// window == 0 means plain causal attention; a positive value additionally
// drops keys older than `window` (matching the kernel's sliding-window path).
//
// If `mean_effective_keys` is non-null it receives the softmax perplexity
// averaged over every (token, head).  A value close to 1 means the attention
// collapsed onto a single key and the test data is degenerate.
void gqa_attention(const bf16 *q, const bf16 *k, const bf16 *v, float *out,
                   const attn_geometry &g, int L_begin, int L_q,
                   int window = 0, double *mean_effective_keys = nullptr);

} // namespace cpu_ref
