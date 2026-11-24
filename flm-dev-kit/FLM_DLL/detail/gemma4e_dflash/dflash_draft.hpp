// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Host (AVX-512) DFlash drafter for gemma4e_dflash (an NPU drafter can replace it later
// and uses this class as its bit-level reference).  See README.md (the DFlash cycle).
#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "typedef.hpp"   // bf16

class dflash_draft_host {
public:
    struct config {
        int layers = 5, D = 2560, ffn = 5120, n_q = 32, n_kv = 8, head_dim = 128;
        int block = 8, mask_id = 4, window = 1024, n_taps = 5;
        int taps[5] = {5, 11, 17, 23, 35};   // HF convention: output of these target layers
        float eps = 1e-6f, rope_theta = 1e6f, embed_scale = 50.59644f;
    };

    dflash_draft_host();
    ~dflash_draft_host();

    /// \brief read `dir`/config.json and the file named by its "draft_weight" (Q4NX, dequantized to bf16/f32 at load)
    bool load(const std::string& dir);
    bool loaded() const;
    const config& cfg() const;

    /// \brief drop the whole draft context (cold draft)
    void reset();
    /// \brief number of target positions the context has absorbed (the next inject() must start here)
    int ctx_end() const;
    /// \brief forget every position >= end (rows injected speculatively for rejected drafts); no-op if end >= ctx_end()
    void truncate(int end);
    /// \brief absorb target taps for n consecutive positions pos0..pos0+n-1 (inject() starting at pos0 != ctx_end() restarts the
    ///        context at pos0: a cold draft after a reset may start anywhere):
    ///        c = rmsnorm(fc . concat(taps)) * enc.output_norm, then per layer K = rope(k_norm(Wk c)), V = Wv c appended to a ring
    /// \param taps [n][n_taps][D] bf16, tap t = output of target layer cfg().taps[t] after layer_scalar, before any final norm
    void inject(const bf16* taps, int n, int pos0);
    /// \brief run the 5-layer draft on the block [last_token, mask x7] at positions pos0..pos0+7 (pos0 = ctx_end())
    /// \param block_emb [8][D] bf16: target embedding rows of the block ids, already multiplied by cfg().embed_scale
    ///        (row 0 = id_last, rows 1..7 = mask_id)
    /// \param hidden out [7][D] bf16: hidden rows of block positions 1..7 BEFORE the final output_norm (the NPU lm_head applies the norm)
    void draft(int pos0, const bf16* block_emb, bf16* hidden);
    /// \brief the draft's final RMSNorm weight [D] (plain `x * w`, Qwen convention), for the lm_head dispatch
    const bf16* output_norm() const;

    /// \brief run only the host reference path in fp32 end to end (for tests; slower, no bf16 rounding between stages)
    void set_fp32_reference(bool on);

private:
    struct Impl;
    Impl* impl_;
};
