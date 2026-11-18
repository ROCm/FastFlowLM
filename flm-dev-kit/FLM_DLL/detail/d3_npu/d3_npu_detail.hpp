// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include "models/d3/d3_npu.hpp"
#include "modules/embedding.hpp"
#include "d3_npu_sequence.hpp"
#include "d3_model_def.hpp"
#include "d3_prefill.hpp"
#include "embedding_q8_0.hpp"
#include "d3_vision.hpp"
#include "d3_positions.hpp"

struct d3_npu::Impl{
    LM_Config config;
    npu_xclbin_manager* npu;
    d3_desc desc;
    std::unique_ptr<d3_npu_sequence> sequence;
    std::unique_ptr<d3_prefill_context> pctx;
    int MAX_L;
    /// rows packed into one pass: 4 GEMM row tiles. Measured on d3-lite (Strix Halo): 16 questions of 330
    /// tokens 156 ms/question at 512 rows, 133 at 2048; 4 questions 150 -> 122. Bit-exact either way.
    int pack_rows = 4 * d3_npu_sequence::MM_ROWS;
    /// row buckets whose scratch and GEMM apps stay between calls: up to max(CACHE_ROWS, pack_rows).
    /// A longer prompt's bucket lives for its call only (d3_prefill_context::trim).
    static constexpr int CACHE_ROWS = 4096;

    // per-layer weights, laid out by d3_desc (the same contract as qwen3_5vl / minicpm_v)
    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> rms_weights;
    std::vector<buffer<bf16>> rope_rms_weights;   //!< attention: [cos/sin | q_norm | k_norm]; linear: the ssm block
    std::vector<buffer<bf16>> alpha_proj_cpu;
    std::vector<buffer<bf16>> beta_proj_cpu;
    buffer<bf16> final_norm;   //!< model.norm.weight, (1 + w) already folded by the converter

    /// @brief bf16 token embedding table when the q4nx carries model.embed_tokens.weight,
    ///        else the tied q8_0 lm_head.weight dequantized (the shipping Qwen3.5 layout).
    std::unique_ptr<Embedding> embedding;
    std::unique_ptr<embedding_q8_0> embedding_q8;

    buffer<bf16> residual;     //!< [R_padded, D] host; trimmed back to the cached rows after a longer call

    /// the vision tower: built (vision_mm / vision_attn registered, vision_weight.q4nx loaded) on the
    /// first call with images, so text-only use never pays for it
    std::unique_ptr<D3VisionEncoder> vision;
    const d3_image_payload* media = nullptr;   //!< the current call's images (nullptr: text)
    const std::vector<int>* state_len = nullptr;   //!< the current call's prefix_bidir state lengths (nullptr: none)
    std::vector<bf16> features;                //!< their merger output, [sum of image tokens][D]
    // encode_state_once: the current call's question images (M3) and video token id (M4)
    int video_id = -1;                         //!< <|video_pad|> of the call (-1: none)
    int qimage_id = 248056;                    //!< pad id of the images inside question blocks
    std::vector<bf16> qfeatures;               //!< merger rows of the call's distinct question images, [tokens][D]
    std::vector<size_t> qrow;                  //!< per distinct question image: its first row in qfeatures
    std::vector<int> qmap;                     //!< per entry of question_images: the distinct image it equals
    std::vector<std::vector<d3_image>> qgrids; //!< per question: the grids of its image_ids (empty: no images)
    std::unique_ptr<d3_state_cache> cache;     //!< the current encode_state_once call's per-layer state cache (host heap)
    bool loaded = false;

    Impl(LM_Config config, npu_xclbin_manager* npu_instance, int max_length);

    void load_weights(Q4NX& q4nx);
    std::vector<buffer<bf16>> encode_batch(const std::vector<std::vector<int>>& ids, const d3_image_payload* payload);

    /// @brief state-once (d3_npu::encode_state_once)
    std::vector<buffer<bf16>> encode_state_once(const d3_state_request& req);
    /// @brief one state-once pass: [the state if with_state] + questions [first_q, last_q); appends their hidden rows
    void _encode_state_pass(const d3_state_request& req, const std::vector<int>& state_pos, int next_pos,
                            bool with_state, size_t first_q, size_t last_q, std::vector<buffer<bf16>>& out);
    /// @brief `media` is set: check that every prompt holds the image runs, run the vision tower once (built on first use) into `features`
    void _prepare_media(const std::vector<std::vector<int>>& ids);
    /// @brief build the vision tower on first use
    void _ensure_vision();
    /// @brief run the vision tower over `m` (in calls of at most vision_max_patches() patches) into rows [tokens][D]
    void _vision_rows(const d3_image_payload& m, std::vector<bf16>& rows);
    /// @brief throws std::invalid_argument unless the grids are positive and even and the pixels match them
    void _check_images(const d3_image_payload& m, const char* who) const;
    /// @brief one packed pass over prompts [first, last) of `ids`; appends their hidden states
    void _encode_pack(const std::vector<std::vector<int>>& ids, size_t first, size_t last,
                      std::vector<buffer<bf16>>& out);
    void _embed(int id, bf16* dst);
    /// @brief throws std::invalid_argument unless every prompt holds the payload's image token runs
    void _check_image_prompts(const std::vector<std::vector<int>>& ids, const d3_image_payload& m) const;
    /// @brief drop what a call over the cached rows allocated (d3_prefill_context::trim, residual)
    void _trim();
};
