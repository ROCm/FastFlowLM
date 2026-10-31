/// \file minicpm_v_4_7_npu.hpp
/// \brief minicpm_v_4_7_npu class -- MiniCPM-V-4.7-1B engine (text tower + SigLIP vision tower)
/// \author FastFlowLM Team
/// \date 2026-10-02
/// \version 0.9.28
///
/// \note CONTRACT FILE. This is the engine ABI: S3 implements it, S6 (the AutoModel
///       wrapper) and S7 (vision) compile against it. Do not edit without the brain's
///       agreement -- see /scratch/michyu/minicpm-1b/CONTRACTS.md.
///
/// MiniCPM-V-4.7-1B's text tower is config-identical to Qwen3.5-0.8B (hidden 1024,
/// 8/2 heads, head_dim 256, FFN 3584, 24 layers, 16x128 GatedDeltaNet heads, conv 4,
/// full_attention_interval 4, rope_theta 1e7, mrope_section [11,11,10]), so the text
/// path reuses the Qwen3.5-0.8B xclbins unchanged. The vision tower is SigLIP and
/// shares nothing with Qwen3.5-VL's ViT.
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "modules/embedding.hpp"
#include "modules/lm_head.hpp"
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "causal_lm.hpp"
#include "flm_state_api.h"
#if USEAVX2
#include <immintrin.h>  // For AVX intrinsics
#endif

/// \brief One decoded image, before and after preprocessing.
///
/// MiniCPM slices a large image into at most `MINICPM_V_MAX_SLICE_NUMS` tiles plus one
/// global view; each tile becomes one of these. `grid_h`/`grid_w` are the patch-grid
/// dimensions fed to the vision tower as `target_sizes`, i.e. `height_resized /
/// MINICPM_V_PATCH_SIZE` and `width_resized / MINICPM_V_PATCH_SIZE`.
typedef struct {
    int height;
    int width;
    int height_resized;  // assigned by image preprocessing
    int width_resized;
    int grid_h;          // patch rows  == height_resized / patch_size
    int grid_w;          // patch cols  == width_resized  / patch_size

    bytes _data;
} minicpm_v_image_t;

/// \brief The multimodal blob handed to prefill() as `void* payload`.
///
/// `_data__processed` is the concatenated, normalised pixel buffer for every image in
/// `images`, laid out tile-major. The engine turns it into soft tokens and splices them
/// over the rows whose id equals `image_token_id` (248056) -- see §7 of
/// ADDING_A_NEW_MODEL.md: those rows must bypass any embedding scale.
typedef struct {
    std::vector<minicpm_v_image_t> images;
    std::vector<bf16> _data__processed;
    unsigned int num_images;
} minicpm_v_image_payload_t;

class minicpm_v_4_7_npu : public causal_lm {
public:
    /// \brief  initialize the minicpm_v_4_7_npu
    /// \param config the configuration
    /// \param npu_instance the npu instance
    minicpm_v_4_7_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~minicpm_v_4_7_npu();

    /// \brief forward the minicpm_v_4_7_npu -- decode one token against the cached KV
    /// \param ids the token id
    /// \return the logits
    buffer<bf16> forward(int ids) override;

    /// \brief prefill the whole prompt
    /// \param ids the token ids
    /// \param payload nullptr for text-only, else a minicpm_v_image_payload_t*
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;

    /// \brief set the context length
    /// \param L the context length
    void set_context_length(int L) override;

    /// \brief load the weights
    /// \param q4nx the q4nx
    void load_weights(Q4NX& q4nx) override;

    /// \brief clear the conversation state
    void clear_context() override;

    /// \brief get the k cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the k cache
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;

    /// \brief get the v cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the v cache
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;

    /// \brief grow the caches without dropping the conversation
    /// \param MAX_L the max length
    void update_max_length(uint32_t MAX_L) override;

    /// \brief get the current context length
    /// \return the current context length
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;

    // ----------------------------------------------------------------------
    // Vision preprocessing parameters.
    //
    // These live in preprocessor_config.json and config.json's vision_config, NOT in
    // the top-level config.json, so they are read through a
    // preprocessor -> config -> literal fallback chain. A checkpoint packaged before
    // preprocessor_config.json existed still preprocesses correctly.
    // ----------------------------------------------------------------------
    unsigned int MINICPM_V_PATCH_SIZE;          ///< vision_config.patch_size            (14)
    unsigned int MINICPM_V_IMAGE_SIZE;          ///< vision_config.image_size            (980)
    unsigned int MINICPM_V_MAX_SLICE_NUMS;      ///< preprocessor.max_slice_nums         (9)
    unsigned int MINICPM_V_INSERT_LAYER_ID;     ///< config.insert_layer_id              (6)
    unsigned int MINICPM_V_SPATIAL_MERGE_SIZE;  ///< 2x2 per merge stage                 (2)
    unsigned int MINICPM_V_TEMPORAL_PATCH_SIZE; ///< always 1 -- MiniCPM has no temporal
                                                ///< patching; video frames are independent
                                                ///< images, unlike Qwen3.5-VL's 2.
    unsigned int MINICPM_V_DOWNSAMPLE_FACTOR;   ///< config.downsample_mode "16x"        (16)
    float MINICPM_V_VISION_RESCALE_FACTOR;      ///< 1/255
    float MINICPM_V_VISION_RESCALE_IMAGE_MEAN;  ///< preprocessor.image_mean[0]          (0.5)
    float MINICPM_V_VISION_RESCALE_IMAGE_STD;   ///< preprocessor.image_std[0]           (0.5)

    /// \brief Read the vision preprocessing parameters.
    ///
    /// \note Called by Impl's constructor.
    ///
    /// Source order follows the shipped-config convention (see
    /// `Qwen3.5-0.8B-NPU2/config.json`): the packaged `config.json` is a FLATTENED
    /// config whose `vision_config` block carries ENGINE-named keys
    /// (`MINICPM_V_PATCH_SIZE`, ...), not the upstream HuggingFace names. Each field
    /// falls back to the upstream HF name, then to the literal from the shipping
    /// MiniCPM-V-4.7-1B checkpoint, so a checkpoint packaged either way works and
    /// nothing throws.
    inline void load_vision_preprocess_parameters(LM_Config& config) {
        const nlohmann::json& vc = config.sub("vision_config");

        MINICPM_V_PATCH_SIZE =
            cfg_get<u32>(vc, "MINICPM_V_PATCH_SIZE", cfg_get<u32>(vc, "patch_size", 14u));
        MINICPM_V_IMAGE_SIZE =
            cfg_get<u32>(vc, "MINICPM_V_IMAGE_SIZE", cfg_get<u32>(vc, "image_size", 980u));
        MINICPM_V_MAX_SLICE_NUMS =
            cfg_get<u32>(vc, "MINICPM_V_MAX_SLICE_NUMS",
                         cfg_get<u32>(config._json_config, "max_slice_nums", 9u));
        MINICPM_V_INSERT_LAYER_ID =
            cfg_get<u32>(vc, "MINICPM_V_INSERT_LAYER_ID",
                         cfg_get<u32>(config._json_config, "insert_layer_id", 6u));
        MINICPM_V_SPATIAL_MERGE_SIZE  = cfg_get<u32>(vc, "MINICPM_V_SPATIAL_MERGE_SIZE", 2u);
        MINICPM_V_TEMPORAL_PATCH_SIZE = cfg_get<u32>(vc, "MINICPM_V_TEMPORAL_PATCH_SIZE", 1u);

        // downsample_mode is a string ("16x" / "4x") upstream, so the packaged config
        // carries the resolved integer instead.
        MINICPM_V_DOWNSAMPLE_FACTOR = cfg_get<u32>(vc, "MINICPM_V_DOWNSAMPLE_FACTOR", 0u);
        if (MINICPM_V_DOWNSAMPLE_FACTOR == 0u) {
            const std::string mode =
                cfg_get<std::string>(config._json_config, "downsample_mode", std::string("16x"));
            MINICPM_V_DOWNSAMPLE_FACTOR = (mode == "4x") ? 4u : 16u;
        }

        MINICPM_V_VISION_RESCALE_FACTOR =
            cfg_get<f32>(vc, "MINICPM_V_VISION_RESCALE_FACTOR", 1.0f / 255.0f);
        MINICPM_V_VISION_RESCALE_IMAGE_MEAN =
            cfg_get<f32>(vc, "MINICPM_V_VISION_RESCALE_IMAGE_MEAN", 0.5f);
        MINICPM_V_VISION_RESCALE_IMAGE_STD =
            cfg_get<f32>(vc, "MINICPM_V_VISION_RESCALE_IMAGE_STD", 0.5f);
    }

private:
    struct Impl;
    Impl* _impl;
};

/// \brief the saved-state table of this engine (flm_state_api.h); nullptr if `engine` is not a minicpm_v_4_7_npu
extern "C" const flm_state_api* minicpm_v_4_7_state_api(void* engine);
