/// \file qwen3vl_npu.hpp
/// \brief qwen3vl_npu class
/// \author FastFlowLM Team
/// \date 2026-01-23
/// \version 0.9.28
/// \note This is a header file for the qwen3vl_npu class
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
#if USEAVX2
#include <immintrin.h>  // For AVX intrinsics
#endif

// The operations this engine declares, and how their keys are spelled. Both
// belong to the model, not to the plugin API: another engine may have no
// layers, or a nest of them, and names its own operations.
namespace gemma4e_ops {

// One shape per call-site convention. A hook always returns
// flm::hook_result<R>: a real result, skip() (the operation already
// happened, nothing further to do), or defer() (run the engine's own
// implementation). R is ert_cmd_state for a blocking call site, xrt::run
// for one that builds a run to start/wait later.
using proj_sig_t = flm::hook_result<ert_cmd_state>(bytes& out, bytes& in, bytes& weights, int64_t layer, int64_t padded);
using proj_async_sig_t = flm::hook_result<xrt::run>(bytes& out, bytes& in, bytes& weights, int64_t layer, int64_t padded);
using attn_core_sig_t = flm::hook_result<ert_cmd_state>(bytes& out, bytes& q, bytes& kv_cache, int64_t layer, int64_t padded, int64_t l_begin_chunked);
using dequant_sig_t = flm::hook_result<ert_cmd_state>(bytes& dequantized, bytes& quantized, int64_t layer, int64_t padded);
using decode_layer_sig_t = flm::hook_result<ert_cmd_state>(bytes& hidden_state_inout, bytes& proj_weights,
    bytes& rms_weights, bytes& rope_rms_weights, bytes& kv_cache, int64_t layer, int64_t context_len);
using decode_layer_async_sig_t = flm::hook_result<xrt::run>(bytes& hidden_state_inout, bytes& proj_weights,
    bytes& rms_weights, bytes& rope_rms_weights, bytes& kv_cache, int64_t layer, int64_t context_len);
using lm_head_sig_t = flm::hook_result<xrt::run>(bytes& logits, bytes& lm_head_weights, bytes& hidden_state);
using audio_conv1d_sig_t = flm::hook_result<ert_cmd_state>(bytes& out, bytes& in, bytes& weights, int64_t layer);

// The vision and audio encoders' own generated sequences take (in, weights,
// out) -- the reverse of the decode path's (out, in, weights) -- so these get
// their own shapes rather than reusing proj_sig_t/proj_async_sig_t.
using layer_proj_sig_t = flm::hook_result<ert_cmd_state>(bytes& in, bytes& weights, bytes& out, int64_t layer, int64_t padded);
using layer_proj_async_sig_t = flm::hook_result<xrt::run>(bytes& in, bytes& weights, bytes& out, int64_t layer, int64_t padded);
// A one-shot embedding/projection step: no layer loop, so no layer or padded argument.
using embed_sig_t = flm::hook_result<ert_cmd_state>(bytes& in, bytes& weights, bytes& out);
// Vision attention takes q/k/v as three separate buffers, not one kv_cache.
using vision_attn_core_sig_t = flm::hook_result<xrt::run>(bytes& out, bytes& q, bytes& k, bytes& v, int64_t layer, int64_t padded);
// The background weight-preload run's sequence takes no buffer arguments at all.
using preload_sig_t = flm::hook_result<xrt::run>();

// Every operation, one name and one typedef each: sliding-window and global
// attention run different kernels, and a skip layer's mlp is a different
// (double-wide) sequence, so each gets its own rather than sharing one
// distinguished by argument. decode.layer additionally has a plain and an
// ".async" name, since the engine dispatches it both ways depending on
// whether NPU preemption is enabled.
namespace op {
using q_swa_proj_func_t = proj_sig_t;
using q_global_proj_func_t = proj_sig_t;
using k_swa_proj_func_t = proj_async_sig_t;
using k_global_proj_func_t = proj_async_sig_t;
using v_swa_proj_func_t = proj_async_sig_t;
using v_global_proj_func_t = proj_async_sig_t;
using o_swa_proj_func_t = proj_sig_t;
using o_global_proj_func_t = proj_sig_t;
using swa_attn_core_func_t = attn_core_sig_t;
using global_attn_core_func_t = attn_core_sig_t;
inline constexpr std::string_view q_swa_proj = "self_attn.q_proj.swa";
inline constexpr std::string_view q_global_proj = "self_attn.q_proj.global";
inline constexpr std::string_view k_swa_proj = "self_attn.k_proj.swa";
inline constexpr std::string_view k_global_proj = "self_attn.k_proj.global";
inline constexpr std::string_view v_swa_proj = "self_attn.v_proj.swa";
inline constexpr std::string_view v_global_proj = "self_attn.v_proj.global";
inline constexpr std::string_view o_swa_proj = "self_attn.o_proj.swa";
inline constexpr std::string_view o_global_proj = "self_attn.o_proj.global";
inline constexpr std::string_view swa_attn_core = "self_attn.core.swa";
inline constexpr std::string_view global_attn_core = "self_attn.core.global";

using gate_proj_func_t = proj_sig_t;
using gate_skip_proj_func_t = proj_sig_t;
using up_proj_func_t = proj_sig_t;
using up_skip_proj_func_t = proj_sig_t;
using down_proj_func_t = proj_sig_t;
using down_skip_proj_func_t = proj_sig_t;
inline constexpr std::string_view gate_proj = "mlp.gate_proj";
inline constexpr std::string_view gate_skip_proj = "mlp.gate_proj.skip";
inline constexpr std::string_view up_proj = "mlp.up_proj";
inline constexpr std::string_view up_skip_proj = "mlp.up_proj.skip";
inline constexpr std::string_view down_proj = "mlp.down_proj";
inline constexpr std::string_view down_skip_proj = "mlp.down_proj.skip";

// One dequant call per gemma4e_layer_type_t, matching the engine's own
// apps[4][matrix] table. Index with int(type) (e_gemma4e_swa_layer=0,
// e_gemma4e_global_layer=1, e_gemma4e_swa_layer_skip=2, e_gemma4e_global_layer_skip=3).
using dequant_func_t = dequant_sig_t;
inline constexpr std::string_view dequant_qkv[4]  = { "dequant.qkv.swa",  "dequant.qkv.global",
                                                      "dequant.qkv.swa_skip",  "dequant.qkv.global_skip" };
inline constexpr std::string_view dequant_o[4]    = { "dequant.o.swa",    "dequant.o.global",
                                                      "dequant.o.swa_skip",    "dequant.o.global_skip" };
inline constexpr std::string_view dequant_gate[4] = { "dequant.gate.swa", "dequant.gate.global",
                                                      "dequant.gate.swa_skip", "dequant.gate.global_skip" };
inline constexpr std::string_view dequant_up[4]   = { "dequant.up.swa",   "dequant.up.global",
                                                      "dequant.up.swa_skip",   "dequant.up.global_skip" };
inline constexpr std::string_view dequant_down[4] = { "dequant.down.swa", "dequant.down.global",
                                                      "dequant.down.swa_skip", "dequant.down.global_skip" };

using swa_layer_func_t = decode_layer_sig_t;
using global_layer_func_t = decode_layer_sig_t;
using swa_skip_layer_func_t = decode_layer_sig_t;
using global_skip_layer_func_t = decode_layer_sig_t;
using swa_layer_async_func_t = decode_layer_async_sig_t;
using global_layer_async_func_t = decode_layer_async_sig_t;
using swa_skip_layer_async_func_t = decode_layer_async_sig_t;
using global_skip_layer_async_func_t = decode_layer_async_sig_t;
inline constexpr std::string_view swa_layer = "decode.layer.swa";
inline constexpr std::string_view global_layer = "decode.layer.global";
inline constexpr std::string_view swa_skip_layer = "decode.layer.swa_skip";
inline constexpr std::string_view global_skip_layer = "decode.layer.global_skip";
inline constexpr std::string_view swa_layer_async = "decode.layer.swa.async";
inline constexpr std::string_view global_layer_async = "decode.layer.global.async";
inline constexpr std::string_view swa_skip_layer_async = "decode.layer.swa_skip.async";
inline constexpr std::string_view global_skip_layer_async = "decode.layer.global_skip.async";

using lm_head_func_t = lm_head_sig_t;
inline constexpr std::string_view lm_head = "lm_head";

using audio_conv1d_func_t = audio_conv1d_sig_t;
inline constexpr std::string_view audio_conv1d = "audio.conv1d";

// The vision encoder: one patch-embedding step, then one attention block and
// one mlp per hidden layer, then one projection into the language model's
// embedding space.
using vision_patch_embed_func_t = embed_sig_t;
using vision_pos_embed_dim0_func_t = embed_sig_t;
using vision_pos_embed_dim1_func_t = embed_sig_t;
using vision_q_proj_func_t = layer_proj_async_sig_t;
using vision_k_proj_func_t = layer_proj_async_sig_t;
using vision_v_proj_func_t = layer_proj_async_sig_t;
using vision_attn_core_func_t = vision_attn_core_sig_t;
using vision_o_proj_func_t = layer_proj_sig_t;
using vision_gate_proj_func_t = layer_proj_async_sig_t;
using vision_up_proj_func_t = layer_proj_async_sig_t;
using vision_down_proj_func_t = layer_proj_sig_t;
using vision_to_language_proj_func_t = embed_sig_t;
inline constexpr std::string_view vision_patch_embed = "vision.patch_embed";
inline constexpr std::string_view vision_pos_embed_dim0 = "vision.pos_embed.dim0";
inline constexpr std::string_view vision_pos_embed_dim1 = "vision.pos_embed.dim1";
inline constexpr std::string_view vision_q_proj = "vision.q_proj";
inline constexpr std::string_view vision_k_proj = "vision.k_proj";
inline constexpr std::string_view vision_v_proj = "vision.v_proj";
inline constexpr std::string_view vision_attn_core = "vision.attn_core";
inline constexpr std::string_view vision_o_proj = "vision.o_proj";
inline constexpr std::string_view vision_gate_proj = "vision.mlp.gate_proj";
inline constexpr std::string_view vision_up_proj = "vision.mlp.up_proj";
inline constexpr std::string_view vision_down_proj = "vision.mlp.down_proj";
inline constexpr std::string_view vision_to_language_proj = "vision.to_language_proj";

// The audio encoder: one sub-sample projection step, then one attention
// block, one conv1d block (its own start/end pointwise projections; the
// depthwise conv1d itself is audio_conv1d above) and one FFN pair per hidden
// layer, then a pre-encode projection and one into the language model's
// embedding space.
using audio_sub_sample_proj_func_t = embed_sig_t;
using audio_q_proj_func_t = layer_proj_async_sig_t;
using audio_k_proj_func_t = layer_proj_async_sig_t;
using audio_v_proj_func_t = layer_proj_async_sig_t;
using audio_o_proj_func_t = layer_proj_sig_t;
using audio_ffn_up_proj_func_t = layer_proj_async_sig_t;
using audio_ffn_down_proj_func_t = layer_proj_sig_t;
using audio_conv1d_start_proj_func_t = layer_proj_sig_t;
using audio_conv1d_end_proj_func_t = layer_proj_sig_t;
using audio_pre_encode_proj_func_t = embed_sig_t;
using audio_to_language_proj_func_t = embed_sig_t;
inline constexpr std::string_view audio_sub_sample_proj = "audio.sub_sample_proj";
inline constexpr std::string_view audio_q_proj = "audio.q_proj";
inline constexpr std::string_view audio_k_proj = "audio.k_proj";
inline constexpr std::string_view audio_v_proj = "audio.v_proj";
inline constexpr std::string_view audio_o_proj = "audio.o_proj";
inline constexpr std::string_view audio_ffn_up_proj = "audio.ffn.up_proj";
inline constexpr std::string_view audio_ffn_down_proj = "audio.ffn.down_proj";
inline constexpr std::string_view audio_conv1d_start_proj = "audio.conv1d.start_proj";
inline constexpr std::string_view audio_conv1d_end_proj = "audio.conv1d.end_proj";
inline constexpr std::string_view audio_pre_encode_proj = "audio.pre_encode_proj";
inline constexpr std::string_view audio_to_language_proj = "audio.to_language_proj";

// The per-layer-input (PLI) path: a model-wide down projection run once per
// batch, and a gate/up pair run at the end of every layer.
using pli_down_proj_func_t = embed_sig_t;
using pli_gate_proj_func_t = layer_proj_sig_t;
using pli_up_proj_func_t = layer_proj_sig_t;
inline constexpr std::string_view pli_down_proj = "pli.down_proj";
inline constexpr std::string_view pli_gate_proj = "pli.gate_proj";
inline constexpr std::string_view pli_up_proj = "pli.up_proj";

// The background run that preloads the next layer's weights while the
// current token executes. Its sequence takes no buffer arguments.
using layer_pre_load_func_t = preload_sig_t;
inline constexpr std::string_view layer_pre_load = "engine.layer_pre_load";
}  // namespace op

}  // namespace gemma4e_ops


// some helper functions for convenience
constexpr int GEMMA4E_IS_GLOBAL_MASK = 0x00000001;
constexpr int GEMMA4E_IS_SKIP_MASK = 0x00000002;

typedef enum :int {
    e_gemma4e_swa_layer = 0,
    e_gemma4e_global_layer = GEMMA4E_IS_GLOBAL_MASK,
    e_gemma4e_swa_layer_skip = GEMMA4E_IS_SKIP_MASK,
    e_gemma4e_global_layer_skip = GEMMA4E_IS_GLOBAL_MASK | GEMMA4E_IS_SKIP_MASK,
    e_gemma4e_total_layer_types = 4
} gemma4e_layer_type_t;

inline bool is_swa_layer(gemma4e_layer_type_t layer) {
    return (layer & GEMMA4E_IS_GLOBAL_MASK) == 0;
}

inline bool is_global_layer(gemma4e_layer_type_t layer) {
    return (layer & GEMMA4E_IS_GLOBAL_MASK) != 0;
}

inline bool is_skip_layer(gemma4e_layer_type_t layer) {
    return (layer & GEMMA4E_IS_SKIP_MASK) != 0;
}

typedef struct {
    int height;
    int width;
    int height_resized;  // assigned by image preprocessing
    int width_resized;
    // int grid_h;
    // int grid_w;

    bytes _data;

} gemma4e_image_t;





struct gemma4e_image_payload_t{
    // original raw data
    std::vector<std::pair<int, int>> image_patch__element_per_patch; // [num_of_image][width, height]
    std::vector<uint32_t> valid_patch_size_per_image; // [num_of_image], the unpadded size per image
    std::vector<std::vector<bf16>> pixel_values; // [num_of_image][image_size], where image_size = height_resized * width_resized * 3
    std::vector< std::vector<int>> image_grid_pairs_per_image; // [num_of_image][num_of_position_id][x, y]
    std::vector<unsigned int> num_soft_tokens_per_image; // [num_of_image]
    unsigned int num_images;
};


struct gemma4e_audio_payload_t{
    // per-audio mel spectrogram data
    std::vector<std::vector<bf16>> mel_spectrograms;               // [num_audios][frames * bins], row-major
    std::vector<int> mel_spectrogram_frames_per_audio;             // [num_audios]
    std::vector<int> mel_spectrogram_bins_per_audio;               // [num_audios]
    unsigned int num_audios = 0;
    std::vector<unsigned int> num_soft_tokens_per_audio; // [num_audios]

};


typedef struct {
    gemma4e_image_payload_t image_payload;
    gemma4e_audio_payload_t audio_payload;
} gemma4e_multi_modal_payload_t;

class gemma4e_npu : public causal_lm{
public:
    /// \brief  initialize the qwen3vl_npu
    /// \param config the configuration
    /// \param npu_instance the npu instance
    gemma4e_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~gemma4e_npu();

    /// \brief forward the qwen3vl_npu
    /// \param ids the ids
    /// \return the output tensor
    buffer<bf16> forward(int ids) override;
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;

    /// \brief set the context length
    /// \param L the context length
    void set_context_length(int L) override;

    /// \brief load the weights
    /// \param q4nx the q4nx
    void load_weights(Q4NX& q4nx) override;

    /// \brief update the max length
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

    /// \brief update the max length
    /// \param MAX_L the max length
    void update_max_length(uint32_t MAX_L) override;

    /// \brief get the current context length
    /// \return the current context length
    int get_current_context_length() override;


    int checkpoint() override;
    int restore() override;

    // parameters for vision preprocessing in Gemma4e
    unsigned int GEMMA4E_VISION_MAX_POSITION_EMBEDDINGS;
    unsigned int GEMMA4E_VISION_NUM_HIDDEN_LAYERS;
    unsigned int GEMMA4E_VISION_NUM_ATTENTION_HEADS;
    unsigned int GEMMA4E_VISION_HIDDEN_SIZE;
    unsigned int GEMMA4E_VISION_INTERMEDIATE_SIZE;
    unsigned int GEMMA4E_VISION_HEAD_DIM;
    unsigned int GEMMA4E_VISION_PATCH_SIZE;
    float GEMMA4E_ROPE_THETA;
    unsigned int GEMMA4E_POOLING_KERNEL_SIZE;
    unsigned int GEMMA4E_POSITION_EMBEDDING_SIZE;
    unsigned int GEMMA4E_VISION_IMAGE_OUTPUT_SIZE;
    float GEMMA4E_VISION_RESCALE_FACTOR;
    float GEMMA4E_VISION_IMAGE_MEAN;
    float GEMMA4E_VISION_IMAGE_STD;

    // parameters for audio preprocessing in Gemma4e
    unsigned int Audio_MM_TILE_M;
    unsigned int Audio_MM_TILE_K;
    unsigned int Audio_MM_TILE_N;
    int Gemma4E_Audio_resample_rate;
    float Gemma4E_Audio_gradient_clipping;
    unsigned int Gemma4E_Audio_Multimodal_Output_SIZE;
    unsigned int Gemma4E_Audio_language_projection_output_size;
    unsigned int Gemma4E_Audio_HIDDEN_SIZE;
    unsigned int Gemma4E_Audio_INTERMEDIATE_SIZE;
    unsigned int Gemma4E_Audio_attention_chunk_size;
    unsigned int Gemma4E_Audio_attention_context_left;
    unsigned int Gemma4E_Audio_attention_context_right;
    unsigned int Gemma4E_Audio_num_attention_heads;
    unsigned int Gemma4E_Audio_num_attention_layers;
    unsigned int Gemma4E_Audio_conv1d_kernel_size;
    unsigned int Gemma4E_Audio_conv1d_stride;
    unsigned int Gemma4E_Audio_conv2d_kernel_size;
    unsigned int Gemma4E_Audio_conv2d_Stride;
    unsigned int Gemma4e_Audio_conv2d_Padding;
    unsigned int Gemma4E_Audio_subsampling_conv_channels_0;
    unsigned int Gemma4E_Audio_subsampling_conv_channels_1;
    float Gemma4E_Audio_attention_softcap;


    inline void load_vision_preprocess_parameters(LM_Config& config){
        // Note: this should be called by Impl:: constructor
        GEMMA4E_VISION_MAX_POSITION_EMBEDDINGS = config.sub("vision_config").value("GEMMA4E_VISION_MAX_POSITION_EMBEDDINGS", -1);
        GEMMA4E_VISION_NUM_HIDDEN_LAYERS   = config.sub("vision_config").value("GEMMA4E_VISION_NUM_HIDDEN_LAYERS", -1);
        GEMMA4E_VISION_NUM_ATTENTION_HEADS = config.sub("vision_config").value("GEMMA4E_VISION_NUM_ATTENTION_HEADS", -1);
        GEMMA4E_VISION_HIDDEN_SIZE         = config.sub("vision_config").value("GEMMA4E_VISION_HIDDEN_SIZE", -1);
        GEMMA4E_VISION_INTERMEDIATE_SIZE   = config.sub("vision_config").value("GEMMA4E_VISION_INTERMEDIATE_SIZE", -1);
        GEMMA4E_VISION_HEAD_DIM            = config.sub("vision_config").value("GEMMA4E_VISION_HEAD_DIM", -1);
        GEMMA4E_VISION_PATCH_SIZE          = config.sub("vision_config").value("GEMMA4E_VISION_PATCH_SIZE", -1);
        GEMMA4E_ROPE_THETA                 = config.sub("vision_config").value("GEMMA4E_ROPE_THETA", -1.0f);
        GEMMA4E_POOLING_KERNEL_SIZE        = config.sub("vision_config").value("GEMMA4E_POOLING_KERNEL_SIZE", -1);
        GEMMA4E_POSITION_EMBEDDING_SIZE    = config.sub("vision_config").value("GEMMA4E_POSITION_EMBEDDING_SIZE", -1);
        GEMMA4E_VISION_IMAGE_OUTPUT_SIZE   = config.sub("vision_config").value("GEMMA4E_VISION_IMAGE_OUTPUT_SIZE", -1);
        GEMMA4E_VISION_RESCALE_FACTOR      = config.sub("vision_config").value("GEMMA4E_VISION_RESCALE_FACTOR", -1.0f);
        GEMMA4E_VISION_IMAGE_MEAN          = config.sub("vision_config").value("GEMMA4E_VISION_IMAGE_MEAN", -1.0f);
        GEMMA4E_VISION_IMAGE_STD           = config.sub("vision_config").value("GEMMA4E_VISION_IMAGE_STD", -1.0f);
    }

    inline void load_audio_preprocess_parameters(LM_Config& config){
        Audio_MM_TILE_M = config.sub("audio_config").value("Audio_MM_TILE_M", 128);
        Audio_MM_TILE_K = config.sub("audio_config").value("Audio_MM_TILE_K", 512);
        Audio_MM_TILE_N = config.sub("audio_config").value("Audio_MM_TILE_N", 64);
        Gemma4E_Audio_resample_rate = config.sub("audio_config").value("Gemma4E_Audio_audio_resample_rate", -1);
        Gemma4E_Audio_gradient_clipping = config.sub("audio_config").value("Gemma4E_Audio_gradient_clipping", -1.0f);
        Gemma4E_Audio_Multimodal_Output_SIZE = config.sub("audio_config").value("Gemma4E_Audio_Multimodal_Output_SIZE", -1);
        Gemma4E_Audio_language_projection_output_size = config.sub("audio_config").value("Gemma4E_Audio_language_projection_output_size", -1);
        Gemma4E_Audio_HIDDEN_SIZE = config.sub("audio_config").value("Gemma4E_Audio_HIDDEN_SIZE", -1);
        Gemma4E_Audio_INTERMEDIATE_SIZE = config.sub("audio_config").value("Gemma4E_Audio_INTERMEDIATE_SIZE", -1);
        Gemma4E_Audio_attention_chunk_size = config.sub("audio_config").value("Gemma4E_Audio_attention_chunk_size", -1);
        Gemma4E_Audio_attention_context_left = config.sub("audio_config").value("Gemma4E_Audio_attention_context_left", -1);
        Gemma4E_Audio_attention_context_right = config.sub("audio_config").value("Gemma4E_Audio_attention_context_right", -1);
        Gemma4E_Audio_num_attention_heads = config.sub("audio_config").value("Gemma4E_Audio_num_attention_heads", -1);
        Gemma4E_Audio_num_attention_layers = config.sub("audio_config").value("Gemma4E_Audio_num_attention_layers", -1);
        Gemma4E_Audio_conv1d_kernel_size = config.sub("audio_config").value("Gemma4E_Audio_conv1d_kernel_size", -1);
        Gemma4E_Audio_conv1d_stride = config.sub("audio_config").value("Gemma4E_Audio_conv1d_stride", -1);
        Gemma4E_Audio_conv2d_kernel_size = config.sub("audio_config").value("Gemma4E_conv2d_kernel_size", -1);
        Gemma4E_Audio_conv2d_Stride = config.sub("audio_config").value("Gemma4E_conv2d_Stride", -1);
        Gemma4e_Audio_conv2d_Padding = config.sub("audio_config").value("Gemma4e_conv2d_Padding", -1);
        Gemma4E_Audio_subsampling_conv_channels_0 = config.sub("audio_config").value("Gemma4E_Audio_subsampling_conv_channels_0", -1);
        Gemma4E_Audio_subsampling_conv_channels_1 = config.sub("audio_config").value("Gemma4E_Audio_subsampling_conv_channels_1", -1);
        Gemma4E_Audio_attention_softcap = config.sub("audio_config").value("Gemma4E_Audio_attention_softcap", -1.0f);
    }

private:
    struct Impl;
    Impl* _impl;
};

