/// \file qwen3_asr_npu.hpp
/// \brief qwen3_asr_npu class
/// \author FastFlowLM Team
/// \date 2026-01-23
/// \version 0.9.28
/// \note This is a header file for the qwen3_asr_npu class
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "qwen3_asr_npu_sequence.hpp"
#include "modules/embedding.hpp"
#include "modules/lm_head.hpp"
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "causal_lm.hpp"
#include "utils/hybrid_state.hpp"
#if USEAVX2
#include <immintrin.h>  // For AVX intrinsics
#endif


/// \brief number of audio tokens (= <|audio_pad|> slots the prompt needs) for `frames` mel frames
int qwen3_asr_audio_tokens(int frames);

/// \brief payload of qwen3_asr_npu::prefill: what goes into the <|audio_pad|> positions of `ids`, in order. The text rows
///        come from the embedding table. Give EITHER the encoder output (audio_embeds, n_audio rows of `hidden` bf16)
///        OR the mel spectrogram (mel, row-major [mel_frames][128] fp32; the engine runs its audio encoder).
typedef struct {
    int audio_pad_id = 151676;
    int n_audio = 0;
    std::vector<bf16> audio_embeds;   ///< [n_audio][hidden]
    int mel_frames = 0;
    std::vector<float> mel;           ///< [mel_frames][128]; n_audio must equal qwen3_asr_audio_tokens(mel_frames)
    /// Streaming: successive requests of one growing recording (mel = all audio so far). The engine then keeps the encoder output of the
    /// closed 8 s windows and the K/V rows of the prompt head + those windows' audio tokens, and only encodes / prefills what is new.
    /// The caller must not clear the context between the steps and must send the same prompt head; stream_reset starts a new recording.
    bool streaming = false;
    bool stream_reset = false;
} qwen3_asr_payload_t;

class qwen3_asr_npu : public causal_lm{
public:
    /// \brief  initialize the qwen3_asr_npu
    /// \param config the configuration
    /// \param npu_instance the npu instance
    qwen3_asr_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~qwen3_asr_npu();

    /// \brief forward the qwen3_asr_npu
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

    // ---- saved states (see flm_state_api.h) --------------------------------
    // The engine keeps no copy of its own context: read_state() copies the live K/V rows into a caller's
    // state, write_state() puts one back (and sets the context length), write_state_kv() puts back only the rows.
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
private:
    struct Impl;
    Impl* _impl;
};

/// \brief the saved-state table of this engine (flm_state_api.h); nullptr if `engine` is not a qwen3_asr_npu
extern "C" const flm_state_api* qwen3_asr_npu_state_api(void* engine);

