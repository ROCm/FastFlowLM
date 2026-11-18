/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file rest_handler.hpp
 * \brief RestHandler class and related declarations
 * \author FastFlowLM Team
 * \date 2025-06-24
 *  \version 0.9.24
 */
#pragma once
#include <optional>
#include <mutex>

#include "AutoModel/all_models.hpp"
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
#include "whisper/modeling_whisper.hpp"
#endif
#include "AutoEmbeddingModel/all_embedding_model.hpp"
#include "AutoDecisionModel/auto_decision_model.hpp"
#include "model_list.hpp"
#include "chat/chat_parser.hpp"
#include "chat/chat_stream.hpp"
#include "program_args.hpp"


#include "model_downloader.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <functional>
#include "prompt_cache.hpp"

using json = nlohmann::ordered_json;

// Forward declaration
struct CancellationToken;
class chat_logprobs_aligner;

///@brief Stream callback type for sending streaming responses
using StreamResponseCallback = std::function<void(const json&, bool)>; // data, is_final

class RestHandler {
public:
    RestHandler(model_list& models, ModelDownloader& downloader, program_args_t& args,
                flm_rt::device* npu_device);
    ~RestHandler();

    void handle_show(const json& request,
        std::function<void(const json&)> send_response,
        StreamResponseCallback send_streaming_response);

    void handle_generate(const json& request, 
                        std::function<void(const json&)> send_response,
                        StreamResponseCallback send_streaming_response,
                        std::shared_ptr<CancellationToken> cancellation_token = nullptr);

    void handle_chat(const json& request,
                    std::function<void(const json&)> send_response, 
                    StreamResponseCallback send_streaming_response,
                    std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    

    void handle_embeddings(const json& request,
                          std::function<void(const json&)> send_response,
                          StreamResponseCallback send_streaming_response,
                          std::function<bool()> cancelled = [] { return false; });
    

    void handle_models(const json& request,
                      std::function<void(const json&)> send_response,
                      StreamResponseCallback send_streaming_response);
    
    void handle_models_openai(const json& request,
                            std::function<void(const json&)> send_response,
                            StreamResponseCallback send_streaming_response);

    void handle_ps(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_version(const json& request,
                       std::function<void(const json&)> send_response,
                       StreamResponseCallback send_streaming_response);
    
    // Placeholder handlers for unimplemented endpoints
    void handle_pull(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_push(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_delete(const json& request,
                      std::function<void(const json&)> send_response,
                      StreamResponseCallback send_streaming_response);
    
    void handle_copy(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_create(const json& request,
                      std::function<void(const json&)> send_response,
                      StreamResponseCallback send_streaming_response);

    void handle_openai_chat_completion(const json& request,
                                      std::function<void(const json&)> send_response,
                                      StreamResponseCallback send_streaming_response,
                                      std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    void handle_openai_audio_transcriptions(const json& request,
                                      std::function<void(const json&)> send_response,
                                      StreamResponseCallback send_streaming_response,
                                      std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    /// GET /v1/systemone/models: the served decision model, in vLLM Semantic Router's
    /// System One list format ({"object": "list", "data": [{"id", "object", "api"}]})
    void handle_systemone_models(std::function<void(const json&)> send_response);

    /// POST /v1/systemone: a System One request for the --decision model
    void handle_systemone(const std::string& raw_body,
                          std::function<void(const json&)> send_response);

    void handle_openai_completion(const json& request,
        std::function<void(const json&)> send_response,
        StreamResponseCallback send_streaming_response,
        std::shared_ptr<CancellationToken> cancellation_token = nullptr);

private:
    bool ensure_model_loaded(const std::string& model_tag,
                             const std::string& request_backend = "", std::string* refusal = nullptr);
    bool is_known_model(const std::string& model_tag);
    void ensure_asr_model_loaded(const std::string& model_tag);
    void ensure_embed_model_loaded(const std::string& model_tag);
    void ensure_decision_model_loaded(const std::string& model_tag);
    void configure_chat_engine_parameters(const json& options, const json& request);
    json build_nstream_response(const std::string& response_text, chat_meta_info_t& meta_info,
                                const chat_parse_options& parse_options,
                                chat_logprobs_aligner* logprobs = nullptr);


    flm_rt::device* npu_device_inst;
    std::string power_mode;
    flm::frontend::FrontendPtr auto_chat_engine;
    std::unique_ptr<AutoDecisionModel> decision_engine;
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    std::unique_ptr<Whisper> whisper_engine;
#endif
    /// \brief the --asr model when it is an r2t2 (Qwen3-ASR) one: a second engine beside the chat model, used only for transcription
    flm::frontend::FrontendPtr asr_sidecar_engine;
    std::string asr_sidecar_tag;
    std::unique_ptr<AutoEmbeddingModel> auto_embedding_engine;
    std::string current_embedding_tag;
    std::mutex embedding_mutex;
    model_list& supported_models;
    ModelDownloader& downloader;
    std::mutex model_info_mutex;
    std::string current_model_tag = "model-faker";
    std::string loaded_backend;
    std::string default_model_tag;
    bool modelscope;
    bool asr;
    /// \brief the prefill chunk length of the loaded model
    int prefill_chunk_len;
    /// \brief the --prefill-chunk-len value, -1 when each model's own is used
    int prefill_chunk_len_arg;
    /// \brief what one request may carry in raw RGB8 frames (--max-raw-frames, --max-raw-mb)
    raw_rgb8_limits_t raw_rgb8_limits;
    int generate_context_id;
    int chat_context_id;
    int ctx_length;
    int img_pre_resize;
    /// \brief the --prefix-cache-mb / --prefix-slots / --prefix-min-gain values, -1 when each model's own default is used
    int prefix_cache_mb = -1;
    int prefix_slots = -1;
    int prefix_min_gain = -1;
    std::string last_question;
    bool preemption;
    /// \brief the --backend value, empty when the flag was not given
    std::string backend;
    flm::backend::BackendSelection loaded_selection;
    PromptCache prompt_cache;
};