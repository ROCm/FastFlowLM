/*!
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file embedding_handler.cpp
 * \brief Embedding model loading and request handling.
 */
#include "rest_handler.hpp"
#include "http_guard.hpp"

///@brief Ensure the embed model is loaded
///@param model_tag the model tag
void RestHandler::ensure_embed_model_loaded(const std::string& model_tag) {
    const std::string tag = supported_models.rectify_model_tag(model_tag);
    if (!supported_models.is_model_supported(tag))
        throw std::invalid_argument("Unknown embedding model: " + model_tag);
    auto info = supported_models.get_model_info(tag).second;
    if (!is_embedding_model(info))
        throw std::invalid_argument("Model is not an embedding model: " + model_tag);
    if (auto_embedding_engine && current_embedding_tag == tag) return;
    switch (downloader.is_model_downloaded(tag, false, /*fast_check=*/true)) {
        case ModelDownloader::ModelStatus::Ready:
            break;
        case ModelDownloader::ModelStatus::Outdated:
        case ModelDownloader::ModelStatus::Missing:
            if (!downloader.pull_model(tag, modelscope))
                throw std::runtime_error("Failed to download embedding model: " + tag);
            info = supported_models.get_model_info(tag).second;
            break;
        case ModelDownloader::ModelStatus::Incompatible:
            throw std::invalid_argument("Embedding model is incompatible: " + tag);
    }
    // Release device resources before constructing the replacement. A failed
    // construction/load leaves no engine or tag, so the next request retries.
    auto_embedding_engine.reset();
    current_embedding_tag.clear();
    auto engine = get_auto_embedding_model(info, npu_device_inst);
    engine->load_model(supported_models.get_model_path(tag), info, preemption);
    auto_embedding_engine = std::move(engine);
    current_embedding_tag = tag;
}

///@brief Handle the embeddings request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_embeddings(const json& request,
                                   std::function<void(const json&)> send_response,
                                   StreamResponseCallback send_streaming_response,
                                   std::function<bool()> cancelled) {
    try {
        if (auto bad = flm::http_guard::validate_embeddings(request)) {
            send_response(*bad);
            return;
        }
        std::lock_guard lock(embedding_mutex);
        std::string model = request.at("model");
        const auto& input = request.at("input");
        const json inputs = input.is_array() ? input : json::array({input});
        ensure_embed_model_loaded(model);
        json embedding_data = json::array();
        uint64_t prompt_tokens = 0;
        json item_request = request;
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (cancelled()) return;
            item_request["input"] = inputs[i];
            auto result = auto_embedding_engine->embed(item_request, cancelled);
            prompt_tokens += result.prompt_tokens;
            embedding_data.push_back({
                {"object", "embedding"},
                {"embedding", result.values},
                {"index", i}
            });
        }
        if (cancelled()) return;

        json response = {
            {"object", "list"},
            {"data", embedding_data},
            {"model", model},
            {"usage", {
                {"prompt_tokens", prompt_tokens},
                {"total_tokens", prompt_tokens}
            }}
        };
        send_response(response);
    }
    catch (const std::exception& e) {
        if (cancelled()) return;
        const int code = dynamic_cast<const std::invalid_argument*>(&e) ||
                         dynamic_cast<const nlohmann::json::exception*>(&e) ? 400 : 500;
        send_response({{"error", {{"message", e.what()},
                                  {"type", code == 400 ? "invalid_request_error" : "server_error"},
                                  {"code", code}}}});
    }
}
