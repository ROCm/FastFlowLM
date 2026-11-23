/*!
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file embedding_handler.cpp
 * \brief Embedding model loading and request handling.
 */
#include "rest_handler.hpp"
#include "embedding_task.hpp"
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
    auto* device = utils::require_npu_device(npu_device_inst, power_mode);
    auto engine = get_auto_embedding_model(info, device);
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
        // The request's task prompt is resolved only for models that use the
        // built-in task prompts (the built-in Gemma_Embedding engine, which
        // prefixes the input with the task's prompt). The model is known only
        // once it is loaded, so resolution happens after
        // ensure_embed_model_loaded, keyed on the same model_kind check
        // get_auto_embedding_model uses to pick the provider path: an
        // extension/provider model never reads prompt_name / task_type and may
        // have its own prompt_name vocabulary, so its requests are left
        // untouched -- no built-in task is resolved, no value is refused, and
        // the raw `task` field is what reaches the provider. For a built-in
        // model, `prompt_name`, with `task_type` as its alias, selects the
        // prefix the engine prepends. An unknown, non-string, or conflicting
        // value is refused with 400 instead of silently embedding as a query;
        // no task keeps today's `task_query` default. One resolved task
        // applies to every element of an `input` array. The integer code is
        // what send_response turns into the HTTP status, so a 400 built here
        // is sent as 400, not 200.
        std::lock_guard lock(embedding_mutex);
        std::string model = request.at("model");
        const auto& input = request.at("input");
        const json inputs = input.is_array() ? input : json::array({input});
        ensure_embed_model_loaded(model);
        std::string resolved_task;
        if (flm::catalog::model_kind(
                supported_models.get_model_info(model).second) !=
            flm::catalog::kEmbeddingModelKind) {
            using TaskStatus = flm::EmbeddingTaskResolution::Status;
            const auto task_resolution = flm::resolve_embedding_task(request);
            if (task_resolution.status == TaskStatus::NotAString) {
                send_response(flm::http_guard::error_body(
                    "'" + task_resolution.field + "' must be a string, one of [" +
                        flm::embedding_task_names_csv() + "].",
                    400, task_resolution.field));
                return;
            }
            if (task_resolution.status == TaskStatus::Unknown) {
                send_response(flm::http_guard::error_body(
                    "unknown " + task_resolution.field + " '" + task_resolution.value +
                        "'. Known: [" + flm::embedding_task_names_csv() +
                        "]. Refusing to substitute one: an embedding under the wrong task "
                        "prompt is correctly shaped and correctly normed, so nothing "
                        "downstream can tell it is wrong.",
                    400, task_resolution.field));
                return;
            }
            if (task_resolution.status == TaskStatus::Conflict) {
                send_response(flm::http_guard::error_body(
                    "'prompt_name' and 'task_type' are aliases and disagree ('" +
                        task_resolution.field + "' says '" + task_resolution.value +
                        "'). Send one, or send the same task in both.",
                    400, task_resolution.field));
                return;
            }
            if (task_resolution.status == TaskStatus::Ok) {
                resolved_task = task_resolution.task;
                // A `task` naming a different task than `prompt_name`/`task_type`
                // is a conflict, not a silent override in either direction. A
                // non-string `task` must still fail here: the resolved task below
                // would otherwise overwrite it and turn a 400 into a 200.
                if (request.contains("task")) {
                    const json& task = request.at("task");
                    if (!task.is_string()) {
                        send_response(flm::http_guard::error_body(
                            "'task' must be a string.", 400, "task"));
                        return;
                    }
                    if (flm::canonical_embedding_task(task.get<std::string>()) != resolved_task) {
                        send_response(flm::http_guard::error_body(
                            "'task' and '" + task_resolution.field + "' disagree ('task' says '" +
                                task.get<std::string>() + "'). Send one task.",
                            400, "task"));
                        return;
                    }
                }
            }
        }
        json embedding_data = json::array();
        uint64_t prompt_tokens = 0;
        json item_request = request;
        if (!resolved_task.empty()) item_request["task"] = resolved_task;
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
