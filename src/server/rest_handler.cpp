/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file rest_handler.cpp
 * \brief RestHandler class and related declarations
 * \author FastFlowLM Team
 * \date 2025-08-05
 *  \version 0.9.24
 */
#include "rest_handler.hpp"
#include "http_guard.hpp"
#include "wstream_buf.hpp"
#include "streaming_ostream.hpp"
#include "streaming_ostream_openai.hpp"
#include "chat_core.hpp"
#include "chat_logprobs.hpp"
#include "protocol_openai_chat.hpp"
#include "image/image_reader.hpp"
#include "image/raw_rgb8.hpp"
#include "base64.hpp"
#include "chat/message_roles.hpp"
#include "chat/positional_tool_results.hpp"
#include "utils/log_privacy.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include "AutoModel/modeling_minicpm_v.hpp"
#include "AutoModel/modeling_qwen3_asr.hpp"
#include "AutoModel/modeling_qwen3_5vl.hpp"
#include "whisper/transcript_format.hpp"
#include <sstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <limits>
#include <locale>
#include <random>
#include "server.hpp"

namespace {
// A failure is a value (chat_error) until a route writes it in its protocol's shape.
// The *Response functions below write the same values as OpenAI's error object.

/// \brief A failure the model reported about this request
chat_error model_error(const ModelRequestError& error) {
    return {error.http_code(), error.type(), error.what(), std::string(), error.session_cleared()};
}

/// \brief An exception on a chat route: the model's own error, or an internal one
chat_error exception_error(const std::exception& error) {
    if (const auto* request_error = dynamic_cast<const ModelRequestError*>(&error))
        return model_error(*request_error);
    return {500, "server_error", error.what(), std::string(), std::nullopt};
}

/// \brief insert() returned false without being cancelled: the reason the model gave, if any
chat_error prefill_failed_error(const chat_meta_info_t& meta_info) {
    const std::string message = meta_info.error_message.empty() ? "Prefill failed (see server log)" : meta_info.error_message;
    return {meta_info.error_code, meta_info.error_code >= 500 ? "server_error" : "invalid_request_error",
            message, std::string(), std::nullopt};
}

/// \brief A model name the catalog does not have: a 404, as OpenAI's
chat_error unknown_model_error(const std::string& model) {
    return {404, "invalid_request_error", "The model '" + model + "' does not exist. Please check with `flm list`.",
            "model", std::nullopt};
}

/// \brief A known model, or the loaded one, cannot be used
/// \note "model-faker" is what a request without "model" names while nothing is loaded: the request is at fault
chat_error model_load_failed_error(const std::string& model, const std::string& refusal) {
    if (model == "model-faker") {
        return {400, "invalid_request_error", "No model is loaded; name one in 'model'.", "model", std::nullopt};
    }
    return {500, "server_error", refusal.empty() ? "Failed to load " + model + " model!" : refusal,
            std::string(), std::nullopt};
}

/// \brief The engine's stop reason as the reason a chat turn ended
chat_finish chat_finish_from(stop_reason_t reason) {
    switch (reason) {
        case EOT_DETECTED: return chat_finish::STOP;
        case MAX_LENGTH_REACHED: return chat_finish::LENGTH;
        case CANCEL_DETECTED: return chat_finish::CANCEL;
        case TOOL_DETECTED: return chat_finish::TOOL_CALLS;
        case ERROR_DETECTED: break;
    }
    return chat_finish::FAILED;
}

json ModelErrorResponse(const ModelRequestError& error) {
    return openai_error_json(model_error(error));
}

json ExceptionResponse(const std::exception& error) {
    if (const auto* model_error = dynamic_cast<const ModelRequestError*>(&error))
        return ModelErrorResponse(*model_error);
    return {{"error", error.what()}};
}

/// \brief The reply to an exception on a /v1 endpoint: an OpenAI error body whose code makes it an HTTP 500
/// \note ExceptionResponse above keeps the Ollama shape ({"error": "..."}, no code, so HTTP 200)
///       that the /api/* handlers send; only the /v1 handlers use this one.
json OpenAIExceptionResponse(const std::exception& error) {
    return openai_error_json(exception_error(error));
}

/// \brief The reply when insert() returned false without being cancelled: the reason the model gave, if any
json PrefillFailedResponse(const chat_meta_info_t& meta_info) {
    return openai_error_json(prefill_failed_error(meta_info));
}

/// \brief The reply to a model name the catalog does not have: a 404, as OpenAI's
json UnknownModelResponse(const std::string& model) {
    return openai_error_json(unknown_model_error(model));
}

/// \brief The reply when a known model, or the loaded one, cannot be used
json ModelLoadFailedResponse(const std::string& model, const std::string& refusal) {
    return openai_error_json(model_load_failed_error(model, refusal));
}

/// \brief One line per finished request in the server log, streamed or not, whatever the client asked for
/// \note The prefix "Request done:" is stable so tools can grep it
void log_request_summary(const chat_meta_info_t& m, size_t kv_used, size_t kv_max) {
    const double prefill_s = static_cast<double>(m.prefill_duration) / 1e9;
    const double decode_s = static_cast<double>(m.decoding_duration) / 1e9;
    const auto new_prompt = m.prompt_tokens > m.cached_prompt_tokens ? m.prompt_tokens - m.cached_prompt_tokens : 0;
    std::ostringstream os;
    os << std::fixed << std::setprecision(3)
       << "Request done: prompt " << m.prompt_tokens << " tokens";
    if (m.cached_prompt_tokens > 0) os << " (" << m.cached_prompt_tokens << " cached)";
    os << " (prefill " << prefill_s << " s" << std::setprecision(2);
    if (prefill_s > 0) os << ", " << new_prompt / prefill_s << " tok/s";
    os << "), generated " << m.generated_tokens << " tokens (decode "
       << std::setprecision(3) << decode_s << " s" << std::setprecision(2);
    if (decode_s > 0) os << ", " << m.generated_tokens / decode_s << " tok/s";
    os << "), kv " << kv_used << "/" << kv_max
       << ", finish " << stop_reason_to_string(m.stop_reason);
    header_print("FLM", os.str());
}

/// \brief Whether a request sets a field
/// \note Some clients send null for every option they leave unset, so null
///       counts as not set; json::value() and conversions throw on it instead.
bool has_value(const json& request, const char* key) {
    auto it = request.find(key);
    return it != request.end() && !it->is_null();
}

/// \brief A request field, or the fallback when it is missing or null
template <typename T>
T value_or(const json& request, const char* key, T fallback) {
    return has_value(request, key) ? request.at(key).get<T>() : fallback;
}
}

static bool starts_with_ci(const std::string& s, const char* prefix) {
    const size_t n = std::strlen(prefix);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i) {
        if (std::tolower(static_cast<unsigned char>(s[i])) != prefix[i]) return false;
    }
    return true;
}

///@brief Check one OpenAI `image_url` content part and return its base64 payload, ready for the models' decoders.
/// Accepts a base64 `data:image/...;base64,` URL (any image media type, any case; the bytes decide the format) or bare
/// base64. On failure returns false with `why` set: the request is malformed or the image can't be used, and silently
/// dropping it would let the model answer about an image it never saw.
static bool image_url_payload(const json& part, ImageReader& reader, std::string& payload, std::string& why) {
    if (!part.contains("image_url") || !part["image_url"].is_object()) {
        why = "'image_url' must be an object like {\"url\": \"data:image/png;base64,...\"}";
        return false;
    }
    const json& image_url = part["image_url"];
    if (!image_url.contains("url") || !image_url["url"].is_string()) {
        why = "'image_url.url' is missing or not a string";
        return false;
    }
    const std::string url = image_url["url"].get<std::string>();
    if (url.empty()) {
        why = "'image_url.url' is empty";
        return false;
    }
    if (starts_with_ci(url, "http://") || starts_with_ci(url, "https://")) {
        why = "remote image URLs are not supported; send the image inline as a base64 data URL "
              "(data:image/png;base64,...)";
        return false;
    }
    std::string encoded;
    if (starts_with_ci(url, "data:")) {
        const size_t comma = url.find(',');
        if (comma == std::string::npos) {
            why = "malformed data URL: no ',' before the image data";
            return false;
        }
        std::string header = url.substr(5, comma - 5);
        std::transform(header.begin(), header.end(), header.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const std::string media_type = header.substr(0, header.find(';'));
        if (media_type.rfind("image/", 0) != 0) {
            why = "data URL media type '" + media_type + "' is not an image";
            return false;
        }
        if (header.size() < 7 || header.compare(header.size() - 7, 7, ";base64") != 0) {
            why = "data URL must be base64-encoded (data:image/...;base64,...)";
            return false;
        }
        payload = url.substr(comma + 1);
    }
    else {
        payload = url;  // bare base64
    }
    if (payload.empty()) {
        why = "the image data is empty";
        return false;
    }
    try {
        encoded = base64::from_base64(payload);
    }
    catch (const std::exception&) {
        why = "the image data is not valid base64";
        return false;
    }
    return reader.validate_image_bytes(encoded, why);
}

///@brief `raw`: where to keep the batches of any `image_raw` parts (null: the model takes none, so such a part is an
/// error). A bad image or image part (and an image outside a user message) sets `error`, and the result must not be used
static json normalize_template(json messages, std::vector<std::shared_ptr<const raw_rgb8_batch_t>>* raw = nullptr,
                               std::string* error = nullptr, const raw_rgb8_limits_t& raw_limits = raw_rgb8_limits_t()) {
    json template_message = json::array();
    size_t raw_frames = 0;
    size_t raw_bytes = 0;
    ImageReader image_reader;
    auto fail = [&](size_t m, size_t c, const std::string& why) {
        if (error) *error = "messages[" + std::to_string(m) + "].content[" + std::to_string(c) + "]: " + why;
        return template_message;
    };

    for (size_t message_index = 0; message_index < messages.size(); ++message_index) {
        auto& message = messages[message_index];
        json new_message = message;
        std::string merged_text;
        nlohmann::ordered_json::array_t merged_images;
        nlohmann::ordered_json::array_t merged_audio;

        if (message["content"].is_string()) {
            // Simple format: just text
            merged_text = message["content"].get<std::string>();
        }
        else if (message["content"].is_array()) {
            // Structured format: extract text and image URLs
            for (size_t content_index = 0; content_index < message["content"].size(); ++content_index) {
                auto& contentItem = message["content"][content_index];
                const bool is_image = contentItem.is_object() && contentItem.contains("type") &&
                                      (contentItem["type"] == "image_url" || contentItem["type"] == "image_raw");
                if (is_image && message.value("role", "") != "user") {
                    // OpenAI takes image parts in user messages only; the chat templates drop or reject the rest.
                    return fail(message_index, content_index, "images are only supported in user messages, not in a message with role '" +
                                (message.contains("role") && message["role"].is_string()
                                     ? message["role"].get<std::string>() : std::string("?")) + "' message");
                }
                if (contentItem.contains("type") && contentItem["type"] == "text") {
                    merged_text += contentItem["text"].get<std::string>();
                }
                else if (contentItem.contains("type") && contentItem["type"] == "image_url") {
                    std::string payload, why;
                    if (!image_url_payload(contentItem, image_reader, payload, why)) {
                        header_print("Warning", "Rejecting image: " + why);
                        return fail(message_index, content_index, why);
                    }
                    merged_images.push_back(payload);
                }
                else if (contentItem.contains("type") && contentItem["type"] == "image_raw") {
                    std::string why;
                    auto batch = std::make_shared<raw_rgb8_batch_t>();
                    if (!raw) {
                        why = "image_raw: raw RGB8 images are not supported by this model";
                    }
                    else if (!contentItem.contains("image_raw")) {
                        why = "image_raw: the part has no 'image_raw' object";
                    }
                    else if (parse_image_raw(contentItem["image_raw"], raw_limits, raw_frames, raw_bytes, *batch, why)) {
                        why.clear();
                    }
                    if (!why.empty()) {
                        return fail(message_index, content_index, why);
                    }
                    raw_frames += (size_t)batch->frames;
                    raw_bytes += batch->data.size();
                    for (int f = 0; f < batch->frames; ++f) {
                        merged_images.push_back({ {"rgb8", {{"batch", raw->size()}, {"frame", f}}} });
                    }
                    raw->push_back(std::move(batch));
                }
                else if (contentItem.contains("type") && contentItem["type"] == "input_audio") {
                    std::string audio_base64 = contentItem["input_audio"]["data"].get<std::string>();
                    if (audio_base64.empty()) {
                        header_print("Warning", "Empty audio, skipping this audio.");
                        continue;
                    }
                    merged_audio.push_back(audio_base64);
                }
            }
        }

        //new_message["role"] = message["role"];
        new_message["content"] = merged_text;
        if (!merged_images.empty()) {
            new_message["images"] = merged_images;
        }
        if (!merged_audio.empty()) {
            new_message["audios"] = merged_audio;
        }
        

        template_message.push_back(new_message);
    }

    return template_message;
}


///@brief Try to parse a JSON value from a string, returning the original string on failure.
static json try_parse_json_value(const std::string& s) {
    try {
        return json::parse(s);
    }
    catch (...) {
        return json(s);
    }
}

///@brief Whether a streaming OpenAI request asked for usage through stream_options.include_usage.
/// Following OpenAI, a stream reports no usage unless asked; a malformed option counts as not asked.
static bool stream_include_usage(const json& request) {
    auto options = request.find("stream_options");
    if (options == request.end() || !options->is_object()) {
        return false;
    }
    auto include_usage = options->find("include_usage");
    return include_usage != options->end() && include_usage->is_boolean() && include_usage->get<bool>();
}

///@brief Read an OpenAI chat request's logprobs and top_logprobs.
///@param top_n set to the alternatives wanted per token, or -1 when logprobs is not asked for
///@param error set to the message for the client when the request is malformed
///@return false when the request is malformed
static bool logprobs_request_from_openai(const json& request, int& top_n, std::string& error) {
    top_n = -1;
    bool enabled = false;
    if (has_value(request, "logprobs")) {
        if (!request.at("logprobs").is_boolean()) {
            error = "logprobs must be a boolean";
            return false;
        }
        enabled = request.at("logprobs").get<bool>();
    }
    if (has_value(request, "top_logprobs")) {
        const json& n = request.at("top_logprobs");
        if (!n.is_number_integer() || n.get<int64_t>() < 0 || n.get<int64_t>() > 100) {
            error = "top_logprobs must be an integer between 0 and 100";
            return false;
        }
        if (!enabled) {
            error = "top_logprobs requires logprobs to be true";
            return false;
        }
        top_n = static_cast<int>(n.get<int64_t>());
    }
    else if (enabled) {
        top_n = 0;
    }
    return true;
}

///@brief The OpenAI request fields that steer sampling but are not plain numbers to set
struct openai_sampling_request {
    /// generated tokens the penalties look at, -1 the whole reply; applies to its own request, like the other sampling fields
    std::optional<int> penalty_window;
    /// restarts the random generator for this request only
    std::optional<uint64_t> seed;
};

///@brief Read an OpenAI request's repeat_last_n (an FLM extension) and seed.
///@param error set to the message for the client when the request is malformed
///@return false when the request is malformed
static bool sampling_request_from_openai(const json& request, openai_sampling_request& sampling, std::string& error) {
    if (has_value(request, "repeat_last_n")) {
        const json& n = request.at("repeat_last_n");
        if (!n.is_number_integer() || (!n.is_number_unsigned() && n.get<int64_t>() < -1)) {
            error = "repeat_last_n must be an integer: -1 for the whole reply, 0 to turn the penalties off, or a window size";
            return false;
        }
        // a window longer than any reply is the whole reply
        sampling.penalty_window = n.is_number_unsigned() && n.get<uint64_t>() > std::numeric_limits<int>::max()
            ? -1 : static_cast<int>(n.get<int64_t>());
    }
    if (has_value(request, "seed")) {
        const json& s = request.at("seed");
        if (!s.is_number_integer()) {
            error = "seed must be an integer";
            return false;
        }
        sampling.seed = s.is_number_unsigned() ? s.get<uint64_t>() : static_cast<uint64_t>(s.get<int64_t>());
    }
    return true;
}

///@brief Read an OpenAI chat request's continue_final_message (vLLM's extension) and check
///       that the request can continue its final message.
///@param messages the request's messages, as sent
///@param constraint the request's tool_choice and response_format
///@param continue_final set to whether the final assistant message is continued
///@param error set to the message for the client when the request is malformed
///@return false when the request is malformed
///@note add_generation_prompt is read only to refuse it alongside continue_final_message, as
///      vLLM does: the two ask for opposite things. On its own it is ignored, as before.
static bool continue_request_from_openai(const json& request, const json& messages,
                                         const chat_constraint_request& constraint,
                                         bool& continue_final, std::string& error) {
    continue_final = false;
    if (has_value(request, "continue_final_message")) {
        if (!request.at("continue_final_message").is_boolean()) {
            error = "continue_final_message must be a boolean";
            return false;
        }
        continue_final = request.at("continue_final_message").get<bool>();
    }
    if (!continue_final) {
        return true;
    }
    if (has_value(request, "add_generation_prompt") && request.at("add_generation_prompt") == true) {
        error = "Cannot set both continue_final_message and add_generation_prompt to true";
        return false;
    }
    const json* last = messages.empty() ? nullptr : &messages.back();
    if (last == nullptr || !last->is_object() || last->value("role", "") != "assistant") {
        error = "continue_final_message requires the last message to have role \"assistant\"";
        return false;
    }
    if (has_value(*last, "tool_calls") && !(last->at("tool_calls").is_array() && last->at("tool_calls").empty())) {
        error = "continue_final_message cannot continue an assistant message that has tool_calls";
        return false;
    }
    // the grammar of a forced call or a JSON answer starts at the beginning of a turn
    if (constraint.json_object || !constraint.json_schema.is_null() ||
        constraint.tool_choice == chat_tool_choice::REQUIRED || constraint.tool_choice == chat_tool_choice::FUNCTION) {
        error = "continue_final_message cannot be combined with response_format or a tool_choice that forces a call";
        return false;
    }
    return true;
}

///@brief Check the final message once normalize_template has flattened it: text to continue, and nothing else
///@return the message for the client, empty when the message can be continued
static std::string continue_final_message_error(const json& messages) {
    const json& last = messages.back();
    if (last.contains("images") || last.contains("audios")) {
        return "continue_final_message requires the final assistant message to be text only";
    }
    const std::string text = last.value("content", std::string());
    if (text.find_first_not_of(" \t\n\r\v\f") == std::string::npos) {
        return "continue_final_message requires text in the final assistant message";
    }
    return "";
}

static void apply_sampling_request(AutoModel& engine, const openai_sampling_request& sampling) {
    if (sampling.penalty_window) {
        engine.set_penalty_window(*sampling.penalty_window);
    }
    if (sampling.seed) {
        engine.set_seed(*sampling.seed);
    }
}

///@brief Convert OpenAI-style assistant tool_calls + following tool messages into the
/// Gemma4 chat-template format, which expects a single assistant message containing
/// both `tool_calls` and `tool_responses` (with `{name, response}` entries).
static json convert_tool_responses_gemma4(json messages) {
    json converted_messages = json::array();

    size_t i = 0;
    while (i < messages.size()) {
        const auto& msg = messages[i];

        if (msg.value("role", "") != "assistant" || !msg.contains("tool_calls") ||
            !msg.at("tool_calls").is_array() || msg.at("tool_calls").empty()) {
            converted_messages.push_back(msg);
            i++;
            continue;
        }

        // Normalize tool_calls: ensure function.arguments is a JSON object (not a string).
        json normalized_tool_calls = json::array();
        // Map tool_call_id -> function name, for matching tool responses below.
        std::unordered_map<std::string, std::string> id_to_name;
        // Preserve call order so unmatched tool responses can fall back positionally.
        std::vector<std::string> call_names_in_order;

        for (const auto& tc : msg.at("tool_calls")) {
            json new_tc = tc;
            std::string name;
            if (new_tc.contains("function") && new_tc["function"].is_object()) {
                auto& fn = new_tc["function"];
                if (fn.contains("name") && fn["name"].is_string()) {
                    name = fn["name"].get<std::string>();
                }
                if (fn.contains("arguments") && fn["arguments"].is_string()) {
                    fn["arguments"] = try_parse_json_value(fn["arguments"].get<std::string>());
                }
            }
            if (new_tc.contains("id") && new_tc["id"].is_string()) {
                id_to_name[new_tc["id"].get<std::string>()] = name;
            }
            call_names_in_order.push_back(name);
            normalized_tool_calls.push_back(new_tc);
        }

        // Collect consecutive following "tool" role messages as tool_responses.
        json tool_responses = json::array();
        size_t j = i + 1;
        size_t response_index = 0;
        while (j < messages.size() && messages[j].value("role", "") == "tool") {
            const auto& tool_msg = messages[j];

            // Resolve the function name: prefer matching by tool_call_id, then by
            // an explicit name field, then by positional order of the tool_calls.
            std::string name;
            if (tool_msg.contains("tool_call_id") && tool_msg["tool_call_id"].is_string()) {
                auto it = id_to_name.find(tool_msg["tool_call_id"].get<std::string>());
                if (it != id_to_name.end()) name = it->second;
            }
            if (name.empty() && tool_msg.contains("name") && tool_msg["name"].is_string()) {
                name = tool_msg["name"].get<std::string>();
            }
            if (name.empty() && response_index < call_names_in_order.size()) {
                name = call_names_in_order[response_index];
            }

            // Parse content into a JSON value if possible.
            json response_value;
            if (tool_msg.contains("content")) {
                const auto& content = tool_msg.at("content");
                if (content.is_string()) {
                    response_value = try_parse_json_value(content.get<std::string>());
                }
                else {
                    response_value = content;
                }
            }
            else {
                response_value = nullptr;
            }

            tool_responses.push_back({
                {"name", name},
                {"response", response_value},
            });

            response_index++;
            j++;
        }

        json merged = {
            {"role", "assistant"},
            {"tool_calls", normalized_tool_calls},
        };
        if (!tool_responses.empty()) {
            merged["tool_responses"] = tool_responses;
        }
        if (msg.contains("content") && !msg.at("content").is_null()) {
            merged["content"] = msg.at("content");
        }
        if (msg.contains("reasoning_content") && !msg.at("reasoning_content").is_null()) {
            merged["reasoning_content"] = msg.at("reasoning_content");
        }

        converted_messages.push_back(merged);
        i = j;
    }

    return converted_messages;
}

///@brief RestHandler constructor
///@param models the model list
///@param downloader the downloader
///@param default_tag the default tag
///@param asr whether to enable asr
///@param embed whether to enable embedding

///@return the rest handler
RestHandler::RestHandler(model_list& models, ModelDownloader& downloader, program_args_t& args,
                         flm_rt::device* npu_device)
    : npu_device_inst(npu_device), power_mode(args.power_mode), supported_models(models),
      downloader(downloader), default_model_tag(args.model_tag), modelscope(args.modelscope),
      asr(args.asr), img_pre_resize(args.img_pre_resize), preemption(args.preemption), backend(args.backend) {

    if (args.ctx_length != -1) {
        this->ctx_length = args.ctx_length >= 512 ? args.ctx_length : 512;
    } else {
        this->ctx_length = -1;
    }
    if (args.prefill_chunk_len != -1) {
        this->prefill_chunk_len_arg = args.prefill_chunk_len >= 512 ? args.prefill_chunk_len : 512;
    }
    else {
        this->prefill_chunk_len_arg = -1;
    }
    this->prefill_chunk_len = this->prefill_chunk_len_arg;
    this->prefix_cache_mb = args.prefix_cache_mb;
    this->prefix_slots = args.prefix_slots;
    this->prefix_min_gain = args.prefix_min_gain;
    this->raw_rgb8_limits.allow_path_dir = args.raw_path_dir;
    if (!args.raw_path_dir.empty()) {
        std::error_code raw_dir_ec;
        if (!std::filesystem::is_directory(args.raw_path_dir, raw_dir_ec)) {
            header_print("WARNING", "--allow-raw-path " << args.raw_path_dir << " is not a directory; raw frame files stay off");
            this->raw_rgb8_limits.allow_path_dir.clear();
        }
    }
    this->raw_rgb8_limits.max_frames = (size_t)std::max(args.max_raw_frames, 1);
    this->raw_rgb8_limits.max_bytes = (size_t)std::max(args.max_raw_mb, 1) * 1024 * 1024;

    default_model_tag = supported_models.rectify_model_tag(default_model_tag);
    const bool embedding_only = supported_models.is_model_supported(default_model_tag) &&
        is_embedding_model(supported_models.get_model_info(default_model_tag).second);
    if (embedding_only) ensure_embed_model_loaded(default_model_tag);
    // Initialize chat bot with default model
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    if (this->asr) {
        ensure_asr_model_loaded(args.asr_tag);
    }
#else
    if (this->asr) {
        header_print("Error", "ASR models are not supported in this build");
    }
#endif
    if (args.embed && !embedding_only) {
        ensure_embed_model_loaded(args.embed_tag);
    }

    if (!args.decision.empty()) {
        ensure_decision_model_loaded(args.decision);
    }

    if (default_model_tag != "model-faker" && !embedding_only) {
        if (!supported_models.is_model_supported(default_model_tag)) {
            header_print("Warning", "Default model tag '" << default_model_tag << "' is not supported. Falling back to 'llama3.2:1b'.");
            this->default_model_tag = "llama3.2:1b";
        }
        if (!ensure_model_loaded(default_model_tag)) {
            header_print("Error", "Failed to load default model: " + default_model_tag);
        }
    }
    else {
        this->current_model_tag = "model-faker";
    }
    this->prompt_cache = PromptCache();
}

///@brief RestHandler destructor
///@return the rest handler
RestHandler::~RestHandler() = default;

///@brief Ensure the model is loaded
///@param model_tag the model tag
bool RestHandler::ensure_model_loaded(const std::string& model_tag,
                                      const std::string& request_backend, std::string* refusal) {
    // Refuse a name the catalog does not have before anything resolves it;
    // an unknown name used to load a substitute model (and on dev04 could
    // end the process). Checked after rectify_model_tag, as get_model_info
    // does, so "Ollama/<tag>" and bare family names still resolve.
    // "model-faker" means no model was chosen.
    if (model_tag != "model-faker" &&
        !supported_models.is_model_supported(
            supported_models.rectify_model_tag(model_tag))) {
        header_print_r("ERROR", "Model tag '" << model_tag
            << "' is not supported. Please check with `flm list`.");
        return false;
    }

    // Nothing was chosen, so there is nothing to load; succeed only if a
    // model is in fact loaded.
    if (model_tag == "model-faker") {
        return auto_chat_engine != nullptr;
    }

    const auto& requested_backend = request_backend.empty() ? backend : request_backend;
    try {
        auto plan = resolve_model_load(supported_models.rectify_model_tag(model_tag), supported_models, requested_backend);
        if (auto_chat_engine && current_model_tag == plan.tag && loaded_selection.backend == plan.selection.backend &&
            loaded_selection.provider == plan.selection.provider) return true;

        auto* device = plan.prepare_device(npu_device_inst, power_mode);
        switch (downloader.is_model_downloaded(plan.tag, false, /*fast_check=*/true)) {
            case ModelDownloader::ModelStatus::Ready:
                break;
            case ModelDownloader::ModelStatus::Outdated:
            case ModelDownloader::ModelStatus::Missing:
                if (!downloader.pull_model(plan.tag, modelscope)) throw std::runtime_error("failed to download " + plan.tag);
                // The pull can select a different catalog source. Preflight its actual entry.
                plan = resolve_model_load(plan.tag, supported_models, requested_backend);
                device = plan.prepare_device(npu_device_inst, power_mode);
                break;
            case ModelDownloader::ModelStatus::Incompatible:
                throw std::runtime_error(plan.tag + " is incompatible with this version of FastFlowLM");
        }
        const auto path = supported_models.get_load_path(plan.tag);
        const int chunk = prefill_chunk_len_arg != -1 ? prefill_chunk_len_arg : plan.info.at("max_prefill_len").get<int>();

        // From here on the two engines must not overlap in accelerator memory.
        {
            std::lock_guard lock(model_info_mutex);
            current_model_tag = "model-faker";
            loaded_backend.clear();
        }
        auto_chat_engine.reset();
        prompt_cache.reset();
        auto engine = flm::frontend::FrontendRegistry::instance().create(plan.family, device);
        engine->set_server_mode(true);
        engine->configure_parameter("img_pre_resize", img_pre_resize);
        engine->set_prefix_limits(prefix_cache_mb, prefix_slots, prefix_min_gain);
        const auto load_started = std::chrono::steady_clock::now();
        engine->load_model(path, plan.info, ctx_length, preemption, plan.selection.backend);
        engine->save_request_settings();
        report_load_time(load_started);
        prefill_chunk_len = chunk;
        positional_tool_results = plan.info.value("positional_tool_results", false)
            && engine->backend_id() == flm::backend::kOgaHipepBackendId;
        {
            std::lock_guard lock(model_info_mutex);
            loaded_backend = engine->backend_id();
            current_model_tag = std::move(plan.tag);
        }
        auto_chat_engine = std::move(engine);
        loaded_selection = std::move(plan.selection);
        return true;
    } catch (const flm::backend::NpuUnavailable& e) {
        if (refusal) *refusal = flm::backend::kNoNpuDevice;
        header_print("ERROR", "Failed to load model: " + std::string(e.what()));
    } catch (const std::exception& e) {
        header_print("ERROR", "Failed to load model: " + std::string(e.what()));
    }
    return false;
}

///@brief Whether ensure_model_loaded's gate would admit the model tag
///@note The same tolerant form as that gate, so a handler can answer a name the
///      catalog does not have with a 404 instead of a failed load.
///@param model_tag the model tag
bool RestHandler::is_known_model(const std::string& model_tag) {
    return model_tag == "model-faker" ||
           supported_models.is_model_supported(supported_models.rectify_model_tag(model_tag));
}

///@brief Ensure the asr model is loaded
///@param model_tag the model tag
void RestHandler::ensure_asr_model_loaded(const std::string& model_tag) {
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    std::string ensure_tag = model_tag;
    const bool sidecar_r2t2 = this->supported_models.get_model_info(ensure_tag).second
        .value("details", json::object()).value("family", std::string()) == "r2t2";
    switch (downloader.is_model_downloaded(ensure_tag, false, /*fast_check=*/true)) {
        case ModelDownloader::ModelStatus::Ready:
            break;
        case ModelDownloader::ModelStatus::Outdated:
        case ModelDownloader::ModelStatus::Missing:
            downloader.pull_model(ensure_tag, modelscope);
            break;
        case ModelDownloader::ModelStatus::Incompatible:
            header_print("ERROR", "Whisper is incompatible with this version of FastFlowLM, skipping... ");
            this->asr = false;
            return;
    }
    if (sidecar_r2t2) {
        // r2t2 transcribes by itself: load it as a second engine beside the chat model
        try {
            auto [tag, engine] = get_auto_model(ensure_tag, supported_models, npu_device_inst, "", power_mode);
            auto [new_tag, info] = this->supported_models.get_model_info(tag);
            engine->set_server_mode(true);
            engine->load_model(this->supported_models.get_load_path(new_tag), info, -1, this->preemption, "");
            engine->save_request_settings();
            this->asr_sidecar_engine = std::move(engine);
            this->asr_sidecar_tag = tag;
        }
        catch (const std::exception& e) {
            header_print("ERROR", "Failed to load ASR model: " + std::string(e.what()));
            exit(EXIT_FAILURE);
        }
        return;
    }
    auto* device = utils::require_npu_device(npu_device_inst, power_mode);
    this->whisper_engine = std::make_unique<Whisper>(device);
    auto [new_ensure_tag, whisper_model_info] = this->supported_models.get_model_info(ensure_tag);
    std::string whisper_model_path = this->supported_models.get_model_path(new_ensure_tag);
    try {
        this->whisper_engine->load_model(whisper_model_path, whisper_model_info, this->preemption);
    }
    catch (const std::exception& e) {
        header_print("ERROR", "Failed to load ASR model: " + std::string(e.what()));
        exit(EXIT_FAILURE);
    }
#else
    throw std::runtime_error("ASR models are not supported in this build");
#endif
}

///@brief Load the --decision model (download it first if needed); exits when it cannot load,
///       like the other sidecars, so a server never runs without the model it was asked for
void RestHandler::ensure_decision_model_loaded(const std::string& model_tag) {
    if (!decisionModelTags.count(model_tag)) {
        std::string supported;
        for (const auto& tag : decisionModelTags) supported += (supported.empty() ? "" : ", ") + tag;
        header_print("ERROR", "Unknown decision model '" << model_tag << "' (supported: " << supported << ")");
        exit(EXIT_FAILURE);
    }
    std::string ensure_tag = model_tag;
    switch (this->downloader.is_model_downloaded(ensure_tag, false, /*fast_check=*/true)) {
        case ModelDownloader::ModelStatus::Ready:
            break;
        case ModelDownloader::ModelStatus::Outdated:
        case ModelDownloader::ModelStatus::Missing:
            this->downloader.pull_model(ensure_tag, this->modelscope);
            break;
        case ModelDownloader::ModelStatus::Incompatible:
            header_print("ERROR", model_tag << " is incompatible with this version of FastFlowLM");
            exit(EXIT_FAILURE);
    }
    auto [resolved_tag, model_info] = this->supported_models.get_model_info(ensure_tag);
    std::string model_path = this->supported_models.get_model_path(resolved_tag);
    try {
        auto* device = utils::require_npu_device(npu_device_inst, power_mode);
        this->decision_engine = std::make_unique<AutoDecisionModel>(device, resolved_tag);
        this->decision_engine->load_model(model_path, model_info, this->preemption);
        this->decision_engine->warmup();
    }
    catch (const std::exception& e) {
        header_print("ERROR", "Failed to load decision model: " + std::string(e.what()));
        exit(EXIT_FAILURE);
    }
}

///@brief GET /v1/systemone/models
void RestHandler::handle_systemone_models(std::function<void(const json&)> send_response) {
    json data = json::array();
    if (this->decision_engine) {
        // what upstream d3_server's /v1/models states: the input limit, the modalities and the image budget
        json entry = {{"id", this->decision_engine->model_tag()}, {"object", "model"}, {"api", "systemone"},
                      {"max_input_tokens", this->decision_engine->max_input_tokens()},
                      {"modalities", this->decision_engine->modalities()},
                      {"image_max_pixels", flm::d3::IMAGE_MAX_PIXELS}};
        if (const auto video = this->decision_engine->video_limits(); !video.is_null()) entry["video"] = video;
        if (this->decision_engine->composed_requests()) {
            // d3:turbo: what its runtime declares (decision_config.json systemone) and this build answers
            entry["question_types"] = this->decision_engine->question_types();
            entry["assets"] = true;
            entry["long_context"] = false;
        }
        data.push_back(std::move(entry));
    }
    send_response({{"object", "list"}, {"data", data}});
}

///@brief POST /v1/systemone (and /v1/decisions)
///@param raw_body the request body, parsed here with Python json semantics
///@param send_response the response callback
void RestHandler::handle_systemone(const std::string& raw_body,
                                   std::function<void(const json&)> send_response) {
    json reply;
    if (!this->decision_engine) {
        reply["__http_status"] = 404;
        reply["__http_body"] = {{"detail", "no decision model is loaded; start the server with --decision d3:edge, d3:lite, d3:nano, d3:mini, d3:flash, d3:turbo or d3:ultra"}};
        send_response(reply);
        return;
    }
    try {
        DecisionReply r = this->decision_engine->system_one(raw_body);
        reply["__http_status"] = r.status;
        reply["__http_body"] = std::move(r.body);
    }
    catch (const std::exception& e) {
        header_print("ERROR", "System One request failed: " << e.what());
        reply["__http_status"] = 500;
        reply["__http_body"] = {{"detail", std::string("the server failed on this request (") + e.what() + ")"}};
    }
    send_response(reply);
}

///@brief Configure chat engine parameters from options and request
///@param options the options JSON object
///@param request the request JSON object
void RestHandler::configure_chat_engine_parameters(const json& options, const json& request,
                                                   std::optional<bool> nested_think) {
    auto_chat_engine->restore_request_settings();
    if (has_value(request, "temperature")) {
        float temperature = request["temperature"];
        auto_chat_engine->set_temperature(temperature);
    }
    if (has_value(request, "top_p")) {
        float top_p = request["top_p"];
        auto_chat_engine->set_topp(top_p);
    }
    if (has_value(request, "top_k")) {
        int top_k = request["top_k"];
        auto_chat_engine->set_topk(top_k);
    }
    if (has_value(request, "min_p")) {
        float min_p = request["min_p"];
        auto_chat_engine->set_minp(min_p);
    }
    if (has_value(request, "presence_penalty")) {
        float presence_penalty = request["presence_penalty"];
        auto_chat_engine->set_presence_penalty(presence_penalty);
    }
    if (has_value(request, "frequency_penalty")) {
        float frequency_penalty = request["frequency_penalty"];
        auto_chat_engine->set_frequency_penalty(frequency_penalty);
    }
    if (has_value(request, "repetition_penalty")) {
        float repetition_penalty = request["repetition_penalty"];
        auto_chat_engine->set_repetition_penalty(repetition_penalty);
    }
    if (has_value(request, "think")) {
        bool enable_thinking = request["think"];
        auto_chat_engine->configure_parameter("enable_think", enable_thinking);
    }
    if (has_value(request, "reasoning_effort")) {
        std::string reasoning_effort = request["reasoning_effort"];
        auto_chat_engine->configure_parameter("reasoning_effort", reasoning_effort);
    }
    // chat_template_kwargs.enable_thinking, already checked against the two above
    if (nested_think.has_value()) {
        auto_chat_engine->configure_parameter("enable_think", *nested_think);
    }
    if (has_value(request, "conversational_mode")) {
        bool conversational_mode = request["conversational_mode"];
        auto_chat_engine->configure_parameter("conversational_mode", conversational_mode);
    }

    if (has_value(request, "image-max-tokens")) {
        int image_max_tokens = request["image-max-tokens"];
        auto_chat_engine->configure_parameter("image_max_tokens", image_max_tokens);
    }
}

///@brief Handle the show request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_show(const json& request,
    std::function<void(const json&)> send_response,
    StreamResponseCallback send_streaming_response) {
    try {
        // A missing key read through const operator[] is undefined behavior (a crash with
        // ordered_json): check the required fields first and answer 400.
        if (auto bad = flm::http_guard::validate_show(request)) {
            send_response(*bad);
            return;
        }
        std::string model = request.at("model").get<std::string>();
        json info = {
            {"modelfile", ""},
            {"parameters", ""},
            {"template", ""},
            {"details", {
                {"parent_model", ""},
                {"format", ""},
                {"family", ""},
                {"families", {""}},
                {"parameter_size", ""},
                {"quantization_level", ""}
                }
            },
            {"model_info", {
                {"general.architecture", "flm" }
            }},
            {"capabilities", {"chat", "vision", "completion"}}
        };


        send_response(info);
    }
    catch (const std::exception& e) {
        json error_response = { {"error", e.what()} };
        send_response(error_response);
    }
}

///@brief Handle the generate request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_generate(const json& request,
                                 std::function<void(const json&)> send_response,
                                 StreamResponseCallback send_streaming_response,
                                 std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        if (auto bad = flm::http_guard::validate_generate(request)) {
            send_response(*bad);
            return;
        }
        std::string prompt = request.at("prompt").get<std::string>();
        bool stream = request.value("stream", true);
        std::string model = request.value("model", current_model_tag);
        const std::string request_backend = request.value("backend", std::string());
        json options = request.value("options", json::object());
       
        int length_limit = request.value("max_tokens", 4096);
        auto load_start_time = time_utils::now();
        // TODO: Use Another Check Function avoid loading again
        std::string refusal;
        if (!ensure_model_loaded(model, request_backend, &refusal)) {
            json error_response = {{"error", refusal.empty() ? "Failed to load " + model + " model!" : refusal}};
            send_response(error_response);
            return;
        }
        auto load_end_time = time_utils::now();
        // It takes no settings, but must not run with what an earlier chat request set
        auto_chat_engine->restore_request_settings();
      
        chat_meta_info_t meta_info;
        lm_uniform_input_t uniformed_input;
        uniformed_input.requested_max_new_tokens =
            normalize_requested_max_new_tokens(
                request.contains("max_tokens")
                    ? std::optional<int>(request.at("max_tokens").get<int>())
                    : std::nullopt);
        meta_info.max_prefill_len = this->prefill_chunk_len;
        meta_info.ignore_eos = value_or(request, "ignore_eos", false);
        meta_info.min_tokens = value_or(request, "min_tokens", 0);
        meta_info.load_duration = (uint64_t)time_utils::duration_ns(load_start_time, load_end_time).first;
        header_print("FLM", "Start generating...");
        
        if (stream) {
            // Streaming response using streaming_ostream
            auto total_start_time = time_utils::now();
            streaming_ostream ostream(model, send_streaming_response, false);
            uniformed_input.prompt = prompt;
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input, [&] { return cancellation_token && cancellation_token->cancelled(); });
                if (!success){
                    json error_response = {{"error", "Max length reached"}};
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream, [&] { return cancellation_token && cancellation_token->cancelled(); });
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            auto total_end_time = time_utils::now();
            auto history = this->auto_chat_engine->get_history();
            // std::cout << "history: " << history.first << std::endl;
            meta_info.total_duration = (uint64_t)time_utils::duration_ns(total_start_time, total_end_time).first;
            ostream.finalize_generate(meta_info, history.second);
            log_request_summary(meta_info, this->auto_chat_engine->get_current_context_length(), this->auto_chat_engine->get_max_length());
        } else {
            // Non-streaming response
            std::stringstream ss;
            wstream_buf obuf(ss);
            std::ostream ostream(&obuf);
            uniformed_input.prompt = prompt;
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input, [&] { return cancellation_token && cancellation_token->cancelled(); });
                if (!success){
                    json error_response = {{"error", "Max length reached"}};
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream, [&] { return cancellation_token && cancellation_token->cancelled(); });
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            std::string response_text = ss.str();
            auto history = this->auto_chat_engine->get_history();
            json response = {
                {"model", model},
                {"response", response_text},
                {"context", history.second},
                {"done", true},
                {"prompt_eval_count", meta_info.prompt_tokens},
                {"eval_count", meta_info.generated_tokens},
                {"total_duration", meta_info.total_duration},
                {"load_duration", meta_info.load_duration},
                {"prompt_eval_duration", meta_info.prefill_duration},
                {"eval_duration", meta_info.decoding_duration},
                {"done_reason", stop_reason_to_string(meta_info.stop_reason)}
            };
            // std::cout << "history: " << history.first << std::endl;
            log_request_summary(meta_info, this->auto_chat_engine->get_current_context_length(), this->auto_chat_engine->get_max_length());
            send_response(response);
        }
    } catch (const std::exception& e) {
        json error_response = ExceptionResponse(e);
        send_response(error_response);
    }
}

///@brief Handle the chat request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_chat(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response,
                             std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        if (auto bad = flm::http_guard::validate_chat(request)) {
            send_response(*bad);
            return;
        }
        nlohmann::ordered_json messages = request.at("messages");
        bool stream = request.value("stream", false);
        std::string model = request.value("model", current_model_tag);
        const std::string request_backend = request.value("backend", std::string());
        json options = request.value("options", json::object());
        int length_limit = options.value("num_predict", 4096);

        auto load_start_time = time_utils::now();
        std::string refusal;
        if (!ensure_model_loaded(model, request_backend, &refusal)) {
            json error_response = {{"error", refusal.empty() ? "Failed to load " + model + " model!" : refusal}};
            send_response(error_response);
            return;
        }
        auto load_end_time = time_utils::now();
       
        configure_chat_engine_parameters(options, request);

        // messages = normalize_messages(messages);
        
        chat_meta_info_t meta_info;
        lm_uniform_input_t uniformed_input;
        uniformed_input.requested_max_new_tokens =
            normalize_requested_max_new_tokens(
                options.contains("num_predict")
                    ? std::optional<int>(options.at("num_predict").get<int>())
                    : std::nullopt);
        meta_info.load_duration = (uint64_t)time_utils::duration_ns(load_start_time, load_end_time).first;
        meta_info.max_prefill_len = this->prefill_chunk_len;
        meta_info.ignore_eos = value_or(request, "ignore_eos", false);
        meta_info.min_tokens = value_or(request, "min_tokens", 0);
        header_print("FLM", "Start generating...");
        if (stream) {
            // Streaming response using streaming_ostream
            auto total_start_time = time_utils::now();
            streaming_ostream ostream(model, send_streaming_response, true);  // true for chat format
            uniformed_input.messages = messages;
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input, [&] { return cancellation_token && cancellation_token->cancelled(); });
                if (!success){
                    json error_response = {{"error", "Max length reached"}};
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream,
                    [&] { return cancellation_token && cancellation_token->cancelled(); });
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            auto total_end_time = time_utils::now();
            meta_info.total_duration = (uint64_t)time_utils::duration_ns(total_start_time, total_end_time).first;
            
            ostream.finalize_chat(meta_info);
            log_request_summary(meta_info, this->auto_chat_engine->get_current_context_length(), this->auto_chat_engine->get_max_length());
            // auto history = this->chat_engine->get_history();
            // std::cout << "history: " << history.first << std::endl;
            this->auto_chat_engine->clear_context();
        } else {
            // Non-streaming response
            uniformed_input.messages = messages;
            auto total_start_time = time_utils::now();
            nullstream nstream;
            //std::string response_text = auto_chat_engine->generate_with_prompt(meta_info, uniformed_input, length_limit, std::cout);
            std::string response_text;
            try {
                response_text = auto_chat_engine->generate_with_prompt(
                    meta_info, uniformed_input, length_limit, nstream,
                    [&] { return cancellation_token && cancellation_token->cancelled(); });
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            // A character the generation stopped partway through can never be finished
            response_text = sanitize_utf8(response_text);
            //std::string response_text = chat_engine->generate_with_prompt(meta_info, prompts, length_limit, std::cout, payload);
            auto total_end_time = time_utils::now();
            meta_info.total_duration = (uint64_t)time_utils::duration_ns(total_start_time, total_end_time).first;
            
            json response = {
                {"model", model},
                {"message", {
                    {"role", "assistant"},
                    {"content", response_text},
                    {"images", nullptr}
                }},
                {"done", true},
                {"prompt_eval_count", meta_info.prompt_tokens},
                {"eval_count", meta_info.generated_tokens},
                {"total_duration", meta_info.total_duration},
                {"load_duration", meta_info.load_duration},
                {"prompt_eval_duration", meta_info.prefill_duration},
                {"eval_duration", meta_info.decoding_duration},
                {"done_reason", stop_reason_to_string(meta_info.stop_reason)}
            };
            log_request_summary(meta_info, this->auto_chat_engine->get_current_context_length(), this->auto_chat_engine->get_max_length());
            send_response(response);

            // auto history = this->chat_engine->get_history();
            // std::cout << "history: " << history.first << std::endl;
            this->auto_chat_engine->clear_context();
        }
    } catch (const std::exception& e) {
        json error_response = ExceptionResponse(e);
        send_response(error_response);
    }
}

///@brief Handle the models request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_models(const json& request,
                               std::function<void(const json&)> send_response,
                               StreamResponseCallback send_streaming_response) {
    try {
        json models = supported_models.get_all_models_ollama();
        send_response(models);
    } catch (const std::exception& e) {
        json error_response = ExceptionResponse(e);
        send_response(error_response);
    }
}

///@brief Handle the version request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_version(const json& request,
                                std::function<void(const json&)> send_response,
                                StreamResponseCallback send_streaming_response) {
    std::string version = __FLM_VERSION__;
    
    json response = {{"version", version}};
    send_response(response);
}

///@brief Handle the models request (open ai)
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_models_openai(const json& request,
    std::function<void(const json&)> send_response,
    StreamResponseCallback send_streaming_response) {
    try {
        send_response(supported_models.get_all_models_openai());
    }
    catch (const std::exception& e) {
        json error_response = { {"error", e.what()} };
        send_response(error_response);
    }
}

///@brief Handle the ps request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_ps(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    try {
        // Generate expires_at timestamp (1 hour from now)
        auto now = std::chrono::system_clock::now();
        auto expires_time = now + std::chrono::hours(1);
        auto expires_time_t = std::chrono::system_clock::to_time_t(expires_time);
        auto expires_tp = std::chrono::system_clock::from_time_t(expires_time_t);
        auto fractional_seconds = std::chrono::duration_cast<std::chrono::microseconds>(expires_time - expires_tp).count();
        
        // Get local time and timezone offset
        std::tm* local_tm = std::localtime(&expires_time_t);
        std::tm* utc_tm = std::gmtime(&expires_time_t);
        
        // Calculate timezone offset in minutes
        int offset_minutes = (local_tm->tm_hour - utc_tm->tm_hour) * 60 + (local_tm->tm_min - utc_tm->tm_min);
        if (local_tm->tm_mday != utc_tm->tm_mday) {
            offset_minutes += (local_tm->tm_mday > utc_tm->tm_mday) ? 1440 : -1440;
        }
        
        std::stringstream expires_ss;
        expires_ss.imbue(std::locale::classic()); // Use C locale to avoid commas
        expires_ss << std::put_time(local_tm, "%Y-%m-%dT%H:%M:%S");
        expires_ss << "." << std::setfill('0') << std::setw(5) << (fractional_seconds / 40); // 5 decimal places
        
        // Format timezone offset
        int offset_hours = offset_minutes / 60;
        int offset_mins = abs(offset_minutes % 60);
        expires_ss << (offset_minutes >= 0 ? "+" : "-") 
                  << std::setfill('0') << std::setw(2) << abs(offset_hours)
                  << ":" << std::setfill('0') << std::setw(2) << offset_mins;
        
        std::string expires_at = expires_ss.str();
        
        std::string tag;
        std::string backend_id;
        {
            std::lock_guard lock(model_info_mutex);
            tag = current_model_tag;
            backend_id = loaded_backend;
        }
        auto info = supported_models.get_model_info(tag).second;
        auto& details = info["details"];
        if (!backend_id.empty()) {
            details["backend"] = backend_id;
            if (const auto device = flm::backend::device_for_backend(backend_id); !device.empty()) {
                details["device"] = device;
            }
        }
        json response = {
            {"models", json::array({
                {
                    {"name", tag},
                    {"model", tag},
                    {"size", info["size"]},
                    {"details", details},
                    {"expires_at", expires_at},
                }
            })}
        };
        // std::cout << "response: " << response.dump(4) << std::endl;
        send_response(response);
    } catch (const std::exception& e) {
        json error_response = ExceptionResponse(e);
        send_response(error_response);
    }
}

///@brief Handle the pull request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_pull(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Pull operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the push request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_push(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Push operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the delete request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_delete(const json& request,
                               std::function<void(const json&)> send_response,
                               StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Delete operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the copy request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_copy(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Copy operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the create request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_create(const json& request,
                               std::function<void(const json&)> send_response,
                               StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Create operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the openai chat completion request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_openai_chat_completion(const json& request,
                                               std::function<void(const json&)> send_response,
                                               StreamResponseCallback send_streaming_response,
                                               std::shared_ptr<CancellationToken> cancellation_token) {
    openai_chat_sink sink(send_response, send_streaming_response, stream_include_usage(request));
    run_chat_turn(request, sink, cancellation_token);
}

///@brief Run one chat turn and report it to a sink
///@note The one place a chat request is validated, the model loaded and configured, the prompt
///      inserted and the answer generated. What the client gets, and in which protocol's shape,
///      is the sink's: this function names no response field.
///@param request the request, in the Chat Completions shape
///@param sink receives the turn: its start, its deltas when streamed, its end, or an error
///@param cancellation_token set when the client goes away
void RestHandler::run_chat_turn(const json& request, chat_sink& sink,
                                std::shared_ptr<CancellationToken> cancellation_token) {
    auto bad_request = [&sink](const std::string& message, const std::string& param = std::string()) {
        sink.on_error({400, "invalid_request_error", message, param, std::nullopt});
    };
    try {
        // Extract OpenAI-style parameters
        if (!has_value(request, "messages") || !request.at("messages").is_array()) {
            bad_request("'messages' is required and must be an array");
            return;
        }
        json current_messages = request.at("messages");
        std::string model = value_or(request, "model", current_model_tag);
        const std::string request_backend = value_or(request, "backend", std::string());
        bool stream = value_or(request, "stream", false);
        int length_limit = value_or(request, "max_tokens", value_or(request, "max_completion_tokens", 4096));
        json tools = value_or(request, "tools", json::array());
        json options = value_or(request, "options", json::object());

        // tool_choice, parallel_tool_calls and response_format become a decoding constraint
        chat_constraint_request constraint_request;
        std::string request_error;
        if (!chat_constraint_request_from_openai(request, tools, constraint_request, request_error)) {
            bad_request(request_error);
            return;
        }
        if (constraint_request.tool_choice == chat_tool_choice::NONE) {
            // the model is not shown the tools at all: banning the call opener alone
            // leaves it writing the call's JSON as plain text
            tools = json::array();
        }
        int logprobs_top_n = -1;
        if (!logprobs_request_from_openai(request, logprobs_top_n, request_error)) {
            bad_request(request_error);
            return;
        }
        openai_sampling_request sampling;
        if (!sampling_request_from_openai(request, sampling, request_error)) {
            bad_request(request_error);
            return;
        }
        bool continue_final = false;
        if (!continue_request_from_openai(request, current_messages, constraint_request, continue_final, request_error)) {
            bad_request(request_error);
            return;
        }

        if (!is_known_model(model)) {
            sink.on_error(unknown_model_error(model));
            return;
        }
        auto load_start_time = time_utils::now();
        std::string refusal;
        if (!ensure_model_loaded(model, request_backend, &refusal)) {
            sink.on_error(model_load_failed_error(model, refusal));
            return;
        }
        auto load_end_time = time_utils::now();
        if (logprobs_top_n >= 0 && !auto_chat_engine->supports_logprobs()) {
            bad_request("logprobs is not supported by " + model);
            return;
        }
        if (continue_final && (auto_chat_engine->single_turn || !auto_chat_engine->supports_continue_final_message())) {
            bad_request("continue_final_message is not supported by " + model);
            return;
        }

        // The Qwen spelling of think, honored by the Qwen3.5 package only; any other model
        // keeps ignoring it. Refused here, before anything streams.
        std::optional<bool> nested_think;
        if (dynamic_cast<Qwen3_5VL*>(auto_chat_engine.get()) != nullptr) {
            if (!chat_thinking_override_from_openai(request, nested_think, request_error)) {
                bad_request(request_error);
                return;
            }
            if (nested_think.has_value() && !*nested_think) {
                constraint_request.allow_reasoning = false;
            }
        }

        configure_chat_engine_parameters(options, request, nested_think);
        apply_sampling_request(*auto_chat_engine, sampling);

        std::vector<size_t> original_indices;
        try {
            current_messages = flm::chat::normalize_messages(current_messages, auto_chat_engine->message_roles(),
                                                              positional_tool_results ? &original_indices : nullptr);
        } catch (const flm::chat::message_error& e) {
            bad_request(e.what());
            return;
        }
        std::vector<std::shared_ptr<const raw_rgb8_batch_t>> raw_rgb8;
        {
            // raw RGB8 frames only go to MiniCPM-V; any other model gets a clear 400 rather than a dropped image
            const bool takes_raw = dynamic_cast<MiniCPM_V*>(auto_chat_engine.get()) != nullptr;
            std::string raw_error;
            current_messages = normalize_template(current_messages, takes_raw ? &raw_rgb8 : nullptr, &raw_error, this->raw_rgb8_limits);
            if (!raw_error.empty()) {
                bad_request(raw_error);
                return;
            }
        }
        if (positional_tool_results) {
            if (auto error = flm::chat::order_parallel_tool_results(current_messages, original_indices)) {
                bad_request(error->message, error->param);
                return;
            }
        }
        if (continue_final) {
            request_error = continue_final_message_error(current_messages);
            if (!request_error.empty()) {
                bad_request(request_error);
                return;
            }
        }

        // see if we can use prompt cache
        chat_meta_info_t meta_info;
        bool can_use_prompt_cache = false;
        if (auto_chat_engine->single_turn) {
            // reject multi-turn requests: any assistant message in the history
            // means the caller is trying to continue a prior turn, which this
            // model cannot support.
            bool has_assistant_msg = false;
            for (const auto& msg : current_messages) {
                if (msg.value("role", "") == "assistant") {
                    has_assistant_msg = true;
                    break;
                }
            }
            if (has_assistant_msg) {
                header_print("Warning", "Single-turn model received multi-turn request (assistant messages detected). Rejecting.");
                bad_request("This model only supports single-turn requests. Multi-turn conversation history (assistant messages) is not allowed.");
                return;
            }
        }
        else {
            // A model switch needs no case of its own: ensure_model_loaded
            // resets the cache whenever it replaces the engine, so the first
            // request on a new engine is a miss.
            cache_match_info_t cache_info;
            const bool engine_manages_cache = auto_chat_engine->manages_prefix_cache();
            if (engine_manages_cache) {
                // The model matches the rendered tokens against the prefixes it
                // saved and prefills the rest; a message-level miss must not
                // clear the context it would resume from. It logs what it reused.
                can_use_prompt_cache = true;
                meta_info.restore_allowed = true;
            }
            else {
                can_use_prompt_cache = prompt_cache.can_use_cache(current_messages, auto_chat_engine->get_chat_template_type(), tools, cache_info);
            }
            if (engine_manages_cache) {
                // nothing to clear
            }
            else if (can_use_prompt_cache) {
                meta_info.restore_allowed = true;
                header_print("FLM", "Use cached prompt!");
                size_t matched_rounds = cache_info.matched_rounds + (auto_chat_engine->check_using_checkpint() ? 0 : 1);
                header_print("FLM", "Matched " + std::to_string(matched_rounds) +
                    " out of " + std::to_string(cache_info.total_rounds) + " messages (" +
                    std::to_string(cache_info.total_rounds - matched_rounds) + " new to prefill).");
            }
            else {
                // cannot use cache, clear and re-insert all
                header_print("FLM", "Prompt cache miss.");
                header_print("FLM", "Clearing context...");
                auto_chat_engine->clear_context();
            }
        }

        if (model.starts_with("gemma4-it")) {
            current_messages = convert_tool_responses_gemma4(current_messages);
        }

        // std::cout << "FLM current_messages: \n" << current_messages.dump(4) << std::endl;

        lm_uniform_input_t uniformed_input;
        uniformed_input.messages = current_messages;
        uniformed_input.raw_rgb8 = std::move(raw_rgb8);
        uniformed_input.tools = tools;
        const std::optional<int> openai_chat_budget = has_value(request, "max_tokens")
            ? std::optional<int>(request.at("max_tokens").get<int>())
            : has_value(request, "max_completion_tokens")
                ? std::optional<int>(request.at("max_completion_tokens").get<int>())
                : std::nullopt;
        uniformed_input.requested_max_new_tokens =
            normalize_requested_max_new_tokens(openai_chat_budget);
        meta_info.load_duration = (uint64_t)time_utils::duration_ns(load_start_time, load_end_time).first;
        meta_info.max_prefill_len = this->prefill_chunk_len;
        meta_info.ignore_eos = value_or(request, "ignore_eos", false);
        meta_info.min_tokens = value_or(request, "min_tokens", 0);
        meta_info.logprobs_top_n = logprobs_top_n;
        meta_info.continue_final_message = continue_final;
        try {
            meta_info.constraint = auto_chat_engine->make_constraint(constraint_request);
        } catch (const std::invalid_argument& e) {
            // the request asks for something this model's format cannot do, or a bad schema
            bad_request(e.what());
            return;
        }
        // tool markup is only a tool call when the request offered tools
        chat_parse_options parse_options;
        parse_options.tools = &tools;
        parse_options.parse_tool_calls = !tools.empty() && constraint_request.tool_choice != chat_tool_choice::NONE;
        parse_options.keep_whitespace = continue_final;
        // a model that reasons first reads the continued message as its reasoning
        parse_options.reasoning_may_close = continue_final && auto_chat_engine->thinking_enabled();

        const chat_format_spec& spec = auto_chat_engine->get_chat_format();
        // where the output starts, once insert() has left the prompt's tail in meta_info
        auto output_prefill = [&] {
            return chat_format_prefill(spec, meta_info.prompt_tail, meta_info.continue_final_message,
                                       parse_options.parse_tool_calls);
        };
        // the end of the turn as the sink is told it; reads the engine, so call it when the turn is over
        auto turn_end = [&](const chat_msg& message, bool generated) {
            // Follow meta_info for the finish reason. A tool call is only visible after
            // parsing the generated text, so promote the stop reason here, and only on an
            // otherwise clean stop: every other case (length, cancel, error) takes
            // precedence, since a truncated or aborted generation must not be reported
            // to the client as a complete tool call.
            if (!message.tool_calls.empty() && meta_info.stop_reason == stop_reason_t::EOT_DETECTED) {
                meta_info.stop_reason = stop_reason_t::TOOL_DETECTED;
            }
            chat_turn_end end;
            end.finish = chat_finish_from(meta_info.stop_reason);
            end.generated = generated;
            end.message = &message;
            end.prompt_tokens = meta_info.prompt_tokens;
            end.cached_prompt_tokens = meta_info.cached_prompt_tokens;
            end.generated_tokens = meta_info.generated_tokens;
            end.load_duration = meta_info.load_duration;
            end.prefill_duration = meta_info.prefill_duration;
            end.decoding_duration = meta_info.decoding_duration;
            if (generated || stream) {
                // a turn cancelled before it began reports its usage only on a stream
                end.context_tokens = this->auto_chat_engine->get_current_context_length();
                end.context_capacity = this->auto_chat_engine->get_max_length();
            }
            end.prefix_cache = meta_info.prefix_cache.valid ? &meta_info.prefix_cache : nullptr;
            return end;
        };
        if (stream){
            cancellation_token->reset();
            sink.on_begin({model, true});
            // declared first, so it outlives the stream whose flush hook reads it
            std::unique_ptr<chat_logprobs_aligner> logprobs;
            chat_turn_ostream ostream(spec, parse_options, sink);  // parses the text, the sink gets the deltas

            header_print("FLM", "Start prefill...");
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input, [&] { return cancellation_token->cancelled(); });
                if (!success) {
                    if (meta_info.stop_reason == CANCEL_DETECTED || cancellation_token->cancelled()) {
                        meta_info.stop_reason = CANCEL_DETECTED;
                        header_print("❌ ", "Prefill Cancelled!");
                        sink.on_end(turn_end(ostream.finish(), false));
                        this->auto_chat_engine->clear_context();
                        this->prompt_cache.reset();
                        return;
                    }

                    sink.on_error(prefill_failed_error(meta_info));
                    this->auto_chat_engine->clear_context();
                    this->prompt_cache.reset();
                    return;
                }
            } catch (const std::exception& e) {
                sink.on_error(exception_error(e));
                this->auto_chat_engine->clear_context();
                this->prompt_cache.reset();
                return;
            }
            chat_turn_start start;
            start.prompt_tokens = meta_info.prompt_tokens;
            start.cached_prompt_tokens = meta_info.cached_prompt_tokens;
            if (meta_info.logprobs_top_n >= 0) {
                // each flush is placed before the parser sees it, so a content delta finds its tokens
                logprobs = std::make_unique<chat_logprobs_aligner>(auto_chat_engine.get(), meta_info.logprobs);
                start.content_logprobs = [&logprobs](const std::string& content) { return logprobs->take_content(content); };
                ostream.start(output_prefill(), [&logprobs](const std::string& written) { logprobs->on_flush(written); });
            }
            else {
                ostream.start(output_prefill());
            }
            sink.on_start(start);
            header_print("FLM", "Start generating...");
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream, [&] { return cancellation_token->cancelled(); });
            } catch (const ModelRequestError& error) {
                sink.on_error(model_error(error));
                if (error.session_cleared()) this->prompt_cache.reset();
                return;
            } catch (const std::exception& e) {
                sink.on_error(exception_error(e));
                this->auto_chat_engine->clear_context();
                this->prompt_cache.reset();
                return;
            }
            if (meta_info.stop_reason == CANCEL_DETECTED) {
                header_print("❌ ", "Generation Cancelled!");
                this->prompt_cache.reset();
            }

            try {
                sink.on_end(turn_end(ostream.finish(), true));
            } catch (const std::exception& e) {
                header_print("ERROR", std::string("Finishing the stream failed: ") + e.what());
                sink.on_error(exception_error(e));
            }
            log_request_summary(meta_info, this->auto_chat_engine->get_current_context_length(), this->auto_chat_engine->get_max_length());
        }
        else {
            sink.on_begin({model, false});
            nullstream nstream;
            // with logprobs the text is still dropped, but each token is placed as it is written
            std::unique_ptr<chat_logprobs_ostream> lstream;
            if (meta_info.logprobs_top_n >= 0) {
                lstream = std::make_unique<chat_logprobs_ostream>(auto_chat_engine.get(), meta_info.logprobs);
            }
            std::ostream& out = lstream ? static_cast<std::ostream&>(*lstream) : nstream;
            std::string response_text;
            header_print("FLM", "Start prefill...");
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input, [&] { return cancellation_token->cancelled(); });
                if (!success) {
                    if (meta_info.stop_reason == CANCEL_DETECTED || cancellation_token->cancelled()) {
                        meta_info.stop_reason = CANCEL_DETECTED;
                        header_print("❌ ", "Prefill Cancelled!");
                        const chat_msg no_message;  // end.message points at it
                        sink.on_end(turn_end(no_message, false));
                        this->auto_chat_engine->clear_context();
                        this->prompt_cache.reset();
                        return;
                    }
                    sink.on_error(prefill_failed_error(meta_info));
                    this->auto_chat_engine->clear_context();
                    this->prompt_cache.reset();
                    return;
                }
            } catch (const std::exception& e) {
                sink.on_error(exception_error(e));
                this->auto_chat_engine->clear_context();
                this->prompt_cache.reset();
                return;
            }
            chat_turn_start start;
            start.prompt_tokens = meta_info.prompt_tokens;
            start.cached_prompt_tokens = meta_info.cached_prompt_tokens;
            if (lstream) {
                start.content_logprobs = [&lstream](const std::string& content) { return lstream->aligner().take_content(content); };
            }
            sink.on_start(start);
            header_print("FLM", "Start generating...");
            try {
                response_text = auto_chat_engine->generate(meta_info, length_limit, out, [&] { return cancellation_token->cancelled(); });
                out.flush();
            } catch (const std::exception& e) {
                sink.on_error(exception_error(e));
                this->auto_chat_engine->clear_context();
                this->prompt_cache.reset();
                return;
            }
            // A character the generation stopped partway through can never be finished
            response_text = sanitize_utf8(response_text);
            chat_msg message = chat_parse(spec, output_prefill() + response_text, false, parse_options);
            for (chat_tool_call& call : message.tool_calls) {
                call.id = chat_new_tool_call_id();
            }
            const chat_turn_end end = turn_end(message, true);
            if (meta_info.stop_reason == CANCEL_DETECTED) {
                header_print("❌ ", "Generation Cancelled!");
                this->prompt_cache.reset();
            }
            log_request_summary(meta_info, this->auto_chat_engine->get_current_context_length(), this->auto_chat_engine->get_max_length());
            sink.on_end(end);
        }

    } catch (const std::exception& e) {
        sink.on_error({500, "server_error", e.what(), std::string(), std::nullopt});
    }
}

///@brief Handle the openai audio transcriptions request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_openai_audio_transcriptions(const json& request,
                                        std::function<void(const json&)> send_response,
                                        StreamResponseCallback send_streaming_response,
                                        std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        if (!request.contains("file") || !request["file"].is_string()) {
            send_response({{"error", {{"message", "the request has no audio file (multipart field \"file\")"}, {"type", "invalid_request_error"}, {"code", 400}}}});
            return;
        }
        std::string model = request["model"];
        std::string file_content = request["file"].get<std::string>();
        std::vector<uint8_t> audio_raw(file_content.begin(), file_content.end());
        std::string language = request.value("language", "");
        std::string prompt = request.value("prompt", "");
        std::string response_format = request.value("response_format", "json");
        const std::string stream_field = request.contains("stream") && request["stream"].is_string() ? request["stream"].get<std::string>() : "";
        const bool stream = request.contains("stream") && (request["stream"].is_boolean() ? request["stream"].get<bool>() : (stream_field == "true" || stream_field == "1"));
        auto form_int = [&request](const char* key, int fallback) {
            if (!request.contains(key)) return fallback;
            if (request[key].is_number_integer()) return request[key].get<int>();
            try { return std::stoi(request[key].get<std::string>()); } catch (...) { throw std::invalid_argument(std::string(key) + " must be an integer"); }
        };
        json response;
        if (response_format != "json" && response_format != "text" && response_format != "verbose_json" &&
            response_format != "srt" && response_format != "vtt") {
            throw std::invalid_argument("unsupported response_format: " + response_format +
                                        " (json, text, verbose_json, srt or vtt)");
        }
        // Segments need the timestamps; json and text are the same transcript without them.
        const bool timed = response_format == "verbose_json" || response_format == "srt" || response_format == "vtt";
        // A model that transcribes by itself (r2t2) answers when it is the one asked for; any other model name is served by
        // the Whisper sidecar (--asr 1), as before.
        Qwen3_ASR* self_asr = nullptr;
        // --asr r2t2:...: the sidecar answers every transcription, so the chat model stays loaded
        if (this->asr_sidecar_engine) {
            self_asr = dynamic_cast<Qwen3_ASR*>(this->asr_sidecar_engine.get());
        } else if (!model.empty() && supported_models.is_model_supported(supported_models.rectify_model_tag(model))) {
            auto [asr_tag, asr_info] = this->supported_models.get_model_info(model);
            if (asr_info.contains("details") && asr_info["details"].value("family", "") == "r2t2") {
                if (!ensure_model_loaded(model)) throw std::runtime_error("failed to load " + model);
                self_asr = dynamic_cast<Qwen3_ASR*>(this->auto_chat_engine.get());
            }
        }
        std::string audio_context;
        std::string language_name;
        double duration = 0.0;
        if (this->asr || self_asr) {
            // Transcripts are user speech: log them only when FLM_LOG_CONTENT or FLM_LOG_RAW_OUTPUT asks.
            const bool log_transcript = flm::log_privacy::raw_output_enabled();
            std::ostream no_log(nullptr);
            std::ostream& transcript_log = log_transcript ? std::cout : no_log;
            if (self_asr && stream) {
                // Append-only text as the audio is processed chunk by chunk (OpenAI transcription events): the model's streaming mode on a whole file
                header_print("FLM", "Transforming audio to text, streaming (" << model << ")...");
                chat_meta_info_t asr_meta;
                asr_meta.max_prefill_len = this->prefill_chunk_len;
                cancellation_token->reset();
                auto sse = [&send_streaming_response](const json& event, bool final) { send_streaming_response(json("data: " + event.dump() + "\n\n"), final); };
                const int chunk_ms = form_int("chunk_size_ms", 1000);
                const int lookahead_ms = form_int("lookahead_ms", 160);
                const int unfixed = form_int("unfixed_token_num", 1);
                try {
                    audio_context = self_asr->transcribe_streaming(asr_meta, audio_raw, language, prompt, chunk_ms, lookahead_ms, unfixed,
                        [&sse](const std::string& delta) { sse({{"type", "transcript.text.delta"}, {"delta", delta}}, false); },
                        language_name, duration, [&] { return cancellation_token->cancelled(); });
                } catch (const std::invalid_argument& e) {
                    send_response({{"error", {{"message", e.what()}, {"type", "invalid_request_error"}, {"code", 400}}}});
                    return;
                }
                sse({{"type", "transcript.text.done"}, {"text", audio_context}}, false);
                send_streaming_response(json("data: [DONE]\n\n"), true);
                return;
            }
            if (self_asr) {
                header_print("FLM", "Transforming audio to text (" << model << ")...");
                chat_meta_info_t asr_meta;
                asr_meta.max_prefill_len = this->prefill_chunk_len;
                transcript_log << "Audio content: " << std::flush;
                audio_context = self_asr->transcribe(asr_meta, audio_raw, language, prompt, request.value("max_tokens", 8192), transcript_log,
                                                     language_name, duration, [&] { return cancellation_token->cancelled(); });
                transcript_log << std::endl;
            } else {
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
            this->whisper_engine->load_audio(audio_raw);
            header_print("FLM", "Transforming audio to text...");
            transcript_log << "Audio content: " << std::flush;
            std::pair<std::string, std::string> audio_result = this->whisper_engine->generate(Whisper::whisper_task_type_t::e_transcribe, true, timed, transcript_log, language, prompt);
            audio_context = audio_result.first;
            language_name = audio_result.second;
            duration = this->whisper_engine->audio_duration();
            transcript_log << std::endl;
#else
            throw std::runtime_error("ASR models are not supported in this build");
#endif
            }

            if (response_format == "json") {
                response = {
                    {"model", model},
                    {"text", audio_context}
                };
            }
            else if (response_format == "text") {
                response = audio_context;  // a string goes out as a text/plain body
            }
            else {
                std::vector<transcript::Segment> segments = transcript::split_segments(audio_context, duration);
                if (response_format == "srt") {
                    response = transcript::to_srt(segments);
                }
                else if (response_format == "vtt") {
                    response = transcript::to_vtt(segments);
                }
                else {
                    // Whisper's decoder keeps no per-segment statistics: tokens,
                    // avg_logprob, compression_ratio and no_speech_prob are
                    // placeholders that no client filter drops a segment on.
                    json segments_json = json::array();
                    std::string text;
                    for (const transcript::Segment& segment : segments) {
                        segments_json.push_back({
                            {"id", (int)segments_json.size()},
                            {"seek", 0},
                            {"start", segment.start},
                            {"end", segment.end},
                            {"text", segment.text},
                            {"tokens", json::array()},
                            {"temperature", 0.0},
                            {"avg_logprob", 0.0},
                            {"compression_ratio", 0.0},
                            {"no_speech_prob", 0.0}
                        });
                        text += segment.text;
                    }
                    // OpenAI names the language in lower case, as openai-whisper's LANGUAGES.
                    std::transform(language_name.begin(), language_name.end(), language_name.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    response = {
                        {"task", "transcribe"},
                        {"language", language_name},
                        {"duration", duration},
                        {"text", text},
                        {"segments", segments_json},
                        {"model", model}
                    };
                }
            }
        }
        else {
            header_print("Warning", "No asr model loaded, cannot load audio file");
        }
        send_response(response);
        //this->whisper_engine->clear_context();
    }
    catch (const std::invalid_argument& e) {
        // A bad language or response_format. A 400, as OpenAI's: clients retry a 500.
        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "invalid_request_error"},
                {"code", 400}
            }}
        };
        send_response(error_response);
    }
    catch (const std::exception& e) {
        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "server_error"},
                {"code", 500}
            }}
        };
        send_response(error_response);
    }
}

///@brief Handle the openai completion request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_openai_completion(const json& request,
    std::function<void(const json&)> send_response,
    StreamResponseCallback send_streaming_response,
    std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        // Extract OpenAI-style parameters
        if (!has_value(request, "prompt") || !request.at("prompt").is_string()) {
            send_response({{"error", {{"message", "'prompt' is required and must be a string"}, {"type", "invalid_request_error"}, {"code", 400}}}});
            return;
        }
        std::string prompt = request.at("prompt");
        std::string model = value_or(request, "model", current_model_tag);
        const std::string request_backend = value_or(request, "backend", std::string());
        bool stream = value_or(request, "stream", false);
        json options = value_or(request, "options", json::object());
        openai_sampling_request sampling;
        std::string request_error;
        if (!sampling_request_from_openai(request, sampling, request_error)) {
            send_response({{"error", {{"message", request_error}, {"type", "invalid_request_error"}, {"code", 400}}}});
            return;
        }

        // direct return if model not supported
        if (!is_known_model(model)) {
            send_response(UnknownModelResponse(model));
            return;
        }

        int length_limit = value_or(request, "max_tokens", 4096);

        std::string refusal;
        if (!ensure_model_loaded(model, request_backend, &refusal)) {
            send_response(ModelLoadFailedResponse(model, refusal));
            return;
        }
        // This shares the engine with /v1/chat/completions and replaces the
        // conversation in it, so the chat cache no longer describes it.
        this->prompt_cache.reset();

        configure_chat_engine_parameters(options, request);
        apply_sampling_request(*auto_chat_engine, sampling);

        chat_meta_info_t meta_info;
        meta_info.max_prefill_len = this->prefill_chunk_len;
        meta_info.ignore_eos = value_or(request, "ignore_eos", false);
        meta_info.min_tokens = value_or(request, "min_tokens", 0);
        lm_uniform_input_t uniformed_input;
        uniformed_input.requested_max_new_tokens =
            normalize_requested_max_new_tokens(
                has_value(request, "max_tokens")
                    ? std::optional<int>(request.at("max_tokens").get<int>())
                    : std::nullopt);
        header_print("FLM", "Start generating...");

        if (stream) {
            // Create a wrapper callback that passes the pre-formatted SSE string directly
            bool stream_started = false;
            auto openai_stream_callback = [&send_streaming_response, &stream_started](const std::string& data, bool is_final) {
                stream_started = true;
                json data_json = data;
                send_streaming_response(data_json, is_final);
                };
            streaming_ostream_openai ostream(model, openai_stream_callback, stream_include_usage(request));  // streaming in completion format
            // Once a chunk is out the headers are sent, so an error can only go
            // out as one more event, and the stream still has to be ended or the
            // client waits on it until its own timeout
            auto send_stream_error = [&](const json& error_response) {
                if (stream_started) {
                    send_streaming_response(json("data: " + error_response.dump() + "\n\n"), false);
                    send_streaming_response(json("data: [DONE]\n\n"), true);
                } else {
                    send_response(error_response);
                }
                };
            uniformed_input.prompt = prompt;
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input, [&] { return cancellation_token && cancellation_token->cancelled(); });
                if (!success) {
                    if (meta_info.stop_reason == CANCEL_DETECTED || (cancellation_token && cancellation_token->cancelled())) {
                        meta_info.stop_reason = CANCEL_DETECTED;
                        header_print("❌ ", "Prefill Cancelled!");
                        ostream.finalize(meta_info);
                        this->auto_chat_engine->clear_context();
                        return;
                    }
                    send_stream_error(PrefillFailedResponse(meta_info));
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                send_stream_error(OpenAIExceptionResponse(e));
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream, [&] { return cancellation_token && cancellation_token->cancelled(); });
            } catch (const ModelRequestError& error) {
                send_stream_error(ModelErrorResponse(error));
                return;
            } catch (const std::exception& e) {
                send_stream_error(OpenAIExceptionResponse(e));
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                ostream.finalize(meta_info);
            } catch (const std::exception& e) {
                header_print("ERROR", std::string("Finishing the stream failed: ") + e.what());
                send_stream_error(OpenAIExceptionResponse(e));
            }
            log_request_summary(meta_info, this->auto_chat_engine->get_current_context_length(), this->auto_chat_engine->get_max_length());

            this->auto_chat_engine->clear_context();
        }
        else {
            std::stringstream ss;
            wstream_buf obuf(ss);
            std::ostream ostream(&obuf);
            uniformed_input.prompt = prompt;
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input, [&] { return cancellation_token && cancellation_token->cancelled(); });
                if (!success) {
                    if (meta_info.stop_reason == CANCEL_DETECTED || (cancellation_token && cancellation_token->cancelled())) {
                        header_print("❌ ", "Prefill Cancelled!");
                        send_response(json());
                        this->auto_chat_engine->clear_context();
                        return;
                    }
                    send_response(PrefillFailedResponse(meta_info));
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = OpenAIExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream, [&] { return cancellation_token && cancellation_token->cancelled(); });
            } catch (const std::exception& e) {
                json error_response = OpenAIExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            std::string response_text = ss.str();
            auto history = this->auto_chat_engine->get_history();

            json response = {
                {"id", "fastflowlm-chat-completion"},
                {"object", "text_completion"},
                {"created", (int)std::time(nullptr)},
                {"model", model},
                {"choices", json::array({
                    {
                        {"text", response_text},
                        {"index", 0},
                        {"logprobs", nullptr},
                        {"finish_reason", stop_reason_to_string(meta_info.stop_reason)}
                    }
                })},
                {"usage", {
                    {"prompt_tokens", meta_info.prompt_tokens},
                    {"prompt_tokens_details", {{"cached_tokens", meta_info.cached_prompt_tokens}}},
                    {"completion_tokens", meta_info.generated_tokens},
                    {"total_tokens", meta_info.prompt_tokens + meta_info.generated_tokens}
                }}
            };
            if (meta_info.prefix_cache.valid) response["usage"]["flm_prefix_cache"] = prefix_cache_usage_json(meta_info.prefix_cache);
            log_request_summary(meta_info, this->auto_chat_engine->get_current_context_length(), this->auto_chat_engine->get_max_length());
            send_response(response);
        }
    }
    catch (const std::exception& e) {
        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "server_error"},
                {"code", 500}
            }}
        };
        send_response(error_response);
    }
}
