/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file protocol_openai_chat.hpp
 * \brief A chat turn written as OpenAI Chat Completions: the error object, the stream chunks, the final object
 * \author FastFlowLM Team
 */
#pragma once

#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <nlohmann/json.hpp>
#include "chat_core.hpp"
#include "prefix_cache_usage.hpp"

/// \brief An error as OpenAI's error object; its code is the HTTP status
inline nlohmann::ordered_json openai_error_json(const chat_error& error) {
    nlohmann::ordered_json body = {{"message", error.message}, {"type", error.type}};
    if (!error.param.empty()) {
        body["param"] = error.param;
    }
    body["code"] = error.status;
    if (error.session_cleared.has_value()) {
        body["session_cleared"] = *error.session_cleared;
    }
    return {{"error", body}};
}

/// \brief finish_reason of a chat completion
inline const char* openai_finish_reason(chat_finish finish) {
    switch (finish) {
        case chat_finish::STOP: return "stop";
        case chat_finish::LENGTH: return "length";
        case chat_finish::CANCEL: return "cancel";
        case chat_finish::FAILED: return "error";
        case chat_finish::TOOL_CALLS: return "tool_calls";
    }
    return "UNKNOWN";
}

/// \brief Writes a chat turn to a /v1/chat/completions client
/// \note Streamed: one chat.completion.chunk per delta, a final chunk with the finish reason,
///       the usage chunk when the client asked for it, then data: [DONE]. Not streamed: one
///       chat.completion object. An error is the error object alone, or, once a chunk is out,
///       one more event followed by data: [DONE].
class openai_chat_sink : public chat_sink {
public:
    using json = nlohmann::ordered_json;

    ///@param send_response sends the one body of a response that is not streamed
    ///@param send_streaming_response sends one server-sent event; true with the last one
    ///@param include_usage stream_options.include_usage: send usage in its own chunk before [DONE]
    openai_chat_sink(std::function<void(const json&)> send_response,
                     std::function<void(const json&, bool)> send_streaming_response, bool include_usage)
        : send_response(std::move(send_response)), send_streaming_response(std::move(send_streaming_response)),
          include_usage(include_usage) {}

    void on_begin(const chat_turn_begin& begin) override {
        model_name = begin.model;
        stream = begin.stream;
        if (stream) {
            // Generate a unique ID for this stream
            generate_stream_id();
            generate_created();
            generate_system_fingerprint();
        }
    }

    void on_start(const chat_turn_start& start) override {
        content_logprobs = start.content_logprobs;
    }

    void on_delta(const chat_msg_delta& d) override {
        json delta = {{"role", "assistant"}};
        // only content has logprobs; reasoning and tool calls go out with null
        json choice_logprobs = nullptr;
        if (d.is_tool_call()) {
            json call = {{"index", d.tool_index}};
            if (!d.tool_id.empty()) {
                // the first delta of a call names it; later ones only add arguments
                call["id"] = d.tool_id;
                call["type"] = "function";
                call["function"] = {{"name", d.tool_name}, {"arguments", d.tool_arguments}};
            }
            else {
                call["function"] = {{"arguments", d.tool_arguments}};
            }
            delta["tool_calls"] = json::array({call});
        }
        else if (!d.reasoning.empty()) {
            delta["reasoning_content"] = d.reasoning;
        }
        else {
            delta["content"] = d.content;
            if (content_logprobs) {
                choice_logprobs = {{"content", content_logprobs(d.content)}, {"refusal", nullptr}};
            }
        }

        json choice = {
            {"index", 0},
            {"delta", delta}
        };
        if (content_logprobs) {
            choice["logprobs"] = choice_logprobs;
        }
        choice["finish_reason"] = nullptr;
        json response = {
            {"id", stream_id},
            {"object", "chat.completion.chunk"},
            {"created", created},
            {"model", model_name},
            {"choices", json::array({choice})}
        };
        if (include_usage) {
            response["usage"] = nullptr;
        }
        send_event("data: " + response.dump() + "\n\n", false);
    }

    void on_end(const chat_turn_end& end) override {
        if (stream) {
            send_final_chunks(end);
        }
        else {
            send_completion(end);
        }
        content_logprobs = nullptr;  // it reads the turn's state, which ends with the turn
    }

    void on_error(const chat_error& error) override {
        // Set before anything is sent: if writing this error fails, the error about that
        // failure is the plain object
        const bool as_event = stream_started && !failed;
        failed = true;
        content_logprobs = nullptr;
        const json error_response = openai_error_json(error);
        // Once a chunk is out the headers are sent, so an error can only go
        // out as one more event, and the stream still has to be ended or the
        // client waits on it until its own timeout. That holds after the final
        // chunk too (cleaning up after a cancelled prompt can fail). Only the
        // first error goes out as an event; a later one is the plain object.
        if (as_event) {
            send_streaming_response(json("data: " + error_response.dump() + "\n\n"), false);
            send_streaming_response(json("data: [DONE]\n\n"), true);
        }
        else {
            send_response(error_response);
        }
    }

private:
    ///@brief Generate a unique stream ID
    void generate_stream_id() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 15);

        std::stringstream ss;
        ss << "chatcmpl-";
        for (int i = 0; i < 24; ++i) {
            ss << std::hex << dis(gen);
        }
        stream_id = ss.str();
    }

    void generate_created() {
        created = static_cast<long>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count());
    }

    void generate_system_fingerprint() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 15);

        std::stringstream ss;
        ss << "fp_";
        for (int i = 0; i < 16; ++i) {
            ss << std::hex << dis(gen);
        }
        system_fingerprint = ss.str();
    }

    ///@brief Send one server-sent event of the stream
    void send_event(const std::string& data, bool is_final) {
        stream_started = true;
        send_streaming_response(json(data), is_final);
    }

    ///@brief The end of a streamed turn: the finish reason, the usage when asked for, [DONE]
    void send_final_chunks(const chat_turn_end& end) {
        json final_choice = {
            {"index", 0},
            {"delta", {
                {"content", nullptr}
            }}
        };
        if (content_logprobs) {
            final_choice["logprobs"] = nullptr;
        }
        final_choice["finish_reason"] = openai_finish_reason(end.finish);
        json final_response = {
            {"id", stream_id},
            {"object", "chat.completion.chunk"},
            {"created", created},
            {"model", model_name},
            {"system_fingerprint", system_fingerprint},
            {"choices", json::array({final_choice})}
        };
        json usage = {
            {"prompt_tokens", end.prompt_tokens},
            {"prompt_tokens_details", {{"cached_tokens", end.cached_prompt_tokens}}},
            {"completion_tokens", end.generated_tokens},
            {"total_tokens", end.prompt_tokens + end.generated_tokens},
            {"active_kv_tokens", end.context_tokens},
            {"max_kv_token_capacity", end.context_capacity},
            {"kv_token_occupancy_rate_percentage", (float)end.context_tokens / (float)end.context_capacity * 100},
            {"load_duration", static_cast<double>(end.load_duration) / 1'000'000'000},
            {"prefill_duration_ttft", static_cast<double>(end.prefill_duration) / 1'000'000'000},
            {"decoding_duration", static_cast<double>(end.decoding_duration) / 1'000'000'000},
            {"prefill_speed_tps", static_cast<double>(end.prompt_tokens - end.cached_prompt_tokens) / static_cast<double>(end.prefill_duration) * 1'000'000'000},
            {"decoding_speed_tps", static_cast<double>(end.generated_tokens) / static_cast<double>(end.decoding_duration) * 1'000'000'000},
        };
        if (end.prefix_cache != nullptr) usage["flm_prefix_cache"] = prefix_cache_usage_json(*end.prefix_cache);
        if (include_usage) {
            // OpenAI: every chunk carries a null usage, then one extra chunk
            // with no choices carries the usage of the whole request
            final_response["usage"] = nullptr;
        }
        send_event("data: " + final_response.dump() + "\n\n", false);
        if (include_usage) {
            json usage_response = {
                {"id", stream_id},
                {"object", "chat.completion.chunk"},
                {"created", created},
                {"model", model_name},
                {"system_fingerprint", system_fingerprint},
                {"choices", json::array()},
                {"usage", usage}
            };
            send_event("data: " + usage_response.dump() + "\n\n", false);
        }
        // the server log shows the usage whether or not the client asked for it
        std::cout << "ChatCompletionChunk: " << final_response << std::endl;
        std::cout << "Usage: " << usage << std::endl;
        // Send the [DONE] message
        send_event("data: [DONE]\n\n", true);
    }

    ///@brief The whole turn as one chat.completion object
    void send_completion(const chat_turn_end& end) {
        if (!end.generated) {
            // cancelled while the prompt was fed: nobody is left to read a body
            send_response(json());
            return;
        }
        const chat_msg& parsed = *end.message;
        json message;
        message["role"] = "assistant";
        if (!parsed.reasoning.empty()) {
            message["reasoning_content"] = parsed.reasoning;
        }

        if (!parsed.tool_calls.empty()) {
            json tool_calls_json = json::array();
            for (size_t i = 0; i < parsed.tool_calls.size(); i++) {
                tool_calls_json.push_back({
                    {"index", i},
                    {"id", parsed.tool_calls[i].id},
                    {"type", "function"},
                    {"function", {
                        {"name", parsed.tool_calls[i].name},
                        {"arguments", parsed.tool_calls[i].arguments}
                    }}
                });
            }
            message["tool_calls"] = tool_calls_json;
            // OpenAI sends null rather than "" when the turn is only tool calls
            message["content"] = parsed.content.empty() ? json(nullptr) : json(parsed.content);
        }
        else {
            message["content"] = parsed.content;
        }

        // Only the content's tokens: reasoning and tool calls have no logprobs
        json choice_logprobs = nullptr;
        if (content_logprobs) {
            choice_logprobs = {{"content", content_logprobs(parsed.content)}, {"refusal", nullptr}};
        }
        json choices = json::array({
            {
                {"index", 0},
                {"message", message},
                {"logprobs", choice_logprobs},
                {"finish_reason", openai_finish_reason(end.finish)}
            }
        });
        json response = {
            {"id", "fastflowlm-chat-completion"},
            {"object", "chat.completion"},
            {"created", static_cast<long long>(std::time(nullptr))},
            {"model", model_name},
            {"choices", choices},
            {"usage", {
                {"prompt_tokens", end.prompt_tokens},
                {"prompt_tokens_details", {{"cached_tokens", end.cached_prompt_tokens}}},
                {"completion_tokens", end.generated_tokens},
                {"total_tokens", end.prompt_tokens + end.generated_tokens},
                {"kv_token_occupancy_rate_percentage", (float)end.context_tokens / (float)end.context_capacity * 100},
                {"load_duration", static_cast<double>(end.load_duration) / 1'000'000'000},
                {"prefill_duration_ttft", static_cast<double>(end.prefill_duration) / 1'000'000'000},
                {"decoding_duration", static_cast<double>(end.decoding_duration) / 1'000'000'000},
                {"prefill_speed_tps", static_cast<double>(end.prompt_tokens - end.cached_prompt_tokens) / static_cast<double>(end.prefill_duration) * 1'000'000'000},
                {"decoding_speed_tps", static_cast<double>(end.generated_tokens) / static_cast<double>(end.decoding_duration) * 1'000'000'000},
            }},
            {"service_tier", "default"}
        };
        if (end.prefix_cache != nullptr) response["usage"]["flm_prefix_cache"] = prefix_cache_usage_json(*end.prefix_cache);
        send_response(response);
    }

    std::function<void(const json&)> send_response;
    std::function<void(const json&, bool)> send_streaming_response;
    ///@brief Whether the client asked for usage, see stream_options.include_usage
    bool include_usage;
    ///@brief Model name
    std::string model_name;
    ///@brief Whether the turn is streamed
    bool stream = false;
    ///@brief A chunk went out, so the response's status and headers are sent
    bool stream_started = false;
    ///@brief An error was reported
    bool failed = false;
    ///@brief Stream ID
    std::string stream_id;
    long created = 0;
    std::string system_fingerprint;
    ///@brief The logprobs of a piece of content; empty when the request asks for none
    std::function<json(const std::string&)> content_logprobs;
};
