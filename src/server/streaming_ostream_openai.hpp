/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file streaming_ostream_openai.hpp
 * \brief Custom ostream for streaming
 * \author FastFlowLM Team
 * \date 2025-06-24
 *  \version 0.9.24
 */
#pragma once

#include <ostream>
#include <streambuf>
#include <functional>
#include <string>
#include <vector>
#include <random>
#include <sstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include "prefix_cache_usage.hpp"
#include "AutoModel/automodel.hpp"
#include "chat/chat_stream.hpp"
#include "chat_logprobs.hpp"

using json = nlohmann::ordered_json;


class AutoModel;

// v1/completions
///@brief Custom streambuf that captures tokens and sends them immediately
///@param model the model
///@param callback the callback
///@param is_chat_format the is chat format
///@return the streaming buf
///@note The packet is sent every std::flush, but only when UTF-8 sequences are complete
class streaming_buf_openai : public std::streambuf {
public:
    ///@brief StreamCallback
    using StreamCallback = std::function<void(const std::string&, bool)>;
    
    ///@param model the model name reported to the client
    ///@param callback receives each server-sent event
    ///@param include_usage stream_options.include_usage: send usage in its own chunk before [DONE]
    streaming_buf_openai(const std::string& model, StreamCallback callback, bool include_usage)
        : model_name(model), stream_callback(callback), include_usage(include_usage), first_chunk(true) {
        // Generate a unique ID for this stream
        generate_stream_id();
        generate_created();
        generate_system_fingerprint();
    }

protected:
    ///@brief Called when buffer is full or flush is requested
    ///@param ch the character
    ///@return the character
    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof()) {
            buffer += static_cast<char>(ch);
        }
        return ch;
    }
    
    ///@brief Called when stream is flushed
    ///@return 0
    int sync() override {
        flush_complete_utf8_sequences(false);
        return 0;
    }

public:
    ///@brief Call this when generation is complete
    void finalize(chat_meta_info_t& meta_info) {
        // Send the remaining content but not an incomplete UTF-8 sequence at
        // its end: a generation cut off by the length limit or a cancel can
        // stop partway through a multi-byte character, the missing bytes will
        // never come, and invalid UTF-8 makes json::dump() throw
        flush_complete_utf8_sequences(false);
        buffer.clear();
        send_final_response(meta_info);
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
        created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
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

    ///@brief Get UTF-8 sequence length from first byte
    ///@param first_byte the first byte
    ///@return sequence length, or 0 if invalid
    size_t get_utf8_sequence_length(unsigned char first_byte) {
        if ((first_byte & 0x80) == 0) {
            return 1; // Single byte sequence
        } else if ((first_byte & 0xE0) == 0xC0) {
            return 2; // Two byte sequence
        } else if ((first_byte & 0xF0) == 0xE0) {
            return 3; // Three byte sequence
        } else if ((first_byte & 0xF8) == 0xF0) {
            return 4; // Four byte sequence
        }
        return 0; // Invalid UTF-8 start byte
    }
    
    ///@brief Flush only complete UTF-8 sequences
    ///@param is_final the is final
    void flush_complete_utf8_sequences(bool is_final) {
        if (buffer.empty()) return;
        
        std::string complete_content;
        size_t pos = 0;
        
        // Process complete UTF-8 sequences
        while (pos < buffer.size()) {
            unsigned char first = static_cast<unsigned char>(buffer[pos]);
            size_t seq_len = get_utf8_sequence_length(first);
            
            if (seq_len == 0) {
                // Invalid UTF-8 start byte, skip it
                pos++;
                continue;
            }
            
            // Check if we have a complete sequence
            if (pos + seq_len > buffer.size()) {
                // Incomplete sequence, stop here
                break;
            }
            
            // Add complete sequence to output
            complete_content.append(buffer, pos, seq_len);
            pos += seq_len;
        }
        
        // Send complete sequences if any
        if (!complete_content.empty()) {
            send_response(complete_content, is_final);
        }
        
        // Remove processed bytes from buffer
        if (pos > 0) {
            buffer.erase(0, pos);
        }
    }
    
    ///@brief Send the response
    ///@param content the content
    ///@param is_final the is final
    void send_response(const std::string& content, bool is_final) {
        json response;
        
        // Content chunk
        response = {
            {"id", stream_id},
            {"object", "text_completion"},
            {"created", created},
            {"system_fingerprint", system_fingerprint},
            {"model", model_name},
            {"choices", json::array({
                {
                    {"text", content},
                    {"index", 0},
                    {"logprobs", nullptr},
                    {"finish_reason", nullptr}
                }
            })}
        };
        if (include_usage) {
            response["usage"] = nullptr;
        }
        
        stream_callback("data: " + response.dump() + "\n\n", is_final);
      
    }

    ///@brief Send the chat final response
    void send_final_response(chat_meta_info_t& meta_info) {
        // Send final chunk with finish_reason only (no empty delta)
        json final_response = {
            {"id", stream_id},
            {"object", "text_completion"},
            {"created", created},
            {"system_fingerprint", system_fingerprint},
            {"model", model_name},
            {"choices", json::array({
                {
                    {"finish_reason", stop_reason_to_string(meta_info.stop_reason)},
                }
            })}
        };
        if (include_usage) {
            // OpenAI: every chunk carries a null usage, then one extra chunk
            // with no choices carries the usage of the whole request
            final_response["usage"] = nullptr;
        }
        stream_callback("data: " + final_response.dump() + "\n\n", false);
        if (include_usage) {
            json usage_response = {
                {"id", stream_id},
                {"object", "text_completion"},
                {"created", created},
                {"system_fingerprint", system_fingerprint},
                {"model", model_name},
                {"choices", json::array()},
                {"usage", {
                    {"prompt_tokens", meta_info.prompt_tokens},
                    {"prompt_tokens_details", {{"cached_tokens", meta_info.cached_prompt_tokens}}},
                    {"completion_tokens", meta_info.generated_tokens},
                    {"total_tokens", meta_info.prompt_tokens + meta_info.generated_tokens}
                }}
            };
            if (meta_info.prefix_cache.valid) usage_response["usage"]["flm_prefix_cache"] = prefix_cache_usage_json(meta_info.prefix_cache);
            stream_callback("data: " + usage_response.dump() + "\n\n", false);
        }
        // Send the [DONE] message
        stream_callback("data: [DONE]\n\n", true);
    }
    

    ///@brief Buffer
    std::string buffer;
    ///@brief Model name
    std::string model_name;
    ///@brief Stream callback
    StreamCallback stream_callback;
    ///@brief Whether the client asked for usage, see stream_options.include_usage
    bool include_usage;
    ///@brief Stream ID
    std::string stream_id;
    long created;
    std::string system_fingerprint;
    ///@brief First chunk flag
    bool first_chunk;
};

///@brief Custom ostream for streaming
///@param model the model
///@param callback the callback
///@param is_chat_format the is chat format
///@return the streaming ostream
class streaming_ostream_openai : public std::ostream {
public:
    streaming_ostream_openai(const std::string& model, streaming_buf_openai::StreamCallback callback, bool include_usage)
        : std::ostream(&buf), buf(model, callback, include_usage) {}
    
    ///@brief Finalize the chat
    void finalize(chat_meta_info_t& meta_info) {
        buf.finalize(meta_info);
    }

private:
    ///@brief Buffer
    streaming_buf_openai buf;
};



// v1/chat/completions
///@brief Custom streambuf that parses the generated text and streams it as chat completion chunks
///@note The packet is sent every std::flush, but only when UTF-8 sequences are complete.
///      Every packet is parsed with the model's chat format; reasoning, content and
///      tool call arguments go out as the deltas between consecutive parses.
class streaming_buf_openai_chat : public std::streambuf {
public:
    ///@brief StreamCallback
    using StreamCallback = std::function<void(const std::string&, bool)>;

    ///@param model the model name reported to the client
    ///@param auto_chat_engine the engine generating the text
    ///@param callback receives each server-sent event
    ///@param options per request parsing options; the tools must outlive the stream
    ///@param include_usage stream_options.include_usage: send usage in its own chunk before [DONE]
    streaming_buf_openai_chat(const std::string& model, AutoModel* auto_chat_engine, StreamCallback callback,
                              chat_parse_options options, bool include_usage)
        : model_name(model), auto_chat_engine(auto_chat_engine), stream_callback(callback),
          include_usage(include_usage), parse_options(options) {
        // Generate a unique ID for this stream
        generate_stream_id();
        generate_created();
        generate_system_fingerprint();
    }

    ///@brief Call this once the prompt is inserted, before generating
    ///@param meta_info the request's meta info, its prompt tail tells where the output starts;
    ///       when it asks for logprobs it must outlive the stream
    void start(const chat_meta_info_t& meta_info) {
        const chat_format_spec& spec = auto_chat_engine->get_chat_format();
        stream = std::make_unique<chat_stream>(spec, chat_format_prefill(spec, meta_info.prompt_tail), parse_options);
        if (meta_info.logprobs_top_n >= 0) {
            logprobs = std::make_unique<chat_logprobs_aligner>(auto_chat_engine, meta_info.logprobs);
        }
    }

protected:
    ///@brief Called when buffer is full or flush is requested
    ///@param ch the character
    ///@return the character
    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof()) {
            buffer += static_cast<char>(ch);
            if (logprobs) {
                logprobs_written += static_cast<char>(ch);
            }
        }
        return ch;
    }

    ///@brief Called when stream is flushed
    ///@return 0
    int sync() override {
        flush_logprobs();
        flush_complete_utf8_sequences();
        return 0;
    }

public:
    ///@brief Call this when generation is complete
    void finalize(chat_meta_info_t& meta_info) {
        flush_logprobs();
        // Send the remaining content but not an incomplete UTF-8 sequence at
        // its end: a generation cut off by the length limit or a cancel can
        // stop partway through a multi-byte character, the missing bytes will
        // never come, and invalid UTF-8 makes json::dump() throw
        flush_complete_utf8_sequences();
        buffer.clear();
        // Let the parser release whatever it held back
        send_response("", true);
        // Only promote to a tool-call finish reason on an otherwise clean stop;
        // length/cancel/error must win, so a truncated or aborted generation is
        // never reported to the client as a complete tool call.
        if (stream && !stream->message().tool_calls.empty()
            && meta_info.stop_reason == stop_reason_t::EOT_DETECTED) {
            meta_info.stop_reason = stop_reason_t::TOOL_DETECTED;
        }
        send_final_response(meta_info);
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
        created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
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

    ///@brief Get UTF-8 sequence length from first byte
    ///@param first_byte the first byte
    ///@return sequence length, or 0 if invalid
    size_t get_utf8_sequence_length(unsigned char first_byte) {
        if ((first_byte & 0x80) == 0) {
            return 1; // Single byte sequence
        }
        else if ((first_byte & 0xE0) == 0xC0) {
            return 2; // Two byte sequence
        }
        else if ((first_byte & 0xF0) == 0xE0) {
            return 3; // Three byte sequence
        }
        else if ((first_byte & 0xF8) == 0xF0) {
            return 4; // Four byte sequence
        }
        return 0; // Invalid UTF-8 start byte
    }

    ///@brief Place the tokens written since the last flush, before the parser sees them
    void flush_logprobs() {
        if (logprobs) {
            logprobs->on_flush(logprobs_written);
            logprobs_written.clear();
        }
    }

    ///@brief Flush only complete UTF-8 sequences
    void flush_complete_utf8_sequences() {
        if (buffer.empty()) return;

        std::string complete_content;
        size_t pos = 0;

        // Process complete UTF-8 sequences
        while (pos < buffer.size()) {
            unsigned char first = static_cast<unsigned char>(buffer[pos]);
            size_t seq_len = get_utf8_sequence_length(first);

            if (seq_len == 0) {
                // Invalid UTF-8 start byte, skip it
                pos++;
                continue;
            }

            // Check if we have a complete sequence
            if (pos + seq_len > buffer.size()) {
                // Incomplete sequence, stop here
                break;
            }

            // Add complete sequence to output
            complete_content.append(buffer, pos, seq_len);
            pos += seq_len;
        }

        // Send complete sequences if any
        if (!complete_content.empty()) {
            send_response(complete_content, false);
        }

        // Remove processed bytes from buffer
        if (pos > 0) {
            buffer.erase(0, pos);
        }
    }

    ///@brief Parse the new text and send what changed
    ///@param content the newly generated text
    ///@param generation_done true once the model has stopped
    void send_response(const std::string& content, bool generation_done) {
        if (!stream) {
            // start() was skipped; parse as if nothing was prefilled
            stream = std::make_unique<chat_stream>(auto_chat_engine->get_chat_format(), "", parse_options);
        }
        for (const chat_msg_delta& d : stream->update(content, generation_done)) {
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
                if (logprobs) {
                    choice_logprobs = {{"content", logprobs->take_content(d.content)}, {"refusal", nullptr}};
                }
            }

            json choice = {
                {"index", 0},
                {"delta", delta}
            };
            if (logprobs) {
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
            stream_callback("data: " + response.dump() + "\n\n", false);
        }
    }

    ///@brief Send the chat final response
    void send_final_response(chat_meta_info_t& meta_info) {
        json final_choice = {
            {"index", 0},
            {"delta", {
                {"content", nullptr}
            }}
        };
        if (logprobs) {
            final_choice["logprobs"] = nullptr;
        }
        final_choice["finish_reason"] = stop_reason_to_string(meta_info.stop_reason);
        json final_response = {
            {"id", stream_id},
            {"object", "chat.completion.chunk"},
            {"created", created},
            {"model", model_name},
            {"system_fingerprint", system_fingerprint},
            {"choices", json::array({final_choice})}
        };
        json usage = {
            {"prompt_tokens", meta_info.prompt_tokens},
            {"prompt_tokens_details", {{"cached_tokens", meta_info.cached_prompt_tokens}}},
            {"completion_tokens", meta_info.generated_tokens},
            {"total_tokens", meta_info.prompt_tokens + meta_info.generated_tokens},
            {"active_kv_tokens", this->auto_chat_engine->get_current_context_length()},
            {"max_kv_token_capacity", this->auto_chat_engine->get_max_length()},
            {"kv_token_occupancy_rate_percentage", (float)this->auto_chat_engine->get_current_context_length() / (float)this->auto_chat_engine->get_max_length() * 100},
            {"load_duration", static_cast<double>(meta_info.load_duration) / 1'000'000'000},
            {"prefill_duration_ttft", static_cast<double>(meta_info.prefill_duration) / 1'000'000'000},
            {"decoding_duration", static_cast<double>(meta_info.decoding_duration) / 1'000'000'000},
            {"prefill_speed_tps", static_cast<double>(meta_info.prompt_tokens - meta_info.cached_prompt_tokens) / static_cast<double>(meta_info.prefill_duration) * 1'000'000'000},
            {"decoding_speed_tps", static_cast<double>(meta_info.generated_tokens) / static_cast<double>(meta_info.decoding_duration) * 1'000'000'000},
        };
        if (meta_info.prefix_cache.valid) usage["flm_prefix_cache"] = prefix_cache_usage_json(meta_info.prefix_cache);
        if (include_usage) {
            // OpenAI: every chunk carries a null usage, then one extra chunk
            // with no choices carries the usage of the whole request
            final_response["usage"] = nullptr;
        }
        stream_callback("data: " + final_response.dump() + "\n\n", false);
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
            stream_callback("data: " + usage_response.dump() + "\n\n", false);
        }
        // the server log shows the usage whether or not the client asked for it
        std::cout << "ChatCompletionChunk: " << final_response << std::endl;
        std::cout << "Usage: " << usage << std::endl;
        // Send the [DONE] message
        stream_callback("data: [DONE]\n\n", true);
    }


    ///@brief Buffer
    std::string buffer;
    ///@brief Model name
    std::string model_name;
    ///@brief auto_chat_engine 
    AutoModel* auto_chat_engine;
    ///@brief Stream callback
    StreamCallback stream_callback;
    ///@brief Whether the client asked for usage, see stream_options.include_usage
    bool include_usage;
    ///@brief Stream ID
    std::string stream_id;
    long created;
    std::string system_fingerprint;
    ///@brief Per request parsing options
    chat_parse_options parse_options;
    ///@brief Parse state of this response, created by start()
    std::unique_ptr<chat_stream> stream;
    ///@brief Set by start() when the request asks for logprobs
    std::unique_ptr<chat_logprobs_aligner> logprobs;
    ///@brief What the model wrote since the last flush, for logprobs
    std::string logprobs_written;
};

///@brief Custom ostream for streaming chat completions
class streaming_ostream_openai_chat : public std::ostream {
public:
    streaming_ostream_openai_chat(const std::string& model, AutoModel* auto_chat_engine, streaming_buf_openai_chat::StreamCallback callback,
                                  chat_parse_options options, bool include_usage)
        : std::ostream(&buf), buf(model, auto_chat_engine, callback, options, include_usage) {}

    ///@brief Call once the prompt is inserted, before generating
    void start(const chat_meta_info_t& meta_info) {
        buf.start(meta_info);
    }

    ///@brief Finalize the chat
    void finalize(chat_meta_info_t& meta_info) {
        buf.finalize(meta_info);
    }

private:
    ///@brief Buffer
    streaming_buf_openai_chat buf;
};
