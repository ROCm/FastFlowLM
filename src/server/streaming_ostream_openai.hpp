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
#include "wstream_buf.hpp"

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

    ///@brief Flush only complete UTF-8 sequences
    ///@param is_final the is final
    void flush_complete_utf8_sequences(bool is_final) {
        if (buffer.empty()) return;
        
        std::string complete_content;
        size_t pos = 0;
        
        // Process complete UTF-8 sequences
        while (pos < buffer.size()) {
            const int len = utf8_char_len(reinterpret_cast<const unsigned char*>(buffer.data()) + pos, buffer.size() - pos);
            if (len == 0) {
                // Not part of a well-formed character (e.g. a character's first byte followed by
                // another token): skip it, json::dump() would throw on it
                pos++;
                continue;
            }
            if (len < 0) {
                // Incomplete sequence, stop here
                break;
            }
            const size_t seq_len = (size_t)len;
            
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
