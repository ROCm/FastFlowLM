/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file streaming_ostream.hpp
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
#include <nlohmann/json.hpp>
#include "AutoModel/automodel.hpp"
#include "wstream_buf.hpp"
#include "harmony_filter.hpp"

using json = nlohmann::ordered_json;

///@brief Custom streambuf that captures tokens and sends them immediately
///@param model the model
///@param callback the callback
///@param is_chat_format the is chat format
///@return the streaming buf
///@note The packet is sent every std::flush, but only when UTF-8 sequences are complete
class streaming_buf : public std::streambuf {
public:
    ///@brief StreamCallback
    using StreamCallback = std::function<void(const json&, bool)>;
    
    streaming_buf(const std::string& model, StreamCallback callback, bool is_chat_format = false)
        : model_name(model), stream_callback(callback), is_chat(is_chat_format) {
            harmony_filter_inst = std::make_unique<harmony_filter>();
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
    void finalize_chat(chat_meta_info_t& meta_info) {
        drop_incomplete_utf8_tail();
        send_chat_final_response(meta_info);
    }
    ///@brief Call this when generation is complete
    ///@param context the context
    void finalize_generate(chat_meta_info_t& meta_info, std::vector<int>& context) {
        drop_incomplete_utf8_tail();
        send_generate_final_response(meta_info, context);
    }

private:
    ///@brief Send the remaining content but not an incomplete UTF-8 sequence at its end
    ///@note A generation cut off by the length limit or a cancel can stop partway
    ///      through a multi-byte character, the missing bytes will never come, and
    ///      invalid UTF-8 makes json::dump() throw
    void drop_incomplete_utf8_tail() {
        flush_complete_utf8_sequences(false);
        buffer.clear();
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
    int is_content = 0;
    int is_template = 0;
    void send_response(const std::string& content, bool is_final) {
        json response;
        
        std::string json_content = "";
        std::string json_reasoning = "";

        harmony_part_t part = harmony_filter_inst->identify_part(content);

        if (model_name == "gpt-oss:20b" || model_name == "gpt-oss" || model_name == "gpt-oss-sg:20b" || model_name == "gpt-oss-sg") {
            json_content = (part == harmony_part_t::response) ? content : "";
            json_reasoning = (part == harmony_part_t::reasoning) ? content : "";
        }
        else {
            json_content = content;
            json_reasoning = "";

        }

        if (is_chat) {
            response = {
                {"model", model_name},
                {"message", {
                    {"role", "assistant"},
                    {"content", json_content},
                    {"thinking", json_reasoning}
                }},
                {"done", is_final}
            };
        } else {
            response = {
                {"model", model_name},
                {"response", json_content},
                {"thinking", json_reasoning},
                {"done", is_final}
            };
        }
        
        stream_callback(response, is_final);
    }

    ///@brief Send the chat final response
    void send_chat_final_response(chat_meta_info_t& meta_info) {
        json response;
        
        response = {
            {"model", model_name},
            {"message", {
                {"role", "assistant"},
                {"content", ""}
            }},
            {"done", true},
            {"done_reason", stop_reason_to_string(meta_info.stop_reason)},
            {"prompt_eval_count", meta_info.prompt_tokens},
            {"eval_count", meta_info.generated_tokens},
            {"total_duration", meta_info.total_duration},
            {"load_duration", meta_info.load_duration},
            {"prompt_eval_duration", meta_info.prefill_duration},
            {"eval_duration", meta_info.decoding_duration}
        };
        
        stream_callback(response, true);
    }
    
    ///@brief Send the generate final response
    ///@param content the content
    void send_generate_final_response(chat_meta_info_t& meta_info, const std::vector<int>& content) {
        json response;
        
        response = {
            {"model", model_name},
            {"response", ""},
            {"context", content},
            {"prompt_eval_count", meta_info.prompt_tokens},
            {"eval_count", meta_info.generated_tokens},
            {"total_duration", meta_info.total_duration},
            {"load_duration", meta_info.load_duration},
            {"prompt_eval_duration", meta_info.prefill_duration},
            {"eval_duration", meta_info.decoding_duration},
            {"done_reason", stop_reason_to_string(meta_info.stop_reason)},
            {"done", true}
        };
        
        stream_callback(response, true);
    }
    ///@brief Buffer
    std::string buffer;
    ///@brief Model name
    std::string model_name;
    ///@brief Stream callback
    StreamCallback stream_callback;
    ///@brief Is chat
    bool is_chat;
    std::unique_ptr<harmony_filter> harmony_filter_inst;
};

///@brief Custom ostream for streaming
///@param model the model
///@param callback the callback
///@param is_chat_format the is chat format
///@return the streaming ostream
class streaming_ostream : public std::ostream {
public:
    streaming_ostream(const std::string& model, streaming_buf::StreamCallback callback, bool is_chat_format = false)
        : std::ostream(&buf), buf(model, callback, is_chat_format) {}
    
    ///@brief Finalize the chat
    void finalize_chat(chat_meta_info_t& meta_info) {
        buf.finalize_chat(meta_info);
    }
    ///@brief Finalize the generate
    ///@param context the context
    void finalize_generate(chat_meta_info_t& meta_info, std::vector<int>& context) {
        buf.finalize_generate(meta_info, context);
    }

private:
    ///@brief Buffer
    streaming_buf buf;
};