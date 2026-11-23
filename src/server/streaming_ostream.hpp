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
#include "chat/chat_stream.hpp"

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
    
    ///@param model the model name reported to the client
    ///@param auto_chat_engine the engine generating the text; its chat format splits the reasoning off
    ///@param callback receives each packet
    ///@param is_chat_format true for /api/chat, false for /api/generate
    streaming_buf(const std::string& model, AutoModel* auto_chat_engine, StreamCallback callback, bool is_chat_format = false)
        : model_name(model), auto_chat_engine(auto_chat_engine), stream_callback(callback), is_chat(is_chat_format) {
        // Ollama's routes have no tools here (WP-21): tool markup stays in the content
        parse_options.parse_tool_calls = false;
    }

    ///@brief Call this once the prompt is inserted, before generating
    ///@param meta_info the request's meta info, its prompt tail tells where the output starts
    void start(const chat_meta_info_t& meta_info) {
        const chat_format_spec& spec = auto_chat_engine->get_chat_format();
        stream = std::make_unique<chat_stream>(
            spec, chat_format_prefill(spec, meta_info.prompt_tail, meta_info.continue_final_message, parse_options.parse_tool_calls),
            parse_options);
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
        send_response("", true);  // let the parser release what it held back
        send_chat_final_response(meta_info);
    }
    ///@brief Call this when generation is complete
    ///@param context the context
    void finalize_generate(chat_meta_info_t& meta_info, std::vector<int>& context) {
        drop_incomplete_utf8_tail();
        send_response("", true);  // let the parser release what it held back
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
    ///@note The reasoning goes out in `thinking` and the answer in `content` / `response`,
    ///      as the model's chat format splits them; an empty `thinking` is left out (Ollama's omitempty)
    void send_response(const std::string& content, bool generation_done) {
        if (!stream) {
            // start() was skipped; parse as if nothing was prefilled
            stream = std::make_unique<chat_stream>(auto_chat_engine->get_chat_format(), "", parse_options);
        }
        for (const chat_msg_delta& d : stream->update(content, generation_done)) {
            send_packet(d.content, d.reasoning);
        }
    }

    ///@brief Send one packet
    void send_packet(const std::string& json_content, const std::string& json_reasoning) {
        if (json_content.empty() && json_reasoning.empty()) return;
        json response;
        if (is_chat) {
            json message = {{"role", "assistant"}, {"content", json_content}};
            if (!json_reasoning.empty()) message["thinking"] = json_reasoning;
            response = {
                {"model", model_name},
                {"message", message},
                {"done", false}
            };
        } else {
            response = {
                {"model", model_name},
                {"response", json_content}
            };
            if (!json_reasoning.empty()) response["thinking"] = json_reasoning;
            response["done"] = false;
        }
        stream_callback(response, false);
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
    ///@brief The engine, for its chat format
    AutoModel* auto_chat_engine;
    ///@brief Is chat
    bool is_chat;
    ///@brief Parsing options
    chat_parse_options parse_options;
    ///@brief Per request parser state
    std::unique_ptr<chat_stream> stream;
};

///@brief Custom ostream for streaming
///@param model the model
///@param callback the callback
///@param is_chat_format the is chat format
///@return the streaming ostream
class streaming_ostream : public std::ostream {
public:
    streaming_ostream(const std::string& model, AutoModel* auto_chat_engine, streaming_buf::StreamCallback callback, bool is_chat_format = false)
        : std::ostream(&buf), buf(model, auto_chat_engine, callback, is_chat_format) {}

    ///@brief Call this once the prompt is inserted, before generating
    void start(const chat_meta_info_t& meta_info) {
        buf.start(meta_info);
    }
    
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