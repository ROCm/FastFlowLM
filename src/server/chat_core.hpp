/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file chat_core.hpp
 * \brief What one chat turn tells the protocol it is served through
 * \author FastFlowLM Team
 * \note A chat turn is run once (RestHandler::run_chat_turn) and reported as events with no
 *       protocol's field names in them. A chat_sink turns the events into one protocol's
 *       response: its stream frames, its final object, its error object. Nothing here needs a
 *       loaded model, so a sink can be tested on a host with a list of deltas.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <streambuf>
#include <string>
#include <utility>
#include <nlohmann/json.hpp>
#include "chat/chat_format.hpp"
#include "chat/chat_msg.hpp"
#include "chat/chat_parser.hpp"
#include "chat/chat_stream.hpp"
#include "prefix_slot_cache.hpp"
#include "wstream_buf.hpp"

/// \brief Why a request failed, before any protocol has put it into its error object
struct chat_error {
    int status = 500;                     ///< the HTTP status that goes with it
    std::string type = "server_error";    ///< invalid_request_error, server_error, model_error
    std::string message;
    std::string param;                    ///< the request field at fault, empty when there is none
    std::optional<bool> session_cleared;  ///< set by a model error: whether the conversation was dropped
};

/// \brief Why the turn ended
enum class chat_finish {
    STOP,        ///< the model ended its turn
    LENGTH,      ///< the token limit or the context was reached
    FAILED,      ///< the model or the engine failed
    CANCEL,      ///< the client went away
    TOOL_CALLS,  ///< the model ended its turn with tool calls
};

/// \brief The request is valid and its model is loaded; nothing has been fed to the model yet
struct chat_turn_begin {
    std::string model;
    bool stream = false;  ///< the client wants the turn as it is generated
};

/// \brief The prompt is in the model; generation starts
struct chat_turn_start {
    int prompt_tokens = 0;         ///< whole prompt, cached prefix included
    int cached_prompt_tokens = 0;  ///< the part of it that was reused
    /// \brief The logprobs entries of the tokens a piece of content was made from, in order;
    ///        empty when the request asked for no logprobs
    /// \note It reads the turn's own state: call it only until on_end or on_error, and drop it there
    std::function<nlohmann::ordered_json(const std::string&)> content_logprobs;
};

/// \brief The turn is over
struct chat_turn_end {
    chat_finish finish = chat_finish::STOP;
    /// \brief false when the turn was cancelled while the prompt was being fed: there is no message
    bool generated = true;
    /// \brief the whole assistant message, tool call ids filled in; never null
    const chat_msg* message = nullptr;
    int prompt_tokens = 0;
    int cached_prompt_tokens = 0;
    int generated_tokens = 0;
    uint64_t load_duration = 0;      ///< nanoseconds
    uint64_t prefill_duration = 0;   ///< nanoseconds
    uint64_t decoding_duration = 0;  ///< nanoseconds
    int context_tokens = 0;              ///< tokens the model's context holds after the turn
    unsigned int context_capacity = 0;   ///< tokens it can hold
    /// \brief what the prefix slot cache did, null for a model that has none
    const prefix_cache_report_t* prefix_cache = nullptr;
};

/// \brief One protocol's way of writing a chat turn to its client
/// \note The calls of one turn: on_begin, on_start, any number of on_delta (only when the turn
///       is streamed), on_end. A turn cancelled during the prompt goes from on_begin to on_end.
///       on_error can come at any point and ends the turn. It can also come after on_end or after
///       another on_error, when cleaning up after the turn fails, or when the sink's own on_error
///       threw: a sink has to take that too, and must not call content_logprobs any more.
class chat_sink {
public:
    virtual ~chat_sink() = default;
    virtual void on_begin(const chat_turn_begin& begin) = 0;
    virtual void on_start(const chat_turn_start& start) = 0;
    virtual void on_delta(const chat_msg_delta& delta) = 0;
    virtual void on_end(const chat_turn_end& end) = 0;
    virtual void on_error(const chat_error& error) = 0;
};

/// \brief The stream a model writes a streamed turn to: parses the text and hands the sink what changed
/// \note A packet goes to the parser at every std::flush, but only whole UTF-8 characters.
///       Every packet is parsed with the model's chat format; reasoning, content and
///       tool call arguments go out as the deltas between consecutive parses.
class chat_turn_streambuf : public std::streambuf {
public:
    ///@param spec the model's chat format; must outlive the stream
    ///@param options per request parsing options; the tools must outlive the stream
    ///@param sink receives every delta
    chat_turn_streambuf(const chat_format_spec& spec, chat_parse_options options, chat_sink& sink)
        : spec(&spec), parse_options(options), sink(sink) {}

    ///@brief Call this once the prompt is inserted, before generating
    ///@param prefill the prompt part the parser must see first (chat_format_prefill)
    ///@param flush_hook told what the model wrote since the last flush, before the parser sees it;
    ///       the request's logprobs are placed there. May be empty.
    void start(std::string prefill, std::function<void(const std::string&)> flush_hook = nullptr) {
        stream = std::make_unique<chat_stream>(*spec, std::move(prefill), parse_options);
        on_flush = std::move(flush_hook);
    }

    ///@brief Call this when generation is complete
    ///@return the whole message, tool call ids filled in
    const chat_msg& finish() {
        flush_written();
        // Send the remaining content but not an incomplete UTF-8 sequence at
        // its end: a generation cut off by the length limit or a cancel can
        // stop partway through a multi-byte character, the missing bytes will
        // never come, and invalid UTF-8 makes json::dump() throw
        flush_complete_utf8_sequences();
        buffer.clear();
        // Let the parser release whatever it held back
        send_deltas("", true);
        return stream->message();
    }

protected:
    ///@brief Called when buffer is full or flush is requested
    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof()) {
            buffer += static_cast<char>(ch);
            if (on_flush) {
                written += static_cast<char>(ch);
            }
        }
        return ch;
    }

    ///@brief Called when stream is flushed
    int sync() override {
        flush_written();
        flush_complete_utf8_sequences();
        return 0;
    }

private:
    ///@brief Report the text written since the last flush, before the parser sees it
    void flush_written() {
        if (on_flush) {
            on_flush(written);
            written.clear();
        }
    }

    ///@brief Flush only complete UTF-8 sequences
    void flush_complete_utf8_sequences() {
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
            send_deltas(complete_content, false);
        }

        // Remove processed bytes from buffer
        if (pos > 0) {
            buffer.erase(0, pos);
        }
    }

    ///@brief Parse the new text and send what changed
    ///@param content the newly generated text
    ///@param generation_done true once the model has stopped
    void send_deltas(const std::string& content, bool generation_done) {
        if (!stream) {
            // start() was skipped; parse as if nothing was prefilled
            stream = std::make_unique<chat_stream>(*spec, "", parse_options);
        }
        for (const chat_msg_delta& d : stream->update(content, generation_done)) {
            sink.on_delta(d);
        }
    }

    ///@brief What the model wrote and the parser has not seen yet
    std::string buffer;
    ///@brief The model's chat format
    const chat_format_spec* spec;
    ///@brief Per request parsing options
    chat_parse_options parse_options;
    ///@brief Where the deltas go
    chat_sink& sink;
    ///@brief Parse state of this response, created by start()
    std::unique_ptr<chat_stream> stream;
    ///@brief Set by start() when the request asks for logprobs
    std::function<void(const std::string&)> on_flush;
    ///@brief What the model wrote since the last flush, for on_flush
    std::string written;
};

///@brief The ostream over chat_turn_streambuf
class chat_turn_ostream : public std::ostream {
public:
    chat_turn_ostream(const chat_format_spec& spec, chat_parse_options options, chat_sink& sink)
        : std::ostream(&buf), buf(spec, options, sink) {}

    ///@brief Call once the prompt is inserted, before generating
    void start(std::string prefill, std::function<void(const std::string&)> on_flush = nullptr) {
        buf.start(std::move(prefill), std::move(on_flush));
    }

    ///@brief Call when generation is complete; the whole message
    const chat_msg& finish() {
        return buf.finish();
    }

private:
    chat_turn_streambuf buf;
};
