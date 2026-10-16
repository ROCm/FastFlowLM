/// \file chat_stream.hpp
/// \brief Per request streaming state: accumulate, re-parse, send the difference
#pragma once

#include <string>
#include <vector>

#include "chat/chat_format.hpp"
#include "chat/chat_msg.hpp"
#include "chat/chat_parser.hpp"

/// \brief Tracks one streamed assistant turn
/// \note The whole text is parsed again on every update. That keeps one code
///       path for streaming and non-streaming, and costs a scan of a few KB per
///       token, which is noise next to a decode step.
class chat_stream {
public:
    /// \param spec the model's format
    /// \param prefill the prompt part the parser must see first (chat_format_prefill)
    /// \param options per request options; the tools pointer must outlive the stream
    chat_stream(const chat_format_spec& spec, std::string prefill, chat_parse_options options);

    /// \brief Feed newly generated text
    /// \param text the new text, may be empty
    /// \param is_final true once generation has stopped
    /// \return what changed since the previous update
    std::vector<chat_msg_delta> update(const std::string& text, bool is_final);

    /// \brief The message as parsed so far, tool call ids filled in
    const chat_msg& message() const {
        return msg_;
    }

private:
    chat_format_spec spec_;
    chat_parse_options options_;
    std::string text_;
    chat_msg msg_;
    std::vector<std::string> tool_ids_;
};

/// \brief A fresh tool call id in the OpenAI style, "call_" and 24 random characters
std::string chat_new_tool_call_id();
