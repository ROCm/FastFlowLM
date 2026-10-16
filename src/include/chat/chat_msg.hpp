/// \file chat_msg.hpp
/// \brief Parsed assistant message and the deltas used to stream it
/// \note A message is rebuilt from the whole generated text on every update;
///       the parser guarantees each field only ever grows, so the difference
///       between two consecutive messages is always a plain suffix.
#pragma once

#include <string>
#include <vector>

/// \brief One function call requested by the model
struct chat_tool_call {
    std::string id;
    std::string name;
    std::string arguments; // JSON text, possibly incomplete while streaming

    bool operator==(const chat_tool_call& other) const {
        return id == other.id && name == other.name && arguments == other.arguments;
    }
};

/// \brief The assistant turn split into its OpenAI parts
struct chat_msg {
    std::string reasoning;
    std::string content;
    std::vector<chat_tool_call> tool_calls;

    bool empty() const {
        return reasoning.empty() && content.empty() && tool_calls.empty();
    }

    bool operator==(const chat_msg& other) const {
        return reasoning == other.reasoning && content == other.content && tool_calls == other.tool_calls;
    }
};

/// \brief What changed between two consecutive parses of the same stream
/// \note At most one of the three groups is set: reasoning text, content text,
///       or one tool call. The tool call carries id and name only in its first
///       delta, as OpenAI clients expect.
struct chat_msg_delta {
    std::string reasoning;
    std::string content;
    int tool_index = -1;
    std::string tool_id;
    std::string tool_name;
    std::string tool_arguments;

    bool is_tool_call() const {
        return tool_index >= 0;
    }
};

/// \brief Compute the deltas that turn prev into cur
/// \param prev the message already sent to the client
/// \param cur the message parsed from the longer text
/// \return the deltas in emission order: reasoning, content, then tool calls
/// \note A field of cur that does not extend the same field of prev is a parser
///       bug; it is reported once and the field is skipped rather than resent.
std::vector<chat_msg_delta> chat_compute_deltas(const chat_msg& prev, const chat_msg& cur);
