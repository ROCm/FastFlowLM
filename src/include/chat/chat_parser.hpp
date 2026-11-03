/// \file chat_parser.hpp
/// \brief Turn a model's raw output into reasoning, content and tool calls
/// \note One parser serves both the streaming and the non-streaming path. The
///       stream re-parses the whole text on every update and sends the
///       difference, which is safe because a partial parse only ever yields a
///       prefix of what the full parse yields:
///         - a marker that may still be forming at the end of the text is held
///           back instead of being shown as text,
///         - reasoning and content are trimmed, so trailing whitespace waits
///           until something follows it,
///         - a tool call appears once its name is complete, its arguments grow
///           as JSON text (see chat_args.hpp).
#pragma once

#include <string>

#include <nlohmann/json.hpp>

#include "chat/chat_format.hpp"
#include "chat/chat_msg.hpp"

/// \brief Per request parsing options
struct chat_parse_options {
    /// \brief the request's tools, used to type the arguments of formats that
    ///        do not say whether a value is a string; may be null
    const nlohmann::ordered_json* tools = nullptr;
    /// \brief extract tool calls; when false the tool markup stays in the content
    bool parse_tool_calls = true;
    /// \brief keep the whitespace around the output's first text, as it was written
    /// \note For continue_final_message: the output continues text the client
    ///       already holds, so its whitespace (" yes" after "The answer is:", or a
    ///       lone "\n") belongs to the answer rather than to the layout. Trailing
    ///       whitespace is held back until the text is final, like any other.
    bool keep_whitespace = false;
    /// \brief the output may close a reasoning block the prompt never opened
    /// \note For continue_final_message on a model that reasons first: such a
    ///       model reads the continued message as its reasoning, so a reasoning
    ///       end it writes before any reasoning start closes that, and the text
    ///       before it is reasoning (as vLLM's DeepSeek-R1 parser has it). Until
    ///       either marker shows, a partial parse yields nothing.
    bool reasoning_may_close = false;
};

/// \brief Parse the assistant output
/// \param spec the model's format
/// \param text the prefill (see chat_format_prefill) followed by the generated text
/// \param is_partial true while the generation is still running
/// \param options per request options
/// \return the parsed message; tool call ids are left empty for the caller
chat_msg chat_parse(const chat_format_spec& spec, const std::string& text, bool is_partial,
                    const chat_parse_options& options);
