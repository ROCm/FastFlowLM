/// \file chat_format.hpp
/// \brief How a model's chat template lays out reasoning and tool calls
/// \note Detected once per model from the template source and the tokenizer's
///       special tokens; the output parser and the decoding constraints are both
///       derived from it, so they can never disagree about the syntax.
#pragma once

#include <functional>
#include <string>
#include <vector>

/// \brief Tool call syntax families
enum class chat_format_kind {
    CONTENT_ONLY,   // no tool calls, optionally <think> reasoning
    HERMES_JSON,    // <tool_call>{"name": .., "arguments": {..}}</tool_call>          (Qwen3, Qwen2.5, Nanbeige)
    QWEN_XML,       // <tool_call><function=x><parameter=k>v</parameter></function></tool_call>  (Qwen3.5+)
    LFM2_PYTHONIC,  // <|tool_call_start|>[f(a=1, b="x")]<|tool_call_end|>               (LFM2)
    GEMMA4,         // <|tool_call>call:f{a:<|"|>x<|"|>}<tool_call|>                     (Gemma4)
    HARMONY,        // <|channel|>analysis / commentary to=functions.f / final           (gpt-oss)
};

/// \brief Everything the parser needs to know about one model's output syntax
struct chat_format_spec {
    chat_format_kind kind = chat_format_kind::CONTENT_ONLY;
    std::string reasoning_start; // empty when the model has no reasoning block
    std::string reasoning_end;
    std::string tool_start;      // opens one tool call block, empty for CONTENT_ONLY and HARMONY
    std::string tool_end;
    std::string stop_marker;     // text after it is never part of the turn, e.g. Gemma4's <|tool_response>
    bool reasoning_forced = false; // the model always opens its turn by reasoning, tag or not (DeepSeek-R1)
    std::vector<std::string> extra_stops; // markers that end the turn but are not in the model's eos list

    bool has_reasoning() const {
        return !reasoning_end.empty();
    }
    bool has_tools() const {
        return kind != chat_format_kind::CONTENT_ONLY;
    }
};

/// \brief Stable name of a format kind, as used in logs and model_list.json
const char* chat_format_name(chat_format_kind kind);

/// \brief Look a kind up by its name
/// \return false when the name is unknown
bool chat_format_from_name(const std::string& name, chat_format_kind& kind);

/// \brief Build the spec of a known kind with its standard markers
/// \param kind the format kind
/// \param with_reasoning whether to recognise <think> blocks (tag based kinds only)
chat_format_spec chat_format_make(chat_format_kind kind, bool with_reasoning);

/// \brief Detect the format from the chat template and the tokenizer
/// \param template_src the jinja chat template source
/// \param is_token true if the string is a single token in the model's vocabulary
/// \return the detected spec, CONTENT_ONLY when nothing matches
/// \note The tokenizer is consulted because some templates never render the
///       markers the model emits, e.g. LFM2.5-Thinking prints tool calls as plain
///       content and the R1 distills never mention <think>.
chat_format_spec chat_format_detect(const std::string& template_src,
                                    const std::function<bool(const std::string&)>& is_token);

/// \brief The part of the prompt the parser has to see before the generated text
/// \param spec the model's format
/// \param prompt_tail the decoded end of the prompt as fed to the model
/// \return text to prepend to the output before parsing, e.g. "<think>" when the
///         template already opened the reasoning block
std::string chat_format_prefill(const chat_format_spec& spec, const std::string& prompt_tail);
