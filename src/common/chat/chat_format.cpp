/// \file chat_format.cpp
/// \brief Chat format detection and prefill computation

#include "chat/chat_format.hpp"

namespace {

struct format_name_entry {
    chat_format_kind kind;
    const char* name;
};

const format_name_entry FORMAT_NAMES[] = {
    {chat_format_kind::CONTENT_ONLY, "content_only"},
    {chat_format_kind::HERMES_JSON, "hermes_json"},
    {chat_format_kind::QWEN_XML, "qwen_xml"},
    {chat_format_kind::LFM2_PYTHONIC, "lfm2_pythonic"},
    {chat_format_kind::GEMMA4, "gemma4"},
    {chat_format_kind::HARMONY, "harmony"},
};

} // namespace

const char* chat_format_name(chat_format_kind kind) {
    for (const auto& entry : FORMAT_NAMES) {
        if (entry.kind == kind) {
            return entry.name;
        }
    }
    return "unknown";
}

bool chat_format_from_name(const std::string& name, chat_format_kind& kind) {
    for (const auto& entry : FORMAT_NAMES) {
        if (name == entry.name) {
            kind = entry.kind;
            return true;
        }
    }
    return false;
}

chat_format_spec chat_format_make(chat_format_kind kind, bool with_reasoning) {
    chat_format_spec spec;
    spec.kind = kind;
    if (with_reasoning) {
        spec.reasoning_start = "<think>";
        spec.reasoning_end = "</think>";
    }
    switch (kind) {
        case chat_format_kind::CONTENT_ONLY:
            break;
        case chat_format_kind::HERMES_JSON:
        case chat_format_kind::QWEN_XML:
            spec.tool_start = "<tool_call>";
            spec.tool_end = "</tool_call>";
            break;
        case chat_format_kind::LFM2_PYTHONIC:
            spec.tool_start = "<|tool_call_start|>";
            spec.tool_end = "<|tool_call_end|>";
            break;
        case chat_format_kind::GEMMA4:
            // Gemma4 always reasons in its own channel, never in <think>
            spec.reasoning_start = "<|channel>thought";
            spec.reasoning_end = "<channel|>";
            spec.tool_start = "<|tool_call>";
            spec.tool_end = "<tool_call|>";
            // the model hands the turn to the tool here; anything after is not its answer
            spec.stop_marker = "<|tool_response>";
            spec.extra_stops = {"<|tool_response>"};
            break;
        case chat_format_kind::HARMONY:
            // channels, not tags; the harmony parser knows its own markers
            spec.reasoning_start.clear();
            spec.reasoning_end.clear();
            // a call ends the turn, the tool answers next; gpt-oss only lists <|return|> as eos
            spec.extra_stops = {"<|call|>"};
            break;
    }
    return spec;
}

chat_format_spec chat_format_detect(const std::string& template_src,
                                    const std::function<bool(const std::string&)>& is_token) {
    auto in_template = [&template_src](const char* marker) {
        return template_src.find(marker) != std::string::npos;
    };
    auto known = [&](const char* marker) {
        return in_template(marker) || is_token(marker);
    };

    const bool with_reasoning = known("<think>") && known("</think>");

    // most specific first: these markers are unique to one family
    if (known("<|channel|>") && known("<|message|>")) {
        return chat_format_make(chat_format_kind::HARMONY, false);
    }
    if (known("<|tool_call>") && known("<|channel>")) {
        return chat_format_make(chat_format_kind::GEMMA4, false);
    }
    if (known("<|tool_call_start|>")) {
        return chat_format_make(chat_format_kind::LFM2_PYTHONIC, with_reasoning);
    }
    if (in_template("<function=") && in_template("<parameter=")) {
        return chat_format_make(chat_format_kind::QWEN_XML, with_reasoning);
    }
    // only trust <tool_call> when the template renders it: some vocabularies carry
    // the token without the model ever being trained to call tools with it
    if (in_template("<tool_call>")) {
        return chat_format_make(chat_format_kind::HERMES_JSON, with_reasoning);
    }
    chat_format_spec spec = chat_format_make(chat_format_kind::CONTENT_ONLY, with_reasoning);
    // DeepSeek-R1 and its distills were trained to always open with reasoning;
    // after a tool result the template does not open <think>, yet they reason anyway
    spec.reasoning_forced = with_reasoning && in_template("<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>");
    return spec;
}

std::string chat_format_prefill(const chat_format_spec& spec, const std::string& prompt_tail) {
    if (spec.kind == chat_format_kind::HARMONY) {
        // the generation prompt ends with an open "<|start|>assistant" header, and
        // may have started a channel too; the header is parsed with the output
        size_t start = prompt_tail.rfind("<|start|>");
        return start == std::string::npos ? std::string() : prompt_tail.substr(start + 9);
    }
    if (!spec.has_reasoning()) {
        return "";
    }
    // a reasoning block opened by the template and not closed yet is where the
    // output starts; one closed by the template (thinking disabled) is not
    size_t open = prompt_tail.rfind(spec.reasoning_start);
    size_t close = prompt_tail.rfind(spec.reasoning_end);
    if (open == std::string::npos || (close != std::string::npos && close > open)) {
        return spec.reasoning_forced ? spec.reasoning_start : "";
    }
    // only the tag: what follows it in the prompt is whitespace, and some
    // decoders render that as byte tokens like <0x0A>
    return spec.reasoning_start;
}
