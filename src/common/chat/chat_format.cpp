/// \file chat_format.cpp
/// \brief Chat format detection and prefill computation

#include "chat/chat_format.hpp"

#include <cctype>

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
    // a template that renders <function=...><parameter=...> is Qwen-style XML even when its vocabulary also carries
    // LFM2's <|tool_call_start|> (LFM3 does: the token is inherited, the model calls tools with <tool_call>)
    if (in_template("<function=") && in_template("<parameter=")) {
        return chat_format_make(chat_format_kind::QWEN_XML, with_reasoning);
    }
    if (known("<|tool_call_start|>")) {
        return chat_format_make(chat_format_kind::LFM2_PYTHONIC, with_reasoning);
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

chat_call_head chat_format_call_head(const chat_format_spec& spec, const std::string& text, size_t pos) {
    static const std::vector<std::string> QWEN_XML_OPENERS = {"<function="};
    static const std::vector<std::string> HERMES_OPENERS = {"{", "["};
    const std::vector<std::string>* openers = spec.kind == chat_format_kind::QWEN_XML    ? &QWEN_XML_OPENERS
                                              : spec.kind == chat_format_kind::HERMES_JSON ? &HERMES_OPENERS
                                                                                           : nullptr;
    if (openers == nullptr) {
        return chat_call_head::CALL;
    }
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r')) {
        pos++;
    }
    const size_t avail = text.size() - pos;
    if (avail == 0) {
        return chat_call_head::UNSURE;
    }
    if (spec.kind == chat_format_kind::HERMES_JSON && text[pos] == '[') {
        // a list of calls is a list of objects; "[docs]" is prose
        return chat_format_call_head(spec, text, pos + 1);
    }
    bool unsure = false;
    for (const auto& opener : *openers) {
        if (text.compare(pos, opener.size(), opener) == 0) {
            return chat_call_head::CALL;
        }
        unsure = unsure || (avail < opener.size() && opener.compare(0, avail, text, pos, avail) == 0);
    }
    return unsure ? chat_call_head::UNSURE : chat_call_head::PROSE;
}

size_t chat_format_tool_end(const chat_format_spec& spec, const std::string& text, size_t from, bool is_partial,
                            size_t& hold) {
    static const std::string FUNCTION_CLOSE = "</function>";
    hold = std::string::npos;
    const size_t first = text.find(spec.tool_end, from);
    if (spec.kind != chat_format_kind::QWEN_XML || first == std::string::npos) {
        return first;
    }
    // look no further than the next call
    size_t next_call = first;
    chat_call_head head = chat_call_head::PROSE;
    while (head == chat_call_head::PROSE && (next_call = text.find(spec.tool_start, next_call)) != std::string::npos) {
        next_call += spec.tool_start.size();
        head = chat_format_call_head(spec, text, next_call);
    }
    const bool settled = head == chat_call_head::CALL || !is_partial;
    for (size_t end = first; end != std::string::npos && (end < next_call || next_call == std::string::npos);
         end = text.find(spec.tool_end, end + 1)) {
        // at most one whitespace character between, as the grammar has it
        size_t e = end > from && std::isspace(static_cast<unsigned char>(text[end - 1])) ? end - 1 : end;
        if (e >= from + FUNCTION_CLOSE.size() &&
            text.compare(e - FUNCTION_CLOSE.size(), FUNCTION_CLOSE.size(), FUNCTION_CLOSE) == 0) {
            return end;
        }
    }
    if (!settled) {
        hold = first;
        return std::string::npos;
    }
    return first;
}

namespace {

/// \brief The markup of a tool call the prompt opened and did not close
std::string open_tool_call(const chat_format_spec& spec, const std::string& prompt_tail) {
    if (spec.tool_start.empty()) {
        return "";
    }
    size_t open = prompt_tail.rfind(spec.tool_start);
    if (open == std::string::npos ||
        chat_format_call_head(spec, prompt_tail, open + spec.tool_start.size()) == chat_call_head::PROSE) {
        return "";
    }
    // the message goes on, so an end tag that only might close the call does not
    size_t hold = std::string::npos;
    if (chat_format_tool_end(spec, prompt_tail, open + spec.tool_start.size(), true, hold) != std::string::npos) {
        return "";
    }
    return prompt_tail.substr(open);
}

} // namespace

std::string chat_format_prefill(const chat_format_spec& spec, const std::string& prompt_tail, bool continuing,
                                bool tool_calls) {
    if (spec.kind == chat_format_kind::HARMONY) {
        // the generation prompt ends with an open "<|start|>assistant" header, and
        // may have started a channel too; the header is parsed with the output
        size_t start = prompt_tail.rfind("<|start|>");
        return start == std::string::npos ? std::string() : prompt_tail.substr(start + 9);
    }
    // a continued message may stop in the middle of a call, which the output finishes
    const std::string call = continuing && tool_calls ? open_tool_call(spec, prompt_tail) : "";
    if (!spec.has_reasoning()) {
        return call;
    }
    // a reasoning block opened by the template and not closed yet is where the
    // output starts; one closed by the template (thinking disabled) is not
    size_t open = prompt_tail.rfind(spec.reasoning_start);
    size_t close = prompt_tail.rfind(spec.reasoning_end);
    if (open == std::string::npos || (close != std::string::npos && close > open)) {
        // a call quoted in the reasoning is not one
        if (!call.empty() && (close == std::string::npos || prompt_tail.size() - call.size() > close)) {
            return call;
        }
        return spec.reasoning_forced && !continuing ? spec.reasoning_start : "";
    }
    // only the tag: what follows it in the prompt is whitespace, and some
    // decoders render that as byte tokens like <0x0A>
    return spec.reasoning_start;
}
