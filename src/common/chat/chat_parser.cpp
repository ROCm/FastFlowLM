/// \file chat_parser.cpp
/// \brief Output parser for every chat format

#include "chat/chat_parser.hpp"

#include <algorithm>
#include <initializer_list>
#include <vector>

#include "chat/chat_args.hpp"

namespace {

using json = nlohmann::ordered_json;

constexpr size_t npos = std::string::npos;

bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::string trim(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && is_ws(s[b])) b++;
    while (e > b && is_ws(s[e - 1])) e--;
    return s.substr(b, e - b);
}

std::string trim_end(const std::string& s) {
    size_t e = s.size();
    while (e > 0 && is_ws(s[e - 1])) e--;
    return s.substr(0, e);
}

size_t skip_ws(const std::string& s, size_t pos) {
    while (pos < s.size() && is_ws(s[pos])) pos++;
    return pos;
}

bool starts_with_at(const std::string& s, size_t pos, const std::string& prefix) {
    return s.compare(pos, prefix.size(), prefix) == 0;
}

/// \brief Drop a tail of s that could be the start of one of the markers
/// \note The tail is held back, not lost: the next parse sees more text and
///       either completes the marker or shows the tail as text.
std::string cut_partial_marker(const std::string& s, std::initializer_list<const std::string*> markers) {
    size_t cut = 0;
    for (const std::string* marker : markers) {
        if (marker->empty()) {
            continue;
        }
        size_t max_len = std::min(s.size(), marker->size() - 1);
        for (size_t len = max_len; len > cut; len--) {
            if (s.compare(s.size() - len, len, *marker, 0, len) == 0) {
                cut = len;
                break;
            }
        }
    }
    return s.substr(0, s.size() - cut);
}

/// \brief Index of the first marker found from pos, and which one it is
size_t find_first(const std::string& s, size_t pos, std::initializer_list<const std::string*> markers,
                  const std::string** which) {
    size_t best = npos;
    for (const std::string* marker : markers) {
        if (marker->empty()) {
            continue;
        }
        size_t at = s.find(*marker, pos);
        if (at < best) {
            best = at;
            *which = marker;
        }
    }
    return best;
}

/// \brief Collects the text parts of one kind and joins them
/// \note Each part is trimmed and empty parts are dropped. Trimming keeps the
///       prefix property: trailing whitespace only shows once text follows it.
///       The first text of a continued message is the exception: its whitespace
///       belongs to the answer, so it is kept, the trailing part once the text is final.
struct text_parts {
    struct part_t {
        std::string raw;
        bool exact; // the first text of a continued message
    };
    std::vector<part_t> parts;

    void add(const std::string& raw, bool exact = false) {
        parts.push_back({raw, exact});
    }

    /// \param is_final no more text follows, so trailing whitespace can be released
    std::string join(bool is_final = false) const {
        const bool others = std::any_of(parts.begin(), parts.end(),
                                        [](const part_t& p) { return !p.exact && !trim(p.raw).empty(); });
        std::string out;
        for (size_t i = 0; i < parts.size(); i++) {
            const auto& [raw, exact] = parts[i];
            // text after it settles its trailing whitespace as surely as the end does
            const bool followed = std::any_of(parts.begin() + i + 1, parts.end(),
                                              [](const part_t& p) { return !trim(p.raw).empty(); });
            std::string part = !exact ? trim(raw) : (is_final || followed) ? raw : trim_end(raw);
            // whitespace alone is the answer only when nothing else is
            if (part.empty() || (exact && others && trim(part).empty())) {
                continue;
            }
            if (!out.empty()) {
                out += "\n\n";
            }
            out += part;
        }
        return out;
    }
};

/// \brief JSON schema types of tool parameters, from the request's tools
class param_types {
public:
    explicit param_types(const json* tools) : tools_(tools) {}

    /// \return the declared type, or empty when the schema does not say
    std::string of(const std::string& tool, const std::string& param) const {
        if (tools_ == nullptr || !tools_->is_array()) {
            return "";
        }
        for (const auto& entry : *tools_) {
            const json& fn = entry.contains("function") ? entry["function"] : entry;
            if (!fn.is_object() || fn.value("name", "") != tool) {
                continue;
            }
            if (!fn.contains("parameters") || !fn["parameters"].contains("properties")) {
                return "";
            }
            const json& props = fn["parameters"]["properties"];
            if (!props.is_object() || !props.contains(param)) {
                return "";
            }
            const json& type = props[param].contains("type") ? props[param]["type"] : json();
            if (type.is_string()) {
                return type.get<std::string>();
            }
            if (type.is_array()) {
                // ["string", "null"] and the like: the first non-null type decides
                for (const auto& t : type) {
                    if (t.is_string() && t.get<std::string>() != "null") {
                        return t.get<std::string>();
                    }
                }
            }
            return "";
        }
        return "";
    }

private:
    const json* tools_;
};

void add_call(std::vector<chat_tool_call>& calls, const std::string& name, const std::string& arguments) {
    chat_tool_call call;
    call.name = name;
    call.arguments = arguments;
    calls.push_back(std::move(call));
}

// ---------------------------------------------------------------- hermes json

/// \brief Decode a JSON string that may still be missing its end
/// \param s the text
/// \param quote the index of the opening quote
/// \return the decoded content seen so far
std::string decode_json_string_prefix(const std::string& s, size_t quote) {
    size_t end = chat_json_value_end(s, quote);
    try {
        if (end != npos) {
            return json::parse(s.substr(quote, end - quote)).get<std::string>();
        }
        // the relaxed reader holds back a half written escape, so what it
        // produced so far is always a well formed JSON string body
        const std::string head = "{\"a\":\"";
        chat_relaxed_result r = chat_relaxed_args_to_json("a=" + s.substr(quote), 0, chat_relaxed_dialect::PYTHON, false);
        if (r.json.size() <= head.size()) {
            return "";
        }
        return json::parse("\"" + r.json.substr(head.size()) + "\"").get<std::string>();
    } catch (const json::exception&) {
        return "";
    }
}

/// \brief Parse one {"name": .., "arguments": ..} object starting at pos
/// \return index past the object, or npos if the text ends inside it
size_t parse_hermes_object(const std::string& s, size_t pos, std::vector<chat_tool_call>& calls) {
    std::string name;
    bool have_name = false;
    std::string arguments;
    bool have_args = false;
    size_t end = npos;

    size_t p = pos + 1;
    while (true) {
        p = skip_ws(s, p);
        if (p >= s.size()) break;
        if (s[p] == ',') { p++; continue; }
        if (s[p] == '}') { end = p + 1; break; }
        if (s[p] != '"') break; // not JSON we understand, stop here
        size_t key_end = chat_json_value_end(s, p);
        if (key_end == npos) break;
        std::string key;
        try {
            key = json::parse(s.substr(p, key_end - p)).get<std::string>();
        } catch (const json::exception&) {
            break;
        }
        p = skip_ws(s, key_end);
        if (p >= s.size() || s[p] != ':') break;
        p = skip_ws(s, p + 1);
        if (p >= s.size()) break;
        size_t value_end = chat_json_value_end(s, p);

        if (key == "name") {
            if (value_end == npos || s[p] != '"') break;
            try {
                name = json::parse(s.substr(p, value_end - p)).get<std::string>();
                have_name = true;
            } catch (const json::exception&) {
                break;
            }
        } else if (key == "arguments" || key == "parameters") {
            have_args = true;
            if (s[p] == '"') {
                // arguments sent as a JSON encoded string
                arguments = decode_json_string_prefix(s, p);
            } else {
                arguments = s.substr(p, (value_end == npos ? s.size() : value_end) - p);
            }
        }
        if (value_end == npos) break;
        p = value_end;
    }

    if (have_name) {
        if (!have_args && end != npos) {
            arguments = "{}";
        }
        add_call(calls, name, arguments);
    }
    return end;
}

void parse_hermes_block(const std::string& body, std::vector<chat_tool_call>& calls) {
    size_t p = skip_ws(body, 0);
    if (p < body.size() && body[p] == '[') {
        p++;
    }
    while (true) {
        p = skip_ws(body, p);
        while (p < body.size() && body[p] == ',') {
            p = skip_ws(body, p + 1);
        }
        if (p >= body.size() || body[p] != '{') {
            return;
        }
        p = parse_hermes_object(body, p, calls);
        if (p == npos) {
            return;
        }
    }
}

// ---------------------------------------------------------------- qwen xml

const std::string XML_FUNCTION_OPEN = "<function=";
const std::string XML_FUNCTION_CLOSE = "</function>";
const std::string XML_PARAM_OPEN = "<parameter=";
const std::string XML_PARAM_CLOSE = "</parameter>";

/// \brief Typed JSON for a complete non-string parameter value
std::string xml_value_to_json(const std::string& value, const std::string& type) {
    if (type == "string") {
        std::string out = "\"";
        chat_json_escape_append(out, value);
        return out + "\"";
    }
    try {
        return json::parse(value).dump();
    } catch (const json::exception&) {
        // not JSON: still understand Python style True/False/None, else a string
        return chat_scalar_to_json(value);
    }
}

/// \brief Arguments of one <function=..> between begin and end
std::string parse_xml_arguments(const std::string& body, size_t begin, size_t end, bool function_closed,
                                const std::string& tool, const param_types& types) {
    std::string out = "{";
    bool first = true;
    size_t p = begin;
    while (true) {
        size_t open = body.find(XML_PARAM_OPEN, p);
        if (open == npos || open >= end) {
            break;
        }
        size_t name_begin = open + XML_PARAM_OPEN.size();
        size_t name_end = body.find('>', name_begin);
        if (name_end == npos || name_end >= end) {
            if (!function_closed) {
                return out;
            }
            break;
        }
        std::string key = trim(body.substr(name_begin, name_end - name_begin));
        size_t value_begin = name_end + 1;

        size_t close = body.find(XML_PARAM_CLOSE, value_begin);
        size_t next = body.find(XML_PARAM_OPEN, value_begin);
        size_t value_end = std::min({close, next, end});
        bool value_closed = value_end < end || function_closed;

        std::string raw = body.substr(value_begin, value_end - value_begin);
        if (!value_closed) {
            raw = cut_partial_marker(raw, {&XML_PARAM_CLOSE, &XML_PARAM_OPEN, &XML_FUNCTION_CLOSE});
        }
        std::string value = trim(raw);
        std::string type = types.of(tool, key);

        if (!first) {
            out += ',';
        }
        first = false;
        out += '"';
        chat_json_escape_append(out, key);
        out += "\":";
        if (!value_closed) {
            if (type == "string") {
                // strings stream as they come; anything else waits for its end
                out += '"';
                chat_json_escape_append(out, value);
            }
            return out;
        }
        out += xml_value_to_json(value, type);
        p = value_end == close ? close + XML_PARAM_CLOSE.size() : value_end;
    }
    if (function_closed) {
        out += '}';
    }
    return out;
}

void parse_qwen_xml_block(const std::string& body, bool closed, const param_types& types,
                          std::vector<chat_tool_call>& calls) {
    size_t p = 0;
    while (true) {
        size_t open = body.find(XML_FUNCTION_OPEN, p);
        if (open == npos) {
            return;
        }
        size_t name_begin = open + XML_FUNCTION_OPEN.size();
        size_t name_end = body.find('>', name_begin);
        if (name_end == npos) {
            return;
        }
        std::string name = trim(body.substr(name_begin, name_end - name_begin));
        size_t function_end = body.find(XML_FUNCTION_CLOSE, name_end);
        bool function_closed = function_end != npos || closed;
        size_t args_end = function_end == npos ? body.size() : function_end;
        add_call(calls, name, parse_xml_arguments(body, name_end + 1, args_end, function_closed, name, types));
        if (function_end == npos) {
            return;
        }
        p = function_end + XML_FUNCTION_CLOSE.size();
    }
}

// ---------------------------------------------------------------- lfm2 / gemma4

void parse_lfm2_block(const std::string& body, bool closed, std::vector<chat_tool_call>& calls) {
    size_t p = skip_ws(body, 0);
    if (p < body.size() && body[p] == '[') {
        p++;
    }
    while (true) {
        p = skip_ws(body, p);
        while (p < body.size() && body[p] == ',') {
            p = skip_ws(body, p + 1);
        }
        if (p >= body.size() || body[p] == ']') {
            return;
        }
        size_t paren = body.find('(', p);
        if (paren == npos) {
            std::string name = trim(body.substr(p));
            if (closed && !name.empty()) {
                add_call(calls, name, "{}");
            }
            return;
        }
        std::string name = trim(body.substr(p, paren - p));
        chat_relaxed_result r = chat_relaxed_args_to_json(body, paren + 1, chat_relaxed_dialect::PYTHON, closed);
        add_call(calls, name, r.json);
        if (!r.closed) {
            return;
        }
        p = r.end;
    }
}

void parse_gemma4_block(const std::string& body, bool closed, std::vector<chat_tool_call>& calls) {
    static const std::string CALL_PREFIX = "call:";
    size_t p = 0;
    while (true) {
        p = skip_ws(body, p);
        if (p >= body.size()) {
            return;
        }
        if (starts_with_at(body, p, CALL_PREFIX)) {
            p += CALL_PREFIX.size();
        }
        size_t brace = body.find('{', p);
        if (brace == npos) {
            std::string name = trim(body.substr(p));
            if (closed && !name.empty()) {
                add_call(calls, name, "{}");
            }
            return;
        }
        std::string name = trim(body.substr(p, brace - p));
        chat_relaxed_result r = chat_relaxed_args_to_json(body, brace, chat_relaxed_dialect::GEMMA4, closed);
        add_call(calls, name, r.json);
        if (!r.closed) {
            return;
        }
        p = r.end;
    }
}

void parse_tool_block(const chat_format_spec& spec, const std::string& body, bool closed,
                      const param_types& types, std::vector<chat_tool_call>& calls) {
    switch (spec.kind) {
        case chat_format_kind::HERMES_JSON:
            parse_hermes_block(body, calls);
            break;
        case chat_format_kind::QWEN_XML:
            parse_qwen_xml_block(body, closed, types, calls);
            break;
        case chat_format_kind::LFM2_PYTHONIC:
            parse_lfm2_block(body, closed, calls);
            break;
        case chat_format_kind::GEMMA4:
            parse_gemma4_block(body, closed, calls);
            break;
        case chat_format_kind::CONTENT_ONLY:
        case chat_format_kind::HARMONY:
            break;
    }
}

// ---------------------------------------------------------------- tag based formats

chat_msg parse_tagged(const chat_format_spec& spec, const std::string& full_text, bool is_partial,
                      const chat_parse_options& options) {
    std::string text = full_text;
    if (!spec.stop_marker.empty()) {
        size_t stop = text.find(spec.stop_marker);
        if (stop != npos) {
            // the turn is over at the marker, whatever the model wrote after it
            text.resize(stop);
            is_partial = false;
        }
    }

    const bool tools = options.parse_tool_calls && !spec.tool_start.empty();
    const std::string no_marker;
    const std::string& reasoning_start = spec.has_reasoning() ? spec.reasoning_start : no_marker;
    const std::string& tool_start = tools ? spec.tool_start : no_marker;
    const std::string& stop_marker = spec.stop_marker;
    const param_types types(options.tools);

    chat_msg msg;
    text_parts reasoning;
    text_parts content;
    bool in_reasoning = false;
    size_t pos = 0;
    if (options.reasoning_may_close && spec.has_reasoning()) {
        const size_t close = text.find(spec.reasoning_end);
        const size_t open = text.find(spec.reasoning_start);
        if (close != npos && (open == npos || close < open)) {
            in_reasoning = true; // what came before the close was reasoning all along
        } else if (open == npos && is_partial) {
            return msg; // reasoning or answer: only a marker, or the end, can tell
        }
    }
    // the first text of a continued message keeps its whitespace
    bool lead_pending = options.keep_whitespace;
    auto lead = [&lead_pending](const std::string& raw) {
        const bool keep = lead_pending && !raw.empty();
        lead_pending = lead_pending && raw.empty();
        return keep;
    };

    while (pos <= text.size()) {
        if (in_reasoning) {
            // the model may open again what the prefill already opened
            size_t first = skip_ws(text, pos);
            if (starts_with_at(text, first, spec.reasoning_start)) {
                pos = first + spec.reasoning_start.size();
                continue;
            }
            size_t rest_len = text.size() - first;
            if (is_partial && rest_len > 0 && rest_len < spec.reasoning_start.size() &&
                spec.reasoning_start.compare(0, rest_len, text, first, rest_len) == 0) {
                break;
            }
            size_t end = text.find(spec.reasoning_end, pos);
            if (end == npos) {
                std::string rest = text.substr(pos);
                if (is_partial) {
                    rest = cut_partial_marker(rest, {&spec.reasoning_end, &stop_marker});
                }
                reasoning.add(rest, lead(rest));
                break;
            }
            const std::string part = text.substr(pos, end - pos);
            // closed with nothing but whitespace in it, the block was layout, not an answer
            const bool exact = lead(part);
            reasoning.add(part, exact && !trim(part).empty());
            pos = end + spec.reasoning_end.size();
            in_reasoning = false;
            continue;
        }

        const std::string* which = nullptr;
        size_t at = find_first(text, pos, {&reasoning_start, &tool_start}, &which);
        if (at == npos) {
            std::string rest = text.substr(pos);
            if (is_partial) {
                rest = cut_partial_marker(rest, {&reasoning_start, &tool_start, &stop_marker});
            }
            content.add(rest, lead(rest));
            break;
        }
        const std::string part = text.substr(pos, at - pos);
        content.add(part, lead(part));
        if (which == &reasoning_start) {
            pos = at + reasoning_start.size();
            in_reasoning = true;
            continue;
        }

        size_t body_begin = at + tool_start.size();
        size_t body_end = text.find(spec.tool_end, body_begin);
        if (body_end == npos) {
            std::string body = text.substr(body_begin);
            if (is_partial) {
                body = cut_partial_marker(body, {&spec.tool_end, &stop_marker});
            }
            // an unterminated block at the very end of a finished generation
            // is parsed as far as it goes, like the stream saw it
            parse_tool_block(spec, body, !is_partial, types, msg.tool_calls);
            break;
        }
        parse_tool_block(spec, text.substr(body_begin, body_end - body_begin), true, types, msg.tool_calls);
        pos = body_end + spec.tool_end.size();
    }

    msg.reasoning = reasoning.join(!is_partial);
    msg.content = content.join(!is_partial);
    if (!msg.tool_calls.empty() && trim(msg.content).empty()) {
        msg.content.clear(); // whitespace before a call is layout, kept only as a whole answer
    }
    return msg;
}

// ---------------------------------------------------------------- harmony

const std::string HARMONY_START = "<|start|>";
const std::string HARMONY_CHANNEL = "<|channel|>";
const std::string HARMONY_MESSAGE = "<|message|>";
const std::string HARMONY_END = "<|end|>";
const std::string HARMONY_CALL = "<|call|>";
const std::string HARMONY_RETURN = "<|return|>";
const std::string HARMONY_RECIPIENT = "to=";
const std::string HARMONY_FUNCTIONS = "functions.";

struct harmony_header {
    std::string channel;
    std::string recipient;
};

harmony_header parse_harmony_header(const std::string& header) {
    harmony_header h;
    size_t channel = header.find(HARMONY_CHANNEL);
    if (channel != npos) {
        size_t b = channel + HARMONY_CHANNEL.size();
        size_t e = b;
        while (e < header.size() && !is_ws(header[e]) && header[e] != '<') e++;
        h.channel = header.substr(b, e - b);
    }
    // the recipient may sit in the role part or after the channel
    size_t to = header.find(HARMONY_RECIPIENT);
    if (to != npos) {
        size_t b = to + HARMONY_RECIPIENT.size();
        size_t e = b;
        while (e < header.size() && !is_ws(header[e]) && header[e] != '<') e++;
        h.recipient = header.substr(b, e - b);
    }
    return h;
}

chat_msg parse_harmony(const std::string& text, bool is_partial, const chat_parse_options& options) {
    chat_msg msg;
    text_parts reasoning;
    text_parts content;

    // the prefill puts us right after a <|start|>, so the text opens with a header
    size_t pos = 0;
    while (pos < text.size()) {
        size_t message = text.find(HARMONY_MESSAGE, pos);
        if (message == npos) {
            break; // header still forming
        }
        harmony_header header = parse_harmony_header(text.substr(pos, message - pos));
        size_t body_begin = message + HARMONY_MESSAGE.size();

        const std::string* which = nullptr;
        size_t body_end = find_first(text, body_begin, {&HARMONY_END, &HARMONY_CALL, &HARMONY_RETURN, &HARMONY_START}, &which);
        bool body_closed = body_end != npos;
        std::string body = text.substr(body_begin, (body_closed ? body_end : text.size()) - body_begin);
        if (!body_closed && is_partial) {
            body = cut_partial_marker(body, {&HARMONY_END, &HARMONY_CALL, &HARMONY_RETURN, &HARMONY_START});
        }

        if (header.recipient.compare(0, HARMONY_FUNCTIONS.size(), HARMONY_FUNCTIONS) == 0) {
            if (options.parse_tool_calls) {
                std::string args = trim(body);
                if (args.empty() && (body_closed || !is_partial)) {
                    args = "{}";
                }
                add_call(msg.tool_calls, header.recipient.substr(HARMONY_FUNCTIONS.size()), args);
            }
        } else if (header.channel == "analysis") {
            reasoning.add(body);
        } else {
            // final, and commentary addressed to the user (preambles)
            content.add(body);
        }

        if (!body_closed) {
            break;
        }
        pos = body_end + which->size();
        if (which != &HARMONY_START) {
            // a closed message; the next one starts at its own <|start|>
            size_t next = text.find(HARMONY_START, pos);
            if (next == npos) {
                break;
            }
            pos = next + HARMONY_START.size();
        }
    }

    msg.reasoning = reasoning.join();
    msg.content = content.join();
    return msg;
}

} // namespace

chat_msg chat_parse(const chat_format_spec& spec, const std::string& text, bool is_partial,
                    const chat_parse_options& options) {
    if (spec.kind == chat_format_kind::HARMONY) {
        return parse_harmony(text, is_partial, options);
    }
    return parse_tagged(spec, text, is_partial, options);
}
