/// \file chat_args.cpp
/// \brief Incremental conversion of tool call arguments to JSON

#include "chat/chat_args.hpp"

#include <cstdint>

#include <nlohmann/json.hpp>

namespace {

const std::string GEMMA4_QUOTE = "<|\"|>";

bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool is_closer(char c) {
    return c == '}' || c == ']' || c == ')';
}

std::string trim(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && is_ws(s[b])) b++;
    while (e > b && is_ws(s[e - 1])) e--;
    return s.substr(b, e - b);
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

bool parse_hex4(const std::string& s, size_t pos, uint32_t& value) {
    if (pos + 4 > s.size()) {
        return false;
    }
    value = 0;
    for (size_t i = pos; i < pos + 4; i++) {
        char c = s[i];
        value <<= 4;
        if (c >= '0' && c <= '9') value |= c - '0';
        else if (c >= 'a' && c <= 'f') value |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') value |= c - 'A' + 10;
        else return false;
    }
    return true;
}

/// \brief Walks relaxed argument text and writes the JSON it means
/// \note Every method returns true when its construct ended normally and false
///       when the text ran out first. On running out the whole call stack
///       unwinds; with input_done each level closes what it opened on the way.
class relaxed_writer {
public:
    relaxed_writer(const std::string& s, size_t pos, chat_relaxed_dialect dialect, bool input_done)
        : s_(s), pos_(pos), dialect_(dialect), done_(input_done) {}

    chat_relaxed_result run() {
        chat_relaxed_result result;
        if (dialect_ == chat_relaxed_dialect::GEMMA4) {
            if (pos_ < s_.size() && s_[pos_] == '{') {
                pos_++;
                result.closed = members('}', ':');
            } else if (done_) {
                out_ = "{}";
            }
        } else {
            result.closed = members(')', '=');
        }
        result.json = std::move(out_);
        result.end = pos_;
        return result;
    }

private:
    enum class key_status { OK, STARVED, SKIP };

    const std::string& s_;
    size_t pos_;
    chat_relaxed_dialect dialect_;
    bool done_;
    std::string out_;

    bool starved() const {
        return pos_ >= s_.size();
    }

    void skip_ws() {
        while (pos_ < s_.size() && is_ws(s_[pos_])) pos_++;
    }

    bool give_up(char closer) {
        if (done_) {
            out_ += closer;
        }
        return false;
    }

    /// \brief true when the rest of the text could still grow into marker
    bool partial_marker_at_end(const std::string& marker) const {
        size_t rem = s_.size() - pos_;
        return rem < marker.size() && marker.compare(0, rem, s_, pos_, rem) == 0;
    }

    /// \brief Delimiter opening a string at pos_, empty if none
    std::string string_delim_at() const {
        if (dialect_ == chat_relaxed_dialect::GEMMA4 && s_.compare(pos_, GEMMA4_QUOTE.size(), GEMMA4_QUOTE) == 0) {
            return GEMMA4_QUOTE;
        }
        char c = s_[pos_];
        if (c == '"' || (dialect_ == chat_relaxed_dialect::PYTHON && c == '\'')) {
            return std::string(1, c);
        }
        return "";
    }

    /// \brief Read string content after its opening delimiter into dst
    /// \param for_json escape what is appended so dst stays JSON string content
    bool scan_string(const std::string& delim, std::string& dst, bool for_json) {
        const bool quote_char = delim.size() == 1;
        while (true) {
            if (starved()) {
                return false;
            }
            if (s_.compare(pos_, delim.size(), delim) == 0) {
                pos_ += delim.size();
                return true;
            }
            if (!quote_char && !done_ && partial_marker_at_end(delim)) {
                return false;
            }
            std::string piece;
            char c = s_[pos_];
            if (quote_char && c == '\\') {
                if (pos_ + 1 >= s_.size()) {
                    return false;
                }
                char e = s_[pos_ + 1];
                size_t used = 2;
                switch (e) {
                    case 'n': piece = "\n"; break;
                    case 't': piece = "\t"; break;
                    case 'r': piece = "\r"; break;
                    case 'b': piece = "\b"; break;
                    case 'f': piece = "\f"; break;
                    case 'u': {
                        uint32_t cp = 0;
                        if (!parse_hex4(s_, pos_ + 2, cp)) {
                            if (pos_ + 6 > s_.size() && !done_) {
                                return false;
                            }
                            piece = "u";
                            break;
                        }
                        used = 6;
                        if (cp >= 0xD800 && cp < 0xDC00) {
                            uint32_t low = 0;
                            if (pos_ + 12 > s_.size() && !done_) {
                                return false;
                            }
                            if (s_.compare(pos_ + 6, 2, "\\u") == 0 && parse_hex4(s_, pos_ + 8, low) &&
                                low >= 0xDC00 && low < 0xE000) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                                used = 12;
                            }
                        }
                        append_utf8(piece, cp);
                        break;
                    }
                    default: piece = std::string(1, e); break;
                }
                pos_ += used;
            } else {
                piece = std::string(1, c);
                pos_++;
            }
            if (for_json) {
                chat_json_escape_append(dst, piece);
            } else {
                dst += piece;
            }
        }
    }

    /// \brief Read a Gemma4 <|"|> string after its opener into dst, JSON escaped
    /// \note Gemma4 strings are raw text, so a <|"|> (or the plain " or ` the
    ///       model sometimes writes instead) only closes the string when the
    ///       value ends right after it; otherwise it is part of the text. Until
    ///       the next non-space character shows which, the candidate is held.
    bool scan_gemma4_string(std::string& dst) {
        while (true) {
            if (starved()) {
                return false;
            }
            size_t len = 0;
            if (s_.compare(pos_, GEMMA4_QUOTE.size(), GEMMA4_QUOTE) == 0) {
                len = GEMMA4_QUOTE.size();
            } else if (s_[pos_] == '"' || s_[pos_] == '`') {
                len = 1;
            } else if (!done_ && partial_marker_at_end(GEMMA4_QUOTE)) {
                return false;
            }
            if (len > 0) {
                size_t next = pos_ + len;
                while (next < s_.size() && is_ws(s_[next])) next++;
                if (next >= s_.size() && !done_) {
                    return false;
                }
                if (next >= s_.size() || s_[next] == ',' || is_closer(s_[next])) {
                    pos_ += len;
                    return true;
                }
                chat_json_escape_append(dst, s_.substr(pos_, len));
                pos_ += len;
                continue;
            }
            chat_json_escape_append(dst, std::string(1, s_[pos_]));
            pos_++;
        }
    }

    bool string_value(const std::string& delim) {
        pos_ += delim.size();
        out_ += '"';
        bool closed = delim == GEMMA4_QUOTE ? scan_gemma4_string(out_) : scan_string(delim, out_, true);
        if (closed || done_) {
            out_ += '"';
        }
        return closed;
    }

    bool scalar() {
        size_t start = pos_;
        int depth = 0;
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == '{' || c == '[' || c == '(') {
                depth++;
            } else if (is_closer(c) || c == ',') {
                if (depth == 0) break;
                if (c != ',') depth--;
            }
            pos_++;
        }
        if (starved() && !done_) {
            // a bare value is only known once it ends: "1" may become "12"
            return false;
        }
        std::string token = s_.substr(start, pos_ - start);
        if (dialect_ == chat_relaxed_dialect::GEMMA4) {
            // a string whose <|"|> opener the model left out still has its closer
            std::string trimmed = trim(token);
            if (trimmed.size() >= GEMMA4_QUOTE.size() &&
                trimmed.compare(trimmed.size() - GEMMA4_QUOTE.size(), GEMMA4_QUOTE.size(), GEMMA4_QUOTE) == 0) {
                token = trimmed.substr(0, trimmed.size() - GEMMA4_QUOTE.size());
                out_ += '"';
                chat_json_escape_append(out_, trim(token));
                out_ += '"';
                return !starved();
            }
        }
        out_ += chat_scalar_to_json(token);
        return !starved();
    }

    bool value() {
        skip_ws();
        if (starved()) {
            if (done_) out_ += "null";
            return false;
        }
        if (dialect_ == chat_relaxed_dialect::GEMMA4 && !done_ && partial_marker_at_end(GEMMA4_QUOTE)) {
            return false;
        }
        std::string delim = string_delim_at();
        if (!delim.empty()) {
            return string_value(delim);
        }
        char c = s_[pos_];
        if (c == '{') {
            pos_++;
            return members('}', ':');
        }
        if (c == '[' || (c == '(' && dialect_ == chat_relaxed_dialect::PYTHON)) {
            pos_++;
            return array();
        }
        return scalar();
    }

    key_status read_key(std::string& key, char sep) {
        if (dialect_ == chat_relaxed_dialect::GEMMA4 && !done_ && partial_marker_at_end(GEMMA4_QUOTE)) {
            return key_status::STARVED;
        }
        std::string delim = string_delim_at();
        if (!delim.empty()) {
            pos_ += delim.size();
            if (!scan_string(delim, key, false)) {
                return key_status::STARVED;
            }
            skip_ws();
            if (starved()) {
                return key_status::STARVED;
            }
            if (s_[pos_] == sep || s_[pos_] == ':' || s_[pos_] == '=') {
                pos_++;
                return key_status::OK;
            }
            return key_status::SKIP;
        }
        size_t start = pos_;
        while (pos_ < s_.size() && s_[pos_] != sep) {
            if (s_[pos_] == ',' || is_closer(s_[pos_])) {
                // a value with no name, nothing to map it to
                return key_status::SKIP;
            }
            pos_++;
        }
        if (starved()) {
            return key_status::STARVED;
        }
        key = trim(s_.substr(start, pos_ - start));
        pos_++;
        return key_status::OK;
    }

    bool members(char close, char sep) {
        out_ += '{';
        bool first = true;
        while (true) {
            skip_ws();
            if (starved()) {
                return give_up('}');
            }
            char c = s_[pos_];
            if (c == close || is_closer(c)) {
                pos_++;
                out_ += '}';
                return true;
            }
            if (c == ',') {
                pos_++;
                continue;
            }
            std::string key;
            key_status status = read_key(key, sep);
            if (status == key_status::STARVED) {
                return give_up('}');
            }
            if (status == key_status::SKIP) {
                if (pos_ < s_.size() && s_[pos_] == ',') pos_++;
                continue;
            }
            if (!first) {
                out_ += ',';
            }
            first = false;
            out_ += '"';
            chat_json_escape_append(out_, key);
            out_ += "\":";
            if (!value()) {
                return give_up('}');
            }
        }
    }

    bool array() {
        out_ += '[';
        bool first = true;
        while (true) {
            skip_ws();
            if (starved()) {
                return give_up(']');
            }
            char c = s_[pos_];
            if (is_closer(c)) {
                pos_++;
                out_ += ']';
                return true;
            }
            if (c == ',') {
                pos_++;
                continue;
            }
            // something starts here, so a value will follow and the comma is safe to send
            if (!first) {
                out_ += ',';
            }
            first = false;
            if (!value()) {
                return give_up(']');
            }
        }
    }
};

} // namespace

void chat_json_escape_append(std::string& out, const std::string& s) {
    static const char HEX[] = "0123456789abcdef";
    for (char ch : s) {
        unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out += HEX[c >> 4];
                    out += HEX[c & 0xF];
                } else {
                    out += ch;
                }
        }
    }
}

size_t chat_json_value_end(const std::string& s, size_t pos) {
    if (pos >= s.size()) {
        return std::string::npos;
    }
    char c = s[pos];
    if (c == '"') {
        for (size_t i = pos + 1; i < s.size(); i++) {
            if (s[i] == '\\') {
                i++;
            } else if (s[i] == '"') {
                return i + 1;
            }
        }
        return std::string::npos;
    }
    if (c == '{' || c == '[') {
        int depth = 0;
        bool in_string = false;
        for (size_t i = pos; i < s.size(); i++) {
            char d = s[i];
            if (in_string) {
                if (d == '\\') i++;
                else if (d == '"') in_string = false;
                continue;
            }
            if (d == '"') {
                in_string = true;
            } else if (d == '{' || d == '[') {
                depth++;
            } else if (d == '}' || d == ']') {
                if (--depth == 0) {
                    return i + 1;
                }
            }
        }
        return std::string::npos;
    }
    // literal: it may still be growing until a delimiter shows up
    for (size_t i = pos; i < s.size(); i++) {
        char d = s[i];
        if (d == ',' || d == '}' || d == ']' || is_ws(d)) {
            return i;
        }
    }
    return std::string::npos;
}

std::string chat_scalar_to_json(const std::string& token) {
    std::string t = trim(token);
    if (t.empty() || t == "None" || t == "null") {
        return "null";
    }
    if (t == "True" || t == "true") {
        return "true";
    }
    if (t == "False" || t == "false") {
        return "false";
    }
    char c = t[0];
    if (c == '-' || (c >= '0' && c <= '9')) {
        try {
            nlohmann::json j = nlohmann::json::parse(t);
            if (j.is_number()) {
                return t;
            }
        } catch (const nlohmann::json::exception&) {
        }
    }
    std::string out = "\"";
    chat_json_escape_append(out, t);
    out += '"';
    return out;
}

chat_relaxed_result chat_relaxed_args_to_json(const std::string& s, size_t pos,
                                              chat_relaxed_dialect dialect, bool input_done) {
    relaxed_writer writer(s, pos, dialect, input_done);
    return writer.run();
}
