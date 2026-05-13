#pragma once

#include <iostream>
#include <iomanip>
#include <string>

inline bool is_windows_path_string(const std::string& value) {
    for (size_t i = 0; i + 2 < value.size(); ++i) {
        bool has_drive = ((value[i] >= 'A' && value[i] <= 'Z') || (value[i] >= 'a' && value[i] <= 'z')) && value[i + 1] == ':';
        if (has_drive && (value[i + 2] == '\\' || value[i + 2] == '/')) {
            return true;
        }
    }
    return value.size() >= 2 && value[0] == '\\' && value[1] == '\\';
}

inline size_t skip_json_spaces(const std::string& text, size_t pos) {
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\r' || text[pos] == '\n')) {
        pos++;
    }
    return pos;
}

inline size_t find_closing_quote(const std::string& text, size_t open_quote_pos) {
    bool escaped = false;
    for (size_t pos = open_quote_pos + 1; pos < text.size(); ++pos) {
        char ch = text[pos];
        if (ch == '"' && !escaped) {
            return pos;
        }
        escaped = ch == '\\' && !escaped;
        if (ch != '\\') {
            escaped = false;
        }
    }
    return std::string::npos;
}

inline bool comma_starts_next_object_member(const std::string& text, size_t comma_pos) {
    size_t key_start = skip_json_spaces(text, comma_pos + 1);
    if (key_start >= text.size() || text[key_start] != '"') {
        return false;
    }

    size_t key_end = find_closing_quote(text, key_start);
    if (key_end == std::string::npos) {
        return false;
    }

    size_t colon_pos = skip_json_spaces(text, key_end + 1);
    return colon_pos < text.size() && text[colon_pos] == ':';
}

inline bool is_string_boundary_quote(const std::string& text, size_t quote_pos, bool is_object_key, bool is_array_value) {
    size_t next_pos = skip_json_spaces(text, quote_pos + 1);
    if (next_pos >= text.size()) {
        return true;
    }

    char next = text[next_pos];
    if (is_object_key) {
        return next == ':';
    }

    if (next == '}' || next == ']') {
        return true;
    }
    if (next == ',') {
        if (is_array_value) {
            return true;
        }
        return comma_starts_next_object_member(text, next_pos);
    }
    return false;
}

inline std::string sanitize_tool_argument_json_strings(const std::string& json_text) {
    std::string output;
    output.reserve(json_text.size());

    for (size_t i = 0; i < json_text.size();) {
        if (json_text[i] != '"') {
            output.push_back(json_text[i++]);
            continue;
        }

        size_t previous_pos = i;
        while (previous_pos > 0 && (json_text[previous_pos - 1] == ' ' || json_text[previous_pos - 1] == '\t' || json_text[previous_pos - 1] == '\r' || json_text[previous_pos - 1] == '\n')) {
            previous_pos--;
        }
        bool is_object_key = previous_pos == 0 || json_text[previous_pos - 1] == '{';
        if (!is_object_key && previous_pos > 0 && json_text[previous_pos - 1] == ',') {
            size_t possible_key_end = find_closing_quote(json_text, i);
            if (possible_key_end != std::string::npos) {
                size_t colon_pos = skip_json_spaces(json_text, possible_key_end + 1);
                is_object_key = colon_pos < json_text.size() && json_text[colon_pos] == ':';
            }
        }
        bool is_array_value = previous_pos > 0 && (json_text[previous_pos - 1] == '[' || (json_text[previous_pos - 1] == ',' && !is_object_key));

        size_t value_start = i + 1;
        size_t value_end = value_start;
        bool escaped = false;
        while (value_end < json_text.size()) {
            char ch = json_text[value_end];
            if (ch == '"' && !escaped && is_string_boundary_quote(json_text, value_end, is_object_key, is_array_value)) {
                break;
            }
            escaped = ch == '\\' && !escaped;
            if (ch != '\\') {
                escaped = false;
            }
            value_end++;
        }

        if (value_end >= json_text.size()) {
            output.append(json_text, i, std::string::npos);
            break;
        }

        std::string value = json_text.substr(value_start, value_end - value_start);
        output.push_back('"');

        auto append_escaped_char = [&output](char ch) {
            switch (ch) {
            case '\n':
                output.append("\\n");
                break;
            case '\r':
                output.append("\\r");
                break;
            case '\t':
                output.append("\\t");
                break;
            case '\b':
                output.append("\\b");
                break;
            case '\f':
                output.append("\\f");
                break;
            case '"':
                output.append("\\\"");
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    unsigned char value = static_cast<unsigned char>(ch);
                    output.append("\\u00");
                    output.push_back(hex[value >> 4]);
                    output.push_back(hex[value & 0x0f]);
                }
                else {
                    output.push_back(ch);
                }
                break;
            }
        };

        if (is_windows_path_string(value)) {
            for (size_t pos = 0; pos < value.size(); ++pos) {
                if (value[pos] == '\\') {
                    output.append("\\\\");
                    if (pos + 1 < value.size() && value[pos + 1] == '\\') {
                        ++pos;
                    }
                }
                else {
                    append_escaped_char(value[pos]);
                }
            }
        }
        else {
            for (size_t pos = 0; pos < value.size(); ++pos) {
                if (value[pos] == '\\' && pos + 1 < value.size()) {
                    output.push_back(value[pos]);
                    output.push_back(value[pos + 1]);
                    ++pos;
                }
                else {
                    append_escaped_char(value[pos]);
                }
            }
        }

        output.push_back('"');
        i = value_end + 1;
    }

    return output;
}