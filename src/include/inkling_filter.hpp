/// \file inkling_filter.hpp
/// \brief cli_inkling_filter: terminal view of an Inkling turn
/// \author FastFlowLM Team
/// \date 2026-11-21
/// \note Portions of this file consist of AI-generated content.
/// \note An Inkling turn is a run of messages, <|message_model|><|content_KIND|>body<|end_message|>.
///       The thinking message is shown as <think>\n...\n</think>\n\n, the other bodies as they
///       are; the markers themselves are dropped. Works per flush: the generator flushes after
///       every token and every marker is a single token.
#pragma once

#include <iostream>
#include <streambuf>
#include <string>
#include <vector>

class inkling_streambuf : public std::streambuf {
private:
    std::vector<char> buffer;
    std::ostream& output_stream;
    bool in_thinking = false;

    static bool is_marker(const std::string& s) {
        return s.size() >= 4 && s.compare(0, 2, "<|") == 0 && s.compare(s.size() - 2, 2, "|>") == 0;
    }

    std::string filter_message(const std::string& message) {
        if (!is_marker(message)) {
            return message;
        }
        if (message == "<|content_thinking|>") {
            in_thinking = true;
            return "<think>\n";
        }
        if (in_thinking && message == "<|end_message|>") {
            in_thinking = false;
            return "\n</think>\n\n";
        }
        return "";
    }

public:
    inkling_streambuf(std::ostream& os) : output_stream(os) { buffer.reserve(256); }

protected:
    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof()) {
            buffer.push_back(static_cast<char>(ch));
        }
        return ch;
    }

    int sync() override {
        if (!buffer.empty()) {
            std::string filtered = filter_message(std::string(buffer.begin(), buffer.end()));
            if (!filtered.empty()) {
                output_stream << filtered;
                output_stream.flush();
            }
            buffer.clear();
        }
        return 0;
    }
};

class cli_inkling_filter : public std::ostream {
private:
    inkling_streambuf buf;

public:
    cli_inkling_filter(std::ostream& os) : std::ostream(&buf), buf(os) {}
};
