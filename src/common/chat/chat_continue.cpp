/// \file chat_continue.cpp
/// \brief Continue the final assistant message instead of starting a new turn

#include "chat/chat_continue.hpp"

#include <stdexcept>

namespace {

// Python's str.strip() set, short of the non-ASCII spaces no template trims to
const char* const WHITESPACE = " \t\n\r\v\f";

std::string lstrip(const std::string& s) {
    const size_t b = s.find_first_not_of(WHITESPACE);
    return b == std::string::npos ? std::string() : s.substr(b);
}

std::string strip(const std::string& s) {
    const std::string left = lstrip(s);
    const size_t e = left.find_last_not_of(WHITESPACE);
    return e == std::string::npos ? std::string() : left.substr(0, e + 1);
}

/// \brief The text the final message ends with: its content, or its last text part
std::string final_text(const nlohmann::ordered_json& message) {
    const auto content = message.find("content");
    if (content != message.end() && content->is_string()) {
        return content->get<std::string>();
    }
    if (content != message.end() && content->is_array()) {
        for (auto part = content->rbegin(); part != content->rend(); ++part) {
            if (part->is_object() && part->contains("text") && (*part)["text"].is_string()) {
                return (*part)["text"].get<std::string>();
            }
        }
    }
    throw std::invalid_argument("continue_final_message is set but the final message has no text to continue");
}

size_t common_prefix(const std::string& a, const std::string& b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) {
        n++;
    }
    return n;
}

} // namespace

std::string chat_continue_final_message(const std::string& rendered, const nlohmann::ordered_json& final_message,
                                        size_t* text_at) {
    const std::string text = final_text(final_message);
    const std::string stripped = strip(text);
    if (stripped.empty()) {
        // every rendering contains "", and cutting there would leave the end of turn in place
        throw std::invalid_argument("continue_final_message is set but the final message has no text to continue");
    }
    const size_t at = rendered.rfind(stripped);
    if (at == std::string::npos) {
        throw std::invalid_argument("continue_final_message is set but the final message does not appear in the chat "
                                    "after applying the chat template; this template rewrites the final assistant message");
    }
    if (text_at != nullptr) {
        *text_at = at;
    }
    // a template that kept the text as it is keeps its trailing whitespace too
    const size_t kept = lstrip(text).size();
    if (rendered.compare(at, kept, text) == 0) {
        return rendered.substr(0, at + kept);
    }
    return rendered.substr(0, at + stripped.size());
}

chat_continued chat_continue_prompt(const chat_render_fn& render, const nlohmann::ordered_json& messages) {
    chat_continued out;
    size_t text_at = 0;
    out.prompt = chat_continue_final_message(render(messages, false), messages.back(), &text_at);

    nlohmann::ordered_json before = messages;
    before.erase(before.size() - 1);
    try {
        // how the template opens a reply after the same conversation
        const std::string opening = render(before, true);
        if (opening.size() > text_at || out.prompt.compare(0, opening.size(), opening) != 0) {
            out.prompt = opening + out.prompt.substr(text_at);
        }
    } catch (const std::exception&) {
        // a template that cannot open a reply here: the cut stands
    }

    size_t turn = 0;
    try {
        // the continued turn starts where the conversation before it ends
        turn = common_prefix(render(before, false), out.prompt);
    } catch (const std::exception&) {
        turn = 0;
    }
    out.turn = out.prompt.substr(turn);
    return out;
}
