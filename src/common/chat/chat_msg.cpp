/// \file chat_msg.cpp
/// \brief Delta computation between two parses of the same assistant turn

#include "chat/chat_msg.hpp"
#include "utils/debug_utils.hpp"

namespace {

/// \brief The part of cur that extends prev, or nothing if it does not extend it
bool suffix_after(const std::string& prev, const std::string& cur, const char* field, std::string& out) {
    if (cur.size() < prev.size() || cur.compare(0, prev.size(), prev) != 0) {
        header_print("WARNING", "chat parser rewrote already streamed " << field << ", dropping the change");
        return false;
    }
    out = cur.substr(prev.size());
    return true;
}

} // namespace

std::vector<chat_msg_delta> chat_compute_deltas(const chat_msg& prev, const chat_msg& cur) {
    std::vector<chat_msg_delta> deltas;

    std::string text;
    if (suffix_after(prev.reasoning, cur.reasoning, "reasoning", text) && !text.empty()) {
        chat_msg_delta d;
        d.reasoning = std::move(text);
        deltas.push_back(std::move(d));
    }
    if (suffix_after(prev.content, cur.content, "content", text) && !text.empty()) {
        chat_msg_delta d;
        d.content = std::move(text);
        deltas.push_back(std::move(d));
    }

    for (size_t i = 0; i < cur.tool_calls.size(); i++) {
        const chat_tool_call& call = cur.tool_calls[i];
        chat_msg_delta d;
        d.tool_index = static_cast<int>(i);
        if (i >= prev.tool_calls.size()) {
            // first sight of this call: id and name go out with whatever arguments exist
            d.tool_id = call.id;
            d.tool_name = call.name;
            d.tool_arguments = call.arguments;
            deltas.push_back(std::move(d));
            continue;
        }
        if (suffix_after(prev.tool_calls[i].arguments, call.arguments, "tool arguments", text) && !text.empty()) {
            d.tool_arguments = std::move(text);
            deltas.push_back(std::move(d));
        }
    }
    return deltas;
}
