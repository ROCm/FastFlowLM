/// \file active_tool_reasoning.hpp
/// \brief Qwen3.5 GPU tool-call reasoning in client-supplied chat history.
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace flm::chat {

/// \brief Restore only client-supplied strings on tool calls after the Qwen GPU template's last real user query.
/// `messages` is the flattened template input, before tool results are reordered; `original_indices` maps it back
/// to this request, including when instruction placement removed a message or merged consecutive user turns.
inline void apply_active_tool_reasoning(nlohmann::ordered_json& messages, const nlohmann::ordered_json& request_messages,
                                        const std::vector<std::size_t>& original_indices) {
    for (std::size_t i = messages.size(); i > 0; --i) {
        const auto& user = messages[i - 1];
        if (user.value("role", "") != "user") continue;
        // The package ignores user-wrapped tool results when finding the last query. Qwen renders images before text.
        std::string_view text = user.at("content").get_ref<const std::string&>();
        const auto first = text.find_first_not_of(" \t\n\r");
        if (first != text.npos && (!user.contains("images") || user.at("images").empty())) {
            text = text.substr(first, text.find_last_not_of(" \t\n\r") - first + 1);
            if (text.starts_with("<tool_response>") && text.ends_with("</tool_response>")) continue;
        }
        for (; i < messages.size(); ++i) {
            auto& message = messages[i];
            const auto calls = message.find("tool_calls");
            if (message.value("role", "") != "assistant" || calls == message.end() || !calls->is_array() || calls->empty()) continue;
            const auto& original = request_messages.at(original_indices.at(i));
            const auto note = original.find("reasoning_content");
            if (note != original.end() && note->is_string()) message["reasoning_content"] = *note;
        }
        return;
    }
}

}  // namespace flm::chat
