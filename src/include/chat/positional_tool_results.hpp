/// \file positional_tool_results.hpp
/// \brief Bind parallel tool results before a positional chat template drops their IDs.
#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flm::chat {

struct tool_history_error {
    std::string param;
    std::string message;
};

/// \brief Order each multi-call assistant turn's contiguous results. Discard the request copy on error.
inline std::optional<tool_history_error> order_parallel_tool_results(
    nlohmann::ordered_json& messages, const std::vector<std::size_t>& original_indices = {}) {
    using json = nlohmann::ordered_json;
    auto has_role = [](const json& msg, const char* role) {
        const auto it = msg.find("role");
        return it != msg.end() && *it == role;
    };
    auto id_field = [](const json& obj, const char* key) -> const std::string* {
        const auto it = obj.find(key);
        return it == obj.end() ? nullptr : it->get_ptr<const std::string*>();
    };
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& assistant = messages[i];
        if (!has_role(assistant, "assistant")) continue;
        const auto calls = assistant.find("tool_calls");
        if (calls == assistant.end() || !calls->is_array() || calls->size() < 2) continue;

        auto path = [&](std::size_t m) {
            return "messages[" + std::to_string(original_indices.empty() ? m : original_indices[m]) + "]";
        };
        auto call_path = [&](std::size_t j) {
            return path(i) + ".tool_calls[" + std::to_string(j) + "].id";
        };
        std::unordered_map<std::string, std::size_t> positions;
        for (std::size_t j = 0; j < calls->size(); ++j) {
            const auto* id = id_field((*calls)[j], "id");
            if (!id || id->empty())
                return tool_history_error{call_path(j), "Parallel tool calls require a non-empty string ID"};
            if (!positions.emplace(*id, j).second)
                return tool_history_error{call_path(j), "Duplicate tool call ID in this assistant turn"};
        }
        std::vector<json> results(calls->size());
        for (std::size_t r = i + 1; r < messages.size() && has_role(messages[r], "tool"); ++r) {
            const auto* id = id_field(messages[r], "tool_call_id");
            auto pos = id ? positions.find(*id) : positions.end();
            if (pos == positions.end() || !results[pos->second].is_null()) {
                return tool_history_error{path(r) + ".tool_call_id",
                    pos == positions.end() ? "Result must identify a tool call in the preceding assistant turn"
                                           : "Duplicate result for this tool call"};
            }
            results[pos->second] = std::move(messages[r]);
        }
        for (std::size_t j = 0; j < results.size(); ++j) {
            if (results[j].is_null())
                return tool_history_error{call_path(j), "No result for this tool call before the next turn or end of history"};
            messages[i + 1 + j] = std::move(results[j]);
        }
        i += results.size();
    }
    return std::nullopt;
}

} // namespace flm::chat
