/// \file responses_request.hpp
/// \brief Convert one self-contained Responses request to the shared chat turn.
#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include "chat/positional_tool_results.hpp"

namespace flm::responses {
using json = nlohmann::ordered_json;

struct request_error : std::invalid_argument {
    std::string param;
    request_error(const std::string& field, const std::string& why)
        : std::invalid_argument(field + ": " + why), param(field) {}
};

inline const json& member(const json& obj, const char* key, const std::string& prefix,
                          json::value_t kind) {
    const auto it = obj.find(key);
    if (it == obj.end() || it->type() != kind)
        throw request_error(prefix + key, "missing or invalid value");
    return *it;
}

inline bool provided(const json& obj, const char* key) {
    const auto it = obj.find(key);
    return it != obj.end() && !it->is_null();
}

inline std::string text_parts(const json& parts, const std::string& field,
                              const char* part_type, bool assistant_message = false) {
    if (!parts.is_array()) throw request_error(field, "expected an array");
    std::string text;
    for (size_t i = 0; i < parts.size(); ++i) {
        const auto path = field + "[" + std::to_string(i) + "].";
        const auto& type = member(parts[i], "type", path, json::value_t::string);
        if (type != part_type && !(assistant_message && type == "input_text"))
            throw request_error(path + "type", std::string("expected ") + part_type);
        text += member(parts[i], "text", path, json::value_t::string).get_ref<const std::string&>();
    }
    return text;
}

inline json to_chat_request(const json& req) {
    if (!req.is_object()) throw request_error("body", "expected an object");
    for (auto key : {"previous_response_id", "conversation", "background"})
        if (req.contains(key)) throw request_error(key, "not supported; send the complete history in input");

    // A final chat_msg groups all text ahead of calls and loses their emission order.
    // Consume the core's ordered deltas for both wire modes; only the sink decides
    // whether to send them now or return the assembled items as one response.
    json chat = {{"model", member(req, "model", "", json::value_t::string)}, {"stream", true}};
    for (auto key : {"stream", "store", "parallel_tool_calls"}) {
        if (!provided(req, key)) continue;
        member(req, key, "", json::value_t::boolean);
        if (std::string_view(key) == "parallel_tool_calls") chat[key] = req[key];
    }
    if (provided(req, "max_output_tokens")) {
        const auto& n = req.at("max_output_tokens");
        if (!n.is_number_integer() || n < 1 || n > (std::numeric_limits<int>::max)())
            throw request_error("max_output_tokens", "expected a positive integer within the token limit");
        chat["max_completion_tokens"] = n;
    }
    for (auto key : {"temperature", "top_p"}) {
        if (provided(req, key)) {
            if (!req[key].is_number()) throw request_error(key, "expected a number");
            chat[key] = req[key];
        }
    }
    if (provided(req, "reasoning")) {
        const auto& r = member(req, "reasoning", "", json::value_t::object);
        if (provided(r, "effort")) chat["reasoning_effort"] = member(r, "effort", "reasoning.", json::value_t::string);
    }
    if (req.contains("text")) {
        const auto& t = member(req, "text", "", json::value_t::object);
        if (t.contains("format")) {
            auto fmt = member(t, "format", "text.", json::value_t::object);
            auto type = member(fmt, "type", "text.format.", json::value_t::string);
            if (type == "json_schema") {
                member(fmt, "name", "text.format.", json::value_t::string);
                member(fmt, "schema", "text.format.", json::value_t::object);
                fmt.erase("type");
                chat["response_format"] = {{"type", type}, {"json_schema", std::move(fmt)}};
            } else if (type == "text" || type == "json_object") {
                chat["response_format"] = {{"type", type}};
            } else throw request_error("text.format.type", "unsupported format");
        }
    }
    if (req.contains("tools")) {
        const auto& tools = member(req, "tools", "", json::value_t::array);
        chat["tools"] = json::array();
        for (size_t i = 0; i < tools.size(); ++i) {
            const auto p = "tools[" + std::to_string(i) + "].";
            if (member(tools[i], "type", p, json::value_t::string) != "function")
                throw request_error(p + "type", "only client-executed function tools are supported");
            auto fn = tools[i];
            if (member(fn, "name", p, json::value_t::string).get_ref<const std::string&>().empty())
                throw request_error(p + "name", "expected a non-empty function name");
            if (provided(fn, "strict")) member(fn, "strict", p, json::value_t::boolean);
            else fn.erase("strict"); // null selects the chat core's default
            if (fn.contains("parameters") && !fn["parameters"].is_null())
                member(fn, "parameters", p, json::value_t::object);
            fn.erase("type");
            chat["tools"].push_back({{"type","function"},{"function",std::move(fn)}});
        }
    }
    if (req.contains("tool_choice")) {
        const auto& choice = req["tool_choice"];
        if (choice.is_string()) chat["tool_choice"] = choice;
        else if (member(choice, "type", "tool_choice.", json::value_t::string) == "function")
            chat["tool_choice"] = {{"type","function"},{"function",{{"name",member(choice, "name", "tool_choice.", json::value_t::string)}}}};
        else throw request_error("tool_choice.type", "only function tools are supported");
    }

    json messages = json::array();
    if (req.contains("instructions") && !req["instructions"].is_null())
        messages.push_back({{"role","system"},{"content",member(req, "instructions", "", json::value_t::string)}});
    if (!req.contains("input")) throw request_error("input", "expected text or ordered input items");
    const auto& input = req["input"];
    if (input.is_string()) messages.push_back({{"role","user"},{"content",input}});
    else if (input.is_array() && !input.empty()) {
        std::unordered_set<std::string> seen, pending;
        auto assistant = [&]() -> json& {
            if (messages.empty() || messages.back()["role"] != "assistant")
                messages.push_back({{"role","assistant"},{"content",""}});
            return messages.back();
        };
        for (size_t i = 0; i < input.size(); ++i) {
            const auto p = "input[" + std::to_string(i) + "].";
            const auto& item = input[i];
            const json type = item.contains("type") ? item["type"] : json("message");
            const bool waiting_for_results = !pending.empty();
            const bool receiving_results = waiting_for_results && messages.back()["role"] == "tool";
            if (type == "message") {
                auto role = member(item, "role", p, json::value_t::string);
                if (role != "user" && role != "assistant" && role != "system" && role != "developer")
                    throw request_error(p + "role", "unsupported message role");
                if (waiting_for_results && (role != "assistant" || receiving_results))
                    throw request_error(p + "role", "expected outstanding function results first");
                if (!item.contains("content")) throw request_error(p + "content", "missing message content");
                const auto& c = item["content"];
                const auto text = c.is_string() ? c.get<std::string>() :
                    text_parts(c, p + "content", role == "assistant" ? "output_text" : "input_text", role == "assistant");
                if (waiting_for_results) assistant()["content"].get_ref<std::string&>() += text;
                else if (role == "assistant" && !messages.empty() && messages.back().contains("reasoning_content") &&
                    messages.back()["content"] == "") messages.back()["content"] = text;
                else messages.push_back({{"role",role},{"content",text}});
            } else if (type == "reasoning") {
                if (receiving_results) throw request_error(p + "type", "expected outstanding function results first");
                if (item.contains("encrypted_content") && !item["encrypted_content"].is_null())
                    throw request_error(p + "encrypted_content", "cannot replay encrypted reasoning");
                auto reasoning = text_parts(member(item, "summary", p, json::value_t::array), p + "summary", "summary_text");
                if (item.contains("content") && !item["content"].is_null()) {
                    auto raw = text_parts(item["content"], p + "content", "reasoning_text");
                    if (!raw.empty()) reasoning = std::move(raw);
                }
                if (!reasoning.empty()) {
                    if (!waiting_for_results && !messages.empty() && messages.back()["role"] == "assistant" &&
                        messages.back()["content"] != "")
                        messages.push_back({{"role","assistant"},{"content",""}});
                    auto& turn = assistant();
                    if (!turn.contains("reasoning_content")) turn["reasoning_content"] = "";
                    turn["reasoning_content"].get_ref<std::string&>() += reasoning;
                }
            } else if (type == "function_call" || type == "function_call_output") {
                const auto id = member(item, "call_id", p, json::value_t::string).get<std::string>();
                if (id.empty()) throw request_error(p + "call_id", "expected a non-empty call ID");
                if (type == "function_call") {
                    if (receiving_results) throw request_error(p + "type", "expected outstanding function results first");
                    if (!seen.insert(id).second) throw request_error(p + "call_id", "duplicate call ID");
                    pending.insert(id);
                    assistant()["tool_calls"].push_back({{"id",id},{"type","function"},{"function",{
                        {"name",member(item, "name", p, json::value_t::string)},
                        {"arguments",member(item, "arguments", p, json::value_t::string)}}}});
                } else {
                    if (!pending.erase(id)) throw request_error(p + "call_id", "no matching outstanding function call");
                    if (!item.contains("output")) throw request_error(p + "output", "missing function output");
                    const auto& out = item["output"];
                    messages.push_back({{"role","tool"},{"tool_call_id",id},{"content",out.is_string() ?
                        out.get<std::string>() : text_parts(out, p + "output", "input_text")}});
                }
            } else throw request_error(p + "type", "unsupported input item");
        }
        if (!pending.empty()) throw request_error("input", "missing output for a function call");
        // Some local templates print tool results without IDs. Put each result next
        // to its corresponding call position even if the client replied in reverse.
        if (auto error = flm::chat::order_parallel_tool_results(messages))
            throw request_error("input", error->message);
    } else throw request_error("input", "expected text or a non-empty array of input items");
    chat["messages"] = std::move(messages);
    return chat;
}
} // namespace flm::responses
