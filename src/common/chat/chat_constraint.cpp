/// \file chat_constraint.cpp
/// \brief Build the decoding constraint a chat request asks for

#include "chat/chat_constraint.hpp"
#include "chat/chat_grammar.hpp"

using json = nlohmann::ordered_json;

bool chat_constraint_request_from_openai(const json& request, const json& tools,
                                         chat_constraint_request& out, std::string& error) {
    out = chat_constraint_request();
    out.tools = &tools;
    out.parallel_tool_calls = request.value("parallel_tool_calls", true);
    // thinking turned off: some models still get an empty think block fed in, but
    // the grammar must not offer room to reason
    if (request.contains("think") && request["think"].is_boolean() && !request["think"].get<bool>()) {
        out.allow_reasoning = false;
    }
    if (request.value("reasoning_effort", "") == "none") {
        out.allow_reasoning = false;
    }

    const json choice = request.value("tool_choice", json("auto"));
    if (choice.is_string()) {
        const std::string mode = choice.get<std::string>();
        if (mode == "auto") {
            out.tool_choice = chat_tool_choice::AUTO;
        } else if (mode == "none") {
            out.tool_choice = chat_tool_choice::NONE;
        } else if (mode == "required") {
            out.tool_choice = chat_tool_choice::REQUIRED;
        } else {
            error = "tool_choice must be \"auto\", \"none\", \"required\" or a function, got \"" + mode + "\"";
            return false;
        }
    } else if (choice.is_object() && choice.value("type", "") == "function" && choice.contains("function") &&
               choice["function"].is_object() && choice["function"].contains("name")) {
        out.tool_choice = chat_tool_choice::FUNCTION;
        out.function_name = choice["function"]["name"].get<std::string>();
    } else {
        error = "tool_choice must be a string or {\"type\": \"function\", \"function\": {\"name\": ...}}";
        return false;
    }

    if (out.tool_choice == chat_tool_choice::REQUIRED || out.tool_choice == chat_tool_choice::FUNCTION) {
        if (!tools.is_array() || tools.empty()) {
            error = "tool_choice requires a tool call but the request has no tools";
            return false;
        }
    }
    if (out.tool_choice == chat_tool_choice::FUNCTION) {
        bool found = false;
        for (const auto& tool : tools) {
            if (tool.contains("function") && tool["function"].value("name", "") == out.function_name) {
                found = true;
            }
        }
        if (!found) {
            error = "tool_choice names function \"" + out.function_name + "\" which is not in tools";
            return false;
        }
    }

    if (request.contains("response_format") && request["response_format"].is_object()) {
        const json& format = request["response_format"];
        const std::string type = format.value("type", "text");
        if (type == "json_object") {
            out.json_object = true;
        } else if (type == "json_schema") {
            if (!format.contains("json_schema") || !format["json_schema"].contains("schema")) {
                error = "response_format json_schema needs json_schema.schema";
                return false;
            }
            out.json_schema = format["json_schema"]["schema"];
        } else if (type != "text") {
            error = "response_format type must be text, json_object or json_schema";
            return false;
        }
    }
    return true;
}

std::shared_ptr<decode_constraint> chat_make_constraint(const chat_format_spec& spec,
                                                        const chat_constraint_request& request,
                                                        const chat_vocab& vocab,
                                                        const std::function<chat_grammar_engine&()>& grammar_engine) {
    if (!chat_structural_tag(spec, request).is_null()) {
        return grammar_engine().make(spec, request);
    }
    if (request.tool_choice == chat_tool_choice::NONE && !spec.tool_start.empty()) {
        // a single token opens every tool call block: taking it away is enough
        std::vector<int> ids = vocab.encode(spec.tool_start);
        if (ids.size() == 1) {
            return std::make_shared<token_ban_constraint>(vocab.size, ids);
        }
    }
    return nullptr;
}
