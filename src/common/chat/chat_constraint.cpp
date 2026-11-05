/// \file chat_constraint.cpp
/// \brief Build the decoding constraint a chat request asks for

#include "chat/chat_constraint.hpp"
#include "chat/chat_grammar.hpp"

using json = nlohmann::ordered_json;

namespace {

/// \brief A field of a request object, null when it is missing
/// \note Some clients send null for every option they leave unset, so null
///       has to read as missing; json::value() throws on it instead.
const json& field(const json& object, const char* key) {
    static const json missing;
    auto it = object.find(key);
    return it == object.end() ? missing : *it;
}

/// \brief Whether a message calls a tool or answers one
bool has_tool_turns(const json& messages) {
    if (!messages.is_array()) {
        return false;
    }
    for (const auto& message : messages) {
        if (!message.is_object()) {
            continue;
        }
        const json& role = field(message, "role");
        const json& calls = field(message, "tool_calls");
        if (role == "tool" || role == "function" || (calls.is_array() && !calls.empty())) {
            return true;
        }
    }
    return false;
}

} // namespace

bool chat_constraint_request_from_openai(const json& request, const json& tools,
                                         chat_constraint_request& out, std::string& error) {
    out = chat_constraint_request();
    out.tools = &tools;
    const json& parallel = field(request, "parallel_tool_calls");
    if (!parallel.is_null() && !parallel.is_boolean()) {
        error = "parallel_tool_calls must be a boolean";
        return false;
    }
    out.parallel_tool_calls = parallel.is_null() || parallel.get<bool>();
    // thinking turned off: some models still get an empty think block fed in, but
    // the grammar must not offer room to reason
    const json& think = field(request, "think");
    if ((think.is_boolean() && !think.get<bool>()) || field(request, "reasoning_effort") == "none") {
        out.allow_reasoning = false;
    }

    const json& choice_field = field(request, "tool_choice");
    const json choice = choice_field.is_null() ? json("auto") : choice_field;
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
    } else if (choice.is_object() && field(choice, "type") == "function" &&
               field(field(choice, "function"), "name").is_string()) {
        out.tool_choice = chat_tool_choice::FUNCTION;
        out.function_name = choice["function"]["name"].get<std::string>();
    } else {
        error = "tool_choice must be a string or {\"type\": \"function\", \"function\": {\"name\": ...}}";
        return false;
    }

    const bool no_tools = !tools.is_array() || tools.empty();
    if (out.tool_choice == chat_tool_choice::REQUIRED || out.tool_choice == chat_tool_choice::FUNCTION) {
        if (no_tools) {
            error = "tool_choice requires a tool call but the request has no tools";
            return false;
        }
    }
    // no tools, yet the conversation shows the model calling them: it calls again in
    // the same markup, which no parser reads for a request without tools. Banning the
    // opener costs speculative decoding, so a plain chat is left alone.
    if (out.tool_choice == chat_tool_choice::AUTO && no_tools && has_tool_turns(field(request, "messages"))) {
        out.tool_choice = chat_tool_choice::NONE;
    }
    if (out.tool_choice == chat_tool_choice::FUNCTION) {
        bool found = false;
        for (const auto& tool : tools) {
            const json& name = field(field(tool, "function"), "name");
            if (name.is_string() && name.get<std::string>() == out.function_name) {
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
        const json& type_field = field(format, "type");
        const std::string type = type_field.is_null() ? "text" : type_field.is_string() ? type_field.get<std::string>() : "";
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
