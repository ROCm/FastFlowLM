/// \file chat_gemma4_tools.cpp
/// \brief Prepare a request's tools for the Gemma4 chat template
#include "chat/chat_gemma4_tools.hpp"

namespace {
/// \brief Rewrite the nullable types in a JSON schema and in the schemas it holds
/// \note Only the schemas whose type the template upper-cases are followed, so literal values
///       such as an enum's are left alone
void write_nullable_types(nlohmann::ordered_json& schema) {
    if (!schema.is_object()) {
        return;
    }
    // [T, "null"] -> T, nullable: true
    auto type = schema.find("type");
    if (type != schema.end() && type->is_array() && type->size() == 2 &&
        ((*type)[0] == "null") != ((*type)[1] == "null")) {
        *type = (*type)[0] == "null" ? (*type)[1] : (*type)[0];
        schema["nullable"] = true;
    }
    if (schema.contains("properties") && schema["properties"].is_object()) {
        for (auto& property : schema["properties"]) {
            write_nullable_types(property);
        }
    }
    // an object without properties has its other keys rendered as parameters, additionalProperties among them
    for (const char* key : {"items", "additionalProperties"}) {
        if (schema.contains(key)) {
            write_nullable_types(schema[key]);
        }
    }
}
} // namespace

nlohmann::ordered_json gemma4_template_tools(nlohmann::ordered_json tools) {
    for (auto& tool : tools) {
        if (tool.is_object() && tool.contains("function") && tool["function"].is_object() &&
            tool["function"].contains("parameters")) {
            write_nullable_types(tool["function"]["parameters"]);
        }
    }
    return tools;
}
