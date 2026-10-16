/// \file chat_constraint.hpp
/// \brief Build the decoding constraint a chat request asks for
/// \note The constraint comes from the same chat_format_spec as the output
///       parser, so whatever the constraint lets through, the parser reads.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat/chat_format.hpp"
#include "chat/decode_constraint.hpp"

/// \brief OpenAI tool_choice
enum class chat_tool_choice {
    AUTO,     // the model decides
    NONE,     // no tool calls
    REQUIRED, // at least one tool call
    FUNCTION, // exactly the named function
};

/// \brief What the request asks to constrain
struct chat_constraint_request {
    chat_tool_choice tool_choice = chat_tool_choice::AUTO;
    std::string function_name;                      // for FUNCTION
    const nlohmann::ordered_json* tools = nullptr;  // the request's tools, may be null
    bool parallel_tool_calls = true;
    bool allow_reasoning = true;                    // false when the request turned thinking off
    bool json_object = false;                       // response_format {"type": "json_object"}
    nlohmann::ordered_json json_schema;             // response_format json_schema's schema, null if none
};

/// \brief The model's vocabulary as the constraint builders need it
struct chat_vocab {
    size_t size = 0; // width of the logits, may exceed the tokenizer's vocabulary
    std::function<std::vector<int>(const std::string&)> encode; // without any BOS
    std::vector<int> stop_tokens;                               // end of turn ids
    std::string tokenizer_json;                                 // path of the HF tokenizer.json
};

class chat_grammar_engine;

/// \brief Parse the OpenAI request fields into a constraint request
/// \param request the chat completion request body
/// \param tools the request's tools, must outlive the result
/// \param error set when a field is malformed
/// \return false on a malformed field
bool chat_constraint_request_from_openai(const nlohmann::ordered_json& request,
                                         const nlohmann::ordered_json& tools,
                                         chat_constraint_request& out, std::string& error);

/// \brief Build the constraint for one request
/// \param grammar_engine returns the model's grammar engine, only called when a grammar is needed
/// \return null when nothing needs constraining
/// \throws std::invalid_argument when the request's grammar cannot be built
std::shared_ptr<decode_constraint> chat_make_constraint(const chat_format_spec& spec,
                                                        const chat_constraint_request& request,
                                                        const chat_vocab& vocab,
                                                        const std::function<chat_grammar_engine&()>& grammar_engine);
