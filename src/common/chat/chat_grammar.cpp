/// \file chat_grammar.cpp
/// \brief Grammar constraints for tool calls and response_format, on xgrammar

#include "chat/chat_grammar.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iterator>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <xgrammar/xgrammar.h>

#include "utils/debug_utils.hpp"

namespace {

using json = nlohmann::ordered_json;

// ---------------------------------------------------------------- structural tag pieces

json any_text(const std::vector<std::string>& excludes = {}) {
    json j = {{"type", "any_text"}};
    if (!excludes.empty()) {
        j["excludes"] = excludes;
    }
    return j;
}

json const_string(const std::string& value) {
    return {{"type", "const_string"}, {"value", value}};
}

json tag(const std::string& begin, json content, json end) {
    return {{"type", "tag"}, {"begin", begin}, {"content", std::move(content)}, {"end", std::move(end)}};
}

json sequence(std::vector<json> elements) {
    return {{"type", "sequence"}, {"elements", std::move(elements)}};
}

json optional_format(json content) {
    return {{"type", "optional"}, {"content", std::move(content)}};
}

/// \note Whitespace is bounded everywhere: a model kept from what it wants to write
///       (a ``` fence before JSON, say) otherwise tends to pad with blanks forever.
json whitespace() {
    return {{"type", "regex"}, {"pattern", "[ \\t\\r\\n]{0,8}"}};
}

constexpr int MAX_JSON_WHITESPACE = 16;  // room for indented JSON, not for endless padding

json schema_content(const json& schema, const char* style = nullptr) {
    json j = {{"type", "json_schema"}, {"json_schema", schema}, {"max_whitespace_cnt", MAX_JSON_WHITESPACE}};
    if (style != nullptr) {
        j["style"] = style;
    }
    return j;
}

/// \brief The arguments schema as Qwen XML spells top level values
/// \note The Qwen3.5+ template renders a parameter with Jinja's |string, so the
///       model learned Python's True/False for booleans, while xgrammar's qwen_xml
///       style only allows JSON's true/false. Strings are written raw in this
///       style, so a string enum lets both through; the parser reads either.
json qwen_xml_schema(json schema) {
    if (!schema.is_object() || !schema.contains("properties") || !schema["properties"].is_object()) {
        return schema;
    }
    for (auto& [name, property] : schema["properties"].items()) {
        if (property.is_object() && property.value("type", json()) == "boolean") {
            json replacement = {{"type", "string"}, {"enum", {"True", "False", "true", "false"}}};
            if (property.contains("description")) replacement["description"] = property["description"];
            property = replacement;
        }
    }
    return schema;
}

json grammar_content(const std::string& ebnf) {
    return {{"type", "grammar"}, {"grammar", ebnf}};
}

// ---------------------------------------------------------------- tools of a request

/// \brief Spell out what a schema leaves implicit where xgrammar needs it
/// \note An array without "items" means any items; the qwen_xml style rejects
///       every value of such an array unless that is written as "items": {}.
json normalize_schema(json schema) {
    if (schema.is_array()) {
        for (auto& item : schema) item = normalize_schema(std::move(item));
        return schema;
    }
    if (!schema.is_object()) {
        return schema;
    }
    if (schema.value("type", json()) == "array" && !schema.contains("items") && !schema.contains("prefixItems")) {
        schema["items"] = json::object();
    }
    for (auto& [key, value] : schema.items()) {
        if (value.is_object() || value.is_array()) {
            value = normalize_schema(std::move(value));
        }
    }
    return schema;
}

struct tool_def {
    std::string name;
    json parameters; // a JSON schema, or true for any arguments
};

std::vector<tool_def> request_tools(const chat_constraint_request& request) {
    std::vector<tool_def> tools;
    if (request.tools == nullptr || !request.tools->is_array()) {
        return tools;
    }
    for (const auto& entry : *request.tools) {
        if (!entry.contains("function") || !entry["function"].is_object()) {
            continue;
        }
        const json& fn = entry["function"];
        tool_def tool;
        tool.name = fn.value("name", "");
        if (tool.name.empty()) {
            continue;
        }
        if (request.tool_choice == chat_tool_choice::FUNCTION && tool.name != request.function_name) {
            continue;
        }
        // as xgrammar does: no schema, or a non-strict tool, still gets well formed arguments
        const bool strict = fn.value("strict", true);
        tool.parameters = strict && fn.contains("parameters") && fn["parameters"].is_object()
            ? normalize_schema(fn["parameters"]) : json(true);
        tools.push_back(std::move(tool));
    }
    return tools;
}

// ---------------------------------------------------------------- argument grammars (EBNF)

std::string ebnf_literal(const std::string& text) {
    std::string out = "\"";
    for (char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        if (c == '\n') { out += "\\n"; continue; }
        out += c;
    }
    return out + "\"";
}

/// \brief The rule for one parameter's value, picked from its schema type
std::string value_rule(const json& property, const char* generic) {
    if (!property.is_object()) {
        return generic;
    }
    if (property.contains("enum") && property["enum"].is_array() && !property["enum"].empty()) {
        bool all_strings = true;
        for (const auto& v : property["enum"]) all_strings = all_strings && v.is_string();
        if (all_strings) {
            return "";  // the caller spells the alternatives out
        }
    }
    const std::string type = property.contains("type") && property["type"].is_string()
        ? property["type"].get<std::string>() : "";
    if (type == "string") return "string";
    if (type == "integer") return "integer";
    if (type == "number") return "number";
    if (type == "boolean") return "boolean";
    if (type == "array") return "array";
    if (type == "object") return "object";
    return generic;
}

/// \brief EBNF for one property's value, from its schema type
/// \param quote_string spells a string enum value the way the dialect writes strings
/// \param array_sep EBNF between two array items
/// \note An array follows its "items" schema, so enums and types inside lists are
///       kept too; objects nested in a value are only held to the dialect's syntax.
std::string property_value(const json& property, const std::function<std::string(const std::string&)>& quote_string,
                           const std::string& array_sep) {
    std::string value = value_rule(property, "value");
    if (value == "array" && property.contains("items") && property["items"].is_object() &&
        !property["items"].empty()) {
        const std::string item = property_value(property["items"], quote_string, array_sep);
        return "(\"[\" (" + item + " (" + array_sep + " " + item + ")*)? \"]\")";
    }
    if (!value.empty()) {
        return value;
    }
    value = "(";
    bool first = true;
    for (const auto& option : property["enum"]) {
        value += (first ? "" : " | ") + quote_string(option.get<std::string>());
        first = false;
    }
    return value + ")";
}

/// \brief Rules for the argument list of one call
/// \param prefix keeps the rule names of different tools apart in one grammar
/// \param kv_sep EBNF between a key and its value
/// \param item_sep EBNF between two arguments
/// \param array_sep EBNF between two items of an array value
/// \param sorted keys come in alphabetical order (Gemma4's template uses dictsort),
///        otherwise in declaration order
/// \return rules whose list rule is <prefix>args; a schema without declared
///         properties gets any key/value list, built from the caller's "pair" rule
/// \note The declared properties come in one fixed order, each required one present
///       and each optional one free to be left out. Rule first<i> is the list
///       starting at property i, rest<i> the remainder after something was written.
std::string argument_list_rules(const json& schema, const std::string& prefix, const std::string& kv_sep,
                                const std::string& item_sep, const std::string& array_sep,
                                const std::function<std::string(const std::string&)>& quote_string, bool sorted) {
    std::ostringstream g;
    const bool declared = schema.is_object() && schema.contains("properties") && schema["properties"].is_object() &&
                          !schema["properties"].empty() && schema.value("additionalProperties", false) == false;
    if (!declared) {
        g << prefix << "args ::= (pair (" << item_sep << " pair)*)?\n";
        return g.str();
    }
    std::vector<std::string> names;
    for (const auto& [name, property] : schema["properties"].items()) {
        names.push_back(name);
    }
    if (sorted) {
        std::sort(names.begin(), names.end());
    }
    std::set<std::string> required;
    for (const auto& name : schema.value("required", json::array())) {
        if (name.is_string()) required.insert(name.get<std::string>());
    }

    const size_t n = names.size();
    auto rule = [&prefix](const char* kind, size_t i) { return prefix + kind + std::to_string(i); };
    for (size_t i = 0; i < n; i++) {
        g << rule("p", i) << " ::= " << ebnf_literal(names[i]) << " " << kv_sep << " "
          << property_value(schema["properties"][names[i]], quote_string, array_sep) << "\n";
    }
    for (size_t i = 0; i <= n; i++) {
        if (i == n) {
            g << rule("first", i) << " ::= \"\"\n" << rule("rest", i) << " ::= \"\"\n";
            continue;
        }
        const std::string with_first = rule("p", i) + " " + rule("rest", i + 1);
        const std::string with_rest = item_sep + " " + rule("p", i) + " " + rule("rest", i + 1);
        if (required.count(names[i])) {
            g << rule("first", i) << " ::= " << with_first << "\n";
            g << rule("rest", i) << " ::= " << with_rest << "\n";
        } else {
            g << rule("first", i) << " ::= (" << with_first << ") | " << rule("first", i + 1) << "\n";
            g << rule("rest", i) << " ::= (" << with_rest << ") | " << rule("rest", i + 1) << "\n";
        }
    }
    g << prefix << "args ::= " << rule("first", 0) << "\n";
    return g.str();
}

/// \brief Numbers with bounded digits: a model kept from writing "13.9 million" will
///        otherwise go on adding digits until it runs out of tokens
const char* BOUNDED_NUMBER_RULES =
    "integer ::= \"-\"? (\"0\" | [1-9] [0-9]{0,15})\n"
    "number ::= integer (\".\" [0-9]{1,15})? ([eE] [+-]? [0-9]{1,3})?\n";

/// \brief Gemma4 call arguments: {key:<|"|>text<|"|>,n:5,flag:true,list:[...]}
std::string gemma4_arguments_grammar(const json& schema) {
    auto quote = [](const std::string& s) { return ebnf_literal("<|\"|>" + s + "<|\"|>"); };
    std::ostringstream g;
    g << "root ::= \"{\" args \"}\"\n"
      << argument_list_rules(schema, "", "\":\"", "\",\"", "\",\"", quote, true)
      << "pair ::= ident \":\" value\n"
      << "ident ::= [a-zA-Z_] [a-zA-Z0-9_]*\n"
      << "value ::= string | number | boolean | \"null\" | object | array\n"
      << "string ::= \"<|\\\"|>\" ([^<] | \"<\" [^|])* \"<|\\\"|>\"\n"
      << BOUNDED_NUMBER_RULES
      << "boolean ::= \"true\" | \"false\"\n"
      << "object ::= \"{\" (pair (\",\" pair)*)? \"}\"\n"
      << "array ::= \"[\" (value (\",\" value)*)? \"]\"\n";
    return g.str();
}

/// \brief The calls inside one LFM2 block: f(key="text", n=5), g(flag=True)
std::string python_calls_grammar(const std::vector<tool_def>& tools, bool parallel) {
    auto quote = [](const std::string& s) { return ebnf_literal("\"" + s + "\""); };
    std::ostringstream g;
    g << "root ::= call" << (parallel ? " (\",\" \" \"? call)*" : "") << "\n";
    g << "call ::= ";
    for (size_t i = 0; i < tools.size(); i++) {
        g << (i ? " | " : "") << "call_" << i;
    }
    g << "\n";
    for (size_t i = 0; i < tools.size(); i++) {
        const std::string prefix = "t" + std::to_string(i) + "_";
        g << "call_" << i << " ::= " << ebnf_literal(tools[i].name + "(") << " " << prefix << "args \")\"\n"
          << argument_list_rules(tools[i].parameters, prefix, "\"=\"", "\",\" \" \"?", "\",\" \" \"?", quote, false);
    }
    g << "pair ::= ident \"=\" value\n"
      << "ident ::= [a-zA-Z_] [a-zA-Z0-9_]*\n"
      << "value ::= string | number | boolean | \"None\" | object | array\n"
      << "string ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" [^\\n])* \"\\\"\" | \"'\" ([^'\\\\] | \"\\\\\" [^\\n])* \"'\"\n"
      << BOUNDED_NUMBER_RULES
      << "boolean ::= \"True\" | \"False\"\n"
      << "object ::= \"{\" (string \":\" \" \"? value (\",\" \" \"? string \":\" \" \"? value)*)? \"}\"\n"
      << "array ::= \"[\" (value (\",\" \" \"? value)*)? \"]\"\n";
    return g.str();
}

// ---------------------------------------------------------------- per format tool tags

struct tool_syntax {
    std::string trigger;
    std::vector<json> tags;
};

tool_syntax tool_tags(const chat_format_spec& spec, const std::vector<tool_def>& tools, bool parallel) {
    tool_syntax syntax;
    if (spec.kind == chat_format_kind::LFM2_PYTHONIC) {
        // LFM2 lists every call of a turn in one block, so one tag holds them all
        syntax.trigger = "<|tool_call_start|>";
        syntax.tags.push_back(tag("<|tool_call_start|>[", grammar_content(python_calls_grammar(tools, parallel)),
                                  "]<|tool_call_end|>"));
        return syntax;
    }
    for (const auto& tool : tools) {
        switch (spec.kind) {
            case chat_format_kind::HERMES_JSON:
                // xgrammar's qwen_3 style
                syntax.trigger = "<tool_call>";
                syntax.tags.push_back(tag("<tool_call>\n{\"name\": \"" + tool.name + "\", \"arguments\": ",
                                          schema_content(tool.parameters), "}\n</tool_call>"));
                break;
            case chat_format_kind::QWEN_XML:
                // xgrammar's qwen_3_5 style
                syntax.trigger = "<tool_call>\n<function=";
                syntax.tags.push_back(tag("<tool_call>\n<function=" + tool.name + ">\n",
                                          schema_content(qwen_xml_schema(tool.parameters), "qwen_xml"),
                                          "\n</function>\n</tool_call>"));
                break;
            case chat_format_kind::GEMMA4:
                syntax.trigger = "<|tool_call>";
                syntax.tags.push_back(tag("<|tool_call>call:" + tool.name,
                                          grammar_content(gemma4_arguments_grammar(tool.parameters)), "<tool_call|>"));
                break;
            case chat_format_kind::LFM2_PYTHONIC:
            case chat_format_kind::CONTENT_ONLY:
            case chat_format_kind::HARMONY:
                break;
        }
    }
    return syntax;
}

/// \brief Optional reasoning block in front of the answer, for tag formats
/// \param allow false when the request turned thinking off; models that always
///        open with reasoning keep the block, their prefill already holds the tag
json reasoning_prefix(const chat_format_spec& spec, bool allow) {
    if (!spec.has_reasoning() || !(allow || spec.reasoning_forced)) {
        return nullptr;
    }
    // the prefill is fed to the grammar too, so a block the template opened is
    // already inside this optional tag when generation starts
    return optional_format(sequence({
        tag(spec.reasoning_start, any_text({spec.reasoning_start, spec.reasoning_end}), spec.reasoning_end),
        whitespace(),
    }));
}

json response_schema(const chat_constraint_request& request) {
    if (!request.json_schema.is_null()) {
        return request.json_schema;
    }
    return json{{"type", "object"}};
}

bool has_response_format(const chat_constraint_request& request) {
    return request.json_object || !request.json_schema.is_null();
}

json harmony_format(const chat_constraint_request& request, const std::vector<tool_def>& tools) {
    // xgrammar's harmony style
    const std::string CALL_END = "<|call|>";
    const json FINAL_END = json::array({"<|end|>", "<|return|>"});
    const std::string SEPARATOR = "<|start|>assistant";

    std::vector<json> call_tags;
    const bool want_tools = request.tool_choice == chat_tool_choice::REQUIRED ||
                            request.tool_choice == chat_tool_choice::FUNCTION ||
                            (request.tool_choice == chat_tool_choice::AUTO && !has_response_format(request));
    if (want_tools) {
        for (const auto& tool : tools) {
            json content = schema_content(tool.parameters);
            // gpt-oss writes "to=functions.x <|constrain|>json"; xgrammar's style has no space
            call_tags.push_back(tag("<|channel|>commentary to=functions." + tool.name + " <|constrain|>json<|message|>", content, CALL_END));
            call_tags.push_back(tag("<|channel|>commentary to=functions." + tool.name + "<|constrain|>json<|message|>", content, CALL_END));
            call_tags.push_back(tag(" to=functions." + tool.name + "<|channel|>commentary <|constrain|>json<|message|>", content, CALL_END));
            call_tags.push_back(tag(" to=functions." + tool.name + "<|channel|>commentary json<|message|>", content, CALL_END));
        }
    }
    std::vector<json> message_tags;
    message_tags.push_back(tag("<|channel|>analysis<|message|>", any_text(), FINAL_END));
    if (request.tool_choice == chat_tool_choice::AUTO || request.tool_choice == chat_tool_choice::NONE) {
        json final_content = has_response_format(request) ? schema_content(response_schema(request)) : any_text();
        message_tags.push_back(tag("<|channel|>final<|message|>", final_content, FINAL_END));
    }

    if (call_tags.empty()) {
        return {{"type", "tags_with_separator"}, {"tags", message_tags}, {"separator", SEPARATOR}, {"at_least_one", true}};
    }
    const bool must_call = request.tool_choice != chat_tool_choice::AUTO;
    if (request.parallel_tool_calls && !must_call) {
        std::vector<json> all = call_tags;
        all.insert(all.end(), message_tags.begin(), message_tags.end());
        return {{"type", "tags_with_separator"}, {"tags", all}, {"separator", SEPARATOR}};
    }
    // any number of reasoning/answer messages, then the call(s), which end the turn
    const bool one_call = !request.parallel_tool_calls || request.tool_choice == chat_tool_choice::FUNCTION;
    json calls = {{"type", "tags_with_separator"}, {"tags", call_tags}, {"separator", SEPARATOR},
                  {"at_least_one", true}, {"stop_after_first", one_call}};
    json messages = {{"type", "tags_with_separator"}, {"tags", message_tags}, {"separator", SEPARATOR},
                     {"at_least_one", true}};
    json messages_then_calls = sequence({messages, const_string(SEPARATOR), calls});
    std::vector<json> options = {calls, messages_then_calls};
    if (!must_call) {
        options.insert(options.begin(), messages);
    }
    return {{"type", "or"}, {"elements", options}};
}

/// \brief Stop tokens the grammar matches as text
/// \note xgrammar never lets a stop token stand for text, so these stay out of the
///       stop list although the model's eos list has them (gpt-oss ends a call with <|call|>)
const std::set<std::string>& text_markers(const chat_format_spec& spec) {
    static const std::set<std::string> harmony = {"<|call|>", "<|return|>", "<|end|>"};
    static const std::set<std::string> none;
    return spec.kind == chat_format_kind::HARMONY ? harmony : none;
}

/// \brief What to drop from the prefill before feeding it to the grammar
std::string prefill_skip(const chat_format_spec& spec) {
    // the harmony prefill opens with the role of the "<|start|>assistant" header,
    // which the grammar does not cover
    return spec.kind == chat_format_kind::HARMONY ? "assistant" : "";
}

// ---------------------------------------------------------------- the constraint

class xgrammar_constraint : public decode_constraint {
public:
    /// \param with_reasoning when the grammar leaves reasoning out, the same grammar with
    ///        it: a model made to open a reasoning block switches to this one
    xgrammar_constraint(const xgrammar::CompiledGrammar& grammar, std::optional<xgrammar::CompiledGrammar> with_reasoning,
                        std::string prefill_skip, size_t vocab_size)
        : matcher_(grammar, std::nullopt, false, MAX_ROLLBACK),
          with_reasoning_(std::move(with_reasoning)),
          skip_(std::move(prefill_skip)),
          words_((vocab_size + 31) / 32) {}

    void start(const std::string& prefill) override {
        std::string text = prefill;
        if (!skip_.empty() && text.compare(0, skip_.size(), skip_) == 0) {
            text = text.substr(skip_.size());
        }
        prefill_ = text;
        if (!text.empty() && !matcher_.AcceptString(text)) {
            give_up("the grammar does not accept the prompt's prefill \"" + text + "\"");
        }
    }

    bool fill_mask(std::vector<uint32_t>& mask) override {
        if (failed_) {
            return false;
        }
        mask.resize(words_);
        int64_t shape[1] = {static_cast<int64_t>(words_)};
        DLTensor tensor;
        tensor.data = mask.data();
        tensor.device = DLDevice{kDLCPU, 0};
        tensor.ndim = 1;
        tensor.dtype = DLDataType{kDLInt, 32, 1};
        tensor.shape = shape;
        tensor.strides = nullptr;
        tensor.byte_offset = 0;
        return matcher_.FillNextTokenBitmask(&tensor);
    }

    void accept(int token) override {
        if (failed_) {
            return;
        }
        if (!matcher_.AcceptToken(token)) {
            give_up("the grammar rejected token " + std::to_string(token));
            return;
        }
        history_.push_back(token);
    }

    void accept_forced(int token) override {
        if (failed_) {
            return;
        }
        if (matcher_.AcceptToken(token)) {
            history_.push_back(token);
            return;
        }
        if (!with_reasoning_) {
            give_up("the grammar rejected forced token " + std::to_string(token));
            return;
        }
        // the model is made to reason although the request turned thinking off:
        // carry on with the grammar that has the reasoning block, from the same point
        xgrammar::GrammarMatcher reasoning(*with_reasoning_, std::nullopt, false, MAX_ROLLBACK);
        with_reasoning_.reset();
        bool ok = prefill_.empty() || reasoning.AcceptString(prefill_);
        for (size_t i = 0; ok && i < history_.size(); i++) {
            ok = reasoning.AcceptToken(history_[i]);
        }
        if (!ok || !reasoning.AcceptToken(token)) {
            give_up("the grammar rejected forced token " + std::to_string(token));
            return;
        }
        matcher_ = std::move(reasoning);
        history_.push_back(token);
    }

    void rollback(int n) override {
        if (!failed_) {
            matcher_.Rollback(n);
            history_.resize(history_.size() > static_cast<size_t>(n) ? history_.size() - n : 0);
        }
    }

private:
    static constexpr int MAX_ROLLBACK = 8;

    void give_up(const std::string& why) {
        // a desynchronised grammar would only produce garbage; let the model finish free
        header_print("WARNING", "Constrained decoding disabled for this request: " << why);
        failed_ = true;
    }

    xgrammar::GrammarMatcher matcher_;
    std::optional<xgrammar::CompiledGrammar> with_reasoning_;
    std::string skip_;
    std::string prefill_;
    std::vector<int> history_;  // accepted tokens, to replay into with_reasoning_
    size_t words_;
    bool failed_ = false;
};

/// \brief An xgrammar error as the client should see it
/// \note xgrammar prefixes its messages with "[hh:mm:ss] <source file>:<line>: ",
///       which tells the client nothing and leaks the build machine's paths.
static std::string client_message(std::string message) {
    if (!message.empty() && message.front() == '[') {
        size_t close = message.find("] ");
        size_t location = close == std::string::npos ? std::string::npos : message.find(": ", close + 2);
        if (location != std::string::npos) {
            message = message.substr(location + 2);
        }
    }
    while (!message.empty() && (message.back() == '\n' || message.back() == ' ')) {
        message.pop_back();
    }
    return message;
}

/// \brief Token strings in id order, as tokenizer.json stores them
std::vector<std::string> encoded_vocab(const nlohmann::json& tokenizer) {
    std::vector<std::string> vocab;
    const auto& model_vocab = tokenizer["model"]["vocab"];
    if (model_vocab.is_object()) {
        for (const auto& [token, id] : model_vocab.items()) {
            size_t i = id.get<size_t>();
            if (i >= vocab.size()) vocab.resize(i + 1);
            vocab[i] = token;
        }
    } else if (model_vocab.is_array()) {
        // unigram: [[piece, score], ...] in id order
        for (const auto& entry : model_vocab) {
            vocab.push_back(entry[0].get<std::string>());
        }
    }
    for (const auto& added : tokenizer.value("added_tokens", nlohmann::json::array())) {
        size_t i = added["id"].get<size_t>();
        if (i >= vocab.size()) vocab.resize(i + 1);
        vocab[i] = added["content"].get<std::string>();
    }
    return vocab;
}

} // namespace

json chat_structural_tag(const chat_format_spec& spec, const chat_constraint_request& request) {
    const std::vector<tool_def> tools = request_tools(request);
    // auto only constrains a format that has a call syntax; required and forced
    // calls on one that has none are an error below
    const bool tool_grammar = request.tool_choice == chat_tool_choice::REQUIRED ||
                              request.tool_choice == chat_tool_choice::FUNCTION ||
                              (request.tool_choice == chat_tool_choice::AUTO && !tools.empty() &&
                               !has_response_format(request) &&
                               (spec.has_tools() || spec.kind == chat_format_kind::HARMONY));
    if (!tool_grammar && !has_response_format(request)) {
        return nullptr;
    }
    if (tool_grammar && tools.empty()) {
        throw std::invalid_argument("tool_choice asks for a tool call but no usable tool is defined");
    }

    json format;
    if (spec.kind == chat_format_kind::HARMONY) {
        format = harmony_format(request, tools);
    } else if (tool_grammar) {
        if (!spec.has_tools()) {
            throw std::invalid_argument(std::string("this model's chat format (") + chat_format_name(spec.kind) +
                                        ") has no tool call syntax to constrain");
        }
        // a forced function is exactly one call
        const bool parallel = request.parallel_tool_calls && request.tool_choice != chat_tool_choice::FUNCTION;
        tool_syntax syntax = tool_tags(spec, tools, parallel);
        json suffix;
        if (request.tool_choice == chat_tool_choice::FUNCTION) {
            suffix = sequence({whitespace(), syntax.tags.front()});
        } else {
            std::vector<std::string> excludes;
            if (spec.has_reasoning()) {
                excludes = {spec.reasoning_start, spec.reasoning_end};
            }
            suffix = {{"type", "triggered_tags"}, {"triggers", {syntax.trigger}}, {"tags", syntax.tags},
                      {"stop_after_first", !request.parallel_tool_calls}};
            if (!excludes.empty()) suffix["excludes"] = excludes;
            if (request.tool_choice == chat_tool_choice::REQUIRED) suffix["at_least_one"] = true;
        }
        json prefix = reasoning_prefix(spec, request.allow_reasoning);
        format = prefix.is_null() ? suffix : sequence({prefix, suffix});
    } else {
        json suffix = sequence({whitespace(), schema_content(response_schema(request))});
        json prefix = reasoning_prefix(spec, request.allow_reasoning);
        format = prefix.is_null() ? suffix : sequence({prefix, suffix});
    }
    return {{"type", "structural_tag"}, {"format", format}};
}

struct chat_grammar_engine::impl {
    std::vector<std::string> vocab;
    size_t vocab_size = 0;
    std::unique_ptr<xgrammar::TokenizerInfo> info;
    std::unique_ptr<xgrammar::GrammarCompiler> compiler;
};

chat_grammar_engine::chat_grammar_engine(const chat_vocab& vocab, const chat_format_spec& spec)
    : impl_(std::make_unique<impl>()) {
    std::ifstream file(vocab.tokenizer_json, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open " + vocab.tokenizer_json + " for the grammar engine");
    }
    const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    impl_->vocab = encoded_vocab(nlohmann::json::parse(content));
    impl_->vocab_size = std::max(vocab.size, impl_->vocab.size());

    std::vector<int32_t> stops;
    const auto& markers = text_markers(spec);
    for (int id : vocab.stop_tokens) {
        if (id < 0 || static_cast<size_t>(id) >= impl_->vocab.size() || !markers.count(impl_->vocab[id])) {
            stops.push_back(id);
        }
    }

    const json metadata = json::parse(xgrammar::TokenizerInfo::DetectMetadataFromHF(content));
    impl_->info = std::make_unique<xgrammar::TokenizerInfo>(
        impl_->vocab, static_cast<xgrammar::VocabType>(metadata.value("vocab_type", 0)),
        static_cast<int>(impl_->vocab_size), stops,
        metadata.value("add_prefix_space", false));
    impl_->compiler = std::make_unique<xgrammar::GrammarCompiler>(*impl_->info);
}

chat_grammar_engine::~chat_grammar_engine() = default;

std::shared_ptr<decode_constraint> chat_grammar_engine::make(const chat_format_spec& spec,
                                                             const chat_constraint_request& request) {
    const json structural_tag = chat_structural_tag(spec, request);
    if (structural_tag.is_null()) {
        return nullptr;
    }
    auto compile = [this](const json& tag) {
        try {
            return impl_->compiler->CompileStructuralTag(tag.dump());
        } catch (const std::exception& e) {
            throw std::invalid_argument("cannot build the decoding grammar: " + client_message(e.what()));
        }
    };
    xgrammar::CompiledGrammar grammar = compile(structural_tag);

    // thinking turned off keeps reasoning out of the grammar, so no spelling of the
    // opener can be sampled; a thinking-only model is still made to open it
    std::optional<xgrammar::CompiledGrammar> with_reasoning;
    if (!request.allow_reasoning && spec.has_reasoning() && !spec.reasoning_forced) {
        chat_constraint_request reasoning_request = request;
        reasoning_request.allow_reasoning = true;
        with_reasoning = compile(chat_structural_tag(spec, reasoning_request));
    }
    try {
        return std::make_shared<xgrammar_constraint>(grammar, with_reasoning, prefill_skip(spec), impl_->vocab_size);
    } catch (const std::exception& e) {
        throw std::invalid_argument("cannot start the decoding grammar: " + client_message(e.what()));
    }
}

bool chat_grammar_engine::accepts(const chat_format_spec& spec, const chat_constraint_request& request,
                                  const std::string& prefill, const std::string& text, std::string& error) {
    const json structural_tag = chat_structural_tag(spec, request);
    if (structural_tag.is_null()) {
        return true;
    }
    xgrammar::GrammarMatcher matcher(impl_->compiler->CompileStructuralTag(structural_tag.dump()));
    std::string full = prefill;
    const std::string skip = prefill_skip(spec);
    if (!skip.empty() && full.compare(0, skip.size(), skip) == 0) {
        full = full.substr(skip.size());
    }
    full += text;
    // feed in small pieces so a rejection can be located
    for (size_t pos = 0; pos < full.size();) {
        size_t n = std::min<size_t>(16, full.size() - pos);
        // do not split a UTF-8 sequence
        while (pos + n < full.size() && (static_cast<unsigned char>(full[pos + n]) & 0xC0) == 0x80) n++;
        if (!matcher.AcceptString(full.substr(pos, n))) {
            error = "rejected at byte " + std::to_string(pos) + ": ..." +
                    full.substr(pos >= 40 ? pos - 40 : 0, std::min<size_t>(pos, 40)) + "[" + full.substr(pos, n) + "]";
            return false;
        }
        pos += n;
    }
    return true;
}
