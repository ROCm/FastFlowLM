/// \file chat_grammar.cpp
/// \brief Grammar constraints for tool calls and response_format, on xgrammar

#include "chat/chat_grammar.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <variant>
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

/// \brief Text a raw string value may not hold: where the call goes on after it
/// \note As EBNF string literals. The separator between two parts of a call is at
///       most one whitespace character (see qwen_xml_arguments), so these are all of
///       its spellings; the parser ends a value, a function and a block the same way.
std::string qwen_xml_value_ends() {
    std::string out;
    for (const char* ws : {"", "\\n", " ", "\\t", "\\r"}) {
        for (const auto& [close, next] : {std::pair{"</parameter>", "<parameter="}, std::pair{"</parameter>", "</function>"},
                                          std::pair{"</function>", "</tool_call>"}, std::pair{"</function>", "<function="}}) {
            out += std::string(out.empty() ? "" : ", ") + "\"" + close + ws + next + "\"";
        }
    }
    return "excludes=(" + out + ")";
}

/// \brief The arguments of a Qwen XML call: xgrammar's qwen_xml grammar for the schema,
///        but a raw string value may hold "</parameter>"
/// \note xgrammar ends a raw string at the first "</parameter>", so a value that holds
///       one (XML, or code that writes tool calls) is cut there, and the model, made to
///       close the call, writes the rest of it as the answer. The value only has to end
///       where the call goes on after it, as the parser reads it: at a "</parameter>"
///       followed by the next parameter or by "</function>". The grammar is rebuilt
///       from xgrammar's own text with just that rule changed; should that ever fail,
///       the stock grammar stands. The templates put one newline between the parts of a
///       call, so one whitespace character is all the grammar allows there, which keeps
///       the list of what a value may not hold short and exact.
json qwen_xml_arguments(const json& schema) {
    // the same tools come with every request of a conversation, and the rebuild costs
    // far more than the rest of the structural tag
    static std::mutex mutex;
    static std::unordered_map<std::string, json> built;
    const std::string key = schema.dump();
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = built.find(key);
        if (it != built.end()) {
            return it->second;
        }
    }
    json content = {{"type", "json_schema"}, {"json_schema", qwen_xml_schema(schema)}, {"max_whitespace_cnt", 1},
                    {"style", "qwen_xml"}};
    static const std::string RULE = "\nxml_string ::= ";
    static const std::string EXCLUDES = "excludes=(\"</parameter>\")";
    try {
        auto grammar = xgrammar::Grammar::FromStructuralTag(json{{"type", "structural_tag"}, {"format", content}}.dump());
        if (std::holds_alternative<xgrammar::Grammar>(grammar)) {
            std::string ebnf = std::get<xgrammar::Grammar>(grammar).ToString();
            size_t rule = ebnf.find(RULE);
            size_t excludes = rule == std::string::npos ? rule : ebnf.find(EXCLUDES, rule);
            size_t rule_end = rule == std::string::npos ? rule : ebnf.find(" ::= ", rule + RULE.size());
            // no raw string in this schema, nothing to change
            if (excludes != std::string::npos && excludes < rule_end) {
                ebnf.replace(excludes, EXCLUDES.size(), qwen_xml_value_ends());
                xgrammar::Grammar::FromEBNF(ebnf); // throws when the text does not read back
                content = grammar_content(ebnf);
            }
        }
    } catch (const std::exception&) {
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (built.size() >= 256) {
        built.clear();
    }
    built.emplace(key, content);
    return content;
}

// ---------------------------------------------------------------- tools of a request

/// \brief Whether a regex is one character class: ".", "\S", "\d", "[a-z]", ...
bool is_one_character(const std::string& pattern) {
    if (pattern == ".") {
        return true;
    }
    if (pattern.size() == 2 && pattern[0] == '\\') {
        return pattern[1] != 'b' && pattern[1] != 'B';  // word boundaries match no character
    }
    if (pattern.size() < 3 || pattern.front() != '[' || pattern.back() != ']') {
        return false;
    }
    for (size_t i = 1; i + 1 < pattern.size(); ++i) {
        if (pattern[i] == '\\') {
            ++i;
        } else if (pattern[i] == ']') {
            return false;
        }
    }
    return true;
}

/// \brief Any character a JSON string holds unescaped, as a regex class
/// \note Not ".": xgrammar keeps '"', '\\' and control characters out of a pattern's
///       matches only when the pattern is printable ASCII. Any other pattern is
///       compiled without that, where "." can close the string and add keys.
const std::string JSON_STRING_CHAR = "[^\"\\\\\\x00-\\x1f]";

/// \brief Where a regex splits into alternatives and holds anchors, outside groups and classes
struct regex_shape {
    std::vector<std::string> alternatives; // split at each top-level "|"
    bool inner_anchor = false;             // a "^" or "$" anywhere but at either end
};

regex_shape scan_regex(const std::string& pattern) {
    regex_shape shape;
    std::string current;
    int depth = 0;
    bool in_class = false;
    for (size_t i = 0; i < pattern.size(); ++i) {
        const char c = pattern[i];
        if (c == '\\' && i + 1 < pattern.size()) {
            current += pattern.substr(i++, 2);
            continue;
        }
        if (in_class) {
            in_class = c != ']';
        } else if (c == '[') {
            in_class = true;
            if (i + 1 < pattern.size() && pattern[i + 1] == '^') current += pattern[i++];
            if (i + 1 < pattern.size() && pattern[i + 1] == ']') current += pattern[i++];
        } else if (c == '(') {
            ++depth;
        } else if (c == ')') {
            --depth;
        } else if (c == '|' && depth == 0) {
            shape.alternatives.push_back(std::move(current));
            current.clear();
            continue;
        } else if (c == '^' || c == '$') {
            const bool at_edge = depth == 0 && (c == '^' ? current.empty()
                                                         : i + 1 == pattern.size() || pattern[i + 1] == '|');
            shape.inner_anchor = shape.inner_anchor || !at_edge;
        }
        current += c;
    }
    shape.alternatives.push_back(std::move(current));
    return shape;
}

/// \brief One alternative of a pattern, padded where it leaves a side unanchored
std::string padded_alternative(const std::string& alternative) {
    const std::string pad = JSON_STRING_CHAR + "*";
    const bool starts = !alternative.empty() && alternative.front() == '^';
    size_t escapes = 0;
    while (escapes + 1 < alternative.size() && alternative[alternative.size() - 2 - escapes] == '\\') ++escapes;
    const bool ends = alternative.size() > (starts ? 1u : 0u) && alternative.back() == '$' && escapes % 2 == 0;
    const size_t begin = starts ? 1 : 0;
    const std::string core = alternative.substr(begin, alternative.size() - begin - (ends ? 1 : 0));
    if (core.empty()) {
        return starts && ends ? std::string() : pad;
    }
    return (starts ? "(" : pad + "(") + core + (ends ? ")" : ")" + pad);
}

/// \brief A JSON Schema string pattern as xgrammar's whole-string regex has to spell it
/// \note JSON Schema finds a pattern anywhere in the string, while xgrammar matches
///       the whole string against it, so "\S" (not blank) would allow exactly one
///       character. A side an alternative leaves unanchored gets padded. Left as sent:
///       a pattern anchored at both ends, one that opens with an inline flag such as
///       "(?i)", which xgrammar only takes there, and one with an anchor inside a group,
///       which xgrammar ignores rather than reads.
std::string whole_string_pattern(const std::string& pattern) {
    const bool inline_flag = pattern.size() > 2 && pattern.compare(0, 2, "(?") == 0 &&
                             std::isalpha(static_cast<unsigned char>(pattern[2]));
    const regex_shape shape = scan_regex(pattern);
    if (inline_flag || shape.inner_anchor) {
        return pattern;
    }
    if (shape.alternatives.size() == 1) {
        const std::string padded = padded_alternative(pattern);
        return padded.find(JSON_STRING_CHAR) == std::string::npos ? pattern : padded;
    }
    std::string out;
    for (const auto& alternative : shape.alternatives) {
        out += (out.empty() ? "(" : "|") + padded_alternative(alternative);
    }
    return out + ")";
}

bool is_string_schema(const json& schema) {
    const json type = schema.value("type", json());
    if (type.is_array()) {
        return std::find(type.begin(), type.end(), json("string")) != type.end();
    }
    return type == "string";
}

/// \brief Longest regex repetition xgrammar builds in place; a longer one becomes a rule
///        of its own, which it cannot build inside a nested object (the check on
///        per_rule_fsm_hashes fails and the request gets no grammar at all).
constexpr int64_t XGRAMMAR_REGEX_REPEAT_LIMIT = 128;

/// \brief The pattern xgrammar is given for a string schema, which may take over its length bounds
/// \note xgrammar drops minLength/maxLength once a pattern is there. A lone character
///       class that only has to occur somewhere leads instead of being padded, so the
///       bounds become the repetition after it and leave the schema. That is narrower
///       than JSON Schema, which lets the class occur anywhere ("\d" with bounds will not
///       write "Room 5"), but what it writes is always valid. Bounds too long for that
///       repetition keep the schema's minLength/maxLength and drop the pattern
///       (std::nullopt): a string past its maxLength is the likelier mistake.
std::optional<std::string> string_pattern(json& schema) {
    const std::string pattern = schema["pattern"].get<std::string>();
    const json min_length = schema.value("minLength", json(0));
    json max_length = schema.value("maxLength", json());
    const bool bounded = schema.contains("minLength") || schema.contains("maxLength");
    if (!bounded || !is_one_character(pattern) || !min_length.is_number_integer() ||
        !(max_length.is_null() || max_length.is_number_integer())) {
        return whole_string_pattern(pattern);
    }
    if (max_length.is_number_integer() && max_length.get<int64_t>() > std::numeric_limits<int32_t>::max()) {
        max_length = json();
    }
    const int64_t low = std::max<int64_t>(min_length.get<int64_t>(), 1) - 1;
    if (max_length.is_number_integer() && max_length.get<int64_t>() - 1 < low) {
        // unsatisfiable; xgrammar ignores the bounds beside a pattern, so this holds to the pattern alone
        return whole_string_pattern(pattern);
    }
    const bool long_repeat = max_length.is_null() ? low > XGRAMMAR_REGEX_REPEAT_LIMIT
                                                  : max_length.get<int64_t>() - 1 > XGRAMMAR_REGEX_REPEAT_LIMIT;
    if (long_repeat) {
        return std::nullopt;
    }
    const std::string high = max_length.is_null() ? std::string() : std::to_string(max_length.get<int64_t>() - 1);
    schema.erase("minLength");
    schema.erase("maxLength");
    return pattern + JSON_STRING_CHAR + "{" + std::to_string(low) + "," + high + "}";
}

/// \brief Spell out what a schema leaves implicit where xgrammar needs it
/// \note An array without "items" means any items; the qwen_xml style rejects
///       every value of such an array unless that is written as "items": {}.
///       Literal values (const, enum, default, examples) are not schemas, and a
///       pattern under "not" would change meaning if rewritten, so neither is entered.
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
    if (is_string_schema(schema) && schema.contains("pattern") && schema["pattern"].is_string()) {
        const std::optional<std::string> pattern = string_pattern(schema);
        if (pattern) {
            schema["pattern"] = *pattern;
        } else {
            schema.erase("pattern");
        }
    }
    static const std::set<std::string> NOT_ENTERED = {"const", "enum", "default", "examples", "not"};
    // keyed by name, so a property called "default" is still a schema
    static const std::set<std::string> SCHEMA_MAPS = {"properties", "patternProperties", "$defs", "definitions",
                                                      "dependentSchemas"};
    for (auto& [key, value] : schema.items()) {
        if (NOT_ENTERED.count(key)) {
            continue;
        }
        if (SCHEMA_MAPS.count(key) && value.is_object()) {
            for (auto& [name, sub] : value.items()) sub = normalize_schema(std::move(sub));
        } else if (value.is_object() || value.is_array()) {
            value = normalize_schema(std::move(value));
        }
    }
    return schema;
}

struct tool_def {
    std::string name;
    json parameters; // a JSON schema, or true for any arguments
    /// the schema as the request sent it, for the argument grammars written here:
    /// normalize_schema folds length bounds into a pattern only xgrammar reads
    json declared;
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
        tool.declared = strict && fn.contains("parameters") && fn["parameters"].is_object()
            ? fn["parameters"] : json(true);
        tool.parameters = tool.declared.is_object() ? normalize_schema(tool.declared) : tool.declared;
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

/// \brief How a dialect writes the values of declared properties
struct value_syntax {
    /// spells a string enum value the way the dialect writes strings
    std::function<std::string(const std::string&)> quote_string;
    /// EBNF between two array items
    std::string array_sep;
    /// EBNF for a string of low..high characters (high -1: no upper bound)
    std::function<std::string(int64_t, int64_t)> bounded_string;
};

/// \brief A string property's minLength/maxLength as {low, high}, high -1 for none
/// \return std::nullopt when the property is not a plain string with usable bounds:
///         none, unsatisfiable ones (left for the model, as before), or ones past
///         XGRAMMAR_REGEX_REPEAT_LIMIT, which would cost a rule per character
std::optional<std::pair<int64_t, int64_t>> string_length_bounds(const json& property) {
    if (property.value("type", json()) != "string" || property.contains("enum")) {
        return std::nullopt;
    }
    const json min_length = property.value("minLength", json(0));
    const json max_length = property.value("maxLength", json(-1));
    if (!min_length.is_number_integer() || !max_length.is_number_integer()) {
        return std::nullopt;
    }
    const int64_t low = std::max<int64_t>(min_length.get<int64_t>(), 0);
    const int64_t high = max_length.get<int64_t>() < 0 ? -1 : max_length.get<int64_t>();
    if ((low == 0 && high < 0) || (high >= 0 && high < low) ||
        (high < 0 ? low : high) > XGRAMMAR_REGEX_REPEAT_LIMIT) {
        return std::nullopt;
    }
    return std::make_pair(low, high);
}

/// \brief EBNF for one property's value, from its schema type
/// \note An array follows its "items" schema, so enums, types and string lengths
///       inside lists are kept too; objects nested in a value are only held to the
///       dialect's syntax.
std::string property_value(const json& property, const value_syntax& syntax) {
    std::string value = value_rule(property, "value");
    if (value == "array" && property.contains("items") && property["items"].is_object() &&
        !property["items"].empty()) {
        const std::string item = property_value(property["items"], syntax);
        return "(\"[\" (" + item + " (" + syntax.array_sep + " " + item + ")*)? \"]\")";
    }
    if (value == "string") {
        if (const auto bounds = string_length_bounds(property)) {
            return syntax.bounded_string(bounds->first, bounds->second);
        }
    }
    if (!value.empty()) {
        return value;
    }
    value = "(";
    bool first = true;
    for (const auto& option : property["enum"]) {
        value += (first ? "" : " | ") + syntax.quote_string(option.get<std::string>());
        first = false;
    }
    return value + ")";
}

/// \brief Rules for the argument list of one call
/// \param prefix keeps the rule names of different tools apart in one grammar
/// \param kv_sep EBNF between a key and its value
/// \param item_sep EBNF between two arguments
/// \param syntax how the dialect writes declared values
/// \param sorted keys come in alphabetical order (Gemma4's template uses dictsort),
///        otherwise in declaration order
/// \return rules whose list rule is <prefix>args; a schema without declared
///         properties gets any key/value list, built from the caller's "pair" rule
/// \note The declared properties come in one fixed order, each required one present
///       and each optional one free to be left out. Rule first<i> is the list
///       starting at property i, rest<i> the remainder after something was written.
std::string argument_list_rules(const json& schema, const std::string& prefix, const std::string& kv_sep,
                                const std::string& item_sep, const value_syntax& syntax, bool sorted) {
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
          << property_value(schema["properties"][names[i]], syntax) << "\n";
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

/// \brief Rules for raw string text of a bounded length, named <prefix><low>_<high>
/// \note Gemma4 and GLM write strings raw up to a mark that starts with "<" (<|"|>,
///       </arg_value>), so the text may not hold "<" followed by the mark's next
///       character. A step takes one character other than "<", or "<" with the
///       character after it (one the dialect allows there), and counts what it took:
///       the length comes out exact, which a {low,high} repetition over the
///       two-character step would not give.
class bounded_raw_strings {
public:
    /// \param prefix starts the rule names
    /// \param after_lt EBNF class of the character after a "<" that a step takes
    /// \param unbounded EBNF for text of any length
    bounded_raw_strings(std::string prefix, std::string after_lt, std::string unbounded)
        : prefix_(std::move(prefix)), after_lt_(std::move(after_lt)), unbounded_(std::move(unbounded)) {}

    /// \brief The rule for text of low..high characters (high -1: no upper bound)
    std::string rule(int64_t low, int64_t high) {
        wanted_.insert({low, high});
        return name(low, high);
    }

    /// \brief The rules every rule() so far refers to
    std::string rules() const {
        std::ostringstream g;
        std::set<std::pair<int64_t, int64_t>> done;
        std::vector<std::pair<int64_t, int64_t>> todo(wanted_.begin(), wanted_.end());
        while (!todo.empty()) {
            const auto [low, high] = todo.back();
            todo.pop_back();
            if (!done.insert({low, high}).second) {
                continue;
            }
            if (low == 0 && high < 0) {
                g << name(low, high) << " ::= " << unbounded_ << "\n";
                continue;
            }
            std::string steps;
            for (int64_t width = 1; width <= 2; width++) {
                if (high >= 0 && high < width) {
                    continue;
                }
                const std::pair<int64_t, int64_t> next = {std::max<int64_t>(low - width, 0),
                                                          high < 0 ? -1 : high - width};
                steps += std::string(steps.empty() ? "" : " | ") + (width == 1 ? "[^<]" : "\"<\" " + after_lt_) +
                         " " + name(next.first, next.second);
                todo.push_back(next);
            }
            if (steps.empty()) {
                g << name(low, high) << " ::= \"\"\n";
            } else if (low == 0) {
                g << name(low, high) << " ::= (" << steps << ")?\n";
            } else {
                g << name(low, high) << " ::= " << steps << "\n";
            }
        }
        return g.str();
    }

private:
    std::string name(int64_t low, int64_t high) const {
        return prefix_ + std::to_string(low) + "_" + (high < 0 ? std::string("x") : std::to_string(high));
    }

    std::string prefix_, after_lt_, unbounded_;
    std::set<std::pair<int64_t, int64_t>> wanted_;
};

/// \brief Gemma4 call arguments: {key:<|"|>text<|"|>,n:5,flag:true,list:[...]}
std::string gemma4_arguments_grammar(const json& schema) {
    // a string's text may hold no "<|"
    bounded_raw_strings strings("gstr_", "[^<|]", "([^<] | \"<\" [^|])*");
    const value_syntax syntax{[](const std::string& s) { return ebnf_literal("<|\"|>" + s + "<|\"|>"); }, "\",\"",
                              [&strings](int64_t low, int64_t high) {
                                  return "(\"<|\\\"|>\" " + strings.rule(low, high) + " \"<|\\\"|>\")";
                              }};
    std::ostringstream g;
    g << "root ::= \"{\" args \"}\"\n"
      << argument_list_rules(schema, "", "\":\"", "\",\"", syntax, true)
      << strings.rules()
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
    // an escape such as \n is one character of the value
    auto bounded = [](int64_t low, int64_t high) {
        const std::string count = "{" + std::to_string(low) + "," + (high < 0 ? "" : std::to_string(high)) + "}";
        return "(\"\\\"\" ([^\"\\\\] | \"\\\\\" [^\\n])" + count + " \"\\\"\" | \"'\" ([^'\\\\] | \"\\\\\" [^\\n])" + count +
               " \"'\")";
    };
    const value_syntax syntax{[](const std::string& s) { return ebnf_literal("\"" + s + "\""); }, "\",\" \" \"?",
                              bounded};
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
          << argument_list_rules(tools[i].declared, prefix, "\"=\"", "\",\" \" \"?", syntax, false);
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

// ---------------------------------------------------------------- glm xml arguments

/// \brief Whether GLM writes a value of this schema raw: it may be a string, or anything
/// \param root the parameters schema, which a "#/$defs/..." reference points into
bool glm_raw_value(const json& property, const json& root, int depth = 0) {
    if (!property.is_object() || depth > 8) {
        return true;
    }
    if (property.contains("$ref")) {
        const std::string ref = property["$ref"].is_string() ? property["$ref"].get<std::string>() : "";
        for (const std::string defs : {"$defs", "definitions"}) {
            const std::string prefix = "#/" + defs + "/";
            if (ref.compare(0, prefix.size(), prefix) == 0 && root.contains(defs) && root[defs].is_object() &&
                root[defs].contains(ref.substr(prefix.size()))) {
                return glm_raw_value(root[defs][ref.substr(prefix.size())], root, depth + 1);
            }
        }
        return false;
    }
    if (property.contains("const")) {
        return property["const"].is_string();
    }
    if (property.contains("enum") && property["enum"].is_array()) {
        return std::any_of(property["enum"].begin(), property["enum"].end(), [](const json& v) { return v.is_string(); });
    }
    if (property.contains("type")) {
        return is_string_schema(property);
    }
    for (const char* key : {"anyOf", "oneOf"}) {
        if (property.contains(key) && property[key].is_array()) {
            return std::any_of(property[key].begin(), property[key].end(),
                               [&](const json& option) { return glm_raw_value(option, root, depth + 1); });
        }
    }
    return !property.contains("allOf") && !property.contains("properties") && !property.contains("items");
}

/// \brief A raw GLM string of low..high characters (high -1: no upper bound)
/// \note The parser ends a value at its first "</arg_value>", so a "<" in the text is
///       followed by neither "/" nor "<" (whose step would let a "</" through).
json glm_bounded_string(int64_t low, int64_t high) {
    bounded_raw_strings strings("s", "[^</]", "([^<] | \"<\" [^</])*");
    const std::string root = "root ::= " + strings.rule(low, high) + "\n";
    return grammar_content(root + strings.rules());
}

/// \brief The arguments of a GLM call: <arg_key>k</arg_key><arg_value>v</arg_value> for each
/// \note Laguna's template writes a string raw and anything else as JSON, with nothing
///       between the parts. The parser ends a value at its first "</arg_value>" and the
///       call at its first "</tool_call>", so a raw value holds neither. As in the
///       other dialects, declared properties come in their order, each optional one
///       free to be left out, and a string keeps its minLength/maxLength; a schema
///       without declared properties takes any keys.
/// \param schema the parameters schema, normalized for xgrammar
/// \param sent the schema as the request sent it, where a string still has its bounds
json glm_arguments(const json& schema, const json& sent) {
    static const std::vector<std::string> RAW_EXCLUDES = {"</tool_call>"};
    const bool declared = schema.is_object() && schema.contains("properties") && schema["properties"].is_object() &&
                          !schema["properties"].empty() && schema.value("additionalProperties", false) == false;
    const json none;
    const json& sent_properties =
        sent.is_object() && sent.contains("properties") && sent["properties"].is_object() ? sent["properties"] : none;
    if (!declared) {
        return {{"type", "star"},
                {"content", sequence({tag("<arg_key>", any_text(RAW_EXCLUDES), "</arg_key>"),
                                      tag("<arg_value>", any_text(RAW_EXCLUDES), "</arg_value>")})}};
    }
    std::set<std::string> required;
    for (const auto& name : schema.value("required", json::array())) {
        if (name.is_string()) required.insert(name.get<std::string>());
    }
    std::vector<json> pairs;
    for (const auto& [name, property] : schema["properties"].items()) {
        json value;
        const bool string_enum = property.is_object() && property.contains("enum") && property["enum"].is_array() &&
                                 !property["enum"].empty() &&
                                 std::all_of(property["enum"].begin(), property["enum"].end(),
                                             [](const json& v) { return v.is_string(); });
        if (string_enum) {
            std::vector<json> options;
            for (const auto& option : property["enum"]) options.push_back(const_string(option.get<std::string>()));
            value = {{"type", "or"}, {"elements", std::move(options)}};
        } else if (glm_raw_value(property, schema)) {
            std::optional<std::pair<int64_t, int64_t>> bounds;
            if (sent_properties.contains(name) && sent_properties[name].is_object()) {
                bounds = string_length_bounds(sent_properties[name]);
            }
            value = bounds ? glm_bounded_string(bounds->first, bounds->second) : any_text(RAW_EXCLUDES);
        } else {
            // a $ref in the property points into the schema it came from
            json sub = property;
            for (const char* defs : {"$defs", "definitions"}) {
                if (schema.contains(defs) && !sub.contains(defs)) sub[defs] = schema[defs];
            }
            value = schema_content(sub);
        }
        json pair = tag("<arg_key>" + name + "</arg_key><arg_value>", std::move(value), "</arg_value>");
        pairs.push_back(required.count(name) ? std::move(pair) : optional_format(std::move(pair)));
    }
    return sequence(std::move(pairs));
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
                // xgrammar's qwen_3_5 style, except for the end: the templates write the newline
                // after each "</parameter>", not before "</function>", so a call without
                // arguments is "<function=f>\n</function>"; the parameter list takes trailing
                // whitespace, which covers the newline after the last one
                syntax.trigger = "<tool_call>\n<function=";
                syntax.tags.push_back(tag("<tool_call>\n<function=" + tool.name + ">\n",
                                          qwen_xml_arguments(tool.parameters),
                                          "</function>\n</tool_call>"));
                break;
            case chat_format_kind::GEMMA4:
                syntax.trigger = "<|tool_call>";
                syntax.tags.push_back(tag("<|tool_call>call:" + tool.name,
                                          grammar_content(gemma4_arguments_grammar(tool.declared)), "<tool_call|>"));
                break;
            case chat_format_kind::GLM_XML:
                syntax.trigger = "<tool_call>";
                syntax.tags.push_back(
                    tag("<tool_call>" + tool.name, glm_arguments(tool.parameters, tool.declared), "</tool_call>"));
                break;
            case chat_format_kind::LFM2_PYTHONIC:
            case chat_format_kind::CONTENT_ONLY:
            case chat_format_kind::HARMONY:
            case chat_format_kind::INKLING:
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
        return normalize_schema(request.json_schema);
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

json inkling_format(const chat_constraint_request& request, const std::vector<tool_def>& tools) {
    // typed messages, each opened by <|message_model|>: the model opens the first one
    // itself, the prompt ends with the user's <|end_message|>
    const std::string MESSAGE = "<|message_model|>";
    const std::string END = "<|end_message|>";

    std::vector<json> call_tags;
    const bool want_tools = request.tool_choice == chat_tool_choice::REQUIRED ||
                            request.tool_choice == chat_tool_choice::FUNCTION ||
                            (request.tool_choice == chat_tool_choice::AUTO && !has_response_format(request));
    if (want_tools) {
        for (const auto& tool : tools) {
            // as the template writes a call: the header names it, the JSON again
            call_tags.push_back(tag(tool.name + "<|content_invoke_tool_json|>{\"name\":" + json(tool.name).dump() + ",\"args\":",
                                    schema_content(tool.parameters), "}" + END));
        }
    }
    // every turn opens with reasoning, at any effort
    std::vector<json> message_tags;
    message_tags.push_back(tag("<|content_thinking|>", any_text(), END));
    if (request.tool_choice == chat_tool_choice::AUTO || request.tool_choice == chat_tool_choice::NONE) {
        json text = has_response_format(request) ? schema_content(response_schema(request)) : any_text();
        message_tags.push_back(tag("<|content_text|>", text, END));
    }

    json format;
    if (call_tags.empty()) {
        format = {{"type", "tags_with_separator"}, {"tags", message_tags}, {"separator", MESSAGE}, {"at_least_one", true}};
        return sequence({const_string(MESSAGE), format});
    }
    const bool must_call = request.tool_choice != chat_tool_choice::AUTO;
    if (request.parallel_tool_calls && !must_call) {
        std::vector<json> all = call_tags;
        all.insert(all.end(), message_tags.begin(), message_tags.end());
        format = {{"type", "tags_with_separator"}, {"tags", all}, {"separator", MESSAGE}, {"at_least_one", true}};
        return sequence({const_string(MESSAGE), format});
    }
    // reasoning/answer messages, then the call(s), which end the turn. A forced call
    // gets one thinking message: the model, kept from answering in text, would
    // otherwise open thinking message after thinking message until the length limit
    const bool one_call = !request.parallel_tool_calls || request.tool_choice == chat_tool_choice::FUNCTION;
    json calls = {{"type", "tags_with_separator"}, {"tags", call_tags}, {"separator", MESSAGE},
                  {"at_least_one", true}, {"stop_after_first", one_call}};
    json messages = {{"type", "tags_with_separator"}, {"tags", message_tags}, {"separator", MESSAGE},
                     {"at_least_one", true}, {"stop_after_first", must_call}};
    json messages_then_calls = sequence({messages, const_string(MESSAGE), calls});
    std::vector<json> options = {calls, messages_then_calls};
    if (!must_call) {
        options.insert(options.begin(), messages);
    }
    return sequence({const_string(MESSAGE), json{{"type", "or"}, {"elements", options}}});
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
        if (text.empty() || matcher_.AcceptString(text)) {
            return;
        }
        // the template opened a reasoning block although the request turned thinking
        // off (a model without a thinking switch): the answer still has to follow it
        if (!use_reasoning_grammar()) {
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
        // the model is made to reason although the request turned thinking off
        if (!use_reasoning_grammar() || !matcher_.AcceptToken(token)) {
            give_up("the grammar rejected forced token " + std::to_string(token));
            return;
        }
        history_.push_back(token);
    }

    void rollback(int n) override {
        if (!failed_) {
            matcher_.Rollback(n);
            history_.resize(history_.size() > static_cast<size_t>(n) ? history_.size() - n : 0);
        }
    }
    bool failed() const override { return failed_; }

private:
    static constexpr int MAX_ROLLBACK = 8;

    /// \brief Carry on with the grammar that has the reasoning block, from the same point
    /// \return false when there is none, or it does not accept what came so far either
    bool use_reasoning_grammar() {
        if (!with_reasoning_) {
            return false;
        }
        xgrammar::GrammarMatcher reasoning(*with_reasoning_, std::nullopt, false, MAX_ROLLBACK);
        with_reasoning_.reset();
        bool ok = prefill_.empty() || reasoning.AcceptString(prefill_);
        for (size_t i = 0; ok && i < history_.size(); i++) {
            ok = reasoning.AcceptToken(history_[i]);
        }
        if (ok) {
            matcher_ = std::move(reasoning);
        }
        return ok;
    }

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
    } else if (spec.kind == chat_format_kind::INKLING) {
        format = inkling_format(request, tools);
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
                // gemma4 opens reasoning with two tokens; the bare <|channel> token must not
                // be sampled either, or it stands in for <|tool_call> (FLM#886)
                if (spec.kind == chat_format_kind::GEMMA4) excludes.push_back("<|channel>");
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
