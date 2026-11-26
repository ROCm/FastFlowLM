/// \file test_chat_tools.cpp
/// \brief Tool calling without a model: the output parser on what models write, the decoding grammar
///        against the calls the chat templates render, and the constraint a request gets. No NPU needed.
#include "chat/chat_constraint.hpp"
#include "chat/chat_format.hpp"
#include "chat/chat_gemma4_tools.hpp"
#include "chat/chat_grammar.hpp"
#include "chat/chat_parser.hpp"
#include "chat/positional_tool_results.hpp"
#include "minja/chat-template.hpp"
#include "minja/minja.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {
int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                             \
        }                                                                           \
    } while (0)

#define CHECK_EQ(actual, expected)                                                  \
    do {                                                                            \
        const std::string a_ = (actual), e_ = (expected);                           \
        if (a_ != e_) {                                                             \
            std::printf("  FAIL %s:%d: %s\n    got:      [%s]\n    expected: [%s]\n", \
                        __FILE__, __LINE__, #actual, a_.c_str(), e_.c_str());       \
            failures++;                                                             \
        }                                                                           \
    } while (0)

using json = nlohmann::ordered_json;

const chat_format_spec QWEN_XML = chat_format_make(chat_format_kind::QWEN_XML, true);
const chat_format_spec HERMES = chat_format_make(chat_format_kind::HERMES_JSON, true);

// How LFM3 and Qwen3.5+ render an assistant's tool calls, verbatim from LFM3's chat_template.jinja
const std::string RENDER_CALLS = R"JINJA(
{%- macro render_value(v) -%}
    {%- if v is string -%}
        {{- v -}}
    {%- elif v is boolean -%}
        {{- "true" if v else "false" -}}
    {%- elif v is none -%}
        {{- "null" -}}
    {%- elif v is mapping or (v is sequence and v is not string) -%}
        {{- v | tojson -}}
    {%- else -%}
        {{- v -}}
    {%- endif -%}
{%- endmacro -%}
{%- macro render_tool_calls(tool_calls) -%}
    {%- for tool_call in tool_calls -%}
        {%- if not loop.first -%}{{- "\n\n" -}}{%- endif -%}
        {%- set func = tool_call["function"] if "function" in tool_call else tool_call -%}
        {%- set func_args = func.get("arguments") -%}
        {{- "<tool_call>\n<function=" + func["name"] + ">\n" -}}
        {%- if func_args is mapping -%}
            {%- for arg_name, arg_value in func_args.items() -%}
                {{- "<parameter=" + arg_name + ">\n" + render_value(arg_value) + "\n</parameter>\n" -}}
            {%- endfor -%}
        {%- endif -%}
        {{- "</function>\n</tool_call>" -}}
    {%- endfor -%}
{%- endmacro -%}
{{- render_tool_calls(calls) -}}
)JINJA";

std::string render_calls(const json& calls) {
    auto tmpl = minja::Parser::parse(RENDER_CALLS, {true, true, false});
    return tmpl->render(minja::Context::make(minja::Value(json{{"calls", calls}})));
}

const chat_format_spec GLM_XML = chat_format_make(chat_format_kind::GLM_XML, true);

// How Laguna-2.1 renders an assistant's tool calls, verbatim from its chat_template.jinja
const std::string GLM_RENDER_CALLS = R"JINJA(
{%- for tool_call in calls -%}
  {%- set function_data = tool_call.function -%}
  {{- '<tool_call>' + function_data.name -}}
  {%- set _args = function_data.arguments -%}
  {%- for k, v in _args.items() -%}
    {{- "<arg_key>" ~ k ~ "</arg_key>" -}}
    {{- "<arg_value>" -}}{{- v | tojson(ensure_ascii=False) if v is not string else v -}}{{- "</arg_value>" -}}
  {%- endfor -%}
  {{- "</tool_call>" -}}
{%- endfor -%}
)JINJA";

std::string render_glm_calls(const json& calls) {
    auto tmpl = minja::Parser::parse(GLM_RENDER_CALLS, {true, true, false});
    return tmpl->render(minja::Context::make(minja::Value(json{{"calls", calls}})));
}

json call(const std::string& name, const json& arguments) {
    return {{"type", "function"}, {"function", {{"name", name}, {"arguments", arguments}}}};
}

json tool(const std::string& name, const json& parameters) {
    json fn = {{"name", name}, {"description", name}};
    if (!parameters.is_null()) fn["parameters"] = parameters;
    return {{"type", "function"}, {"function", fn}};
}

json object_schema(const json& properties, const json& required = json::array()) {
    return {{"type", "object"}, {"properties", properties}, {"required", required}};
}

const json TOOLS = json::array({
    tool("ping", object_schema(json::object())),
    tool("now", nullptr),
    tool("status", {{"type", "object"}}),
    tool("get_weather", object_schema({{"city", {{"type", "string"}}},
                                       {"unit", {{"type", "string"}, {"enum", {"celsius", "fahrenheit"}}}}},
                                      {"city"})),
    tool("write_file", object_schema({{"path", {{"type", "string"}}}, {"content", {{"type", "string"}}}},
                                     {"path", "content"})),
    tool("set_flag", object_schema({{"on", {{"type", "boolean"}}}, {"count", {{"type", "integer"}}}}, {"on"})),
    tool("lookup", {{"type", "object"},
                    {"properties",
                     {{"zip", {{"anyOf", {{{"type", "string"}}, {{"type", "null"}}}}}},
                      {"limit", {{"anyOf", {{{"type", "integer"}}, {{"type", "null"}}}}}},
                      {"code", {{"$ref", "#/$defs/Code"}}},
                      {"level", {{"enum", {"1", "2"}}}},
                      {"note", {{"type", {"string", "null"}}}},
                      {"anything", json::object()},
                      {"filter", {{"type", "object"}, {"properties", {{"a", {{"type", "integer"}}}}}}}}},
                    {"$defs", {{"Code", {{"type", "string"}}}}}}),
});

chat_msg parse(const chat_format_spec& spec, const std::string& text, bool is_partial = false,
               bool parse_tool_calls = true, bool reasoning_may_close = false) {
    chat_parse_options options;
    options.tools = &TOOLS;
    options.parse_tool_calls = parse_tool_calls;
    options.reasoning_may_close = reasoning_may_close;
    return chat_parse(spec, text, is_partial, options);
}

bool is_prefix(const std::string& prefix, const std::string& s) {
    return s.compare(0, prefix.size(), prefix) == 0 && prefix.size() <= s.size();
}

/// \brief Streaming safety: every partial parse only extends the previous one and leads to the final one
void check_prefix_stable(const chat_format_spec& spec, const std::string& text, const char* label) {
    const chat_msg final_msg = parse(spec, text);
    chat_msg prev;
    for (size_t i = 0; i <= text.size(); i++) {
        const chat_msg cur = i == text.size() ? final_msg : parse(spec, text.substr(0, i), true);
        bool ok = is_prefix(prev.reasoning, cur.reasoning) && is_prefix(prev.content, cur.content) &&
                  prev.tool_calls.size() <= cur.tool_calls.size();
        for (size_t c = 0; ok && c < prev.tool_calls.size(); c++) {
            ok = prev.tool_calls[c].name == cur.tool_calls[c].name &&
                 is_prefix(prev.tool_calls[c].arguments, cur.tool_calls[c].arguments);
        }
        if (!ok) {
            std::printf("  FAIL %s: the parse of %zu bytes does not extend the one before\n", label, i);
            failures++;
            return;
        }
        prev = cur;
    }
}

// ---------------------------------------------------------------- grammar

/// \brief A grammar engine over a vocabulary of single characters, enough to match text
class grammar_fixture {
public:
    explicit grammar_fixture(const chat_format_spec& spec = QWEN_XML) : spec_(spec) {
        json vocab = json::object();
        for (int c = 1; c < 127; c++) vocab[std::string(1, static_cast<char>(c))] = c;
        // a plain decoder: every token is the text it stands for
        json tokenizer = {{"model", {{"type", "BPE"}, {"vocab", vocab}}},
                          {"decoder", {{"type", "Fuse"}}},
                          {"added_tokens", json::array()}};
        path_ = (std::filesystem::temp_directory_path() / "flm_test_chat_tools_tokenizer.json").string();
        std::ofstream(path_) << tokenizer.dump();
        chat_vocab v;
        v.size = 128;
        v.tokenizer_json = path_;
        engine_ = std::make_unique<chat_grammar_engine>(v, spec_);
    }
    ~grammar_fixture() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    /// \brief Whether the request's grammar takes the text, without reporting
    bool takes(const chat_constraint_request& request, const std::string& text) {
        std::string error;
        return engine_->accepts(spec_, request, "", text, error);
    }

    bool accepts(const chat_constraint_request& request, const std::string& text, const char* label) {
        std::string error;
        if (!engine_->accepts(spec_, request, "", text, error)) {
            std::printf("  FAIL %s: the grammar rejects the template's call: %s\n", label, error.c_str());
            failures++;
            return false;
        }
        return true;
    }

private:
    chat_format_spec spec_;
    std::string path_;
    std::unique_ptr<chat_grammar_engine> engine_;
};

chat_constraint_request tools_request(chat_tool_choice choice = chat_tool_choice::AUTO, const std::string& fn = "") {
    chat_constraint_request request;
    request.tools = &TOOLS;
    request.tool_choice = choice;
    request.function_name = fn;
    request.allow_reasoning = false;
    return request;
}

void test_qwen_xml_grammar_takes_template_calls() {
    std::printf("qwen_xml grammar takes every call the template renders\n");
    grammar_fixture grammar;
    const std::vector<std::pair<const char*, json>> calls = {
        {"no arguments", json::array({call("ping", json::object())})},
        {"no parameters schema", json::array({call("now", json::object())})},
        {"bare object schema", json::array({call("status", json::object())})},
        {"one argument", json::array({call("get_weather", {{"city", "Paris"}})})},
        {"optional argument", json::array({call("get_weather", {{"city", "Paris"}, {"unit", "celsius"}})})},
        {"value holding </parameter>",
         json::array({call("write_file", {{"path", "a.xml"},
                                          {"content", "<config>\n  <parameter name=\"x\">1</parameter>\n</config>"}})})},
        {"value holding </function>", json::array({call("write_file", {{"path", "a"}, {"content", "x</function>y"}})})},
        {"boolean and integer", json::array({call("set_flag", {{"on", true}, {"count", 3}})})},
        {"parallel, one without arguments",
         json::array({call("ping", json::object()), call("get_weather", {{"city", "Tokyo"}})})},
    };
    for (const auto& [label, c] : calls) {
        grammar.accepts(tools_request(), render_calls(c), label);
        grammar.accepts(tools_request(chat_tool_choice::REQUIRED), render_calls(c), label);
    }
    grammar.accepts(tools_request(chat_tool_choice::FUNCTION, "ping"), render_calls(json::array({call("ping", json::object())})),
                    "forced call without arguments");
    grammar.accepts(tools_request(), "Let me check.\n" + render_calls(json::array({call("ping", json::object())})),
                    "text before the call");

    // a string value that may hold "</parameter>" still leaves the rest of the call constrained
    std::printf("qwen_xml grammar still holds the call after a string value\n");
    const std::string after_value =
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n";
    CHECK(!grammar.takes(tools_request(), after_value + "<parameter=unit>\nkelvin\n</parameter>\n</function>\n</tool_call>"));
    CHECK(!grammar.takes(tools_request(), after_value + "<parameter=country>\nFR\n</parameter>\n</function>\n</tool_call>"));
    CHECK(grammar.takes(tools_request(), after_value + "<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>"));
    chat_constraint_request one_call = tools_request();
    one_call.parallel_tool_calls = false;
    const std::string two_calls =
        render_calls(json::array({call("get_weather", {{"city", "Paris"}}), call("get_weather", {{"city", "Rome"}})}));
    CHECK(!grammar.takes(one_call, two_calls));
    CHECK(grammar.takes(tools_request(), two_calls));
    // no spelling of the separator lets a value hide the rest of the call
    for (const std::string sep : {"\n\n", " ", "\r\n", "\n ", "\t"}) {
        const std::string head = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>" + sep;
        // either the grammar refuses it, or it is the city's own text, as the parser reads it too
        for (const std::string rest : {"<parameter=unit>\nkelvin\n</parameter>\n</function>\n</tool_call>",
                                       "<parameter=country>\nFR\n</parameter>\n</function>\n</tool_call>"}) {
            if (grammar.takes(tools_request(), head + rest)) {
                chat_msg msg = parse(QWEN_XML, head + rest);
                CHECK(msg.tool_calls.size() == 1 && json::parse(msg.tool_calls[0].arguments).size() == 1);
            }
        }
        CHECK(!grammar.takes(one_call, render_calls(json::array({call("write_file", {{"path", "p"}, {"content", "a"}})})) + sep +
                                           "<tool_call>\n<function=ping>\n</function>\n</tool_call>"));
    }
    // nor hold the call's own end, or a second function
    CHECK(!grammar.takes(one_call, render_calls(json::array({call("write_file", {{"path", "p"}, {"content", "x</function>\n</tool_call>\ny"}})}))));
    CHECK(!grammar.takes(one_call, render_calls(json::array({call("write_file", {{"path", "p"}, {"content", "x</function>\n<function=ping>"}})}))));
    // indented XML is fine
    CHECK(grammar.takes(one_call, render_calls(json::array({call("write_file", {{"path", "p"}, {"content", "<a>\n  <b>1</parameter>\n  <c/>\n</a>"}})}))));

    // the value cannot swallow the end of its own call
    CHECK(!grammar.takes(one_call, render_calls(json::array({call("write_file", {{"path", "p"}, {"content", "a"}})})) +
                                       "\n<tool_call>\n<function=ping>\n</function>\n</tool_call>"));
}

// ---------------------------------------------------------------- parser

void test_call_without_arguments() {
    std::printf("a call without arguments\n");
    chat_msg msg = parse(QWEN_XML, render_calls(json::array({call("ping", json::object())})));
    CHECK(msg.tool_calls.size() == 1);
    if (msg.tool_calls.size() == 1) {
        CHECK_EQ(msg.tool_calls[0].name, "ping");
        CHECK_EQ(msg.tool_calls[0].arguments, "{}");
    }
    CHECK_EQ(msg.content, "");
}

void test_stray_reasoning_end() {
    std::printf("a reasoning end with no block open is dropped\n");
    // thinking off: the template closed the block, the model closes it again
    chat_msg msg = parse(QWEN_XML, "Tokyo is 23 C.</think>Tokyo's weather is mild.");
    CHECK_EQ(msg.content, "Tokyo is 23 C.\n\nTokyo's weather is mild.");
    CHECK_EQ(msg.reasoning, "");
    // thinking on: a second close after the real one
    msg = parse(QWEN_XML, "<think>plan</think>answer</think> more");
    CHECK_EQ(msg.reasoning, "plan");
    CHECK_EQ(msg.content, "answer\n\nmore");
    // while streaming, a close still being written is held back
    CHECK_EQ(parse(QWEN_XML, "answer</thi", true).content, "answer");
    check_prefix_stable(QWEN_XML, "Tokyo is 23 C.</think>Tokyo's weather is mild.", "stray close");
    // the answer's own text: quoted, in code, set apart in prose, or a second one
    CHECK_EQ(parse(QWEN_XML, "Use `s.split('</think>')` to drop it.").content, "Use `s.split('</think>')` to drop it.");
    CHECK_EQ(parse(QWEN_XML, "The tag </think> closes it.").content, "The tag </think> closes it.");
    CHECK_EQ(parse(QWEN_XML, "A.</think>B.</think>C").content, "A.\n\nB.\n\nC");
    // what the model writes: its last word, the close, the answer
    CHECK_EQ(parse(QWEN_XML, "I said \"18 C and cloudy.\"</think>You asked about Paris.").content,
             "I said \"18 C and cloudy.\"\n\nYou asked about Paris.");
    CHECK_EQ(parse(QWEN_XML, "with city \"Tokyo\".</think>```json\n{}\n```").content, "with city \"Tokyo\".\n\n```json\n{}\n```");
    // while streaming, a close set apart waits for the next character
    CHECK_EQ(parse(QWEN_XML, "The tag </think>", true).content, "The tag");
    check_prefix_stable(QWEN_XML, "I said \"x.\"</think>You asked. The tag </think> closes.", "quoted close");
    CHECK_EQ(parse(QWEN_XML, "<think>x</think>```\nre.sub('</think>', '', s)\n```").content,
             "```\nre.sub('</think>', '', s)\n```");
    check_prefix_stable(QWEN_XML, "The tag </think> closes it. A.</think>B", "kept close");
}

void test_tool_tag_in_prose() {
    std::printf("the tool tag named in prose is text, not a call\n");
    const std::string prose = "Each call goes in a <tool_call> tag, closed by </tool_call>. More text follows.";
    chat_msg msg = parse(QWEN_XML, prose);
    CHECK_EQ(msg.content, prose);
    CHECK(msg.tool_calls.empty());
    check_prefix_stable(QWEN_XML, prose, "prose tag");

    const std::string mixed = "I will use <tool_call> now.\n" + render_calls(json::array({call("ping", json::object())}));
    msg = parse(QWEN_XML, mixed);
    CHECK_EQ(msg.content, "I will use <tool_call> now.");
    CHECK(msg.tool_calls.size() == 1);
    check_prefix_stable(QWEN_XML, mixed, "prose tag then call");

    // hermes: a call is a JSON object
    msg = parse(HERMES, "Wrap it as <tool_call> like this. Done.");
    CHECK_EQ(msg.content, "Wrap it as <tool_call> like this. Done.");
    msg = parse(HERMES, "<tool_call>\n{\"name\": \"ping\", \"arguments\": {}}\n</tool_call>");
    CHECK(msg.tool_calls.size() == 1);
    CHECK_EQ(parse(HERMES, "See <tool_call>[docs] then do X").content, "See <tool_call>[docs] then do X");
    msg = parse(HERMES, "<tool_call>\n[{\"name\": \"ping\", \"arguments\": {}}]\n</tool_call>");
    CHECK(msg.tool_calls.size() == 1);

    // a tag at the very end of a stream may still open a call
    CHECK_EQ(parse(QWEN_XML, "Sure.\n<tool_call>\n", true).content, "Sure.");
    // and a cut off generation leaves no tag behind
    CHECK_EQ(parse(QWEN_XML, "Sure.\n<tool_call>\n").content, "Sure.");
}

void test_value_holding_closing_tags() {
    std::printf("a string value may hold </parameter> and </function>\n");
    const std::string content = "<config>\n  <parameter name=\"x\">1</parameter>\n</config>\nend</function>.";
    const std::string text = render_calls(json::array({call("write_file", {{"path", "a.xml"}, {"content", content}})}));
    chat_msg msg = parse(QWEN_XML, text);
    CHECK(msg.tool_calls.size() == 1);
    if (msg.tool_calls.size() == 1) {
        CHECK_EQ(msg.tool_calls[0].arguments, json({{"path", "a.xml"}, {"content", content}}).dump());
    }
    CHECK_EQ(msg.content, "");
    check_prefix_stable(QWEN_XML, text, "value with closing tags");

    // the block's end tag inside a value
    const std::string doc_tags = "Calls are wrapped in <tool_call> and end with </tool_call>.\nThat is all.";
    const std::string tags_call = render_calls(json::array({call("write_file", {{"path", "n.md"}, {"content", doc_tags}})}));
    msg = parse(QWEN_XML, tags_call);
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == json({{"path", "n.md"}, {"content", doc_tags}}).dump());
    CHECK_EQ(msg.content, "");
    check_prefix_stable(QWEN_XML, tags_call, "value with the end tag");

    // markers written a byte at a time never retract streamed text
    check_prefix_stable(QWEN_XML, "<tool_call>\n<function=write_file>\n<parameter=content>\nx<</parameter>\n</function>\n</tool_call>",
                        "value ending in <");
    check_prefix_stable(QWEN_XML, "<tool_call>\n<function=write_file>\n<parameter=content>\nhello</<parameter=path>\np\n"
                                  "</parameter>\n</function>\n</tool_call>",
                        "value ending in </");

    // as the grammar has it, a "<parameter=" before the "</parameter>" that closes a value is part of it
    const std::string doc = "Calls look like\n<parameter=city>\nParis";
    const std::string doc_call = render_calls(json::array({call("write_file", {{"path", "d.md"}, {"content", doc}})}));
    msg = parse(QWEN_XML, doc_call);
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == json({{"path", "d.md"}, {"content", doc}}).dump());
    check_prefix_stable(QWEN_XML, doc_call, "value with a parameter tag");

    // a call that left out "</function>" still ends at its end tag
    msg = parse(QWEN_XML, "<tool_call>\n<function=write_file>\n<parameter=content>\nx\n</parameter>\n</tool_call>\nThe answer follows.");
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == "{\"content\":\"x\"}");
    CHECK_EQ(msg.content, "The answer follows.");
    const std::string two = "<tool_call>\n<function=write_file>\n<parameter=content>\nx\n</parameter>\n</tool_call>\n"
                            "<tool_call>\n<function=set_flag>\n<parameter=count>\n5\n</parameter>\n</function>\n</tool_call>";
    msg = parse(QWEN_XML, two);
    CHECK(msg.tool_calls.size() == 2);
    if (msg.tool_calls.size() == 2) {
        CHECK_EQ(msg.tool_calls[0].arguments, "{\"content\":\"x\"}");
        CHECK_EQ(msg.tool_calls[1].arguments, "{\"count\":5}");
    }
    check_prefix_stable(QWEN_XML, two, "call without </function>");
    msg = parse(QWEN_XML, "<tool_call>\n<function=write_file>\n<parameter=content>\nx\n</tool_call>\nDone.");
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == "{\"content\":\"x\"}");
    CHECK_EQ(msg.content, "Done.");
    msg = parse(QWEN_XML, "<tool_call>\n<function=ping>\n</function>\nok\n</tool_call>\nAfter text.");
    CHECK_EQ(msg.content, "After text.");
    // and a closing tag spelled out with more space is still one, when nothing else closes the value
    msg = parse(QWEN_XML, "<tool_call>\n<function=write_file>\n<parameter=content>\nx\n</parameter>\n\n</function>\n</tool_call>");
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == "{\"content\":\"x\"}");
    for (const std::string text : {"<tool_call><function=write_file><parameter=content></parameter><</function>>",
                                   "<tool_call>\n<function=write_file>\n<parameter=content>\nx</parameter>\n</</function>\n"
                                   "</parameter>\n</function>\n</tool_call>"}) {
        check_prefix_stable(QWEN_XML, text, "held function end");
    }

    // a value the model left open ends at the next parameter, or at the function's end
    msg = parse(QWEN_XML, "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n<parameter=unit>\ncelsius\n"
                          "</function>\n</tool_call>");
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == "{\"city\":\"Paris\",\"unit\":\"celsius\"}");
}

void test_argument_types() {
    std::printf("arguments take the types their schema allows\n");
    auto args = [](const json& arguments) {
        chat_msg msg = parse(QWEN_XML, render_calls(json::array({call("lookup", arguments)})));
        return msg.tool_calls.size() == 1 ? msg.tool_calls[0].arguments : std::string("<no call>");
    };
    CHECK_EQ(args({{"zip", "90210"}}), "{\"zip\":\"90210\"}");    // Optional[str]
    CHECK_EQ(args({{"zip", "true"}}), "{\"zip\":\"true\"}");
    CHECK_EQ(args({{"zip", nullptr}}), "{\"zip\":null}");
    CHECK_EQ(args({{"zip", ""}}), "{\"zip\":\"\"}");
    CHECK_EQ(args({{"limit", 5}}), "{\"limit\":5}");             // Optional[int]
    CHECK_EQ(args({{"limit", nullptr}}), "{\"limit\":null}");
    CHECK_EQ(args({{"code", "007"}}), "{\"code\":\"007\"}");     // a $ref to a string
    CHECK_EQ(args({{"level", "1"}}), "{\"level\":\"1\"}");       // a string enum
    CHECK_EQ(args({{"note", "42"}}), "{\"note\":\"42\"}");       // ["string", "null"]
    CHECK_EQ(args({{"anything", 42}}), "{\"anything\":42}");     // no type: read as it looks
    CHECK_EQ(args({{"filter", {{"a", 1}}}}), "{\"filter\":{\"a\":1}}");

    chat_msg msg = parse(QWEN_XML, render_calls(json::array({call("set_flag", {{"on", true}, {"count", 3}})})));
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == "{\"on\":true,\"count\":3}");
    // Python's True from templates that render with |string
    msg = parse(QWEN_XML, "<tool_call>\n<function=set_flag>\n<parameter=on>\nTrue\n</parameter>\n</function>\n</tool_call>");
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == "{\"on\":true}");

    // plain strings stream as they come
    msg = parse(QWEN_XML, "<tool_call>\n<function=write_file>\n<parameter=content>\nhello wor", true);
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == "{\"content\":\"hello wor");
}

void test_streams_extend() {
    std::printf("every stream of tool calls only grows\n");
    const std::vector<std::string> texts = {
        render_calls(json::array({call("ping", json::object()), call("get_weather", {{"city", "Tokyo"}})})),
        "Let me check.\n" + render_calls(json::array({call("get_weather", {{"city", "Paris"}, {"unit", "celsius"}})})),
        "<think>Should call.</think>" + render_calls(json::array({call("set_flag", {{"on", false}})})),
        render_calls(json::array({call("lookup", {{"zip", "90210"}, {"limit", 3}, {"filter", {{"a", 1}}}})})),
        render_calls(json::array({call("write_file", {{"path", "p"}, {"content", "a</parameter>\nb</parameter>c"}})})),
    };
    for (size_t i = 0; i < texts.size(); i++) {
        check_prefix_stable(QWEN_XML, texts[i], ("stream " + std::to_string(i)).c_str());
    }
}

// ---------------------------------------------------------------- continue_final_message

void test_continued_call() {
    std::printf("a continued message that stops inside a call\n");
    const std::string tail = "Checking now.\n<tool_call>\n<function=get_weather>\n<parameter=city>\n";
    const std::string prefill = chat_format_prefill(QWEN_XML, tail, true, true);
    CHECK_EQ(prefill, "<tool_call>\n<function=get_weather>\n<parameter=city>\n");
    chat_msg msg = parse(QWEN_XML, prefill + "Paris\n</parameter>\n</function>\n</tool_call>");
    CHECK(msg.tool_calls.size() == 1);
    if (msg.tool_calls.size() == 1) {
        CHECK_EQ(msg.tool_calls[0].name, "get_weather");
        CHECK_EQ(msg.tool_calls[0].arguments, "{\"city\":\"Paris\"}");
    }
    CHECK_EQ(msg.content, "");

    // with thinking on, the call streams and stays a call even when its value closes a think tag
    const std::string tail2 = "<tool_call>\n<function=write_file>\n<parameter=content>\nab";
    const std::string prefill2 = chat_format_prefill(QWEN_XML, tail2, true, true);
    CHECK_EQ(prefill2, tail2);
    msg = parse(QWEN_XML, prefill2 + "cd", true, true, true);
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == "{\"content\":\"abcd");
    msg = parse(QWEN_XML, prefill2 + "cd.split('</think>')\n</parameter>\n</function>\n</tool_call>", false, true, true);
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == "{\"content\":\"abcd.split('</think>')\"}");
    // a value that mentions the end tag leaves the call open
    CHECK_EQ(chat_format_prefill(QWEN_XML, "<tool_call>\n<function=write_file>\n<parameter=content>\nwrite </tool_call> then", true, true),
             "<tool_call>\n<function=write_file>\n<parameter=content>\nwrite </tool_call> then");

    // without tools the markup is the client's own text
    CHECK_EQ(chat_format_prefill(QWEN_XML, tail, true, false), "");
    // nor is a call when not continuing
    CHECK_EQ(chat_format_prefill(QWEN_XML, tail, false, true), "");
    // a closed call, the tag in prose, or a call quoted in reasoning hand nothing on
    CHECK_EQ(chat_format_prefill(QWEN_XML, render_calls(json::array({call("ping", json::object())})) + "\nDone", true, true), "");
    CHECK_EQ(chat_format_prefill(QWEN_XML, "I write a <tool_call> tag", true, true), "");
    CHECK_EQ(chat_format_prefill(QWEN_XML, "<think>maybe <tool_call>\n<function=ping></think>Sure", true, true), "");
    CHECK_EQ(chat_format_prefill(QWEN_XML, "<think>maybe <tool_call>\n<function=ping>", true, true), "<think>");
}

// ---------------------------------------------------------------- the request's constraint

void test_no_tools_after_tool_turns() {
    std::printf("a request without tools after tool turns may not call\n");
    const json history = json::array({
        {{"role", "user"}, {"content", "Weather in Paris?"}},
        {{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({call("get_weather", {{"city", "Paris"}})})}},
        {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "18 C"}},
        {{"role", "user"}, {"content", "And Rome?"}},
    });
    const json no_tools = json::array();
    auto choice = [](const json& request, const json& tools) {
        chat_constraint_request out;
        std::string error;
        CHECK(chat_constraint_request_from_openai(request, tools, out, error));
        return out.tool_choice;
    };
    CHECK(choice({{"messages", history}}, no_tools) == chat_tool_choice::NONE);
    CHECK(choice({{"messages", history}, {"tools", no_tools}, {"tool_choice", "auto"}}, no_tools) == chat_tool_choice::NONE);
    CHECK(choice({{"messages", history}, {"tools", TOOLS}}, TOOLS) == chat_tool_choice::AUTO);
    // a plain chat keeps no constraint at all, so speculative decoding stays on
    const json plain = {{"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})}};
    CHECK(choice(plain, no_tools) == chat_tool_choice::AUTO);

    chat_constraint_request request;
    std::string error;
    chat_constraint_request_from_openai({{"messages", history}}, no_tools, request, error);
    chat_vocab vocab;
    vocab.size = 64;
    vocab.encode = [](const std::string& text) { return text == "<tool_call>" ? std::vector<int>{7} : std::vector<int>{1, 2}; };
    auto never = []() -> chat_grammar_engine& { throw std::logic_error("no grammar expected"); };
    std::shared_ptr<decode_constraint> constraint = chat_make_constraint(QWEN_XML, request, vocab, never);
    CHECK(constraint != nullptr);
    if (constraint) {
        std::vector<uint32_t> mask;
        CHECK(constraint->fill_mask(mask));
        CHECK(mask.size() == 2 && (mask[0] & (1u << 7)) == 0 && (mask[0] & (1u << 6)) != 0);
    }
    chat_constraint_request plain_request;
    chat_constraint_request_from_openai(plain, no_tools, plain_request, error);
    CHECK(chat_make_constraint(QWEN_XML, plain_request, vocab, never) == nullptr);
}

void test_reasoning_effort_levels() {
    std::printf("reasoning_effort parses the seven OpenAI levels and rejects anything else\n");
    auto parse = [](const json& request, std::optional<std::string>& out, std::string& error) {
        error.clear();
        return chat_reasoning_effort_from_openai(request, out, error);
    };
    std::optional<std::string> out;
    std::string error;

    // absent or null reads as no override
    CHECK(parse(json::object(), out, error) && !out.has_value());
    CHECK(parse({{"reasoning_effort", nullptr}}, out, error) && !out.has_value());

    // the seven OpenAI levels map to the four canonical FLM levels
    const std::vector<std::pair<std::string, std::string>> expected = {
        {"none", "none"},
        {"minimal", "low"},
        {"low", "low"},
        {"medium", "medium"},
        {"high", "high"},
        {"xhigh", "high"},
        {"max", "high"},
    };
    for (const auto& [raw, canonical] : expected) {
        CHECK(parse({{"reasoning_effort", raw}}, out, error));
        CHECK(out.has_value() && *out == canonical);
        // grammar allow_reasoning agrees with the parser
        chat_constraint_request constraint;
        std::string constraint_error;
        const json no_tools = json::array();
        CHECK(chat_constraint_request_from_openai({{"reasoning_effort", raw}}, no_tools, constraint, constraint_error));
        CHECK(constraint.allow_reasoning == (canonical != "none"));
    }

    // undefined strings and wrong JSON types are refused and name the field
    const std::vector<json> bad_values = {
        "bogus", "", "HIGH", "no_think", "0.5", 1, 0.5, true, false, json::array(), json::object(),
    };
    for (const auto& bad : bad_values) {
        CHECK(!parse({{"reasoning_effort", bad}}, out, error));
        CHECK(!out.has_value());
        CHECK(error.find("reasoning_effort") != std::string::npos);
    }
}

void test_thinking_switch_in_chat_template_kwargs() {
    std::printf("chat_template_kwargs.enable_thinking is the Qwen spelling of think\n");
    // returns: 0 no override, 1 override on, 2 override off, -1 rejected (message in error)
    auto resolve = [](const json& request, std::string* message = nullptr) {
        std::optional<bool> override_value;
        std::string error;
        const bool ok = chat_thinking_override_from_openai(request, override_value, error);
        if (message != nullptr) {
            *message = error;
        }
        return !ok ? -1 : !override_value.has_value() ? 0 : *override_value ? 1 : 2;
    };
    CHECK(resolve({{"chat_template_kwargs", {{"enable_thinking", true}}}}) == 1);
    CHECK(resolve({{"chat_template_kwargs", {{"enable_thinking", false}}}}) == 2);
    // absent or null at either level means no override, as for every other optional field
    CHECK(resolve(json::object()) == 0);
    CHECK(resolve({{"chat_template_kwargs", nullptr}}) == 0);
    CHECK(resolve({{"chat_template_kwargs", json::object()}}) == 0);
    CHECK(resolve({{"chat_template_kwargs", {{"enable_thinking", nullptr}}}}) == 0);
    // only the named key: a neighbouring template variable changes nothing
    CHECK(resolve({{"chat_template_kwargs", {{"add_generation_prompt", false}, {"enable_thinking", true}}}}) == 1);
    CHECK(resolve({{"chat_template_kwargs", {{"preserve_thinking", true}}}}) == 0);
    // wrong types name the field
    std::string message;
    CHECK(resolve({{"chat_template_kwargs", "yes"}}, &message) == -1);
    CHECK(message.find("chat_template_kwargs") != std::string::npos);
    CHECK(resolve({{"chat_template_kwargs", {{"enable_thinking", "yes"}}}}, &message) == -1);
    CHECK(message.find("chat_template_kwargs.enable_thinking") != std::string::npos);
    CHECK(resolve({{"chat_template_kwargs", {{"enable_thinking", 1}}}}) == -1);
    // an agreeing native control is accepted, a contradicting one is not
    CHECK(resolve({{"think", true}, {"chat_template_kwargs", {{"enable_thinking", true}}}}) == 1);
    CHECK(resolve({{"think", false}, {"chat_template_kwargs", {{"enable_thinking", false}}}}) == 2);
    CHECK(resolve({{"reasoning_effort", "none"}, {"chat_template_kwargs", {{"enable_thinking", false}}}}) == 2);
    CHECK(resolve({{"reasoning_effort", "high"}, {"chat_template_kwargs", {{"enable_thinking", true}}}}) == 1);
    CHECK(resolve({{"reasoning_effort", "minimal"}, {"chat_template_kwargs", {{"enable_thinking", true}}}}) == 1);
    CHECK(resolve({{"reasoning_effort", "xhigh"}, {"chat_template_kwargs", {{"enable_thinking", true}}}}) == 1);
    CHECK(resolve({{"reasoning_effort", "max"}, {"chat_template_kwargs", {{"enable_thinking", true}}}}) == 1);
    CHECK(resolve({{"think", false}, {"chat_template_kwargs", {{"enable_thinking", true}}}}, &message) == -1);
    CHECK(message.find("think") != std::string::npos);
    CHECK(resolve({{"think", true}, {"chat_template_kwargs", {{"enable_thinking", false}}}}) == -1);
    CHECK(resolve({{"reasoning_effort", "none"}, {"chat_template_kwargs", {{"enable_thinking", true}}}}) == -1);
    CHECK(resolve({{"reasoning_effort", "low"}, {"chat_template_kwargs", {{"enable_thinking", false}}}}) == -1);
    CHECK(resolve({{"reasoning_effort", "minimal"}, {"chat_template_kwargs", {{"enable_thinking", false}}}}, &message) == -1);
    CHECK(message.find("reasoning_effort") != std::string::npos);
    CHECK(resolve({{"reasoning_effort", "xhigh"}, {"chat_template_kwargs", {{"enable_thinking", false}}}}, &message) == -1);
    CHECK(message.find("reasoning_effort") != std::string::npos);
    CHECK(resolve({{"reasoning_effort", "max"}, {"chat_template_kwargs", {{"enable_thinking", false}}}}, &message) == -1);
    CHECK(message.find("reasoning_effort") != std::string::npos);
    // null native controls count as absent
    CHECK(resolve({{"think", nullptr}, {"chat_template_kwargs", {{"enable_thinking", false}}}}) == 2);
    // without the nested field the native controls are not examined here
    CHECK(resolve({{"think", false}, {"reasoning_effort", "high"}}) == 0);
}

void test_gemma4_nullable_types() {
    std::printf("gemma4 template gets nullable types in its own form\n");
    const json tools = json::array({tool(
        "f", object_schema({
                 {"note", {{"type", {"string", "null"}}}},
                 {"tags", {{"type", "array"}, {"items", {{"type", {"null", "integer"}}}}}},
                 {"either", {{"type", {"string", "number"}}}},
                 {"either_or_null", {{"type", {"string", "number", "null"}}}},
                 {"shape", {{"type", "object"}, {"enum", {{{"type", {"string", "null"}}}}}}},
                 {"labels", {{"type", "object"}, {"additionalProperties", {{"type", {"string", "null"}}}}}},
             }))});
    const json out = gemma4_template_tools(tools)[0]["function"]["parameters"]["properties"];
    CHECK(out["note"] == (json{{"type", "string"}, {"nullable", true}}));
    CHECK(out["tags"]["items"] == (json{{"type", "integer"}, {"nullable", true}}));
    CHECK(out["either"] == (json{{"type", {"string", "number"}}}));
    CHECK(out["either_or_null"] == (json{{"type", {"string", "number", "null"}}}));
    // a literal value is not a schema, even when it looks like one
    CHECK(out["shape"]["enum"][0] == (json{{"type", {"string", "null"}}}));
    CHECK(out["labels"]["additionalProperties"] == (json{{"type", "string"}, {"nullable", true}}));
    // what the template does with a parameter's type, which minja cannot do to a list
    auto tmpl = minja::Parser::parse("{{ value['type'] | upper }}{% if value['nullable'] %} nullable{% endif %}",
                                     {true, true, false});
    CHECK_EQ(tmpl->render(minja::Context::make(minja::Value(json{{"value", out["note"]}}))), "STRING nullable");
}

/// \brief One tool whose string parameter has length bounds, as test_tool_calls.py sends it
const json NOTE_TOOLS = json::array({tool(
    "save_note", object_schema({{"text", {{"type", "string"}, {"minLength", 2}, {"maxLength", 6}, {"pattern", "\\S"}}},
                                {"tags", {{"type", "array"}, {"items", {{"type", "string"}, {"maxLength", 3}}}}}},
                               {"text"}))});

chat_constraint_request note_request() {
    chat_constraint_request request = tools_request(chat_tool_choice::FUNCTION, "save_note");
    request.tools = &NOTE_TOOLS;
    return request;
}

void test_gemma4_string_length() {
    std::printf("gemma4 grammar holds a string to its minLength/maxLength\n");
    grammar_fixture grammar(chat_format_make(chat_format_kind::GEMMA4, true));
    auto note = [](const std::string& text, const std::string& more = "") {
        return "<|tool_call>call:save_note{" + more + "text:<|\"|>" + text + "<|\"|>}<tool_call|>";
    };
    CHECK(grammar.takes(note_request(), note("ab")));
    CHECK(grammar.takes(note_request(), note("abcdef")));
    CHECK(!grammar.takes(note_request(), note("a")));
    CHECK(!grammar.takes(note_request(), note("abcdefg")));
    // "<" counts as one character, and "<|" still cannot occur in the text
    CHECK(grammar.takes(note_request(), note("a<b")));
    CHECK(grammar.takes(note_request(), note("abcd<e")));
    CHECK(!grammar.takes(note_request(), note("abcde<f")));
    CHECK(grammar.takes(note_request(), note("<a")));
    CHECK(!grammar.takes(note_request(), note("a<|b")));
    // array items keep their bounds too
    CHECK(grammar.takes(note_request(), note("ab", "tags:[<|\"|>x<|\"|>,<|\"|>xyz<|\"|>],")));
    CHECK(!grammar.takes(note_request(), note("ab", "tags:[<|\"|>wxyz<|\"|>],")));
}

void test_lfm2_string_length() {
    std::printf("lfm2 grammar holds a string to its minLength/maxLength\n");
    grammar_fixture grammar(chat_format_make(chat_format_kind::LFM2_PYTHONIC, true));
    auto note = [](const std::string& text) {
        return "<|tool_call_start|>[save_note(text=" + text + ")]<|tool_call_end|>";
    };
    CHECK(grammar.takes(note_request(), note("\"abcdef\"")));
    CHECK(grammar.takes(note_request(), note("'ab'")));
    CHECK(!grammar.takes(note_request(), note("\"a\"")));
    CHECK(!grammar.takes(note_request(), note("\"abcdefg\"")));
    // an escape is one character of the value
    CHECK(grammar.takes(note_request(), note("\"abcd\\\"e\"")));
    CHECK(!grammar.takes(note_request(), note("\"abcde\\\"f\"")));
}

chat_constraint_request string_schema_request(const json& property) {
    chat_constraint_request request;
    request.allow_reasoning = false;
    request.json_schema = {{"type", "object"}, {"properties", {{"v", property}}}, {"required", {"v"}},
                           {"additionalProperties", false}};
    return request;
}

void test_string_pattern() {
    std::printf("a string pattern matches anywhere, as JSON Schema reads it\n");
    grammar_fixture grammar(HERMES);
    auto takes = [&grammar](const json& property, const std::string& value) {
        return grammar.takes(string_schema_request(property), "{\"v\": \"" + value + "\"}");
    };
    auto pattern = [](const std::string& p) { return json{{"type", "string"}, {"pattern", p}}; };
    CHECK(takes(pattern("\\S"), "Paris"));
    CHECK(takes(pattern("\\S"), "a"));
    CHECK(!takes(pattern("\\S"), " "));
    CHECK(takes(pattern("\\d{5}"), "zip 90210"));
    // anchored at both ends: as sent
    CHECK(takes(pattern("^\\d{4}$"), "2024"));
    CHECK(!takes(pattern("^\\d{4}$"), "x2024"));
    CHECK(takes(pattern("^ab"), "abc"));
    CHECK(!takes(pattern("^ab"), "cab"));
    CHECK(takes(pattern("^$"), ""));
    CHECK(!takes(pattern("^$"), "a"));
    // each alternative keeps its own anchor
    CHECK(takes(pattern("^a|b"), "ax"));
    CHECK(takes(pattern("^a|b"), "xbx"));
    CHECK(!takes(pattern("^a|b"), "xa"));
    // an anchor inside a group, which xgrammar would ignore: left as sent
    CHECK(takes(pattern("(?:^abc)"), "abc"));
    CHECK(!takes(pattern("(?:^abc)"), "xabc"));
    // padding never ends the string: a non-ASCII pattern is compiled without xgrammar's
    // JSON string guard, where ".*" took "x\", \"k\": \"y" and added a key
    CHECK(takes(pattern("(\xC3\xA9|x)"), "ax"));
    CHECK(!takes(pattern("(\xC3\xA9|x)"), "x\", \"k\": \"y"));
    // a lone class folds its length bounds
    const json bounded = {{"type", "string"}, {"minLength", 2}, {"maxLength", 5}, {"pattern", "\\S"}};
    CHECK(takes(bounded, "abcde"));
    CHECK(!takes(bounded, "abcdef"));
    CHECK(!takes(bounded, "a"));
    // past xgrammar's repetition limit the bounds stay and the pattern goes, nested too
    const json caption = {{"type", "string"}, {"minLength", 1}, {"maxLength", 512}, {"pattern", "\\S"}};
    chat_constraint_request nested = string_schema_request(
        {{"type", "object"}, {"properties", {{"caption", caption}}}, {"required", {"caption"}}, {"additionalProperties", false}});
    CHECK(grammar.takes(nested, "{\"v\": {\"caption\": \"Monthly sales\"}}"));
}

void test_string_pattern_inline_flag() {
    std::printf("a pattern opening with an inline flag is left as sent\n");
    grammar_fixture grammar(HERMES);
    // padded, it would read ".*((?i)abc).*", which xgrammar cannot build
    CHECK(grammar.takes(string_schema_request({{"type", "string"}, {"pattern", "(?i)abc"}}), "{\"v\": \"abc\"}"));
}

void test_schema_literals_untouched() {
    std::printf("literal values and \"not\" are not rewritten as schemas\n");
    grammar_fixture grammar(HERMES);
    // a property named like a keyword is still a schema
    chat_constraint_request request;
    request.allow_reasoning = false;
    request.json_schema = {{"type", "object"},
                           {"properties", {{"default", {{"type", "string"}, {"pattern", "\\S"}}}}},
                           {"required", {"default"}},
                           {"additionalProperties", false}};
    CHECK(grammar.takes(request, "{\"default\": \"Paris\"}"));
    // an enum value shaped like a string schema is a value: it is matched as sent
    request.json_schema = {{"type", "object"},
                           {"properties", {{"v", {{"enum", {{{"type", "string"}, {"pattern", "\\S"}}}}}}}},
                           {"required", {"v"}},
                           {"additionalProperties", false}};
    CHECK(grammar.takes(request, "{\"v\": {\"type\":\"string\",\"pattern\":\"\\\\S\"}}"));
}

void test_positional_tool_results() {
    auto shelf = [](const char* id, const char* args) {
        auto c = call("read_shelf", args);
        c["id"] = id;
        return c;
    };
    const json assistant = {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({
        shelf("z", R"({"shelf":"red"})"), shelf("a", R"({"shelf":"blue"})"), shelf("m", R"({"shelf":"green"})")})}};
    const json red = {{"role", "tool"}, {"tool_call_id", "z"}, {"content", "731"}};
    const json blue = {{"role", "tool"}, {"tool_call_id", "a"}, {"content", "482"}};
    const json green = {{"role", "tool"}, {"tool_call_id", "m"}, {"content", json::array({
        {{"type", "text"}, {"text", "993"}}})}};
    const json boundary = {{"role", "user"}, {"content", "again"}};
    const json group = json::array({assistant, red, blue, green});
    // Two outstanding groups may reuse the same IDs; keep every part of the messages intact.
    for (const json& separator : {boundary, json{{"role", "assistant"}, {"content", "done"}}}) {
        json history = {assistant, green, red, blue, separator, assistant, blue, green, red};
        CHECK(!flm::chat::order_parallel_tool_results(history));
        CHECK(history == json::array({assistant, red, blue, green, separator, assistant, red, blue, green}));
    }
    auto invalid = [](json history, const std::string& path) {
        auto err = flm::chat::order_parallel_tool_results(history);
        CHECK(err && err->param == path && !err->message.empty());
    };
    for (const json& bad_id : {json(nullptr), json(7), json("")}) {
        json history = group;
        history[0]["tool_calls"][1]["id"] = bad_id;
        invalid(history, "messages[0].tool_calls[1].id");
        history = group;
        history[2]["tool_call_id"] = bad_id;
        invalid(history, "messages[2].tool_call_id");
    }
    json missing = group;
    missing[0]["tool_calls"][1].erase("id");
    invalid(missing, "messages[0].tool_calls[1].id");
    invalid(json::array({assistant, blue, green, boundary, red}), "messages[0].tool_calls[0].id");
    invalid(json::array({assistant, blue, green, assistant, red}), "messages[0].tool_calls[0].id");
    invalid(json::array({assistant, blue, green}), "messages[0].tool_calls[0].id");
    invalid(json::array({assistant, red, blue, green, red}), "messages[4].tool_call_id");
    json single = assistant;
    single["tool_calls"] = json::array({assistant["tool_calls"][0]});
    json history = json::array({single, red});
    CHECK(!flm::chat::order_parallel_tool_results(history));
    CHECK(history == json::array({single, red}));
}

// ---------------------------------------------------------------- glm xml (Laguna 2.1)

void test_glm_detection() {
    std::printf("a template that writes <arg_key>/<arg_value> is GLM XML, ahead of Hermes\n");
    auto no_token = [](const std::string&) { return false; };
    const chat_format_spec spec = chat_format_detect("<think></think>" + GLM_RENDER_CALLS, no_token);
    CHECK_EQ(chat_format_name(spec.kind), "glm_xml");
    CHECK(spec.has_reasoning());
    CHECK_EQ(spec.tool_start, "<tool_call>");
    CHECK_EQ(spec.tool_end, "</tool_call>");
    // Nemotron 3.5 writes Qwen XML inside <tool_call>
    CHECK(chat_format_detect("<tool_call>\n<function=f>\n<parameter=k>\n", no_token).kind == chat_format_kind::QWEN_XML);
    CHECK(chat_format_detect("<tool_call>\n{\"name\": f}\n</tool_call>", no_token).kind == chat_format_kind::HERMES_JSON);
    // Hy-Memory (NPU) puts <tool_sep> after the name: it stays on Hermes
    CHECK(chat_format_detect("<tool_calls>\n<tool_call>f<tool_sep>\n<arg_key>k</arg_key>\n<arg_value>v</arg_value>\n"
                             "</tool_call>\n</tool_calls>",
                             no_token)
              .kind == chat_format_kind::HERMES_JSON);
}

void test_glm_parse() {
    std::printf("glm xml: the calls Laguna's template renders parse back\n");
    auto one = [](const json& c) {
        chat_msg msg = parse(GLM_XML, render_glm_calls(json::array({c})));
        return msg.tool_calls.size() == 1 ? msg.tool_calls[0].name + " " + msg.tool_calls[0].arguments
                                          : std::string("<no call>");
    };
    CHECK_EQ(one(call("ping", json::object())), "ping {}");
    CHECK_EQ(one(call("get_weather", {{"city", "Paris"}, {"unit", "celsius"}})),
             "get_weather {\"city\":\"Paris\",\"unit\":\"celsius\"}");
    CHECK_EQ(one(call("set_flag", {{"on", true}, {"count", 3}})), "set_flag {\"on\":true,\"count\":3}");
    // a string is written raw, so a string schema keeps what looks like a number; the rest is JSON
    CHECK_EQ(one(call("lookup", {{"zip", "90210"}, {"limit", 5}, {"code", "007"}, {"filter", {{"a", 1}}}})),
             "lookup {\"zip\":\"90210\",\"limit\":5,\"code\":\"007\",\"filter\":{\"a\":1}}");
    CHECK_EQ(one(call("write_file", {{"path", "a.md"}, {"content", "line 1\nuse <tool_call> here\n"}})),
             "write_file {\"path\":\"a.md\",\"content\":\"line 1\\nuse <tool_call> here\\n\"}");
    CHECK_EQ(one(call("write_file", {{"path", "caf\xC3\xA9"}, {"content", "\xE2\x9C\x93"}})),
             "write_file {\"path\":\"caf\xC3\xA9\",\"content\":\"\xE2\x9C\x93\"}");

    std::printf("glm xml: text, reasoning and parallel calls around it\n");
    const std::string two =
        render_glm_calls(json::array({call("ping", json::object()), call("get_weather", {{"city", "Tokyo"}})}));
    chat_msg msg = parse(GLM_XML, "<think>Need both.</think>Checking." + two);
    CHECK_EQ(msg.reasoning, "Need both.");
    CHECK_EQ(msg.content, "Checking.");
    CHECK(msg.tool_calls.size() == 2);
    // a call opens with a name, then <arg_key> or </tool_call>; the tag in prose is text
    const std::string prose = "Each call goes in a <tool_call> tag, closed by </tool_call>. A <tool_call>name is fine.";
    msg = parse(GLM_XML, prose);
    CHECK_EQ(msg.content, prose);
    CHECK(msg.tool_calls.empty());
    check_prefix_stable(GLM_XML, prose, "glm prose tag");

    std::printf("glm xml: streams only grow\n");
    const std::vector<std::string> texts = {
        two,
        "<think>plan</think>Let me check." +
            render_glm_calls(json::array({call("lookup", {{"zip", "90210"}, {"limit", 3}, {"filter", {{"a", 1}}}})})),
        render_glm_calls(json::array({call("write_file", {{"path", "p"}, {"content", "a</arg_key>b<arg_value>c"}})})),
    };
    for (size_t i = 0; i < texts.size(); i++) {
        check_prefix_stable(GLM_XML, texts[i], ("glm stream " + std::to_string(i)).c_str());
    }
    // a string streams as it comes, held back from a close still being written
    msg = parse(GLM_XML, "<tool_call>write_file<arg_key>content</arg_key><arg_value>hello wor", true);
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == "{\"content\":\"hello wor");
    msg = parse(GLM_XML, "<tool_call>write_file<arg_key>content</arg_key><arg_value>hello</arg_v", true);
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == "{\"content\":\"hello");
    // a name still being written is no call yet, nor text
    msg = parse(GLM_XML, "Sure.<tool_call>get_wea", true);
    CHECK_EQ(msg.content, "Sure.");
    CHECK(msg.tool_calls.empty());
    // a generation cut off inside a value keeps what it wrote
    msg = parse(GLM_XML, "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris");
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].arguments == "{\"city\":\"Paris\"}");
    // a continued message that stops inside a call hands the call on
    CHECK_EQ(chat_format_prefill(GLM_XML, "Checking.<tool_call>get_weather<arg_key>city</arg_key><arg_value>", true, true),
             "<tool_call>get_weather<arg_key>city</arg_key><arg_value>");
}

void test_glm_grammar() {
    std::printf("glm xml grammar takes every call Laguna's template renders\n");
    grammar_fixture grammar(GLM_XML);
    const std::vector<std::pair<const char*, json>> calls = {
        {"no arguments", json::array({call("ping", json::object())})},
        {"no parameters schema", json::array({call("now", json::object())})},
        {"bare object schema", json::array({call("status", {{"any", "thing"}, {"n", 2}})})},
        {"one argument", json::array({call("get_weather", {{"city", "Paris"}})})},
        {"optional argument", json::array({call("get_weather", {{"city", "Paris"}, {"unit", "celsius"}})})},
        {"value with markup", json::array({call("write_file", {{"path", "a.xml"}, {"content", "<a>\n  <arg_value>1</a>\n"}})})},
        {"boolean and integer", json::array({call("set_flag", {{"on", true}, {"count", 3}})})},
        {"raw strings and JSON values",
         json::array({call("lookup", {{"zip", "90210"}, {"limit", 5}, {"code", "007"}, {"level", "2"}, {"note", "n"},
                                      {"anything", "x"}, {"filter", {{"a", 1}}}})})},
        {"parallel", json::array({call("ping", json::object()), call("get_weather", {{"city", "Tokyo"}})})},
    };
    for (const auto& [label, c] : calls) {
        grammar.accepts(tools_request(), render_glm_calls(c), label);
        grammar.accepts(tools_request(chat_tool_choice::REQUIRED), render_glm_calls(c), label);
    }
    grammar.accepts(tools_request(chat_tool_choice::FUNCTION, "get_weather"),
                    render_glm_calls(json::array({call("get_weather", {{"city", "Paris"}})})), "forced call");
    grammar.accepts(tools_request(), "Let me check." + render_glm_calls(json::array({call("ping", json::object())})),
                    "text before the call");

    std::printf("glm xml grammar holds a call to its tool's schema\n");
    auto takes = [&grammar](const json& c) { return grammar.takes(tools_request(), render_glm_calls(json::array({c}))); };
    CHECK(!takes(call("get_weather", {{"city", "Paris"}, {"unit", "kelvin"}})));
    CHECK(!takes(call("get_weather", {{"city", "Paris"}, {"country", "FR"}})));
    CHECK(!takes(call("get_weather", {{"unit", "celsius"}})));
    CHECK(!takes(call("set_flag", {{"on", "yes"}})));
    CHECK(!takes(call("unknown", json::object())));
    // a raw value holds neither its own end nor the call's
    CHECK(!grammar.takes(tools_request(), "<tool_call>write_file<arg_key>path</arg_key><arg_value>p</arg_value>"
                                          "<arg_key>content</arg_key><arg_value>a</tool_call>b</arg_value></tool_call>"));
    chat_constraint_request one_call = tools_request();
    one_call.parallel_tool_calls = false;
    CHECK(!grammar.takes(one_call, render_glm_calls(json::array({call("ping", json::object()), call("ping", json::object())}))));
}

void test_glm_string_length() {
    std::printf("glm xml grammar holds a raw string to its minLength/maxLength\n");
    grammar_fixture grammar(GLM_XML);
    auto note = [](const std::string& text, const std::string& more = "") {
        return "<tool_call>save_note<arg_key>text</arg_key><arg_value>" + text + "</arg_value>" + more + "</tool_call>";
    };
    CHECK(grammar.takes(note_request(), note("ab")));
    CHECK(grammar.takes(note_request(), note("abcdef")));
    CHECK(!grammar.takes(note_request(), note("a")));
    CHECK(!grammar.takes(note_request(), note("abcdefg")));
    // "<" counts as one character, and "</" still cannot occur in the text
    CHECK(grammar.takes(note_request(), note("a<b")));
    CHECK(grammar.takes(note_request(), note("abcd<e")));
    CHECK(!grammar.takes(note_request(), note("abcde<f")));
    CHECK(!grammar.takes(note_request(), note("a</b")));
    // array items, written as JSON, keep their bounds too
    CHECK(grammar.takes(note_request(), note("ab", "<arg_key>tags</arg_key><arg_value>[\"x\", \"xyz\"]</arg_value>")));
    CHECK(!grammar.takes(note_request(), note("ab", "<arg_key>tags</arg_key><arg_value>[\"wxyz\"]</arg_value>")));
}

// ---------------------------------------------------------------- minja on the new templates

void test_minja_tojson_ensure_ascii() {
    std::printf("minja's tojson takes ensure_ascii, as the Laguna 2.1 template writes it\n");
    auto render = [](const std::string& source, const json& context) {
        auto tmpl = minja::Parser::parse(source, {true, true, false});
        return tmpl->render(minja::Context::make(minja::Value(context)));
    };
    const json ctx = {{"v", {{"city", "Z\xC3\xBCrich \xE2\x9C\x93 \xF0\x9F\x98\x80"}, {"n", 1}}}};
    const std::string raw = "{\"city\": \"Z\xC3\xBCrich \xE2\x9C\x93 \xF0\x9F\x98\x80\", \"n\": 1}";
    CHECK_EQ(render("{{ v | tojson }}", ctx), raw);
    CHECK_EQ(render("{{ v | tojson(ensure_ascii=False) }}", ctx), raw);
    CHECK_EQ(render("{{ v | tojson(ensure_ascii=True) }}", ctx),
             "{\"city\": \"Z\\u00fcrich \\u2713 \\ud83d\\ude00\", \"n\": 1}");
    CHECK_EQ(render("{{ v | tojson(indent=2, ensure_ascii=False) }}", json{{"v", {{"a", 1}}}}), "{\n  \"a\": 1\n}");
}

// Laguna-XS-2.1's chat_template.jinja without its comments and its literal BOS
const std::string LAGUNA_TEMPLATE = R"JINJA(
{%- set enable_thinking = enable_thinking | default(false) -%}
{%- set add_generation_prompt = add_generation_prompt | default(false) -%}
{%- set system_message = "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations." -%}
{%- if messages and messages[0].role == "system" -%}
  {%- set system_message = messages[0].content -%}
  {%- set messages = messages[1:] -%}
{%- endif -%}
{%- set has_sys = system_message and system_message.strip() -%}
{%- if has_sys or tools or enable_thinking -%}
  {{- "<system>" -}}
  {%- if has_sys -%}
    {{- system_message.rstrip() -}}
    {%- if tools -%}{{- "\n\n" -}}{%- endif -%}
  {%- endif -%}
  {%- if tools -%}
    {{- "### Tools\n\n" -}}
    {{- "You may call functions to assist with the user query.\n" -}}
    {{- "All available function signatures are listed below:\n" -}}
    {{- "<available_tools>\n" -}}
    {%- for tool in tools -%}
      {{- (tool | tojson) ~ "\n" -}}
    {%- endfor -%}
    {{- "</available_tools>" -}}
  {%- endif -%}
  {{- "</system>\n" -}}
{%- endif -%}
{%- for message in messages -%}
  {%- set content = message.content if message.content is string else "" -%}
  {%- if message.role == "user" -%}
    {{- "<user>" + content + "</user>\n" -}}
  {%- elif message.role == "assistant" -%}
    {%- generation -%}
      {{- "<assistant>" -}}
      {%- set reasoning_content = '' -%}
      {%- if message.reasoning is string -%}
        {%- set reasoning_content = message.reasoning -%}
      {%- elif message.reasoning_content is string -%}
        {%- set reasoning_content = message.reasoning_content -%}
      {%- endif -%}
      {%- if enable_thinking -%}
        {{- '<think>' + reasoning_content + '</think>' -}}
      {%- else -%}
        {{- '</think>' -}}
      {%- endif -%}
      {%- if content -%}
        {{- content -}}
      {%- endif -%}
      {%- if message.tool_calls -%}
        {%- for tool_call in message.tool_calls -%}
          {%- set function_data = tool_call.function -%}
          {{- '<tool_call>' + function_data.name -}}
          {%- set _args = function_data.arguments -%}
          {%- for k, v in _args.items() -%}
            {{- "<arg_key>" ~ k ~ "</arg_key>" -}}
            {{- "<arg_value>" -}}{{- v | tojson(ensure_ascii=False) if v is not string else v -}}{{- "</arg_value>" -}}
          {%- endfor -%}
          {{- "</tool_call>" -}}
        {%- endfor -%}
      {%- endif -%}
      {{- "</assistant>\n" -}}
    {%- endgeneration -%}
  {%- elif message.role == "tool" -%}
    {{- "<tool_response>" + content + "</tool_response>\n" -}}
  {%- elif message.role == "system" -%}
    {{- "<system>" + content + "</system>\n" -}}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
  {{- "<assistant>" -}}
  {%- if enable_thinking -%}
    {{- '<think>' -}}
  {%- else -%}
    {{- '</think>' -}}
  {%- endif -%}
{%- endif -%}
)JINJA";

void test_glm_template_capabilities() {
    std::printf("minja renders Laguna's tool calls and tool responses itself, not as polyfilled JSON\n");
    minja::chat_template laguna(LAGUNA_TEMPLATE, "", "");
    const minja::chat_template_caps& caps = laguna.original_caps();
    CHECK(caps.supports_tools);
    CHECK(caps.supports_tool_calls);
    CHECK(caps.requires_object_arguments);
    CHECK(caps.supports_tool_responses);
    // Hy-Memory's calls, with <tool_sep> after the name, keep the polyfills
    minja::chat_template hy_memory("{%- for message in messages -%}"
                                   "{%- if message.tool_calls -%}"
                                   "{%- for tool in message.tool_calls -%}"
                                   "{{- '<tool_call>' + tool.function.name + '<tool_sep>\\n' -}}"
                                   "{%- for key, value in tool.function.arguments.items() -%}"
                                   "{{- '<arg_key>' + key + '</arg_key>\\n<arg_value>' + value + '</arg_value>\\n' -}}"
                                   "{%- endfor -%}"
                                   "{{- '</tool_call>\\n' -}}"
                                   "{%- endfor -%}"
                                   "{%- else -%}{{- message.content -}}{%- endif -%}"
                                   "{%- endfor -%}",
                                   "", "");
    CHECK(!hy_memory.original_caps().supports_tool_calls);
    minja::chat_template_inputs inputs;
    // as an OpenAI client sends it back: the arguments are a JSON string
    inputs.messages = json::array({
        {{"role", "user"}, {"content", "Weather in Paris?"}},
        {{"role", "assistant"}, {"content", ""},
         {"tool_calls", json::array({call("get_weather", "{\"city\": \"Paris\", \"days\": 3}")})}},
        {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "18 C"}},
    });
    inputs.tools = TOOLS;
    const std::string prompt = laguna.apply(inputs);
    CHECK(prompt.find("<assistant></think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
                      "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call></assistant>\n"
                      "<tool_response>18 C</tool_response>\n<assistant></think>") != std::string::npos);
    CHECK(prompt.find("\"tool_calls\"") == std::string::npos);
}

void test_inkling() {
    std::printf("inkling: typed messages, reasoning, answers and calls\n");
    const chat_format_spec spec = chat_format_make(chat_format_kind::INKLING, false);
    CHECK(chat_format_detect("{{ '<|message_model|>' }}", [](const std::string& t) {
              return t == "<|content_invoke_tool_json|>";
          }).kind == chat_format_kind::INKLING);
    // as the model writes a turn; the end token is dropped before parsing, or kept
    const std::string answer = "<|message_model|><|content_thinking|>84 * 3 = 252, / 2 = 126.<|end_message|>"
                               "<|message_model|><|content_text|>126<|end_message|>";
    chat_msg msg = parse(spec, answer);
    CHECK_EQ(msg.reasoning, "84 * 3 = 252, / 2 = 126.");
    CHECK_EQ(msg.content, "126");
    CHECK(msg.tool_calls.empty());
    CHECK_EQ(parse(spec, answer + "<|content_model_end_sampling|>").content, "126");
    check_prefix_stable(spec, answer, "inkling answer");

    const std::string calls = "<|message_model|><|content_thinking|>Need the weather.<|end_message|>"
                              "<|message_model|>get_weather<|content_invoke_tool_json|>"
                              "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"unit\":\"celsius\"}}<|end_message|>"
                              "<|message_model|>ping<|content_invoke_tool_json|>{\"name\":\"ping\",\"args\":{}}<|end_message|>";
    msg = parse(spec, calls);
    CHECK_EQ(msg.reasoning, "Need the weather.");
    CHECK_EQ(msg.content, "");
    CHECK(msg.tool_calls.size() == 2);
    if (msg.tool_calls.size() == 2) {
        CHECK_EQ(msg.tool_calls[0].name, "get_weather");
        CHECK_EQ(msg.tool_calls[0].arguments, "{\"city\":\"Paris\",\"unit\":\"celsius\"}");
        CHECK_EQ(msg.tool_calls[1].name, "ping");
        CHECK_EQ(msg.tool_calls[1].arguments, "{}");
    }
    check_prefix_stable(spec, calls, "inkling calls");
    // tool calls not asked for are dropped, not shown as text
    msg = parse(spec, calls, false, false);
    CHECK(msg.tool_calls.empty() && msg.content.empty());
    // a call with no JSON stands by its header
    msg = parse(spec, "<|message_model|>now<|content_invoke_tool_json|><|end_message|>");
    CHECK(msg.tool_calls.size() == 1 && msg.tool_calls[0].name == "now" && msg.tool_calls[0].arguments == "{}");
    // text before any message is the answer's
    msg = parse(spec, "hello");
    CHECK_EQ(msg.content, "hello");
    // a stream stops at a half written marker
    msg = parse(spec, "<|message_model|><|content_text|>Hi<|end_mes", true);
    CHECK_EQ(msg.content, "Hi");

    // a continued answer hands its open message on
    CHECK_EQ(chat_format_prefill(spec, "<|end_message|><|message_model|><|content_text|>The", true), "<|message_model|><|content_text|>The");
    CHECK_EQ(chat_format_prefill(spec, "<|message_user|><|content_text|>hi<|end_message|>"), "");

    grammar_fixture grammar(spec);
    const std::string think = "<|message_model|><|content_thinking|>Let me see.<|end_message|>";
    const std::string weather = "<|message_model|>get_weather<|content_invoke_tool_json|>"
                                "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"unit\":\"celsius\"}}<|end_message|>";
    CHECK(grammar.accepts(tools_request(), think + weather, "inkling auto call"));
    CHECK(grammar.accepts(tools_request(), think + "<|message_model|><|content_text|>Hi.<|end_message|>", "inkling auto text"));
    CHECK(grammar.accepts(tools_request(chat_tool_choice::REQUIRED), think + weather, "inkling required"));
    CHECK(grammar.accepts(tools_request(chat_tool_choice::FUNCTION, "get_weather"), weather, "inkling function"));
    CHECK(!grammar.takes(tools_request(chat_tool_choice::REQUIRED), think + "<|message_model|><|content_text|>Hi.<|end_message|>"));
    // a forced call gets one thinking message: a second one would let the model loop
    CHECK(!grammar.takes(tools_request(chat_tool_choice::REQUIRED), think + think + weather));
    CHECK(grammar.accepts(tools_request(), think + think + weather, "inkling auto two thoughts"));
    CHECK(!grammar.takes(tools_request(chat_tool_choice::FUNCTION, "ping"), weather));
    CHECK(!grammar.takes(tools_request(), think + "<|message_model|>get_weather<|content_invoke_tool_json|>"
                                                  "{\"name\":\"get_weather\",\"args\":{\"unit\":\"celsius\"}}<|end_message|>"));
    chat_constraint_request schema = string_schema_request({{"type", "string"}});
    CHECK(grammar.accepts(schema, think + "<|message_model|><|content_text|>{\"v\": \"x\"}<|end_message|>", "inkling schema"));
    CHECK(!grammar.takes(schema, think + "<|message_model|><|content_text|>x<|end_message|>"));
}

} // namespace

int main() {
    test_positional_tool_results();
    test_qwen_xml_grammar_takes_template_calls();
    test_call_without_arguments();
    test_stray_reasoning_end();
    test_tool_tag_in_prose();
    test_value_holding_closing_tags();
    test_argument_types();
    test_streams_extend();
    test_continued_call();
    test_no_tools_after_tool_turns();
    test_reasoning_effort_levels();
    test_thinking_switch_in_chat_template_kwargs();
    test_gemma4_nullable_types();
    test_gemma4_string_length();
    test_lfm2_string_length();
    test_string_pattern();
    test_string_pattern_inline_flag();
    test_schema_literals_untouched();
    test_glm_detection();
    test_glm_parse();
    test_glm_grammar();
    test_glm_string_length();
    test_minja_tojson_ensure_ascii();
    test_glm_template_capabilities();
    test_inkling();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
