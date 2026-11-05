/// \file test_chat_tools.cpp
/// \brief Tool calling without a model: the output parser on what models write, the decoding grammar
///        against the calls the chat templates render, and the constraint a request gets. No NPU needed.
#include "chat/chat_constraint.hpp"
#include "chat/chat_format.hpp"
#include "chat/chat_grammar.hpp"
#include "chat/chat_parser.hpp"
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
    grammar_fixture() {
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
        engine_ = std::make_unique<chat_grammar_engine>(v, QWEN_XML);
    }
    ~grammar_fixture() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    /// \brief Whether the request's grammar takes the text, without reporting
    bool takes(const chat_constraint_request& request, const std::string& text) {
        std::string error;
        return engine_->accepts(QWEN_XML, request, "", text, error);
    }

    bool accepts(const chat_constraint_request& request, const std::string& text, const char* label) {
        std::string error;
        if (!engine_->accepts(QWEN_XML, request, "", text, error)) {
            std::printf("  FAIL %s: the grammar rejects the template's call: %s\n", label, error.c_str());
            failures++;
            return false;
        }
        return true;
    }

private:
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

} // namespace

int main() {
    test_qwen_xml_grammar_takes_template_calls();
    test_call_without_arguments();
    test_stray_reasoning_end();
    test_tool_tag_in_prose();
    test_value_holding_closing_tags();
    test_argument_types();
    test_streams_extend();
    test_continued_call();
    test_no_tools_after_tool_turns();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
