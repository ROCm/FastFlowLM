/// \file test_chat_gemma4_args.cpp
/// \brief Gemma4 tool call arguments (also Nemotron-3.5-Bolt, same template): a string is raw text between <|"|>
///        markers, so code with quotes in it (np.genfromtxt("a.csv", delimiter=",")) must come through whole
///        (FastFlowLM#900). Also the plain-quote closers the reader still accepts, and streaming. No NPU needed.
#include "chat/chat_parser.hpp"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {
int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                             \
        }                                                                           \
    } while (0)

using nlohmann::json;

const std::string Q = "<|\"|>";

chat_msg parse(const std::string& text, bool partial = false) {
    static const chat_format_spec spec = chat_format_make(chat_format_kind::GEMMA4, false);
    return chat_parse(spec, text, partial, chat_parse_options{});
}

std::string call(const std::string& args) {
    return "<|tool_call>call:file_write{" + args + "}<tool_call|>";
}

/// \brief the arguments of the only call in text, discarded when the parse is not exactly one call
json one_call_args(const std::string& text) {
    chat_msg m = parse(text);
    if (m.tool_calls.size() != 1) {
        return json::value_t::discarded;
    }
    return json::parse(m.tool_calls[0].arguments, nullptr, false);
}

void test_quotes_in_code() {
    std::printf("quotes in code\n");
    // the report: a quote followed by a comma or a brace inside the value
    const std::string np = "import numpy as np\ndata = np.genfromtxt(\"train.csv\", delimiter=\",\")\nprint(data)";
    CHECK(one_call_args(call("content:" + Q + np + Q + ",path:" + Q + "a.py" + Q)) ==
          (json{{"content", np}, {"path", "a.py"}}));
    const std::string d = "d = {\"a\": 1}\nx = f(\"q\",\"r\")";
    CHECK(one_call_args(call("content:" + Q + d + Q + ",path:" + Q + "a.py" + Q)) ==
          (json{{"content", d}, {"path", "a.py"}}));
    // a quote before a closer, a quote before what looks like the next key
    const std::string e = "d = {\"a\": \"b\"}";
    CHECK(one_call_args(call("content:" + Q + e + Q + ",path:" + Q + "a.py" + Q)) ==
          (json{{"content", e}, {"path", "a.py"}}));
    const std::string js = "const o = {name: \"x\", age: 3};";
    CHECK(one_call_args(call("content:" + Q + js + Q + ",path:" + Q + "a.js" + Q)) ==
          (json{{"content", js}, {"path", "a.js"}}));
    const std::string ticks = "echo `date`, now: `x`";
    CHECK(one_call_args(call("content:" + Q + ticks + Q)) == (json{{"content", ticks}}));
    // a closer right after the quote, last argument
    const std::string last = "print(\"a\", \"b\")";
    CHECK(one_call_args(call("path:" + Q + "a.py" + Q + ",content:" + Q + last + Q)) ==
          (json{{"content", last}, {"path", "a.py"}}));
    // quotes in nested values and arrays
    CHECK(one_call_args(call("o:{s:" + Q + "f(\"a\",\"b\")" + Q + ",n:5},l:[" + Q + "x\",\"y" + Q + "," + Q + "z" + Q + "]")) ==
          (json{{"o", {{"s", "f(\"a\",\"b\")"}, {"n", 5}}}, {"l", {"x\",\"y", "z"}}}));
    // the single quote case from the report still works
    CHECK(one_call_args(call("content:" + Q + "d = {'a': 1}" + Q)) == (json{{"content", "d = {'a': 1}"}}));
    // two calls in one turn, the first with quotes
    chat_msg two = parse(call("content:" + Q + "f(\"a\",\"b\")" + Q) + call("path:" + Q + "p" + Q));
    CHECK(two.tool_calls.size() == 2);
    if (two.tool_calls.size() == 2) {
        CHECK(json::parse(two.tool_calls[0].arguments) == (json{{"content", "f(\"a\",\"b\")"}}));
        CHECK(json::parse(two.tool_calls[1].arguments) == (json{{"path", "p"}}));
    }
}

void test_plain_quote_closers() {
    std::printf("plain quote closers\n");
    // the model sometimes closes with a plain quote or a backtick instead of <|"|>
    CHECK(one_call_args(call("content:" + Q + "hello\",path:" + Q + "a.py" + Q)) ==
          (json{{"content", "hello"}, {"path", "a.py"}}));
    CHECK(one_call_args(call("content:" + Q + "hello`,path:" + Q + "a.py" + Q)) ==
          (json{{"content", "hello"}, {"path", "a.py"}}));
    CHECK(one_call_args(call("path:" + Q + "a.py" + Q + ",content:" + Q + "hello\"")) ==
          (json{{"content", "hello"}, {"path", "a.py"}}));
    CHECK(one_call_args(call("l:[" + Q + "a\"," + Q + "b" + Q + "]")) == (json{{"l", {"a", "b"}}}));
    // an opener the model left out
    CHECK(one_call_args(call("content:hello" + Q + ",n:3")) == (json{{"content", "hello"}, {"n", 3}}));
    // a call cut off inside a string still gives valid JSON
    chat_msg cut = parse("<|tool_call>call:f{a:" + Q + "x\"y", false);
    CHECK(cut.tool_calls.size() == 1 && !json::parse(cut.tool_calls[0].arguments, nullptr, false).is_discarded());
}

void test_streaming() {
    std::printf("streaming\n");
    const std::string text = call("content:" + Q + "f(\"a\",\"b\"), c: \"d\"" + Q + ",path:" + Q + "a.py" + Q);
    chat_msg full = parse(text);
    CHECK(full.tool_calls.size() == 1);
    std::string prev;
    for (size_t cut = 0; cut <= text.size(); cut++) {
        chat_msg p = parse(text.substr(0, cut), cut < text.size());
        if (p.tool_calls.empty()) {
            continue;
        }
        const std::string& a = p.tool_calls[0].arguments;
        // a partial parse only ever yields a prefix of the full one
        CHECK(a.size() <= full.tool_calls[0].arguments.size() &&
              full.tool_calls[0].arguments.compare(0, a.size(), a) == 0);
        CHECK(a.size() >= prev.size() && a.compare(0, prev.size(), prev) == 0);
        prev = a;
    }
}

// ———————————————————————————————————————————————
// random calls written the way the Gemma4 template writes them, with values full of quotes, commas and braces
// ———————————————————————————————————————————————
std::mt19937_64 rng(900);
int rnd(int n) {
    return static_cast<int>(rng() % n);
}

const char* const FRAGS[] = {
    "\"", "\",", "\",\"", "\"}", "\"]", "\"},", "\"}}", "\"})", ", key:", "\", key:", "\",k:", "\", name: \"x\"",
    "{a:\"b\", c:1}", "{\"a\": \"b\", \"c\": 1}", "f(\"q\",\"r\")", "np.genfromtxt(\"train.csv\", delimiter=\",\")",
    "print(\"hi\")", "x = [\"a\", \"b\"]", "`", "``", "`x`, y:", "'", "'a', 'b'", "\\", "\\n", "\\\"", "\n", "\t", " ",
    "  ", "{", "}", "[", "]", "(", ")", ",", ":", ", ", " ,", "<", "<|", "|>", "<tool", "call:", "}call:x{",
    "}\ncall:y{a:1", "é", "日本語", "abc", "x", "key:", "a:b", "http://x.y/z?q=1,2", "{\"k\":[1,2,{\"z\":\"w\"}]}",
    "if (a) { b(\"c\"); }", "const o = {name: \"x\", age: 3};", "SELECT \"a\", \"b\" FROM t;", "\"\"", "\"\"\"",
    "\",\n", "\" ,", "\"\n}", "\" }", "\" ]", "\",]", "\",}", "\"}, {\"", "echo \"a\",b", "a\",b", ",\"", ",`", "\"`", "`\""};
constexpr int N_FRAGS = sizeof(FRAGS) / sizeof(FRAGS[0]);

std::string rand_string() {
    std::string s;
    for (int i = 0, n = 1 + rnd(8); i < n; i++) {
        s += FRAGS[rnd(N_FRAGS)];
    }
    // a real marker never occurs inside a value
    return s.find(Q) == std::string::npos ? s : "x\"y";
}

json rand_value(int depth) {
    switch (rnd(depth > 1 ? 6 : 8)) {
        case 0: case 1: case 2: case 5: return rand_string();
        case 3: return rnd(1000) - 300;
        case 4: return rnd(2) == 0;
        case 6: {
            json o = json::object();
            for (int i = 0, n = rnd(3); i < n; i++) o["k" + std::to_string(i)] = rand_value(depth + 1);
            return o;
        }
        default: {
            json a = json::array();
            for (int i = 0, n = rnd(4); i < n; i++) a.push_back(rand_value(depth + 1));
            return a;
        }
    }
}

std::string encode(const json& v) {
    if (v.is_string()) return Q + v.get<std::string>() + Q;
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    if (v.is_number_integer()) return std::to_string(v.get<long long>());
    std::string o;
    bool first = true;
    if (v.is_object()) {
        for (auto& [k, x] : v.items()) {
            o += (first ? "" : ",") + k + ":" + encode(x);
            first = false;
        }
        return "{" + o + "}";
    }
    for (auto& x : v) {
        o += (first ? "" : ",") + encode(x);
        first = false;
    }
    return "[" + o + "]";
}

void test_random_calls() {
    std::printf("random calls\n");
    int bad = 0;
    for (int it = 0; it < 3000 && bad < 3; it++) {
        std::string text;
        std::vector<json> expect;
        for (int c = 0, nc = 1 + rnd(2); c < nc; c++) {
            json args = json::object();
            for (int i = 0, n = rnd(4); i < n; i++) args["a" + std::to_string(i)] = rand_value(0);
            text += "<|tool_call>call:f" + encode(args) + "<tool_call|>";
            expect.push_back(args);
        }
        chat_msg m = parse(text);
        bool ok = m.tool_calls.size() == expect.size();
        for (size_t i = 0; ok && i < expect.size(); i++) {
            ok = json::parse(m.tool_calls[i].arguments, nullptr, false) == expect[i];
        }
        // streaming: every prefix parses to a prefix of the full result; a cut-off call is still valid JSON
        for (size_t cut = 0; ok && cut <= text.size(); cut++) {
            if (cut < text.size() && (text[cut] & 0xC0) == 0x80) {
                continue; // not inside a UTF-8 character
            }
            chat_msg p = parse(text.substr(0, cut), true);
            ok = p.tool_calls.size() <= m.tool_calls.size();
            for (size_t i = 0; ok && i < p.tool_calls.size(); i++) {
                const std::string& a = p.tool_calls[i].arguments;
                ok = m.tool_calls[i].arguments.compare(0, a.size(), a) == 0;
            }
            for (auto& tc : parse(text.substr(0, cut), false).tool_calls) {
                ok = ok && !json::parse(tc.arguments, nullptr, false).is_discarded();
            }
        }
        if (!ok) {
            bad++;
            std::printf("  mismatch on: %s\n", text.c_str());
        }
    }
    CHECK(bad == 0);
}
} // namespace

int main() {
    test_quotes_in_code();
    test_plain_quote_closers();
    test_streaming();
    test_random_calls();
    if (failures > 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}
