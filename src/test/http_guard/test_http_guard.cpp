/// \file test_http_guard.cpp
/// \brief http_guard.hpp: route normalization, the NPU route list against server.cpp's route table,
///        Origin/Host policy, CORS headers, and the 400s the Ollama-style handlers answer for a
///        missing or mistyped field. No NPU needed.
#include "http_guard.hpp"

#include <cstdio>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>

using namespace flm::http_guard;

namespace {
int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                             \
        }                                                                           \
    } while (0)

std::string header(const HeaderList& h, const std::string& name) {
    for (const auto& [k, v] : h)
        if (lower(k) == lower(name)) return v;
    return "";
}

bool is_400_for(const std::optional<ojson>& e, const char* param) {
    if (!e) return false;
    const ojson& err = (*e)["error"];
    const bool code_ok = err["code"].is_number_integer() && err["code"].get<int>() == 400;
    const bool param_ok = param == nullptr ? true : (err["param"].is_string() && err["param"].get<std::string>() == param);
    return code_ok && param_ok && err["type"] == "invalid_request_error" && err["message"].is_string();
}

void test_route_path() {
    std::printf("route path\n");
    CHECK(route_path("/v1/models") == "/v1/models");
    CHECK(route_path("/v1/models?x=1") == "/v1/models");
    CHECK(route_path("/v1/models/") == "/v1/models");
    CHECK(route_path("/v1/models//?a=b#f") == "/v1/models");
    CHECK(route_path("/") == "/");
    CHECK(route_path("") == "/");
    CHECK(route_path("/?x") == "/");
}

/// Every POST route whose handler runs the model must be in requires_npu_access, so it takes the
/// NPU lock or queue. Reads server.cpp; a handler this test does not know must be classified here.
void test_engine_routes_take_npu_queue() {
    std::printf("engine routes take the NPU queue\n");
    std::ifstream in(FLM_SERVER_CPP);
    CHECK(in.good());
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string src = ss.str();

    const std::set<std::string> engine = {"handle_generate", "handle_chat", "handle_embeddings",
        "handle_openai_chat_completion", "handle_openai_completion", "handle_openai_audio_transcriptions",
        "handle_systemone"};
    const std::set<std::string> no_engine = {"handle_show", "handle_ps", "handle_models", "handle_models_openai",
        "handle_version", "handle_pull", "handle_systemone_models"};

    const std::regex reg(R"re(register_handler\(\s*"(GET|POST)"\s*,\s*("([^"]+)"|path))re");
    const std::regex call(R"(rest_handler->(handle_\w+))");
    const std::regex for_paths(R"re(for\s*\(\s*const\s+char\s*\*\s*path\s*:\s*\{([^}]*)\})re");
    const std::regex literal(R"re("([^"]+)")re");

    std::vector<std::smatch> regs;
    for (auto it = std::sregex_iterator(src.begin(), src.end(), reg); it != std::sregex_iterator(); ++it) regs.push_back(*it);
    CHECK(regs.size() >= 15);  // the route table was found at all

    int engine_routes = 0;
    for (size_t i = 0; i < regs.size(); ++i) {
        const size_t begin = regs[i].position(0);
        const size_t end = i + 1 < regs.size() ? static_cast<size_t>(regs[i + 1].position(0)) : src.size();
        const std::string body = src.substr(begin, end - begin);
        const std::string method = regs[i][1];

        std::vector<std::string> paths;
        if (regs[i][3].matched) {
            paths.push_back(regs[i][3]);
        } else {  // register_handler("POST", path, ...) inside `for (const char* path : {...})`
            const std::string before = src.substr(begin > 400 ? begin - 400 : 0, begin > 400 ? 400 : begin);
            std::smatch m;
            std::string last;
            for (auto it = std::sregex_iterator(before.begin(), before.end(), for_paths); it != std::sregex_iterator(); ++it) last = (*it)[1];
            for (auto it = std::sregex_iterator(last.begin(), last.end(), literal); it != std::sregex_iterator(); ++it) paths.push_back((*it)[1]);
            CHECK(!paths.empty());
        }

        std::smatch c;
        if (!std::regex_search(body, c, call)) continue;  // inline handler (cancel, npu/status): no engine
        const std::string handler = c[1];
        const bool known = engine.count(handler) || no_engine.count(handler);
        if (!known) std::printf("  unknown handler %s: add it to one of the sets in this test\n", handler.c_str());
        CHECK(known);
        for (const auto& path : paths) {
            if (engine.count(handler)) {
                ++engine_routes;
                if (!requires_npu_access(method, path)) std::printf("  %s %s calls %s but skips the NPU queue\n", method.c_str(), path.c_str(), handler.c_str());
                CHECK(requires_npu_access(method, path));
            }
        }
    }
    CHECK(engine_routes >= 9);
    CHECK(requires_npu_access("POST", "/api/embeddings"));
    CHECK(!requires_npu_access("GET", "/api/embeddings"));
    CHECK(!requires_npu_access("POST", "/api/show"));
}

void test_origins() {
    std::printf("origins\n");
    OriginPolicy p;  // the default: CORS on, local origins only
    CHECK(origin_allowed(p, ""));
    CHECK(origin_allowed(p, "http://localhost:3000"));
    CHECK(origin_allowed(p, "http://LOCALHOST:5173/"));
    CHECK(origin_allowed(p, "https://127.0.0.1"));
    CHECK(origin_allowed(p, "http://127.1.2.3:8080"));
    CHECK(origin_allowed(p, "http://[::1]:8080"));
    CHECK(origin_allowed(p, "http://app.localhost:3000"));
    CHECK(origin_allowed(p, "app://obsidian.md"));
    CHECK(origin_allowed(p, "vscode-webview://abc"));
    CHECK(!origin_allowed(p, "http://evil.example"));
    CHECK(!origin_allowed(p, "http://localhost.evil.example"));
    CHECK(!origin_allowed(p, "http://127.0.0.1.evil.example"));
    CHECK(!origin_allowed(p, "null"));
    CHECK(!origin_allowed(p, "http://192.168.1.10:3000"));
    CHECK(!origin_allowed(p, "http://0.0.0.0:3000"));  // Host accepts it; an Origin does not
    CHECK(!origin_allowed(p, "http://[::]:3000"));
    CHECK(origin_allowed(p, "chrome-extension://abcdefghijklmnop"));
    CHECK(origin_allowed(p, "moz-extension://1234-5678"));
    // malformed origins a browser never sends: userinfo, a path, a non-numeric port, CR/LF
    CHECK(!origin_allowed(p, "http://localhost:3000@evil.example"));
    CHECK(!origin_allowed(p, "http://localhost:x.evil"));
    CHECK(!origin_allowed(p, "http://localhost/path"));
    CHECK(!origin_allowed(p, "http://localhost:3000\r\nSet-Cookie: a=b"));
    CHECK(!origin_allowed(p, "http://local host"));
    CHECK(origin_allowed(p, "http://[::1]:8080"));

    set_extra_origins(p, " http://nas:3000/ , https://webui.example.com ");
    CHECK(origin_allowed(p, "http://nas:3000"));
    CHECK(origin_allowed(p, "HTTPS://WEBUI.EXAMPLE.COM"));
    CHECK(!origin_allowed(p, "http://nas:3001"));

    set_extra_origins(p, "*");
    CHECK(p.allow_any);
    CHECK(origin_allowed(p, "http://evil.example"));
    CHECK(origin_allowed(p, "null"));
}

void test_hosts() {
    std::printf("hosts\n");
    OriginPolicy p;
    p.check_host = false;  // non-loopback bind: Host is not checked
    CHECK(host_allowed(p, "evil.example"));

    p.check_host = true;
    p.hostname = "MyBox";
    CHECK(host_allowed(p, ""));
    CHECK(host_allowed(p, "localhost:52625"));
    CHECK(host_allowed(p, "127.0.0.1:52625"));
    CHECK(host_allowed(p, "[::1]:52625"));
    CHECK(host_allowed(p, "foo.localhost"));
    CHECK(host_allowed(p, "10.0.0.5:52625"));
    CHECK(host_allowed(p, "192.168.1.20"));
    CHECK(host_allowed(p, "172.20.1.1"));
    CHECK(host_allowed(p, "mybox:52625"));
    CHECK(host_allowed(p, "mybox.local"));
    CHECK(host_allowed(p, "printer.internal"));
    CHECK(host_allowed(p, "0.0.0.0"));
    CHECK(host_allowed(p, "0.0.0.0:52625"));
    CHECK(host_allowed(p, "[::]:52625"));
    CHECK(host_allowed(p, "[::]"));
    CHECK(host_allowed(p, "::"));
    CHECK(!host_allowed(p, "0.0.0.1"));
    CHECK(!host_allowed(p, "evil.example"));
    CHECK(!host_allowed(p, "evil.example:52625"));
    CHECK(!host_allowed(p, "172.32.0.1"));
    CHECK(!host_allowed(p, "8.8.8.8"));
    CHECK(!host_allowed(p, "localhost.evil.example"));

    set_extra_origins(p, "https://llm.example.com");  // a reverse proxy that keeps its Host
    CHECK(host_allowed(p, "llm.example.com"));
    CHECK(host_allowed(p, "llm.example.com:443"));
    CHECK(!host_allowed(p, "evil.example"));
    set_extra_origins(p, "*");  // opts out of the Host check too
    CHECK(host_allowed(p, "evil.example"));

    CHECK(is_loopback_bind("127.0.0.1"));
    CHECK(is_loopback_bind("::1"));
    CHECK(is_loopback_bind("localhost"));
    CHECK(!is_loopback_bind("0.0.0.0"));
    CHECK(!is_loopback_bind("192.168.1.2"));
    CHECK(!parse_ipv4("1.2.3"));
    CHECK(!parse_ipv4("1.2.3.4.5"));
    CHECK(!parse_ipv4("256.1.1.1"));
    CHECK(parse_ipv4("127.0.0.1") == 0x7F000001u);
}

void test_cors_headers() {
    std::printf("cors headers\n");
    OriginPolicy off;
    off.cors = false;
    CHECK(cors_headers(off, "http://localhost:3000").empty());
    CHECK(cors_headers(off, "http://localhost:3000", "content-type", true).empty());
    off.allow_any = true;
    CHECK(cors_headers(off, "http://evil.example").empty());  // --cors 0 wins over --cors-origins *

    OriginPolicy local;
    // not a browser, or not allowed: nothing granted, but caches must still key on Origin
    for (const char* o : {"", "http://evil.example", "http://localhost:3000\r\nX: y"}) {
        auto none = cors_headers(local, o);
        CHECK(header(none, "Access-Control-Allow-Origin").empty());
        CHECK(header(none, "Access-Control-Allow-Methods").empty());
        CHECK(header(none, "Vary") == "Origin");
    }
    auto h = cors_headers(local, "http://localhost:3000");
    CHECK(header(h, "Access-Control-Allow-Origin") == "http://localhost:3000");
    CHECK(header(h, "Vary") == "Origin");
    CHECK(header(h, "Access-Control-Allow-Headers").find("X-Stainless-OS") != std::string::npos);

    auto pre = cors_headers(local, "http://localhost:3000", "x-stainless-os, openai-beta, content-type", true);
    CHECK(header(pre, "Access-Control-Allow-Headers") == "x-stainless-os, openai-beta, content-type");
    CHECK(header(pre, "Access-Control-Allow-Methods") == "GET, POST, OPTIONS");
    CHECK(!header(pre, "Access-Control-Max-Age").empty());
    auto inject = cors_headers(local, "http://localhost:3000", "x\r\nSet-Cookie: a=b", true);
    CHECK(header(inject, "Access-Control-Allow-Headers") == kDefaultAllowHeaders);

    OriginPolicy any;
    set_extra_origins(any, "*");
    auto a = cors_headers(any, "");
    CHECK(header(a, "Access-Control-Allow-Origin") == "*");
    CHECK(header(a, "Vary").empty());
}

void test_required_fields() {
    std::printf("required fields\n");
    const ojson empty = ojson::object();
    CHECK(is_400_for(validate_show(empty), "model"));
    CHECK(is_400_for(validate_show(ojson{{"model", 5}}), "model"));
    CHECK(is_400_for(validate_show(ojson{{"model", nullptr}}), "model"));
    CHECK(!validate_show(ojson{{"model", "llama3.2:1b"}}));
    CHECK(is_400_for(validate_show(ojson()), nullptr));       // empty body
    CHECK(is_400_for(validate_show(ojson::array()), nullptr)); // not an object

    CHECK(is_400_for(validate_generate(empty), "prompt"));
    CHECK(is_400_for(validate_generate(ojson{{"prompt", ojson::array()}}), "prompt"));
    CHECK(!validate_generate(ojson{{"prompt", ""}}));

    CHECK(is_400_for(validate_chat(empty), "messages"));
    CHECK(is_400_for(validate_chat(ojson{{"messages", "hi"}}), "messages"));
    CHECK(!validate_chat(ojson{{"messages", ojson::array()}}));

    CHECK(is_400_for(validate_embeddings(empty), "model"));
    CHECK(is_400_for(validate_embeddings(ojson{{"model", "m"}}), "input"));
    CHECK(is_400_for(validate_embeddings(ojson{{"model", "m"}, {"input", 3}}), "input"));
    CHECK(is_400_for(validate_embeddings(ojson{{"model", "m"}, {"input", ojson::array()}}), "input"));
    CHECK(is_400_for(validate_embeddings(ojson{{"model", "m"}, {"input", ojson::array({"a", 1})}}), "input"));
    CHECK(!validate_embeddings(ojson{{"model", "m"}, {"input", "a"}}));
    CHECK(!validate_embeddings(ojson{{"model", "m"}, {"input", ojson::array({"a", "b"})}}));

    CHECK(is_400_for(validate_cancel_request_id(ojson{{"request_id", 123}}), "request_id"));
    CHECK(!validate_cancel_request_id(ojson{{"request_id", "req_1"}}));
    CHECK(!validate_cancel_request_id(empty));  // absent: the handler's own message (WP-02)

    // send_response turns error.code into the HTTP status: it must be an integer.
    const auto e = error_body("x", 400, "model");
    CHECK(e["error"]["code"].is_number_integer());
    CHECK(error_body("x", 403)["error"]["param"].is_null());
}
}  // namespace

int main() {
    test_route_path();
    test_engine_routes_take_npu_queue();
    test_origins();
    test_hosts();
    test_cors_headers();
    test_required_fields();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}
