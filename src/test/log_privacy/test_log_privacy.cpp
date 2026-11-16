/// \file test_log_privacy.cpp
/// \brief WP-15: the server log carries no user text unless FLM_LOG_CONTENT asks for it, the switches read the
///        environment as documented, and model downloads verify TLS unless FLM_INSECURE_DOWNLOAD=1. No NPU needed.
#include "utils/json_log.hpp"
#include "utils/log_privacy.hpp"
#include "download_tls.hpp"

#include <curl/curl.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                             \
        }                                                                           \
    } while (0)

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

// One body per route, with a marker wherever a client puts text: message content (string and parts), system,
// prompt, input, response_format, tools, stop, user, images and audio.
const char* kBody = R"({
  "model": "qwen3:0.6b", "stream": true, "temperature": 0.25, "max_tokens": 7, "user": "MARKuser",
  "messages": [
    {"role": "system", "content": "MARKsystem be brief"},
    {"role": "user", "content": [{"type": "text", "text": "MARKpart"},
                                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,MARKimage", "detail": "low"}},
                                 {"type": "input_audio", "input_audio": {"data": "MARKaudio", "format": "wav"}}]},
    {"role": "assistant", "content": "MARKreply", "images": ["MARKollamaimage"]}
  ],
  "prompt": "MARKprompt", "system": "MARKgensystem", "input": ["MARKinput1", "MARKinput2"], "suffix": "MARKsuffix",
  "stop": ["MARKstop"], "tool_choice": "auto", "encoding_format": "float",
  "response_format": {"type": "json_schema", "json_schema": {"name": "MARKschema", "schema": {"type": "object",
                      "description": "MARKdescription", "enum": ["MARKenum"]}}},
  "tools": [{"type": "function", "function": {"name": "MARKtool", "description": "MARKtooldesc"}}]
})";

void TestRequestLogHasNoContent() {
    using json = nlohmann::json;
    const std::string logged = json_log::summary(kBody);
    CHECK(!has(logged, "MARK"));
    CHECK(!has(logged, "\n"));  // one line
    const json line = json::parse(logged);
    // the top-level parameters: the model tag, numbers and flags as they are
    CHECK(line["model"] == "qwen3:0.6b");
    CHECK(line["stream"] == true);
    CHECK(line["temperature"] == 0.25);
    CHECK(line["max_tokens"] == 7);
    // text is its length in characters, an array its item count, an object its key count
    CHECK(line["prompt"] == "<10 chars>");
    CHECK(line["tool_choice"] == "<4 chars>");
    CHECK(line["messages"] == "<3 items>");
    CHECK(line["input"] == "<2 items>");
    CHECK(line["response_format"] == "<2 keys>");
    CHECK(json_log::hidden("\xc3\xa9\xf0\x9f\x98\x80x") == "<3 chars>");  // code points, not bytes
    CHECK(json::parse(json_log::summary(R"({"prompt": ""})"))["prompt"] == "<0 chars>");
    // nothing below the top level is shown: not a role, a type, nor a key
    CHECK(!has(logged, "assistant"));
    CHECK(!has(logged, "image_url"));
    CHECK(!has(logged, "json_schema"));
    CHECK(!has(logged, "content"));

    // text under a key that names a setting elsewhere (tool-call arguments, metadata) stays out
    const std::string nested = json_log::summary(R"({"model": "qwen3:0.6b", "metadata": {"role": "MARKrole"},
        "messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "MARKname", "arguments":
        {"detail": "MARKdetail", "type": "MARKtype", "model": "MARKmodel", "object": "MARKobject"}}}]}]})");
    CHECK(!has(nested, "MARK"));
    CHECK(!has(nested, "detail"));
    CHECK(json::parse(nested)["messages"] == "<1 items>");
    // a schema's property names stay out
    CHECK(!has(json_log::summary(R"({"response_format": {"type": "json_schema", "json_schema": {"schema":
        {"properties": {"MARKproperty": {"type": "string"}}}}}})"), "MARK"));
    // text put where a setting goes: a model that does not read as a tag is its length; other strings always are
    const json misplaced = json::parse(json_log::summary(
        R"({"model": "MARK text as the model", "tool_choice": "MARKchoice", "keep_alive": "MARKalive"})"));
    CHECK(misplaced["model"] == "<22 chars>");
    CHECK(misplaced["tool_choice"] == "<10 chars>");
    CHECK(misplaced["keep_alive"] == "<9 chars>");
    CHECK(json::parse(json_log::summary("{\"model\": \"" + std::string(65, 'm') + "\"}"))["model"] == "<65 chars>");
    CHECK(json::parse(json_log::summary("{\"model\": \"mod\xc3\xa9le\"}"))["model"] == "<6 chars>");
    // a parameter whose name does not read as a name is counted, not shown
    const std::string named = json_log::summary("{\"model\": \"m\", \"MARK as a key\": 1, \"MARK\xc3\xa9\": 2, \"seed\": 3}");
    CHECK(!has(named, "MARK"));
    CHECK(json::parse(named)["<other parameters>"] == 2);
    CHECK(json::parse(named)["seed"] == 3);
    CHECK(json_log::is_name("qwen3:0.6b", 64));
    CHECK(!json_log::is_name("", 64));
    CHECK(!json_log::is_name("two words", 64));
    CHECK(!json_log::is_name("tab\there", 64));
    // a body that is not an object
    CHECK(json_log::summary("\"MARKtop\"") == "\"<7 chars>\"");
    CHECK(json_log::summary(R"(["MARKa", "MARKb"])") == "\"<2 items>\"");
    CHECK(json_log::summary("7") == "7");
    // megabytes of image data are neither printed nor kept
    const std::string big = "{\"prompt\": \"" + std::string(100000, 'x') + "\", \"images\": [\"" + std::string(100000, 'y') + "\"]}";
    CHECK(json_log::summary(big) == R"({"images":"<1 items>","prompt":"<100000 chars>"})");

    // the request target: the path, without the query
    CHECK(json_log::target("/v1/models", false) == "/v1/models");
    CHECK(json_log::target("/v1/chat/completions?note=MARKquery&x=1", false) == "/v1/chat/completions?<query hidden>");
    CHECK(json_log::target("/v1/chat/completions?note=MARKquery", true) == "/v1/chat/completions?note=MARKquery");

    // with content on, the body reads as before (long strings still cut)
    const std::string full = json_log::dump(json_log::parse(kBody));
    CHECK(has(full, "MARKprompt"));
    CHECK(has(full, "MARKsystem be brief"));
    // still throws on a malformed body, as nlohmann::json::parse does
    bool threw = false;
    try { json_log::summary("{\"prompt\": "); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    // a parse error quotes the text it last read: the log leaves that out unless content is on
    threw = false;
    try { json_log::summary("{\"messages\": [{\"role\": \"user\", \"content\": \"MARKsecret\n more\"}]}"); }
    catch (const std::exception& e) {
        threw = true;
        CHECK(has(e.what(), "MARKsecret"));
        CHECK(!has(json_log::what(e, false), "MARK"));
        CHECK(has(json_log::what(e, false), "parse error"));
        CHECK(has(json_log::what(e, true), "MARKsecret"));
        CHECK(json_log::what(e, true) == json_log::what(e));
    }
    CHECK(threw);
}

void TestSwitches() {
    using namespace flm::log_privacy;
    set_env("FLM_LOG_CONTENT", nullptr);
    set_env("FLM_LOG_RAW_OUTPUT", nullptr);
    CHECK(!content_enabled());
    CHECK(!raw_output_enabled());
    set_env("FLM_LOG_CONTENT", "0");
    set_env("FLM_LOG_RAW_OUTPUT", "0");
    CHECK(!content_enabled());
    CHECK(!raw_output_enabled());
    set_env("FLM_LOG_RAW_OUTPUT", "1");
    CHECK(!content_enabled());
    CHECK(raw_output_enabled());
    set_env("FLM_LOG_RAW_OUTPUT", nullptr);
    set_env("FLM_LOG_CONTENT", "1");
    CHECK(content_enabled());
    CHECK(raw_output_enabled());
    set_env("FLM_LOG_CONTENT", nullptr);
}

void TestTlsDefaults() {
    set_env("FLM_INSECURE_DOWNLOAD", nullptr);
    const download_utils::TlsSettings tls = download_utils::tls_settings();
    CHECK(tls.verify_peer == 1);
    CHECK(tls.verify_host == 2);
#ifdef _WIN32
    CHECK(tls.native_ca);
    CHECK(tls.revoke_best_effort);
#endif
    set_env("FLM_INSECURE_DOWNLOAD", "0");
    CHECK(download_utils::tls_settings().verify_peer == 1);
    set_env("FLM_INSECURE_DOWNLOAD", nullptr);
}

void TestTlsDowngrade() {
    set_env("FLM_INSECURE_DOWNLOAD", "1");
    const download_utils::TlsSettings tls = download_utils::tls_settings();
    CHECK(tls.verify_peer == 0);
    CHECK(tls.verify_host == 0);
    // a request carrying the Hugging Face token stays verified
    CHECK(download_utils::verified_tls_settings().verify_peer == 1);
    CHECK(download_utils::verified_tls_settings().verify_host == 2);
    set_env("FLM_INSECURE_DOWNLOAD", nullptr);
}

void TestTlsFailureMessage() {
    // the codes is_tls_error names are curl's
    static_assert(CURLE_SSL_CONNECT_ERROR == 35);
    static_assert(CURLE_PEER_FAILED_VERIFICATION == 60);
    static_assert(CURLE_SSL_CACERT_BADFILE == 77);
    static_assert(CURLE_SSL_ISSUER_ERROR == 83);
    CHECK(download_utils::is_tls_error(CURLE_PEER_FAILED_VERIFICATION));
    CHECK(download_utils::is_tls_error(CURLE_SSL_CONNECT_ERROR));
    CHECK(!download_utils::is_tls_error(CURLE_COULDNT_RESOLVE_HOST));
    CHECK(!download_utils::is_tls_error(CURLE_OPERATION_TIMEDOUT));
    CHECK(has(download_utils::tls_failure_hint(), "FLM_INSECURE_DOWNLOAD=1"));
    CHECK(has(download_utils::tls_failure_hint(), "root certificate"));
    CHECK(has(download_utils::tls_failure_hint(), "Hugging Face token"));
}
}  // namespace

int main() {
    TestRequestLogHasNoContent();
    TestSwitches();
    TestTlsDefaults();
    TestTlsDowngrade();
    TestTlsFailureMessage();
    if (failures) { std::printf("%d FAILED\n", failures); return 1; }
    std::printf("all passed\n");
    return 0;
}
