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
    const std::string logged = json_log::dump(json_log::parse(kBody, false));
    CHECK(!has(logged, "MARK"));
    // settings, numbers and flags stay readable
    CHECK(has(logged, "\"qwen3:0.6b\""));
    CHECK(has(logged, "\"system\""));      // a role value, and the key of the generate field
    CHECK(has(logged, "\"assistant\""));
    CHECK(has(logged, "\"image_url\""));
    CHECK(has(logged, "\"json_schema\""));
    CHECK(has(logged, "\"auto\""));
    CHECK(has(logged, "\"float\""));
    CHECK(has(logged, "\"low\""));
    CHECK(has(logged, "0.25"));
    CHECK(has(logged, "true"));
    // text becomes its length in characters
    const auto parsed = json_log::parse(kBody, false);
    CHECK(parsed["prompt"] == "<10 chars>");
    CHECK(parsed["messages"][0]["content"] == "<19 chars>");
    CHECK(parsed["input"][1] == "<10 chars>");
    CHECK(json_log::hidden("\xc3\xa9\xf0\x9f\x98\x80x") == "<3 chars>");  // code points, not bytes
    // a top-level string and an empty body string are hidden too
    CHECK(json_log::parse("\"MARKtop\"", false) == "<7 chars>");
    CHECK(json_log::parse(R"({"prompt": ""})", false)["prompt"] == "<0 chars>");
    // with content kept, the body reads as before (long strings still cut)
    const std::string full = json_log::dump(json_log::parse(kBody, true));
    CHECK(has(full, "MARKprompt"));
    CHECK(has(full, "MARKsystem be brief"));
    CHECK(json_log::parse(kBody) == json_log::parse(kBody, true));
    const std::string big = "{\"prompt\": \"" + std::string(100000, 'x') + "\"}";
    CHECK(json_log::parse(big, false)["prompt"] == "<100000 chars>");
    CHECK(json_log::dump(json_log::parse(big, true)).size() < 400);
    // still throws on a malformed body, as nlohmann::json::parse does
    bool threw = false;
    try { json_log::parse("{\"prompt\": ", false); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    // a parse error quotes the text it last read: the log leaves that out unless content is on
    threw = false;
    try { json_log::parse("{\"messages\": [{\"role\": \"user\", \"content\": \"MARKsecret\n more\"}]}", false); }
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
