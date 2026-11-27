/// \file test_error_model.cpp
/// \brief error_model.hpp: protocol routing, OpenAI / Ollama / System One writers, M4 prefill
///        status resolution, M6 model_error mapping, Q7 session_cleared placement, exception
///        classification, stream error frames, and UTF-8-safe JSON serialization. No NPU needed.
#include "error_model.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

using namespace flm::error_model;

namespace {
int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                             \
        }                                                                           \
    } while (0)

void test_protocol_routing() {
    std::printf("protocol routing\n");
    CHECK(protocol_for_path("/v1/chat/completions") == Protocol::OpenAI);
    CHECK(protocol_for_path("/v1/models?x=1") == Protocol::OpenAI);
    CHECK(protocol_for_path("/no/such/route") == Protocol::OpenAI);
    CHECK(protocol_for_path("/api/chat") == Protocol::Ollama);
    CHECK(protocol_for_path("/api/chat/") == Protocol::Ollama);
    CHECK(protocol_for_path("/api/no-such-route?a=1") == Protocol::Ollama);
    CHECK(protocol_for_path("/api") == Protocol::Ollama);
    CHECK(protocol_for_path("/apiv2/chat") == Protocol::OpenAI);
    CHECK(protocol_for_path("/v1/systemone") == Protocol::SystemOne);
    CHECK(protocol_for_path("/v1/systemone/models") == Protocol::SystemOne);
    CHECK(protocol_for_path("/v1/decisions") == Protocol::SystemOne);
    CHECK(protocol_for_path("/v1/decisions/") == Protocol::SystemOne);
    CHECK(protocol_for_path("/v1/systemone_other") == Protocol::OpenAI);
}

void test_writers_and_mapping() {
    std::printf("writers and mapping\n");
    ErrorValue ev;
    ev.status = 404;
    ev.type = "invalid_request_error";
    ev.message = "The model 'x' does not exist.";
    ev.param = "model";

    // OpenAI keeps integer status in error.code (Q2), always includes param.
    const ojson oai = body_for(Protocol::OpenAI, ev);
    CHECK(oai["error"]["code"].is_number_integer() && oai["error"]["code"].get<int>() == 404);
    CHECK(oai["error"]["type"] == "invalid_request_error");
    CHECK(oai["error"]["param"] == "model");
    CHECK(oai["error"]["message"] == ev.message);
    CHECK(!oai["error"].contains("session_cleared"));

    // Ollama renders {"error": "msg"} without session_cleared when unset.
    const ojson oll = body_for(Protocol::Ollama, ev);
    CHECK(oll.size() == 1 && oll["error"] == ev.message);

    // System One renders {"detail": "msg"}.
    const ojson s1 = body_for(Protocol::SystemOne, ev);
    CHECK(s1.size() == 1 && s1["detail"] == ev.message);

    // Absent param becomes explicit JSON null on OpenAI.
    const ojson no_param = make_error(400, "bad input");
    CHECK(no_param["error"].contains("param") && no_param["error"]["param"].is_null());
    CHECK(no_param["error"]["type"] == "invalid_request_error");
    CHECK(no_param["error"]["code"] == 400);

    // M6 & Q8: legacy "model_error" never reaches the wire; 404 defaults to invalid_request_error
    // (not_found_error is explicit for unrouted paths only).
    CHECK(normalize_type("model_error", 400) == "invalid_request_error");
    CHECK(normalize_type("model_error", 404) == "invalid_request_error");
    CHECK(default_type_for(404) == "invalid_request_error");
    CHECK(normalize_type("model_error", 500) == "server_error");
    CHECK(normalize_type("model_error", 503) == "service_unavailable_error");
    CHECK(normalize_type("not_found_error", 404) == "not_found_error");

    // Q7: session_cleared appears inside error on OpenAI and top-level on Ollama whenever set.
    ErrorValue cleared{500, "model_error", "backend poisoned", "", true};
    const ojson oai_cleared = body_for(Protocol::OpenAI, cleared);
    CHECK(oai_cleared["error"]["type"] == "server_error");
    CHECK(oai_cleared["error"]["param"].is_null());
    CHECK(oai_cleared["error"]["session_cleared"] == true);
    const ojson oll_cleared = body_for(Protocol::Ollama, cleared);
    CHECK(oll_cleared["error"] == "backend poisoned");
    CHECK(oll_cleared["session_cleared"] == true);

    ErrorValue kept{503, "model_error", "busy", "", false};
    CHECK(body_for(Protocol::OpenAI, kept)["error"]["session_cleared"] == false);
    CHECK(body_for(Protocol::Ollama, kept)["session_cleared"] == false);

    // Out-of-range status clamps to 500 so ErrorValue can never emit HTTP 200.
    ErrorValue bad_status{200, "", "oops"};
    CHECK(body_for(Protocol::OpenAI, bad_status)["error"]["code"] == 500);
    CHECK(body_for(Protocol::OpenAI, bad_status)["error"]["type"] == "server_error");
}

void test_extract_error() {
    std::printf("extract_error\n");
    // Legacy inline /v1 error omitting param is extracted and normalized with param: null.
    const ojson inline_v1 = {{"error", {{"message", "bad"}, {"type", "invalid_request_error"}, {"code", 400}}}};
    const auto ev = extract_error(inline_v1);
    CHECK(ev.has_value());
    if (ev) {
        CHECK(ev->status == 400 && ev->type == "invalid_request_error" && ev->message == "bad" && ev->param.empty());
        const ojson norm = body_for(Protocol::OpenAI, *ev);
        CHECK(norm["error"].contains("param") && norm["error"]["param"].is_null());
    }

    // Legacy "model_error" + session_cleared round-trips cleanly.
    const ojson model_err = {{"error", {{"message", "reset"}, {"type", "model_error"}, {"code", 500}, {"session_cleared", true}}}};
    const auto mev = extract_error(model_err);
    CHECK(mev.has_value() && mev->type == "server_error" && mev->session_cleared);

    // Non-structured errors, normal responses, and System One __http_status payloads are not extracted.
    CHECK(!extract_error(ojson{{"error", "legacy string"}}).has_value());
    CHECK(!extract_error(ojson{{"object", "list"}, {"data", ojson::array()}}).has_value());
    CHECK(!extract_error(ojson{{"__http_status", 422}, {"__http_body", {{"detail", ojson::array({{{"msg", "bad"}}})}}}}).has_value());
    CHECK(!extract_error(ojson("plain text")).has_value());
    CHECK(!extract_error(ojson{{"error", {{"code", 200}, {"message", "ok"}}}}).has_value());
}

void test_exceptions_and_prefill() {
    std::printf("exceptions and prefill\n");
    // json::type_error, out_of_range, invalid_iterator, std::invalid_argument -> 400.
    try {
        ojson j = {{"stream", "yes"}};
        (void)j.at("stream").get<bool>();
        CHECK(false);
    } catch (const std::exception& e) {
        const auto ev = exception_error(e);
        CHECK(ev.status == 400 && ev.type == "invalid_request_error");
    }
    try {
        ojson j = ojson::object();
        (void)j.at("missing");
        CHECK(false);
    } catch (const std::exception& e) {
        const auto ev = exception_error(e);
        CHECK(ev.status == 400 && ev.type == "invalid_request_error");
    }
    {
        const std::invalid_argument e("bad argument");
        const auto ev = exception_error(e);
        CHECK(ev.status == 400 && ev.type == "invalid_request_error" && ev.message == "bad argument");
    }
    // Internal parse_error (e.g. model file) and generic runtime_error stay 500.
    try {
        (void)ojson::parse("{bad");
        CHECK(false);
    } catch (const std::exception& e) {
        const auto ev = exception_error(e);
        CHECK(ev.status == 500 && ev.type == "server_error");
    }
    {
        const std::runtime_error e("engine crash");
        const auto ev = exception_error(e);
        CHECK(ev.status == 500 && ev.type == "server_error" && ev.message == "engine crash");
    }

    // M4: chat_meta_info_t::error_code defaults to 0 (unspecified).
    // Empty message -> 500 server_error; non-empty message with default 0 -> 400 (preserves 5 models);
    // explicit 4xx/5xx wins.
    const auto reasonless = prefill_error(0, "");
    CHECK(reasonless.status == 500 && reasonless.type == "server_error" && !reasonless.message.empty());
    const auto implicit_client = prefill_error(0, "image decode failed");
    CHECK(implicit_client.status == 400 && implicit_client.type == "invalid_request_error" &&
          implicit_client.message == "image decode failed");
    const auto explicit_500 = prefill_error(500, "npu fault");
    CHECK(explicit_500.status == 500 && explicit_500.type == "server_error");
    const auto explicit_400 = prefill_error(400, "prompt too long");
    CHECK(explicit_400.status == 400 && explicit_400.type == "invalid_request_error");
    const auto explicit_503 = prefill_error(503, "busy");
    CHECK(explicit_503.status == 503 && explicit_503.type == "service_unavailable_error");
}

void test_stream_and_utf8() {
    std::printf("stream frames and utf-8 replacement\n");
    // Invalid UTF-8 bytes in error message must not throw on dump_json or stream_error_frame.
    ErrorValue bad_utf8{400, "invalid_request_error", std::string("bad bytes: \xFF\xFE")};
    const std::string oai_frame = stream_error_frame(Protocol::OpenAI, bad_utf8);
    CHECK(oai_frame.rfind("data: ", 0) == 0);
    CHECK(oai_frame.size() >= 2 && oai_frame.substr(oai_frame.size() - 2) == "\n\n");
    CHECK(oai_frame.find("event:") == std::string::npos);
    const ojson parsed_oai = ojson::parse(oai_frame.substr(6, oai_frame.size() - 8));
    CHECK(parsed_oai["error"]["code"] == 400 && parsed_oai["error"]["param"].is_null());

    const std::string oll_frame = stream_error_frame(Protocol::Ollama, bad_utf8);
    CHECK(!oll_frame.empty() && oll_frame.back() == '\n');
    const ojson parsed_oll = ojson::parse(oll_frame);
    CHECK(parsed_oll["error"].is_string());
}

void test_stream_abort() {
    std::printf("ending a stream that failed after it started\n");
    const ErrorValue err{500, "server_error", "engine fault"};

    // OpenAI: one error event, then [DONE].
    const auto oai = stream_abort_frames(Protocol::OpenAI, err);
    CHECK(oai.has_value());
    if (oai) {
        CHECK(*oai == stream_error_frame(Protocol::OpenAI, err) + "data: [DONE]\n\n");
    }

    // Ollama: one NDJSON line {"error": "<message>"} and nothing after it, as Ollama ends a failed stream.
    const auto oll = stream_abort_frames(Protocol::Ollama, err);
    CHECK(oll.has_value());
    if (oll) {
        CHECK(!oll->empty() && oll->back() == '\n' && oll->find('\n') == oll->size() - 1);
        const ojson line = ojson::parse(*oll);
        CHECK(line.size() == 1 && line["error"] == "engine fault");
    }

    // System One does not stream: the connection is dropped.
    CHECK(!stream_abort_frames(Protocol::SystemOne, err).has_value());
}

}  // namespace

int main() {
    test_protocol_routing();
    test_writers_and_mapping();
    test_extract_error();
    test_exceptions_and_prefill();
    test_stream_and_utf8();
    test_stream_abort();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}
