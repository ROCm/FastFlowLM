/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file error_model.hpp
 * \brief Protocol-independent error value and wire writers for OpenAI (/v1/*), Ollama (/api/*),
 *        and System One (/v1/systemone, /v1/decisions). Header-only and free of Boost/XRT/models.
 */
#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace flm::error_model {

using ojson = nlohmann::ordered_json;

enum class Protocol { OpenAI, Ollama, SystemOne, Anthropic };

struct ErrorValue {
    int status = 500;
    std::string type;
    std::string message;
    std::string param;
    std::optional<bool> session_cleared;
};

inline int normalize_status(int status) {
    return (status >= 400 && status <= 599) ? status : 500;
}

inline std::string default_type_for(int status) {
    switch (normalize_status(status)) {
        case 501: return "not_supported_error";
        case 503: return "service_unavailable_error";
        default:  return normalize_status(status) >= 500 ? "server_error" : "invalid_request_error";
    }
}

/// Map empty or legacy internal "model_error" to a wire-valid OpenAI error type.
inline std::string normalize_type(std::string_view type, int status) {
    if (type.empty() || type == "model_error") {
        return default_type_for(status);
    }
    return std::string(type);
}

inline Protocol protocol_for_path(std::string_view target) {
    const auto cut = target.find_first_of("?#");
    std::string_view path = target.substr(0, cut);
    while (path.size() > 1 && path.back() == '/') path.remove_suffix(1);
    if (path == "/v1/messages") return Protocol::Anthropic;
    if (path == "/api" || path.rfind("/api/", 0) == 0) return Protocol::Ollama;
    if (path == "/v1/systemone" || path.rfind("/v1/systemone/", 0) == 0 ||
        path == "/v1/decisions" || path.rfind("/v1/decisions/", 0) == 0) {
        return Protocol::SystemOne;
    }
    return Protocol::OpenAI;
}

/// Serialize JSON safely for wire errors: invalid UTF-8 bytes are replaced instead of throwing.
inline std::string dump_json(const ojson& body) {
    return body.dump(-1, ' ', false, ojson::error_handler_t::replace);
}

/// Render an ErrorValue in the target protocol's JSON wire shape.
inline ojson body_for(Protocol protocol, const ErrorValue& raw) {
    const int status = normalize_status(raw.status);
    if (protocol == Protocol::Anthropic) {
        const char* type = status == 404 ? "not_found_error" : status == 401 ? "authentication_error" :
                           status == 403 ? "permission_error" : status == 413 ? "request_too_large" :
                           status == 429 ? "rate_limit_error" : status == 503 || status == 529 ? "overloaded_error" :
                           status >= 500 ? "api_error" : "invalid_request_error";
        const auto message = raw.param.empty() ? raw.message : raw.param + ": " + raw.message;
        return {{"type","error"},{"error",{{"type",type},{"message",message}}}};
    }
    if (protocol == Protocol::Ollama) {
        ojson out = {{"error", raw.message}};
        if (raw.session_cleared.has_value()) out["session_cleared"] = *raw.session_cleared;
        return out;
    }
    if (protocol == Protocol::SystemOne) {
        return ojson{{"detail", raw.message}};
    }
    ojson err = {
        {"message", raw.message},
        {"type", normalize_type(raw.type, status)},
        {"param", raw.param.empty() ? ojson(nullptr) : ojson(raw.param)},
        {"code", status},
    };
    if (raw.session_cleared.has_value()) err["session_cleared"] = *raw.session_cleared;
    return ojson{{"error", std::move(err)}};
}

/// Canonical internal/OpenAI JSON payload carried through send_response.
inline ojson make_error(const ErrorValue& err) {
    return body_for(Protocol::OpenAI, err);
}

inline ojson make_error(int status, const std::string& message, const std::string& param = {},
                        const std::string& type = {}) {
    ErrorValue ev;
    ev.status = status;
    ev.type = type;
    ev.message = message;
    ev.param = param;
    return make_error(ev);
}

/// Extract an ErrorValue from a structured {"error": {"code": 400..599, ...}} response body.
inline std::optional<ErrorValue> extract_error(const ojson& response_data) {
    if (!response_data.is_object()) return std::nullopt;
    const auto err_it = response_data.find("error");
    if (err_it == response_data.end() || !err_it->is_object()) return std::nullopt;
    const auto code_it = err_it->find("code");
    if (code_it == err_it->end() || !code_it->is_number_integer()) return std::nullopt;
    const int status = code_it->get<int>();
    if (status < 400 || status > 599) return std::nullopt;

    ErrorValue ev;
    ev.status = status;
    if (auto it = err_it->find("message"); it != err_it->end() && it->is_string()) {
        ev.message = it->get<std::string>();
    }
    if (auto it = err_it->find("type"); it != err_it->end() && it->is_string()) {
        ev.type = normalize_type(it->get<std::string>(), status);
    } else {
        ev.type = default_type_for(status);
    }
    if (auto it = err_it->find("param"); it != err_it->end() && it->is_string()) {
        ev.param = it->get<std::string>();
    }
    if (auto it = err_it->find("session_cleared"); it != err_it->end() && it->is_boolean()) {
        ev.session_cleared = it->get<bool>();
    }
    return ev;
}

/// Where an exception was caught: while the handler reads the request, or inside the engine
/// (prefill, generation, finishing the reply) after the request was accepted.
enum class Stage { Request, Engine };

/// std::stoi and friends throw std::invalid_argument with a bare message ("stoi" on libstdc++,
/// "invalid stoi argument" on MSVC). Request parsing names the field instead, so these are internal.
inline bool is_std_conversion_error(const std::exception& e) {
    if (!dynamic_cast<const std::invalid_argument*>(&e)) return false;
    std::string_view what = e.what();
    constexpr std::string_view msvc_prefix = "invalid ";
    constexpr std::string_view msvc_suffix = " argument";
    if (what.size() > msvc_prefix.size() + msvc_suffix.size() && what.substr(0, msvc_prefix.size()) == msvc_prefix &&
        what.substr(what.size() - msvc_suffix.size()) == msvc_suffix) {
        what = what.substr(msvc_prefix.size(), what.size() - msvc_prefix.size() - msvc_suffix.size());
    }
    for (std::string_view fn : {"stoi", "stol", "stoll", "stoul", "stoull", "stof", "stod", "stold"}) {
        if (what == fn) return true;
    }
    return false;
}

/// Map standard/JSON exceptions escaping a handler to 400 (client type/arg fault) or 500.
/// JSON access errors are the client's only while the request is read; inside the engine they come
/// from model files or model output. Engines reject bad input with std::invalid_argument, which stays
/// 400 at either stage, except a bare std::sto* conversion failure, which is never the client's.
inline ErrorValue exception_error(const std::exception& e, Stage stage = Stage::Request) {
    const bool json_access = dynamic_cast<const nlohmann::json::type_error*>(&e) ||
                             dynamic_cast<const nlohmann::json::out_of_range*>(&e) ||
                             dynamic_cast<const nlohmann::json::invalid_iterator*>(&e);
    const bool client_fault = json_access ? stage == Stage::Request
                                          : dynamic_cast<const std::invalid_argument*>(&e) && !is_std_conversion_error(e);
    if (client_fault) {
        return ErrorValue{400, "invalid_request_error", e.what()};
    }
    return ErrorValue{500, "server_error", e.what()};
}

/// Resolve a non-cancelled prefill failure (chat_meta_info_t::error_code defaults to 0):
/// explicit >= 400 wins; unspecified with empty message is 500; unspecified with message is 400.
inline ErrorValue prefill_error(int error_code, const std::string& error_message) {
    const int status = (error_code >= 400 && error_code <= 599)
        ? error_code
        : (error_message.empty() ? 500 : 400);
    const std::string msg = error_message.empty() ? "Prefill failed (see server log)" : error_message;
    return ErrorValue{status, default_type_for(status), msg};
}

/// Format one mid-stream error frame (OpenAI SSE data event or Ollama NDJSON line).
inline std::string stream_error_frame(Protocol protocol, const ErrorValue& err) {
    const std::string payload = dump_json(body_for(protocol, err));
    if (protocol == Protocol::Anthropic) return "event: error\ndata: " + payload + "\n\n";
    if (protocol == Protocol::OpenAI) {
        return "data: " + payload + "\n\n";
    }
    return payload + "\n";
}

/// The last bytes of a stream that fails after it started: OpenAI one error event then [DONE],
/// Ollama one {"error": ...} line (as Ollama does). System One does not stream: nullopt, drop it.
inline std::optional<std::string> stream_abort_frames(Protocol protocol, const ErrorValue& err) {
    switch (protocol) {
        case Protocol::OpenAI: return stream_error_frame(protocol, err) + "data: [DONE]\n\n";
        case Protocol::Anthropic:
        case Protocol::Ollama: return stream_error_frame(protocol, err);
        case Protocol::SystemOne: break;
    }
    return std::nullopt;
}

}  // namespace flm::error_model
