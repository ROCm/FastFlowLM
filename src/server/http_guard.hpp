/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file http_guard.hpp
 * \brief Request checks the HTTP server applies before a handler runs: route normalization,
 *        which routes take the NPU queue, Origin/Host policy, CORS headers, and the required
 *        fields of the Ollama-style handlers. Header-only and free of Boost/XRT so the host
 *        tests in src/test/http_guard build it on a plain runner.
 */
#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flm::http_guard {

using ojson = nlohmann::ordered_json;

// ---------------------------------------------------------------------------------------------
// Routing

/// The path a route is looked up by: the request target without its query or fragment and
/// without trailing slashes ("/v1/models?x=1" and "/v1/models/" both give "/v1/models").
inline std::string route_path(std::string_view target) {
    const auto cut = target.find_first_of("?#");
    std::string path(target.substr(0, cut));
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    if (path.empty()) path = "/";
    return path;
}

/// Routes that run the model and so must take the NPU lock or queue. Every POST route whose
/// handler reaches the engine belongs here; src/test/http_guard checks server.cpp against it.
inline bool requires_npu_access(const std::string& method, const std::string& path) {
    if (method != "POST") return false;
    return path == "/api/generate" || path == "/api/chat" || path == "/api/embeddings" ||
           path == "/v1/chat/completions" || path == "/v1/completions" ||
           path == "/v1/audio/transcriptions" || path == "/v1/embeddings" ||
           path == "/v1/systemone" || path == "/v1/decisions";
}

// ---------------------------------------------------------------------------------------------
// Origin / Host policy and CORS headers

struct OriginPolicy {
    /// --cors: write Access-Control-* headers at all.
    bool cors = true;
    /// --cors-origins "*": every Origin is allowed and ACAO is "*" (the behavior before WP-06).
    bool allow_any = false;
    /// --cors-origins: extra origins allowed besides the local ones, normalized by
    /// normalize_origin.
    std::vector<std::string> extra_origins;
    /// The server listens on a loopback address only; then a request must also name a local Host
    /// (blocks DNS rebinding). Off for --host 0.0.0.0 and other non-loopback binds.
    bool check_host = false;
    /// This machine's host name, accepted as a Host on a loopback bind.
    std::string hostname;
};

inline std::string lower(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

inline std::string trim(std::string_view s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return std::string(s.substr(b, e - b));
}

/// Lower-case, trimmed, without a trailing slash: "HTTP://Example.com/" -> "http://example.com".
inline std::string normalize_origin(std::string_view origin) {
    std::string o = lower(trim(origin));
    while (!o.empty() && o.back() == '/') o.pop_back();
    return o;
}

/// Parse --cors-origins: a comma-separated list; "*" anywhere allows every origin.
inline void set_extra_origins(OriginPolicy& policy, std::string_view list) {
    policy.allow_any = false;
    policy.extra_origins.clear();
    size_t start = 0;
    while (start <= list.size()) {
        const size_t comma = list.find(',', start);
        const std::string item = normalize_origin(list.substr(start, comma == std::string_view::npos ? std::string_view::npos : comma - start));
        if (item == "*") policy.allow_any = true;
        else if (!item.empty()) policy.extra_origins.push_back(item);
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
}

/// Split "host:port", "[v6]:port", "host" into the host part (brackets removed, lower-case).
inline std::string host_part(std::string_view authority) {
    std::string a = lower(trim(authority));
    if (!a.empty() && a.front() == '[') {
        const auto close = a.find(']');
        return close == std::string::npos ? a : a.substr(1, close - 1);
    }
    const auto colon = a.find(':');
    if (colon != std::string::npos && a.find(':', colon + 1) == std::string::npos) a.resize(colon);
    while (!a.empty() && a.back() == '.') a.pop_back();  // "localhost." is localhost
    return a;
}

inline bool ends_with(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

/// Parse a dotted IPv4 literal.
inline std::optional<std::uint32_t> parse_ipv4(std::string_view s) {
    std::uint32_t value = 0;
    for (int part = 0; part < 4; ++part) {
        const size_t dot = s.find('.');
        const std::string_view octet = s.substr(0, dot);
        if (octet.empty() || octet.size() > 3 ||
            !std::all_of(octet.begin(), octet.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
            return std::nullopt;
        const unsigned v = static_cast<unsigned>(std::stoul(std::string(octet)));
        if (v > 255) return std::nullopt;
        value = (value << 8) | v;
        if ((dot == std::string_view::npos) != (part == 3)) return std::nullopt;  // exactly four parts
        if (dot != std::string_view::npos) s.remove_prefix(dot + 1);
    }
    return value;
}

inline bool is_loopback_name(std::string_view host) {
    if (host == "localhost" || ends_with(host, ".localhost") || host == "::1") return true;
    const auto v4 = parse_ipv4(host);
    return v4 && (*v4 >> 24) == 127;
}

/// A private-network address: Ollama accepts these as Host on a loopback bind too.
inline bool is_private_ip(std::string_view host) {
    if (const auto v4 = parse_ipv4(host)) {
        const std::uint32_t a = *v4;
        return (a >> 24) == 10 || (a >> 20) == ((172u << 4) | 1u) || (a >> 16) == ((192u << 8) | 168u) ||
               (a >> 16) == ((169u << 8) | 254u);
    }
    // IPv6 unique-local fc00::/7 and link-local fe80::/10
    return host.size() > 2 && host.find(':') != std::string_view::npos &&
           (host.rfind("fc", 0) == 0 || host.rfind("fd", 0) == 0 || host.rfind("fe8", 0) == 0 ||
            host.rfind("fe9", 0) == 0 || host.rfind("fea", 0) == 0 || host.rfind("feb", 0) == 0);
}

/// Is the bind address a loopback address ("127.0.0.1", "::1", "localhost")?
inline bool is_loopback_bind(std::string_view bind_host) {
    return is_loopback_name(host_part(bind_host));
}

/// The host of an http(s) origin "scheme://host[:port]" (lower-case, brackets removed), or
/// nothing when the origin is not http(s) or has userinfo, a path or a non-numeric port.
inline std::optional<std::string> origin_http_host(std::string_view origin) {
    const std::string o = normalize_origin(origin);
    std::string_view rest;
    if (o.rfind("http://", 0) == 0) rest = std::string_view(o).substr(7);
    else if (o.rfind("https://", 0) == 0) rest = std::string_view(o).substr(8);
    else return std::nullopt;
    if (rest.empty() || rest.find_first_of("/@?#\\") != std::string_view::npos) return std::nullopt;
    std::string_view port;
    if (rest.front() == '[') {
        const auto close = rest.find(']');
        if (close == std::string_view::npos) return std::nullopt;
        port = rest.substr(close + 1);
    } else {
        const auto colon = rest.find(':');
        if (colon != std::string_view::npos) port = rest.substr(colon);
    }
    if (!port.empty() && (port.size() < 2 || port.front() != ':' ||
                          !std::all_of(port.begin() + 1, port.end(), [](unsigned char c) { return std::isdigit(c) != 0; })))
        return std::nullopt;
    return host_part(rest);
}

/// Printable ASCII without spaces: what a browser sends, and safe to echo into a header.
inline bool origin_well_formed(std::string_view origin) {
    const std::string o = trim(origin);
    return !o.empty() && std::all_of(o.begin(), o.end(), [](unsigned char c) { return c > 0x20 && c < 0x7f; });
}

/// Origins a local server accepts without configuration: pages served from this machine
/// (http/https on localhost, 127.0.0.0/8, [::1], any port) and desktop-app schemes. "null"
/// (sandboxed iframes, file pages) is not accepted: any site can produce it. Browser-extension
/// schemes are accepted, as Ollama does (Page Assist and similar UIs).
inline bool is_local_origin(std::string_view origin) {
    if (!origin_well_formed(origin)) return false;
    const std::string o = normalize_origin(origin);
    for (std::string_view scheme : {"app://", "tauri://", "vscode-webview://", "vscode-file://", "file://",
                                    "chrome-extension://", "moz-extension://", "safari-web-extension://"}) {
        if (o.rfind(scheme, 0) == 0) return true;
    }
    const auto host = origin_http_host(o);
    return host && is_loopback_name(*host);
}

/// May a request carrying this Origin header be served? No Origin (curl, SDKs, server-side
/// callers) is always allowed.
inline bool origin_allowed(const OriginPolicy& policy, std::string_view origin) {
    if (trim(origin).empty() || policy.allow_any) return true;
    if (!origin_well_formed(origin)) return false;
    if (is_local_origin(origin)) return true;
    const std::string o = normalize_origin(origin);
    return std::find(policy.extra_origins.begin(), policy.extra_origins.end(), o) != policy.extra_origins.end();
}

/// May a request carrying this Host header be served? Only checked on a loopback bind, where a
/// foreign name means the browser was pointed here by DNS rebinding. The hosts of --cors-origins
/// entries are accepted, and --cors-origins "*" turns the check off.
inline bool host_allowed(const OriginPolicy& policy, std::string_view host_header) {
    if (!policy.check_host || policy.allow_any) return true;  // --cors-origins "*" opts out of both checks
    const std::string host = host_part(host_header);
    if (host.empty()) return true;  // HTTP/1.0 without Host
    if (is_loopback_name(host) || is_private_ip(host)) return true;
    // The unspecified address: a client with base_url http://0.0.0.0:port reaches a loopback bind
    // on Linux. Ollama accepts it too. Host only; it is not a local Origin.
    if (host == "0.0.0.0" || host == "::") return true;
    if (ends_with(host, ".local") || ends_with(host, ".internal")) return true;
    for (const auto& extra : policy.extra_origins) {  // a reverse proxy named in --cors-origins keeps its Host
        if (const auto h = origin_http_host(extra); h && *h == host) return true;
    }
    const std::string self = lower(policy.hostname);
    return !self.empty() && (host == self || host == self + ".local");
}

/// The SDK headers a browser preflight asks for (openai-node sends x-stainless-*, OpenAI-Beta).
/// Used when the preflight does not list its headers.
inline constexpr const char* kDefaultAllowHeaders =
    "Authorization, Content-Type, User-Agent, Accept, X-Requested-With, OpenAI-Beta, "
    "OpenAI-Organization, OpenAI-Project, X-Stainless-Arch, X-Stainless-Async, "
    "X-Stainless-Custom-Poll-Interval, X-Stainless-Helper-Method, X-Stainless-Lang, "
    "X-Stainless-OS, X-Stainless-Package-Version, X-Stainless-Poll-Helper, "
    "X-Stainless-Retry-Count, X-Stainless-Runtime, X-Stainless-Runtime-Version, X-Stainless-Timeout";

/// Echo a preflight's Access-Control-Request-Headers if it is a plain header list, else the
/// default list. Anything but token characters, commas and spaces is refused (header injection).
inline std::string allow_headers_for(std::string_view requested) {
    const std::string r = trim(requested);
    if (r.empty()) return kDefaultAllowHeaders;
    for (unsigned char c : r) {
        if (!(std::isalnum(c) || c == '-' || c == '_' || c == ',' || c == ' ')) return kDefaultAllowHeaders;
    }
    return r;
}

using HeaderList = std::vector<std::pair<std::string, std::string>>;

/// The Access-Control-* headers for a response (or a preflight) to a request with this Origin.
/// Empty when CORS is off or the origin is not allowed; no header is ever written with --cors 0.
inline HeaderList cors_headers(const OriginPolicy& policy, std::string_view origin,
                               std::string_view requested_headers = {}, bool preflight = false) {
    HeaderList h;
    if (!policy.cors) return h;
    const std::string o = trim(origin);
    if (policy.allow_any) {
        h.emplace_back("Access-Control-Allow-Origin", "*");
    } else {
        h.emplace_back("Vary", "Origin");  // the answer depends on Origin, so caches must key on it
        if (o.empty() || !origin_allowed(policy, o)) return h;  // not a browser, or not allowed
        h.emplace_back("Access-Control-Allow-Origin", o);       // well formed: no CR/LF reaches the header
    }
    h.emplace_back("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    h.emplace_back("Access-Control-Allow-Headers", preflight ? allow_headers_for(requested_headers) : std::string(kDefaultAllowHeaders));
    if (preflight) h.emplace_back("Access-Control-Max-Age", "600");
    return h;
}

// ---------------------------------------------------------------------------------------------
// Errors (interim shape until WP-02's make_error) and required fields

/// {"error": {message, type, param, code}}: the integer code is what send_response turns into the
/// HTTP status, so a 400 built here is sent as 400, not 200.
inline ojson error_body(const std::string& message, int code, const std::string& param = {},
                        const std::string& type = "invalid_request_error") {
    ojson e = {{"message", message}, {"type", type}, {"param", param.empty() ? ojson(nullptr) : ojson(param)}, {"code", code}};
    return ojson{{"error", e}};
}

/// Missing or not a string -> 400 naming the field.
inline std::optional<ojson> require_string(const ojson& body, const char* key) {
    if (!body.is_object()) return error_body("The request body must be a JSON object.", 400);
    const auto it = body.find(key);
    if (it == body.end() || it->is_null()) return error_body(std::string("Missing required parameter: '") + key + "'.", 400, key);
    if (!it->is_string()) return error_body(std::string("Invalid type for '") + key + "': expected a string.", 400, key);
    return std::nullopt;
}

/// Missing or not an array -> 400 naming the field.
inline std::optional<ojson> require_array(const ojson& body, const char* key) {
    if (!body.is_object()) return error_body("The request body must be a JSON object.", 400);
    const auto it = body.find(key);
    if (it == body.end() || it->is_null()) return error_body(std::string("Missing required parameter: '") + key + "'.", 400, key);
    if (!it->is_array()) return error_body(std::string("Invalid type for '") + key + "': expected an array.", 400, key);
    return std::nullopt;
}

/// POST /api/show: "model" (string).
inline std::optional<ojson> validate_show(const ojson& r) { return require_string(r, "model"); }

/// POST /api/generate: "prompt" (string). Ollama loads the model when prompt is absent; that is
/// WP-21's call, so until then a missing prompt is a 400 rather than undefined behavior.
inline std::optional<ojson> validate_generate(const ojson& r) { return require_string(r, "prompt"); }

/// POST /api/chat: "messages" (array).
inline std::optional<ojson> validate_chat(const ojson& r) { return require_array(r, "messages"); }

/// Embedding requests name a model and supply input; providers validate each input item's contents.
inline std::optional<ojson> validate_embeddings(const ojson& r) {
    if (auto e = require_string(r, "model")) return e;
    const auto it = r.find("input");
    if (it == r.end()) return error_body("Missing required parameter: 'input'.", 400, "input");
    if (it->is_array() && it->empty()) return error_body("'input' must not be an empty batch.", 400, "input");
    return std::nullopt;
}

/// POST /api/cancel: "request_id" must be a string when present.
inline std::optional<ojson> validate_cancel_request_id(const ojson& r) {
    if (r.is_object()) {
        const auto it = r.find("request_id");
        if (it != r.end() && !it->is_string())
            return error_body("Invalid type for 'request_id': expected a string.", 400, "request_id");
    }
    return std::nullopt;
}

}  // namespace flm::http_guard
