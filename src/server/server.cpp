/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file server.cpp
 * \brief WebServer class and related declarations
 * \author FastFlowLM Team
 * \date 2025-06-24
 *  \version 0.9.24
 */
#include "server.hpp"
#include "rest_handler.hpp"
#include "utils/json_log.hpp"
#include "utils/log_privacy.hpp"
#include <sstream>
#include <thread>
#include <iostream>
#include <iomanip>
#include <locale>
#include <array>

namespace {
/// Populate a non-streaming HTTP error response in the target protocol's wire shape.
void apply_error_response(http::response<http::string_body>& res,
                          flm::error_model::Protocol proto,
                          const flm::error_model::ErrorValue& err) {
    const int status = flm::error_model::normalize_status(err.status);
    res.result(static_cast<http::status>(status));
    res.set(http::field::content_type, "application/json");
    res.body() = flm::error_model::dump_json(flm::error_model::body_for(proto, err));
    if (status == 503) {
        res.set(http::field::retry_after, "1");
    }
    res.prepare_payload();
}

/// Reject a connection over the connection limit with a real HTTP 503 instead of a bare close.
/// The request is not read; after the reply the write side is shut down and the read side is
/// drained (bounded by a timer) so the kernel does not answer unread request bytes with a RST.
struct ConnectionRejecter : std::enable_shared_from_this<ConnectionRejecter> {
    // The io_context runs on several threads: a strand keeps the timeout's close() from
    // racing the write/read handlers on the same socket.
    net::strand<net::any_io_executor> strand;
    tcp::socket socket;
    net::steady_timer timer;
    std::string reply;
    std::array<char, 4096> sink{};

    explicit ConnectionRejecter(tcp::socket s)
        : strand(net::make_strand(s.get_executor())), socket(std::move(s)), timer(strand) {}

    /// cors_wildcard: write "Access-Control-Allow-Origin: *". The request is never read, so its
    /// Origin is unknown; only a server configured to allow every origin can grant it here.
    void start(size_t limit, bool cors_wildcard) {
        const flm::error_model::ErrorValue err{
            503,
            "service_unavailable_error",
            "Server is at its connection limit (" + std::to_string(limit) +
                "). Please try again later."
        };
        const std::string body = flm::error_model::dump_json(
            flm::error_model::body_for(flm::error_model::Protocol::OpenAI, err));
        reply = std::string("HTTP/1.1 503 Service Unavailable\r\n"
                "Content-Type: application/json\r\n"
                "Retry-After: 1\r\n"
                "Connection: close\r\n") +
                (cors_wildcard ? "Access-Control-Allow-Origin: *\r\n" : "") +
                "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
        auto self = shared_from_this();
        timer.expires_after(std::chrono::seconds(5));
        timer.async_wait(net::bind_executor(strand, [self](beast::error_code ec) {
            if (!ec) { beast::error_code ignore; self->socket.close(ignore); }
        }));
        // Start the write on the strand too, so it is ordered against the timer handler.
        net::dispatch(strand, [self] {
            net::async_write(self->socket, net::buffer(self->reply),
                net::bind_executor(self->strand, [self](beast::error_code ec, std::size_t) {
                    if (ec) { self->finish(); return; }
                    beast::error_code ignore;
                    self->socket.shutdown(tcp::socket::shutdown_send, ignore);
                    self->drain();
                }));
        });
    }

    void drain() {
        auto self = shared_from_this();
        socket.async_read_some(net::buffer(sink),
            net::bind_executor(strand, [self](beast::error_code ec, std::size_t) {
                if (ec) { self->finish(); return; }
                self->drain();
            }));
    }

    void finish() {
        beast::error_code ignore;
        timer.cancel();
        socket.close(ignore);
    }
};

/// Write the Access-Control-* headers http_guard chose (none with --cors 0).
template <class Response>
void set_cors_headers(Response& res, const flm::http_guard::HeaderList& headers) {
    for (const auto& [name, value] : headers) res.set(name, value);
}
} // namespace


// Global NPU access control
std::mutex g_npu_access_mutex;
std::atomic<bool> g_npu_in_use{false};

std::atomic<int> g_npu_active_requests{0};

///@brief get current time string, format: hh:mm:ss mm:dd:yyyy
///@return the current time string
std::string get_current_time_string() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%H:%M:%S %m:%d:%Y");
    return ss.str();
}

// Helper: Truncate a UTF-8 string by code points, not bytes
std::string utf8_truncate_middle(const std::string& input, size_t head_count, size_t tail_count) {
    // Walk the string to count code points
    size_t codepoints = 0;
    std::vector<size_t> positions; // positions of codepoint starts
    for (size_t i = 0; i < input.size();) {
        positions.push_back(i);
        unsigned char c = static_cast<unsigned char>(input[i]);
        size_t char_len = 1;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;
        else char_len = 1; // fallback: treat as single byte
        i += char_len;
        codepoints++;
    }
    if (codepoints <= head_count + tail_count) return input;
    // Get head and tail
    std::string head = input.substr(0, positions[head_count]);
    std::string tail = input.substr(positions[codepoints - tail_count]);
    return head + "..." + tail;
}

///@brief brief print request
///@param request the request, already parsed with json_log::parse (long strings cut)
///@note prints user text: called only when FLM_LOG_CONTENT asks for it (json_log::summary otherwise)
void brief_print_message_request(nlohmann::json request) {
    // if request has "messages" or "message" field, the context in meassages shall be printed briefly
    
    // split ollama and openai message type of image
    
    if (request.contains("messages")) {
        for (auto& message : request["messages"]) {
            // Handle string-based content
            if (message["content"].is_string()) {
                std::string content = message["content"].get<std::string>();
                if (content.size() > 20) {
                    message["content"] = utf8_truncate_middle(content, 10, 10);
                }
            }
            // Handle array-based structured content
            else if (message["content"].is_array()) {
                for (auto& contentItem : message["content"]) {
                    if (contentItem.contains("type") && contentItem["type"] == "text") {
                        std::string text = contentItem["text"].get<std::string>();
                        if (text.size() > 20) {
                            contentItem["text"] = utf8_truncate_middle(text, 10, 10);
                        }
                    }
                    else if (contentItem.contains("type") && contentItem["type"] == "image_url" &&
                             contentItem["image_url"].is_object() && contentItem["image_url"].contains("url") &&
                             contentItem["image_url"]["url"].is_string()) {
                        std::string image_url = contentItem["image_url"]["url"].get<std::string>();
                        if (image_url.size() > 20) {
                            contentItem["image_url"]["url"] = utf8_truncate_middle(image_url, 10, 10);
                        }
                    }
                    else if (contentItem.contains("type") && contentItem["type"] == "input_audio") {
                        std::string audio_base64 = contentItem["input_audio"]["data"].get<std::string>();
                        if (audio_base64.size() > 20) {
                            contentItem["input_audio"]["data"] = utf8_truncate_middle(audio_base64, 10, 10);
                        }
                    }
                }
            }
            // Handle legacy "images" array
            if (message.contains("images")) {
                nlohmann::ordered_json::array_t images = message.value("images", nlohmann::ordered_json::array());
                nlohmann::ordered_json::array_t new_images;
                for (auto& image : images) {
                    std::string image_str = image.get<std::string>();
                    if (image_str.size() > 20) {
                        new_images.push_back(utf8_truncate_middle(image_str, 10, 10));
                    }
                    else {
                        new_images.push_back(image_str);
                    }
                }
                message["images"] = new_images;
            }
        }
    }

    // TODO: improve tools logging like only print the tool name and elide the arguments
    // TODO: Support debug level
    if (request.contains("tools")) {
        request["tools"] = "...";
    }

    header_print("LOG", "Body: ");
    std::string brief_body = json_log::dump(request);
    std::cout << brief_body << std::endl;
}

///@brief brief print request
///@param request the request
void brief_print_message_response(nlohmann::json request) {

}

// NPU Access Manager implementation
bool AcceleratorAccessManager::try_acquire_npu_access() {
    std::lock_guard<std::mutex> lock(g_npu_access_mutex);
    if (g_npu_in_use.load()) {
        return false; // NPU is already in use
    }
    g_npu_in_use.store(true);
    g_npu_active_requests.fetch_add(1);
    return true;
}

void AcceleratorAccessManager::release_npu_access() {

    header_print("🔵 ", "Accelerator lock released");
    std::lock_guard<std::mutex> lock(g_npu_access_mutex);
    g_npu_in_use.store(false);
    g_npu_active_requests.fetch_sub(1);
}

bool AcceleratorAccessManager::is_npu_available() {
    return !g_npu_in_use.load();
}

int AcceleratorAccessManager::get_active_npu_requests() {
    return g_npu_active_requests.load();
}

///@brief HttpSession class implementation
///@param socket the socket
///@param server the server
HttpSession::HttpSession(tcp::socket socket, WebServer& server)
    : socket_(std::move(socket))
    , server_(server)
    , is_streaming_(false) {
    // Set socket timeout
    socket_.set_option(tcp::socket::keep_alive(false));
    // Avoid abortive close that can lead to client-side broken pipe on large uploads
    socket_.set_option(tcp::socket::linger(false, 0));
    
    // Debug: Log TCP connection formation
    header_print("🔗 ", "TCP connection established - Remote: " + socket_.remote_endpoint().address().to_string() + ":" + std::to_string(socket_.remote_endpoint().port()));
}

///@brief destructor: a session never outlives its slot in the connection limit
HttpSession::~HttpSession() {
    release_connection();
}

///@brief release this connection's slot (idempotent; every close path may call it)
void HttpSession::release_connection() {
    if (!connection_released_.exchange(true)) {
        server_.active_connections_.fetch_sub(1);
    }
}

///@brief start
void HttpSession::start(bool cors) {
    read_request(cors);
}

void HttpSession::abort_open_stream(flm::error_model::Protocol proto, const flm::error_model::ErrorValue& err) {
    if (stream_finished_.load()) {
        return;
    }
    if (proto == flm::error_model::Protocol::OpenAI) {
        write_streaming_response(json(flm::error_model::stream_error_frame(proto, err)), false);
        write_streaming_response(json("data: [DONE]\n\n"), true);
    } else {
        close_connection();
    }
}

///@brief close connection
void HttpSession::close_connection() {
    stream_finished_.store(true);
    boost::system::error_code ec;
    
    // Debug: Log TCP connection disconnection
    try {
        header_print("🔒 ", "TCP connection closing - Remote: " + socket_.remote_endpoint().address().to_string() + ":" + std::to_string(socket_.remote_endpoint().port()));
    } catch (const std::exception& e) {
        header_print("🔒 ", "TCP connection closing - Remote endpoint unavailable");
    }
    socket_.shutdown(tcp::socket::shutdown_both, ec);
    socket_.cancel(ec);
    release_connection();
}

///@brief read request
void HttpSession::read_request(bool cors) {
    auto self = shared_from_this();

    // Use a parser to control the maximum accepted body size
    auto parser = std::make_shared<http::request_parser<http::string_body>>();
    parser->body_limit(self->server_.get_max_body_size_bytes());

    http::async_read(self->socket_, self->buffer_, *parser,
        [self, parser, cors](beast::error_code ec, std::size_t bytes_transferred) {
            if (!ec) {
                header_print("TCP", "Read " + std::to_string(bytes_transferred) + " bytes from socket");
                // Move the parsed message into our request object
                self->req_ = parser->release();
                self->handle_request(cors);
                return;
            }

            // Handle body too large explicitly with 413
            if (ec == http::error::body_limit) {
                // The header was read: keep it so the reply's CORS headers and protocol see it.
                self->req_ = parser->release();
                self->res_ = {};
                self->res_.version(11);
                self->res_.keep_alive(false);
                const auto proto = flm::error_model::protocol_for_path(
                    std::string_view(self->req_.target().data(), self->req_.target().size()));
                const flm::error_model::ErrorValue err{
                    413,
                    "invalid_request_error",
                    "Request payload too large (max " + std::to_string(self->server_.get_max_body_size_bytes()) + " bytes)."
                };
                apply_error_response(self->res_, proto, err);
                self->write_response();
                return;
            }

            // Connection closed or other error, decrement connection counter
            try {
                header_print("🔒 ", "TCP connection closed - Remote: " + self->socket_.remote_endpoint().address().to_string() + ":" + std::to_string(self->socket_.remote_endpoint().port()));
            } catch (...) {
                header_print("🔒 ", "TCP connection closed - Remote endpoint unavailable");
            }
            self->release_connection();
        });
}

///@brief handle request

void HttpSession::handle_request(bool cors) {
    // --- Origin and Host checks, before anything else (preflight included) ---
    // A web page may send "simple" requests here without a preflight, and they have effects
    // (/api/pull, /api/cancel, model switches) even when the page cannot read the reply. A page
    // from a foreign origin, or one reached through a foreign Host name (DNS rebinding), is
    // refused before routing.
    {
        const auto& policy = server_.origin_policy();
        const std::string origin(req_[http::field::origin]);
        const std::string host(req_[http::field::host]);
        const bool origin_ok = flm::http_guard::origin_allowed(policy, origin);
        if (!origin_ok || !flm::http_guard::host_allowed(policy, host)) {
            const std::string message = !origin_ok
                ? "Origin '" + origin + "' is not allowed. Start flm serve with --cors-origins \"" + origin +
                      "\" to allow it (\"*\" allows every origin)."
                : "Host '" + host + "' is not allowed for a server bound to a loopback address. If a proxy "
                  "forwards this name, add its origin to --cors-origins (e.g. \"https://" + host + "\").";
            header_print("🚫 ", message);
            const auto proto = flm::error_model::protocol_for_path(
                std::string_view(req_.target().data(), req_.target().size()));
            res_ = {};
            res_.version(req_.version());
            res_.keep_alive(false);
            apply_error_response(res_, proto, flm::error_model::ErrorValue{403, "invalid_request_error", message});
            write_response();
            return;
        }
    }

    // --- BEGIN: Preflight (OPTIONS) request handling ---
    if (req_.method() == http::verb::options && cors) {
        header_print("✈️ ", "Handling Preflight (OPTIONS) request");

        // reponse empty body for OPTIONS
        auto options_res = std::make_shared<http::response<http::empty_body>>();
        options_res->version(req_.version());
        options_res->result(http::status::ok);
        options_res->keep_alive(req_.keep_alive());

        set_cors_headers(*options_res, flm::http_guard::cors_headers(server_.origin_policy(),
            req_[http::field::origin], req_[http::field::access_control_request_headers], true));

        options_res->prepare_payload();
        auto self = shared_from_this();
        http::async_write(socket_, *options_res,
            [self, cors, options_res](beast::error_code ec, std::size_t) {
                if (ec) {
                    return;
                }
                // waiting the true response
                self->read_request(cors);
            });
        return;
    }
    // --- END: Preflight (OPTIONS) request handling ---


    // IF not OPTIONS
    // Reset response
    res_ = {};
    res_.version(req_.version());
    res_.keep_alive(req_.keep_alive());

    // Handle the request through the server. Nothing may escape into ioc.run(): answer 500
    // (or, mid-stream, drop the connection) instead of leaving the client without a reply.
    bool deferred = false;
    const auto proto = flm::error_model::protocol_for_path(
        std::string_view(req_.target().data(), req_.target().size()));
    try {
        deferred = server_.handle_request(req_, res_, socket_, shared_from_this());
    } catch (const std::exception& e) {
        header_print("LOG", "Unhandled exception in request: " + std::string(e.what()));
        deferred = false;
        auto err = flm::error_model::exception_error(e);
        if (err.status == 500) err.message = "Internal server error: " + err.message;
        if (try_begin_response()) {
            res_ = {};
            res_.version(req_.version());
            res_.keep_alive(false);
            apply_error_response(res_, proto, err);
        } else if (is_streaming_.load()) {
            abort_open_stream(proto, err);
        }
    } catch (...) {
        header_print("LOG", "Unknown unhandled exception in request");
        deferred = false;
        const flm::error_model::ErrorValue err{500, "server_error", "Internal server error"};
        if (try_begin_response()) {
            res_ = {};
            res_.version(req_.version());
            res_.keep_alive(false);
            apply_error_response(res_, proto, err);
        } else if (is_streaming_.load()) {
            abort_open_stream(proto, err);
        }
    }

    // Clear the buffer after processing to prevent data accumulation
    buffer_.consume(buffer_.size());

    if (!deferred && !is_streaming_.load()) {
        write_response();
    } else {
        // For streaming responses, we need to handle connection cleanup differently
        // The connection will be closed when streaming ends
    }
}

///@brief write response
void HttpSession::write_response() {
    if (is_streaming_.load()) {
        return;
    }
    auto self = shared_from_this();

    // header_print("⬆️ ", "Outgoing Response: ");
    // header_print("LOG", "Time stamp: " << get_current_time_string()); // hh:mm:ss mm:dd:yyyy
    
    // Check if this is one of the endpoints where we should skip printing the response body
    std::string target = std::string(req_.target());
    bool skip_body_print = (target == "/api/ps" || target == "/api/tags" || target == "/api/version" || target == "/v1/models" || target == "/api/show");
    
    //if (!skip_body_print) {
    //    try{
    //        nlohmann::json response_json = json::parse(res_.body());
    //        brief_print_message_request(response_json);
    //        // std::cout << "response_json: " << response_json.dump(4) << std::endl;
    //    } catch (const std::exception& e) {
    //        header_print("LOG", "Body: ");
    //        std::cout << res_.body() << std::endl;
    //    }
    //}

    std::cout << "================================================" << std::endl;

    // Every non-streaming exit (normal, 4xx/5xx, 404, 413, 503 queue full) passes here.
    set_cors_headers(res_, flm::http_guard::cors_headers(server_.origin_policy(), req_[http::field::origin]));


    http::async_write(socket_, res_,
        [self](beast::error_code ec, std::size_t) {
            self->socket_.shutdown(tcp::socket::shutdown_both, ec);
            // Decrement connection counter for non-keep-alive connections
            self->release_connection();
            try {
                header_print("🔒 ", "TCP connection closed - Remote: " + self->socket_.remote_endpoint().address().to_string() + ":" + std::to_string(self->socket_.remote_endpoint().port()));
            }
            catch (...) {
                header_print("🔒 ", "TCP connection closed - Remote endpoint unavailable");
            }

        });
}

///@brief write response from callback (for queued requests)
void HttpSession::write_response_from_callback() {
    // This function is called by the send_response lambda
    // when a queued request is finally processed.
    write_response();
}

void HttpSession::set_cancellation_token(std::shared_ptr<CancellationToken> token) {
    cancellation_token_ = token;
    start_client_disconnect_monitor(token);
}

void HttpSession::start_client_disconnect_monitor(std::shared_ptr<CancellationToken> token) {
    if (!token || token->cancelled() || !socket_.is_open()) {
        return;
    }

    auto self = shared_from_this();
    socket_.async_wait(tcp::socket::wait_read,
        [self, token](beast::error_code ec) {
            if (!token || token->cancelled() || token->completed()) {
                return;
            }

            if (ec) {
                if (ec != net::error::operation_aborted && !token->completed()) {
                    token->cancel();
                    header_print("🔴 ", "Client socket wait failed; cancelling active request: " + ec.message());
                }
                return;
            }

            char peek_buffer = 0;
            boost::system::error_code peek_ec;
            std::size_t bytes_read = self->socket_.receive(
                net::buffer(&peek_buffer, 1),
                boost::asio::socket_base::message_peek,
                peek_ec);

            if ((peek_ec || bytes_read == 0) && !token->completed()) {
                token->cancel();
                header_print("🔴 ", "Client disconnected; cancelling active request");
                return;
            }
        });
}

///@brief write streaming response
///@param data the data
///@param is_final the is final
void HttpSession::write_streaming_response(const json& data, bool is_final) {
    if (stream_finished_.load()) {
        return;
    }
    if (!is_streaming_.load()) {
        if (!try_begin_response()) {
            return;
        }
        is_streaming_.store(true);
        // Initialize streaming response headers
        res_.result(http::status::ok);
        res_.set(http::field::content_type, "application/x-ndjson");
        res_.set(http::field::cache_control, "no-cache");
        res_.set(http::field::connection, "keep-alive");
        res_.set(http::field::transfer_encoding, "chunked");
        
        // Send headers immediately using raw socket write
        std::string headers = "HTTP/1.1 200 OK\r\n";
        headers += "Content-Type: text/event-stream\r\n";
        headers += "Cache-Control: no-cache\r\n";
        headers += "Connection: keep-alive\r\n";
        headers += "Transfer-Encoding: chunked\r\n";
        for (const auto& [name, value] : flm::http_guard::cors_headers(server_.origin_policy(), req_[http::field::origin])) {
            headers += name + ": " + value + "\r\n";
        }
        headers += "\r\n";
        
        // Send headers synchronously
        boost::system::error_code ec;
        net::write(socket_, net::buffer(headers), ec);
        if (ec) {
            stream_finished_.store(true);
            if (cancellation_token_ && !cancellation_token_->completed()) {
                cancellation_token_->cancel();
            }
            boost::system::error_code ignore_ec;
            socket_.close(ignore_ec);
            release_connection();
            return;
        }
    }
    
    // Send this chunk immediately
    send_chunk_data(data, is_final);
}

///@brief send chunk data
///@param data the data
///@param is_final the is final
void HttpSession::send_chunk_data(const json& data, bool is_final) {
    if (stream_finished_.load()) {
        return;
    }
    const std::string chunk_content = data.is_string()
        ? data.get<std::string>()
        : flm::error_model::dump_json(data) + "\n";

    boost::system::error_code ec;
    if (!chunk_content.empty()) {
        // HTTP chunked format: size in hex + \r\n + data + \r\n
        std::ostringstream chunk_size;
        chunk_size << std::hex << chunk_content.length();
        std::string http_chunk = chunk_size.str() + "\r\n" + chunk_content + "\r\n";

        net::write(socket_, net::buffer(http_chunk), ec);
        if (ec) {
            stream_finished_.store(true);
            if (cancellation_token_ && !cancellation_token_->completed()) {
                cancellation_token_->cancel();
            }
            boost::system::error_code ignore_ec;
            socket_.close(ignore_ec);
            release_connection();
            return;
        }
    }

    if (is_final && !stream_finished_.exchange(true)) {
        // Send final chunk (0-length chunk to end stream) at most once
        std::string final_chunk = "0\r\n\r\n";
        net::write(socket_, net::buffer(final_chunk), ec);
        header_print("🔒 ", "Closing TCP connection (streaming, non-keep-alive)");
        socket_.shutdown(tcp::socket::shutdown_both, ec);
        socket_.cancel(ec);
        // Decrement connection counter for non-keep-alive connections
        release_connection();
    }
}

///@brief WebServer implementation
///@param port the port
WebServer::WebServer(std::string host, int port, bool cors) : acceptor(ioc, {net::ip::make_address(host), static_cast<unsigned short>(port)}), running(false), port(port), cors(cors) {
    origin_policy_.cors = cors;
    // On a loopback bind a request must also name a local Host (DNS rebinding); a server bound
    // to 0.0.0.0 or a LAN address is reached by names this process cannot know.
    origin_policy_.check_host = flm::http_guard::is_loopback_bind(host);
    boost::system::error_code ec;
    origin_policy_.hostname = net::ip::host_name(ec);
}

///@brief destructor
WebServer::~WebServer() {
    stop();
}

///@brief start
void WebServer::start() {
    running = true;
    do_accept();
    
    // Run the I/O service on multiple threads for better concurrency
    for (size_t i = 0; i < io_thread_count_; ++i) {
        io_threads_.emplace_back([this]() {
            // An exception escaping a handler unwinds run(); asio allows calling run() again, so
            // keep this thread serving instead of losing one I/O thread per escaped exception.
            for (;;) {
                try {
                    ioc.run();
                    break;  // run() returned normally: the server is stopping
                } catch (const std::exception& e) {
                    header_print("LOG", "Error in WebServer I/O thread (continuing): " + std::string(e.what()));
                } catch (...) {
                    header_print("LOG", "Unknown error in WebServer I/O thread (continuing)");
                }
                if (!running) break;
            }
        });
    }
    
    header_print("FLM", "WebServer started on port " + std::to_string(port) + " with " + std::to_string(io_thread_count_) + " I/O threads");
}

///@brief stop
void WebServer::stop() {
    running = false;
    ioc.stop();
    
    // Wait for all I/O threads to finish
    for (auto& thread : io_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    io_threads_.clear();
}

///@brief register active request
///@param request_id the request ID
///@param token the cancellation token
void WebServer::register_active_request(const std::string& request_id, std::shared_ptr<CancellationToken> token) {
    std::lock_guard<std::mutex> lock(active_requests_mutex_);
    active_requests_[request_id] = token;
}

///@brief unregister active request
///@param request_id the request ID
void WebServer::unregister_active_request(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(active_requests_mutex_);
    active_requests_.erase(request_id);
}

///@brief cancel request
///@param request_id the request ID
///@return true if request was found and cancelled
bool WebServer::cancel_request(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(active_requests_mutex_);
    auto it = active_requests_.find(request_id);
    if (it != active_requests_.end()) {
        it->second->cancel();
        active_requests_.erase(it);
        return true;
    }
    return false;
}

///@brief register handler
///@param method the method
///@param path the path
///@param handler the handler
void WebServer::register_handler(const std::string& method, const std::string& path, RequestHandler handler) {
    std::string key = method + " " + path;
    routes[key] = std::move(handler);
}

///@brief do accept
void WebServer::do_accept() {
    acceptor.async_accept(
        [this](beast::error_code ec, tcp::socket socket) {
            if (!ec) {
                // Check connection limit
                if (active_connections_.load() >= max_connections_) {
                    header_print("LOG", "Connection limit reached (" + std::to_string(max_connections_) + "), rejecting new connection with 503");
                    std::make_shared<ConnectionRejecter>(std::move(socket))->start(max_connections_, origin_policy_.cors && origin_policy_.allow_any);
                } else {
                    // Increment connection counter
                    active_connections_.fetch_add(1);
                    
                    // Create a new session for this connection
                    auto session = std::make_shared<HttpSession>(std::move(socket), *this);
                    session->start(cors);
                }
            }
            if (running) {
                do_accept();
            }
        });
}

///@brief process_next_npu_request Handles one queued NPU task at a time
void WebServer::process_next_npu_request() {
    npu_request_coordinator_.complete_current(
        [this](AcceleratorRequestCoordinator::Task task) {
            const auto remaining = npu_request_coordinator_.size();
            header_print("🟡 ", "Dequeuing accelerator request (" +
                std::to_string(remaining) + " remaining)...");
            net::post(ioc, std::move(task));
        },
        [] { AcceleratorAccessManager::release_npu_access(); },
        std::chrono::milliseconds(333));
}

///@brief handle request
///@param req the request
///@param res the response
///@param socket the socket
///@param session the session
///@return true if the response is deferred (queued), false otherwise
bool WebServer::handle_request(http::request<http::string_body>& req,
    http::response<http::string_body>& res,
    tcp::socket& socket,
    std::shared_ptr<HttpSession> session) {
    // Log request details
    std::cout << "================================================" << std::endl;
    header_print("⬇️ ", "Incoming Request: " << req.method_string());
    header_print("LOG", "Time stamp: " << get_current_time_string()); // hh:mm:ss mm:dd:yyyy
    // user text stays out of the log unless FLM_LOG_CONTENT asks for it: no query, and a summary of the body
    const bool log_content = flm::log_privacy::content_enabled();
    header_print("LOG", "Target: " << json_log::target(std::string_view(req.target().data(), req.target().size()), log_content));
    header_print("LOG", "Version: " << req.version());
    header_print("LOG", "Keep-Alive: " << req.keep_alive());
    json request_json_log;
    bool is_json = false;
    try {
        if (!req.body().empty()) {
            std::string content_type = std::string(req[http::field::content_type]);

            if (content_type.find("application/json") != std::string::npos) {
                // Decide on the Content-Type, not on whether this log-only parse succeeds:
                // a malformed body must still reach the parse below and get its 400.
                is_json = true;
                if (log_content) {
                    // cut long strings (image / audio data) as they are read: never logged, nor copied, whole
                    brief_print_message_request(json_log::parse(req.body()));
                } else {
                    header_print("LOG", "Body (" << req.body().size() << " bytes; FLM_LOG_CONTENT=1 shows the text): "
                                            << json_log::summary(req.body()));
                }
            }
            else if (content_type.find("multipart/form-data") != std::string::npos) {
                // print some request info 
            }
        }
    }
    catch (const std::exception& e) {
        header_print("LOG", "Error parsing request body: " + json_log::what(e, log_content));
    }

    // Route lookup
    // Routes match on the path alone: "/v1/models?x=1" and "/v1/models/" are "/v1/models".
    const std::string path = flm::http_guard::route_path(std::string_view(req.target().data(), req.target().size()));
    std::string key = std::string(req.method_string()) + " " + path;
    auto it = routes.find(key);
    if (it == routes.end()) {
        // No route: respond 404 synchronously in the path's wire protocol.
        apply_error_response(res, flm::error_model::protocol_for_path(path),
                             flm::error_model::ErrorValue{404, "not_found_error", "Not Found"});
        return false;
    }

    const auto route_proto = flm::error_model::protocol_for_path(path);
    // Decide if this handler needs exclusive NPU access.
    bool needs_npu = requires_npu_access(std::string(req.method_string()), path);

    // Store stable pointers for deferred execution to avoid reference lifetime issues.
    auto* req_ptr = &req;
    auto* res_ptr = &res;

    // Define a task lambda with is_deferred flag
    auto process_task = [this, it, route_proto, req_ptr, res_ptr, session, needs_npu, key, path, is_json](bool is_deferred) {
        auto& req_ref = *req_ptr;
        auto& res_ref = *res_ptr;
        AcceleratorRequestCompletionGuard completion([this, needs_npu] {
            if (needs_npu) process_next_npu_request();
        });

        // Parse JSON request body
        json request_json;
        try {
            if (!req_ref.body().empty()) {
                if(is_json && !reads_raw_body(std::string(req_ref.method_string()), path))
                    request_json = json::parse(req_ref.body());
                //else 
                    //do some parse to the MultiPart request?
                    //std::map<std::string, MultipartPart> parts = parse_multipart(req_ref);
            }
        }
        catch (const std::exception& e) {
            header_print("LOG", "Invalid JSON request body: " + json_log::what(e, flm::log_privacy::content_enabled()));
            apply_error_response(res_ref, route_proto,
                                 flm::error_model::ErrorValue{400, "invalid_request_error", "Invalid JSON: " + json_log::what(e)});

            // Only write from callback when deferred
            if (is_deferred && session) session->write_response_from_callback();
            return;
        }

        auto cancellation_token = std::make_shared<CancellationToken>(session);
        session->set_cancellation_token(cancellation_token);

        std::string request_id;
        if (request_json.is_object() && request_json.contains("request_id") &&
            request_json["request_id"].is_string()) {
            request_id = request_json["request_id"].get<std::string>();
        }
        else {
            static std::atomic<int> counter{ 0 };
            request_id = "req_" + std::to_string(counter.fetch_add(1));
        }

        register_active_request(request_id, cancellation_token);

        // catch is_deferred
        auto send_response = [res_ptr, session, this, route_proto, request_id, needs_npu, is_deferred, cancellation_token](const json& response_data) {
            if (session && session->is_streaming()) {
                cancellation_token->complete();
                unregister_active_request(request_id);
                if (!session->stream_finished()) {
                    const auto err = flm::error_model::extract_error(response_data)
                        .value_or(flm::error_model::ErrorValue{500, "server_error", "Internal server error"});
                    session->abort_open_stream(route_proto, err);
                }
                return;
            }
            // Prepare the payload before claiming the session response slot so a throw here
            // cannot lock out a subsequent error reply.
            http::response<http::string_body> staged;
            staged.version(res_ptr->version());
            staged.keep_alive(res_ptr->keep_alive());
            // A handler that owns its status (System One: 422 {"detail": ...}) wraps its reply as
            // {"__http_status": code, "__http_body": body}.
            if (response_data.is_object() && response_data.contains("__http_status")) {
                staged.result(static_cast<http::status>(response_data["__http_status"].get<int>()));
                staged.body() = flm::error_model::dump_json(response_data["__http_body"]);
                staged.set(http::field::content_type, "application/json");
                staged.prepare_payload();
            }
            else if (const auto err = flm::error_model::extract_error(response_data)) {
                apply_error_response(staged, route_proto, *err);
            }
            else if (response_data.is_string()) {
                // A plain-text body: the text, srt and vtt transcription formats.
                staged.result(http::status::ok);
                staged.body() = response_data.get<std::string>();
                staged.set(http::field::content_type, "text/plain; charset=utf-8");
                staged.prepare_payload();
            }
            else {
                staged.result(http::status::ok);
                staged.body() = flm::error_model::dump_json(response_data);
                staged.set(http::field::content_type, "application/json");
                staged.prepare_payload();
            }
            if (session && !session->try_begin_response()) {
                cancellation_token->complete();
                unregister_active_request(request_id);
                return;
            }
            *res_ptr = std::move(staged);
            cancellation_token->complete();
            unregister_active_request(request_id);

            if (is_deferred && session) {
                session->write_response_from_callback();
            }
        };

        auto send_streaming_response = [session, this, request_id, needs_npu, cancellation_token](const json& data, bool is_final) {
            if (is_final) {
                cancellation_token->complete();
            }

            if (session) {
                session->write_streaming_response(data, is_final);
            }
            if (is_final) {
                unregister_active_request(request_id);
            }
        };

        auto finish_with_error = [&](const flm::error_model::ErrorValue& err) {
            cancellation_token->complete();
            unregister_active_request(request_id);
            if (session && session->is_streaming()) {
                session->abort_open_stream(route_proto, err);
                return;
            }
            if (session && !session->try_begin_response()) {
                return;
            }
            apply_error_response(res_ref, route_proto, err);
            if (is_deferred && session) {
                session->write_response_from_callback();
            }
        };

        try {
            it->second(req_ref, send_response, send_streaming_response, session, cancellation_token);
        }
        catch (const json::parse_error& e) {
            // A route that parses the body itself (any Content-Type) got a malformed one: 400.
            // A parse error on anything else (model output, a file) stays a 500.
            const bool bad_body = !json::accept(req_ref.body());
            finish_with_error(bad_body
                ? flm::error_model::ErrorValue{400, "invalid_request_error", "Invalid JSON: " + json_log::what(e)}
                : flm::error_model::ErrorValue{500, "server_error", std::string("Handler exception: ") + e.what()});
        }
        catch (const ModelRequestError& e) {
            flm::error_model::ErrorValue err;
            err.status = e.http_code();
            err.type = flm::error_model::normalize_type(e.type(), e.http_code());
            err.message = e.what();
            err.param = e.param();
            err.session_cleared = e.session_cleared();
            finish_with_error(err);
        }
        catch (const std::exception& e) {
            auto err = flm::error_model::exception_error(e);
            if (err.status == 500) err.message = "Handler exception: " + err.message;
            finish_with_error(err);
        }
        catch (...) {
            finish_with_error(flm::error_model::ErrorValue{500, "server_error", "Unknown handler exception"});
        }
    }; // --- End of process_task lambda ---


    // --- Logic to run or queue the task ---

    if (!needs_npu) {
        process_task(false); //  return false
        return false;
    }

    const auto admission = npu_request_coordinator_.admit([this, process_task]() { process_task(true); },
                                                          [] { return AcceleratorAccessManager::try_acquire_npu_access(); });
    if (admission == AcceleratorRequestCoordinator::Admission::run_now) {
        header_print("🟢 ", "Accelerator locked");
        process_task(false); //  return false
        return false;
    }

    if (admission == AcceleratorRequestCoordinator::Admission::queue_full) {
        apply_error_response(res, route_proto, flm::error_model::ErrorValue{
            503,
            "service_unavailable_error",
            "NPU is in use and request queue is full (limit: " +
                std::to_string(npu_request_coordinator_.capacity()) +
                "). Please try again later."
        });
        header_print("🚫 ", "Accelerator busy and queue full, request denied: " + key);
        return false;
    }

    header_print("🕒 ", "Accelerator busy, request queued (" +
        std::to_string(npu_request_coordinator_.size()) + "/" +
        std::to_string(npu_request_coordinator_.capacity()) + "): " + key);
    return true;
}

///@brief create lm server
///@param models the model list
///@param downloader the downloader
///@param default_tag the default tag
///@param port the port
///@return the server
std::unique_ptr<WebServer> create_lm_server(model_list& models, ModelDownloader& downloader, program_args_t& args,
                                            flm_rt::device* npu_device) {
    auto server = std::make_unique<WebServer>(args.host, args.port, args.cors);
    server->set_cors_origins(args.cors_origins);
    auto rest_handler = std::make_shared<RestHandler>(models, downloader, args, npu_device);
    
    // Register Ollama-compatible routes
    server->register_handler("POST", "/api/show",
        [rest_handler](const http::request<http::string_body>& req,
            std::function<void(const json&)> send_response,
            std::function<void(const json&, bool)> send_streaming_response,
            std::shared_ptr<HttpSession> session,
            std::shared_ptr<CancellationToken> cancellation_token) {
                json request_json;
                if (!req.body().empty()) {
                    request_json = json::parse(req.body());
                }
                rest_handler->handle_show(request_json, send_response, send_streaming_response);
        });

    server->register_handler("POST", "/api/generate", 
        [rest_handler](const http::request<http::string_body>& req, 
                      std::function<void(const json&)> send_response,
                      std::function<void(const json&, bool)> send_streaming_response,
                      std::shared_ptr<HttpSession> session,
                      std::shared_ptr<CancellationToken> cancellation_token) {
            json request_json;
            if (!req.body().empty()) {
                request_json = json::parse(req.body());
            }
            rest_handler->handle_generate(request_json, send_response, send_streaming_response, cancellation_token);
        });
    
    server->register_handler("POST", "/api/chat",
        [rest_handler](const http::request<http::string_body>& req,
                      std::function<void(const json&)> send_response,
                      std::function<void(const json&, bool)> send_streaming_response,
                      std::shared_ptr<HttpSession> session,
                      std::shared_ptr<CancellationToken> cancellation_token) {
            json request_json;
            if (!req.body().empty()) {
                request_json = json::parse(req.body());
            }
            rest_handler->handle_chat(request_json, send_response, send_streaming_response, cancellation_token);
        });

    server->register_handler("GET", "/api/ps",
        [rest_handler](const http::request<http::string_body>& req,
                      std::function<void(const json&)> send_response,
                      std::function<void(const json&, bool)> send_streaming_response,
                      std::shared_ptr<HttpSession> session,
                      std::shared_ptr<CancellationToken> cancellation_token) {
            json request_json;
            rest_handler->handle_ps(request_json, send_response, send_streaming_response);
        });

    server->register_handler("POST", "/api/embeddings",
        [rest_handler](const http::request<http::string_body>& req,
                      std::function<void(const json&)> send_response,
                      std::function<void(const json&, bool)> send_streaming_response,
                      std::shared_ptr<HttpSession> session,
                      std::shared_ptr<CancellationToken> cancellation_token) {
            json request_json;
            if (!req.body().empty()) {
                request_json = json::parse(req.body());
            }
            rest_handler->handle_embeddings(request_json, send_response, send_streaming_response,
                    [&] { return cancellation_token && cancellation_token->cancelled(); });
        });
    
    server->register_handler("GET", "/api/tags",
        [rest_handler](const http::request<http::string_body>& req,
                      std::function<void(const json&)> send_response,
                      std::function<void(const json&, bool)> send_streaming_response,
                      std::shared_ptr<HttpSession> session,
                      std::shared_ptr<CancellationToken> cancellation_token) {
            json request_json;
            rest_handler->handle_models(request_json, send_response, send_streaming_response);
        });
    
    server->register_handler("GET", "/api/version",
        [rest_handler](const http::request<http::string_body>& req,
                      std::function<void(const json&)> send_response,
                      std::function<void(const json&, bool)> send_streaming_response,
                      std::shared_ptr<HttpSession> session,
                      std::shared_ptr<CancellationToken> cancellation_token) {
            json request_json;
            rest_handler->handle_version(request_json, send_response, send_streaming_response);
        });
    
    // Add NPU status endpoint
    server->register_handler("GET", "/api/npu/status",
        [](const http::request<http::string_body>& req,
           std::function<void(const json&)> send_response,
           std::function<void(const json&, bool)> send_streaming_response,
           std::shared_ptr<HttpSession> session,
           std::shared_ptr<CancellationToken> cancellation_token) {
            json response = {
                {"npu_available", AcceleratorAccessManager::is_npu_available()},
                {"active_requests", AcceleratorAccessManager::get_active_npu_requests()},
                {"message", AcceleratorAccessManager::is_npu_available() ? "NPU is available" : "NPU is currently in use"}
            };
            send_response(response);
        });
    
    // Add other endpoints...
    server->register_handler("POST", "/api/pull",
        [rest_handler](const http::request<http::string_body>& req,
                      std::function<void(const json&)> send_response,
                      std::function<void(const json&, bool)> send_streaming_response,
                      std::shared_ptr<HttpSession> session,
                      std::shared_ptr<CancellationToken> cancellation_token) {
            json request_json;
            if (!req.body().empty()) {
                request_json = json::parse(req.body());
            }
            rest_handler->handle_pull(request_json, send_response, send_streaming_response);
        });
    
    // Add OpenAI endpoints
    server->register_handler("GET", "/v1/version",
        [rest_handler](const http::request<http::string_body>& req,
            std::function<void(const json&)> send_response,
            std::function<void(const json&, bool)> send_streaming_response,
            std::shared_ptr<HttpSession> session,
            std::shared_ptr<CancellationToken> cancellation_token) {
                json request_json;
                rest_handler->handle_version(request_json, send_response, send_streaming_response);
        });

    server->register_handler("GET", "/v1/models",
        [rest_handler](const http::request<http::string_body>& req,
            std::function<void(const json&)> send_response,
            std::function<void(const json&, bool)> send_streaming_response,
            std::shared_ptr<HttpSession> session,
            std::shared_ptr<CancellationToken> cancellation_token) {
                json request_json;
                rest_handler->handle_models_openai(request_json, send_response, send_streaming_response);
        });

    server->register_handler("POST", "/v1/embeddings",
        [rest_handler](const http::request<http::string_body>& req,
            std::function<void(const json&)> send_response,
            std::function<void(const json&, bool)> send_streaming_response,
            std::shared_ptr<HttpSession> session,
            std::shared_ptr<CancellationToken> cancellation_token) {
                json request_json;
                if (!req.body().empty()) {
                    request_json = json::parse(req.body());
                }
                rest_handler->handle_embeddings(request_json, send_response, send_streaming_response,
                    [&] { return cancellation_token && cancellation_token->cancelled(); });
        });

    // System One decisions (d3:lite via --decision); /v1/decisions is the alias vLLM Semantic
    // Router also accepts. The handler parses the raw body itself.
    for (const char* path : {"/v1/systemone", "/v1/decisions"}) {
        server->register_handler("POST", path,
            [rest_handler](const http::request<http::string_body>& req,
                std::function<void(const json&)> send_response,
                std::function<void(const json&, bool)> send_streaming_response,
                std::shared_ptr<HttpSession> session,
                std::shared_ptr<CancellationToken> cancellation_token) {
                    rest_handler->handle_systemone(req.body(), send_response);
            });
    }

    server->register_handler("GET", "/v1/systemone/models",
        [rest_handler](const http::request<http::string_body>& req,
            std::function<void(const json&)> send_response,
            std::function<void(const json&, bool)> send_streaming_response,
            std::shared_ptr<HttpSession> session,
            std::shared_ptr<CancellationToken> cancellation_token) {
                rest_handler->handle_systemone_models(send_response);
        });

    server->register_handler("POST", "/v1/chat/completions",
        [rest_handler](const http::request<http::string_body>& req,
                      std::function<void(const json&)> send_response,
                      std::function<void(const json&, bool)> send_streaming_response,
                      std::shared_ptr<HttpSession> session,
                      std::shared_ptr<CancellationToken> cancellation_token) {
            json request_json;
            if (!req.body().empty()) {
                request_json = json::parse(req.body());
            }
            rest_handler->handle_openai_chat_completion(request_json, send_response, send_streaming_response, cancellation_token);
        });
    
    server->register_handler("POST", "/v1/responses",
        [rest_handler](const http::request<http::string_body>& req, auto send_response,
                       auto send_streaming_response, std::shared_ptr<HttpSession>, auto cancellation_token) {
            rest_handler->handle_openai_responses(json::parse(req.body()), std::move(send_response),
                                                 std::move(send_streaming_response), std::move(cancellation_token));
        });

    server->register_handler("POST", "/v1/audio/transcriptions",
        [rest_handler](const http::request<http::string_body>& req,
            std::function<void(const json&)> send_response,
            std::function<void(const json&, bool)> send_streaming_response,
            std::shared_ptr<HttpSession> session,
            std::shared_ptr<CancellationToken> cancellation_token) {
                std::map<std::string, MultipartPart> parts;
                try {
                    parts = parse_multipart(req);
                } catch (const std::exception& e) {   // not a multipart/form-data body: the client's mistake, not a server error
                    send_response(flm::error_model::make_error(
                        400, std::string("the request must be multipart/form-data with a \"file\" field: ") + e.what()));
                    return;
                }
                json request_json;
                request_json["model"] = parts["model"].content;
                request_json["file"] = parts["file"].content;
                for (const char* field : {"language", "prompt", "response_format", "stream", "chunk_size_ms", "lookahead_ms", "unfixed_token_num"}) {
                    auto it = parts.find(field);
                    if (it != parts.end()) {
                        request_json[field] = it->second.content;
                    }
                }
                rest_handler->handle_openai_audio_transcriptions(request_json, send_response, send_streaming_response, cancellation_token);
        });

    server->register_handler("POST", "/v1/completions",
        [rest_handler](const http::request<http::string_body>& req,
            std::function<void(const json&)> send_response,
            std::function<void(const json&, bool)> send_streaming_response,
            std::shared_ptr<HttpSession> session,
            std::shared_ptr<CancellationToken> cancellation_token) {
                json request_json;
                if (!req.body().empty()) {
                    request_json = json::parse(req.body());
                }
                rest_handler->handle_openai_completion(request_json, send_response, send_streaming_response, cancellation_token);
        });


    // Add cancel endpoint - capture server by raw pointer
    WebServer* server_ptr = server.get();
    server->register_handler("POST", "/api/cancel",
        [server_ptr](const http::request<http::string_body>& req,
                     std::function<void(const json&)> send_response,
                     std::function<void(const json&, bool)> send_streaming_response,
                     std::shared_ptr<HttpSession> session,
                     std::shared_ptr<CancellationToken> cancellation_token) {
            json request_json;
            if (!req.body().empty()) {
                request_json = json::parse(req.body());
            }
            
            if (!request_json.contains("request_id")) {
                json error_response = {{"error", "request_id is required"}};
                send_response(error_response);
                return;
            }
            
            if (auto bad = flm::http_guard::validate_cancel_request_id(request_json)) {
                send_response(*bad);
                return;
            }
            const std::string request_id = request_json.at("request_id").get<std::string>();

            // Try to cancel the request
            bool cancelled = server_ptr->cancel_request(request_id);
            
            json response;
            if (cancelled) {
                response["cancelled"] = true;
                response["message"] = "Request cancelled successfully";
            } else {
                response["cancelled"] = false;
                response["message"] = "Request not found or already completed";
            }
            send_response(response);
        });
    
    return server;
}