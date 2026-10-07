// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "iron/worker.hpp"

#if !defined(_WIN32)
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace flm::iron {
namespace {

constexpr std::uint32_t kMaxHeader = 1 << 20;

std::string PythonExecutable() {
    const char* value = std::getenv("FLM_IRON_PYTHON");
    return value && *value ? value : "python3";
}

std::string WorkerModule() {
    const char* value = std::getenv("FLM_IRON_WORKER_MODULE");
    return value && *value ? value : "flm_iron.worker";
}

}  // namespace

Worker::Worker(const std::filesystem::path& model_path,
               std::uint32_t context_length, std::string model) {
#if defined(_WIN32)
    throw std::runtime_error("The IRON backend requires POSIX process support");
#else
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
        throw std::runtime_error("Cannot create the IRON worker socket");
    }
    process_ = fork();
    if (process_ < 0) throw std::runtime_error("Cannot start the IRON worker");
    if (process_ == 0) {
        dup2(sockets[1], STDIN_FILENO);
        dup2(sockets[1], STDOUT_FILENO);
        close(sockets[0]);
        close(sockets[1]);
        const std::string python = PythonExecutable();
        const std::string module = WorkerModule();
        execlp(python.c_str(), python.c_str(), "-m", module.c_str(),
               static_cast<char*>(nullptr));
        _exit(127);
    }
    close(sockets[1]);
    input_ = output_ = sockets[0];
    try {
        request({{"command", model == "embedding" ? "load_embedding" : "load_llama"},
                 {"model_path", model_path.string()},
                 {"context_length", context_length}});
    } catch (...) {
        close(input_);
        waitpid(process_, nullptr, 0);
        input_ = output_ = process_ = -1;
        throw;
    }
#endif
}

Worker::~Worker() {
#if !defined(_WIN32)
    if (input_ >= 0) {
        try { request({{"command", "close"}}); } catch (...) {}
        close(input_);
    }
    if (process_ > 0) waitpid(process_, nullptr, 0);
#endif
}

void Worker::write_all(const void* data, std::size_t size) {
#if !defined(_WIN32)
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    while (size) {
        const ssize_t written = send(input_, cursor, size, MSG_NOSIGNAL);
        if (written <= 0) throw std::runtime_error("IRON worker write failed");
        cursor += written;
        size -= static_cast<std::size_t>(written);
    }
#endif
}

void Worker::read_all(void* data, std::size_t size) {
#if !defined(_WIN32)
    auto* cursor = static_cast<std::uint8_t*>(data);
    while (size) {
        const ssize_t received = read(output_, cursor, size);
        if (received <= 0) throw std::runtime_error("IRON worker exited");
        cursor += received;
        size -= static_cast<std::size_t>(received);
    }
#endif
}

nlohmann::json Worker::request(const nlohmann::json& command,
                               const void* payload,
                               std::size_t payload_size) {
    try {
        const std::string header = command.dump();
        const std::uint32_t header_size = static_cast<std::uint32_t>(header.size());
        write_all(&header_size, sizeof(header_size));
        write_all(header.data(), header.size());
        if (payload_size) write_all(payload, payload_size);

        std::uint32_t response_size = 0;
        read_all(&response_size, sizeof(response_size));
        if (response_size == 0 || response_size > kMaxHeader) {
            throw std::runtime_error("IRON worker returned an invalid header");
        }
        std::string response(response_size, '\0');
        read_all(response.data(), response.size());
        auto result = nlohmann::json::parse(response);
        if (!result.value("ok", false)) {
            throw std::runtime_error(result.value("error", "IRON worker failed"));
        }
        return result;
    } catch (...) {
        poisoned_ = true;
        throw;
    }
}

std::vector<std::uint16_t> Worker::logits(const std::vector<int>& tokens) {
    auto response = request({{"command", "logits"}, {"count", tokens.size()}},
                            tokens.data(), tokens.size() * sizeof(int));
    const std::size_t count = response.at("count").get<std::size_t>();
    std::vector<std::uint16_t> values(count);
    try {
        read_all(values.data(), values.size() * sizeof(values[0]));
    } catch (...) {
        poisoned_ = true;
        throw;
    }
    return values;
}

void Worker::reset() { request({{"command", "reset"}}); }

std::vector<float> Worker::embed(const std::string& text,
                                 const std::string& task,
                                 std::uint32_t dimensions) {
    auto response = request({{"command", "embed"},
                             {"task", task},
                             {"dimensions", dimensions},
                             {"size", text.size()}},
                            text.data(), text.size());
    const std::size_t count = response.at("count").get<std::size_t>();
    std::vector<float> values(count);
    try {
        read_all(values.data(), values.size() * sizeof(values[0]));
    } catch (...) {
        poisoned_ = true;
        throw;
    }
    return values;
}

}  // namespace flm::iron