// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "iron/worker.hpp"

#define BOOST_PROCESS_USE_STD_FS
#include <boost/process.hpp>

#if !defined(_WIN32)
#include <csignal>
#endif
#include <cstdlib>
#include <stdexcept>

namespace flm::iron {
namespace {

constexpr std::uint32_t kMaxHeader = 1 << 20;

std::string PythonExecutable() {
    const char* value = std::getenv("FLM_IRON_PYTHON");
    if (value && *value) return value;
#if defined(_WIN32)
    return "python";
#else
    return "python3";
#endif
}

std::string WorkerModule() {
    const char* value = std::getenv("FLM_IRON_WORKER_MODULE");
    return value && *value ? value : "iron.lm_server";
}

std::filesystem::path ResolveExecutable(const std::string& executable) {
    const std::filesystem::path path(executable);
    return path.has_parent_path() ? path : boost::process::search_path(path);
}

void IgnoreBrokenPipeSignal() {
#if !defined(_WIN32)
    static const bool ignored = std::signal(SIGPIPE, SIG_IGN) != SIG_ERR;
    if (!ignored) throw std::runtime_error("Cannot ignore SIGPIPE");
#endif
}

}  // namespace

struct Worker::Impl {
    boost::process::opstream input;
    boost::process::ipstream output;
    boost::process::child process;

    Impl(const std::string& python, const std::string& module)
        : process(ResolveExecutable(python),
                  boost::process::args({"-m", module}),
                  boost::process::std_in < input,
                  boost::process::std_out > output) {}

    void close_pipes() noexcept {
        try { input.pipe().close(); } catch (...) {}
        try { output.pipe().close(); } catch (...) {}
    }

    void stop() noexcept {
        close_pipes();
        std::error_code error;
        if (process.running(error)) process.terminate(error);
        process.wait(error);
    }
};

Worker::Worker(const std::filesystem::path& model_path, std::string model,
               std::uint32_t context_length) {
    IgnoreBrokenPipeSignal();
    impl_ = std::make_unique<Impl>(PythonExecutable(), WorkerModule());
    try {
        nlohmann::json command = {{"command", "load_model"},
                                  {"model", std::move(model)},
                                  {"model_path", model_path.string()}};
        if (context_length) command["context_length"] = context_length;
        request(command);
    } catch (...) {
        impl_->stop();
        throw;
    }
}

Worker::~Worker() {
    if (impl_) {
        try {
            request({{"command", "close"}});
        } catch (...) {
            impl_->stop();
            return;
        }
        impl_->close_pipes();
        std::error_code error;
        impl_->process.wait(error);
    }
}

void Worker::write_all(const void* data, std::size_t size) {
    impl_->input.write(static_cast<const char*>(data), size);
    if (!impl_->input) throw std::runtime_error("IRON worker write failed");
}

void Worker::read_all(void* data, std::size_t size) {
    impl_->output.read(static_cast<char*>(data), size);
    if (!impl_->output) throw std::runtime_error("IRON worker exited");
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
        impl_->input.flush();
        if (!impl_->input) throw std::runtime_error("IRON worker write failed");

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