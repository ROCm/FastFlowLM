// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "iron/worker.hpp"

#include "../phi4_core/test_support.hpp"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

namespace {

void SetEnvironment(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

// The test executable doubles as the server, using the same launch arguments.
int ServeFake() {
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    auto read = [](void* data, std::size_t size) {
        TEST_REQUIRE(std::cin.read(static_cast<char*>(data), size));
    };
    auto send = [](const nlohmann::json& header,
                   const void* payload = nullptr, std::size_t size = 0) {
        const auto data = header.dump();
        const std::uint32_t length = data.size();
        std::cout.write(reinterpret_cast<const char*>(&length), sizeof(length));
        std::cout.write(data.data(), data.size());
        if (size) std::cout.write(static_cast<const char*>(payload), size);
        std::cout.flush();
    };
    std::string model;
    std::uint32_t length;
    while (std::cin.read(reinterpret_cast<char*>(&length), sizeof(length))) {
        std::string header(length, '\0');
        read(header.data(), header.size());
        const auto command = nlohmann::json::parse(header);
        const auto name = command.at("command").get<std::string>();
        if (name == "load_model") {
            model = command.at("model").get<std::string>();
            if (model == "llama") TEST_REQUIRE(command.at("context_length") == 32);
            else TEST_REQUIRE(!command.contains("context_length"));
            send({{"ok", true}});
            if (model == "dies") return 0;
        } else if (name == "logits") {
            std::vector<std::int32_t> tokens(command.at("count").get<std::size_t>());
            read(tokens.data(), tokens.size() * sizeof(tokens[0]));
            if (model == "bad_count") {
                send({{"ok", true}});
            } else if (model == "bad_header") {
                const std::uint32_t zero = 0;
                std::cout.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
                std::cout.flush();
            } else if (model == "error") {
                send({{"ok", false}, {"error", "model failed"}});
            } else if (model == "truncated") {
                send({{"ok", true}, {"count", 2}}, "x", 1);
                return 0;
            } else {
                const std::uint16_t values[] = {
                    static_cast<std::uint16_t>(tokens.back()),
                    static_cast<std::uint16_t>(tokens.size())};
                send({{"ok", true}, {"count", 2}}, values, sizeof(values));
            }
        } else if (name == "embed") {
            std::string text(command.at("size").get<std::size_t>(), '\0');
            read(text.data(), text.size());
            const float values[] = {static_cast<float>(text.size()),
                                    command.at("dimensions").get<float>()};
            send({{"ok", true}, {"count", 2}}, values, sizeof(values));
        } else {
            TEST_REQUIRE(name == "reset" || name == "close");
            send({{"ok", true}});
            if (name == "close") return 0;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        return argc == 3 && std::string(argv[1]) == "-m" &&
                       std::string(argv[2]) == "fake_server"
                   ? ServeFake() : 1;
    }
    SetEnvironment("FLM_IRON_WORKER_MODULE", "fake_server");

    flm::iron::Worker llama("/unused", "llama", 32);
    const auto logits = llama.logits(std::vector<int>{4, 10, 26});
    TEST_REQUIRE(logits == std::vector<std::uint16_t>({26, 3}));
    llama.reset();
    TEST_REQUIRE(!llama.poisoned());

    flm::iron::Worker embedding("/unused", "embedding_gemma2");
    const auto result = embedding.embed("hello", "query", 128);
    TEST_REQUIRE(result == std::vector<float>({5.0f, 128.0f}));

    for (const char* model : {"bad_count", "bad_header", "error", "truncated"}) {
        flm::iron::Worker broken("/unused", model);
        RequireThrows([&] { broken.logits({1}); });
        TEST_REQUIRE(broken.poisoned());
        RequireContains(RequireThrows([&] { broken.reset(); }), "reload");
    }

    bool pipe_failed = false;
    try {
        flm::iron::Worker exits("/unused", "dies");
        exits.reset();
    } catch (const std::runtime_error&) {
        pipe_failed = true;
    }
    TEST_REQUIRE(pipe_failed);

    bool failed = false;
    SetEnvironment("FLM_IRON_WORKER_MODULE", "missing_worker");
    try {
        flm::iron::Worker dead("/unused", "llama", 32);
    } catch (const std::runtime_error&) {
        failed = true;
    }
    TEST_REQUIRE(failed);
    return 0;
}
