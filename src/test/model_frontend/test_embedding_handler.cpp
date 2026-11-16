/// \file test_embedding_handler.cpp
/// \brief Exercise the production embedding request handler without a device or network.
#include "rest_handler.hpp"
#include "../phi4_core/test_support.hpp"
#include <fstream>
#include <future>
#include <map>

// Only the constructor and downloader are substituted. The real handler,
// catalog, factory and C plugin execute every request below.
RestHandler::RestHandler(model_list& models, ModelDownloader& downloads,
                         program_args_t& args, flm_rt::device* device)
    : npu_device_inst(device), embeddings_enabled(args.embed), supported_models(models), downloader(downloads),
      modelscope(args.modelscope), preemption(args.preemption) {}
RestHandler::~RestHandler() = default;
AutoDecisionModel::~AutoDecisionModel() = default;
ImageReader::~ImageReader() = default;
download_utils::CurlInitializer::CurlInitializer() = default;
download_utils::CurlInitializer::~CurlInitializer() = default;

namespace {
using Status = ModelDownloader::ModelStatus;
std::map<std::string, Status> statuses;
int pulls = 0;
bool pull_succeeds = false;
bool npu_available = true;
}
flm_rt::device* utils::require_npu_device(flm_rt::device* device, const std::string&) {
    static flm_rt::device fake_device;
    if (!npu_available) throw std::runtime_error(flm::backend::kNoNpuDevice);
    return device ? device : &fake_device;
}
ModelDownloader::ModelDownloader(model_list& models) : supported_models(models) {}
ModelDownloader::ModelStatus ModelDownloader::is_model_downloaded(
    const std::string& tag, bool, bool fast_check) {
    TEST_REQUIRE(fast_check);
    return statuses.at(tag);
}
bool ModelDownloader::pull_model(const std::string&, bool, bool) {
    ++pulls;
    return pull_succeeds;
}

void test_embedding_handler() {
    auto& registry = flm::frontend::FrontendRegistry::instance();
    registry.load_extension(FLM_TEST_EMBEDDING_MANIFEST, {});
    nlohmann::json entries;
    for (const char* name : {"first", "second", "missing", "incompatible", "fail-load", "unsupported-load", "invalid-load"}) {
        entries[name] = {{"1", {{"name", name}, {"model_kind", "embedding"},
            {"details", {{"family", "test-embedding"}}}, {"url", "https://example.invalid"},
            {"files", {"config.json"}}, {"flm_min_version", "1.0.0"}}}};
        statuses[std::string(name) + ":1"] = Status::Ready;
    }
    entries["generation"] = entries["first"];
    entries["generation"]["1"]["model_kind"] = "generation";
    entries["generation"]["1"]["details"]["family"] = "llama3";
    statuses["missing:1"] = Status::Missing;
    statuses["incompatible:1"] = Status::Incompatible;
    const auto directory = std::filesystem::path(FLM_TEST_EMBEDDING_MANIFEST).parent_path();
    std::string path = (directory / "handler-models.json").string(), root = directory.string();
    std::ofstream(path) << nlohmann::json({{"model_path", "models"}, {"models", entries}});
    model_list models(path, root);
    TEST_REQUIRE(models.is_model_supported("generation:1"));
    bool found_embedding = false;
    const auto catalog = models.get_all_models_openai();
    for (const auto& entry : catalog.at("data")) {
        if (entry.at("id") != "first:1") continue;
        found_embedding = true;
        TEST_REQUIRE(entry.at("owned_by") == "FastFlowLM");
        TEST_REQUIRE(!entry.contains("backend") && !entry.contains("device"));
    }
    TEST_REQUIRE(found_embedding);
    ModelDownloader downloads(models);
    program_args_t args{};
    RestHandler disabled(models, downloads, args, nullptr);
    disabled.handle_embeddings({{"model", "missing:1"}, {"input", "hello"}},
        [](const json& response) { TEST_REQUIRE(response.at("error").at("code") == 400);
            TEST_REQUIRE(response.at("error").at("type") == "invalid_request_error"); }, {});
    TEST_REQUIRE(pulls == 0);
    args.embed = true;
    RestHandler server(models, downloads, args, nullptr);
    const auto request = [&](const std::string& tag, json input = "hello", json options = json::object()) {
        json response;
        options["model"] = tag;
        options["input"] = input;
        server.handle_embeddings(options,
            [&](const json& value) { response = value; }, {});
        return response;
    };
    const auto check = [&](const std::string& tag, float marker) {
        const auto response = request(tag);
        TEST_REQUIRE(response.at("data").at(0).at("embedding") == json({5, 3, marker}));
    };
    npu_available = false;
    auto no_device = request("first:1");
    npu_available = true;
    TEST_REQUIRE(no_device.contains("error"));
    TEST_REQUIRE(no_device["error"]["message"] == flm::backend::kNoNpuDevice);
    for (const json& bad : {json(nullptr), json(42), json::array({"ok", false})})
        TEST_REQUIRE(request("missing:1", bad).at("error").at("code") == 400);
    TEST_REQUIRE(request("missing:1", "hello", {{"task", 42}}).at("error").at("code") == 400);
    TEST_REQUIRE(request("missing:1", "hello", {{"dimensions", 0}}).at("error").at("code") == 400);
    TEST_REQUIRE(pulls == 0);
    check("first:1", 0.5f);
    TEST_REQUIRE(request("first", "hello", {{"task", nullptr}, {"dimensions", nullptr}})
                     .at("data").at(0).at("embedding") == json({5, 3, 0.5f}));
    TEST_REQUIRE(request("first", "cancel").at("error").at("code") == 499);
    TEST_REQUIRE(request("first", "hello", {{"task", "provider-task"}}).at("data").at(0).at("embedding").at(1) == 4);
    TEST_REQUIRE(request("first", "hello", {{"task", "unknown"}}).at("error").at("code") == 400);
    TEST_REQUIRE(request("first", "hello", {{"task", 42}}).at("error").at("code") == 400);
    check("first", 0.5f);
    check("Ollama/first", 0.5f);
    for (const char* tag : {"missing:1", "incompatible:1", "absent:1", "Ollama/absent", "generation:1"}) {
        const auto error = request(tag).at("error");
        TEST_REQUIRE(error.at("code") == (std::string(tag) == "missing:1" ? 500 : 400));
        TEST_REQUIRE(error.at("type") == (std::string(tag) == "missing:1" ? "server_error" : "invalid_request_error"));
        check("first", 0.5f);
    }
    TEST_REQUIRE(pulls == 1);
    for (const char* tag : {"fail-load:1", "unsupported-load:1", "invalid-load:1"}) {
        const auto error = request(tag).at("error");
        TEST_REQUIRE(error.at("code") == 500);
        TEST_REQUIRE(error.at("type") == "server_error");
        check("first", 0.5f); // Retry after releasing the old model for a failed load.
    }
    statuses["second:1"] = Status::Outdated;
    TEST_REQUIRE(request("second").contains("error"));
    check("first", 0.5f);
    pull_succeeds = true;
    check("Ollama/second:1", 2.0f);
    TEST_REQUIRE(pulls == 3);
    statuses["second:1"] = Status::Ready;
    const auto batch = request("second", json::array({"hi", "hello"}));
    TEST_REQUIRE(batch.at("data").size() == 2);
    TEST_REQUIRE(batch.at("usage").at("prompt_tokens") == 7);
    TEST_REQUIRE(request("second", {{"text", "hi"}, {"title", "abc"}},
                         {{"encoding_format", "float"}, {"user", "client"}})
                     .at("data").at(0).at("embedding").at(0) == 5);
    for (const auto& dimensions : {json(0), json(-1), json(4), json(1.5), json("2")})
        TEST_REQUIRE(request("second", "hi", {{"dimensions", dimensions}}).at("error").at("code") == 400);
    server.handle_embeddings({{"model", "second"}, {"input", "hi"}},
        [](const json& response) { TEST_REQUIRE(response.at("error").at("code") == 499); },
        {}, [] { return true; });
    TEST_REQUIRE(batch.at("data").at(0).at("embedding").at(0) == 2);
    TEST_REQUIRE(request("second", 42).contains("error"));
    TEST_REQUIRE(request("second", json::array()).at("error").at("code") == 400);
    for (const auto& malformed : {json::object(), json{{"model", 42}}, json{{"model", "second"}}}) {
        server.handle_embeddings(malformed, [](const json& response) {
            TEST_REQUIRE(response.at("error").at("code") == 400);
            TEST_REQUIRE(response.at("error").contains("param"));
        }, {});
    }
    auto first = std::async(std::launch::async, [&] {
        for (int i = 0; i < 20; ++i) check("first", 0.5f);
    });
    auto second = std::async(std::launch::async, [&] {
        for (int i = 0; i < 20; ++i) check("second", 2.0f);
    });
    first.get();
    second.get();
    registry.unload_extensions();
}
