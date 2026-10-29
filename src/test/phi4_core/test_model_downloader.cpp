#include "download_model.hpp"
#include "model_downloader.hpp"
#include "test_support.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;

// The corelib model is packaged differently from its FastFlowLM namesake --
// a Q8_0 GGUF against an NPU2/Q4NX build -- so it is a separate tag, and the
// "-core" suffix is what routes it to the corelib kernels.
constexpr const char* kCoreTag = "phi4-mini-it-core:4b";
constexpr const char* kFlmTag = "phi4-mini-it:4b";
// The tag is the model_info.json key and the directory name, so all three read
// alike and nothing has to redirect between them.
constexpr const char* kCoreModelInfoKey = "phi4-mini-it-core:4b";
constexpr const char* kCoreDirName = "phi4-mini-it-core";
constexpr const char* kUnslothRevision = "78eb92a46fc37e6b524df991ed9aca9bc6aa7b80";
constexpr const char* kMicrosoftRevision = "cfbefacb99257ffa30c83adab238a50856ac3083";

/// \brief the shipped catalog entry as the machine that can run it resolves it
/// \param tag the model tag to resolve
/// \param platform the NPU generation to resolve it for
/// \note Goes through model_list so these tests exercise the real filtering
///       path rather than the raw JSON, with a corelib build's backend list --
///       a tag whose flow this build lacks, or whose silicon this is not, is
///       pruned.
nlohmann::json ResolvedModel(const char* tag, const char* platform) {
    std::string path = FLM_SOURCE_DIR "/model_list.json";
    std::string exe_dir = ".";
    model_list models(path, exe_dir, platform, {"flm", "core"});
    TEST_REQUIRE(models.is_model_supported(tag));
    return models.get_model_info(tag).second;
}

nlohmann::json ReadJson(const fs::path& path) {
    std::ifstream stream(path);
    TEST_REQUIRE(stream.is_open());
    return nlohmann::json::parse(stream);
}

void Write(const fs::path& path, std::string_view bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    TEST_REQUIRE(stream.good());
}

std::string Read(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::size_t CountOccurrences(std::string_view text, std::string_view needle) {
    std::size_t count = 0;
    for (std::size_t position = text.find(needle); position != std::string_view::npos;
         position = text.find(needle, position + needle.size())) {
        ++count;
    }
    return count;
}

fs::path TempDirectory(std::string_view name) {
    const auto path = fs::temp_directory_path() / ("flm-task5-" + std::string(name));
    std::error_code ignored;
    fs::remove_all(path, ignored);
    fs::create_directories(path);
    return path;
}

std::string FileUrl(const fs::path& path) {
    std::string value = fs::absolute(path).generic_string();
#ifdef _WIN32
    return "file:///" + value;
#else
    return "file://" + value;
#endif
}

void TestCoreCatalogHasExactlyFourFilesAndExpectedDirectoryName() {
    const auto model = ResolvedModel(kCoreTag, "dev04");
    const std::vector<std::string> expected = {
        "Phi-4-mini-instruct.Q8_0.gguf", "tokenizer.json",
        "tokenizer_config.json", "config.json"};
    TEST_REQUIRE(model.at("name") == kCoreDirName);
    // The tag is the key, so there is nothing to redirect.
    TEST_REQUIRE(!model.contains("model_info_key"));
    TEST_REQUIRE(std::string(kCoreTag) == kCoreModelInfoKey);
    TEST_REQUIRE(model.at("backend") == "core");
    TEST_REQUIRE(model.at("files").get<std::vector<std::string>>() == expected);
    TEST_REQUIRE(model.at("size").get<std::uint64_t>() == 4100140571ULL);
}

void TestGgufUrlContainsUnslothRevisionAndFilename() {
    const auto model = ResolvedModel(kCoreTag, "dev04");
    const auto source = resolve_file_source(model, "Phi-4-mini-instruct.Q8_0.gguf", false);
    TEST_REQUIRE(source.url == std::string("https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/") +
                                  kUnslothRevision + "/Phi-4-mini-instruct.Q8_0.gguf?download=true");
}

void TestThreeFrontendUrlsContainMicrosoftRevisionAndFilename() {
    const auto model = ResolvedModel(kCoreTag, "dev04");
    for (const std::string filename : {"tokenizer.json", "tokenizer_config.json", "config.json"}) {
        const auto source = resolve_file_source(model, filename, false);
        TEST_REQUIRE(source.url == std::string("https://huggingface.co/microsoft/Phi-4-mini-instruct/resolve/") +
                                      kMicrosoftRevision + "/" + filename + "?download=true");
    }
}

void TestExistingSingleSourceEntryKeepsItsCurrentUrl() {
    // The FastFlowLM phi4, which a corelib build still lists and still runs on
    // the flm kernels: nothing about the corelib tag may reach it.
    const auto model = ResolvedModel(kFlmTag, "aie2p");
    TEST_REQUIRE(model.at("backend") == "flm");
    TEST_REQUIRE(model.at("name") == "Phi4-mini-Instruct-NPU2");
    TEST_REQUIRE(!model.contains("file_sources"));
    TEST_REQUIRE(!model.contains("model_info_key"));
    const auto source = resolve_file_source(model, "config.json", false);
    TEST_REQUIRE(source.url ==
                 "https://huggingface.co/FastFlowLM/Phi4-mini-Instruct-NPU2/resolve/main/config.json?download=true");
}

void TestUnknownFileSourceKeyAndMissingUrlOrRevisionFail() {
    nlohmann::json model = {
        {"url", "https://example.invalid/base"},
        {"files", {"config.json"}},
        {"file_sources", {{"unknown.json", {{"url", "https://example.invalid/source"},
                                              {"revision", std::string(40, 'a')}}}}}};
    RequireContains(RequireThrows([&] { resolve_file_source(model, "config.json", false); }),
                    "unknown file_sources key");

    model["file_sources"] = {{"config.json", {{"revision", std::string(40, 'a')}}}};
    RequireContains(RequireThrows([&] { resolve_file_source(model, "config.json", false); }), "url");
    model["file_sources"] = {{"config.json", {{"url", "https://example.invalid/source"}}}};
    RequireContains(RequireThrows([&] { resolve_file_source(model, "config.json", false); }), "revision");
    model["file_sources"] = {{"config.json", {{"url", "https://example.invalid/source"},
                                               {"revision", "NOT-A-COMMIT"}}}};
    RequireContains(RequireThrows([&] { resolve_file_source(model, "config.json", false); }), "revision");
}

void TestActualCoreCatalogTreatsPinnedConfigWithoutFlmVersionAsCompatible() {
    const auto root = TempDirectory("actual-catalog-version");
    // Take the resolved entry and re-home it in a temp catalog, under the same
    // family: the tag is what says these are corelib artifacts.
    const auto model = ResolvedModel(kCoreTag, "dev04");
    const nlohmann::json catalog = {
        {"model_path", "models"},
        {"models", {{"phi4-mini-it-core", {{"4b", model}}}}}};
    const auto catalog_path = root / "model_list.json";
    Write(catalog_path, catalog.dump());
    std::string catalog_string = catalog_path.string();
    std::string root_string = root.string();
    // The tag names the corelib flow and the entry the next generation, so say
    // both are here.
    model_list models(catalog_string, root_string, "dev04", {"flm", "core"});
    const auto model_path = root / "models" / kCoreDirName;
    for (const auto& filename : model.at("files")) {
        Write(model_path / filename.get<std::string>(), "placeholder");
    }
    Write(model_path / "config.json",
          R"({"architectures":["Phi3ForCausalLM"],"model_type":"phi3"})");

    ModelDownloader downloader(models);
    TEST_REQUIRE(downloader.is_model_downloaded(kCoreTag, true, true) ==
                 ModelDownloader::ModelStatus::Ready);
}

void TestModelInfoHasExactSizeAndSha256ForEveryRequiredFile() {
    const auto all_info = ReadJson(FLM_SOURCE_DIR "/model_info.json");
    const auto& records = all_info.at(kCoreModelInfoKey);
    const std::vector<std::tuple<std::string, std::uint64_t, std::string>> expected = {
        {"Phi-4-mini-instruct.Q8_0.gguf", 4084611040ULL, "26188c6050d525376a88b04514c236c5e28a36730f1e936f2a00314212b7ba42"},
        {"tokenizer.json", 15524095ULL, "382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea"},
        {"tokenizer_config.json", 2932ULL, "9c9b6bc0c94d95f69f826c41069a3e8b387ac3ced89601d201886e99240ac9db"},
        {"config.json", 2504ULL, "ac65d86061d3d0d704ee2511fd0eb8713ef19eb6eedba17c3080a4165d5b933b"}};
    TEST_REQUIRE(records.size() == expected.size());
    std::uint64_t total = 0;
    for (const auto& [path, size, sha256] : expected) {
        const auto match = std::find_if(records.begin(), records.end(), [&](const auto& record) {
            return record.at("path") == path;
        });
        TEST_REQUIRE(match != records.end());
        TEST_REQUIRE(match->at("size").get<std::uint64_t>() == size);
        TEST_REQUIRE(match->at("sha256") == sha256);
        total += size;
    }
    TEST_REQUIRE(total == 4100140571ULL);
}

struct DownloaderFixture {
    fs::path root = TempDirectory("ready");
    fs::path catalog_path = root / "model_list.json";
    fs::path info_path = root / "model_info.json";
    std::string catalog_string;
    std::string root_string;
    model_list models;

    DownloaderFixture()
        : catalog_string(catalog_path.string()), root_string(root.string()), models() {
        const nlohmann::json catalog = {
            {"model_path", "models"},
            {"models", {{"test-model", {{"1b", {
                {"name", "test-model"}, {"url", "https://example.invalid/repo"},
                {"file_url", "https://example.invalid/api"}, {"flm_min_version", "1.0.3"},
                {"details", {{"family", "phi4"}}},
                // Present but empty: this is what marks the entry as
                // pinned-upstream, and the per-file lookup is expected to fail
                // on it rather than the entry silently going unverified.
                {"file_sources", nlohmann::json::object()},
                {"files", {"config.json", "a.bin", "b.bin", "c.bin"}}
            }}}}}}};
        const nlohmann::json info = {{"test-model:1b", {
            {{"path", "config.json"}, {"size", 57}, {"sha256", "b8bfba5e42c4cb0b8660ea39fec6fefafddc42fb6a0b17d472177fb7683b2290"}},
            {{"path", "a.bin"}, {"size", 5}, {"sha256", "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8"}},
            {{"path", "b.bin"}, {"size", 4}, {"sha256", "f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753"}},
            {{"path", "c.bin"}, {"size", 5}, {"sha256", "be9d587defa1f0c09ef49eb17e206983a5f8f8289e4281860bd0ee5a19592c67"}}
        }}};
        Write(catalog_path, catalog.dump());
        Write(info_path, info.dump());
#ifdef _WIN32
        _putenv_s("FLM_MODELINFO_PATH", info_path.string().c_str());
#else
        setenv("FLM_MODELINFO_PATH", info_path.string().c_str(), 1);
#endif
        models = model_list(catalog_string, root_string);
    }

    fs::path model_path() const { return root / "models" / "test-model"; }
    void WriteValidFiles() const {
        Write(model_path() / "config.json",
              R"({"architectures":["Phi3ForCausalLM"],"model_type":"phi3"})");
        Write(model_path() / "a.bin", "alpha");
        Write(model_path() / "b.bin", "beta");
        Write(model_path() / "c.bin", "gamma");
    }
};

void TestModelIsReadyOnlyWhenAllFourFinalFilesValidate() {
    DownloaderFixture fixture;
    fixture.WriteValidFiles();
    ModelDownloader downloader(fixture.models);
    TEST_REQUIRE(downloader.is_model_downloaded("test-model:1b") == ModelDownloader::ModelStatus::Ready);
    Write(fixture.model_path() / "b.bin", "BETA");
    TEST_REQUIRE(downloader.is_model_downloaded("test-model:1b") == ModelDownloader::ModelStatus::Missing);
}

void TestPartFileNeverMakesModelReady() {
    DownloaderFixture fixture;
    fixture.WriteValidFiles();
    fs::rename(fixture.model_path() / "c.bin", fixture.model_path() / "c.bin.part");
    ModelDownloader downloader(fixture.models);
    TEST_REQUIRE(downloader.is_model_downloaded("test-model:1b") == ModelDownloader::ModelStatus::Missing);
}

void TestLegacyReadyCheckDoesNotHashOrDeleteWeights() {
    const auto root = TempDirectory("legacy-ready");
    const auto catalog_path = root / "model_list.json";
    const auto info_path = root / "model_info.json";
    const nlohmann::json catalog = {
        {"model_path", "models"},
        {"models", {{"legacy-model", {{"1b", {
            {"name", "legacy-model"}, {"url", "https://example.invalid/repo"},
            {"file_url", "https://example.invalid/api"}, {"flm_min_version", "1.0.3"},
            {"files", {"config.json", "model.bin"}}
        }}}}}}};
    const nlohmann::json info = {{"legacy-model:1b", {
        {{"path", "config.json"}, {"size", 23}, {"oid", std::string(40, '0')}},
        {{"path", "model.bin"}, {"size", 8}, {"oid", std::string(40, '0')}}
    }}};
    Write(catalog_path, catalog.dump());
    Write(info_path, info.dump());
#ifdef _WIN32
    _putenv_s("FLM_MODELINFO_PATH", info_path.string().c_str());
#else
    setenv("FLM_MODELINFO_PATH", info_path.string().c_str(), 1);
#endif
    std::string catalog_string = catalog_path.string();
    std::string root_string = root.string();
    model_list models(catalog_string, root_string);
    const auto model_path = root / "models" / "legacy-model";
    Write(model_path / "config.json", R"({"flm_version":"1.0.3"})");
    Write(model_path / "model.bin", "bad-data");

    ModelDownloader downloader(models);
    TEST_REQUIRE(downloader.is_model_downloaded("legacy-model:1b") ==
                 ModelDownloader::ModelStatus::Ready);
    TEST_REQUIRE(Read(model_path / "model.bin") == "bad-data");

    Write(model_path / "config.json", R"({"model_type":"legacy"})");
    TEST_REQUIRE(downloader.is_model_downloaded("legacy-model:1b", true, true) ==
                 ModelDownloader::ModelStatus::Outdated);
}

void TestPullAndCheckRejectModelscopeBeforePinnedReadyStateChecks() {
    DownloaderFixture fixture;
    ModelDownloader downloader(fixture.models);
    std::ostringstream output;
    auto* previous = std::cout.rdbuf(output.rdbuf());
    const bool pull_ok = downloader.pull_model("test-model:1b", true);
    const bool check_ok = downloader.check_model("test-model:1b", true, true);
    std::cout.rdbuf(previous);

    TEST_REQUIRE(!pull_ok);
    TEST_REQUIRE(!check_ok);
    TEST_REQUIRE(CountOccurrences(output.str(),
        "pinned Hugging Face per-file sources are required") == 2);
}

void TestCheckHashesPinnedFilesExactlyOnce() {
    DownloaderFixture fixture;
    fixture.WriteValidFiles();
    ModelDownloader downloader(fixture.models);
    std::ostringstream output;
    auto* previous = std::cout.rdbuf(output.rdbuf());
    const bool ok = downloader.check_model("test-model:1b", false, false);
    std::cout.rdbuf(previous);

    TEST_REQUIRE(ok);
    TEST_REQUIRE(CountOccurrences(output.str(), "Checking file:") == 4);
}

void TestStartupStatusDoesNotRehashButCheckStillDoes() {
    // Re-hashing the 4 GB GGUF on every launch cost ~28 s, 62% of startup, and
    // buys nothing a pull-time verification has not already established. The
    // run/serve paths ask for status only; `flm check` remains the full check.
    DownloaderFixture fixture;
    fixture.WriteValidFiles();
    ModelDownloader downloader(fixture.models);

    std::ostringstream fast;
    auto* previous = std::cout.rdbuf(fast.rdbuf());
    const auto fast_status = downloader.is_model_downloaded("test-model:1b", false, true);
    std::cout.rdbuf(previous);
    TEST_REQUIRE(fast_status == ModelDownloader::ModelStatus::Ready);
    TEST_REQUIRE(CountOccurrences(fast.str(), "Checking file:") == 0);

    std::ostringstream full;
    previous = std::cout.rdbuf(full.rdbuf());
    const bool ok = downloader.check_model("test-model:1b", false, false);
    std::cout.rdbuf(previous);
    TEST_REQUIRE(ok);
    TEST_REQUIRE(CountOccurrences(full.str(), "Checking file:") == 4);
}

constexpr const char* kPreferredTag = "pref-model:1b";
constexpr const char* kPhi3Config = R"({"architectures":["Phi3ForCausalLM"],"model_type":"phi3"})";

void UseModelInfo(const fs::path& path) {
#ifdef _WIN32
    _putenv_s("FLM_MODELINFO_PATH", path.string().c_str());
#else
    setenv("FLM_MODELINFO_PATH", path.string().c_str(), 1);
#endif
}

/// \brief an entry whose preferred source and own source are local directories
struct PreferredSourceFixture {
    fs::path root;
    fs::path base_source;
    fs::path preferred_source;
    std::string catalog_string;
    std::string root_string;
    model_list models;

    explicit PreferredSourceFixture(std::string_view name)
        : root(TempDirectory(name)),
          base_source(root / "base" / "resolve"),
          preferred_source(root / "amd" / "resolve") {
        Write(base_source / "config.json", kPhi3Config);
        Write(base_source / "base.gguf", "q8-weights");
        Write(preferred_source / "config.json", kPhi3Config);
        Write(preferred_source / "amd.gguf", "q4");
        const auto record = [](const fs::path& file) {
            return nlohmann::json{{"path", file.filename().string()},
                                  {"size", fs::file_size(file)},
                                  {"sha256", download_utils::calculate_file_sha256(file.string())}};
        };
        const nlohmann::json catalog = {
            {"model_path", "models"},
            {"models", {{"pref-model", {{"1b", {
                {"name", "pref-model"},
                {"url", FileUrl(base_source)},
                {"file_url", "https://example.invalid/api"},
                {"flm_min_version", "1.0.3"},
                {"details", {{"family", "qwen3"}, {"quantization_level", "Q8_0"}}},
                {"file_sources", nlohmann::json::object()},
                {"files", {"config.json", "base.gguf"}},
                {"preferred_source", {
                    {"url", FileUrl(preferred_source)},
                    {"use_hf_token", true},
                    {"details", {{"quantization_level", "Q4_G128"}}},
                    {"files", {"config.json", "amd.gguf"}},
                    {"model_info_key", "pref-model:1b@amd"}}}
            }}}}}}};
        const nlohmann::json info = {
            {"pref-model:1b", {record(base_source / "config.json"),
                               record(base_source / "base.gguf")}},
            {"pref-model:1b@amd", {record(preferred_source / "config.json"),
                                   record(preferred_source / "amd.gguf")}}};
        Write(root / "model_list.json", catalog.dump());
        Write(root / "model_info.json", info.dump());
        UseModelInfo(root / "model_info.json");
        catalog_string = (root / "model_list.json").string();
        root_string = root.string();
        models = model_list(catalog_string, root_string);
    }

    fs::path model_path() const { return root / "models" / "pref-model"; }

    std::vector<std::string> resolved_files() const {
        return models.get_model_info(kPreferredTag).second.at("files").get<std::vector<std::string>>();
    }
};

void TestShippedPreferredSourcesArePinnedAmdGgufs() {
    const auto catalog = ReadJson(FLM_SOURCE_DIR "/model_list.json");
    const auto all_info = ReadJson(FLM_SOURCE_DIR "/model_info.json");
    std::set<std::string> tags;
    for (const auto& [family, sizes] : catalog.at("models").items()) {
        for (const auto& [size, entry] : sizes.items()) {
            if (!entry.contains("preferred_source")) continue;
            const std::string tag = family + ":" + size;
            tags.insert(tag);
            const auto& preferred = entry.at("preferred_source");
            TEST_REQUIRE(preferred.at("url").get<std::string>().rfind("https://huggingface.co/amd/", 0) == 0);
            TEST_REQUIRE(preferred.at("use_hf_token") == true);

            const auto& files = preferred.at("files");
            const auto& records = all_info.at(preferred.at("model_info_key").get<std::string>());
            TEST_REQUIRE(records.size() == files.size());
            std::uint64_t total = 0;
            std::size_t replaced = 0;
            for (const auto& file : files) {
                const auto record = std::find_if(records.begin(), records.end(), [&](const auto& value) {
                    return value.at("path") == file;
                });
                TEST_REQUIRE(record != records.end());
                TEST_REQUIRE(record->at("sha256").get<std::string>().size() == 64);
                total += record->at("size").get<std::uint64_t>();
                const auto& own = entry.at("files");
                if (std::find(own.begin(), own.end(), file) == own.end()) {
                    TEST_REQUIRE(file.get<std::string>().ends_with(".gguf"));
                    ++replaced;
                }
            }
            TEST_REQUIRE(replaced == 1);
            TEST_REQUIRE(total == preferred.at("size").get<std::uint64_t>());
        }
    }
    TEST_REQUIRE(tags == std::set<std::string>({
        "phi4-mini-it-core:4b", "qwen3-core:1.7b", "qwen3-core:4b", "qwen3-core:8b",
        "qwen3.5-core:4b", "qwen3.5-core:9b", "qwen3.6-moe-core:35b-a3b", "qwen3.8-core:27b",
        "gemma4-it-core:e2b", "gemma4-it-core:e4b"}));
}

const std::vector<std::string> kBaseFiles = {"config.json", "base.gguf"};
const std::vector<std::string> kPreferredFiles = {"config.json", "amd.gguf"};

void TestPreferredSourceResolvesByWhatIsOnDisk() {
    PreferredSourceFixture fixture("preferred-resolve");
    auto info = fixture.models.get_model_info(kPreferredTag).second;
    TEST_REQUIRE(fixture.resolved_files() == kBaseFiles);
    TEST_REQUIRE(!info.contains("preferred_source"));
    TEST_REQUIRE(!resolve_file_source(info, "base.gguf", false).use_hf_token);
    TEST_REQUIRE(fixture.models.preferred_source_files(kPreferredTag) ==
                 std::vector<std::string>{"amd.gguf"});

    Write(fixture.model_path() / "amd.gguf.part", "q");
    TEST_REQUIRE(fixture.resolved_files() == kBaseFiles);

    Write(fixture.model_path() / "amd.gguf", "q4");
    info = fixture.models.get_model_info(kPreferredTag).second;
    TEST_REQUIRE(fixture.resolved_files() == kPreferredFiles);
    TEST_REQUIRE(info.at("details").at("family") == "qwen3");
    TEST_REQUIRE(info.at("details").at("quantization_level") == "Q4_G128");
    TEST_REQUIRE(info.at("model_info_key") == "pref-model:1b@amd");
    const auto source = resolve_file_source(info, "amd.gguf", false);
    TEST_REQUIRE(source.use_hf_token);
    TEST_REQUIRE(source.url == FileUrl(fixture.preferred_source) + "/amd.gguf?download=true");

    fixture.models.choose_source(kPreferredTag, false);
    TEST_REQUIRE(fixture.resolved_files() == kBaseFiles);
    fixture.models.choose_source("pref-model", std::nullopt);
    TEST_REQUIRE(fixture.resolved_files() == kPreferredFiles);
}

void TestPullUsesReachablePreferredSourceAndKeepsItOncePulled() {
    PreferredSourceFixture fixture("preferred-pull");
    ModelDownloader downloader(fixture.models);
    std::ostringstream output;
    auto* previous = std::cout.rdbuf(output.rdbuf());
    const bool pulled = downloader.pull_model(kPreferredTag);
    std::cout.rdbuf(previous);

    TEST_REQUIRE(pulled);
    TEST_REQUIRE(CountOccurrences(output.str(), "Using preferred source") == 1);
    TEST_REQUIRE(Read(fixture.model_path() / "amd.gguf") == "q4");
    TEST_REQUIRE(!fs::exists(fixture.model_path() / "base.gguf"));
    TEST_REQUIRE(fixture.resolved_files() == kPreferredFiles);

    // Pulled already, so an unreachable preferred source changes nothing.
    fs::remove_all(fixture.preferred_source);
    std::ostringstream again;
    previous = std::cout.rdbuf(again.rdbuf());
    const bool repulled = downloader.pull_model(kPreferredTag);
    std::cout.rdbuf(previous);
    TEST_REQUIRE(repulled);
    TEST_REQUIRE(CountOccurrences(again.str(), "Model already downloaded") == 1);
    TEST_REQUIRE(CountOccurrences(again.str(), "Falling back") == 0);
    TEST_REQUIRE(!fs::exists(fixture.model_path() / "base.gguf"));
}

void TestPullFallsBackWhenPreferredSourceIsUnreachable() {
    PreferredSourceFixture fixture("preferred-fallback");
    fs::remove_all(fixture.preferred_source);
    ModelDownloader downloader(fixture.models);
    std::ostringstream output;
    auto* previous = std::cout.rdbuf(output.rdbuf());
    const bool pulled = downloader.pull_model(kPreferredTag);
    std::cout.rdbuf(previous);

    TEST_REQUIRE(pulled);
    TEST_REQUIRE(CountOccurrences(output.str(), "Preferred source is not accessible") == 1);
    TEST_REQUIRE(CountOccurrences(output.str(), "Falling back to: " + FileUrl(fixture.base_source)) == 1);
    TEST_REQUIRE(Read(fixture.model_path() / "base.gguf") == "q8-weights");
    TEST_REQUIRE(!fs::exists(fixture.model_path() / "amd.gguf"));
    TEST_REQUIRE(fixture.resolved_files() == kBaseFiles);
}

download_utils::DownloadRequest Request(const fs::path& source, const fs::path& destination,
                                        std::uint64_t size, std::string hash) {
    return {FileUrl(source), destination, size, download_utils::HashAlgorithm::Sha256, std::move(hash)};
}

void TestResumeAppendsToPartThenAtomicallyPromotes() {
    const auto root = TempDirectory("resume");
    const auto source = root / "source.bin";
    const auto destination = root / "destination.bin";
    Write(source, "abcdefgh");
    Write(destination.string() + ".part", "abcd");
    TEST_REQUIRE(download_utils::download_file_atomic(
        Request(source, destination, 8, "9c56cc51b374c3ba189210d5b6d4bf57790d351c96c47c02190ecf1e430635ab")));
    TEST_REQUIRE(Read(destination) == "abcdefgh");
    TEST_REQUIRE(!fs::exists(destination.string() + ".part"));
}

void TestWrongSizeOrHashNeverReplacesAValidFinalFile() {
    const auto root = TempDirectory("wrong");
    const auto source = root / "source.bin";
    const auto destination = root / "destination.bin";
    Write(source, "ABCDEFGH");
    Write(destination, "abcdefgh");
    TEST_REQUIRE(!download_utils::download_file_atomic(
        Request(source, destination, 8, "9c56cc51b374c3ba189210d5b6d4bf57790d351c96c47c02190ecf1e430635ab")));
    TEST_REQUIRE(Read(destination) == "abcdefgh");
    TEST_REQUIRE(!fs::exists(destination.string() + ".part"));
}

void TestInterruptedTransferKeepsPartForNextResume() {
    const auto root = TempDirectory("interrupted");
    const auto destination = root / "destination.bin";
    Write(destination.string() + ".part", "abcd");
    auto request = Request(root / "missing.bin", destination, 8,
                           "9c56cc51b374c3ba189210d5b6d4bf57790d351c96c47c02190ecf1e430635ab");
    TEST_REQUIRE(!download_utils::download_file_atomic(request));
    TEST_REQUIRE(Read(destination.string() + ".part") == "abcd");
    TEST_REQUIRE(!fs::exists(destination));
}

void TestSuccessfulForceDownloadAtomicallyReplacesFinalFile() {
    const auto root = TempDirectory("replace");
    const auto source = root / "source.bin";
    const auto destination = root / "destination.bin";
    Write(source, "ABCDEFGH");
    Write(destination, "abcdefgh");
    TEST_REQUIRE(download_utils::download_file_atomic(
        Request(source, destination, 8, "9ac2197d9258257b1ae8463e4214e4cd0a578bc1517f2415928b91be4283fc48")));
    TEST_REQUIRE(Read(destination) == "ABCDEFGH");
    TEST_REQUIRE(!fs::exists(destination.string() + ".part"));
}
}  // namespace

int main() {
    RunTest(TestCoreCatalogHasExactlyFourFilesAndExpectedDirectoryName, "core catalog");
    RunTest(TestGgufUrlContainsUnslothRevisionAndFilename, "GGUF URL");
    RunTest(TestThreeFrontendUrlsContainMicrosoftRevisionAndFilename, "frontend URLs");
    RunTest(TestExistingSingleSourceEntryKeepsItsCurrentUrl, "legacy URL");
    RunTest(TestUnknownFileSourceKeyAndMissingUrlOrRevisionFail, "source validation");
    RunTest(TestActualCoreCatalogTreatsPinnedConfigWithoutFlmVersionAsCompatible,
            "core pinned config compatibility");
    RunTest(TestModelInfoHasExactSizeAndSha256ForEveryRequiredFile, "model metadata");
    RunTest(TestModelIsReadyOnlyWhenAllFourFinalFilesValidate, "ready integrity");
    RunTest(TestPartFileNeverMakesModelReady, "part is not ready");
    RunTest(TestLegacyReadyCheckDoesNotHashOrDeleteWeights, "legacy ready behavior");
    RunTest(TestPullAndCheckRejectModelscopeBeforePinnedReadyStateChecks,
            "modelscope rejection");
    RunTest(TestCheckHashesPinnedFilesExactlyOnce, "single check verification");
    RunTest(TestStartupStatusDoesNotRehashButCheckStillDoes, "startup status skips rehash");
    RunTest(TestShippedPreferredSourcesArePinnedAmdGgufs, "shipped preferred sources");
    RunTest(TestPreferredSourceResolvesByWhatIsOnDisk, "preferred source resolution");
    RunTest(TestPullUsesReachablePreferredSourceAndKeepsItOncePulled, "preferred source pull");
    RunTest(TestPullFallsBackWhenPreferredSourceIsUnreachable, "preferred source fallback");
    RunTest(TestResumeAppendsToPartThenAtomicallyPromotes, "resume and promote");
    RunTest(TestWrongSizeOrHashNeverReplacesAValidFinalFile, "invalid transfer isolation");
    RunTest(TestInterruptedTransferKeepsPartForNextResume, "interrupted transfer");
    RunTest(TestSuccessfulForceDownloadAtomicallyReplacesFinalFile, "atomic replacement");
}
