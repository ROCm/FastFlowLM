#include "download_model.hpp"
#include "model_downloader.hpp"
#include "test_support.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;

constexpr const char* kAie4Tag = "phi4-mini-it-aie4:4b";
constexpr const char* kUnslothRevision = "78eb92a46fc37e6b524df991ed9aca9bc6aa7b80";
constexpr const char* kMicrosoftRevision = "cfbefacb99257ffa30c83adab238a50856ac3083";

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

void TestAie4CatalogHasExactlyFourFilesAndExpectedDirectoryName() {
    const auto catalog = ReadJson(FLM_SOURCE_DIR "/model_list.json");
    const auto& model = catalog.at("models").at("phi4-mini-it-aie4").at("4b");
    const std::vector<std::string> expected = {
        "Phi-4-mini-instruct.Q8_0.gguf", "tokenizer.json",
        "tokenizer_config.json", "config.json"};
    TEST_REQUIRE(model.at("name") == "phi4-mini-it-aie4");
    TEST_REQUIRE(model.at("files").get<std::vector<std::string>>() == expected);
    TEST_REQUIRE(model.at("size").get<std::uint64_t>() == 4100140571ULL);
}

void TestGgufUrlContainsUnslothRevisionAndFilename() {
    const auto catalog = ReadJson(FLM_SOURCE_DIR "/model_list.json");
    const auto& model = catalog.at("models").at("phi4-mini-it-aie4").at("4b");
    const auto source = resolve_file_source(model, "Phi-4-mini-instruct.Q8_0.gguf", false);
    TEST_REQUIRE(source.url == std::string("https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/") +
                                  kUnslothRevision + "/Phi-4-mini-instruct.Q8_0.gguf?download=true");
}

void TestThreeFrontendUrlsContainMicrosoftRevisionAndFilename() {
    const auto catalog = ReadJson(FLM_SOURCE_DIR "/model_list.json");
    const auto& model = catalog.at("models").at("phi4-mini-it-aie4").at("4b");
    for (const std::string filename : {"tokenizer.json", "tokenizer_config.json", "config.json"}) {
        const auto source = resolve_file_source(model, filename, false);
        TEST_REQUIRE(source.url == std::string("https://huggingface.co/microsoft/Phi-4-mini-instruct/resolve/") +
                                      kMicrosoftRevision + "/" + filename + "?download=true");
    }
}

void TestExistingSingleSourceEntryKeepsItsCurrentUrl() {
    const auto catalog = ReadJson(FLM_SOURCE_DIR "/model_list.json");
    const auto& model = catalog.at("models").at("phi4-mini-it").at("4b");
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

void TestModelInfoHasExactSizeAndSha256ForEveryRequiredFile() {
    const auto all_info = ReadJson(FLM_SOURCE_DIR "/model_info.json");
    const auto& records = all_info.at(kAie4Tag);
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
                {"file_url", "https://example.invalid/api"}, {"flm_min_version", "0.0.0"},
                {"files", {"config.json", "a.bin", "b.bin", "c.bin"}}
            }}}}}}};
        const nlohmann::json info = {{"test-model:1b", {
            {{"path", "config.json"}, {"size", 2}, {"sha256", "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"}},
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
        Write(model_path() / "config.json", "{}");
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
        Request(source, destination, 7, "9ac2197d9258257b1ae8463e4214e4cd0a578bc1517f2415928b91be4283fc48")));
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
    RunTest(TestAie4CatalogHasExactlyFourFilesAndExpectedDirectoryName, "AIE4 catalog");
    RunTest(TestGgufUrlContainsUnslothRevisionAndFilename, "GGUF URL");
    RunTest(TestThreeFrontendUrlsContainMicrosoftRevisionAndFilename, "frontend URLs");
    RunTest(TestExistingSingleSourceEntryKeepsItsCurrentUrl, "legacy URL");
    RunTest(TestUnknownFileSourceKeyAndMissingUrlOrRevisionFail, "source validation");
    RunTest(TestModelInfoHasExactSizeAndSha256ForEveryRequiredFile, "model metadata");
    RunTest(TestModelIsReadyOnlyWhenAllFourFinalFilesValidate, "ready integrity");
    RunTest(TestPartFileNeverMakesModelReady, "part is not ready");
    RunTest(TestResumeAppendsToPartThenAtomicallyPromotes, "resume and promote");
    RunTest(TestWrongSizeOrHashNeverReplacesAValidFinalFile, "invalid transfer isolation");
    RunTest(TestInterruptedTransferKeepsPartForNextResume, "interrupted transfer");
    RunTest(TestSuccessfulForceDownloadAtomicallyReplacesFinalFile, "atomic replacement");
}
