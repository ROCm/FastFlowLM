/// \file test_local_model_oga.cpp
/// \brief flm serve --model-dir with an OGA package: the entry a model folder becomes and how model_list takes it
/// \note The folders here are small fakes: the entry is built from the package's
///       genai_config.json and the file names, never from the weights.
#include "AutoModel/local_model_oga.hpp"
#include "model_list.hpp"
#include "../phi4_core/test_support.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

/// \brief a temp folder with exactly this name, inside a unique parent, that goes away with the test
struct TempFolder {
    fs::path root;
    fs::path path;
    explicit TempFolder(const std::string& name) {
        static std::atomic<int> counter{0};
        root = fs::temp_directory_path() /
               ("flm_local_model_oga_" + std::to_string(counter++) + "_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        path = root / name;
        fs::create_directories(path);
    }
    ~TempFolder() {
        std::error_code error;
        fs::remove_all(root, error);
    }
};

void write_file(const fs::path& file, const std::string& text) {
    fs::create_directories(file.parent_path());
    std::ofstream out(file, std::ios::binary);
    out << text;
}

/// \brief a vision-language package: three graphs, each with a .data file
nlohmann::json vlm_genai_config() {
    return {
        {"model", {
            {"context_length", 262144},
            {"decoder", {{"filename", "text.onnx"}}},
            {"embedding", {{"filename", "embedding.onnx"}}},
            {"vision", {{"filename", "vision.onnx"}, {"config_filename", "processor_config.json"}}}}},
        {"search", {{"max_length", 262144}, {"past_present_share_buffer", true}}}};
}

/// \brief lay out a package; sizes are the byte counts of the strings written
void make_vlm_package(const fs::path& root, const std::string& chat_template = "{{ messages }}") {
    write_file(root / "genai_config.json", vlm_genai_config().dump());
    write_file(root / "config.json", "{}");
    write_file(root / "tokenizer.json", "tokenizer");
    write_file(root / "tokenizer_config.json", "{}");
    write_file(root / "chat_template.jinja", chat_template);
    write_file(root / "processor_config.json", "{}");
    write_file(root / "text.onnx", "graph");
    write_file(root / "text.onnx.data", "0123456789");
    write_file(root / "embedding.onnx", "graph");
    write_file(root / "embedding.onnx.data", "01234");
    write_file(root / "vision.onnx", "graph");
    write_file(root / "vision.onnx.data", "012");
    // present in a Hugging Face checkout, not loaded by flm
    write_file(root / "README.md", "readme");
    write_file(root / ".gitattributes", "attributes");
    write_file(root / "generation_config.json", "{}");
}

std::set<std::string> files_of(const nlohmann::json& entry) {
    const auto list = entry.at("files").get<std::vector<std::string>>();
    return {list.begin(), list.end()};
}

/// \brief a catalog with one ordinary entry, read as a build with the oga-hipep backend
model_list open_catalog(model_list::backend_provider_inventory_t providers = {},
                        std::vector<std::string> backends = {"flm", "oga-hipep"}) {
    const auto path = fs::temp_directory_path() / "flm_local_model_oga_catalog.json";
    {
        std::ofstream out(path);
        out << nlohmann::json{
            {"model_path", "models"},
            {"models", {{"demo-oga-hipep", {{"1b", {
                {"name", "Demo"},
                {"size", 1},
                {"default_context_length", 4096},
                {"max_prefill_len", 512},
                {"details", {{"family", "qwen3.5"}, {"parameter_size", "1B"},
                             {"quantization_level", "q"}}}}}}}}}}.dump();
    }
    std::string list_path = path.string();
    std::string exe_dir = ".";
    return model_list(list_path, exe_dir, "aie2p", std::move(backends), {}, {},
                      std::move(providers));
}

const model_list::backend_provider_inventory_t kQwenProviders = {
    {"qwen3.5", {{"oga-hipep", {"builtin"}}}}};

void test_a_folder_becomes_an_entry() {
    TempFolder folder("Qwen3.8-27B_gpuep_1.0_gpu_128");
    make_vlm_package(folder.path);

    const auto local = flm::catalog::build_local_model_entry(folder.path, "qwen3.5", "aie2p");
    TEST_REQUIRE(local.model_type == "qwen3.8-27b_gpuep_1.0_gpu_128-oga-hipep");
    TEST_REQUIRE(local.size == "local");
    TEST_REQUIRE(local.tag == "qwen3.8-27b_gpuep_1.0_gpu_128-oga-hipep:local");

    const auto& entry = local.entry;
    TEST_REQUIRE(fs::path(entry.at("name").get<std::string>()) == fs::absolute(folder.path).lexically_normal());
    TEST_REQUIRE(entry.at("local") == true);
    TEST_REQUIRE(entry.at("backend") == "oga-hipep");
    TEST_REQUIRE(entry.at("url") == "local");
    TEST_REQUIRE(entry.at("flm_min_version") == "0.0.0");
    TEST_REQUIRE(entry.at("details").at("family") == "qwen3.5");
    TEST_REQUIRE(entry.at("vlm") == true);
    TEST_REQUIRE(entry.at("max_prefill_len") == 4096);
    // the package allows 262144; the default stays at what the shipped entries use
    TEST_REQUIRE(entry.at("default_context_length") == 32768);
    // no hashing: a pinned entry would verify and delete files
    TEST_REQUIRE(!entry.contains("file_sources"));

    const std::set<std::string> expected = {
        "genai_config.json", "config.json", "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "processor_config.json", "text.onnx", "text.onnx.data",
        "embedding.onnx", "embedding.onnx.data", "vision.onnx", "vision.onnx.data"};
    TEST_REQUIRE(files_of(entry) == expected);

    std::uint64_t total = 0;
    for (const auto& name : expected) total += fs::file_size(folder.path / name);
    TEST_REQUIRE(entry.at("size").get<std::uint64_t>() == total);
}

void test_a_text_package_is_not_a_vlm_and_a_small_context_is_kept() {
    TempFolder folder("text-model");
    write_file(folder.path / "genai_config.json",
               nlohmann::json{{"model", {{"decoder", {{"filename", "model.onnx"}}}}},
                              {"search", {{"max_length", 2048}}}}.dump());
    write_file(folder.path / "config.json", "{}");
    write_file(folder.path / "tokenizer.json", "t");
    write_file(folder.path / "model.onnx", "g");

    const auto local = flm::catalog::build_local_model_entry(folder.path, "qwen2", "aie2p");
    TEST_REQUIRE(local.entry.at("vlm") == false);
    TEST_REQUIRE(local.entry.at("default_context_length") == 2048);
    TEST_REQUIRE(files_of(local.entry) ==
                 (std::set<std::string>{"genai_config.json", "config.json", "tokenizer.json", "model.onnx"}));
}

void test_the_chat_template_decides_thinking_and_tools() {
    TempFolder plain("plain");
    make_vlm_package(plain.path, "{{ messages }}");
    const auto none = flm::catalog::build_local_model_entry(plain.path, "qwen3.5", "aie2p").entry;
    TEST_REQUIRE(none.at("details").at("think") == false);
    TEST_REQUIRE(none.at("tools") == false);

    TempFolder reasoning("reasoning");
    make_vlm_package(reasoning.path, "{% if tools %}...{% endif %}{% if enable_thinking %}<think>{% endif %}");
    const auto both = flm::catalog::build_local_model_entry(reasoning.path, "qwen3.5", "aie2p").entry;
    TEST_REQUIRE(both.at("details").at("think") == true);
    TEST_REQUIRE(both.at("tools") == true);
}

void test_tag_names_are_safe() {
    // lower case, only letters, digits, '.', '_' and '-'; never empty
    TempFolder odd("My Model (v2)");
    make_vlm_package(odd.path);
    const auto local = flm::catalog::build_local_model_entry(odd.path, "qwen3.5", "aie2p");
    const std::string stem = local.model_type.substr(0, local.model_type.size() - std::string("-oga-hipep").size());
    TEST_REQUIRE(local.model_type.ends_with("-oga-hipep"));
    for (unsigned char c : stem) {
        TEST_REQUIRE(std::islower(c) || std::isdigit(c) || c == '.' || c == '_' || c == '-');
    }
    TEST_REQUIRE(stem.find("--") == std::string::npos);
    TEST_REQUIRE(stem.front() != '-' && stem.back() != '-');
    TEST_REQUIRE(stem == "my-model-v2");

    // a trailing separator still names the folder
    const auto with_slash = flm::catalog::build_local_model_entry(odd.path / "", "qwen3.5", "aie2p");
    TEST_REQUIRE(with_slash.model_type == local.model_type);
}

void test_a_bad_folder_says_what_is_wrong() {
    const auto absent = RequireThrows<std::runtime_error>([] {
        flm::catalog::build_local_model_entry(fs::temp_directory_path() / "flm_no_such_model_folder", "qwen3.5", "aie2p");
    });
    RequireContains(absent, "not found");

    TempFolder empty("empty");
    RequireContains(RequireThrows<std::runtime_error>([&] {
        flm::catalog::build_local_model_entry(empty.path, "qwen3.5", "aie2p");
    }), "genai_config.json");

    TempFolder partial("partial");
    make_vlm_package(partial.path);
    fs::remove(partial.path / "vision.onnx.data");
    fs::remove(partial.path / "text.onnx");
    fs::remove(partial.path / "tokenizer.json");
    const auto missing = RequireThrows<std::runtime_error>([&] {
        flm::catalog::build_local_model_entry(partial.path, "qwen3.5", "aie2p");
    });
    // every missing file is named, so one run is enough to fix the folder
    RequireContains(missing, "text.onnx");
    RequireContains(missing, "tokenizer.json");
    // a .data file is only expected where one exists: dropping it loses a file, not the graph
    TEST_REQUIRE(missing.find("vision.onnx.data") == std::string::npos);

    TempFolder broken("broken-json");
    write_file(broken.path / "genai_config.json", "{ not json");
    RequireContains(RequireThrows<std::runtime_error>([&] {
        flm::catalog::build_local_model_entry(broken.path, "qwen3.5", "aie2p");
    }), "genai_config.json");

    RequireContains(RequireThrows<std::runtime_error>([&] {
        flm::catalog::build_local_model_entry(empty.path, "", "aie2p");
    }), "--family");
}

void test_a_package_cannot_name_files_outside_its_folder() {
    TempFolder folder("escape");
    write_file(folder.path / "genai_config.json",
               nlohmann::json{{"model", {{"decoder", {{"filename", "../outside.onnx"}}}}}}.dump());
    write_file(folder.path / "config.json", "{}");
    write_file(folder.path / "tokenizer.json", "t");
    RequireContains(RequireThrows<std::runtime_error>([&] {
        flm::catalog::build_local_model_entry(folder.path, "qwen3.5", "aie2p");
    }), "outside");
}

void test_model_list_takes_the_entry() {
    TempFolder folder("qwen-local");
    make_vlm_package(folder.path, "{% if tools %}x{% endif %}<think>");
    auto local = flm::catalog::build_local_model_entry(folder.path, "qwen3.5", "aie2p");
    const std::string folder_name = local.entry.at("name").get<std::string>();

    auto models = open_catalog(kQwenProviders);
    TEST_REQUIRE(!models.is_model_supported(local.tag));
    models.add_local_model(local.model_type, local.size, local.entry);

    TEST_REQUIRE(models.is_model_supported(local.tag));
    TEST_REQUIRE(models.is_model_supported("demo-oga-hipep:1b"));   // the catalog is otherwise untouched

    // the model path is the folder itself, whatever the catalog's model root is
    TEST_REQUIRE(fs::path(models.get_model_path(local.tag)) == fs::path(folder_name));
    TEST_REQUIRE(fs::path(models.get_load_path(local.tag)) == fs::path(folder_name));

    const auto [tag, info] = models.get_model_info(local.tag);
    TEST_REQUIRE(tag == local.tag);
    TEST_REQUIRE(info.at("backend") == "oga-hipep");
    TEST_REQUIRE(info.at("backend_provider") == "builtin");
    TEST_REQUIRE(info.at("details").at("family") == "qwen3.5");

    // the bare family resolves to the one size it has
    TEST_REQUIRE(models.rectify_model_tag(local.model_type) == local.tag);

    // /v1/models and the Ollama listing read the entry like any other
    const auto capability = model_list::capabilities_of(info);
    TEST_REQUIRE(capability.completion);
    TEST_REQUIRE(capability.vision);
    TEST_REQUIRE(capability.tools);
    TEST_REQUIRE(capability.thinking);
    TEST_REQUIRE(capability.context_length == 32768);

    // keep the listings alive: iterating .at() of a temporary would read freed memory
    const auto openai = models.get_all_models_openai();
    const auto ollama_listing = models.get_all_models_ollama();
    bool listed = false;
    for (const auto& item : openai.at("data")) {
        if (item.at("id") == local.tag) {
            listed = true;
            TEST_REQUIRE(item.at("backend") == "oga-hipep");
            TEST_REQUIRE(item.at("device") == "gpu");
        }
    }
    TEST_REQUIRE(listed);
    bool ollama = false;
    for (const auto& item : ollama_listing.at("models")) {
        ollama = ollama || item.at("name") == local.tag;
    }
    TEST_REQUIRE(ollama);
}

void test_model_list_refuses_what_this_build_cannot_run() {
    TempFolder folder("refuse");
    make_vlm_package(folder.path);
    const auto local = flm::catalog::build_local_model_entry(folder.path, "qwen3.5", "aie2p");

    // a build without the oga-hipep backend
    auto no_backend = open_catalog({}, {"flm"});
    RequireContains(RequireThrows<std::runtime_error>([&] {
        no_backend.add_local_model(local.model_type, local.size, local.entry);
    }), "oga-hipep");
    TEST_REQUIRE(!no_backend.is_model_supported(local.tag));

    // a build with it, but not for this family
    auto no_family = open_catalog(model_list::backend_provider_inventory_t{
        {"gemma3", {{"oga-hipep", {"builtin"}}}}});
    RequireContains(RequireThrows<std::runtime_error>([&] {
        no_family.add_local_model(local.model_type, local.size, local.entry);
    }), "qwen3.5");
    TEST_REQUIRE(!no_family.is_model_supported(local.tag));

    // a tag that exists is never replaced
    auto taken = open_catalog(kQwenProviders);
    taken.add_local_model(local.model_type, local.size, local.entry);
    RequireContains(RequireThrows<std::runtime_error>([&] {
        taken.add_local_model(local.model_type, local.size, local.entry);
    }), "already exists");
    auto shipped = open_catalog(kQwenProviders);
    RequireContains(RequireThrows<std::runtime_error>([&] {
        shipped.add_local_model("demo-oga-hipep", "1b", local.entry);
    }), "already exists");
}

void test_model_list_checks_the_platform_an_entry_names() {
    TempFolder folder("platform-bound");
    make_vlm_package(folder.path);
    auto local = flm::catalog::build_local_model_entry(folder.path, "qwen3.5", "aie2p");

    // an entry that names this machine's platform is taken, and the key is not kept
    auto mine = local.entry;
    mine["supported_platforms"] = nlohmann::json::array({"aie2p"});
    auto taken = open_catalog(kQwenProviders);
    taken.add_local_model(local.model_type, local.size, mine);
    TEST_REQUIRE(taken.is_model_supported(local.tag));
    TEST_REQUIRE(!taken.get_model_info(local.tag).second.contains("supported_platforms"));

    // one that names another platform is refused
    auto other = local.entry;
    other["supported_platforms"] = nlohmann::json::array({"dev04"});
    auto refused = open_catalog(kQwenProviders);
    RequireContains(RequireThrows<std::runtime_error>([&] {
        refused.add_local_model(local.model_type, local.size, other);
    }), "another NPU");
    TEST_REQUIRE(!refused.is_model_supported(local.tag));
}

/// \brief a folder holding something else than an OGA package is refused and says what it wants
void test_a_folder_of_another_format_is_not_an_oga_package() {
    TempFolder folder("not-oga");
    write_file(folder.path / "config.json", "{}");
    write_file(folder.path / "model.q4nx", "weights");
    const auto message = RequireThrows<std::runtime_error>([&] {
        flm::catalog::build_local_model_entry(folder.path, "qwen3.5", "aie2p");
    });
    RequireContains(message, "not a model package");
    RequireContains(message, "genai_config.json");
}

/// \brief opt-in: a real package against the shipped entry of the same model
/// \note FLM_TEST_LOCAL_MODEL_DIR is the folder, FLM_TEST_LOCAL_MODEL_TAG the
///       shipped tag it should agree with (e.g. qwen3.8-oga-hipep:27b); the
///       family is read from that entry. Without both, the test is skipped.
void test_a_real_package_matches_the_shipped_entry() {
    const char* dir = std::getenv("FLM_TEST_LOCAL_MODEL_DIR");
    const char* tag = std::getenv("FLM_TEST_LOCAL_MODEL_TAG");
    if (!dir || !*dir || !tag || !*tag) {
        std::cout << "  (skipped: FLM_TEST_LOCAL_MODEL_DIR and FLM_TEST_LOCAL_MODEL_TAG are not both set)\n";
        return;
    }
    std::ifstream catalog_file(FLM_TEST_MODEL_LIST_PATH);
    TEST_REQUIRE(catalog_file.is_open());
    const auto catalog = nlohmann::json::parse(catalog_file);
    const std::string wanted = tag;
    const auto colon = wanted.find(':');
    TEST_REQUIRE(colon != std::string::npos);
    const auto& shipped = catalog.at("models").at(wanted.substr(0, colon)).at(wanted.substr(colon + 1));

    const auto local = flm::catalog::build_local_model_entry(
        dir, shipped.at("details").at("family").get<std::string>(), "aie2p");
    TEST_REQUIRE(local.backend == "oga-hipep");
    // the same files, the same capabilities; only the source differs
    TEST_REQUIRE(files_of(local.entry) == files_of(shipped));
    TEST_REQUIRE(local.entry.at("vlm") == shipped.value("vlm", false));
    TEST_REQUIRE(local.entry.at("tools") == shipped.value("tools", false));
    TEST_REQUIRE(local.entry.at("details").at("think") == shipped.at("details").value("think", false));
    TEST_REQUIRE(local.entry.at("default_context_length") == shipped.at("default_context_length"));
    TEST_REQUIRE(local.entry.at("max_prefill_len") == shipped.at("max_prefill_len"));
    std::cout << "  " << dir << " agrees with " << wanted << " (" << files_of(local.entry).size()
              << " files, " << local.entry.at("size") << " bytes)\n";
}

}  // namespace

int main() {
    RunTest(test_a_folder_becomes_an_entry, "a folder becomes an entry");
    RunTest(test_a_text_package_is_not_a_vlm_and_a_small_context_is_kept,
            "a text package is not a vlm and a small context is kept");
    RunTest(test_the_chat_template_decides_thinking_and_tools,
            "the chat template decides thinking and tools");
    RunTest(test_tag_names_are_safe, "tag names are safe");
    RunTest(test_a_bad_folder_says_what_is_wrong, "a bad folder says what is wrong");
    RunTest(test_a_package_cannot_name_files_outside_its_folder,
            "a package cannot name files outside its folder");
    RunTest(test_model_list_takes_the_entry, "model_list takes the entry");
    RunTest(test_model_list_refuses_what_this_build_cannot_run,
            "model_list refuses what this build cannot run");
    RunTest(test_model_list_checks_the_platform_an_entry_names,
            "model_list checks the platform an entry names");
    RunTest(test_a_folder_of_another_format_is_not_an_oga_package,
            "a folder of another format is not an OGA package");
    RunTest(test_a_real_package_matches_the_shipped_entry,
            "a real package matches the shipped entry");
    std::cout << "All local OGA model tests passed\n";
    return 0;
}
