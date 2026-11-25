/// \file test_local_model_npu.cpp
/// \brief flm serve --model-dir with an NPU package: the entry a model folder becomes, the kernels it runs on and how model_list takes it
/// \note The folders here are small fakes: the entry is built from the package's
///       config.json and the file names, never from the weights.
#include "AutoModel/local_model_npu.hpp"
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
               ("flm_local_model_npu_" + std::to_string(counter++) + "_" +
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

std::set<std::string> files_of(const nlohmann::json& entry) {
    const auto list = entry.at("files").get<std::vector<std::string>>();
    return {list.begin(), list.end()};
}

/// \brief a catalog with one ordinary entry, read as a build with the oga-hipep backend
model_list open_catalog(model_list::backend_provider_inventory_t providers = {},
                        std::vector<std::string> backends = {"flm", "oga-hipep"}) {
    const auto path = fs::temp_directory_path() / "flm_local_model_npu_catalog.json";
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
                             {"quantization_level", "q"}}}}}}},
                // a shipped NPU model: its name is the folder its xclbins are filed under
                {"demo-vl", {{"4b", {
                {"name", "Demo-VL-NPU2"},
                {"size", 1},
                {"supported_platforms", {"aie2p"}},
                {"default_context_length", 4096},
                {"max_prefill_len", 512},
                {"details", {{"family", "qwen3vl"}, {"parameter_size", "4B"},
                             {"quantization_level", "q"}}}}}}}}}}.dump();
    }
    std::string list_path = path.string();
    std::string exe_dir = ".";
    return model_list(list_path, exe_dir, "aie2p", std::move(backends), {}, {},
                      std::move(providers));
}

/// \brief an NPU package: weights, network config, tokenizer, and a vision tower for a VLM
/// \note The Hugging Face repo also holds .xclbin files and a README; flm runs
///       on its own kernels, so the entry lists neither.
void make_npu_package(const fs::path& root, bool vision, const std::string& chat_template = "",
                      int max_position_embeddings = 40960) {
    write_file(root / "config.json", nlohmann::json{{"max_position_embeddings", max_position_embeddings},
                                                    {"flm_version", "0.9.22"}}.dump());
    write_file(root / "model.q4nx", "weights");
    write_file(root / "tokenizer.json", "tokenizer");
    write_file(root / "tokenizer_config.json", nlohmann::json{{"chat_template", chat_template}}.dump());
    if (vision) write_file(root / "vision_weight.q4nx", "vision");
    write_file(root / "attn.xclbin", "kernel");
    write_file(root / "README.md", "readme");
}

void test_an_npu_folder_becomes_an_entry() {
    TempFolder folder("Qwen3-VL-4B-NPU2");
    make_npu_package(folder.path, true, "{% if tools %}x{% endif %}");

    const auto local = flm::catalog::build_local_model_entry(folder.path, "qwen3vl", "aie2p");
    // an NPU tag has no backend suffix, as the shipped ones do not
    TEST_REQUIRE(local.model_type == "qwen3-vl-4b-npu2");
    TEST_REQUIRE(local.tag == "qwen3-vl-4b-npu2:local");
    TEST_REQUIRE(local.backend == "flm");

    const auto& entry = local.entry;
    TEST_REQUIRE(entry.at("backend") == "flm");
    TEST_REQUIRE(entry.at("local") == true);
    TEST_REQUIRE(entry.at("details").at("family") == "qwen3vl");
    TEST_REQUIRE(entry.at("supported_platforms") == nlohmann::json::array({"aie2p"}));
    TEST_REQUIRE(entry.at("vlm") == true);
    TEST_REQUIRE(entry.at("tools") == true);
    TEST_REQUIRE(entry.at("details").at("think") == false);
    TEST_REQUIRE(!entry.contains("file_sources"));
    // the package allows 40960; the default stays at what the shipped entries use
    TEST_REQUIRE(entry.at("default_context_length") == 32768);
    TEST_REQUIRE(files_of(entry) == (std::set<std::string>{
        "config.json", "model.q4nx", "tokenizer.json", "tokenizer_config.json", "vision_weight.q4nx"}));

    std::uint64_t total = 0;
    for (const auto& name : files_of(entry)) total += fs::file_size(folder.path / name);
    TEST_REQUIRE(entry.at("size").get<std::uint64_t>() == total);
}

void test_an_npu_text_package_and_its_context() {
    TempFolder folder("text-only");
    make_npu_package(folder.path, false, "<think>", 2048);
    const auto local = flm::catalog::build_local_model_entry(folder.path, "qwen3", "aie2p");
    TEST_REQUIRE(local.entry.at("vlm") == false);
    TEST_REQUIRE(local.entry.at("tools") == false);
    TEST_REQUIRE(local.entry.at("details").at("think") == true);
    TEST_REQUIRE(local.entry.at("default_context_length") == 2048);
    TEST_REQUIRE(files_of(local.entry).count("vision_weight.q4nx") == 0);

    // a chat_template.jinja next to the weights wins over the one in tokenizer_config.json
    write_file(folder.path / "chat_template.jinja", "{% if tools %}x{% endif %}");
    const auto with_file = flm::catalog::build_local_model_entry(folder.path, "qwen3", "aie2p");
    TEST_REQUIRE(with_file.entry.at("tools") == true);
    TEST_REQUIRE(with_file.entry.at("details").at("think") == false);
    TEST_REQUIRE(files_of(with_file.entry).count("chat_template.jinja") == 1);
}

void test_an_npu_folder_that_cannot_be_run_says_why() {
    // a folder of another format, and an empty one, are not NPU packages
    TempFolder other("other-format");
    write_file(other.path / "genai_config.json", "{}");
    write_file(other.path / "config.json", "{}");
    RequireContains(RequireThrows<std::runtime_error>([&] {
        flm::catalog::build_local_model_entry(other.path, "qwen3", "aie2p");
    }), "model.q4nx");

    TempFolder neither("neither");
    write_file(neither.path / "config.json", "{}");
    const auto none = RequireThrows<std::runtime_error>([&] {
        flm::catalog::build_local_model_entry(neither.path, "qwen3", "aie2p");
    });
    RequireContains(none, "not a model package");
    RequireContains(none, "model.q4nx");

    TempFolder partial("partial-npu");
    make_npu_package(partial.path, false);
    fs::remove(partial.path / "tokenizer.json");
    fs::remove(partial.path / "config.json");
    const auto missing = RequireThrows<std::runtime_error>([&] {
        flm::catalog::build_local_model_entry(partial.path, "qwen3", "aie2p");
    });
    RequireContains(missing, "tokenizer.json");
    RequireContains(missing, "config.json");

    TempFolder broken("broken-config");
    make_npu_package(broken.path, false);
    write_file(broken.path / "config.json", "{ not json");
    RequireContains(RequireThrows<std::runtime_error>([&] {
        flm::catalog::build_local_model_entry(broken.path, "qwen3", "aie2p");
    }), "config.json");

    // an NPU package is bound to a generation, so the caller has to name one
    TempFolder ok("needs-platform");
    make_npu_package(ok.path, false);
    RequireContains(RequireThrows<std::runtime_error>([&] {
        flm::catalog::build_local_model_entry(ok.path, "qwen3", "");
    }), "platform");
}

const model_list::backend_provider_inventory_t kNpuProviders = {
    {"qwen3vl", {{"flm", {"builtin"}}}}};

void test_model_list_takes_an_npu_entry() {
    TempFolder folder("npu-local");
    make_npu_package(folder.path, true, "{% if tools %}x{% endif %}");
    const auto local = flm::catalog::build_local_model_entry(folder.path, "qwen3vl", "aie2p");
    const std::string folder_name = local.entry.at("name").get<std::string>();

    auto models = open_catalog(kNpuProviders);
    // the kernels of a folder with another name come from a shipped model of the family
    TEST_REQUIRE(models.npu_folders_of_family("qwen3vl") == std::vector<std::string>{"Demo-VL-NPU2"});
    TEST_REQUIRE(models.npu_folders_of_family("qwen3.5").empty());   // that one is an OGA entry
    models.add_local_model(local.model_type, local.size, local.entry);
    TEST_REQUIRE(models.npu_folders_of_family("qwen3vl") == std::vector<std::string>{"Demo-VL-NPU2"});   // not the local one
    TEST_REQUIRE(models.is_model_supported(local.tag));
    TEST_REQUIRE(fs::path(models.get_load_path(local.tag)) == fs::path(folder_name));

    const auto [tag, info] = models.get_model_info(local.tag);
    TEST_REQUIRE(info.at("backend") == "flm");
    TEST_REQUIRE(info.at("backend_provider") == "builtin");
    // the platform was checked, and is not kept: execution does not need it
    TEST_REQUIRE(!info.contains("supported_platforms"));
    const auto capability = model_list::capabilities_of(info);
    TEST_REQUIRE(capability.vision);
    TEST_REQUIRE(capability.tools);
    TEST_REQUIRE(capability.context_length == 32768);

    const auto openai = models.get_all_models_openai();
    bool listed = false;
    for (const auto& item : openai.at("data")) {
        if (item.at("id") == local.tag) {
            listed = true;
            TEST_REQUIRE(item.at("backend") == "flm");
            TEST_REQUIRE(item.at("device") == "npu");
        }
    }
    TEST_REQUIRE(listed);

    // a package for another generation, a build without the backend, a family without it
    const auto other = flm::catalog::build_local_model_entry(folder.path, "qwen3vl", "dev04");
    auto wrong_silicon = open_catalog(kNpuProviders);
    RequireContains(RequireThrows<std::runtime_error>([&] {
        wrong_silicon.add_local_model(other.model_type, other.size, other.entry);
    }), "another NPU");
    TEST_REQUIRE(!wrong_silicon.is_model_supported(other.tag));

    auto no_backend = open_catalog({}, {"oga-hipep"});
    RequireContains(RequireThrows<std::runtime_error>([&] {
        no_backend.add_local_model(local.model_type, local.size, local.entry);
    }), "flm");
    auto no_family = open_catalog(model_list::backend_provider_inventory_t{
        {"gemma3", {{"flm", {"builtin"}}}}});
    RequireContains(RequireThrows<std::runtime_error>([&] {
        no_family.add_local_model(local.model_type, local.size, local.entry);
    }), "qwen3vl");
}

/// \brief opt-in: a real package against the shipped entry of the same model
/// \note FLM_TEST_LOCAL_MODEL_DIR is the folder, FLM_TEST_LOCAL_MODEL_TAG the
///       shipped tag it should agree with (e.g. qwen3vl-it:4b); the
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
    TEST_REQUIRE(local.backend == "flm");
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

/// \brief a catalog with a local NPU model added, and an xclbins folder holding the given kernel folders
struct KernelSetup {
    TempFolder package;
    TempFolder xclbins;
    model_list models;
    flm::catalog::LocalModelEntry local;

    KernelSetup(const std::string& package_name, const std::vector<std::string>& kernel_folders)
        : package(package_name), xclbins("xclbins"), models(open_catalog(kNpuProviders)) {
        make_npu_package(package.path, false);
        for (const auto& name : kernel_folders) fs::create_directories(xclbins.path / name);
        local = flm::catalog::build_local_model_entry(package.path, "qwen3vl", "aie2p");
        models.add_local_model(local.model_type, local.size, local.entry);
    }
    std::string kernels() const {
        return models.get_model_info(local.tag).second.value("xclbin_folder", std::string());
    }
};

void test_the_kernels_of_a_folder_named_like_a_shipped_model() {
    KernelSetup setup("Demo-VL-NPU2", {"Demo-VL-NPU2"});
    TEST_REQUIRE(flm::catalog::select_npu_kernels(setup.models, setup.local.tag, "qwen3vl", "", setup.xclbins.path).empty());
    TEST_REQUIRE(setup.kernels().empty());
}

void test_the_only_shipped_model_of_the_family_lends_its_kernels() {
    KernelSetup setup("my-own-name", {"Demo-VL-NPU2"});
    const auto note = flm::catalog::select_npu_kernels(setup.models, setup.local.tag, "qwen3vl", "", setup.xclbins.path);
    RequireContains(note, "Demo-VL-NPU2");
    TEST_REQUIRE(setup.kernels() == "Demo-VL-NPU2");
}

void test_the_kernels_can_be_named() {
    KernelSetup setup("my-own-name", {"Demo-VL-NPU2", "Other-NPU2"});
    // no folder under the package's name and more than one candidate would be a guess; a name settles it
    TEST_REQUIRE(flm::catalog::select_npu_kernels(setup.models, setup.local.tag, "qwen3vl", "Other-NPU2", setup.xclbins.path).empty());
    TEST_REQUIRE(setup.kernels() == "Other-NPU2");
}

void test_kernels_that_cannot_be_chosen_say_why() {
    // no kernels filed for the family
    KernelSetup none("my-own-name", {});
    const auto nothing = RequireThrows<std::runtime_error>([&] {
        flm::catalog::select_npu_kernels(none.models, none.local.tag, "qwen3vl", "", none.xclbins.path);
    });
    RequireContains(nothing, "--xclbin-folder");
    RequireContains(nothing, "my-own-name");

    // a name that is not a folder
    KernelSetup unknown("my-own-name", {"Demo-VL-NPU2"});
    const auto missing = RequireThrows<std::runtime_error>([&] {
        flm::catalog::select_npu_kernels(unknown.models, unknown.local.tag, "qwen3vl", "No-Such-NPU2", unknown.xclbins.path);
    });
    RequireContains(missing, "No-Such-NPU2");
    RequireContains(missing, "Demo-VL-NPU2");
    TEST_REQUIRE(unknown.kernels().empty());
}

void test_xclbin_folder_is_for_an_npu_model() {
    // a model on another backend takes no kernels folder, and a request for one is refused
    TempFolder folder("gpu-model");
    write_file(folder.path / "genai_config.json", "{}");
    auto models = open_catalog({}, {"flm", "oga-hipep"});
    auto entry = nlohmann::json{
        {"name", folder.path.string()}, {"size", 1}, {"local", true}, {"backend", "oga-hipep"},
        {"default_context_length", 4096}, {"max_prefill_len", 512},
        {"details", {{"family", "qwen3.5"}, {"parameter_size", "local"}, {"quantization_level", "unknown"}}}};
    models.add_local_model("gpu-model-oga-hipep", "local", entry);
    TEST_REQUIRE(flm::catalog::select_npu_kernels(models, "gpu-model-oga-hipep:local", "qwen3.5", "", folder.path).empty());
    RequireContains(RequireThrows<std::runtime_error>([&] {
        flm::catalog::select_npu_kernels(models, "gpu-model-oga-hipep:local", "qwen3.5", "Demo-VL-NPU2", folder.path);
    }), "--xclbin-folder");
}

}  // namespace

int main() {
    RunTest(test_an_npu_folder_becomes_an_entry, "an NPU folder becomes an entry");
    RunTest(test_an_npu_text_package_and_its_context, "an NPU text package and its context");
    RunTest(test_an_npu_folder_that_cannot_be_run_says_why, "an NPU folder that cannot be run says why");
    RunTest(test_model_list_takes_an_npu_entry, "model_list takes an NPU entry");
    RunTest(test_the_kernels_of_a_folder_named_like_a_shipped_model,
            "the kernels of a folder named like a shipped model");
    RunTest(test_the_only_shipped_model_of_the_family_lends_its_kernels,
            "the only shipped model of the family lends its kernels");
    RunTest(test_the_kernels_can_be_named, "the kernels can be named");
    RunTest(test_kernels_that_cannot_be_chosen_say_why, "kernels that cannot be chosen say why");
    RunTest(test_xclbin_folder_is_for_an_npu_model, "--xclbin-folder is for an NPU model");
    RunTest(test_a_real_package_matches_the_shipped_entry,
            "a real package matches the shipped entry");
    std::cout << "All local NPU model tests passed\n";
    return 0;
}