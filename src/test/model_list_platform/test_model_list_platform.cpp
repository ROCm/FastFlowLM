/// \file test_model_list_platform.cpp
/// \brief Platform filtering / override merging in model_list, plus the
///        npu_platform helpers and a sweep of the shipped catalog.
/// \note  Deliberately free of NPU hardware: everything here is catalog logic,
///        so it builds and runs on Linux CI where the phi4_corelib_aie4 suite
///        cannot.
#include "model_list.hpp"
#include "utils/npu_platform.hpp"
#include "../phi4_corelib_aie4/test_support.hpp"

#include <cstdlib>
#include <fstream>
#include <string>

namespace {

constexpr const char* kCatalogPath = FLM_TEST_MODEL_LIST_PATH;
constexpr const char* kPhiTag = "phi4-mini-it:4b";

/// \brief build a model_list over the shipped catalog for one platform
model_list open_catalog(const std::string& platform) {
    std::string path = kCatalogPath;
    std::string exe_dir = ".";
    return model_list(path, exe_dir, platform);
}

nlohmann::json read_catalog() {
    std::ifstream stream(kCatalogPath);
    TEST_REQUIRE(stream.is_open());
    return nlohmann::json::parse(stream);
}

void test_aie2p_entry_is_unchanged() {
    auto models = open_catalog("aie2p");
    const auto [tag, info] = models.get_model_info(kPhiTag);
    TEST_REQUIRE(tag == kPhiTag);
    TEST_REQUIRE(info.at("name") == "Phi4-mini-Instruct-NPU2");
    TEST_REQUIRE(info.contains("ms_url"));
    TEST_REQUIRE(!info.at("details").contains("execution_backend"));
    TEST_REQUIRE(!info.contains("file_sources"));
    TEST_REQUIRE(info.at("default_context_length") == 32768);
    TEST_REQUIRE(info.at("flm_min_version") == "0.9.25");
    // The aie2p catalog is the full catalog, and the retired aie4 family is gone.
    TEST_REQUIRE(models.is_model_supported("llama3.2:1b"));
    TEST_REQUIRE(!models.is_model_supported("phi4-mini-it-aie4:4b"));
    TEST_REQUIRE(!models.is_model_supported("phi4-mini-it-aie4"));
}

void test_aie4_entry_is_merged() {
    auto models = open_catalog("aie4");
    TEST_REQUIRE(models.all_tags.size() == 2);
    TEST_REQUIRE(models.is_model_supported("phi4-mini-it"));
    TEST_REQUIRE(models.is_model_supported(kPhiTag));

    const auto [tag, info] = models.get_model_info(kPhiTag);
    TEST_REQUIRE(tag == kPhiTag);
    // The override wins where it speaks...
    TEST_REQUIRE(info.at("name") == "phi4-mini-it-aie4");
    TEST_REQUIRE(info.at("details").at("execution_backend") == "corelib_aie4_gguf");
    TEST_REQUIRE(info.at("default_context_length") == 4096);
    TEST_REQUIRE(info.at("flm_min_version") == "1.0.3");
    TEST_REQUIRE(info.at("model_info_key") == "phi4-mini-it-aie4:4b");
    TEST_REQUIRE(info.at("files").size() == 4);
    TEST_REQUIRE(info.at("file_sources").size() == 3);
    // ...a null in the patch deletes the key...
    TEST_REQUIRE(!info.contains("ms_url"));
    // ...and the rest of details survives the recursive merge.
    TEST_REQUIRE(info.at("details").at("family") == "phi4");
    TEST_REQUIRE(info.at("details").at("parameter_size") == "4B");
    // Bookkeeping keys never reach the caller.
    TEST_REQUIRE(!info.contains("supported_platforms"));
    TEST_REQUIRE(!info.contains("platform_overrides"));
}

void test_pruned_lookups_do_not_throw() {
    auto models = open_catalog("aie4");
    // Both of these used to dereference the pruned llama3.2 family.
    const auto [missing_tag, missing_info] = models.get_model_info("bogus:9b");
    TEST_REQUIRE(missing_tag == kPhiTag);
    TEST_REQUIRE(missing_info.at("default_context_length") == 4096);
    TEST_REQUIRE(models.rectify_model_tag("llama3.2") == "llama3.2");
    TEST_REQUIRE(models.rectify_model_tag("phi4-mini-it") == kPhiTag);
}

void test_platform_helpers() {
    TEST_REQUIRE(utils::parse_platform("aie2p") == utils::npu_platform::aie2p);
    TEST_REQUIRE(utils::parse_platform("aie4") == utils::npu_platform::aie4);
    TEST_REQUIRE(!utils::parse_platform("not-a-platform").has_value());
    TEST_REQUIRE(utils::parse_platform(utils::platform_id(utils::npu_platform::aie4)) ==
                 utils::npu_platform::aie4);
    TEST_REQUIRE(utils::default_npu_platform() == utils::npu_platform::aie2p);

    // The generation is whatever this binary was built for, nothing else.
#ifdef FLM_ENABLE_AIE4
    TEST_REQUIRE(utils::build_npu_platform() == utils::npu_platform::aie4);
#else
    TEST_REQUIRE(utils::build_npu_platform() == utils::npu_platform::aie2p);
#endif
}

void test_shipped_catalog_is_well_formed() {
    const auto catalog = read_catalog();
    TEST_REQUIRE(!catalog.at("models").contains("phi4-mini-it-aie4"));

    for (const auto& [family, sizes] : catalog.at("models").items()) {
        for (const auto& [size, entry] : sizes.items()) {
            const std::string tag = family + ":" + size;
            TEST_REQUIRE(entry.contains("supported_platforms"));
            const auto& supported = entry.at("supported_platforms");
            if (!supported.is_array() || supported.empty()) {
                throw std::runtime_error(tag + ": supported_platforms must be a non-empty array");
            }
            for (const auto& value : supported) {
                if (!value.is_string() ||
                    !utils::parse_platform(value.get<std::string>()).has_value()) {
                    throw std::runtime_error(tag + ": unknown platform in supported_platforms");
                }
            }
            if (!entry.contains("platform_overrides")) continue;
            const auto& overrides = entry.at("platform_overrides");
            if (!overrides.is_object()) {
                throw std::runtime_error(tag + ": platform_overrides must be an object");
            }
            for (const auto& [platform, patch] : overrides.items()) {
                bool declared = false;
                for (const auto& value : supported) {
                    if (value.get<std::string>() == platform) declared = true;
                }
                if (!declared) {
                    throw std::runtime_error(
                        tag + ": platform_overrides has '" + platform +
                        "', which is not in supported_platforms");
                }
                if (!patch.is_object()) {
                    throw std::runtime_error(tag + ": override for '" + platform +
                                             "' must be an object");
                }
            }
        }
    }
}

void test_missing_key_means_aie2p_only() {
    // A catalog written before supported_platforms existed must keep its
    // original meaning rather than vanishing or leaking onto aie4.
    const auto path = std::filesystem::temp_directory_path() /
                      "flm_legacy_model_list.json";
    nlohmann::json legacy = {
        {"model_path", "models"},
        {"models", {{"legacy", {{"1b", {{"name", "Legacy"}}}}}}}};
    {
        std::ofstream out(path);
        out << legacy.dump(2);
    }
    std::string list_path = path.string();
    std::string exe_dir = ".";
    model_list aie2p(list_path, exe_dir, "aie2p");
    TEST_REQUIRE(aie2p.is_model_supported("legacy:1b"));
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    RunTest(test_aie2p_entry_is_unchanged, "aie2p entry is unchanged");
    RunTest(test_aie4_entry_is_merged, "aie4 entry is merged");
    RunTest(test_pruned_lookups_do_not_throw, "pruned lookups do not throw");
    RunTest(test_platform_helpers, "platform helpers");
    RunTest(test_shipped_catalog_is_well_formed, "shipped catalog is well formed");
    RunTest(test_missing_key_means_aie2p_only, "missing key means aie2p only");
    std::cout << "All model_list platform tests passed\n";
    return 0;
}
