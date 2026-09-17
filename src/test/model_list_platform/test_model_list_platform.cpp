/// \file test_model_list_platform.cpp
/// \brief Platform filtering / override merging in model_list, plus the
///        npu_platform helpers and a sweep of the shipped catalog.
/// \note  Deliberately free of NPU hardware: everything here is catalog logic,
///        so it builds and runs on Linux CI where the phi4_rai suite
///        cannot.
#include "model_list.hpp"
#include "utils/npu_platform.hpp"
#include "../phi4_rai/test_support.hpp"

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

void test_stx_entry_is_unchanged() {
    auto models = open_catalog("stx");
    const auto [tag, info] = models.get_model_info(kPhiTag);
    TEST_REQUIRE(tag == kPhiTag);
    TEST_REQUIRE(info.at("name") == "Phi4-mini-Instruct-NPU2");
    TEST_REQUIRE(info.contains("ms_url"));
    TEST_REQUIRE(!info.contains("file_sources"));
    TEST_REQUIRE(info.at("default_context_length") == 32768);
    TEST_REQUIRE(info.at("flm_min_version") == "0.9.25");
    // The stx catalog is the full catalog, and the aie_next-only entry is gone.
    TEST_REQUIRE(models.is_model_supported("llama3.2:1b"));
    TEST_REQUIRE(!models.is_model_supported("phi4-mini-it-rai:4b"));
    TEST_REQUIRE(!models.is_model_supported("phi4-mini-it-rai"));
}

void test_aie_next_entry_is_merged() {
    auto models = open_catalog("aie_next");
    TEST_REQUIRE(models.all_tags.size() == 2);
    TEST_REQUIRE(models.is_model_supported("phi4-mini-it"));
    TEST_REQUIRE(models.is_model_supported(kPhiTag));

    const auto [tag, info] = models.get_model_info(kPhiTag);
    TEST_REQUIRE(tag == kPhiTag);
    // The override wins where it speaks...
    TEST_REQUIRE(info.at("name") == "phi4-mini-it-rai");
    TEST_REQUIRE(info.at("default_context_length") == 4096);
    TEST_REQUIRE(info.at("flm_min_version") == "1.0.3");
    TEST_REQUIRE(info.at("model_info_key") == "phi4-mini-it-rai:4b");
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
    // And the entry names no backend: the hardware it was selected for is the
    // backend, so there is nothing left for the catalog to say about it.
    TEST_REQUIRE(!info.contains("supported_backends"));
    TEST_REQUIRE(!info.at("details").contains("execution_backend"));
}

void test_pruned_lookups_do_not_throw() {
    auto models = open_catalog("aie_next");
    // Both of these used to dereference the pruned llama3.2 family.
    const auto [missing_tag, missing_info] = models.get_model_info("bogus:9b");
    TEST_REQUIRE(missing_tag == kPhiTag);
    TEST_REQUIRE(missing_info.at("default_context_length") == 4096);
    TEST_REQUIRE(models.rectify_model_tag("llama3.2") == "llama3.2");
    TEST_REQUIRE(models.rectify_model_tag("phi4-mini-it") == kPhiTag);
}

void test_platform_helpers() {
    TEST_REQUIRE(utils::parse_platform("stx") == utils::npu_platform::stx);
    TEST_REQUIRE(utils::parse_platform("aie_next") == utils::npu_platform::aie_next);
    TEST_REQUIRE(!utils::parse_platform("not-a-platform").has_value());
    TEST_REQUIRE(utils::parse_platform(utils::platform_id(utils::npu_platform::aie_next)) ==
                 utils::npu_platform::aie_next);
    TEST_REQUIRE(utils::default_npu_platform() == utils::npu_platform::stx);

    // The generation is whatever this binary was built for, nothing else.
#ifdef FLM_ENABLE_RAI
    TEST_REQUIRE(utils::build_npu_platform() == utils::npu_platform::aie_next);
#else
    TEST_REQUIRE(utils::build_npu_platform() == utils::npu_platform::stx);
#endif
}

void test_shipped_catalog_is_well_formed() {
    const auto catalog = read_catalog();
    TEST_REQUIRE(!catalog.at("models").contains("phi4-mini-it-rai"));

    for (const auto& [family, sizes] : catalog.at("models").items()) {
        for (const auto& [size, entry] : sizes.items()) {
            const std::string tag = family + ":" + size;

            // Nothing in the catalog names a backend any more.
            if (entry.contains("supported_backends")) {
                throw std::runtime_error(tag + ": supported_backends is retired");
            }
            if (entry.contains("details") &&
                entry.at("details").contains("execution_backend")) {
                throw std::runtime_error(tag + ": execution_backend is retired");
            }

            // An entry is only tagged if it runs somewhere other than stx, so
            // the common case is no key at all. A key that says only ["stx"]
            // is not wrong, just noise, and this keeps it from creeping back.
            const nlohmann::json supported =
                entry.value("supported_platforms", nlohmann::json::array());
            if (entry.contains("supported_platforms")) {
                if (!supported.is_array() || supported.empty()) {
                    throw std::runtime_error(tag + ": supported_platforms must be a non-empty array");
                }
                bool beyond_stx = false;
                for (const auto& value : supported) {
                    if (!value.is_string() ||
                        !utils::parse_platform(value.get<std::string>()).has_value()) {
                        throw std::runtime_error(tag + ": unknown platform in supported_platforms");
                    }
                    if (value.get<std::string>() != "stx") beyond_stx = true;
                }
                if (!beyond_stx) {
                    throw std::runtime_error(
                        tag + ": supported_platforms says only stx, which is "
                              "the default -- drop the key");
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

void test_missing_key_means_stx_only() {
    // An untagged entry is stx-only. That is the rule the shipped catalog
    // leans on -- only aie_next support gets a tag -- so it is worth pinning down
    // from both sides: the untagged entry must appear on stx and must not
    // leak onto aie_next.
    const auto path = std::filesystem::temp_directory_path() /
                      "flm_legacy_model_list.json";
    nlohmann::json untagged = {
        {"model_path", "models"},
        {"models",
         {{"legacy", {{"1b", {{"name", "Legacy"}}}}},
          {"both",
           {{"1b",
             {{"name", "Both"},
              {"supported_platforms", {"stx", "aie_next"}}}}}}}}};
    {
        std::ofstream out(path);
        out << untagged.dump(2);
    }
    std::string list_path = path.string();
    std::string exe_dir = ".";

    model_list stx(list_path, exe_dir, "stx");
    TEST_REQUIRE(stx.is_model_supported("legacy:1b"));
    TEST_REQUIRE(stx.is_model_supported("both:1b"));

    model_list aie_next(list_path, exe_dir, "aie_next");
    TEST_REQUIRE(!aie_next.is_model_supported("legacy:1b"));
    TEST_REQUIRE(aie_next.is_model_supported("both:1b"));

    std::filesystem::remove(path);
}

}  // namespace

int main() {
    RunTest(test_stx_entry_is_unchanged, "stx entry is unchanged");
    RunTest(test_aie_next_entry_is_merged, "aie_next entry is merged");
    RunTest(test_pruned_lookups_do_not_throw, "pruned lookups do not throw");
    RunTest(test_platform_helpers, "platform helpers");
    RunTest(test_shipped_catalog_is_well_formed, "shipped catalog is well formed");
    RunTest(test_missing_key_means_stx_only, "missing key means stx only");
    std::cout << "All model_list platform tests passed\n";
    return 0;
}
