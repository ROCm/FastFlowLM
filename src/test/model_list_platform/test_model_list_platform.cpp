/// \file test_model_list_platform.cpp
/// \brief Platform and backend filtering in model_list, plus the npu_platform
///        helpers and a sweep of the shipped catalog.
/// \note  Deliberately free of NPU hardware: everything here is catalog logic,
///        so it builds and runs on Linux CI where the phi4_rai suite cannot.
#include "model_list.hpp"
#include "utils/npu_platform.hpp"
#include "../phi4_rai/test_support.hpp"

#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kCatalogPath = FLM_TEST_MODEL_LIST_PATH;
constexpr const char* kPhiTag = "phi4-mini-it:4b";
constexpr const char* kRaiTag = "phi4-mini-it-rai:4b";

/// \brief every generation the build knows about, for the catalog sweep
const std::set<std::string> kAllPlatforms = {
    std::string(utils::platform_id(utils::npu_platform::aie2p)),
    std::string(utils::platform_id(utils::npu_platform::aie_next))};

/// \brief open the shipped catalog as one machine and build read it
/// \param platform the generation the device reports
/// \param backends the kernel flows this build is to be taken as linking
model_list open_catalog(const std::string& platform,
                        std::vector<std::string> backends) {
    std::string path = kCatalogPath;
    std::string exe_dir = ".";
    return model_list(path, exe_dir, platform, std::move(backends));
}

/// \brief the catalog as a stock build on shipping silicon reads it
model_list open_stock() { return open_catalog("aie2p", {"flm"}); }

/// \brief the catalog as a corelib build on the next generation reads it
model_list open_rai() { return open_catalog("aie_next", {"flm", "rai"}); }

nlohmann::json read_json(const char* path) {
    std::ifstream stream(path);
    TEST_REQUIRE(stream.is_open());
    return nlohmann::json::parse(stream);
}

/// \brief write a catalog to a temp file and hand back its path
std::filesystem::path write_catalog(const char* name, const nlohmann::json& catalog) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path);
    out << catalog.dump(2);
    return path;
}

void test_stock_build_offers_phi4_with_its_own_artifacts() {
    // The regression this design exists to prevent. A stock install has one
    // phi4 -- the FastFlowLM one -- and nothing about a corelib model may
    // prune it.
    auto models = open_stock();
    TEST_REQUIRE(models.is_model_supported(kPhiTag));
    TEST_REQUIRE(models.rectify_model_tag("phi4-mini-it") == kPhiTag);

    const auto [tag, info] = models.get_model_info(kPhiTag);
    TEST_REQUIRE(tag == kPhiTag);
    TEST_REQUIRE(info.at("name") == "Phi4-mini-Instruct-NPU2");
    TEST_REQUIRE(info.at("default_context_length") == 32768);
    TEST_REQUIRE(info.at("flm_min_version") == "0.9.25");
    TEST_REQUIRE(info.contains("ms_url"));
    TEST_REQUIRE(!info.contains("file_sources"));
    // The tag names no flow, so it is the FastFlowLM one.
    TEST_REQUIRE(info.at("backend") == "flm");
    // Bookkeeping never reaches the caller.
    TEST_REQUIRE(!info.contains("supported_backend"));

    TEST_REQUIRE(models.is_model_supported("llama3.2:1b"));
    // A build with no corelib kernels cannot offer a corelib model.
    TEST_REQUIRE(!models.is_model_supported(kRaiTag));
    TEST_REQUIRE(!models.is_model_supported("phi4-mini-it-rai"));
}

void test_a_corelib_build_offers_the_corelib_phi4() {
    // The corelib entry is described by its own artifacts -- a Q8_0 GGUF, not
    // the NPU2/Q4NX package -- under its own tag, and it is offered by a build
    // that has corelib's kernels, on the silicon they were built for.
    auto corelib = open_rai();
    TEST_REQUIRE(corelib.is_model_supported(kRaiTag));

    const auto [tag, info] = corelib.get_model_info(kRaiTag);
    TEST_REQUIRE(tag == kRaiTag);
    TEST_REQUIRE(corelib.rectify_model_tag("phi4-mini-it-rai") == kRaiTag);
    TEST_REQUIRE(info.at("name") == "phi4-mini-it-rai");
    TEST_REQUIRE(info.at("default_context_length") == 4096);
    TEST_REQUIRE(info.at("flm_min_version") == "1.0.3");
    TEST_REQUIRE(info.at("files").size() == 4);
    TEST_REQUIRE(info.at("file_sources").size() == 3);
    // Its own package, pulled from its own upstream: no ModelScope mirror, and
    // the tag is the model_info.json key, so nothing has to redirect it.
    TEST_REQUIRE(!info.contains("ms_url"));
    TEST_REQUIRE(!info.contains("model_info_key"));
    // Same engine family, other kernels.
    TEST_REQUIRE(info.at("details").at("family") == "phi4");
    TEST_REQUIRE(info.at("details").at("parameter_size") == "4B");
    TEST_REQUIRE(info.at("backend") == "rai");
    TEST_REQUIRE(!info.contains("supported_platforms"));

    // Having the kernels is not having the silicon. On aie2p the entry names
    // another generation, so it is gone -- linking corelib does not make a
    // corelib model appear on a machine that cannot run it -- while the
    // FastFlowLM entries beside it are untouched.
    auto corelib_on_aie2p = open_catalog("aie2p", {"flm", "rai"});
    TEST_REQUIRE(!corelib_on_aie2p.is_model_supported(kRaiTag));
    TEST_REQUIRE(corelib_on_aie2p.is_model_supported(kPhiTag));
    TEST_REQUIRE(corelib_on_aie2p.get_model_info(kPhiTag).second.at("backend") == "flm");
}

void test_a_catalog_can_filter_down_to_nothing() {
    // The other corner of the same rule, and the one that reads like a bug if
    // it is not stated: a build with only the FastFlowLM kernels, on silicon
    // none of the FastFlowLM entries name. Every aie2p entry is pruned by the
    // generation and the one aie_next entry by the kernels it would need, so
    // the install offers nothing -- and that is a legible state, not a crash.
    // It says so on stderr and keeps going; the commands fail one at a time,
    // which is what makes `flm --help` still work on such a build.
    auto nothing = open_catalog("aie_next", {"flm"});
    TEST_REQUIRE(nothing.all_tags.empty());
    TEST_REQUIRE(!nothing.is_model_supported(kPhiTag));
    TEST_REQUIRE(!nothing.is_model_supported(kRaiTag));
    TEST_REQUIRE(!nothing.is_model_supported("llama3.2:1b"));
    // Asking anyway is an error with a reason, not a fallback to a model that
    // is not there.
    bool threw = false;
    try {
        (void)nothing.get_model_info(kPhiTag);
    } catch (const std::exception&) {
        threw = true;
    }
    TEST_REQUIRE(threw);
}

void test_the_tag_name_decides_the_backend() {
    // The rule, on a catalog small enough to see: a family ending in "-rai" is
    // corelib's, anything else is the FastFlowLM kernels'. Nothing else in the
    // entry is consulted, which is why the two never contend for a tag.
    const auto path = write_catalog(
        "flm_backend_model_list.json",
        {{"model_path", "models"},
         {"models",
          {{"demo", {{"1b", {{"name", "Demo"}}}}},
           {"demo-rai", {{"1b", {{"name", "Demo-rai"}}}}}}}});
    std::string list_path = path.string();
    std::string exe_dir = ".";

    // Neither entry says anything about silicon, so only the kernels decide:
    // without the corelib ones the rai tag is not a model this binary can run,
    // and the entries beside it are untouched.
    model_list stock(list_path, exe_dir, "aie2p", {"flm"});
    TEST_REQUIRE(stock.is_model_supported("demo:1b"));
    TEST_REQUIRE(!stock.is_model_supported("demo-rai:1b"));
    TEST_REQUIRE(stock.get_model_info("demo:1b").second.at("backend") == "flm");
    // Both lookups used to dereference the pruned entry. A pruned family keeps
    // its own name rather than resolving to a size, and an unknown tag comes
    // back as something the caller can safely report on.
    TEST_REQUIRE(stock.rectify_model_tag("demo-rai") == "demo-rai");
    TEST_REQUIRE(stock.rectify_model_tag("demo") == "demo:1b");
    const auto [missing_tag, missing_info] = stock.get_model_info("bogus:9b");
    TEST_REQUIRE(stock.is_model_supported(missing_tag));
    TEST_REQUIRE(missing_info.contains("name"));

    model_list corelib(list_path, exe_dir, "aie2p", {"flm", "rai"});
    TEST_REQUIRE(corelib.is_model_supported("demo:1b"));
    TEST_REQUIRE(corelib.is_model_supported("demo-rai:1b"));
    TEST_REQUIRE(corelib.get_model_info("demo-rai:1b").second.at("backend") == "rai");
    TEST_REQUIRE(corelib.get_model_info("demo:1b").second.at("backend") == "flm");

    std::filesystem::remove(path);
}

void test_an_entry_is_pruned_by_the_silicon_it_names() {
    // The other axis, on its own: same build, same kernels, different machine.
    const auto path = write_catalog(
        "flm_platform_model_list.json",
        {{"model_path", "models"},
         {"models",
          {{"shipping",
            {{"1b", {{"name", "Shipping"}, {"supported_platforms", {"aie2p"}}}}}},
           {"next",
            {{"1b", {{"name", "Next"}, {"supported_platforms", {"aie_next"}}}}}},
           {"either",
            {{"1b",
              {{"name", "Either"},
               {"supported_platforms", {"aie2p", "aie_next"}}}}}}}}});
    std::string list_path = path.string();
    std::string exe_dir = ".";

    model_list aie2p(list_path, exe_dir, "aie2p", {"flm"});
    TEST_REQUIRE(aie2p.is_model_supported("shipping:1b"));
    TEST_REQUIRE(aie2p.is_model_supported("either:1b"));
    TEST_REQUIRE(!aie2p.is_model_supported("next:1b"));
    // The key is bookkeeping: once the entry is kept, the generation is
    // settled and nothing downstream has to know it was ever asked.
    TEST_REQUIRE(!aie2p.get_model_info("shipping:1b").second.contains("supported_platforms"));

    model_list aie_next(list_path, exe_dir, "aie_next", {"flm"});
    TEST_REQUIRE(!aie_next.is_model_supported("shipping:1b"));
    TEST_REQUIRE(aie_next.is_model_supported("either:1b"));
    TEST_REQUIRE(aie_next.is_model_supported("next:1b"));

    std::filesystem::remove(path);
}

void test_missing_key_means_every_generation() {
    // Silence is the historical default, on both axes: an entry that names no
    // silicon runs on all of it, and a tag that does not say "-rai" runs on the
    // FastFlowLM kernels, which every family registers in every build. The
    // shipped catalog still spells the platform out -- this is about catalogs
    // older than the key, and about hand-made development trees.
    const auto path = write_catalog(
        "flm_legacy_model_list.json",
        {{"model_path", "models"},
         {"models", {{"legacy", {{"1b", {{"name", "Legacy"}}}}}}}});
    std::string list_path = path.string();
    std::string exe_dir = ".";

    for (const char* platform : {"aie2p", "aie_next"}) {
        model_list models(list_path, exe_dir, platform, {"flm"});
        TEST_REQUIRE(models.is_model_supported("legacy:1b"));
        TEST_REQUIRE(models.get_model_info("legacy:1b").second.at("backend") == "flm");
    }

    std::filesystem::remove(path);
}

void test_platform_helpers() {
    TEST_REQUIRE(utils::parse_platform("aie2p") == utils::npu_platform::aie2p);
    TEST_REQUIRE(utils::parse_platform("aie_next") == utils::npu_platform::aie_next);
    TEST_REQUIRE(!utils::parse_platform("not-a-platform").has_value());
    TEST_REQUIRE(!utils::parse_platform("stx").has_value());
    TEST_REQUIRE(utils::parse_platform(utils::platform_id(utils::npu_platform::aie_next)) ==
                 utils::npu_platform::aie_next);
    TEST_REQUIRE(utils::default_npu_platform() == utils::npu_platform::aie_next);

    // The stand-in consults nothing -- not the environment, not which kernels
    // were linked -- so it is the default here and in a corelib build alike.
    // Changing what an install can see means changing default_npu_platform()
    // and rebuilding, which is the one place to look.
    TEST_REQUIRE(utils::get_device() == utils::default_npu_platform());
    TEST_REQUIRE(utils::get_device() == utils::get_device());
}

void test_shipped_catalog_is_well_formed() {
    const auto catalog = read_json(kCatalogPath);
    bool saw_rai = false;

    for (const auto& [family, sizes] : catalog.at("models").items()) {
        const bool rai_family = family.size() > 4 &&
                                family.compare(family.size() - 4, 4, "-rai") == 0;
        for (const auto& [size, entry] : sizes.items()) {
            const std::string tag = family + ":" + size;

            // Retired spellings. Each of these once meant what
            // "supported_platforms" plus the tag name now mean between them,
            // and a catalog carrying one would be filtered on rules nothing
            // reads.
            for (const char* dead : {"platform_overrides", "supported_backend",
                                     "supported_backends", "backend"}) {
                if (entry.contains(dead)) {
                    throw std::runtime_error(tag + ": " + dead + " is retired");
                }
            }
            if (entry.contains("details") &&
                entry.at("details").contains("execution_backend")) {
                throw std::runtime_error(tag + ": execution_backend is retired");
            }

            if (!entry.contains("supported_platforms")) {
                throw std::runtime_error(tag + ": no supported_platforms");
            }
            const auto& supported = entry.at("supported_platforms");
            if (!supported.is_array() || supported.empty()) {
                throw std::runtime_error(
                    tag + ": supported_platforms must be a non-empty array");
            }
            std::set<std::string> named;
            for (const auto& value : supported) {
                if (!value.is_string() ||
                    !utils::parse_platform(value.get<std::string>()).has_value()) {
                    throw std::runtime_error(
                        tag + ": unknown platform in supported_platforms");
                }
                named.insert(value.get<std::string>());
            }
            TEST_REQUIRE(named.size() <= kAllPlatforms.size());

            // corelib runs on the next generation and the FastFlowLM kernels on
            // what is shipping, so a tag that says one and a platform key that
            // says the other describes a package nothing can run.
            if (rai_family != (named.count("aie_next") != 0)) {
                throw std::runtime_error(
                    tag + ": the tag name and supported_platforms disagree "
                          "about which generation this package is for");
            }
            if (rai_family) saw_rai = true;
        }
    }

    // A corelib model is packaged differently from its FastFlowLM namesake, so
    // it has its own tag; if that stops being true, everything above is
    // checking a rule the catalog no longer follows.
    TEST_REQUIRE(saw_rai);
    TEST_REQUIRE(catalog.at("models").contains("phi4-mini-it"));
    TEST_REQUIRE(catalog.at("models").contains("phi4-mini-it-rai"));
}

}  // namespace

int main() {
    RunTest(test_stock_build_offers_phi4_with_its_own_artifacts,
            "stock build offers phi4 with its own artifacts");
    RunTest(test_a_corelib_build_offers_the_corelib_phi4,
            "a corelib build offers the corelib phi4");
    RunTest(test_the_tag_name_decides_the_backend,
            "the tag name decides the backend");
    RunTest(test_a_catalog_can_filter_down_to_nothing,
            "a catalog can filter down to nothing");
    RunTest(test_an_entry_is_pruned_by_the_silicon_it_names,
            "an entry is pruned by the silicon it names");
    RunTest(test_missing_key_means_every_generation,
            "missing key means every generation");
    RunTest(test_platform_helpers, "platform helpers");
    RunTest(test_shipped_catalog_is_well_formed,
            "shipped catalog is well formed");
    std::cout << "All model_list platform tests passed\n";
    return 0;
}
