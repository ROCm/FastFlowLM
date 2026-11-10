/// \file test_model_list_platform.cpp
/// \brief Platform and backend filtering in model_list, plus the npu_platform
///        helpers and a sweep of the shipped catalog.
/// \note  Deliberately free of NPU hardware: everything here is catalog logic,
///        so it builds and runs on Linux CI where the phi4_core suite cannot.
#include "model_list.hpp"
#include "utils/npu_platform.hpp"
#include "../phi4_core/test_support.hpp"

#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kCatalogPath = FLM_TEST_MODEL_LIST_PATH;
constexpr const char* kPhiTag = "phi4-mini-it:4b";
constexpr const char* kCoreTag = "phi4-mini-it-core:4b";

/// \brief every generation the build knows about, for the catalog sweep
const std::set<std::string> kAllPlatforms = {
    std::string(utils::platform_id(utils::npu_platform::aie2p)),
    std::string(utils::platform_id(utils::npu_platform::dev04))};

/// \brief open the shipped catalog as one machine and build read it
/// \param platform the generation the device reports
/// \param backends the kernel flows this build is to be taken as linking
model_list open_catalog(const std::string& platform,
                        std::vector<std::string> backends,
                        model_list::backend_provider_inventory_t providers = {}) {
    std::string path = kCatalogPath;
    std::string exe_dir = ".";
    return model_list(
        path, exe_dir, platform, std::move(backends), {}, {},
        std::move(providers));
}

/// \brief the catalog as a stock build on shipping silicon reads it
model_list open_stock() { return open_catalog("aie2p", {"flm"}); }

/// \brief the catalog as a corelib build on the next generation reads it
model_list open_core() { return open_catalog("dev04", {"flm", "core"}); }

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

/// \brief an overlay entry plus the download metadata every overlay must carry
nlohmann::json WithDownloadMetadata(nlohmann::json entry) {
    entry["url"] = "https://huggingface.co/FastFlowLM/Test-Model";
    entry["flm_min_version"] = "1.0.0";
    entry["files"] = {"config.json"};
    return entry;
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
    TEST_REQUIRE(!models.is_model_supported(kCoreTag));
    TEST_REQUIRE(!models.is_model_supported("phi4-mini-it-core"));
}

void test_a_corelib_build_offers_the_corelib_phi4() {
    // The corelib entry is described by its own artifacts -- AMD's Q4_G128 GGUF, not
    // the NPU2/Q4NX package -- under its own tag, and it is offered by a build
    // that has corelib's kernels, on the silicon they were built for.
    auto corelib = open_core();
    TEST_REQUIRE(corelib.is_model_supported(kCoreTag));

    const auto [tag, info] = corelib.get_model_info(kCoreTag);
    TEST_REQUIRE(tag == kCoreTag);
    TEST_REQUIRE(corelib.rectify_model_tag("phi4-mini-it-core") == kCoreTag);
    TEST_REQUIRE(info.at("name") == "phi4-mini-it-core");
    TEST_REQUIRE(info.at("default_context_length") == 131072);
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
    TEST_REQUIRE(info.at("backend") == "core");
    TEST_REQUIRE(!info.contains("supported_platforms"));

    // Having the kernels is not having the silicon. On aie2p the entry names
    // another generation, so it is gone -- linking corelib does not make a
    // corelib model appear on a machine that cannot run it -- while the
    // FastFlowLM entries beside it are untouched.
    auto corelib_on_aie2p = open_catalog("aie2p", {"flm", "core"});
    TEST_REQUIRE(!corelib_on_aie2p.is_model_supported(kCoreTag));
    TEST_REQUIRE(corelib_on_aie2p.is_model_supported(kPhiTag));
    TEST_REQUIRE(corelib_on_aie2p.get_model_info(kPhiTag).second.at("backend") == "flm");
}

void test_a_catalog_can_filter_down_to_nothing() {
    // The other corner of the same rule, and the one that reads like a bug if
    // it is not stated: a build with only the FastFlowLM kernels, on silicon
    // none of the FastFlowLM entries name. Every aie2p entry is pruned by the
    // generation and the one dev04 entry by the kernels it would need, so
    // the install offers nothing -- and that is a legible state, not a crash.
    // It says so on stderr and keeps going; the commands fail one at a time,
    // which is what makes `flm --help` still work on such a build.
    auto nothing = open_catalog("dev04", {"flm"});
    TEST_REQUIRE(nothing.all_tags.empty());
    TEST_REQUIRE(!nothing.is_model_supported(kPhiTag));
    TEST_REQUIRE(!nothing.is_model_supported(kCoreTag));
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
    // The rule, on a catalog small enough to see: a family ending in "-core" is
    // corelib's, anything else is the FastFlowLM kernels'. Nothing else in the
    // entry is consulted, which is why the two never contend for a tag.
    const auto path = write_catalog(
        "flm_backend_model_list.json",
        {{"model_path", "models"},
         {"models",
          {{"demo", {{"1b", {{"name", "Demo"}}}}},
           {"demo-core", {{"1b", {{"name", "Demo-core"}}}}}}}});
    std::string list_path = path.string();
    std::string exe_dir = ".";

    // Neither entry says anything about silicon, so only the kernels decide:
    // without the corelib ones the core tag is not a model this binary can run,
    // and the entries beside it are untouched.
    model_list stock(list_path, exe_dir, "aie2p", {"flm"});
    TEST_REQUIRE(stock.is_model_supported("demo:1b"));
    TEST_REQUIRE(!stock.is_model_supported("demo-core:1b"));
    TEST_REQUIRE(stock.get_model_info("demo:1b").second.at("backend") == "flm");
    // Both lookups used to dereference the pruned entry. A pruned family keeps
    // its own name rather than resolving to a size, and an unknown tag comes
    // back as something the caller can safely report on.
    TEST_REQUIRE(stock.rectify_model_tag("demo-core") == "demo-core");
    TEST_REQUIRE(stock.rectify_model_tag("demo") == "demo:1b");
    const auto [missing_tag, missing_info] = stock.get_model_info("bogus:9b");
    TEST_REQUIRE(stock.is_model_supported(missing_tag));
    TEST_REQUIRE(missing_info.contains("name"));

    model_list corelib(list_path, exe_dir, "aie2p", {"flm", "core"});
    TEST_REQUIRE(corelib.is_model_supported("demo:1b"));
    TEST_REQUIRE(corelib.is_model_supported("demo-core:1b"));
    TEST_REQUIRE(corelib.get_model_info("demo-core:1b").second.at("backend") == "core");
    TEST_REQUIRE(corelib.get_model_info("demo:1b").second.at("backend") == "flm");

    std::filesystem::remove(path);
}

void test_the_oga_hipep_suffix_names_its_backend() {
    const auto path = write_catalog(
        "flm_oga_hipep_model_list.json",
        {{"model_path", "models"},
         {"models",
          {{"demo", {{"1b", {{"name", "Demo"}}}}},
           {"demo-core", {{"1b", {{"name", "Demo-core"}}}}},
           {"demo-oga-hipep", {{"1b", {{"name", "Demo-oga-hipep"}}}}}}}});
    std::string list_path = path.string();
    std::string exe_dir = ".";

    model_list stock(list_path, exe_dir, "aie2p", {"flm"});
    TEST_REQUIRE(!stock.is_model_supported("demo-oga-hipep:1b"));

    model_list gpu(list_path, exe_dir, "aie2p", {"flm", "oga-hipep"});
    TEST_REQUIRE(gpu.is_model_supported("demo-oga-hipep:1b"));
    TEST_REQUIRE(!gpu.is_model_supported("demo-core:1b"));
    TEST_REQUIRE(gpu.get_model_info("demo-oga-hipep:1b").second.at("backend") == "oga-hipep");
    TEST_REQUIRE(gpu.get_model_info("demo:1b").second.at("backend") == "flm");

    model_list both(list_path, exe_dir, "dev04", {"core", "flm", "oga-hipep"});
    TEST_REQUIRE(both.get_model_info("demo-core:1b").second.at("backend") == "core");
    TEST_REQUIRE(both.get_model_info("demo-oga-hipep:1b").second.at("backend") == "oga-hipep");

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
            {{"1b", {{"name", "Next"}, {"supported_platforms", {"dev04"}}}}}},
           {"either",
            {{"1b",
              {{"name", "Either"},
               {"supported_platforms", {"aie2p", "dev04"}}}}}}}}});
    std::string list_path = path.string();
    std::string exe_dir = ".";

    model_list aie2p(list_path, exe_dir, "aie2p", {"flm"});
    TEST_REQUIRE(aie2p.is_model_supported("shipping:1b"));
    TEST_REQUIRE(aie2p.is_model_supported("either:1b"));
    TEST_REQUIRE(!aie2p.is_model_supported("next:1b"));
    // The key is bookkeeping: once the entry is kept, the generation is
    // settled and nothing downstream has to know it was ever asked.
    TEST_REQUIRE(!aie2p.get_model_info("shipping:1b").second.contains("supported_platforms"));

    model_list dev04(list_path, exe_dir, "dev04", {"flm"});
    TEST_REQUIRE(!dev04.is_model_supported("shipping:1b"));
    TEST_REQUIRE(dev04.is_model_supported("either:1b"));
    TEST_REQUIRE(dev04.is_model_supported("next:1b"));

    std::filesystem::remove(path);
}

void test_missing_key_means_every_generation() {
    // Silence is the historical default, on both axes: an entry that names no
    // silicon runs on all of it, and a tag that does not say "-core" runs on the
    // FastFlowLM kernels, which every family registers in every build. The
    // shipped catalog still spells the platform out -- this is about catalogs
    // older than the key, and about hand-made development trees.
    const auto path = write_catalog(
        "flm_legacy_model_list.json",
        {{"model_path", "models"},
         {"models", {{"legacy", {{"1b", {{"name", "Legacy"}}}}}}}});
    std::string list_path = path.string();
    std::string exe_dir = ".";

    for (const char* platform : {"aie2p", "dev04"}) {
        model_list models(list_path, exe_dir, platform, {"flm"});
        TEST_REQUIRE(models.is_model_supported("legacy:1b"));
        TEST_REQUIRE(models.get_model_info("legacy:1b").second.at("backend") == "flm");
    }

    std::filesystem::remove(path);
}

void test_extension_catalogs_merge_without_overrides() {
    const auto base = write_catalog(
        "flm_extension_base_catalog.json",
        {{"model_path", "models"},
         {"models",
          {{"base",
            {{"1b",
              {{"name", "Base"},
               {"details", {{"family", "base-frontend"}}}}}}}}}});
    std::string base_path = base.string();
    std::string exe_dir = ".";
    model_list models(
        base_path, exe_dir, "aie2p", {"flm"},
        {flm::catalog::load_model_catalog_overlay(
            FLM_TEST_VALID_CATALOG_OVERLAY)},
        {"base-frontend", "demo-frontend"},
        {{"base-frontend", {{"flm", {"builtin"}}}},
         {"demo-frontend", {{"flm", {"demo-provider"}}}}});
    TEST_REQUIRE(models.is_model_supported("base:1b"));
    TEST_REQUIRE(models.is_model_supported("extension-demo:1b"));
    TEST_REQUIRE(
        models.get_model_info("extension-demo:1b")
            .second.at("details").at("family") == "demo-frontend");
    TEST_REQUIRE(
        models.get_model_info("extension-demo:1b")
            .second.at("backend_provider") == "demo-provider");

    const auto duplicate = write_catalog(
        "flm_extension_duplicate_catalog.json",
        {{"schema_version", 1},
         {"models",
          {{"base",
            {{"1b",
              WithDownloadMetadata({{"name", "Replacement"},
               {"backend", "flm"},
               {"backend_provider", "builtin"},
               {"default_context_length", 4096},
               {"max_prefill_len", 512},
               {"details", {{"family", "base-frontend"}}}})}}}}}});
    RequireContains(
        RequireThrows([&] {
            model_list duplicate_models(
                base_path, exe_dir, "aie2p", {"flm"},
                {flm::catalog::load_model_catalog_overlay(
                    duplicate.string())},
                {"base-frontend"},
                {{"base-frontend", {{"flm", {"builtin"}}}}});
        }),
        "duplicates model tag 'base:1b'");

    const auto redirect = write_catalog(
        "flm_extension_redirect_alias_catalog.json",
        {{"schema_version", 1},
         {"models",
          {{"base",
            {{"0b",
              WithDownloadMetadata({{"name", "Alias redirect"},
               {"backend", "flm"},
               {"backend_provider", "builtin"},
               {"default_context_length", 4096},
               {"max_prefill_len", 512},
               {"details", {{"family", "base-frontend"}}}})}}}}}});
    RequireContains(
        RequireThrows([&] {
            model_list redirect_models(
                base_path, exe_dir, "aie2p", {"flm"},
                {flm::catalog::load_model_catalog_overlay(
                    redirect.string())},
                {"base-frontend"},
                {{"base-frontend", {{"flm", {"builtin"}}}}});
        }),
        "changes existing bare model tag 'base' from 'base:1b' to 'base:0b'");

    const auto later_size = write_catalog(
        "flm_extension_later_size_catalog.json",
        {{"schema_version", 1},
         {"models",
          {{"base",
            {{"2b",
              WithDownloadMetadata({{"name", "Additional size"},
               {"backend", "flm"},
               {"backend_provider", "builtin"},
               {"default_context_length", 4096},
               {"max_prefill_len", 512},
               {"details", {{"family", "base-frontend"}}}})}}}}}});
    model_list extended_models(
        base_path, exe_dir, "aie2p", {"flm"},
        {flm::catalog::load_model_catalog_overlay(
            later_size.string())},
        {"base-frontend"},
        {{"base-frontend", {{"flm", {"builtin"}}}}});
    TEST_REQUIRE(extended_models.is_model_supported("base:2b"));
    TEST_REQUIRE(extended_models.rectify_model_tag("base") == "base:1b");

    const auto unknown = write_catalog(
        "flm_extension_unknown_catalog.json",
        {{"schema_version", 1},
         {"models",
          {{"unknown",
            {{"1b",
              WithDownloadMetadata({{"name", "Unknown"},
               {"backend", "flm"},
               {"backend_provider", "builtin"},
               {"default_context_length", 4096},
               {"max_prefill_len", 512},
               {"details", {{"family", "missing-frontend"}}}})}}}}}});
    RequireContains(
        RequireThrows([&] {
            model_list unknown_models(
                base_path, exe_dir, "aie2p", {"flm"},
                {flm::catalog::load_model_catalog_overlay(
                    unknown.string())},
                {"base-frontend"});
        }),
        "unregistered frontend family 'missing-frontend'");

    const auto unknown_provider = write_catalog(
        "flm_extension_unknown_provider_catalog.json",
        {{"schema_version", 1},
         {"models",
          {{"provider-test",
            {{"1b",
              WithDownloadMetadata({{"name", "Unknown provider"},
               {"backend", "flm"},
               {"backend_provider", "missing-provider"},
               {"default_context_length", 4096},
               {"max_prefill_len", 512},
               {"details", {{"family", "base-frontend"}}}})}}}}}});
    RequireContains(
        RequireThrows([&] {
            model_list unknown_provider_models(
                base_path, exe_dir, "aie2p", {"flm"},
                {flm::catalog::load_model_catalog_overlay(
                    unknown_provider.string())},
                {"base-frontend"},
                {{"base-frontend", {{"flm", {"builtin"}}}}});
        }),
        "base-frontend/flm/missing-provider");

    RequireContains(
        RequireThrows([&] {
            model_list invalid_schema_models(
                base_path, exe_dir, "aie2p", {"flm"},
                {flm::catalog::load_model_catalog_overlay(
                    FLM_TEST_INVALID_CATALOG_OVERLAY)},
                {"base-frontend"});
        }),
        "must use schema_version 1");

    const auto schema = read_json(FLM_TEST_CATALOG_OVERLAY_SCHEMA);
    TEST_REQUIRE(
        schema.at("$id") ==
        "https://github.com/ROCm/FastFlowLM/schemas/"
        "model-catalog-overlay-v1.schema.json");
    TEST_REQUIRE(schema.at("properties")
                     .at("schema_version")
                     .at("const") == 1);
    TEST_REQUIRE(schema.at("type") == "object");
    TEST_REQUIRE(schema.at("additionalProperties") == false);
    TEST_REQUIRE(
        schema.at("required") ==
        nlohmann::json::array({"schema_version", "models"}));
    const auto& model_entry = schema.at("$defs").at("modelEntry");
    TEST_REQUIRE(
        model_entry.at("required") ==
        nlohmann::json::array(
            {"name", "backend", "backend_provider",
             "default_context_length", "max_prefill_len", "details",
             "url", "flm_min_version", "files"}));
    TEST_REQUIRE(
        model_entry.at("properties")
            .at("details")
            .at("required") ==
        nlohmann::json::array({"family"}));

    std::filesystem::remove(base);
    std::filesystem::remove(duplicate);
    std::filesystem::remove(redirect);
    std::filesystem::remove(later_size);
    std::filesystem::remove(unknown);
    std::filesystem::remove(unknown_provider);
}

void test_extension_overlay_requires_explicit_backend_provider() {
    const auto base = write_catalog(
        "flm_direct_builtin_model_list.json",
        {{"model_path", "models"},
         {"models",
          {{"base-model",
            {{"1b",
              {{"name", "Base model"},
               {"details", {{"family", "base-frontend"}}}}}}}}}});
    const auto overlay = write_catalog(
        "flm_direct_builtin_overlay.json",
        {{"schema_version", 1},
         {"models",
          {{"direct-model",
            {{"1b",
              {{"name", "Direct built-in model"},
               {"details", {{"family", "direct-frontend"}}}}}}}}}});
    std::string catalog_path = base.string();
    std::string exe_dir = ".";
    RequireContains(
        RequireThrows([&] {
            model_list models(
                catalog_path, exe_dir, "aie2p", {"flm"},
                {flm::catalog::load_model_catalog_overlay(
                    overlay.string())},
                {"base-frontend", "direct-frontend"},
                {{"base-frontend", {{"flm", {"builtin"}}}}});
        }),
        "backend_provider");
    std::filesystem::remove(base);
    std::filesystem::remove(overlay);
}

void test_overlay_keeps_builtin_bare_tag_after_platform_filter() {
    // Validation sees base:1b first and accepts base:2b as a later size, but
    // dev04 filters base:1b out; the bare tag must still reach base:3b.
    const auto base = write_catalog(
        "flm_filtered_alias_base_catalog.json",
        {{"model_path", "models"},
         {"models",
          {{"base",
            {{"1b",
              {{"name", "Base 1B"},
               {"supported_platforms", {"aie2p"}},
               {"details", {{"family", "base-frontend"}}}}},
             {"3b",
              {{"name", "Base 3B"},
               {"supported_platforms", {"dev04"}},
               {"details", {{"family", "base-frontend"}}}}}}}}}});
    const auto overlay = write_catalog(
        "flm_filtered_alias_overlay.json",
        {{"schema_version", 1},
         {"models",
          {{"base",
            {{"2b",
              WithDownloadMetadata(
                  {{"name", "Extension 2B"},
                   {"supported_platforms", {"dev04"}},
                   {"backend", "flm"},
                   {"backend_provider", "builtin"},
                   {"default_context_length", 4096},
                   {"max_prefill_len", 512},
                   {"details", {{"family", "base-frontend"}}}})}}}}}});
    std::string base_path = base.string();
    std::string exe_dir = ".";
    model_list models(
        base_path, exe_dir, "dev04", {"flm"},
        {flm::catalog::load_model_catalog_overlay(overlay.string())},
        {"base-frontend"},
        {{"base-frontend", {{"flm", {"builtin"}}}}});
    TEST_REQUIRE(!models.is_model_supported("base:1b"));
    TEST_REQUIRE(models.is_model_supported("base:2b"));
    TEST_REQUIRE(models.rectify_model_tag("base") == "base:3b");
    std::filesystem::remove(base);
    std::filesystem::remove(overlay);
}

void test_overlay_requires_download_metadata() {
    // flm pull reads these for every catalog entry, so an overlay missing one
    // must fail at load time instead of crashing the downloader later.
    const nlohmann::json complete = WithDownloadMetadata(
        {{"name", "Download model"},
         {"backend", "flm"},
         {"backend_provider", "external"},
         {"default_context_length", 4096},
         {"max_prefill_len", 512},
         {"details", {{"family", "download"}}}});
    for (const auto& [field, invalid] :
         std::vector<std::pair<std::string, nlohmann::json>>{
             {"url", nullptr},
             {"flm_min_version", nullptr},
             {"flm_min_version", ""},
             {"files", nullptr},
             {"files", nlohmann::json::array()},
             {"files", nlohmann::json::array({""})}}) {
        nlohmann::json entry = complete;
        if (invalid.is_null()) {
            entry.erase(field);
        } else {
            entry[field] = invalid;
        }
        const auto overlay = write_catalog(
            "flm_overlay_without_download_metadata.json",
            {{"schema_version", 1},
             {"models", {{"download-model", {{"1b", entry}}}}}});
        RequireContains(
            RequireThrows([&] {
                (void)flm::catalog::load_model_catalog_overlay(
                    overlay.string());
            }),
            field);
        std::filesystem::remove(overlay);
    }
}

void test_multimodal_overlay_requires_frontend_provider() {
    const auto overlay = write_catalog(
        "flm_multimodal_overlay_without_frontend.json",
        {{"schema_version", 1},
         {"models",
          {{"multimodal-model",
            {{"1b",
              WithDownloadMetadata({{"name", "Multimodal model"},
               {"backend", "flm"},
               {"backend_provider", "external"},
               {"default_context_length", 4096},
               {"max_prefill_len", 512},
               {"vlm", true},
               {"details", {{"family", "multimodal"}}}})}}}}}});
    RequireContains(
        RequireThrows([&] {
            (void)flm::catalog::load_model_catalog_overlay(
                overlay.string());
        }),
        "frontend_provider");
    std::filesystem::remove(overlay);
}

void test_overlay_paths_stay_inside_the_model_folder() {
    // flm pull writes, and flm remove deletes, under <model root>/<name>/<file>.
    const nlohmann::json complete = WithDownloadMetadata(
        {{"name", "Path model"},
         {"backend", "flm"},
         {"backend_provider", "external"},
         {"default_context_length", 4096},
         {"max_prefill_len", 512},
         {"details", {{"family", "path"}}}});
    const auto load = [](const nlohmann::json& entry) {
        const auto overlay = write_catalog(
            "flm_overlay_paths.json",
            {{"schema_version", 1},
             {"models", {{"path-model", {{"1b", entry}}}}}});
        const auto remove = [&] { std::filesystem::remove(overlay); };
        try {
            (void)flm::catalog::load_model_catalog_overlay(overlay.string());
        } catch (...) {
            remove();
            throw;
        }
        remove();
    };

    nlohmann::json nested = complete;
    nested["files"] = {"config.json", "decoder/model.onnx"};
    nested["xclbin_folder"] = "Path-Model-Kernels";
    load(nested);

    // COM with a superscript 4 is not reserved; Windows reserves only the
    // superscripts 1 to 3.
    for (const std::string name :
         {"CONSOLE", "COM10", "LPT0", "Model.v2", "COM\xE2\x81\xB4"}) {
        nlohmann::json entry = complete;
        entry["name"] = name;
        load(entry);
    }

    // Windows opens "Model." and "Model " as "Model", and CON, NUL.txt,
    // com1 or COM with a superscript 1 as devices.
    for (const std::string name :
         {"..", ".", "a/b", "a\\b", "C:", "C:model", "/root", "\\\\server",
          "Model.", "Model ", "CON", "nul.txt", "Com1", "LPT9.bin", "AUX .x",
          "COM\xC2\xB9", "com\xC2\xB2", "LPT\xC2\xB3.bin"}) {
        nlohmann::json entry = complete;
        entry["name"] = name;
        RequireContains(RequireThrows([&] { load(entry); }), "'name'");
    }
    for (const std::string file :
         {"../x", "/x", "a/../b", "./a", "a//b", "a/", "a\\b", "C:\\x",
          "C:x", "config.json.", "a /x", "a/CON", "prn.json",
          "a/COM\xC2\xB3", "lpt\xC2\xB9.json", "LPT\xC2\xB2/x"}) {
        nlohmann::json entry = complete;
        entry["files"] = {"config.json", file};
        RequireContains(RequireThrows([&] { load(entry); }), "files");
    }
    {
        nlohmann::json entry = complete;
        entry["xclbin_folder"] = "..";
        RequireContains(RequireThrows([&] { load(entry); }), "xclbin_folder");
    }
    {
        nlohmann::json entry = complete;
        entry["file_sources"] = {{"../tokenizer.json", {{"url", "x"}}}};
        RequireContains(RequireThrows([&] { load(entry); }), "file_sources");
    }
    {
        nlohmann::json entry = complete;
        entry["preferred_source"] = {{"name", "../Other"}};
        RequireContains(RequireThrows([&] { load(entry); }), "'name'");
        entry["preferred_source"] = {{"files", {"../x"}}};
        RequireContains(RequireThrows([&] { load(entry); }), "files");
    }

    // flm pull and flm run use the entry preferred_source merge-patches in,
    // so every field it sets is checked as the entry's own would be.
    for (const auto& [preferred, field] :
         std::vector<std::pair<nlohmann::json, std::string>>{
             {{{"xclbin_folder", ".."}}, "xclbin_folder"},
             {{{"file_sources", {{"../tokenizer.json", {{"url", "x"}}}}}},
              "file_sources"},
             {{{"files", nullptr}}, "files"},
             {{{"url", ""}}, "url"},
             {{{"backend_provider", ""}}, "backend_provider"},
             {{{"vlm", true}}, "frontend_provider"},
             {{{"preferred_source", nlohmann::json::object()}},
              "preferred_source"}}) {
        nlohmann::json entry = complete;
        entry["preferred_source"] = preferred;
        RequireContains(RequireThrows([&] { load(entry); }), field);
    }
    {
        nlohmann::json entry = complete;
        entry["preferred_source"] = {
            {"url", "https://example.com/amd"},
            {"files", {"config.json", "amd/model.onnx"}},
            {"details", {{"quantization_level", "Q4"}}}};
        load(entry);
    }
}

void test_overlay_cannot_reuse_another_catalogs_folder() {
    // flm remove deletes the entry's whole folder, so an extension entry that
    // reused a built-in folder would delete the built-in model's weights.
    const auto base = write_catalog(
        "flm_folder_base_catalog.json",
        {{"model_path", "models"},
         {"models",
          {{"base",
            {{"1b",
              {{"name", "Base-Model-NPU2"},
               {"details", {{"family", "base-frontend"}}}}}}}}}});
    const auto entry = [](const char* name) {
        return WithDownloadMetadata({{"name", name},
                                     {"backend", "flm"},
                                     {"backend_provider", "demo-provider"},
                                     {"default_context_length", 4096},
                                     {"max_prefill_len", 512},
                                     {"details", {{"family", "demo-frontend"}}}});
    };
    std::string base_path = base.string();
    std::string exe_dir = ".";
    const model_list::backend_provider_inventory_t providers = {
        {"base-frontend", {{"flm", {"builtin"}}}},
        {"demo-frontend", {{"flm", {"demo-provider"}}}}};
    const auto open = [&](const std::filesystem::path& overlay) {
        model_list models(
            base_path, exe_dir, "aie2p", {"flm"},
            {flm::catalog::load_model_catalog_overlay(overlay.string())},
            {"base-frontend", "demo-frontend"}, providers);
        return models.is_model_supported("demo:1b");
    };

    for (const char* name : {"Base-Model-NPU2", "base-model-npu2"}) {
        const auto overlay = write_catalog(
            "flm_folder_reuse_overlay.json",
            {{"schema_version", 1},
             {"models", {{"demo", {{"1b", entry(name)}}}}}});
        RequireContains(RequireThrows([&] { (void)open(overlay); }),
                        "uses model folder");
        std::filesystem::remove(overlay);
    }

    // Two sizes of one extension may share a folder, as built-in tags do.
    const auto shared = write_catalog(
        "flm_folder_shared_overlay.json",
        {{"schema_version", 1},
         {"models",
          {{"demo",
            {{"1b", entry("Demo-Model-NPU2")},
             {"2b", entry("Demo-Model-NPU2")}}}}}});
    TEST_REQUIRE(open(shared));
    std::filesystem::remove(shared);

    auto preferred = entry("Demo-Model-NPU2");
    preferred["preferred_source"] = {{"name", "base-model-npu2"}};
    const auto preferred_reuse = write_catalog(
        "flm_folder_preferred_reuse_overlay.json",
        {{"schema_version", 1}, {"models", {{"demo", {{"1b", preferred}}}}}});
    RequireContains(RequireThrows([&] { (void)open(preferred_reuse); }),
                    "uses model folder");
    std::filesystem::remove(preferred_reuse);
    std::filesystem::remove(base);
}

void test_overlay_preferred_source_selects_available_providers() {
    // A preferred_source may not switch to a backend provider or frontend
    // family the loaded extensions do not offer.
    const auto base = write_catalog(
        "flm_preferred_provider_base_catalog.json",
        {{"model_path", "models"},
         {"models",
          {{"base",
            {{"1b",
              {{"name", "Base-Model-NPU2"},
               {"details", {{"family", "base-frontend"}}}}}}}}}});
    std::string base_path = base.string();
    std::string exe_dir = ".";
    const model_list::backend_provider_inventory_t providers = {
        {"base-frontend", {{"flm", {"builtin"}}}},
        {"demo-frontend", {{"flm", {"demo-provider"}}}}};
    for (const auto& [preferred, message] :
         std::vector<std::pair<nlohmann::json, std::string>>{
             {nlohmann::json::object(), ""},
             {{{"backend_provider", "other-provider"}},
              "unavailable backend provider"},
             {{{"backend", "core"}}, "unavailable backend provider"},
             {{{"details", {{"family", "missing-frontend"}}}},
              "unregistered frontend family"}}) {
        auto demo = WithDownloadMetadata(
            {{"name", "Demo-Model-NPU2"},
             {"backend", "flm"},
             {"backend_provider", "demo-provider"},
             {"default_context_length", 4096},
             {"max_prefill_len", 512},
             {"details", {{"family", "demo-frontend"}}},
             {"preferred_source", preferred}});
        const auto overlay = write_catalog(
            "flm_preferred_provider_overlay.json",
            {{"schema_version", 1}, {"models", {{"demo", {{"1b", demo}}}}}});
        const auto open = [&] {
            model_list models(
                base_path, exe_dir, "aie2p", {"flm", "core"},
                {flm::catalog::load_model_catalog_overlay(overlay.string())},
                {"base-frontend", "demo-frontend"}, providers);
            return models.is_model_supported("demo:1b");
        };
        if (message.empty()) {
            TEST_REQUIRE(open());
        } else {
            RequireContains(RequireThrows(open), message);
        }
        std::filesystem::remove(overlay);
    }
    std::filesystem::remove(base);
}

void test_provider_inventory_preserves_all_implicit_builtin_models() {
    const model_list::backend_provider_inventory_t incomplete_inventory = {
        {"phi4", {{"flm", {"builtin"}}}}};

    for (const auto& [platform, backends] :
         std::vector<std::pair<std::string, std::vector<std::string>>>{
             {"aie2p", {"flm"}},
             {"dev04", {"flm", "core"}}}) {
        const auto baseline = open_catalog(platform, backends);
        const auto with_inventory =
            open_catalog(platform, backends, incomplete_inventory);
        TEST_REQUIRE(with_inventory.all_tags == baseline.all_tags);
    }
}

void test_platform_helpers() {
    TEST_REQUIRE(utils::parse_platform("aie2p") == utils::npu_platform::aie2p);
    TEST_REQUIRE(utils::parse_platform("dev04") == utils::npu_platform::dev04);
    TEST_REQUIRE(!utils::parse_platform("not-a-platform").has_value());
    TEST_REQUIRE(!utils::parse_platform("stx").has_value());
    TEST_REQUIRE(utils::parse_platform(utils::platform_id(utils::npu_platform::dev04)) ==
                 utils::npu_platform::dev04);
    TEST_REQUIRE(utils::default_npu_platform() == utils::npu_platform::aie2p);

    // corelib names the NPU; only its AIEM is dev04, and anything else,
    // no NPU included, is the default.
    TEST_REQUIRE(utils::platform_from_corelib_arch("AIEM") == utils::npu_platform::dev04);
    TEST_REQUIRE(utils::platform_from_corelib_arch("AIE2P") == utils::npu_platform::aie2p);
    TEST_REQUIRE(utils::platform_from_corelib_arch("") == utils::npu_platform::aie2p);

    // This hardware-free target does not define FLM_ENABLE_CORE, so the
    // compile-time stand-in is the shipping aie2p generation.
    TEST_REQUIRE(utils::get_device() == utils::default_npu_platform());
    TEST_REQUIRE(utils::get_device() == utils::get_device());
}

void test_shipped_catalog_is_well_formed() {
    const auto catalog = read_json(kCatalogPath);
    bool saw_core = false;

    for (const auto& [family, sizes] : catalog.at("models").items()) {
        const bool core_family = family.size() > 5 &&
                                 family.compare(family.size() - 5, 5, "-core") == 0;
        const bool gpu_family = family.ends_with("-oga-hipep");
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

            // An oga-hipep package runs on the GPU, so no NPU generation
            // decides whether it runs here.
            if (gpu_family) {
                if (entry.contains("supported_platforms")) {
                    throw std::runtime_error(tag + ": an oga-hipep package names no NPU generation");
                }
                continue;
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
            if (core_family != (named.count("dev04") != 0)) {
                throw std::runtime_error(
                    tag + ": the tag name and supported_platforms disagree "
                          "about which generation this package is for");
            }
            if (core_family) saw_core = true;
        }
    }

    // A corelib model is packaged differently from its FastFlowLM namesake, so
    // it has its own tag; if that stops being true, everything above is
    // checking a rule the catalog no longer follows.
    TEST_REQUIRE(saw_core);
    TEST_REQUIRE(catalog.at("models").contains("phi4-mini-it"));
    TEST_REQUIRE(catalog.at("models").contains("phi4-mini-it-core"));
}

}  // namespace

int main() {
    RunTest(test_stock_build_offers_phi4_with_its_own_artifacts,
            "stock build offers phi4 with its own artifacts");
    RunTest(test_a_corelib_build_offers_the_corelib_phi4,
            "a corelib build offers the corelib phi4");
    RunTest(test_the_tag_name_decides_the_backend,
            "the tag name decides the backend");
    RunTest(test_the_oga_hipep_suffix_names_its_backend,
            "the oga-hipep suffix names its backend");
    RunTest(test_a_catalog_can_filter_down_to_nothing,
            "a catalog can filter down to nothing");
    RunTest(test_an_entry_is_pruned_by_the_silicon_it_names,
            "an entry is pruned by the silicon it names");
    RunTest(test_missing_key_means_every_generation,
            "missing key means every generation");
    RunTest(test_extension_catalogs_merge_without_overrides,
            "extension catalogs merge without overrides");
    RunTest(test_extension_overlay_requires_explicit_backend_provider,
            "extension overlay requires explicit backend provider");
    RunTest(test_overlay_keeps_builtin_bare_tag_after_platform_filter,
            "overlay keeps built-in bare tag after platform filter");
    RunTest(test_overlay_requires_download_metadata,
            "overlay requires download metadata");
    RunTest(test_multimodal_overlay_requires_frontend_provider,
            "multimodal overlay requires frontend provider");
    RunTest(test_overlay_paths_stay_inside_the_model_folder,
            "overlay paths stay inside the model folder");
    RunTest(test_overlay_cannot_reuse_another_catalogs_folder,
            "overlay cannot reuse another catalog's folder");
    RunTest(test_overlay_preferred_source_selects_available_providers,
            "overlay preferred_source selects available providers");
    RunTest(test_provider_inventory_preserves_all_implicit_builtin_models,
            "provider inventory preserves all implicit built-in models");
    RunTest(test_platform_helpers, "platform helpers");
    RunTest(test_shipped_catalog_is_well_formed,
            "shipped catalog is well formed");
    std::cout << "All model_list platform tests passed\n";
    return 0;
}
