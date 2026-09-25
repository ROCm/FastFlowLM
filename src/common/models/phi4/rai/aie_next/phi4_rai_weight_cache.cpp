/// \file phi4_rai_weight_cache.cpp
/// \brief On-disk cache of the packed weights
#include "models/phi4/rai/aie_next/phi4_rai_weight_cache.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <string_view>

namespace flm::phi4 {
namespace {

constexpr const char* kIndexName = "phi4-rai-weights.json";
constexpr const char* kDataName = "phi4-rai-weights.bin";
/// bumped when the on-disk layout changes in a way older indexes cannot express
constexpr int kIndexFormat = 1;

bool IsDisabled(std::string_view value) {
    return value == "0" || value == "off" || value == "OFF" || value == "false";
}

}  // namespace

bool WeightCacheKey::operator==(const WeightCacheKey& other) const {
    return gguf_size == other.gguf_size &&
           gguf_write_time == other.gguf_write_time &&
           corelib_major == other.corelib_major &&
           corelib_minor == other.corelib_minor &&
           corelib_patch == other.corelib_patch &&
           group_size == other.group_size &&
           weight_count == other.weight_count;
}

std::optional<std::filesystem::path> WeightCacheDirectory(
    const std::filesystem::path& model_path) {
    const char* configured = std::getenv("FLM_RAI_WEIGHT_CACHE");
    if (configured && *configured) {
        if (IsDisabled(configured)) return std::nullopt;
        return std::filesystem::path(configured);
    }
    return model_path;
}

std::filesystem::path WeightCacheDataPath(const std::filesystem::path& directory) {
    return directory / kDataName;
}

WeightCacheKey MakeWeightCacheKey(const std::filesystem::path& gguf_path,
                                  std::uint32_t corelib_major,
                                  std::uint32_t corelib_minor,
                                  std::uint32_t corelib_patch,
                                  std::uint32_t group_size,
                                  std::uint64_t weight_count) {
    WeightCacheKey key;
    key.corelib_major = corelib_major;
    key.corelib_minor = corelib_minor;
    key.corelib_patch = corelib_patch;
    key.group_size = group_size;
    key.weight_count = weight_count;
    std::error_code error;
    const auto size = std::filesystem::file_size(gguf_path, error);
    if (!error) key.gguf_size = size;
    const auto written = std::filesystem::last_write_time(gguf_path, error);
    if (!error) key.gguf_write_time = written.time_since_epoch().count();
    return key;
}

std::uint64_t RemoveWeightCache(const std::filesystem::path& directory) {
    std::uint64_t reclaimed = 0;
    const std::string index(kIndexName);
    const std::string data(kDataName);
    for (const auto& name : {data, data + ".tmp", index, index + ".tmp"}) {
        std::error_code error;
        const auto path = directory / name;
        const auto size = std::filesystem::file_size(path, error);
        if (error) continue;
        if (std::filesystem::remove(path, error) && !error) reclaimed += size;
    }
    return reclaimed;
}

std::optional<WeightCacheIndex> ReadWeightCacheIndex(
    const std::filesystem::path& directory, const WeightCacheKey& expected) {
    try {
        const auto index_path = directory / kIndexName;
        std::ifstream input(index_path, std::ios::binary);
        if (!input) return std::nullopt;
        const auto document = nlohmann::json::parse(input, nullptr, false);
        if (document.is_discarded()) return std::nullopt;
        if (document.value("format", 0) != kIndexFormat) return std::nullopt;

        WeightCacheIndex index;
        index.key.gguf_size = document.value("gguf_size", std::uint64_t{0});
        index.key.gguf_write_time = document.value("gguf_write_time", std::int64_t{0});
        index.key.corelib_major = document.value("corelib_major", std::uint32_t{0});
        index.key.corelib_minor = document.value("corelib_minor", std::uint32_t{0});
        index.key.corelib_patch = document.value("corelib_patch", std::uint32_t{0});
        index.key.group_size = document.value("group_size", std::uint32_t{0});
        index.key.weight_count = document.value("weight_count", std::uint64_t{0});
        if (!(index.key == expected)) return std::nullopt;

        const auto spans = document.find("spans");
        if (spans == document.end() || !spans->is_array()) return std::nullopt;
        if (spans->size() != expected.weight_count) return std::nullopt;
        index.spans.reserve(spans->size());
        for (const auto& span : *spans) {
            if (!span.is_object()) return std::nullopt;
            CachedWeightSpan entry;
            entry.offset = span.value("offset", std::uint64_t{0});
            entry.size = span.value("size", std::uint64_t{0});
            if (entry.size == 0) return std::nullopt;
            index.spans.push_back(entry);
        }

        // The data file has to be at least as long as the last span claims, or
        // the index is describing a file that was truncated under it.
        std::error_code error;
        const auto data_size =
            std::filesystem::file_size(WeightCacheDataPath(directory), error);
        if (error) return std::nullopt;
        for (const auto& span : index.spans) {
            if (span.offset + span.size > data_size) return std::nullopt;
        }
        return index;
    } catch (...) {
        // A damaged cache is a miss, never a failed load.
        return std::nullopt;
    }
}

bool WriteWeightCacheIndex(const std::filesystem::path& directory,
                           const WeightCacheKey& key,
                           const std::vector<CachedWeightSpan>& spans) {
    try {
        nlohmann::json document;
        document["format"] = kIndexFormat;
        document["gguf_size"] = key.gguf_size;
        document["gguf_write_time"] = key.gguf_write_time;
        document["corelib_major"] = key.corelib_major;
        document["corelib_minor"] = key.corelib_minor;
        document["corelib_patch"] = key.corelib_patch;
        document["group_size"] = key.group_size;
        document["weight_count"] = key.weight_count;
        auto array = nlohmann::json::array();
        for (const auto& span : spans) {
            array.push_back({{"offset", span.offset}, {"size", span.size}});
        }
        document["spans"] = std::move(array);

        // Write to a temporary and rename, so a reader never sees a half index
        // pointing into a data file it does not describe.
        const auto final_path = directory / kIndexName;
        const auto temporary = directory / (std::string(kIndexName) + ".tmp");
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) return false;
            output << document.dump();
            if (!output) return false;
        }
        std::error_code error;
        std::filesystem::rename(temporary, final_path, error);
        if (error) {
            std::filesystem::remove(temporary, error);
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace flm::phi4
