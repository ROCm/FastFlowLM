#pragma once

/// \file phi4_corelib_weight_cache.hpp
/// \brief On-disk cache of the packed weights, so requantization is paid once
/// \note Requantizing the 161 weights from Q8_0 is effectively the whole of
///       model load. corelib can hand the packed bytes back
///       (ryzenai_corelib_weights_copy_data) and load them again later
///       (..._weights_create_from_file, which maps rather than copies), so the
///       refit need only happen the first time a given GGUF is loaded.
///
///       corelib's own guidance is that caching only pays for a large blob and
///       that for an ordinary matmul packing can be faster than reading a
///       precomputed one back. Whether it pays here is a measurement, which is
///       why the cache reports what it cost.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace flm::phi4 {

/// \brief where one packed weight lives inside the cache file
struct CachedWeightSpan {
    std::uint64_t offset{};
    std::uint64_t size{};
};

/// \brief the identity a cache is only valid for
/// \note Everything that can change the packed bytes has to be in here. The
///       GGUF is identified by size and write time rather than by content: the
///       content hash of a 4 GB file costs more than the packing the cache
///       exists to avoid, which would defeat it. corelib independently rejects
///       a slice whose length is not exactly what the descriptor packs to, so a
///       stale-but-plausible cache cannot quietly become wrong weights.
struct WeightCacheKey {
    std::uint64_t gguf_size{};
    std::int64_t gguf_write_time{};
    std::uint32_t corelib_major{}, corelib_minor{}, corelib_patch{};
    std::uint32_t group_size{};
    std::uint64_t weight_count{};

    bool operator==(const WeightCacheKey& other) const;
};

/// \brief an index read back from disk, when one is present and current
struct WeightCacheIndex {
    WeightCacheKey key;
    std::vector<CachedWeightSpan> spans;
};

/// \brief the cache directory for a model, or nullopt when caching is disabled
/// \param model_path the directory the GGUF lives in
/// \return the directory to hold the cache, honouring FLM_AIE4_WEIGHT_CACHE
/// \note Unset means the cache sits beside the model. Set to a path redirects
///       it; set to "0" or "off" disables caching entirely.
std::optional<std::filesystem::path> WeightCacheDirectory(
    const std::filesystem::path& model_path);

/// \brief compute the identity of the cache a given GGUF would produce
WeightCacheKey MakeWeightCacheKey(const std::filesystem::path& gguf_path,
                                  std::uint32_t corelib_major,
                                  std::uint32_t corelib_minor,
                                  std::uint32_t corelib_patch,
                                  std::uint32_t group_size,
                                  std::uint64_t weight_count);

/// \brief read the index beside a cache file, if it matches the expected key
/// \return the index, or nullopt when absent, unreadable or stale
/// \note Never throws: a damaged cache is a cache miss, not a failed load.
std::optional<WeightCacheIndex> ReadWeightCacheIndex(
    const std::filesystem::path& directory, const WeightCacheKey& expected);

/// \brief write the index describing a freshly written cache file
/// \return true when the index landed, false when it could not be written
/// \note Written last and renamed into place, so a cache file without a
///       matching index is never mistaken for a usable one.
bool WriteWeightCacheIndex(const std::filesystem::path& directory,
                           const WeightCacheKey& key,
                           const std::vector<CachedWeightSpan>& spans);

/// \brief the cache file itself, beside its index
std::filesystem::path WeightCacheDataPath(const std::filesystem::path& directory);

/// \brief delete a cache that is not going to be used
/// \param directory the cache directory
/// \return how many bytes were reclaimed
/// \note Called before repacking, so a cache that no longer matches its GGUF
///       stops occupying two gigabytes from the moment it is known to be
///       useless rather than from whenever the next write happens to succeed.
///       Also clears the temporaries an interrupted write leaves behind.
///       Never throws: failing to delete is not a reason to fail a load.
std::uint64_t RemoveWeightCache(const std::filesystem::path& directory);

}  // namespace flm::phi4
