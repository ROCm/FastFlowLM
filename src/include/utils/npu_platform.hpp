/// \file npu_platform.hpp
/// \brief NPU generation detection (aie2p vs aie4)
/// \note  The NPU generation is identified by its AIE column count:
///          8 columns -> aie2p (Strix / Krackan Point)
///          3 columns -> aie4  (Medusa Point)
///        The catalog in model_list.json declares which generations each model
///        supports, and model_list prunes itself to the detected generation.
#pragma once

#include "device_runtime.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace utils {

/// \brief NPU generation the running machine exposes
enum class npu_platform {
    aie2p,  ///< Strix / Krackan Point, 8 AIE columns
    aie4    ///< Medusa Point, 3 AIE columns
};

/// \brief the catalog id for a platform, as written in model_list.json
/// \param platform the platform
/// \return "aie2p" or "aie4"
constexpr std::string_view platform_id(npu_platform platform) {
    return platform == npu_platform::aie4 ? std::string_view("aie4")
                                          : std::string_view("aie2p");
}

/// \brief the platform every entry is assumed to support when it says nothing
constexpr npu_platform default_npu_platform() { return npu_platform::aie2p; }

/// \brief parse a catalog platform id
/// \param text the id, e.g. "aie4"
/// \return the platform, or nullopt when the id is unknown
std::optional<npu_platform> parse_platform(std::string_view text);

/// \brief map an AIE column count onto a generation
/// \param columns the column count reported by the NPU
/// \return aie4 for narrow (<= 4 column) parts, aie2p otherwise
npu_platform platform_from_columns(int columns);

/// \brief whether a column count belongs to a generation FLM knows about
/// \param columns the column count reported by the NPU
/// \return true for 3 (aie4) or >= 8 (aie2p)
bool is_known_column_count(int columns);

/// \brief read the AIE column count out of the platform description
/// \param device an open NPU device
/// \return the column count
/// \note Only available on the XRT backend; HRX exposes no platform query, so
///       on HRX builds the Linux amdxdna ioctl path in probe_npu_columns is used
///       instead.
#ifndef FLM_USE_HRX
int get_platform_columns(const flm_rt::device& device);
#endif

/// \brief best-effort read of the NPU's AIE column count
/// \param device an open NPU device, or nullptr when none could be opened
/// \return the column count, or nullopt when no NPU could be reached
/// \note Never throws; a machine without an NPU or driver simply gets nullopt.
std::optional<int> probe_npu_columns(const flm_rt::device* device) noexcept;

/// \brief detect which NPU generation this machine is
/// \param device an open NPU device, or nullptr when none could be opened
/// \param detail optional; receives a short human-readable reason for the result
/// \return the detected platform, falling back to aie2p when nothing is readable
/// \note Never throws. FLM_PLATFORM ("aie2p"/"aie4") overrides detection and is
///       intended for testing only. The result is computed once per process.
npu_platform detect_npu_platform(const flm_rt::device* device,
                                 std::string* detail = nullptr) noexcept;

}  // namespace utils
