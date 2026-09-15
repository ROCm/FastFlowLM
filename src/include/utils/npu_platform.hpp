/// \file npu_platform.hpp
/// \brief the NPU generation this build targets (aie2p vs aie4)
/// \note  The two generations do not share an engine: aie2p (Strix / Krackan
///        Point) runs FastFlowLM's own NPU kernels, aie4 runs ryzenai-corelib,
///        and a build carries exactly one of the two. FLM_ENABLE_AIE4 selects
///        which, so the generation is a build-time fact rather than something
///        to go and ask the hardware. The catalog in model_list.json declares
///        which generations each model supports, and model_list prunes itself
///        to the one this build was made for.
#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace utils {

/// \brief NPU generation an FLM build targets
enum class npu_platform {
    aie2p,  ///< Strix / Krackan Point
    aie4    ///< driven through ryzenai-corelib
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

/// \brief the NPU generation this build has engines for
/// \return aie4 when built with FLM_ENABLE_AIE4, aie2p otherwise
/// \note There is no binary that carries both engines, so this is the whole of
///       platform selection: no probe, and nothing for a user to configure
///       beyond choosing the build that matches their machine.
constexpr npu_platform build_npu_platform() {
#ifdef FLM_ENABLE_AIE4
    return npu_platform::aie4;
#else
    return npu_platform::aie2p;
#endif
}

/// \brief parse a catalog platform id
/// \param text the id, e.g. "aie4"
/// \return the platform, or nullopt when the id is unknown
std::optional<npu_platform> parse_platform(std::string_view text);

}  // namespace utils
