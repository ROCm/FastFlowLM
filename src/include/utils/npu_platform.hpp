/// \file npu_platform.hpp
/// \brief the NPU generation this build targets (stx vs aie_next)
/// \note  A build carries engines for exactly one generation: the two share
///        none, and FLM_ENABLE_RAI picks which at compile time, so the
///        generation is a build-time fact rather than something to go and ask
///        the hardware. The catalog in model_list.json declares which
///        generations each model supports, and model_list prunes itself to the
///        one this build was made for. Which *kernels* serve a generation is a
///        separate question with its own vocabulary -- see kFlmBackendId and
///        kRaiBackendId in AutoModel/model_backend.hpp.
/// \note  aie_next names silicon that has not been announced. When it ships,
///        this enumerator, the string platform_id() returns, and the catalog
///        key in model_list.json are the whole of the rename; nothing on disk
///        and nothing a user has downloaded is named after it.
#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace utils {

/// \brief NPU generation an FLM build targets
enum class npu_platform {
    stx,      ///< Strix / Krackan Point
    aie_next  ///< the next NPU generation, not yet announced
};

/// \brief the catalog id for a platform, as written in model_list.json
/// \param platform the platform
/// \return "stx" or "aie_next"
constexpr std::string_view platform_id(npu_platform platform) {
    return platform == npu_platform::aie_next ? std::string_view("aie_next")
                                              : std::string_view("stx");
}

/// \brief the platform every entry is assumed to support when it says nothing
constexpr npu_platform default_npu_platform() { return npu_platform::stx; }

/// \brief the NPU generation this build has engines for
/// \return aie_next when built with FLM_ENABLE_RAI, stx otherwise
/// \note There is no binary that carries both engines, so this is the whole of
///       platform selection: no probe, and nothing for a user to configure
///       beyond choosing the build that matches their machine.
constexpr npu_platform build_npu_platform() {
#ifdef FLM_ENABLE_RAI
    return npu_platform::aie_next;
#else
    return npu_platform::stx;
#endif
}

/// \brief parse a catalog platform id
/// \param text the id, e.g. "aie_next"
/// \return the platform, or nullopt when the id is unknown
std::optional<npu_platform> parse_platform(std::string_view text);

}  // namespace utils
