/// \file npu_platform.hpp
/// \brief the NPU generation a build runs on (aie2p vs dev04)
/// \note  The generation is a property of the silicon, not of the build: it
///        says which model artifacts this machine can run at all, and the
///        catalog in model_list.json names one on every entry. Which *kernels*
///        execute them is a second, independent axis with its own vocabulary --
///        see kFlmBackendId and kCoreBackendId in AutoModel/model_backend.hpp --
///        and neither one implies the other.
/// \note  dev04 names silicon that has not been announced. When it ships,
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
    aie2p,    ///< the AIE2P generation: Strix / Strix Halo / Krackan Point
    dev04     ///< the next NPU generation, not yet announced
};

/// \brief the catalog id for a platform, as written in model_list.json
/// \param platform the platform
/// \return "aie2p" or "dev04"
constexpr std::string_view platform_id(npu_platform platform) {
    return platform == npu_platform::dev04 ? std::string_view("dev04")
                                              : std::string_view("aie2p");
}

/// \brief the generation assumed when nothing identifies one
constexpr npu_platform default_npu_platform() { return npu_platform::aie2p; }

/// \brief the generation corelib's name for this machine's NPU stands for
/// \param arch what corelib reports: "AIEM", "AIE2P", or "" for none it knows
constexpr npu_platform platform_from_corelib_arch(std::string_view arch) {
    return arch == "AIEM" ? npu_platform::dev04 : default_npu_platform();
}

/// \brief ask the machine which NPU generation it has
/// \return the generation of the NPU in this host
/// \note A build that links corelib asks corelib, which reads the device
///       itself: FastFlowLM keeps no table of devices. Every dev04 model is a
///       corelib one, so a build without corelib, or one whose corelib cannot
///       be loaded, answers default_npu_platform(). Asked once per process.
npu_platform get_device();

/// \brief parse a catalog platform id
/// \param text the id, e.g. "dev04"
/// \return the platform, or nullopt when the id is unknown
std::optional<npu_platform> parse_platform(std::string_view text);

}  // namespace utils
