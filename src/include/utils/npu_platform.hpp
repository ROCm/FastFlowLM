/// \file npu_platform.hpp
/// \brief the NPU generation a build runs on (aie2p vs aie_next)
/// \note  The generation is a property of the silicon, not of the build: it
///        says which model artifacts this machine can run at all, and the
///        catalog in model_list.json names one on every entry. Which *kernels*
///        execute them is a second, independent axis with its own vocabulary --
///        see kFlmBackendId and kRaiBackendId in AutoModel/model_backend.hpp --
///        and neither one implies the other.
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
    aie2p,    ///< the AIE2P generation: Strix / Strix Halo / Krackan Point
    aie_next  ///< the next NPU generation, not yet announced
};

/// \brief the catalog id for a platform, as written in model_list.json
/// \param platform the platform
/// \return "aie2p" or "aie_next"
constexpr std::string_view platform_id(npu_platform platform) {
    return platform == npu_platform::aie_next ? std::string_view("aie_next")
                                              : std::string_view("aie2p");
}

/// \brief the generation assumed when nothing has read the device
/// \note Until the probe in get_device() is real this is not a default but the
///       whole answer, so it decides what every install can see. It is
///       aie_next while the corelib path is what is being brought up: that is
///       the generation corelib's kernels are built for, and an install that
///       claimed aie2p could not offer them at all.
/// \note The cost is exact and intended: every entry in model_list.json except
///       the corelib one names aie2p, so a build that does not link corelib
///       offers nothing here. That is not a broken install, it is an install
///       for silicon none of its models were built for, and model_list says so
///       in one line. Set this back to aie2p to get those 42 models back.
constexpr npu_platform default_npu_platform() { return npu_platform::aie_next; }

/// \brief ask the machine which NPU generation it has
/// \return the generation of the NPU in this host
/// \note A stand-in, and deliberately the dumbest one that can be written: the
///       real probe reads the device and is not part of this tree, so until it
///       lands this returns default_npu_platform() and consults nothing at all
///       -- not the environment, and not which kernels were linked. A build
///       that links corelib is still a build on whatever silicon it is running
///       on, and an install that offers a model it cannot run is worse than one
///       that offers nothing. The call sites need no change when the probe
///       arrives: they already treat the generation as a run-time answer.
npu_platform get_device();

/// \brief parse a catalog platform id
/// \param text the id, e.g. "aie_next"
/// \return the platform, or nullopt when the id is unknown
std::optional<npu_platform> parse_platform(std::string_view text);

}  // namespace utils
