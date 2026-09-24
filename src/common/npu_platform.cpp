/// \file npu_platform.cpp
/// \brief which NPU generation this host has (aie2p vs aie_next)
#include "utils/npu_platform.hpp"

namespace utils {

npu_platform get_device() {
    // No device read here: the probe that tells the generations apart is not in
    // this tree, and nothing else in the build is allowed to stand in for it.
    // Linking corelib in particular is not evidence -- which kernels were
    // compiled is a fact about the build, the generation is a fact about the
    // machine, and letting the first answer the second is how the two axes get
    // quietly welded back together.
    return default_npu_platform();
}

std::optional<npu_platform> parse_platform(std::string_view text) {
    if (text == platform_id(npu_platform::aie2p)) return npu_platform::aie2p;
    if (text == platform_id(npu_platform::aie_next)) return npu_platform::aie_next;
    return std::nullopt;
}

}  // namespace utils
