/// \file npu_platform.cpp
/// \brief the NPU generation this build targets (aie2p vs aie4)
#include "utils/npu_platform.hpp"

namespace utils {

std::optional<npu_platform> parse_platform(std::string_view text) {
    if (text == platform_id(npu_platform::aie2p)) return npu_platform::aie2p;
    if (text == platform_id(npu_platform::aie4)) return npu_platform::aie4;
    return std::nullopt;
}

}  // namespace utils
