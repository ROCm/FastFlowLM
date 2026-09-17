/// \file npu_platform.cpp
/// \brief the NPU generation this build targets (stx vs aie_next)
#include "utils/npu_platform.hpp"

namespace utils {

std::optional<npu_platform> parse_platform(std::string_view text) {
    if (text == platform_id(npu_platform::stx)) return npu_platform::stx;
    if (text == platform_id(npu_platform::aie_next)) return npu_platform::aie_next;
    return std::nullopt;
}

}  // namespace utils
