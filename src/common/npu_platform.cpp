/// \file npu_platform.cpp
/// \brief which NPU generation this host has (aie2p vs dev04)
#include "utils/npu_platform.hpp"

#if defined(FLM_ENABLE_CORE)
#include "core/corelib_runtime.hpp"
#include "utils/utils.hpp"
#elif defined(FLM_NPU_UTIL)
/// flm_util (lib/<runtime>): "aie2p", "dev04", or "" when it knows no NPU here
extern "C" __declspec(dllimport) const char* flm_util_npu_platform(void);
#endif

namespace utils {

npu_platform get_device() {
#if defined(FLM_ENABLE_CORE)
    static const npu_platform platform = platform_from_corelib_arch(
        flm::corelib::CorelibRuntime::NpuArch(get_executable_directory()));
    return platform;
#elif defined(FLM_NPU_UTIL)
    static const npu_platform platform =
        parse_platform(flm_util_npu_platform()).value_or(default_npu_platform());
    return platform;
#else
    return default_npu_platform();
#endif
}

std::optional<npu_platform> parse_platform(std::string_view text) {
    if (text == platform_id(npu_platform::aie2p)) return npu_platform::aie2p;
    if (text == platform_id(npu_platform::dev04)) return npu_platform::dev04;
    return std::nullopt;
}

}  // namespace utils
