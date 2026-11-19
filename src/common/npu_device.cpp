/// \file npu_device.cpp
/// \brief Open the shared NPU device only when a model needs it.
#include "utils/utils.hpp"
#include "AutoModel/backend_contract.hpp"

#include <cstdlib>
#include <mutex>
#include <system_error>

#if defined(_WIN32) && defined(FLM_ENABLE_OGA_HIPEP)
#include <delayimp.h>

static HRESULT load_xrt_imports() {
    __try {
        return __HrLoadAllImportsForDll("xrt_coreutil.dll");
    } __except (GetExceptionCode() == VcppException(ERROR_SEVERITY_ERROR, ERROR_MOD_NOT_FOUND) ||
                GetExceptionCode() == VcppException(ERROR_SEVERITY_ERROR, ERROR_PROC_NOT_FOUND)
                    ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return HRESULT_FROM_WIN32(HRESULT_CODE(GetExceptionCode()));
    }
}
#endif

namespace utils {

bool valid_npu_power_mode(std::string_view mode) {
    return mode == "default" || mode == "powersaver" || mode == "balanced" ||
           mode == "performance" || mode == "turbo";
}

static flm_rt::device* open_npu_device() {
    try {
#if defined(_WIN32) && defined(FLM_ENABLE_OGA_HIPEP)
        const auto status = load_xrt_imports();
        if (FAILED(status)) {
            throw std::system_error(HRESULT_CODE(status), std::system_category(), "xrt_coreutil.dll");
        }
#endif
        static flm_rt::device device(0);
        return &device;
    } catch (const std::exception& error) {
        throw flm::backend::NpuUnavailable(std::string(flm::backend::kNoNpuDevice) + ": " + error.what());
    }
}

flm_rt::device* require_npu_device(flm_rt::device* shared_device,
                                   const std::string& power_mode) {
#ifdef _WIN32
    if (!valid_npu_power_mode(power_mode)) {
        throw std::invalid_argument("Invalid power mode: " + power_mode);
    }
#else
    (void)power_mode;
#endif
    auto* device = shared_device ? shared_device : open_npu_device();
#ifdef _WIN32
    static std::once_flag power_configured;
    std::call_once(power_configured, [&] {
        const std::string cmd = "cd \"C:\\Windows\\System32\\AMD\" && .\\xrt-smi.exe configure --pmode " +
                                power_mode + " > NUL 2>&1";
        header_print("FLM", "Configuring NPU Power Mode to " + power_mode +
                             (power_mode == "performance" ? " (flm default)" : ""));
        (void)std::system(cmd.c_str());
    });
#endif
    return device;
}

}  // namespace utils
