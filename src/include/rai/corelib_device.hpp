/// \file corelib_device.hpp
/// \brief Access to the NPU device that ryzenai-corelib owns.
/// \note  FastFlowLM consumes corelib through the C ABI in <ryzenai/corelib.h>
///        (see corelib_api.hpp), and ryzenai_corelib_get_device() is the device
///        accessor in it. corelib's own ryzenai::corelib::GetDevice() is an
///        inline wrapper around that same call rather than an exported entry
///        point, so there is nothing to link against and no reason to reach for
///        the C++ layer: going through CorelibApi keeps every corelib call in
///        this tree on one resolved function table.
#pragma once

#if defined(FLM_USE_HRX)
#error "FLM_ENABLE_RAI requires the XRT backend (FLM_USE_HRX=OFF)"
#endif

#include "device_runtime.hpp"
#include "rai/corelib_runtime.hpp"

namespace flm::corelib {

/// \brief the device corelib dispatches on, shared with every engine here
/// \param runtime an initialized corelib runtime
/// \return corelib's device, or nullptr when this machine has no NPU
/// \note Valid until ryzenai_corelib_cleanup(); the process runtime outlives
///       every use of the returned pointer.
/// \note flm_rt is an alias for xrt on this path, so corelib's device already
///       has the type the engines take. corelib hands it out const and the
///       engines want it mutable; the constness is cast away rather than a
///       second device opened, because a buffer object created against a
///       different xrt::device for the same NPU binds without error and then
///       never completes.
inline flm_rt::device* SharedDevice(const CorelibRuntime& runtime) noexcept {
    const void* device = runtime.api()->functions().get_device();
    if (device == nullptr) {
        return nullptr;
    }
    return const_cast<flm_rt::device*>(
        reinterpret_cast<const flm_rt::device*>(device));
}

}  // namespace flm::corelib
