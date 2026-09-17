/// \file corelib_device.hpp
/// \brief Access to the xrt::device that ryzenai-corelib owns.
/// \note  FastFlowLM consumes corelib through the C ABI in <ryzenai/corelib.h>
///        (see corelib_api.hpp), which has no device accessor: the device lives
///        behind corelib's C++ entry point ryzenai::corelib::GetDevice(). The
///        declaration is reproduced here rather than pulled from a corelib C++
///        header so that this tree keeps depending on exactly one corelib header;
///        it resolves at link time against the statically linked corelib
///        (RYZENAI_CORELIB_STATIC), so a signature drift is a link error, not a
///        silent mismatch.
#pragma once

#if defined(FLM_USE_HRX)
#error "FLM_ENABLE_AIE4 requires the XRT backend (FLM_USE_HRX=OFF)"
#endif

#include "device_runtime.hpp"

namespace ryzenai::corelib {

/// \brief the device corelib initialized; valid until ryzenai_corelib_cleanup()
/// \return corelib's device, shared with every FLM engine
const xrt::device& GetDevice();

}  // namespace ryzenai::corelib
