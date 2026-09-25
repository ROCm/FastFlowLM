/// \file flm_runtime.hpp
/// \brief Neutral NPU-runtime namespace alias (flm_rt) selecting the active
///        backend at build time. FLM_USE_HRX=ON maps flm_rt to the hrx C++
///        shim (over libhrx); otherwise flm_rt maps to XRT. Engine sources use
///        flm_rt:: so a single tree compiles against either runtime.
#pragma once

#if defined(FLM_USE_HRX)
#include "hrx_cpp/hrx_cpp.hpp"
namespace flm_rt = hrx;
#else
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/experimental/xrt_kernel.h"
namespace flm_rt = xrt;
#endif
