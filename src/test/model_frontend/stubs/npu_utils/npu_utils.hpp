/// \file npu_utils.hpp
/// \brief Hardware-free stand-in for the NPU xclbin manager.
/// \note Shadows the real header so automodel.cpp builds without a device
///       runtime. Extension providers that ask for an xclbin manager are not
///       exercised by these tests.
#pragma once

#include <device_runtime.hpp>

enum class npu_device { device_npu2 };

class npu_xclbin_manager {
public:
    npu_xclbin_manager(npu_device, flm_rt::device*, bool) {}
};
