// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/npu/vision_bo.hpp
/// \brief Sync the part of a staging BO a dispatch actually uses.
///
/// The tower's staging BOs are grow-only and shared: design A's x and y are
/// sized by the widest op of the largest image (block fc2's x, block fc1's y:
/// ~66 MB each at 720p), and every one of the 113 design-A dispatches then
/// flushed and invalidated BOTH IN FULL -- `buffer::sync_*_device()` takes no
/// range. That was 145 ms of a 1270 ms 720p encode. A range sync costs what the
/// op moves.
///
/// HRX's bo has no ranged flush, so that build keeps the whole-BO sync.
#ifndef __MINICPM_V_4_7_NPU_VISION_BO_HPP__
#define __MINICPM_V_4_7_NPU_VISION_BO_HPP__

#include "buffer.hpp"

#include <cstddef>

namespace mcvm_npu {

template <typename T>
inline void sync_to_device_n(buffer<T>& b, size_t n) {
#if defined(FLM_USE_HRX)
    (void)n;
    b.sync_to_device();
#else
    if (n == 0) return;
    b.bo().sync(XCL_BO_SYNC_BO_TO_DEVICE, n * sizeof(T), 0);
#endif
}

template <typename T>
inline void sync_from_device_n(buffer<T>& b, size_t n) {
#if defined(FLM_USE_HRX)
    (void)n;
    b.sync_from_device();
#else
    if (n == 0) return;
    b.bo().sync(XCL_BO_SYNC_BO_FROM_DEVICE, n * sizeof(T), 0);
#endif
}

}  // namespace mcvm_npu

#endif  // __MINICPM_V_4_7_NPU_VISION_BO_HPP__
