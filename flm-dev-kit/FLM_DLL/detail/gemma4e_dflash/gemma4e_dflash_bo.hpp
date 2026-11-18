// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Runtime-neutral helpers for gemma4e_dflash (XRT and HRX builds).
#pragma once
#include <cstddef>
#include "buffer.hpp"

namespace gemma4e_dflash_bo {
/// \brief push host writes of the byte range [off, off+len) of `b` to the device now, without touching the rest of the buffer
/// \note XRT: a ranged bo sync. HRX: a ranged cache flush of the persistent host mapping (the buffer's dirty bits are left alone:
///       the range is coherent after the flush and the rest of the buffer is not affected, so a later whole-buffer flush is not needed)
template <typename T>
inline void flush_range(buffer<T>& b, size_t off_bytes, size_t len_bytes) {
    if (len_bytes == 0) return;
#if defined(FLM_USE_HRX)
    hrx::hrx_report(hrx_buffer_flush_range(b.bo().handle(), off_bytes, len_bytes), "gemma4e_dflash flush_range");
#else
    b.bo().sync(XCL_BO_SYNC_BO_TO_DEVICE, len_bytes, off_bytes);
#endif
}
}  // namespace gemma4e_dflash_bo
