/// \file device_runtime.hpp
/// \brief Hardware-free runtime API for model-extension contract tests.
/// \note This intentionally models the small runtime surface used by buffer.hpp
///       so tests compile the production buffer layout and adapter code.
#pragma once

#include <cstddef>

using xclDeviceHandle = void*;

enum {
    XCL_BO_SYNC_BO_TO_DEVICE = 0,
    XCL_BO_SYNC_BO_FROM_DEVICE = 1,
};

namespace flm_rt {

class device {
public:
    operator xclDeviceHandle() const noexcept {
        return const_cast<device*>(this);
    }
};

class bo {
public:
    bo() = default;
    bo(device&, std::size_t size) : size_(size) {}

    template <typename T>
    T map() {
        return nullptr;
    }

    std::size_t size() const noexcept { return size_; }
    void sync(int) noexcept {}
    void flush() noexcept {}
    void invalidate() noexcept {}

private:
    std::size_t size_ = 0;
};

namespace ext {
using bo = flm_rt::bo;
}

}
