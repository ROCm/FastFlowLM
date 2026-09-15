/// \file npu_platform.cpp
/// \brief NPU generation detection (aie2p vs aie4)
#include "utils/npu_platform.hpp"
#include "utils/debug_utils.hpp"

#include "nlohmann/json.hpp"

#include <cstdlib>
#include <mutex>
#include <stdexcept>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <libdrm/drm.h>
#include <cerrno>
#include <cstring>
#include <string>
#include "npu_utils/amdxdna_accel.h"
#endif

namespace utils {
namespace {

#ifndef _WIN32
/// \brief ask the amdxdna driver how many AIE columns the first NPU has
/// \return the column count, or nullopt when no amdxdna device answers
/// \note This is the only probe available on the HRX backend, which exposes no
///       platform query. It opens no XRT/HRX device, so it is also the cheap
///       path for commands that never touch the NPU.
std::optional<int> query_amdxdna_columns() noexcept {
    for (int i = 0; i < 16; ++i) {
        const std::string dev_name = "/dev/accel/accel" + std::to_string(i);
        const int fd = open(dev_name.c_str(), O_RDWR);
        if (fd < 0) {
            if (errno == ENOENT) break;
            continue;
        }
        amdxdna_drm_query_aie_metadata query_aie_metadata;
        std::memset(&query_aie_metadata, 0, sizeof(query_aie_metadata));
        amdxdna_drm_get_info get_info = {
            .param = DRM_AMDXDNA_QUERY_AIE_METADATA,
            .buffer_size = sizeof(query_aie_metadata),
            .buffer = (unsigned long)&query_aie_metadata,
        };
        const bool ok = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info) == 0;
        close(fd);
        if (ok && query_aie_metadata.cols > 0) {
            return static_cast<int>(query_aie_metadata.cols);
        }
    }
    return std::nullopt;
}
#endif

/// \brief read FLM_PLATFORM, warning once about an unusable value
/// \return the override, or nullopt when unset or unparseable
std::optional<npu_platform> platform_override() {
    const char* configured = std::getenv("FLM_PLATFORM");
    if (!configured || !*configured) return std::nullopt;
    const auto parsed = parse_platform(configured);
    if (!parsed) {
        header_print_r("ERROR", "Ignoring unknown FLM_PLATFORM '"
                                    << configured << "'; expected aie2p or aie4");
    }
    return parsed;
}

}  // namespace

std::optional<npu_platform> parse_platform(std::string_view text) {
    if (text == platform_id(npu_platform::aie2p)) return npu_platform::aie2p;
    if (text == platform_id(npu_platform::aie4)) return npu_platform::aie4;
    return std::nullopt;
}

npu_platform platform_from_columns(int columns) {
    // Medusa Point exposes 3 columns; every aie2p part exposes 8. Treat anything
    // narrow as aie4 so a future 4-column part is not silently misread as aie2p.
    return columns <= 4 ? npu_platform::aie4 : npu_platform::aie2p;
}

bool is_known_column_count(int columns) {
    return columns == 3 || columns >= 8;
}

#ifndef FLM_USE_HRX
int get_platform_columns(const flm_rt::device& device) {
    const auto platform = nlohmann::json::parse(
        device.get_info<flm_rt::info::device::platform>());
    const auto columns_text = platform.at("platforms").at(0)
        .at("static_region").at("total_columns").get<std::string>();
    size_t parsed_length = 0;
    const int columns = std::stoi(columns_text, &parsed_length);
    if (parsed_length != columns_text.size() || columns <= 0) {
        throw std::runtime_error("Invalid platform total_columns: " + columns_text);
    }
    return columns;
}
#endif

std::optional<int> probe_npu_columns(const flm_rt::device* device) noexcept {
#ifndef FLM_USE_HRX
    if (device != nullptr) {
        try {
            return get_platform_columns(*device);
        } catch (...) {
            // Fall through to the driver probe below; an unreadable platform
            // description must not be fatal for `flm list` and friends.
        }
    }
#else
    (void)device;
#endif
#ifndef _WIN32
    return query_amdxdna_columns();
#else
    return std::nullopt;
#endif
}

npu_platform detect_npu_platform(const flm_rt::device* device,
                                 std::string* detail) noexcept {
    static std::once_flag once;
    static npu_platform cached = default_npu_platform();
    static std::string cached_detail;

    std::call_once(once, [device]() {
        try {
            if (const auto forced = platform_override()) {
                cached = *forced;
                cached_detail = "forced by FLM_PLATFORM";
                return;
            }
            if (const auto columns = probe_npu_columns(device)) {
                cached = platform_from_columns(*columns);
                cached_detail = std::to_string(*columns) + " AIE columns";
                return;
            }
            cached_detail = "no NPU detected, assuming " +
                            std::string(platform_id(default_npu_platform()));
        } catch (...) {
            cached = default_npu_platform();
            cached_detail = "detection failed, assuming " +
                            std::string(platform_id(default_npu_platform()));
        }
    });

    if (detail != nullptr) *detail = cached_detail;
    return cached;
}

}  // namespace utils
