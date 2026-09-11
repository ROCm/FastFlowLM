#pragma once

#include <filesystem>

#if defined(FLM_CORELIB_TESTING)
#include <functional>
#include <utility>
#endif

namespace flm::file_access {

#if defined(FLM_CORELIB_TESTING)
using OpenObserver = std::function<void(const std::filesystem::path&)>;
inline OpenObserver open_observer;

inline void SetOpenObserver(OpenObserver observer) {
    open_observer = std::move(observer);
}

inline void ObserveOpen(const std::filesystem::path& path) {
    if (open_observer) open_observer(path);
}
#else
inline void ObserveOpen(const std::filesystem::path&) {}
#endif

} // namespace flm::file_access
