#pragma once

#include "corelib/corelib_api.hpp"

#include <filesystem>
#include <memory>
#include <mutex>
#if defined(FLM_CORELIB_TESTING)
#include <functional>
#endif

namespace flm::corelib {

class CorelibRuntime final {
public:
    ~CorelibRuntime();
    static std::shared_ptr<CorelibRuntime> GetOrCreate(
        const std::filesystem::path& executable_dir);
    static std::shared_ptr<CorelibRuntime> CreateForTest(
        std::shared_ptr<CorelibApi> api);
    static void ShutdownProcess();
#if defined(FLM_CORELIB_TESTING)
    static void SetDestructionObserverForTest(std::function<void(bool)> observer);
#endif
    std::unique_lock<std::mutex> AcquireExecution();
    const std::shared_ptr<CorelibApi>& api() const noexcept;
    const std::filesystem::path& loaded_library_path() const noexcept;

private:
    explicit CorelibRuntime(std::shared_ptr<CorelibApi> api);
    static std::shared_ptr<CorelibRuntime> CreateReady(
        std::shared_ptr<CorelibApi> api);

    std::shared_ptr<CorelibApi> api_;
    std::mutex execution_mutex_;
};

}  // namespace flm::corelib
