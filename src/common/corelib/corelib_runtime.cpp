#include "corelib/corelib_runtime.hpp"

#include <stdexcept>
#include <utility>

namespace flm::corelib {
namespace {
std::mutex process_mutex;
std::shared_ptr<CorelibRuntime> process_runtime;
}

CorelibRuntime::CorelibRuntime(std::shared_ptr<CorelibApi> api)
    : api_(std::move(api)) {}

std::shared_ptr<CorelibRuntime> CorelibRuntime::CreateReady(
    std::shared_ptr<CorelibApi> api) {
    if (!api) throw std::invalid_argument("corelib API is null");
    api->Check(api->functions().selftest_dependencies(),
               "ryzenai_corelib_selftest_dependencies");
    if (!api->functions().has_device_context()) {
        throw std::runtime_error("corelib has no AIE4 device context");
    }
    return std::shared_ptr<CorelibRuntime>(new CorelibRuntime(std::move(api)));
}

std::shared_ptr<CorelibRuntime> CorelibRuntime::GetOrCreate(
    const std::filesystem::path& executable_dir) {
    std::lock_guard lock(process_mutex);
    if (!process_runtime) {
        auto api = CorelibApi::Load(CorelibApi::ResolveLibraryPath(executable_dir));
        process_runtime = CreateReady(std::move(api));
    }
    return process_runtime;
}

std::shared_ptr<CorelibRuntime> CorelibRuntime::CreateForTest(
    std::shared_ptr<CorelibApi> api) {
    auto runtime = CreateReady(std::move(api));
    std::lock_guard lock(process_mutex);
    if (process_runtime) {
        throw std::runtime_error("corelib runtime already exists");
    }
    process_runtime = runtime;
    return runtime;
}

void CorelibRuntime::ShutdownProcess() {
    std::lock_guard process_lock(process_mutex);
    if (!process_runtime) return;

    std::lock_guard execution_lock(process_runtime->execution_mutex_);
    if (process_runtime->api_->live_object_count() != 0) {
        throw std::runtime_error("cannot shut down with live corelib objects");
    }
    process_runtime->api_->functions().cleanup();
    process_runtime->api_.reset();
    process_runtime.reset();
}

std::unique_lock<std::mutex> CorelibRuntime::AcquireExecution() {
    return std::unique_lock<std::mutex>(execution_mutex_);
}

const std::shared_ptr<CorelibApi>& CorelibRuntime::api() const noexcept {
    return api_;
}

}  // namespace flm::corelib
