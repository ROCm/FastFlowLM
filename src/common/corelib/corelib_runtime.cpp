#include "corelib/corelib_runtime.hpp"

#include <stdexcept>
#include <utility>

namespace flm::corelib {
namespace {
std::mutex process_mutex;
std::shared_ptr<CorelibRuntime> process_runtime;
#if defined(FLM_CORELIB_TESTING)
std::function<void(bool)> destruction_observer;
bool shutdown_execution_lock_held = false;
#endif
}

CorelibRuntime::CorelibRuntime(std::shared_ptr<CorelibApi> api)
    : api_(std::move(api)) {}

CorelibRuntime::~CorelibRuntime() {
#if defined(FLM_CORELIB_TESTING)
    if (destruction_observer) destruction_observer(shutdown_execution_lock_held);
#endif
}

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

    auto runtime = process_runtime;
    std::unique_lock execution_lock(runtime->execution_mutex_);
#if defined(FLM_CORELIB_TESTING)
    shutdown_execution_lock_held = true;
#endif
    if (runtime->api_->live_object_count() != 0) {
#if defined(FLM_CORELIB_TESTING)
        shutdown_execution_lock_held = false;
#endif
        throw std::runtime_error("cannot shut down with live corelib objects");
    }
    runtime->api_->functions().cleanup();
    runtime->api_.reset();
    process_runtime.reset();
    execution_lock.unlock();
#if defined(FLM_CORELIB_TESTING)
    shutdown_execution_lock_held = false;
#endif
    runtime.reset();
}

#if defined(FLM_CORELIB_TESTING)
void CorelibRuntime::SetDestructionObserverForTest(
    std::function<void(bool)> observer) {
    destruction_observer = std::move(observer);
}
#endif

std::unique_lock<std::mutex> CorelibRuntime::AcquireExecution() {
    return std::unique_lock<std::mutex>(execution_mutex_);
}

const std::shared_ptr<CorelibApi>& CorelibRuntime::api() const noexcept {
    return api_;
}

}  // namespace flm::corelib
