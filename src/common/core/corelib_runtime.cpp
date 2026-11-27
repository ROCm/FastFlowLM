#include "core/corelib_runtime.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace flm::corelib {
namespace {
std::mutex process_mutex;
std::shared_ptr<CorelibRuntime> process_runtime;
/// \brief the API loaded for this process, kept so a runtime that failed to
///        come up, or the NPU probe, does not load it again
std::shared_ptr<CorelibApi> process_api;

/// \note The caller holds process_mutex.
std::shared_ptr<CorelibApi> ProcessApi(const std::filesystem::path& executable_dir) {
    if (!process_api) {
#if defined(FLM_CORELIB_LINK_STATIC)
        (void)executable_dir;
        process_api = CorelibApi::LoadStatic();
#else
        process_api = CorelibApi::Load(CorelibApi::ResolveLibraryPath(executable_dir));
#endif
    }
    return process_api;
}
#if defined(FLM_CORELIB_TESTING)
std::function<void(bool)> destruction_observer;
bool shutdown_execution_lock_held = false;
#endif
}

CorelibRuntime::CorelibRuntime(std::shared_ptr<CorelibApi> api, bool npu_direct)
    : api_(std::move(api)), npu_direct_(npu_direct) {}

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
    // Without one corelib still runs a session, on the GPU or the host, which
    // is the silent slowness the core backend is meant to rule out. An AIE2P
    // NPU runs corelib's kernels only behind an xclbin, which FastFlowLM never
    // hands it, so only AIEM counts.
    const auto& functions = api->functions();
    if (!functions.has_feature(ryzenai_corelib_feature_npu_aiem)) {
        const char* name = functions.has_feature(ryzenai_corelib_feature_npu_aie2p)
                               ? functions.feature_name(ryzenai_corelib_feature_npu_aie2p)
                               : nullptr;
        throw std::runtime_error(std::string("corelib has no usable NPU (") +
                                 (name ? name : "none") + ")");
    }
    // AIEM's default is a corelib built with --npu-direct. Without it, or under
    // RYZENAI_CORELIB_NPU_DIRECT=0, the models still run, on DD's kernels alone.
    // has_feature answers for the build alone: corelib reads the variable as
    // each stream is created, and runs none there under "0".
    const bool built = functions.has_feature(ryzenai_corelib_feature_npu_direct);
    const char* setting = std::getenv("RYZENAI_CORELIB_NPU_DIRECT");
    const bool disabled = setting && std::string_view(setting) == "0";
    if (!built || disabled) {
        std::cerr << "[FLM]  WARNING: corelib runs without NPU direct ("
                  << (built ? "RYZENAI_CORELIB_NPU_DIRECT=0"
                            : "a library built without build.py --npu-direct")
                  << "): attention runs on DD's kernels, slower on long prompts"
                  << std::endl;
    }
    return std::shared_ptr<CorelibRuntime>(
        new CorelibRuntime(std::move(api), built && !disabled));
}

std::shared_ptr<CorelibRuntime> CorelibRuntime::GetOrCreate(
    const std::filesystem::path& executable_dir) {
    std::lock_guard lock(process_mutex);
    if (!process_runtime) process_runtime = CreateReady(ProcessApi(executable_dir));
    return process_runtime;
}

std::string CorelibRuntime::NpuArch(const std::filesystem::path& executable_dir) noexcept {
    try {
        std::lock_guard lock(process_mutex);
        const auto api = process_runtime ? process_runtime->api_ : ProcessApi(executable_dir);
        const auto& functions = api->functions();
        for (const auto npu : {ryzenai_corelib_feature_npu_aiem,
                               ryzenai_corelib_feature_npu_aie2p}) {
            if (!functions.has_feature(npu)) continue;
            const char* name = functions.feature_name(npu);
            return name ? name : "";
        }
        return "";
    } catch (...) {
        return "";
    }
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
    process_api.reset();
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

bool CorelibRuntime::npu_direct() const noexcept {
    return npu_direct_;
}

const std::filesystem::path& CorelibRuntime::loaded_library_path() const noexcept {
    static const std::filesystem::path empty;
    return api_ ? api_->loaded_library_path() : empty;
}

}  // namespace flm::corelib
