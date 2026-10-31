#pragma once

#include <ryzenai/corelib.h>

#if RYZENAI_CORELIB_VERSION_MAJOR != 0 || RYZENAI_CORELIB_VERSION_MINOR != 53 || \
    RYZENAI_CORELIB_VERSION_PATCH != 0
#error "FastFlowLM requires ryzenai-corelib headers exactly 0.53.0"
#endif

// ryzenai_corelib_testing_npu_arch, until corelib's public header names the
// NPU's architecture. After the pin: a testing hook may change in any release.
#include <ryzenai/corelib/testing.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#define FLM_CORELIB_FUNCTIONS(X)                                                   \
    X(get_version, ryzenai_corelib_get_version)                                    \
    X(status_to_string, ryzenai_corelib_status_to_string)                          \
    X(get_last_error_message, ryzenai_corelib_get_last_error_message)              \
    X(selftest_dependencies, ryzenai_corelib_selftest_dependencies)                \
    X(npu_usable, ryzenai_corelib_npu_usable)                                      \
    X(npu_arch, ryzenai_corelib_testing_npu_arch)                                  \
    X(object_release, ryzenai_corelib_object_release)                              \
    X(create_stream, ryzenai_corelib_create_stream)                                \
    X(stream_synchronize, ryzenai_corelib_stream_synchronize)                      \
    X(create_device_tensor, ryzenai_corelib_create_device_tensor)                  \
    X(create_tensor_window, ryzenai_corelib_create_tensor_window)                  \
    X(create_host_view, ryzenai_corelib_create_host_view)                          \
    X(tensor_write, ryzenai_corelib_tensor_write)                                  \
    X(tensor_read, ryzenai_corelib_tensor_read)                                    \
    X(stream_enable_kernel_bundle, ryzenai_corelib_stream_enable_kernel_bundle)    \
    X(matmul_enum_kernels, ryzenai_corelib_matmul_bf16_enum_kernels)               \
    X(matmul_weights_create, ryzenai_corelib_matmul_bf16_weights_create)           \
    X(matmul, ryzenai_corelib_matmul_bf16)                                         \
    X(matmul_act_enum_kernels, ryzenai_corelib_matmul_act_bf16_enum_kernels)       \
    X(ssmlp_enum_kernels, ryzenai_corelib_ssmlp_bf16_enum_kernels)                 \
    X(ssmlp_weights_create, ryzenai_corelib_ssmlp_bf16_weights_create)             \
    X(weights_copy_data, ryzenai_corelib_weights_copy_data)                        \
    X(ssmlp, ryzenai_corelib_ssmlp_bf16)                                           \
    X(model_get_info, ryzenai_corelib_model_get_info)                              \
    X(model_forward, ryzenai_corelib_model_forward)                                \
    X(model_reset, ryzenai_corelib_model_reset)                                    \
    X(create_model_from_path, ryzenai_corelib_create_model_from_path)              \
    X(model_create_session, ryzenai_corelib_model_create_session)                  \
    X(model_state_save, ryzenai_corelib_model_state_save)                          \
    X(model_state_restore, ryzenai_corelib_model_state_restore)                    \
    X(cleanup, ryzenai_corelib_cleanup)

namespace flm::corelib {

struct CorelibVersion {
    std::uint32_t major;
    std::uint32_t minor;
    std::uint32_t patch;
};

class CorelibError final : public std::runtime_error {
public:
    CorelibError(ryzenai_corelib_status status,
                 std::string call,
                 std::string detail,
                 std::string status_text);
    ryzenai_corelib_status status() const noexcept;
    const std::string& call() const noexcept;
    const std::string& detail() const noexcept;

private:
    ryzenai_corelib_status status_;
    std::string call_;
    std::string detail_;
};

struct CorelibFunctions {
#define FLM_DECLARE_CORELIB_FUNCTION(member, symbol) decltype(&::symbol) member{};
    FLM_CORELIB_FUNCTIONS(FLM_DECLARE_CORELIB_FUNCTION)
#undef FLM_DECLARE_CORELIB_FUNCTION
};

class CorelibApi final {
public:
    using Resolver = std::function<void*(std::string_view)>;
    /// \brief load corelib from a shared library at run time
    /// \note LoadLibraryEx on Windows, dlopen elsewhere. Unused by a build that
    ///       links corelib in (see LoadStatic), but still compiled there.
    static std::shared_ptr<CorelibApi> Load(const std::filesystem::path& library);
#if defined(FLM_CORELIB_LINK_STATIC)
    /// \brief bind to the corelib linked into this binary
    /// \return the API bound to the linked symbols
    /// \note The static build has no library to pick, so FLM_CORE_CORELIB_PATH
    ///       has nothing to select and is reported as ignored. On Windows an
    ///       import library still leaves a DLL that the loader binds before
    ///       main, so loaded_library_path() reports the module the linked
    ///       symbols live in.
    static std::shared_ptr<CorelibApi> LoadStatic();
#endif
    /// \brief the file of the module this process has loaded under `name`
    /// \return an absolute path, or empty when no such module is loaded or the
    ///         platform has no way to ask
    static std::filesystem::path LoadedModulePath(const wchar_t* name);
    static std::shared_ptr<CorelibApi> ResolveForTest(
        Resolver resolver, std::filesystem::path loaded_library_path = {});
    static std::filesystem::path ResolveLibraryPath(
        const std::filesystem::path& executable_dir);
    const CorelibFunctions& functions() const noexcept;
    CorelibVersion runtime_version() const noexcept;
    const std::filesystem::path& loaded_library_path() const noexcept;
    void Check(ryzenai_corelib_status status, std::string_view call) const;
    void RegisterObject() const noexcept;
    void Release(void* object) const noexcept;
    std::size_t live_object_count() const noexcept;

private:
    explicit CorelibApi(Resolver resolver,
                        std::filesystem::path loaded_library_path = {});

    Resolver resolver_;
    CorelibFunctions functions_{};
    CorelibVersion runtime_version_{};
    std::filesystem::path loaded_library_path_;
    mutable std::atomic<std::size_t> live_object_count_{0};
};

}  // namespace flm::corelib
