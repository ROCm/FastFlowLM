#pragma once

#include <ryzenai/corelib.h>

#if RYZENAI_CORELIB_VERSION_MAJOR != 0 || RYZENAI_CORELIB_VERSION_MINOR != 3 || \
    RYZENAI_CORELIB_VERSION_PATCH != 0
#error "FastFlowLM requires ryzenai-corelib headers exactly 0.3.0"
#endif

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
    X(has_device_context, ryzenai_corelib_has_device_context)                      \
    X(object_release, ryzenai_corelib_object_release)                              \
    X(create_stream, ryzenai_corelib_create_stream)                                \
    X(stream_synchronize, ryzenai_corelib_stream_synchronize)                      \
    X(create_device_tensor, ryzenai_corelib_create_device_tensor)                  \
    X(create_tensor_window, ryzenai_corelib_create_tensor_window)                  \
    X(tensor_write, ryzenai_corelib_tensor_write)                                  \
    X(tensor_read, ryzenai_corelib_tensor_read)                                    \
    X(tensor_get_byte_size, ryzenai_corelib_tensor_get_byte_size)                  \
    X(tensor_get_data_type, ryzenai_corelib_tensor_get_data_type)                  \
    X(matmul_pad_shape, ryzenai_corelib_matmul_bf16_pad_shape)                     \
    X(matmul_weights_create_gguf_requantized,                                      \
      ryzenai_corelib_matmul_bf16_weights_create_gguf_requantized)                 \
    X(matmul, ryzenai_corelib_matmul_bf16)                                         \
    X(ssmlp_pad_rows, ryzenai_corelib_ssmlp_bf16_pad_rows)                         \
    X(ssmlp_weights_create_gguf_requantized,                                       \
      ryzenai_corelib_ssmlp_bf16_weights_create_gguf_requantized)                  \
    X(ssmlp, ryzenai_corelib_ssmlp_bf16)                                           \
    X(rmsnorm_weights_create_scale, ryzenai_corelib_rmsnorm_bf16_weights_create_scale) \
    X(rmsnorm_pad_rows, ryzenai_corelib_rmsnorm_bf16_pad_rows)                     \
    X(rmsnorm, ryzenai_corelib_rmsnorm_bf16)                                       \
    X(flat_mha_pad_rows, ryzenai_corelib_flat_mha_bf16_pad_rows)                   \
    X(flat_mha, ryzenai_corelib_flat_mha_bf16)                                     \
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
    static std::shared_ptr<CorelibApi> Load(const std::filesystem::path& dll);
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
