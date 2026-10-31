/// \file oga_api.hpp
/// \brief The onnxruntime-genai (OGA) C API, loaded at run time from a hip-ep package
/// \note flm links nothing from OGA: the package is installed separately, so
///       its DLLs are found and loaded when a model first needs them, and every
///       entry point is resolved by name into OgaFunctions.
#pragma once

#include "onnxruntime-genai/ort_genai_c.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#define FLM_OGA_FUNCTIONS(X)                                                       \
    X(result_get_error, OgaResultGetError)                                         \
    X(destroy_result, OgaDestroyResult)                                            \
    X(destroy_string, OgaDestroyString)                                            \
    X(create_config, OgaCreateConfig)                                              \
    X(config_overlay, OgaConfigOverlay)                                            \
    X(config_set_provider_option, OgaConfigSetProviderOption)                      \
    X(destroy_config, OgaDestroyConfig)                                            \
    X(create_model_from_config, OgaCreateModelFromConfig)                          \
    X(model_get_device_type, OgaModelGetDeviceType)                                \
    X(destroy_model, OgaDestroyModel)                                              \
    X(create_generator_params, OgaCreateGeneratorParams)                           \
    X(generator_params_set_search_number, OgaGeneratorParamsSetSearchNumber)       \
    X(destroy_generator_params, OgaDestroyGeneratorParams)                         \
    X(create_generator, OgaCreateGenerator)                                        \
    X(destroy_generator, OgaDestroyGenerator)                                      \
    X(generator_append_tokens, OgaGenerator_AppendTokens)                          \
    X(generator_get_logits, OgaGenerator_GetLogits)                                \
    X(tensor_get_type, OgaTensorGetType)                                           \
    X(tensor_get_shape_rank, OgaTensorGetShapeRank)                                \
    X(tensor_get_shape, OgaTensorGetShape)                                         \
    X(tensor_get_data, OgaTensorGetData)                                           \
    X(destroy_tensor, OgaDestroyTensor)

namespace flm::oga {

/// \brief a Windows file version, major.minor.build.revision
struct FileVersion {
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t build = 0;
    std::uint32_t revision = 0;
};

/// \brief the oldest onnxruntime-genai whose C API matches the vendored header
/// \note hip-ep 0.5.1 ships OGA 0.14, which runs this model only on the CPU.
inline constexpr FileVersion kMinimumOgaVersion{0, 17, 0, 0};

/// \brief one place a hip-ep package may be, in the order they are tried
struct PackageCandidate {
    /// \brief what named the root, e.g. "AMDRT_PKG_ROOT"
    std::string source;
    /// \brief the package root, empty when the source names none
    std::filesystem::path root;
};

/// \brief a hip-ep package that has every DLL and a new enough OGA
struct Package {
    std::string source;
    /// \brief the directory holding the DLLs: root\\bin, or root when it has no bin
    std::filesystem::path bin;
    /// \brief onnxruntime-genai.dll's file version
    FileVersion oga_version;
};

/// \brief "major.minor.build"
std::string VersionText(FileVersion version);

/// \brief a DLL's fixed file version, empty when it has no version resource
std::optional<FileVersion> ReadFileVersion(const std::filesystem::path& file);

/// \brief the value of an environment variable as a path, empty when it is unset
using PackageEnvironment = std::function<std::filesystem::path(const wchar_t* name)>;

/// \brief where discovery looked, in the order it tried
struct PackageSearch {
    std::vector<PackageCandidate> candidates;
    /// \brief default folders that were not searched or hold no version folder, for the error
    std::vector<std::string> notes;
};

/// \brief where this build looks for a hip-ep package, in order
/// \note AMDRT_PKG_ROOT, when set, is the only candidate. Otherwise the
///       candidates are the version folders under %LOCALAPPDATA%\hip-ep and
///       %ProgramFiles%\hip-ep, newest version first. A folder name that is not a
///       version comes after all of them; equal versions keep the %LOCALAPPDATA% one first.
PackageSearch SearchPackages(const PackageEnvironment& environment);

/// \brief check one candidate
/// \return the package, or why the candidate does not qualify
std::variant<Package, std::string> InspectPackage(const PackageCandidate& candidate);

/// \brief the first candidate that qualifies
/// \param notes appended to the error when no candidate qualifies
/// \throws std::runtime_error naming every candidate and why it was rejected
Package FindPackage(const std::vector<PackageCandidate>& candidates,
                    const std::vector<std::string>& notes = {});

struct OgaFunctions {
#define FLM_DECLARE_OGA_FUNCTION(member, symbol) decltype(&::symbol) member{};
    FLM_OGA_FUNCTIONS(FLM_DECLARE_OGA_FUNCTION)
#undef FLM_DECLARE_OGA_FUNCTION
};

/// \brief an OGA call that returned an error
class OgaError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// \brief the OGA C API of one hip-ep package
class OgaApi final {
public:
    using Resolver = std::function<void*(std::string_view)>;

    /// \brief the API of the hip-ep package that FindPackage(SearchPackages()) picks
    /// \note The cache holds no owning reference. The API lives as long as an
    ///       owner holds it; loading again after the last owner went finds the
    ///       DLLs already mapped. The DLLs are never unloaded here, and
    ///       OgaShutdown() is never called: flm swaps models inside one process,
    ///       and on hip-ep 0.5.2 (OGA 0.17.0) creating a generator after
    ///       OgaShutdown() crashes inside onnxruntime.dll (access violation under
    ///       OgaCreateGenerator), although ort_genai_c.h says GenAI may be used
    ///       again. Measured with `flm serve`: oga-hipep, then an NPU model, then
    ///       oga-hipep again crashes with the call and passes without it; ending
    ///       `flm run` with /bye exits with code 0 either way.
    static std::shared_ptr<const OgaApi> GetOrCreate();

    /// \brief GetOrCreate with `load` in place of the hip-ep loader
    static std::shared_ptr<const OgaApi> GetOrCreateWith(
        const std::function<std::shared_ptr<const OgaApi>()>& load);

    /// \brief an API whose entry points come from `resolver`
    static std::shared_ptr<const OgaApi> ResolveForTest(Resolver resolver, Package package = {});

    const OgaFunctions& functions() const noexcept;
    const Package& package() const noexcept;

    /// \brief throw OgaError if `result` holds an error, and destroy it either way
    void Check(OgaResult* result, std::string_view what) const;

    /// \brief a deleter that hands an OGA object back to its Destroy function
    template <typename T>
    struct Destroy {
        void(OGA_API_CALL* destroy)(T*) = nullptr;
        void operator()(T* object) const noexcept {
            if (object) destroy(object);
        }
    };

    template <typename T>
    using Owned = std::unique_ptr<T, Destroy<T>>;

private:
    OgaApi(Resolver resolver, Package package);

    OgaFunctions functions_{};
    Package package_;
};

}  // namespace flm::oga
