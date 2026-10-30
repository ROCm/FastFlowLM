#include "oga/oga_api.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winver.h>

namespace flm::oga {
namespace {
/// \brief the DLLs a hip-ep package must have; the first is OGA itself
constexpr std::array<std::string_view, 5> kPackageFiles{
    "onnxruntime-genai.dll", "onnxruntime.dll", "amdgpu-ep.dll", "hip-backend.dll", "hipgpu.dll"};

constexpr const wchar_t* kPackageRootVariable = L"AMDRT_PKG_ROOT";
constexpr std::string_view kHipEpFolder = "hip-ep";
/// \brief the hip-ep release whose OGA this backend is built against
constexpr const char* kMinimumHipEp = "0.5.2";

/// \brief a folder hip-ep may be unzipped under; `label` is how the error names it
struct DefaultFolder {
    const wchar_t* variable;
    const char* label;
};
/// \brief in this order, a tie between equal versions prefers the first
constexpr std::array<DefaultFolder, 2> kDefaultFolders{{
    {L"LOCALAPPDATA", "%LOCALAPPDATA%"},
    {L"ProgramFiles", "%ProgramFiles%"},
}};

bool AtLeast(FileVersion version, FileVersion minimum) {
    return std::tie(version.major, version.minor, version.build, version.revision) >=
           std::tie(minimum.major, minimum.minor, minimum.build, minimum.revision);
}

std::filesystem::path EnvironmentPath(const wchar_t* name) {
    const DWORD needed = GetEnvironmentVariableW(name, nullptr, 0);
    if (needed == 0) return {};
    std::wstring value(needed, L'\0');
    value.resize(GetEnvironmentVariableW(name, value.data(), needed));
    return value;
}

/// \brief the numeric parts of the first dotted number in a folder name, such as "0.5.2" in
/// "hipep_0.5.2_windows"; empty when the name has no dotted number
std::optional<std::vector<std::uint32_t>> ParseVersionName(std::string_view name) {
    for (std::size_t start = 0; start < name.size(); ++start) {
        // Read a digit run from its first digit only. A run too large for 32 bits
        // must not be read again from a later digit.
        if (start > 0 && name[start - 1] >= '0' && name[start - 1] <= '9') continue;
        std::string_view rest = name.substr(start);
        std::vector<std::uint32_t> parts;
        for (;;) {
            std::uint32_t part = 0;
            const char* const begin = rest.data();
            const auto [end, error] = std::from_chars(begin, begin + rest.size(), part);
            if (error != std::errc()) break;
            parts.push_back(part);
            rest.remove_prefix(static_cast<std::size_t>(end - begin));
            if (rest.empty() || rest.front() != '.') break;
            rest.remove_prefix(1);
        }
        if (parts.size() > 1) return parts;
    }
    return std::nullopt;
}

/// \return negative, zero or positive as `a` is older than, equal to or newer than `b`
int CompareVersions(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
    for (std::size_t i = 0; i < std::max(a.size(), b.size()); ++i) {
        const std::uint32_t x = i < a.size() ? a[i] : 0;
        const std::uint32_t y = i < b.size() ? b[i] : 0;
        if (x != y) return x < y ? -1 : 1;
    }
    return 0;
}

/// \brief a version folder under one of the default folders
struct FoundVersion {
    std::filesystem::path path;
    std::string source;
    std::string name;
    std::optional<std::vector<std::uint32_t>> parts;
    /// \brief the index into kDefaultFolders
    std::size_t folder = 0;
};

/// \brief the order discovery tries folders in: newest version first, names that are not versions
/// last, then the default folder order, then the name
bool TriedBefore(const FoundVersion& a, const FoundVersion& b) {
    if (a.parts.has_value() != b.parts.has_value()) return a.parts.has_value();
    if (a.parts && b.parts) {
        if (const int order = CompareVersions(*a.parts, *b.parts); order != 0) return order > 0;
    }
    if (a.folder != b.folder) return a.folder < b.folder;
    return a.name < b.name;
}

HMODULE LoadFromPackage(const std::filesystem::path& file) {
    HMODULE module = LoadLibraryExW(file.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        throw std::runtime_error("failed to load '" + file.string() + "' (Windows error " +
                                 std::to_string(GetLastError()) + ")");
    }
    return module;
}

/// \brief what GetOrCreate shares with the API it hands out
struct Cache {
    std::mutex mutex;
    std::weak_ptr<const OgaApi> api;
};

Cache& GetCache() {
    static Cache cache;
    return cache;
}

/// \brief load the package that FindPackage picks, and resolve its OGA entry points
std::shared_ptr<const OgaApi> LoadHipEp() {
    const PackageSearch search = SearchPackages(EnvironmentPath);
    Package package = FindPackage(search.candidates, search.notes);
    if (GetEnvironmentVariableW(L"HIPDNN_EP_STRICT", nullptr, 0) == 0)
        _putenv_s("HIPDNN_EP_STRICT", "1");
    AddDllDirectory(package.bin.c_str());
    // A System32 onnxruntime.dll of another version would otherwise satisfy
    // onnxruntime-genai.dll's import.
    LoadFromPackage(package.bin / "onnxruntime.dll");
    HMODULE genai = LoadFromPackage(package.bin / kPackageFiles.front());
    OgaApi::Resolver resolver = [genai](std::string_view name) -> void* {
        const std::string terminated(name);
        return reinterpret_cast<void*>(GetProcAddress(genai, terminated.c_str()));
    };
    return OgaApi::ResolveForTest(std::move(resolver), std::move(package));
}
}  // namespace

std::string VersionText(FileVersion version) {
    return std::to_string(version.major) + "." + std::to_string(version.minor) + "." +
           std::to_string(version.build);
}

std::optional<FileVersion> ReadFileVersion(const std::filesystem::path& file) {
    DWORD ignored = 0;
    const DWORD size = GetFileVersionInfoSizeW(file.c_str(), &ignored);
    if (size == 0) return std::nullopt;
    std::vector<std::byte> data(size);
    if (!GetFileVersionInfoW(file.c_str(), 0, size, data.data())) return std::nullopt;
    VS_FIXEDFILEINFO* info = nullptr;
    UINT length = 0;
    if (!VerQueryValueW(data.data(), L"\\", reinterpret_cast<void**>(&info), &length) ||
        length < sizeof(VS_FIXEDFILEINFO)) {
        return std::nullopt;
    }
    return FileVersion{HIWORD(info->dwFileVersionMS), LOWORD(info->dwFileVersionMS),
                       HIWORD(info->dwFileVersionLS), LOWORD(info->dwFileVersionLS)};
}

PackageSearch SearchPackages(const PackageEnvironment& environment) {
    PackageSearch search;
    if (std::filesystem::path root = environment(kPackageRootVariable); !root.empty()) {
        search.candidates.push_back({"AMDRT_PKG_ROOT", std::move(root)});
        return search;
    }

    std::vector<FoundVersion> found;
    for (std::size_t index = 0; index < kDefaultFolders.size(); ++index) {
        const DefaultFolder& folder = kDefaultFolders[index];
        const std::filesystem::path base = environment(folder.variable);
        if (base.empty()) {
            search.notes.push_back(std::string(folder.label) + " is not set");
            continue;
        }
        const std::filesystem::path hipep = base / kHipEpFolder;
        std::error_code error;
        if (!std::filesystem::is_directory(hipep, error)) {
            search.notes.push_back("'" + hipep.string() + "' does not exist");
            continue;
        }
        const std::size_t before = found.size();
        const std::filesystem::directory_iterator end;
        std::filesystem::directory_iterator entry(hipep, error);
        for (; !error && entry != end; entry.increment(error)) {
            std::error_code entry_error;
            if (!entry->is_directory(entry_error)) continue;
            std::string name = entry->path().filename().string();
            const std::string source = std::string(folder.label) + "\\" +
                                       std::string(kHipEpFolder) + "\\" + name;
            found.push_back({entry->path(), source, name, ParseVersionName(name), index});
        }
        if (found.size() == before)
            search.notes.push_back("'" + hipep.string() + "' has no version folder");
    }
    std::sort(found.begin(), found.end(), TriedBefore);
    for (FoundVersion& version : found)
        search.candidates.push_back({std::move(version.source), std::move(version.path)});
    return search;
}

std::variant<Package, std::string> InspectPackage(const PackageCandidate& candidate) {
    if (candidate.root.empty()) return std::string("not set");
    std::error_code error;
    if (!std::filesystem::is_directory(candidate.root, error))
        return "'" + candidate.root.string() + "' is not a directory";
    const std::filesystem::path bin = std::filesystem::is_directory(candidate.root / "bin", error)
                                          ? candidate.root / "bin"
                                          : candidate.root;
    std::string missing;
    for (const std::string_view file : kPackageFiles) {
        if (std::filesystem::is_regular_file(bin / file, error)) continue;
        missing += missing.empty() ? "" : ", ";
        missing += file;
    }
    if (!missing.empty()) return "'" + bin.string() + "' has no " + missing;
    const std::filesystem::path oga = bin / kPackageFiles.front();
    const std::optional<FileVersion> version = ReadFileVersion(oga);
    if (!version) return "'" + oga.string() + "' has no file version";
    if (!AtLeast(*version, kMinimumOgaVersion)) {
        return "'" + oga.string() + "' is OGA " + VersionText(*version) + ", older than " +
               VersionText(kMinimumOgaVersion);
    }
    return Package{candidate.source, bin, *version};
}

Package FindPackage(const std::vector<PackageCandidate>& candidates,
                    const std::vector<std::string>& notes) {
    std::string reasons;
    for (const PackageCandidate& candidate : candidates) {
        auto inspected = InspectPackage(candidate);
        if (auto* package = std::get_if<Package>(&inspected)) return std::move(*package);
        reasons += "\n  " + candidate.source + ": " + std::get<std::string>(inspected);
    }
    for (const std::string& note : notes) reasons += "\n  " + note;
    throw std::runtime_error(
        "no usable hip-ep package for the oga-hipep backend:" + reasons +
        "\nInstall hip-ep " + kMinimumHipEp + " or later (OGA " + VersionText(kMinimumOgaVersion) +
        " or later). hip-ep is a zip with no installer. Unzip it into %LOCALAPPDATA%\\hip-ep\\<version>"
        " or %ProgramFiles%\\hip-ep\\<version>, so that its bin directory holds the DLLs (a root"
        " without bin also works). Or set AMDRT_PKG_ROOT to the package root: when it is set, no"
        " other folder is tried.");
}

OgaApi::OgaApi(Resolver resolver, Package package) : package_(std::move(package)) {
#define FLM_RESOLVE_OGA_FUNCTION(member, symbol)                                   \
    {                                                                              \
        void* address = resolver(#symbol);                                         \
        if (!address) throw std::runtime_error("missing OGA symbol: " #symbol);    \
        functions_.member = reinterpret_cast<decltype(functions_.member)>(address); \
    }
    FLM_OGA_FUNCTIONS(FLM_RESOLVE_OGA_FUNCTION)
#undef FLM_RESOLVE_OGA_FUNCTION
}

std::shared_ptr<const OgaApi> OgaApi::ResolveForTest(Resolver resolver, Package package) {
    if (!resolver) throw std::invalid_argument("OGA resolver is empty");
    return std::shared_ptr<const OgaApi>(new OgaApi(std::move(resolver), std::move(package)));
}

std::shared_ptr<const OgaApi> OgaApi::GetOrCreate() { return GetOrCreateWith(LoadHipEp); }

std::shared_ptr<const OgaApi> OgaApi::GetOrCreateWith(
    const std::function<std::shared_ptr<const OgaApi>()>& load) {
    Cache& cache = GetCache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (std::shared_ptr<const OgaApi> api = cache.api.lock()) return api;
    std::shared_ptr<const OgaApi> api = load();
    cache.api = api;
    return api;
}

const OgaFunctions& OgaApi::functions() const noexcept { return functions_; }
const Package& OgaApi::package() const noexcept { return package_; }

void OgaApi::Check(OgaResult* result, std::string_view what) const {
    if (!result) return;
    const char* message = functions_.result_get_error(result);
    std::string text(what);
    text += " failed: ";
    text += message ? message : "unknown error";
    functions_.destroy_result(result);
    throw OgaError(text);
}

}  // namespace flm::oga
