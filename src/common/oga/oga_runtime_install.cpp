// Copyright (c) 2026 Advanced Micro Devices, Inc.

#include "oga_runtime_install.hpp"

#include "utils/debug_utils.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <sstream>
#include <system_error>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace flm::oga {
namespace {

constexpr const wchar_t* kLocalAppDataVariable = L"LOCALAPPDATA";
constexpr std::string_view kManagedFolder = "hip-ep";
constexpr std::string_view kInstallerFolder = "hip-ep-install";
constexpr std::string_view kCacheFolder = "cache";
constexpr std::string_view kLockFolder = "locks";
constexpr std::string_view kStagingFolder = "staging";
constexpr std::string_view kQuarantineFolder = "quarantine";
constexpr std::string_view kManagedPrefix = "flm-";
constexpr std::string_view kMarkerName = ".flm-hipep-install.json";
constexpr std::string_view kMarkerSchema = "flm-hipep-install/v1";
constexpr std::string_view kMarkerOwner = "FastFlowLM";
constexpr std::uint64_t kInstallSafetyMargin = 64ULL * 1024ULL * 1024ULL;

const HipEpRuntimeRelease& BuiltInRelease() {
    static const HipEpRuntimeRelease release{
        "0.5.2",
        "amdgpu.zip",
        "https://github.com/ROCm/hip-ep/releases/download/v0.5.2/amdgpu.zip",
        130920185ULL,
        323570048ULL,
        "de3e4e2241e6466712be530bd66f77fcce44fc005910f23a874ce8adf4de7ef4",
    };
    return release;
}

std::string Text(const std::filesystem::path& path) {
    const auto utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

std::string BytesText(std::uint64_t bytes) {
    std::ostringstream stream;
    stream << (bytes >> 20) << " MiB";
    return stream.str();
}

bool PathFitsAnsi(const std::filesystem::path& path) {
    if (GetACP() == CP_UTF8) return true;
    BOOL used_default = FALSE;
    const std::wstring wide = path.native();
    const int needed = WideCharToMultiByte(CP_ACP, WC_NO_BEST_FIT_CHARS, wide.c_str(), -1, nullptr, 0, nullptr,
                                           &used_default);
    return needed > 0 && !used_default;
}

std::optional<std::filesystem::path> ShortPath(const std::filesystem::path& path) {
    const DWORD needed = GetShortPathNameW(path.c_str(), nullptr, 0);
    if (needed == 0) return std::nullopt;
    std::wstring short_name(needed, L'\0');
    const DWORD written = GetShortPathNameW(path.c_str(), short_name.data(), needed);
    if (written == 0) return std::nullopt;
    short_name.resize(written);
    return std::filesystem::path(short_name);
}

std::optional<std::filesystem::path> DefaultDownloadPath(const std::filesystem::path& path) {
    if (PathFitsAnsi(path)) return path;
    if (auto parent = ShortPath(path.parent_path())) {
        const std::filesystem::path alias = *parent / path.filename();
        if (PathFitsAnsi(alias)) return alias;
    }
    return std::nullopt;
}

std::string WindowsMessage(DWORD error) {
    return std::error_code(static_cast<int>(error), std::system_category()).message();
}

bool Exists(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::exists(path, error);
}

void EnsureDirectories(const std::filesystem::path& path, std::string_view label) {
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) throw std::runtime_error(std::string(label) + ": failed to create '" + Text(path) + "': " +
                                        error.message());
}

void RemoveTree(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::remove_all(path, error);
}

bool IsReparsePoint(const std::filesystem::path& path) {
    const DWORD attributes = GetFileAttributesW(path.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES) return false;
    return (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
}

void EnsureNotReparseIfExists(const std::filesystem::path& path, std::string_view label) {
    if (Exists(path) && IsReparsePoint(path))
        throw std::runtime_error(std::string(label) + ": '" + Text(path) + "' is a reparse point");
}

struct ManagedPaths {
    std::filesystem::path local_app_data;
    std::filesystem::path package_root;
    std::filesystem::path installer_root;
    std::filesystem::path cache_root;
    std::filesystem::path lock_root;
    std::filesystem::path staging_root;
    std::filesystem::path quarantine_root;
    std::filesystem::path final_dir;
    std::filesystem::path archive_path;
    std::filesystem::path lock_path;
};

ManagedPaths BuildPaths(const PackageEnvironment& environment, const HipEpRuntimeRelease& release) {
    const std::filesystem::path local_app_data = environment(kLocalAppDataVariable);
    if (local_app_data.empty()) {
        throw std::runtime_error("%LOCALAPPDATA% is not set, so flm cannot auto-install hip-ep");
    }
    ManagedPaths paths;
    paths.local_app_data = local_app_data;
    paths.package_root = local_app_data / kManagedFolder;
    paths.installer_root = local_app_data / kInstallerFolder;
    paths.cache_root = paths.installer_root / kCacheFolder;
    paths.lock_root = paths.installer_root / kLockFolder;
    paths.staging_root = paths.installer_root / kStagingFolder;
    paths.quarantine_root = paths.installer_root / kQuarantineFolder;
    const std::string final_name = std::string(kManagedPrefix) + release.version;
    paths.final_dir = paths.package_root / final_name;
    paths.archive_path = paths.cache_root / release.archive_name;
    paths.lock_path = paths.lock_root / (release.version + ".lock");
    return paths;
}

std::string MarkerJson(const HipEpRuntimeRelease& release) {
    const nlohmann::json marker = {
        {"owner", kMarkerOwner},
        {"schema", kMarkerSchema},
        {"runtime", "hip-ep"},
        {"version", release.version},
        {"archive_sha256", release.sha256},
    };
    return marker.dump(2);
}

/// \note The archive is not compared: flm releases may pin different archives of one hip-ep
///       version, and each must be able to repair what another installed.
bool IsManagedInstall(const std::filesystem::path& root, const HipEpRuntimeRelease& release) {
    std::ifstream stream(root / kMarkerName, std::ios::binary);
    if (!stream.is_open()) return false;
    nlohmann::json marker;
    try {
        stream >> marker;
        return marker.value("owner", "") == kMarkerOwner && marker.value("schema", "") == kMarkerSchema &&
               marker.value("runtime", "") == "hip-ep" && marker.value("version", "") == release.version;
    } catch (...) {
        return false;
    }
}

void WriteMarker(const std::filesystem::path& root, const HipEpRuntimeRelease& release) {
    std::ofstream stream(root / kMarkerName, std::ios::binary | std::ios::trunc);
    if (!stream.is_open())
        throw std::runtime_error("failed to write installer marker under '" + Text(root) + "'");
    stream << MarkerJson(release);
    if (!stream.good())
        throw std::runtime_error("failed to finish installer marker under '" + Text(root) + "'");
}

struct Hooks {
    HipEpRuntimeRelease release;
    std::function<bool(const download_utils::DownloadRequest&, std::function<void(double)>)> download;
    std::function<void(const std::filesystem::path&, const std::filesystem::path&)> extract_archive;
    std::function<std::optional<std::filesystem::path>(const std::filesystem::path&)> download_path;
    std::function<std::filesystem::space_info(const std::filesystem::path&)> space;
    std::function<void(std::string_view)> log;
    std::chrono::milliseconds lock_wait;
    std::chrono::milliseconds lock_poll;
};

Hooks ResolveHooks(const HipEpInstallHooks& hooks) {
    Hooks resolved;
    resolved.release = hooks.release.value_or(BuiltInRelease());
    resolved.download = hooks.download ? hooks.download : download_utils::download_file_atomic;
    resolved.extract_archive = hooks.extract_archive ? hooks.extract_archive : testing::ExtractHipEpArchiveWithTar;
    resolved.download_path = hooks.download_path ? hooks.download_path : DefaultDownloadPath;
    resolved.space = hooks.space ? hooks.space
                                 : [](const std::filesystem::path& path) {
                                       std::error_code error;
                                       const std::filesystem::space_info info = std::filesystem::space(path, error);
                                       if (error)
                                           throw std::runtime_error("failed to read free space for '" + Text(path) +
                                                                    "': " + error.message());
                                       return info;
                                   };
    resolved.log = hooks.log ? hooks.log : [](std::string_view text) { header_print("FLM", text); };
    resolved.lock_wait = hooks.lock_wait;
    resolved.lock_poll = hooks.lock_poll;
    return resolved;
}

class ScopedCleanup {
public:
    explicit ScopedCleanup(std::filesystem::path path) : path_(std::move(path)) {}
    ~ScopedCleanup() {
        if (!armed_) return;
        RemoveTree(path_);
    }
    void release() noexcept { armed_ = false; }

private:
    std::filesystem::path path_;
    bool armed_ = true;
};

class UniqueHandle {
public:
    UniqueHandle() = default;
    explicit UniqueHandle(HANDLE handle) : handle_(handle) {}
    ~UniqueHandle() {
        if (valid()) CloseHandle(handle_);
    }
    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;
    UniqueHandle(UniqueHandle&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    UniqueHandle& operator=(UniqueHandle&& other) noexcept {
        if (this == &other) return *this;
        if (valid()) CloseHandle(handle_);
        handle_ = other.handle_;
        other.handle_ = nullptr;
        return *this;
    }
    HANDLE get() const noexcept { return handle_; }
    bool valid() const noexcept { return handle_ && handle_ != INVALID_HANDLE_VALUE; }

private:
    HANDLE handle_ = nullptr;
};

UniqueHandle AcquireLock(const ManagedPaths& paths, const Hooks& hooks) {
    EnsureDirectories(paths.lock_root, "installer lock");
    const ULONGLONG deadline = GetTickCount64() + static_cast<ULONGLONG>(hooks.lock_wait.count());
    bool reported_wait = false;
    for (;;) {
        HANDLE handle = CreateFileW(paths.lock_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        if (handle != INVALID_HANDLE_VALUE) return UniqueHandle(handle);
        const DWORD error = GetLastError();
        if (error != ERROR_SHARING_VIOLATION && error != ERROR_ACCESS_DENIED) {
            throw std::runtime_error("failed to open installer lock '" + Text(paths.lock_path) + "': " +
                                     WindowsMessage(error));
        }
        if (!reported_wait) {
            hooks.log("Waiting for another FLM process to finish installing hip-ep");
            reported_wait = true;
        }
        if (GetTickCount64() >= deadline) {
            throw std::runtime_error("timed out waiting for the hip-ep installer lock '" + Text(paths.lock_path) +
                                     "'");
        }
        Sleep(static_cast<DWORD>(std::max<std::int64_t>(1, hooks.lock_poll.count())));
    }
}

std::optional<Package> FirstUsablePackage(const PackageSearch& search) {
    for (const PackageCandidate& candidate : search.candidates) {
        auto inspected = InspectPackage(candidate);
        if (auto* package = std::get_if<Package>(&inspected)) return *package;
    }
    return std::nullopt;
}

std::filesystem::path NarrowPathOrThrow(const std::filesystem::path& path, const Hooks& hooks) {
    const std::optional<std::filesystem::path> chosen = hooks.download_path(path);
    if (!chosen) {
        throw std::runtime_error("the cache path '" + Text(path) +
                                 "' cannot be represented for the shared downloader, and no short-path alias is "
                                 "available. Set AMDRT_PKG_ROOT to a manually extracted hip-ep package");
    }
    return *chosen;
}

std::string Sha256ForFile(const std::filesystem::path& path, const Hooks& hooks) {
    return download_utils::calculate_file_sha256(NarrowPathOrThrow(path, hooks).string());
}

bool ArchiveMatches(const ManagedPaths& paths, const Hooks& hooks) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(paths.archive_path, error)) return false;
    if (std::filesystem::file_size(paths.archive_path, error) != hooks.release.archive_size || error) return false;
    return Sha256ForFile(paths.archive_path, hooks) == hooks.release.sha256;
}

void EnsureSpaceBudget(const ManagedPaths& paths, const Hooks& hooks) {
    const std::filesystem::space_info info = hooks.space(paths.local_app_data);
    const std::uint64_t archive_need = ArchiveMatches(paths, hooks) ? 0 : hooks.release.archive_size;
    const std::uint64_t budget = hooks.release.expanded_size + archive_need + kInstallSafetyMargin;
    if (info.available < budget) {
        throw std::runtime_error("not enough free space under %LOCALAPPDATA% for hip-ep install: need at least " +
                                 BytesText(budget) + ", have " + BytesText(info.available));
    }
}

void EnsureArchiveCached(const ManagedPaths& paths, const Hooks& hooks) {
    EnsureDirectories(paths.cache_root, "installer cache");
    if (ArchiveMatches(paths, hooks)) {
        hooks.log("Reusing cached hip-ep archive " + Text(paths.archive_path));
        return;
    }
    std::error_code error;
    std::filesystem::remove(paths.archive_path, error);
    const download_utils::DownloadRequest request{
        hooks.release.url,
        NarrowPathOrThrow(paths.archive_path, hooks),
        hooks.release.archive_size,
        download_utils::HashAlgorithm::Sha256,
        hooks.release.sha256,
    };
    hooks.log("Downloading hip-ep " + hooks.release.version + " (" + BytesText(hooks.release.archive_size) +
              ") to " + Text(paths.final_dir));
    if (!hooks.download(request, [](double) {})) {
        throw std::runtime_error("failed to download hip-ep from " + hooks.release.url);
    }
    if (!ArchiveMatches(paths, hooks)) {
        throw std::runtime_error("cached hip-ep archive '" + Text(paths.archive_path) +
                                 "' did not match the pinned SHA-256 after download");
    }
}

/// \brief what hipgpu.dll needs beside the DLLs InspectPackage checks, all under the archive root
void ValidateArchiveLayout(const std::filesystem::path& root) {
    const std::array<std::filesystem::path, 3> required{
        root / "amdhip64_7.dll",
        root / "hipblaslt.dll",
        root / "hipblaslt" / "library",
    };
    for (const auto& path : required) {
        std::error_code error;
        const bool ok = path.has_extension() ? std::filesystem::is_regular_file(path, error)
                                             : std::filesystem::is_directory(path, error);
        if (!ok) throw std::runtime_error("extracted hip-ep archive is missing '" + Text(path) + "'");
    }
}

Package InspectOrThrow(std::string source, const std::filesystem::path& root) {
    auto inspected = InspectPackage({std::move(source), root});
    if (auto* package = std::get_if<Package>(&inspected)) return *package;
    throw std::runtime_error(std::get<std::string>(inspected));
}

std::filesystem::path UniqueChild(const std::filesystem::path& root, std::string_view prefix) {
    const DWORD pid = GetCurrentProcessId();
    const DWORD ticks = GetTickCount();
    return root / (std::string(prefix) + "-" + std::to_string(pid) + "-" + std::to_string(ticks));
}

void PrepareExistingTarget(const ManagedPaths& paths, const Hooks& hooks) {
    EnsureDirectories(paths.package_root, "managed hip-ep root");
    EnsureNotReparseIfExists(paths.package_root, "managed hip-ep root");
    if (!Exists(paths.final_dir)) return;
    EnsureNotReparseIfExists(paths.final_dir, "managed hip-ep package");
    if (!IsManagedInstall(paths.final_dir, hooks.release)) {
        throw std::runtime_error("refusing to replace '" + Text(paths.final_dir) +
                                 "' because it is not owned by FastFlowLM");
    }
    EnsureDirectories(paths.quarantine_root, "installer quarantine");
    const std::filesystem::path quarantine = UniqueChild(paths.quarantine_root, "broken-runtime");
    std::error_code error;
    std::filesystem::rename(paths.final_dir, quarantine, error);
    if (error) {
        throw std::runtime_error("failed to quarantine broken managed runtime '" + Text(paths.final_dir) + "': " +
                                 error.message());
    }
    hooks.log("Quarantined broken managed hip-ep runtime to " + Text(quarantine));
}

Package InstallManagedPackage(const PackageEnvironment& environment, const Hooks& hooks) {
    const ManagedPaths paths = BuildPaths(environment, hooks.release);
    EnsureDirectories(paths.installer_root, "installer root");
    EnsureNotReparseIfExists(paths.package_root, "managed hip-ep root");
    EnsureNotReparseIfExists(paths.installer_root, "installer root");
    EnsureNotReparseIfExists(paths.cache_root, "installer cache root");
    EnsureNotReparseIfExists(paths.lock_root, "installer lock root");
    EnsureNotReparseIfExists(paths.staging_root, "installer staging root");
    EnsureNotReparseIfExists(paths.quarantine_root, "installer quarantine root");
    EnsureNotReparseIfExists(paths.final_dir, "managed hip-ep package");
    EnsureNotReparseIfExists(paths.archive_path, "installer archive");
    EnsureNotReparseIfExists(std::filesystem::path(paths.archive_path.native() + L".part"), "installer archive part");
    UniqueHandle lock = AcquireLock(paths, hooks);

    if (auto package = FirstUsablePackage(SearchPackages(environment))) return *package;

    PrepareExistingTarget(paths, hooks);
    EnsureSpaceBudget(paths, hooks);
    EnsureArchiveCached(paths, hooks);

    EnsureDirectories(paths.staging_root, "installer staging root");
    const std::filesystem::path staging = UniqueChild(paths.staging_root, "runtime");
    EnsureDirectories(staging, "installer staging");
    ScopedCleanup cleanup(staging);
    hooks.log("Extracting hip-ep " + hooks.release.version + " under " + Text(staging));
    hooks.extract_archive(paths.archive_path, staging);
    ValidateArchiveLayout(staging);
    WriteMarker(staging, hooks.release);
    Package staged = InspectOrThrow("%LOCALAPPDATA%\\hip-ep\\" + std::string(kManagedPrefix) + hooks.release.version,
                                    staging);
    std::error_code error;
    std::filesystem::rename(staging, paths.final_dir, error);
    if (error) {
        throw std::runtime_error("failed to move hip-ep into '" + Text(paths.final_dir) + "': " + error.message());
    }
    cleanup.release();
    hooks.log("Installed hip-ep " + hooks.release.version + " to " + Text(paths.final_dir));
    return InspectOrThrow(std::move(staged.source), paths.final_dir);
}

std::wstring QuoteForCreateProcess(std::wstring_view value) {
    std::wstring quoted;
    quoted.push_back(L'"');
    std::size_t slashes = 0;
    for (const wchar_t ch : value) {
        if (ch == L'\\') {
            ++slashes;
            quoted.push_back(ch);
            continue;
        }
        if (ch == L'"') quoted.append(slashes, L'\\');
        slashes = 0;
        if (ch == L'"') quoted.push_back(L'\\');
        quoted.push_back(ch);
    }
    quoted.append(slashes, L'\\');
    quoted.push_back(L'"');
    return quoted;
}

std::filesystem::path SystemTar() {
    std::wstring system(MAX_PATH, L'\0');
    const UINT size = GetSystemDirectoryW(system.data(), static_cast<UINT>(system.size()));
    if (size == 0 || size >= system.size())
        throw std::runtime_error("failed to locate the Windows system directory");
    system.resize(size);
    return std::filesystem::path(system) / "tar.exe";
}

}  // namespace

void testing::ExtractHipEpArchiveWithTar(const std::filesystem::path& archive,
                                         const std::filesystem::path& staging_root) {
    constexpr DWORD kTarWaitMs = 10 * 60 * 1000;
    const std::filesystem::path tar = SystemTar();
    if (!Exists(tar)) throw std::runtime_error("Windows tar.exe not found at '" + Text(tar) + "'");
    UniqueHandle job(CreateJobObjectW(nullptr, nullptr));
    if (!job.valid()) throw std::runtime_error("failed to create a Windows job object for tar.exe");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation, &info, sizeof(info))) {
        throw std::runtime_error("failed to configure the tar.exe job object");
    }
    // Windows tar can lose non-ACP characters even when started with CreateProcessW.
    const auto archive_argument = ShortPath(archive).value_or(archive);
    const auto staging_argument = ShortPath(staging_root).value_or(staging_root);
    std::wstring command = QuoteForCreateProcess(tar.native()) + L" -xf " +
                           QuoteForCreateProcess(archive_argument.native()) + L" -C " +
                           QuoteForCreateProcess(staging_argument.native());
    command.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(tar.c_str(), command.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr, &startup, &process)) {
        throw std::runtime_error("failed to start tar.exe for '" + Text(archive) + "': " +
                                 WindowsMessage(GetLastError()));
    }
    UniqueHandle thread(process.hThread);
    UniqueHandle handle(process.hProcess);
    if (!AssignProcessToJobObject(job.get(), handle.get())) {
        TerminateProcess(handle.get(), 1);
        throw std::runtime_error("failed to assign tar.exe to its job object");
    }
    if (ResumeThread(thread.get()) == static_cast<DWORD>(-1)) {
        TerminateProcess(handle.get(), 1);
        throw std::runtime_error("failed to resume tar.exe");
    }
    const DWORD wait = WaitForSingleObject(handle.get(), kTarWaitMs);
    if (wait == WAIT_TIMEOUT) {
        TerminateProcess(handle.get(), 1);
        throw std::runtime_error("tar.exe timed out while extracting '" + Text(archive) + "'");
    }
    if (wait != WAIT_OBJECT_0) {
        throw std::runtime_error("failed while waiting for tar.exe on '" + Text(archive) + "'");
    }
    DWORD exit_code = 0;
    if (!GetExitCodeProcess(handle.get(), &exit_code)) {
        throw std::runtime_error("failed to read tar.exe exit code for '" + Text(archive) + "': " +
                                 WindowsMessage(GetLastError()));
    }
    if (exit_code != 0) {
        throw std::runtime_error("tar.exe failed to extract '" + Text(archive) + "' (exit " +
                                 std::to_string(exit_code) + ")");
    }
}

Package EnsureHipEpPackageForTest(const PackageEnvironment& environment, const HipEpInstallHooks& raw_hooks) {
    const Hooks hooks = ResolveHooks(raw_hooks);
    const PackageSearch search = SearchPackages(environment);
    if (!search.candidates.empty() && search.candidates.front().source == "AMDRT_PKG_ROOT") {
        return FindPackage(search.candidates, search.notes);
    }
    if (auto package = FirstUsablePackage(search)) return *package;
    try {
        return InstallManagedPackage(environment, hooks);
    } catch (const std::exception& install_error) {
        std::string message = install_error.what();
        try {
            FindPackage(search.candidates, search.notes);
        } catch (const std::exception& discovery_error) {
            message += "\nDiscovery before auto-install failed with:\n";
            message += discovery_error.what();
        }
        throw std::runtime_error(message);
    }
}

Package EnsureHipEpPackage(const PackageEnvironment& environment) {
    Package package = EnsureHipEpPackageForTest(environment, {});
    // RegisterAmdGpuProvider passes OGA the library path as narrow text; a short path keeps it ASCII.
    package.bin = ShortPath(package.bin).value_or(package.bin);
    return package;
}

}  // namespace flm::oga
