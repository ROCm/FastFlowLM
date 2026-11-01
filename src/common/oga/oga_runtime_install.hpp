// Copyright (c) 2026 Advanced Micro Devices, Inc.

/// \file oga_runtime_install.hpp
/// \brief Private helpers that find or install the fixed hip-ep runtime for oga-hipep
#pragma once

#include "oga/oga_api.hpp"
#include "download_model.hpp"

#include <chrono>
#include <functional>
#include <optional>
#include <string>

namespace flm::oga {

/// \brief the fixed hip-ep archive this build auto-installs
struct HipEpRuntimeRelease {
    std::string version;
    std::string archive_name;
    std::string url;
    std::uint64_t archive_size = 0;
    std::uint64_t expanded_size = 0;
    std::string sha256;
};

/// \brief test-only overrides for the private installer
struct HipEpInstallHooks {
    std::optional<HipEpRuntimeRelease> release;
    std::function<bool(const download_utils::DownloadRequest&, std::function<void(double)>)> download;
    std::function<void(const std::filesystem::path&, const std::filesystem::path&)> extract_archive;
    std::function<std::optional<std::filesystem::path>(const std::filesystem::path&)> download_path;
    std::function<std::filesystem::space_info(const std::filesystem::path&)> space;
    std::function<void(std::string_view)> log;
    // Allow the shared downloader's 60-minute timeout plus extraction to complete.
    std::chrono::milliseconds lock_wait = std::chrono::minutes(75);
    std::chrono::milliseconds lock_poll = std::chrono::milliseconds(250);
};

/// \brief find a usable hip-ep package, or auto-install the fixed per-user runtime
Package EnsureHipEpPackage(const PackageEnvironment& environment);

/// \brief test hook for the private installer
Package EnsureHipEpPackageForTest(const PackageEnvironment& environment,
                                  const HipEpInstallHooks& hooks);

namespace testing {

/// \brief the CreateProcessW tar extractor the installer uses in production
void ExtractHipEpArchiveWithTar(const std::filesystem::path& archive,
                                const std::filesystem::path& staging_root);

}  // namespace testing

}  // namespace flm::oga
