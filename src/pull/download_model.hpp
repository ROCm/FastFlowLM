/// \file download_model.hpp
/// \brief Download model class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This class for curl download
#pragma once

#include <string>
#include <filesystem>
#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#include <cstdio>
#include "nlohmann/json.hpp"

namespace download_utils {

enum class HashAlgorithm { Sha256, GitBlobSha1 };

struct DownloadRequest {
    std::string url;
    std::filesystem::path destination;
    std::uint64_t expected_size;
    HashAlgorithm hash_algorithm;
    std::string expected_hash;
    bool use_hf_token = false;
};

std::string calculate_file_sha256(const std::string& file_path);
std::string calculate_git_blob_oid(const std::string& file_path);

// Cursor control functions
void hide_cursor();
void show_cursor();

// Callback function for libcurl to write data to a file
size_t write_data_to_file(void* ptr, size_t size, size_t nmemb, FILE* stream);

// Callback function for libcurl to write data to a string
size_t write_data_to_string(void* ptr, size_t size, size_t nmemb, std::string* userdata);

// Progress callback function
int progress_callback(void* clientp, double dltotal, double dlnow, double ultotal, double ulnow);

// Download a file from URL to a local file
bool download_file(const std::string& url, const std::string& local_path, bool is_lfs, std::string remote_oid,
                   std::function<void(double)> progress_cb = nullptr);

// Download to a same-directory temporary file, verify it, then atomically promote it.
bool download_file_atomic(
    const DownloadRequest& request,
    std::function<void(double)> progress_cb = nullptr);

// Download content from URL to a string
std::string download_string(const std::string& url);

// The Hugging Face access token from HF_TOKEN, HUGGING_FACE_HUB_TOKEN or the
// token file `hf auth login` writes; empty when there is none.
std::string huggingface_token();

// Whether the server would serve url: a 2xx or 3xx answer to a HEAD request,
// or for a non-HTTP URL, a transfer that succeeds.
bool url_accessible(const std::string& url, bool use_hf_token = false);

// Download multiple files with progress tracking
bool download_multiple_files(const nlohmann::json downloads,
                           std::function<void(size_t, size_t)> progress_cb = nullptr);

// Initialize CURL library (call this once at program startup)
bool init_curl();

// Cleanup CURL library (call this once at program shutdown)
void cleanup_curl();

// RAII wrapper for CURL initialization
class CurlInitializer {
public:
    CurlInitializer();
    ~CurlInitializer();
};

} // namespace download_utils 