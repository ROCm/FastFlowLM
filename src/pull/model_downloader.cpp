/// \file model_downloader.cpp
/// \brief Model downloader class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This class is used to download models from the huggingface
#include "model_downloader.hpp"
#include "utils/utils.hpp"
#include "download_model.hpp"
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cctype>
#include <unordered_set>

namespace {

std::string percent_encode_filename(std::string_view filename) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string encoded;
    for (const unsigned char ch : filename) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' || ch == '.' || ch == '~') {
            encoded.push_back(static_cast<char>(ch));
        } else {
            encoded.push_back('%');
            encoded.push_back(kHex[ch >> 4]);
            encoded.push_back(kHex[ch & 0x0f]);
        }
    }
    return encoded;
}

bool is_hex_revision(const std::string& revision) {
    return revision.size() == 40 &&
           std::all_of(revision.begin(), revision.end(), [](unsigned char ch) {
               return std::isxdigit(ch) != 0;
           });
}

/// \brief load the pinned per-file records describing a model's download
/// \param model_info the resolved catalog entry
/// \param model_tag the resolved "family:size" tag
/// \note A model whose artifacts differ per NPU generation shares one tag across
///       platforms, so its entry names the record set explicitly via
///       "model_info_key"; everything else is keyed by its tag.
nlohmann::json load_model_file_records(const nlohmann::json& model_info,
                                       const std::string& model_tag) {
    const std::string key = model_info.value("model_info_key", model_tag);
    std::ifstream stream(utils::find_model_info());
    if (!stream.is_open()) {
        throw std::runtime_error("model_info.json could not be opened");
    }
    return nlohmann::json::parse(stream).at(key);
}

const nlohmann::json& find_file_record(const nlohmann::json& records,
                                       const std::string& filename) {
    const auto record = std::find_if(records.begin(), records.end(), [&](const auto& value) {
        return value.at("path") == filename;
    });
    if (record == records.end()) {
        throw std::runtime_error("missing model_info record for " + filename);
    }
    return *record;
}

struct ResolvedModelFile {
    ModelFileSource source;
    std::uint64_t size;
    bool is_lfs;
    download_utils::HashAlgorithm hash_algorithm;
    std::string hash;
};

/// \brief whether an entry's files come from pinned upstream revisions
/// \param model_info the resolved model_list.json entry
/// \return true when the entry carries a "file_sources" map
/// \note These entries are assembled from third-party repos at a fixed commit
///       rather than published by FastFlowLM, so every file is hash-verified
///       against the same pinned metadata it was downloaded with. The core
///       Phi-4 GGUF is the only such entry today, but the rule is about how the
///       files are sourced, not about which backend consumes them.
bool uses_pinned_upstream_integrity(const nlohmann::json& model_info) {
    // Presence, not contents: resolve_file_source decides the same way, and a
    // malformed map should fail there rather than quietly downgrade an entry to
    // the unverified path here.
    return model_info.contains("file_sources");
}

}  // namespace

ModelFileSource resolve_file_source(const nlohmann::json& model_info,
                                    std::string_view filename,
                                    bool use_modelscope) {
    if (model_info.contains("file_sources")) {
        if (use_modelscope) {
            throw std::runtime_error("pinned Hugging Face per-file sources are required; --modelscope is not supported");
        }
        const auto& sources = model_info.at("file_sources");
        if (!sources.is_object()) {
            throw std::runtime_error("file_sources must be an object");
        }
        std::unordered_set<std::string> files;
        for (const auto& file : model_info.at("files")) {
            files.insert(file.get<std::string>());
        }
        for (const auto& [key, value] : sources.items()) {
            if (!files.contains(key)) {
                throw std::runtime_error("unknown file_sources key: " + key);
            }
            if (!value.is_object() || value.size() != 2 ||
                !value.contains("url") || !value.at("url").is_string() ||
                value.at("url").get<std::string>().empty()) {
                throw std::runtime_error("file source requires exactly a non-empty string url and revision");
            }
            if (!value.contains("revision") || !value.at("revision").is_string() ||
                !is_hex_revision(value.at("revision").get<std::string>())) {
                throw std::runtime_error("file source revision must be a 40-character hexadecimal string");
            }
        }
        const auto override = sources.find(std::string(filename));
        if (override != sources.end()) {
            const std::string base = override->at("url");
            const std::string revision = override->at("revision");
            return {base + "/resolve/" + revision + "/" +
                        percent_encode_filename(filename) + "?download=true",
                    revision};
        }
    }

    const std::string base_url = use_modelscope
        ? model_info.at("ms_url").get<std::string>()
        : model_info.at("url").get<std::string>();
    const bool use_hf_token = !use_modelscope && model_info.value("use_hf_token", false);
    if (base_url.find("resolve") != std::string::npos) {
        return {base_url + "/" + std::string(filename) + "?download=true", {}, use_hf_token};
    }
    return {base_url + "/resolve/main/" + std::string(filename) + "?download=true", {}, use_hf_token};
}

namespace {

ResolvedModelFile resolve_model_file(const nlohmann::json& model_info,
                                     const nlohmann::json& records,
                                     const std::string& filename,
                                     bool use_modelscope) {
    const auto& record = find_file_record(records, filename);
    const bool is_lfs = record.contains("lfs");
    const bool has_explicit_sha256 = record.contains("sha256");
    return {
        resolve_file_source(model_info, filename, use_modelscope),
        record.at("size").get<std::uint64_t>(),
        is_lfs,
        has_explicit_sha256 || is_lfs
            ? download_utils::HashAlgorithm::Sha256
            : download_utils::HashAlgorithm::GitBlobSha1,
        has_explicit_sha256
            ? record.at("sha256").get<std::string>()
            : (is_lfs ? record.at("lfs").at("oid").get<std::string>()
                      : record.at("oid").get<std::string>())};
}

}  // namespace

/// \brief Constructor
/// \param models the model list
/// \return the model downloader
ModelDownloader::ModelDownloader(model_list& models) 
    : supported_models(models), curl_init() {
}

/// \brief Check if the model is downloaded
/// \param model_tag the model tag
/// \return true if the model is downloaded, false otherwise
ModelDownloader::ModelStatus ModelDownloader::is_model_downloaded(const std::string& model_tag, bool sub_process_mode, bool fast_check) {
    const auto [new_tag, info] = supported_models.get_model_info(model_tag);
    const ModelStatus own = own_model_status(new_tag, sub_process_mode, fast_check);
    const auto base = supported_models.get_base_model(new_tag);
    if (!base) return own;

    // A tag with a base_model is ready only if its base is too. The worse of
    // the two wins: Missing, then Incompatible, then Outdated.
    const ModelStatus base_status = own_model_status(*base, sub_process_mode, fast_check);
    if (base_status != ModelStatus::Ready && !sub_process_mode && !fast_check) {
        header_print("WARNING", new_tag + " needs its base model " + *base +
                     (base_status == ModelStatus::Missing ? ", which is missing or incomplete."
                      : base_status == ModelStatus::Outdated ? ", which is outdated."
                      : ", which is incompatible with this FastFlowLM."));
    }
    const auto rank = [](ModelStatus st) {
        switch (st) {
            case ModelStatus::Missing: return 3;
            case ModelStatus::Incompatible: return 2;
            case ModelStatus::Outdated: return 1;
            default: return 0;
        }
    };
    return rank(base_status) > rank(own) ? base_status : own;
}

/// \brief Readiness of one tag's own folder (never looks through base_model)
ModelDownloader::ModelStatus ModelDownloader::own_model_status(const std::string& model_tag, bool sub_process_mode, bool fast_check) {
    const auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
    const bool strict_integrity = uses_pinned_upstream_integrity(model_info);
    auto missing_files = get_missing_files(new_model_tag);
    bool is_config_file_missing = std::find(missing_files.begin(), missing_files.end(), "config.json") != missing_files.end();
    ModelStatus modelstatus = ModelStatus::Missing;

    if (!is_config_file_missing) {
        modelstatus = check_model_compatibility(new_model_tag, sub_process_mode);

        if (modelstatus == ModelStatus::Outdated) {
            if (!fast_check) {
                header_print("FLM", "Checking outdated files...");
                verify_and_clean_files(new_model_tag, false, sub_process_mode);
            }
        }
        else if (modelstatus == ModelStatus::Ready) {
            if (!missing_files.empty() ||
                (strict_integrity && !fast_check &&
                 !verify_and_clean_files(new_model_tag, false, sub_process_mode))) {
                modelstatus = ModelStatus::Missing;
            }
        }
    }
    return modelstatus;
}

/// \brief Check if the model is compatible with the current FLM version
/// \param model_tag the model tag
/// \return true if the model is compatible, false otherwise
ModelDownloader::ModelStatus ModelDownloader::check_model_compatibility(const std::string& model_tag, bool sub_process_mode) {
    auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
    LM_Config config;
    config.from_pretrained(this->supported_models.get_model_path(new_model_tag));
    std::string flm_min_version = model_info["flm_min_version"];
    // The pinned Microsoft frontend config is upstream-native and intentionally
    // has no FLM version. Its catalog contract supplies the compatibility floor.
    std::string flm_version = uses_pinned_upstream_integrity(model_info)
        ? flm_min_version
        : config.flm_version;
    int l_l, m_l, r_l; //left, middle, right on local version
    int l_r, m_r, r_r; //left, middle, right on requried version
    int l_f, m_f, r_f; //left, middle, right on flm version
    sscanf(__FLM_VERSION__, "%d.%d.%d", &l_f, &m_f, &r_f);
    sscanf(flm_version.c_str(), "%d.%d.%d", &l_l, &m_l, &r_l);
    sscanf(flm_min_version.c_str(), "%d.%d.%d", &l_r, &m_r, &r_r);
    uint32_t local_version_u32 = l_l * 1000000 + m_l * 1000 + r_l;
    uint32_t required_version_u32 = l_r * 1000000 + m_r * 1000 + r_r;
    uint32_t flm_version_u32 = l_f * 1000000 + m_f * 1000 + r_f;

    if (local_version_u32 > flm_version_u32) {
        if (!sub_process_mode) {
            header_print("WARNING", "Local model " + model_tag + " version: " + flm_version + " > " + __FLM_VERSION__);
            header_print("WARNING", "Please update FLM to the latest version.");
        }
        return ModelStatus::Incompatible;
    }
    if (local_version_u32 < required_version_u32) {
        if (!sub_process_mode) {
            header_print("WARNING", "Local model " + model_tag + " version: " + flm_version + " < " + flm_min_version);
            // header_print("FLM", "Re-pulling latest model...");
        }
        return ModelStatus::Outdated;
    }
    return ModelStatus::Ready;
}
/// \brief Pull the model
/// \param model_tag the model tag
/// \param force_redownload true if the model should be downloaded even if it is already downloaded
/// \return true if the model is downloaded, false otherwise
bool ModelDownloader::pull_model(const std::string& model_tag, bool use_modelscope, bool force_redownload) {
    // A tag with a base_model needs the base's files too: pull the base first
    // (same source, never forced: --force re-downloads the tag's own files),
    // and stop if it fails.
    const auto base = supported_models.get_base_model(model_tag);
    if (base) {
        header_print("FLM", "Pulling required base model " + *base + " ...");
        if (!pull_own_model(*base, use_modelscope, false)) {
            header_print("ERROR", "Could not pull the required base model " + *base + ".");
            return false;
        }
    }
    return pull_own_model(model_tag, use_modelscope, force_redownload);
}

bool ModelDownloader::pull_own_model(const std::string& model_tag, bool use_modelscope, bool force_redownload) {
    try {
        // Once the pull is over, the files on disk say again which source it got.
        struct SourceChoiceReset {
            model_list& models;
            const std::string& tag;
            ~SourceChoiceReset() { models.choose_source(tag, std::nullopt); }
        } reset_source_choice{supported_models, model_tag};
        choose_pull_source(model_tag, use_modelscope, force_redownload);

        // Get model info
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::string model_name = model_info["name"];
        std::string model_server = use_modelscope ? "ModelScope" : "HuggingFace";
        if (use_modelscope && model_info.contains("file_sources")) {
            // Validate this before any ready-state early return.
            resolve_file_source(model_info, model_info.at("files").at(0).get<std::string>(), true);
        }
        
        header_print("FLM", "Pulling model from " + model_server + "...");
        header_print("FLM", "Model: " + new_model_tag);
        header_print("FLM", "Name: " + model_name);

        ModelDownloader::ModelStatus status = own_model_status(new_model_tag, false, false);
        switch (status) {
            case ModelStatus::Ready:
                if (!force_redownload) {
                    header_print("FLM", "Model already downloaded. Use --force to re-download.");
                    return true;
                }
                break;
            case ModelStatus::Missing:
                if (uses_pinned_upstream_integrity(model_info)) {
                    // Preserve valid finals, but remove corrupt pinned finals before
                    // deciding which files need to be downloaded.
                    verify_and_clean_files(new_model_tag, use_modelscope, true);
                }
                break;
            case ModelStatus::Outdated:
                break;
            case ModelStatus::Incompatible:
                return true;
        }
        
        // Get missing files
        auto missing_files = get_missing_files(new_model_tag);
        if (missing_files.empty() && !force_redownload) {
            header_print("FLM", "All files already present.");
            return true;
        }
        
        if (!missing_files.empty()) {
            header_print("FLM", "Missing files (" + std::to_string(missing_files.size()) + "):");
            for (const auto& file : missing_files) {
                std::cout << "  - " << file << std::endl;
            }
        } else {
            header_print("FLM", "All required files are present.");
        }
        
        // Show present files if any
        auto present_files = get_present_files(new_model_tag);
        if (!present_files.empty()) {
            header_print("FLM", "Present files (" + std::to_string(present_files.size()) + "):");
            for (const auto& file : present_files) {
                std::cout << "  - " << file << std::endl;
            }
        }
        
        // Build download list
        auto download_list = build_download_list(new_model_tag, use_modelscope, force_redownload);
        auto downloads = download_list.first;
        float sum_fize_size = download_list.second;
        if (downloads.empty()) {
            header_print("FLM", "No files to download for model: " + new_model_tag);
            return !uses_pinned_upstream_integrity(model_info) ||
                   verify_and_clean_files(new_model_tag, use_modelscope);
        }
        
        header_print("FLM", "Downloading " + std::to_string(downloads.size()) + " missing files...");

        header_print("FLM", "Files to download (" << std::fixed << std::setprecision(2) << sum_fize_size << " MB): ");
        for (const auto& download : downloads) {
            float file_size = download["size"];
            std::string filename = download["file"];
            std::cout << "  - " << filename << " ("
                << std::fixed << std::setprecision(2) << file_size << " MB)"
                << std::endl;
        }
        
        // Download files with progress
        bool success = download_utils::download_multiple_files(downloads, get_progress_callback());

        if (success) {
            header_print("FLM", "Model downloaded successfully!");
            
            // Verify every final file using the same pinned metadata used to download it.
            auto final_missing = get_missing_files(new_model_tag);
            const bool verified = final_missing.empty() &&
                (!uses_pinned_upstream_integrity(model_info) ||
                 verify_and_clean_files(new_model_tag, use_modelscope));
            if (verified) {
                header_print("FLM", "All files verified successfully.");
            } else {
                header_print("WARNING", "Some files are missing or failed verification after download.");
            }
            return verified;
        } else {
            header_print("ERROR", "Failed to download model files.");
            return false;
        }
        
    } catch (const std::exception& e) {
        header_print("ERROR", "Exception during download: " + std::string(e.what()));
        return false;
    }
}

/// \brief Settle which source a pull uses
/// \param model_tag the model tag
/// \param use_modelscope true when pulling from ModelScope, which mirrors no
///        preferred source
/// \param force_redownload true if the files are to be fetched again
/// \note The preferred source is used when it already supplied the files on
///       disk or answers now; otherwise the entry's own source is.
void ModelDownloader::choose_pull_source(const std::string& model_tag, bool use_modelscope,
                                         bool force_redownload) {
    const auto own_files = supported_models.preferred_source_files(model_tag);
    if (own_files.empty()) return;

    supported_models.choose_source(model_tag, true);
    const auto [new_model_tag, preferred] = supported_models.get_model_info(model_tag);
    const std::string preferred_url = preferred.at("url");
    const std::string model_path = supported_models.get_model_path(new_model_tag);
    const bool pulled = std::all_of(own_files.begin(), own_files.end(), [&](const std::string& file) {
        return file_exists(get_model_file_path(model_path, file));
    });
    if (pulled && !force_redownload) return;

    if (!use_modelscope) {
        const auto probe = resolve_file_source(preferred, own_files.front(), false);
        if (download_utils::url_accessible(probe.url, probe.use_hf_token)) {
            header_print("FLM", "Using preferred source: " + preferred_url);
            return;
        }
        header_print("FLM", "Preferred source is not accessible: " + preferred_url);
    }
    supported_models.choose_source(model_tag, false);
    const std::string fallback_url = supported_models.get_model_info(model_tag).second.at("url");
    header_print("FLM", "Falling back to: " + fallback_url);
}

/// \brief Model not found
/// \param model_tag the model tag
void ModelDownloader::model_not_found(const std::string& model_tag) {
    header_print("ERROR", "Model not found: " + model_tag);
    header_print("ERROR", "Supported models: ");
    nlohmann::json models = supported_models.get_all_models();
    for (const auto& model : models["models"]) {
        header_print("ERROR", "  - " + model["name"].get<std::string>());
    }
}

/// \brief Get missing files
/// \param model_tag the model tag
/// \return the missing files
std::vector<std::string> ModelDownloader::get_missing_files(const std::string& model_tag) {
    std::vector<std::string> missing_files;

    try {
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::string model_name = model_info["name"];
        std::string model_path = supported_models.get_model_path(new_model_tag);
        std::vector<std::string> model_files = model_info["files"];

        // Check if this is a VLM model (default to false if key doesn't exist)

        // Check each required model file
        for (int i = 0; i < model_files.size(); ++i) {
            std::string filename = model_files[i];
            std::string file_path = get_model_file_path(model_path, filename);
            if (!file_exists(file_path)) {
                missing_files.push_back(filename);
            }
        }
    } catch (const std::exception& e) {
        header_print("ERROR", "Error checking missing files: " + std::string(e.what()));
    }

    return missing_files;
}

/// \brief Get present files
/// \param model_tag the model tag
/// \return the present files
std::vector<std::string> ModelDownloader::get_present_files(const std::string& model_tag) {
    std::vector<std::string> present_files;
    
    try {
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::string model_name = model_info["name"];
        std::string model_path = supported_models.get_model_path(new_model_tag);
        std::vector<std::string> model_files = model_info["files"];

        // Check if this is a VLM model (default to false if key doesn't exist)
        
        // Check each required model file
        for (int i = 0; i < model_files.size(); ++i) {
            std::string filename = model_files[i];
            std::string file_path = get_model_file_path(model_path, filename);
            if (file_exists(file_path)) {
                present_files.push_back(filename);
            }
        }     
    } catch (const std::exception& e) {
        header_print("ERROR", "Error checking present files: " + std::string(e.what()));
    }
    
    return present_files;
}

/// \brief Get progress callback
/// \return the progress callback
std::function<void(size_t, size_t)> ModelDownloader::get_progress_callback() {
    return [](size_t completed, size_t total) {
        if (total > 0) {
            double percentage = (static_cast<double>(completed) / total) * 100.0;
            std::cout << "\r[FLM]  Overall progress:  " << completed << "/" << total << " files" << std::flush;
            
            std::cout << std::endl;
        }
    };
}

/// \brief Check if the file exists
/// \param file_path the file path
/// \return true if the file exists, false otherwise
bool ModelDownloader::file_exists(const std::string& file_path) {
    return std::filesystem::exists(file_path) && std::filesystem::is_regular_file(file_path);
}

/// \brief Get the model file path
/// \param model_path the model path
/// \param filename the filename
/// \return the model file path
std::string ModelDownloader::get_model_file_path(const std::string& model_path, const std::string& filename) {
    std::filesystem::path full_path = std::filesystem::path(model_path) / filename;
    return full_path.string();
}

/// \brief Build the download list
/// \param model_tag the model tag
/// \return the download list
std::pair<nlohmann::json, float> ModelDownloader::build_download_list(
    const std::string& model_tag, bool modelscope, bool force_redownload) {
    nlohmann::json downloads = nlohmann::json::array();
    float sum_file_size = 0;

    auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
    const std::vector<std::string> model_files = model_info.at("files");
    const std::string model_path = supported_models.get_model_path(new_model_tag);
    std::filesystem::create_directories(model_path);
    const nlohmann::json records = load_model_file_records(model_info, new_model_tag);

    for (const auto& filename : model_files) {
        const std::string local_path = get_model_file_path(model_path, filename);
        if (!force_redownload && file_exists(local_path)) {
            continue;
        }

        const auto file = resolve_model_file(model_info, records, filename, modelscope);
        const float file_size = static_cast<float>(file.size) / 1024 / 1024;
        sum_file_size += file_size;
        downloads.push_back({
            {"file", filename},
            {"size", file_size},
            {"expected_size", file.size},
            {"url", file.source.url},
            {"localpath", local_path},
            {"oid", file.hash},
            {"is_lfs", file.is_lfs},
            {"hash_algorithm", file.hash_algorithm == download_utils::HashAlgorithm::Sha256
                                   ? "sha256" : "git_blob_sha1"},
            {"use_hf_token", file.source.use_hf_token},
        });
    }
    return {downloads, sum_file_size};
}

/// \brief Remove a model and all its files
/// \param model_tag the model tag
/// \return true if the model was successfully removed, false otherwise
bool ModelDownloader::remove_model(const std::string& model_tag, bool sub_process_mode) {
    try {
        // Check if model exists in supported models by trying to get its info
        try {
            supported_models.get_model_info(model_tag);
        } catch (const std::exception& e) {
            header_print("ERROR", "Model not found: " + model_tag);
            model_not_found(model_tag);
            return false;
        }
        
        // Removing a base leaves its dependants without their main weights.
        // Warn, never cascade.
        if (!sub_process_mode) {
            for (const auto& dependant : installed_dependants(model_tag)) {
                header_print("WARNING", "Installed model " + dependant + " requires " + model_tag +
                             " and will not run until it is pulled again.");
            }
        }

        // Get model path (a tag with a base_model removes only its own folder)
        std::string model_path = supported_models.get_model_path(model_tag);
        
        // Check if model directory exists
        if (!std::filesystem::exists(model_path)) {
            header_print("FLM", "Model directory does not exist: " + model_path);
            return true; // Consider it already removed
        }

        if (!sub_process_mode) {
            header_print("FLM", "Removing model: " + model_tag);
            header_print("FLM", "Path: " + model_path);
        }
        
        // Remove all files in the model directory
        size_t removed_files = 0;
        for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
            if (entry.is_regular_file()) {
                std::filesystem::remove(entry.path());
                removed_files++;
            }
        }
        
        // Remove the model directory itself
        if (std::filesystem::remove(model_path)) {
            if(!sub_process_mode)
                header_print("FLM", "Successfully removed " + std::to_string(removed_files) + " files and model directory.");
            return true;
        } else {
            header_print("ERROR", "Failed to remove model directory: " + model_path);
            return false;
        }
        
    } catch (const std::exception& e) {
        header_print("ERROR", "Exception during model removal: " + std::string(e.what()));
        return false;
    }
}

/// \brief Check hash of model files
/// \param model_tag the model tag
/// \return true if all files are present and compatible, false otherwise
bool ModelDownloader::check_model(const std::string& model_tag, bool use_modelscope, bool sub_process_mode) {
    // The base is checked on its own; then the tag's own files.
    if (const auto base = supported_models.get_base_model(model_tag)) {
        if (!check_own_model(*base, use_modelscope, sub_process_mode)) {
            header_print("FLM", "Base model incomplete: " + *base + " (needed by " + model_tag + ").");
            return false;
        }
    }
    return check_own_model(model_tag, use_modelscope, sub_process_mode);
}

/// \brief Tags that need this one as their base_model and have files on disk
std::vector<std::string> ModelDownloader::installed_dependants(const std::string& model_tag) {
    std::vector<std::string> out;
    for (const auto& dependant : supported_models.get_dependants(model_tag)) {
        std::error_code error;
        if (std::filesystem::exists(supported_models.get_model_path(dependant), error)) out.push_back(dependant);
    }
    return out;
}

bool ModelDownloader::check_own_model(const std::string& model_tag, bool use_modelscope, bool sub_process_mode) {
    auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
    if (use_modelscope && model_info.contains("file_sources")) {
        try {
            resolve_file_source(
                model_info, model_info.at("files").at(0).get<std::string>(), true);
        }
        catch (const std::exception& error) {
            header_print("ERROR", error.what());
            return false;
        }
    }
    header_print("FLM", "Checking model: " + new_model_tag + "...\n");

    // check_model owns the one full integrity pass below. Status discovery must
    // remain presence/version-only so a pinned 4.1 GB model is not hashed twice.
    ModelStatus status = own_model_status(new_model_tag, sub_process_mode, true);
    switch (status) {
        case ModelStatus::Missing:
            header_print("FLM", "Model not found: " + new_model_tag);
            header_print("FLM", "Use `flm pull " + new_model_tag + "` to download it.");
            return false;
        case ModelStatus::Incompatible:
            header_print("FLM", "Model is incompatible with this version of FastFlowLM: " + new_model_tag);
            header_print("FLM", "Use `flm pull " + new_model_tag + "` to re-download it.");
            return false;
        case ModelStatus::Outdated:
        case ModelStatus::Ready: {
            bool ok = verify_and_clean_files(new_model_tag, use_modelscope, sub_process_mode);
            if (!ok)
                header_print("FLM", "Model check completed with errors. Use `flm pull " + new_model_tag + "` to re-download corrupted files.");
            else
                header_print("FLM", "Model check completed successfully. All files are present and compatible.");
            return ok;
        }
    }
    return false;
}

/// \brief Verify each model file's hash against HuggingFace metadata and
///        remove any corrupted files. Files that pass verification are kept.
/// \param model_tag the model tag
/// \param sub_process_mode if true, suppress informational logging
/// \return true if all files passed verification, false otherwise
bool ModelDownloader::verify_and_clean_files(const std::string& model_tag, bool use_modelscope, bool sub_process_mode) {
    bool any_error = false;
    try {
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::vector<std::string> model_files = model_info["files"];
        std::string model_path = supported_models.get_model_path(new_model_tag);
        const nlohmann::json records = load_model_file_records(model_info, new_model_tag);

        for (const auto& filename : model_files) {
            if (!sub_process_mode) {
                header_print("FLM", "Checking file: " + filename + "...");
            }

            const auto file = resolve_model_file(
                model_info, records, filename, use_modelscope);
            std::string local_path = get_model_file_path(model_path, filename);

            // If the file isn't present locally, there's nothing to verify or
            // remove; treat as an error so the caller knows a re-pull is needed.
            if (!file_exists(local_path)) {
                any_error = true;
                header_print("FLM", "File missing: " + filename);
                continue;
            }

            const std::string local_oid =
                file.hash_algorithm == download_utils::HashAlgorithm::Sha256
                    ? download_utils::calculate_file_sha256(local_path)
                    : download_utils::calculate_git_blob_oid(local_path);
            std::error_code size_error;
            const auto local_size = std::filesystem::file_size(local_path, size_error);
            const bool size_matches = !size_error && local_size == file.size;

            if (size_matches && local_oid == file.hash) {
                if (!sub_process_mode) {
                    header_print("FLM", "Success!");
                }
            }
            else {
                if (!sub_process_mode) {
                    header_print("FLM", "Fail!");
                    header_print("FLM", "Removing corrupted file: " + filename + "...");
                }

                if (std::filesystem::remove(local_path)) {
                    if (!sub_process_mode) {
                        header_print("FLM", "Successfully removed " + filename + "!");
                    }
                }
                else {
                    header_print("ERROR", "Failed to remove corrupted file: " + filename);
                }
                any_error = true;
            }
        }
    }
    catch (const std::exception& e) {
        header_print("ERROR", "Exception during file verification: " + std::string(e.what()));
        any_error = true;
    }
    return !any_error;
}