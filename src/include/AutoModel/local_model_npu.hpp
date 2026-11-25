/// \file local_model_npu.hpp
/// \brief The NPU package format of a model folder (flm backend, NPU)
/// \note An NPU package is marked by model.q4nx: the weights, with the network
///       config, the tokenizer and, for a vision-language model, the vision
///       tower. Including this header makes build_local_model_entry read such
///       folders. A package is built for one NPU generation, which the folder
///       does not say, so its entry names the platform the caller runs on and
///       model_list refuses it anywhere else.
/// \note The package's own .xclbin files are not used: flm runs its kernels
///       from xclbins/<model name>. select_npu_kernels picks that folder.
#pragma once
#include "AutoModel/backend_contract.hpp"
#include "AutoModel/local_model_entry.hpp"
#include "model_list.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace flm::catalog {

/// \brief the file that marks an NPU package
inline constexpr const char* kNpuWeightsFile = "model.q4nx";

/// \brief read an NPU package folder
/// \param folder the absolute model folder
/// \param platform the NPU platform id this process runs on; required
/// \throws std::runtime_error if no platform is given or config.json is unreadable
inline LocalPackageFacts read_npu_package(const std::filesystem::path& folder, const std::string& platform) {
    using local_detail::is_file;
    LocalPackageFacts facts;
    // The weights, the network config and the tokenizer are what the loader reads;
    // the vision tower is there only for a VLM.
    facts.files = {"config.json", kNpuWeightsFile, "tokenizer.json"};
    for (const char* optional : {"tokenizer_config.json", "chat_template.jinja", "vision_weight.q4nx"}) {
        if (is_file(folder / optional)) facts.files.insert(optional);
    }
    facts.vision = is_file(folder / "vision_weight.q4nx");
    if (platform.empty()) {
        throw std::runtime_error("an NPU package needs the NPU platform it runs on");
    }
    if (is_file(folder / "config.json")) {
        const auto config = local_detail::parse_file(folder / "config.json", "config.json");
        if (config.contains("max_position_embeddings") &&
            config["max_position_embeddings"].is_number_integer()) {
            facts.package_limit = config["max_position_embeddings"].get<int>();
        }
    }
    // The kernels belong to one generation; an entry on another one is refused
    // by model_list, as a shipped entry would be.
    facts.entry_extra["supported_platforms"] = nlohmann::json::array({platform});
    return facts;
}

/// \brief the NPU format, known to build_local_model_entry once this header is included
inline const bool kNpuPackageRegistered = register_local_package_format(
    {"NPU package", kNpuWeightsFile, flm::backend::kFlmBackendId, "", read_npu_package});

/// \brief make a local NPU model run on kernels that exist
/// \param models the catalog the local model was added to
/// \param tag the local model's tag
/// \param family details.family of the package (--family)
/// \param requested the folder under xclbins the user named (--xclbin-folder), "" if none
/// \param xclbin_root the folder that holds the xclbins folders
/// \return a note for the user when a folder was chosen for them, else ""
/// \throws std::runtime_error naming the kernel folders to choose from, or if
///         a folder was requested for a model that does not run on the NPU
/// \note The engine looks the kernels up under the model folder's name. A
///       folder named like a shipped model needs nothing. Under another name
///       the kernels of a shipped model are used: the one asked for, else the
///       only shipped model of the family. The kernels belong to the model's
///       dimensions and nothing checks that a package matches them (a mismatch
///       is wrong output or a hang), so a guess is made only when the family
///       has exactly one shipped kernel folder.
inline std::string select_npu_kernels(model_list& models, const std::string& tag, const std::string& family,
                                      const std::string& requested, const std::filesystem::path& xclbin_root) {
    namespace fs = std::filesystem;
    const auto info = models.get_model_info(tag).second;
    if (info.value(flm::backend::kBackendKey, std::string()) != flm::backend::kFlmBackendId) {
        if (!requested.empty()) {
            throw std::runtime_error("--xclbin-folder is for an NPU model folder; " + tag + " does not run on the NPU");
        }
        return "";
    }
    const std::string folder_name = fs::path(info.at("name").get<std::string>()).filename().string();
    const auto has = [&](const std::string& name) {
        std::error_code error;
        return fs::is_directory(xclbin_root / name, error);
    };
    std::vector<std::string> candidates;
    for (const auto& name : models.npu_folders_of_family(family)) {
        if (has(name)) candidates.push_back(name);
    }
    const std::string choices = local_detail::join(candidates);

    if (!requested.empty()) {
        if (!has(requested)) {
            throw std::runtime_error("--xclbin-folder " + requested + ": no such folder under " +
                                     xclbin_root.string() +
                                     (choices.empty() ? std::string() : " (family " + family + ": " + choices + ")"));
        }
        if (requested != folder_name) models.set_local_xclbin_folder(tag, requested);
        return "";
    }
    if (has(folder_name)) return "";
    if (candidates.size() == 1) {
        models.set_local_xclbin_folder(tag, candidates.front());
        return "No kernels are filed under '" + folder_name + "'; running on those of " + candidates.front() +
               " (the only " + family + " model; --xclbin-folder picks another)";
    }
    throw std::runtime_error("NPU kernels are looked up under the model folder's name and none are filed under '" +
                             folder_name + "'; pass --xclbin-folder <name>" +
                             (choices.empty() ? " (no shipped model of family " + family + " found under " + xclbin_root.string() + ")"
                                              : ", one of: " + choices));
}

}  // namespace flm::catalog
