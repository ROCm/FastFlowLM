/// \file local_model_oga.hpp
/// \brief The OGA package format of a model folder (oga-hipep backend, GPU)
/// \note An OGA package is marked by genai_config.json. That file names the
///       graphs the package loads, its context limit and whether it has a
///       vision tower. Including this header makes build_local_model_entry
///       read such folders.
#pragma once
#include "AutoModel/backend_contract.hpp"
#include "AutoModel/local_model_entry.hpp"

#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>

namespace flm::catalog {

/// \brief the file that marks an OGA package
inline constexpr const char* kOgaConfigFile = "genai_config.json";

namespace local_detail {

/// \brief every string a genai_config.json names under "filename" or "config_filename"
/// \note These are the files the package loads: the decoder, the embedding and
///       vision graphs, and the processor config. Anything the config does not
///       name is not checked here.
inline void collect_named_files(const nlohmann::json& node, std::set<std::string>& out) {
    if (node.is_object()) {
        for (const auto& [key, value] : node.items()) {
            if ((key == "filename" || key == "config_filename") && value.is_string()) {
                out.insert(value.get<std::string>());
            } else {
                collect_named_files(value, out);
            }
        }
    } else if (node.is_array()) {
        for (const auto& value : node) collect_named_files(value, out);
    }
}

/// \brief whether a name is one relative path inside the model folder
inline bool stays_inside(const std::string& name) {
    if (name.empty()) return false;
    const std::filesystem::path relative = std::filesystem::path(name).lexically_normal();
    return !relative.has_root_path() && !relative.empty() && *relative.begin() != "..";
}

}  // namespace local_detail

/// \brief read an OGA package folder
/// \param folder the absolute model folder
/// \throws std::runtime_error if genai_config.json is unreadable or names a file outside the folder
inline LocalPackageFacts read_oga_package(const std::filesystem::path& folder, const std::string&) {
    using local_detail::is_file;
    LocalPackageFacts facts;
    const nlohmann::json genai = local_detail::parse_file(folder / kOgaConfigFile, "genai_config.json");

    // The files the package loads, plus the .data file next to each graph.
    facts.files = {"genai_config.json", "config.json", "tokenizer.json"};
    std::set<std::string> named_files;
    local_detail::collect_named_files(genai, named_files);
    for (const auto& name : named_files) {
        if (!local_detail::stays_inside(name)) {
            throw std::runtime_error("genai_config.json names a file outside the model folder: " + name);
        }
        facts.files.insert(name);
        if (name.size() > 5 && name.compare(name.size() - 5, 5, ".onnx") == 0 &&
            is_file(folder / (name + ".data"))) {
            facts.files.insert(name + ".data");
        }
    }
    for (const char* optional : {"tokenizer_config.json", "chat_template.jinja", "processor_config.json"}) {
        if (is_file(folder / optional)) facts.files.insert(optional);
    }

    const auto model_node = genai.contains("model") && genai["model"].is_object()
        ? genai["model"] : nlohmann::json::object();
    facts.vision = model_node.contains("vision");
    if (genai.contains("search") && genai["search"].is_object() &&
        genai["search"].contains("max_length") && genai["search"]["max_length"].is_number_integer()) {
        facts.package_limit = genai["search"]["max_length"].get<int>();
    } else if (model_node.contains("context_length") && model_node["context_length"].is_number_integer()) {
        facts.package_limit = model_node["context_length"].get<int>();
    }
    return facts;
}

/// \brief the OGA format, known to build_local_model_entry once this header is included
inline const bool kOgaPackageRegistered = register_local_package_format(
    {"OGA package", kOgaConfigFile, flm::backend::kOgaHipepBackendId, flm::backend::kOgaHipepFamilySuffix,
     read_oga_package});

}  // namespace flm::catalog
