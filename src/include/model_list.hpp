/// \file model_list.hpp
/// \brief model_list class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This class is used to manage the model list.
#pragma once
#include "nlohmann/json.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <any>
#include <unordered_set>
#include <filesystem>
#include <stdexcept>
// Only the header_print* macros are needed here. Including utils/utils.hpp
// instead would drag in the NPU runtime headers and make this class impossible
// to unit test without an XRT/HRX toolchain.
#include "utils/debug_utils.hpp"

/// \note This class is used to manage the model list.
class model_list {
    public:
        std::unordered_set<std::string> all_tags;
        /// \brief constructor
        model_list(){}


        /// \brief constructor
        /// \param list_path the path to the model list
        /// \param exe_dir the executable directory for resolving relative paths
        /// \param platform the NPU generation this machine has ("aie2p" or
        ///        "aie_next"); an entry whose supported_platforms names a
        ///        different generation is pruned, because these artifacts do
        ///        not run on this silicon
        /// \param backends the kernel flows this build links, as backend ids;
        ///        an entry whose flow is not among them is pruned too
        /// \note Two independent axes. The generation says what the silicon can
        ///       run; the backend list says which kernels were linked in. Which
        ///       flow runs an entry is not a third thing to look up: a tag is
        ///       served by exactly one flow and its name says which, so a
        ///       family ending in "-rai" is ryzenai-corelib's and anything else
        ///       is the FastFlowLM kernels'. That is why the two never contend
        ///       for a tag -- phi4-mini-it and phi4-mini-it-rai are separate
        ///       entries, with their own artifacts and their own silicon.
        model_list(std::string& list_path, std::string& exe_dir,
                   std::string platform = "aie2p",
                   std::vector<std::string> backends = {"flm"}){
            this->list_path = list_path;
            this->platform_ = std::move(platform);
            this->backends_ = std::move(backends);
            std::ifstream config_file(list_path);
            if (!config_file.is_open()) {
                std::cerr << "Failed to open config file: " << list_path << std::endl;
                exit(1);
            }
            this->config = nlohmann::json::parse(config_file);
            // Resolve model_root_path relative to executable directory
            std::string relative_model_path = this->config["model_path"];
            std::filesystem::path root_path = std::filesystem::path(exe_dir) / relative_model_path;
            this->model_root_path = root_path.string();
            config_file.close();

            // Prune before indexing: all_tags must describe what this machine can
            // actually run, so an unsupported tag fails at validation instead of
            // failing much later inside the model backend.
            this->apply_support_filter();

            // Populate all_tags set
            for (const auto& [model_type, sizes] : this->config["models"].items()) {
                // insert default tag without size
                all_tags.insert(model_type);
                for (const auto& [size, model_info] : sizes.items()) {
                    all_tags.insert(model_type + ":" + size);
                }
            }

            // An empty catalog is a real state, not a crash: it is what an
            // install looks like when every entry names other silicon or other
            // kernels. Say so once, here, where the reason is still known --
            // "Models:" followed by nothing tells the user only that something
            // is wrong -- and then let the commands answer for themselves.
            if (all_tags.empty()) {
                header_print_r("ERROR",
                               "No models in " + this->list_path +
                                   " run on this NPU (" + this->platform_ +
                                   ") with the kernels this build has (" +
                                   this->backend_summary() + ")");
            }
        }

        /// \brief the NPU generation this catalog was filtered for
        /// \return "aie2p" or "aie_next"
        const std::string& platform() const { return this->platform_; }

        /// \brief get the model info
        /// \param tag the tag of the model
        /// \return the model info
        std::pair<std::string, nlohmann::json> get_model_info(const std::string tag) const {
            static std::string last_error_tag = "";
            std::string new_tag = rectify_model_tag(tag);
            bool model_found = false;
            // get model type, the string before ':' in the tag
            std::string model_type;
            std::string model_size;

            if (new_tag.find(':') != std::string::npos) {
                model_type = new_tag.substr(0, new_tag.find(':'));
                model_size = new_tag.substr(new_tag.find(':') + 1);
            }
            else {
                model_type = new_tag;
                model_size = "";
            }
            
            // find the model subset first, compare with the key of the model
            bool model_subset_found = false;
            for (const auto& [key, model] : this->config["models"].items()) {
                if (key == model_type) {
                    model_subset_found = true;
                    break;
                }
            } 
            if (model_subset_found) {
                bool model_found = false;
                for (const auto& [key, model] : this->config["models"][model_type].items()) {
                    if (key == model_size) { // if the size is found, return the model
                        model_found = true;
                        return std::make_pair(new_tag, model);
                    }
                } 
                if (!model_found) {
                    auto fallback = this->fallback_model();
                    if (last_error_tag != new_tag) {
                        last_error_tag = new_tag;
                        header_print_r("ERROR", "Model not found: " + model_size + " in subset " + model_type);
                        header_print_r("ERROR", "Using default model: " + fallback.first);
                    }
                    return fallback;
                }
            }
            else{
                auto fallback = this->fallback_model();
                if (last_error_tag != new_tag) {
                    last_error_tag = new_tag;
                    header_print_r("ERROR", "Model subset not found: " << model_type << "; using default model: " << fallback.first);
                }
                return fallback;
            }
            return this->fallback_model();
        }

        /// \brief cut the tag, some program adds a prefix to the tag, like "Ollama/llama3.2-1B", we need to cut the prefix
        /// \param tag the tag of the model
        /// \return the model type, string
        std::string cut_tag(const std::string tag) const {
            std::string new_tag = tag;
            if (tag.find('/') != std::string::npos) {
                new_tag = tag.substr(tag.find('/') + 1);
            }
            return new_tag;
        }

        /// \brief rectify the model tag, remove / and replace with actuall tag if size is not specified
        /// \param original_tag the original tag of the model
        /// \return the rectified model tag, string
        std::string rectify_model_tag(const std::string original_tag) const {
            std::string new_tag = this->cut_tag(original_tag);
            // check if size is specified
            if (new_tag.find(':') == std::string::npos) {
                const std::string model_type = new_tag;
                // A family this build has no kernels for (or a misspelled one) has no
                // sizes to pick from. Return the tag untouched so get_model_info
                // reports it rather than dereferencing a null subset.
                const auto& models = this->config["models"];
                if (!models.contains(model_type) || models.at(model_type).empty()) {
                    return new_tag;
                }
                // get the first size in the subset
                std::string model_size = models.at(model_type).begin().key();
                new_tag = model_type + ":" + model_size;
            }
            return new_tag;
        }

        /// \brief get the model root path
        /// \return the model root path, string
        std::string get_model_root_path(){
            return this->model_root_path;
        }

        /// \brief get all the models
        /// \return all the models in json
        nlohmann::json get_all_models(){
            nlohmann::json response = {
                {"models", nlohmann::json::array()}
            };

            for (const auto& [model_type, model_subset] : this->config["models"].items()) {
                for (const auto& [size, model_info] : model_subset.items()) {
                    nlohmann::json model_entry = model_info;
                    model_entry["name"] = model_type + ":" + size;
                    model_entry["model"] = model_type + ":" + size;
                    response["models"].push_back(model_entry);
                }
            }
            return response;
        }

        /// \brief get all the models
        /// \return all the models in json
        nlohmann::json get_all_models_ollama() {
            nlohmann::json response = {
                {"models", nlohmann::json::array()}
            };

            for (const auto& [model_type, model_subset] : this->config["models"].items()) {
                if (model_type == "whisper-v3") continue;
                else if (model_type == "embed-gemma") continue;
                for (const auto& [size, model_info] : model_subset.items()) {
                    nlohmann::json model_entry = {
                        {"name", model_type + ":" + size},
                        {"model", model_type + ":" + size},
                        {"details", {
                            {"family", model_info["details"]["family"]},
                            {"parameter_size", model_info["details"]["parameter_size"]},
                            {"quantization_level", model_info["details"]["quantization_level"]}
                        }}
                    };
                    response["models"].push_back(model_entry);
                }
            }
            return response;
        }

        /// \brief get all the models
        /// \return all the models in json
        nlohmann::json get_all_models_openai() {
            nlohmann::json response = {
                {"object", "list"},
                {"data", nlohmann::json::array()},
                {"object", "list" }
            };

            std::time_t now = std::time(nullptr);

            for (const auto& [model_type, model_subset] : this->config["models"].items()) {
                if (model_type == "whisper-v3") continue;
                else if (model_type == "embed-gemma") continue;
                for (const auto& [size, model_info] : model_subset.items()) {
                    // id uses the same "type:size" convention; created uses current epoch seconds
                    nlohmann::json model_entry = {
                        {"id", model_type + ":" + size},
                        {"object", "model"},
                        {"created", static_cast<long long>(now)},
                        {"owned_by", "FastFlowLM"}
                    };
                    response["data"].push_back(model_entry);
                }
            }

            return response;
        }

        /// \brief get the model path
        /// \param tag the tag of the model
        /// \return the model path, string
        std::string get_model_path(const std::string& tag){
            std::string new_tag = this->rectify_model_tag(tag);
            auto [new_tag_unused, model_info] = this->get_model_info(new_tag);
            std::string model_name = model_info["name"];
            std::filesystem::path full_path = std::filesystem::path(this->model_root_path) / model_name;
            return full_path.string();
        }

        bool is_model_supported(const std::string& tag) {
            return all_tags.find(tag) != all_tags.end();
        }
        
    private:
        std::string list_path;
        nlohmann::json config;
        std::string model_root_path;
        std::string platform_;
        /// \brief the backend ids this build links
        /// \note Spelled as plain strings rather than reached through
        ///       flm::backend: AutoModel/model_backend.hpp pulls in causal_lm
        ///       and the NPU runtime headers, and keeping those out is what
        ///       lets this class be unit tested without an XRT toolchain. The
        ///       key name mirrors flm::backend::kBackendKey.
        std::vector<std::string> backends_;

        /// \brief the backend ids, rendered for an error message
        std::string backend_summary() const {
            std::string joined;
            for (const auto& id : this->backends_) {
                if (!joined.empty()) joined += ", ";
                joined += id;
            }
            return joined.empty() ? std::string("none") : joined;
        }

        /// \brief the kernel flow that runs a tag, read off the tag itself
        /// \param model_type the family part of the tag, e.g. "phi4-mini-it-rai"
        /// \return "rai" for a family ending in "-rai", "flm" otherwise
        /// \note The name is the mechanism, not a label on top of one. A
        ///       corelib model is packaged differently from its FastFlowLM
        ///       namesake -- different weights, different quantization -- so it
        ///       is a different model to pull and a different tag to ask for,
        ///       and there is nothing left for a catalog key to disambiguate.
        static std::string backend_for_family(const std::string& model_type) {
            static const std::string suffix = "-rai";
            if (model_type.size() > suffix.size() &&
                model_type.compare(model_type.size() - suffix.size(),
                                   suffix.size(), suffix) == 0) {
                return "rai";
            }
            return "flm";
        }

        /// \brief whether an entry claims support for this machine's silicon
        /// \param entry the size entry
        /// \param tag the "family:size" tag, used only in error messages
        /// \return true if the entry runs on this->platform_
        /// \note An entry that says nothing runs on every generation. The
        ///       shipped catalog names one on every entry, because artifacts
        ///       are built for a generation and saying so is how a model that
        ///       cannot run here fails at validation rather than deep inside a
        ///       kernel; the default is what keeps an older catalog, or a
        ///       hand-made development tree, readable.
        bool entry_supports(const nlohmann::json& entry, const std::string& tag) const {
            const auto supported = entry.find("supported_platforms");
            if (supported == entry.end()) return true;
            if (!supported->is_array() || supported->empty()) {
                throw std::runtime_error(
                    "supported_platforms must be a non-empty array: " + tag);
            }
            for (const auto& value : *supported) {
                if (!value.is_string()) {
                    throw std::runtime_error(
                        "supported_platforms must contain strings: " + tag);
                }
                if (value.get<std::string>() == this->platform_) return true;
            }
            return false;
        }

        /// \brief whether this build links a given kernel flow
        bool build_has(const std::string& id) const {
            for (const auto& have : this->backends_) {
                if (have == id) return true;
            }
            return false;
        }

        /// \brief drop entries this build cannot run and settle their backend
        /// \note After this runs the config has exactly the shape it had before
        ///       any of this existed, plus one key: "backend", the flow that
        ///       will run the entry. It is written here rather than read from
        ///       the file because the tag is what decides it, and the tag is
        ///       not part of the entry -- by the time AutoModel sees the entry,
        ///       the name it came from is gone. The generation, by contrast, is
        ///       settled once the entry is kept, so its key is erased.
        void apply_support_filter() {
            std::vector<std::string> empty_families;

            for (auto& [model_type, model_subset] : this->config["models"].items()) {
                std::vector<std::string> unsupported_sizes;
                const std::string backend = backend_for_family(model_type);

                for (auto& [size, model_info] : model_subset.items()) {
                    const std::string tag = model_type + ":" + size;
                    if (!entry_supports(model_info, tag) ||
                        !this->build_has(backend)) {
                        unsupported_sizes.push_back(size);
                        continue;
                    }
                    model_info.erase("supported_platforms");
                    model_info["backend"] = backend;
                }

                for (const auto& size : unsupported_sizes) model_subset.erase(size);
                if (model_subset.empty()) empty_families.push_back(model_type);
            }

            for (const auto& model_type : empty_families) {
                this->config["models"].erase(model_type);
            }
        }

        /// \brief the entry to fall back on when a tag cannot be resolved
        /// \return the fallback tag and its info
        /// \note llama3.2:1b is the historical default, but it is pruned on
        ///       silicon that cannot run it, so fall back to whatever survived.
        std::pair<std::string, nlohmann::json> fallback_model() const {
            const auto& models = this->config["models"];
            if (models.contains("llama3.2") &&
                models["llama3.2"].contains("1b")) {
                return std::make_pair("llama3.2:1b", models["llama3.2"]["1b"]);
            }
            for (const auto& [model_type, model_subset] : models.items()) {
                for (const auto& [size, model_info] : model_subset.items()) {
                    return std::make_pair(model_type + ":" + size, model_info);
                }
            }
            throw std::runtime_error("No models available for NPU platform " +
                                     this->platform_ + " with backends " +
                                     this->backend_summary());
        }

};
