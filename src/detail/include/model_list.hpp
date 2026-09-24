/// \file model_list.hpp
/// \brief model_list class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.10
/// \note This class is used to manage the model list.
#pragma once
#include "nlohmann/json.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "utils/utils.hpp"

#define __FLM_VERSION__ "0.9.6"

/// \note This class is used to manage the model list.
class model_list {
    public:
        /// \brief files required for the model
        static constexpr const char* model_files[] = {
            "config.json",
            "tokenizer.json",
            "attn.xclbin",
            "mm.xclbin",
            "dequant.xclbin",
            "layer.xclbin",
            "lm_head.xclbin",
            "model.q4nx",
            "tokenizer_config.json"
        };
        static constexpr const char* vision_model_files[] = {
            "vision_attn.xclbin",
            "vision_mm.xclbin",
            "vision_weight.q4nx"
        };
        /// \brief number of model files
        static constexpr int model_files_count = sizeof(model_files) / sizeof(model_files[0]);
        
        /// \brief number of vision model files
        static constexpr int vision_model_files_count = sizeof(vision_model_files) / sizeof(vision_model_files[0]);

        /// \brief constructor
        model_list(){}

        /// \brief constructor
        /// \param list_path the path to the model list
        /// \param exe_dir the executable directory for resolving relative paths
        model_list(std::string& list_path, std::string& exe_dir){
            this->list_path = list_path;
            std::ifstream config_file(list_path);
            if (!config_file.is_open()) {
                std::cerr << "Failed to open config file: " << list_path << std::endl;
                exit(1);
            }
            this->config = nlohmann::json::parse(config_file);
            // Resolve model_root_path relative to executable directory
            std::string relative_model_path = this->config["model_path"];
            #ifdef _WIN32
            this->model_root_path = exe_dir + "\\" + relative_model_path;
            #else
            this->model_root_path = exe_dir + "/" + relative_model_path;
            #endif
            config_file.close();
        }

        /// \brief get the model info
        /// \param tag the tag of the model
        /// \return the model info
        nlohmann::json get_model_info(const std::string& tag){
            static std::string last_error_tag = "";
            std::string new_tag = this->cut_tag(tag);
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
                // find the model in the subset
                // check if a size is specified in the tag
                // if not use the first model in the subset
                if (model_size.empty()) {
                    model_size = this->config["models"][model_type].begin().key();
                    return this->config["models"][model_type][model_size];
                }
                bool model_found = false;
                for (const auto& [key, model] : this->config["models"][model_type].items()) {
                    if (key == model_size) { // if the size is found, return the model
                        model_found = true;
                        return model;
                    }
                } 
                if (!model_found) {
                    if (last_error_tag != new_tag) {
                        last_error_tag = new_tag;
                        header_print("ERROR", "Model not found: " + model_size + " in subset " + model_type);
                        header_print("ERROR", "Using default model: llama3.2-1B");
                    }
                    return this->config["models"]["llama3.2"]["1b"];
                }
            }
            else{
                if (last_error_tag != new_tag) {
                    last_error_tag = new_tag;
                    header_print("ERROR", "Model subset not found: " << model_type << "; using default model: llama3.2-1B");
                }
                return this->config["models"]["llama3.2"]["1b"];
            }
            return this->config["models"]["llama3.2"]["1b"];
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
                    nlohmann::json model_entry = {
                        {"name", model_type + ":" + size},
                        {"model", model_type + ":" + size},
                        {"modified_at", "2024-03-28T00:00:00Z"},
                        {"details", {
                            {"format", "gguf"},
                            {"family", model_type},
                            {"parameter_size", size},
                            {"quantization_level", "Q4_0"}
                        }}
                    };
                    response["models"].push_back(model_entry);
                }
            }
            return response;
        }

        /// \brief get the model path
        /// \param tag the tag of the model
        /// \return the model path, string
        std::string get_model_path(const std::string& tag){
            std::string new_tag = this->cut_tag(tag);
            std::string model_name = this->get_model_info(new_tag)["name"];
            #ifdef _WIN32
            std::string model_path = this->model_root_path + "\\" + model_name;
            #else
            std::string model_path = this->model_root_path + "/" + model_name;
            #endif
            return model_path;
        }

        /// \brief cut the tag, some program adds a prefix to the tag, like "Ollama/llama3.2-1B", we need to cut the prefix
        /// \param tag the tag of the model
        /// \return the model type, string
        std::string cut_tag(const std::string& tag){
            std::string new_tag = tag;
            if (tag.find('/') != std::string::npos) {
                new_tag = tag.substr(tag.find('/') + 1);
            }
            return new_tag;
        }
        
    private:
        std::string list_path;
        nlohmann::json config;
        std::string model_root_path;

};