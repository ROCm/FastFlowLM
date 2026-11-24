// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>
#include "tensor_utils/safe_tensors.hpp"

/// \brief Tensors of a Hugging Face checkpoint, by their HF names.
/// \param path a checkpoint folder with model.safetensors.index.json, or one .safetensors file
class hf_checkpoint {
public:
    explicit hf_checkpoint(const std::string &path) : root_(path) {
        if (!std::filesystem::is_directory(path)) {
            single_file_ = path;
            return;
        }
        const std::string index_path = path + "/model.safetensors.index.json";
        std::ifstream index_file(index_path);
        if (!index_file.is_open()) {
            std::cerr << "No " << index_path << " (expected an HF checkpoint folder)" << std::endl;
            exit(1);
        }
        nlohmann::json index = nlohmann::json::parse(index_file);
        for (auto &[tensor_name, shard] : index.at("weight_map").items()) {
            shard_of_[tensor_name] = shard.get<std::string>();
        }
    }

    tensor_metadata get_tensor_metadata(const std::string &tensor_name) {
        return file_for(tensor_name).get_tensor_metadata(tensor_name);
    }

    void load_weights(bytes &weight_buffer, const std::string &tensor_name) {
        file_for(tensor_name).load_weights(weight_buffer, tensor_name);
    }

private:
    SafeTensors &file_for(const std::string &tensor_name) {
        std::string file_path = single_file_;
        if (file_path.empty()) {
            auto shard = shard_of_.find(tensor_name);
            if (shard == shard_of_.end()) {
                std::cerr << "Tensor " << tensor_name << " is not in " << root_ << std::endl;
                exit(1);
            }
            file_path = root_ + "/" + shard->second;
        }
        std::unique_ptr<SafeTensors> &opened = opened_[file_path];
        if (!opened) {
            opened = std::make_unique<SafeTensors>(file_path);
        }
        return *opened;
    }

    std::string root_;
    std::string single_file_;
    std::map<std::string, std::string> shard_of_;
    std::map<std::string, std::unique_ptr<SafeTensors>> opened_;
};
