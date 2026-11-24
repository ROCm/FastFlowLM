// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file safe_tensors.hpp
/// \brief SafeTensors class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.10
/// \note This class is used to load weights from a safe-tensors file.
#pragma once

#include "typedef.hpp"
#include "buffer.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

/*
Safetensor data format:

|8 Bytes, N, size of the header|N Bytes, json header|binary data|
Json header format:
{
    "TENSOR_NAME_1": {
        "shape": [1, 1, 1, 1], (List integers)
        "dtype": "float16", (String)
        "data_offsets": [BEGIN, END]
    },
    "TENSOR_NAME_2": {
        ...
    },
    ...
    "__metadata__": {
        ...
    }
}

*/

/// \brief Tensor metadata
typedef struct {
    std::string name;
    std::vector<size_t> shape;
    std::string dtype;
    std::vector<size_t> offsets;
    size_t size;
    size_t byte_size;
} tensor_metadata;

/// \brief SafeTensors class
class SafeTensors{
private:
    std::string model_path;
    std::ifstream file;
    nlohmann::json metadata;

    size_t _get_data_size(tensor_metadata tensor_meta){
        const std::vector<std::pair<std::string, size_t>> data_size = {
            std::make_pair("BF16", 2),
            std::make_pair("FP16", 2),
            std::make_pair("F32", 4),
            std::make_pair("BOOL", 1),
            std::make_pair("U8", 1),
            std::make_pair("I8", 1),
            std::make_pair("BOOL", 1),
            std::make_pair("I16", 2),
            std::make_pair("I32", 4),
            std::make_pair("I64", 8),
            std::make_pair("Q4NX", 1) // special extention for q4nx
        };

        std::string dtype = tensor_meta.dtype;
        for (size_t i = 0; i < data_size.size(); i++){
            if (data_size[i].first == dtype){
                return data_size[i].second;
            }
        }
        std::cerr << "Unknown dtype: " << dtype << std::endl;
        exit(1);
    }
    void _load_tensors(){
        this->_open_file();

        // read first 8 bytes (size of the header)
        unsigned long long header_size;
        this->file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
        
        // read the json header from the current byte to byte size of header_size

        std::string json_str(header_size, '\0');
        this->file.read(&json_str[0], header_size);
        nlohmann::json json_data = nlohmann::json::parse(json_str);

        // read __metadata__
        JSON_GET(this->metadata, json_data, "__metadata__", nlohmann::json(), nlohmann::json);
        
        // read each entry
        // __metadata__ is skipped explicitly: it is free form, and some writers put a
        // "dtype" key in it, which would otherwise make it look like a tensor entry.
        for (auto& [key, value] : json_data.items()){
            if (key != "__metadata__" && value.contains("dtype") && value.contains("shape")){
                tensor_metadata tensor_meta;
                std::vector<size_t> shape = value["shape"];
                std::vector<size_t> offsets = value["data_offsets"];
                tensor_meta.name = key;
                tensor_meta.shape = shape;
                tensor_meta.dtype = std::string(value["dtype"]);
                tensor_meta.offsets = offsets;
                // correct shape and compute and offsets
                tensor_meta.size = 1;
                for (size_t i = 0; i < shape.size(); i++){
                    if (shape[i] == 0){
                        shape[i] = 1;
                        tensor_meta.shape[i] = 1; // 0 dim is not valid, set to 1
                    }
                    tensor_meta.size *= shape[i];
                }
                size_t data_size = this->_get_data_size(tensor_meta);
                if (data_size > 0){
                    tensor_meta.byte_size = data_size * tensor_meta.size;
                    if (tensor_meta.dtype == "Q4NX"){
                        tensor_meta.byte_size = tensor_meta.byte_size * 5 / 8;
                    }
                } else {
                    tensor_meta.byte_size = -1;
                }
                for (size_t i = 0; i < offsets.size(); i++){
                    tensor_meta.offsets[i] += (8 + header_size); // 8 bytes for the header_size, header_size bytes for the json header
                }
                this->tensors_data.push_back(tensor_meta);
                LOG_VERBOSE_IF_ELSE(1, tensor_meta.shape.size() == 2, 
                "Loaded tensor: " << tensor_meta.name << " with shape: " << tensor_meta.shape[0] << "x" << tensor_meta.shape[1] << " and dtype: " << tensor_meta.dtype << ", offset: " << tensor_meta.offsets[0] << " and byte_size: " << tensor_meta.byte_size,
                "Loaded tensor: " << tensor_meta.name << " with shape: " << tensor_meta.shape[0] << "x1" << " and dtype: " << tensor_meta.dtype << ", offset: " << tensor_meta.offsets[0] << " and byte_size: " << tensor_meta.byte_size);
            }
        }
        LOG_VERBOSE(1, "Loaded " << this->tensors_data.size() << " tensors");
    }

    void _open_file(){
        if (!this->file.is_open()){
            this->file.open(this->model_path, std::ios::binary);
            if (!this->file.is_open()){
                // Fallback: if the path ends with ".q4nx", retry with ".safetensors"
                const std::string q4nx_ext = ".q4nx";
                if (this->model_path.size() >= q4nx_ext.size() &&
                    this->model_path.compare(this->model_path.size() - q4nx_ext.size(), q4nx_ext.size(), q4nx_ext) == 0){
                    std::string fallback_path = this->model_path.substr(0, this->model_path.size() - q4nx_ext.size()) + ".safetensors";
                    this->file.clear();
                    this->file.open(fallback_path, std::ios::binary);
                    if (this->file.is_open()){
                        this->model_path = fallback_path;
                    }
                }
                if (!this->file.is_open()){
                    std::cerr << "Failed to open file: " << this->model_path << std::endl;
                    exit(1);
                }
            }
        }
    }

protected:
    std::vector<tensor_metadata> tensors_data;

public:
    /// \brief Constructor
    /// \param model_path the model path
    SafeTensors(const std::string& model_path){
        
        this->model_path = model_path;
        // std::cout << "Using model: " << model_path << std::endl;
        this->_open_file();
        this->_load_tensors();
    }

    /// \brief Destructor
    ~SafeTensors(){
        if (this->file.is_open()){
            this->file.close();
        }
    }

    /// \brief Load the weights
    /// \param weight_buffer the weight buffer
    /// \param weights_name the weights name
    /// \return the weights name
    std::string load_weights(bytes& weight_buffer, std::string weights_name){
        this->_open_file();
        // std::cout << "Loading weights: " << weights_name << std::endl;

        for (size_t i = 0; i < this->tensors_data.size(); i++){
            if (this->tensors_data[i].name == weights_name){
                if (weight_buffer.size() == 0){
                    weight_buffer.resize(this->tensors_data[i].byte_size);
                }
                else if (this->tensors_data[i].byte_size != weight_buffer.size()){
                    std::cerr << weights_name << " size mismatch: " << this->tensors_data[i].byte_size << " != " << weight_buffer.size() << std::endl;
                    exit(1);
                }
                this->file.seekg(this->tensors_data[i].offsets[0]);
                this->file.read(reinterpret_cast<char*>(weight_buffer.data()), this->tensors_data[i].byte_size);
                return this->tensors_data[i].dtype;
            }
        }
        std::cerr << "Weight not found: " << weights_name << std::endl;
        exit(1);
    }

    /// \brief Load the weights with offset
    /// \param weight_buffer the weight buffer
    /// \param weights_name the weights name
    /// \param byte_offset the byte offset in the weight buffer
    /// \return the weights name
    std::string load_weights(bytes& weight_buffer, std::string weights_name, size_t byte_offset){
        this->_open_file();
        // std::cout << "Loading weights: " << weights_name << " at offset: " << byte_offset << std::endl;

        for (size_t i = 0; i < this->tensors_data.size(); i++){
            if (this->tensors_data[i].name == weights_name){
                if (weight_buffer.size() == 0){
                    std::cerr << weights_name << " buffer size is 0" << std::endl;
                    exit(1);
                }
                if (weight_buffer.size() < byte_offset + this->tensors_data[i].byte_size){
                    std::cerr << weights_name << " buffer size insufficient: " << weight_buffer.size() << " < " << byte_offset << " + " << this->tensors_data[i].byte_size << std::endl;
                    exit(1);
                }
                this->file.seekg(this->tensors_data[i].offsets[0]);
                this->file.read(reinterpret_cast<char*>(weight_buffer.data() + byte_offset), this->tensors_data[i].byte_size);
                return this->tensors_data[i].dtype;
            }
        }
        std::cerr << "Weight not found: " << weights_name << std::endl;
        exit(1);
    }

    /// \brief Get the tensor metadata
    /// \param tensor_name the tensor name
    /// \return the tensor metadata
    tensor_metadata get_tensor_metadata(std::string tensor_name){
        for (size_t i = 0; i < this->tensors_data.size(); i++){
            if (this->tensors_data[i].name == tensor_name){
                return this->tensors_data[i];
            }
        }
        std::cerr << "Tensor not found: " << tensor_name << std::endl;
        exit(1);
    }


    /// \brief Write the safetensors
    /// \param output_path the output path
    void write_safetensors(std::string output_path){
        // to be implemented
    }

    /// \brief Get the metadata
    /// \return the metadata
    nlohmann::json get_metadata(){
        return this->metadata;
    }

    /// \brief Switch the model
    /// \param new_model_path the new model path
    void switch_model(std::string new_model_path){
        this->file.close();
        this->model_path = new_model_path;
        this->tensors_data.clear();
        this->_open_file();
        this->_load_tensors();
    }
};
