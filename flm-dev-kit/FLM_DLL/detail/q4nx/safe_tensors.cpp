// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

// =============================================================================
// File        : safe_tensors.cpp
// Created     : 2025-03-28
// Description : This is a source file for the SafeTensors class.
//               It is used to load weights from a safe-tensors file.
// =============================================================================

#include "tensor_utils/safe_tensors.hpp"

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

SafeTensors::SafeTensors(const std::string& model_path){
    this->model_path = model_path;
    // std::cout << "Using model: " << model_path << std::endl;
    this->_open_file();
    this->_load_tensors();
}

void SafeTensors::_open_file(){
    if (!this->file.is_open()){
        this->file.open(this->model_path, std::ios::binary);
        if (!this->file.is_open()){
            std::cerr << "Failed to open file: " << this->model_path << std::endl;
            exit(1);
        }
    }
}

SafeTensors::~SafeTensors(){
    if (this->file.is_open()){
        this->file.close();
    }
}

void SafeTensors::switch_model(std::string new_model_path){
    this->file.close();
    this->model_path = new_model_path;
    this->tensors_data.clear();
    this->_open_file();
    this->_load_tensors();
}

void SafeTensors::_load_tensors(){
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

size_t SafeTensors::_get_data_size(tensor_metadata tensor_meta){
    const std::vector<std::pair<std::string, size_t>> data_size = {
        std::make_pair("BF16", 2),
        std::make_pair("FP16", 2),
        std::make_pair("F32", 4),
        std::make_pair("U8", 1),
        std::make_pair("I8", 1),
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

std::string SafeTensors::load_weights(bytes& weight_buffer, std::string weights_name){
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

std::string SafeTensors::load_weights(bytes& weight_buffer, std::string weights_name, size_t byte_offset){
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

void SafeTensors::write_safetensors(std::string output_path){
    // to be implemented
}

nlohmann::json SafeTensors::get_metadata(){
    return this->metadata;
}

bool SafeTensors::has_tensor(std::string tensor_name){
    for (size_t i = 0; i < this->tensors_data.size(); i++){
        if (this->tensors_data[i].name == tensor_name){
            return true;
        }
    }
    return false;
}

tensor_metadata SafeTensors::get_tensor_metadata(std::string tensor_name){
    for (size_t i = 0; i < this->tensors_data.size(); i++){
        if (this->tensors_data[i].name == tensor_name){
            return this->tensors_data[i];
        }
    }
    std::cerr << "Tensor not found: " << tensor_name << std::endl;
    exit(1);
}
