// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file q4_npu_eXpress.hpp
/// \brief Q4NX class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.10
/// \note This class is used to quantize the model.
#pragma once

#include "safe_tensors.hpp"

// compared with the safetensor, the q4nx data type is added.
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

// Q4NX quantization formula:
// q = round(r / s + zp)
// s = scale
// r = real value
// zp = zero point
// q = quantized value
// Dequantization formula:
// r = s * (q - zp)


// Q4NX quantization config
// pr: row parallel size
// chunk_size_row: row chunk size
// chunk_size_col: column chunk size
// group_size: group size
// bits_per_weight: bits per weight
// Use 4 bits for weight
// Use bf16 for scale and zero point
// zp is saved as zp' = s * zp,
// so r = s * q - zp'


// We want to make sure each datablock is 1024 bytes:
//     1024 bytes -> 256 weight groups -> 256 x 32 = 8192 weights
//     if col_chunk_size = 256, then 8192 weights -> 8192 / 256 = 32 rows

// Final config:
// Tile size: 32 x 256, 32 x (256 / 32) = 32 x 8 groups -> 32 x 8 x 2 scales and zero points (bf16)
// 32 x 8 x 2 x 2 = 1024 bytes
// In each tile we have 32 x 256 weights, 4 bits per weight, which is 32 x 256 / 2 = 4096 bytes
// To match the 1024 data packing, we need 4096 / 1024 = 4 sub-tiles
// 32 / 4 = 8 rows, which is not good, we want row_parallel_size = 

/// \brief Q4NX class
/// \note This class is for Q4NX reordering and model conversion.
class Q4NX : public SafeTensors {
private:
    static constexpr int parallel_size = 16; // row parallel size
    static constexpr int row_chunk_size = 32; // row chunk size
    static constexpr int col_chunk_size = 256; // column chunk size
    static constexpr int group_size = 32; // group size

    static constexpr int qb = (group_size * 4 + 16 + 16) / group_size; // bits per weight
    static constexpr int weight_per_chunk = row_chunk_size * col_chunk_size; // weight per chunk
    static constexpr int num_groups_per_chunk_row = col_chunk_size / group_size; // number of groups per chunk row
    static constexpr int num_groups_per_row_parallel = row_chunk_size / parallel_size; // number of groups per row parallel
    static constexpr size_t group_size_bytes = weight_per_chunk * qb; // group size in bytes
    bool is_a_quantized_model;
    int layers;
    nlohmann::json _process_json_header(std::vector<tensor_metadata>& new_tensors_data){
        // generate the header
        nlohmann::json json_header;
        json_header["__metadata__"] = {
            {"row_parallel_size", parallel_size},
            {"row_chunk_size", row_chunk_size},
            {"col_chunk_size", col_chunk_size},
            {"group_size", group_size},
            {"scale_type", "bf16"},
            {"zero_point_type", "bf16"}
        };
        for (size_t i = 0; i < new_tensors_data.size(); i++) {
            json_header[new_tensors_data[i].name] = {
                {"shape", new_tensors_data[i].shape},
                {"dtype", new_tensors_data[i].dtype},
                {"data_offsets", new_tensors_data[i].offsets}
            };
        }
        return json_header;
    }

    void _grab_metadata(){
        this->is_a_quantized_model = false;
        this->layers = 0;
        for (size_t i = 0; i < this->tensors_data.size(); i++) {
            if (this->tensors_data[i].dtype == "Q4NX") {
                this->is_a_quantized_model = true;
                // recalculate the byte size
                this->tensors_data[i].byte_size = this->tensors_data[i].size * qb / 8;
            }
            if (this->tensors_data[i].name.find("input_layernorm.weight") != std::string::npos) {
                int layer_idx;
                sscanf(this->tensors_data[i].name.c_str(), "model.layers.%d.input_layernorm.weight", &layer_idx);
                this->layers = std::max(this->layers, layer_idx + 1);
            }
        }
        LOG_VERBOSE(1, "Layers: " << this->layers);
        LOG_VERBOSE(1, "Is quantized: " << this->is_a_quantized_model);
    }

public:
    struct npu_quantize_block{
        bf16 scales[num_groups_per_chunk_row][row_chunk_size]; // per-group scales and mins, shall view as 8 rows and 32 columns
        bf16 mins[num_groups_per_chunk_row][row_chunk_size]; // per-group scales and mins
        uint8_t qs[row_chunk_size / parallel_size][col_chunk_size][parallel_size / 2];           // 4-bit quantized values
    };
    typedef npu_quantize_block q4nx;

    /// \brief Constructor
    /// \param model_path the model path
    Q4NX(std::string model_path) : SafeTensors(model_path + "/model.q4nx") {
        this->_grab_metadata();
    }

    /// \brief Dequantize the weight
    /// \param weight the weight
    /// \param q4nx_weight the q4nx weight
    /// \param columns the columns
    template<typename T>
    static void q4nx_dequantize(bytes& weight, bytes& q4nx_weight, const int columns){
        const int n_npu_blocks_per_row = columns / col_chunk_size;
        const int n_npu_blocks = q4nx_weight.size() / sizeof(npu_quantize_block);
        const npu_quantize_block* npu_blocks = (npu_quantize_block*)q4nx_weight.data();
        if (weight.size() == 0){
            weight.resize(n_npu_blocks * weight_per_chunk * sizeof(T));
        }
        else if (weight.size() != n_npu_blocks * weight_per_chunk * sizeof(T)){
            std::cerr << "Weight size mismatch: " << weight.size() << " != " << n_npu_blocks * weight_per_chunk * sizeof(T) << std::endl;
            exit(1);
        }
        
        // create a row pointer of each row
        // T (*dst_ptr)[columns] = (T (*)[columns])weight.data();
        T* dst_ptr = (T*)weight.data();
        for (int i = 0; i < n_npu_blocks; i++) {
            const int row_offset = (i / n_npu_blocks_per_row) * row_chunk_size;
            const int col_offset = (i % n_npu_blocks_per_row) * col_chunk_size;
            float temp_array[row_chunk_size][col_chunk_size];
            for (int r = 0; r < row_chunk_size / parallel_size; r++) {
                for (int c = 0; c < col_chunk_size; c++) {
                    const int sub_block_id = c / group_size;
                    float local_S[parallel_size];
                    float local_M[parallel_size];
                    for (int rr = 0; rr < parallel_size; rr++) {
                        local_S[rr] = (float)(npu_blocks[i].scales[sub_block_id][r * parallel_size + rr]);
                        local_M[rr] = (float)(npu_blocks[i].mins[sub_block_id][r * parallel_size + rr]);
                    }
                    for (int k = 0; k < parallel_size / 2; k++) {
                        uint8_t q = npu_blocks[i].qs[r][c][k];
                        float row_0 = local_S[k * 2] * ((q & 0xF) - local_M[k * 2]);
                        float row_1 = local_S[k * 2 + 1] * ((q >> 4) - local_M[k * 2 + 1]);
                        temp_array[r * parallel_size + k * 2][c] = row_0;
                        temp_array[r * parallel_size + k * 2 + 1][c] = row_1;
                    }
                }
            }
            // copy to the dst
            for (int r = 0; r < row_chunk_size; r++) {
                for (int c = 0; c < col_chunk_size; c++) {
                    dst_ptr[(row_offset + r) * columns + col_offset + c] = (T)temp_array[r][c];
                }
            }
        }
    }

    /// \brief Dequantize the weight
    /// \param weight the weight
    /// \param q4nx_weight the q4nx weight
    /// \param columns the columns
    template<typename T>
    static void q4nx_dequantize(buffer<T>& weight, buffer<u32>& q, buffer<bf16>& scale, buffer<i32>& zero_point, const int columns){
        // to be implemented
        size_t total_size = scale.size() * group_size;

        if (weight.size() == 0){
            weight.resize(total_size);
        }
        else if (weight.size() != total_size){
            std::cerr << "Weight size mismatch: " << weight.size() << " != " << total_size << std::endl;
            exit(1);
        }
        
        int rows = total_size / columns;
        int cols = columns;

        T* weight_ptr = (T*)weight.data();
        bf16* scale_ptr = scale.data();
        i32* zero_point_ptr = zero_point.data();
        u32* q_ptr = q.data();
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols / group_size; j++){
                bf16 s = *scale_ptr++;
                int zp = *zero_point_ptr++;

                for (int k = 0; k < group_size / 8; k++){
                    u32 q_val = *q_ptr++;
                    for (int l = 0; l < 8; l++){
                        i8 q_val_i8 = (i8)(q_val & 0xF);
                        q_val >>= 4;
                        bf16 w = bf16((float)s * (q_val_i8 - zp));
                        *weight_ptr = (T)w;
                        weight_ptr++;

                    }
                }
            }
        }
    }



    /// \brief Get the weight per chunk
    /// \return the weight per chunk
    size_t get_weight_per_chunk() const{
        return weight_per_chunk;
    }

    /// \brief Get the block size
    /// \return the block size
    size_t get_block_size() const{
        return sizeof(npu_quantize_block);
    }
};