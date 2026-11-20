// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

// =============================================================================
// File        : q4_npu_eXpress.cpp
// Created     : 2025-03-30
// Description : This file contains the implementation of the q4_npu_eXpress
//               quantization recording.
// =============================================================================

#include "tensor_utils/q4_npu_eXpress.hpp"

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

namespace {
    static constexpr int parallel_size = 16; // row parallel size
    static constexpr int row_chunk_size = 32; // row chunk size
    static constexpr int col_chunk_size = 256; // column chunk size
    static constexpr int group_size = 32; // group size

    static constexpr int qb = (group_size * 4 + 16 + 16) / group_size; // bits per weight
    static constexpr int weight_per_chunk = row_chunk_size * col_chunk_size; // weight per chunk
    static constexpr int num_groups_per_chunk_row = col_chunk_size / group_size; // number of groups per chunk row
    static constexpr int num_groups_per_row_parallel = row_chunk_size / parallel_size; // number of groups per row parallel
    static constexpr size_t group_size_bytes = weight_per_chunk * qb; // group size in bytes
}
struct Q4NX::npu_quantize_block{
    bf16 scales[num_groups_per_chunk_row][row_chunk_size]; // per-group scales and mins, shall view as 8 rows and 32 columns
    bf16 mins[num_groups_per_chunk_row][row_chunk_size]; // per-group scales and mins
    uint8_t qs[row_chunk_size / parallel_size][col_chunk_size][parallel_size / 2];           // 4-bit quantized values
};
Q4NX::Q4NX(std::string model_path) : SafeTensors() {
    // check if model.q4nx exists, if not, try model.safetensors
    std::string q4nx_model_path = model_path + "/model.q4nx";
    // check if q4nx model exists
    std::ifstream q4nx_file(q4nx_model_path);
    if (q4nx_file.good()) {
        this->switch_model(q4nx_model_path);
    }
    else {
        // load the safetensors model. Say which path was probed: the fallback
        // used to be silent, so a model.q4nx that is present but not openable
        // (wrong model_path, a non-ASCII path under MSVC's narrow ifstream)
        // surfaced only as "Failed to open file: .../model.safetensors".
        std::cerr << "No model.q4nx at " << q4nx_model_path
                  << ", falling back to model.safetensors" << std::endl;
        this->switch_model(model_path + "/model.safetensors");
    }
    this->_grap_metadata();
}

void Q4NX::_grap_metadata(){
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

// construct the header
nlohmann::json Q4NX::_process_json_header(std::vector<tensor_metadata>& new_tensors_data) {
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

// convert the tensor to a q4nx tensor
tensor_metadata Q4NX::_convert_to_q4nx(tensor_metadata tensor_meta, bool is_quantized, size_t& offset) {
    tensor_metadata temp;
    if (is_quantized) {
        assert(tensor_meta.shape.size() == 2); // only 2D tensor can be quantized
        size_t rows = tensor_meta.shape[0];
        size_t cols = tensor_meta.shape[1] * 8;
        if (rows % row_chunk_size != 0) {
            LOG_VERBOSE(1, "Padding rows to " << rows << " for lm_head");
            rows = (rows / row_chunk_size + 1) * row_chunk_size;
        }
        size_t weights = rows * cols;
        temp.name = tensor_meta.name;
        temp.dtype = "Q4NX";
        temp.shape = {size_t(rows), size_t(cols)};
        temp.size = weights;
        temp.byte_size = weights * qb / 8;
        temp.offsets = {offset, offset + temp.byte_size};
        offset += temp.byte_size;
    } else {
        temp = tensor_meta;
        if (temp.dtype == "F32") {
            temp.dtype = "BF16";
            temp.byte_size = temp.byte_size / 2;
        }
        temp.offsets = {offset, offset + temp.byte_size};
        offset += temp.byte_size;
    }
    return temp;
}

void Q4NX::_to_bf16(bytes& weight) {
    buffer<u8> temp(weight.size());
    temp.copy_from(weight);
    weight.resize(temp.size() / 2);
    bf16* dst_ptr = (bf16*)weight.data();
    f32* src_ptr = (f32*)temp.data();
    for (size_t i = 0; i < weight.size() / 2; i++) {
        *dst_ptr++ = (bf16)(*src_ptr++);
    }
}

void Q4NX::_q4nx_reorder(bytes& q4nx_weight, buffer<u32>& weight, buffer<bf16>& scale, buffer<i32>& zero_point, int columns) {
    int scale_cols = columns / group_size;
    int scale_rows = scale.size() / scale_cols;
    int rows = scale_rows;
    int cols = columns;

    // row zero padding
    if (rows % row_chunk_size != 0) {
        int rows_padded = (rows / row_chunk_size + 1) * row_chunk_size;

        buffer<u32> weight_buffer(weight.size());
        buffer<bf16> scale_buffer(scale.size());
        buffer<i32> zero_point_buffer(zero_point.size());

        weight_buffer.copy_from(weight);
        scale_buffer.copy_from(scale);
        zero_point_buffer.copy_from(zero_point);

        weight.resize(rows_padded * cols / 8);
        scale.resize((size_t)rows_padded * scale_cols);
        zero_point.resize((size_t)rows_padded * scale_cols);

        weight.copy_from(weight_buffer);
        scale.copy_from(scale_buffer);
        zero_point.copy_from(zero_point_buffer);

        for (int i = rows; i < rows_padded; i++){
            // weight
            for (int j = 0; j < cols / 8; j++){
                weight[i * cols / 8 + j] = 0;
            }
            // scale
            for (int j = 0; j < scale_cols; j++){
                scale[i * scale_cols + j] = 0;
            }
            // zero_point
            for (int j = 0; j < scale_cols; j++){
                zero_point[i * scale_cols + j] = 0;
            }
        }
        rows = rows_padded;
    }

    // compute total size
    size_t total_size = rows * cols / 2;

    total_size += scale.size() * 2;
    total_size += zero_point.size() * 2;
    assert(total_size == q4nx_weight.size());

    int row_chunks = rows / row_chunk_size;
    int col_chunks = cols / col_chunk_size;

    npu_quantize_block* q4nx_block = (npu_quantize_block*)q4nx_weight.data();
    const int row_groups = row_chunk_size / parallel_size;
    const int col_groups = col_chunk_size / group_size;

    for (int i = 0; i < row_chunks; i++){
        for (int j = 0; j < col_chunks; j++){
            int block_idx = i * col_chunks + j;
            int row_offset = i * row_chunk_size;
            int scale_col_offset = j * col_groups;
            int weight_col_offset = j * col_chunk_size / 8;

            // write all scales and zero points in this chunk
            // ordered as scales[row_chunk_size / row_parallel_size][col_chunk_size / group_size][row_parallel_size]
            // reorder from [rows][cols]

            for (int r = 0; r < row_chunk_size; r++) {
                for (int sub_block = 0; sub_block < col_groups; sub_block++) {
                    int scale_idx = (row_offset + r) * scale_cols + scale_col_offset + sub_block;
                    q4nx_block[block_idx].scales[sub_block][r] = (bf16)(scale[scale_idx]);
                    int zp = zero_point[scale_idx];
                    q4nx_block[block_idx].mins[sub_block][r] = (bf16)(zp);
                }
            }
            // write weights
            u8 weights[row_chunk_size][col_chunk_size];
            const int weight_cols = cols / 8;
            for (int r = 0; r < row_chunk_size; r++){
                for (int c = 0; c < col_chunk_size / 8; c++){
                    int row_idx = row_offset + r;
                    int col_idx = weight_col_offset + c;
                    u32 weight_val = weight[row_idx * weight_cols + col_idx];
                    for (int l = 0; l < 8; l++){
                        weights[r][c * 8 + l] = (weight_val >> (l * 4)) & 0xF;
                    }
                }
            }
            // reorder weights
            for (int g = 0; g < row_groups; g++){
                for (int h = 0; h < col_chunk_size; h++){
                    for (int k = 0; k < parallel_size / 2; k++){
                        int weight_row = g * parallel_size + k * 2;
                        int weight_col = h;
                        u8 w_0 = weights[weight_row][weight_col];
                        u8 w_1 = weights[weight_row + 1][weight_col];
                        q4nx_block[block_idx].qs[g][h][k] = (w_1 << 4) | w_0;
                    }
                }
            }
        }
    }
}

void Q4NX::convert_model(std::string output_path) {
    assert(!this->is_a_quantized_model);
    std::ofstream output_file(output_path, std::ios::out | std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        exit(1);
    }

    this->_convert_llama(output_file);

    output_file.close();

    this->switch_model(output_path);
    this->_grap_metadata();
}

template<typename T>
void Q4NX::q4nx_dequantize(bytes& weight, bytes& q4nx_weight, const int columns){
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
template void Q4NX::q4nx_dequantize<bf16>(bytes& weight, bytes& q4nx_weight, const int columns);
template void Q4NX::q4nx_dequantize<f32>(bytes& weight, bytes& q4nx_weight, const int columns);

template<typename T>
void Q4NX::q4nx_dequantize(buffer<T>& weight, buffer<u32>& q, buffer<bf16>& scale, buffer<i32>& zero_point, const int columns){
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
template void Q4NX::q4nx_dequantize<bf16>(buffer<bf16>& weight, buffer<u32>& q, buffer<bf16>& scale, buffer<i32>& zero_point, const int columns);
template void Q4NX::q4nx_dequantize<f32>(buffer<f32>& weight, buffer<u32>& q, buffer<bf16>& scale, buffer<i32>& zero_point, const int columns);

// convert the llama model to a q4nx model
// weights that are quantized in Llama are:
// - lm_head.weight
// - model.layers.*.self_attn.q_proj.weight
// - model.layers.*.self_attn.k_proj.weight
// - model.layers.*.self_attn.v_proj.weight
// - model.layers.*.self_attn.o_proj.weight
// - model.layers.*.mlp.down_proj.weight
// - model.layers.*.mlp.gate_proj.weight
// - model.layers.*.mlp.up_proj.weight
void Q4NX::_convert_llama(std::ofstream& output_file) {
    // model architecture:
    // model.embed_tokens.weight with size: 262668288 and dtype: F32
    // model.norm.weight with size: 2048 and dtype: F32
    // lm_head.weight with size: 32833536 and dtype: I32
    // lm_head.weight_scale with size: 8208384 and dtype: F32
    // lm_head.weight_zero_point with size: 8208384 and dtype: I32
    // model.layers.0.input_layernorm.weight with size: 2048 and dtype: F32
    // model.layers.0.mlp.down_proj.weight with size: 2097152 and dtype: I32
    // model.layers.0.mlp.down_proj.weight_scale with size: 524288 and dtype: F32
    // model.layers.0.mlp.down_proj.weight_zero_point with size: 524288 and dtype: I32
    // model.layers.0.mlp.gate_proj.weight with size: 2097152 and dtype: I32
    // model.layers.0.mlp.gate_proj.weight_scale with size: 524288 and dtype: F32
    // model.layers.0.mlp.gate_proj.weight_zero_point with size: 524288 and dtype: I32
    // model.layers.0.mlp.up_proj.weight with size: 2097152 and dtype: I32
    // model.layers.0.mlp.up_proj.weight_scale with size: 524288 and dtype: F32
    // model.layers.0.mlp.up_proj.weight_zero_point with size: 524288 and dtype: I32
    // model.layers.0.post_attention_layernorm.weight with size: 2048 and dtype: F32
    // model.layers.0.self_attn.k_proj.weight with size: 131072 and dtype: I32
    // model.layers.0.self_attn.k_proj.weight_scale with size: 32768 and dtype: F32
    // model.layers.0.self_attn.k_proj.weight_zero_point with size: 32768 and dtype: I32
    // model.layers.0.self_attn.o_proj.weight with size: 524288 and dtype: I32
    // model.layers.0.self_attn.o_proj.weight_scale with size: 131072 and dtype: F32
    // model.layers.0.self_attn.o_proj.weight_zero_point with size: 131072 and dtype: I32
    // model.layers.0.self_attn.q_proj.weight with size: 524288 and dtype: I32
    // model.layers.0.self_attn.q_proj.weight_scale with size: 131072 and dtype: F32
    // model.layers.0.self_attn.q_proj.weight_zero_point with size: 131072 and dtype: I32
    // model.layers.0.self_attn.v_proj.weight with size: 131072 and dtype: I32
    // model.layers.0.self_attn.v_proj.weight_scale with size: 32768 and dtype: F32
    // model.layers.0.self_attn.v_proj.weight_zero_point with size: 32768 and dtype: I32
    size_t total_tensors = 3 + (3 + 4 + 2) * this->layers; // lm_head, embed_tokens, norm
                                                                              // up, gate, down, q, k, v, o, input_layernorm, post_attention_layernorm

    std::vector<tensor_metadata> new_tensors_data;
    tensor_metadata temp_meta;
    size_t offset = 0;
    // process lm_head
    temp_meta = this->get_tensor_metadata("lm_head.weight");
    temp_meta = this->_convert_to_q4nx(temp_meta, true, offset);
    new_tensors_data.push_back(temp_meta);
    // process embed_tokens, only offset is required to be updated
    temp_meta = this->get_tensor_metadata("model.embed_tokens.weight");
    temp_meta = this->_convert_to_q4nx(temp_meta, false, offset);
    new_tensors_data.push_back(temp_meta);
    // process norm, only offset is required to be updated
    temp_meta = this->get_tensor_metadata("model.norm.weight");
    temp_meta = this->_convert_to_q4nx(temp_meta, false, offset);
    new_tensors_data.push_back(temp_meta);
    // process layers
    for (int i = 0; i < this->layers; i++) {
        // process input_layernorm
        temp_meta = this->get_tensor_metadata("model.layers." + std::to_string(i) + ".input_layernorm.weight");
        temp_meta = this->_convert_to_q4nx(temp_meta, false, offset);
        new_tensors_data.push_back(temp_meta);
        // process self_attn.q_proj
        temp_meta = this->get_tensor_metadata("model.layers." + std::to_string(i) + ".self_attn.q_proj.weight");
        temp_meta = this->_convert_to_q4nx(temp_meta, true, offset);
        new_tensors_data.push_back(temp_meta);
        // process self_attn.k_proj
        temp_meta = this->get_tensor_metadata("model.layers." + std::to_string(i) + ".self_attn.k_proj.weight");
        temp_meta = this->_convert_to_q4nx(temp_meta, true, offset);
        new_tensors_data.push_back(temp_meta);
        // process self_attn.v_proj
        temp_meta = this->get_tensor_metadata("model.layers." + std::to_string(i) + ".self_attn.v_proj.weight");
        temp_meta = this->_convert_to_q4nx(temp_meta, true, offset);
        new_tensors_data.push_back(temp_meta);
        // process self_attn.o_proj
        temp_meta = this->get_tensor_metadata("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight");
        temp_meta = this->_convert_to_q4nx(temp_meta, true, offset);
        new_tensors_data.push_back(temp_meta);
        // process mlp.down_proj
        temp_meta = this->get_tensor_metadata("model.layers." + std::to_string(i) + ".mlp.down_proj.weight");
        temp_meta = this->_convert_to_q4nx(temp_meta, true, offset);
        new_tensors_data.push_back(temp_meta);
        // process mlp.gate_proj
        temp_meta = this->get_tensor_metadata("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight");
        temp_meta = this->_convert_to_q4nx(temp_meta, true, offset);
        new_tensors_data.push_back(temp_meta);
        // process mlp.up_proj
        temp_meta = this->get_tensor_metadata("model.layers." + std::to_string(i) + ".mlp.up_proj.weight");
        temp_meta = this->_convert_to_q4nx(temp_meta, true, offset);
        new_tensors_data.push_back(temp_meta);
        // process post_attention_layernorm
        temp_meta = this->get_tensor_metadata("model.layers." + std::to_string(i) + ".post_attention_layernorm.weight");
        temp_meta = this->_convert_to_q4nx(temp_meta, false, offset);
        new_tensors_data.push_back(temp_meta);
    }
    assert(new_tensors_data.size() == total_tensors);

    nlohmann::json json_header = this->_process_json_header(new_tensors_data);
    size_t header_size = json_header.dump(4).size();
    unsigned long long header_size_ll = header_size;
    output_file.write(reinterpret_cast<const char*>(&header_size_ll), sizeof(header_size_ll));
    output_file.write(json_header.dump(4).c_str(), header_size);
    size_t total_input_bytes = 0;
    size_t total_output_bytes = 0;
    // write the tensors
    for (size_t i = 0; i < new_tensors_data.size(); i++) { // must in order
        tensor_metadata original_meta_weight = this->get_tensor_metadata(new_tensors_data[i].name);
        tensor_metadata tensor_meta = new_tensors_data[i];
        if (tensor_meta.dtype == "Q4NX") {
            tensor_metadata original_meta_scale = this->get_tensor_metadata(tensor_meta.name + "_scale");
            tensor_metadata original_meta_zero_point = this->get_tensor_metadata(tensor_meta.name + "_zero_point");
            total_input_bytes += original_meta_weight.byte_size + original_meta_scale.byte_size + original_meta_zero_point.byte_size;
            buffer<u32> weight(original_meta_weight.size);
            this->load_weights(weight, original_meta_weight.name);
            buffer<bf16> scale(original_meta_scale.size);
            this->load_weights(scale, original_meta_scale.name);
            buffer<i32> zero_point(original_meta_zero_point.size);
            this->load_weights(zero_point, original_meta_zero_point.name);

            buffer<u8> q4nx_weight(tensor_meta.byte_size);
            this->_q4nx_reorder(q4nx_weight, weight, scale, zero_point, original_meta_scale.shape[1] * group_size);
            output_file.write(reinterpret_cast<const char*>(q4nx_weight.data()), q4nx_weight.size());
            output_file.flush();
            total_output_bytes += q4nx_weight.size();
        }
        else {
            bytes weight(original_meta_weight.byte_size);
            this->load_weights(weight, original_meta_weight.name);
            if (original_meta_weight.dtype == "F32") {
                this->_to_bf16(weight);
            }
            output_file.write(reinterpret_cast<const char*>(weight.data()), weight.size());
            output_file.flush();
            total_output_bytes += weight.size();
            total_input_bytes += tensor_meta.byte_size;
        }
    }

    total_input_bytes /= 1024 * 1024;
    total_output_bytes /= 1024 * 1024;
    LOG_VERBOSE(1, "Total input GB: " << total_input_bytes / 1024.0f);
    LOG_VERBOSE(1, "Total output GB: " << total_output_bytes / 1024.0f);
}

size_t Q4NX::get_weight_per_chunk() const {
    return weight_per_chunk;
}

size_t Q4NX::get_block_size() const {
    return sizeof(npu_quantize_block);
}
