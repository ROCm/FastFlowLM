// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __WEIGHT_DESC_HPP__
#define __WEIGHT_DESC_HPP__

#include <array>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cassert>

#include "utils/debug_utils.hpp"

#define IS_Q8_MASK    0x1
#define HAS_ZP_MASK   0x2
#define HAS_BIAS_MASK 0x4

const int FLM_TENSOR_MAX_DIMS = 4;

constexpr int QXNX_ROW_BLOCK_SIZE = 16;
constexpr int QXNX_COL_BLOCK_SIZE = 256;

typedef enum: uint8_t {
    flm_q40 = 0,
    flm_q40b = HAS_BIAS_MASK,
    flm_q41 = HAS_ZP_MASK,
    flm_q41b = HAS_ZP_MASK | HAS_BIAS_MASK,
    flm_q80 = IS_Q8_MASK,
    flm_q80b = IS_Q8_MASK | HAS_BIAS_MASK,
    flm_q81 = IS_Q8_MASK | HAS_ZP_MASK,
    flm_q81b = IS_Q8_MASK | HAS_ZP_MASK | HAS_BIAS_MASK,
    flm_u8,
    flm_i8,
    flm_u16,
    flm_i16,
    flm_u32,
    flm_i32,
    flm_u64,
    flm_i64,
    flm_f32,
    flm_bf16,
    flm_unknown
} flm_dtype_t;

inline bool is_quantize(flm_dtype_t qtype) { return qtype < flm_u8; }

inline size_t get_quantization_byte_size(size_t elems, flm_dtype_t qtype) {
    static constexpr size_t slice_size = 512; // Byte
    static constexpr size_t chunk_elems = QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE;
    static constexpr size_t chunk_groups = chunk_elems / 32; // group size is 32
    assert(is_quantize(qtype)); // does not apply to other dtype
    assert(elems % chunk_elems == 0);

    size_t chunks = elems / chunk_elems;

    // Count the quant + scale payload in bytes, not in 512B slices. A 16x256 q4 chunk is
    // 2048B of nibbles plus 256B of scales = 2304B, i.e. 4.5 slices -- the old truncating
    // division rounded that down to 4 and under-sized every weight by 11%. At a 32-row
    // block the payload happened to be slice-aligned (4608B = 9 slices), so this only
    // surfaced once the tile shrank to 16 rows.
    size_t chunk_bytes = (qtype & IS_Q8_MASK) ? chunk_elems : chunk_elems / 2; // / 2 for q4
    chunk_bytes += chunk_groups * sizeof(bf16);                                // scales

    // The zero-point and bias slices are still slice-granular on disk.
    if (qtype & HAS_ZP_MASK) {
        chunk_bytes += slice_size;
    }

    if (qtype & HAS_BIAS_MASK) {
        chunk_bytes += slice_size;
    }

    return chunks * chunk_bytes;
}


inline size_t get_dtype_byte(flm_dtype_t dtype) {
    int dtype_size = -1;
    switch (dtype) {
        case flm_u8:
        case flm_i8:
            dtype_size=  1;
            break;
        case flm_u16:
        case flm_i16:
        case flm_bf16:
            dtype_size = 2;
            break;
        case flm_u32:
        case flm_i32:
        case flm_f32:
            dtype_size = 4;
            break;
        case flm_u64:
        case flm_i64:
            dtype_size =  8;
            break;
        case flm_unknown:
        default:
            dtype_size = 1; // quantized / unknown types have no fixed per-element size
            break;
    }

    return dtype_size;
}


struct flm_shape_t {
    std::array<int64_t, FLM_TENSOR_MAX_DIMS> _data;

    flm_shape_t() { _data.fill(1); }

    flm_shape_t(std::initializer_list<int64_t> init) {
        _data.fill(1);
        size_t i = 0;
        for (int64_t v : init) {
            if (i >= FLM_TENSOR_MAX_DIMS) break;
            _data[i++] = v;
        }
    }

    int64_t&       operator[](size_t idx)       { return _data[idx]; }
    const int64_t& operator[](size_t idx) const { return _data[idx]; }

    auto begin()       { return _data.begin(); }
    auto end()         { return _data.end();   }
    auto begin() const { return _data.begin(); }
    auto end()   const { return _data.end();   }
    constexpr size_t size() const { return FLM_TENSOR_MAX_DIMS; }
    void fill(int64_t v) { _data.fill(v); }
    size_t elems() { return _data[0] * _data[1] * _data[2] * _data[3]; }

    bool operator==(const flm_shape_t& o) const { return _data == o._data; }
    bool operator!=(const flm_shape_t& o) const { return _data != o._data; }
};


inline std::pair<size_t, flm_shape_t> quantize_rectify(flm_dtype_t& dtype, flm_shape_t& shape){
    // last 2 dims must satisfy 32x256 block
    assert(shape[0] % QXNX_COL_BLOCK_SIZE == 0);
    assert(shape[1] % QXNX_ROW_BLOCK_SIZE == 0);

    assert(shape[2] > 0);
    assert(shape[3] > 0);

    size_t total_size = get_quantization_byte_size(shape.elems(), dtype);

    flm_shape_t new_shape = shape;

    new_shape[0] = shape[0] / QXNX_COL_BLOCK_SIZE;
    new_shape[1] = shape[1] / QXNX_ROW_BLOCK_SIZE;

    return std::pair<size_t, flm_shape_t> (total_size, new_shape);
}

struct weight_desc_t {
public:
    flm_shape_t shape;
    flm_dtype_t dtype;
    std::string name;
    size_t offset;

    bool added;
    bool loaded;

    weight_desc_t() {}

    weight_desc_t(flm_dtype_t dtype, flm_shape_t shape, std::string name) :
        shape(shape), dtype(dtype), name(name), offset(0), added(false), loaded(false) {}

    void indp() { added = true; }
    void load() { loaded = true; }

    bool ready() { return added & loaded; }

    template <typename... Args>
    std::string format_name(Args... args) {
        int size = std::snprintf(nullptr, 0, name.c_str(), args...);
        assert(size >= 0);
        std::string out(size, '\0');
        std::snprintf(&out[0], size + 1, name.c_str(), args...);
        return out;
    }

    size_t get_size() {
        if (is_quantize(dtype)) {
            auto new_size_shape = quantize_rectify(dtype, shape);
            return new_size_shape.first;
        }
        else {
            return shape.elems() * get_dtype_byte(dtype);
        }
    }
};

class weight_container{
    size_t total_size_counter;
public:
    weight_container() {
        total_size_counter = 0;
    }

    size_t add_weight(flm_dtype_t& dtype, flm_shape_t& shape){
        size_t offset = total_size_counter;
        if (is_quantize(dtype)) {
            auto new_size_shape = quantize_rectify(dtype, shape);
            total_size_counter += new_size_shape.first;
        }
        else {
            total_size_counter += shape.elems() * get_dtype_byte(dtype);
        }
        return offset;
    }

    void add_weight(weight_desc_t& weight){
        weight.offset = this->add_weight(weight.dtype, weight.shape);
        weight.added = true;
        LOG_VERBOSE(1, "Added weight '" << weight.name << "' at offset " << weight.offset
                       << ", total size now " << total_size_counter);
    }

    size_t get_size() { return total_size_counter; }
};

#endif