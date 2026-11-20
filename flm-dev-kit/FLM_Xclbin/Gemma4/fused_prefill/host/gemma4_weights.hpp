// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __GEMMA4_WEIGHTS_HPP__
#define __GEMMA4_WEIGHTS_HPP__
#pragma once

// Reading Gemma4-E2B's q4nx projections off disk.
//
// The other dequant_mm tests synthesize their weights, so they can quantize on
// the host and keep the pre-quantization matrix as a reference.  Here the
// weights arrive already quantized, so the reference has to be built the other
// way round -- q4_1_unpack reconstructs exactly what the dequant core rebuilds,
// and the CPU matmul runs on that.  Anything the quantizer lost is lost to both
// sides equally, which is what makes the comparison a test of the kernel rather
// than of the quantizer.

#include <cassert>
#include <string>
#include <vector>

#include "typedef.hpp"
#include "utils/utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "weights_desc.hpp"

// Q4_1: 4 bits of quant per weight plus a bf16 scale and a bf16 min per group
// of 32, i.e. 5/8 byte = 5 bpw, 5120 B for a 32x256 block.
inline size_t q4_1_size(size_t elems) { return elems * 5 / 8; }

/// \brief Interleave adjacent 32-row blocks the way the row-pair cores read them.
///
/// Lifted from the dequant_mm tests' util.hpp.  A core pair splits a 64-row slab
/// between its two rows, so the two 32-row halves have to alternate block by
/// block along the row of chunks rather than following each other.
inline void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks = 2)
{
    const int a_block_size = q4_1_size(32 * 256);
    const int blocks_per_row = col / 256;
    const int rows = src.size() / a_block_size / blocks_per_row;

    u8 *dst_ptr = dst;
    std::vector<u8 *> src_ptr(vertical_blocks);
    for (int i = 0; i < vertical_blocks; i++)
        src_ptr[i] = src.data() + i * a_block_size * blocks_per_row;

    for (int r = 0; r < rows; r += vertical_blocks)
    {
        for (int c = 0; c < blocks_per_row; c++)
            for (int i = 0; i < vertical_blocks; i++)
            {
                memcpy(dst_ptr, src_ptr[i], a_block_size);
                dst_ptr += a_block_size;
                src_ptr[i] += a_block_size;
            }
        for (int i = 0; i < vertical_blocks; i++)
        {
            src_ptr[i] += (vertical_blocks - 1) * a_block_size * blocks_per_row;
            if (src_ptr[i] + a_block_size * blocks_per_row > src.end())
                src_ptr[i] = src.data(); // useless padding
        }
    }
}

// Gemma4-E2B-IT, from config.json's text_config.  Layer types alternate
// 4 sliding : 1 full, so layer 0 is sliding and layer 4 is the first full one --
// and the two differ in head dim, 256 against global_head_dim 512, which is
// exactly the split this xclbin's two attention columns are built for.
struct gemma4_cfg {
    static constexpr int hidden_size        = 1536;
    static constexpr int heads              = 8;
    static constexpr int kv_heads           = 1;      // MQA
    static constexpr int head_dim           = 256;    // sliding layers
    static constexpr int global_head_dim    = 512;    // full layers
    static constexpr int sliding_window     = 512;
    static constexpr int intermediate_size  = 6144;
    static constexpr int num_hidden_layers  = 35;

    static bool is_full(int layer) { return layer % 5 == 4; }
    static int  dh(int layer)      { return is_full(layer) ? global_head_dim : head_dim; }
};

inline std::string g4_name(int layer, const char* proj) {
    return "model.layers." + std::to_string(layer) + ".self_attn." + proj + ".weight";
}

/// \brief Load one projection and hand back both the packed and unpacked forms.
///
/// \param packed   reordered for the cores, ready to sync to the device
/// \param w_deq    Dout x Din row major, what the dequant core reconstructs
inline void load_proj(Q4NX& q4nx, const std::string& name, int Dout, int Din,
                      u8* packed, bf16* w_deq) {
    buffer<u8> raw;
    q4nx.load_weights(raw, name);
    const size_t bytes = q4_1_size((size_t)Dout * Din);
    if (raw.size() != bytes) {
        std::cerr << "weight " << name << ": got " << raw.size() << " bytes, expected "
                  << bytes << " for " << Dout << "x" << Din << std::endl;
        exit(1);
    }
    q4_1_unpack(w_deq, raw.data(), Dout, Din);
    reorder_cpy(packed, raw, Din, 2);
}

#endif
