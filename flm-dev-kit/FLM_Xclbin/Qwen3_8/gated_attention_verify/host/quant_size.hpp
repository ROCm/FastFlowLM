// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __QUANT_SIZE_HPP__
#define __QUANT_SIZE_HPP__
#include <cstddef>

// On-device byte footprint of a q4nx tensor, per the block structs in
// kernels/model_spec.h. A 32x256 chunk is 4736 B for q4k and 8704 B for q80.
// q4k is 4.625 bits/weight, so it is not a whole number of 512 B slices --
// anything that walks it must step by whole blocks.

constexpr size_t q4k_bytes(size_t elems) {
    return elems / 2            // 4-bit quants
         + elems / 32 * 2       // uint8 scale + uint8 min per group
         + elems / 256 * 4;     // bf16 S + bf16 M per super-block
}

constexpr size_t q80_bytes(size_t elems) {
    return elems                // 8-bit quants
         + elems / 32 * 2;      // bf16 scale per group
}

#endif
