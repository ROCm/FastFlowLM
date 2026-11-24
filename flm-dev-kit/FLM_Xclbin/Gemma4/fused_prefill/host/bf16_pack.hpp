// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
/// @file bf16_pack.hpp
/// @brief Pack plain bf16 weights into the stream the fused mm overlay's bf16
///        passthrough mode expects.
///
/// The overlay was built for q4_1: one 5120 B packet carries a whole
/// 32 x 256 w tile and the row pair unpacks half of it each.  bf16 weights are
/// 16384 B per tile, and only 4096 B of a packet may carry tile payload, so a
/// tile arrives as four packets.  Each packet is
///
///     [ even core's 2048 B ][ odd core's 2048 B ][ 1024 B pad ]
///
/// which keeps the core-side indexing identical to the quant planes: a core
/// reads at (row / PARALLEL_ROWS) * 2048 either way.  See `_bf16_w_copy` in
/// kernel/q4_core.h -- the two must agree byte for byte.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace bf16_pack {

// Must match kernel/common.hpp.
static constexpr int ROW_BLOCK   = 32;    // n per w tile
static constexpr int COL_BLOCK   = 256;   // k per w tile
static constexpr int TILE_ELEMS  = ROW_BLOCK * COL_BLOCK;          // 8192
static constexpr int PACKETS     = 4;
static constexpr int PLANE_ELEMS = 1024;                            // 2048 B
static constexpr int PACKET_ELEMS = 5120 / 2;                       // 2560, incl. pad
static constexpr int SLAB_N      = 64;    // output columns one mm column owns

/// @brief position of w[n][k] inside a tile, in mmul B order
///        [n / 8][k / 8][k % 8][n % 8].
static inline int tile_index(int n, int k) {
    return (n / 8) * (COL_BLOCK / 8) * 64 + (k / 8) * 64 + (k % 8) * 8 + (n % 8);
}

/// @brief bytes the packed stream needs for a Dout x Din bf16 weight.
static inline size_t packed_bytes(size_t Dout, size_t Din) {
    // 4 packets x 2560 elems per 8192-elem tile = 2.5 bytes per weight, which is
    // exactly 4x q4_1's 0.625 -- the sequence generator relies on that identity.
    return (Dout * Din / TILE_ELEMS) * PACKETS * PACKET_ELEMS * sizeof(uint16_t);
}

/// @brief Pack `w` (row major [Dout][Din], n-major as the mm consumes it) into
///        `out`.  Tiles are emitted in the order the cores consume them: slab
///        major, then k slice, then the two n slices of the slab.
///
/// @param out  at least packed_bytes(Dout, Din) bytes
/// @param w    [Dout][Din] bf16
template <typename BF16>
void pack(BF16* out, const BF16* w, size_t Dout, size_t Din) {
    const size_t slabs   = Dout / SLAB_N;
    const size_t kslices = Din / COL_BLOCK;

    BF16* p = out;
    for (size_t s = 0; s < slabs; s++) {
        for (size_t ks = 0; ks < kslices; ks++) {
            for (int sub = 0; sub < SLAB_N / ROW_BLOCK; sub++) {
                // gather one 32 x 256 tile into mmul B order
                BF16 tile[TILE_ELEMS];
                const size_t n0 = s * SLAB_N + (size_t)sub * ROW_BLOCK;
                for (int n = 0; n < ROW_BLOCK; n++) {
                    const BF16* src = w + (n0 + n) * Din + ks * COL_BLOCK;
                    for (int k = 0; k < COL_BLOCK; k++) tile[tile_index(n, k)] = src[k];
                }
                // and cut it into the four packets the two cores pick up
                for (int q = 0; q < PACKETS; q++) {
                    memset(p, 0, PACKET_ELEMS * sizeof(BF16));
                    memcpy(p,               tile + (size_t)q * PLANE_ELEMS,
                           PLANE_ELEMS * sizeof(BF16));
                    memcpy(p + PLANE_ELEMS, tile + TILE_ELEMS / 2 + (size_t)q * PLANE_ELEMS,
                           PLANE_ELEMS * sizeof(BF16));
                    p += PACKET_ELEMS;
                }
            }
        }
    }
}

} // namespace bf16_pack
