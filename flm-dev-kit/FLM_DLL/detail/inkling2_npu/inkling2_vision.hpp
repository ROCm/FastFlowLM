// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_vision.hpp
/// \brief the Inkling-2 image tower (vision_weight.q4nx -> one 2304-wide row per image token)
///
///   An image is cut into 40 x 40 pixel cells (no resize; the right and bottom edges are padded), one language
///   model token per cell. Each cell holds 2 x 2 patches of 20 x 20 pixels. The tower embeds the patches (a
///   linear map of the 1200 pixel values, plus a position embedding interpolated from a learned 32 x 32 table),
///   puts a class token in front, and runs 23 pre-LN transformer layers over all 4 N + 1 rows (16 heads x 64,
///   bidirectional, axial 2D rotary position on q and k, layer scale, exact-GELU MLP). The class token is dropped,
///   each cell's four patches are concatenated (4096) and projected to the text model's width (GELU in between).
///   The rows replace the embeddings of the image placeholder tokens as they are, with no RMSNorm.
///
///   The host path here is the reference (fp32 activations, bf16 weights read in place). The matrices are in the
///   layout the vision matmul kernel reads: 64-row x 512-column tiles, zero padded to whole 512 x 512 blocks.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace inkling2_vision {

using u16 = uint16_t;

constexpr int PATCH = 20;                    // patch edge in pixels
constexpr int MERGE = 2;                     // patches per cell edge
constexpr int CELL = PATCH * MERGE;          // 40 pixels
constexpr int PATCH_K = PATCH * PATCH * 3;   // 1200 values per patch, [py][px][c]
constexpr int DIM = 1024;                    // tower width
constexpr int NHEAD = 16;
constexpr int HDIM = 64;
constexpr int FFN = 4096;
constexpr int NLAYER = 23;
constexpr int POS_SIDE = 32;                 // the learned position table is 32 x 32
constexpr int MERGED = MERGE * MERGE * DIM;  // 4096: the four patches of a cell
constexpr int OUT = 2304;                    // the text model's width
constexpr int IMAGE_TOKEN_ID = 200054;       // the placeholder the prompt repeats once per cell

/// \brief an image's cell grid; one image token per cell, raster order
struct grid {
    int rows = 0, cols = 0;
    int tokens() const { return rows * cols; }
    int patches() const { return tokens() * MERGE * MERGE; }
    int seq() const { return patches() + 1; }   // with the class token
};

/// \brief the cells that cover a height x width image
grid grid_for(int height, int width);

/// \brief RGB8 rows (height x width x 3) -> pixels[tokens][40][40][3] in [-1, 1]: padded with -1 (0..255 scale) to
///        whole cells, then (v / 255 - 0.5) / 0.5. pixels holds grid_for(height, width).tokens() * 4800 floats.
void patchify(const uint8_t* rgb, int height, int width, float* pixels);

/// \brief the device half of the tower: the matrix products and the attention, behind the host flow of
///        tower::encode (layer norms, residuals, position embeddings and the rotary tables stay on the host).
///        All matrices are bf16 in the matmul kernel's tile layout (see the file comment); activations cross
///        as fp32 rows; an output may alias the input of the same call.
struct accel {
    /// \brief W[N][K] as 64-row x 512-column tiles zero padded to whole blocks, and its bias (null: none)
    struct mat {
        const u16* tiles = nullptr;
        int N = 0, K = 0;
        const float* bias = nullptr;
    };
    struct layer_mats {
        mat qkv, out, fc1, fc2;
    };
    virtual ~accel() = default;
    /// \brief take every matrix (copied into device buffers; the pointers need not outlive the call)
    virtual void bind(const mat& patch, const std::vector<layer_mats>& layers, const mat& proj_in, const mat& proj_out) = 0;
    /// \brief y[T][DIM] = patches[T][PATCH_K] W^T
    virtual void embed(const float* patches, int T, float* y) = 0;
    /// \brief delta[L][DIM] = out_proj(attention(rotate(qkv_proj(h)))); h = the layer-normed rows [L][DIM];
    ///        cos / sin: rotary angles [L][HDIM / 2], pair i = dims (2i, 2i + 1)
    virtual void attention(int layer, const float* h, int L, const float* cos, const float* sin, float* delta) = 0;
    /// \brief delta[L][DIM] = fc2(gelu(fc1(h)))
    virtual void mlp(int layer, const float* h, int L, float* delta) = 0;
    /// \brief out[N][OUT] = proj_out(gelu(proj_in(rows))), rows [N][MERGED]
    virtual void merge(const float* rows, int N, float* out) = 0;
};

class tower {
public:
    tower();
    ~tower();
    tower(const tower&) = delete;
    tower& operator=(const tower&) = delete;

    /// \brief map vision_weight.q4nx and locate every tensor; throws on a missing or misshapen one
    void load(const std::string& vision_q4nx);

    /// \brief run the matrix products and the attention on a device from now on (nullptr: back to the host path);
    ///        the weights are bound now. Not available while "merger_fc1" is traced.
    void set_accel(std::unique_ptr<accel> a);

    /// \brief the image tokens of one image: pixels as patchify makes them -> out_bf16[tokens][OUT] (round to nearest
    ///        even); with out_f32, the same rows in fp32 before that rounding
    void encode(const float* pixels, const grid& g, u16* out_bf16, float* out_f32 = nullptr) const;

    /// \brief DIAG: set, it sees the rows after "embed" (before ln_pre), "ln_pre", "layer<i>" for every layer, and
    ///        "merger_fc1" (before the GELU); rows x cols fp32, the class token first
    std::function<void(const char* tag, const float* x, int rows, int cols)> trace;

private:
    struct impl;
    std::unique_ptr<impl> p_;
};

}  // namespace inkling2_vision
