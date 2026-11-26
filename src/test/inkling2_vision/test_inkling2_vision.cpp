// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// grid_for and patchify of the Inkling-2 image tower: one 40 x 40 pixel cell per image token in raster order, the right
// and bottom edges padded with -1 (0..255 scale), then (v / 255 - 0.5) / 0.5 on every value, padding included.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "inkling2_vision.hpp"

using namespace inkling2_vision;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static float norm(float v) { return (v / 255.f - 0.5f) / 0.5f; }
static bool near(float a, float b) { return std::fabs(a - b) < 1e-6f; }

int main() {
    // the grid: whole cells, rounded up; one token per cell; four patches per cell and a class token
    CHECK(grid_for(40, 40).rows == 1 && grid_for(40, 40).cols == 1);
    CHECK(grid_for(41, 40).rows == 2 && grid_for(41, 40).cols == 1);
    CHECK(grid_for(480, 640).rows == 12 && grid_for(480, 640).cols == 16 && grid_for(480, 640).tokens() == 192);
    CHECK(grid_for(813, 1000).rows == 21 && grid_for(813, 1000).cols == 25 && grid_for(813, 1000).tokens() == 525);
    CHECK(grid_for(1, 1).tokens() == 1 && grid_for(1, 1).patches() == 4 && grid_for(1, 1).seq() == 5);
    bool threw = false;
    try { grid_for(0, 10); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { grid_for(10, -1); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    // a 50 x 70 image: 2 x 2 cells. Pixel (y, x) = (3y + x, 5y + 2x, y ^ x) mod 256; the rest of the cells is padding.
    const int H = 50, W = 70;
    std::vector<uint8_t> rgb((size_t)H * W * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            rgb[((size_t)y * W + x) * 3 + 0] = (uint8_t)(3 * y + x);
            rgb[((size_t)y * W + x) * 3 + 1] = (uint8_t)(5 * y + 2 * x);
            rgb[((size_t)y * W + x) * 3 + 2] = (uint8_t)(y ^ x);
        }
    const grid g = grid_for(H, W);
    CHECK(g.rows == 2 && g.cols == 2);
    std::vector<float> px((size_t)g.tokens() * CELL * CELL * 3, 12345.f);
    patchify(rgb.data(), H, W, px.data());
    int bad_real = 0, bad_pad = 0;
    for (int tok = 0; tok < g.tokens(); tok++)
        for (int yy = 0; yy < CELL; yy++)
            for (int xx = 0; xx < CELL; xx++) {
                const int y = (tok / g.cols) * CELL + yy, x = (tok % g.cols) * CELL + xx;
                const float* o = px.data() + (((size_t)tok * CELL + yy) * CELL + xx) * 3;
                for (int c = 0; c < 3; c++) {
                    const float want = (y < H && x < W) ? norm(rgb[((size_t)y * W + x) * 3 + c]) : norm(-1.f);
                    if (!near(o[c], want)) ((y < H && x < W) ? bad_real : bad_pad)++;
                }
            }
    CHECK(bad_real == 0);
    CHECK(bad_pad == 0);
    CHECK(near(norm(-1.f), -1.0078431f));   // the padding is slightly below the 0 pixel's -1

    std::printf(failures ? "inkling2_vision: %d failure(s)\n" : "inkling2_vision: all passed\n", failures);
    return failures ? 1 : 0;
}
