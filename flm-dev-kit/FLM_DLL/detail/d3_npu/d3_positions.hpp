// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
/// \file d3_positions.hpp
/// \brief Qwen3.5 3D M-RoPE position planning for prompts holding image / video runs (header only, no device
///        or XRT dependency: tools/test_positions_cpu.cpp checks it against transformers' get_rope_index).
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

/// @brief Qwen3.5's get_rope_index for one token sequence (or one block of a longer one).
///
/// Text positions count up from `pos0`; a visual run (the expanded pad tokens of one image or of one temporal
/// group of a video, `grid_h / 2 * grid_w / 2` tokens) starting at position s gives its tokens
/// (t, h, w) = (s, s + row, s + col) in 2 x 2 merge order, and the text after it resumes at
/// s + max(grid_h, grid_w) / 2 (not grid_w / 2: portrait images differ).
/// Both pad ids count as run starts: `image_id`, and `video_id` when it is >= 0. Every run takes the next
/// grid of `grids` in order (video groups are entries of the same list, in prompt order); a run starts at a
/// visual token and takes exactly its grid's token count. A visual token with no grid left is taken as text
/// (the callers check run structure first). Positions are written as 3 ints per token (t, h, w).
/// @param grids    array of `n_grids` elements with `.grid_h` / `.grid_w` (d3_image)
/// @param used     if not null, receives the number of grids the sequence consumed
/// @return the position the next token would take
template <class Image>
inline int d3_plan_positions(const std::vector<int>& ids, int image_id, int video_id, const Image* grids, size_t n_grids,
                             int pos0, int* pos, size_t* used = nullptr) {
    int cur = pos0;
    size_t image = 0, i = 0;
    while (i < ids.size()) {
        const bool visual = ids[i] == image_id || (video_id >= 0 && ids[i] == video_id);
        if (visual && image < n_grids) {
            const int gh = grids[image].grid_h / 2, gw = grids[image].grid_w / 2;
            for (int r = 0; r < gh && i < ids.size(); r++) {
                for (int c = 0; c < gw && i < ids.size(); c++, i++) {
                    pos[i * 3 + 0] = cur;
                    pos[i * 3 + 1] = cur + r;
                    pos[i * 3 + 2] = cur + c;
                }
            }
            cur += std::max(gh, gw);
            image++;
        } else {
            pos[i * 3 + 0] = pos[i * 3 + 1] = pos[i * 3 + 2] = cur++;
            i++;
        }
    }
    if (used) *used = image;
    return cur;
}

/// @brief the visual runs of `ids` must match `grids` exactly: as many runs as grids, each run holding
///        grid_h * grid_w / 4 visual tokens and followed by a non-visual token or the end (runs are separated by text; a
///        run followed directly by another visual token is an error). Throws std::invalid_argument.
template <class Image>
inline void d3_check_visual_runs(const std::vector<int>& ids, int image_id, int video_id, const Image* grids, size_t n_grids,
                                 const char* who) {
    auto visual = [&](int id) { return id == image_id || (video_id >= 0 && id == video_id); };
    size_t image = 0, i = 0;
    const std::string w = std::string("d3_npu: ") + who;
    const std::string e = "d3_npu: ";
    while (i < ids.size()) {
        if (!visual(ids[i])) { i++; continue; }
        if (image == n_grids) throw std::invalid_argument(w + " holds more image token runs than images");
        const size_t n = (size_t)grids[image].grid_h * grids[image].grid_w / 4;
        for (size_t k = 0; k < n; k++, i++) {
            if (i >= ids.size() || !visual(ids[i])) throw std::invalid_argument(e + "an image token run is shorter than its image");
        }
        if (i < ids.size() && visual(ids[i])) throw std::invalid_argument(e + "an image token run is longer than its image");
        image++;
    }
    if (image != n_grids) throw std::invalid_argument(w + " holds fewer image token runs than images");
}
