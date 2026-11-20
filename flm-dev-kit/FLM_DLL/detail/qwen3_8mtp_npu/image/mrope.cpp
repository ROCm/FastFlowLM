// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/mrope.cpp
#include "image/mrope.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace qwen38 {

std::vector<int32_t> build_mrope(const int* tokens, int L,
                                 int32_t image_token_id,
                                 const image_grid* grids, int num_images,
                                 int spatial_merge_size,
                                 mrope_cursor& cursor,
                                 std::vector<image_span>* spans,
                                 int32_t video_token_id) {
    const int m = spatial_merge_size;
    std::vector<int32_t> pos((size_t)3 * L);
    auto set = [&](int t, int32_t a, int32_t b, int32_t c) {
        pos[(size_t)0 * L + t] = a;
        pos[(size_t)1 * L + t] = b;
        pos[(size_t)2 * L + t] = c;
    };

    int t = 0;
    while (t < L) {
        // A video token is not text. Falling through to the branch below would
        // give it a text position and never consume a grid, so the whole rest
        // of the window would be off by the video's extent -- silently.
        if (video_token_id >= 0 && tokens[t] == video_token_id)
            throw std::runtime_error(
                "window row " + std::to_string(t) + " is a video token (id " +
                std::to_string(video_token_id) + "): this engine has no video "
                "path. The m-rope law below handles frames, but the vision "
                "tower's attention mask is a single full block, which is only "
                "correct within one frame");

        if (tokens[t] != image_token_id) {
            set(t, cursor.pos, cursor.pos, cursor.pos);
            cursor.pos++;
            t++;
            continue;
        }

        if (cursor.image >= num_images)
            throw std::runtime_error(
                "window row " + std::to_string(t) + " is an image token but the "
                "payload only carries " + std::to_string(num_images) +
                " image(s) -- the prompt and the payload disagree");

        const image_grid& g  = grids[cursor.image];
        const int mt    = g.t;             // frames, after temporal patching
        const int mh    = g.h / m;
        const int mw    = g.w / m;
        const int frame = mh * mw;         // merged blocks in one frame
        const int total = mt * frame;
        if (mw <= 0 || mh <= 0 || total <= 0)
            throw std::runtime_error("image " + std::to_string(cursor.image) +
                                     " has an empty merged grid");
        // The position law below is frame-general, but the tower's attention
        // mask is not -- see mrope.hpp. Refuse rather than encode positions the
        // rest of the pipeline cannot honour.
        if (mt != 1)
            throw std::runtime_error(
                "image " + std::to_string(cursor.image) + " has t = " +
                std::to_string(mt) + " frames: this engine has no video path, "
                "and the vision tower's full attention mask is only correct "
                "within one frame");

        // First row of this image anywhere: pin its base position.
        if (cursor.consumed == 0) cursor.image_start_pos = cursor.pos;
        const int32_t st = cursor.image_start_pos;

        const int begin = t;
        const int first = cursor.consumed;
        while (t < L && tokens[t] == image_token_id && cursor.consumed < total) {
            const int s = cursor.consumed;
            // s indexes the merged grid in raster order, frames outermost:
            // frame s/frame, then block (r/mw, r%mw) within it. mt == 1 here,
            // so the t channel is flat -- keep the general form anyway, so the
            // law in the header and the code that implements it are the same
            // statement.
            const int f = s / frame;
            const int r = s % frame;
            set(t, st + (int32_t)f, st + (int32_t)(r / mw), st + (int32_t)(r % mw));
            cursor.consumed++;
            t++;
        }

        if (spans)
            spans->push_back(image_span{begin, t - begin, cursor.image, first});

        if (cursor.consumed == total) {
            // The image is finished: the next text token resumes past the
            // image's extent, not past its token count. HF's rule is
            // `max over the three channels + 1`, i.e. max(mt, mh, mw).
            cursor.pos = st + (int32_t)std::max({mt, mh, mw});
            cursor.image++;
            cursor.consumed = 0;
        } else if (t < L) {
            // Ran out of image tokens before the grid was exhausted, mid-window.
            throw std::runtime_error(
                "image " + std::to_string(cursor.image) + " needs " +
                std::to_string(total) + " prompt rows but the window supplied " +
                std::to_string(cursor.consumed) + " before a non-image token");
        }
        // else: the window ended mid-image; the cursor carries the rest.
    }

    return pos;
}

}  // namespace qwen38
