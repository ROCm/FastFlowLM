/// \file video_preprocess.hpp
/// \brief The d3 video processor on the CPU: what upstream's `Qwen3VLVideoProcessor` (transformers 5.17, configured
///        by d3_runtime.configure_video_processor) and `Qwen3VLProcessor.replace_video_token` do to a sampled clip.
///        Plain functions, no NPU. test_d3_video compares them with the real processor (pixel goldens).
///
/// ENGINE CONTRACT (what the engine payload takes per request, see `VideoPayload`):
///   * one `d3_image` {grid_h, grid_w} per TEMPORAL GROUP, in prompt order, after the request's images; the
///     engine's patch embedding already consumes `[3][2][16][16]` per patch, so a group of two frames is
///     exactly one "image" whose two temporal slices differ;
///   * `pixels` = the groups' patches concatenated (= upstream `pixel_values_videos` row order), bf16, per patch
///     [3 channels][2 frames][16][16], patches in 2 x 2 merge order;
///   * the prompt holds, per group, `<t seconds><|vision_start|>` + grid_h*grid_w/4 x <|video_pad|> +
///     `<|vision_end|>`; every video pad run is a position run exactly like an image's (3D M-RoPE `mrope_positions`).
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "video/video_sampling.hpp"

namespace flm::d3::video {

constexpr const char* VIDEO_PAD_TOKEN = "<|video_pad|>";
constexpr const char* VISION_START = "<|vision_start|>";
constexpr const char* VISION_END = "<|vision_end|>";
constexpr int VIDEO_PATCH = 16;
constexpr int VIDEO_MERGE = 2;
constexpr int VIDEO_TEMPORAL = 2;
constexpr int VIDEO_MAX_PIXELS = 200704;       //!< per frame (0.2 MP)
constexpr int VIDEO_MIN_PIXELS = 4096;
constexpr long long VIDEO_MAX_TOKENS = 16384;  //!< all videos of one request
constexpr long long MAX_VIDEO_BYTES = 32000000;
constexpr long long MAX_VIDEO_SOURCE_PIXELS = 8294400;
constexpr int PATCH_VALUES = 3 * VIDEO_TEMPORAL * VIDEO_PATCH * VIDEO_PATCH;   //!< 1,536 per patch

/// The checkpoint's video processor settings as the d3 runtime configures them.
struct VideoProcessorConfig {
    int patch = VIDEO_PATCH;
    int merge = VIDEO_MERGE;
    int temporal = VIDEO_TEMPORAL;
    long long shortest_edge = VIDEO_MIN_PIXELS;                     //!< size["shortest_edge"]
    long long longest_edge = static_cast<long long>(VIDEO_MAX_PIXELS) * VIDEO_MAX_FRAMES;   //!< size["longest_edge"]
    int max_video_tokens = VIDEO_MAX_PIXELS / (VIDEO_PATCH * VIDEO_MERGE * VIDEO_PATCH * VIDEO_MERGE);   //!< 196
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std[3] = {0.5f, 0.5f, 0.5f};
};

/// A clip as decoded and sampled: `frames` is `n x height x width x 3` uint8 RGB with an EVEN frame count (as
/// upstream's `Video`: the last frame is repeated when needed), `indices` are the frames' source numbers.
struct VideoClip {
    int width = 0;
    int height = 0;
    int n = 0;
    std::vector<uint8_t> frames;
    double fps = 0.0;            //!< source frame rate (VIDEO_FPS for frame lists)
    std::vector<int> indices;    //!< source frame number of each frame
    int total = 0;               //!< the source's frame count (the estimate the decoder gave)

    const uint8_t* frame(int i) const { return frames.data() + static_cast<size_t>(i) * width * height * 3; }
};

/// What the processor makes of a clip's geometry.
struct VideoGrid {
    int resized_h = 0, resized_w = 0;   //!< pixels after the resize
    int grid_t = 0, grid_h = 0, grid_w = 0;   //!< video_grid_thw
    int tokens_per_group() const { return grid_h * grid_w / 4; }
    long long tokens() const { return static_cast<long long>(grid_t) * tokens_per_group(); }
};

/// The video processor's `smart_resize` (Python's round() is half to even; `ValueError` texts kept).
/// Throws VideoError.
std::pair<int, int> smart_resize_video(int num_frames, int height, int width, int temporal_factor, int factor,
                                       long long min_pixels, long long max_pixels);

/// The grid the processor gives `n_frames` frames of `height` x `width` (the `cap_pixels_per_frame` budget:
/// pixels per frame = max(min(max_video_tokens * 32^2, longest_edge // n), int(shortest_edge * 1.05))).
/// Throws VideoError (aspect ratio over 200 ...).
VideoGrid plan_video_grid(int n_frames, int height, int width, const VideoProcessorConfig& cfg = {});

/// `Processor(videos=[frames])` up to the patches, fp32: the resized frames (torchvision's antialiased bicubic on
/// uint8, bit-exact on AVX2/AVX512 CPUs), (x - 127.5) / 127.5 (the fused rescale and normalize), per-patch
/// [3][2][16][16] in 2 x 2 merge order, groups in time order. Size = grid_t * grid_h * grid_w * 1,536.
std::vector<float> preprocess_video(const VideoClip& clip, const VideoGrid& grid, const VideoProcessorConfig& cfg = {});

/// The same as bf16 bits (round to nearest even, as torch's .to(bfloat16)), appended to `out`.
void preprocess_video_bf16(const VideoClip& clip, const VideoGrid& grid, std::vector<uint16_t>& out,
                           const VideoProcessorConfig& cfg = {});

/// The resized frames alone (`n x resized_h x resized_w x 3` uint8): torchvision `resize(bicubic, antialias)`.
std::vector<uint8_t> resize_clip(const VideoClip& clip, const VideoGrid& grid);

/// float -> bfloat16 bits, round to nearest even (NaN stays NaN).
uint16_t float_to_bf16_bits(float v);

/// torchvision.transforms.v2.functional.resize(uint8 [h, w, 3], bicubic, antialias=True) as ATen's AVX path
/// computes it (int16 weights of variable precision, horizontal then vertical, 8-bit between the passes).
std::vector<uint8_t> resize_bicubic_u8(const uint8_t* src, int in_w, int in_h, int out_w, int out_h);

// ---- text --------------------------------------------------------------------------------------------------

/// `_calculate_timestamps`: one time per temporal group, from the frames' source numbers.
std::vector<double> group_timestamps(const std::vector<int>& indices, double fps, int temporal = VIDEO_TEMPORAL);

/// `replace_video_token`: per group `<{t:.1f} seconds><|vision_start|>` + pads + `<|vision_end|>`. With
/// `expand` the pads are `tokens_per_group` repeats, else one pad per group (tokenize that, then
/// `expand_pad_runs`).
std::string video_placeholder(const std::vector<double>& times, int tokens_per_group, bool expand = true);

/// Replace the video placeholders (`<|video_pad|>`, one per video, in order) of a chat text by the videos' texts
/// (`get_text_with_replacements` for videos).
std::string expand_video_text(const std::string& text, const std::vector<std::string>& video_texts);

/// Token ids with one `pad_id` per group: each pad becomes a run of `runs[k]` pads (k-th pad in order).
std::vector<int> expand_pad_runs(const std::vector<int>& ids, int pad_id, const std::vector<int>& runs);

/// The planned input tokens of all videos must stay within VIDEO_MAX_TOKENS; throws VideoError with upstream's text.
void check_video_token_budget(long long planned);

// ---- the engine's view -------------------------------------------------------------------------------------

/// One request's videos, ready for the engine payload (d3_image_payload: `images` = `groups`, `pixels`).
struct VideoPayload {
    std::vector<std::pair<int, int>> groups;   //!< (grid_h, grid_w) per temporal group, all videos in order
    std::vector<uint16_t> pixels;              //!< bf16 bits, see the contract above
    int video_token_id = -1;                   //!< `<|video_pad|>`'s id (the tokenizer's)
};

/// One track window as the engine's per-window call reads it (a box at a moment / a track): the window's grid after the
/// processor's resize (its own frame budget: a short last window gets bigger frames), its groups' times and the pixels
/// of all its groups (bf16, `grid.grid_t * grid_h * grid_w` patches of PATCH_VALUES). `pixels` is released
/// after the window's call (`release_pixels`), the rest stays for the answers.
struct VideoWindow {
    VideoGrid grid;
    std::vector<int> indices;       //!< source frame numbers (even count, as decoded)
    double fps = 0.0;
    int total = 0;
    std::vector<double> times;      //!< `group_times`: one time per temporal group (the moments)
    std::vector<uint16_t> pixels;   //!< bf16 bits; empty once released
    void release_pixels() { std::vector<uint16_t>().swap(pixels); }
    /// bf16 values of one group (empty when released)
    std::vector<uint16_t> group_pixels(int g) const {
        const size_t n = static_cast<size_t>(grid.grid_h) * grid.grid_w * PATCH_VALUES;
        if (pixels.size() < (static_cast<size_t>(g) + 1) * n) return {};
        return std::vector<uint16_t>(pixels.begin() + static_cast<long>(n * g), pixels.begin() + static_cast<long>(n * (g + 1)));
    }
};

/// Append one clip's groups (and pixels) to the payload.
void append_video(VideoPayload& payload, const VideoClip& clip, const VideoGrid& grid,
                  const VideoProcessorConfig& cfg = {});

/// Qwen3.5's `get_rope_index` for one prompt, 3D positions as (t, h, w) per token (`out[3 * i + axis]`):
/// text counts up; a media run of grid_h/2 x grid_w/2 tokens (an image, or ONE temporal group of a video: every
/// token equal to `image_id` or `video_id` starts the next group in `groups`) at position s gets
/// (s, s + row, s + col) and the text after it resumes at s + max(grid_h, grid_w) / 2. Returns the next
/// position (what follows the last token). Throws VideoError when the runs do not match `groups`.
int mrope_positions(const std::vector<int>& ids, int image_id, int video_id,
                    const std::vector<std::pair<int, int>>& groups, std::vector<int>& out);

}  // namespace flm::d3::video
