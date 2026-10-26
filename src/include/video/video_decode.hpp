/// \file video_decode.hpp
/// \brief Decoding a video input into the sampled clip the d3 model reads. Two parts:
///        * `FrameBackend` / `FrameReader`: a sequential frame source (what OpenCV's VideoCapture gives upstream:
///          grab, retrieve, a frame size, a rate and a frame-count estimate). The FFmpeg implementation loads the
///          shared libraries at run time (`make_ffmpeg_backend`), no build-time dependency. Tests use fakes.
///        * `decode_clip` / `decode_windows` / `frames_clip`: upstream's `_decode_video`, `_decode_windows` and
///          `_frames_video` on top of any reader, backend independent.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "video/video_preprocess.hpp"
#include "video/video_sampling.hpp"

namespace flm::d3::video {

constexpr double FLM_MAX_VIDEO_SECONDS = 60.0;   //!< design decision A4 (upstream's server says 300)

/// What the container says about the video stream.
struct StreamInfo {
    int width = 0;
    int height = 0;
    double fps = 0.0;        //!< 0 or non-finite = unknown
    long long total = 0;     //!< the frame-count estimate (CAP_PROP_FRAME_COUNT); <= 0 = unknown
};

/// A sequential reader over one opened video (cv2.VideoCapture semantics).
class FrameReader {
public:
    virtual ~FrameReader() = default;
    virtual const StreamInfo& info() const = 0;
    /// Decode the next frame; false at the end (or on an unrecoverable error).
    virtual bool grab() = 0;
    /// The last grabbed frame as interleaved RGB (`width * height * 3`); false when it cannot be converted.
    virtual bool retrieve(std::vector<uint8_t>& rgb) = 0;
};

class FrameBackend {
public:
    virtual ~FrameBackend() = default;
    virtual std::string name() const = 0;
    /// False with the reason when the decoder cannot be used (libraries missing, unsupported version).
    virtual bool available(std::string* why) const = 0;
    /// Open encoded bytes (mp4, webm, mkv, mov, gif ...). Null with `error` when they cannot be opened.
    virtual std::unique_ptr<FrameReader> open(const std::string& bytes, std::string* error) = 0;
};

/// The refusal text for a missing decoder (documented in README: which FFmpeg libraries to install).
std::string backend_unavailable_message(const std::string& why);

/// FFmpeg's shared libraries (libavformat, libavcodec, libavutil, libswscale), loaded with dlopen / LoadLibrary
/// the first time it is needed. `dir` (or the FLM_FFMPEG_DIR environment variable) is searched first, then the
/// system's library path. Never null; `available()` tells whether the libraries and their versions fit.
std::shared_ptr<FrameBackend> make_ffmpeg_backend(const std::string& dir = std::string());

struct DecodeLimits {
    bool strict = true;                                       //!< the server's checks (source pixels, duration)
    long long max_source_pixels = MAX_VIDEO_SOURCE_PIXELS;
    double max_seconds = FLM_MAX_VIDEO_SECONDS;
};

/// `_decode_video`: open, check, sample (`sample_indices`), read sequentially, pad to an even frame count.
/// `end_frame` > 0 reads only the frames before it. Throws VideoError with upstream's texts.
VideoClip decode_clip(FrameBackend& backend, const std::string& bytes, const DecodeLimits& limits = {},
                      int end_frame = 0);

/// `_decode_windows`: one pass over the clip, split into the track windows of `window_plan(total, fps, rate)`.
std::vector<VideoClip> decode_windows(FrameBackend& backend, const std::string& bytes, int rate,
                                      const DecodeLimits& limits = {});
/// The same, handing each window to `sink` as soon as it is read, so a caller can resize or consume it and drop the
/// full-size frames (a 60 s 1080p clip at rate 2 is 240 frames, 1.5 GB if all windows are kept). Throws like the
/// vector form; `sink` is called for every window that kept a frame.
void decode_windows(FrameBackend& backend, const std::string& bytes, int rate, const DecodeLimits& limits,
                    const std::function<void(VideoClip&&)>& sink);

/// An already decoded RGB image (a frame of a frame list).
struct RgbFrame {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
};

/// `_frames_video`: `kept` are the frames `frame_list_indices(total)` selects out of a list of `total` frames
/// (the caller decodes only those); 2 fps timestamps, padded to an even count. Throws
/// VideoError("all frames of a video must have the same size").
VideoClip frames_clip(const std::vector<RgbFrame>& kept, const std::vector<int>& indices, int total);

}  // namespace flm::d3::video
