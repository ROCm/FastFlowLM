/// \file video_input.hpp
/// \brief One `videos[i]` request value -> a sampled clip: `load_video(value, strict=True)` of d3_runtime plus FLM's
///        frame-list extension. Strict means what the server accepts: inline data only.
///
///   * "data:video/<mp4|webm|quicktime|x-matroska>[;params];base64,<data>" (32,000,000 bytes, upstream's types) and,
///     as an extension, "data:video/gif" / "data:image/gif": decoded by the FrameBackend (FFmpeg, loaded at run time);
///   * {"frames": ["data:image/png;base64,...", ...]} (extension): RGB frames as PNG / JPEG / WebP data URLs,
///     read as a clip sampled at 2 per second and thinned to 32 frames (`_frames_video`). Needs no video decoder.
#pragma once

#include <functional>
#include <memory>
#include <string>

#include "decision/d3_contract.hpp"
#include "video/video_decode.hpp"

namespace flm::d3::video {

struct VideoLoadOptions {
    std::shared_ptr<FrameBackend> backend;   //!< null: encoded videos are refused (frame lists still work)
    DecodeLimits limits;
    /// Decodes one inspected image file into RGB (the application's image decoder). Required for frame lists;
    /// returns false (with a message) on a corrupt file.
    std::function<bool(const ImageFile&, RgbFrame&, std::string*)> decode_image;
    bool allow_frame_lists = true;
    bool allow_gif = true;
};

constexpr int MAX_FRAME_LIST = 2048;

/// Throws VideoError with the text after "videos[i]: " (upstream's where defined), VideoBackendUnavailable when an
/// encoded video needs a decoder that is not there.
VideoClip load_video(const PyValue& value, const VideoLoadOptions& options);

/// `load_video_windows(value, rate)` of the track / box-at-a-moment path: the clip decoded once, in one pass, into its
/// windows (`window_plan`), each already resized and patchified (`VideoWindow::pixels`). Only a data URL string is
/// taken (frame lists are not: upstream's text). Throws VideoError, VideoBackendUnavailable.
std::vector<VideoWindow> load_video_windows(const PyValue& value, int rate, const VideoLoadOptions& options,
                                            const VideoProcessorConfig& cfg = {});

/// `PrepareOptions::load_video` for these options (the d3_contract hook): `load_video` as a shared clip.
std::function<std::shared_ptr<const VideoClip>(size_t, const PyValue&)> make_video_loader(VideoLoadOptions options);

}  // namespace flm::d3::video
