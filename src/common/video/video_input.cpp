#include "video/video_input.hpp"

#include <algorithm>

namespace flm::d3::video {

namespace {

std::string thousands(long long n) {
    const std::string digits = std::to_string(n < 0 ? -n : n);
    std::string out;
    for (size_t i = 0; i < digits.size(); i++) {
        if (i && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return out;
}

const char* const NOT_INLINE =
    "cannot load this video with strict=True; pass the clip inline as a data:video/... URL";

VideoClip load_frames(const PyValue& value, const VideoLoadOptions& o) {
    for (const auto& m : value.members) {
        if (m.first != "frames") {
            throw VideoError("a video given as frames is {\"frames\": [data:image/... URLs]}, got the key " +
                             repr(PyValue::string(m.first)));
        }
    }
    const PyValue* list = value.get("frames");
    if (!list || !list->is_array()) {
        throw VideoError("a video given as frames is {\"frames\": [data:image/... URLs]}");
    }
    if (list->items.empty()) throw VideoError("a video given as frames needs at least one frame");
    if (list->items.size() > static_cast<size_t>(MAX_FRAME_LIST)) {
        throw VideoError("a video given as frames takes at most " + thousands(MAX_FRAME_LIST) + " frames");
    }
    if (!o.decode_image) throw VideoError("frames cannot be decoded in this build");
    // every frame is inspected (type, size, limits); only the frames the clip keeps are decoded
    std::vector<ImageFile> files;
    long long bytes = 0;
    for (size_t i = 0; i < list->items.size(); i++) {
        try {
            files.push_back(inspect_image(list->items[i]));
        } catch (const ContractError& e) {
            throw VideoError("frames[" + std::to_string(i) + "]: " + e.what());
        }
        bytes += static_cast<long long>(files.back().data.size());
        if (bytes > MAX_VIDEO_BYTES) {
            throw VideoError("the frames hold over " + thousands(MAX_VIDEO_BYTES) + " bytes");
        }
        const ImageFile& f = files.back();
        if (static_cast<long long>(f.width) * f.height > MAX_VIDEO_SOURCE_PIXELS) {
            throw VideoError("video frames must have at most " + thousands(MAX_VIDEO_SOURCE_PIXELS) + " pixels");
        }
        if (f.width != files[0].width || f.height != files[0].height) {
            throw VideoError("all frames of a video must have the same size");
        }
    }
    const int total = static_cast<int>(files.size());
    const std::vector<int> indices = frame_list_indices(total);
    std::vector<RgbFrame> kept;
    for (int i : indices) {
        RgbFrame fr;
        std::string err;
        if (!o.decode_image(files[static_cast<size_t>(i)], fr, &err)) {
            throw VideoError("frames[" + std::to_string(i) + "]: " +
                             (err.empty() ? "the image could not be decoded" : err));
        }
        if (fr.width != files[0].width || fr.height != files[0].height ||
            fr.rgb.size() != static_cast<size_t>(fr.width) * fr.height * 3) {
            throw VideoError("frames[" + std::to_string(i) + "]: the decoded frame does not match its header");
        }
        kept.push_back(std::move(fr));
    }
    return frames_clip(kept, indices, total);
}

}  // namespace

namespace {

/// `_video_data_url_payload(value, strict=True)` and the backend check: the encoded bytes of a data URL video.
/// Throws VideoError, VideoBackendUnavailable.
std::string video_bytes(const PyValue& value, const VideoLoadOptions& o) {
    if (value.text.compare(0, 5, "data:") != 0) throw VideoError(NOT_INLINE);
    std::string text, rest;
    try {
        try {
            rest = split_data_url(value.text, "video", text);
        } catch (const ContractError& first) {
            if (!o.allow_gif) throw;
            try {
                rest = split_data_url(value.text, "image", text);
            } catch (const ContractError&) {
                throw first;
            }
            if (rest.substr(0, rest.find(';')) != "gif") throw first;
        }
        const std::string subtype = rest.substr(0, rest.find(';'));
        const bool gif = o.allow_gif && subtype == "gif";
        if (!gif && subtype != "mp4" && subtype != "webm" && subtype != "quicktime" && subtype != "x-matroska") {
            throw ContractError("video data URLs take the types mp4, webm, quicktime, x-matroska, got " +
                                repr(PyValue::string(subtype)));
        }
        std::string bytes = decode_base64(text, MAX_VIDEO_BYTES);
        if (!o.backend) throw VideoBackendUnavailable(backend_unavailable_message("no decoder configured"));
        return bytes;
    } catch (const ContractError& e) {
        throw VideoError(e.what());
    }
}

}  // namespace

VideoClip load_video(const PyValue& value, const VideoLoadOptions& o) {
    if (value.is_object() && o.allow_frame_lists) return load_frames(value, o);
    if (!value.is_string()) throw VideoError(NOT_INLINE);
    if (value.text.empty()) {
        throw VideoError("a video is a path, an http(s) URL, a data:video/...;base64 URL or a frame array");
    }
    const std::string bytes = video_bytes(value, o);
    return decode_clip(*o.backend, bytes, o.limits);
}

std::vector<VideoWindow> load_video_windows(const PyValue& value, int rate, const VideoLoadOptions& o,
                                            const VideoProcessorConfig& cfg) {
    if (!value.is_string() || value.text.empty()) {
        throw VideoError("a video Box needs a path, an http(s) URL or a data:video/...;base64 URL");
    }
    const std::string bytes = video_bytes(value, o);
    std::vector<VideoWindow> out;
    // each window is resized and patchified as soon as it is read: only its bf16 pixels (38 MB at 32 frames of 0.2 MP) stay
    decode_windows(*o.backend, bytes, rate, o.limits, [&](VideoClip&& clip) {
        VideoWindow w;
        w.grid = plan_video_grid(clip.n, clip.height, clip.width, cfg);
        preprocess_video_bf16(clip, w.grid, w.pixels, cfg);
        w.indices = clip.indices;
        w.fps = clip.fps;
        w.total = clip.total;
        w.times = group_times(clip.indices, clip.fps);
        out.push_back(std::move(w));
    });
    return out;
}

std::function<std::shared_ptr<const VideoClip>(size_t, const PyValue&)> make_video_loader(VideoLoadOptions options) {
    return [options = std::move(options)](size_t, const PyValue& value) {
        return std::make_shared<const VideoClip>(load_video(value, options));
    };
}

}  // namespace flm::d3::video
