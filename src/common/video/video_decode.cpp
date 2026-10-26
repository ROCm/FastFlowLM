#include "video/video_decode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace flm::d3::video {

namespace {

std::string thousands(long long n) {
    const std::string digits = std::to_string(n < 0 ? -n : n);
    std::string out;
    for (size_t i = 0; i < digits.size(); i++) {
        if (i && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return n < 0 ? "-" + out : out;
}

std::string seconds_text(double s) {
    // f"{MAX_VIDEO_SECONDS}" with an integral limit prints as an int in the message ("300"), as ours ("60")
    if (s == std::floor(s)) return std::to_string(static_cast<long long>(s));
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", s);
    return buf;
}

VideoClip make_clip(std::vector<std::vector<uint8_t>>& frames, std::vector<int> indices, double fps, int total,
                    int width, int height) {
    VideoClip c;
    pad_to_even(frames, indices);
    c.width = width;
    c.height = height;
    c.n = static_cast<int>(frames.size());
    c.frames.reserve(static_cast<size_t>(c.n) * width * height * 3);
    for (const auto& f : frames) c.frames.insert(c.frames.end(), f.begin(), f.end());
    c.fps = fps;
    c.indices = std::move(indices);
    c.total = total;
    return c;
}

std::unique_ptr<FrameReader> open_or_throw(FrameBackend& backend, const std::string& bytes) {
    std::string why;
    if (!backend.available(&why)) throw VideoBackendUnavailable(backend_unavailable_message(why));
    std::string error;
    auto reader = backend.open(bytes, &error);
    if (!reader) throw VideoError(error.empty() ? "the video could not be opened" : error);
    return reader;
}

}  // namespace

std::string backend_unavailable_message(const std::string& why) {
    return "video inputs are not available: video decoding backend not available" +
           (why.empty() ? std::string() : " (" + why + ")");
}

VideoClip decode_clip(FrameBackend& backend, const std::string& bytes, const DecodeLimits& limits, int end_frame) {
    auto reader = open_or_throw(backend, bytes);
    const StreamInfo info = reader->info();
    if (info.width <= 0 || info.height <= 0) throw VideoError("the video has no frame size");
    if (limits.strict && static_cast<long long>(info.width) * info.height > limits.max_source_pixels) {
        throw VideoError("video frames must have at most " + thousands(limits.max_source_pixels) + " pixels");
    }
    const double fps = effective_fps(info.fps);
    long long total = info.total;
    if (total <= 0) {   // the container does not say: count the frames, then read again
        const long long limit = limits.strict ? static_cast<long long>(limits.max_seconds * fps) : -1;
        total = 0;
        while (reader->grab()) {
            total++;
            if (limit >= 0 && total > limit) break;
        }
        reader = open_or_throw(backend, bytes);
    }
    if (end_frame > 0) total = std::min<long long>(total, end_frame);
    if (total <= 0) throw VideoError("the video has no frames");
    if (limits.strict && static_cast<double>(total) / fps > limits.max_seconds) {
        throw VideoError("each video must be at most " + seconds_text(limits.max_seconds) + " seconds long");
    }
    const std::vector<int> indices = sample_indices(static_cast<int>(total), fps);
    std::vector<char> wanted(static_cast<size_t>(indices.back()) + 1, 0);
    for (int i : indices) wanted[static_cast<size_t>(i)] = 1;
    std::vector<std::vector<uint8_t>> frames;
    std::vector<int> kept;
    for (int number = 0; number <= indices.back(); number++) {
        if (!reader->grab()) continue;   // upstream carries on: a failed grab skips the frame number
        if (!wanted[static_cast<size_t>(number)]) continue;
        std::vector<uint8_t> rgb;
        if (reader->retrieve(rgb) && !rgb.empty()) {
            frames.push_back(std::move(rgb));
            kept.push_back(number);
        }
    }
    if (frames.empty()) throw VideoError("no frame of the video could be decoded");
    return make_clip(frames, kept, fps, static_cast<int>(total), info.width, info.height);
}

void decode_windows(FrameBackend& backend, const std::string& bytes, int rate, const DecodeLimits& limits,
                    const std::function<void(VideoClip&&)>& sink) {
    auto reader = open_or_throw(backend, bytes);
    const StreamInfo info = reader->info();
    const double fps = effective_fps(info.fps);
    const long long total = info.total;
    if (total <= 0) throw VideoError("the video has no frames");
    if (static_cast<double>(total) / fps > limits.max_seconds) {
        throw VideoError("each video must be at most " + seconds_text(limits.max_seconds) + " seconds long");
    }
    bool any = false;
    int number = 0;
    for (const std::vector<int>& window : window_plan(static_cast<int>(total), fps, rate)) {
        std::vector<std::pair<int, std::vector<uint8_t>>> have;     // (frame number, pixels) read for this window
        const int last = window.back();
        while (number <= last) {
            if (!reader->grab()) break;
            if (std::binary_search(window.begin(), window.end(), number)) {
                std::vector<uint8_t> rgb;
                if (reader->retrieve(rgb) && !rgb.empty()) have.emplace_back(number, std::move(rgb));
            }
            number++;
        }
        std::vector<std::vector<uint8_t>> frames;
        std::vector<int> kept;
        for (int i : window) {   // `kept = [i for i in window if i in frames]`: repeated indices repeat the frame
            for (const auto& h : have) {
                if (h.first == i) {
                    frames.push_back(h.second);
                    kept.push_back(i);
                    break;
                }
            }
        }
        if (!kept.empty()) {
            any = true;
            sink(make_clip(frames, kept, fps, static_cast<int>(total), info.width, info.height));
        }
    }
    if (!any) throw VideoError("no frame of the video could be decoded");
}

std::vector<VideoClip> decode_windows(FrameBackend& backend, const std::string& bytes, int rate,
                                      const DecodeLimits& limits) {
    std::vector<VideoClip> out;
    decode_windows(backend, bytes, rate, limits, [&](VideoClip&& c) { out.push_back(std::move(c)); });
    return out;
}

VideoClip frames_clip(const std::vector<RgbFrame>& kept, const std::vector<int>& indices, int total) {
    if (kept.empty()) throw VideoError("a video given as frames needs at least one frame");
    for (const RgbFrame& f : kept) {
        if (f.width != kept[0].width || f.height != kept[0].height) {
            throw VideoError("all frames of a video must have the same size");
        }
    }
    std::vector<std::vector<uint8_t>> frames;
    for (const RgbFrame& f : kept) frames.push_back(f.rgb);
    return make_clip(frames, indices, VIDEO_FPS, total, kept[0].width, kept[0].height);
}

}  // namespace flm::d3::video
