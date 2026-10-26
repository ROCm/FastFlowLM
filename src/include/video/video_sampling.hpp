/// \file video_sampling.hpp
/// \brief Which frames of a clip the d3 model reads, and the track windows of a box / track question.
///        Pure functions, no decoder and no NPU: byte-exact ports of upstream's `video_sample_indices`,
///        `_frames_video` thinning, `_video` padding (d3_runtime.py) and `window_plan`, `group_times`
///        (d3_compose.py), pin v1.1.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace flm::d3::video {

/// A problem with one video input; the message is upstream's ValueError text (the caller adds "videos[i]: ").
struct VideoError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// Thrown when no decoder can be used: the request is refused with a 422 "video inputs are not available: ...".
struct VideoBackendUnavailable : VideoError {
    using VideoError::VideoError;
};

constexpr double VIDEO_FPS = 2.0;            //!< sampling rate of a decoded clip (frames per second)
constexpr int VIDEO_MIN_FRAMES = 4;
constexpr int VIDEO_MAX_FRAMES = 32;
constexpr int WINDOW_FRAMES = 32;            //!< d3_compose.WINDOW_FRAMES: frames of one track window
constexpr double VIDEO_UNKNOWN_FPS = 24.0;   //!< a source rate outside VIDEO_SOURCE_FPS counts as this
constexpr double VIDEO_SOURCE_FPS_MIN = 0.1;
constexpr double VIDEO_SOURCE_FPS_MAX = 1000.0;

/// The source frame rate the runtime uses: `fps` when it is finite and in [0.1, 1000], else 24.
double effective_fps(double fps);

/// `video_sample_indices(total, fps)`: 2 frames per second, at least 4 and at most 32 (and at most `total`),
/// spread evenly over the whole clip: `np.linspace(0, total - 1, count).round()` (half to even).
std::vector<int> sample_indices(int total, double fps);

/// `_frames_video` thinning of a frame array: all frames up to 32, else 32 evenly spread
/// (`np.linspace(0, total - 1, 32).round()`). The clip's timestamps then use VIDEO_FPS.
std::vector<int> frame_list_indices(int total);

/// `_video`: an odd frame count gets the last frame (and its index) repeated once. `Item` is any frame type.
template <class Item>
void pad_to_even(std::vector<Item>& frames, std::vector<int>& indices) {
    if (frames.size() % 2) {
        frames.push_back(frames.back());
        indices.push_back(indices.back());
    }
}
/// The same for indices alone.
inline void pad_indices_to_even(std::vector<int>& indices) {
    if (indices.size() % 2) indices.push_back(indices.back());
}

/// `d3_compose.window_plan(total, fps, rate)`: the source frames of each track window; `rate` updates per
/// second means one frame every fps / (2 * rate) source frames; windows of at most 32 frames, each padded to
/// an even count and to at least 4 frames by repeating its last index.
std::vector<std::vector<int>> window_plan(int total, double fps, int rate);

/// `d3_compose.group_times`: the timestamp of each pair of frames, the mean of the two frames' times
/// (`index / fps`); an odd count repeats the last index (the processor's `_calculate_timestamps`).
std::vector<double> group_times(const std::vector<int>& indices, double fps);

/// One moment of a track: a temporal group of a window and its time in seconds.
struct Moment {
    int window = 0;
    int group = 0;      //!< index of the temporal group inside its window
    double time = 0.0;
};

/// Every group of every window in time order (`_video_box`'s `moments`).
std::vector<Moment> moments_of(const std::vector<std::vector<int>>& windows, double fps);

/// `_video_box`'s `at` handling: throws VideoError("at <at> s is outside the clip") for at < 0 or
/// at > last moment + 1.0; else the moment nearest to `at` (the first of equal distances).
Moment nearest_moment(const std::vector<Moment>& moments, double at);

/// A refusal of a box / track question about a video: upstream's QuestionRefused (reason + message) or
/// invalid_question (reason empty).
struct VideoQuestionIssue {
    std::string reason;     //!< "video_needed", "video_target_ambiguous", or "" for invalid_question
    std::string message;
};

/// The checks `_video_box` makes before decoding: exactly one video and `rate` in {1, 2} (box always 1).
/// Returns nothing when the question may run.
std::optional<VideoQuestionIssue> check_video_box(const std::string& kind, int n_videos, int rate);

/// FLM's row budget (design decision A5; not in upstream): every moment is one question row. Throws
/// VideoError with a clear text when `rows` is over `limit`.
constexpr int DEFAULT_MOMENT_ROW_LIMIT = 256;
void check_moment_rows(long long rows, int limit = DEFAULT_MOMENT_ROW_LIMIT);

}  // namespace flm::d3::video
