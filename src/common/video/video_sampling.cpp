#include "video/video_sampling.hpp"

#include <algorithm>
#include <cmath>

#include "decision/py_json.hpp"

namespace flm::d3::video {

double effective_fps(double fps) {
    if (!(std::isfinite(fps) && VIDEO_SOURCE_FPS_MIN <= fps && fps <= VIDEO_SOURCE_FPS_MAX)) return VIDEO_UNKNOWN_FPS;
    return fps;
}

namespace {

/// `np.linspace(0, total - 1, count).round().astype(int)`: numpy computes arange(count) * step with
/// step = (stop - start) / (count - 1), then forces the last element to `stop`; round is half to even.
std::vector<int> linspace_round(int total, int count) {
    std::vector<int> out(static_cast<size_t>(count));
    if (count == 1) {
        out[0] = 0;
        return out;
    }
    const double delta = static_cast<double>(total - 1);
    const double step = delta / static_cast<double>(count - 1);
    for (int i = 0; i < count; i++) {
        const double v = (step == 0.0 ? 0.0 : static_cast<double>(i) * step);
        out[static_cast<size_t>(i)] = static_cast<int>(std::nearbyint(v));
    }
    out.back() = total - 1;
    return out;
}

}  // namespace

std::vector<int> sample_indices(int total, double fps) {
    if (total < 1) throw VideoError("the video has no frames");
    // count = int(total / fps * VIDEO_FPS); Python's int() truncates toward zero
    int count = static_cast<int>(static_cast<double>(total) / fps * VIDEO_FPS);
    count = std::min(std::min(std::max(count, VIDEO_MIN_FRAMES), VIDEO_MAX_FRAMES), total);
    return linspace_round(total, count);
}

std::vector<int> frame_list_indices(int total) {
    if (total < 1) throw VideoError("a video given as frames needs at least one frame");
    if (total > VIDEO_MAX_FRAMES) return linspace_round(total, VIDEO_MAX_FRAMES);
    std::vector<int> out(static_cast<size_t>(total));
    for (int i = 0; i < total; i++) out[static_cast<size_t>(i)] = i;
    return out;
}

std::vector<std::vector<int>> window_plan(int total, double fps, int rate) {
    const double step = fps / (2.0 * rate);
    const int count = std::max(1, static_cast<int>(static_cast<double>(total) / step));
    std::vector<int> indices(static_cast<size_t>(count));
    for (int k = 0; k < count; k++) {
        // min(total - 1, int(round(k * step))): Python's round() is half to even
        const double r = std::nearbyint(static_cast<double>(k) * step);
        indices[static_cast<size_t>(k)] = static_cast<int>(std::min(static_cast<double>(total - 1), r));
    }
    std::vector<std::vector<int>> windows;
    for (size_t i = 0; i < indices.size(); i += WINDOW_FRAMES) {
        std::vector<int> w(indices.begin() + static_cast<long>(i),
                           indices.begin() + static_cast<long>(std::min(indices.size(), i + WINDOW_FRAMES)));
        if (w.size() % 2) w.push_back(w.back());
        while (w.size() < 4) w.push_back(w.back());
        windows.push_back(std::move(w));
    }
    return windows;
}

std::vector<double> group_times(const std::vector<int>& indices_in, double fps) {
    std::vector<int> indices = indices_in;
    if (indices.empty()) return {};
    pad_indices_to_even(indices);
    std::vector<double> seconds(indices.size());
    for (size_t i = 0; i < indices.size(); i++) seconds[i] = static_cast<double>(indices[i]) / fps;
    std::vector<double> out;
    for (size_t i = 0; i + 1 < seconds.size(); i += 2) out.push_back((seconds[i] + seconds[i + 1]) / 2);
    return out;
}

std::vector<Moment> moments_of(const std::vector<std::vector<int>>& windows, double fps) {
    std::vector<Moment> out;
    for (size_t w = 0; w < windows.size(); w++) {
        const std::vector<double> times = group_times(windows[w], fps);
        for (size_t g = 0; g < times.size(); g++) out.push_back({static_cast<int>(w), static_cast<int>(g), times[g]});
    }
    return out;
}

Moment nearest_moment(const std::vector<Moment>& moments, double at) {
    if (moments.empty()) throw VideoError("the video has no frames");
    if (at < 0 || at > moments.back().time + 1.0) {
        throw VideoError("at " + float_repr(at) + " s is outside the clip");
    }
    Moment best = moments.front();
    double best_d = std::fabs(best.time - at);
    for (const Moment& m : moments) {
        const double d = std::fabs(m.time - at);
        if (d < best_d) {
            best = m;
            best_d = d;
        }
    }
    return best;
}

std::optional<VideoQuestionIssue> check_video_box(const std::string& kind, int n_videos, int rate) {
    if (n_videos < 1) return VideoQuestionIssue{"video_needed", "a " + kind + " question needs a video"};
    if (n_videos > 1) {
        return VideoQuestionIssue{"video_target_ambiguous",
                                  "a video " + kind + " needs exactly one video, got " + std::to_string(n_videos)};
    }
    if (rate != 1 && rate != 2) return VideoQuestionIssue{"", "rate must be 1 or 2 updates per second"};
    return std::nullopt;
}

void check_moment_rows(long long rows, int limit) {
    if (rows > limit) {
        throw VideoError("the box and track questions need " + std::to_string(rows) + " video moments, over the " +
                         std::to_string(limit) +
                         " this build runs for one request; send fewer track questions, rate 1 or a shorter video");
    }
}

}  // namespace flm::d3::video
