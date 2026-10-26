#include "video/video_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "decision/py_json.hpp"

namespace flm::d3::video {

namespace {

/// Python's `f"{n:,}"`
std::string thousands(long long n) {
    const std::string digits = std::to_string(n < 0 ? -n : n);
    std::string out;
    for (size_t i = 0; i < digits.size(); i++) {
        if (i && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return n < 0 ? "-" + out : out;
}

long long round_even(double v) { return static_cast<long long>(std::nearbyint(v)); }

// ---- ATen's antialiased bicubic on uint8 (the AVX path of upsample_bicubic2d, weights as int16) ------------

double cubic_filter(double x) {   // HelperInterpCubic::aa_filter with a = -0.5 (Keys cubic)
    constexpr double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2.0) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
    return 0.0;
}

struct Taps {
    std::vector<int> first;       // xmin per output
    std::vector<int> size;        // taps per output
    std::vector<int16_t> w;       // out * stride weights
    int stride = 0;
    int precision = 0;
};

Taps make_taps(int in, int out) {
    // _compute_index_ranges_weights<double> + _compute_index_ranges_int16_weights
    const double scale = static_cast<double>(in) / static_cast<double>(out);
    const double support = scale >= 1.0 ? 4 * 0.5 * scale : 4 * 0.5;
    const int max_interp = static_cast<int>(std::ceil(support)) * 2 + 1;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    Taps t;
    t.first.resize(static_cast<size_t>(out));
    t.size.resize(static_cast<size_t>(out));
    t.stride = max_interp;
    std::vector<double> wt(static_cast<size_t>(out) * max_interp, 0.0);
    double wt_max = 0.0;
    for (int i = 0; i < out; i++) {
        const double center = scale * (i + 0.5);
        const long long xmin = std::max(static_cast<long long>(center - support + 0.5), 0LL);
        long long xsize = std::min(static_cast<long long>(center + support + 0.5), static_cast<long long>(in)) - xmin;
        xsize = std::clamp<long long>(xsize, 0, max_interp);
        double* p = wt.data() + static_cast<size_t>(i) * max_interp;
        double total = 0.0;
        for (long long j = 0; j < xsize; j++) {
            const double w = cubic_filter((static_cast<double>(j + xmin) - center + 0.5) * invscale);
            p[j] = w;
            total += w;
        }
        if (total != 0.0) {
            for (long long j = 0; j < xsize; j++) {
                p[j] /= total;
                wt_max = std::max(wt_max, p[j]);
            }
        }
        t.first[static_cast<size_t>(i)] = static_cast<int>(xmin);
        t.size[static_cast<size_t>(i)] = static_cast<int>(xsize);
    }
    unsigned precision = 0;
    for (precision = 0; precision < 22; ++precision) {
        const int next = static_cast<int>(0.5 + wt_max * (1 << (precision + 1)));
        if (next >= (1 << 15)) break;
    }
    t.precision = static_cast<int>(precision);
    t.w.assign(wt.size(), 0);
    for (size_t i = 0; i < wt.size(); i++) {
        const double v = wt[i] * (1 << precision);
        t.w[i] = static_cast<int16_t>(v < 0 ? static_cast<int>(-0.5 + v) : static_cast<int>(0.5 + v));
    }
    return t;
}

inline uint8_t clip8(int32_t v) { return static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v); }

}  // namespace

std::vector<uint8_t> resize_bicubic_u8(const uint8_t* src, int in_w, int in_h, int out_w, int out_h) {
    std::vector<uint8_t> mid;
    const uint8_t* rows = src;
    if (out_w != in_w) {
        const Taps t = make_taps(in_w, out_w);
        const int32_t round = 1 << (t.precision - 1);
        mid.resize(static_cast<size_t>(out_w) * in_h * 3);
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < in_h; y++) {
            const uint8_t* line = src + static_cast<size_t>(y) * in_w * 3;
            uint8_t* dst = mid.data() + static_cast<size_t>(y) * out_w * 3;
            for (int x = 0; x < out_w; x++) {
                const int xmin = t.first[static_cast<size_t>(x)], n = t.size[static_cast<size_t>(x)];
                const int16_t* k = t.w.data() + static_cast<size_t>(x) * t.stride;
                for (int c = 0; c < 3; c++) {
                    int32_t s = round;
                    for (int j = 0; j < n; j++) s += static_cast<int32_t>(line[(xmin + j) * 3 + c]) * k[j];
                    dst[x * 3 + c] = clip8(s >> t.precision);
                }
            }
        }
        rows = mid.data();
    }
    if (out_h == in_h) return std::vector<uint8_t>(rows, rows + static_cast<size_t>(out_w) * in_h * 3);
    const Taps t = make_taps(in_h, out_h);
    const int32_t round = 1 << (t.precision - 1);
    std::vector<uint8_t> out(static_cast<size_t>(out_w) * out_h * 3);
    const size_t row = static_cast<size_t>(out_w) * 3;
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < out_h; y++) {
        const int ymin = t.first[static_cast<size_t>(y)], n = t.size[static_cast<size_t>(y)];
        const int16_t* k = t.w.data() + static_cast<size_t>(y) * t.stride;
        uint8_t* dst = out.data() + static_cast<size_t>(y) * row;
        for (size_t x = 0; x < row; x++) {
            int32_t s = round;
            for (int j = 0; j < n; j++) s += static_cast<int32_t>(rows[(static_cast<size_t>(ymin) + j) * row + x]) * k[j];
            dst[x] = clip8(s >> t.precision);
        }
    }
    return out;
}

std::pair<int, int> smart_resize_video(int num_frames, int height, int width, int temporal_factor, int factor,
                                       long long min_pixels, long long max_pixels) {
    if (num_frames < temporal_factor) {
        throw VideoError("t:" + std::to_string(num_frames) + " must be larger than temporal_factor:" +
                         std::to_string(temporal_factor));
    }
    if (height < factor || width < factor) {
        const double scale = std::max(static_cast<double>(factor) / height, static_cast<double>(factor) / width);
        height = static_cast<int>(height * scale);
        width = static_cast<int>(width * scale);
    }
    const double ratio = static_cast<double>(std::max(height, width)) / static_cast<double>(std::min(height, width));
    if (ratio > 200) throw VideoError("absolute aspect ratio must be smaller than 200, got " + float_repr(ratio));
    long long h_bar = round_even(static_cast<double>(height) / factor) * factor;
    long long w_bar = round_even(static_cast<double>(width) / factor) * factor;
    const long long t_bar = round_even(static_cast<double>(num_frames) / temporal_factor) * temporal_factor;
    if (t_bar * h_bar * w_bar > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(static_cast<long long>(num_frames) * height * width) /
                                      static_cast<double>(max_pixels));
        h_bar = std::max<long long>(factor, static_cast<long long>(std::floor(height / beta / factor)) * factor);
        w_bar = std::max<long long>(factor, static_cast<long long>(std::floor(width / beta / factor)) * factor);
    } else if (t_bar * h_bar * w_bar < min_pixels) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) /
                                      (static_cast<double>(num_frames) * height * width));
        h_bar = static_cast<long long>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<long long>(std::ceil(width * beta / factor)) * factor;
    }
    return {static_cast<int>(h_bar), static_cast<int>(w_bar)};
}

VideoGrid plan_video_grid(int n_frames, int height, int width, const VideoProcessorConfig& cfg) {
    const int factor = cfg.patch * cfg.merge;
    // cap_pixels_per_frame: min(max_video_tokens * factor^2, longest_edge // n), floored at int(1.05 * shortest_edge)
    const long long frame_cap = static_cast<long long>(cfg.max_video_tokens) * factor * factor;
    const long long per_frame =
        std::max(std::min(frame_cap, cfg.longest_edge / std::max(n_frames, 1)),
                 static_cast<long long>(static_cast<double>(cfg.shortest_edge) * 1.05));
    const auto hw = smart_resize_video(n_frames, height, width, cfg.temporal, factor, cfg.shortest_edge,
                                       per_frame * n_frames);
    VideoGrid g;
    g.resized_h = hw.first;
    g.resized_w = hw.second;
    g.grid_h = hw.first / cfg.patch;
    g.grid_w = hw.second / cfg.patch;
    g.grid_t = (n_frames + cfg.temporal - 1) / cfg.temporal;
    return g;
}

std::vector<uint8_t> resize_clip(const VideoClip& clip, const VideoGrid& grid) {
    const size_t frame = static_cast<size_t>(grid.resized_w) * grid.resized_h * 3;
    std::vector<uint8_t> out(frame * clip.n);
    for (int i = 0; i < clip.n; i++) {
        const std::vector<uint8_t> r =
            resize_bicubic_u8(clip.frame(i), clip.width, clip.height, grid.resized_w, grid.resized_h);
        std::memcpy(out.data() + frame * i, r.data(), frame);
    }
    return out;
}

uint16_t float_to_bf16_bits(float v) {
    uint32_t b;
    std::memcpy(&b, &v, 4);
    if ((b & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<uint16_t>((b >> 16) | 0x40);   // NaN
    b += 0x7FFFu + ((b >> 16) & 1u);
    return static_cast<uint16_t>(b >> 16);
}

namespace {

/// Visit the clip's patch values in processor order: `sink(float)` per value.
template <class Sink>
void for_each_value(const VideoClip& clip, const VideoGrid& grid, const VideoProcessorConfig& cfg, Sink&& sink) {
    const std::vector<uint8_t> r = resize_clip(clip, grid);
    const size_t frame = static_cast<size_t>(grid.resized_w) * grid.resized_h * 3;
    const int p = cfg.patch, m = cfg.merge, T = cfg.temporal;
    // the processor's fused rescale + normalize: (x - mean * 255) / (std * 255), in fp32
    float mean[3], stdv[3];
    for (int c = 0; c < 3; c++) {
        mean[c] = static_cast<float>(cfg.mean[c] * 255.0);
        stdv[c] = static_cast<float>(cfg.std[c] * 255.0);
    }
    for (int g = 0; g < grid.grid_t; g++) {
        for (int hb = 0; hb < grid.grid_h / m; hb++) {
            for (int wb = 0; wb < grid.grid_w / m; wb++) {
                for (int mh = 0; mh < m; mh++) {
                    for (int mw = 0; mw < m; mw++) {
                        const int y0 = (hb * m + mh) * p, x0 = (wb * m + mw) * p;
                        for (int c = 0; c < 3; c++) {
                            for (int t = 0; t < T; t++) {
                                // an odd clip repeats its last frame (the processor's padding)
                                const int fi = std::min(g * T + t, clip.n - 1);
                                const uint8_t* f = r.data() + frame * fi;
                                for (int py = 0; py < p; py++) {
                                    const uint8_t* line = f + (static_cast<size_t>(y0 + py) * grid.resized_w + x0) * 3 + c;
                                    for (int px = 0; px < p; px++) {
                                        sink((static_cast<float>(line[px * 3]) - mean[c]) / stdv[c]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

std::vector<float> preprocess_video(const VideoClip& clip, const VideoGrid& grid, const VideoProcessorConfig& cfg) {
    std::vector<float> out;
    out.reserve(static_cast<size_t>(grid.grid_t) * grid.grid_h * grid.grid_w * PATCH_VALUES);
    for_each_value(clip, grid, cfg, [&](float v) { out.push_back(v); });
    return out;
}

void preprocess_video_bf16(const VideoClip& clip, const VideoGrid& grid, std::vector<uint16_t>& out,
                           const VideoProcessorConfig& cfg) {
    out.reserve(out.size() + static_cast<size_t>(grid.grid_t) * grid.grid_h * grid.grid_w * PATCH_VALUES);
    for_each_value(clip, grid, cfg, [&](float v) { out.push_back(float_to_bf16_bits(v)); });
}

std::vector<double> group_timestamps(const std::vector<int>& indices, double fps, int temporal) {
    std::vector<int> idx = indices;
    if (idx.empty()) return {};
    while (idx.size() % static_cast<size_t>(temporal)) idx.push_back(idx.back());
    std::vector<double> seconds(idx.size());
    for (size_t i = 0; i < idx.size(); i++) seconds[i] = static_cast<double>(idx[i]) / fps;
    std::vector<double> out;
    for (size_t i = 0; i < seconds.size(); i += static_cast<size_t>(temporal)) {
        out.push_back((seconds[i] + seconds[i + static_cast<size_t>(temporal) - 1]) / 2);
    }
    return out;
}

std::string video_placeholder(const std::vector<double>& times, int tokens_per_group, bool expand) {
    std::string out;
    char buf[64];
    for (double t : times) {
        std::snprintf(buf, sizeof(buf), "<%.1f seconds>", t);
        out += buf;
        out += VISION_START;
        const int n = expand ? tokens_per_group : 1;
        for (int i = 0; i < n; i++) out += VIDEO_PAD_TOKEN;
        out += VISION_END;
    }
    return out;
}

std::string expand_video_text(const std::string& text, const std::vector<std::string>& video_texts) {
    std::string out;
    size_t at = 0, video = 0;
    const std::string pad = VIDEO_PAD_TOKEN;
    for (;;) {
        const size_t hit = text.find(pad, at);
        if (hit == std::string::npos || video >= video_texts.size()) break;
        out.append(text, at, hit - at);
        out += video_texts[video++];
        at = hit + pad.size();
    }
    out.append(text, at, std::string::npos);
    return out;
}

std::vector<int> expand_pad_runs(const std::vector<int>& ids, int pad_id, const std::vector<int>& runs) {
    std::vector<int> out;
    size_t k = 0;
    for (int id : ids) {
        if (id == pad_id && k < runs.size()) {
            out.insert(out.end(), static_cast<size_t>(runs[k++]), pad_id);
        } else {
            out.push_back(id);
        }
    }
    return out;
}

void check_video_token_budget(long long planned) {
    if (planned > VIDEO_MAX_TOKENS) {
        throw VideoError("the videos take " + thousands(planned) + " input tokens, over the " +
                         thousands(VIDEO_MAX_TOKENS) +
                         " tokens a request may spend on videos; send fewer or shorter videos");
    }
}

void append_video(VideoPayload& payload, const VideoClip& clip, const VideoGrid& grid, const VideoProcessorConfig& cfg) {
    for (int g = 0; g < grid.grid_t; g++) payload.groups.emplace_back(grid.grid_h, grid.grid_w);
    preprocess_video_bf16(clip, grid, payload.pixels, cfg);
}

int mrope_positions(const std::vector<int>& ids, int image_id, int video_id,
                    const std::vector<std::pair<int, int>>& groups, std::vector<int>& out) {
    out.assign(ids.size() * 3, 0);
    int cur = 0;
    size_t group = 0, i = 0;
    while (i < ids.size()) {
        if (ids[i] == image_id || ids[i] == video_id) {
            if (group >= groups.size()) throw VideoError("the prompt holds more media token runs than grids");
            const int gh = groups[group].first / 2, gw = groups[group].second / 2;
            for (int r = 0; r < gh; r++) {
                for (int c = 0; c < gw; c++, i++) {
                    if (i >= ids.size() || (ids[i] != image_id && ids[i] != video_id)) {
                        throw VideoError("a media token run is shorter than its grid");
                    }
                    out[i * 3 + 0] = cur;
                    out[i * 3 + 1] = cur + r;
                    out[i * 3 + 2] = cur + c;
                }
            }
            cur += std::max(gh, gw);
            group++;
        } else {
            out[i * 3] = out[i * 3 + 1] = out[i * 3 + 2] = cur++;
            i++;
        }
    }
    if (group != groups.size()) throw VideoError("the prompt holds fewer media token runs than grids");
    return cur;
}

}  // namespace flm::d3::video
