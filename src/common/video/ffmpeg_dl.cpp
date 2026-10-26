/// FFmpeg as a run-time dependency: libavformat / libavcodec / libavutil / libswscale are opened with dlopen
/// (LoadLibrary on Windows) and the C API is called through function pointers, so FastFlowLM builds and links
/// without FFmpeg headers or libraries. Only plain functions are used, plus a handful of struct fields whose
/// offsets are the same in FFmpeg 6.x, 7.x and 8.x (checked against 6.1 headers by test_d3_video):
///   AVFormatContext: pb, nb_streams, streams;  AVStream: index, codecpar, time_base, nb_frames, avg_frame_rate;
///   AVFrame: data, linesize, width, height, format;  AVPacket: stream_index;  AVIOContext: buffer.
/// Video is read like OpenCV's FFmpeg capture does (what upstream uses): the first video stream, sequential
/// decoding, swscale SWS_BICUBIC to RGB24 (BT.601), frame count = nb_frames else duration * fps + 0.5.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>

#include "video/video_decode.hpp"

#ifndef FLM_NO_FFMPEG_DL
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace flm::d3::video {

namespace {

// ---- minimal declarations of the FFmpeg C API -------------------------------------------------------------
struct AVRational { int num, den; };
struct AVFormatContext;
struct AVCodecContext;
struct AVCodec;
struct AVStream;
struct AVPacket;
struct AVFrame;
struct AVIOContext;
struct SwsContext;

constexpr int AVMEDIA_TYPE_VIDEO = 0;
constexpr int AV_PIX_FMT_RGB24 = 2;
constexpr int SWS_BICUBIC = 4;
constexpr int AVERROR_EOF_ = -541478725;   // FFERRTAG('E','O','F',' ')
constexpr int AVSEEK_SIZE = 0x10000;
constexpr int AV_LOG_QUIET = -8;
#ifdef _WIN32
constexpr int AVERROR_EAGAIN = -11;   // MinGW's EAGAIN
#else
constexpr int AVERROR_EAGAIN = -11;   // Linux; the BSDs (35) are not supported targets
#endif

// byte offsets (see the file comment)
constexpr size_t OFF_FMT_PB = 32, OFF_FMT_NB_STREAMS = 44, OFF_FMT_STREAMS = 48, OFF_FMT_DURATION = 72;
constexpr size_t OFF_ST_INDEX = 8, OFF_ST_CODECPAR = 16, OFF_ST_TIME_BASE = 32, OFF_ST_DURATION = 48,
                 OFF_ST_NB_FRAMES = 56, OFF_ST_AVG_FRAME_RATE = 88;
constexpr size_t OFF_FRAME_DATA = 0, OFF_FRAME_LINESIZE = 64, OFF_FRAME_WIDTH = 104, OFF_FRAME_HEIGHT = 108,
                 OFF_FRAME_FORMAT = 116;
constexpr size_t OFF_PKT_STREAM_INDEX = 36;
constexpr size_t OFF_IO_BUFFER = 8;

template <class T>
T field(const void* base, size_t off) {
    T v;
    std::memcpy(&v, static_cast<const char*>(base) + off, sizeof(T));
    return v;
}

struct Api {
    // libavutil
    void* (*av_malloc)(size_t) = nullptr;
    void (*av_freep)(void*) = nullptr;
    void (*av_log_set_level)(int) = nullptr;
    AVFrame* (*av_frame_alloc)() = nullptr;
    void (*av_frame_free)(AVFrame**) = nullptr;
    void (*av_frame_unref)(AVFrame*) = nullptr;
    double (*av_display_rotation_get)(const int32_t*) = nullptr;
    // libavformat
    AVFormatContext* (*avformat_alloc_context)() = nullptr;
    int (*avformat_open_input)(AVFormatContext**, const char*, const void*, void**) = nullptr;
    int (*avformat_find_stream_info)(AVFormatContext*, void**) = nullptr;
    void (*avformat_close_input)(AVFormatContext**) = nullptr;
    int (*av_find_best_stream)(AVFormatContext*, int, int, int, const AVCodec**, int) = nullptr;
    int (*av_read_frame)(AVFormatContext*, AVPacket*) = nullptr;
    AVIOContext* (*avio_alloc_context)(unsigned char*, int, int, void*, int (*)(void*, uint8_t*, int),
                                       int (*)(void*, uint8_t*, int), int64_t (*)(void*, int64_t, int)) = nullptr;
    void (*avio_context_free)(AVIOContext**) = nullptr;
    uint8_t* (*av_stream_get_side_data)(const AVStream*, int, size_t*) = nullptr;
    unsigned (*avformat_version)() = nullptr;
    // libavcodec
    AVCodecContext* (*avcodec_alloc_context3)(const AVCodec*) = nullptr;
    void (*avcodec_free_context)(AVCodecContext**) = nullptr;
    int (*avcodec_parameters_to_context)(AVCodecContext*, const void*) = nullptr;
    int (*avcodec_open2)(AVCodecContext*, const AVCodec*, void**) = nullptr;
    int (*avcodec_send_packet)(AVCodecContext*, const AVPacket*) = nullptr;
    int (*avcodec_receive_frame)(AVCodecContext*, AVFrame*) = nullptr;
    AVPacket* (*av_packet_alloc)() = nullptr;
    void (*av_packet_free)(AVPacket**) = nullptr;
    void (*av_packet_unref)(AVPacket*) = nullptr;
    // libswscale
    SwsContext* (*sws_getContext)(int, int, int, int, int, int, int, void*, void*, const double*) = nullptr;
    int (*sws_scale)(SwsContext*, const uint8_t* const*, const int*, int, int, uint8_t* const*, const int*) = nullptr;
    void (*sws_freeContext)(SwsContext*) = nullptr;
};

#ifdef FLM_NO_FFMPEG_DL
using Handle = void*;
Handle open_lib(const std::string&) { return nullptr; }
void* sym(Handle, const char*) { return nullptr; }
std::string sep() { return "/"; }
std::string lib_name(const char*, int) { return std::string(); }
#elif defined(_WIN32)
using Handle = HMODULE;
Handle open_lib(const std::string& name) { return LoadLibraryA(name.c_str()); }
void* sym(Handle h, const char* n) { return reinterpret_cast<void*>(GetProcAddress(h, n)); }
std::string sep() { return "\\"; }
std::string lib_name(const char* stem, int major) { return std::string(stem) + "-" + std::to_string(major) + ".dll"; }
#else
using Handle = void*;
Handle open_lib(const std::string& name) { return dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL); }
void* sym(Handle h, const char* n) { return dlsym(h, n); }
std::string sep() { return "/"; }
std::string lib_name(const char* stem, int major) {
#ifdef __APPLE__
    return std::string("lib") + stem + "." + std::to_string(major) + ".dylib";
#else
    return std::string("lib") + stem + ".so." + std::to_string(major);
#endif
}
#endif

/// The set of majors that go together: (avformat, avcodec, avutil, swscale) of FFmpeg 6, 7, 8.
struct Release { int format, codec, util, sws; };
const Release RELEASES[] = {{62, 62, 60, 9}, {61, 61, 59, 8}, {60, 60, 58, 7}};

struct Loaded {
    Api api;
    bool ok = false;
    std::string why;
};

Loaded load_all(const std::string& dir_in) {
    Loaded L;
    std::string dir = dir_in;
    if (dir.empty()) {
        if (const char* env = std::getenv("FLM_FFMPEG_DIR")) dir = env;
    }
    std::string last;
    for (const Release& r : RELEASES) {
        const auto open = [&](const char* stem, int major) -> Handle {
            if (!dir.empty()) {
                if (Handle h = open_lib(dir + sep() + lib_name(stem, major))) return h;
            }
            return open_lib(lib_name(stem, major));
        };
        Handle util = open("avutil", r.util), codec = open("avcodec", r.codec), fmt = open("avformat", r.format),
               sws = open("swscale", r.sws);
        if (!util || !codec || !fmt || !sws) {
#ifdef FLM_NO_FFMPEG_DL
            last = "this build has no FFmpeg run-time loading (FLM_VIDEO_FFMPEG=OFF)";
#else
            last = "could not load libavformat/libavcodec/libavutil/libswscale of FFmpeg 6, 7 or 8";
#endif
            continue;
        }
        Api& a = L.api;
        bool all = true;
        const auto need = [&](auto& fn, Handle h, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(sym(h, name));
            if (!fn) {
                all = false;
                last = std::string("FFmpeg library lacks ") + name;
            }
        };
        need(a.av_malloc, util, "av_malloc");
        need(a.av_freep, util, "av_freep");
        need(a.av_log_set_level, util, "av_log_set_level");
        need(a.av_frame_alloc, util, "av_frame_alloc");
        need(a.av_frame_free, util, "av_frame_free");
        need(a.av_frame_unref, util, "av_frame_unref");
        need(a.avformat_alloc_context, fmt, "avformat_alloc_context");
        need(a.avformat_open_input, fmt, "avformat_open_input");
        need(a.avformat_find_stream_info, fmt, "avformat_find_stream_info");
        need(a.avformat_close_input, fmt, "avformat_close_input");
        need(a.av_find_best_stream, fmt, "av_find_best_stream");
        need(a.av_read_frame, fmt, "av_read_frame");
        need(a.avio_alloc_context, fmt, "avio_alloc_context");
        need(a.avio_context_free, fmt, "avio_context_free");
        need(a.avformat_version, fmt, "avformat_version");
        need(a.avcodec_alloc_context3, codec, "avcodec_alloc_context3");
        need(a.avcodec_free_context, codec, "avcodec_free_context");
        need(a.avcodec_parameters_to_context, codec, "avcodec_parameters_to_context");
        need(a.avcodec_open2, codec, "avcodec_open2");
        need(a.avcodec_send_packet, codec, "avcodec_send_packet");
        need(a.avcodec_receive_frame, codec, "avcodec_receive_frame");
        need(a.av_packet_alloc, codec, "av_packet_alloc");
        need(a.av_packet_free, codec, "av_packet_free");
        need(a.av_packet_unref, codec, "av_packet_unref");
        need(a.sws_getContext, sws, "sws_getContext");
        need(a.sws_scale, sws, "sws_scale");
        need(a.sws_freeContext, sws, "sws_freeContext");
        // optional: display-matrix rotation (removed from newer FFmpeg: then frames are not rotated)
        a.av_stream_get_side_data = reinterpret_cast<decltype(a.av_stream_get_side_data)>(sym(fmt, "av_stream_get_side_data"));
        a.av_display_rotation_get = reinterpret_cast<decltype(a.av_display_rotation_get)>(sym(util, "av_display_rotation_get"));
        if (!all) continue;
        if ((a.avformat_version() >> 16) != static_cast<unsigned>(r.format)) {
            last = "unexpected libavformat version";
            continue;
        }
        a.av_log_set_level(AV_LOG_QUIET);
        L.ok = true;
        return L;
    }
    L.why = last.empty() ? "FFmpeg shared libraries not found" : last;
    return L;
}

/// Encoded bytes as an FFmpeg input.
struct MemoryInput {
    const std::string* data;
    size_t pos = 0;
};

int read_packet(void* opaque, uint8_t* buf, int size) {
    auto* in = static_cast<MemoryInput*>(opaque);
    const size_t left = in->data->size() - in->pos;
    if (left == 0) return AVERROR_EOF_;
    const size_t n = std::min(left, static_cast<size_t>(size));
    std::memcpy(buf, in->data->data() + in->pos, n);
    in->pos += n;
    return static_cast<int>(n);
}

int64_t seek_packet(void* opaque, int64_t offset, int whence) {
    auto* in = static_cast<MemoryInput*>(opaque);
    if (whence & AVSEEK_SIZE) return static_cast<int64_t>(in->data->size());
    whence &= ~0x20000;   // AVSEEK_FORCE
    int64_t target;
    switch (whence) {
        case 0: target = offset; break;
        case 1: target = static_cast<int64_t>(in->pos) + offset; break;
        case 2: target = static_cast<int64_t>(in->data->size()) + offset; break;
        default: return -1;
    }
    if (target < 0 || target > static_cast<int64_t>(in->data->size())) return -1;
    in->pos = static_cast<size_t>(target);
    return target;
}

class FfmpegReader : public FrameReader {
public:
    FfmpegReader(const Api& api, const std::string& bytes) : api_(api), input_{&bytes, 0} {}

    ~FfmpegReader() override {
        if (sws_) api_.sws_freeContext(sws_);
        if (frame_) api_.av_frame_free(&frame_);
        if (packet_) api_.av_packet_free(&packet_);
        if (codec_) api_.avcodec_free_context(&codec_);
        if (fmt_) api_.avformat_close_input(&fmt_);
        if (io_) {
            unsigned char* buf = field<unsigned char*>(io_, OFF_IO_BUFFER);
            api_.av_freep(&buf);   // the buffer FFmpeg may have reallocated; avio_context_free does not free it
            std::memset(reinterpret_cast<char*>(io_) + OFF_IO_BUFFER, 0, sizeof(void*));
            api_.avio_context_free(&io_);
        }
    }

    bool open(std::string* error) {
        constexpr int kBuf = 1 << 16;
        auto* buffer = static_cast<unsigned char*>(api_.av_malloc(kBuf));
        if (!buffer) { *error = "out of memory"; return false; }
        io_ = api_.avio_alloc_context(buffer, kBuf, 0, &input_, read_packet, nullptr, seek_packet);
        if (!io_) { api_.av_freep(&buffer); *error = "out of memory"; return false; }
        fmt_ = api_.avformat_alloc_context();
        if (!fmt_) { *error = "out of memory"; return false; }
        const void* pb = io_;
        std::memcpy(reinterpret_cast<char*>(fmt_) + OFF_FMT_PB, &pb, sizeof(pb));   // ic->pb: custom I/O
        if (api_.avformat_open_input(&fmt_, "", nullptr, nullptr) < 0) {   // frees fmt_ on failure
            fmt_ = nullptr;
            *error = "the video could not be opened";
            return false;
        }
        if (api_.avformat_find_stream_info(fmt_, nullptr) < 0) { *error = "the video could not be opened"; return false; }
        const AVCodec* decoder = nullptr;
        stream_ = api_.av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
        if (stream_ < 0 || !decoder) { *error = "the video could not be opened"; return false; }
        const unsigned n = field<unsigned>(fmt_, OFF_FMT_NB_STREAMS);
        if (static_cast<unsigned>(stream_) >= n) { *error = "the video could not be opened"; return false; }
        AVStream* st = field<AVStream**>(fmt_, OFF_FMT_STREAMS)[stream_];
        codec_ = api_.avcodec_alloc_context3(decoder);
        if (!codec_ || api_.avcodec_parameters_to_context(codec_, field<void*>(st, OFF_ST_CODECPAR)) < 0 ||
            api_.avcodec_open2(codec_, decoder, nullptr) < 0) {
            *error = "invalid video data (the decoder could not be opened)";
            return false;
        }
        frame_ = api_.av_frame_alloc();
        packet_ = api_.av_packet_alloc();
        if (!frame_ || !packet_) { *error = "out of memory"; return false; }
        // stream info, as OpenCV's capture reads it. The frame size comes from the first decoded frame (the
        // AVCodecParameters layout differs between FFmpeg versions); that frame is handed out by the first grab().
        if (pull()) {
            pending_ = true;
            info_.width = field<int>(frame_, OFF_FRAME_WIDTH);
            info_.height = field<int>(frame_, OFF_FRAME_HEIGHT);
        }
        const AVRational avg = field<AVRational>(st, OFF_ST_AVG_FRAME_RATE);
        info_.fps = avg.den > 0 ? static_cast<double>(avg.num) / avg.den : 0.0;
        const int64_t nb = field<int64_t>(st, OFF_ST_NB_FRAMES);
        const AVRational tb = field<AVRational>(st, OFF_ST_TIME_BASE);
        const int64_t sdur = field<int64_t>(st, OFF_ST_DURATION);
        if (nb > 0) {
            info_.total = nb;
        } else if (info_.fps > 0) {
            // duration_sec * fps + 0.5: the container's duration when the stream has none (FFmpeg 6 layout only)
            double sec = 0.0;
            if (sdur > 0 && tb.den > 0) sec = static_cast<double>(sdur) * tb.num / tb.den;
            if (sec <= 0.0 && (api_.avformat_version() >> 16) == 60) {
                const int64_t d = field<int64_t>(fmt_, OFF_FMT_DURATION);
                if (d > 0) sec = static_cast<double>(d) / 1000000.0;
            }
            info_.total = sec > 0 ? static_cast<long long>(sec * info_.fps + 0.5) : 0;
        }
        rotation_ = display_rotation(st);
        if (rotation_ == 90 || rotation_ == 270) std::swap(info_.width, info_.height);
        return true;
    }

    const StreamInfo& info() const override { return info_; }

    bool grab() override {
        if (pending_) {
            pending_ = false;
            return true;
        }
        return pull();
    }

    bool pull() {
        for (;;) {
            api_.av_frame_unref(frame_);
            const int r = api_.avcodec_receive_frame(codec_, frame_);
            if (r == 0) return true;
            if (r != AVERROR_EAGAIN) return false;   // EOF (after the flush) or an error
            if (drained_) return false;
            // feed the decoder with the next packet of the video stream
            for (;;) {
                const int rr = api_.av_read_frame(fmt_, packet_);
                if (rr < 0) {
                    api_.avcodec_send_packet(codec_, nullptr);   // flush
                    drained_ = true;
                    break;
                }
                const bool mine = field<int>(packet_, OFF_PKT_STREAM_INDEX) == stream_;
                if (mine) api_.avcodec_send_packet(codec_, packet_);
                api_.av_packet_unref(packet_);
                if (mine) break;
            }
        }
    }

    bool retrieve(std::vector<uint8_t>& rgb) override {
        const int w = field<int>(frame_, OFF_FRAME_WIDTH), h = field<int>(frame_, OFF_FRAME_HEIGHT);
        const int fmt = field<int>(frame_, OFF_FRAME_FORMAT);
        if (w <= 0 || h <= 0 || fmt < 0) return false;
        if (!sws_ || sws_w_ != w || sws_h_ != h || sws_fmt_ != fmt) {
            if (sws_) api_.sws_freeContext(sws_);
            sws_ = api_.sws_getContext(w, h, fmt, w, h, AV_PIX_FMT_RGB24, SWS_BICUBIC, nullptr, nullptr, nullptr);
            sws_w_ = w; sws_h_ = h; sws_fmt_ = fmt;
            if (!sws_) return false;
        }
        // swscale's SIMD stores may run a few bytes past the last row: pad the buffer, then trim
        std::vector<uint8_t> plain(static_cast<size_t>(w) * h * 3 + 64);
        uint8_t* dst[1] = {plain.data()};
        const int dst_stride[1] = {w * 3};
        const uint8_t* src[8];
        int src_stride[8];
        std::memcpy(src, static_cast<const char*>(static_cast<const void*>(frame_)) + OFF_FRAME_DATA, sizeof(src));
        std::memcpy(src_stride, static_cast<const char*>(static_cast<const void*>(frame_)) + OFF_FRAME_LINESIZE, sizeof(src_stride));
        if (api_.sws_scale(sws_, src, src_stride, 0, h, dst, dst_stride) <= 0) return false;
        plain.resize(static_cast<size_t>(w) * h * 3);
        rotate(plain, w, h, rgb);
        return true;
    }

private:
    int display_rotation(AVStream* st) const {
        if (!api_.av_stream_get_side_data || !api_.av_display_rotation_get) return 0;
        size_t size = 0;
        constexpr int AV_PKT_DATA_DISPLAYMATRIX = 5;
        const uint8_t* m = api_.av_stream_get_side_data(st, AV_PKT_DATA_DISPLAYMATRIX, &size);
        if (!m || size < 36) return 0;
        int32_t matrix[9];
        std::memcpy(matrix, m, sizeof(matrix));
        const double ccw = api_.av_display_rotation_get(matrix);
        if (!std::isfinite(ccw)) return 0;
        int cw = static_cast<int>(std::lround(-ccw)) % 360;   // clockwise degrees to turn the frame upright
        if (cw < 0) cw += 360;
        return cw == 90 || cw == 180 || cw == 270 ? cw : 0;
    }

    void rotate(const std::vector<uint8_t>& in, int w, int h, std::vector<uint8_t>& out) const {
        if (rotation_ == 0) { out = in; return; }
        out.resize(in.size());
        const auto px = [&](int x, int y) { return in.data() + (static_cast<size_t>(y) * w + x) * 3; };
        if (rotation_ == 180) {
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) std::memcpy(out.data() + (static_cast<size_t>(y) * w + x) * 3, px(w - 1 - x, h - 1 - y), 3);
        } else if (rotation_ == 90) {   // clockwise: out is h wide, w tall
            for (int y = 0; y < w; y++)
                for (int x = 0; x < h; x++) std::memcpy(out.data() + (static_cast<size_t>(y) * h + x) * 3, px(y, h - 1 - x), 3);
        } else {                        // 270
            for (int y = 0; y < w; y++)
                for (int x = 0; x < h; x++) std::memcpy(out.data() + (static_cast<size_t>(y) * h + x) * 3, px(w - 1 - y, x), 3);
        }
    }

    const Api& api_;
    MemoryInput input_;
    AVFormatContext* fmt_ = nullptr;
    AVIOContext* io_ = nullptr;
    AVCodecContext* codec_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* packet_ = nullptr;
    SwsContext* sws_ = nullptr;
    int sws_w_ = 0, sws_h_ = 0, sws_fmt_ = -1;
    int stream_ = -1;
    int rotation_ = 0;
    bool drained_ = false;
    bool pending_ = false;
    StreamInfo info_;
};

class FfmpegBackend : public FrameBackend {
public:
    explicit FfmpegBackend(std::string dir) : dir_(std::move(dir)) {}
    std::string name() const override { return "ffmpeg"; }
    bool available(std::string* why) const override {
        const Loaded& l = loaded();
        if (!l.ok && why) *why = l.why;
        return l.ok;
    }
    std::unique_ptr<FrameReader> open(const std::string& bytes, std::string* error) override {
        const Loaded& l = loaded();
        if (!l.ok) {
            if (error) *error = l.why;
            return nullptr;
        }
        auto reader = std::make_unique<FfmpegReader>(l.api, bytes);
        std::string e;
        if (!reader->open(&e)) {
            if (error) *error = e;
            return nullptr;
        }
        return reader;
    }

private:
    const Loaded& loaded() const {
        std::call_once(once_, [&] { loaded_ = load_all(dir_); });
        return loaded_;
    }
    std::string dir_;
    mutable std::once_flag once_;
    mutable Loaded loaded_;
};

}  // namespace

std::shared_ptr<FrameBackend> make_ffmpeg_backend(const std::string& dir) {
    return std::make_shared<FfmpegBackend>(dir);
}

}  // namespace flm::d3::video
