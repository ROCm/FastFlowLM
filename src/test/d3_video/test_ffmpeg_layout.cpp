/// Checks the field offsets ffmpeg_dl.cpp hard-codes against the FFmpeg headers this is compiled with (FFmpeg 6.1 here;
/// build against 7.x / 8.x headers to check those). Compile-time only: a mismatch fails the build of this target.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/frame.h>
}
#include <cstddef>
#include <cstdio>

static_assert(offsetof(AVFormatContext, pb) == 32, "AVFormatContext.pb");
static_assert(offsetof(AVFormatContext, nb_streams) == 44, "AVFormatContext.nb_streams");
static_assert(offsetof(AVFormatContext, streams) == 48, "AVFormatContext.streams");
static_assert(offsetof(AVStream, codecpar) == 16, "AVStream.codecpar");
static_assert(offsetof(AVStream, time_base) == 32, "AVStream.time_base");
static_assert(offsetof(AVStream, duration) == 48, "AVStream.duration");
static_assert(offsetof(AVStream, nb_frames) == 56, "AVStream.nb_frames");
static_assert(offsetof(AVStream, avg_frame_rate) == 88, "AVStream.avg_frame_rate");
static_assert(offsetof(AVFrame, data) == 0, "AVFrame.data");
static_assert(offsetof(AVFrame, linesize) == 64, "AVFrame.linesize");
static_assert(offsetof(AVFrame, width) == 104, "AVFrame.width");
static_assert(offsetof(AVFrame, height) == 108, "AVFrame.height");
static_assert(offsetof(AVFrame, format) == 116, "AVFrame.format");
static_assert(offsetof(AVPacket, stream_index) == 36, "AVPacket.stream_index");
static_assert(offsetof(AVIOContext, buffer) == 8, "AVIOContext.buffer");

int main() {
    std::printf("ffmpeg layout ok (libavformat %d.%d)\n", LIBAVFORMAT_VERSION_MAJOR, LIBAVFORMAT_VERSION_MINOR);
    return 0;
}
