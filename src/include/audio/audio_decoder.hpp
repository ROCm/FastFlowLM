/// \file audio_decoder.hpp
/// \brief Container/codec decoding for AudioReader (no FFmpeg)
/// \author FastFlowLM Team
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace audio {

/// Decoded PCM at the source's native sample rate and channel count.
struct DecodedAudio {
    /// Interleaved float32 samples, nominally in [-1.0, 1.0].
    std::vector<float> samples;
    int sample_rate = 0;
    int channels = 0;
};

/// Decode a complete audio file held in memory.
///
/// Supported: WAV (RIFF/RF64/W64/AIFF), FLAC (native and in Ogg), MP3,
/// Ogg Vorbis, Ogg Opus and WebM/Matroska Opus. Anything else, notably AAC
/// (.m4a/.mp4/ADTS), fails with a descriptive message in \p error.
bool decode_audio_file(const uint8_t* data, size_t size, DecodedAudio& out, std::string& error);

} // namespace audio
