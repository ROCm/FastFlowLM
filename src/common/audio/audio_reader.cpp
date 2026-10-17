/// \file audio_reader.cpp
/// \brief AudioReader implementation – decode (audio_decoder) and resample (audio_resampler)
/// \author FastFlowLM Team

#include "audio/audio_reader.hpp"
#include "audio/audio_decoder.hpp"
#include "audio/audio_resampler.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cmath>

// ---------------------------------------------------------------------------
// AudioReader
// ---------------------------------------------------------------------------
AudioReader::AudioReader() = default;

AudioReader::~AudioReader() = default;

// ---------------------------------------------------------------------------
bool AudioReader::load_audio(const std::string& filename,
                             audio_data_t& out_audio,
                             int target_sample_rate,
                             MonoDownmixMode downmix)
{
    if (!std::filesystem::exists(filename)) {
        std::cerr << "Error: audio file not found: " << filename << std::endl;
        return false;
    }

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    const std::streamsize file_size = file.is_open() ? static_cast<std::streamsize>(file.tellg()) : -1;
    if (file_size <= 0) {
        std::cerr << "Error: could not open audio file: " << filename << std::endl;
        return false;
    }
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(file_size));
    if (!file.read(reinterpret_cast<char*>(data.data()), file_size)) {
        std::cerr << "Error: could not read audio file: " << filename << std::endl;
        return false;
    }

    return decode_audio(data.data(), data.size(), out_audio, target_sample_rate, downmix);
}

// ---------------------------------------------------------------------------
bool AudioReader::load_audio_from_memory(const uint8_t* data, size_t size,
                                         audio_data_t& out_audio,
                                         int target_sample_rate,
                                         MonoDownmixMode downmix)
{
    if (!data || size == 0) {
        std::cerr << "Error: empty audio buffer" << std::endl;
        return false;
    }
    return decode_audio(data, size, out_audio, target_sample_rate, downmix);
}

// ---------------------------------------------------------------------------
// Core decode + downmix + resample logic (shared by file and memory paths)
// ---------------------------------------------------------------------------
bool AudioReader::decode_audio(const uint8_t* data, size_t size,
                               audio_data_t& out_audio,
                               int target_sample_rate,
                               MonoDownmixMode downmix)
{
    if (target_sample_rate <= 0) {
        std::cerr << "Error: invalid target sample rate " << target_sample_rate << std::endl;
        return false;
    }

    audio::DecodedAudio decoded;
    std::string error;
    if (!audio::decode_audio_file(data, size, decoded, error)) {
        std::cerr << "Error: could not decode audio: " << error << std::endl;
        return false;
    }

    const int original_sample_rate = decoded.sample_rate;
    const int original_channels    = decoded.channels;
    std::vector<float> samples     = std::move(decoded.samples);

    // Downmix to mono before resampling (cheaper, and the order libswresample
    // used): every channel weighted by the same coefficient.
    const bool do_downmix = (downmix != MonoDownmixMode::NONE) && (original_channels > 1);
    int out_channels = original_channels;
    if (do_downmix) {
        const float coeff = downmix == MonoDownmixMode::RMS
            // Energy-preserving: 1/sqrt(N) per channel (FFmpeg default)
            ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(original_channels)))
            // Simple average: 1/N per channel (matches librosa's to_mono)
            : 1.0f / static_cast<float>(original_channels);
        const size_t frames = samples.size() / original_channels;
        std::vector<float> mono(frames);
        for (size_t f = 0; f < frames; ++f) {
            float sum = 0.0f;
            for (int c = 0; c < original_channels; ++c) {
                sum += samples[f * original_channels + c];
            }
            mono[f] = sum * coeff;
        }
        samples = std::move(mono);
        out_channels = 1;
    }

    samples = audio::resample(samples, out_channels, original_sample_rate, target_sample_rate);

    // Fill output struct
    out_audio.samples              = std::move(samples);
    out_audio.num_samples          = out_audio.samples.size();
    out_audio.sample_rate          = target_sample_rate;
    out_audio.original_sample_rate = original_sample_rate;
    out_audio.channels             = out_channels;
    out_audio.original_channels    = original_channels;
    out_audio.num_frames           = (out_channels > 0)
                                     ? out_audio.num_samples / out_channels
                                     : 0;
    out_audio.duration_seconds     = static_cast<double>(out_audio.num_frames) / target_sample_rate;

    return out_audio.num_samples > 0;
}

// ---------------------------------------------------------------------------
bool AudioReader::clip_audio_length(audio_data_t& audio, double max_duration_second)
{
    if (audio.sample_rate <= 0 || audio.channels <= 0 || max_duration_second <= 0.0) {
        return false;
    }

    if (audio.duration_seconds <= max_duration_second) {
        return true; // nothing to clip
    }

    size_t max_frames = static_cast<size_t>(max_duration_second * audio.sample_rate);
    size_t max_samples = max_frames * audio.channels;

    if (max_samples < audio.samples.size()) {
        audio.samples.resize(max_samples);
        audio.num_samples = max_samples;
        audio.num_frames = max_frames;
        audio.duration_seconds = static_cast<double>(max_frames) / audio.sample_rate;
    }

    return true;
}
