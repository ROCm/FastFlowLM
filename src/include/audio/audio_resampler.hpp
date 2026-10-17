/// \file audio_resampler.hpp
/// \brief Sample-rate conversion for AudioReader (no FFmpeg)
/// \author FastFlowLM Team
#pragma once

#include <vector>

namespace audio {

/// Resample interleaved float32 PCM from \p in_rate to \p out_rate.
///
/// Uses the filter design libswresample (FFmpeg) applies by default, so output
/// stays close to what AudioReader produced before: Kaiser-windowed sinc
/// (beta 9), cutoff 0.97 of the lower Nyquist, filter_size 32, exact rational
/// phases when they fit in 1024 (otherwise 1024 phases with linear
/// interpolation), and mirrored signal edges.
std::vector<float> resample(const std::vector<float>& in, int channels, int in_rate, int out_rate);

} // namespace audio
