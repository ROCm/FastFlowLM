/// \file audio_resampler.cpp
/// \brief Sample-rate conversion matching libswresample's default filter design
/// \author FastFlowLM Team

#include "audio/audio_resampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

namespace audio {
namespace {

// libswresample defaults (libswresample/options.c).
constexpr int kFilterSize = 32;
constexpr int kPhaseShift = 10;
constexpr double kCutoff = 0.97;
constexpr double kKaiserBeta = 9.0;

// Zeroth-order modified Bessel function of the first kind (power series).
double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    const double q = x * x / 4.0;
    for (int k = 1; k < 64; ++k) {
        term *= q / (static_cast<double>(k) * k);
        sum += term;
        if (term < sum * 1e-17) {
            break;
        }
    }
    return sum;
}

// Kaiser-windowed sinc filter bank, one row of `taps` coefficients per phase,
// normalized by the DC gain of phase 0 (as build_filter() in
// libswresample/resample.c does).
std::vector<float> build_filter_bank(double factor, int taps, int phases) {
    const int center = (taps - 1) / 2;
    std::vector<float> bank(static_cast<size_t>(phases + 1) * taps);
    std::vector<double> row(taps);
    double norm = 0.0;
    for (int ph = 0; ph <= phases; ++ph) {
        for (int i = 0; i < taps; ++i) {
            const double x = M_PI * ((i - center) - static_cast<double>(ph) / phases) * factor;
            double y = x == 0.0 ? 1.0 : std::sin(x) / x;
            const double w = 2.0 * x / (factor * taps * M_PI);
            y *= bessel_i0(kKaiserBeta * std::sqrt(std::max(1.0 - w * w, 0.0)));
            row[i] = y;
            if (ph == 0) {
                norm += y;
            }
        }
        for (int i = 0; i < taps; ++i) {
            bank[static_cast<size_t>(ph) * taps + i] = static_cast<float>(row[i] / norm);
        }
    }
    return bank;
}

// Index into the signal extended the way libswresample extends it: mirrored
// about the first sample at the start (x[-k] = x[k], invert_initial_buffer)
// and about the end at the tail (x[n+k] = x[n-1-k], resample_flush). The
// filter never reaches further than its half-length past either edge.
inline int64_t mirror(int64_t i, int64_t n) {
    if (i < 0) {
        i = -i;
    } else if (i >= n) {
        i = 2 * n - 1 - i;
    }
    return std::clamp<int64_t>(i, 0, n - 1);
}

} // namespace

std::vector<float> resample(const std::vector<float>& in, int channels, int in_rate, int out_rate) {
    if (in_rate == out_rate || in.empty() || channels <= 0 || in_rate <= 0 || out_rate <= 0) {
        return in;
    }
    const int64_t in_frames = static_cast<int64_t>(in.size()) / channels;

    const double factor = std::min(out_rate * kCutoff / in_rate, 1.0);
    // libswresample rounds the filter length up to an even number of taps.
    int taps = std::max(static_cast<int>(std::ceil(kFilterSize / factor)), 1);
    if (taps > 1) {
        taps += taps & 1;
    }
    const int center = (taps - 1) / 2;

    // Output sample n sits at input position n * in_rate / out_rate. With the
    // ratio reduced to in_step / out_step, the fractional part takes exactly
    // out_step values; use one filter phase each when that fits in the default
    // 1024 phases, otherwise 1024 phases with linear interpolation.
    const int g = std::gcd(in_rate, out_rate);
    const int64_t in_step = in_rate / g;
    const int64_t out_step = out_rate / g;
    const bool exact = out_step <= (1 << kPhaseShift);
    const int phases = exact ? static_cast<int>(out_step) : (1 << kPhaseShift);
    const std::vector<float> bank = build_filter_bank(factor, taps, phases);

    const int64_t out_frames = (in_frames * out_step + in_step - 1) / in_step;
    std::vector<float> out(static_cast<size_t>(out_frames) * channels);

    for (int64_t n = 0; n < out_frames; ++n) {
        const int64_t pos = n * in_step;
        const int64_t idx = pos / out_step;
        const int64_t rem = pos % out_step;
        const int64_t first = idx - center;
        const bool interior = first >= 0 && first + taps <= in_frames;

        for (int c = 0; c < channels; ++c) {
            auto sample = [&](int i) -> float {
                const int64_t k = interior ? first + i : mirror(first + i, in_frames);
                return in[static_cast<size_t>(k) * channels + c];
            };
            float v;
            if (exact) {
                const float* h = &bank[static_cast<size_t>(rem) * taps];
                v = 0.0f;
                for (int i = 0; i < taps; ++i) {
                    v += sample(i) * h[i];
                }
            } else {
                const double fpos = static_cast<double>(rem) * phases / out_step;
                const int ph = static_cast<int>(fpos);
                const float frac = static_cast<float>(fpos - ph);
                const float* h1 = &bank[static_cast<size_t>(ph) * taps];
                const float* h2 = h1 + taps;
                float v1 = 0.0f;
                float v2 = 0.0f;
                for (int i = 0; i < taps; ++i) {
                    const float s = sample(i);
                    v1 += s * h1[i];
                    v2 += s * h2[i];
                }
                v = v1 + (v2 - v1) * frac;
            }
            out[static_cast<size_t>(n) * channels + c] = v;
        }
    }
    return out;
}

} // namespace audio
