/// \file py_math.hpp
/// \brief The few numeric behaviours of Python 3.12 / numpy that the d3 contract depends on bit for bit.
#pragma once

#include <cmath>
#include <cstddef>

namespace flm::d3 {

/// Python's built-in `sum()` over floats: since 3.12 it is Neumaier-compensated, so a plain loop
/// differs from upstream in the last bits.
struct PySum {
    double s = 0.0, c = 0.0;
    void add(double x) {
        double t = s + x;
        c += std::fabs(s) >= std::fabs(x) ? (s - t) + x : (x - t) + s;
        s = t;
    }
    double value() const { return (c != 0.0 && std::isfinite(c)) ? s + c : s; }
};

/// numpy's `DOUBLE_pairwise_sum` (what `ndarray.sum(axis=-1)` does over a contiguous row): blocks of up to
/// 128 elements summed by eight interleaved accumulators, larger sums split in two halves.
inline double np_pairwise_sum(const double* a, size_t n) {
    if (n < 8) {
        double res = 0.0;
        for (size_t i = 0; i < n; i++) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; j++) r[j] = a[j];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8) {
            for (int j = 0; j < 8; j++) r[j] += a[i + j];
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_pairwise_sum(a, n2) + np_pairwise_sum(a + n2, n - n2);
}

/// Python's `x % 1.0` for a finite x (the sign of the result follows the divisor).
inline double py_mod1(double x) {
    double m = std::fmod(x, 1.0);
    if (m != 0.0) {
        if (m < 0.0) m += 1.0;
    } else {
        m = std::copysign(0.0, 1.0);
    }
    return m;
}

}  // namespace flm::d3
