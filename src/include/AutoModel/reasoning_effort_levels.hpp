// Copyright (c) 2026 Advanced Micro Devices, Inc.

/// \file reasoning_effort_levels.hpp
/// \brief The reasoning_effort values the Hy-Memory and Inkling-2 chat templates read
/// \note Header only, so the CPU tests check them without the NPU engines behind the models.
#pragma once

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>

namespace flm {

/// \brief Hy-Memory's template level for a reasoning_effort
/// \param effort one of the four canonical levels (none, low, medium, high) or Hy-Memory's own no_think
/// \return "no_think", "low" or "high"; empty for anything else
inline std::optional<std::string> hy_memory_reasoning_effort(const std::string& effort) {
    if (effort == "none" || effort == "no_think") return std::string("no_think");
    if (effort == "low") return std::string("low");
    if (effort == "medium" || effort == "high") return std::string("high");
    return std::nullopt;
}

/// \brief Read an Inkling-2 numeric effort such as "0.5": a plain decimal number in [0, 0.99]
/// \param level set to the number when it is one
/// \return false for anything else (hex, exponents with p, signs, spaces, inf, nan, out of range)
inline bool parse_inkling_numeric_effort(const std::string& effort, double& level) {
    if (effort.empty() || (!std::isdigit(static_cast<unsigned char>(effort[0])) && effort[0] != '.')) {
        return false;
    }
    if (effort.find_first_of("xXpP") != std::string::npos) {
        return false;
    }
    char* end = nullptr;
    const double parsed = std::strtod(effort.c_str(), &end);
    if (end == effort.c_str() || end != effort.c_str() + effort.size()) {
        return false;
    }
    if (!std::isfinite(parsed) || parsed < 0.0 || parsed > 0.99) {
        return false;
    }
    level = parsed;
    return true;
}

/// \brief whether `effort` is an Inkling-2 level name or a number in [0, 0.99]
inline bool inkling_valid_effort(const std::string& effort) {
    static const char* names[] = {"none", "minimal", "low", "medium", "high", "xhigh", "max"};
    for (const char* name : names) {
        if (effort == name) return true;
    }
    double level = 0.0;
    return parse_inkling_numeric_effort(effort, level);
}

}  // namespace flm
