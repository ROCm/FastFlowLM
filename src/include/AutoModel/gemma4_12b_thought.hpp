// Copyright (c) 2026 Advanced Micro Devices, Inc.

/// \file gemma4_12b_thought.hpp
/// \brief When Gemma4 12B drops the empty thought channel from a prompt
#pragma once

#include <cstddef>
#include <vector>

namespace flm {

/// \brief token ids of `<|channel>thought\n<channel|>`
inline constexpr int kGemma4ChannelBegin = 100;
inline constexpr int kGemma4ThoughtToken = 45518;
inline constexpr int kGemma4NewlineToken = 107;
inline constexpr int kGemma4ChannelEnd = 101;

/// \brief whether `tokens` ends with the empty thought channel
inline bool gemma4_ends_with_empty_thought(const std::vector<int>& tokens) {
    if (tokens.size() < 4) return false;
    const std::size_t n = tokens.size();
    return tokens[n - 4] == kGemma4ChannelBegin && tokens[n - 3] == kGemma4ThoughtToken &&
           tokens[n - 2] == kGemma4NewlineToken && tokens[n - 1] == kGemma4ChannelEnd;
}

/// \brief thinking is off and this is a new turn, so generate() opens the channel
inline bool gemma4_feed_empty_thought(bool enable_think, bool continue_final_message) {
    return !enable_think && !continue_final_message;
}

/// \brief tokens insert() removes so generate() can feed the channel once
/// \note A prompt that does not end with the channel, such as a tool result,
///       keeps every token. generate() still opens the channel after it.
inline std::size_t gemma4_empty_thought_trim(bool enable_think, bool continue_final_message,
                                             const std::vector<int>& tokens) {
    if (!gemma4_feed_empty_thought(enable_think, continue_final_message)) return 0;
    return gemma4_ends_with_empty_thought(tokens) ? 4 : 0;
}

}  // namespace flm
