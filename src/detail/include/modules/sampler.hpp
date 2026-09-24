/// \file sampler.hpp
/// \brief sampler class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.10
/// \note This class is used to sample the tokens.
#pragma once

#include "typedef.hpp"
#include <deque>

/// \brief sampler config
/// \param temperature the temperature
/// \param top_k the top k
/// \param top_p the top p
/// \param rep_penalty the rep penalty
/// \param freq_penalty the freq penalty
/// \param rep_penalty_window the rep penalty window
typedef struct sampler_config_{
    float temperature = 1.0f;
    int top_k = 5;
    float top_p = 0.9f;
    float rep_penalty = 1.0f;
    float freq_penalty = 1.0f;
    int rep_penalty_window = 1024;
    int freq_penalty_window = 1024;  // Window size for frequency penalty
} sampler_config;

typedef std::pair<float, int> logits_t;
typedef std::vector<logits_t> logits_list_t;

class Sampler{
public:
    std::vector<float> logits;
    int in_features;
    std::vector<int> counters;
    logits_list_t top_k_logits;
    float rep_penalty;
    float freq_penalty;
    float temperature;
    int top_k;
    float top_p;
    int total_tokens;
    std::vector<int> token_positions;
    
    // Ring buffer for frequency tracking
    std::deque<int> token_history;
    size_t freq_penalty_window;
    size_t rep_penalty_window;

    /// \brief Constructor
    /// \param in_features the input features
    /// \param config the configuration
    Sampler(){};
    Sampler(int in_features, sampler_config& config);

    /// \brief Reset the penalties
    /// \note The function will reset the penalties
    /// \note The function will reset the token positions
    /// \note The function will reset the token history
    /// \note The function will reset the total tokens
    /// \note The function will reset the token positions
    void reset_penalties();

    /// \brief Sample the token
    /// \param x the input buffer
    /// \return the sampled token
    int sample(buffer<bf16>& x);
};