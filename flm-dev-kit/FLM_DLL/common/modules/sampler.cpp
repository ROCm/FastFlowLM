// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file sampler.cpp
/// \brief sampler class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.10
/// \note This is a header file for the sampler class
#pragma once

#include "modules/sampler.hpp"

#include <deque>
#include <cstdlib>    // for std::rand, std::srand, RAND_MAX
#include <ctime>      // for std::time
#include <algorithm>  // for std::sort
#include <cmath>      // for std::exp

/// \brief Constructor
/// \param in_features the input features
/// \param config the configuration
Sampler::Sampler(int in_features, sampler_config& config) {
    this->in_features           = in_features;
    this->logits.resize(in_features);
    this->counters.resize(in_features, 0);
    this->token_positions.resize(in_features, -1);
    this->top_k_logits.resize(config.top_k);

    this->temperature           = config.temperature;
    this->top_k                 = config.top_k;
    this->top_p                 = config.top_p;
    this->total_tokens          = 0;

    this->rep_penalty           = config.rep_penalty;
    this->freq_penalty          = config.freq_penalty;
    this->rep_penalty_window    = config.rep_penalty_window;
    this->freq_penalty_window   = config.freq_penalty_window;

    this->token_history.clear();
}

/// \brief Reset the penalties
/// \note The function will reset the penalties
/// \note The function will reset the token positions
/// \note The function will reset the token history
/// \note The function will reset the total tokens
/// \note The function will reset the token positions
void Sampler::reset_penalties() {
    for (int i = 0; i < in_features; i++) {
        this->counters[i]            = 0;
        this->token_positions[i]     = -1;
    }
    this->total_tokens = 0;
    this->token_history.clear();
}

/// \brief Sample the token
/// \param x the input buffer
/// \return the sampled token
int Sampler::sample(buffer<bf16>& x) {
    // Re‐seed the PRNG each call:
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    //
    // 1) COPY FROM `x` → `this->logits[]`
    //
    #if USEAVX2
    const int simd_width = 8;
    int i = 0;
    for (; i <= in_features - simd_width; i += simd_width) {
        __m128i bf16_vals_x = _mm_loadu_si128(
            (__m128i*)&x[i]
        );
        __m256 fp32_vals_x = bf16o_fp32(bf16_vals_x);
        _mm256_storeu_ps(&this->logits[i], fp32_vals_x);
    }
    for (; i < in_features; i++) {
        this->logits[i] = (float)x[i];
    }
    #else
    for (int i = 0; i < in_features; i++) {
        this->logits[i] = (float)x[i];
    }
    #endif

    //
    // 2) APPLY REPETITION + FREQUENCY PENALTIES
    //
    for (int token_id = 0; token_id < in_features; token_id++) {
        // --- (A) Repetition Penalty (sliding window = rep_penalty_window) ---
        int last_pos = this->token_positions[token_id]; // –1 if never seen
        if (last_pos >= 0 && this->rep_penalty_window > 0 && this->rep_penalty != 1.0f) {
            size_t distance = this->total_tokens - last_pos;
            if (distance < this->rep_penalty_window) {
                // Apply penalty based on sign: if logit < 0 then multiply, else divide
                if (this->logits[token_id] < 0.0f) {
                    this->logits[token_id] = this->logits[token_id] * this->rep_penalty;
                } else {
                    this->logits[token_id] = this->logits[token_id] / this->rep_penalty;
                }
            }
        }

        // --- (B) Frequency Penalty (sliding window = freq_penalty_window) ---
        if (this->freq_penalty_window > 0 && this->freq_penalty != 1.0f) {
            float normalized_freq = (float)this->counters[token_id] /
                                    (float)this->freq_penalty_window;
            // Apply frequency penalty: subtract penalty * frequency
            this->logits[token_id] =
                this->logits[token_id] - (this->freq_penalty - 1.0f) * normalized_freq;
        }
    }

    //
    // 3) TEMPERATURE NORMALIZATION
    //    Subtract max_logit and divide by temperature
    //
    float max_logit = -1e9f;
    for (int i = 0; i < in_features; i++) {
        if (this->logits[i] > max_logit) {
            max_logit = this->logits[i];
        }
    }

    #if USEAVX2
    {
        int i = 0;
        __m256 v_temp = _mm256_set1_ps(this->temperature);
        __m256 v_max  = _mm256_set1_ps(max_logit);
        for (; i <= in_features - simd_width; i += simd_width) {
            __m256 v_logits = _mm256_loadu_ps(&this->logits[i]);
            __m256 v_sub    = _mm256_sub_ps(v_logits, v_max);
            __m256 v_div    = _mm256_div_ps(v_sub, v_temp);
            _mm256_storeu_ps(&this->logits[i], v_div);
        }
        for (; i < in_features; i++) {
            this->logits[i] = (this->logits[i] - max_logit) / this->temperature;
        }
    }
    #else
    for (int i = 0; i < in_features; i++) {
        this->logits[i] = (this->logits[i] - max_logit) / this->temperature;
    }
    #endif

    //
    // 4) TOP‐K + TOP‐P FILTERING
    //
    // 4.1 Initialize top_k buffer to (−∞, 0)
    float min_logit = -1e9f;
    int   min_index = 0;
    this->top_k_logits.clear();
    this->top_k_logits.resize(this->top_k, std::make_pair(min_logit, min_index));

    // 4.2 Collect top_k highest logits
    for (int i = 0; i < in_features; i++) {
        if (this->logits[i] > min_logit) {
            this->top_k_logits[min_index] = std::make_pair(this->logits[i], i);
            // Re‐find the new minimum among top_k entries
            float new_min_val = this->top_k_logits[0].first;
            int   new_min_idx = 0;
            for (int j = 1; j < this->top_k; j++) {
                if (this->top_k_logits[j].first < new_min_val) {
                    new_min_val = this->top_k_logits[j].first;
                    new_min_idx = j;
                }
            }
            min_logit = new_min_val;
            min_index = new_min_idx;
        }
    }

    // 4.3 Sort top_k entries in descending order
    std::sort(
        this->top_k_logits.begin(),
        this->top_k_logits.end(),
        [](const logits_t& a, const logits_t& b) {
            return a.first > b.first;
        }
    );

    // 4.4 Convert logits→exp(logit) and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < this->top_k; i++) {
        this->top_k_logits[i].first = std::exp(this->top_k_logits[i].first);
        sum_exp += this->top_k_logits[i].first;
    }
    float inv_sum_exp = 1.0f / sum_exp;

    // 4.5 Find cutoff index for top_p
    float running_prob = 0.0f;
    int   top_p_index  = this->top_k - 1;
    for (int i = 0; i < this->top_k; i++) {
        running_prob += this->top_k_logits[i].first * inv_sum_exp;
        if (running_prob > this->top_p) {
            top_p_index = i;
            break;
        }
    }

    // 4.6 Renormalize only up to top_p_index
    float sum_exp_p = 0.0f;
    for (int i = 0; i <= top_p_index; i++) {
        sum_exp_p += this->top_k_logits[i].first;
    }
    float inv_sum_exp_p = 1.0f / sum_exp_p;
    for (int i = 0; i <= top_p_index; i++) {
        this->top_k_logits[i].first *= inv_sum_exp_p;
    }

    //
    // 5) SAMPLE ONE TOKEN FROM THE FINAL DISTRIBUTION
    //
    float u   = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    float cdf = 0.0f;
    int   sampled_index = this->top_k_logits[top_p_index].second;
    for (int i = 0; i <= top_p_index; i++) {
        cdf += this->top_k_logits[i].first;
        if (u < cdf) {
            sampled_index = this->top_k_logits[i].second;
            break;
        }
    }

    //
    // 6) UPDATE RING BUFFER (token_history), COUNTERS, POSITIONS, total_tokens
    //
    if (this->freq_penalty_window > 0) {
        // Push new token and update its counter
        this->token_history.push_back(sampled_index);
        this->counters[sampled_index]++;

        // If buffer exceeds window, pop oldest and decrement its counter
        if (this->token_history.size() > this->freq_penalty_window) {
            int oldest = this->token_history.front();
            this->token_history.pop_front();
            this->counters[oldest]--;
        }
    }

    // Update last‐seen position for repetition penalty
    this->token_positions[sampled_index] = this->total_tokens;

    // Advance global token count
    this->total_tokens++;

    return sampled_index;
}