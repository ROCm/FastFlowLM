/// \file sampler.hpp
/// \brief sampler class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This class is used to sample the tokens.
#pragma once

#include "typedef.hpp"
#include <cstdint>
#include <deque>
#include <random>
#include <unordered_map>

/// \brief sampler config
/// \param temperature the temperature
/// \param top_k the top k
/// \param top_p the top p
/// \param rep_penalty the rep penalty
/// \param freq_penalty the freq penalty
/// \param rep_penalty_window the rep penalty window
typedef struct sampler_config_{
    int top_k = 40;
    float top_p = 0.9f;
    float min_p = 0.1f;
    float temperature = 0.8f;
    float rep_penalty = 1.0f;
    float freq_penalty = 0.0f;
    float pre_penalty = 0.0f;
    int rep_penalty_window = 64;
    int freq_penalty_window = 64;  // Window size for frequency penalty
    int repeat_last_n = 64;
    bool use_optimized_sampling = true;
    bool has_rng_seed = false;
    uint64_t rng_seed = 0;
} sampler_config;

//typedef std::pair<float, int> logits_t;

typedef struct {
    float logits;
    int token_id;
    float prob;
} logits_t;

typedef std::vector<logits_t> logits_list_t;

class Sampler{
public:
    std::vector<float> logits;
    int in_features;
    std::vector<int> counters;
    logits_list_t top_k_logits;

    /// Set by the model for a request that asked for logprobs, and cleared at
    /// the start of every request whether it asked or not. It stays set for
    /// the whole request: every token that reaches the caller's text needs its
    /// own distribution. A request that does not ask never arms it, so it pays
    /// nothing.
    bool capture_top_logprobs = false;
    /// How many entries to keep. Bounded by the caller, OpenAI allows 20.
    int capture_count = 0;

    /// The top entries by logit at the moment of the capture, and the
    /// log-sum-exp of the WHOLE vocabulary at the same moment.
    ///
    /// Sorted by logit and cut to the count the caller asked for, so it is NOT
    /// the list the draw used and the drawn token need not be in it: with
    /// top_logprobs: 1 and a draw at rank 3 it is not. That is what `chosen`
    /// is for. Entries behind the top-p and min-p cut are listed although they
    /// had no chance of being drawn.
    logits_list_t captured_top_k;
    /// Set by the capture, cleared when a request arms it. Guards against
    /// reading values that belong to an earlier draw or request.
    bool captured = false;
    float captured_logsumexp = 0.0f;
    float rep_penalty;
    float freq_penalty;
    float pre_penalty;
    int top_k;
    float top_p;
    float min_p;
    float temperature;
    int total_tokens;
    std::vector<int> token_positions;
    
    // Ring buffer for frequency tracking
    std::deque<int> token_history;
    size_t freq_penalty_window;
    size_t rep_penalty_window;
    size_t repeat_last_n;
    
    std::unordered_map<int, int> token_counts_sparse;

    std::mt19937_64 rng_{};
    std::uniform_real_distribution<float> uniform_dist_{0.0f, 1.0f};
    bool use_optimized_sampling = true;

    /// \brief Constructor
    /// \param in_features the input features
    /// \param config the configuration
    Sampler() = default;
    Sampler(int in_features, sampler_config& config);
    void set_seed(uint64_t seed);

    /// \brief Reset the penalties
    /// \note The function will reset the penalties
    /// \note The function will reset the token positions
    /// \note The function will reset the token history
    /// \note The function will reset the total tokens
    /// \note The function will reset the token positions
    void reset_penalties();

    void softmax_inplace();
    void softmax_with_topp_minp(float top_p_threshold, float min_p_threshold);
    
    void sampler_penalty_apply();
    void sampler_penalty_apply_sparse();
    
    /// \brief argmax shortcut taken when top_k == 1
    /// \param x the raw logits
    /// \return the winning token id
    /// \note softmax, top-p, min-p and temperature are all order preserving, so
    ///       with k == 1 the winner is just the argmax of the (penalised) logits.
    ///       Going through the general path would allocate and partial_sort a
    ///       vocab-sized array of triples and run three full-vocab softmaxes for
    ///       a result that never depends on any of it.
    /// \param record_in_history whether the drawn token joins the repetition
    ///        and frequency windows
    /// \note Pass false for a draw the caller throws away. Such a token was
    ///       never generated text, and counting it pushes the tokens that
    ///       follow it down -- by count * freq_penalty + pre_penalty each time
    ///       it occurs.
    int sample_greedy(buffer<bf16>& x, const std::vector<uint32_t>* allowed = nullptr, bool record_in_history = true);

    /// \brief Capture the current top-k if a request asked for it
    /// \note Called by the sampling paths right after the top-k is settled and
    ///       before the temperature. Stays armed until the next request
    ///       disarms it; `captured` marks that this draw has values.
    void capture_if_requested();
    /// \brief log-sum-exp over the whole vocabulary
    /// \note Reads this->logits, which the sampling paths have filled by then
    float log_sum_exp_all() const;

    void sampler_topk_apply(int k);
    void sampler_topp_apply(float p);
    void sampler_minp_apply(float p);
    void sampler_temp_apply(float temp);
    int sample_from_probs();
    void ring_buffer_update(int sampled_index);
    void ring_buffer_update_sparse(int sampled_index);
    
    /// \brief Sample the token
    /// \param x the input buffer, only read
    /// \param allowed optional bitmask, bit i of word i/32 set when token i may be
    ///        sampled; applied to the sampler's own copy of the logits
    /// \return the sampled token
    /// \param record_in_history whether the drawn token joins the repetition
    ///        and frequency windows; see sample_greedy
    int sample(buffer<bf16>& x, const std::vector<uint32_t>* allowed = nullptr, bool record_in_history = true);

    /// \brief Push every token the bitmask does not allow to -inf in this->logits
    void sampler_allowed_apply(const std::vector<uint32_t>* allowed);
};