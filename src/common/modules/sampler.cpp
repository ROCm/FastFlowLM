/// \file sampler.cpp
/// \brief sampler class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This is a header file for the sampler class
#pragma once

#include "modules/sampler.hpp"

#include <array>
#include <cstring>
#include <numeric>
#include <deque>
#include <algorithm>  // for std::sort
#include <chrono>
#include <cmath>      // for std::exp
#include <limits>

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
    this->min_p                 = config.min_p;
    this->total_tokens          = 0;

    this->rep_penalty           = config.rep_penalty;
    this->freq_penalty          = config.freq_penalty;
    this->pre_penalty           = config.pre_penalty;
    this->rep_penalty_window    = config.rep_penalty_window;
    this->freq_penalty_window   = config.freq_penalty_window;
    this->repeat_last_n        = config.repeat_last_n;
    this->use_optimized_sampling = config.use_optimized_sampling;

    this->token_history.clear();

    if (config.has_rng_seed || config.rng_seed != 0) {
        rng_.seed(config.rng_seed);
    } else {
        auto seed = std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count();
        rng_.seed(static_cast<uint64_t>(seed));
    }
    uniform_dist_ = std::uniform_real_distribution<float>(0.0f, 1.0f);
}

void Sampler::set_seed(uint64_t seed) {
    rng_.seed(seed);
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
    this->token_counts_sparse.clear();    
    this->total_tokens = 0;
    this->token_history.clear();
}

void Sampler::softmax_inplace() {
    if (this->top_k_logits.empty()) 
        return;

    float max_l = this->top_k_logits[0].logits;
    for (int i = 1; i < this->top_k_logits.size(); ++i) {
        if (this->top_k_logits[i].logits > max_l) {
            max_l = this->top_k_logits[i].logits;
        }
    }

    double sum = 0.0;
    for (auto& kv : this->top_k_logits) {
        double x = std::exp(double(kv.logits - max_l));
        kv.prob = float(x); 
        sum += x;
    }

    float inv_sum = 1.0f / float(sum);
    for (auto& kv : this->top_k_logits) {
        kv.prob *= inv_sum;
    }
}

void Sampler::softmax_with_topp_minp(float top_p_threshold, float min_p_threshold) {
    if (this->top_k_logits.empty())
        return;

    // Find max for numerical stability
    float max_logit = this->top_k_logits[0].logits;
    for (int i = 1; i < this->top_k_logits.size(); ++i) {
        if (this->top_k_logits[i].logits > max_logit) {
            max_logit = this->top_k_logits[i].logits;
        }
    }

    // Compute exp(x - max) and accumulate sum
    double sum = 0.0;
    for (auto& kv : this->top_k_logits) {
        double x = std::exp(double(kv.logits - max_logit));
        kv.prob = float(x);
        sum += x;
    }

    // Normalize
    float inv_sum = 1.0f / float(sum);
    for (auto& kv : this->top_k_logits) {
        kv.prob *= inv_sum;
    }

    // Apply top_p filter (cumulative probability threshold)
    if (top_p_threshold < 1.0f) {
        float cum = 0.0f;
        int last_idx = 0;
        for (int i = 0; i < this->top_k_logits.size(); ++i) {
            cum += this->top_k_logits[i].prob;
            last_idx = i + 1;
            if (cum >= top_p_threshold) {
                break;
            }
        }
        if (last_idx < this->top_k_logits.size()) {
            this->top_k_logits.resize(last_idx);
        }
    }

    // Apply min_p filter using the same logit criterion as sampler_minp_apply().
    if (min_p_threshold > 0.0f && min_p_threshold <= 1.0f) {
        float min_logit_threshold = max_logit + std::log(min_p_threshold);

        int valid_count = 0;
        for (int i = 0; i < this->top_k_logits.size(); ++i) {
            if (this->top_k_logits[i].logits >= min_logit_threshold) {
                this->top_k_logits[valid_count] = this->top_k_logits[i];
                valid_count++;
            }
        }
        this->top_k_logits.resize(valid_count);
    }

    // renormalize after filtering
    if (top_p_threshold < 1.0f || (min_p_threshold > 0.0f && min_p_threshold <= 1.0f)) {
        double filtered_sum = 0.0;
        for (auto& kv : this->top_k_logits) {
            filtered_sum += kv.prob;
        }
        
        if (filtered_sum > 0.0) {
            float filtered_inv_sum = 1.0f / float(filtered_sum);
            for (auto& kv : this->top_k_logits) {
                kv.prob *= filtered_inv_sum;
            }
        }
    }
}

void Sampler::sampler_penalty_apply() {
    if ((this->repeat_last_n == 0) ||
        (this->rep_penalty == 1.0f && this->freq_penalty == 0.0f && this->pre_penalty == 0.0f)) {
        return;
    }

    // Apply frequency and presence penalties
    for (int token_id = 0; token_id < in_features; token_id++) {
        int count = this->counters[token_id];
        if (count <= 0) {
            continue; 
        }
        assert(count > 0 && count <= this->repeat_last_n);

        float logit = this->logits[token_id];

        if (this->logits[token_id] <= 0.0f) {
            this->logits[token_id] *= this->rep_penalty;
        }
        else {
            this->logits[token_id] /= this->rep_penalty;
        }

        this->logits[token_id] -= (float(count) * this->freq_penalty + float(count > 0) * this->pre_penalty);
    }
}

void Sampler::sampler_penalty_apply_sparse() {
    if ((this->repeat_last_n == 0) ||
        (this->rep_penalty == 1.0f && this->freq_penalty == 0.0f && this->pre_penalty == 0.0f)) {
        return;
    }

    for (auto const& [token_id, count] : this->token_counts_sparse) {
        if (count <= 0) continue;

        if (this->logits[token_id] <= 0.0f) {
            this->logits[token_id] *= this->rep_penalty;
        } else {
            this->logits[token_id] /= this->rep_penalty;
        }
        
        this->logits[token_id] -= (float(count) * this->freq_penalty + 
                                  float(count > 0) * this->pre_penalty);
    }
}

void Sampler::sampler_topk_apply(int k) {
    k = std::min(k, in_features);
    if (k <= 0) {
        return;
    }

    // bf16 logits tie exactly often enough to matter over a 100k+ vocab, so a
    // bare `>` would leave the winner of a tie to the selection's internals.
    // Lowest id wins instead: the order is reproducible and matches
    // sample_greedy()'s argmax.
    const auto ranks_before = [](const logits_t& a, const logits_t& b) {
        if (a.logits != b.logits) { return a.logits > b.logits; }
        return a.token_id < b.token_id;
    };

    // One pass with a k-entry heap whose front is the worst survivor, instead of
    // materialising and partially sorting a vocab-sized array every token.
    auto& top = this->top_k_logits;
    top.clear();
    top.reserve(k);
    for (int i = 0; i < k; i++) {
        top.push_back(logits_t{ this->logits[i], i, 0.0f });
    }
    std::make_heap(top.begin(), top.end(), ranks_before);
    for (int i = k; i < in_features; i++) {
        // ids only grow, so a logit equal to the worst survivor's never displaces it
        if (this->logits[i] > top.front().logits) {
            std::pop_heap(top.begin(), top.end(), ranks_before);
            top.back() = logits_t{ this->logits[i], i, 0.0f };
            std::push_heap(top.begin(), top.end(), ranks_before);
        }
    }
    std::sort_heap(top.begin(), top.end(), ranks_before);
}

void Sampler::sampler_topp_apply(float p) {
    if (p >= 1.0f) {
        return;
    }

    float cum = 0.0f;
    int last_idx = top_k_logits.size();
    for (int i = 0; i < top_k_logits.size(); ++i) {
        cum += top_k_logits[i].prob;
        if (cum >= p) {
            last_idx = i + 1;
            break;
        }
    }
    if (last_idx < top_k_logits.size()) {
        top_k_logits.resize(last_idx);
    }
}

void Sampler::sampler_minp_apply(float p) {
    if (p <= 0.0f || p > 1) return;

    float max_logit = this->top_k_logits[0].logits;

    // logit >= max_logit + log(p)
    float min_logit = max_logit + std::log(p);
    for (int i = 0; i < this->top_k_logits.size(); i++) {
        if (this->top_k_logits[i].logits < min_logit) {
            this->top_k_logits.resize(i);
            break;
        }
    }
}

void Sampler::sampler_temp_apply(float temp) {
    if (temp == 0) { // Greedy sampling
        for (int i = 1; i < this->top_k_logits.size(); i++) {
            this->top_k_logits[i].logits = -std::numeric_limits<float>::infinity();
        }
        return;
    }

    for (int i = 0; i < this->top_k_logits.size(); i++) {
        this->top_k_logits[i].logits /= temp;
    }
}

int Sampler::sample_from_probs() {
    float u = this->uniform_dist_(this->rng_);
    
    float cdf = 0.0f;
    for (int i = 0; i < this->top_k_logits.size(); i++) {
        cdf += this->top_k_logits[i].prob;
        if (u <= cdf) {
            return this->top_k_logits[i].token_id;
        }
    }
    return this->top_k_logits.back().token_id;
}

void Sampler::ring_buffer_update(int sampled_index) {
    // UPDATE RING BUFFER (token_history), COUNTERS, POSITIONS, total_tokens
    if (this->repeat_last_n > 0) {
        // Push new token and update its counter
        this->token_history.push_back(sampled_index);
        this->counters[sampled_index]++;

        // If buffer exceeds window, pop oldest and decrement its counter
        if (this->token_history.size() > this->repeat_last_n) {
            int oldest = this->token_history.front();
            this->token_history.pop_front();
            this->counters[oldest]--;
        }
    }

    // Update last‐seen position for repetition penalty
    this->token_positions[sampled_index] = this->total_tokens;

    // Advance global token count
    this->total_tokens++;
}

void Sampler::ring_buffer_update_sparse(int sampled_index) {
    if (this->repeat_last_n > 0) {
        this->token_history.push_back(sampled_index);

        // Keep legacy counters synchronized for compatibility with existing callers.
        this->counters[sampled_index]++;
        this->token_counts_sparse[sampled_index]++;

        if (this->token_history.size() > this->repeat_last_n) {
            int oldest = this->token_history.front();
            this->token_history.pop_front();

            this->counters[oldest]--;
            if (--this->token_counts_sparse[oldest] <= 0) {
                this->token_counts_sparse.erase(oldest);
            }
        }
    }

    this->token_positions[sampled_index] = this->total_tokens;
    this->total_tokens++;
}

namespace {
/// exp() for every value a bf16 can hold.
///
/// The logits arrive as bf16 from the engine and the widening to fp32 is
/// exact, so this->logits can only take 65536 distinct values. Looking the
/// exponential up is the same number std::exp would return -- precomputed,
/// not approximated -- and it turns a quarter of a million exp() calls into as
/// many reads. 512 KB, which fits in L2.
const std::array<double, 65536>& bf16_exp_table() {
    static const std::array<double, 65536> table = [] {
        std::array<double, 65536> t{};
        for (uint32_t b = 0; b < 65536; b++) {
            const uint32_t bits = b << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            t[b] = std::exp(static_cast<double>(f));
        }
        return t;
    }();
    return table;
}
}  // namespace

float Sampler::log_sum_exp_all() const {
    if (this->logits.empty()) {
        return 0.0f;
    }
    const auto& table = bf16_exp_table();
    double sum = 0.0;
    for (int i = 0; i < this->in_features; i++) {
        uint32_t bits;
        std::memcpy(&bits, &this->logits[i], sizeof(bits));
        // The low 16 bits are non-zero for anything the penalties moved off
        // the bf16 grid; those take the exact call. The branch predicts well
        // and costs nothing measurable.
        sum += (bits & 0xFFFFu) != 0u
            ? std::exp(static_cast<double>(this->logits[i]))
            : table[bits >> 16];
    }
    if (std::isfinite(sum) && sum > 0.0) {
        return static_cast<float>(std::log(sum));
    }
    // Nothing on the bf16 grid produced a usable sum -- reachable only for a
    // logit beyond about +-700, where exp overflows a double. Fall back to the
    // shifted form. Note what it does NOT cover: +-inf and NaN are not
    // meaningful here either way, and a logit of +inf would come back as 0.
    float max_logit = -std::numeric_limits<float>::infinity();
    for (float v : this->logits) {
        if (v > max_logit) {
            max_logit = v;
        }
    }
    if (!std::isfinite(max_logit)) {
        return 0.0f;
    }
    double shifted = 0.0;
    for (float v : this->logits) {
        shifted += std::exp(static_cast<double>(v) - max_logit);
    }
    return max_logit + static_cast<float>(std::log(shifted));
}

void Sampler::capture_if_requested() {
    if (!this->capture_top_logprobs) {
        return;
    }
    // Own top-N by logit, taken from the full logits rather than from
    // top_k_logits: the report should not depend on the sampler's top_k, and
    // the greedy path keeps only one entry there.
    const int n = std::min(this->capture_count, this->in_features);
    this->captured_top_k.clear();
    this->captured_logsumexp = this->log_sum_exp_all();
    if (n > 0 && static_cast<size_t>(n) <= this->top_k_logits.size()) {
        // top_k_logits is already sorted by logit with the same tie-break, so
        // its beginning IS the top-n. No second sort, no vocab-sized index
        // vector -- which matters because this runs once per token.
        this->captured_top_k.assign(this->top_k_logits.begin(),
                                    this->top_k_logits.begin() + n);
    } else if (n > 0) {
        std::vector<int> idx(this->in_features);
        std::iota(idx.begin(), idx.end(), 0);
        // Same order as sampler_topk_apply: bf16 logits over 100k+ entries
        // are often equal, partial_sort is not stable, and a tie broken
        // differently would put another token at the top than the one the
        // draw would have taken.
        std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                          [this](int a, int b) {
                              if (this->logits[a] != this->logits[b]) {
                                  return this->logits[a] > this->logits[b];
                              }
                              return a < b;
                          });
        this->captured_top_k.reserve(n);
        for (int i = 0; i < n; i++) {
            this->captured_top_k.push_back(logits_t{ this->logits[idx[i]], idx[i], 0.0f });
        }
    }
    // Stays armed: every token that reaches the caller's text needs its own
    // distribution. `captured` marks that THIS draw has one.
    this->captured = true;
}

/// \brief Sample the token
/// \param x the input buffer
/// \return the sampled token
int Sampler::sample_greedy(buffer<bf16>& x, bool record_in_history) {
    const bool penalties_active =
        (this->repeat_last_n != 0) &&
        (this->rep_penalty != 1.0f || this->freq_penalty != 0.0f || this->pre_penalty != 0.0f);

    float best = -std::numeric_limits<float>::infinity();
    int best_id = 0;

    if (penalties_active) {
        // penalties reorder the logits, so they have to land before the argmax.
        // that needs the fp32 copy, but nothing past it does.
        #if USEAVX2
        const int simd_width = 8;
        int i = 0;
        for (; i <= in_features - simd_width; i += simd_width) {
            __m128i bf16_vals_x = _mm_loadu_si128((__m128i*)&x[i]);
            __m256 fp32_vals_x = bf16o_fp32(bf16_vals_x);
            _mm256_storeu_ps(&this->logits[i], fp32_vals_x);
        }
        for (; i < in_features; i++) {
            this->logits[i] = x[i];
        }
        #else
        for (int i = 0; i < in_features; i++) {
            this->logits[i] = x[i];
        }
        #endif
        sampler_penalty_apply_sparse();
        for (int i = 0; i < in_features; i++) {
            if (this->logits[i] > best) {
                best = this->logits[i];
                best_id = i;
            }
        }
    }
    else {
        // Nothing rewrites the logits, so scan them where they lie: no
        // vocab-sized fp32 buffer is written at all. The logits come back in
        // device-mapped memory where element-at-a-time reads are punishing, so
        // the scan pulls 8 at a time and keeps the running argmax in lanes.
        #if USEAVX2
        const int simd_width = 8;
        int i = 0;
        __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        __m256i vidx = _mm256_set1_epi32(0);
        __m256i vcur = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i vstep = _mm256_set1_epi32(simd_width);
        for (; i <= in_features - simd_width; i += simd_width) {
            __m256 v = bf16o_fp32(_mm_loadu_si128((__m128i*)&x[i]));
            __m256 gt = _mm256_cmp_ps(v, vmax, _CMP_GT_OQ);
            vmax = _mm256_blendv_ps(vmax, v, gt);
            vidx = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(vidx),
                                                        _mm256_castsi256_ps(vcur), gt));
            vcur = _mm256_add_epi32(vcur, vstep);
        }
        alignas(32) float lane_max[simd_width];
        alignas(32) int lane_idx[simd_width];
        _mm256_store_ps(lane_max, vmax);
        _mm256_store_si256((__m256i*)lane_idx, vidx);
        for (int j = 0; j < simd_width; j++) {
            // strictly greater keeps the lowest index on a tie, in both loops
            if (lane_max[j] > best) {
                best = lane_max[j];
                best_id = lane_idx[j];
            }
        }
        for (; i < in_features; i++) {
            float v = x[i];
            if (v > best) {
                best = v;
                best_id = i;
            }
        }
        #else
        for (int i = 0; i < in_features; i++) {
            float v = x[i];
            if (v > best) {
                best = v;
                best_id = i;
            }
        }
        #endif
    }

    // callers that inspect the surviving candidates still get a consistent view
    this->top_k_logits.assign(1, logits_t{ best, best_id, 1.0f });
    if (this->capture_top_logprobs) {
        // log_sum_exp_all() reads this->logits, which the branch above only
        // fills when penalties are active. Fill it here the same way sample()
        // does: the logits come back in device-mapped memory where
        // element-at-a-time reads are punishing, so this pulls eight at a time.
        if (!penalties_active) {
            #if USEAVX2
            const int simd_width = 8;
            int i = 0;
            for (; i <= in_features - simd_width; i += simd_width) {
                __m128i bf16_vals_x = _mm_loadu_si128((__m128i*)&x[i]);
                _mm256_storeu_ps(&this->logits[i], bf16o_fp32(bf16_vals_x));
            }
            for (; i < in_features; i++) {
                this->logits[i] = x[i];
            }
            #else
            for (int i = 0; i < in_features; i++) {
                this->logits[i] = x[i];
            }
            #endif
        }
        this->capture_if_requested();
    }
    if (record_in_history) {
        ring_buffer_update_sparse(best_id);
    }
    return best_id;
}

int Sampler::sample(buffer<bf16>& x, bool record_in_history) {
    // PRNG is seeded once at construction, no per-call seeding needed
    if (this->top_k == 1) {
        return sample_greedy(x, record_in_history);
    }

    // COPY FROM `x` → `this->logits[]`
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
        this->logits[i] = x[i];
    }
    #else
    for (int i = 0; i < in_features; i++) {
        this->logits[i] = x[i];
    }
    #endif

    sampler_penalty_apply_sparse();
    sampler_topk_apply(this->top_k);

    this->capture_if_requested();
    if (this->use_optimized_sampling) {
        softmax_with_topp_minp(this->top_p, this->min_p);
        sampler_temp_apply(this->temperature);
        softmax_inplace();
    } else {
        // Legacy behavior is kept as the default for output compatibility.
        softmax_inplace();
        sampler_topp_apply(this->top_p);
        softmax_inplace();
        sampler_minp_apply(this->min_p);
        sampler_temp_apply(this->temperature);
        softmax_inplace();
    }

    int sampled_index = sample_from_probs();
    if (record_in_history) {
        ring_buffer_update_sparse(sampled_index);
    }

    return sampled_index;
}