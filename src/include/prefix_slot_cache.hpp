/// \file prefix_slot_cache.hpp
/// \brief Token-level prefix cache over a few saved engine states, shared by every model family that can save one
/// \note The cache owns the snapshots ("slots"): each one is an opaque flm_state (what the engine cannot rewind, plus the
///       attention K/V rows after a chosen position) read out of the engine with flm_state_api::state_read. It remembers
///       which tokens each was saved after, resumes a new prompt from the longest one that is a prefix of it
///       (flm_state_api::state_write), and decides what to throw away when memory or slots run out. The engine only ever
///       holds the context it is running: it has no slot numbers, no slot count and no cache policy.
///       It does not know any model: the model wrapper hands it tokens and image hashes, and the engine.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "flm_state_api.h"

/// \brief one saved engine context: owns an opaque flm_state, freed through the engine library that made it
class saved_state {
public:
    saved_state(const flm_state_api* api, flm_state* state) : api_(api), state_(state) {}
    saved_state(saved_state&& o) noexcept : api_(o.api_), state_(o.state_) { o.state_ = nullptr; }
    saved_state(const saved_state&) = delete;
    saved_state& operator=(const saved_state&) = delete;
    ~saved_state() { if (state_) api_->state_free(state_); }
    flm_state* get() const { return state_; }
    int context_length() const { return api_->state_context_length(state_); }
    int kv_from() const { return api_->state_kv_from(state_); }
    size_t estimate_bytes(int length, int from) const { return api_->state_estimate_bytes(state_, length, from); }
private:
    const flm_state_api* api_;
    flm_state* state_;
};

/// \brief 64-bit content hash of a byte range, continuing from `seed`
uint64_t prefix_hash_bytes(const void* data, size_t bytes, uint64_t seed);

/// \brief content hash of each image of a payload, in order, so a snapshot is only reused for the same pictures
/// \note Works with any vision payload that has `images[i].grid_h / grid_w` and the concatenated pixel buffer
///       `_data__processed` (patch * patch * 3 * temporal values per grid cell).
template <class Payload>
std::vector<uint64_t> prefix_image_hashes(const Payload& payload, size_t patch, size_t temporal) {
    std::vector<uint64_t> hashes;
    const auto& pixels = payload._data__processed;
    size_t offset = 0;
    for (const auto& img : payload.images) {
        const size_t count = static_cast<size_t>(img.grid_h) * patch * static_cast<size_t>(img.grid_w) * patch * 3u * temporal;
        uint64_t h = 0xcbf29ce484222325ull ^ (static_cast<uint64_t>(img.grid_h) << 32 | static_cast<uint64_t>(img.grid_w));
        if (offset + count <= pixels.size()) {
            h = prefix_hash_bytes(pixels.data() + offset, count * sizeof(pixels[0]), h);
        }
        hashes.push_back(h);
        offset += count;
    }
    return hashes;
}

/// \brief what the prefix cache did for one request: logged, and reported to the client in `usage.flm_prefix_cache`
struct prefix_cache_report_t {
    bool valid = false;           ///< the model has a slot cache and this request went through it
    bool hit = false;             ///< the request resumed from a slot
    std::string miss_reason;      ///< why not, when it did not
    int slot = -1;                ///< id of the slot resumed from (ids are never reused)
    bool slot_boundary = false;   ///< that slot is a boundary slot (system + tools head) rather than an end slot
    int matched_tokens = 0;       ///< tokens of the prompt it covered
    int prompt_tokens = 0;        ///< tokens the cache considered (the prompt without its generation prompt)
    struct saved_t {
        int slot = 0;
        bool boundary = false;
        int tokens = 0;           ///< prompt length the slot was saved after
        int kv_from = 0;          ///< first position whose K/V rows it holds
        size_t bytes = 0;
    };
    std::vector<saved_t> saved;   ///< snapshots this request added
    int dropped = 0;              ///< snapshots this request evicted
    int slots_used = 0;           ///< after the request
    int slots_max = 0;
    size_t bytes_used = 0;
    size_t bytes_budget = 0;
};

class PrefixSlotCache {
public:
    /// \brief most snapshots --prefix-slots may ask for
    static constexpr int kMaxSlots = 32;

    /// \brief the placeholder token id of images; slots count the image runs they contain
    int image_token_id = -1;
    /// \brief a shared head shorter than this is not worth a slot: re-processing it is cheap
    size_t min_boundary = 512;
    /// \brief an end snapshot must add at least this many tokens over a saved slot that covers its prompt
    size_t min_end_gain = 64;

    /// \brief --prefix-cache-mb / --prefix-slots (-1 = default; 0 slots = off)
    void configure(int cache_mb, int slots) {
        cache_mb_ = cache_mb;
        slots_ = slots;
    }
    /// \brief attach the engine and the state table its library exports (nullptr api detaches); slots of a former engine are forgotten
    /// \note The engine's library exports a flm_state_api (see flm_state_api.h); without one nothing is saved.
    void attach(void* engine, const flm_state_api* api);
    bool enabled() const { return engine_ != nullptr && api_ != nullptr && slots_ != 0; }

    /// \brief result of resume()
    struct resume_t {
        bool resumed = false;      ///< the engine now holds a slot's state
        size_t match_len = 0;      ///< tokens of the prompt it covers
        std::string miss_reason;   ///< when not resumed: why no slot could serve the prompt
    };
    /// \brief resume from the longest saved prefix of `full_tokens` and write that state back into the engine
    /// \param full_tokens what the prompt leaves in the engine
    /// \param image_hashes content hash of each image of the prompt, in order
    /// \param matched_tokens set to the tokens of the slot used (what the engine now holds)
    /// \note Only slots strictly shorter than the prompt count: one token must still be processed to get logits.
    /// \note On a restore failure every slot is forgotten and resumed is false.
    resume_t resume(const std::vector<int>& full_tokens, const std::vector<uint64_t>& image_hashes, std::vector<int>& matched_tokens);

    /// \brief read the engine's current context into a new slot; `tokens` is what the engine holds
    /// \param boundary true for the end of system + tools (kv_from 0, evicted last)
    /// \param kv_from first position whose K/V rows the slot saves (0 = self-contained)
    void save(const std::vector<int>& tokens, const std::vector<uint64_t>& image_hashes, bool boundary, int kv_from);

    /// \brief the kv_from an end-of-prompt slot should use: `boundary` when a self-contained slot holds the first
    ///        `boundary` tokens of `tokens`, else 0
    int end_kv_from(const std::vector<int>& tokens, size_t boundary) const;

    /// \brief tokens whose K/V rows the engine's cache holds now (what the last request left in it)
    void set_live(const std::vector<int>& tokens) { live_tokens_ = tokens; }
    /// \param announce log that the engine's context was dropped while the slots were kept (a clear or a cancel, not a cold start)
    void clear_live(bool announce = false);

    /// \brief what the cache did since begin_report(): the lookup, the snapshots added and dropped, and how full it is
    void begin_report() { report_ = prefix_cache_report_t(); report_.valid = true; }
    prefix_cache_report_t report() const;

    /// \brief forget every slot (and the live tokens)
    void forget_all();

    size_t slot_count() const { return slots_table_.size(); }

private:
    struct slot_t {
        std::unique_ptr<saved_state> state;  // what the engine held after `tokens`, with its length and kv_from
        std::vector<int> tokens;             // what the slot was saved after
        std::vector<uint64_t> image_hashes;  // content of the images inside `tokens`
        uint64_t last_used = 0;
        int id = 0;                          // told apart in logs and usage; never reused
        bool boundary = false;               // end of system + tools; evicted last
        /// first position whose K/V rows the slot holds
        int kv_from() const { return state->kv_from(); }
    };

    size_t bytes_used() const;
    size_t budget_bytes() const;
    size_t slot_limit() const;
    size_t count_image_runs(const std::vector<int>& tokens, size_t len) const;

    void* engine_ = nullptr;           // the engine (causal_lm*), opaque to the cache
    const flm_state_api* api_ = nullptr;
    std::vector<slot_t> slots_table_;
    std::vector<int> live_tokens_;
    uint64_t clock_ = 0;
    int cache_mb_ = -1;
    int slots_ = -1;
    bool announced_ = false;
    bool warned_too_big_ = false;
    int next_id_ = 1;
    prefix_cache_report_t report_;
};
