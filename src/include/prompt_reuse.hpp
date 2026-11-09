/// \file prompt_reuse.hpp
/// \brief How much of a new prompt the engine already holds, decided once for the prompt's tokens and its media
/// \note _shared_insert reuses the engine's context when the token history is a prefix of the new prompt: it skips
///       those tokens and prefills the rest. A model with images or audio has to drop the items inside that prefix
///       from its payload before the call, so it has to know the skip in advance and agree with _shared_insert on
///       it exactly: a disagreement prefills image rows with no pixels (the engine refuses) or splices one image's
///       pixels into another's rows. Both sides ask plan_prompt_reuse() here, so there is one rule.
///       No model and no engine in here: CPU-testable (src/test/prompt_reuse).
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

/// \brief one image or audio clip of a prompt: the tokens it was expanded into
struct prompt_media_t {
    static constexpr int kImage = 0;
    static constexpr int kAudio = 1;
    static constexpr int kKinds = 2;
    int kind = kImage;     ///< which payload list it sits in; each list is dropped from the front
    size_t begin = 0;      ///< its first token in the prompt
    size_t end = 0;        ///< one past its last token
    uint64_t hash = 0;     ///< content hash (see prefix_image_hashes), 0 when the model does not hash this kind
};

/// \brief what plan_prompt_reuse() decided
struct prompt_reuse_t {
    enum reason_t {
        kNoHistory,     ///< the engine holds nothing: prefill everything
        kPrefix,        ///< the history is a strict prefix of the prompt: skip it
        kDiverged,      ///< the history is not a prefix of the prompt
        kWholePrompt,   ///< the history covers the whole prompt: sampling needs the last token's logits, read it all in again
        kInsideMedia,   ///< the history ends inside an item's tokens: that item's rows cannot be split, read it all in again
        kMediaChanged,  ///< an item inside the history has other content than the one the engine read (same size, other picture)
    };
    reason_t reason = kNoHistory;
    size_t skip = 0;                                   ///< leading tokens of the prompt the engine keeps
    bool clear = false;                                ///< the engine's context has to be dropped first (skip is then 0)
    std::array<size_t, prompt_media_t::kKinds> skipped_media{};  ///< per kind: leading items wholly inside the skip
    size_t media_end = 0;                              ///< one past the last token of the items that are prefilled, relative to the skip (0: none)
    std::array<size_t, prompt_media_t::kKinds> media_end_by_kind{};  ///< the same, per kind
};

/// \brief decide how much of `tokens` the engine keeps, given that it holds `history`
/// \param history what the engine holds (AutoModel::token_history, after any restore or slot resume)
/// \param tokens the prompt exactly as it will be prefilled (generation-prompt tokens a model feeds itself already cut)
/// \param media every image and audio clip of `tokens`, in prompt order per kind
/// \param history_media the items of `history` as they were read in, with their hashes; nullptr when they were checked
///        already (a prefix slot compares image hashes itself when it resumes)
/// \note The tokens of an image only depend on its size, so the tokens alone cannot tell a picture from another one of the
///       same size: an item inside the history is only kept when its hash is the one the engine read. An item with hash 0
///       is not checked.
inline prompt_reuse_t plan_prompt_reuse(const std::vector<int>& history, const std::vector<int>& tokens,
                                        const std::vector<prompt_media_t>& media = {},
                                        const std::vector<prompt_media_t>* history_media = nullptr) {
    prompt_reuse_t plan;
    if (history.empty()) {
        plan.reason = prompt_reuse_t::kNoHistory;
    } else {
        const size_t limit = std::min(history.size(), tokens.size());
        const size_t matched = static_cast<size_t>(
            std::mismatch(history.begin(), history.begin() + limit, tokens.begin()).first - history.begin());
        if (matched != history.size()) {
            plan.reason = prompt_reuse_t::kDiverged;
            plan.clear = true;
        } else if (matched >= tokens.size()) {
            plan.reason = prompt_reuse_t::kWholePrompt;
            plan.clear = true;
        } else {
            plan.reason = prompt_reuse_t::kPrefix;
            plan.skip = matched;
            for (const prompt_media_t& m : media) {
                if (m.begin < plan.skip && m.end > plan.skip) {
                    plan.reason = prompt_reuse_t::kInsideMedia;
                    plan.clear = true;
                    plan.skip = 0;
                    break;
                }
            }
            for (const prompt_media_t& m : media) {
                if (history_media == nullptr || plan.skip == 0 || m.end > plan.skip || m.hash == 0) continue;
                const bool same = std::any_of(history_media->begin(), history_media->end(), [&m](const prompt_media_t& h) {
                    return h.kind == m.kind && h.begin == m.begin && h.end == m.end && h.hash == m.hash;
                });
                if (!same) {
                    plan.reason = prompt_reuse_t::kMediaChanged;
                    plan.clear = true;
                    plan.skip = 0;
                    break;
                }
            }
        }
    }
    for (const prompt_media_t& m : media) {
        if (m.end <= plan.skip) {
            plan.skipped_media[m.kind]++;
        } else if (m.end > m.begin) {   // an item with no tokens has no rows to fit in the first chunk
            plan.media_end_by_kind[m.kind] = std::max(plan.media_end_by_kind[m.kind], m.end - plan.skip);
            plan.media_end = std::max(plan.media_end, m.end - plan.skip);
        }
    }
    return plan;
}

/// \brief the items of `tokens` whose tokens are runs of `token_id`, `counts[i]` tokens for item i, in order
/// \note An item may span several runs (a picture and its slices); its tokens are the next counts[i] `token_id`
///       tokens. When the prompt has fewer such tokens than the counts add up to, the last items end at its end.
inline void prompt_media_from_counts(const std::vector<int>& tokens, int token_id, const std::vector<size_t>& counts,
                                     int kind, std::vector<prompt_media_t>& out) {
    size_t pos = 0;
    for (size_t count : counts) {
        while (pos < tokens.size() && tokens[pos] != token_id) pos++;
        prompt_media_t m;
        m.kind = kind;
        m.begin = pos;
        size_t seen = 0;
        size_t last = pos;
        while (pos < tokens.size() && seen < count) {
            if (tokens[pos] == token_id) {
                seen++;
                last = pos;
            }
            pos++;
        }
        m.end = seen > 0 ? last + 1 : m.begin;
        out.push_back(m);
    }
}

/// \brief the items of `tokens` that sit between an `open_id` and a `close_id` token (both included), in order
inline void prompt_media_from_brackets(const std::vector<int>& tokens, int open_id, int close_id, int kind,
                                       std::vector<prompt_media_t>& out) {
    bool open = false;
    prompt_media_t m;
    m.kind = kind;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == open_id && !open) {
            open = true;
            m.begin = i;
        } else if (tokens[i] == close_id && open) {
            open = false;
            m.end = i + 1;
            out.push_back(m);
        }
    }
}
