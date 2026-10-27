/// \file prefix_cache_usage.hpp
/// \brief the `usage.flm_prefix_cache` object: what the slot cache did for a request (a FLM extension, ignored by OpenAI clients)
#pragma once
#include <nlohmann/json.hpp>

#include "prefix_slot_cache.hpp"

inline nlohmann::ordered_json prefix_cache_usage_json(const prefix_cache_report_t& r) {
    using json = nlohmann::ordered_json;
    json out = {
        {"lookup", r.hit ? "hit" : "miss"},
        {"prompt_tokens", r.prompt_tokens},
    };
    if (r.hit) {
        out["slot"] = r.slot;
        out["slot_type"] = r.slot_boundary ? "boundary" : "end";
        out["matched_tokens"] = r.matched_tokens;
    } else {
        out["miss_reason"] = r.miss_reason;
    }
    json saved = json::array();
    for (const auto& s : r.saved) {
        saved.push_back({{"slot", s.slot}, {"slot_type", s.boundary ? "boundary" : "end"}, {"tokens", s.tokens}, {"kv_from", s.kv_from}, {"bytes", s.bytes}});
    }
    out["saved"] = saved;
    out["dropped"] = r.dropped;
    out["slots_used"] = r.slots_used;
    out["slots_max"] = r.slots_max;
    out["bytes_used"] = r.bytes_used;
    out["bytes_budget"] = r.bytes_budget;
    return out;
}
