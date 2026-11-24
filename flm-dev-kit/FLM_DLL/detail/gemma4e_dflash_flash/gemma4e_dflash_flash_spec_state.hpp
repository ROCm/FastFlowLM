// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Shared private state of the DFlash speculation code (gemma4e_dflash_flash_spec.cpp, gemma4e_dflash_flash_verify.cpp).
#pragma once
#include "gemma4e_dflash_flash_detail.hpp"
#include "dflash_draft.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>
#include <string>
#include <vector>

namespace dfl {
using clk = std::chrono::steady_clock;
inline long long us_since(clk::time_point t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
}
inline float bf16_to_f32(bf16 v) {
    uint16_t h;
    std::memcpy(&h, &v, 2);
    uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
// argmax over bf16 logits; ties go to the lowest id (same rule as the sampler's greedy path)
inline int argmax_bf16(const bf16* p, int n) {
    int best = 0;
    float bv = bf16_to_f32(p[0]);
    for (int i = 1; i < n; i++) {
        float v = bf16_to_f32(p[i]);
        if (v > bv) { bv = v; best = i; }
    }
    return best;
}
inline constexpr int kTaps = 5;
inline constexpr int kRing = 1024;      // tap ring depth = the draft's attention window
inline constexpr int kBlock = 8;        // rows of one verify block (id_last + 7 drafts)
inline constexpr int kTapLayers[kTaps] = {5, 11, 17, 23, 35};
}  // namespace dfl

struct gemma4e_dflash_flash::Impl::Spec {
    dflash_draft_host draft;
    bool draft_loaded = false;
    bool env_off = false;               // FLM_DFLASH=0, or no fused verify (see load_draft)
    bool seq_forced = false;            // FLM_DFLASH_VERIFY=seq: speculate with the sequential verify (8 decode passes, debug / reference)

    std::vector<int> stop_ids;
    std::vector<int> ids;               // token id of every row held in the KV caches (size must equal the context length)

    // taps of the last kRing positions, slot = position % kRing: [kRing][kTaps][D]
    std::vector<bf16> tap_ring;
    long long tap_lo = 0, tap_hi = 0;   // positions [tap_lo, tap_hi) are valid
    std::vector<bf16> tap_cur;          // taps of the token being decoded: [kTaps][D]
    int fwd_pos = 0;

    // warm restore (flash pin): ids and tap range of the checkpointed rows. The context is capped at 1024 = kRing, so the tap ring slots
    // of those rows are never overwritten by later rows and stay valid; only the bookkeeping has to be put back.
    bool ckpt_taps_ok = false;
    int ckpt_len = 0;
    std::vector<int> ckpt_ids;

    std::vector<bf16> logit_rows;       // [kBlock][vocab]
    int rows_valid = 0;
    int verify_ctx0 = 0, verify_n = 0;

    std::vector<int> swa_layers;        // kv cache indices of the sliding layers (non-skip)
    std::vector<bf16> target_norm;      // model.norm.weight (x slot 1) while a draft norm is swapped in

    bool prime_pending = false;         // first draft cycle after a prefill absorbs the prompt taps
    uint64_t prime_us = 0;

    // adaptive guard (see draft_ready): speculation is only worth it while  AL * t_plain(ctx) > margin * t_cycle(ctx)
    bool guard_off = false;
    double ema_al = 2.6;                // running mean of tokens per cycle, optimistic at the start of every request
    int plain_since = 0;                // plain steps since the last speculative cycle (probe trigger)
    // measured costs (ms, running means; 0 = not measured yet: the fitted model of draft_ready is used instead)
    double ema_t_cycle = 0, ema_t_plain = 0;
    dfl::clk::time_point cycle_t0, fwd_t0;
    bool in_cycle = false, in_verify_seq = false;

    // fused verify path (gemma4e_dflash_flash_verify.cpp)
    struct Verify;
    Verify* verify = nullptr;
    int verify_rows = 0;                // KV rows the last verify wrote: n+1 (sequential) or kBlock (fused, padded rows included)
    bool verify_check = false;          // FLM_DFLASH_VERIFY=check: run both paths and compare
    long long chk_rows = 0, chk_argmax_diff = 0, chk_argmax_diff_big = 0;   // big = sequential top-1/top-2 margin above 2 bf16 ulps
    double chk_cos_sum = 0;
    double chk_min_cos = 1.0;
    long long t_v_prep = 0, t_v_seq = 0, t_v_run = 0, t_v_taps = 0, t_v_lm = 0;   // fused verify phase times
    long long t_v_prep_work = 0;        // duration of verify_prepare itself (runs on another thread)
    long long fused_cycles = 0;

    // speculative draft-context injection (overlap): right after a verify, the taps of ALL its rows are injected on a background
    // thread while the caller samples / accepts; dflash_commit() joins it and truncates the draft context to the kept rows.
    // Injection of a row depends on that row's taps and position only, so injecting rejected rows and truncating is exact.
    std::thread inj_thread;
    bool inj_async = true;              // FLM_DFLASH_ASYNC_INJECT=0 disables
    long long t_inj_wait = 0;           // time the main thread waited for the background injection

    // FLM_DFLASH_DUMP=<prefix>: debug dumps for the AL analysis (FLM_Xclbin/Gemma4/verify_decoding/notes/al_and_host.md)
    //   <prefix>.inj : records {int64 pos0, int32 n} + [n][kTaps][D] bf16 of every draft-context injection
    //   <prefix>.drf : records {int32 pos0, int32 id_last, int32 drafts[7]} + [7][D] bf16 draft hidden rows (pre output_norm)
    FILE* dump_inj = nullptr;
    FILE* dump_drf = nullptr;

    // statistics
    long long cycles = 0, tokens = 0, accepted = 0, drafted = 0;
    long long t_draft = 0, t_draft_head = 0, t_verify = 0, t_accept = 0, t_commit = 0, t_prime = 0;
};


using dfl::us_since; using dfl::bf16_to_f32; using dfl::argmax_bf16; using dfl::kTaps; using dfl::kRing; using dfl::kBlock; using dfl::kTapLayers; using dfl::clk;
