// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/npu/decode_prof.hpp
/// \brief Where a decode cycle's wall clock goes, split host vs device.
///
/// The decode sibling of `dequant_mm_prof` (speculate/npu/proj_op.hpp), and it
/// exists for the same reason: module 1's per-stage split is what showed prefill
/// to be 50% host, and nothing equivalent covered modules 2 and 3.
///
/// A decode cycle is one verify pass -- 64 fused layer dispatches and one
/// `lm_head_8` -- plus up to `MTP_STEPS` draft steps. REPLACEMENTWORK.md's
/// measurements bracket the *whole* of a layer (25.35 ms per delta layer,
/// 205.1 ms per verify row) and cannot say how much of that is the weight
/// stream and how much is the host staging for it. That is the number every
/// decode optimisation has to be argued against, so it is the number this
/// header produces.
///
/// Built only under `make DECODE_PROFILE=1` (`-DQWEN3_8MTP_DECODE_PROFILE`, a
/// -D flag with no make dependency -- `make clean` first, memory:
/// make-does-not-track-debug-level). Zero cost when off: every macro below
/// expands to `do {} while (0)` and the counters do not exist.
///
/// Not thread safe, and it does not need to be. Every counter is touched from
/// the driver thread only; the one piece of decode work that runs on a worker
/// is `npu_delta_layer::_replay`, and what is timed there is the *launch*, not
/// the fold, which is the honest thing to charge the critical path anyway.
#ifndef __QWEN3_8MTP_NPU_DECODE_PROF_HPP__
#define __QWEN3_8MTP_NPU_DECODE_PROF_HPP__

#ifdef QWEN3_8MTP_DECODE_PROFILE

#include "utils/utils.hpp"

#include <string>

namespace q38npu {

/// Microseconds, accumulated across the process and reset per verify pass.
///
/// The three `*_dispatch_us` counters are the device's; everything else is the
/// host's. Their sum is not quite the pass's wall clock -- the sweep's own
/// `memcpy(x, y)` and the argmax sit outside any bracket -- and `decode_prof`
/// prints the residual rather than hiding it.
struct decode_prof {
    // ---- the 48 gated-delta-net layers -------------------------------------
    static inline double delta_dispatch_us = 0;  ///< the blocking NPU call
    static inline double delta_write_x_us  = 0;  ///< fp32 -> bf16 + flush
    static inline double delta_state_us    = 0;  ///< S memcpy + conv narrow + flush
    static inline double delta_read_us     = 0;  ///< the two sync_from_device
    static inline double delta_widen_us    = 0;  ///< h2 bf16 -> fp32
    static inline double delta_carry_us    = 0;  ///< conv carry + arena stage
    static inline double delta_join_us     = 0;  ///< waiting on the previous fold
    static inline long   delta_passes      = 0;

    // ---- the 16 full-attention layers ---------------------------------------
    static inline double attn_dispatch_us = 0;
    static inline double attn_stage_us    = 0;   ///< rope table + x + both flushes
    static inline double attn_read_us     = 0;
    static inline double attn_widen_us    = 0;
    static inline double attn_ctx_us      = 0;   ///< set_ctx(): sequence regeneration
    static inline long   attn_passes      = 0;

    // ---- the projection and the draft head ----------------------------------
    /// `head_us` brackets the whole `npu_lm_head_8::run`, which contains a
    /// dispatch; `head_dispatch_us` is that dispatch. Charging the difference
    /// -- and only the difference -- to the host is the point: the first cut of
    /// this header did not split them and read the ~29 ms weight stream as host
    /// time, which inflated the host share by a factor of three.
    static inline double head_us          = 0;   ///< lm_head_8, whole call
    static inline double head_dispatch_us = 0;   ///< ...of which, the device
    static inline double argmax_us   = 0;
    static inline double mtp_dispatch_us = 0;
    static inline double mtp_seq_us  = 0;        ///< gen_mtp_layer_seq per step
    static inline double mtp_host_us = 0;        ///< staging, rope, widen
    static inline long   mtp_steps   = 0;

    static inline time_utils::time_point mark{};

    static void reset() {
        delta_dispatch_us = delta_write_x_us = delta_state_us = 0;
        delta_read_us = delta_widen_us = delta_carry_us = delta_join_us = 0;
        attn_dispatch_us = attn_stage_us = attn_read_us = attn_widen_us = 0;
        attn_ctx_us = head_us = head_dispatch_us = argmax_us = 0;
        mtp_dispatch_us = mtp_seq_us = mtp_host_us = 0;
        delta_passes = attn_passes = mtp_steps = 0;
    }

    /// One block per line, in ms, with the per-dispatch average alongside.
    /// `wall_us` is the caller's own bracket around the whole sweep, so the
    /// residual line is the work no counter claims -- the sweep's `memcpy(x, y)`
    /// and anything else outside a bracket.
    static void report(const std::string& what, double wall_us) {
        const double dev  = delta_dispatch_us + attn_dispatch_us +
                            mtp_dispatch_us + head_dispatch_us;
        const double host = delta_write_x_us + delta_state_us + delta_read_us +
                            delta_widen_us + delta_carry_us + delta_join_us +
                            attn_stage_us + attn_read_us + attn_widen_us +
                            attn_ctx_us + (head_us - head_dispatch_us) +
                            argmax_us + mtp_seq_us + mtp_host_us;
        auto line = [](const char* name, double us, long n) {
            header_print("decode-prof",
                         name << ": " << us / 1000.0 << " ms"
                              << (n ? " (" + std::to_string(us / (double)n) +
                                          " us x " + std::to_string(n) + ")"
                                    : std::string()));
        };
        header_print("decode-prof", "---- " << what << " ----");
        line("delta.dispatch", delta_dispatch_us, delta_passes);
        line("delta.write_x",  delta_write_x_us,  delta_passes);
        line("delta.state_in", delta_state_us,    delta_passes);
        line("delta.read",     delta_read_us,     delta_passes);
        line("delta.widen",    delta_widen_us,    delta_passes);
        line("delta.carry",    delta_carry_us,    delta_passes);
        line("delta.join",     delta_join_us,     delta_passes);
        line("attn.dispatch",  attn_dispatch_us,  attn_passes);
        line("attn.set_ctx",   attn_ctx_us,       attn_passes);
        line("attn.stage",     attn_stage_us,     attn_passes);
        line("attn.read",      attn_read_us,      attn_passes);
        line("attn.widen",     attn_widen_us,     attn_passes);
        line("lm_head_8.dispatch", head_dispatch_us, 0);
        line("lm_head_8.host", head_us - head_dispatch_us, 0);
        line("argmax",         argmax_us,         0);
        line("mtp.dispatch",   mtp_dispatch_us,   mtp_steps);
        line("mtp.gen_seq",    mtp_seq_us,        mtp_steps);
        line("mtp.host",       mtp_host_us,       mtp_steps);
        header_print("decode-prof",
                     "TOTAL: " << wall_us / 1000.0 << " ms = "
                               << dev / 1000.0 << " device + "
                               << host / 1000.0 << " host + "
                               << (wall_us - dev - host) / 1000.0
                               << " unattributed  ["
                               << 100.0 * host / (wall_us > 0 ? wall_us : 1)
                               << "% host]");
    }
};

} // namespace q38npu

/// Time `expr` into `counter`. `expr` may be void or may have a value; the
/// value is returned, so a dispatch can be wrapped in place:
///     const ert_cmd_state st = DPROF_T(delta_dispatch_us, eng_->app(...));
#define DPROF_T(counter, expr)                                                 \
    [&]{ const auto __d0 = time_utils::now();                                  \
         auto __r = (expr);                                                    \
         q38npu::decode_prof::counter +=                                       \
             time_utils::duration_us(__d0, time_utils::now()).first;           \
         return __r; }()

/// The void form, for statements rather than expressions.
#define DPROF_V(counter, stmt)                                                 \
    do { const auto __d0 = time_utils::now();                                  \
         stmt;                                                                 \
         q38npu::decode_prof::counter +=                                       \
             time_utils::duration_us(__d0, time_utils::now()).first;           \
    } while (0)

/// Open / close brackets for a region too long or too declaration-heavy to sit
/// inside `DPROF_V`. `tag` makes the cursor's name unique so two regions can
/// share a scope; it is not otherwise used.
#define DPROF_B(tag)          const auto __dprof_##tag = time_utils::now()
#define DPROF_E(tag, counter) q38npu::decode_prof::counter +=                  \
    time_utils::duration_us(__dprof_##tag, time_utils::now()).first

#define DPROF_COUNT(counter) (q38npu::decode_prof::counter++)
#define DPROF_RESET()        q38npu::decode_prof::reset()
#define DPROF_REPORT(tag, what)                                                \
    q38npu::decode_prof::report(                                               \
        what, time_utils::duration_us(__dprof_##tag, time_utils::now()).first)

#else   // !QWEN3_8MTP_DECODE_PROFILE

#define DPROF_T(counter, expr)  (expr)
#define DPROF_V(counter, stmt)  do { stmt; } while (0)
#define DPROF_B(tag)            do {} while (0)
#define DPROF_E(tag, counter)   do {} while (0)
#define DPROF_COUNT(counter)    do {} while (0)
#define DPROF_RESET()           do {} while (0)
#define DPROF_REPORT(tag, what) do {} while (0)

#endif  // QWEN3_8MTP_DECODE_PROFILE

#endif  // __QWEN3_8MTP_NPU_DECODE_PROF_HPP__
