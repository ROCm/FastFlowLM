// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/npu/vision_attn_reconfig.hpp
/// \brief Opt-in: run the tower's attention dispatches in vision_mm_a's hw_context between two in-stream
///        reconfigurations (vision_mm_a -> vision_attn -> vision_mm_a), instead of switching hw_contexts twice.
///
/// A block's order is qkv (design A) -> attention -> proj / fc1 / fc2 / next qkv (design A): two context
/// switches of ~2.4 ms around every attention dispatch (54 per encode, ~130 ms). As one submission
/// [reconfigure to attn, attention, reconfigure back to A] the round trip costs 1.5 ms instead of 4.9 ms
/// (measured on the shipped overlays), ~95 ms per encode.
///
/// The mechanism and its safeguards are the text path's (minicpm_v_reconfig.hpp, which explains the quiesce
/// of pending lock acquires that makes the replay safe). The same three gates, each with the two-context
/// path as the fallback:
///   built      FLM_MINICPM_V_VISION_SINGLE_CTX=1, the NPU runs without preemption, both CDOs parse
///   validated  a known-answer test at load: attention of 512 deterministic rows through the round trip must equal
///              the two-context result bit for bit, and a design-A matmul run before and after the round trips
///              must give identical bits (the array state is restored); on an NPU this process has to itself
///   usable()   per call: validated, not disabled by an earlier failure, every hw_context on the NPU is ours
/// A failed round trip disables the path (and the text path: one kill switch) for the process; the same attention is then run through
/// vision_attn's own hw_context, which also makes the firmware reload the design A PDI for the next matmul.
#ifndef __MINICPM_V_4_7_NPU_VISION_ATTN_RECONFIG_HPP__
#define __MINICPM_V_4_7_NPU_VISION_ATTN_RECONFIG_HPP__

#include "minicpm_v_reconfig.hpp"

#include <atomic>
#include <cstdlib>
#include <string>

#if FLM_MINICPM_RECONFIG_SUPPORTED
#include "npu_utils/npu_utils.hpp"
#endif

namespace mcv_npu {

#if FLM_MINICPM_RECONFIG_SUPPORTED
struct vision_attn_reconfig {
    using chain_t = minicpm_reconfig::Chain<npu_app, xrt::run>;
    chain_t to_attn;   //!< vision_mm_a -> vision_attn
    chain_t to_mm;     //!< vision_attn -> vision_mm_a
    npu_app_manager* home = nullptr;   //!< vision_mm_a's manager: the hw_context everything runs in
#ifdef FLM_MINICPM_RECONFIG_FAULT_INJECT
    // test build only: the Nth entry into the attention configuration (FLM_MCPM_VFAULT_AT=N, 1-based, counted after the self-test) replays
    // design A's configuration instead of vision_attn's (the old prelude does not hang these overlays), so the attention dispatch runs on
    // the wrong array and hangs: proves timeout -> disable -> redo through vision_attn's own context
    chain_t to_attn_unquiesced;
    std::atomic<long> enters{0};
    long fault_at = 0;
    bool selftest_done = false;
#endif
    bool built = false;
    bool validated = false;
    std::atomic<long> trips{0};     //!< attention dispatches run through the round trip
    std::atomic<long> skipped{0};   //!< dispatches that took the two-context path because another process was on the NPU

    static bool requested() {
        // default on (self-test, exclusivity check and fallback as for the text path); FLM_MINICPM_V_VISION_SINGLE_CTX=0 switches it off
        static const bool v = [] { const char* e = std::getenv("FLM_MINICPM_V_VISION_SINGLE_CTX"); return !(e && std::atoi(e) == 0); }();
        return v;
    }

    ~vision_attn_reconfig() {
        if (built) header_print_r("INFO", "minicpm_v: vision single-context attention: " << trips.load() << " dispatches by round trip, " << skipped.load() << " skipped (NPU shared)");
    }

    /// \param dir the model's xclbin directory (vision_mm_a.xclbin)
    /// \param attn the attention xclbin in use (vision_attn.xclbin or vision_attn_kv.xclbin)
    void build(bool preemption_enabled, npu_app_manager* mm_a, const std::string& dir, const std::string& attn) {
        if (!requested() || mm_a == nullptr) return;
        if (preemption_enabled) {
            header_print_r("WARNING", "minicpm_v: vision single-context reconfiguration needs the NPU without preemption, using separate contexts");
            return;
        }
        try {
            home = mm_a;
            auto mk = [&] { return mm_a->create_app(); };
            to_attn.build(attn, dir + "/vision_mm_a.xclbin", mk);
            to_mm.build(dir + "/vision_mm_a.xclbin", attn, mk);
#ifdef FLM_MINICPM_RECONFIG_FAULT_INJECT
            to_attn_unquiesced.build(dir + "/vision_mm_a.xclbin", dir + "/vision_mm_a.xclbin", mk);
            if (const char* f = std::getenv("FLM_MCPM_VFAULT_AT")) fault_at = std::atol(f);
#endif
            built = true;
        } catch (const std::exception& e) {
            header_print_r("WARNING", "minicpm_v: vision single-context reconfiguration unavailable (" << e.what() << "), using separate contexts");
        }
    }

    /// usable for this call (re-checked every time: another process can appear at any moment)
    bool usable() {
        if (!built || !validated || minicpm_reconfig::disabled()) return false;
        if (minicpm_reconfig::npu_exclusive()) return true;
        skipped++;
        return false;
    }
};
#else
struct vision_attn_reconfig {
    static bool requested() { return false; }
};
#endif

}  // namespace mcv_npu

#endif  // __MINICPM_V_4_7_NPU_VISION_ATTN_RECONFIG_HPP__
