// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/vision_par.hpp
/// \brief How many threads the vision tower's host stages ask for.
///
/// Ported unchanged from Qwen3.8-27B's tower (`qwen3_8mtp_npu/image/`), whose
/// geometry this one shares. The table below is THAT tower's measurement on this
/// box; the mechanism it found -- a team that covers every core starves the
/// thread submitting the blocking dispatches -- is a property of libgomp and the
/// dispatch pattern, not of the model, so the rule carries over.
///
/// ## What was measured
///
/// One image at the operating point, L = 1200 -- 111 `vision_mm` dispatches and
/// 27 `vision_attn` ones -- on a 32-thread part, after the host stages were
/// vectorised. Best of four encodes, same binary, `FLM_MINICPM_V_VISION_THREADS`
/// sweeping the team size:
///
/// | threads | default (spin) | `OMP_WAIT_POLICY=passive` |
/// |---:|---:|---:|
/// |  1 |  793 ms | -- |
/// |  4 |  668 ms | -- |
/// |  8 |  663 ms | 683 ms |
/// | 16 |  **644 ms** | 682 ms |
/// | 24 |  650 ms | -- |
/// | 28 |  639 ms | -- |
/// | 30 |  735 ms | -- |
/// | 31 |  784 ms | -- |
/// | 32 | 1036 ms | 684 ms |
///
/// The spin column is flat from 4 to 28 and then falls off a cliff. **The cliff
/// is at `nproc`, not at a bandwidth knee** -- it starts at 30, doubles by 32,
/// and 28 is as fast as 16. So the mechanism is not "these stages stop scaling";
/// it is that on the default policy libgomp leaves the whole team spinning
/// after every parallel region, this path issues ~140 *blocking* dispatches per
/// image, and a team that covers every core leaves nothing for the thread that
/// has to submit them and poll for completion. Sizing the team to the machine
/// is what makes a runnable thread wait behind a spinning one.
///
/// It is the same runtime behaviour module 1 found from the other side
/// (memory: openmp-spin-wait-vs-blocking-dispatch), and the same caveat holds:
/// the policy cannot be set from here, because libgomp reads `OMP_WAIT_POLICY`
/// and `GOMP_SPINCOUNT` in its own ELF constructor, before any of ours.
///
/// ## Why there is no policy sniff here
///
/// Because capping costs nothing on the arm that does not need it. Under
/// `passive` the whole column is 682-684 ms whatever the team size, so one
/// unconditional rule is both simpler and strictly better than two arms -- and
/// the capped spin arm (650 ms) beats *every* passive cell anyway, which is the
/// number that matters, since the default is what a user who sets nothing gets.
///
/// ## The rule
///
/// `hw - max(2, hw/4)`: leave a quarter of the machine, and never fewer than
/// two cores, for the dispatch thread and the driver. On this part that is 24,
/// inside the flat region with room on both sides -- the point is to be off the
/// cliff, not to sit on the minimum, since 4 through 28 are within 5% of each
/// other and picking 16 over 24 would be fitting noise.
///
/// `FLM_MINICPM_V_VISION_THREADS=n` overrides it, for re-running that table on a part
/// with a different core count or a different DDR width.
#ifndef __MINICPM_V_VISION_PAR_HPP__
#define __MINICPM_V_VISION_PAR_HPP__
#pragma once

#include <algorithm>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mcv {
namespace vision {

/// Threads for one of the tower's elementwise host stages.
inline int par_threads() {
    static const int v = [] () -> int {
#ifdef _OPENMP
        const int hw = omp_get_max_threads();
#else
        const int hw = 1;
#endif
        if (const char* e = std::getenv("FLM_MINICPM_V_VISION_THREADS")) {
            const int n = std::atoi(e);
            if (n > 0) return std::min(n, hw);
        }
        return std::max(1, hw - std::max(2, hw / 4));
    }();
    return v;
}

}  // namespace vision
}  // namespace mcv

#endif  // __MINICPM_V_VISION_PAR_HPP__
