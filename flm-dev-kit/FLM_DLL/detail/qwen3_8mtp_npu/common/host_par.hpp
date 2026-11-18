// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/host_par.hpp
/// \brief The one gate that decides whether a host-side row loop gets threads.
///
/// Moved out of `speculate/npu/proj_op.hpp`, which is where it started and where
/// it is still re-exported from, because the marshalling loops are no longer the
/// only callers: the per-row elementwise stages in `cpu_ref::attention_layer`
/// and `cpu_ref::delta_net_layer` want the same decision, and those headers are
/// underneath the NPU ones and cannot include them.
///
/// ## Why these loops want threads
///
/// The fp32 <-> bf16 loops on this path -- x staging, the y widen, the
/// scatters -- are elementwise, independent, and were 2.8 s of a 11.2 s
/// L = 1024 window. They are not memory bound: the widest of them, the SwiGLU
/// un-weave, moved ~7 GB per window at 4 GB/s, an order of magnitude under
/// DDR. The cost was per-element ALU work, because `biovault::bfloat16_t
/// (float)` is a `std::fpclassify` switch (biovault_bfloat16.h:124) -- branchy,
/// so gcc cannot vectorise a loop containing it, ~0.6 ns an element.
///
/// **That is fixed, and it was the bigger half.** `common/cpu_simd.hpp` does
/// the conversion 16 lanes at a time, branch-free and bit-identical (checked
/// exhaustively over all 2^32 floats by `tools/verify_cpu_simd.cpp`), and every
/// marshalling loop calls it. One vectorised core beats all 32 scalar ones by
/// 4-8x. What is left for threads is a second-order ~2x on top of that -- still
/// worth having where it is free, which is why the gate stays.
///
/// **The arithmetic stages are a different case and they are the newer
/// callers.** `rms_norm`, the sigmoid gate and the gated RMSNorm are fp32 in and
/// fp32 out with no bf16 constructor in them, so gcc already vectorises their
/// inner loops -- there is no 4-8x sitting on the table the way there was for
/// the marshalling, and threads are the *only* lever left. They are also much
/// more of the window than the marshalling now is: the two norms and the gate
/// run 48 times per window in the linear block and 16 in the attention one, over
/// `L x 5120` and `L x 6144`, all of it on one core.
///
/// ## Why it is conditional
///
/// **Measured, L = 1024, round 1 wall, same binary, after the vectorisation:**
///
/// | | default (spin) | `OMP_WAIT_POLICY=passive` |
/// |---|---:|---:|
/// | serial   | 9.82 s | -- |
/// | threaded | 9.84 s | **8.95 s** |
///
/// Threading is worth 9% under a sleeping barrier and is a wash -- it was a 6%
/// regression before the vectorisation -- under a spinning one. The reason is
/// the shape of this path: a marshalling loop, then a *blocking* NPU dispatch
/// of ~10 ms, ~1500 times per window. On the default policy libgomp leaves the
/// team spinning after every region, so 31 cores burn through the dispatch; the
/// device barely notices (dispatch 5400 vs 5347 ms) but the host loops around
/// it run slower, which is what a power/frequency budget spent on spinning
/// looks like. A plain `#pragma omp parallel for` benchmark with no dispatch in
/// it shows the opposite sign -- 124 ms spinning against 709 ms passive over
/// 20000 tiny regions -- so this is not a general statement about wait
/// policies.
///
/// The policy cannot be set from here. libgomp reads `GOMP_SPINCOUNT` and
/// `OMP_WAIT_POLICY` in its own ELF constructor, which runs before any
/// constructor of ours and long before `main`; `setenv` from either has no
/// effect (verified). So the only thing this library can do is notice.
///
/// `FLM_Q38_HOST_PAR=1` forces threading on and `=0` off. `prefill_cpu`'s
/// constructor prints which way it went.
///
/// ## The small-L guard
///
/// The same loops are the draft head's, at L = 1..4. One row of a D = 5120 projection
/// is 5120 elements and stays on one core, where a fork would cost more than
/// the loop.
#ifndef __QWEN3_8MTP_NPU_HOST_PAR_HPP__
#define __QWEN3_8MTP_NPU_HOST_PAR_HPP__

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace q38npu {

/// Elements below which a host row loop stays serial -- or `SIZE_MAX`, when
/// threading it is a *regression* on this process's OpenMP runtime.
inline size_t host_par_min() {
    static const size_t v = [] () -> size_t {
        constexpr size_t ON = 1u << 16, OFF = (size_t)-1;
        if (const char* e = std::getenv("FLM_Q38_HOST_PAR"))
            return e[0] == '0' ? OFF : ON;
        // Either knob alone puts libgomp's barriers to sleep; accept either
        // spelling rather than requiring both, since `OMP_WAIT_POLICY` is the
        // standard one and `GOMP_SPINCOUNT=0` is the GCC-specific one.
        const char* wp = std::getenv("OMP_WAIT_POLICY");
        const char* sc = std::getenv("GOMP_SPINCOUNT");
        const bool passive = (wp && (wp[0] == 'p' || wp[0] == 'P')) ||
                             (sc && std::strcmp(sc, "0") == 0);
        return passive ? ON : OFF;
    }();
    return v;
}

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_HOST_PAR_HPP__
