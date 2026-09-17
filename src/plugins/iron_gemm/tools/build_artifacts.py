"""Build the IRON artifacts this plugin dispatches, for Gemma4 E2B.

    IRON_PATH=<iron> python3 build_artifacts.py

Both operators take their shape as a runtime parameter, so each contributes one
xclbin and one instruction stream per shape -- which is what keeps a model's
weight shapes inside the driver's budget of 16 hardware contexts.

The dequant operator reads the block order FastFlowLM's loader writes to DRAM,
so it consumes the engine's per-layer weight buffer in place.

The artifacts committed alongside this plugin were built at IRON 3b9e4bb1b.
"""

import os
import sys

sys.path.insert(0, os.environ.get("IRON_PATH", ""))

from iron.operators.flm.dequant.op import DequantBFP  # noqa: E402
from iron.operators.flm.gemm.op import GEMM  # noqa: E402

D, I, SKIP_I = 1536, 6144, 12288
DQ, DK = 4096, 512
SWA_DQ, SWA_DK = 2048, 256
M = 256

# Every prefill GEMM of Gemma4-E2B as (K, N, epilogue). Only gate activates.
GEMM_SHAPES = [
    (D, DQ, "none"), (D, SWA_DQ, "none"),            # q
    (D, DK, "none"), (D, SWA_DK, "none"),            # k, v
    (DQ, D, "none"), (SWA_DQ, D, "none"),            # o
    (D, I, "none"), (D, I, "gelu"),                  # up, gate
    (D, SKIP_I, "none"), (D, SKIP_I, "gelu"),        # up, gate on skip layers
    (I, D, "none"), (SKIP_I, D, "none"),             # down
]

# The dequant shapes those need. q, k and v are dequantized together at their
# combined width -- they are adjacent out-features and the packed order is
# column-block-major over N, so one dispatch feeds all three. gate and up are
# interleaved in the layer's buffer every RUN out-features, so they stride.
RUN, PERIOD = 512, 1024
DEQUANT_SHAPES = [
    (D, DQ + 2 * DK, False), (D, SWA_DQ + 2 * SWA_DK, False),   # q, k, v
    (D, DQ, False), (D, SWA_DQ, False),                          # q alone, skip layers
    (DQ, D, False), (SWA_DQ, D, False),                          # o
    (D, I, True), (D, SKIP_I, True),                             # gate, up
    (I, D, False), (SKIP_I, D, False),                           # down
]


def main():
    ops = [GEMM(M=M, K=k, N=n, rounding="floor", epilogue=epi) for k, n, epi in GEMM_SHAPES]
    ops += [
        DequantBFP(
            K=k, N=n,
            run_out_features=RUN if il else None,
            run_period_out_features=PERIOD if il else None,
        )
        for k, n, il in DEQUANT_SHAPES
    ]

    for i, op in enumerate(ops, 1):
        print(f"[{i:2d}/{len(ops)}] {op.name}", flush=True)
        op.compile()

    print("artifacts are in ./build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
