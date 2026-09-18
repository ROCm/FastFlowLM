"""Build the IRON artifacts this plugin dispatches, for one model.

    IRON_PATH=<iron> python3 build_artifacts.py <model_dir> [--m 256]

The shapes come from the model's own weights, the same way the plugin derives
them: one side of every projection is the hidden size and the other follows from
the q4nx tensor's byte count. So a new model needs no edit here.

Both operators take their shape as a runtime parameter, so each contributes one
xclbin and one instruction stream per shape -- which is what keeps a model's
weight shapes inside the driver's budget of 16 hardware contexts.

The dequant operator reads the block order FastFlowLM's loader writes to DRAM,
so it consumes the engine's per-layer weight buffer in place.

The artifacts committed alongside this plugin were built at IRON 3b9e4bb1b.
"""

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.environ.get("IRON_PATH", ""))

from iron.operators.flm.dequant.op import DequantBFP  # noqa: E402
from iron.operators.flm.gemm.op import GEMM  # noqa: E402

# The loader alternates up and gate every RUN out-features.
RUN, PERIOD = 512, 1024


def shapes(model_dir):
    """(gemm, dequant) shape sets for every projection the plugin dispatches."""
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    d = cfg["hidden_size"]
    non_skip = cfg["num_hidden_layers"] - cfg["num_kv_shared_layers"]

    with open(os.path.join(model_dir, "model.q4nx"), "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))

    def other(layer, proj):
        meta = header[f"model.layers.{layer}.{proj}.weight"]
        lo, hi = meta["data_offsets"]
        return (hi - lo) * 8 // 5 // d

    gemm, dequant = set(), set()
    for layer in range(cfg["num_hidden_layers"]):
        dq, dk = other(layer, "self_attn.q_proj"), other(layer, "self_attn.k_proj")
        inter = other(layer, "mlp.up_proj")

        gemm |= {(d, dq, "none"), (d, dk, "none"), (dq, d, "none"),
                 (d, inter, "none"), (d, inter, "gelu"), (inter, d, "none")}
        # q, k and v are dequantized together where the layer has all three;
        # the kv-sharing layers hold only q.
        dequant |= {(d, dq if layer >= non_skip else dq + 2 * dk, False),
                    (dq, d, False), (inter, d, False), (d, inter, True)}
    return sorted(gemm), sorted(dequant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--m", type=int, default=256, help="padded chunk length")
    args = ap.parse_args()

    gemm, dequant = shapes(args.model_dir)
    ops = [GEMM(M=args.m, K=k, N=n, rounding="floor", epilogue=epi) for k, n, epi in gemm]
    ops += [
        DequantBFP(K=k, N=n,
                   run_out_features=RUN if il else None,
                   run_period_out_features=PERIOD if il else None)
        for k, n, il in dequant
    ]

    for i, op in enumerate(ops, 1):
        print(f"[{i:2d}/{len(ops)}] {op.name}", flush=True)
        op.compile()

    print("artifacts are in ./build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
