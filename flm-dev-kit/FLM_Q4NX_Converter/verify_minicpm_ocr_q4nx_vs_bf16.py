# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""What did quantization actually cost MiniCPM-OCR-1.0? Weight-level, q4nx vs HF bf16.

    source activate.sh
    python verify_minicpm_ocr_q4nx_vs_bf16.py [--min-cos 0.999] [-v]

`verify_minicpm_v_q4nx.py` answers a different question: it scores model.q4nx against the GGUF it
was built from, so it proves the *converter* is faithful. It cannot see the loss that llama.cpp
already baked in when it wrote a Q4_K GGUF from the bf16 checkpoint. This script closes that gap by
going all the way back to `model.safetensors`, which is the only ground truth there is.

It is the weight-level half of the guide's rung 3. The other half -- running the reference forward
with these weights and scoring the logits -- is a separate job; a tensor can score well here and
still matter end to end, and vice versa, so neither replaces the other.

Only the language tower is covered. The vision tower is bf16 on both sides (no quantization at all,
only zero-padding to the array tile), so there is nothing to lose there and
`verify_minicpm_v_vision_q4nx.py` already checks the padding.
"""
import argparse
import json
import os
import re
import struct
import sys

import numpy as np
from safetensors import safe_open

from verify_minicpm_v_q4nx import cosine, load_q4nx_tensor, rel_rmse

HF_DIR = "/path/to/models/MiniCPM-OCR-1.0"
NPU_DIR = "/path/to/models/MiniCPM-OCR-1.0-NPU2"

HEAD_DIM = 256
INTERVAL = 4          # every 4th layer is full attention
VOCAB = 248100        # lm_head is zero-padded up from this to a whole row block
LM = "model.language_model."

# What each storage format should cost at the *weight* level, against bf16.
#
# One threshold for everything is the wrong instrument here: 0.999 is the right bar for a bf16
# tensor that should be near-exact, and far too tight for a 4.5-bit one, where it flags all 150
# Q4_K tensors and tells you nothing. These are expectations about the format, not about this
# model -- a Q4_K tensor at 0.9966 is ordinary, and the same tensor at 0.95 is a bug.
#
# Note these are *weight* cosines. They do not predict end-to-end accuracy, in either direction;
# that needs the forward half of rung 3.
MIN_COS = {"F32": 0.999999, "BF16": 0.9999, "q80": 0.9999, "q4k": 0.995, "q41": 0.995, "q40": 0.99}


def hf_name(q4nx: str) -> str | None:
    """The `model.safetensors` tensor a q4nx language tensor came from, or None if there is none.

    Mostly a prefix swap. The interesting cases are the ones where the q4nx name does not mean
    what it looks like:

      * `self_attn.gate_proj.weight` is only an attention gate on full-attention layers, and those
        carry it *inside* q_proj instead (see `unreshape`). On a linear-attention layer the same
        name holds GatedDeltaNet's z projection, `in_proj_z`.
      * `lm_head.weight` has no source of its own: the embeddings are tied.
    """
    if q4nx == "lm_head.weight":
        return LM + "embed_tokens.weight"          # tied
    if q4nx in ("model.embed_tokens.weight", "model.norm.weight"):
        return LM + q4nx[len("model."):]

    m = re.match(r"model\.layers\.(\d+)\.(.+)$", q4nx)
    if not m:
        return None
    layer, rest = int(m.group(1)), m.group(2)
    base = f"{LM}layers.{layer}."
    is_full = layer % INTERVAL == INTERVAL - 1

    if rest.endswith(".bf16.weight"):
        rest = rest.replace(".bf16.weight", ".weight")   # twin of the quantized tensor

    linear = {
        "linear_attn.qkv_proj.weight":      "linear_attn.in_proj_qkv.weight",
        "linear_attn.ssm_alpha_proj.weight": "linear_attn.in_proj_a.weight",
        "linear_attn.ssm_beta_proj.weight":  "linear_attn.in_proj_b.weight",
        "linear_attn.ssm_out_proj.weight":  "linear_attn.out_proj.weight",
        "linear_attn.ssm_conv1d.weight":    "linear_attn.conv1d.weight",
        "linear_attn.ssm_norm.weight":      "linear_attn.norm.weight",
        "linear_attn.ssm_a":                "linear_attn.A_log",
        "linear_attn.ssm_dt.bias":          "linear_attn.dt_bias",
        "self_attn.gate_proj.weight":       "linear_attn.in_proj_z.weight",
    }
    if not is_full and rest in linear:
        return base + linear[rest]
    if rest in ("input_layernorm.weight", "post_attention_layernorm.weight") or rest.startswith("mlp."):
        return base + rest
    if is_full and rest.startswith("self_attn."):
        return base + rest
    return None


# RMSNorm weights that llama.cpp stores with the +1 folded in, so the kernel can use a bare
# multiply rather than (1 + w). Determined by measuring, not assumed: every one of these matched
# `q4nx == hf + 1` to within a bf16 ulp, and everything else matched identity exactly.
# `linear_attn.ssm_norm.weight` is deliberately NOT here -- the GatedDeltaNet's internal norm keeps
# the HF convention, and folding a +1 into it would be a silent 1-ulp-of-nothing bug.
PLUS_ONE_NORMS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)


def to_reference(q4nx: str, hf_weight: np.ndarray) -> np.ndarray:
    """The HF tensor expressed the way the GGUF stores it.

    llama.cpp does not copy weights verbatim; it bakes two conventions in on the way out. Comparing
    against the raw HF tensor without replaying them does not report a small error, it reports
    nonsense -- cosine 0.20 on a layernorm, -0.39 on `ssm_a` -- and invites you to go hunting for a
    quantizer bug that is not there.
    """
    if q4nx == "model.norm.weight" or q4nx.endswith(PLUS_ONE_NORMS):
        return hf_weight + 1.0
    if q4nx.endswith("linear_attn.ssm_a"):
        return -np.exp(hf_weight)           # the decay, precomputed from A_log
    return hf_weight


def unreshape(q4nx: str, got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Undo the converter's layout changes so `got` can be compared with `ref` elementwise."""
    layer = re.search(r"layers\.(\d+)\.", q4nx)
    is_full = layer is not None and int(layer.group(1)) % INTERVAL == INTERVAL - 1

    if is_full and q4nx.endswith("self_attn.q_proj.weight"):
        # qwen35.py packs '(g p h) c -> (p g h) c' with p=2: HF interleaves q and its output gate
        # per head, the converter splits them into two contiguous halves. Interleave them back.
        cols = got.shape[1]
        got = got.reshape(2, -1, HEAD_DIM, cols).transpose(1, 0, 2, 3).reshape(-1, cols)
    elif q4nx.endswith("ssm_conv1d.weight"):
        got = got.T                                   # q4nx [4, 6144] vs HF [6144, 1, 4]
    elif q4nx.endswith(("ssm_alpha_proj.weight", "ssm_beta_proj.weight")):
        got = got[:ref.shape[0]]                      # 16 rows duplicated to 32
    elif q4nx == "lm_head.weight":
        got = got[:VOCAB]                             # zero-padded to a whole row block

    return got.reshape(ref.shape) if got.size == ref.size else got


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf", default=HF_DIR)
    ap.add_argument("--npu", default=NPU_DIR)
    ap.add_argument("--min-cos", type=float, default=None,
                    help="override the per-format gate below with a single number")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    q = safe_open(os.path.join(args.npu, "model.q4nx"), "pt")
    hf = safe_open(os.path.join(args.hf, "model.safetensors"), "pt")
    hf_keys = set(hf.keys())

    failures, rows, skipped = [], [], []
    for k in sorted(q.keys()):
        src = hf_name(k)
        if src is None or src not in hf_keys:
            skipped.append(k)
            continue

        fmt, got = load_q4nx_tensor(q, k)
        ref = to_reference(k, hf.get_tensor(src).float().numpy())   # bf16 -> fp32 is exact
        got = unreshape(k, got, ref)

        if got.shape != ref.shape:
            failures.append(f"{k}: shape {got.shape} vs {ref.shape} (from {src})")
            continue

        cos, rr = cosine(got, ref), rel_rmse(got, ref)
        rows.append((cos, rr, fmt, k))
        gate = args.min_cos if args.min_cos is not None else MIN_COS.get(fmt, 0.995)
        bad = not (cos > gate)
        if bad:
            failures.append(f"{k}: {fmt} cos {cos:.6f} relRMSE {rr:.2e} vs bf16 (gate {gate})")
        if bad or args.verbose:
            print(f"    {'FAIL' if bad else 'ok  '} {k:58s} {fmt:5s} cos {cos:.7f} relRMSE {rr:.2e}")

    # Per-format summary: this is the number that says what the recipe cost.
    print(f"\nquantization cost vs HF bf16, {len(rows)} language tensors "
          f"({len(skipped)} skipped: vision/no source)")
    by_fmt = {}
    for cos, rr, fmt, k in rows:
        by_fmt.setdefault(fmt, []).append((cos, rr, k))
    print(f"    {'fmt':6s} {'n':>4s} {'min cos':>11s} {'mean cos':>11s} {'max relRMSE':>12s} "
          f"{'gate':>9s}  worst")
    for fmt, v in sorted(by_fmt.items()):
        cs = [x[0] for x in v]
        worst = min(v, key=lambda x: x[0])
        gate = args.min_cos if args.min_cos is not None else MIN_COS.get(fmt, 0.995)
        print(f"    {fmt:6s} {len(v):4d} {min(cs):11.7f} {np.mean(cs):11.7f} "
              f"{max(x[1] for x in v):12.2e} {gate:9.6f}  {worst[2]}")

    if rows:
        w = min(rows)
        print(f"\n    worst overall: {w[3]}  cos {w[0]:.7f}  relRMSE {w[1]:.2e}")

    if failures:
        print(f"\nFAIL ({len(failures)})")
        for f in failures[:20]:
            print("  -", f)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
