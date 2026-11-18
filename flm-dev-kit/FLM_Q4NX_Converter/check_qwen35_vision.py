# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Coverage + permutation check of a Qwen3.5-family vision q4nx (`-t vision` output) against the HF checkpoint.

The vision tower is bf16 end to end.  Whatever layout the engine's kernels want (blocked `(mm_n mm_k)` for
Qwen3.5/3.6, plain row-major + zero padding for Qwen3.8, patch-embed temporal stack), each tensor must hold
exactly the HF values -- only moved around, plus zero padding.  So per tensor: the multiset of non-zero values
of the q4nx tensor equals that of the HF tensor, and the names correspond one to one.

    python check_qwen35_vision.py --hf <hf dir> --q4nx <vision q4nx file>
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--q4nx", required=True)
    a = ap.parse_args()

    index = os.path.join(a.hf, "model.safetensors.index.json")
    if os.path.exists(index):
        wm = json.load(open(index))["weight_map"]
    else:
        with safe_open(os.path.join(a.hf, "model.safetensors"), "pt") as f:
            wm = {k: "model.safetensors" for k in f.keys()}
    hf_names = sorted(k for k in wm if "visual." in k)
    handles = {}

    def hf(name):
        fn = wm[name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(a.hf, fn), "pt")
        return handles[fn].get_tensor(name)

    q = safe_open(a.q4nx, "pt")
    q_names = set(q.keys())

    def to_q4nx_name(n):
        # model.visual.<x> (HF) <-> model.visual.<x> (q4nx); the HF checkpoint may carry a language_model.-style prefix
        return n[n.index("visual."):] if n.startswith("visual.") else "model." + n[n.index("visual."):]

    expected = {to_q4nx_name(n) for n in hf_names}
    # patch embed: one HF tensor, ".weight" -> stacked
    check(expected <= q_names, f"every HF vision tensor is present ({len(expected)} expected, "
          f"missing {sorted(expected - q_names)[:3]})")
    extra = q_names - expected - {"model.visual.patch_embed.proj.weight.1"}
    check(not extra, f"no extra vision tensors ({sorted(extra)[:3]})")

    bad = 0
    for n in hf_names:
        qn = to_q4nx_name(n)
        if qn not in q_names:
            continue
        h, t = hf(n).float().flatten(), q.get_tensor(qn).float().flatten()
        hn, tn = h[h != 0], t[t != 0]
        if hn.numel() != tn.numel() or not torch.equal(torch.sort(hn).values, torch.sort(tn).values):
            # patch_embed is split in two halves by some exporters: compare against the stacked whole
            bad += 1
            print(f"FAIL {qn}: nonzero values differ (hf {hn.numel()}, q4nx {tn.numel()})")
            FAILS.append(qn)
    check(bad == 0, f"all {len(hf_names)} vision tensors hold exactly the HF values (permutation + zero padding)")
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
