# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Compare Inkling-2-Mini's vision_weight.q4nx with the HF checkpoint (model.visual.*), tensor by tensor.

The HF tensors go through the same layout steps as q4nx/models/inkling2.py (q, k, v stacked into qkv,
matmul weights zero-padded to whole 512 x 512 tiles and rearranged into 64 x 512 blocks), then every
element must be equal: both sides are BF16, so there is no tolerance.  Also checked: every checkpoint
tensor is used exactly once, every file tensor is BF16, the padding is exactly zero.

Usage (from FLM_Q4NX_Converter/, converter venv active):

    python verify_inkling2_vision_q4nx.py --out OUT_DIR --hf MODEL.safetensors

convert.py runs it for `-t vision --verify-hf MODEL.safetensors`.  Exit status 0 only if nothing differs.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open

MM_K, MM_N = 512, 64
B = "model.visual.backbone."


def hf_to_q4nx(f):
    """Checkpoint tensors -> ({q4nx name: expected tensor}, checkpoint tensors not covered)."""
    names = [k for k in f.keys() if k.startswith("model.visual.")]
    out, used = {}, set()

    def get(n):
        used.add(n)
        return f.get_tensor(n).to(torch.bfloat16)

    def mm(w):
        pad = max(MM_K, MM_N)
        n, k = w.shape
        padded = F.pad(w, (0, (-k) % pad, 0, (-n) % pad))
        return rearrange(padded, "(N mm_n) (K mm_k) -> N K (mm_n mm_k)", mm_n=MM_N, mm_k=MM_K).contiguous()

    e = "model.visual.backbone.embeddings."
    for n in ("class_embedding", "cls_position_embedding", "position_embedding"):
        out[B + "embeddings." + n] = get(e + n)
    out[B + "embeddings.patch_embedding.weight"] = mm(get(e + "patch_embedding.weight"))
    for n in ("weight", "bias"):
        out[B + "ln_pre." + n] = get(B + "ln_pre." + n)
    nl = 1 + max(int(k.split(".")[4]) for k in names if k.startswith(B + "layers."))
    for i in range(nl):
        L = f"{B}layers.{i}."
        for n in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias", "ls_1.gamma", "ls_2.gamma"):
            out[L + n] = get(L + n)
        for kind in ("weight", "bias"):
            qkv = torch.cat([get(f"{L}self_attn.{x}_proj.{kind}") for x in "qkv"], 0)
            out[L + "self_attn.qkv_proj." + kind] = mm(qkv) if kind == "weight" else qkv
        for n in ("self_attn.out_proj", "mlp.c_fc", "mlp.c_proj"):
            out[f"{L}{n}.weight"] = mm(get(f"{L}{n}.weight"))
            out[f"{L}{n}.bias"] = get(f"{L}{n}.bias")
    for n in ("proj_in", "proj_out"):
        out[f"model.visual.{n}.weight"] = mm(get(f"model.visual.{n}.weight"))
    unused = sorted(set(names) - used)
    return out, unused


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="folder holding vision_weight.q4nx")
    ap.add_argument("--hf", required=True, help="HF model.safetensors")
    ap.add_argument("--gguf", help="(ignored; convert.py passes the source GGUF)")
    ap.add_argument("--max-hf-diff", type=float, default=0.0, help="largest accepted fraction of differing elements")
    args = ap.parse_args()

    bad = 0
    with safe_open(args.hf, "pt") as hf, safe_open(os.path.join(args.out, "vision_weight.q4nx"), "pt") as q:
        want, unused = hf_to_q4nx(hf)
        have = set(q.keys())
        for n in sorted(have - set(want)):
            print(f"[FAIL] file tensor {n} is not in the checkpoint"); bad += 1
        for n in sorted(set(want) - have):
            print(f"[FAIL] checkpoint tensor {n} is missing from the file"); bad += 1
        for n in unused:
            print(f"[FAIL] checkpoint tensor {n} is not covered"); bad += 1
        nd = nt = 0
        for n, ref in want.items():
            if n not in have:
                continue
            w = q.get_tensor(n)
            if w.dtype != torch.bfloat16:
                print(f"[FAIL] {n}: dtype {w.dtype}"); bad += 1; continue
            if w.shape != ref.shape:
                print(f"[FAIL] {n}: shape {tuple(w.shape)} expected {tuple(ref.shape)}"); bad += 1; continue
            d = (w.view(torch.int16) != ref.view(torch.int16)).sum().item()
            nd += d; nt += w.numel()
            if d:
                print(f"[FAIL] {n}: {d} of {w.numel()} elements differ"); bad += 1
    frac = nd / max(nt, 1)
    print(f"{len(want)} tensors, {nt} elements, {nd} differ ({frac:.2e})")
    if frac > args.max_hf_diff:
        bad += 1
    print("PASS" if not bad else f"FAIL ({bad} problems)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
