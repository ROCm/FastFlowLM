#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Dequantize every tensor of a DFlash2 dflash_weight.q4nx back to fp32 and compare it with the HF checkpoint.

usage: check_dflash2_q4nx.py <hf model.safetensors> <dflash_weight.q4nx> [--max-l2-q4k 0.09] [name-substring ...]
Limits (relative L2): Q4_K matrices <= 0.09 (the quantizer gives ~0.07), BF16 tensors exact. Exit 1 on any violation.
`dequant_q4k` is the inverse of __Q4NX_Converter._pack_q4k (also imported by FLM_DLL/detail/qwen3_8mtp_npu/tools/dflash2_ref.py).
"""
import argparse
import sys

import numpy as np
import torch
from safetensors import safe_open

RB, CB, GS = 32, 256, 32     # row block, column block, quantization group


def _bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def dequant_q4k(blob, rows, cols):
    """[rows/32][cols/256][4736 B] FLM q4_k chunks -> [rows, cols] fp32."""
    n = (rows // RB) * (cols // CB)
    b = blob.view(torch.uint8).numpy().reshape(n, -1)
    assert b.shape[1] == 4736, b.shape
    ng = CB // GS
    s8 = b[:, :ng * RB].reshape(n, ng, RB).astype(np.float32)
    m8 = b[:, ng * RB:2 * ng * RB].reshape(n, ng, RB).astype(np.float32)
    off = 2 * ng * RB
    qs = b[:, off:off + CB * RB // 2].reshape(n, CB, RB // 2)
    off += CB * RB // 2
    S = _bf16(np.ascontiguousarray(b[:, off:off + RB * 2]).view(np.uint16))            # [n, 32]
    Mn = _bf16(np.ascontiguousarray(b[:, off + RB * 2:off + RB * 4]).view(np.uint16))   # stored negated
    q = np.empty((n, CB, RB // 2, 2), np.float32)
    q[..., 0] = qs & 0xF            # even row in the low nibble
    q[..., 1] = qs >> 4
    q = q.reshape(n, CB, RB).transpose(0, 2, 1)                                         # [n, rows, cols]
    gs = np.repeat(s8.transpose(0, 2, 1), GS, axis=2)
    gm = np.repeat(m8.transpose(0, 2, 1), GS, axis=2)
    w = S[:, :, None] * gs * q + Mn[:, :, None] * gm
    return w.reshape(rows // RB, cols // CB, RB, CB).transpose(0, 2, 1, 3).reshape(rows, cols)


def is_matmul(name):
    return name == "fc.weight" or name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                                                  "gate_proj.weight", "up_proj.weight", "down_proj.weight"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf")
    ap.add_argument("pkg")
    ap.add_argument("--max-l2-q4k", type=float, default=0.09)
    ap.add_argument("pats", nargs="*")
    a = ap.parse_args()
    src, pkg = safe_open(a.hf, "pt"), safe_open(a.pkg, "pt")
    bad = 0
    for k in sorted(src.keys()):
        if a.pats and not any(p in k for p in a.pats):
            continue
        w, q = src.get_tensor(k), pkg.get_tensor("model." + k)
        if is_matmul(k):
            d = torch.from_numpy(dequant_q4k(q, *w.shape))
            e = ((d - w.float()).norm() / w.float().norm()).item()
            ok = e <= a.max_l2_q4k
            print(f"{'ok ' if ok else 'BAD'} {k:62s} Q4_K relL2 {e:.4f}", flush=True)
        else:
            ok = q.dtype == torch.bfloat16 and q.shape == w.shape and torch.equal(q, w.to(torch.bfloat16))
            print(f"{'ok ' if ok else 'BAD'} {k:62s} BF16 exact", flush=True)
        bad += not ok
    print("violations:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
