#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Dequantize every tensor of a dflash_weight.q4nx back to fp32 and compare with the source GGUF.

usage: check_dflash_q4nx.py <dflash.gguf> <dflash_weight.q4nx> [--max-l2-q41 0.1] [--json out.json]
Limits (relative L2): Q4_1 <= 0.1, BF16 / norm tensors <= 4e-3 (bf16 rounding). Exit 1 on any violation.
"""
import argparse, json, sys
import numpy as np, torch
from gguf import GGUFReader, dequantize
from safetensors.torch import load_file

RB, CB, GS, PS = 32, 256, 32, 16     # row block, col block, quant group, parallel (row stride)

def unpack_q4nx(blob: torch.Tensor, rows: int, cols: int) -> np.ndarray:
    """Inverse of __Q4NX_Converter._pack_q4nx for a Q4_1 tensor ([rows, cols] = [out, in])."""
    n_chunks = (rows // RB) * (cols // CB)
    b = blob.view(torch.uint8).numpy().reshape(n_chunks, -1)
    ng = CB // GS
    dsz = ng * RB * 2
    d = np.ascontiguousarray(b[:, :dsz]).view(np.uint16).reshape(n_chunks, ng, RB)
    m = np.ascontiguousarray(b[:, dsz:2 * dsz]).view(np.uint16).reshape(n_chunks, ng, RB)
    f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
    d, m = f(d), f(m)
    qs = b[:, 2 * dsz:].reshape(n_chunks, RB // PS, CB, PS // 2)          # n g c r
    q = np.empty((n_chunks, RB // PS, CB, PS // 2, 2), np.uint8)
    q[..., 0] = qs & 0xF; q[..., 1] = qs >> 4                              # n g c r b
    q = q.reshape(n_chunks, RB // PS, CB, PS).transpose(0, 1, 3, 2).reshape(n_chunks, RB, CB)  # n (g s) c
    w = q.astype(np.float32).reshape(n_chunks, RB, ng, GS) * d.transpose(0, 2, 1)[..., None] + m.transpose(0, 2, 1)[..., None]
    w = w.reshape(n_chunks, RB, CB)
    w = w.reshape(rows // RB, cols // CB, RB, CB).transpose(0, 2, 1, 3).reshape(rows, cols)
    return w

def unblock_bf16(w: torch.Tensor) -> np.ndarray:                          # [N/32, K, 32] -> [N, K]
    return w.float().permute(0, 2, 1).reshape(-1, w.shape[1]).numpy()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("gguf"); ap.add_argument("q4nx")
    ap.add_argument("--max-l2-q41", type=float, default=0.1); ap.add_argument("--json")
    a = ap.parse_args()
    r = GGUFReader(a.gguf); t = load_file(a.q4nx)
    cfg = json.load(open("configs/dflash.json"))["name_map"]
    import re
    fwd = {}
    for v in cfg.values():
        for bid in range(5):
            fwd[v["gguf_name"].format(bid=bid)] = v["q4nx_name"].format(bid=bid)
    bad, rows = 0, []
    for g in r.tensors:
        qn = fwd[g.name]
        ref = dequantize(g.data, g.tensor_type) if g.tensor_type.name not in ("F32", "BF16") else None
        if g.tensor_type.name == "F32": ref = np.array(g.data, np.float32)
        elif g.tensor_type.name == "BF16": ref = (g.data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        ref = ref.reshape(tuple(g.shape.tolist()[::-1]))
        x = t[qn]
        if ref.ndim == 1: got, kind, lim = x.float().numpy(), "bf16-vec", 4e-3
        elif x.dtype == torch.bfloat16: got, kind, lim = unblock_bf16(x), "bf16-blk", 4e-3
        else: got, kind, lim = unpack_q4nx(x, *ref.shape), "q4_1", a.max_l2_q41
        l2 = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))
        ok = l2 <= lim and got.shape == ref.shape
        bad += (not ok)
        rows.append(dict(name=g.name, kind=kind, shape=list(ref.shape), rel_l2=l2, ok=ok))
        print(f"{'OK ' if ok else 'BAD'} {g.name:28s} {kind:9s} {str(list(ref.shape)):14s} relL2={l2:.4f}")
    print(f"tensors {len(rows)}, bad {bad}")
    if a.json: json.dump(rows, open(a.json, "w"), indent=1)
    sys.exit(1 if bad else 0)
main()
