#!./venv/bin/python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Check an Inkling-2 (inkling2_loop) model.q4nx against the GGUF it came from.

Decodes the packed q4k / q41 / q8nx blocks back to float32 and compares them
with the GGUF dequantization:

  * every GGUF tensor has its q4nx counterpart (names from configs/inkling2_loop.json)
  * F32 tensors are bit-exact, BF16 tensors equal the bf16 rounding of the GGUF values
  * quantized tensors: 4/8-bit codes bit-identical to the GGUF's; Q4_K values differ
    only by the super-block re-fit of _pack_q4k, Q4_1 / Q8_0 only by bf16 scales
  * Q4_1 column padding (expert down_proj, 1152 -> 1280) is all zero

Usage:
    ./verify_inkling2_q4nx.py --gguf model.gguf --q4nx out/model.q4nx [--layers 0,5,6,17,23] [--experts 0,77,127]
"""
import argparse
import json
import os
import re

import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType, dequantize
from safetensors import safe_open

from verify_hy_mem_q4nx import (ROWS, COLS, GROUP, GROUPS, Q4K_BLOCK_BYTES, Q8_BLOCK_BYTES,
                                decode_q4k, decode_q8, bf16_bytes_to_f32, blocks_to_matrix,
                                gguf_q4k_codes, gguf_q8_codes, stats)

Q41_BLOCK_BYTES = ROWS * GROUPS * 2 * 2 + ROWS * COLS // 2                   # 5120


def decode_q41(blob: np.ndarray):
    """[P, U, 5120] int8 -> (weights, codes).  bf16 scales[8 groups][32 rows], bf16 mins
    (same shape), then uint4 q[256 cols][16 bytes] with rows paired per byte (even row = low nibble).
    w = s * q + m."""
    P, U, _ = blob.shape
    b = blob.reshape(P * U, Q41_BLOCK_BYTES).view(np.uint8)
    n = GROUPS * ROWS * 2
    s = bf16_bytes_to_f32(b[:, :n]).reshape(-1, GROUPS, ROWS)
    m = bf16_bytes_to_f32(b[:, n:2 * n]).reshape(-1, GROUPS, ROWS)
    qb = b[:, 2 * n:].reshape(-1, COLS, ROWS // 2)
    q = np.stack([qb & 0x0F, qb >> 4], axis=-1).reshape(-1, COLS, ROWS).transpose(0, 2, 1)
    scale = np.repeat(s.transpose(0, 2, 1), GROUP, axis=2)
    minv = np.repeat(m.transpose(0, 2, 1), GROUP, axis=2)
    w = scale * q.astype(np.float32) + minv
    return blocks_to_matrix(w, P, U), blocks_to_matrix(q, P, U)


def gguf_q41_codes(data: np.ndarray, cols: int) -> np.ndarray:
    # block_q4_1: fp16 d, fp16 m, 16 bytes; low nibbles are elements 0..15, high 16..31
    qb = data.view(np.uint8).reshape(-1, 20)[:, 4:]
    return np.concatenate([qb & 0x0F, qb >> 4], axis=1).reshape(-1, cols)


DECODERS = {
    GGMLQuantizationType.Q4_K: (Q4K_BLOCK_BYTES, decode_q4k, gguf_q4k_codes),
    GGMLQuantizationType.Q4_1: (Q41_BLOCK_BYTES, decode_q41, gguf_q41_codes),
    GGMLQuantizationType.Q8_0: (Q8_BLOCK_BYTES, decode_q8, gguf_q8_codes),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--q4nx", required=True)
    ap.add_argument("--layers", default="0,5,6,17,23", help="layers whose quantized tensors are decoded")
    ap.add_argument("--experts", default="0,77,127", help="experts decoded per MoE tensor")
    args = ap.parse_args()
    layers = {int(x) for x in args.layers.split(",")}
    experts = [int(x) for x in args.experts.split(",")]

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "inkling2_loop.json")
    cfg = json.load(open(cfg_path))
    reader = GGUFReader(args.gguf)
    g = {t.name: t for t in reader.tensors}
    nx = safe_open(args.q4nx, framework="pt")
    failures = []

    def fail(msg):
        print(f"  FAIL {msg}")
        failures.append(msg)

    # ---- 1. name map: every GGUF tensor -> (q4nx name, target type)
    mapping = {}
    for e in cfg["name_map"].values():
        target = e.get("default_tensor_type", cfg["default_tensor_type"])
        pat = re.compile("^" + re.escape(e["gguf_name"]).replace(r"\{bid\}", r"(\d+)") + "$")
        for name in g:
            mt = pat.match(name)
            if mt:
                q = e["q4nx_name"].format(bid=mt.group(1)) if mt.groups() else e["q4nx_name"]
                mapping[name] = (q, GGMLQuantizationType[target], int(mt.group(1)) if mt.groups() else None)

    print("== inventory")
    for name in sorted(set(g) - set(mapping)):
        fail(f"GGUF tensor {name} has no config entry")
    have = set(nx.keys())
    want = {q for q, _, _ in mapping.values()}
    for q in sorted(want - have):
        fail(f"missing {q}")
    for q in sorted(have - want):
        fail(f"unexpected {q}")
    print(f"  {len(have)} tensors in file, {len(want)} expected")

    # ---- 2. float tensors (all layers)
    print("== F32 / BF16 tensors")
    n_float = 0
    for gname, (qname, target, _) in sorted(mapping.items()):
        if target in DECODERS or qname not in have:
            continue
        t = g[gname]
        ref = torch.from_numpy(np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32))
        got = nx.get_tensor(qname)
        want_dtype = torch.float32 if target == GGMLQuantizationType.F32 else torch.bfloat16
        if got.dtype != want_dtype:
            fail(f"{qname}: dtype {got.dtype}, expected {want_dtype}")
            continue
        if not torch.equal(got, ref.to(want_dtype)):
            fail(f"{qname}: not exact, {stats(got.float().numpy(), ref.numpy())}")
        n_float += 1
    print(f"  {n_float} float tensors checked")

    # ---- 3. quantized tensors (selected layers, selected experts)
    print("== quantized tensors (codes must match the GGUF)")
    for gname, (qname, target, bid) in sorted(mapping.items()):
        if target not in DECODERS or qname not in have or (bid is not None and bid not in layers):
            continue
        t = g[gname]
        if t.tensor_type != target:
            fail(f"{gname}: GGUF {t.tensor_type.name}, config {target.name}")
            continue
        block_bytes, decode, gcodes = DECODERS[target]
        cols = int(t.shape[0])
        blob_all = nx.get_tensor(qname).numpy()
        if blob_all.shape[-1] != block_bytes:
            fail(f"{qname}: block of {blob_all.shape[-1]} B, expected {block_bytes}")
            continue
        ref_all = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(-1, cols)
        codes_all = gcodes(t.data, cols)
        rows_per = int(t.shape[1])
        pieces = experts if len(t.shape) == 3 else [None]
        for e in pieces:
            r0, r1 = (0, ref_all.shape[0]) if e is None else (e * rows_per, (e + 1) * rows_per)
            label = qname if e is None else f"{qname}[expert {e}]"
            w, codes = decode(blob_all[r0 // ROWS: r1 // ROWS])
            pad_w, w, codes = w[:, cols:], w[:, :cols], codes[:, :cols]
            same = np.array_equal(codes.astype(np.int16), codes_all[r0:r1].astype(np.int16))
            print(f"  {'ok  ' if same else 'FAIL'} {target.name:4s} {label:62s} {stats(w, ref_all[r0:r1])}")
            if not same:
                fail(f"{label}: codes differ from GGUF")
            if pad_w.size and np.any(pad_w != 0):
                fail(f"{label}: padding columns are not zero")

    print()
    if failures:
        print(f"{len(failures)} FAILURES")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
