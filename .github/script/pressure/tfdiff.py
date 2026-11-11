#!/usr/bin/env python3
"""Compares two phase S dump folders (out/tf from two platforms): pixels, tower rows and final hidden rows.

    python tfdiff.py <tf dir A> <tf dir B>

Records are matched in order (order.json lists the cases). bf16 values are compared exactly, then by max |diff|
and by cosine similarity per row, so the first stage that departs shows where the platforms stop agreeing.
"""
import json
import os
import struct
import sys

import numpy as np


def bf16(buf, n):
    u = np.frombuffer(buf, dtype=np.uint16, count=n).astype(np.uint32) << 16
    return u.view(np.float32)


def records(path, tag):
    out = []
    if not os.path.exists(path):
        return out
    data = open(path, "rb").read()
    i = 0
    while i < len(data):
        t = data[i:i + 4].decode()
        i += 4
        if t == "TFP1":
            nv, = struct.unpack_from("<i", data, i); i += 4
            grids = [struct.unpack_from("<ii", data, i + 8 * k) for k in range(nv)]; i += 8 * nv
            pd, = struct.unpack_from("<i", data, i); i += 4
            n = sum(h * w for h, w in grids) * pd
            out.append({"grids": grids, "x": bf16(data[i:i + 2 * n], n).reshape(-1, pd)}); i += 2 * n
        elif t in ("TFH1", "TFE1"):
            rows, D = struct.unpack_from("<ii", data, i); i += 8
            ids = None
            if t == "TFH1":
                ids = np.frombuffer(data, dtype=np.int32, count=rows, offset=i); i += 4 * rows
            n = rows * D
            out.append({"ids": ids, "x": bf16(data[i:i + 2 * n], n).reshape(rows, D)}); i += 2 * n
        else:
            raise SystemExit(f"{path}: unknown tag {t!r} at {i - 4}")
    return out


def cmp(a, b):
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    same = np.mean(a == b)
    d = np.abs(a - b)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    cos = np.sum(a * b, axis=-1) / np.maximum(na * nb, 1e-30)
    return (f"identical {same * 100:6.2f}%  max|d| {d.max():.4g}  mean|d| {d.mean():.3g}  "
            f"row cos min {cos.min():.6f} last {cos[-1]:.6f}")


def main():
    A, B = sys.argv[1], sys.argv[2]
    oa = json.load(open(os.path.join(A, "order.json"), encoding="utf-8"))
    ob = json.load(open(os.path.join(B, "order.json"), encoding="utf-8"))
    print(f"A = {oa['platform']}  B = {ob['platform']}")
    cases = [c["case"] for c in oa["order"]]
    imgs = [c for c in cases if c.startswith("img")]
    for kind, fn in (("pixels", "pixels.bin"), ("tower", "imgemb.bin")):
        ra, rb = records(os.path.join(A, fn), kind), records(os.path.join(B, fn), kind)
        for k, name in enumerate(imgs):
            if k < len(ra) and k < len(rb):
                print(f"{name:8s} {kind:7s} {cmp(ra[k]['x'], rb[k]['x'])}")
    ha, hb = records(os.path.join(A, "hidden.bin"), "h"), records(os.path.join(B, "hidden.bin"), "h")
    for k, name in enumerate(cases):
        if k < len(ha) and k < len(hb):
            ids = "same ids" if np.array_equal(ha[k]["ids"], hb[k]["ids"]) else "IDS DIFFER"
            print(f"{name:8s} hidden  {cmp(ha[k]['x'], hb[k]['x'])}  ({len(ha[k]['ids'])} rows, {ids})")
    for x, y in zip(oa["order"], ob["order"]):
        print(f"{x['case']:8s} text {'SAME' if x['text'] == y['text'] else 'differs'}")


if __name__ == "__main__":
    main()
