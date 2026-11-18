#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Reference and scorer for the projection-only test (host/test_proj8.hpp, `run.exe -t proj8`).

  proj8_ref.py gen   --layer N --rows R --dir D [--seed S]   writes D/x_{qkv,o,up,down}.bin (bf16 [t][K]) and D/ref_*.npy (fp32 [t][N])
  proj8_ref.py check --layer N --rows R --dir D [--tol 0.02]  reads D/y_*.bin written by the host and scores them

Real Bolt weights from the shipped model.q4nx (layer N): qkv = [q;k;v] (q only on skip layers), o, up|gate in the round order the
engine streams them (512 rows up, 512 rows gate, ...), down.  Dequantisation = the q4nx block layout of kernels/q4_k.h (validated by the
one-tile toy used for the projection core study).  The reference is fp32 on the dequantised weights and bf16-rounded x; the device result is bf16.
"""
import argparse, json, os, sys
import numpy as np
import torch
from safetensors import safe_open

RB, CB, GS, PS = 32, 256, 32, 16
MODEL = os.environ.get('BOLT_Q4NX', 'q4nx_files/Gemma4-E4B-IT-NPU2/model.q4nx')   # the nemotron3.5-bolt:e4b package
D, I, NH, NKV, DH_G, DH_S, NL, SHARED, PERIOD = 2560, 10240, 8, 2, 512, 256, 42, 18, 6


def unpack_q4nx(blob, rows, cols):
    n_chunks = (rows // RB) * (cols // CB)
    b = blob.reshape(n_chunks, -1)
    ng = CB // GS; dsz = ng * RB * 2
    d = np.ascontiguousarray(b[:, :dsz]).view(np.uint16).reshape(n_chunks, ng, RB)
    m = np.ascontiguousarray(b[:, dsz:2 * dsz]).view(np.uint16).reshape(n_chunks, ng, RB)
    f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
    d, m = f(d), f(m)
    qs = b[:, 2 * dsz:].reshape(n_chunks, RB // PS, CB, PS // 2)
    q = np.empty((n_chunks, RB // PS, CB, PS // 2, 2), np.uint8)
    q[..., 0] = qs & 0xF; q[..., 1] = qs >> 4
    q = q.reshape(n_chunks, RB // PS, CB, PS).transpose(0, 1, 3, 2).reshape(n_chunks, RB, CB)
    w = q.astype(np.float32).reshape(n_chunks, RB, ng, GS) * d.transpose(0, 2, 1)[..., None] + m.transpose(0, 2, 1)[..., None]
    w = w.reshape(n_chunks, RB, CB)
    return w.reshape(rows // RB, cols // CB, RB, CB).transpose(0, 2, 1, 3).reshape(rows, cols)


def bf16bits(a):
    t = torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(torch.bfloat16)
    return t.view(torch.int16).numpy().view(np.uint16)


def bf16val(a):
    return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(torch.bfloat16).float().numpy()


def from_bf16bits(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def layer_info(layer):
    swa = (layer % PERIOD) < (PERIOD - 1)
    skip = layer >= NL - SHARED
    dh = DH_S if swa else DH_G
    return dict(swa=swa, skip=skip, dq=dh * NH, dk=dh * NKV)


def tensor(f, name, rows, cols):
    blob = np.asarray(f.get_tensor(name)).view(np.uint8).reshape(-1)
    return unpack_q4nx(blob, rows, cols)


def phases(layer):
    li = layer_info(layer)
    qkv_rows = li['dq'] if li['skip'] else li['dq'] + 2 * li['dk']
    return li, [("qkv", D, qkv_rows), ("o", li['dq'], D), ("up", D, 2 * I), ("down", I, D)]


def weights(layer):
    li, ph = phases(layer)
    p = f"model.layers.{layer}."
    out = {}
    with safe_open(MODEL, 'np') as f:
        q = tensor(f, p + "self_attn.q_proj.weight", li['dq'], D)
        if li['skip']:
            out["qkv"] = q
        else:
            k = tensor(f, p + "self_attn.k_proj.weight", li['dk'], D)
            v = tensor(f, p + "self_attn.v_proj.weight", li['dk'], D)
            out["qkv"] = np.concatenate([q, k, v], 0)
        out["o"] = tensor(f, p + "self_attn.o_proj.weight", D, li['dq'])
        up = tensor(f, p + "mlp.up_proj.weight", I, D)
        gate = tensor(f, p + "mlp.gate_proj.weight", I, D)
        ug = np.empty((2 * I, D), np.float32)
        for c in range(I // 512):       # engine round order: 512 up rows, 512 gate rows, ...
            ug[(2 * c) * 512:(2 * c + 1) * 512] = up[c * 512:(c + 1) * 512]
            ug[(2 * c + 1) * 512:(2 * c + 2) * 512] = gate[c * 512:(c + 1) * 512]
        out["up"] = ug
        out["down"] = tensor(f, p + "mlp.down_proj.weight", D, I)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--layer", type=int, default=5); ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--dir", required=True); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--tol", type=float, default=0.025)
    ap.add_argument("--phases", default="qkv,o,up,down")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    li, ph = phases(a.layer)
    want = a.phases.split(",")
    if a.cmd == "gen":
        W = weights(a.layer)
        rng = np.random.default_rng(a.seed)
        meta = dict(layer=a.layer, rows=a.rows, swa=li['swa'], skip=li['skip'], phases=[(n, k, nn) for n, k, nn in ph])
        for n, k, nn in ph:
            if n not in want: continue
            x = bf16val(rng.standard_normal((a.rows, k)).astype(np.float32))
            bf16bits(x).tofile(f"{a.dir}/x_{n}.bin")
            ref = x @ W[n].T
            np.save(f"{a.dir}/ref_{n}.npy", ref)
            print(f"gen {n}: x [{a.rows}][{k}] -> ref [{a.rows}][{nn}] rms {np.sqrt((ref**2).mean()):.3f}")
        json.dump(meta, open(f"{a.dir}/meta.json", "w"))
    else:
        bad = 0
        for n, k, nn in ph:
            if n not in want: continue
            fn = f"{a.dir}/y_{n}.bin"
            if not os.path.exists(fn):
                print(f"{n}: MISSING {fn}"); bad += 1; continue
            y = from_bf16bits(np.fromfile(fn, np.uint16)).reshape(a.rows, nn)
            ref = np.load(f"{a.dir}/ref_{n}.npy")
            rel = [float(np.linalg.norm(y[t] - ref[t]) / max(np.linalg.norm(ref[t]), 1e-9)) for t in range(a.rows)]
            cos = [float(y[t] @ ref[t] / max(np.linalg.norm(y[t]) * np.linalg.norm(ref[t]), 1e-9)) for t in range(a.rows)]
            nan = bool(np.isnan(y).any())
            ok = (max(rel) <= a.tol) and not nan
            bad += 0 if ok else 1
            print(f"{n:5s} [{a.rows}x{nn}] relL2 max {max(rel):.4f} mean {np.mean(rel):.4f}  cos min {min(cos):.6f}  nan={nan}  {'PASS' if ok else 'FAIL'}")
        print("proj8 check:", "PASS" if bad == 0 else f"FAIL ({bad})")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
