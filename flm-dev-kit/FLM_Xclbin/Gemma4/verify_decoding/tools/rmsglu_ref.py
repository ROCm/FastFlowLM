#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Reference and scorer for the rms/GLU stage design (host/test_rmsglu.hpp, `run.exe -t rmsglu`, DESIGN=rmsglu).

  rmsglu_ref.py gen   --layer N --dir D [--seed S]   writes D/x_rms.bin ([8][D] bf16 layer input), D/x_o.bin ([8][DQ] bf16, stands in for the attention
                                                     output) and the fp32/bf16-stepped reference chain D/ref_*.npy
  rmsglu_ref.py check --layer N --dir D [--tol 0.03]  scores D/y_qkv.bin (qkv projection out, [8][N_qkv]) and D/y_out.bin ([8][D] layer output rows)

The chain mirrors the device (bf16 rounding where the device stores bf16, fp32 inside a stage), with the real Bolt layer weights and norm weights:
  n1 = n_in(x); qkv = Wqkv n1 | o = Wo x_o; h1 = x + n_pa(o); n2 = n_pff(h1); up|gate = Wug n2; hid = gelu_tanh(gate) * up; d = n_pf(Wdown hid); h2 = h1 + d
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proj8_ref as P
from safetensors import safe_open

D, I = P.D, P.I
EPS = 1e-6


def norm_w(layer):
    with safe_open(P.MODEL, 'pt') as f:
        return [f.get_tensor(f"model.layers.{layer}.{n}.weight").float().numpy()
                for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm")]


def nrm(v, w):
    """y = bf16(bf16-weighted v * 1/sqrt(mean(v^2) + eps)), one row at a time like the kernel"""
    ms = (v.astype(np.float64) ** 2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + EPS)
    return P.bf16val((v * w).astype(np.float32) * inv.astype(np.float32))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--layer", type=int, default=5); ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--dir", required=True); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--tol", type=float, default=0.03)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    li, ph = P.phases(a.layer)
    n_qkv = ph[0][2]; dq = li['dq']
    L = a.rows
    if a.cmd == "gen":
        W = P.weights(a.layer)
        wn = norm_w(a.layer)
        rng = np.random.default_rng(a.seed)
        x = P.bf16val(rng.standard_normal((L, D)).astype(np.float32))
        xo = P.bf16val((0.5 * rng.standard_normal((L, dq))).astype(np.float32))
        P.bf16bits(x).tofile(f"{a.dir}/x_rms.bin")
        P.bf16bits(xo).tofile(f"{a.dir}/x_o.bin")
        n1 = nrm(x, wn[0])
        qkv = n1 @ W["qkv"].T
        o = P.bf16val(xo @ W["o"].T)
        h1 = P.bf16val(x + nrm(o, wn[1]))
        n2 = nrm(h1, wn[2])
        ug = P.bf16val(n2 @ W["up"].T)
        up = np.concatenate([ug[:, (2 * c) * 512:(2 * c + 1) * 512] for c in range(I // 512)], 1)
        gate = np.concatenate([ug[:, (2 * c + 1) * 512:(2 * c + 2) * 512] for c in range(I // 512)], 1)
        hid = P.bf16val(gelu_tanh(gate) * up)
        down = P.bf16val(hid @ W["down"].T)
        h2 = P.bf16val(h1 + nrm(down, wn[3]))
        for k, v in dict(qkv=qkv, o=o, h1=h1, n2=n2, ug=ug, hid=hid, down=down, h2=h2).items():
            np.save(f"{a.dir}/ref_{k}.npy", v)
        json.dump(dict(layer=a.layer, rows=L, swa=li['swa'], skip=li['skip'], n_qkv=n_qkv, dq=dq), open(f"{a.dir}/meta.json", "w"))
        print(f"gen layer {a.layer}: qkv rms {np.sqrt((qkv**2).mean()):.3f}  h2 rms {np.sqrt((h2**2).mean()):.3f}  |d|/|h1| {np.linalg.norm(h2-h1)/np.linalg.norm(h1):.3f}")
        return
    bad = 0

    def score(name, y, ref, tol):
        nonlocal bad
        rel = [float(np.linalg.norm(y[t] - ref[t]) / max(np.linalg.norm(ref[t]), 1e-9)) for t in range(L)]
        cos = [float(y[t] @ ref[t] / max(np.linalg.norm(y[t]) * np.linalg.norm(ref[t]), 1e-9)) for t in range(L)]
        nan = bool(np.isnan(y).any())
        ok = max(rel) <= tol and min(cos) >= 0.9995 and not nan
        bad += 0 if ok else 1
        print(f"{name:12s} [{L}x{y.shape[1]}] relL2 max {max(rel):.4f} mean {np.mean(rel):.4f}  cos min {min(cos):.6f}  nan={nan}  {'PASS' if ok else 'FAIL'}")

    def depermute(fn, rounds):      # raw rounds [round][b = 16][t][32] -> token-major [t][rounds * 512]
        raw = P.from_bf16bits(np.fromfile(fn, np.uint16))[: rounds * 4096]
        return raw.reshape(rounds, 16, L, 32).transpose(2, 0, 1, 3).reshape(L, rounds * 512)

    for name, fn, n, rf in (("qkv", f"{a.dir}/y_qkv.bin", n_qkv, "ref_qkv"), ("h2 (layer out)", f"{a.dir}/y_out.bin", D, "ref_h2")):
        if not os.path.exists(fn):
            print(f"{name}: MISSING {fn}"); bad += 1; continue
        score(name, P.from_bf16bits(np.fromfile(fn, np.uint16)).reshape(L, n), np.load(f"{a.dir}/{rf}.npy"), a.tol)
    if os.path.exists(f"{a.dir}/tap_o.raw"):
        W = P.weights(a.layer)
        score("o proj", depermute(f"{a.dir}/tap_o.raw", 5), np.load(f"{a.dir}/ref_o.npy"), a.tol)
        ug = depermute(f"{a.dir}/tap_ug.raw", 40)
        score("up|gate (ph2)", ug, np.load(f"{a.dir}/ref_ug.npy"), a.tol)
        up = np.concatenate([ug[:, (2 * c) * 512:(2 * c + 1) * 512] for c in range(I // 512)], 1)
        gate = np.concatenate([ug[:, (2 * c + 1) * 512:(2 * c + 2) * 512] for c in range(I // 512)], 1)
        hid_dev = P.bf16val(gelu_tanh(gate) * up)                 # the GLU applied to the device's own up|gate
        down_exp = P.bf16val(hid_dev @ W["down"].T)
        score("glu+down", depermute(f"{a.dir}/tap_down.raw", 5), down_exp, a.tol)
    print("rmsglu check:", "PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
