#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""check_lm8.py DUMP_DIR [model.q4nx]: exact fp32 reference of the lm_head (rmsnorm*w -> Q4NX GEMM -> softcap 30) from the real
Bolt weights, compared with the dumped logits of the 8-row xclbin and of the single-row xclbin (host.cpp --dump)."""
import sys, numpy as np, torch
from safetensors import safe_open
RB, CB, GS, PS = 32, 256, 32, 16
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
    return w.reshape(rows // RB, cols // CB, RB, CB).transpose(0, 2, 1, 3).reshape(rows, cols)
def rd(n): return torch.from_numpy(np.fromfile(n, dtype=np.int16)).view(torch.bfloat16).float().numpy()
d = sys.argv[1]; path = sys.argv[2] if len(sys.argv) > 2 else '../verify_decoding/q4nx_files/Gemma4-E4B-IT-NPU2/model.q4nx'
D, V = 2560, 262144
h = rd(f'{d}/hidden_A.bin').reshape(8, D); nw = rd(f'{d}/nw.bin')
L8 = rd(f'{d}/logits8_A.bin').reshape(8, V); L1 = rd(f'{d}/logits1_A.bin').reshape(8, V) if len(sys.argv) < 4 else None
xn = h * nw[None, :] / np.sqrt((h ** 2).mean(1, keepdims=True) + 1e-6)
xn = torch.from_numpy(xn).to(torch.bfloat16).float().numpy()          # the kernels hand bf16 activations to the matmul
f = safe_open(path, 'np'); blob = np.asarray(f.get_tensor('lm_head.weight')).view(np.uint8).reshape(-1)
ref = np.zeros((8, V), np.float32)
CH = 8192
for r0 in range(0, V, CH):
    Wc = unpack_q4nx(blob[r0 * D * 5 // 8:(r0 + CH) * D * 5 // 8].copy(), CH, D)
    ref[:, r0:r0 + CH] = xn @ Wc.T
ref_cap = 30 * np.tanh(ref / 30)
def stats(name, L):
    out = []
    for t in range(8):
        e = L[t] - ref_cap[t]; rl2 = np.linalg.norm(e) / np.linalg.norm(ref_cap[t])
        cos = float(L[t] @ ref_cap[t] / (np.linalg.norm(L[t]) * np.linalg.norm(ref_cap[t])))
        top = np.argsort(-ref[t])[:2]
        out.append(rl2)
        print(f'{name} row {t}: relL2={rl2:.5f} cos={cos:.6f} argmax={int(L[t].argmax())} exact_argmax={int(ref_cap[t].argmax())} top2_margin={float(ref[t][top[0]] - ref[t][top[1]]):.4f}')
    return np.array(out)
e8 = stats('L8', L8)
if L1 is not None:
    e1 = stats('L1', L1)
    ratio = e8 / e1
    print('relL2(L8 vs exact) / relL2(L1 vs exact): per row', np.round(ratio, 3), ' max', ratio.max())
    ok = bool((e8 <= 1.4 * e1 + 0.003).all())
    print('ACCURACY', 'PASS' if ok else 'FAIL', '(L8 error <= 1.4 x L1 error + 0.003 on every row)')
    sys.exit(0 if ok else 1)
