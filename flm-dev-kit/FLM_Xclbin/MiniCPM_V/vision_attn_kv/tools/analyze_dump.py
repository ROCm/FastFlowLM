#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Find what the NPU actually computed, by testing layout hypotheses offline.

Reads the q/k/v/o dumps written by host.cpp and scores the device output
against a set of candidate references. Whichever candidate matches tells us
which index is wrong, without another 20-minute bitstream rebuild.
"""
import numpy as np, sys, os

B = os.path.join(os.path.dirname(__file__), '..', 'build')
L, Lp, H, DH = map(int, open(os.path.join(B, 'dump_meta.txt')).read().split())
D = H * DH
print(f"L={L} L_padded={Lp} heads={H} head_dim={DH} D={D}")

def rd(n, rows):
    a = np.fromfile(os.path.join(B, f'dump_{n}.bin'), dtype=np.float32)
    return a.reshape(rows, D)

q, k, v = rd('q', L), rd('k', L), rd('v', L)
o = rd('o', Lp)[:L]
scale = 1.0 / np.sqrt(DH)

def attn(qh, kh, vh):
    s = (qh @ kh.T) * scale
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=1, keepdims=True)) @ vh

def rel(a, b):
    return float(np.sqrt(((a - b) ** 2).sum() / max((b ** 2).sum(), 1e-30)))

def head(x, h, perm=None, dh=DH):
    """head h of x; perm optionally permutes the dh columns"""
    s = x[:, h * dh:(h + 1) * dh]
    return s if perm is None else s[:, perm]

print("\n== baseline: head h vs reference head h ==")
for h in range(min(4, H)):
    print(f"  head {h}: relL2 {rel(head(o,h), attn(head(q,h), head(k,h), head(v,h))):.4f}")

# Hypothesis A: the d-column order inside a head is permuted. The mmul C
# operand is dim-block major, so a plausible scramble is an 8x9 <-> 9x8
# transpose of the 72 columns (9 blocks of 8).
print("\n== A: column block transpose within a head (9x8 <-> 8x9) ==")
p89 = np.arange(72).reshape(9, 8).T.reshape(-1)
p98 = np.arange(72).reshape(8, 9).T.reshape(-1)
for name, p in (("9x8->8x9", p89), ("8x9->9x8", p98)):
    r = np.mean([rel(head(o,h), attn(head(q,h),head(k,h),head(v,h))[:, p]) for h in range(4)])
    print(f"  {name}: mean relL2 {r:.4f}")

# Hypothesis B: heads are interleaved differently, i.e. the DDR head stride
# assumed by seq_gen does not match D=1152.
print("\n== B: output head h actually holds reference head g ==")
refs = [attn(head(q,g), head(k,g), head(v,g)) for g in range(H)]
for h in range(min(4, H)):
    best = sorted(((rel(head(o,h), refs[g]), g) for g in range(H)))[:3]
    print(f"  out head {h}: " + ", ".join(f"g={g} {r:.4f}" for r, g in best))

# Hypothesis C: v is read with a different head stride than q/k, which would
# leave the softmax right but mix the value rows.
print("\n== C: correct softmax, but V taken from another head ==")
for h in range(min(4, H)):
    s = (head(q,h) @ head(k,h).T) * scale
    s -= s.max(axis=1, keepdims=True); e = np.exp(s); p = e / e.sum(axis=1, keepdims=True)
    best = sorted(((rel(head(o,h), p @ head(v,g)), g) for g in range(H)))[:3]
    print(f"  out head {h}: " + ", ".join(f"v_head={g} {r:.4f}" for r, g in best))

# Hypothesis D: row (token) permutation of the output.
print("\n== D: output row t actually holds reference row of another t ==")
r0 = refs[0]
for t in [0, 1, 2, 8]:
    d = np.linalg.norm(r0 - o[t, :DH], axis=1)
    print(f"  out row {t}: closest ref rows {np.argsort(d)[:4].tolist()}")
