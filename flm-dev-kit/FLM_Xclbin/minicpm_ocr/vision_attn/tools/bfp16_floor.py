#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Is the residual error the bfp16 mmul floor, or a bug?

The kernel is built with -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16, so every
mmul<8,8,8,bf16> quantises both operands onto a shared-exponent-per-8 grid.
This models that and reports the relL2 a *correct* implementation would score,
to compare against what the device actually scored.

The mantissa is 7 bits, not the 8 the docs imply - fitted by matching the
measured bias, which dropped device-vs-model from 0.11 to 0.003.
"""
import numpy as np, os

B = os.path.join(os.path.dirname(__file__), '..', 'build')
L, Lp, H, DH = map(int, open(os.path.join(B, 'dump_meta.txt')).read().split())
D = H * DH
rd = lambda n, r: np.fromfile(os.path.join(B, f'dump_{n}.bin'), dtype=np.float32).reshape(r, D)
q, k, v, o = rd('q', L), rd('k', L), rd('v', L), rd('o', Lp)[:L]
scale = 1.0 / np.sqrt(DH)


def to_bf16(x):
    """round-to-nearest-even fp32 -> bf16 -> fp32"""
    u = x.astype(np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return (((u + r) & 0xFFFF0000)).view(np.float32)


def floor_bf16(x):
    """accum->bf16. With conv_even rounding set in the kernel this is
    round-to-nearest-even, not the truncation the name suggests."""
    return to_bf16(x)


def bfp16(x, axis=-1, block=8):
    """shared-exponent-per-8 with 8-bit mantissa, along `axis` (the reduction)"""
    x = np.asarray(x, np.float32)
    sh = list(x.shape)
    n = sh[axis]
    pad = (-n) % block
    if pad:
        pw = [(0, 0)] * x.ndim
        pw[axis] = (0, pad)
        x = np.pad(x, pw)
    xm = np.moveaxis(x, axis, -1)
    s = xm.shape
    xb = xm.reshape(*s[:-1], -1, block)
    amax = np.abs(xb).max(axis=-1, keepdims=True)
    e = np.where(amax > 0, np.floor(np.log2(amax + 1e-45)), 0.0)
    # 7 mantissa bits within the block, fitted against the device.
    step = np.exp2(e - 6.0)
    # The kernel now calls aie::set_rounding(conv_even), so the quantisation is
    # round-to-nearest-even, not the floor it used to be. Flooring here made the
    # model predict a constant negative bias the device no longer has.
    xq = np.rint(xb / step) * step
    xq = xq.reshape(*s)
    xq = np.moveaxis(xq, -1, axis)
    if pad:
        sl = [slice(None)] * xq.ndim
        sl[axis] = slice(0, n)
        xq = xq[tuple(sl)]
    return xq


def rel(a, b):
    return float(np.sqrt(((a - b) ** 2).sum() / (b ** 2).sum()))


def attn_exact(h):
    qh, kh, vh = q[:, h*DH:(h+1)*DH], k[:, h*DH:(h+1)*DH], v[:, h*DH:(h+1)*DH]
    s = (qh @ kh.T) * scale
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(1, keepdims=True)) @ vh


def attn_bfp16(h):
    """same dataflow as the kernel: bfp16 mmul for QK and for P@V, bf16 P"""
    qh, kh, vh = q[:, h*DH:(h+1)*DH], k[:, h*DH:(h+1)*DH], v[:, h*DH:(h+1)*DH]
    s = (bfp16(qh, 1) @ bfp16(kh, 1).T) * scale
    s = floor_bf16(s)
    s -= s.max(1, keepdims=True)
    e = np.exp2(floor_bf16(s * np.float32(1.4426950408889634)))
    e = floor_bf16(e)
    p = e / e.sum(1, keepdims=True)
    return bfp16(floor_bf16(p), 1) @ bfp16(vh, 0)


print(f"L={L} heads={H} head_dim={DH}\n")
print(f"{'head':>5} {'device vs exact':>16} {'bfp16model vs exact':>21} {'device vs bfp16model':>22}")
for h in range(min(6, H)):
    ex = attn_exact(h)
    mo = attn_bfp16(h)
    dv = o[:, h*DH:(h+1)*DH]
    print(f"{h:>5} {rel(dv,ex):>16.4f} {rel(mo,ex):>21.4f} {rel(dv,mo):>22.4f}")

print("\nColumn 2 is the error a correct implementation cannot avoid: the mmuls")
print("run in bfp16 (shared exponent per 8, ~7 mantissa bits) while the reference")
print("is exact fp32. Column 1 within ~2x of it means the port is at that floor.")
print("A constant negative offset here would mean truncation rounding is back;")
print("the kernel sets aie::set_rounding(conv_even) to keep the error zero-mean.")
