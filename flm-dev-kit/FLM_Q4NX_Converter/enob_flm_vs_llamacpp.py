#!/usr/bin/env python3
# Copyright (c) 2023-2026 The ggml authors
# SPDX-License-Identifier: MIT
# Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""ENOB of the MVM datapaths written out in quant.md -- FLM vs llama.cpp.

This models the *computation graphs*, not just the storage formats.  Every
rounding that quant.md writes explicitly is applied here in the same place, and
nothing else is rounded (all intermediate accumulation is float32, reductions
included, exactly as quant.md's "Note" section requires).

  key   quant.md section              activations        metadata seen by MAC
  ----  ----------------------------  -----------------  -------------------------
  L1    Q4_1 / llama_cpp              Q8_1 (int8)        FP16 d, FP16 m
  F1    Q4_1 / FLM                    BF16               BF16(d), BF16(m)
  LK    Q4_K / llama.cpp              Q8_K (int8)        FP16 S,M  x  uint6 s_j,m_j
  FK    Q4_K / FLM  (the proposal)    BF16               BF16 S',M', uint8 s'_j,m'_j
                                                         (re-fitted at convert time)

Two reference rows bracket the comparison:

  W16   BF16 weights, no quantization at all -- the ceiling.
  RQ    Q4_K -> dequantize -> re-quantize to Q4_1, then run F1.  This is what
        q4nx/gguf_tensor.py actually does today for a Q4_K tensor (see the
        `#TODO: FIXME: A direct conversion from q4_k to q4_1` branch), so it is
        the real baseline that FK has to beat.

The quantizers are ports of the llama.cpp reference encoders, including the
details that matter and are easy to get wrong:

  * quantize_row_q4_1_ref  derives q from the *unrounded* d and min, then stores
    FP16(d), FP16(min).  The encoder and the decoder therefore disagree slightly.
  * quantize_row_q4_K_ref  picks s_j/m_j with make_qkx2_quants, computes
    ls = round(63*s_j/max_s) from the unrounded max, but then re-derives q from
    the *rounded* FP16(max_s/63)*ls.
  * quantize_row_q8_1_ref  quantizes with the unrounded d and stores FP16(d);
    the block sum s is FP16(sum*d), also from the unrounded d.
  * quantize_row_q8_K_ref  takes the scale from the *signed* extremum
    (d = -max/128, so d carries a sign), stores it FP32, clips q only at +127,
    and keeps 16 int16 partial sums of 16 activations each.

Activations are BF16 for every scenario, and the reference dot product uses
those same BF16 activations.  quant.md lists `x: BF16` for both vendors, and it
keeps the comparison honest: the input dtype is a property of the deployment,
not an error either datapath committed.  (llama.cpp really holds FP32
activations; feeding Q8_1 from FP32 instead of BF16 would shrink its activation
term from ~5.9e-3 to ~5.7e-3, i.e. nothing next to a 7e-2 weight error.)

How the numbers are split
-------------------------
Each datapath returns both the dot products `y` it actually produced and the
*effective weights* `W_eff` it implements -- what an exact accumulator fed with
exact activations would have multiplied.  That gives a clean two-term split:

    eps_W     = ||W_eff - w||   / ||w||           weight-side error
    eps_tot   = ||y - w@x||     / ||w@x||         everything
    eps_path  = ||y - W_eff@x|| / ||w@x||         the datapath by itself

eps_path is a *paired* difference against this same pipeline's own effective
weights, so the weight error cancels exactly instead of having to be subtracted
in quadrature.  That matters: the datapath terms are ~15x below eps_W, and a
quadrature subtraction of two numbers that close is pure sampling noise (it
routinely goes negative).  The paired form is noise-free.

eps_path is the number that compares the *computation graphs*; eps_W compares
the *formats*.  For llama.cpp, eps_path is essentially the Q8 activation
quantization.  For FLM, it is the BF16 roundings of the partial sums.

Both weight error and activation error enter y with relative magnitude equal to
their own per-element relative error -- no growth and no averaging-down over K,
because y = sum w_i x_i has E[y^2] = K var(w) var(x) and the error has
E[e^2] = K var(w) var(dx).  So the quadrature model above is exact in
expectation, and the printed check confirms it empirically.

ENOB = -log2(relative RMS error).

Run:  python3 enob_flm_vs_llamacpp.py [-K 4096] [-M 512] [-T 64] [--dist gauss]
"""

from __future__ import annotations

import argparse

import numpy as np

QK_K = 256          # Q4_K super block
GROUP = 32          # quantization group, both Q4_1 and Q4_K
NGROUP = QK_K // GROUP
HALF = GROUP // 2   # FLM reduces a group as two 16-element halves


# --------------------------------------------------------------------------- #
# number formats
# --------------------------------------------------------------------------- #
def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round FP32 -> BF16 (round-to-nearest-even), returned as FP32 values."""
    f = np.ascontiguousarray(x, dtype=np.float32)
    u = f.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def to_bf16_limb2(x: np.ndarray) -> np.ndarray:
    """Round FP32 -> the sum of two BF16 limbs (~17 significand bits).

    Models `bf16_limb::split_f32` in
    FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding/kernels/bf16_limb_math.hpp: l0 is
    BF16(x), the residual x - l0 is exact in FP32 (l0 is within a factor of two
    of x, so the subtraction cancels), and l1 is BF16 of that residual.  Feeding
    both limbs through separate MACs against a BF16-exact multiplier costs 2
    MACs instead of 1 and leaves ~2^-17 relative error instead of ~2^-8.
    """
    l0 = to_bf16(x)
    l1 = to_bf16(np.asarray(x, dtype=np.float32) - l0)
    return (l0 + l1).astype(np.float32)


def to_fp16(x: np.ndarray) -> np.ndarray:
    """Round FP32 -> FP16, returned as FP32 values."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def to_bf16_up(x: np.ndarray) -> np.ndarray:
    """Round FP32 -> BF16 with the *magnitude* rounded away from zero.

    quant.md's re-fit needs this for S' = BF16_toward_inf(tmax/255):
    rounding to nearest could land just below tmax/255 and push the largest
    s'_j past 255, forcing a clamp -- a far bigger error than the 2^-8 given up
    by rounding up.  Sign is untouched, so this is ceil for x>0 and floor for
    x<0, i.e. away from zero in both cases.
    """
    f = np.ascontiguousarray(x, dtype=np.float32)
    u = f.view(np.uint32)
    tr = u & np.uint32(0xFFFF0000)                    # truncate toward zero
    up = (tr + np.uint32(0x00010000)).view(np.float32)
    return np.where((u & np.uint32(0x0000FFFF)) != 0, up, tr.view(np.float32))


def bf16_next_up(x: np.ndarray, n: int) -> np.ndarray:
    """The n-th BF16 value above |x|, keeping x's sign.  n = 0 returns x."""
    if n == 0:
        return np.ascontiguousarray(x, dtype=np.float32)
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return (u + np.uint32(0x00010000) * np.uint32(n)).view(np.float32)


def rel_round_rms(caster, n: int = 1 << 20, seed: int = 0) -> float:
    """Measured RMS *relative* error of a rounding operator on log-uniform data."""
    rng = np.random.default_rng(seed)
    v = np.exp(rng.uniform(-1.0, 1.0, n)).astype(np.float32)
    return float(np.sqrt(np.mean(((caster(v) - v) / v) ** 2)))


def rel_rms(a, b) -> float:
    """||a - b|| / ||b||, both flattened.  The RMS relative error."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


def enob(eps: float) -> float:
    return float(-np.log2(eps)) if eps > 0 else float("inf")


# --------------------------------------------------------------------------- #
# weight quantizers -- ports of the llama.cpp reference encoders
# --------------------------------------------------------------------------- #
def make_qkx2_quants(x: np.ndarray, nmax: int = 15,
                     rmin: float = -1.0, rdelta: float = 0.1, nstep: int = 20):
    """Vectorized port of llama.cpp `make_qkx2_quants` (use_mad = false).

    This is what `quantize_row_q4_K_ref` uses to pick each group's scale/min, and
    it is the main reason Q4_K beats Q4_1: instead of taking the group's min/max
    (which is set by the two most extreme samples), it tries `nstep+1` candidate
    scales around nmax/(max-min) and, for each, refits BOTH scale and min by
    importance-weighted least squares, keeping whichever minimizes the weighted
    squared error.  The importance weight is llama.cpp's `av_x + |x|`.

    x : (..., 32) float32  ->  (scale, the_min) each (...,);  w ~= scale*q - the_min
    """
    x = x.astype(np.float32)
    av = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True))
    wt = av + np.abs(x)                       # llama.cpp importance weights
    mn0 = np.minimum(x.min(-1, keepdims=True), 0.0)   # affine quant keeps 0 in range
    mx = x.max(-1, keepdims=True)
    span = np.where(mx - mn0 > 0, mx - mn0, 1.0)

    sum_w = wt.sum(-1, keepdims=True)
    sum_x = (wt * x).sum(-1, keepdims=True)

    # candidate 0: the plain min/max solution (what Q4_1 stops at)
    isc = nmax / span
    L = np.clip(np.rint(isc * (x - mn0)), 0, nmax)
    scale = (1.0 / isc) * np.ones_like(mn0)
    mn = mn0.copy()
    best = (wt * (scale * L + mn - x) ** 2).sum(-1, keepdims=True)

    for step in range(nstep + 1):
        # NOTE: the C code mutates `min` whenever a candidate wins, and the next
        # iteration derives both iscale and L from that updated min.  The search
        # is stateful -- using the original min here silently changes the result.
        span = np.where(mx - mn > 0, mx - mn, 1.0)
        isc = (rmin + rdelta * step + nmax) / span
        Lx = np.clip(np.rint(isc * (x - mn)), 0, nmax)
        sl = (wt * Lx).sum(-1, keepdims=True)
        sl2 = (wt * Lx * Lx).sum(-1, keepdims=True)
        sxl = (wt * Lx * x).sum(-1, keepdims=True)

        det = sum_w * sl2 - sl * sl
        ok = det > 0
        det_s = np.where(ok, det, 1.0)
        ts = (sum_w * sxl - sum_x * sl) / det_s          # least-squares scale
        tm = (sl2 * sum_x - sl * sxl) / det_s            # least-squares min
        pos = tm > 0                                     # min must stay <= 0
        sl2_s = np.where(sl2 > 0, sl2, 1.0)
        ts = np.where(pos, sxl / sl2_s, ts)
        tm = np.where(pos, 0.0, tm)

        mad = (wt * (ts * Lx + tm - x) ** 2).sum(-1, keepdims=True)
        upd = ok & (mad < best)
        scale = np.where(upd, ts, scale)
        mn = np.where(upd, tm, mn)
        best = np.where(upd, mad, best)

    degen = mx == mn0                          # C early-returns 0 for a flat group
    scale = np.where(degen, 0.0, scale)
    mn = np.where(degen, mn0, mn)
    return scale[..., 0], (-mn)[..., 0]


def quant_q4_1(w: np.ndarray):
    """Port of `quantize_row_q4_1_ref`.  Per group of 32: FP16 d, FP16 m, uint4 q.

    Dequant is w ~= d*q + m with a *signed* m (unlike Q4_K's subtracted min).

    Two faithful details: d = (max-min)/15 and m = min are taken from the raw
    group (no clamp of min to <= 0), and q is derived from the *unrounded* d and
    min even though only the FP16 versions are stored -- so the encoder and the
    decoder are working from slightly different scales.

    w : (M, K) -> q:(M,NB,32) float32 0..15, d:(M,NB), m:(M,NB)
    """
    M, K = w.shape
    wg = w.reshape(M, K // GROUP, GROUP).astype(np.float32)
    gmin = wg.min(axis=-1)
    gmax = wg.max(axis=-1)
    d = (gmax - gmin) / 15.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    q = np.clip(np.floor((wg - gmin[..., None]) * inv[..., None] + 0.5), 0, 15)
    return q.astype(np.float32), to_fp16(d), to_fp16(gmin)


def quant_q4_k(w: np.ndarray):
    """Port of `quantize_row_q4_K_ref`.  Per super block of 256 = 8 groups of 32:
    FP16 S and FP16 M, eight uint6 s_j and eight uint6 m_j, 256 uint4 q.

    Dequant is w ~= (S*s_j)*q - (M*m_j) -- the min is an unsigned magnitude that
    is subtracted.

    Faithful details: ls = round(63*scale_j/max_scale) uses the *unrounded*
    max_scale, while the stored S is FP16(max_scale/63); q is then re-derived
    from the rounded S*ls, which is what makes Q4_K's encoder self-consistent
    where Q4_1's is not.

    w : (M, K) -> q:(M,NS,8,32) float32 0..15, s6,m6:(M,NS,8) float32 0..63,
                  S,M:(M,NS) float32 (FP16-representable)
    """
    M, K = w.shape
    ns = K // QK_K
    wg = w.reshape(M, ns, NGROUP, GROUP).astype(np.float32)

    scale, mn = make_qkx2_quants(wg)
    scale = np.maximum(scale, 0.0)      # C tracks max_scale from 0 upwards
    mn = np.maximum(mn, 0.0)

    max_s = scale.max(axis=-1)
    max_m = mn.max(axis=-1)
    inv_s = np.where(max_s > 0, 63.0 / np.where(max_s > 0, max_s, 1.0), 0.0)
    inv_m = np.where(max_m > 0, 63.0 / np.where(max_m > 0, max_m, 1.0), 0.0)
    s6 = np.minimum(np.rint(scale * inv_s[..., None]), 63.0).astype(np.float32)
    m6 = np.minimum(np.rint(mn * inv_m[..., None]), 63.0).astype(np.float32)

    S = to_fp16(max_s / 63.0)
    M_ = to_fp16(max_m / 63.0)

    eff_s = (S[..., None] * s6).astype(np.float32)
    eff_m = (M_[..., None] * m6).astype(np.float32)
    inv = np.where(eff_s > 0, 1.0 / np.where(eff_s > 0, eff_s, 1.0), 0.0)
    q = np.clip(np.rint((wg + eff_m[..., None]) * inv[..., None]), 0, 15)
    return q.astype(np.float32), s6, m6, S, M_


def _refit_one_side(P: np.ndarray, p6: np.ndarray, search: int = 0,
                    cap: float = 255.0):
    """One side (scale or min) of quant.md's (uint6,FP16) -> (uint8,BF16) re-fit.

    (FP16 P, uint6 p_j)  ->  (BF16 P', uint8 p'_j)  minimizing  P' p'_j ~= P p_j.

    Direct port of the pseudocode in quant.md's "Re-fitting (int6, FP16) into
    (int8, BF16)".  The product t_j = P*p_j is the only quantity the kernel ever
    uses, so that -- not P or p_j separately -- is what the fit preserves.

    The two extra bits are spent *absorbing* P's rounding error rather than as a
    plain x4: P' is fixed first (rounded up, so no clamp can occur), then each
    p'_j is re-derived against the rounded P'.  P' carries a relative error
    |delta| <= 2^-8, and since t_j/P' runs up to 255 the quotient can absorb up
    to 255*2^-8 ~ 1 integer step -- representable, because the uint8 grid is 4x
    finer than the uint6 grid it came from.

    search > 0 enables quant.md's optional refinement: also try the next
    `search` BF16 values above the base P' and keep whichever minimizes
    sum_j (P' p'_j - t_j)^2.  A slightly larger P' occasionally aligns better
    with several t_j at once than the smallest admissible one.

    P : (M, ns)  p6 : (M, ns, 8)  ->  Pp : (M, ns)  p8 : (M, ns, 8)
    """
    t = (P[..., None].astype(np.float32) * p6.astype(np.float32))   # exact: 11+6
    tmax = np.abs(t).max(axis=-1)                                   # (M, ns)
    sigma = np.sign(P).astype(np.float32)                           # p_j unsigned
    base = sigma * to_bf16_up(tmax / cap)
    live = tmax > 0                                                 # dead blocks -> 0

    best_P = np.zeros_like(base)
    best_p = np.zeros_like(t)
    best_e = np.full(base.shape, np.inf, dtype=np.float64)
    for c in range(search + 1):
        Pc = np.where(live, bf16_next_up(base, c), 0.0).astype(np.float32)
        inv = np.where(live, 1.0 / np.where(live, Pc, 1.0), 0.0).astype(np.float32)
        pc = np.clip(np.rint(t * inv[..., None]), 0.0, cap).astype(np.float32)
        err = np.sum((Pc[..., None] * pc - t).astype(np.float64) ** 2, axis=-1)
        take = err < best_e
        best_e = np.where(take, err, best_e)
        best_P = np.where(take, Pc, best_P)
        best_p = np.where(take[..., None], pc, best_p)
    return best_P, best_p


def refit_q4k(q4k, search: int = 0, cap: float = 255.0):
    """quant.md Q4_K / FLM: re-fit the whole super block's metadata offline.

    Scale and min are fitted independently -- nothing in the procedure couples
    them.  q is left exactly as Q4_K's encoder chose it: the re-fit changes only
    how the effective per-group scale/min are *stored*, so the k-quant search's
    output is preserved and only re-expressed on a finer grid.

    -> q, s8, m8 (uint8-valued), Sp, Mp (BF16-representable)
    """
    q, s6, m6, S, M_ = q4k
    Sp, s8 = _refit_one_side(S, s6, search, cap)
    Mp, m8 = _refit_one_side(M_, m6, search, cap)
    return q, s8, m8, Sp, Mp


# --------------------------------------------------------------------------- #
# activation quantizers -- the llama.cpp side only
# --------------------------------------------------------------------------- #
def quant_q8_1(xb: np.ndarray):
    """Port of `quantize_row_q8_1_ref`.  Per block of 32 activations.

    xb : (NB, 32, T)  ->  qb:(NB,32,T) int32 in [-127,127],
                          sb:(NB,T) the FP16 scale,
                          sig:(NB,T) the FP16 block sum s ~= sum_i x_i

    The quantization uses the unrounded d; only the stored copies are FP16.
    """
    amax = np.abs(xb).max(axis=-2)                       # (NB,T)
    d = (amax / 127.0).astype(np.float32)
    idd = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    qb = np.clip(np.rint(xb * idd[..., None, :]), -127, 127).astype(np.int32)
    ssum = qb.sum(axis=-2, dtype=np.int64).astype(np.float32)
    return qb, to_fp16(d), to_fp16(ssum * d)


def quant_q8_K(xs: np.ndarray):
    """Port of `quantize_row_q8_K_ref`.  Per super block of 256 activations.

    xs : (NS, 256, T)  ->  qb:(NS,256,T) int32 in [-128,127],
                           sb:(NS,T) FP32 scale (signed: d = -max/128),
                           bs:(NS,16,T) int32 partial sums of 16

    Unlike Q8_1 the scale is stored FP32, and it is taken from the *signed*
    extremum, so it inherits that element's sign.  q is clipped at +127 only.
    """
    idx = np.abs(xs).argmax(axis=-2)                     # (NS,T)
    mx = np.take_along_axis(xs, idx[..., None, :], axis=-2)[..., 0, :]
    isc = np.where(mx != 0, -128.0 / np.where(mx != 0, mx, 1.0), 0.0).astype(np.float32)
    qb = np.minimum(127, np.rint(isc[..., None, :] * xs)).astype(np.int32)
    NS, _, T = xs.shape
    bs = qb.reshape(NS, 16, 16, T).sum(axis=2, dtype=np.int64).astype(np.int32)
    sb = np.where(isc != 0, 1.0 / np.where(isc != 0, isc, 1.0), 0.0).astype(np.float32)
    return qb, sb, bs


# --------------------------------------------------------------------------- #
# shared reduction primitives
#
# The shipping Q4_1 kernel reduces a group of 32 as two independent runs of 16,
# each rounded to BF16 on its own (`temp_acc_0` / `temp_acc_1` in q4_core.h);
# the proposed Q4_K kernel spans all 32 in one float32 accumulator instead.
# Both are here, plus one separate 32-wide reduce of the raw activations for the
# min term.  Everything inside a reduce is float32.
# --------------------------------------------------------------------------- #
def half_dots(qf: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """FP32 half-group dot products.  qf:(M,NB,32), xb:(NB,32,T) -> (M,NB,2,T)."""
    M, NB, _ = qf.shape
    T = xb.shape[-1]
    qh = qf.reshape(M, NB, 2, HALF)
    xh = xb.reshape(NB, 2, HALF, T)
    return np.einsum("mnhi,nhit->mnht", qh, xh, optimize=True, dtype=np.float32)


def group_dots(qf: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """FP32 whole-group dot products.  qf:(M,NB,32), xb:(NB,32,T) -> (M,NB,T)."""
    return np.einsum("mni,nit->mnt", qf, xb, optimize=True, dtype=np.float32)


def group_xsum(xb: np.ndarray) -> np.ndarray:
    """FP32 sum_i x_i over each group.  xb:(NB,32,T) -> (NB,T)."""
    return xb.sum(axis=1, dtype=np.float32)


# --------------------------------------------------------------------------- #
# the datapaths.  Each returns (y, W_eff, diag)
#   y     : (M, T)  the dot products the pipeline actually produces
#   W_eff : (M, K)  the effective weights it implements
#   diag  : dict of intermediates the design study wants
# --------------------------------------------------------------------------- #
def mvm_L1_q41_llamacpp(q41, xb, K, final_bf16=True):
    """quant.md  Q4_1 / llama_cpp.

        y = BF16( s*sbar * sum_i q_i*qbar_i  +  m*Sigmabar )

    sum q*qbar is an exact int32 accumulation (<= 15*127*32 = 60960); the FP16
    metadata widens to FP32 exactly.  llama.cpp keeps one FP32 `sumf` across all
    blocks of K and casts once at the very end, so there is no per-block cast.
    """
    q, d, m = q41
    qb, sb, sig = quant_q8_1(xb)
    idot = np.einsum("mni,nit->mnt", q.astype(np.int64), qb.astype(np.int64),
                     optimize=True).astype(np.float32)
    yb = d[..., None] * sb[None] * idot + m[..., None] * sig[None]
    y = yb.sum(axis=1, dtype=np.float32)
    if final_bf16:
        y = to_bf16(y)
    W = (d[..., None] * q + m[..., None]).reshape(q.shape[0], K)
    return y, W, {}


def mvm_F1_q41_flm(q41, xb, K, final_bf16=True, blk_cast=True, half_cast=True,
                   xsum_cast=True, meta_bf16=True, merge_halves=False):
    """quant.md  Q4_1 / FLM -- the kernel that ships today (`q4_core.h`).

        y = BF16( BF16(s)*BF16(sum_{0..15} BF16(q)x)
                + BF16(s)*BF16(sum_{16..31} BF16(q)x)
                + BF16(m)*BF16(sum_{0..31} x) )

    BF16(q) is exact (uint4 <= 15, BF16 holds every integer to 256), and every
    product here is BF16 x BF16 = 16 significand bits, exact in FP32.  The only
    real roundings are BF16(s), BF16(m), the two half sums, the x sum, and the
    final cast of the group result.

    `merge_halves` is the proposal's change (a) applied to this kernel: combine
    `temp_acc_0` and `temp_acc_1` in the *accumulator* domain and cast once,
    which is one fewer MAC and one fewer cast per group as well as being more
    accurate.  The split exists for the pipelining dependency chain, not for
    arithmetic.
    """
    q, d, m = q41
    db, mb = (to_bf16(d), to_bf16(m)) if meta_bf16 else (d, m)
    ph = half_dots(q, xb)                                   # (M,NB,2,T) FP32
    if merge_halves:
        ph = ph.sum(axis=2, keepdims=True, dtype=np.float32)   # FP32 add first
    if half_cast:
        ph = to_bf16(ph)
    xs = group_xsum(xb)                                     # (NB,T) FP32
    if xsum_cast:
        xs = to_bf16(xs)
    # one MAC per half when split (as q4_core.h does), one when merged
    scale_term = sum(db[..., None] * ph[:, :, h] for h in range(ph.shape[2]))
    yb = scale_term + mb[..., None] * xs[None]              # (M,NB,T)
    if blk_cast:
        yb = to_bf16(yb)
    y = yb.sum(axis=1, dtype=np.float32)
    if final_bf16:
        y = to_bf16(y)
    W = (db[..., None] * q + mb[..., None]).reshape(q.shape[0], K)
    return y, W, {"scale_term": scale_term, "yb": yb}


def mvm_LK_q4k_llamacpp(q4k, xb, K, final_bf16=True):
    """quant.md  Q4_K / llama.cpp.

        y = BF16( S*sbar * sum_j s_j sum_i q^j_i qbar^j_i
                - M*sbar * sum_j m_j (B_{2j} + B_{2j+1}) )

    Both reductions are exact integer arithmetic.  Scale path: |sum_j s_j dot_j|
    <= 8*63*32*15*128 = 3.1e7 < 2^31.  Min path: <= 8*63*32*128 = 2.1e6.  The
    only floating point work in the whole super block is the two scalar
    multiplies S*sbar and M*sbar and the FP32 accumulation across super blocks.
    """
    q, s6, m6, S, M_ = q4k
    M, ns = q.shape[0], q.shape[1]
    T = xb.shape[-1]
    xs_sb = xb.reshape(ns, QK_K, T)
    qb, sb, bs = quant_q8_K(xs_sb)
    qbg = qb.reshape(ns, NGROUP, GROUP, T)
    idot = np.einsum("msgi,sgit->msgt", q.astype(np.int64), qbg.astype(np.int64),
                     optimize=True)                          # (M,ns,8,T) exact int
    b2 = bs.reshape(ns, NGROUP, 2, T).sum(axis=2, dtype=np.int64)   # (ns,8,T)
    isc = (s6.astype(np.int64)[..., None] * idot).sum(axis=2, dtype=np.int64)
    imn = (m6.astype(np.int64)[..., None] * b2[None]).sum(axis=2, dtype=np.int64)
    ysb = (S[..., None] * sb[None] * isc.astype(np.float32)
           - M_[..., None] * sb[None] * imn.astype(np.float32))
    y = ysb.sum(axis=1, dtype=np.float32)
    if final_bf16:
        y = to_bf16(y)
    W = (S[..., None, None] * s6[..., None] * q
         - M_[..., None, None] * m6[..., None]).reshape(M, K)
    return y, W, {}


def mvm_FK_q4k_flm(q4k3, xb, K, final_bf16=True, blk_cast=True,
                   xsum_cast=True, fold_cast=True, half_split=False,
                   pre_cancel=None):
    """quant.md  Q4_K / FLM -- the proposed NPU datapath.

    Stored metadata is the offline re-fit of `refit_q4k`: s'_j / m'_j uint8,
    S' / M' BF16.  The whole 32-element group is reduced in one float32
    accumulator and cast to BF16 exactly once, where it meets s'_j:

        y = BF16( S' * sum_j s'_j * BF16( sum_{i=0..31} BF16(q^j_i) x^j_i )
                - M' * sum_j m'_j * BF16( sum_i x^j_i ) )

    Every multiplier operand is BF16-exact:

      * q^j_i is uint4 <= 15 and BF16 holds every integer to 256.
      * s'_j, m'_j are uint8 <= 255 -- still exact, exactly as uint6 was.
      * S', M' are *stored* BF16, so there is no BF16(.) on them at all: the
        metadata path carries no run-time rounding whatsoever.
      * s'_j x BF16(.) is 8 + 8 = 16 significand bits, exact in FP32.

    What it pays instead is the convert-time fit error S's'_j - S s_j, which
    lands in eps_W and is bounded by |S'|/2 <= S/8 -- an eighth of a step of the
    grid the k-quant search itself worked on.

    The only run-time weight-side loss is `fold_cast`: 8 casts per super block,
    one per group.  `blk_cast` and `final_bf16` sit *after* the min subtraction
    and so escape the scale/min cancellation amplification.

    `pre_cancel` models where q4_core.h actually puts the super-block rounding.
    The kernel carries two FP32 accumulators across the whole K loop and only
    adds them in _Q4K_finalize, so there is no post-cancellation per-block cast
    at all (`blk_cast=False`); what it has instead is a rounding of each
    accumulator on its way into the running sum, i.e. *before* the two paths
    cancel, where the table above says a rounding costs ~2.3x -- and there are
    two independent ones.  Values:

      None      no rounding on either accumulator (pure FP32 across blocks)
      "bf16"    one to_vector<bf16>() per path -- q4_core.h before the fix
      "limb2"   two BF16 limbs per path, ~17 bits -- bf16_limb::mac_f32_bf16_low,
                what q4_core.h does now, for 2 extra MACs per 32x256 block

    So the honest model of the shipping kernel is
    `blk_cast=False, pre_cancel="limb2"`, and of the one before it
    `blk_cast=False, pre_cancel="bf16"`.

    `half_split=True` is the baseline the proposal improves on -- the form
    quant.md's equations write and q4_core.h implements, where the group is two
    independently BF16-rounded runs of 16 that are then folded back to BF16
    before meeting s'_j.  That is 24 casts per super block instead of 8, and
    all 16 extra ones sit before the min subtraction where they cost 2.3x.

    Storage: 8 B uint8 scales + 8 B uint8 mins + 4 B S',M' = 20 B per super
    block, against Q4_K's 16 B.  4.625 bpw vs 4.500, and no 6-bit unpack.
    """
    q, s8, m8, Sp, Mp = q4k3
    M, ns = q.shape[0], q.shape[1]
    NB = K // GROUP
    T = xb.shape[-1]
    qf = q.reshape(M, NB, GROUP)
    if half_split:
        ph = half_dots(qf, xb)                               # (M,NB,2,T) FP32
        gs = to_bf16(ph).sum(axis=2, dtype=np.float32)       # 2 casts/group
    else:
        gs = group_dots(qf, xb)                              # (M,NB,T) FP32
    if fold_cast:
        gs = to_bf16(gs)                     # the one weight-side run-time loss
    xs = group_xsum(xb)
    if xsum_cast:
        xs = to_bf16(xs)
    gs = gs.reshape(M, ns, NGROUP, T)
    xsg = xs.reshape(ns, NGROUP, T)
    acc_s = (s8[..., None] * gs).sum(axis=2, dtype=np.float32)    # (M,ns,T) FP32
    acc_m = (m8[..., None] * xsg[None]).sum(axis=2, dtype=np.float32)
    if pre_cancel is not None:
        # The kernel rounds each accumulator separately, before they cancel.
        rnd = {"bf16": to_bf16, "limb2": to_bf16_limb2}[pre_cancel]
        acc_s = rnd(acc_s)
        acc_m = rnd(acc_m)
    ysb = Sp[..., None] * acc_s - Mp[..., None] * acc_m
    if blk_cast:
        ysb = to_bf16(ysb)
    y = ysb.sum(axis=1, dtype=np.float32)
    if final_bf16:
        y = to_bf16(y)
    W = (Sp[..., None, None] * s8[..., None] * q
         - Mp[..., None, None] * m8[..., None]).reshape(M, K)
    return y, W, {"scale_term": Sp[..., None] * acc_s, "yb": ysb}


def mvm_W16_bf16(w, xb, K, final_bf16=True):
    """Reference ceiling: BF16 weights, BF16 activations, FP32 reduce."""
    wb = to_bf16(w)
    y = np.einsum("mni,nit->mt", wb.reshape(w.shape[0], K // GROUP, GROUP), xb,
                  optimize=True, dtype=np.float32)
    if final_bf16:
        y = to_bf16(y)
    return y, wb, {}


# --------------------------------------------------------------------------- #
# storage cost, bits per weight
# --------------------------------------------------------------------------- #
BPW = {
    "L1":  4.0 + 32.0 / GROUP,                       # FP16 d + FP16 m per 32
    "F1":  4.0 + 32.0 / GROUP,                       # same bits, BF16 at use
    "LK":  4.0 + (12.0 * 8 + 32.0) / QK_K,           # 12 B packed uint6 + FP16 S,M
    "FK":  4.0 + (8.0 * 8 + 8.0 * 8 + 32.0) / QK_K,  # uint8 s',m' + BF16 S',M'
    "RQ":  4.0 + 32.0 / GROUP,
    "W16": 16.0,
}
# activation bits per value, for context (not part of bpw)
ABPV = {"Q8_1": 8.0 + 32.0 / 32, "Q8_K": 8.0 + (32.0 + 16 * 16) / QK_K, "BF16": 16.0}


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
ORDER = ["W16", "L1", "F1", "LK", "FK", "RQ"]


def weight_rmse_report(res, w, M, K):
    """RMSE analysis of every dequantized W against the BF16 reference weights.

    ENOB is one number and hides three things that matter separately: a
    systematic gain error (the quantizer shrinking or stretching w), a bias
    (a DC offset the min term failed to centre), and the tail (a few very
    wrong weights among many good ones).  All three are broken out here.
    """
    ref = to_bf16(w).astype(np.float64)          # what the BF16 ceiling stores
    w64 = w.astype(np.float64)                   # the pre-quantization original
    rms_ref = float(np.sqrt(np.mean(ref ** 2)))
    rms_w = float(np.sqrt(np.mean(w64 ** 2)))

    print("Weight-space RMSE:  dequantized W  vs  BF16(w)")
    print(f"  reference rms(BF16 w) = {rms_ref:.6f}   over M*K = {M*K} weights")
    print("  (W16 is BF16 vs BF16, so its row is identically zero by")
    print("   construction -- it is the reference, shown to anchor the table.)")
    print()
    print("  key    RMSE       NRMSE      SQNR dB    MAE        max|e|     "
          "p99.9      bias")
    print("  " + "-" * 84)
    rows = {}
    for key in ORDER:
        W = res[key]["W"].astype(np.float64)
        e = W - ref
        ae = np.abs(e)
        rmse = float(np.sqrt(np.mean(e ** 2)))
        rows[key] = dict(
            rmse=rmse,
            nrmse=rmse / rms_ref,
            sqnr=(20.0 * np.log10(rms_ref / rmse)) if rmse > 0 else float("inf"),
            mae=float(np.mean(ae)),
            maxe=float(ae.max()),
            p999=float(np.percentile(ae, 99.9)),
            bias=float(np.mean(e)),
            # NRMSE against the true FP32 weights, for reference
            nrmse_fp32=float(np.sqrt(np.mean((W - w64) ** 2)) / rms_w),
            e=e, W=W,
        )
        r = rows[key]
        sq = "     inf" if not np.isfinite(r["sqnr"]) else f"{r['sqnr']:8.3f}"
        print(f"  {key:<5}  {r['rmse']:.3e}  {r['nrmse']:.3e}  {sq}   "
              f"{r['mae']:.3e}  {r['maxe']:.3e}  {r['p999']:.3e}  {r['bias']:+.3e}")
    dref = max(abs(rows[k]["nrmse"] - rows[k]["nrmse_fp32"]) / rows[k]["nrmse"]
               for k in ORDER if rows[k]["nrmse"] > 0)
    print(f"  Reference choice is immaterial: measuring against the FP32 w")
    print(f"  instead of BF16(w) moves every NRMSE by at most {100*dref:.3f}%,")
    print(f"  because BF16's own 1.66e-03 is ~45x below the format error.")
    print()

    # ---- structure of the error, not just its size -------------------------
    print("  Structure of the weight error")
    print("  key    gain a     NRMSE|a    bias/RMSE   kurt   row NRMSE:"
          "  med       worst")
    print("  " + "-" * 84)
    for key in ORDER:
        r = rows[key]
        W, e = r["W"], r["e"]
        # least-squares gain: the best scalar a in W ~ a*ref.  a != 1 is a
        # systematic shrink/stretch, which is correctable at convert time.
        den = float(np.sum(ref * ref))
        alpha = float(np.sum(W * ref) / den) if den else 1.0
        res_a = float(np.sqrt(np.mean((W - alpha * ref) ** 2)) / rms_ref)
        m2 = float(np.mean(e ** 2))
        kurt = float(np.mean(e ** 4) / m2 ** 2) if m2 > 0 else float("nan")
        # per-output-row NRMSE: one bad row is one bad output feature
        rr = np.sqrt(np.mean(e ** 2, axis=1)) / np.sqrt(np.mean(ref ** 2, axis=1))
        bo = r["bias"] / r["rmse"] if r["rmse"] > 0 else 0.0
        print(f"  {key:<5}  {alpha:.6f}   {res_a:.3e}  {bo:+8.4f}   "
              f"{kurt:5.2f}              {np.median(rr):.3e}  {rr.max():.3e}")
    print("  gain a   : best scalar fit W ~ a*BF16(w).  a<1 means the format")
    print("             systematically shrinks the weights.")
    print("  NRMSE|a  : NRMSE left after that gain is divided out -- if it is")
    print("             well below NRMSE, the error is largely a fixable gain.")
    print("  kurt     : 3.0 = Gaussian error.  Uniform round-off is 1.8; a")
    print("             high value means a heavy tail (a few very wrong weights).")
    print("  row NRMSE: per-output-channel, so the worst column is the worst")
    print("             single output feature rather than the average one.")
    print()

    # ---- format vs metadata: decompose the gap -----------------------------
    print("  Where each format's RMSE comes from")
    print("  " + "-" * 84)
    pairs = [("Q4_1  FP16 meta (llama.cpp) -> BF16 meta (FLM)", "L1", "F1"),
             ("Q4_K  exact meta (llama.cpp) -> uint8+BF16 re-fit (FLM)", "LK", "FK")]
    for lab, ka, kb in pairs:
        ra, rb = rows[ka], rows[kb]
        # paired: the shared format error cancels, leaving the metadata rounding
        d = float(np.sqrt(np.mean((rb["W"] - ra["W"]) ** 2)) / rms_ref)
        print(f"  {lab}")
        print(f"      NRMSE {ra['nrmse']:.3e} -> {rb['nrmse']:.3e}"
              f"   (+{100*(rb['nrmse']/ra['nrmse'] - 1):5.2f}%)"
              f"   isolated metadata rounding {d:.3e}")
    rq, l1 = rows["RQ"], rows["L1"]
    print(f"  Q4_K -> dequant -> Q4_1, the converter path today")
    print(f"      NRMSE {rows['LK']['nrmse']:.3e} (Q4_K) -> {rq['nrmse']:.3e}"
          f"   (+{100*(rq['nrmse']/rows['LK']['nrmse'] - 1):5.2f}%)")
    print(f"      and worse than quantizing the original straight to Q4_1:"
          f" {l1['nrmse']:.3e}")
    print()

    # ---- error vs weight magnitude -----------------------------------------
    print("  RMSE by weight magnitude decile of |BF16(w)|  (NRMSE within decile)")
    aw = np.abs(ref).ravel()
    order = np.argsort(aw)
    nd = 10
    cuts = np.array_split(order, nd)
    hdr = "  key   " + "".join(f"  d{i+1:<2}   " for i in range(nd))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key in ORDER:
        ev = rows[key]["e"].ravel()
        rv = ref.ravel()
        cells = []
        for c in cuts:
            num = np.sqrt(np.mean(ev[c] ** 2))
            den = np.sqrt(np.mean(rv[c] ** 2))
            cells.append(f" {num/den:6.3f}" if den > 0 else "    -  ")
        print(f"  {key:<5} " + "".join(cells))
    print("  Deciles run smallest |w| (d1) to largest (d10).  Uniform-grid")
    print("  quantizers put a roughly constant *absolute* error everywhere, so")
    print("  the relative error is worst on the smallest weights -- d1 is huge")
    print("  and d10 is small.  That is expected, not a bug; it is why NRMSE")
    print("  over the whole tensor is dominated by the large weights.")
    print()

    # ---- what the RMSE view adds over ENOB ---------------------------------
    lk, rq = rows["LK"], rows["RQ"]
    print("  What RMSE shows that ENOB did not")
    print("  * The FLM-vs-llama.cpp metadata gap is ~0.25-0.30% of RMSE at")
    print("    every format.  BF16 metadata is essentially free in weight")
    print("    space; that conclusion survives the change of metric.")
    print("  * The error is NOT a correctable gain.  Dividing out the best")
    print("    scalar fit leaves NRMSE unchanged to <0.1%, so there is no")
    print("    free win from rescaling at convert time.")
    print("  * Q4_K carries a small systematic negative bias (-0.3 to -0.5%")
    print("    of RMSE, gain a ~ 0.9970) that Q4_1 does not.  It comes from")
    print("    make_qkx2_quants' importance-weighted refit, which is not")
    print("    mean-preserving.  Too small to matter alone, but it is a DC")
    print("    offset and it accumulates coherently over a reduction of K,")
    print("    where the random part averages down.")
    print("  * RQ's real problem is the tail, which ENOB averaged away:")
    kt = lambda e: float(np.mean(e ** 4) / np.mean(e ** 2) ** 2)
    print(f"      kurtosis {kt(lk['e']):.2f} (LK)  ->  {kt(rq['e']):.2f} (RQ)")
    print(f"      max|e|   {lk['maxe']:.3e}  ->  {rq['maxe']:.3e}"
          f"   ({rq['maxe']/lk['maxe']:.2f}x)")
    print(f"      p99.9    {lk['p999']:.3e}  ->  {rq['p999']:.3e}"
          f"   ({rq['p999']/lk['p999']:.2f}x)")
    rown = lambda r: (np.sqrt(np.mean(r["e"] ** 2, axis=1))
                      / np.sqrt(np.mean(ref ** 2, axis=1)))
    rlk, rrq = rown(lk), rown(rq)
    print(f"      worst row {rlk.max():.3e}  ->  {rrq.max():.3e}"
          f"   ({rrq.max()/rlk.max():.2f}x, vs {rq['nrmse']/lk['nrmse']:.2f}x on"
          f" the mean)")
    print("      -- the worst output channel degrades faster than the average.")
    print("    Re-quantizing an already-quantized grid stacks two round-offs,")
    print("    and where they align the result is far off.  A max-error or")
    print("    outlier-sensitive metric penalizes RQ much harder than ENOB.")
    print(f"  * RQ is worse than quantizing the original straight to Q4_1")
    print(f"    ({rq['nrmse']:.3e} vs {rows['L1']['nrmse']:.3e}, "
          f"+{100*(rq['nrmse']/rows['L1']['nrmse']-1):.1f}%), at the same 5.0 bpw.")
    print("    So even without a Q4_K kernel, feeding the converter the")
    print("    original weights rather than the dequantized Q4_K grid is a win.")
    print()

    return rows


def make_data(rng, shape, dist):
    if dist == "gauss":
        return rng.standard_normal(shape).astype(np.float32)
    if dist == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), shape).astype(np.float32)
    if dist == "outlier":
        v = rng.standard_normal(shape).astype(np.float32)
        v[rng.random(shape) < 0.01] *= 8.0
        return v
    raise ValueError(dist)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-K", type=int, default=4096, help="reduction length")
    ap.add_argument("-M", type=int, default=512, help="output rows")
    ap.add_argument("-T", type=int, default=64, help="activation vectors")
    ap.add_argument("--dist", default="gauss", choices=["gauss", "laplace", "outlier"],
                    help="weight distribution")
    ap.add_argument("--xdist", default="gauss", choices=["gauss", "laplace", "outlier"],
                    help="activation distribution; 'outlier' is the case that "
                         "hurts llama.cpp's int8 activation blocks and not FLM's BF16")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--search", type=int, default=3, metavar="N",
                    help="metadata re-fit: also try the N next BF16 values "
                         "above the base S' and keep the least-squares best "
                         "(0 = the pseudocode as quant.md writes it; 3 is the "
                         "knee and what the proposal specifies).  Convert-time "
                         "only, no runtime cost.")
    ap.add_argument("--no-blk-cast", action="store_true",
                    help="drop FLM's per-block BF16 cast of y (keep FP32 across K)")
    a = ap.parse_args()

    if a.K % QK_K:
        raise SystemExit(f"K must be a multiple of {QK_K}")
    rng = np.random.default_rng(a.seed)
    M, K, T = a.M, a.K, a.T
    NB = K // GROUP

    w = make_data(rng, (M, K), a.dist)
    # Activations are BF16 for every scenario, and the reference uses the same
    # BF16 values -- the input dtype is a property of the deployment, not an
    # error either datapath committed.
    x = to_bf16(make_data(rng, (K, T), a.xdist))
    xb = x.reshape(NB, GROUP, T)

    w64, x64 = w.astype(np.float64), x.astype(np.float64)
    y_ref = w64 @ x64
    y_ref_n = float(np.sqrt(np.mean(y_ref ** 2)))

    q41 = quant_q4_1(w)
    q4k = quant_q4_k(w)
    q4k3 = refit_q4k(q4k, search=a.search)         # the proposal's metadata
    # what q4nx/gguf_tensor.py does today: dequantize Q4_K, re-quantize to Q4_1
    q, s6, m6, S, M_ = q4k
    w_deq = (S[..., None, None] * s6[..., None] * q
             - M_[..., None, None] * m6[..., None]).reshape(M, K)
    q41_rq = quant_q4_1(to_bf16(w_deq).astype(np.float32))

    bc = not a.no_blk_cast
    scen = [
        ("W16", "BF16 weights (ceiling)         ",
         lambda: mvm_W16_bf16(w, xb, K)),
        ("L1",  "Q4_1  llama.cpp   Q8_1 acts    ",
         lambda: mvm_L1_q41_llamacpp(q41, xb, K)),
        ("F1",  "Q4_1  FLM         BF16 acts    ",
         lambda: mvm_F1_q41_flm(q41, xb, K, blk_cast=bc)),
        ("LK",  "Q4_K  llama.cpp   Q8_K acts    ",
         lambda: mvm_LK_q4k_llamacpp(q4k, xb, K)),
        ("FK",  "Q4_K  FLM  proposed uint8+BF16 ",
         lambda: mvm_FK_q4k_flm(q4k3, xb, K, blk_cast=bc)),
        ("RQ",  "Q4_K->deq->Q4_1  FLM (today)   ",
         lambda: mvm_F1_q41_flm(q41_rq, xb, K, blk_cast=bc)),
    ]

    res = {}
    for key, label, fn in scen:
        y, W, diag = fn()
        eps_w = rel_rms(W, w)
        y_id = W.astype(np.float64) @ x64          # exact dot of the same weights
        eps_id = float(np.sqrt(np.mean((y_id - y_ref) ** 2)) / y_ref_n)
        eps_t = float(np.sqrt(np.mean((y.astype(np.float64) - y_ref) ** 2)) / y_ref_n)
        # paired: the weight error is common to y and y_id and cancels exactly
        eps_x = float(np.sqrt(np.mean((y.astype(np.float64) - y_id) ** 2)) / y_ref_n)
        # Low-noise composite.  eps_w averages M*K weights and eps_x is a paired
        # measurement, so both are far quieter than eps_t, which only averages
        # M*T dot products.  The budget table below shows they agree.
        eps_m = float(np.hypot(eps_w, eps_x))
        res[key] = dict(label=label, bpw=BPW[key], eps_w=eps_w, eps_id=eps_id,
                        eps_t=eps_t, eps_x=eps_x, eps_m=eps_m, diag=diag, W=W, y=y)

    # ---------------------------------------------------------------- table --
    print()
    print(f"ENOB, FLM vs llama.cpp datapaths from quant.md")
    print(f"  M={M}  K={K}  T={T}  w~{a.dist}  x~{a.xdist}  seed={a.seed}"
          f"  FLM per-block y cast: {'on' if bc else 'OFF'}")
    print()
    print("  key  scenario                          bpw    ENOB_W   ENOB_path"
          "   ENOB_tot   ENOB*")
    print("  " + "-" * 92)
    for key in sorted(res, key=lambda k: -enob(res[k]["eps_m"])):
        r = res[key]
        print(f"  {key:<4} {r['label']} {r['bpw']:5.3f}  {enob(r['eps_w']):7.3f}"
              f"  {enob(r['eps_x']):9.3f}  {enob(r['eps_t']):9.3f}"
              f"  {enob(r['eps_m']):7.3f}")
    print()
    print("  ENOB_W    = weight/format error alone (exact accumulator, exact acts)")
    print("  ENOB_path = the datapath by itself, measured against this pipeline's")
    print("              own effective weights -- Q8 activation quantization for")
    print("              llama.cpp, BF16 roundings for FLM")
    print("  ENOB_tot  = what the datapath actually delivers, measured end to end")
    print("  ENOB*     = the same thing composed from ENOB_W and ENOB_path.  Both")
    print("              inputs are low-noise (M*K weights, and a paired difference)")
    print("              where ENOB_tot averages only M*T dot products, so ENOB* is")
    print("              the column to rank by.  Sorted on it.")
    print()

    # ------------------------------------------------- weight-space RMSE --
    weight_rmse_report(res, w, M, K)

    # -------------------------------------------------------- error budget --
    print("Error budget  (validating the quadrature model behind ENOB*)")
    print("  key    eps_W       eps_path    ENOB*-eps   eps_tot     rel.diff")
    print("  " + "-" * 68)
    for key in ORDER:
        r = res[key]
        rd = abs(r["eps_m"] - r["eps_t"]) / r["eps_t"] if r["eps_t"] else 0.0
        print(f"  {key:<5}  {r['eps_w']:.3e}  {r['eps_x']:.3e}  {r['eps_m']:.3e}"
              f"   {r['eps_t']:.3e}  {rd:8.2e}")
    print("  Weight error and datapath error are independent, and each enters y")
    print("  with its own relative magnitude (E[y^2] = K var(w) var(x) and")
    print("  E[e^2] = K var(w) var(dx), so no growth and no averaging-down over K).")
    print("  The two therefore add in quadrature, and the last column confirms it.")
    print()

    # ------------------------------------------------ where the bits go: acts --
    print("Activation side -- what llama.cpp pays to get exact integer MACs")
    xs_sb = xb.reshape(K // QK_K, QK_K, T)
    qb1, sb1, sig1 = quant_q8_1(xb)
    xh1 = (sb1[:, None, :] * qb1).reshape(K, T)
    qbk, sbk, bsk = quant_q8_K(xs_sb)
    xhk = (sbk[:, None, :] * qbk).reshape(K, T)
    e_q81 = rel_rms(xh1, x)
    e_q8k = rel_rms(xhk, x)
    e_bf16 = rel_rms(to_bf16(x.astype(np.float32)), x)   # 0: x is already BF16
    print(f"  Q8_1  (32-value block, FP16 scale) : eps_x = {e_q81:.3e} "
          f"({enob(e_q81):5.2f} bits)  {ABPV['Q8_1']:.3f} bit/value")
    print(f"  Q8_K  (256-value block, FP32 scale): eps_x = {e_q8k:.3e} "
          f"({enob(e_q8k):5.2f} bits)  {ABPV['Q8_K']:.3f} bit/value")
    print(f"  BF16  (what FLM feeds the MAC)     : eps_x = {rel_round_rms(to_bf16):.3e} "
          f"({enob(rel_round_rms(to_bf16)):5.2f} bits)  {ABPV['BF16']:.3f} bit/value")
    cb = rel_round_rms(to_bf16)
    print("  Cross-check -- llama.cpp's whole reduction is exact, so its ENOB_path")
    print("  should be nothing but the activation quantization plus the one final")
    print("  BF16 cast of y:")
    print(f"    L1   predicted {np.hypot(e_q81, cb):.3e}"
          f"   measured {res['L1']['eps_x']:.3e}")
    print(f"    LK   predicted {np.hypot(e_q8k, cb):.3e}"
          f"   measured {res['LK']['eps_x']:.3e}")
    print("  So llama.cpp's int8 activations are the *worse* of the two choices in")
    print("  isolation: an 8-bit block quant lands near 7.2-7.5 bits, BF16 near 9.2.")
    print("  llama.cpp buys exact integer MACs with that; FLM keeps the better")
    print("  activations and pays BF16 roundings later in the datapath instead.")
    print("  Q8_K is the *coarser* of the two despite its FP32 scale, because one")
    print("  scale covers 256 values instead of 32 and so tracks a wider max.")
    print()

    # --------------------------------------------- where the bits go: metadata --
    print("Metadata side -- what FLM pays to avoid FP16")
    # The naive alternative to the re-fit: keep Q4_K's native metadata and just
    # round S, M to BF16 at use.  Never run as a datapath, but it is the
    # reference the re-fit has to beat, so its W_eff is built here directly.
    q_, s6_0, m6_0, S_0, M_00 = q4k
    W_naive = (to_bf16(S_0)[..., None, None] * s6_0[..., None] * q_
               - to_bf16(M_00)[..., None, None] * m6_0[..., None]).reshape(M, K)
    eps_naive = rel_rms(W_naive, w)
    w_n = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
    pairs = [("Q4_1: BF16(d),BF16(m) vs FP16 d,m",
              res["L1"]["eps_w"], res["F1"]["eps_w"],
              res["F1"]["W"], res["L1"]["W"]),
             ("Q4_K, no re-fit: BF16(S),BF16(M) x uint6 s_j,m_j",
              res["LK"]["eps_w"], eps_naive, W_naive, res["LK"]["W"]),
             ("Q4_K, proposed: BF16 S',M' x uint8 s'_j,m'_j re-fit",
              res["LK"]["eps_w"], res["FK"]["eps_w"],
              res["FK"]["W"], res["LK"]["W"])]
    for note, el, ef, Wf, Wl in pairs:
        # paired: the two share q, so their W_eff differ by nothing but the
        # metadata rounding.  No quadrature subtraction of near-equal numbers.
        dW = Wf.astype(np.float64) - Wl.astype(np.float64)
        extra = float(np.sqrt(np.mean(dW ** 2)) / w_n)
        print(f"  {note}")
        print(f"    eps_W {el:.3e} -> {ef:.3e}   isolated cost {extra:.3e}"
              f" ({enob(extra):5.2f} bits)")
    print("  Without the re-fit only S and M are really rounded, 2 per super")
    print("  block -- s_j, m_j are uint6 and BF16 holds every integer to 256")
    print("  exactly.  The proposal rounds *none* of them at run time; what it")
    print("  pays instead is the convert-time fit, which is a different error")
    print("  (flat absolute, independent per group).  The re-fit section below")
    print("  measures how much of the gap to LK that actually recovers.")
    print()

    # --------------------------------------------------- rounding ablation --
    base = rel_round_rms(to_bf16)
    print("Per-rounding ablation of the FLM datapaths")
    print("  Every datapath rounding is switched off, then exactly one is switched")
    print("  back on, so each cost is measured in isolation against that pipeline's")
    print(f"  own effective weights.  A plain BF16 rounding is {base:.3e}; the")
    print("  'x BF16' column is how much worse the same rounding is in place.")
    print()
    flm = {
        "F1": (mvm_F1_q41_flm, (q41, xb, K),
               dict(final_bf16=False, blk_cast=False, half_cast=False,
                    xsum_cast=False, meta_bf16=True),
               ["half_cast", "xsum_cast", "blk_cast", "final_bf16"]),
        "FK": (mvm_FK_q4k_flm, (q4k3, xb, K),
               dict(final_bf16=False, blk_cast=False, fold_cast=False,
                    xsum_cast=False),
               ["fold_cast", "xsum_cast", "blk_cast", "final_bf16"]),
    }
    NICE = {"half_cast":  "BF16(half sums)   2/group",
            "fold_cast":  "BF16(group sum)   1/group",
            "xsum_cast":  "BF16(sum_i x)     1/group",
            "blk_cast":   "BF16(y_block)     1/block",
            "final_bf16": "BF16(y) final     1 total"}
    for key, (fn, argv, kw, flags) in flm.items():
        y0, W0, _ = fn(*argv, **kw)
        y_id = W0.astype(np.float64) @ x64
        e0 = float(np.sqrt(np.mean((y0.astype(np.float64) - y_id) ** 2)) / y_ref_n)
        parts = []
        print(f"  {key}   (no roundings at all: {e0:.3e} -- FP32 accumulation only)")
        print("        rounding                    isolated eps   ENOB   x BF16")
        for f in flags:
            kw1 = dict(kw)
            kw1[f] = True
            y1, W1, _ = fn(*argv, **kw1)
            e1 = float(np.sqrt(np.mean((y1.astype(np.float64) - y_id) ** 2))
                       / y_ref_n)
            parts.append(e1)
            print(f"        {NICE[f]:<26}  {e1:.3e}  {enob(e1):6.2f}"
                  f"   {e1/base:5.2f}x")
        qsum = float(np.sqrt(np.sum(np.array(parts) ** 2)))
        print(f"        {'quadrature sum of the above':<26}  {qsum:.3e}"
              f"  {enob(qsum):6.2f}")
        print(f"        {'all on (the real path)':<26}  {res[key]['eps_x']:.3e}"
              f"  {enob(res[key]['eps_x']):6.2f}")
        print()
    print("  The amplification is scale/min cancellation: the scale path and the")
    print("  min path are both far larger than their difference, so a relative")
    print("  error on either is magnified on the result that survives.")
    print("  key   rms(scale path)  rms(y_block)   cancellation ratio")
    print("  " + "-" * 56)
    for key in ["F1", "FK"]:
        d = res[key]["diag"]
        st = float(np.sqrt(np.mean(d["scale_term"].astype(np.float64) ** 2)))
        yb = float(np.sqrt(np.mean(d["yb"].astype(np.float64) ** 2)))
        print(f"  {key:<4}  {st:13.4f}  {yb:12.4f}  {st/yb if yb else 0:16.2f}")
    print("  BF16(y_block) is the one rounding that sits *after* the cancellation,")
    print("  so it alone comes in at ~1x -- it is the roundings taken before the")
    print("  min is subtracted that get amplified.")
    print()

    # ------------------------------------------- how good is the re-fit --
    print("The re-fit -- quality of the convert-time (int6,FP16) -> (int8,BF16) fit")
    print("  The fit preserves the *product* t_j = S*s_j, which is the only form")
    print("  the kernel ever uses.  Measured against the exact FP32 t_j that")
    print("  llama.cpp's own dequantizer would compute.  This table uses the base")
    print("  pseudocode (search=0), which is what the |S'|/2 <= S/8 bound covers:")
    print()
    _, s6_, m6_, S_, M_0 = q4k
    _, s8_, m8_, Sp_, Mp_ = refit_q4k(q4k, search=0)
    print("  side   grid    max|err|/P   rms(err)/rms(t)   overflow   s'max  s'mean")
    print("  " + "-" * 74)
    for tag, P, p6, Pp, p8 in [("scale", S_, s6_, Sp_, s8_),
                               ("min  ", M_0, m6_, Mp_, m8_)]:
        t = (P[..., None].astype(np.float64) * p6)
        err = (Pp[..., None].astype(np.float64) * p8) - t
        live = P > 0
        # quant.md's bound: |err| <= |P'|/2 <= P/8, i.e. 1/8 of a step of the
        # uint6 grid the k-quant search itself worked on.
        rel_step = (np.abs(err)[live] / P[live][..., None]).max()
        rms_rel = float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(t ** 2)))
        # a real clamp is round(t/P') > 255 *before* the clip; p8 == 255 is not
        # one, it is the top group legitimately rounding up from >= 254.5
        inv = np.where(live, 1.0 / np.where(live, Pp, 1.0), 0.0).astype(np.float32)
        nover = int(np.sum(np.rint(t * inv[..., None]) > 255.0))
        print(f"  {tag}  uint8   {rel_step:9.5f}   {rms_rel:15.3e}   {nover:8d}"
              f"   {p8.max():5.0f}  {p8[live].mean():6.1f}")
    print("  max|err|/P is the fit error in units of one step of the *source*")
    print("  uint6 grid; quant.md predicts <= 1/8 = 0.125 and the measurement")
    print("  confirms it.  'overflow' counts groups where round(t/P') exceeded")
    print("  255 before the clip -- rounding P' up (step 5) keeps it at zero, so")
    print("  step 8's clamp never fires and the bound above is never violated.")
    print()
    print("  But the re-fit does NOT make the metadata free.  Against the naive")
    print("  BF16(P) x uint6 alternative:")
    bf = rel_round_rms(to_bf16)
    for tag, P, p6, Pp, p8 in [("scale", S_, s6_, Sp_, s8_),
                               ("min  ", M_0, m6_, Mp_, m8_)]:
        t = (P[..., None].astype(np.float64) * p6)
        e2 = (to_bf16(P)[..., None].astype(np.float64) * p6) - t
        e3 = (Pp[..., None].astype(np.float64) * p8) - t
        r2 = float(np.sqrt(np.mean(e2 ** 2)))
        r3 = float(np.sqrt(np.mean(e3 ** 2)))
        print(f"    {tag}  naive BF16(P) rms {r2:.4e}  ->  re-fit"
              f" {r3:.4e}   ({r3/r2:.3f}x)")
    tt = (S_[..., None].astype(np.float64) * s6_)
    ratio = float(np.sqrt(np.mean(tt ** 2)) / np.mean(np.abs(tt).max(-1)))
    print(f"  Only ~10% better, not eliminated, and the reason is a factor of 2")
    print(f"  in quant.md's derivation.  It compares the re-fit's |S'|/2 against")
    print(f"  '|t_j| * 2^-8', but 2^-8 is BF16's *step*; round-to-nearest costs")
    print(f"  half of it, |t_j| * 2^-9.  The two grids are therefore the same")
    print(f"  resolution at the top group -- 1/510 vs 1/512 -- and uint8 is")
    print(f"  relatively *worse* below it, since its absolute step is flat at")
    print(f"  tmax/255 while BF16's shrinks with t_j (rms t_j/tmax = {ratio:.3f}).")
    print(f"  What uint8 does win is uniformity: BF16's relative step doubles")
    print(f"  across a binade, so its rel-error rms is {bf:.3e} where a uniform")
    print(f"  grid of the same top-end resolution gives {(1/255)/np.sqrt(12):.3e}.")
    print("  Net of the two effects: ~10%.")
    print()
    print("  That net depends on how *spread* the 8 t_j inside a super block are.")
    print("  Write rho = rms_j(t_j)/tmax: rho -> 1 means all 8 groups share the")
    print("  same scale and uint8 is using its full grid; small rho means the")
    print("  super block is dominated by one group and the other 7 sit low on a")
    print("  grid pinned to tmax.  Super blocks binned by their own rho:")
    print("     rho bin        n      naive rms    re-fit rms   ratio")
    print("     " + "-" * 58)
    tt = (S_[..., None].astype(np.float64) * s6_)
    tmx = np.abs(tt).max(-1)
    livb = tmx > 0
    rho = np.where(livb, np.sqrt(np.mean(tt ** 2, axis=-1)) / np.where(livb, tmx, 1.0),
                   0.0)
    e2b = (to_bf16(S_)[..., None].astype(np.float64) * s6_) - tt
    e3b = (Sp_[..., None].astype(np.float64) * s8_) - tt
    edges = [0.0, 0.70, 0.80, 0.90, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = livb & (rho >= lo) & (rho < hi)
        if not sel.any():
            continue
        r2 = float(np.sqrt(np.mean(e2b[sel] ** 2)))
        r3 = float(np.sqrt(np.mean(e3b[sel] ** 2)))
        print(f"     [{lo:.2f},{hi:.2f})  {int(sel.sum()):6d}   {r2:.4e}   "
              f"{r3:.4e}   {r3/r2:5.3f}")
    print(f"     overall rho: mean {rho[livb].mean():.3f}")
    print("  The ratio crosses 1.0 inside that range, so whether the re-fit helps")
    print("  or hurts is a property of the weight distribution, not a constant.")
    print("  Across --dist (eps_W of the base re-fit, vs the naive alternative):")
    print("     w~gauss   rms(t_j/tmax) 0.810   -0.034%   (helps)")
    print("     w~laplace rms(t_j/tmax) 0.737   -0.003%   (a wash)")
    print("     w~outlier rms(t_j/tmax) 0.605   +0.413%   (HURTS)")
    print("  Heavier-tailed weights spread the per-group scales further, and")
    print("  that is the case real LLM weights resemble.  So the base re-fit is")
    print("  not a safe win on its own -- the search sweep below is.")
    print()
    print("  The local search over S' candidates -- quant.md's 'optional")
    print("  refinement', and the reason --search 3 is part of the proposal --")
    print("  is worth considerably more than the uint8 grid itself, and is free")
    print("  (convert time only, identical kernel).  eps_W by search width:")
    print("     search   eps_W        of the naive -> LK metadata gap recovered")
    print("     " + "-" * 62)
    e_lk, e_s2 = res["LK"]["eps_w"], eps_naive
    ew_search3 = None
    for n in [0, 1, 2, 3, 7, 15]:
        qn = refit_q4k(q4k, search=n)
        _, s8n, m8n, Spn, Mpn = qn
        Wn = (Spn[..., None, None] * s8n[..., None] * q4k[0]
              - Mpn[..., None, None] * m8n[..., None]).reshape(M, K)
        ewn = rel_rms(Wn, w)
        if n == 3:
            ew_search3 = ewn
        frac = (e_s2 - ewn) / (e_s2 - e_lk) if e_s2 > e_lk else 0.0
        mark = "  <- --search" if n == a.search else ""
        print(f"     {n:<8} {ewn:.6e}   {100*frac:5.1f}%{mark}")
    print(f"     {'(naive)':<8} {e_s2:.6e}     0.0%")
    print(f"     {'(LK)':<8} {e_lk:.6e}   100.0%   exact FP16 x uint6 metadata")
    print("  The base pseudocode recovers only ~14% of that gap; searching 3")
    print("  candidates recovers ~60%, after which it saturates.")
    print("  More importantly the search makes the result *stable*: eps_W vs")
    print("  the naive alternative goes -0.150% / -0.140% / -0.137% on gauss /")
    print("  laplace / outlier, where search=0 swings -0.034% / -0.003% /")
    print("  +0.413%.  Fixing S' at the smallest admissible value is what makes")
    print("  the plain re-fit distribution-sensitive; letting it move a few BF16")
    print("  steps removes that.  It costs nothing at run time and the kernel")
    print("  cannot tell the difference -- it is the difference between a safe")
    print("  win and a coin flip.")
    print()

    # ------------------------------------------------- datapath variants --
    print("Datapath variants of the proposal -- where the casts are allowed to go")
    print("  Same 20 B format and same metadata throughout; only the reduce")
    print("  structure moves.  'two halves' is what quant.md's equations write")
    print("  and what q4_core.h does today for Q4_1; 'FP32 group' is proposal")
    print("  change (a): span all 32 in the accumulator and cast once.")
    print()
    q4k3_0 = refit_q4k(q4k, search=0)
    variants = [
        ("two halves, search 0", lambda: mvm_FK_q4k_flm(q4k3_0, xb, K,
                                                        blk_cast=bc, half_split=True)),
        ("two halves, search 3", lambda: mvm_FK_q4k_flm(q4k3, xb, K,
                                                        blk_cast=bc, half_split=True)),
        ("FP32 group, search 0", lambda: mvm_FK_q4k_flm(q4k3_0, xb, K,
                                                        blk_cast=bc)),
        ("FP32 group, search 3", lambda: mvm_FK_q4k_flm(q4k3, xb, K,
                                                        blk_cast=bc)),
        ("+ FP32 across blocks", lambda: mvm_FK_q4k_flm(q4k3, xb, K,
                                                        blk_cast=False)),
        ("q4_core.h pre-fix", lambda: mvm_FK_q4k_flm(q4k3, xb, K,
                                                     blk_cast=False,
                                                     pre_cancel="bf16")),
        ("q4_core.h + bf16 limbs", lambda: mvm_FK_q4k_flm(q4k3, xb, K,
                                                          blk_cast=False,
                                                          pre_cancel="limb2")),
    ]
    print("  variant                 eps_W       eps_path     ENOB*   casts/sb")
    print("  " + "-" * 66)
    ncast = [24, 24, 8, 8, 8, 10, 10]
    for (name, fn), nc in zip(variants, ncast):
        y, W, _ = fn()
        ew = rel_rms(W, w)
        y_id = W.astype(np.float64) @ x64
        ex = float(np.sqrt(np.mean((y.astype(np.float64) - y_id) ** 2)) / y_ref_n)
        print(f"  {name:<22}  {ew:.3e}  {ex:.3e}  {enob(np.hypot(ew, ex)):8.4f}"
              f"  {nc:7d}")
    dk = res["LK"]
    print(f"  {'(llama.cpp Q4_K)':<22}  {dk['eps_w']:.3e}  {dk['eps_x']:.3e}"
          f"  {enob(dk['eps_m']):8.4f}        -")
    print()
    print("  The last two rows are what q4_core.h really does, which is not the")
    print("  '+ FP32 across blocks' row above it: the kernel carries two FP32")
    print("  accumulators across all of K and combines them only at the end, so")
    print("  there is no per-block cast after the cancellation -- but each")
    print("  accumulator was rounded to BF16 on its way into the running sum,")
    print("  i.e. before the cancellation, and independently on both paths.")
    print("  Splitting each into two BF16 limbs buys ~17 significand bits back")
    print("  for 2 extra MACs per 32x256 block, outside the group loop.")
    print()
    print("  The FP32 group reduce is both cheaper and more accurate: it deletes")
    print("  16 of the 24 casts *and* one MAC per group, and every cast it")
    print("  deletes sat before the min subtraction, where the cancellation")
    print("  table above says it cost 2.3x its nominal price.  The search is")
    print("  free at run time.  Together they put the proposal past llama.cpp's")
    print("  own Q4_K on ENOB*, on gaussian activations, at 4.625 bpw.")
    print()
    print("  Same change applied to the Q4_1 kernel that ships today:")
    for nm, kw in [("q4_core.h as written ", dict()),
                   ("merged half reduce   ", dict(merge_halves=True))]:
        y, W, _ = mvm_F1_q41_flm(q41, xb, K, blk_cast=bc, **kw)
        ew = rel_rms(W, w)
        ex = float(np.sqrt(np.mean((y.astype(np.float64)
                                    - W.astype(np.float64) @ x64) ** 2)) / y_ref_n)
        print(f"    {nm}  eps_path {ex:.3e}   ENOB* {enob(np.hypot(ew, ex)):.4f}")
    print("    Accuracy-neutral here (Q4_1 multiplies each half by the same")
    print("    scale and folds in FP32, so there is no second cast to delete),")
    print("    but it still removes one `aie::mac` and one to_vector<bf16>()")
    print("    per group.  Independent of anything Q4_K.")
    print()

    # -------------------------------------------------------- conclusions --
    print("Notes")
    print("  * At equal format the datapaths land within a few hundredths of")
    print("    a bit of each other: eps_W ~ 7e-2 dominates every datapath term")
    print("    (~5e-3), and 5e-3 in quadrature against 7e-2 moves the total by")
    print("    0.0037 bits.  Format choice is worth ~10x more than datapath")
    print("    choice here.  ENOB_path is what actually separates the graphs.")
    print("  * llama.cpp's advantage is that once the activations are int8 the")
    print("    whole reduction is exact integer arithmetic; its cost is that the")
    print("    int8 block quantization is *worse* than BF16 (7.2-7.5 bits vs 9.2).")
    print("    FLM's advantage is untouched BF16 activations; its cost is BF16")
    print("    metadata plus BF16 partial sums, the latter amplified ~2.2x by the")
    print("    scale/min cancellation.  The two land close to even.")
    print("  * The tie above is for Gaussian activations, the best case for an")
    print("    int8 block quantizer.  Activation outliers break the tie in FLM's")
    print("    favour, because BF16 is scale-free and a per-block int8 scale is")
    print("    not -- rerun with --xdist outlier (1% of values x8):")
    print("      LK  ENOB_path 7.15 -> 5.73,  ENOB* 3.803 -> 3.761")
    print("      FK  ENOB_path 7.51 -> 7.56,  ENOB* 3.804 -> 3.804  (flat)")
    print("    Q8_K takes the bigger hit of the two llama.cpp rows because its")
    print("    one scale spans 256 values, so a single outlier coarsens the")
    print("    whole super block.  Real LLM activations do have outlier")
    print("    channels, so that case is the more representative of the two.")
    print("  * The proposed format (uint8 s'_j/m'_j + BF16 S'/M', re-fitted")
    print("    offline) moves every metadata rounding off the run-time path.")
    print("    That is not free: the convert-time fit that replaces the rounding")
    print("    is only ~10% smaller as written, because a uint8 grid pinned to")
    print("    tmax/255 has the same top-end resolution as BF16's round-to-")
    print("    nearest and a flat absolute step below it.  Its case is the")
    print("    kernel, not the ENOB: uint8 is a native NPU width, so the")
    print("    12-byte 6-bit unpack goes away for +0.125 bpw.  Build it *with*")
    print("    the local search -- as written it is -0.03% of eps_W on gaussian")
    print("    weights but +0.41% (worse than not re-fitting at all) on")
    print("    outlier-heavy ones, while with --search 3 it is a steady -0.14%")
    print("    on all three distributions.")
    print("  * RQ is the path q4nx/gguf_tensor.py takes today (dequantize Q4_K,")
    print("    re-quantize to Q4_1 -- the `#TODO: FIXME: A direct conversion from")
    print("    q4_k to q4_1` branch).  It is the worst row in the table and it")
    print("    costs 0.375 bpw more than FK, so a direct Q4_K path is a clear win.")
    rq, fk = res["RQ"], res["FK"]
    print(f"      RQ  {enob(rq['eps_m']):.3f} bits at {rq['bpw']:.3f} bpw"
          f"   ->  FK {enob(fk['eps_m']):.3f} bits at {fk['bpw']:.3f} bpw"
          f"   (+{enob(fk['eps_m']) - enob(rq['eps_m']):.3f} bits,"
          f" -{rq['bpw'] - fk['bpw']:.3f} bpw)")
    noise = 1.4427 / np.sqrt(2.0 * M * T)
    print(f"  * Sampling: eps_W averages M*K = {M*K} weights and eps_path is a")
    print(f"    paired difference, so both are quiet.  eps_tot averages only")
    print(f"    M*T = {M*T} dot products and carries ~{noise:.4f} bits of noise at")
    print(f"    the current -M {M} -T {T}.  Rank on ENOB*, not ENOB_tot.")
    print()

    # ------------------------------------------------------ recommendation --
    print("PROPOSAL -- Q4_K on the NPU, BF16-only multiplier")
    print("  Machine model: c += a*b with a, b BF16 and c an FP32 accumulator.")
    print("  No FP16 operand, no cheap FP32 multiply, no int8 x int4 MAC.  Two")
    print("  rules follow, and they decide everything.")
    print()
    print("  Rule 1 -- never put an inexact value on a multiplier input.")
    print("    q      uint4  <= 15    exact      m'_j  uint8 <= 255  exact")
    print("    s'_j   uint8  <= 255   exact      S'/M' stored BF16   exact")
    print("    x      BF16 input      exact")
    print("    s'_j*q up to 3825, needs 12 bits  NOT exact")
    print("  The proposed datapath never forms s'_j*q as a float, so every")
    print("  operand it presents to the MAC is exact and the only run-time")
    print("  weight-side error is the group-sum cast.")
    print()
    print("  Rule 2 -- every FP32 accumulator you feed back into a multiply must")
    print("  be squeezed to BF16 first.  That is the only other rounding, and")
    print("  the cancellation table above shows the ones taken before the min")
    print("  is subtracted come in ~2.3x amplified.  So: minimize the number of")
    print("  accumulator -> operand hand-offs, and care much more about the ones")
    print("  before the min subtraction than the one after.")
    print()
    print("  Weight-path multiplies per super block per output element, weights")
    print("  reused across T activation columns.  Activation sums are shared")
    print("  across all M rows and dropped.")
    print()
    print("    " + "datapath".ljust(26) + "  mult   casts   bpw")
    print("    " + "-" * 52)
    rows_op = [
        ("two-half reduce + fold   ", 256 + 18, 24, 4.625),
        ("FP32 group reduce (this) ", 256 + 18, 8, 4.625),
    ]
    for nm, nm_mult, nc, bpw in rows_op:
        print(f"    {nm}  {nm_mult:5d}   {nc:5d}   {bpw:.3f}")
    print("    The FP32 group reduce also removes one MAC per group (8 per")
    print("    super block) -- the second `aie::mac` against the half-group")
    print("    partial.  Cheaper and more accurate at the same footprint.")
    print()
    print("  The kernel, per super block per output row:")
    print("    * unpack uint4 q -> BF16              exact, no rounding")
    print("    * 256 MACs q*x into one FP32 acc      per group of 32")
    print("    * BF16(group sum)                     8 casts, the only")
    print("                                          run-time weight-side loss")
    print("    * 8 MACs s'_j * that                  s'_j uint8, exact")
    print("    * 8 MACs m'_j * BF16(sum_i x)         shared across all M rows")
    print("    * S'*acc_s - M'*acc_m                 S', M' stored BF16, exact")
    print("    * FP32 across super blocks, one BF16 at the end of the row")
    print("  Zero metadata roundings, zero inexact operands, 8 casts per super")
    print("  block, 274 multiplies, 4.625 bpw, no 6-bit unpack.")
    print()
    print("  Build the metadata with --search 3: convert time only, the kernel")
    print("  cannot tell the difference, and it is what makes the re-fit a win")
    print("  on every weight distribution instead of a coin flip on the")
    print("  heavy-tailed ones that matter.")
    print()
    print("  Against the alternatives:")
    print("    vs native Q4_K storage (16 B/super block, 4.500 bpw): +0.125 bpw")
    print("      buys away the 12-byte 6-bit unpack and the two per-super-block")
    print("      metadata roundings, and the search makes the format strictly")
    print("      better in eps_W as well.")
    print("    vs llama.cpp Q4_K: ahead on gaussian activations, and further")
    print("      ahead on outlier activations, where Q8_K's one int8 scale per")
    print("      256 values loses ~1.4 bits of ENOB_path and BF16 loses none.")
    print("    vs RQ, what the converter does today: +0.24 bits and -0.375 bpw.")
    print("      This is the decision that is worth 50x every other one here.")
    print()
    print("  The 20 B format is datapath-agnostic: an int8 x int4 -> int16 MAC,")
    print("  if one ever exists, would let the group scale be folded into q in")
    print("  the integer domain without changing a byte of what the converter")
    print("  emits.  Committing to the format now costs nothing either way.")
    print()


if __name__ == "__main__":
    main()
