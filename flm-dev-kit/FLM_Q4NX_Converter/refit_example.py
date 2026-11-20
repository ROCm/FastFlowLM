#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Worked numbers for refit_example.md.

Compares two ways of moving Q4_K's (FP16 S, uint6 s_j) metadata into
(BF16 S', uint8 s'_j):

  naive : S' = BF16(S/4),  s'_j = 4 s_j        -- cast and pad
  re-fit: S' = BF16_up(t_max/255) [+ search],  s'_j = round(t_j / S')

Error is measured directly against the exact effective scale t_j = S s_j.
Since the dequantized weight is w_i = t_j q_i (with q exact), the relative
error on w is identical to the relative error on t -- no MVM needed.

    python3 refit_example.py
"""
import numpy as np

# ---------------------------------------------------------------- bf16 helpers
def bf16(x):
    """Round to nearest BF16 (ties to even)."""
    u = np.array(np.float32(x)).view(np.uint32).reshape(-1)
    r = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32).astype(np.float64).reshape(np.shape(x))

def bf16_up(x):
    """Round to BF16 away from zero (never below |x|)."""
    u = np.array(np.float32(x)).view(np.uint32).reshape(-1)
    r = np.where(u & np.uint32(0xFFFF), np.uint32(0x10000), np.uint32(0))
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32).astype(np.float64).reshape(np.shape(x))

def bf16_nth(x, n):
    """The n-th BF16 value above x (n=0 returns x)."""
    u = np.array(np.float32(x)).view(np.uint32).reshape(-1)
    return (u + np.uint32(n) * np.uint32(0x10000)).view(np.float32).astype(np.float64).reshape(np.shape(x))

rms = lambda v: float(np.sqrt((np.asarray(v, float) ** 2).mean()))

# ---------------------------------------------------------------- the two fits
def naive(S, s):
    Sp = float(bf16(S / 4))
    return Sp, (4 * s).astype(int)

def refit(S, s, search=0):
    t = S * s
    base = float(bf16_up(t.max() / 255))
    best = None
    for k in range(search + 1):
        Sp = float(bf16_nth(base, k))
        sp = np.clip(np.round(t / Sp), 0, 255)
        sse = float(((Sp * sp - t) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, Sp, sp.astype(int), k)
    return best[1], best[2], best[3]

# ---------------------------------------------------------------- single block
def show(title, S, s, search=3, ncand=4):
    s = np.asarray(s, float)
    t = S * s
    Sn, sn = naive(S, s)
    en = Sn * sn - t
    d = (float(bf16(S)) - S) / S

    print("=" * 74)
    print(title)
    print("  S       = %-18.12g  BF16(S) = %-18.12g" % (S, bf16(S)))
    print("  delta   = %+.4e  (%.2f x 2^-8)" % (d, abs(d) / 2**-8))
    print("  rho     = %.4f   t_max = %.8g" % (rms(t) / t.max(), t.max()))
    print("  s_j     =", s.astype(int))
    print("  t_j     =", np.array(["%.8g" % v for v in t]))
    print("  -- naive: S' = %.10g" % Sn)
    print("     s'_j =", sn)
    print("     err  =", np.array(["%+.4e" % v for v in en]))
    print("     rel rms = %.4e   max = %.4e" % (rms(en) / rms(t), abs(en).max()))
    print("  -- candidates:")
    base = float(bf16_up(t.max() / 255))
    for k in range(ncand):
        Sp = float(bf16_nth(base, k))
        sp = np.clip(np.round(t / Sp), 0, 255)
        e = Sp * sp - t
        print("     k=%d  S'=%-16.10g s'=%-38s SSE=%.6e" % (k, Sp, str(sp.astype(int)), (e**2).sum()))
    Sp, sp, k = refit(S, s, search)
    er = Sp * sp - t
    print("  -- re-fit (search %d picked k=%d): S' = %.10g" % (search, k, Sp))
    print("     s'_j =", sp)
    print("     err  =", np.array(["%+.4e" % v for v in er]))
    print("     rel rms = %.4e   max = %.4e   bound S/8 = %.4e" % (rms(er) / rms(t), abs(er).max(), S / 8))
    print()


# ---------------------------------------------------------------- aggregate
def sweep(N=200000, seed=7):
    rng = np.random.default_rng(seed)

    def fit_batch(S, s, search):
        t = S[:, None] * s
        Sn = bf16(S / 4).reshape(-1)
        en = Sn[:, None] * (4 * s) - t
        base = bf16_up(t.max(1) / 255).reshape(-1)
        bE = bS = None
        for k in range(search + 1):
            Sp = bf16_nth(base, k).reshape(-1)[:, None]
            e = Sp * np.clip(np.round(t / Sp), 0, 255) - t
            sse = (e**2).sum(1)
            if bE is None:
                bE, bS = e, sse
            else:
                m = sse < bS
                bE[m], bS[m] = e[m], sse[m]
        return t, en, bE

    print("=" * 74)
    print("Aggregate: %d random super-blocks, S ~ random FP16, s_j spread swept" % N)
    print("  rho     naive        re-fit       ratio")
    for p in (4.0, 2.0, 1.0, 0.5, 0.15):
        mant = rng.integers(0, 1024, N) / 1024 + 1.0
        S = np.float64(np.float16(mant)) * 2.0 ** rng.integers(-14, 2, N)
        a = rng.uniform(0, 1, (N, 8)) ** p
        s = np.clip(np.round(a / a.max(1, keepdims=True) * 63), 1, 63)
        t, en, e3 = fit_batch(S, s, 3)
        rho = float(np.mean(np.sqrt((t**2).mean(1)) / t.max(1)))
        print("  %.3f   %.4e   %.4e   %.3f"
              % (rho, rms(en) / rms(t), rms(e3) / rms(t), rms(e3) / rms(en)))
    print()


def delta_stats():
    d = np.array([(float(bf16(1.0 + m / 1024)) - (1.0 + m / 1024)) / (1.0 + m / 1024)
                  for m in range(1024)])
    print("=" * 74)
    print("BF16(S) relative error over all 1024 FP16 mantissas in [1,2):")
    print("  exact (delta = 0): %d/1024 = %.1f%%" % ((d == 0).sum(), 100 * (d == 0).mean()))
    print("  rms |delta| = %.4e     max |delta| = %.4e   (2^-8 = %.4e)"
          % (rms(d), abs(d).max(), 2**-8))
    print()


if __name__ == "__main__":
    delta_stats()
    show("EX1  worst-case S (exact BF16 tie), moderate spread",
         1.0 + 2**-8, [63, 58, 51, 47, 44, 40, 35, 29])
    show("EX2  BF16-exact S -- naive is exact, re-fit must not lose",
         1.0625, [63, 58, 51, 47, 44, 40, 35, 29])
    show("EX3  heavy-tailed s_j (low rho) -- the search picks k=1, not k=0",
         float(np.float16(0.0217895507812)), [63, 19, 14, 11, 9, 7, 6, 4])
    sweep()
