#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Bit-accurate emulation of the FLM Q4_K NPU decoding datapath.

Emulates, operation for operation, what a single AIE core executes for one
mvm() over a Q4_K weight tensor:

  proj_main.cc   b_col_reduce_add[l] = bf16(reduce_add(load_v<32>(x + 32l)))
  q4_core.h      _Q4K_zero / _Q4K_x_bf16 (per 32x256 block) / _Q4K_finalize

Every value that lives in bf16 on the core is rounded to bf16 (RNE) here, every
accumulator is fp32 (accfloat is 32 bits -- accum.hpp:108), and every
accumulation is stepped in the same order the kernel steps it, so each fp32
rounding lands where the hardware's lands.

Reference is the exact ggml Q4_K dequantisation of the same weights, evaluated
in float64.  That is what llama.cpp computes against, so the error reported
here IS FLM's excess over llama.cpp -- refit loss plus datapath loss.
"""

import argparse
import numpy as np
import torch

from q4nx.gguf_tensor import GGUFTensor, _refit_one_side

F32 = np.float32

# ---------------------------------------------------------------------------
# bf16
# ---------------------------------------------------------------------------

def bf16(x):
    """Round float32 -> bfloat16 -> float32, round-to-nearest-even.

    Matches to_vector<bfloat16>() under the core's default rounding mode.
    """
    u = np.ascontiguousarray(x, dtype=F32).view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    r = (u + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    # keep NaN/Inf well-behaved (not reachable here, but cheap)
    return r.view(F32)


def reduce_add_bf16(v):
    """aie::reduce_add over a bf16 vector: fp32 pairwise tree, bf16 on the way out.

    v: (..., N) with N a power of two, entries already bf16-exact.
    """
    a = np.asarray(v, dtype=F32).copy()
    n = a.shape[-1]
    while n > 1:
        n //= 2
        a = (a[..., :n] + a[..., n:]).astype(F32)
    return bf16(a[..., 0])


def split_f32(a):
    """bf16_limb::split_f32 -- two bf16 limbs, a ~= l0 + l1."""
    l0 = bf16(a)
    l1 = bf16((np.asarray(a, F32) - l0).astype(F32))   # a - l0 is exact in fp32
    return l0, l1


def mac_f32_bf16_low(acc, a, b):
    """acc += a*b with a fp32 (split into 2 limbs) and b bf16-exact. Smallest first."""
    l0, l1 = split_f32(a)
    acc = (acc + b * l1).astype(F32)
    acc = (acc + b * l0).astype(F32)
    return acc


def mac_f32_bf16_safe(acc, a, b):
    """3-limb variant, fp32-exact."""
    l0 = bf16(a)
    r1 = (np.asarray(a, F32) - l0).astype(F32)
    l1 = bf16(r1)
    l2 = bf16((r1 - l1).astype(F32))
    acc = (acc + b * l2).astype(F32)
    acc = (acc + b * l1).astype(F32)
    acc = (acc + b * l0).astype(F32)
    return acc


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------

GGML_GROUP_SIZE = 32
NIBBLE_BIAS     = 128    # q4_core.h: uint4 -> bf16 by placing 0x43 over the nibble
SUPER_BLOCK = 256


def npu_mvm(x_bf16, s8, m8, S, Mneg, q,
            group_sum_exact=False,      # ablation: skip the one bf16 group rounding
            sb_limb="low",              # "low" | "safe" | "bf16" | "exact"
            fold_min_per_sb=False,      # the a_M-per-super-block variant
            reduce_exact=False,         # ablation: exact sum_i x_i
            group_limb=False,           # FIX: 2-limb split of the group sum
            reduce_limb=False,          # FIX: 2-limb split of b_col_reduce_add
            nibble_bias=None):          # "limb2" | "limb3" | "f32": the +128
                                        # operand trick, and how sum_i x_i is
                                        # represented when paying it back
    """One mvm() over K columns, N rows, exactly as the core runs it.

    s8, m8 : (N, K/32)   uint8 per-group scale / min, as float32
    S      : (N, K/256)  bf16 super-block scale
    Mneg   : (N, K/256)  bf16 super-block min, ALREADY NEGATED (kernel adds)
    q      : (N, K)      uint4 quants as float32
    Returns (N,) float32.
    """
    N, K = q.shape
    nblk = K // SUPER_BLOCK

    # ---- proj_main.cc: b_col_reduce_add, one bf16 per group, shared by all rows
    xg = x_bf16.reshape(-1, GGML_GROUP_SIZE)
    b_red_f32 = xg.astype(np.float64).sum(-1).astype(F32)
    if reduce_exact:
        b_red, b_red_lo = b_red_f32, np.zeros_like(b_red_f32)
    elif reduce_limb:
        b_red, b_red_lo = split_f32(b_red_f32)
    else:
        b_red, b_red_lo = reduce_add_bf16(xg), np.zeros_like(b_red_f32)

    # q4_core.h: the group accumulator starts at the scale path's zero point
    # rather than at zero, because the MAC operand is 128 + q, not q. How
    # precisely sum_i x_i is represented here is the whole question: the bias
    # term runs ~17x the group sum it corrects, so a representation good to
    # 2^-b leaves a residual of 128/|q_mean| * 2^-b on the answer.
    if nibble_bias == "limb2":                 # the two limbs the min path uses
        l0, l1 = split_f32(b_red_f32)
        zero_point = [-F32(NIBBLE_BIAS) * l1, -F32(NIBBLE_BIAS) * l0]
    elif nibble_bias == "limb3":               # one more limb, ~25 bits
        l0 = bf16(b_red_f32)
        r1 = (b_red_f32 - l0).astype(F32)
        l1 = bf16(r1)
        l2 = bf16((r1 - l1).astype(F32))
        zero_point = [-F32(NIBBLE_BIAS) * l2, -F32(NIBBLE_BIAS) * l1,
                      -F32(NIBBLE_BIAS) * l0]
    elif nibble_bias == "f32":                 # a single fp32 vector subtract
        zero_point = [-F32(NIBBLE_BIAS) * b_red_f32]
    elif nibble_bias in ("defer2", "defer3"):
        # The other place the bias can be paid: not into the group accumulator,
        # but into acc_s, once the group sum has already been scaled by s'_j.
        # The correction is 128 S' sum_j s'_j sum_i x_i -- the same shape the
        # min path already carries -- so it can ride the scale accumulator all
        # the way to the end of the row and be settled once.
        l0 = bf16(b_red_f32)
        r1 = (b_red_f32 - l0).astype(F32)
        l1 = bf16(r1)
        defer_limbs = [l1, l0] if nibble_bias == "defer2" else \
                      [bf16((r1 - l1).astype(F32)), l1, l0]
        zero_point = None
    elif nibble_bias in ("row2", "row3"):
        # Row-level payback.  The bias is 128 sum_b S'_b sum_j s'_j sum_i x_i,
        # which is a scale-path term of exactly the min path's shape, so it can
        # ride its own accumulator pair (acc_b per super-block, c_b across the
        # whole K loop) and be settled once, after the 32-row block is done.
        # The kernel then never mentions 128 inside the group loop.
        l0 = bf16(b_red_f32)
        r1 = (b_red_f32 - l0).astype(F32)
        l1 = bf16(r1)
        row_limbs = [l1, l0] if nibble_bias == "row2" else \
                    [bf16((r1 - l1).astype(F32)), l1, l0]
        zero_point = None
    elif nibble_bias is None:
        zero_point = None
    else:
        raise ValueError(nibble_bias)
    defer = nibble_bias in ("defer2", "defer3")
    row = nibble_bias in ("row2", "row3")

    c_s = np.zeros(N, F32)
    c_m = np.zeros(N, F32)
    c_b = np.zeros(N, F32)                                # row-level bias path
    c_1 = np.zeros(N, F32)                                # folded variant

    for b in range(nblk):
        acc_s = np.zeros(N, F32)
        acc_m = np.zeros(N, F32)
        acc_b = np.zeros(N, F32)

        for g in range(SUPER_BLOCK // GGML_GROUP_SIZE):
            gi = b * 8 + g
            qcol = q[:, gi * 32:(gi + 1) * 32]            # bf16-exact (uint4)
            xcol = x_bf16[gi * 32:(gi + 1) * 32]          # bf16

            # inner MAC loop: fp32 accumulator, one rounded add per column
            temp = np.zeros(N, F32)
            if zero_point is not None:
                # 128 + q as the operand; the accumulator opens at -128 sum_i x_i
                for z in zero_point:
                    temp = (temp + z[gi]).astype(F32)
                qcol = qcol + F32(NIBBLE_BIAS)
            elif defer or row:
                qcol = qcol + F32(NIBBLE_BIAS)   # nothing paid back yet
            for t in range(GGML_GROUP_SIZE):
                temp = (temp + qcol[:, t] * xcol[t]).astype(F32)

            # s'/m' are uint8 -> bf16 exact
            if group_sum_exact:
                acc_s = (acc_s.astype(np.float64)
                         + temp.astype(np.float64) * s8[:, gi]).astype(F32)
            elif group_limb == "safe":
                acc_s = mac_f32_bf16_safe(acc_s, temp, s8[:, gi])
            elif group_limb:
                acc_s = mac_f32_bf16_low(acc_s, temp, s8[:, gi])
            else:
                acc_s = (acc_s + bf16(temp) * s8[:, gi]).astype(F32)

            if defer:
                # -128 s'_j sum_i x_i into the scale accumulator, smallest limb
                # first. 128 * s'_j is exact (s'_j is a uint8 in bf16, 128 a
                # power of two), so each product is exact; only the fp32 adds
                # round.
                for lb in defer_limbs:
                    acc_s = (acc_s - F32(NIBBLE_BIAS) * s8[:, gi] * lb[gi]).astype(F32)

            if row:
                # sum_j s'_j sum_i x_i, smallest limb first.  Same two MACs the
                # min path costs, and no 128 anywhere in the loop.
                for lb in row_limbs:
                    acc_b = (acc_b + s8[:, gi] * lb[gi]).astype(F32)

            if group_sum_exact and reduce_exact:
                acc_m = (acc_m.astype(np.float64)
                         + m8[:, gi].astype(np.float64) * b_red[gi]).astype(F32)
            else:
                if reduce_limb:   # smallest limb first, as bf16_limb does
                    acc_m = (acc_m + m8[:, gi] * b_red_lo[gi]).astype(F32)
                acc_m = (acc_m + m8[:, gi] * b_red[gi]).astype(F32)

        a_S, a_M = S[:, b], Mneg[:, b]
        if fold_min_per_sb:
            # combine inside the block, one running accumulator
            t = mac_f32_bf16_low(np.zeros(N, F32), acc_s, a_S)
            t = mac_f32_bf16_low(t, acc_m, a_M)
            c_1 = (c_1 + t).astype(F32)
        elif sb_limb == "low":
            c_s = mac_f32_bf16_low(c_s, acc_s, a_S)
            c_m = mac_f32_bf16_low(c_m, acc_m, a_M)
        elif sb_limb == "safe":
            c_s = mac_f32_bf16_safe(c_s, acc_s, a_S)
            c_m = mac_f32_bf16_safe(c_m, acc_m, a_M)
        elif sb_limb == "safe_s":       # what the kernel does: acc_s carries the
            c_s = mac_f32_bf16_safe(c_s, acc_s, a_S)   # bias and needs 3 limbs,
            c_m = mac_f32_bf16_low(c_m, acc_m, a_M)    # acc_m does not
        elif sb_limb == "bf16":                            # pre-fix behaviour
            c_s = (c_s + bf16(acc_s) * a_S).astype(F32)
            c_m = (c_m + bf16(acc_m) * a_M).astype(F32)
        elif sb_limb == "exact":
            c_s = (c_s.astype(np.float64) + acc_s.astype(np.float64) * a_S).astype(F32)
            c_m = (c_m.astype(np.float64) + acc_m.astype(np.float64) * a_M).astype(F32)
        if row:
            # acc_b holds a quantity ~15x the answer, so its fold takes the
            # 3-limb form regardless of sb_limb -- it is once per super-block.
            c_b = mac_f32_bf16_safe(c_b, acc_b, a_S)

    if fold_min_per_sb:
        return c_1
    if row:
        # _Q4K_finalize: the only place 128 appears, once per 32-row block
        return (c_s + c_m - F32(NIBBLE_BIAS) * c_b).astype(F32)
    return (c_s + c_m).astype(F32)      # _Q4K_finalize


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def prepare(tensor_bytes, K, rows):
    """ggml Q4_K bytes -> (ggml exact W, FLM refit metadata)."""
    t, u, q = GGUFTensor.unpack_q4_k(tensor_bytes, K)
    t, u, q = t[:rows].numpy(), u[:rows].numpy(), q[:rows].numpy()

    # ggml reference: w = t_j q - u_j, exact
    W_ggml = (np.repeat(t, 32, axis=1).astype(np.float64) * q
              - np.repeat(u, 32, axis=1).astype(np.float64))

    # FLM refit, exactly as _pack_q4k does it (per 256-col super-block)
    gps = SUPER_BLOCK // GGML_GROUP_SIZE
    S, s8 = _refit_one_side(t.reshape(-1, gps), search=3)
    M, m8 = _refit_one_side(u.reshape(-1, gps), search=3)
    S = S.reshape(rows, -1)
    M = M.reshape(rows, -1)
    s8 = s8.reshape(t.shape)
    m8 = m8.reshape(u.shape)

    # what the core actually loads: S and -M stored as bf16
    S_b = bf16(S)
    Mneg_b = bf16(-M)

    W_flm = (np.repeat(S_b, gps, axis=1).astype(np.float64) * s8 .astype(np.float64)
             ).repeat(32, axis=1) * q \
        + (np.repeat(Mneg_b, gps, axis=1).astype(np.float64) * m8.astype(np.float64)
           ).repeat(32, axis=1)
    return W_ggml, W_flm, s8.astype(F32), m8.astype(F32), S_b, Mneg_b, q.astype(F32)


def rel(err, ref):
    return float(np.sqrt((err ** 2).mean()) / np.sqrt((ref ** 2).mean()))


def enob(r):
    return -np.log2(r * np.sqrt(12)) if r > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="gguf_files/qwen3.6-q4k.gguf")
    ap.add_argument("--tensor", default="blk.0.attn_qkv.weight")
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from gguf import GGUFReader
    r = GGUFReader(args.gguf)
    tt = next(t for t in r.tensors if t.name == args.tensor)
    K = int(tt.shape[0])
    print(f"{args.tensor}  K={K}  rows={args.rows}  ({tt.tensor_type.name})\n")

    W_ggml, W_flm, s8, m8, S, Mneg, q = prepare(tt.data, K, args.rows)

    print(f"  weight refit (FLM W' vs ggml W)          rel L2 = "
          f"{rel(W_flm - W_ggml, W_ggml):.3e}\n")

    rng = np.random.default_rng(args.seed)
    variants = {
        "NPU as shipped                       ": dict(),
        "  ablate: group_sum kept fp32        ": dict(group_sum_exact=True),
        "  ablate: super-block MAC exact      ": dict(sb_limb="exact"),
        "  ablate: b_col_reduce exact         ": dict(reduce_exact=True),
        "  ablate: all three                  ": dict(group_sum_exact=True,
                                                      sb_limb="exact",
                                                      reduce_exact=True),
        "variant: 3-limb (mac_f32_bf16_safe)  ": dict(sb_limb="safe"),
        "variant: fold a_M per super-block    ": dict(fold_min_per_sb=True),
        "PRE-FIX: plain bf16 super-block MAC  ": dict(sb_limb="bf16"),
        "FIX A: limb the group sum            ": dict(group_limb=True),
        "FIX B: limb b_col_reduce_add         ": dict(reduce_limb=True),
        "FIX A+B                              ": dict(group_limb=True,
                                                      reduce_limb=True),
        "NIBBLE BIAS: pay back w/ 2 bf16 limbs": dict(group_limb=True,
                                                      reduce_limb=True,
                                                      nibble_bias="limb2"),
        "NIBBLE BIAS: pay back w/ 3 bf16 limbs": dict(group_limb=True,
                                                      reduce_limb=True,
                                                      nibble_bias="limb3"),
        "NIBBLE BIAS: pay back in fp32        ": dict(group_limb=True,
                                                      reduce_limb=True,
                                                      nibble_bias="f32"),
        "NIBBLE BIAS: defer to acc_s, 2 limbs ": dict(group_limb=True,
                                                      reduce_limb=True,
                                                      nibble_bias="defer2"),
        "NIBBLE BIAS: defer to acc_s, 3 limbs ": dict(group_limb=True,
                                                      reduce_limb=True,
                                                      nibble_bias="defer3"),
        "NIBBLE BIAS: defer, 2 limbs, safe grp": dict(group_limb="safe",
                                                      reduce_limb=True,
                                                      nibble_bias="defer2"),
        "NIBBLE BIAS: defer, 3 limbs, safe grp": dict(group_limb="safe",
                                                      reduce_limb=True,
                                                      nibble_bias="defer3"),
        "NIBBLE BIAS: row acc, 2 limbs, low grp": dict(group_limb=True,
                                                       reduce_limb=True,
                                                       nibble_bias="row2"),
        "NIBBLE BIAS: row acc, 2 limbs, safe grp": dict(group_limb="safe",
                                                        reduce_limb=True,
                                                        nibble_bias="row2"),
        "NIBBLE BIAS: row acc, 3 limbs, safe grp": dict(group_limb="safe",
                                                        reduce_limb=True,
                                                        nibble_bias="row3"),
        "NIBBLE BIAS: row acc, 2 limbs, all safe": dict(group_limb="safe",
                                                        sb_limb="safe",
                                                        reduce_limb=True,
                                                        nibble_bias="row2"),
        "NIBBLE BIAS: row acc, 3 limbs, all safe": dict(group_limb="safe",
                                                        sb_limb="safe",
                                                        reduce_limb=True,
                                                        nibble_bias="row3"),
        "NIBBLE BIAS: row acc, 2 limbs, low+sbsafe": dict(group_limb=True,
                                                        sb_limb="safe",
                                                        reduce_limb=True,
                                                        nibble_bias="row2"),
        "NIBBLE BIAS: row acc, 2 limbs, AS BUILT": dict(group_limb="safe",
                                                        sb_limb="safe_s",
                                                        reduce_limb=True,
                                                        nibble_bias="row2"),
        "NIBBLE BIAS: row acc, 3 limbs, as built": dict(group_limb="safe",
                                                        sb_limb="safe_s",
                                                        reduce_limb=True,
                                                        nibble_bias="row3"),
        "NIBBLE BIAS: defer, 3 limbs + safe   ": dict(group_limb=True,
                                                      reduce_limb=True,
                                                      sb_limb="safe",
                                                      nibble_bias="defer3"),
    }
    acc = {k: [] for k in variants}
    acc_tot = {k: [] for k in variants}

    for trial in range(args.trials):
        x = bf16(rng.standard_normal(K).astype(F32))
        y_flm = W_flm @ x.astype(np.float64)     # exact matmul, FLM weights
        y_ggml = W_ggml @ x.astype(np.float64)   # exact matmul, ggml weights (llama.cpp)
        for name, kw in variants.items():
            y = npu_mvm(x, s8, m8, S, Mneg, q, **kw)
            acc[name].append(rel(y - y_flm, y_ggml))       # datapath only
            acc_tot[name].append(rel(y - y_ggml, y_ggml))  # vs llama.cpp

    print(f"  {'':37s} {'datapath':>11s} {'  total vs llama.cpp':>22s}")
    for name in variants:
        d = float(np.mean(acc[name]))
        t = float(np.mean(acc_tot[name]))
        print(f"  {name:37s} {d:11.3e}   {t:9.3e}  ({enob(t):5.2f} bits)")

    print("\n  'datapath' = NPU vs an exact matmul with the SAME refit weights.")
    print("  'total'    = NPU vs an exact matmul with the ggml weights = excess over llama.cpp.")


if __name__ == "__main__":
    main()
