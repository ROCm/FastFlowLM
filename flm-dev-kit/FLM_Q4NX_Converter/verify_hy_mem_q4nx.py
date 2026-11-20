#!./venv/bin/python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Check a Hy-Memory-30B-A3B model.q4nx against the GGUF it came from.

Decodes the packed q4k / q8nx blocks back to float32 and compares them with the
GGUF dequantization (and, with --hf, with the original BF16 checkpoint):

  * every expected tensor is present with the expected dtype and shape
  * BF16 / F32 tensors (embedding, norms, router, expert bias) are exact
  * Q4_K: the 4-bit codes are bit-identical to the GGUF's; the values differ
    only by the (BF16 S', uint8 s') super-block re-fit of _pack_q4k
  * Q8_0 (lm_head): codes identical, scales are bf16_RNE(fp16 scale)

Usage:
    ./verify_hy_mem_q4nx.py --gguf language.gguf --q4nx out/model.q4nx [--hf model.safetensors]
"""
import argparse

import numpy as np
import torch
from gguf import GGUFReader, dequantize
from safetensors import safe_open

ROWS = 32            # row_block_size
COLS = 256           # col_block_size
GROUP = 32           # weights per Q4_K / Q8_0 group
GROUPS = COLS // GROUP
Q4K_BLOCK_BYTES = ROWS * GROUPS * 2 + ROWS * COLS // 2 + ROWS * 2 * 2     # 4736
Q8_BLOCK_BYTES = ROWS * GROUPS * 2 + ROWS * COLS                          # 4352


# ---------------------------------------------------------------- block decoders
def decode_q4k(blob: np.ndarray):
    """[P, U, 4736] int8 -> (weights [P*32, U*256] fp32, codes [P*32, U*256] uint8).

    Block byte layout (_pack_q4k), everything column major over the block:
        uint8 s8[8 groups][32 rows]    per-group scale
        uint8 m8[8 groups][32 rows]    per-group min
        uint4 q [256 cols][16 bytes]   rows paired per byte, even row = low nibble
        bf16  S [32 rows]              super-block scale
        bf16  Mneg[32 rows]            super-block min, stored negated
    w[r, c] = S[r] * s8[g, r] * q[r, c] + Mneg[r] * m8[g, r],   g = c // 32
    """
    P, U, _ = blob.shape
    b = blob.reshape(P * U, Q4K_BLOCK_BYTES).view(np.uint8)

    # cut the block into its five planes
    o = 0
    s8 = b[:, o:o + GROUPS * ROWS].reshape(-1, GROUPS, ROWS).astype(np.float32); o += GROUPS * ROWS
    m8 = b[:, o:o + GROUPS * ROWS].reshape(-1, GROUPS, ROWS).astype(np.float32); o += GROUPS * ROWS
    qb = b[:, o:o + COLS * ROWS // 2].reshape(-1, COLS, ROWS // 2);              o += COLS * ROWS // 2
    S = bf16_bytes_to_f32(b[:, o:o + ROWS * 2]);                                 o += ROWS * 2
    Mneg = bf16_bytes_to_f32(b[:, o:o + ROWS * 2]);                              o += ROWS * 2
    assert o == Q4K_BLOCK_BYTES

    # unpack nibbles: byte k of a column holds rows 2k (low) and 2k+1 (high)
    q = np.stack([qb & 0x0F, qb >> 4], axis=-1).reshape(-1, COLS, ROWS)   # [blk, col, row]
    q = q.transpose(0, 2, 1)                                              # [blk, row, col]

    # broadcast group values over their 32 columns: [blk, g, row] -> [blk, row, col]
    scale = np.repeat((S[:, None, :] * s8).transpose(0, 2, 1), GROUP, axis=2)
    minv = np.repeat((Mneg[:, None, :] * m8).transpose(0, 2, 1), GROUP, axis=2)
    w = scale * q.astype(np.float32) + minv

    return blocks_to_matrix(w, P, U), blocks_to_matrix(q, P, U)


def decode_q8(blob: np.ndarray):
    """[P, U, 4352] int8 -> (weights, codes).  bf16 scales[8 groups][32 rows], then
    int8 q[256 cols][32 rows] (parallel_size == row_block_size, so plain column major)."""
    P, U, _ = blob.shape
    b = blob.reshape(P * U, Q8_BLOCK_BYTES)
    d = bf16_bytes_to_f32(b[:, :GROUPS * ROWS * 2].view(np.uint8)).reshape(-1, GROUPS, ROWS)
    q = b[:, GROUPS * ROWS * 2:].reshape(-1, COLS, ROWS).transpose(0, 2, 1)      # [blk, row, col]
    scale = np.repeat(d.transpose(0, 2, 1), GROUP, axis=2)
    w = scale * q.astype(np.float32)
    return blocks_to_matrix(w, P, U), blocks_to_matrix(q, P, U)


def bf16_bytes_to_f32(raw: np.ndarray) -> np.ndarray:
    # bf16 is the top half of an fp32: shift the 16 bits up and reinterpret
    u16 = np.ascontiguousarray(raw).view(np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def blocks_to_matrix(x: np.ndarray, P: int, U: int) -> np.ndarray:
    # [P*U, 32, 256] -> [P*32, U*256]
    return x.reshape(P, U, ROWS, COLS).transpose(0, 2, 1, 3).reshape(P * ROWS, U * COLS)


# ---------------------------------------------------------------- GGUF side
def gguf_q4k_codes(data: np.ndarray, cols: int) -> np.ndarray:
    """Raw Q4_K nibbles in natural (row, col) order (same walk as GGUFTensor.unpack_q4_k)."""
    blocks = data.view(np.uint8).reshape(-1, 144)
    qb = blocks[:, 16:].reshape(-1, 4, 32)
    q = np.stack([qb & 0x0F, qb >> 4], axis=2).reshape(-1, 256)
    return q.reshape(-1, cols)


def gguf_q8_codes(data: np.ndarray, cols: int) -> np.ndarray:
    blocks = data.view(np.uint8).reshape(-1, 34)
    return blocks[:, 2:].view(np.int8).reshape(-1, cols)


# ---------------------------------------------------------------- metrics
def stats(a: np.ndarray, ref: np.ndarray) -> str:
    a = a.astype(np.float64).ravel()
    ref = ref.astype(np.float64).ravel()
    cos = a @ ref / (np.linalg.norm(a) * np.linalg.norm(ref))
    rel = np.linalg.norm(a - ref) / np.linalg.norm(ref)
    return f"cos {cos:.7f}  relRMSE {rel:.2e}  maxabs {np.abs(a - ref).max():.2e}"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--q4nx", required=True)
    ap.add_argument("--hf", default=None, help="optional original BF16 model.safetensors")
    ap.add_argument("--layers", default="0,1,23,47", help="layers to decode in full")
    ap.add_argument("--experts", default="0,77,127", help="experts to decode per MoE layer")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    experts = [int(x) for x in args.experts.split(",")]

    reader = GGUFReader(args.gguf)
    g = {t.name: t for t in reader.tensors}
    nx = safe_open(args.q4nx, framework="pt")
    hf = safe_open(args.hf, framework="pt") if args.hf else None
    failures = []

    def fail(msg):
        print(f"  FAIL {msg}")
        failures.append(msg)

    # ---- 1. inventory: names, dtypes, shapes for every layer
    print("== inventory")
    n_layers = 48
    expect = {
        "model.embed_tokens.weight": (torch.bfloat16, (120832, 2048)),
        "model.norm.weight": (torch.bfloat16, (2048,)),
        "lm_head.weight": (torch.int8, (120832 // ROWS, 2048 // COLS, Q8_BLOCK_BYTES)),
    }

    def q4k_shape(rows, cols):
        return (torch.int8, (rows // ROWS, cols // COLS, Q4K_BLOCK_BYTES))

    for L in range(n_layers):
        p = f"model.layer.{L}."
        expect[p + "self_attn.q_proj.weight"] = q4k_shape(4096, 2048)
        expect[p + "self_attn.k_proj.weight"] = q4k_shape(512, 2048)
        expect[p + "self_attn.v_proj.weight"] = q4k_shape(512, 2048)
        expect[p + "self_attn.o_proj.weight"] = q4k_shape(2048, 4096)
        expect[p + "self_attn.q_norm.weight"] = (torch.bfloat16, (128,))
        expect[p + "self_attn.k_norm.weight"] = (torch.bfloat16, (128,))
        expect[p + "input_layernorm.weight"] = (torch.bfloat16, (2048,))
        expect[p + "post_attention_layernorm.weight"] = (torch.bfloat16, (2048,))
        if L == 0:
            expect[p + "mlp.gate_proj.weight"] = q4k_shape(6912, 2048)
            expect[p + "mlp.up_proj.weight"] = q4k_shape(6912, 2048)
            expect[p + "mlp.down_proj.weight"] = q4k_shape(2048, 6912)
        else:
            expect[p + "moe_router.weight"] = (torch.bfloat16, (2048, 128))
            expect[p + "moe_expert_bias"] = (torch.float32, (128,))
            expect[p + "mlp.gate_exps_proj.weight"] = q4k_shape(128 * 768, 2048)
            expect[p + "mlp.up_exps_proj.weight"] = q4k_shape(128 * 768, 2048)
            expect[p + "mlp.down_exps_proj.weight"] = q4k_shape(128 * 2048, 768)
            expect[p + "mlp.share_gate_exps_proj.weight"] = q4k_shape(768, 2048)
            expect[p + "mlp.share_up_exps_proj.weight"] = q4k_shape(768, 2048)
            expect[p + "mlp.share_down_exps_proj.weight"] = q4k_shape(2048, 768)

    have = set(nx.keys())
    for name in sorted(set(expect) - have):
        fail(f"missing {name}")
    for name in sorted(have - set(expect)):
        fail(f"unexpected {name}")
    for name in sorted(set(expect) & have):
        sl = nx.get_slice(name)
        dtype, shape = expect[name]
        got_shape = tuple(sl.get_shape())
        got_dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "I8": torch.int8}.get(sl.get_dtype())
        if got_shape != shape or got_dtype != dtype:
            fail(f"{name}: {sl.get_dtype()} {got_shape}, expected {dtype} {shape}")
    print(f"  {len(have)} tensors in file, {len(expect)} expected")

    # ---- 2. exact tensors
    print("== BF16 / F32 tensors (must be exact)")

    def gdeq(name):
        t = g[name]
        return np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32)

    def check_exact(q4nx_name, ref):
        got = nx.get_tensor(q4nx_name).float().numpy()
        same = np.array_equal(got, ref)
        print(f"  {'ok  ' if same else 'FAIL'} {q4nx_name}")
        if not same:
            fail(f"{q4nx_name} not exact, {stats(got, ref)}")

    check_exact("model.embed_tokens.weight", gdeq("token_embd.weight"))
    check_exact("model.norm.weight", gdeq("output_norm.weight"))
    for L in layers:
        p = f"model.layer.{L}."
        check_exact(p + "self_attn.q_norm.weight", gdeq(f"blk.{L}.attn_q_norm.weight"))
        check_exact(p + "self_attn.k_norm.weight", gdeq(f"blk.{L}.attn_k_norm.weight"))
        check_exact(p + "input_layernorm.weight", gdeq(f"blk.{L}.attn_norm.weight"))
        check_exact(p + "post_attention_layernorm.weight", gdeq(f"blk.{L}.ffn_norm.weight"))
        if L > 0:
            check_exact(p + "moe_router.weight", gdeq(f"blk.{L}.ffn_gate_inp.weight").T)
            check_exact(p + "moe_expert_bias", gdeq(f"blk.{L}.exp_probs_b"))

    # ---- 3. quantized tensors
    print("== quantized tensors (codes must match; values differ only by the scale re-fit)")

    def check_quant(q4nx_name, gguf_name, hf_names=None, expert_rows=None):
        """expert_rows: rows per expert; decodes only args.experts of a 3D tensor."""
        t = g[gguf_name]
        cols = int(t.shape[0])
        blob_all = nx.get_tensor(q4nx_name).numpy()
        is_q8 = blob_all.shape[-1] == Q8_BLOCK_BYTES

        ref_all = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(-1, cols)
        if is_q8:
            codes_all = gguf_q8_codes(t.data, cols)
        else:
            codes_all = gguf_q4k_codes(t.data, cols)

        pieces = experts if expert_rows else [None]
        for e in pieces:
            if e is None:
                r0, r1, label = 0, ref_all.shape[0], q4nx_name
            else:
                r0, r1, label = e * expert_rows, (e + 1) * expert_rows, f"{q4nx_name}[expert {e}]"
            blob = blob_all[r0 // ROWS: r1 // ROWS]
            w, codes = decode_q8(blob) if is_q8 else decode_q4k(blob)
            ref = ref_all[r0:r1]
            same_codes = np.array_equal(codes.astype(np.int16), codes_all[r0:r1].astype(np.int16))
            line = f"  {'ok  ' if same_codes else 'FAIL'} {label:58s} vs GGUF {stats(w, ref)}"
            if hf is not None and hf_names is not None:
                hn = hf_names if e is None else hf_names.format(e=e)
                line += f" | vs HF cos {stats(w, hf.get_tensor(hn).float().numpy()).split()[1]}"
            print(line)
            if not same_codes:
                fail(f"{label}: codes differ from GGUF")

    check_quant("lm_head.weight", "output.weight", "lm_head.weight")
    for L in layers:
        p, b, h = f"model.layer.{L}.", f"blk.{L}.", f"model.layers.{L}."
        check_quant(p + "self_attn.q_proj.weight", b + "attn_q.weight", h + "self_attn.q_proj.weight")
        check_quant(p + "self_attn.k_proj.weight", b + "attn_k.weight", h + "self_attn.k_proj.weight")
        check_quant(p + "self_attn.v_proj.weight", b + "attn_v.weight", h + "self_attn.v_proj.weight")
        check_quant(p + "self_attn.o_proj.weight", b + "attn_output.weight", h + "self_attn.o_proj.weight")
        if L == 0:
            for k in ("gate", "up", "down"):
                check_quant(p + f"mlp.{k}_proj.weight", b + f"ffn_{k}.weight", h + f"mlp.{k}_proj.weight")
        else:
            for k in ("gate", "up", "down"):
                rows = 2048 if k == "down" else 768
                check_quant(p + f"mlp.{k}_exps_proj.weight", b + f"ffn_{k}_exps.weight",
                            h + f"mlp.experts.{{e}}.{k}_proj.weight", expert_rows=rows)
                check_quant(p + f"mlp.share_{k}_exps_proj.weight", b + f"ffn_{k}_shexp.weight",
                            h + f"mlp.shared_mlp.{k}_proj.weight")

    print()
    if failures:
        print(f"{len(failures)} FAILURES")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
