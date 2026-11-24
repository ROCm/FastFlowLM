# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Round-trip check for MiniCPM-V-4.7-1B's model.q4nx against its source GGUF.

Decodes every tensor in model.q4nx back to float32 and scores it against
gguf.dequantize() of the GGUF tensor it came from, after replaying the converter's
known reshapes (full-attention q_proj de-interleave, conv1d transpose, alpha/beta
16 -> 32 row duplication, lm_head row padding).  Also checks the S1 gate items that
can be checked from the artifact alone:

  * every tensor round-trips at cosine > --min-cos (default 0.999)
  * ssm_{alpha,beta}_proj.weight are Q8_0 with 32 rows, and every linear layer
    has both .bf16.weight twins
  * every name in minicpm_v_model_def.hpp exists with the descriptor's dtype
  * the packaged folder has config.json, tokenizer.json, tokenizer_config.json,
    chat_template.jinja

Usage (from FLM_Q4NX_Converter/, with the converter venv active):

    python verify_minicpm_v_q4nx.py
    python verify_minicpm_v_q4nx.py --layers 0 3 --verbose

The dequant_q4nx() decoder is importable on its own: it reads any 32x256-chunk
q4nx blob (Q4_K / Q8_0 / Q4_1 / Q4_0, keep_block_in_2D) back to (rows, cols).

Exit status is 0 only if every check passes.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from gguf import GGUFReader, dequantize
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DEFAULT_GGUF = "/path/to/models/MiniCPM-V-4.7-1B-hf/language-q4k-llamacpp.gguf"
DEFAULT_OUT = "/path/to/models/MiniCPM-V-4.7-1B-NPU2"
DEFAULT_CONFIG = os.path.join(HERE, "configs", "minicpm_v_1b.json")
DEFAULT_MODEL_DEF = os.path.join(REPO, "FLM_DLL", "detail", "minicpm_v_4_7_npu", "minicpm_v_model_def.hpp")

PACKAGED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")

ROW_BLOCK, COL_BLOCK, GROUP = 32, 256, 32
GROUPS = COL_BLOCK // GROUP  # 8 groups of 32 columns per chunk

# Bytes per 32x256 chunk -> format (model_converter._pack_q4k / _pack_q8nx / _pack_q4nx).
BLOCK_BYTES = {
    4736: "q4k",   # uint8 s[8][32] | uint8 m[8][32] | uint4 q[256][16] | bf16 S[32] | bf16 -M[32]
    8704: "q80",   # bf16 s[8][32]  | int8 q[256][32]
    5120: "q41",   # bf16 s[8][32]  | bf16 m[8][32] | uint4 q[256][16]
    4608: "q40",   # bf16 s[8][32]  | int4 q[256][16]
}

# minicpm_v_model_def.hpp dtype -> what that looks like in the q4nx file
DESC_DTYPE = {"flm_q4k": "q4k", "flm_q80": "q80", "flm_q41": "q41", "flm_q40": "q40",
              "flm_bf16": "BF16", "flm_f32": "F32"}


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _nibbles(qb: np.ndarray, signed: bool) -> np.ndarray:
    """(..., col, 16) bytes -> (..., col, 32) rows; even row in the low nibble."""
    lo = (qb & 0x0F).astype(np.int8)
    hi = (qb >> 4).astype(np.int8)
    q = np.stack([lo, hi], axis=-1).reshape(*qb.shape[:-1], ROW_BLOCK)
    if signed:
        q = np.where(q >= 8, q - 16, q).astype(np.int8)
    return q.astype(np.float32)


def q4nx_format(blob: np.ndarray) -> str:
    """Format name of a quantized q4nx blob, from its per-chunk byte count."""
    fmt = BLOCK_BYTES.get(blob.shape[-1]) if blob.ndim == 3 else None
    if fmt is None:
        raise ValueError(f"not a 32x256-chunk q4nx blob: shape {blob.shape}")
    return fmt


def dequant_q4nx(blob: np.ndarray) -> np.ndarray:
    """Decode an int8 q4nx blob of shape (row_blocks, col_blocks, chunk_bytes) to
    float32 (row_blocks*32, col_blocks*256).  Inverse of model_converter._pack_*."""
    fmt = q4nx_format(blob)
    b = np.ascontiguousarray(blob).view(np.uint8)
    p, u, _ = b.shape

    def u16(lo, hi):
        return np.ascontiguousarray(b[..., lo:hi]).view(np.uint16)

    if fmt == "q4k":
        s8 = b[..., 0:256].reshape(p, u, GROUPS, ROW_BLOCK).astype(np.float32)
        m8 = b[..., 256:512].reshape(p, u, GROUPS, ROW_BLOCK).astype(np.float32)
        q = _nibbles(b[..., 512:4608].reshape(p, u, COL_BLOCK, ROW_BLOCK // 2), signed=False)
        S = _bf16_to_f32(u16(4608, 4672)).reshape(p, u, 1, ROW_BLOCK)
        negM = _bf16_to_f32(u16(4672, 4736)).reshape(p, u, 1, ROW_BLOCK)
        scale = np.repeat(S * s8, GROUP, axis=2)         # (p, u, col, row)
        bias = np.repeat(negM * m8, GROUP, axis=2)
        w = scale * q + bias
    elif fmt == "q80":
        s = _bf16_to_f32(u16(0, 512)).reshape(p, u, GROUPS, ROW_BLOCK)
        q = b[..., 512:].view(np.int8).reshape(p, u, COL_BLOCK, ROW_BLOCK).astype(np.float32)
        w = np.repeat(s, GROUP, axis=2) * q
    elif fmt == "q41":
        s = _bf16_to_f32(u16(0, 512)).reshape(p, u, GROUPS, ROW_BLOCK)
        m = _bf16_to_f32(u16(512, 1024)).reshape(p, u, GROUPS, ROW_BLOCK)
        q = _nibbles(b[..., 1024:].reshape(p, u, COL_BLOCK, ROW_BLOCK // 2), signed=False)
        w = np.repeat(s, GROUP, axis=2) * q + np.repeat(m, GROUP, axis=2)
    else:  # q40
        s = _bf16_to_f32(u16(0, 512)).reshape(p, u, GROUPS, ROW_BLOCK)
        q = _nibbles(b[..., 512:].reshape(p, u, COL_BLOCK, ROW_BLOCK // 2), signed=True)
        w = np.repeat(s, GROUP, axis=2) * q

    # (p, u, col, row) -> (p, row, u, col) -> (rows, cols)
    return w.transpose(0, 3, 1, 2).reshape(p * ROW_BLOCK, u * COL_BLOCK)


def load_q4nx_tensor(f, name: str):
    """(format, float32 array) for one q4nx tensor."""
    t = f.get_tensor(name)
    dtype = f.get_slice(name).get_dtype()
    if dtype == "I8":
        blob = t.numpy()
        return q4nx_format(blob), dequant_q4nx(blob)
    import torch
    return dtype, t.to(torch.float32).numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return 1.0 if not a.any() and not b.any() else 0.0
    return float(a @ b / den)


def rel_rmse(a: np.ndarray, ref: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    den = np.sqrt(np.mean(ref ** 2))
    return float(np.sqrt(np.mean((a.astype(np.float64) - ref) ** 2)) / den) if den else 0.0


def build_name_map(config_path: str, gguf_names) -> dict:
    """q4nx name -> GGUF name, expanded the way model_converter._create_name_maps does.
    The .bf16 twins are expanded for every layer that has the quantized original,
    since the converter synthesizes them."""
    cfg = json.load(open(config_path))
    out = {}
    for info in cfg["name_map"].values():
        g, q = info["gguf_name"], info["q4nx_name"]
        # S7a added the vision name_map to this same config, but those tensors are
        # converted by `-t vision` into vision_weight.q4nx, not model.q4nx. Skip them
        # here or every one reads as missing from the language file.
        # verify_minicpm_v_vision_q4nx.py is their checker.
        if q.startswith("model.visual."):
            continue
        if "{bid}" not in g:
            out[q] = g
            continue
        src = g.replace(".bf16.weight", ".weight")
        rx = re.compile("^" + re.escape(src).replace(r"\{bid\}", r"(\d+)") + "$")
        for n in gguf_names:
            m = rx.match(n)
            if m:
                bid = int(m.group(1))
                out[q.format(bid=bid)] = g.format(bid=bid)
    return out


def reference(reader_tensors: dict, gguf_name: str, q4nx_name: str, meta: dict,
              alias: dict | None = None) -> np.ndarray:
    """gguf.dequantize() of the source tensor, with the converter's reshapes replayed.

    `alias` redirects a GGUF name the converter synthesized rather than read. A model with
    tied embeddings has no `output.weight` in its GGUF -- the converter copies it from
    `token_embd.weight` -- so scoring the head means scoring it against that copy's source.
    """
    gguf_name = (alias or {}).get(gguf_name, gguf_name)
    src = reader_tensors[gguf_name.replace(".bf16.weight", ".weight")]
    ref = np.asarray(dequantize(src.data, src.tensor_type), dtype=np.float32)

    layer = re.search(r"layers\.(\d+)\.", q4nx_name)
    full_attn = layer is not None and int(layer.group(1)) % meta["interval"] == meta["interval"] - 1
    if full_attn and q4nx_name.endswith("self_attn.q_proj.weight"):
        # qwen35.py: '(g p h) c -> (p g h) c', p=2, h=DH  -- q rows first, then gate rows
        cols = ref.shape[1]
        ref = ref.reshape(-1, 2, meta["head_dim"], cols).transpose(1, 0, 2, 3).reshape(-1, cols)
    elif q4nx_name.endswith("ssm_conv1d.weight"):
        ref = ref.T
    return ref


def compare(q4nx_name: str, fmt: str, got: np.ndarray, ref: np.ndarray):
    """Align got to ref and return (cos, rel_rmse, note); note flags a layout problem."""
    note = ""
    if q4nx_name.endswith(("ssm_alpha_proj.weight", "ssm_beta_proj.weight")):
        # qwen35.py duplicates the 16 rows to 32: repeat 'd c -> (r d) c', r=2
        r = ref.shape[0]
        if got.shape[0] == 2 * r and not np.array_equal(got[:r], got[r:]):
            note = "row copies differ"
        got = got[:r]
    if got.ndim == 2 and got.shape[0] > ref.shape[0]:
        pad = got[ref.shape[0]:]
        if pad.any():
            note = f"{pad.shape[0]} padded rows are not zero"
        got = got[:ref.shape[0]]
    if got.ndim == 2 and got.shape[1] > ref.shape[1]:
        got = got[:, :ref.shape[1]]
    if got.shape != ref.shape:
        return float("nan"), float("nan"), f"shape {got.shape} vs reference {ref.shape}"
    return cosine(got, ref), rel_rmse(got, ref), note


def descriptor_names(model_def: str):
    """[(dtype, name_template)] from the weight_desc_t(...) lines of the model def."""
    rx = re.compile(r'weight_desc_t\(\s*(flm_\w+)\s*,[^"]*"([^"]+)"\s*\)')
    return sorted(set(rx.findall(open(model_def).read())), key=lambda x: x[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--out", default=DEFAULT_OUT, help="packaged model folder holding model.q4nx")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--model-def", default=DEFAULT_MODEL_DEF)
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--tied-lm-head", action="store_true",
                    help="the model ties lm_head to the embedding (tie_word_embeddings), so its "
                         "GGUF has no output.weight; score lm_head.weight against "
                         "token_embd.weight, which is what the converter copied it from")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="only round-trip these layers (plus the non-layer tensors); default all")
    ap.add_argument("--verbose", "-v", action="store_true", help="print every tensor, not just failures")
    args = ap.parse_args()

    failures = []
    reader = GGUFReader(args.gguf)
    gguf_tensors = {t.name: t for t in reader.tensors}
    meta = {"interval": int(reader.fields["qwen35.full_attention_interval"].contents()),
            "head_dim": int(reader.fields["qwen35.attention.value_length"].contents())}
    n_layers = int(reader.fields["qwen35.block_count"].contents())
    linear_layers = [l for l in range(n_layers) if l % meta["interval"] != meta["interval"] - 1]

    q4nx_path = os.path.join(args.out, "model.q4nx")
    f = safe_open(q4nx_path, "pt")
    keys = set(f.keys())
    name_map = build_name_map(args.config, gguf_tensors.keys())

    # ---- 1. round trip -------------------------------------------------------
    print(f"[1] round trip vs gguf.dequantize(), cosine > {args.min_cos}")
    unmapped = sorted(keys - name_map.keys())
    for k in unmapped:
        failures.append(f"{k}: in model.q4nx but has no GGUF source in the name map")
    missing = sorted(name_map.keys() - keys)
    for k in missing:
        failures.append(f"{k}: expected from {name_map[k]} but absent from model.q4nx")

    gguf_alias = {"output.weight": "token_embd.weight"} if args.tied_lm_head else {}

    worst = (1.0, None)
    by_fmt = {}
    for k in sorted(keys & name_map.keys()):
        layer = re.search(r"layers\.(\d+)\.", k)
        if args.layers is not None and layer is not None and int(layer.group(1)) not in args.layers:
            continue
        fmt, got = load_q4nx_tensor(f, k)
        ref = reference(gguf_tensors, name_map[k], k, meta, gguf_alias)
        cos, rr, note = compare(k, fmt, got, ref)
        by_fmt.setdefault(fmt, []).append(cos)
        if cos < worst[0]:
            worst = (cos, k)
        bad = not (cos > args.min_cos) or note
        if bad:
            failures.append(f"{k}: cos {cos:.6f} relRMSE {rr:.2e} {note}".rstrip())
        if bad or args.verbose:
            print(f"    {'FAIL' if bad else 'ok  '} {k:62s} {fmt:5s} cos {cos:.7f} relRMSE {rr:.2e} {note}")
    for fmt, c in sorted(by_fmt.items()):
        print(f"    {fmt:5s} {len(c):4d} tensors   min cos {min(c):.7f}   mean cos {np.mean(c):.7f}")
    print(f"    worst: {worst[1]} at {worst[0]:.7f}")

    # ---- 2. alpha/beta Q8_0 + bf16 twins ------------------------------------
    print("[2] ssm_{alpha,beta}_proj: Q8_0 with 32 rows, .bf16 twins on every linear layer")
    n_twins = 0
    for l in linear_layers:
        for proj in ("alpha", "beta"):
            q = f"model.layers.{l}.linear_attn.ssm_{proj}_proj.weight"
            tw = f"model.layers.{l}.linear_attn.ssm_{proj}_proj.bf16.weight"
            if q not in keys:
                failures.append(f"{q}: missing")
            else:
                blob = f.get_slice(q)
                shape = blob.get_shape()
                if blob.get_dtype() != "I8" or len(shape) != 3 or BLOCK_BYTES.get(shape[-1]) != "q80" \
                        or shape[0] * ROW_BLOCK != 32:
                    failures.append(f"{q}: expected Q8_0 with 32 rows, got {blob.get_dtype()} {shape}")
            if tw not in keys:
                failures.append(f"{tw}: missing")
            else:
                n_twins += 1
                if f.get_slice(tw).get_dtype() != "BF16":
                    failures.append(f"{tw}: expected BF16, got {f.get_slice(tw).get_dtype()}")
    print(f"    {len(linear_layers)} linear layers, {n_twins} twins found (expect {2 * len(linear_layers)})")

    # ---- 3. descriptor names and dtypes -------------------------------------
    print(f"[3] names/dtypes vs {os.path.relpath(args.model_def, REPO)}")
    n_checked = 0
    for dtype, tmpl in descriptor_names(args.model_def):
        present = [l for l in range(n_layers) if (tmpl % l if "%d" in tmpl else tmpl) in keys]
        if not present:
            failures.append(f"{tmpl}: named by the descriptor, absent from model.q4nx")
            continue
        for l in present:
            name = tmpl % l if "%d" in tmpl else tmpl
            sl = f.get_slice(name)
            fmt = BLOCK_BYTES.get(sl.get_shape()[-1]) if sl.get_dtype() == "I8" else sl.get_dtype()
            n_checked += 1
            if fmt != DESC_DTYPE.get(dtype, dtype):
                failures.append(f"{name}: descriptor says {dtype}, model.q4nx has {fmt}")
    print(f"    {n_checked} tensors checked")

    # ---- 4. packaging ---------------------------------------------------------
    print(f"[4] packaged folder {args.out}")
    for fn in PACKAGED_FILES:
        if not os.path.isfile(os.path.join(args.out, fn)):
            failures.append(f"{fn}: missing from {args.out}")
    cfg_path = os.path.join(args.out, "config.json")
    if os.path.isfile(cfg_path):
        cfg = json.load(open(cfg_path))
        stray = [k for k in cfg if k.startswith("_comment")]
        if stray:
            failures.append(f"config.json still has comment keys: {stray}")
        # vision_model_weight is what makes _normalize_multi_modal() set is_vlm, so it
        # must agree with whether the vision tower actually shipped. Asserting the
        # invariant rather than "never present" keeps this check meaningful once S7b
        # lands instead of turning it into a false failure the moment vision works.
        has_vw = "vision_model_weight" in cfg
        vw_file = os.path.isfile(os.path.join(args.out, "vision_weight.q4nx"))
        if has_vw and not vw_file:
            failures.append("config.json declares vision_model_weight but no "
                            "vision_weight.q4nx is present; is_vlm would be true "
                            "with nothing behind it")
        if vw_file and not has_vw:
            print("    note: vision_weight.q4nx present but config.json has no "
                  "vision_model_weight -- is_vlm stays false, image input rejected "
                  "(expected until S7b enables the VLM path)")

    print()
    if failures:
        print(f"FAIL ({len(failures)})")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
