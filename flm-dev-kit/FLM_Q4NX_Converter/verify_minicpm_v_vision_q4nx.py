# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Round-trip check for MiniCPM-V-4.7-1B's vision_weight.q4nx against its mmproj GGUF.

vision_weight.q4nx is plain safetensors, every tensor BF16.  Each one is scored
against gguf.dequantize() of the GGUF tensor(s) it came from, after replaying
what q4nx/models/minicpm_v.py does to it:

  * attn.qkv (27 blocks + vit_merger) is attn_q / attn_k / attn_v stacked on
    the row axis, [q; k; v], biases likewise
  * MM weights are zero-padded on N (rows) and K (cols) up to the vision_mm
    tile -- the pad must be exactly zero, and is cut off before scoring

Checks:

  [1] every tensor round-trips at cosine > --min-cos (default 0.999), and every
      GGUF tensor is consumed by exactly one q4nx tensor
  [2] every tensor is BF16 and the MM weights have exactly the padded shapes
      the Qwen3_8 vision_mm designs need (ffn_up N 4608, ffn_down K 4608,
      ds_ffn_up N 17280, ds_ffn_down K 17280, qkv N 3456, mm.down N 1024)
  [3] with --hf, the same tensors against the BF16 HF checkpoint, GATED: the
      fraction of differing elements must stay under --max-hf-diff (default
      1e-3).  A GGUF whose tower was exported from the same weights differs only
      in the few elements an F16 file flushes to subnormals (|w| < 6.1e-5).  A
      GGUF exported from another snapshot or through another rounding differs in
      tens of percent of every matrix by one bf16 ulp: that cannot be seen by
      the round trip in [1] (which only compares against the GGUF itself), yet
      moves image_embeds to a cosine of ~0.95 against the HF model.

Usage (from FLM_Q4NX_Converter/, with the converter venv active):

    python verify_minicpm_v_vision_q4nx.py
    python verify_minicpm_v_vision_q4nx.py --hf /path/to/models/MiniCPM-V-4.7-1B-hf/model.safetensors -v

Exit status is 0 only if [1] and [2] pass.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from gguf import GGUFReader, dequantize
from safetensors import safe_open

from verify_minicpm_v_q4nx import cosine, rel_rmse

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_GGUF = "/path/to/models/MiniCPM-V-4.7-1B-hf/mmproj-model-f16.gguf"
DEFAULT_OUT = "/path/to/models/MiniCPM-V-4.7-1B-NPU2"
DEFAULT_CONFIG = os.path.join(HERE, "configs", "minicpm_v_1b.json")

D, F, F_DS, MERGED, OUT = 1152, 4304, 17216, 4608, 1024

#: GGUF tensors that are deliberately not part of vision_weight.q4nx.  v.tok_embd_sep (4 x 1024, in the GGUF re-uploaded on
#: 2026-10-11) is four rows of the language model's own token embedding (slice start/end, newline, image start) that the
#: GGUF consumer wants next to the tower; FLM reads those tokens from its own embedding table.
IGNORED_GGUF = {"v.tok_embd_sep"}

#: q4nx name (with {bid} for the 27 blocks) -> exact stored shape.  The four
#: padded ones are the point of the exercise; the rest pin the tile fit.
EXPECTED_SHAPES = {
    "model.visual.blocks.{bid}.attn.qkv.weight":        (3 * D, D),     # 3456 = 9 * 384
    "model.visual.blocks.{bid}.attn.qkv.bias":          (3 * D,),
    "model.visual.blocks.{bid}.attn.proj.weight":       (D, D),
    "model.visual.blocks.{bid}.mlp.linear_fc1.weight":  (4608, D),      # N 4304 -> 4608
    "model.visual.blocks.{bid}.mlp.linear_fc1.bias":    (4608,),
    "model.visual.blocks.{bid}.mlp.linear_fc2.weight":  (D, 4608),      # K 4304 -> 4608
    "model.visual.vit_merger.attn.qkv.weight":          (3 * D, D),
    "model.visual.vit_merger.attn.proj.weight":         (D, D),
    "model.visual.vit_merger.mlp.linear_fc1.weight":    (17280, MERGED),  # N 17216 -> 17280
    "model.visual.vit_merger.mlp.linear_fc1.bias":      (17280,),
    "model.visual.vit_merger.mlp.linear_fc2.weight":    (D, 17280),       # K 17216 -> 17280
    "model.visual.merger.linear_fc1.weight":            (MERGED, MERGED),
    "model.visual.merger.linear_fc2.weight":            (OUT, MERGED),    # N 1024, design B, unpadded
    "model.visual.merger.linear_fc2.bias":              (OUT,),
    "model.visual.patch_embed.proj.weight":             (D, 3, 14, 14),
    "model.visual.pos_embed.weight":                    (4900, D),
}

#: q4nx suffix -> HF key suffix, for --hf.  Blocks are model.vision_tower.encoder.layers.N.
HF_NAMES = {
    "model.visual.patch_embed.proj":   "model.vision_tower.embeddings.patch_embedding",
    "model.visual.pos_embed":          "model.vision_tower.embeddings.position_embedding",
    "model.visual.merger.norm":        "model.vision_tower.post_layernorm",
    "model.visual.merger.pre_norm":    "model.merger.mlp.0.pre_norm",
    "model.visual.merger.linear_fc1":  "model.merger.mlp.0.linear_1",
    "model.visual.merger.linear_fc2":  "model.merger.mlp.0.linear_2",
    "model.visual.vit_merger.norm1":   "model.vision_tower.vit_merger.layer_norm1",
    "model.visual.vit_merger.pre_norm": "model.vision_tower.vit_merger.pre_norm",
    "model.visual.vit_merger.mlp.linear_fc1": "model.vision_tower.vit_merger.linear_1",
    "model.visual.vit_merger.mlp.linear_fc2": "model.vision_tower.vit_merger.linear_2",
    "model.visual.vit_merger.attn.proj": "model.vision_tower.vit_merger.self_attn.out_proj",
}
HF_BLOCK = {"norm1": "layer_norm1", "norm2": "layer_norm2", "attn.proj": "self_attn.out_proj",
            "mlp.linear_fc1": "mlp.fc1", "mlp.linear_fc2": "mlp.fc2"}


def build_sources(config_path: str, gguf_names) -> dict:
    """q4nx name -> [GGUF source names].  attn_qkv does not exist in the GGUF;
    the converter builds it from attn_q, attn_k, attn_v, in that order."""
    cfg = json.load(open(config_path))
    out = {}
    for info in cfg["name_map"].values():
        g, q = info["gguf_name"], info["q4nx_name"]
        if not (g.startswith("v.") or g.startswith("mm.")):
            continue                                  # language entry
        probe = g.replace("attn_qkv", "attn_q")
        if "{bid}" in probe:
            rx = re.compile("^" + re.escape(probe).replace(r"\{bid\}", r"(\d+)") + "$")
            bids = sorted(int(m.group(1)) for n in gguf_names for m in [rx.match(n)] if m)
        else:
            bids = [None] if probe in gguf_names else []
        for bid in bids:
            gn = g.format(bid=bid) if bid is not None else g
            qn = q.format(bid=bid) if bid is not None else q
            out[qn] = ([gn.replace("attn_qkv", f"attn_{p}") for p in "qkv"]
                       if "attn_qkv" in gn else [gn])
    return out


def reference(gguf_tensors: dict, sources) -> np.ndarray:
    parts = [np.asarray(dequantize(gguf_tensors[s].data, gguf_tensors[s].tensor_type),
                        dtype=np.float32) for s in sources]
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


def unpad(got: np.ndarray, ref: np.ndarray):
    """Cut got down to ref's shape; (trimmed, note) where note flags a non-zero pad."""
    note = ""
    if got.ndim != ref.ndim or any(g < r for g, r in zip(got.shape, ref.shape)):
        return got, f"shape {got.shape} smaller than / not like source {ref.shape}"
    core = tuple(slice(0, r) for r in ref.shape)
    mask = np.ones(got.shape, dtype=bool)
    mask[core] = False
    if mask.any() and got[mask].any():
        note = f"{int((got[mask] != 0).sum())} padded elements are not zero"
    return got[core], note


def bf16_rne(x: np.ndarray) -> np.ndarray:
    """float32 -> nearest-even bf16, back as float32 (torch's .to(bfloat16))."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def expected_shape(name: str):
    for tmpl, shape in EXPECTED_SHAPES.items():
        rx = "^" + re.escape(tmpl).replace(r"\{bid\}", r"\d+") + "$"
        if re.match(rx, name):
            return shape
    return None


def hf_key(name: str):
    m = re.match(r"^model\.visual\.blocks\.(\d+)\.(.+)\.(weight|bias)$", name)
    if m:
        bid, mid, kind = m.groups()
        if mid == "attn.qkv":
            return [f"model.vision_tower.encoder.layers.{bid}.self_attn.{p}_proj.{kind}" for p in "qkv"]
        return [f"model.vision_tower.encoder.layers.{bid}.{HF_BLOCK[mid]}.{kind}"]
    stem, kind = name.rsplit(".", 1)
    if stem == "model.visual.vit_merger.attn.qkv":
        return [f"model.vision_tower.vit_merger.self_attn.{p}_proj.{kind}" for p in "qkv"]
    return [f"{HF_NAMES[stem]}.{kind}"] if stem in HF_NAMES else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--out", default=DEFAULT_OUT, help="folder holding vision_weight.q4nx")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--hf", default=None, help="HF model.safetensors to also compare against (gated by --max-hf-diff)")
    ap.add_argument("--max-hf-diff", type=float, default=1e-3,
                    help="largest allowed fraction of elements that differ from the HF checkpoint (default 1e-3)")
    ap.add_argument("--verbose", "-v", action="store_true", help="print every tensor")
    args = ap.parse_args()

    failures = []
    reader = GGUFReader(args.gguf)
    gguf_tensors = {t.name: t for t in reader.tensors}
    q4nx_path = os.path.join(args.out, "vision_weight.q4nx")
    f = safe_open(q4nx_path, "pt")
    keys = set(f.keys())
    sources = build_sources(args.config, gguf_tensors.keys())

    # ---- 1. round trip -------------------------------------------------------
    print(f"[1] round trip vs gguf.dequantize(), cosine > {args.min_cos}")
    for k in sorted(keys - sources.keys()):
        failures.append(f"{k}: in vision_weight.q4nx but has no GGUF source in the name map")
    for k in sorted(sources.keys() - keys):
        failures.append(f"{k}: expected from {sources[k]} but absent")
    used = [s for k in keys & sources.keys() for s in sources[k]]
    for s in sorted(set(gguf_tensors) - set(used) - IGNORED_GGUF):
        failures.append(f"{s}: GGUF tensor not consumed by any q4nx tensor")
    for s in sorted({s for s in used if used.count(s) > 1}):
        failures.append(f"{s}: consumed more than once")

    worst, n_exact, n_total, cos_all = (1.0, None), 0, 0, []
    got_cache = {}
    for k in sorted(keys & sources.keys()):
        got = f.get_tensor(k).to(torch.float32).numpy()
        ref = reference(gguf_tensors, sources[k])
        trimmed, note = unpad(got, ref)
        if trimmed.shape != ref.shape:
            cos, rr = float("nan"), float("nan")
        else:
            cos, rr = cosine(trimmed, ref), rel_rmse(trimmed, ref)
            # The only loss allowed is the per-element bf16 rounding.
            exact = np.array_equal(trimmed, bf16_rne(ref))
            n_exact += exact
            if not exact:
                note = (note + " not bit-exact bf16(source)").strip()
        n_total += 1
        cos_all.append(cos)
        if cos < worst[0] or worst[1] is None:
            worst = (cos, k)
        bad = not (cos > args.min_cos) or bool(note)
        if bad:
            failures.append(f"{k}: cos {cos:.7f} relRMSE {rr:.2e} {note}".rstrip())
        if bad or args.verbose:
            print(f"    {'FAIL' if bad else 'ok  '} {k:58s} {str(tuple(got.shape)):16s} "
                  f"cos {cos:.12f} relRMSE {rr:.2e} {note}")
        got_cache[k] = got.shape
    print(f"    {n_total} tensors from {len(gguf_tensors)} GGUF tensors; min cos {min(cos_all):.12f}, "
          f"mean {np.mean(cos_all):.12f}; {n_exact}/{n_total} bit-exact to bf16(source)")
    print(f"    worst: {worst[1]} at {worst[0]:.12f}")

    # ---- 2. dtype and padded shapes ------------------------------------------
    print("[2] dtype BF16, padded MM shapes")
    n_shape = 0
    for k in sorted(keys):
        dt = f.get_slice(k).get_dtype()
        if dt != "BF16":
            failures.append(f"{k}: dtype {dt}, expected BF16")
        want = expected_shape(k)
        if want is not None:
            n_shape += 1
            if tuple(got_cache.get(k, f.get_slice(k).get_shape())) != want:
                failures.append(f"{k}: shape {tuple(f.get_slice(k).get_shape())}, expected {want}")
    for tmpl, want in EXPECTED_SHAPES.items():
        if "{bid}" not in tmpl:
            print(f"    {tmpl:58s} {want}")
        else:
            print(f"    {tmpl:58s} {want}  x27")
    print(f"    {n_shape} tensors checked against the table")

    # ---- 3. against the HF checkpoint (gated) ---------------------------------
    if args.hf:
        print(f"[3] vs HF checkpoint {args.hf} (differing elements <= {args.max_hf_diff:.0e})")
        hf = safe_open(args.hf, "pt")
        hf_keys = set(hf.keys())
        worst_hf, n_eq, n_el, n_t = (1.0, None), 0, 0, 0
        for k in sorted(keys):
            hk = hf_key(k)
            if hk is None or not all(h in hf_keys for h in hk):
                failures.append(f"{k}: no HF counterpart {hk}")
                continue
            ref = torch.cat([hf.get_tensor(h) for h in hk], dim=0).to(torch.float32).numpy()
            got, _ = unpad(f.get_tensor(k).to(torch.float32).numpy(), ref)
            cos = cosine(got, ref)
            eq = int((got == ref).sum())
            n_eq += eq; n_el += ref.size; n_t += 1
            if (ref.size - eq) / ref.size > args.max_hf_diff:
                failures.append(f"{k}: {ref.size - eq}/{ref.size} elements differ from the HF checkpoint "
                                f"({(ref.size - eq) / ref.size:.2e} > {args.max_hf_diff:.0e})")
            if cos < worst_hf[0] or worst_hf[1] is None:
                worst_hf = (cos, k)
            if args.verbose:
                print(f"    {k:58s} cos {cos:.12f} differ {ref.size - eq}/{ref.size} "
                      f"maxabs {np.abs(got - ref).max():.2e}")
        print(f"    {n_t} tensors; worst {worst_hf[1]} at {worst_hf[0]:.12f}; "
              f"{n_el - n_eq}/{n_el} elements differ ({(n_el - n_eq) / n_el:.2e})")

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
