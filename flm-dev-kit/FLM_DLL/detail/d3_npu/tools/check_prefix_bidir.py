#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""End-to-end check of the d3_npu engine's prefix_bidir mode against the upstream BF16 CPU runtime.

    check_prefix_bidir.py --hf <BF16 HF checkpoint dir> --upstream <dir holding d3_runtime.py (turbo)> \
        --pkg <q4nx package dir named like its xclbins folder> --xclbin-root <dir with xclbins/<pkg name>/{mm,attn}.xclbin> \
        --harness <d3_engine_run> --prompts <jsonl rows with "ids"> [--work DIR]

For a few prompts and several state lengths S it computes the final-norm hidden state of the last token with
  * the reference: upstream ``PromptEncoder`` (BF16, CPU, plain SDPA path) with ``StateBlocks`` (prefix_bidir) and
    without (noncausal_full_attention), on the BF16 checkpoint, whatever its decision_config says;
  * the engine through ``d3_engine_run --state-file`` (prefix_bidir) and without it (noncausal), with attention on
    the host (D3_ATTN_ARRAY_MIN=0) and on the array (D3_ATTN_ARRAY_MIN=1: every prompt of one or more query-key pair).
and prints cosines. The checkpoint need not have been trained with the mask (the probabilities of a noncausal
model are meaningless under it): what is checked is that the engine computes the same function as the
reference, to the cosine the noncausal comparison reaches (about 0.999 for a GPTQ package), and that the
mask changes the result (prefix_bidir vs noncausal differ well beyond that).
Exit code 0 when every prefix_bidir engine-vs-reference cosine is >= 0.99 and at most --max-drop (default 0.003)
below the noncausal engine-vs-reference cosine of the same prompt, and the engine follows the mask wherever the
reference's hidden state is moved by it visibly (cos prefix_bidir vs noncausal < 0.995).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def bf16_rows(path: Path, width: int) -> np.ndarray:
    words = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return words.view(np.float32).reshape(-1, width)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--xclbin-root", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--prompts", required=True, help="jsonl, one object per row with 'ids' (token ids)")
    ap.add_argument("--work", default="check_prefix_bidir_work")
    ap.add_argument("--short", default="120,300,600", help="prompt lengths (nearest rows) for the host-attention cases")
    ap.add_argument("--long", default="2100,2600", help="prompt lengths for the array-attention cases")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-drop", type=float, default=0.003, help="allowed cosine drop of prefix_bidir below the noncausal comparison")
    ap.add_argument("--reference-only", action="store_true", help="write the reference and stop")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(args.prompts) if line.strip()]

    def nearest(n: int) -> list[int]:
        return min(rows, key=lambda r: abs(len(r["ids"]) - n))["ids"]

    # (ids, S) cases: boundary-ish and odd state lengths, one-token and many-token questions
    cases: list[tuple[list[int], int]] = []
    for n in map(int, args.short.split(",")):
        ids = nearest(n)
        L = len(ids)
        for S in (1, 127, 128, 129, L // 2, L - 1):
            if 0 < S < L:
                cases.append((ids, S))
    for n in map(int, args.long.split(",")):
        ids = nearest(n)
        L = len(ids)
        for S in (129, L // 2, L - 3):
            cases.append((ids, S))
    print(f"{len(cases)} cases, lengths {sorted({len(c[0]) for c in cases})}")

    # ---- reference
    ref_file = work / "ref.npz"
    if not ref_file.exists():
        import torch

        torch.set_num_threads(args.threads)
        sys.path.insert(0, args.upstream)
        import d3_runtime as rt

        backbone = rt.backbone_class(rt.backbone_type(Path(args.hf))).from_pretrained(
            args.hf, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cpu"}
        )
        backbone.eval().requires_grad_(False)
        encoder = rt.PromptEncoder(backbone.language_model)
        embed = backbone.get_input_embeddings()
        prefix, plain = [], []
        cache: dict[int, np.ndarray] = {}
        with torch.no_grad():
            for ids, S in cases:
                x = embed(torch.tensor([ids]))
                present = torch.ones(1, len(ids), dtype=torch.long)
                out = encoder(x, present, None, rt.StateBlocks.of(len(ids), [len(ids)], [S]))
                prefix.append(out[0, -1].float().numpy())
                key = id(ids)
                if key not in cache:
                    cache[key] = encoder(x, present)[0, -1].float().numpy()
                plain.append(cache[key])
                print(f"reference S={S} L={len(ids)} done", flush=True)
        np.savez(ref_file, prefix=np.stack(prefix), plain=np.stack(plain))
    ref = np.load(ref_file)
    if args.reference_only:
        return 0

    # ---- engine
    with open(work / "prompts.txt", "w") as f:
        for i, (ids, S) in enumerate(cases):
            f.write(f"c{i} " + " ".join(map(str, ids)) + "\n")
    (work / "state.txt").write_text("\n".join(str(S) for _, S in cases) + "\n")
    (work / "zero.txt").write_text("\n".join("0" for _ in cases) + "\n")
    width = ref["prefix"].shape[1]
    results = {}
    for tag, state, array_min in (("host_prefix", "state.txt", "0"), ("array_prefix", "state.txt", "1"),
                                  ("host_plain", "zero.txt", "0"), ("array_plain", "zero.txt", "1")):
        env = dict(os.environ, D3_ATTN_ARRAY_MIN=array_min)
        out = work / f"out_{tag}.bin"
        cmd = [args.harness, args.pkg, args.xclbin_root, str(work / "prompts.txt"), str(out), "--state-file", str(work / state)]
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL, timeout=3600)
        results[tag] = bf16_rows(out, width)

    ok = True
    print(f"\n{'S':>6} {'L':>6} | cos vs reference (prefix_bidir): host-attn  array-attn | plain engine vs plain ref: host array | ref prefix vs plain | engine prefix vs ref plain (host, array)")
    worst_drop = -1.0   # how far below its noncausal counterpart a prefix_bidir cosine falls (same prompt, same attention)
    worst = 1.0
    discriminated = 0
    for i, (ids, S) in enumerate(cases):
        c = {t: cos(results[t][i], ref["prefix"][i] if t.endswith("prefix") else ref["plain"][i]) for t in results}
        gap = cos(ref["prefix"][i], ref["plain"][i])
        wrong = {t: cos(results[t][i], ref["plain"][i]) for t in ("host_prefix", "array_prefix")}
        print(f"{S:>6} {len(ids):>6} | {c['host_prefix']:.6f}  {c['array_prefix']:.6f} | {c['host_plain']:.6f} {c['array_plain']:.6f} | {gap:.6f} | {wrong['host_prefix']:.6f} {wrong['array_prefix']:.6f}")
        for mode in ("host", "array"):
            worst = min(worst, c[f"{mode}_prefix"])
            worst_drop = max(worst_drop, c[f"{mode}_plain"] - c[f"{mode}_prefix"])
            if gap < 0.995:   # the mask moves the reference by more than the engine's error: it must be followed
                discriminated += 1
                if not c[f"{mode}_prefix"] > wrong[f"{mode}_prefix"]:
                    print(f"FAIL: S={S} L={len(ids)} {mode}: the engine is closer to the noncausal reference than to the prefix_bidir one")
                    ok = False
    print(f"\nworst prefix_bidir engine-vs-reference cosine {worst:.6f}; largest drop below the same prompt's noncausal cosine {worst_drop:.6f}")
    print(f"cases where the mask is visible (reference gap < 0.995) and was followed: {discriminated}")
    if worst < 0.99 or worst_drop > args.max_drop:
        print(f"FAIL: prefix_bidir cosine below 0.99 or more than {args.max_drop} below the noncausal comparison")
        ok = False
    if discriminated == 0:
        print("note: no case where the mask moves the reference visibly; the comparison cannot distinguish the modes")
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
