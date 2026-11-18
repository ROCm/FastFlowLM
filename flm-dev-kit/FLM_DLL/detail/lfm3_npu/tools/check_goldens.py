#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Check the LFM3 engine (via lfm3_engine_run) against frozen goldens.

  python check_goldens.py --harness <lfm3_engine_run> --pkg <package dir> --goldens <goldens dir> [--qref <qref dir>] [--prompts a b]

Per prompt (ids come from <goldens>/<name>.json):
  logits   cos / top-1 of the last-position logits against <qref>/<name>.first_logits.pt (default <goldens>/qref_qad)
  greedy   how many of the Q-ref's greedy tokens the engine reproduces before the first difference
  split    prefill in two calls gives bit-identical logits to one call (continuation of conv state and KV)
Exit status is non-zero if a logits cosine is below --min-cos or a split run differs.
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--harness", required=True); ap.add_argument("--pkg", required=True); ap.add_argument("--goldens", required=True)
ap.add_argument("--qref", default=None); ap.add_argument("--prompts", nargs="*", default=None)
ap.add_argument("--min-cos", type=float, default=0.995); ap.add_argument("--greedy", type=int, default=48)
ap.add_argument("--npu", action="store_true", help="run the NPU stages (LFM3_PREFILL selects them)"); ap.add_argument("--xclbin-root", default=None)
ap.add_argument("--max-length", type=int, default=4096); ap.add_argument("--extra", nargs="*", default=[], help="extra harness args")
a = ap.parse_args()
qref = a.qref or os.path.join(a.goldens, "qref_qad")
bad = 0
for fn in sorted(os.listdir(a.goldens)):
    if not fn.endswith(".json"): continue
    name = fn[:-5]
    if a.prompts and name not in a.prompts: continue
    g = json.load(open(os.path.join(a.goldens, fn)))
    r = json.load(open(os.path.join(qref, fn)))
    ref = torch.load(os.path.join(qref, name + ".first_logits.pt"), weights_only=True)
    with tempfile.TemporaryDirectory() as td:
        ids = os.path.join(td, "ids.txt"); open(ids, "w").write(" ".join(map(str, g["prompt_ids"])))
        out = os.path.join(td, "logits.bin"); out2 = os.path.join(td, "logits2.bin")
        base = [a.harness, "--pkg", a.pkg, "--ids-file", ids, "--max-length", str(a.max_length)] + a.extra + (["--npu"] if a.npu else []) + (["--xclbin-root", a.xclbin_root] if a.xclbin_root else [])
        p = subprocess.run(base + ["--logits-out", out, "--greedy", str(a.greedy)], capture_output=True, text=True)
        if p.returncode: print(name, "FAILED", p.stderr[-400:]); bad += 1; continue
        gen = [int(v) for v in p.stdout.split("greedy:")[1].split()] if "greedy:" in p.stdout else []
        timing = " ".join(l for l in p.stderr.splitlines() if l.startswith(("prefill_ms", "decode_ms")))
        p2 = subprocess.run(base + ["--logits-out", out2, "--split-prefill"], capture_output=True, text=True)
        l1 = np.fromfile(out, dtype=np.float32); l2 = np.fromfile(out2, dtype=np.float32)
    l = torch.from_numpy(l1)
    cos = torch.nn.functional.cosine_similarity(l, ref, dim=0).item()
    top = int(l.argmax()) == int(ref.argmax())
    want = r["greedy_ids"]
    n = next((i for i, (x, y) in enumerate(zip(gen, want)) if x != y), min(len(gen), len(want)))
    split_same = bool(np.array_equal(l1, l2))
    split_cos = torch.nn.functional.cosine_similarity(l, torch.from_numpy(l2), dim=0).item()
    # host runs: a split prefill must be bit-identical; NPU runs: pass shapes change the bf16 rounding, so require a tight cosine
    split_ok = split_same if not a.npu else split_cos >= 0.998
    ok = cos >= a.min_cos and split_ok
    bad += (not ok)
    print(f"{name:15s} L={len(g['prompt_ids']):5d} cos={cos:.6f} top1={'Y' if top else 'N'} greedy_match={n}/{len(want)} split_identical={'Y' if split_same else 'N'} split_cos={split_cos:.5f} {'OK' if ok else 'FAIL'}  [{timing}]", flush=True)
sys.exit(1 if bad else 0)
