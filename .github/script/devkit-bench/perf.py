#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Runs `flm bench` for one model, on the dev-kit build and optionally on the
package it was overlaid on, back to back on the same runner.

    perf.py --flm-bin <overlaid flm> [--base-flm-bin <unmodified flm>]
            --model <tag> --out-dir <dir> [--ctx-len 8192]
            [--timeout 3600] [--pull]

The runners have no perf-pinned machine, so the number that judges a pull
request is the A/B ratio of these two runs on one box (compare.py); its
comparison with the nightly baseline is informational, and only made on the
same machine kind (machine.py).

--pull runs `flm pull <tag>` first, which downloads a missing model and
hashes every file of one already on the runner against the catalog's pins
(model weights live in a shared cache on the runner).

flm bench writes bench_<tag>_<date>[_<cpu>].csv into its working
directory, one row per context length (1k, 2k, 4k, ... up to -c or the
model's max_bench_length), so each run gets its own empty directory and its
CSV is kept as <out-dir>/<variant>.csv, variant "devkit" or "base".
FLM_XCLBIN_PATH is set to each binary's own directory, so each reads its
own xclbins/ tree. status.json records each variant's status: ok,
timeout, failed (exit N) or no CSV.

Exit 1 when the dev-kit run did not produce a CSV; a failed base run only
warns (there is then no A/B ratio). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def annotate(level: str, msg: str) -> None:
    print(f"::{level}::{msg}", flush=True)


def safe_tag(model: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in model)


def run_bench(flm_bin: str, model: str, work: Path, ctx_len: int, timeout: float) -> tuple[str, float]:
    """(status, seconds) of one `flm bench` in an empty work directory."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cmd = [flm_bin, "bench", model]
    if ctx_len > 0:
        cmd += ["-c", str(ctx_len)]
    env = dict(os.environ, FLM_XCLBIN_PATH=str(Path(flm_bin).resolve().parent))
    print(f"[perf] {' '.join(cmd)} (in {work})", flush=True)
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=work, env=env, timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        return "timeout", time.monotonic() - started
    seconds = time.monotonic() - started
    csvs = sorted(work.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if code != 0:
        return f"failed (exit {code})", seconds
    if not csvs:
        return "no CSV", seconds
    shutil.move(str(csvs[-1]), str(work.parent / f"{work.name}.csv"))
    return "ok", seconds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flm-bin", required=True, help="the overlaid (dev-kit) flm")
    ap.add_argument("--base-flm-bin", default="", help="the same package without the overlay")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ctx-len", type=int, default=0, help="flm bench -c (0: the model's maximum)")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--pull", action="store_true", help="flm pull the model first (verifies its files)")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    status: dict = {"model": args.model, "ctx_len": args.ctx_len, "variants": {}}
    if args.pull:
        try:
            code = subprocess.run([args.flm_bin, "pull", args.model], timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        status["pull"] = "ok" if code == 0 else f"failed ({code})"
        if code != 0:
            annotate("error", f"flm pull {args.model} failed ({code}); nothing to bench")
            (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            return 1
    variants = [("devkit", args.flm_bin)]
    if args.base_flm_bin:
        # Base first, then the dev-kit build: any warm-up effect favours
        # the change, so a slowdown reported is not an artefact of order.
        variants.insert(0, ("base", args.base_flm_bin))
    for name, flm_bin in variants:
        result, seconds = run_bench(flm_bin, args.model, out / name, args.ctx_len, args.timeout)
        shutil.rmtree(out / name, ignore_errors=True)
        status["variants"][name] = {"status": result, "seconds": round(seconds, 1)}
        if result != "ok":
            annotate("error" if name == "devkit" else "warning",
                     f"flm bench {args.model} ({name}): {result}")
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return 0 if status["variants"]["devkit"]["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
