#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Runs one accuracy bench (GSM8K or DocVQA) for one model against its own
`flm serve`.

    run_acc.py --bench gsm8k|docvqa --flm-bin <flm> --model <tag>
               --out-dir <dir> [--n 50] [--data eval.jsonl]
               [--max-unanswered 5] [--time-budget-s 0]

Starts `flm serve <tag> -r 0` on a free port with npu_test's FlmServer (the
harness the NPU tests use: waits for /v1/version, stops it with SIGINT then
SIGKILL), runs gsm8k_run.py or run_vqa.py against it and scores the
predictions with gsm8k_score.py or score_anls.py, which write summary.json
and report.md into <out-dir> and apply acc_common.gate: the benches' one
hard gate (no predictions, or more than --max-unanswered unanswered). The
server's log stays in <out-dir>. FLM_XCLBIN_PATH is the flm binary's own
directory, so it reads the overlaid package's xclbins/ tree.

Exit 0 when the run is usable, 1 otherwise (scoring and the report happen
either way). Standard library only.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from npu_test.server import FlmServer, ServerError  # noqa: E402

BENCHES = {
    "gsm8k": ("gsm8k_run.py", "gsm8k_score.py"),
    "docvqa": ("run_vqa.py", "score_anls.py"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", required=True, choices=sorted(BENCHES))
    ap.add_argument("--flm-bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--data", default="", help="DocVQA: eval.jsonl from docvqa_fetch.py")
    ap.add_argument("--max-unanswered", type=int, default=5)
    ap.add_argument("--time-budget-s", type=float, default=0.0)
    ap.add_argument("--ready-timeout", type=float, default=900.0,
                    help="seconds for flm serve to load the model and answer")
    args = ap.parse_args(argv)
    if args.bench == "docvqa" and not args.data:
        ap.error("--data is required for docvqa")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds = out / "predictions.jsonl"
    run_script, score_script = (str(HERE / s) for s in BENCHES[args.bench])
    os.environ["FLM_XCLBIN_PATH"] = str(Path(args.flm_bin).resolve().parent)

    server = FlmServer(args.flm_bin, name=args.bench, log_dir=out, args=[args.model, "-r", "0"],
                       ready_timeout=args.ready_timeout)
    run_rc = 0
    try:
        server.start()
        cmd = [sys.executable, run_script, "--model", args.model, "--base-url", server.base_url,
               "--out", str(preds), "--time-budget-s", str(args.time_budget_s)]
        if args.bench == "gsm8k":
            cmd += ["--n", str(args.n)]
        else:
            cmd += ["--data", args.data, "--limit", str(args.n)]
        run_rc = subprocess.run(cmd).returncode
    except ServerError as exc:
        print(f"::error::{args.bench} {args.model}: flm serve did not start: {exc}", flush=True)
        print(server.log_tail(100), flush=True)
        run_rc = 1
    finally:
        server.stop()

    if not preds.is_file() or preds.stat().st_size == 0:
        print(f"::error::{args.bench} {args.model}: no predictions were recorded.", flush=True)
        return 1
    score_rc = subprocess.run([sys.executable, score_script, str(preds), "--model", args.model,
                               "--out-dir", str(out), "--max-unanswered", str(args.max_unanswered)]).returncode
    return 1 if run_rc or score_rc else 0


if __name__ == "__main__":
    sys.exit(main())
