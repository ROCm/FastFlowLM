#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Accuracy for GSM8K predictions written by gsm8k_run.py.

Ported from FastFlowLM_IRON's .github/scripts/llm-acc-bench.

Accuracy = 100 * (# correct) / (# total). A question counts as correct when
the parsed final numeric answer exactly matches the reference answer.

A reply that is still reasoning when it stops (no final answer) is scored as
a wrong answer, not unanswered. Unanswered questions -- failed (no usable
reply from the server) or unparsed (a reply with no number in it) -- count as incorrect and stay in the
denominator, so a run that lost questions isn't silently inflated relative to
the baseline. Counters, report.md and the --max-unanswered gate are shared
with score_anls.py -- see acc_common.py.

Usage:
  python gsm8k_score.py predictions.jsonl [--model llama3.2:1b --out-dir results/ --max-unanswered 5]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc_common as ac  # noqa: E402


def score_file(path, model):
    rows = ac.load_predictions(path, "id")
    counts = ac.tally(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    accuracy = 100.0 * correct / counts["total"] if counts["total"] else 0.0
    meta = ac.load_meta(path)
    summary = {"file": path, "accuracy": accuracy, "correct": correct, **counts,
               "server": meta.get("server"), "config": meta.get("config")}
    if model:
        summary["model"] = model
    return summary, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("predictions", nargs="+", help="predictions.jsonl file(s) from gsm8k_run.py")
    p.add_argument("--model", help="Model tag to embed in summary.json (for cross-model rollups).")
    p.add_argument("--out-dir", help="If set, write summary.json and report.md here.")
    p.add_argument("--max-unanswered", type=int, default=None,
                   help="Exit non-zero if more questions than this went unanswered (failed + unparsed), "
                   "or if nothing was recorded. An accuracy of 0 only warns. "
                   "summary.json and report.md are written either way.")
    args = p.parse_args(argv)

    summaries, rc = [], 0
    for path in args.predictions:
        s, rows = score_file(path, args.model)
        summaries.append(s)
        print(f"{os.path.basename(path)}:  accuracy={s['accuracy']:.1f}%   ({s['correct']}/{s['total']} correct, "
              f"{s['unanswered']} unanswered = {s['failed']} failed"
              f"{' [' + ac.failures_text(s) + ']' if s['failed'] else ''} + {s['unparsed']} unparsed; "
              f"{s['reasoning_only']} still reasoning (scored wrong); "
              f"{s['truncated']} hit max_tokens)")
        label = f"GSM8K {args.model or os.path.basename(path)}"
        rc |= ac.gate(label, "accuracy", s["accuracy"], s, args.max_unanswered)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            ac.write_report(args.out_dir, label,
                            f"Accuracy {s['accuracy']:.1f}% ({s['correct']}/{s['total']} correct)",
                            s, rows, "id", "gt")

    if args.out_dir:
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summaries[0] if len(summaries) == 1 else summaries, f, indent=2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
