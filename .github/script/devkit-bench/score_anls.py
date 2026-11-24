#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Official DocVQA metric: ANLS (Average Normalized Levenshtein Similarity).

For each question, score = max over ground-truth answers of
    (1 - NL(pred, gt))   if that value >= 0.5   else 0
where NL is the Levenshtein distance normalized by the longer string length.
ANLS = 100 * mean(score). Validated to reproduce the published DocVQA 94.83.

A reply that is still reasoning when it stops (nothing left once the <think>
block is stripped) is a wrong answer and scores 0, but is not unanswered.
Failed questions (no usable reply from the server) score 0 too and stay in
the denominator. Counters, report.md and the --max-unanswered gate are shared with
gsm8k_score.py -- see acc_common.py.

Usage:
  python score_anls.py predictions.jsonl [more.jsonl ...]
  python score_anls.py predictions.jsonl --model qwen2.5vl-it:3b --out-dir results/ --max-unanswered 5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc_common as ac  # noqa: E402

THRESHOLD = 0.5


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s):
    return " ".join(str(s).strip().lower().split())


def anls_one(pred, gts):
    p = norm(pred)
    best = 0.0
    for gt in gts:
        g = norm(gt)
        if not p and not g:
            best = max(best, 1.0)
            continue
        m = max(len(p), len(g))
        if m == 0:
            continue
        sim = 1.0 - levenshtein(p, g) / m
        best = max(best, sim if sim >= THRESHOLD else 0.0)
    return best


def score_file(path, model):
    rows = ac.load_predictions(path, "question_id")
    counts = ac.tally(rows)
    # pred is the generation with any think block already stripped by run_vqa.py
    total = sum(anls_one(r.get("pred") or "", r["answers"]) for r in rows)
    anls = 100.0 * total / counts["total"] if counts["total"] else 0.0
    meta = ac.load_meta(path)
    # "n" predates "total" and is kept for older readers of the baseline.
    summary = {"file": path, "anls": anls, "n": counts["total"], **counts,
               "server": meta.get("server"), "config": meta.get("config")}
    if model:
        summary["model"] = model
    return summary, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions", nargs="+", help="predictions.jsonl file(s) from run_vqa.py")
    ap.add_argument("--model", help="Model tag to embed in summary.json (for cross-model rollups).")
    ap.add_argument("--out-dir", help="If set, write summary.json and report.md here.")
    ap.add_argument("--max-unanswered", type=int, default=None,
                    help="Exit non-zero if more questions than this went unanswered (failed + unparsed), "
                    "or if nothing was recorded. An ANLS of 0 only warns. "
                    "summary.json and report.md are written either way.")
    args = ap.parse_args(argv)

    summaries, rc = [], 0
    for path in args.predictions:
        s, rows = score_file(path, args.model)
        summaries.append(s)
        print(f"{os.path.basename(path)}:  ANLS={s['anls']:.2f}   (n={s['total']}, "
              f"{s['unanswered']} unanswered = {s['failed']} failed"
              f"{' [' + ac.failures_text(s) + ']' if s['failed'] else ''} + {s['unparsed']} unparsed; "
              f"{s['reasoning_only']} still reasoning (scored wrong); "
              f"{s['truncated']} hit max_tokens)")
        label = f"DocVQA {args.model or os.path.basename(path)}"
        rc |= ac.gate(label, "ANLS", s["anls"], s, args.max_unanswered)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            ac.write_report(args.out_dir, label, f"ANLS {s['anls']:.2f} (n={s['total']})",
                            s, rows, "question_id", "answers")

    if args.out_dir:
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summaries[0] if len(summaries) == 1 else summaries, f, indent=2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
