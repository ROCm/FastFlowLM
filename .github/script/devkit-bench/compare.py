#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Comparisons and soft gates of the dev-kit benches (pure functions).

report.py renders what these return.

Perf (flm bench CSVs: context_length_k, prefill/decoding avg tok/s, TTFT):
  A/B        the dev-kit build against the package it was overlaid on, run
             back to back on the same runner (perf.py). A context whose
             decoding or prefill rate drops more than PERF_WARN_PCT is a
             warning: the one perf signal that does not depend on which
             machine kind a job landed on.
  baseline   against the nightly baseline's numbers for the same machine
             kind only (machine.py); informational, never a warning, as the
             runners differ run to run even within a kind.
Accuracy (summary.json of gsm8k_score.py / score_anls.py):
  baseline   delta in points against the nightly baseline; more than
             ACC_WARN_POINTS down is a warning (n=50 has a standard deviation
             of about 6 points). acc_common.gate's verdict (summary.usable)
             is the hard gate and fails the bench job itself.

Soft gates only: a warning is shown, never fails anything. Promote them only
with variance data from ~30 nightly runs.

Everything read here comes from artifacts, so it is parsed with csv/json
only, numbers through float() (NaN and infinities rejected) and model ids
against MODEL_ID; anything else is dropped. Standard library only.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
from typing import Optional

PERF_WARN_PCT = 7.0
ACC_WARN_POINTS = 10.0
# Unanswered questions above this many are flagged (the benches' default
# --max-unanswered).
FLAG_UNANSWERED = 5
MODEL_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
PERF_METRICS = (("decoding_avg_toks_per_s", "decode"), ("prefill_avg_toks_per_s", "prefill"))
PERF_COLUMNS = ("context_length_k", "ttft_avg_s") + tuple(k for k, _ in PERF_METRICS)
ACC_KEYS = {"gsm8k": "accuracy", "docvqa": "anls"}
COUNTERS = ("total", "unanswered", "failed", "unparsed", "reasoning_only", "truncated")
FAILURE_KINDS = {"timeout", "connection_error", "http_error", "bad_response_shape", "empty_reply", "aborted"}


def number(value) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def count(value) -> Optional[int]:
    x = number(value)
    return int(x) if x is not None and x >= 0 and x == int(x) and x < 1e9 else None


def valid_model(model) -> bool:
    return isinstance(model, str) and bool(MODEL_ID.match(model))


# ── Perf ─────────────────────────────────────────────────────────────────────

def parse_perf_csv(text: str) -> dict[float, dict]:
    """context_length_k -> {column: float}, from a flm bench CSV. Rows with a
    missing or non-numeric context are dropped, as are bad cells."""
    out: dict[float, dict] = {}
    for row in csv.DictReader(io.StringIO(text)):
        ctx = number(row.get("context_length_k"))
        if ctx is None or ctx <= 0 or ctx > 4096:
            continue
        values = {k: number(row.get(k)) for k in PERF_COLUMNS[1:]}
        out[ctx] = {k: v for k, v in values.items() if v is not None and v >= 0}
    return out


def perf_rows_from_json(rows) -> dict[float, dict]:
    """The baseline's perf rows ([{context_length_k, ...}]), validated."""
    if not isinstance(rows, list):
        return {}
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=PERF_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        if isinstance(r, dict):
            writer.writerow({k: r.get(k, "") for k in PERF_COLUMNS})
    return parse_perf_csv(text.getvalue())


def perf_rows_to_json(points: dict[float, dict]) -> list[dict]:
    return [{"context_length_k": ctx, **vals} for ctx, vals in sorted(points.items())]


def pct(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or not old:
        return None
    return 100.0 * (new - old) / old


def compare_perf(new: dict[float, dict], old: dict[float, dict], warn: bool) -> dict:
    """{"rows": [{ctx, <metric>: (old, new, delta%)}], "warnings": [...]}.
    warn: apply PERF_WARN_PCT (the A/B comparison only)."""
    rows, warnings = [], []
    for ctx in sorted(new):
        row = {"context_length_k": ctx}
        for key, label in PERF_METRICS:
            n, o = new[ctx].get(key), old.get(ctx, {}).get(key)
            d = pct(n, o)
            row[label] = (o, n, d)
            if warn and d is not None and d < -PERF_WARN_PCT:
                warnings.append(f"{label} at {ctx:g}k is {d:+.1f}% vs the base package "
                                f"(over {PERF_WARN_PCT:g}%)")
        rows.append(row)
    return {"rows": rows, "warnings": warnings}


# ── Accuracy ─────────────────────────────────────────────────────────────────

def clean_acc_summary(summary, bench: str) -> Optional[dict]:
    """The fields of a summary.json the comparison and the baseline use,
    validated; None when it is not one."""
    if not isinstance(summary, dict) or bench not in ACC_KEYS:
        return None
    score = number(summary.get(ACC_KEYS[bench]))
    if score is None or not 0 <= score <= 100:
        return None
    out = {"score": round(score, 2), "usable": summary.get("usable") is True}
    for k in COUNTERS:
        out[k] = count(summary.get(k)) or 0
    failures = summary.get("failures")
    out["failures"] = ({k: count(v) or 0 for k, v in failures.items() if k in FAILURE_KINDS}
                       if isinstance(failures, dict) else {})
    lat = number(summary.get("mean_latency_s"))
    out["mean_latency_s"] = round(lat, 2) if lat is not None and lat >= 0 else None
    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    n, asked = count(config.get("n")), count(config.get("asked"))
    out["n"] = n
    out["partial"] = n is not None and asked is not None and asked < n
    server = summary.get("server")
    out["server"] = server[:120] if isinstance(server, str) else None
    return out


def acc_notes(pr: dict, base: Optional[dict]) -> list[str]:
    """Why a reviewer should look closer, beyond the delta (fixed wording)."""
    notes = []
    if not pr["usable"]:
        notes.append("failed the bench gate (no predictions, or too many unanswered)")
    if pr["score"] == 0:
        notes.append("scored 0")
    if pr["unanswered"]:
        flag = "⚠️ " if pr["unanswered"] > FLAG_UNANSWERED else ""
        kinds = ", ".join(f"{k} ×{n}" for k, n in sorted(pr["failures"].items()) if n)
        notes.append(f"{flag}{pr['unanswered']} unanswered" + (f" ({kinds})" if kinds else ""))
    if pr["reasoning_only"]:
        notes.append(f"{pr['reasoning_only']} still reasoning when stopped")
    if pr["truncated"]:
        notes.append(f"{pr['truncated']} hit max_tokens")
    if pr["partial"]:
        notes.append(f"time budget spent: {pr['total']} of {pr['n']} questions asked")
    if base and base.get("server") and pr.get("server") and base["server"] != pr["server"]:
        notes.append("FLM version differs from the baseline's")
    return notes


def compare_acc(pr: dict, base: Optional[dict]) -> dict:
    """{"score", "baseline", "delta", "notes", "warnings"}."""
    old = base.get("score") if base else None
    delta = None if old is None else round(pr["score"] - old, 2)
    warnings = []
    if delta is not None and delta < -ACC_WARN_POINTS:
        warnings.append(f"{delta:+.1f} points vs the nightly baseline (over {ACC_WARN_POINTS:g})")
    return {"score": pr["score"], "baseline": old, "delta": delta, "notes": acc_notes(pr, base),
            "warnings": warnings}


def load_json_text(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return None
