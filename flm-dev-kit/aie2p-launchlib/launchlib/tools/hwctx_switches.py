#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# hwctx_switches.py OUT_DIR [CONTEXT_MS] [WARM_MS] -- reads hwctx_survey.py's
# output: for each model and request, the commands submitted, the hw_context
# switches (two consecutive submits to different hw_contexts) and the contexts
# used. The estimate: a hw_context switch costs CONTEXT_MS (default 2.6, the
# Whisper encoder's, 8 columns), a launchlib warm switch WARM_MS (default 0.5);
# "share" is switches * CONTEXT_MS of the request's wall time, "gain" the part
# warm switches would save: switches * (CONTEXT_MS - WARM_MS).

import json
import statistics
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
context_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 2.6
warm_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5


def events(trace):
    out = []
    for line in trace.read_text().splitlines():
        f = line.split()
        if f and f[0] == "X":
            out.append((int(f[1]), int(f[2])))
    return out


print(f"hw_context switch {context_ms} ms, warm switch {warm_ms} ms")
print(f"{'model':24} {'request':8} {'wall ms':>8} {'tokens':>12} {'submits':>8} {'switches':>8} {'ctxs':>5} "
      f"{'sw/token':>8} {'share':>6} {'gain':>6} {'gap+ms':>6}")
for js in sorted(out_dir.glob("*.json")):
    record = json.loads(js.read_text())
    trace = js.with_suffix(".trace")
    if "error" in record or not trace.exists():
        print(f"{record['model']:24} not measured: {record.get('error', 'no trace')[:80]}")
        continue
    xs = events(trace)
    for r in record["requests"]:
        inside = [h for t, h in xs if r["start_ns"] <= t <= r["end_ns"]]
        switches = sum(1 for a, b in zip(inside, inside[1:]) if a != b)
        wall = (r["end_ns"] - r["start_ns"]) / 1e6
        u = r["usage"] or {}
        prompt, completion = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        per = completion if r["kind"] == "decode" else prompt
        per_token = f"{switches / per:.2f}" if per else "-"
        share = 100 * switches * context_ms / wall if wall else 0
        gain = 100 * switches * (context_ms - warm_ms) / wall if wall else 0
        # A check of CONTEXT_MS: the median gap before a submit to another context
        # minus that before one to the same context (the commands differ too, so rough).
        times = [t for t, h in xs if r["start_ns"] <= t <= r["end_ns"]]
        to_other = [(b - a) / 1e6 for a, b, x, y in zip(times, times[1:], inside, inside[1:]) if x != y]
        to_same = [(b - a) / 1e6 for a, b, x, y in zip(times, times[1:], inside, inside[1:]) if x == y]
        extra = f"{statistics.median(to_other) - statistics.median(to_same):6.2f}" if to_other and to_same else "     -"
        note = f" ERROR {r['error'][:40]}" if r.get("error") else ""
        print(f"{record['model']:24} {r['kind']:8} {wall:8.0f} {f'{prompt}+{completion}':>12} {len(inside):8} "
              f"{switches:8} {len(set(inside)):5} {per_token:>8} {share:5.0f}% {gain:5.0f}% {extra}{note}")
