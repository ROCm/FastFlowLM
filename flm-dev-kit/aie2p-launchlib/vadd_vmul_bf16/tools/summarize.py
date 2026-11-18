#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# summarize.py -- the medians of the RESULT lines of sweep.sh's logs.
#
#   summarize.py OUT_DIR
#
# One row per run (COLSxCORES-BLOCKS), us per launch, the median over the
# repetitions. A repetition with a failed check or an error is left out and
# counted. "program B" is the program of all cores (the build's own report).

import collections
import glob
import os
import re
import statistics
import sys

COLUMNS = [
    ("rerun", "rerun"),
    ("alt full", "alternate__full"), ("alt pm", "alternate__pm"),
    ("**switch full**", "switch_full"), ("**switch pm**", "switch_pm"),
    ("expr full (3)", "expression__full"), ("expr pm (3)", "expression__pm"),
    ("1 ctx, 1 at a time", "ctx_one"), ("2 ctx, 1 at a time", "ctx_two"), ("**ctx switch**", "ctx_switch"),
]


def parse(run):
    """COLSxCORES-BLOCKS[-pair] -> cols, cores, blocks, layout."""
    cols, rest = run.split("x")
    parts = rest.split("-")
    cores, blocks = int(parts[0]), int(parts[1])
    layout = "pair" if (len(parts) > 2 or cores > 2) else "split"
    return int(cols), cores, blocks, layout


def key(run):
    cols, cores, blocks, layout = parse(run)
    return layout == "pair", blocks, cols * cores, cols


def main():
    out = sys.argv[1]
    runs, bad, program = collections.defaultdict(list), collections.Counter(), {}
    for log in glob.glob(os.path.join(out, "run_*.log")):
        run = re.match(r"run_(.+)_\d+\.log", os.path.basename(log)).group(1)
        text = open(log).read()
        m = re.search(r"cold_vadd\s+\d+ ops\s+\d+ B \(program (\d+) B\)", text)
        if m:
            program[run] = int(m.group(1))
        if "check FAIL" in text or "error:" in text or "RESULT switch_full" not in text:
            bad[run] += 1
            continue
        runs[run].append({k: float(v) for k, v in re.findall(r"^RESULT (\S+) (\S+)$", text, re.M)})
    head = ("| array | cores | layout | blocks/core | N (bf16) | program B | runs | "
            + " | ".join(c for c, _ in COLUMNS) + " |")
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for run in sorted(set(runs) | set(bad), key=key):
        cols, cores, blocks, layout = parse(run)
        n = 256 * cols * cores * blocks
        cells = []
        for _, k in COLUMNS:
            vals = [r[k] for r in runs[run] if k in r]
            cells.append(f"{statistics.median(vals):.1f}" if vals else "-")
        runs_txt = f"{len(runs[run])}" + (f" (+{bad[run]} failed)" if bad[run] else "")
        print(f"| {cols}x{cores} | {cols * cores} | {layout} | {blocks} | {n} | {program.get(run, '-')} | {runs_txt} | "
              + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
