#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Synthetic state-once requests for d3_engine_run: per N in --ns one group "g<N>" of N prompts that share their first S
tokens (the state) and differ after it (question blocks of Q +- JITTER tokens). Token ids come from a pool of real prompts
(a prompts.txt as d3_engine_run reads it), so activations look like text. Writes <out>.txt (all questions) and <out>.state
(S per prompt; --image-run n puts a run of n <|image_pad|> tokens inside the state, for d3_engine_run --image); with --rowcap K also <out>_row.txt / _row.state / _row.map (row -> line of <out>.txt) holding only the first K questions of every group (the
per-row path is K times as expensive per question; the questions of a group are the same ones, so rows compare 1:1).

  make_state_once_prompts.py POOL.txt OUT_PREFIX --state S --ns 1,4,16,64,255 [--q 110] [--jitter 20] [--seed 1] [--rowcap 4]

Run the same files with and without --state-once: the per-row path (encode_batch + state_len) and encode_state_once.
"""
import argparse, random

ap = argparse.ArgumentParser()
ap.add_argument("pool"); ap.add_argument("out")
ap.add_argument("--state", type=int, required=True); ap.add_argument("--ns", required=True)
ap.add_argument("--q", type=int, default=110); ap.add_argument("--jitter", type=int, default=20)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--image-run", type=int, default=0); ap.add_argument("--rowcap", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
pool = []
for line in open(a.pool):
    pool += [int(x) for x in line.split()[1:]]
pool = [t for t in pool if t < 240000]   # real text tokens, no specials
def seq(n):
    start = rng.randrange(0, len(pool) - n) if len(pool) > n else 0
    s = pool[start:start + n]
    while len(s) < n: s += pool[:n - len(s)]
    return s
ns = [int(x) for x in a.ns.split(",")]
files = [open(a.out + ".txt", "w"), open(a.out + ".state", "w")]
rows = [open(a.out + "_row.txt", "w"), open(a.out + "_row.state", "w"), open(a.out + "_row.map", "w")] if a.rowcap else None
idx = 0
for n in ns:
    state = seq(a.state - a.image_run)
    if a.image_run:   # an image's pad run (<|image_pad|> = 248056) inside the state, counted in S
        k = len(state) // 2
        state = state[:k] + [248056] * a.image_run + state[k:]
    for i in range(n):
        q = max(1, a.q + rng.randint(-a.jitter, a.jitter))
        line = f"g{n} " + " ".join(map(str, state + seq(q))) + "\n"
        files[0].write(line); files[1].write(f"{a.state}\n")
        if rows and i < a.rowcap:
            rows[0].write(line); rows[1].write(f"{a.state}\n"); rows[2].write(f"{idx}\n")
        idx += 1
