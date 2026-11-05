#!/usr/bin/env python3
"""Sum the [prefill-prof] lines of a test_long run (engine built with PREFILL_PROFILE=1).

    python3 prof_summary.py test_long.log [--skip 1] [--md]

Every result row of test_long closes one prompt; the profile lines before it belong
to that prompt. Steps are summed over all layers (and all 512-token pieces), then
the median over the repeats of one length is taken, after dropping the first
--skip repeats (they build sequences and buffers). See README.md.
"""
import argparse
import re
import statistics
from collections import OrderedDict, defaultdict

PROF_LINE = re.compile(r"\[prefill-prof\]\s+([^:]+:[^:]+?):\s+([0-9.]+)\s+(us|ms|s)\b")
RESULT_ROW = re.compile(r"^\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+")
TO_MS = {"us": 1e-3, "ms": 1.0, "s": 1e3}

# spans that enclose other ticks: shown for reference, never added to the total
CONTAINERS = {"layer:attn_block_total", "layer:mlp_block_total", "prefill:layers", "prefill:total(mm)"}

GROUPS = OrderedDict([
    ("NPU gemm (q, kv, o, up+gate, down)", ["attn:q_run", "attn:kv_start", "attn:kv_wait", "attn:o_run",
                                            "mlp:upgate_run", "mlp:down_run"]),
    ("NPU attention (mha)", ["attn:mha"]),
    ("CPU rms norms", ["layer:input_rms_norm", "layer:post_attn_rms_norm"]),
    ("CPU rope + qk norm", ["attn:q_rope_norm", "attn:k_rope_norm"]),
    ("CPU glu multiply", ["mlp:glu_mul"]),
    ("CPU residual add + x memset", ["layer:residual_add", "layer:x_memset"]),
    ("CPU fill_kv_cache", ["attn:fill_kv_cache"]),
    ("CPU _update_bfp16", ["prefill:update_bfp16"]),
    ("syncs (whole-BO)", ["layer:x_sync_to", "attn:q_sync_from", "attn:q_sync_to", "attn:kv_sync_from",
                          "attn:kv_sync_to", "attn:o_sync_from", "mlp:upgate_sync_from", "mlp:hid_sync_to",
                          "mlp:down_sync_from"]),
    ("sequence get + alloc", ["attn:alloc+seq_get", "mlp:seq_gen+alloc"]),
    ("embedding", ["prefill:embedding"]),
    ("lm_head (incl. xclbin switch)", ["prefill:lm_head"]),
])


def parse(path):
    """list of (tokens, prefill_ms, {step: ms summed over the prompt})"""
    prompts = []
    steps = defaultdict(float)
    header_seen = False
    for line in open(path, errors="replace"):
        if "prefill(ms)" in line:
            header_seen = True
            steps = defaultdict(float)   # drop the warm-up call
            continue
        m = PROF_LINE.search(line)
        if m:
            steps[m.group(1).strip()] += float(m.group(2)) * TO_MS[m.group(3)]
            continue
        m = RESULT_ROW.match(line)
        if header_seen and m:
            prompts.append((int(m.group(1)), float(m.group(2)), dict(steps)))
            steps = defaultdict(float)
    return prompts


def median_per_length(prompts, skip):
    by_len = OrderedDict()
    for n, total, steps in prompts:
        by_len.setdefault(n, []).append((total, steps))
    out = OrderedDict()
    for n, runs in by_len.items():
        kept = runs[skip:] if len(runs) > skip else runs
        names = set().union(*(s.keys() for _, s in kept))
        med = {k: statistics.median(s.get(k, 0.0) for _, s in kept) for k in names}
        out[n] = (statistics.median(t for t, _ in kept), med, len(kept))
    return out


def grouped(total, med):
    rows = OrderedDict()
    used = set()
    for g, names in GROUPS.items():
        rows[g] = sum(med.get(k, 0.0) for k in names)
        used.update(names)
    leftover = [k for k in med if k not in used and k not in CONTAINERS]
    if leftover:
        rows["other ticks (" + ", ".join(sorted(leftover)) + ")"] = sum(med[k] for k in leftover)
    rows["not covered by ticks"] = total - sum(rows.values())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--skip", type=int, default=1, help="repeats of each length to drop first")
    ap.add_argument("--md", action="store_true", help="markdown tables")
    ap.add_argument("--steps", action="store_true", help="also print every single step")
    args = ap.parse_args()

    prompts = []
    for p in args.logs:
        prompts += parse(p)
    table = median_per_length(prompts, args.skip)
    lengths = list(table)

    group_rows = OrderedDict()
    for n in lengths:
        total, med, _ = table[n]
        for g, v in grouped(total, med).items():
            group_rows.setdefault(g, {})[n] = v

    sep = " | " if args.md else "  "
    head = ["step"] + [f"{n} tok" for n in lengths]
    print(("| " if args.md else "") + sep.join(head) + (" |" if args.md else ""))
    if args.md:
        print("|" + "---|" * len(head))
    for g, vals in group_rows.items():
        cells = [g] + [f"{vals.get(n, 0.0):.1f} ({100 * vals.get(n, 0.0) / table[n][0]:.0f}%)" for n in lengths]
        print(("| " if args.md else "") + sep.join(cells) + (" |" if args.md else ""))
    cells = ["**total prefill ms**" if args.md else "total prefill ms"] + [f"{table[n][0]:.1f}" for n in lengths]
    print(("| " if args.md else "") + sep.join(cells) + (" |" if args.md else ""))
    cells = ["repeats used"] + [str(table[n][2]) for n in lengths]
    print(("| " if args.md else "") + sep.join(cells) + (" |" if args.md else ""))

    if args.steps:
        print()
        names = sorted(set().union(*(table[n][1].keys() for n in lengths)))
        print(("| " if args.md else "") + sep.join(head) + (" |" if args.md else ""))
        if args.md:
            print("|" + "---|" * len(head))
        for k in names:
            cells = [k] + [f"{table[n][1].get(k, 0.0):.2f}" for n in lengths]
            print(("| " if args.md else "") + sep.join(cells) + (" |" if args.md else ""))


if __name__ == "__main__":
    main()
