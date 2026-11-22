#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Turns the bench jobs' results into bench-summary.md and, nightly, the next
baseline.

    report.py --results <dir> --matrix <json file> --mode pr|nightly|refresh|dispatch
              --head-sha <sha> [--build-run-id N] [--baseline <dir>]
              [--baseline-run-id N] --summary <file> [--new-baseline <dir>]
              [--outputs <file>]

<results> holds one directory per bench job (artifact
devkit-bench-result-<key>-<model>), each with meta.json, machine.json,
perf/{status.json,devkit.csv,base.csv}, gsm8k/summary.json and
docvqa/summary.json as the job got to them. <baseline> holds the previous
nightly's baseline.json.

The summary (at most MAX_BYTES, cut with a note beyond that) has, per
runtime: perf A/B (dev-kit against the base package on the same runner)
with the change against the nightly baseline for the same machine kind,
GSM8K and DocVQA against the nightly baseline, the soft-gate warnings and
the jobs that failed or produced nothing. A pull request (--mode pr) and
the dev-kit full refresh (--mode refresh) are given no baseline and run no
A/B: their summary is the run's numbers, compared with nothing (a
comparison is to be designed later). It holds only numbers, validated
model ids and machine kinds, and fixed wording: no model output, log or
other free text from an artifact. Per-question reports stay in the result
artifacts.

baseline.json (schema 1):
  {"schema": 1, "sha": <commit benched>, "run_id": <the run that benched>,
   "updated": "YYYY-MM-DD",
   "entries": {<key, e.g. lin-xrt>: {<model>: {
       "gsm8k":  {score, total, unanswered, ..., server, sha, run_id},
       "docvqa": {...},
       "perf":   {<machine kind>: {"rows": [...], "sha", "run_id"}}}}}}
--new-baseline writes the previous one updated with this run: each model's
usable accuracy results and successful perf runs replace its old ones; a
model whose bench failed keeps its old numbers; others are kept as they
were (an upsert, as IRON's merge-pr.yml did into baseline/).

--outputs (the step's $GITHUB_OUTPUT) gets hard-gate=fail when some model's
accuracy run failed acc_common's hard gate (no predictions, or too many
unanswered), else hard-gate=pass: a bench job that failed for any other
reason (perf giving no result, a pull, a timeout, a lost runner) is not a
hard-gate failure. Build and Test's summary fails a pull request on
hard-gate=fail only.

Exit 0; the bench jobs carry the hard gate. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare  # noqa: E402
import machine  # noqa: E402

MAX_BYTES = 60000
KEYS = ("lin-xrt", "lin-hrx", "win-xrt", "win-hrx")
KEY_NAMES = {"lin-xrt": "Linux XRT", "lin-hrx": "Linux HRX", "win-xrt": "Windows XRT",
             "win-hrx": "Windows HRX"}
SHA = re.compile(r"^[0-9a-f]{40}$")
PERF_STATUS = re.compile(r"^(ok|timeout|no CSV|failed \((exit )?-?\d+\)|failed \(timeout\))$")


def read_json(path: Path):
    try:
        if path.stat().st_size > 5_000_000:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_text(path: Path, limit: int = 1_000_000) -> str:
    try:
        if path.stat().st_size > limit:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ── Loading ──────────────────────────────────────────────────────────────────

def load_result(d: Path) -> Optional[dict]:
    """One bench job's results, validated; None when it has no usable meta."""
    meta = read_json(d / "meta.json")
    if not isinstance(meta, dict) or meta.get("key") not in KEYS or not compare.valid_model(meta.get("model")):
        return None
    info = read_json(d / "machine.json")
    kind = info.get("kind") if isinstance(info, dict) else None
    out = {"key": meta["key"], "model": meta["model"], "vlm": meta.get("vlm") is True,
           "kind": kind if machine.valid_kind(kind) else "unknown", "perf": None, "acc": {}}
    status = read_json(d / "perf" / "status.json")
    if isinstance(status, dict) and isinstance(status.get("variants"), dict):
        variants = {}
        for name in ("devkit", "base"):
            v = status["variants"].get(name)
            s = v.get("status") if isinstance(v, dict) else None
            if isinstance(s, str) and PERF_STATUS.match(s):
                variants[name] = s
        out["perf"] = {
            "status": variants,
            "devkit": compare.parse_perf_csv(read_text(d / "perf" / "devkit.csv")),
            "base": compare.parse_perf_csv(read_text(d / "perf" / "base.csv")),
        }
    for bench in compare.ACC_KEYS:
        s = compare.clean_acc_summary(read_json(d / bench / "summary.json"), bench)
        if s is not None:
            out["acc"][bench] = s
    return out


def load_results(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()):
        r = load_result(d)
        if r is not None:
            out.append(r)
    return out


def parse_baseline(data) -> dict:
    """baseline.json, validated: anything malformed is dropped."""
    empty = {"schema": 1, "sha": "", "run_id": None, "updated": "", "entries": {}}
    if not isinstance(data, dict) or data.get("schema") != 1:
        return empty
    out = dict(empty)
    if isinstance(data.get("sha"), str) and SHA.match(data["sha"]):
        out["sha"] = data["sha"]
    out["run_id"] = compare.count(data.get("run_id"))
    if isinstance(data.get("updated"), str) and re.match(r"^\d{4}-\d{2}-\d{2}$", data["updated"]):
        out["updated"] = data["updated"]
    entries = data.get("entries") if isinstance(data.get("entries"), dict) else {}
    for key, models in entries.items():
        if key not in KEYS or not isinstance(models, dict):
            continue
        for model, entry in models.items():
            if not compare.valid_model(model) or not isinstance(entry, dict):
                continue
            clean: dict = {}
            for bench in compare.ACC_KEYS:
                b = entry.get(bench)
                if isinstance(b, dict):
                    s = compare.clean_acc_summary({compare.ACC_KEYS[bench]: b.get("score"), **b,
                                                   "config": {"n": b.get("n")}}, bench)
                    if s is not None:
                        # Stored as upsert_baseline writes it: usable results only.
                        s = {k: v for k, v in s.items() if k not in ("usable", "partial")}
                        s.update(_provenance(b))
                        clean[bench] = s
            perf = entry.get("perf") if isinstance(entry.get("perf"), dict) else {}
            kinds = {}
            for kind, p in perf.items():
                if machine.valid_kind(kind) and isinstance(p, dict):
                    rows = compare.perf_rows_from_json(p.get("rows"))
                    if rows:
                        kinds[kind] = {"rows": compare.perf_rows_to_json(rows), **_provenance(p)}
            if kinds:
                clean["perf"] = kinds
            if clean:
                out["entries"].setdefault(key, {})[model] = clean
    return out


def _provenance(d: dict) -> dict:
    sha = d.get("sha") if isinstance(d.get("sha"), str) and SHA.match(d.get("sha") or "") else ""
    return {"sha": sha, "run_id": compare.count(d.get("run_id"))}


# ── Baseline ─────────────────────────────────────────────────────────────────

def upsert_baseline(old: dict, results: list[dict], head_sha: str, run_id: Optional[int]) -> dict:
    new = json.loads(json.dumps(old))
    new.update(schema=1, sha=head_sha, run_id=run_id, updated=time.strftime("%Y-%m-%d", time.gmtime()))
    where = {"sha": head_sha, "run_id": run_id}
    for r in results:
        entry = new["entries"].setdefault(r["key"], {}).setdefault(r["model"], {})
        for bench, s in r["acc"].items():
            if s["usable"] and not s["partial"]:
                entry[bench] = {**{k: v for k, v in s.items() if k not in ("usable", "partial")}, **where}
        perf = r["perf"]
        if perf and perf["status"].get("devkit") == "ok" and perf["devkit"] and machine.valid_kind(r["kind"]):
            entry.setdefault("perf", {})[r["kind"]] = {"rows": compare.perf_rows_to_json(perf["devkit"]),
                                                       **where}
        if not entry:
            del new["entries"][r["key"]][r["model"]]
    return new


# ── Summary ──────────────────────────────────────────────────────────────────

def fmt(x: Optional[float], spec: str = ".1f") -> str:
    return "—" if x is None else format(x, spec)


def fmt_delta(d: Optional[float], unit: str = "%") -> str:
    return "—" if d is None else f"{d:+.1f}{unit}"


def perf_section(results: list[dict], baseline: dict, key: str, warnings: list) -> list[str]:
    lines = ["", "#### Perf (tok/s; dev-kit against the base package on the same runner)", "",
             "| Model | Machine | Context | Decode base → dev-kit | Δ | Prefill base → dev-kit | Δ "
             "| Decode Δ vs nightly (same kind) |",
             "|---|---|---|---|---|---|---|---|"]
    any_row = False
    for r in results:
        perf = r["perf"]
        if not perf:
            continue
        status = perf["status"]
        if status.get("devkit") != "ok":
            lines.append(f"| {r['model']} | {r['kind']} | — | dev-kit run: {status.get('devkit', 'no result')} "
                         f"| | | | |")
            any_row = True
            continue
        ab = compare.compare_perf(perf["devkit"], perf["base"], warn=True)
        warnings += [f"{KEY_NAMES[key]} {r['model']}: {w}" for w in ab["warnings"]]
        base_kind = (baseline["entries"].get(key, {}).get(r["model"], {}).get("perf", {}).get(r["kind"])
                     if machine.valid_kind(r["kind"]) else None)
        nightly = compare.compare_perf(perf["devkit"], compare.perf_rows_from_json(base_kind["rows"])
                                       if base_kind else {}, warn=False)
        for row, nrow in zip(ab["rows"], nightly["rows"]):
            do, dn, dd = row["decode"]
            po, pn, pd = row["prefill"]
            vs = fmt_delta(nrow["decode"][2]) if base_kind else "no baseline for this kind"
            lines.append(f"| {r['model']} | {r['kind']} | {row['context_length_k']:g}k | {fmt(do)} → {fmt(dn)} "
                         f"| {fmt_delta(dd)} | {fmt(po)} → {fmt(pn)} | {fmt_delta(pd)} | {vs} |")
            any_row = True
        if status.get("base") not in (None, "ok"):
            lines.append(f"| {r['model']} | {r['kind']} | — | base run: {status.get('base', 'not run')}; "
                         f"no A/B | | | | |")
    return lines if any_row else []


def acc_section(results: list[dict], baseline: dict, key: str, bench: str, warnings: list,
                problems: list) -> list[str]:
    title = {"gsm8k": "GSM8K accuracy (%)", "docvqa": "DocVQA ANLS"}[bench]
    lines = ["", f"#### {title}", "", "| Model | Nightly baseline | This run | Δ (points) | Notes |",
             "|---|---|---|---|---|"]
    any_row = False
    for r in results:
        s = r["acc"].get(bench)
        if s is None:
            continue
        base = baseline["entries"].get(key, {}).get(r["model"], {}).get(bench)
        c = compare.compare_acc(s, base)
        warnings += [f"{KEY_NAMES[key]} {r['model']} {bench}: {w}" for w in c["warnings"]]
        if not s["usable"]:
            problems.append(f"{KEY_NAMES[key]} {r['model']} {bench}: failed the bench gate")
        lines.append(f"| {r['model']} | {fmt(c['baseline'])} | {fmt(c['score'])} | {fmt_delta(c['delta'], '')} "
                     f"| {'; '.join(c['notes'])} |")
        any_row = True
    return lines if any_row else []


def hard_gate_failures(results: list[dict]) -> list[str]:
    """key model bench of every accuracy run that failed acc_common's gate."""
    return [f"{r['key']} {r['model']} {bench}" for r in results for bench in compare.ACC_KEYS
            if r["acc"].get(bench) is not None and not r["acc"][bench]["usable"]]


def render(results: list[dict], baseline: dict, matrix: list[dict], *, mode: str, head_sha: str,
           build_run_id: Optional[int], baseline_run_id: Optional[int]) -> str:
    warnings: list[str] = []
    problems: list[str] = []
    body: list[str] = []
    got = {(r["key"], r["model"]) for r in results}
    for key in KEYS:
        mine = sorted((r for r in results if r["key"] == key), key=lambda r: r["model"])
        expected = sorted({e["model"] for e in matrix if e.get("key") == key and compare.valid_model(e.get("model"))})
        if not mine and not expected:
            continue
        body += ["", f"### {KEY_NAMES[key]}"]
        body += perf_section(mine, baseline, key, warnings)
        for bench in compare.ACC_KEYS:
            body += acc_section(mine, baseline, key, bench, warnings, problems)
        missing = [m for m in expected if (key, m) not in got]
        if missing:
            problems.append(f"{KEY_NAMES[key]}: no results from the bench job of {', '.join(missing)}")
    head = [f"## Dev-kit bench ({mode}) on `{head_sha[:10]}`", ""]
    where = []
    if build_run_id:
        where.append(f"Build and Test run {build_run_id}")
    if mode == "refresh":
        where.append("dev-kit full refresh: this run's numbers only, no comparison")
    elif mode == "pr":
        where.append("the pull request's staged packages: this run's numbers only, no comparison "
                     "(to be designed later)")
    elif baseline_run_id and baseline.get("sha"):
        where.append(f"nightly baseline: run {baseline_run_id} on `{baseline['sha'][:10]}`"
                     + (f" ({baseline['updated']})" if baseline.get("updated") else ""))
    else:
        where.append("no nightly baseline yet")
    head.append(" · ".join(where))
    if problems:
        head += ["", "**❌ Failed** (hard gate: no predictions or too many unanswered, or no results):", ""]
        head += [f"- {p}" for p in problems]
    if warnings:
        head += ["", f"**⚠️ Warnings** (soft gates: perf more than {compare.PERF_WARN_PCT:g}% slower than the "
                     f"base package on the same runner, accuracy more than {compare.ACC_WARN_POINTS:g} points "
                     f"below the nightly baseline):", ""]
        head += [f"- {w}" for w in warnings]
    if not problems and not warnings:
        head += ["", "✅ No hard-gate failure and no soft-gate warning."]
    tail = ["", "_Perf: flm bench, decode and prefill average tok/s per context. A/B is the one perf signal "
            "that judges a change; the runners differ by machine kind, so the nightly comparison is "
            "informational and only made for the same kind. Accuracy: GSM8K (first 50 questions) and "
            "DocVQA (when enabled), greedy; unanswered questions count as wrong. Per-question reports, "
            "server logs and CSVs: the devkit-bench-result-* artifacts._"]
    return cap("\n".join(head + body + tail) + "\n")


def cap(text: str, limit: int = MAX_BYTES) -> str:
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    note = "\n\n_(cut: the summary is over its size limit; see the devkit-bench-result-* artifacts)_\n"
    keep = data[:limit - len(note.encode("utf-8"))].decode("utf-8", errors="ignore")
    keep = keep[:keep.rfind("\n")] if "\n" in keep else keep
    return keep + note


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--matrix", default="")
    ap.add_argument("--mode", choices=("pr", "nightly", "refresh", "dispatch"), required=True)
    ap.add_argument("--head-sha", required=True)
    ap.add_argument("--build-run-id", default="")
    ap.add_argument("--run-id", default="", help="this run (Build and Test's, or a dispatched devkit-bench.yml's)")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--baseline-run-id", default="")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--new-baseline", default="")
    ap.add_argument("--outputs", default="", help="appends hard-gate=pass|fail (GITHUB_OUTPUT)")
    args = ap.parse_args(argv)
    if not SHA.match(args.head_sha):
        ap.error("--head-sha must be a full commit id")
    results = load_results(Path(args.results))
    matrix = read_json(Path(args.matrix)) if args.matrix else []
    matrix = matrix if isinstance(matrix, list) else []
    matrix = [e for e in matrix if isinstance(e, dict)]
    baseline = parse_baseline(read_json(Path(args.baseline) / "baseline.json") if args.baseline else None)
    text = render(results, baseline, matrix, mode=args.mode, head_sha=args.head_sha,
                  build_run_id=compare.count(args.build_run_id),
                  baseline_run_id=compare.count(args.baseline_run_id))
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(text, encoding="utf-8")
    print(text)
    if args.outputs:
        with open(args.outputs, "a", encoding="utf-8") as fh:
            fh.write(f"hard-gate={'fail' if hard_gate_failures(results) else 'pass'}\n")
    if args.new_baseline:
        out = Path(args.new_baseline)
        out.mkdir(parents=True, exist_ok=True)
        new = upsert_baseline(baseline, results, args.head_sha, compare.count(args.run_id))
        (out / "baseline.json").write_text(json.dumps(new, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
