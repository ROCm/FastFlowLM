#!/usr/bin/env python3
"""Render d3 MMMU-Pro vision results as one Markdown report, platforms side by side.

Reads every *_results.json that d3_mmmu_test.py wrote under the given directory
(only those in a d3-results folder when the artifacts also hold baseline runs)
and prints GitHub-flavoured Markdown to stdout, for $GITHUB_STEP_SUMMARY.

d3_mmmu_test.py also calls render() for the report of a single run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from asr_summary import PLATFORM_ORDER, cell, label


def fraction(correct, total) -> str:
    if correct is None or not total:
        return "-"
    return f"{correct}/{total} ({correct / total * 100:.1f}%)"


def verdict(result: dict) -> str:
    return "**FAIL**" if any(c["status"] == "FAIL" for c in result.get("checks", [])) else "**PASS**"


def render(results: list[dict], title: str = "d3 MMMU-Pro vision test") -> str:
    results = sorted(results, key=lambda r: (PLATFORM_ORDER.index(r["platform"])
                                             if r["platform"] in PLATFORM_ORDER else 99,
                                             r["platform"]))
    lines = [f"# {title}", ""]
    if not results:
        return "\n".join(lines + ["_No results were produced._", ""])

    first = results[0]
    data = first.get("eval", {})
    compared = any(r.get("baseline") for r in results)
    lines += [
        f"{first.get('model')} on {data.get('count')} questions of {data.get('source')}, "
        f"dataset commit `{str(data.get('revision'))[:12]}`, seed {data.get('seed')}. "
        "Builds under test: flm " + ", ".join(sorted({str(r.get('version')) for r in results}))
        + ". Compared against: " + ("main's latest build, run first on the same machine"
                                    if compared else "nothing") + ".",
        "",
        "| Measure | " + " | ".join(label(r["platform"]) for r in results) + " |",
        "|---|" + "---|" * len(results),
    ]

    def row(name: str, render_cell) -> None:
        lines.append(f"| {name} | " + " | ".join(render_cell(r) for r in results) + " |")

    def base(result: dict) -> dict:
        return result.get("baseline") or {}

    row("Verdict", verdict)
    row("Accuracy", lambda r: fraction(r.get("correct"), r.get("questions")))
    if compared:
        row("main's accuracy", lambda r: fraction(base(r).get("correct"), base(r).get("questions")))
        row("Same answer as main", lambda r: fraction(base(r).get("agree"), base(r).get("questions")))
        row("Mean \\|dp\\| vs main", lambda r: "-" if base(r).get("mean_dp") is None
            else f"{base(r)['mean_dp']:.4f}")
    row("Chance level", lambda r: "-" if r.get("chance") is None else f"{r['chance'] * 100:.1f}%")
    row("Median latency (s)", lambda r: "-" if r.get("median_seconds") is None
        else f"{r['median_seconds']:.2f}" + (f" (main {base(r)['median_seconds']:.2f})"
                                             if base(r).get("median_seconds") else ""))
    row("Median input tokens", lambda r: str(r.get("median_input_tokens") or "-"))
    lines.append("")

    lines += ["**Failed checks**", ""]
    failed = False
    for result in results:
        rows = [c for c in result.get("checks", []) if c["status"] != "ok"]
        if not rows:
            continue
        failed = True
        lines += [f"*{label(result['platform'])}*", ""]
        lines += [f"- **{c['status']}** {cell(c['check'])}" + (f": {cell(c['detail'])}" if c["detail"] else "")
                  for c in rows]
        lines.append("")
    if not failed:
        lines += ["_None._", ""]
    return "\n".join(lines)


def load_results(root: Path) -> list[dict]:
    paths = sorted(root.rglob("*_results.json"))
    # Artifacts also carry the baseline's own run; report the builds under test.
    if any("d3-results" in p.parts for p in paths):
        paths = [p for p in paths if "d3-results" in p.parts]
    results = []
    for path in paths:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"<!-- skipped {path}: {exc} -->")
            continue
        if "platform" in result and "checks" in result:
            results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", help="Directory holding the downloaded artifacts.")
    args = parser.parse_args(argv)
    print(render(load_results(Path(args.directory))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
