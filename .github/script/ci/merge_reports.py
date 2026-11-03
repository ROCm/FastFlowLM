#!/usr/bin/env python3
"""Merges the npu_test reports of one Build and Test run into one summary.

    merge_reports.py <dir with npu-test-* artifacts> [--output summary.md]

Each package's job already puts its own report on its job page; this puts
them side by side on the run page, so one look shows which packages and
runners failed which suites.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = []
    for path in sorted(args.reports.glob("**/results.json")):
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)

    lines = ["## NPU tests", ""]
    if not reports:
        lines.append("No test reports were uploaded (tests skipped, or every test job failed "
                     "before writing one).")
    else:
        suites = sorted({s for r in reports for s in r["suites"]})
        lines += ["| Package | Tier | flm | " + " | ".join(suites) + " | Time |",
                  "|---|---|---|" + "---|" * len(suites) + "---|"]
        for report in reports:
            cells = []
            for suite in suites:
                results = [r for r in report["results"] if r["suite"] == suite]
                if suite not in report["suites"]:
                    cells.append("")
                elif any(r["status"] == "fail" for r in results):
                    failed = sum(1 for r in results if r["status"] == "fail")
                    cells.append(f"❌ {failed}")
                elif any(r["status"] == "warn" for r in results):
                    cells.append("⚠️")
                elif results:
                    cells.append("✅")
                else:
                    cells.append("–")
            lines.append(f"| {report['label']} | {report['tier']} | {report['flm_version']} | "
                         + " | ".join(cells) + f" | {report['seconds'] / 60:.0f} min |")
        lines += ["", "❌ n: n failed cases, ⚠️ warnings only, – nothing to run. "
                  "Each test job's page lists the individual cases.", ""]

        failures = [(rep["label"], r) for rep in reports for r in rep["results"]
                    if r["status"] == "fail"]
        if failures:
            lines += ["<details><summary>Failed cases</summary>", "",
                      "| Package | Suite | Case | Model | Detail |", "|---|---|---|---|---|"]
            for label, r in failures[:200]:
                detail = r["detail"].replace("|", "\\|").replace("\n", " ")[:200]
                lines.append(f"| {label} | {r['suite']} | {r['case']} | {r['model']} | {detail} |")
            if len(failures) > 200:
                lines.append(f"| | | | | {len(failures) - 200} more not shown |")
            lines += ["", "</details>", ""]

    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
