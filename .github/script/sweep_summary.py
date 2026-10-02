#!/usr/bin/env python3
"""Render the sweep summary JSON files as a Markdown report.

Reads every *_summary.json written by the sweep scripts under the given
directory and prints a GitHub-flavoured Markdown report to stdout. Intended
for $GITHUB_STEP_SUMMARY, but works fine on a terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_summaries(root: Path) -> list[dict]:
    summaries = []
    for path in sorted(root.rglob("*_summary.json")):
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"<!-- skipped {path}: {exc} -->")
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Directory holding the downloaded artifacts.")
    parser.add_argument("--linux-result", default="", help="Job result for the Linux sweep.")
    parser.add_argument("--windows-result", default="", help="Job result for the Windows sweep.")
    args = parser.parse_args(argv)

    out: list[str] = ["# Model Sweep Results", ""]

    if args.linux_result or args.windows_result:
        out += ["| Platform | Overall |", "| --- | --- |"]
        if args.linux_result:
            out.append(f"| Linux | {args.linux_result} |")
        if args.windows_result:
            out.append(f"| Windows | {args.windows_result} |")
        out.append("")

    summaries = load_summaries(Path(args.directory))
    if not summaries:
        out += ["_No sweep summaries were produced._", ""]
        print("\n".join(out))
        return 0

    out += [
        "| Task | Platform | FLM | Models | Rows | Failures | Not run | Restarts |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        tested = item.get("models_tested", 0)
        selected = item.get("models_selected", tested)
        out.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                item.get("task", "?"),
                item.get("platform", "?"),
                item.get("flm_version", "?"),
                f"{tested}/{selected}" if tested != selected else tested,
                item.get("rows_written", 0),
                len(item.get("failures", [])),
                len(item.get("not_attempted", [])),
                item.get("server_restarts", 0),
            )
        )
    out.append("")

    failing = [item for item in summaries if item.get("failures")]
    out += ["## Failures", ""]
    if not failing:
        out += ["_None._", ""]
    for item in failing:
        out.append(f"**{item.get('task', '?')} / {item.get('platform', '?')}**")
        out.append("")
        for failure in item["failures"]:
            # Collapse newlines so a multi-line server error stays one bullet.
            out.append("- " + " ".join(str(failure).split()))
        out.append("")

    # Models the sweep never got to, because the server stopped coming back.
    # Reported apart from failures so they are not mistaken for tested models.
    skipped = [item for item in summaries if item.get("not_attempted")]
    if skipped:
        out += ["## Not attempted", ""]
        for item in skipped:
            out.append(f"**{item.get('task', '?')} / {item.get('platform', '?')}**")
            out.append("")
            for model in item["not_attempted"]:
                out.append(f"- {model}")
            out.append("")

    # Not a problem, but worth stating: these models were tested with one
    # round instead of two, by design rather than by accident.
    single_turn = [item for item in summaries if item.get("single_turn")]
    if single_turn:
        out += [
            "## Single-turn models",
            "",
            "Follow-up round skipped by design; refusing a second turn is "
            "correct behaviour for these.",
            "",
        ]
        for item in single_turn:
            listed = ", ".join(f"`{m}`" for m in item["single_turn"])
            out.append(
                f"- **{item.get('task', '?')} / {item.get('platform', '?')}**: {listed}"
            )
        out.append("")

    # Anything the sweep decided on its own, with the reason it decided it.
    # Routine start/stop is omitted; a restart is never routine.
    NOISE = {"server_start", "server_stop"}
    eventful = [
        (item, [e for e in item.get("events", []) if e.get("event") not in NOISE])
        for item in summaries
    ]
    eventful = [(item, events) for item, events in eventful if events]
    if eventful:
        out += ["## Server events", ""]
        for item, events in eventful:
            out.append(f"**{item.get('task', '?')} / {item.get('platform', '?')}**")
            out.append("")
            out += ["| At | Event | Reason | Detail |", "| --- | --- | --- | --- |"]
            for event in events:
                detail = " ".join(str(event.get("detail", "")).split())
                models = event.get("models") or event.get("model")
                if models:
                    listed = models if isinstance(models, str) else ", ".join(models)
                    detail = f"{detail} ({listed})" if detail else listed
                out.append(
                    "| +{}s | {} | {} | {} |".format(
                        event.get("elapsed_s", "?"),
                        event.get("event", "?"),
                        event.get("reason", ""),
                        detail,
                    )
                )
            out.append("")
            out.append(f"Full server output: `{item.get('server_log', 'server log')}`")
            out.append("")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
