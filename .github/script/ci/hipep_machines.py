#!/usr/bin/env python3
"""Lists the machines Test Windows HIPEP runs on, as its job matrix.

    hipep_machines.py [--tier pr] [--changed-models a,b] [--only stx-halo,npu-runners]
                      [--github-output]

.github/hipep-machines.json has one entry per kind of machine, each tested
by a job of its own:

  id               short name (stx-halo, npu-runners), in the job's name and
                   in its report's label
  runs-on          the labels its runners register with. Several runners may
                   share them; a run takes whichever is free. With the NPU
                   test runners' own (npu, sweep) the job shares those
                   runners, holding one for its run as an NPU test job does;
                   a label only some of them carry (oga-hipep) picks those
  enabled          false leaves it out of every run: a kind with no runner
                   yet, or with all of them down (a job no runner takes waits
                   a day before it fails, and holds up the run's verdict)
  max-model-gb     the largest oga-hipep model, by its catalog size, that the
                   machine's memory holds; larger ones are skipped there. 0
                   (the default) for no limit
  max-tier         the highest tier its job runs: install, pr or full (the
                   default). A run of a higher tier runs this one there, so
                   runners that cannot hold every model, or that the NPU
                   nightly needs, test the pr tier's models nightly
  changed-models   false leaves the models whose catalog entries the change
                   added or edited to the kinds that test them (true, the
                   default): a change to many entries downloads each new
                   revision within the job, which would hold a shared runner
                   for hours

--tier is the run's tier; each machine in the matrix carries the tier its job
runs. --changed-models, the change's (changes.py's hipep-models), goes to the
machines that test them, and an empty list to the rest. --only, from the
FLM_HIPEP_MACHINES repository variable, runs the named kinds instead of the
enabled ones ("none" for none), to take one out or put it back without a
commit. Prints the matrix, a JSON list that is empty when no kind runs;
--github-output also writes it to $GITHUB_OUTPUT as "machines".

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = ".github/hipep-machines.json"
ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
# npu_test's tiers, lowest first.
TIERS = ("install", "pr", "full")


def load(text: str) -> list[dict]:
    """The config's machines, checked; ValueError names the first bad one."""
    machines = json.loads(text).get("machines")
    if not isinstance(machines, list):
        raise ValueError('no "machines" list')
    out, seen = [], set()
    for entry in machines:
        if not isinstance(entry, dict):
            raise ValueError(f"not an object: {entry!r}")
        mid = entry.get("id")
        if not isinstance(mid, str) or not ID.match(mid) or mid in seen:
            raise ValueError(f"id {mid!r}: missing, repeated, or not lower-case letters, digits and -")
        seen.add(mid)
        labels = entry.get("runs-on")
        if (not isinstance(labels, list) or not all(isinstance(l, str) and l for l in labels)
                or "self-hosted" not in labels):
            raise ValueError(f"{mid}: runs-on must be a list of labels, self-hosted among them")
        enabled = entry.get("enabled", True)
        limit = entry.get("max-model-gb", 0)
        max_tier = entry.get("max-tier", "full")
        changed = entry.get("changed-models", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"{mid}: enabled must be true or false")
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit < 0:
            raise ValueError(f"{mid}: max-model-gb must be a number of GB, 0 for no limit")
        if max_tier not in TIERS:
            raise ValueError(f"{mid}: max-tier must be one of {', '.join(TIERS)}")
        if not isinstance(changed, bool):
            raise ValueError(f"{mid}: changed-models must be true or false")
        out.append({"id": mid, "runs-on": labels, "enabled": enabled, "max-model-gb": limit,
                    "max-tier": max_tier, "changed-models": changed})
    return out


def pick(machines: list[dict], only: str) -> tuple[list[dict], list[str]]:
    """The machines to run on, and the names in only that match none."""
    names = [n.strip() for n in only.split(",") if n.strip()]
    if not names:
        return [m for m in machines if m["enabled"]], []
    if names == ["none"]:
        return [], []
    known = {m["id"] for m in machines}
    return [m for m in machines if m["id"] in names], [n for n in names if n not in known]


def tier_on(machine: dict, tier: str) -> str:
    """The tier a machine's job runs: the run's, or its max-tier if lower."""
    return min(tier, machine["max-tier"], key=TIERS.index)


def matrix(machines: list[dict], tier: str, changed: str = "") -> str:
    """The job matrix's include list, on one line for $GITHUB_OUTPUT."""
    return json.dumps([{"id": m["id"], "runs-on": m["runs-on"], "max-model-gb": m["max-model-gb"],
                        "tier": tier_on(m, tier),
                        "changed-models": changed if m["changed-models"] else ""}
                       for m in machines],
                      separators=(",", ":"))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / CONFIG)
    parser.add_argument("--tier", choices=TIERS, default="pr", help="The run's tier (default: pr).")
    parser.add_argument("--changed-models", default="",
                        help="Comma-separated models whose catalog entries the change touched.")
    parser.add_argument("--only", default="",
                        help='Comma-separated kinds to run instead of the enabled ones; "none" for none.')
    parser.add_argument("--github-output", action="store_true",
                        help="Also append the matrix to $GITHUB_OUTPUT.")
    args = parser.parse_args(argv)

    try:
        machines = load(args.config.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"::error file={CONFIG}::{exc}")
        return 2
    chosen, unknown = pick(machines, args.only)
    for name in unknown:
        print(f"::warning::FLM_HIPEP_MACHINES names {name}, which {CONFIG} does not list")
    value = matrix(chosen, args.tier, args.changed_models.strip())
    names = [f"{m['id']} ({tier_on(m, args.tier)} tier)" for m in chosen]
    print(f"HIPEP machines: {', '.join(names) or 'none'}")
    print(f"machines={value}")
    if args.github_output:
        target = os.environ.get("GITHUB_OUTPUT")
        if not target:
            print("--github-output given but $GITHUB_OUTPUT is not set", file=sys.stderr)
            return 2
        with open(target, "a", encoding="utf-8") as out:
            out.write(f"machines={value}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
