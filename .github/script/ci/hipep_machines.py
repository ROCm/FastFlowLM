#!/usr/bin/env python3
"""Lists the machines Test Windows HIPEP runs on, as its job matrix.

    hipep_machines.py [--tier pr] [--changed-models a,b] [--families a,b] [--models a,b]
                      [--only stx-halo,npu-runners] [--github-output]

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
  changed-models   false leaves the models the change gives new weights
                   (added, republished or moved onto the GPU) to the kinds
                   that test them (true, the default): a change to many
                   entries downloads each new revision within the job, which
                   would hold a shared runner for hours. The job requires them (npu_test
                   --required-models): it downloads them whatever their size,
                   and fails if one does not pull, bar one larger than its
                   max-model-gb, a skip. When no kind that runs takes them
                   (FLM_HIPEP_MACHINES names only kinds with false), every
                   kind that runs does, with a warning, so that a change's
                   models are never silently untested
  standby          true: below the full tier, the kind runs only when the
                   other kinds that run cannot test the change: none of them
                   runs, the change has models it takes (changed-models), or
                   a model the run tests (one per --families family, as
                   npu_test's PR_GPU picks it, or the --models a dispatch
                   names) is larger than every one of their max-model-gb.
                   So the STX Halo tests the pull requests that need it, and
                   the shared runners the rest. false (the default): it runs
                   on every tier

--tier is the run's tier; each machine in the matrix carries the tier its job
runs. --changed-models, the change's (changes.py's hipep_models), goes to the
machines that test them, and an empty list to the rest. --only, from the
FLM_HIPEP_MACHINES repository variable, runs the named kinds instead of the
enabled ones ("none" for none), to take one out or put it back without a
commit. --families (changes.py's hipep_families) and --models (a dispatch's)
decide whether a standby kind runs, from the sizes in src/model_list.json. Prints the matrix, a JSON list that is empty when no kind runs;
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
MODEL_LIST = "src/model_list.json"
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
        standby = entry.get("standby", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"{mid}: enabled must be true or false")
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit < 0:
            raise ValueError(f"{mid}: max-model-gb must be a number of GB, 0 for no limit")
        if max_tier not in TIERS:
            raise ValueError(f"{mid}: max-tier must be one of {', '.join(TIERS)}")
        if not isinstance(changed, bool):
            raise ValueError(f"{mid}: changed-models must be true or false")
        if not isinstance(standby, bool):
            raise ValueError(f"{mid}: standby must be true or false")
        out.append({"id": mid, "runs-on": labels, "enabled": enabled, "max-model-gb": limit,
                    "max-tier": max_tier, "changed-models": changed, "standby": standby})
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


def gpu_sizes(text: str) -> dict[str, tuple[str, float]]:
    """oga-hipep model id -> (family, size in GB), from model_list.json text."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from changes import family_of, load_catalog, runs_on_gpu  # noqa: E402
    out = {}
    for mid, entry in load_catalog(text).items():
        if runs_on_gpu(mid, entry):
            size = entry.get("size")
            out[mid] = (family_of(entry), size / 1e9 if isinstance(size, (int, float)) else 0.0)
    return out


def tested_models(sizes: dict[str, tuple[str, float]], families: set, models: list) -> list:
    """The oga-hipep models a run below the full tier tests (bar required
    ones): the models a dispatch names, else one per family, as npu_test's
    Plan.chat picks them (PR_GPU; some families only when touched)."""
    if models:
        return [m for m in models if m in sizes]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from npu_test.plan import PR_GPU, PR_GPU_WHEN_TOUCHED  # noqa: E402
    picked = []
    for family in sorted({fam for fam, _ in sizes.values()}):
        if family in PR_GPU_WHEN_TOUCHED and family not in families:
            continue
        members = [m for m, (fam, _) in sizes.items() if fam == family]
        pick = PR_GPU.get(family)
        picked.append(pick if pick in members else min(members, key=lambda m: (sizes[m][1], m)))
    return picked


def why_standby_runs(machine: dict, others: list[dict], changed: str,
                     tested: list, sizes: dict[str, tuple[str, float]]) -> Optional[str]:
    """Why a standby kind runs below the full tier, beside the other kinds
    that run (others), or None when they test everything it would."""
    if not others:
        return "no other kind runs"
    if changed and machine["changed-models"]:
        return f"it tests the change's models ({changed})"
    for model in tested:
        size = sizes.get(model, ("", 0.0))[1]
        if all(0 < m["max-model-gb"] < size for m in others):
            return f"{model} ({size:.0f} GB) is larger than the other kinds hold"
    return None


def standby(chosen: list[dict], tier: str, changed: str, families: set, models: list,
            catalog_text: Optional[str]) -> tuple[list[dict], list[str]]:
    """chosen less the standby kinds that need not run, and a line per
    standby kind on whether it runs. Every kind runs on the full tier, and
    a standby one runs when the catalog cannot be read."""
    if tier == "full":
        return chosen, []
    others = [m for m in chosen if not m["standby"]]
    sizes, tested = {}, []
    if catalog_text is not None:
        sizes = gpu_sizes(catalog_text)
        tested = tested_models(sizes, families, models)
    keep, notes = [], []
    for m in chosen:
        if not m["standby"]:
            keep.append(m)
            continue
        why = (why_standby_runs(m, others, changed, tested, sizes) if catalog_text is not None
               else f"{MODEL_LIST} unreadable")
        notes.append(f"{m['id']} (standby): " + (f"runs, {why}" if why else "not needed"))
        if why:
            keep.append(m)
    return keep, notes


def tier_on(machine: dict, tier: str) -> str:
    """The tier a machine's job runs: the run's, or its max-tier if lower."""
    return min(tier, machine["max-tier"], key=TIERS.index)


def takes_changed(machines: list[dict]) -> set[str]:
    """The ids of the machines that test the change's models: those whose
    changed-models is true, or every one when none of them is."""
    takers = {m["id"] for m in machines if m["changed-models"]}
    return takers or {m["id"] for m in machines}


def matrix(machines: list[dict], tier: str, changed: str = "") -> str:
    """The job matrix's include list, on one line for $GITHUB_OUTPUT."""
    takers = takes_changed(machines)
    return json.dumps([{"id": m["id"], "runs-on": m["runs-on"], "max-model-gb": m["max-model-gb"],
                        "tier": tier_on(m, tier),
                        "changed-models": changed if m["id"] in takers else ""}
                       for m in machines],
                      separators=(",", ":"))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / CONFIG)
    parser.add_argument("--tier", choices=TIERS, default="pr", help="The run's tier (default: pr).")
    parser.add_argument("--changed-models", default="",
                        help="Comma-separated models whose catalog entries the change touched.")
    parser.add_argument("--families", default="",
                        help="Comma-separated families the run treats as touched (changes.py's hipep_families).")
    parser.add_argument("--models", default="",
                        help="Comma-separated models a dispatch names, the only ones the run tests.")
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / MODEL_LIST)
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
    changed = args.changed_models.strip()
    try:
        catalog_text = args.catalog.read_text(encoding="utf-8")
    except OSError:
        catalog_text = None
    split = lambda v: [x.strip() for x in v.split(",") if x.strip()]  # noqa: E731
    chosen, notes = standby(chosen, args.tier, changed, set(split(args.families)),
                            split(args.models), catalog_text)
    for note in notes:
        print(note)
    if changed and not chosen:
        print(f"::warning::no HIPEP machine runs, so nothing tests the oga-hipep models this "
              f"change gives new weights: {changed}")
    elif changed and not any(m["changed-models"] for m in chosen):
        print(f"::warning::no HIPEP machine that runs tests changed models, so "
              f"{', '.join(m['id'] for m in chosen)} test{'s' if len(chosen) == 1 else ''} "
              f"them, bar those larger than its max-model-gb: {changed}")
    value = matrix(chosen, args.tier, changed)
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
