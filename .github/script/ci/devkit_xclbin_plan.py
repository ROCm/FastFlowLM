#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Plans the dev kit's kernel builds (flm-dev-kit/FLM_Xclbin) for CI.

    devkit_xclbin_plan.py --mode changed --base <sha> [--head <sha>] [--shards N] [--per-shard K]
    devkit_xclbin_plan.py --mode changed --files <path>... [--base <sha>]
    devkit_xclbin_plan.py --mode assigned|all [--shards N] [--per-shard K]

Prints one JSON object (and, with --github-output, appends plan=, matrix=,
count=, empty=, touched-models=, bench-models= and mode= to $GITHUB_OUTPUT):

  {"mode": <mode actually used>,
   "units": [{"unit", "model_type" (null when the unit does not fan out),
              "artifact_name" (xclbins-<unit, / as _>[_<MODEL_TYPE>])}, ...],
   "shards": [[<unit entry>, ...], ...],
   "touched_models": [<kernel_assignment.json models using a built artifact>],
   "bench_models": [<the models worth benching: see below>]}

bench_models: in mode changed, the models
using a build selected because its source (or its build_units.json entry)
changed, plus the models whose kernel_assignment.json entry was added or
changed. A canary build, or one rebuilt only because a changed model uses
it, does not make its other users worth benching. In modes assigned and all,
touched_models.

A build is one (unit, MODEL_TYPE) pair of flm-dev-kit/FLM_Xclbin/
build_units.json ("make bitstream [MODEL_TYPE=...]" in that unit).

  all       every build of every unit
  assigned  the builds whose artifact kernel_assignment.json names (what the
            product ships)
  changed   from the files changed between the merge base of --base and
            --head (git diff --name-only base...head, as changes.py), or
            --files:
            - a file under FLM_Xclbin selects the nearest unit at or above
              its directory (walking up); a file under a directory with no
              unit above it (shared code such as VisionAttention/makefiles/,
              or a family's parent directory) selects every unit below the
              nearest ancestor that has units below it. No dependency map
              exists yet, so this is conservative. A selected unit builds the
              MODEL_TYPEs kernel_assignment.json uses, or all of them when it
              uses none
            - kernel_assignment.json: the artifacts of the models whose entry
              was added or changed, exactly
            - rtp_addresses.lock.txt: the artifacts whose stanzas were
              added, changed or removed (every artifact the lock names when
              the base's lock is unknown), so the RTP check runs on them
            - build_units.json: the units whose entry was added or changed:
              the MODEL_TYPEs added to a unit, or when none was (a new unit
              without MODEL_TYPEs, one removed), as for a changed unit
            - a file under a unit that only the base's build_units.json has
              (the unit was removed or renamed): nothing for that file
            - toolchain.image, toolchain.lock, docker/aie-build*, and the
              dev-kit CI (changes.py's DEVKIT_CI, which covers this script,
              the devkit-* actions and workflows): the canary builds, the
              kernels docker/aie-build.pdi.sha256 pins, not every unit.
              changes.py gives toolchain.lock and docker/aie-build* no area
              (devkit-aie-image.yml checks them), so those two only add the
              canary when another change already selected the xclbin area
            - a shipped kernel, src/xclbins/<folder>/<slot>.xclbin (added,
              replaced or removed): the artifact kernel_assignment.json
              puts in <slot> for every model whose folder is <folder>, so
              devkit_xclbin_compare.py reports how the rebuild compares with
              the shipped file. Nothing when no model assigns that slot (the
              catalog check in devkit_check.py reports such a file). These
              builds are not bench_models: their sources did not change
            - documentation (*.md), .gitignore files, host-only code
              (host_common/) and the FLM_Xclbin scripts no bitstream build
              reads (setup.sh, build_drivers.sh, dimhelper.py): nothing
            An unknown base (empty, all zeros, or one git cannot diff, as in
            a shallow clone) plans mode all, as changes.py does, and shards
            it as mode all is: up to --fallback-shards shards (default 16),
            filling each, whatever --shards and --per-shard say.

Shards: the builds are split into at most --shards contiguous shards of
nearly equal size, one per --per-shard builds (default: as many as
--shards), keeping a unit's MODEL_TYPEs together where the sizes allow.
Every build counts the same: the measured cost (7-61 s per build) is small
next to a runner's start-up time.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from changes import DEVKIT_CI, OUTSIDE_BUILD_AND_TEST, SHIPPED_KERNEL, xclbin_no_bitstream  # noqa: E402
from devkit_rtp_lock import LOCK_REL as RTP_LOCK, changed_artifacts  # noqa: E402

XCLBIN = "flm-dev-kit/FLM_Xclbin"
BUILD_UNITS = f"{XCLBIN}/build_units.json"
ASSIGNMENT = f"{XCLBIN}/kernel_assignment.json"
PINS = "docker/aie-build.pdi.sha256"
MODES = ("changed", "assigned", "all")

# The kernel toolchain: a change rebuilds the canary builds. changes.py
# selects the xclbin area for toolchain.image only; toolchain.lock and
# docker/aie-build* select no area (devkit-aie-image.yml checks them), so they
# add the canary only when another change already runs this plan.
TOOLCHAIN = re.compile(r"^(docker/aie-build[^/]*|flm-dev-kit/FLM_Xclbin/toolchain\.(lock|image))$")
# Files under FLM_Xclbin no "make bitstream" reads (and changes.py
# xclbin_no_bitstream: units' host program sources and visualize.py).
NO_BITSTREAM = re.compile(
    r"^flm-dev-kit/FLM_Xclbin/(.*\.md|(.*/)?\.gitignore|host_common/.*|setup\.sh|build_drivers\.sh"
    r"|dimhelper\.py)$")
# Used when PINS is missing or lists nothing.
DEFAULT_CANARY = (("Dequant_Q4_0_w128x512", None), ("MM/host7_mt_variant_bias", "GENERIC"))


def artifact_name(unit: str, model_type: Optional[str]) -> str:
    safe = unit.replace("/", "_")
    return f"xclbins-{safe}_{model_type}" if model_type else f"xclbins-{safe}"


def builds_of(unit: str, model_types: list) -> list[dict]:
    return [{"unit": unit, "model_type": mt, "artifact_name": artifact_name(unit, mt)}
            for mt in (model_types or [None])]


def load_units(text: Optional[str]) -> dict[str, list]:
    """unit -> model_types from build_units.json text; {} when unreadable."""
    try:
        units = json.loads(text or "")["units"]
        return {u: list(v.get("model_types") or []) for u, v in units.items()}
    except (ValueError, KeyError, TypeError, AttributeError):
        return {}


def load_assignment(text: Optional[str]) -> dict[str, dict]:
    """model -> entry from kernel_assignment.json text; {} when unreadable."""
    try:
        models = json.loads(text or "")["models"]
        return models if isinstance(models, dict) else {}
    except (ValueError, KeyError, TypeError):
        return {}


def used_artifacts(assignment: dict[str, dict]) -> set[str]:
    return {a for m in assignment.values() for a in (m.get("kernels") or {}).values()}


def load_canary(pins_text: Optional[str], units: dict[str, list]) -> list[tuple[str, Optional[str]]]:
    """The (unit, MODEL_TYPE) pairs the image check pins ("-" = no MODEL_TYPE)."""
    out = []
    for line in (pins_text or "").splitlines():
        parts = line.split()
        if len(parts) >= 4 and not parts[0].startswith("#"):
            unit, mt = parts[2], (None if parts[3] == "-" else parts[3])
            if unit in units and (mt is None or mt in units[unit]):
                out.append((unit, mt))
    return out or [(u, mt) for u, mt in DEFAULT_CANARY if u in units]


def unit_for(path: str, units: dict[str, list], gone: frozenset | set = frozenset()) -> list[str]:
    """The units a changed file under FLM_Xclbin selects (see the docstring).
    gone: units only the base has; a file under one of them selects nothing."""
    rel = path[len(XCLBIN) + 1:]
    parts = rel.split("/")[:-1]
    for i in range(len(parts), 0, -1):
        p = "/".join(parts[:i])
        if p in units:
            return [p]
        if p in gone:
            return []
    for i in range(len(parts), -1, -1):
        prefix = "/".join(parts[:i])
        below = sorted(u for u in units if not prefix or u.startswith(prefix + "/"))
        if below:
            return below
    return []


class Git:
    def __init__(self, repo: Path, base: str, head: str):
        self.repo, self.base, self.head = repo, base, head

    def _git(self, *args: str) -> Optional[str]:
        try:
            res = subprocess.run(["git", "-c", "core.quotePath=false", "-C", str(self.repo), *args],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
        except OSError:
            return None
        return res.stdout if res.returncode == 0 else None

    def merge_base(self) -> Optional[str]:
        out = self._git("merge-base", self.base, self.head)
        return out.strip() if out else None

    def files(self) -> Optional[list[str]]:
        out = self._git("diff", "--name-only", "--no-renames", f"{self.base}...{self.head}")
        return None if out is None else [l for l in out.splitlines() if l.strip()]

    def show(self, rev: str, path: str) -> Optional[str]:
        return self._git("show", f"{rev}:{path}")


class Plan:
    def __init__(self, units: dict[str, list], assignment: dict[str, dict]):
        self.units, self.assignment = units, assignment
        self.used = used_artifacts(assignment)
        self.selected: dict[str, dict] = {}
        self.reasons: list[str] = []
        # Artifacts selected because their source changed, and the models
        # whose assignment changed: what bench_models is made of.
        self.source: set[str] = set()
        self.assign_models: set[str] = set()

    def add(self, unit: str, model_types: list, source: bool = False) -> None:
        """These builds of unit ([] = the unit's one build without MODEL_TYPE);
        source: selected because the unit's source or entry changed."""
        for b in builds_of(unit, model_types):
            self.selected.setdefault(b["artifact_name"], b)
            if source:
                self.source.add(b["artifact_name"])

    def add_unit(self, unit: str) -> None:
        """A unit whose source changed: the MODEL_TYPEs the assignment uses, else all."""
        mts = self.units.get(unit)
        if mts is None:
            return
        if not mts:
            self.add(unit, [], source=True)
            return
        wanted = [mt for mt in mts if artifact_name(unit, mt) in self.used]
        self.add(unit, wanted or mts, source=True)

    def add_artifact(self, artifact: str) -> bool:
        for unit, mts in self.units.items():
            for b in builds_of(unit, mts):
                if b["artifact_name"] == artifact:
                    self.selected.setdefault(artifact, b)
                    return True
        return False

    def add_all(self, only_used: bool) -> None:
        for unit in sorted(self.units):
            for b in builds_of(unit, self.units[unit]):
                if not only_used or b["artifact_name"] in self.used:
                    self.selected.setdefault(b["artifact_name"], b)

    def builds(self) -> list[dict]:
        return sorted(self.selected.values(), key=lambda b: (b["unit"], b["model_type"] or ""))


def changed_plan(plan: Plan, files: list[str], base_units: dict[str, list],
                 base_assignment: Optional[dict], canary: list,
                 base_lock: Optional[str] = None, head_lock: Optional[str] = None) -> None:
    # Units removed or renamed: their old paths select nothing (a walk-up
    # from them finds no Makefile at head).
    gone = set(base_units) - set(plan.units)
    for path in files:
        if OUTSIDE_BUILD_AND_TEST.match(path):
            # No Build and Test job uses it (the CI unit tests,
            # merge_reports.py): changes.py selects no dev-kit area for it,
            # so it adds no canary here either.
            continue
        if TOOLCHAIN.match(path) or DEVKIT_CI.match(path):
            if canary:
                plan.reasons.append(f"{path}: canary builds")
            for unit, mt in canary:
                plan.add(unit, [mt] if mt else [])
        elif path == ASSIGNMENT:
            base = base_assignment or {}
            changed = sorted(m for m, e in plan.assignment.items() if base.get(m) != e)
            plan.reasons.append(f"{path}: entries added or changed: {', '.join(changed) or 'none'}")
            plan.assign_models.update(changed)
            for m in changed:
                for a in sorted(set((plan.assignment[m].get("kernels") or {}).values())):
                    if not plan.add_artifact(a):
                        plan.reasons.append(f"warning: {m} names {a}, which no build unit produces")
        elif path == RTP_LOCK:
            changed = changed_artifacts(base_lock, head_lock)
            plan.reasons.append(f"{path}: RTP stanzas changed: {', '.join(changed) or 'none'}")
            for a in changed:
                if not plan.add_artifact(a):
                    plan.reasons.append(f"warning: {path} names {a}, which no build unit produces")
        elif path == BUILD_UNITS:
            changed = sorted(u for u, mts in plan.units.items() if base_units.get(u) != mts)
            plan.reasons.append(f"{path}: units added or changed: {', '.join(changed) or 'none'}")
            for u in changed:
                # The MODEL_TYPEs added: none of them can be assigned yet, so
                # add_unit would skip them. Otherwise (a new unit without
                # MODEL_TYPEs, one removed) as a changed unit.
                old = base_units.get(u)
                added = [mt for mt in plan.units[u] if old is None or mt not in old]
                if added:
                    plan.add(u, added, source=True)
                else:
                    plan.add_unit(u)
        elif SHIPPED_KERNEL.match(path):
            add_shipped_kernel(plan, path)
        elif path.startswith(XCLBIN + "/") and not (NO_BITSTREAM.match(path)
                                                    or xclbin_no_bitstream(path, plan.units)):
            for u in unit_for(path, plan.units, gone):
                plan.add_unit(u)


def add_shipped_kernel(plan: Plan, path: str) -> None:
    """src/xclbins/<folder>/<slot>.xclbin: the artifact each model using
    <folder> assigns to <slot>, to compare with the shipped file."""
    folder, slot = path[len("src/xclbins/"):].rsplit("/", 1)
    hits = {m: e["kernels"][slot] for m, e in sorted(plan.assignment.items())
            if e.get("folder") == folder and slot in (e.get("kernels") or {})}
    if not hits:
        plan.reasons.append(f"{path}: shipped kernel no {ASSIGNMENT} model assigns: nothing to build")
        return
    for a in sorted(set(hits.values())):
        if not plan.add_artifact(a):
            plan.reasons.append(f"warning: {path} is {a}, which no build unit produces")
    plan.reasons.append(f"{path}: shipped kernel of {', '.join(hits)} -> {', '.join(sorted(set(hits.values())))}, "
                        "rebuilt to compare")


def shard(builds: list[dict], max_shards: int, per_shard: int = 0) -> list[list[dict]]:
    """Contiguous shards of nearly equal size: at most max_shards, one per
    per_shard builds (0: as many as max_shards)."""
    if not builds:
        return []
    per = per_shard or math.ceil(len(builds) / max_shards)
    n = max(1, min(max_shards, math.ceil(len(builds) / per)))
    size, extra = divmod(len(builds), n)
    out, i = [], 0
    for k in range(n):
        step = size + (1 if k < extra else 0)
        out.append(builds[i:i + step])
        i += step
    return out


def touched_models(assignment: dict[str, dict], builds: list[dict]) -> list[str]:
    built = {b["artifact_name"] for b in builds}
    return sorted(m for m, e in assignment.items() if built & set((e.get("kernels") or {}).values()))


def make_plan(mode: str, units: dict[str, list], assignment: dict[str, dict], *,
              files: Optional[list[str]] = None, base_units: Optional[dict] = None,
              base_assignment: Optional[dict] = None, canary: Optional[list] = None,
              base_lock: Optional[str] = None, head_lock: Optional[str] = None,
              max_shards: int = 4, per_shard: int = 0, fallback_shards: int = 16) -> dict:
    plan = Plan(units, assignment)
    if mode == "changed" and files is None:
        # Sharded as mode all is, not as the small change the caller expected.
        plan.reasons.append("changed files unknown: planning mode all")
        mode = "all"
        max_shards, per_shard = max(max_shards, fallback_shards), 0
    if mode == "changed":
        changed_plan(plan, files, base_units or {}, base_assignment,
                     canary if canary is not None else load_canary(None, units),
                     base_lock, head_lock)
    else:
        plan.add_all(only_used=(mode == "assigned"))
    builds = plan.builds()
    touched = touched_models(assignment, builds)
    if mode == "changed":
        bench = sorted(m for m, e in assignment.items()
                       if m in plan.assign_models or plan.source & set((e.get("kernels") or {}).values()))
    else:
        bench = touched
    return {"mode": mode, "units": builds, "shards": shard(builds, max_shards, per_shard),
            "touched_models": touched, "bench_models": bench, "reasons": plan.reasons}


def read(repo: Path, path: str) -> Optional[str]:
    try:
        return (repo / path).read_text(encoding="utf-8")
    except OSError:
        return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=MODES, default="changed")
    parser.add_argument("--base", default="", help="Base commit (changed); empty or all-zero means unknown.")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--files", nargs="*", help="Changed files, instead of diffing base...head.")
    parser.add_argument("--shards", type=int, default=4, help="Most shards (default 4).")
    parser.add_argument("--per-shard", type=int, default=0,
                        help="Builds per shard before another shard is added (default: fill --shards).")
    parser.add_argument("--fallback-shards", type=int, default=16,
                        help="Most shards when mode changed falls back to all (default 16).")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--github-output", action="store_true",
                        help="Also append the outputs to $GITHUB_OUTPUT.")
    args = parser.parse_args(argv)
    if args.shards < 1 or args.fallback_shards < 1 or args.per_shard < 0:
        parser.error("--shards and --fallback-shards must be at least 1 and --per-shard not negative")

    git = None
    if args.mode == "changed" and args.base.strip("0 "):
        mb = Git(args.repo, args.base, args.head).merge_base()
        git = Git(args.repo, mb, args.head) if mb else None

    # The head side from git when there is a range, so the plan matches the
    # diff whatever is checked out; the checkout otherwise.
    def head_text(path: str) -> Optional[str]:
        text = git.show(git.head, path) if git else None
        return text if text is not None else read(args.repo, path)

    units = load_units(head_text(BUILD_UNITS))
    assignment = load_assignment(head_text(ASSIGNMENT))
    if not units:
        print(f"devkit_xclbin_plan: error: {BUILD_UNITS} has no units", file=sys.stderr)
        return 1

    files = base_units = base_assignment = base_lock = head_lock = None
    if args.mode == "changed":
        if args.files is not None:
            files = [f.replace("\\", "/") for f in args.files]
        elif git:
            files = git.files()
        if git:
            base_units = load_units(git.show(git.base, BUILD_UNITS))
            # No file at the base (it is new): every entry counts as added.
            base_assignment = load_assignment(git.show(git.base, ASSIGNMENT))
            # None (no lock at the base, or git cannot tell): every stanza.
            base_lock = git.show(git.base, RTP_LOCK)
        head_lock = head_text(RTP_LOCK)
        canary = load_canary(head_text(PINS), units)
    else:
        canary = None

    result = make_plan(args.mode, units, assignment, files=files, base_units=base_units,
                       base_assignment=base_assignment, canary=canary,
                       base_lock=base_lock, head_lock=head_lock,
                       max_shards=args.shards, per_shard=args.per_shard,
                       fallback_shards=args.fallback_shards)
    for reason in result.pop("reasons"):
        print(f"devkit_xclbin_plan: {reason}", file=sys.stderr)
    print(f"devkit_xclbin_plan: mode {result['mode']}: {len(result['units'])} builds in "
          f"{len(result['shards'])} shards, {len(result['touched_models'])} models served, "
          f"{len(result['bench_models'])} to bench", file=sys.stderr)
    text = json.dumps(result, separators=(",", ":"))
    print(text)
    if args.github_output:
        target = os.environ.get("GITHUB_OUTPUT")
        if not target:
            print("--github-output given but $GITHUB_OUTPUT is not set", file=sys.stderr)
            return 2
        outputs = {
            "plan": text,
            "matrix": json.dumps({"shard": list(range(len(result["shards"])))}, separators=(",", ":")),
            "count": str(len(result["units"])),
            "empty": str(not result["units"]).lower(),
            "touched-models": json.dumps(result["touched_models"], separators=(",", ":")),
            "bench-models": json.dumps(result["bench_models"], separators=(",", ":")),
            "mode": result["mode"],
        }
        with open(target, "a", encoding="utf-8") as fh:
            fh.writelines(f"{k}={v}\n" for k, v in outputs.items())
    return 0


if __name__ == "__main__":
    sys.exit(main())
