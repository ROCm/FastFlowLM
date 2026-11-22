#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Assembles the kernels a dev-kit CI run built into per-model folders.

    devkit_assemble_xclbins.py --artifacts <dir> --out <dir> [--assignment <json>]
                               [--fail-on-partial] [--github-output]

<artifacts> holds the built kernels one folder per build, named after the
build's artifact (xclbins-<unit, / as _>[_<MODEL_TYPE>]), at any depth: the
devkit-xclbins-shard-<i> artifacts downloaded side by side
(<artifacts>/devkit-xclbins-shard-0/xclbins-Dequant_Q4_0_w128x512/...) or
merged. For every model of flm-dev-kit/FLM_Xclbin/kernel_assignment.json,
each {<slot>.xclbin: artifact} (or <slot>.switches, d3's packed control-packet
switches) whose artifact was built is copied to
<out>/<model folder>/<slot>.xclbin, as FastFlowLM's xclbins/ tree lays them
out. Of an
artifact's .xclbin files, the one named <slot>.xclbin is taken (a unit that
builds several designs names each after the slot it fills); otherwise the
only one, and an artifact with several and none of that name is ambiguous
(the slot counts as missing, with a warning). rtp_addresses.txt stays in the
build artifacts: no product reads it from the model folder.

Models may share a folder; two that put different artifacts in the same
<folder>/<slot> are an error (exit 1), not a silent overwrite. A folder or
slot that is not a plain relative path ([A-Za-z0-9_.+-] parts, no "..", a
slot ending in .xclbin or .switches) is never written: the slot counts as missing.

Prints one line per model (OK, PARTIAL with its missing slots, SKIP) and
a JSON summary: {"complete": [...], "partial": [...], "touched": [complete +
partial]}.
With --github-output, appends complete-models=, partial-models=,
touched-models= (JSON lists), found= and count= (files copied) to
$GITHUB_OUTPUT, and a table to $GITHUB_STEP_SUMMARY when that is set.
--fail-on-partial exits 1 when a model got some but not all of its kernels.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSIGNMENT = REPO_ROOT / "flm-dev-kit/FLM_Xclbin/kernel_assignment.json"

# What a model folder and a slot may be: they become paths under --out.
SAFE_FOLDER = re.compile(r"[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)*")
# A slot is a kernel (<name>.xclbin) or the d3 control-packet switches (<name>.switches, one packed file).
SAFE_SLOT = re.compile(r"[A-Za-z0-9_.+-]+\.(xclbin|switches)")


def safe_folder(folder: object) -> bool:
    return (isinstance(folder, str) and bool(SAFE_FOLDER.fullmatch(folder))
            and not any(p in (".", "..") for p in folder.split("/")))


def safe_slot(slot: object) -> bool:
    return isinstance(slot, str) and bool(SAFE_SLOT.fullmatch(slot))


def find_artifacts(root: Path) -> dict[str, Path]:
    """Artifact name -> its folder: every directory named xclbins-* under root.
    Two folders with the same name are an error (two shards built it)."""
    out: dict[str, Path] = {}
    for d in sorted(p for p in root.rglob("xclbins-*") if p.is_dir()):
        if d.name in out:
            raise ValueError(f"{d.name} appears twice: {out[d.name]} and {d}")
        out[d.name] = d
    return out


def pick(folder: Path, slot: str) -> tuple[Optional[Path], str]:
    """The file of folder for slot (an .xclbin, or a .switches file for a .switches slot), or (None, why not)."""
    suffix = Path(slot).suffix
    found = sorted(p for p in folder.rglob("*" + suffix) if p.is_file())
    if not found:
        return None, f"no {suffix}"
    for p in found:
        if p.name == slot:
            return p, ""
    if len(found) > 1:
        return None, f"ambiguous: {len(found)} {suffix} files, none named {slot}"
    return found[0], ""


def assemble(assignment: dict, artifacts: dict[str, Path], out: Path) -> dict:
    complete, partial, missing, copied = [], [], {}, 0
    written: dict[Path, str] = {}
    for model, entry in assignment.get("models", {}).items():
        folder, kernels = entry["folder"], entry.get("kernels", {})
        got, miss = 0, []
        for slot, artifact in kernels.items():
            if not safe_folder(folder) or not safe_slot(slot):
                miss.append(f"{slot} <- {artifact} (unsafe folder/slot)")
                continue
            dest = out / folder / slot
            if dest in written and written[dest] != artifact:
                raise ValueError(f"{folder}/{slot}: {written[dest]} and {artifact} (model {model})")
            # Claimed whether or not it was built: the conflict is in the map.
            written[dest] = artifact
            src = artifacts.get(artifact)
            if src is None:
                miss.append(f"{slot} <- {artifact} (not built)")
                continue
            path, why = pick(src, slot)
            if path is None:
                miss.append(f"{slot} <- {artifact} ({why})")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            got += 1
            copied += 1
        if got and got == len(kernels):
            complete.append(model)
            print(f"  OK      {model} ({got}/{len(kernels)}) -> {folder}")
        elif got:
            partial.append(model)
            missing[model] = miss
            print(f"  PARTIAL {model} ({got}/{len(kernels)}) -> {folder}")
            for m in miss:
                print(f"            missing: {m}")
        else:
            print(f"  SKIP    {model} (nothing built for it)")
    return {"complete": complete, "partial": partial, "touched": complete + partial,
            "missing": missing, "count": copied}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--fail-on-partial", action="store_true")
    parser.add_argument("--github-output", action="store_true")
    args = parser.parse_args(argv)

    try:
        assignment = json.loads(args.assignment.read_text(encoding="utf-8"))
        artifacts = find_artifacts(args.artifacts) if args.artifacts.is_dir() else {}
    except (OSError, ValueError) as e:
        print(f"::error::{e}")
        return 1
    print(f"{len(artifacts)} built artifacts under {args.artifacts}")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        r = assemble(assignment, artifacts, args.out)
    except ValueError as e:
        print(f"::error::{e}")
        return 1
    used = {a for m in assignment.get("models", {}).values() for a in m.get("kernels", {}).values()}
    for name in sorted(set(artifacts) - used):
        print(f"  (built, no model uses it: {name})")
    if r["partial"]:
        print(f"::warning::Models with some but not all kernels built: {', '.join(r['partial'])}")
    print(json.dumps({k: r[k] for k in ("complete", "partial", "touched")}))

    if args.github_output:
        target = os.environ.get("GITHUB_OUTPUT")
        if not target:
            print("--github-output given but $GITHUB_OUTPUT is not set", file=sys.stderr)
            return 2
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(f"complete-models={json.dumps(r['complete'], separators=(',', ':'))}\n")
            fh.write(f"partial-models={json.dumps(r['partial'], separators=(',', ':'))}\n")
            fh.write(f"touched-models={json.dumps(r['touched'], separators=(',', ':'))}\n")
            fh.write(f"found={'true' if r['count'] else 'false'}\n")
            fh.write(f"count={r['count']}\n")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write("### Assembled xclbins\n\n")
                fh.write(f"- Complete models: **{len(r['complete'])}**\n")
                fh.write(f"- Partial models: **{len(r['partial'])}**"
                         f"{': ' + ', '.join(r['partial']) if r['partial'] else ''}\n")
                fh.write(f"- Files: {r['count']}\n")
    if args.fail_on_partial and r["partial"]:
        print(f"::error::{len(r['partial'])} models assembled with missing kernels")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
