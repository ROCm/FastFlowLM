#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""The RTP address lock of the dev kit's kernels (flm-dev-kit/FLM_Xclbin).

    devkit_rtp_lock.py generate <shards dir> -o <lock> [--update] [--note TEXT]...
                               [--assignment <kernel_assignment.json>]
    devkit_rtp_lock.py check <shards dir> [--lock <lock>] [--github-output]

Some kernel builds write the addresses of their runtime parameters (RTPs)
next to the xclbin: build/[<MODEL_TYPE>/]xclbins/*rtp_addresses*.txt, one
"<name> <address>" per line. The engines (flm-dev-kit/FLM_DLL/detail/<engine>,
and FastFlowLM's src/detail/<engine>) do not read that file: they hard-code
the addresses (gemma4e_npu_sequence.cpp's e2b_rtp_addresses, for one). A
kernel change that moves an RTP therefore breaks its engine with no error:
the engine writes the parameter where it no longer is, and the model hangs or
outputs garbage. flm-dev-kit/FLM_Xclbin/rtp_addresses.lock.txt records every
such file CI builds of a kernel some model uses (kernel_assignment.json), so
that a moved RTP fails CI instead. Builds no model uses are left out: no
engine loads them, so their addresses bind nothing (a stanza for one would
look protected while no engine rule applies to it).

<shards dir> holds the builds one folder per artifact, named
xclbins-<unit, / as _>[_<MODEL_TYPE>], at any depth: the
devkit-xclbins-shard-<i> artifacts of a run downloaded side by side
(gh run download <run> -R ROCm/FastFlowLM -p 'devkit-xclbins-shard-*' -D <dir>).

Lock format: "#" comment lines (a header), then one stanza per RTP file,
sorted by artifact then file name:

    [<artifact>/<file name>]
    <name> <address>
    ...

Lines are normalized: blank lines dropped, runs of whitespace made one space,
the lines sorted (an engine reads names, not their order). A blank line ends
a stanza; inside one, every line must be "<name> <address>" (comments go in
the header only).

generate  writes the lock from the RTP files under <shards dir> of the
          artifacts some kernel_assignment.json model uses. --update keeps
          the stanzas of the lock already at -o whose artifact was not built
          in <shards dir> (an artifact that was built replaces all of its
          stanzas, and loses them when it wrote no RTP file) and the notes of
          its header. --note adds a line to the header (the run it came from,
          artifacts it lacked); with --update, after the kept notes.
check     compares every artifact built under <shards dir> that a model uses
          with the lock: an RTP file whose content differs, one the lock does
          not have, or a lock stanza of a built artifact that wrote no such
          file fails (exit 1), naming the models that use the kernel
          (kernel_assignment.json) and the engines that serve them
          (flm-dev-kit/FLM_DLL/engines.json). Artifacts not built, or used by
          no model, are not checked. With --github-output, writes a section
          to $GITHUB_STEP_SUMMARY when that is set.

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

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
LOCK_REL = "flm-dev-kit/FLM_Xclbin/rtp_addresses.lock.txt"
LOCK = REPO_ROOT / LOCK_REL
ASSIGNMENT = REPO_ROOT / "flm-dev-kit/FLM_Xclbin/kernel_assignment.json"
ENGINES = REPO_ROOT / "flm-dev-kit/FLM_DLL/engines.json"
DETAIL = "flm-dev-kit/FLM_DLL/detail"
SCRIPT = ".github/script/ci/devkit_rtp_lock.py"

RTP_FILE = re.compile(r".*rtp_addresses.*\.txt")
STANZA = re.compile(r"\[(xclbins-[A-Za-z0-9_.+-]+)/([A-Za-z0-9_.+-]+)\]")

HEADER = f"""\
Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
Portions of this file consist of AI-generated content.

RTP addresses of the dev-kit kernels, as CI builds them: one stanza per
*rtp_addresses*.txt written by a build of a kernel some
kernel_assignment.json model uses, "[<artifact>/<file>]" and its sorted
"<name> <address>" lines (no comments inside a stanza; a blank line ends
one). The engines hard-code these addresses, so CI (devkit-xclbins,
{SCRIPT} check) fails when a build's file differs
from its stanza: a moved RTP needs the engines' constants
(FLM_DLL/detail/<engine>/, and FastFlowLM's src/detail/<engine>/) and this
file updated together, and devkit_check.py fails a pull request that changes
a stanza without touching every engine serving a model that uses it.
Regenerate from a run's shards (keeping the artifacts it did not build):
  gh run download <run id> -R ROCm/FastFlowLM -p 'devkit-xclbins-shard-*' -D <dir>
  python3 {SCRIPT} generate <dir> -o {LOCK_REL} --update"""

Lock = dict  # (artifact, file name) -> tuple of normalized lines


def normalize(text: str) -> tuple[str, ...]:
    return tuple(sorted(" ".join(l.split()) for l in text.splitlines() if l.strip()))


def find_builds(root: Path) -> dict[str, Path]:
    """Artifact name -> its folder: every directory named xclbins-* under root.
    Two folders with the same name are an error (two shards built it)."""
    out: dict[str, Path] = {}
    for d in sorted(p for p in root.rglob("xclbins-*") if p.is_dir()):
        if d.name in out:
            raise ValueError(f"{d.name} appears twice: {out[d.name]} and {d}")
        out[d.name] = d
    return out


def rtp_files(builds: dict[str, Path]) -> Lock:
    """(artifact, file name) -> normalized lines of every RTP file built."""
    out: Lock = {}
    for name, folder in builds.items():
        for p in sorted(folder.iterdir()):
            if p.is_file() and RTP_FILE.fullmatch(p.name):
                out[(name, p.name)] = normalize(p.read_text(encoding="utf-8", errors="replace"))
    return out


def parse_lock(text: str) -> tuple[Lock, list[str]]:
    """(entries, header lines without '# ') of a lock file; ValueError when malformed."""
    entries: Lock = {}
    header: list[str] = []
    key = None
    lines: list[str] = []
    for n, raw in enumerate(text.splitlines() + [""], 1):
        line = raw.strip()
        if key is None:
            if not line:
                continue
            if line.startswith("#"):
                if not entries:
                    header.append(raw[2:] if raw.startswith("# ") else raw[1:])
                continue
            m = STANZA.fullmatch(line)
            if not m:
                raise ValueError(f"line {n}: expected [<artifact>/<file>], found {line!r}")
            key = (m.group(1), m.group(2))
            if key in entries:
                raise ValueError(f"line {n}: {key[0]}/{key[1]} appears twice")
            lines = []
        elif line:
            if line.startswith("#") or STANZA.fullmatch(line) or len(line.split()) != 2:
                raise ValueError(f"line {n}: expected '<name> <address>' in {key[0]}/{key[1]} (end a "
                                 f"stanza with a blank line; comments only in the header), found {line!r}")
            lines.append(line)
        else:
            entries[key] = normalize("\n".join(lines))
            key = None
    return entries, header


def header_notes(header: list[str]) -> list[str]:
    """The note lines of a parsed header: those after HEADER's last line (or,
    when a lock predates the current HEADER wording, after as many lines)."""
    last = HEADER.splitlines()[-1]
    if last in header:
        return header[len(header) - header[::-1].index(last):]
    return header[len(HEADER.splitlines()):]


def render_lock(entries: Lock, notes: list[str] = ()) -> str:
    out = [f"# {l}" if l else "#" for l in HEADER.splitlines()]
    for note in dict.fromkeys(notes):
        out.append(f"# {note}" if note else "#")
    for (artifact, name) in sorted(entries):
        out += ["", f"[{artifact}/{name}]", *entries[(artifact, name)]]
    return "\n".join(out) + "\n"


def changed_artifacts(base_text: Optional[str], head_text: Optional[str]) -> list[str]:
    """The artifacts whose stanzas differ between two lock texts: added,
    changed or removed. A None base (no lock there, or unknown) gives every
    artifact of the head; an unreadable lock counts as empty (devkit_check.py
    reports it)."""
    def entries(text: Optional[str]) -> Lock:
        try:
            return parse_lock(text or "")[0]
        except ValueError:
            return {}
    head = entries(head_text)
    if base_text is None:
        return sorted({a for a, _ in head})
    base = entries(base_text)
    return sorted({k[0] for k in set(base) | set(head) if base.get(k) != head.get(k)})


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def assigned(assignment: dict) -> set[str]:
    """The artifacts some kernel_assignment.json model uses."""
    return {a for e in (assignment.get("models") or {}).values()
            for a in (e.get("kernels") or {}).values()}


def users(artifact: str, assignment: dict, engines: dict) -> tuple[list[str], list[str]]:
    """(models using artifact, engine dirs serving them; or what runs them)."""
    models = sorted(m for m, e in (assignment.get("models") or {}).items()
                    if artifact in (e.get("kernels") or {}).values())
    serving = set()
    for engine, e in (engines.get("engines") or {}).items():
        if set(e.get("models") or []) & set(models):
            serving.add(f"{DETAIL}/{engine}/")
    for m in models:
        other = (engines.get("served_elsewhere") or {}).get(m)
        if other:
            serving.add(f"{m}: {other}")
    return models, sorted(serving)


def diff_lines(old: tuple, new: tuple, limit: int = 8) -> list[str]:
    gone, added = sorted(set(old) - set(new)), sorted(set(new) - set(old))
    out = [f"- {l}" for l in gone[:limit]] + [f"+ {l}" for l in added[:limit]]
    more = max(0, len(gone) - limit) + max(0, len(added) - limit)
    if more:
        out.append(f"  ... {more} more")
    return out


def check(built: dict[str, Path], found: Lock, lock: Lock, assignment: dict,
          engines: dict) -> list[dict]:
    """One problem per RTP file of a built artifact some model uses that
    disagrees with the lock."""
    problems = []
    names = set(built) & assigned(assignment)
    found = {k: v for k, v in found.items() if k[0] in names}
    for key in sorted(set(found) | {k for k in lock if k[0] in names}):
        artifact, name = key
        if key in found and key in lock:
            if found[key] == lock[key]:
                continue
            what, detail = "differs from the lock", diff_lines(lock[key], found[key])
        elif key in found:
            what, detail = "is not in the lock", [f"+ {l}" for l in found[key][:8]]
        else:
            what, detail = "is in the lock, but the build wrote no such file", []
        models, serving = users(artifact, assignment, engines)
        problems.append({"artifact": artifact, "file": name, "what": what, "detail": detail,
                         "models": models, "engines": serving})
    return problems


def cmd_generate(args) -> int:
    try:
        builds = find_builds(args.shards)
    except ValueError as e:
        print(f"::error::{e}")
        return 1
    if not builds:
        print(f"::error::no xclbins-* build folders under {args.shards}")
        return 1
    used = assigned(load_json(args.assignment))
    if not used:
        print(f"::error::{args.assignment}: no model uses any artifact (unreadable?)")
        return 1
    found = rtp_files(builds)
    entries = {k: v for k, v in found.items() if k[0] in used}
    unused = sorted({k[0] for k in found} - used)
    if unused:
        print(f"left out {len(unused)} artifacts no model uses: {', '.join(unused)}")
    notes = []
    if args.update and args.output.exists():
        try:
            old, old_header = parse_lock(args.output.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"::error::{args.output}: {e}")
            return 1
        notes = header_notes(old_header)
        kept = {k: v for k, v in old.items() if k[0] not in builds and k[0] in used}
        entries.update(kept)
        print(f"kept {len(kept)} stanzas of artifacts not built under {args.shards}")
    args.output.write_text(render_lock(entries, notes + (args.note or [])), encoding="utf-8")
    print(f"{args.output}: {len(entries)} RTP files of {len({k[0] for k in entries})} artifacts "
          f"({len(builds)} artifacts built)")
    return 0


def cmd_check(args) -> int:
    try:
        lock, _ = parse_lock(args.lock.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"::error::{args.lock}: {e}")
        return 1
    try:
        builds = find_builds(args.shards) if args.shards.is_dir() else {}
    except (OSError, ValueError) as e:
        print(f"::error::{args.shards}: {e}")
        return 1
    found = rtp_files(builds)
    assignment = load_json(args.assignment)
    used = assigned(assignment)
    problems = check(builds, found, lock, assignment, load_json(args.engines))
    checked = {k for k in set(found) | set(lock) if k[0] in builds and k[0] in used}
    unused = sorted({k[0] for k in found} - used)
    print(f"{len(builds)} artifacts built, {len(checked)} RTP files checked against {args.lock}")
    if unused:
        print(f"not checked, as no model uses them: {', '.join(unused)}")
    for p in problems:
        print(f"::error title=RTP addresses changed::{p['artifact']}/{p['file']} {p['what']}. "
              f"Models using it: {', '.join(p['models']) or 'none'}. Engines serving them: "
              f"{', '.join(p['engines']) or 'none'}. Update the engines' hard-coded RTP addresses "
              f"and {LOCK_REL} together.")
        for l in p["detail"]:
            print(f"    {l}")
    if problems:
        print(f"\n{len(problems)} RTP file(s) disagree with {LOCK_REL}. The engines hard-code these "
              "addresses: update each engine's RTP constants (and FastFlowLM's src/detail/<engine>/ "
              "copy) together with the lock. To regenerate the lock from this run:\n"
              "  gh run download <run id> -R ROCm/FastFlowLM -p 'devkit-xclbins-shard-*' -D <dir>\n"
              f"  python3 {SCRIPT} generate <dir> -o {LOCK_REL} --update")
    else:
        print("RTP addresses match the lock")
    summary = os.environ.get("GITHUB_STEP_SUMMARY") if args.github_output else None
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("### RTP addresses\n\n")
            if not problems:
                fh.write(f"{len(checked)} RTP files of the built kernels match `{LOCK_REL}`.\n\n")
            else:
                fh.write(f"**{len(problems)} RTP file(s) disagree with `{LOCK_REL}`.** Update the "
                         "engines' hard-coded RTP addresses and the lock together.\n\n")
                fh.write("| Artifact | File | Problem | Models | Engines |\n|---|---|---|---|---|\n")
                for p in problems:
                    fh.write(f"| `{p['artifact']}` | `{p['file']}` | {p['what']} | "
                             f"{', '.join(p['models']) or '—'} | "
                             f"{', '.join(f'`{e}`' for e in p['engines']) or '—'} |\n")
                fh.write("\n")
    return 1 if problems else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="Write the lock from a run's shards.")
    g.add_argument("shards", type=Path)
    g.add_argument("-o", "--output", type=Path, default=LOCK)
    g.add_argument("--update", action="store_true",
                   help="Keep the existing lock's stanzas of artifacts not built under <shards>.")
    g.add_argument("--note", action="append",
                   help="A line for the header (repeatable; with --update, after the kept notes).")
    g.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    c = sub.add_parser("check", help="Compare a run's shards with the lock.")
    c.add_argument("shards", type=Path)
    c.add_argument("--lock", type=Path, default=LOCK)
    c.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    c.add_argument("--engines", type=Path, default=ENGINES)
    c.add_argument("--github-output", action="store_true", help="Also write $GITHUB_STEP_SUMMARY.")
    args = parser.parse_args(argv)
    return cmd_generate(args) if args.cmd == "generate" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
