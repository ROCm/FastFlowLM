#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Fail when a model loader sets a repetition penalty default on the wrong scale.

    check_sampler_defaults.py [--repo <root>]

The sampler (src/common/modules/sampler.cpp) applies three penalties to the
last repeat_last_n tokens. rep_penalty scales a seen token's logit down (a
positive logit is divided by it, a negative one multiplied), so 1.0 is neutral. freq_penalty and pre_penalty are subtracted from it (count * freq +
pre), so 0 is neutral and 1.0 is a large penalty. The dev-kit sampler under
flm-dev-kit/ uses a different formula in which 1.0 is neutral; values copied
from it, or chosen as if every penalty were multiplicative, left loaders
subtracting 1.0 or more per repeat and broke tool calls and copied text.

Every `freq_penalty = <n>`, `pre_penalty = <n>` and `rep_penalty = <n>`
assignment in src/ (tests excepted) is checked:
  - freq_penalty / pre_penalty below 0 is an error;
  - freq_penalty / pre_penalty above 0 is an error unless the same line has a
    // comment giving its source (for example the model card's
    presence_penalty), so a reviewer can tell a chosen value from a scale slip;
  - rep_penalty below 1.0 is an error (it would reward repeats).
Prints "check_sampler_defaults: error: <path>:<line>: <why>: <text>" per hit
and exits 1 when there is one.

Standard library and git only. Run by build-and-test.yml's changes job on
every event; its tests are test_check_sampler_defaults.py.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]

ASSIGN = re.compile(r"\b(freq_penalty|pre_penalty|rep_penalty)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+))f?\s*;(.*)$")
SOURCE_DIR = "src/"
SKIP_DIRS = ("src/test/",)
SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx")


def scan_text(text: str) -> list[tuple[int, str, str]]:
    """(line number, why, assignment) for every bad default in text."""
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        match = ASSIGN.search(line)
        if not match or "//" in line[:match.start()]:
            continue  # no assignment, or one inside a comment
        name, value, rest = match.group(1), float(match.group(2)), match.group(3)
        found = match.group(0)[:match.end(2) - match.start(0)]
        if name == "rep_penalty":
            if value < 1.0:
                hits.append((number, "rep_penalty scales seen tokens down, 1.0 is neutral; below 1.0 rewards repeats", found))
        elif value < 0.0:
            hits.append((number, f"{name} is subtracted, 0 is neutral; below 0 rewards repeats", found))
        elif value > 0.0 and "//" not in rest:
            hits.append((number, f"{name} is subtracted, 0 is neutral; a non-zero default needs a "
                                 "same-line // comment giving its source", found))
    return hits


def tracked_files(repo: Path) -> Optional[list[str]]:
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "--", SOURCE_DIR], check=True,
                             capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def scan_repo(repo: Path) -> tuple[list[str], int]:
    """(errors, files scanned) for the C/C++ files git tracks under src/ in repo."""
    files = tracked_files(repo)
    if files is None:
        return [f"{repo}: git ls-files failed"], 0
    errors = []
    scanned = 0
    for rel in files:
        if not rel.endswith(SUFFIXES) or rel.startswith(SKIP_DIRS):
            continue
        try:
            text = (repo / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue  # deleted in the work tree, or a submodule
        scanned += 1
        for number, why, found in scan_text(text):
            errors.append(f"{rel}:{number}: {why}: {found}")
    return errors, scanned


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    errors, scanned = scan_repo(args.repo)
    for err in errors:
        print(f"check_sampler_defaults: error: {err}")
    if errors:
        print(f"check_sampler_defaults: {len(errors)} hit(s); see the docstring", file=sys.stderr)
        return 1
    print(f"check_sampler_defaults: ok ({scanned} source files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
