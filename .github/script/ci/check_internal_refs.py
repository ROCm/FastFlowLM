#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Fail when a tracked file names internal repositories or process details.

    check_internal_refs.py [--repo <root>]

The dev kit (flm-dev-kit/) came from a private development tree. Tracked files
must not name that tree, its pull requests, commits, branches or CI, internal
ticket IDs, personal paths, or NPU crash details (keep the engineering warning,
phrased generically). Every text file git tracks is scanned against PATTERNS;
binary files (a NUL byte in the first 8 KB) are skipped, as are the paths in
ALLOWED (each with its reason). "IRON" alone is not matched: it is also the
public mlir-aie Python API ("IRON Python", "IRON's object FIFOs"), so only the
forms that mean the repository are. Prints "check_internal_refs: error:
<path>:<line>: <pattern>: <text>" per hit and exits 1 when there is one.

Standard library and git only. Run by build-and-test.yml's changes job on
every event; its tests are test_check_internal_refs.py.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]

# name -> pattern. Narrow on purpose: a hit should always be worth fixing.
PATTERNS = {
    "internal repository": re.compile(r"FastFlowLM_IRON|\bFLM_IRON\b|AMD-ROCm-Internal", re.IGNORECASE),
    "internal ticket": re.compile(r"\bSCL-\d+"),
    "internal PR or issue": re.compile(r"\bIRON\s*#\d+|\bIRON PR\b"),
    "internal commit or branch": re.compile(r"\bIRON\s+`?[0-9a-f]{7,40}\b|\bIRON (?:HEAD|CI)\b|\bIRON tree\b"),
    "internal CI or tooling": re.compile(r"\bIRON's (?:CI|local|wheel-aie-env|ironenv|bench-models|walk-up|sampling)\b"
                                         r"|\bIRON's [\w./-]+\.(?:py|ya?ml|json|sh)\b|\btook IRON\b"
                                         r"|\b(?:in|from|of) the IRON[,.;:]"),
    "internal process": re.compile(r"\bPhase 2b exclusion\b|\b(?:pending|for an?) Legal (?:review|decision)\b"),
    "crash details": re.compile(r"\bbugcheck\b|VIDEO_SCHEDULER_INTERNAL_ERROR", re.IGNORECASE),
    # /scratch/$USER, $(USER) and ${USER} are generic; a literal name is not.
    "personal path": re.compile(r"/scratch/(?!\$\(USER\)|\$\{?USER\b)[A-Za-z0-9_.-]+"),
}

# path -> why it may contain the patterns.
ALLOWED = {
    ".github/script/ci/check_internal_refs.py": "defines the patterns",
}

BINARY_PROBE = 8192


def scan_text(text: str) -> list[tuple[int, str, str]]:
    """(line number, pattern name, matched text) for every hit in text."""
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        for name, pattern in PATTERNS.items():
            match = pattern.search(line)
            if match:
                hits.append((number, name, match.group(0)))
    return hits


def tracked_files(repo: Path) -> Optional[list[str]]:
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], check=True,
                             capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def scan_repo(repo: Path) -> tuple[list[str], int]:
    """(errors, files scanned) for the files git tracks in repo."""
    files = tracked_files(repo)
    if files is None:
        return [f"{repo}: git ls-files failed"], 0
    errors = []
    scanned = 0
    for rel in files:
        if rel in ALLOWED:
            continue
        try:
            data = (repo / rel).read_bytes()
        except OSError:
            continue  # deleted in the work tree, or a submodule
        if b"\0" in data[:BINARY_PROBE]:
            continue
        scanned += 1
        for number, name, found in scan_text(data.decode("utf-8", "replace")):
            errors.append(f"{rel}:{number}: {name}: {found}")
    return errors, scanned


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    errors, scanned = scan_repo(args.repo)
    for err in errors:
        print(f"check_internal_refs: error: {err}")
    if errors:
        print(f"check_internal_refs: {len(errors)} hit(s); reword them (see the docstring)", file=sys.stderr)
        return 1
    print(f"check_internal_refs: ok ({scanned} text files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
