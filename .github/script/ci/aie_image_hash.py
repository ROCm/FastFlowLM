#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""The toolchain hash: the tag of the dev kit's kernel build image.

    aie_image_hash.py [<toolchain.lock> <Dockerfile>]

Prints the first 16 hex digits of the sha256 of the two files' normalised
text, flm-dev-kit/FLM_Xclbin/toolchain.lock then docker/aie-build.Dockerfile
by default (paths relative to the current directory, the repository root in
CI). devkit-aie-image.yml tags ghcr.io/rocm/fastflowlm/aie-build
:<toolchain hash> with it, and devkit-xclbins.yml looks that tag up, so both
must compute it here.

Normalised: line endings made "\\n", trailing whitespace stripped, and blank
lines and comment lines (first non-blank character "#") dropped, so an edit
to the comments of either file keeps the tag and does not rebuild and
republish an identical image. Docker ignores those lines, and every
toolchain.lock line that is not KEY=VALUE is one of them. The exception is a
Dockerfile's parser directives (# syntax=, # escape=, # check=) at the top of
the file: they are comment-shaped but change the build, so they are kept.
Any other change, a value, an instruction, a directive, changes the hash.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

LOCK = "flm-dev-kit/FLM_Xclbin/toolchain.lock"
DOCKERFILE = "docker/aie-build.Dockerfile"
HASH_LEN = 16

# BuildKit reads a directive only before the first comment, blank line or
# instruction; the names are case-insensitive, spaces around "=" allowed.
# Blank lines among them are skipped here, not treated as the end: keeping a
# line Docker reads as a comment costs at most one extra rebuild.
DIRECTIVE = re.compile(r"^#\s*(syntax|escape|check)\s*=", re.IGNORECASE)


def normalise(text: str, directives: bool = False) -> str:
    """<text> without blank and comment lines, each kept line stripped at its
    end and ended by "\\n". With <directives>, the parser directives at the top
    are kept."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    in_header = directives
    for line in lines:
        line = line.rstrip()
        stripped = line.lstrip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if in_header and DIRECTIVE.match(stripped):
                out.append(stripped)
                continue
            in_header = False
            continue
        in_header = False
        out.append(line)
    return "".join(f"{line}\n" for line in out)


def toolchain_hash(lock_text: str, dockerfile_text: str) -> str:
    """The first HASH_LEN hex digits of sha256(normalised lock + Dockerfile)."""
    data = normalise(lock_text) + normalise(dockerfile_text, directives=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:HASH_LEN]


def read(path: str) -> str:
    return Path(path).read_bytes().decode("utf-8-sig")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lock", nargs="?", default=LOCK, help=f"default {LOCK}")
    ap.add_argument("dockerfile", nargs="?", default=DOCKERFILE, help=f"default {DOCKERFILE}")
    args = ap.parse_args(argv)
    try:
        print(toolchain_hash(read(args.lock), read(args.dockerfile)))
    except (OSError, UnicodeDecodeError) as e:
        print(f"aie_image_hash.py: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
