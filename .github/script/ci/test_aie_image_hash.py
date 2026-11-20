# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for aie_image_hash.py: python3 -m unittest discover -s ci -p 'test_*.py' -t ci"""

from __future__ import annotations

import io
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aie_image_hash as ah  # noqa: E402

REPO = HERE.parents[2]

LOCK = """\
# Copyright header
#
# The toolchain, pinned.

# Base image.
UBUNTU_IMAGE=ubuntu:24.04@sha256:00
PEANO_VERSION=1.2.3
"""

DOCKERFILE = """\
# syntax=docker/dockerfile:1
# Copyright header
#
# What the image is.
ARG UBUNTU_IMAGE
FROM ${UBUNTU_IMAGE} AS build
ARG PEANO_VERSION
# Peano, by version.
RUN pip install llvm-aie==${PEANO_VERSION} \\
    && echo done
"""


def h(lock: str = LOCK, dockerfile: str = DOCKERFILE) -> str:
    return ah.toolchain_hash(lock, dockerfile)


class HashFormat(unittest.TestCase):
    def test_sixteen_hex(self):
        self.assertRegex(h(), r"^[0-9a-f]{16}$")

    def test_current_files(self):
        lock, dockerfile = REPO / ah.LOCK, REPO / ah.DOCKERFILE
        if not (lock.is_file() and dockerfile.is_file()):
            self.skipTest("not in a FastFlowLM checkout")
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(ah.main([str(lock), str(dockerfile)]), 0)
        self.assertRegex(out.getvalue(), r"^[0-9a-f]{16}\n$")


class SameHash(unittest.TestCase):
    def test_comment_edit(self):
        self.assertEqual(h(LOCK.replace("pinned.", "pinned, with care."),
                           DOCKERFILE.replace("# Peano, by version.", "# Peano, the AIE compiler.")), h())

    def test_comment_added_and_removed(self):
        self.assertEqual(h(LOCK + "# A trailing note.\n",
                           DOCKERFILE.replace("# Peano, by version.\n", "")), h())

    def test_indented_comment(self):
        self.assertEqual(h(dockerfile=DOCKERFILE.replace("ARG PEANO_VERSION\n",
                                                         "ARG PEANO_VERSION\n    # note\n")), h())

    def test_blank_lines(self):
        self.assertEqual(h(LOCK.replace("\n", "\n\n"), "\n" + DOCKERFILE + "\n   \n"), h())

    def test_trailing_whitespace(self):
        self.assertEqual(h(LOCK.replace("1.2.3\n", "1.2.3  \t\n"),
                           DOCKERFILE.replace("AS build\n", "AS build \n")), h())

    def test_line_endings(self):
        self.assertEqual(h(LOCK.replace("\n", "\r\n"), DOCKERFILE.replace("\n", "\r\n")), h())
        self.assertEqual(h(LOCK.replace("\n", "\r"), DOCKERFILE), h())

    def test_no_final_newline(self):
        self.assertEqual(h(LOCK.rstrip("\n"), DOCKERFILE.rstrip("\n")), h())

    def test_blank_line_between_directives(self):
        base = "# syntax=docker/dockerfile:1\n# check=error=true\nFROM x\n"
        self.assertEqual(h(dockerfile=base.replace("\n# check", "\n\n# check")), h(dockerfile=base))


class DifferentHash(unittest.TestCase):
    def test_value_change(self):
        self.assertNotEqual(h(LOCK.replace("1.2.3", "1.2.4")), h())

    def test_key_added(self):
        self.assertNotEqual(h(LOCK + "XRT_VERSION=2\n"), h())

    def test_instruction_change(self):
        self.assertNotEqual(h(dockerfile=DOCKERFILE.replace("echo done", "echo ok")), h())

    def test_indentation_inside_instruction(self):
        self.assertNotEqual(h(dockerfile=DOCKERFILE.replace("    && echo", "  && echo")), h())

    def test_directive_change(self):
        self.assertNotEqual(h(dockerfile=DOCKERFILE.replace("dockerfile:1", "dockerfile:1.7")), h())

    def test_directive_added_or_removed(self):
        self.assertNotEqual(h(dockerfile="# escape=`\n" + DOCKERFILE), h())
        self.assertNotEqual(h(dockerfile="# check=error=true\n" + DOCKERFILE), h())
        self.assertNotEqual(h(dockerfile=DOCKERFILE.replace("# syntax=docker/dockerfile:1\n", "")), h())

    def test_directive_spelling(self):
        self.assertNotEqual(h(dockerfile=DOCKERFILE.replace("# syntax=", "#  SYNTAX = ")), h())

    def test_lock_and_dockerfile_kept_apart(self):
        # The same line moved from one file to the other is a different input.
        self.assertNotEqual(h(LOCK + "ARG X\n", DOCKERFILE), h(LOCK, "ARG X\n" + DOCKERFILE))


class Directives(unittest.TestCase):
    def test_only_at_the_top(self):
        # After a comment or an instruction, "# syntax=" is a comment.
        late = DOCKERFILE.replace("# Peano, by version.", "# syntax=docker/dockerfile:1.7")
        self.assertEqual(h(dockerfile=late), h())
        after_comment = "# A note\n# syntax=docker/dockerfile:1\nFROM x\n"
        self.assertEqual(h(dockerfile=after_comment), h(dockerfile="FROM x\n"))

    def test_kept_in_dockerfile_only(self):
        # toolchain.lock has no directives: "# syntax=" there is a comment.
        self.assertEqual(h("# syntax=docker/dockerfile:1\n" + LOCK), h())

    def test_normalise(self):
        self.assertEqual(ah.normalise(DOCKERFILE, directives=True).splitlines()[:2],
                         ["# syntax=docker/dockerfile:1", "ARG UBUNTU_IMAGE"])
        self.assertNotIn("#", ah.normalise(LOCK))


class Cli(unittest.TestCase):
    def test_files(self):
        with tempfile.TemporaryDirectory() as d:
            lock, dockerfile = Path(d) / "toolchain.lock", Path(d) / "Dockerfile"
            lock.write_bytes(("\ufeff" + LOCK.replace("\n", "\r\n")).encode("utf-8"))
            dockerfile.write_text(DOCKERFILE, encoding="utf-8", newline="\n")
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(ah.main([str(lock), str(dockerfile)]), 0)
            self.assertEqual(out.getvalue(), h() + "\n")

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertEqual(ah.main([str(Path(d) / "nope"), str(Path(d) / "nope2")]), 1)
            self.assertTrue(re.search(r"nope", err.getvalue()))


if __name__ == "__main__":
    unittest.main()
