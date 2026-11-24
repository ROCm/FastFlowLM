# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for check_internal_refs.py: python3 -m unittest discover -s ci -p 'test_*.py' -t ci"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_internal_refs as cir  # noqa: E402

# Built from parts so that a grep of this file finds no whole example.
REPO = "FastFlowLM" + "_IRON"


def names(text: str) -> list[str]:
    return [name for _, name, _ in cir.scan_text(text)]


class ScanTextTest(unittest.TestCase):
    def test_internal_forms_hit(self):
        cases = {
            f"imported from {REPO}": "internal repository",
            "Ported from the internal FLM" + "_IRON CI.": "internal repository",
            "github.com/AMD-ROCm-" + "Internal/x": "internal repository",
            "SCL-" + "5105 holds it": "internal ticket",
            "(As IRON" + "#148.)": "internal PR or issue",
            "IRON" + " PR #137": "internal PR or issue",
            "kept held in IRON" + " 0123abcd9": "internal commit or branch",
            "it is in git at IRON" + " `2261dffc2`.": "internal commit or branch",
            "rebuilt from IRON" + " HEAD": "internal commit or branch",
            "built by IRON" + " CI": "internal commit or branch",
            "the IRON" + " tree's quant_size.hpp": "internal commit or branch",
            "the version IRON's" + " wheel-aie-env action pinned": "internal CI or tooling",
            "the same sample as IRON's" + " build_dataset.py": "internal CI or tooling",
            "as IRON's" + " merge-pr.yml did": "internal CI or tooling",
            "(IRON's" + " sampling)": "internal CI or tooling",
            "GSM8K alone took " + "IRON ~1050 runner-minutes": "internal CI or tooling",
            "Fixed in the IRON" + ", with the history": "internal CI or tooling",
            "is still in the IRON" + ". But": "internal CI or tooling",
            "(Phase 2b" + " exclusion candidate)": "internal process",
            "is pending Legal" + " review": "internal process",
            "ended in bug" + "check 0x0": "crash details",
            "(VIDEO_SCHEDULER_" + "INTERNAL_ERROR)": "crash details",
            "cp /scratch/" + "someone/models .": "personal path",
        }
        for text, name in cases.items():
            with self.subTest(text=text):
                self.assertIn(name, names(text))

    def test_public_iron_and_generic_paths_pass(self):
        for text in (
            "the IRON Python design in iron/attn_aie.py",
            "IRON's task API and IRON's allocator",
            "RTP addresses are pinned in the IRON designs",
            "Fixed in the IRON design (`iron/gate.py`), with the history",
            "IRON's object FIFOs; the IRON `runtime_sequence`; the IRON `init=` value",
            "IRON → MLIR → aiecc.py",
            "IRON CT[r][c] = (col c, row r+2)",
            "built on [IRON](https://github.com/amd/iron)",
            "export FLM_MODEL_PATH=\"/scratch/$USER\"",
            "REFERENCE ?= /scratch/$(USER)/ref.safetensors",
            "prefix ${USER}: /scratch/${USER}/flm_exe_core",
            "Legal notices are in THIRD_PARTY_NOTICES.txt",
        ):
            with self.subTest(text=text):
                self.assertEqual(names(text), [])

    def test_line_numbers(self):
        hits = cir.scan_text("ok\nfrom " + REPO + "\nok\n")
        self.assertEqual([(n, name) for n, name, _ in hits], [(2, "internal repository")])

    def test_allowed_paths_exist(self):
        root = HERE.parents[2]
        for rel, reason in cir.ALLOWED.items():
            with self.subTest(path=rel):
                self.assertTrue((root / rel).is_file())
                self.assertTrue(reason)


class ScanRepoTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)

    def tearDown(self):
        self._tmp.cleanup()

    def add(self, rel: str, data: bytes):
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        subprocess.run(["git", "-C", str(self.repo), "add", rel], check=True)

    def test_hits_binary_allowed_and_untracked(self):
        self.add("docs/a.md", f"line\nsee {REPO}\n".encode())
        self.add("img.bin", b"\0\1" + REPO.encode())
        self.add(".github/script/ci/check_internal_refs.py", REPO.encode())
        (self.repo / "untracked.md").write_text(REPO)
        errors, scanned = cir.scan_repo(self.repo)
        self.assertEqual(errors, [f"docs/a.md:2: internal repository: {REPO}"])
        self.assertEqual(scanned, 1)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cir.main(["--repo", str(self.repo)]), 1)
        self.assertIn("check_internal_refs: error: docs/a.md:2:", out.getvalue())

    def test_clean_repo_passes(self):
        self.add("README.md", b"IRON Python kernels, /scratch/$USER/x\n")
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cir.main(["--repo", str(self.repo)]), 0)
        self.assertIn("ok (1 text files)", out.getvalue())

    def test_not_a_repo(self):
        with tempfile.TemporaryDirectory() as plain:
            errors, scanned = cir.scan_repo(Path(plain) / "missing")
        self.assertEqual(scanned, 0)
        self.assertEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
