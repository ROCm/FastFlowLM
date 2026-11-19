# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for devkit_assemble_xclbins.py: python3 -m unittest discover -s ci -p 'test_*.py' -t ci"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import devkit_assemble_xclbins as da  # noqa: E402

ASSIGNMENT = {"models": {
    "m:1b": {"folder": "M-1B", "kernels": {"dequant.xclbin": "xclbins-Dequant",
                                            "layer.xclbin": "xclbins-Dec_A"}},
    "m:3b": {"folder": "M-3B", "kernels": {"dequant.xclbin": "xclbins-Dequant",
                                            "layer.xclbin": "xclbins-Dec_B",
                                            "vision_mm_a.xclbin": "xclbins-Two",
                                            "vision_mm_b.xclbin": "xclbins-Two"}},
    "m:9b": {"folder": "M-9B", "kernels": {"layer.xclbin": "xclbins-Dec_C"}},
    "m:amb": {"folder": "M-AMB", "kernels": {"mm.xclbin": "xclbins-Two"}},
}}


class Assemble(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.art = self.root / "artifacts"
        # Two shards, each with per-build folders.
        self.put("devkit-xclbins-shard-0/xclbins-Dequant/dequant.xclbin", "D")
        self.put("devkit-xclbins-shard-0/xclbins-Dec_A/layer.xclbin", "A")
        self.put("devkit-xclbins-shard-0/xclbins-Dec_A/rtp_addresses.txt", "rtp")
        self.put("devkit-xclbins-shard-1/xclbins-Two/vision_mm_a.xclbin", "2a")
        self.put("devkit-xclbins-shard-1/xclbins-Two/vision_mm_b.xclbin", "2b")
        self.put("devkit-xclbins-shard-1/xclbins-Dec_C/rtp_addresses.txt", "only rtp")
        self.put("devkit-xclbins-shard-1/xclbins-Unused/x.xclbin", "u")

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, rel, text):
        p = self.art / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def test_assemble(self):
        out = self.root / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            r = da.assemble(ASSIGNMENT, da.find_artifacts(self.art), out)
        self.assertEqual(r["complete"], ["m:1b"])
        self.assertEqual(r["partial"], ["m:3b"])
        self.assertEqual(r["touched"], ["m:1b", "m:3b"])
        self.assertEqual(r["missing"]["m:3b"], ["layer.xclbin <- xclbins-Dec_B (not built)"])
        # Single-file artifact under a different name -> the slot's name.
        self.assertEqual((out / "M-1B/layer.xclbin").read_text(), "A")
        self.assertEqual((out / "M-3B/dequant.xclbin").read_text(), "D")
        # Exact basename wins in a multi-design artifact.
        self.assertEqual((out / "M-3B/vision_mm_a.xclbin").read_text(), "2a")
        self.assertEqual((out / "M-3B/vision_mm_b.xclbin").read_text(), "2b")
        # No xclbin built (m:9b), and ambiguous (m:amb): nothing for them.
        self.assertFalse((out / "M-9B").exists())
        self.assertFalse((out / "M-AMB").exists())
        self.assertEqual(sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()), [
            "M-1B/dequant.xclbin", "M-1B/layer.xclbin", "M-3B/dequant.xclbin",
            "M-3B/vision_mm_a.xclbin", "M-3B/vision_mm_b.xclbin"])

    def test_pick(self):
        p, why = da.pick(self.art / "devkit-xclbins-shard-1/xclbins-Two", "mm.xclbin")
        self.assertIsNone(p)
        self.assertIn("ambiguous", why)
        self.assertEqual(da.pick(self.art / "devkit-xclbins-shard-1/xclbins-Dec_C", "layer.xclbin"),
                         (None, "no .xclbin"))

    def test_duplicate_artifact_is_an_error(self):
        self.put("devkit-xclbins-shard-1/xclbins-Dequant/dequant.xclbin", "D2")
        with self.assertRaises(ValueError):
            da.find_artifacts(self.art)

    def test_same_folder_and_slot_must_be_one_artifact(self):
        out = self.root / "out"
        same = {"models": {"a": {"folder": "F", "kernels": {"dequant.xclbin": "xclbins-Dequant"}},
                           "b": {"folder": "F", "kernels": {"dequant.xclbin": "xclbins-Dequant"}}}}
        with contextlib.redirect_stdout(io.StringIO()):
            r = da.assemble(same, da.find_artifacts(self.art), out)
        self.assertEqual(r["complete"], ["a", "b"])
        clash = {"models": {"a": {"folder": "F", "kernels": {"layer.xclbin": "xclbins-Dec_A"}},
                            "b": {"folder": "F", "kernels": {"layer.xclbin": "xclbins-Dec_B"}}}}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "F/layer.xclbin"):
            da.assemble(clash, da.find_artifacts(self.art), out)

    def test_unsafe_folder_or_slot_is_never_written(self):
        out = self.root / "out"
        for folder, slot in ((str(self.root / "abs"), "dequant.xclbin"), ("../escape", "dequant.xclbin"),
                             ("ok/../../escape", "dequant.xclbin"), ("ok", "../dequant.xclbin"),
                             ("ok", "dequant.bin"), ("", "dequant.xclbin")):
            with self.subTest(folder=folder, slot=slot):
                bad = {"models": {"m": {"folder": folder, "kernels": {slot: "xclbins-Dequant"}}}}
                with contextlib.redirect_stdout(io.StringIO()):
                    r = da.assemble(bad, da.find_artifacts(self.art), out)
                self.assertEqual((r["touched"], r["count"]), ([], 0))
        # Beside a safe slot, the unsafe one is reported as missing.
        mixed = {"models": {"m": {"folder": "M", "kernels": {"dequant.xclbin": "xclbins-Dequant",
                                                             "../x.xclbin": "xclbins-Dequant"}}}}
        with contextlib.redirect_stdout(io.StringIO()):
            r = da.assemble(mixed, da.find_artifacts(self.art), out)
        self.assertEqual(r["missing"]["m"], ["../x.xclbin <- xclbins-Dequant (unsafe folder/slot)"])
        (out / "M/dequant.xclbin").unlink()
        self.assertFalse((self.root / "abs").exists())
        self.assertFalse((self.root / "escape").exists())
        self.assertEqual([p for p in out.rglob("*") if p.is_file()], [])

    def test_cli(self):
        amap = self.root / "ka.json"
        amap.write_text(json.dumps(ASSIGNMENT))
        gh = self.root / "gh_output"
        env = dict(os.environ, GITHUB_OUTPUT=str(gh))
        env.pop("GITHUB_STEP_SUMMARY", None)
        cmd = [sys.executable, str(HERE / "devkit_assemble_xclbins.py"), "--artifacts", str(self.art),
               "--out", str(self.root / "out"), "--assignment", str(amap), "--github-output"]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        lines = dict(l.split("=", 1) for l in gh.read_text().splitlines())
        self.assertEqual(json.loads(lines["touched-models"]), ["m:1b", "m:3b"])
        self.assertEqual(json.loads(lines["complete-models"]), ["m:1b"])
        self.assertEqual((lines["found"], lines["count"]), ("true", "5"))
        res = subprocess.run(cmd + ["--fail-on-partial"], capture_output=True, text=True, env=env)
        self.assertEqual(res.returncode, 1)


if __name__ == "__main__":
    unittest.main()
