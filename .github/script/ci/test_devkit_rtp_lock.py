# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for devkit_rtp_lock.py: python3 -m unittest discover -s ci -p 'test_*.py' -t ci"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import devkit_rtp_lock as rl  # noqa: E402

ASSIGNMENT = {"models": {
    "g:2b": {"folder": "G-2B", "kernels": {"layer.xclbin": "xclbins-G_decoding_E2B",
                                            "dequant.xclbin": "xclbins-Dequant"}},
    "g:4b": {"folder": "G-4B", "kernels": {"layer.xclbin": "xclbins-G_decoding_E4B"}},
    "v:1b": {"folder": "V-1B", "kernels": {"layer.xclbin": "xclbins-G_decoding_E4B"}},
    "s:1b": {"folder": "S-1B", "kernels": {"vision.xclbin": "xclbins-Smol"}},
}}
ENGINES = {"engines": {"g_npu": {"models": ["g:2b", "g:4b"]}, "v_npu": {"models": ["v:1b"]}},
           "served_elsewhere": {"s:1b": "a pybind module"}}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.shards = self.tmp / "shards"
        self.write("devkit-xclbins-shard-0/xclbins-G_decoding_E2B/layer.xclbin", "x")
        self.write("devkit-xclbins-shard-0/xclbins-G_decoding_E2B/rtp_addresses.txt",
                   "RTP_B   2048\n\nRTP_A 1024\n")
        self.write("devkit-xclbins-shard-1/xclbins-G_decoding_E4B/layer.xclbin", "x")
        self.write("devkit-xclbins-shard-1/xclbins-G_decoding_E4B/rtp_addresses.txt", "RTP_A 1\n")
        self.write("devkit-xclbins-shard-1/xclbins-G_decoding_E4B/fused8_rtp_addresses.txt", "RTP_F 8\n")
        self.write("devkit-xclbins-shard-1/xclbins-Dequant/dequant.xclbin", "x")
        # Built, writes RTPs, but no model uses it: neither locked nor checked.
        self.write("devkit-xclbins-shard-1/xclbins-Unused/layer.xclbin", "x")
        self.write("devkit-xclbins-shard-1/xclbins-Unused/rtp_addresses.txt", "R 7\n")
        self.lock = self.tmp / "lock.txt"
        (self.tmp / "a.json").write_text(json.dumps(ASSIGNMENT), encoding="utf-8")
        (self.tmp / "e.json").write_text(json.dumps(ENGINES), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write(self, rel: str, text: str) -> None:
        p = self.shards / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def run_cli(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            rc = rl.main(list(args))
        return rc, out.getvalue()

    def generate(self, *extra: str) -> int:
        return self.run_cli("generate", str(self.shards), "-o", str(self.lock),
                            "--assignment", str(self.tmp / "a.json"), *extra)[0]

    def check(self) -> tuple[int, str]:
        return self.run_cli("check", str(self.shards), "--lock", str(self.lock),
                            "--assignment", str(self.tmp / "a.json"), "--engines", str(self.tmp / "e.json"))


class Format(Fixture):
    def test_generate_is_sorted_and_normalized(self):
        self.assertEqual(self.generate("--note", "from run 1"), 0)
        text = self.lock.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# Copyright © Advanced Micro Devices"))
        self.assertIn("# from run 1\n", text)
        body = text[text.index("\n["):]
        self.assertEqual(body, "\n[xclbins-G_decoding_E2B/rtp_addresses.txt]\nRTP_A 1024\nRTP_B 2048\n"
                               "\n[xclbins-G_decoding_E4B/fused8_rtp_addresses.txt]\nRTP_F 8\n"
                               "\n[xclbins-G_decoding_E4B/rtp_addresses.txt]\nRTP_A 1\n")
        entries, header = rl.parse_lock(text)
        self.assertEqual(len(entries), 3)
        self.assertIn("from run 1", header)
        # Generating again gives the same bytes.
        self.generate("--note", "from run 1")
        self.assertEqual(self.lock.read_text(encoding="utf-8"), text)

    def test_parse_errors(self):
        for text, fragment in (("RTP_A 1\n", "expected [<artifact>/<file>]"),
                               ("[xclbins-A/r.txt]\nR 1\n\n[xclbins-A/r.txt]\nR 1\n", "appears twice"),
                               ("[../x/r.txt]\nR 1\n", "expected"),
                               # A comment inside a stanza is no address line.
                               ("[xclbins-A/r.txt]\n# moved in PR 12\nR 1\n", "comments only in the header"),
                               # No blank line between stanzas: B would vanish into A.
                               ("[xclbins-A/r.txt]\nR 1\n[xclbins-B/r.txt]\nR 2\n", "end a stanza with a blank"),
                               ("[xclbins-A/r.txt]\nR 1 2\n", "expected '<name> <address>'")):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, fragment.replace("[", r"\[")):
                    rl.parse_lock(text)

    def test_changed_artifacts(self):
        base = "[xclbins-A/r.txt]\nR 1\n\n[xclbins-B/r.txt]\nR 2\n\n[xclbins-C/r.txt]\nR 3\n"
        head = "[xclbins-A/r.txt]\nR 1\n\n[xclbins-B/r.txt]\nR 9\n\n[xclbins-D/r.txt]\nR 4\n"
        self.assertEqual(rl.changed_artifacts(base, head), ["xclbins-B", "xclbins-C", "xclbins-D"])
        self.assertEqual(rl.changed_artifacts(None, head), ["xclbins-A", "xclbins-B", "xclbins-D"])
        # Whitespace and order are no change.
        self.assertEqual(rl.changed_artifacts(base, base.replace("R 1", "R    1  ")), [])

    def test_update_keeps_artifacts_not_built(self):
        self.lock.write_text("[xclbins-Smol/rtp_addresses.txt]\nR 5\n\n"
                             "[xclbins-Other/rtp_addresses.txt]\nR 5\n\n"
                             "[xclbins-G_decoding_E4B/old_rtp_addresses.txt]\nR 6\n", encoding="utf-8")
        self.generate("--update")
        entries, _ = rl.parse_lock(self.lock.read_text(encoding="utf-8"))
        # Smol kept; Other dropped (no model uses it); E4B was built, so its
        # stanzas are this run's only.
        self.assertIn(("xclbins-Smol", "rtp_addresses.txt"), entries)
        self.assertNotIn(("xclbins-Other", "rtp_addresses.txt"), entries)
        self.assertNotIn(("xclbins-G_decoding_E4B", "old_rtp_addresses.txt"), entries)
        self.assertEqual(len(entries), 4)

    def test_update_keeps_header_notes(self):
        self.generate("--note", "Generated from run 1, all builds.")
        shutil.rmtree(self.shards / "devkit-xclbins-shard-0")
        self.generate("--update", "--note", "E4B from run 2.")
        entries, header = rl.parse_lock(self.lock.read_text(encoding="utf-8"))
        self.assertEqual(rl.header_notes(header), ["Generated from run 1, all builds.", "E4B from run 2."])
        self.assertIn(("xclbins-G_decoding_E2B", "rtp_addresses.txt"), entries)
        # Without --update, the notes are only those given.
        self.generate("--note", "run 3")
        self.assertEqual(rl.header_notes(rl.parse_lock(self.lock.read_text(encoding="utf-8"))[1]), ["run 3"])

    def test_artifacts_no_model_uses_are_left_out(self):
        self.generate()
        entries, _ = rl.parse_lock(self.lock.read_text(encoding="utf-8"))
        self.assertFalse(any(a == "xclbins-Unused" for a, _ in entries))


class Check(Fixture):
    def setUp(self):
        super().setUp()
        self.generate()

    def test_matching_run_passes(self):
        rc, out = self.check()
        self.assertEqual(rc, 0, out)
        self.assertIn("4 artifacts built, 3 RTP files checked", out)
        self.assertIn("not checked, as no model uses them: xclbins-Unused", out)

    def test_moved_rtp_fails_naming_models_and_engines(self):
        self.write("devkit-xclbins-shard-1/xclbins-G_decoding_E4B/rtp_addresses.txt", "RTP_A 2\n")
        rc, out = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("xclbins-G_decoding_E4B/rtp_addresses.txt differs from the lock", out)
        self.assertIn("Models using it: g:4b, v:1b", out)
        self.assertIn("flm-dev-kit/FLM_DLL/detail/g_npu/, flm-dev-kit/FLM_DLL/detail/v_npu/", out)
        self.assertIn("- RTP_A 1", out)
        self.assertIn("+ RTP_A 2", out)
        self.assertIn("generate <dir> -o", out)

    def test_new_and_vanished_files_fail(self):
        self.write("devkit-xclbins-shard-1/xclbins-Smol/vision.xclbin", "x")
        self.write("devkit-xclbins-shard-1/xclbins-Smol/rtp_addresses.txt", "R 1\n")
        os.remove(self.shards / "devkit-xclbins-shard-1/xclbins-G_decoding_E4B/fused8_rtp_addresses.txt")
        rc, out = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("xclbins-Smol/rtp_addresses.txt is not in the lock", out)
        self.assertIn("s:1b: a pybind module", out)
        self.assertIn("fused8_rtp_addresses.txt is in the lock, but the build wrote no such file", out)

    def test_artifacts_not_built_are_not_checked(self):
        shutil.rmtree(self.shards / "devkit-xclbins-shard-1")
        rc, out = self.check()
        self.assertEqual(rc, 0, out)
        self.assertIn("1 artifacts built, 1 RTP files checked", out)

    def test_errors_name_their_source(self):
        # The same artifact in two shards is the shards' fault, not the lock's.
        self.write("devkit-xclbins-shard-2/xclbins-Dequant/dequant.xclbin", "x")
        rc, out = self.check()
        self.assertEqual(rc, 1)
        self.assertIn(f"::error::{self.shards}: xclbins-Dequant appears twice", out)
        self.lock.write_text("[xclbins-A/r.txt]\n# c\nR 1\n", encoding="utf-8")
        rc, out = self.check()
        self.assertEqual(rc, 1)
        self.assertIn(f"::error::{self.lock}: line 2", out)

    def test_summary(self):
        self.write("devkit-xclbins-shard-1/xclbins-G_decoding_E4B/rtp_addresses.txt", "RTP_A 2\n")
        summary = self.tmp / "summary.md"
        os.environ["GITHUB_STEP_SUMMARY"] = str(summary)
        try:
            rc, _ = self.run_cli("check", str(self.shards), "--lock", str(self.lock), "--github-output",
                                 "--assignment", str(self.tmp / "a.json"), "--engines", str(self.tmp / "e.json"))
        finally:
            del os.environ["GITHUB_STEP_SUMMARY"]
        self.assertEqual(rc, 1)
        self.assertIn("| `xclbins-G_decoding_E4B` | `rtp_addresses.txt` | differs from the lock | g:4b, v:1b |",
                      summary.read_text(encoding="utf-8"))


class RealLock(unittest.TestCase):
    def test_repository_lock_parses(self):
        entries, header = rl.parse_lock(rl.LOCK.read_text(encoding="utf-8"))
        self.assertTrue(entries)
        self.assertTrue(any("Copyright" in l for l in header))
        # Stored as generate writes it.
        self.assertEqual(rl.render_lock(entries, rl.header_notes(header)),
                         rl.LOCK.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
