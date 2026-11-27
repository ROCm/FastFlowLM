# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for check_sampler_defaults.py: python3 -m unittest discover -s ci -p 'test_*.py' -t ci"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_sampler_defaults as csd  # noqa: E402


def lines(text: str) -> list[int]:
    return [number for number, _, _ in csd.scan_text(text)]


class ScanTextTest(unittest.TestCase):
    def test_scale_slips_hit(self):
        for text in (
            "    config.freq_penalty = 1.0;",
            "    config.freq_penalty = 1.05;",
            "    config.pre_penalty = 1.0f;",
            "    config.pre_penalty = 1.5f;",
            "    config.freq_penalty = .5;",
            "    config.pre_penalty = 2;  /* no line comment */",
            "    config.freq_penalty = -0.5f; // a source",
            "    config.rep_penalty = 0.0;",
            "    config.rep_penalty = 0.95f; // a source",
        ):
            with self.subTest(text=text):
                self.assertEqual(lines(text), [1])

    def test_neutral_and_sourced_pass(self):
        for text in (
            "    config.freq_penalty = 0.0f;",
            "    config.freq_penalty = 0;",
            "    config.pre_penalty = 0.0;",
            "    config.pre_penalty = 1.5f; // Qwen3.5 card: presence_penalty 1.5",
            "    config.rep_penalty = 1.0;",
            "    config.rep_penalty = 1.05;",
            "    float freq_penalty = 0.0f;",
            "    float rep_penalty = 1.0f;",
            "    this->freq_penalty = config.freq_penalty;",
            "    if (this->freq_penalty == 1.0f) {",
            "    // config.freq_penalty = 1.0; would be wrong",
        ):
            with self.subTest(text=text):
                self.assertEqual(lines(text), [])

    def test_reports_line_numbers(self):
        text = "a\n    config.freq_penalty = 1.0;\nb\n    config.pre_penalty = 1.0f;\n"
        self.assertEqual(lines(text), [2, 4])


class RepoTest(unittest.TestCase):
    def _repo(self, files: dict[str, str]) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        for rel, text in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
        return root

    def test_flags_loaders_only_under_src(self):
        root = self._repo({
            "src/common/AutoModel/modeling_x.cpp": "config.pre_penalty = 1.0f;\n",
            "src/test/sampler/test_sampler.cpp": "config.pre_penalty = 1.0f;\n",
            "flm-dev-kit/FLM_DLL/common/modules/sampler.cpp": "config.freq_penalty = 1.0f;\n",
            "src/README.md": "config.freq_penalty = 1.0;\n",
        })
        errors, scanned = csd.scan_repo(root)
        self.assertEqual(scanned, 1)
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("src/common/AutoModel/modeling_x.cpp:1: pre_penalty"))

    def test_main_exit_codes(self):
        bad = self._repo({"src/a.cpp": "config.freq_penalty = 1.05;\n"})
        good = self._repo({"src/a.cpp": "config.freq_penalty = 0.0f;\n"})
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(csd.main(["--repo", str(good)]), 0)
        self.assertIn("ok (1 source files)", out.getvalue())
        with redirect_stdout(io.StringIO()) as out, unittest.mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(csd.main(["--repo", str(bad)]), 1)
        self.assertIn("src/a.cpp:1:", out.getvalue())

    def test_this_repository_is_clean(self):
        errors, scanned = csd.scan_repo(csd.REPO_ROOT)
        self.assertGreater(scanned, 0)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
