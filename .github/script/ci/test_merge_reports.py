"""Tests for merge_reports.py: python3 .github/script/ci/test_merge_reports.py

Build and Test's summary job runs merge_reports.py on every run, before its
Verdict step (what release.yml waits for), so a crash there blocks a release.
A change to it alone runs no NPU test (changes.py), so these tests are what
checks it: the formatting, and the reports as the npu_test harness writes
them (Context.write_report)."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import merge_reports  # noqa: E402
from npu_test.harness import FAIL, PASS, SKIP, WARN, Context, Result  # noqa: E402


def report(label="Linux XRT", suites=("chat",), results=(), seconds=60.0, tier="pr", version="1.0.0") -> dict:
    return {"label": label, "tier": tier, "flm_version": version, "suites": list(suites), "seconds": seconds,
            "counts": {}, "results": [dict({"model": "", "detail": "", "seconds": 0.0, "data": {}}, **r)
                                      for r in results]}


def res(suite, status, case="c", model="m:1b", detail="") -> dict:
    return {"suite": suite, "case": case, "status": status, "model": model, "detail": detail}


class Merge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, name: str, data) -> None:
        path = self.dir / name / "results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")

    def run_merge(self, *extra: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = merge_reports.main([str(self.dir), *extra])
        return rc, out.getvalue(), err.getvalue()

    def table(self, text: str) -> list[list[str]]:
        rows = [l for l in text.splitlines() if l.startswith("| ") and not l.startswith("| Package | Suite")]
        return [[c.strip() for c in l.strip().strip("|").split("|")] for l in rows]

    def test_no_reports(self):
        for setup in (lambda: None, lambda: self.put("npu-test-x", "{not json")):
            with self.subTest():
                setup()
                rc, text, _ = self.run_merge()
                self.assertEqual(rc, 0)
                self.assertIn("## NPU tests", text)
                self.assertIn("No test reports were uploaded", text)
        # A missing directory too: the download step found no artifact.
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(merge_reports.main([str(self.dir / "absent")]), 0)
        self.assertIn("No test reports were uploaded", out.getvalue())

    def test_columns_are_the_union_of_suites(self):
        self.put("a", report("A", ["chat", "api"], [res("chat", PASS), res("api", PASS)]))
        self.put("b", report("B", ["chat", "asr"], [res("chat", PASS), res("asr", PASS)]))
        _, text, _ = self.run_merge()
        rows = self.table(text)
        self.assertEqual(rows[0], ["Package", "Tier", "flm", "api", "asr", "chat", "Time"])
        a, b = rows[1], rows[2]
        self.assertEqual((a[0], a[3:6]), ("A", ["✅", "", "✅"]))
        self.assertEqual((b[0], b[3:6]), ("B", ["", "✅", "✅"]))

    def test_cell_states(self):
        self.put("a", report("A", ["chat", "api", "asr", "decision", "serving"], [
            res("chat", FAIL), res("chat", FAIL), res("chat", PASS),
            res("api", WARN), res("api", PASS),
            res("asr", PASS),
            res("serving", SKIP)]))
        _, text, _ = self.run_merge()
        row = self.table(text)[1]
        cells = dict(zip(["api", "asr", "chat", "decision", "serving"], row[3:8]))
        self.assertEqual(cells, {"chat": "❌ 2", "api": "⚠️", "asr": "✅", "decision": "–", "serving": "✅"})

    def test_time(self):
        self.put("a", report("A", seconds=89.9))
        self.put("b", report("B", seconds=29))
        _, text, _ = self.run_merge()
        self.assertEqual([r[-1] for r in self.table(text)[1:]], ["1 min", "0 min"])

    def test_failure_details(self):
        self.put("a", report("A", ["chat"], [res("chat", FAIL, detail="a|b\nc"),
                                             res("chat", FAIL, case="long", detail="x" * 300)]))
        _, text, _ = self.run_merge()
        self.assertIn("<details><summary>Failed cases</summary>", text)
        self.assertIn("| A | chat | c | m:1b | a\\|b c |", text)
        self.assertIn("| A | chat | long | m:1b | " + "x" * 200 + " |", text)

    def test_failures_are_capped(self):
        for n, more in ((0, None), (200, None), (201, "| | | | | 1 more not shown |")):
            with self.subTest(n=n):
                self.put("a", report("A", ["chat"], [res("chat", FAIL, case=f"c{i}") for i in range(n)]
                                     + [res("chat", PASS)]))
                _, text, _ = self.run_merge()
                self.assertEqual("<details>" in text, n > 0)
                self.assertEqual(text.count("| A | chat | c"), min(n, 200))
                if more:
                    self.assertIn(more, text)
                else:
                    self.assertNotIn("more not shown", text)

    def test_unreadable_report_is_skipped(self):
        self.put("a", report("A", ["chat"], [res("chat", PASS)]))
        self.put("b", "{broken")
        rc, text, err = self.run_merge()
        self.assertEqual(rc, 0)
        self.assertIn("skipping", err)
        self.assertEqual([r[0] for r in self.table(text)[1:]], ["A"])

    def test_output_file(self):
        self.put("a", report("A"))
        out = self.dir / "summary.md"
        _, text, _ = self.run_merge("--output", str(out))
        # print() adds one newline after the text written to the file.
        self.assertEqual(out.read_text(encoding="utf-8") + "\n", text)

    def test_reports_the_harness_writes(self):
        # The contract: merge_reports reads what Context.write_report writes.
        for i, (label, suites, results) in enumerate((
                ("Linux XRT portable", ["chat", "api"],
                 [Result("chat", "basic", PASS, "qwen3:0.6b", seconds=1.5),
                  Result("api", "tools", FAIL, "qwen3:0.6b", "HTTP 500: k_proj_run failed | timeout"),
                  Result("api", "shape", WARN, "qwen3:0.6b", "slow")]),
                ("Windows HRX", ["chat", "asr"],
                 [Result("chat", "basic", PASS, "gemma3:1b"), Result("asr", "wav", SKIP, "whisper-v3:turbo")])), 1):
            ctx = Context(flm_bin="flm", label=label, tier="pr", plan=None,
                          out_dir=self.dir / f"npu-test-{i}", request_timeout=1.0)
            with contextlib.redirect_stdout(io.StringIO()):
                for r in results:
                    ctx.record(r)
            ctx.version = "1.2.3"
            ctx.write_report(suites, 125.0)
        rc, text, err = self.run_merge()
        self.assertEqual((rc, err), (0, ""))
        rows = self.table(text)
        self.assertEqual(rows[0], ["Package", "Tier", "flm", "api", "asr", "chat", "Time"])
        self.assertEqual(rows[1], ["Linux XRT portable", "pr", "1.2.3", "❌ 1", "", "✅", "2 min"])
        self.assertEqual(rows[2], ["Windows HRX", "pr", "1.2.3", "", "✅", "✅", "2 min"])
        self.assertIn("| Linux XRT portable | api | tools | qwen3:0.6b | HTTP 500: k_proj_run failed \\| timeout |",
                      text)


if __name__ == "__main__":
    unittest.main()
