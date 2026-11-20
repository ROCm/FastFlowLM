# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for the dev-kit bench scripts (.github/script/devkit-bench): scoring,
the gate, report rendering, comparisons and soft gates, machine kinds, model
selection, run and artifact lookup, and the comment workflow's invariants.

Here rather than beside the scripts so the changes job's
`unittest discover -s ci` runs them.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "devkit-bench"
sys.path.insert(0, str(BENCH))

import acc_common as ac  # noqa: E402
import compare  # noqa: E402
import docvqa_fetch  # noqa: E402
import gsm8k_run  # noqa: E402
import machine  # noqa: E402
import report  # noqa: E402
import resolve  # noqa: E402
import score_anls  # noqa: E402
import select_models  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SHA_A = "a" * 40
SHA_B = "b" * 40


def quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


class Gsm8k(unittest.TestCase):
    def test_extract_answer(self):
        self.assertEqual(gsm8k_run.extract_answer("work...\n#### 1,234"), 1234)
        self.assertEqual(gsm8k_run.extract_answer("#### $18.50"), 18.5)
        self.assertEqual(gsm8k_run.extract_answer("so 3 then 7"), 7)
        self.assertIsNone(gsm8k_run.extract_answer("no number"))
        self.assertEqual(gsm8k_run.extract_answer("<think>#### 5</think>#### 6"), 6)
        self.assertIsNone(gsm8k_run.extract_answer("<think>12 and 13"))

    def test_numbers_equal(self):
        self.assertTrue(gsm8k_run.numbers_equal(18, 18.00001))
        self.assertFalse(gsm8k_run.numbers_equal(None, 1))
        self.assertFalse(gsm8k_run.numbers_equal(1, 2))

    def test_dataset(self):
        rows = [json.loads(l) for l in (BENCH / "data" / "gsm8k_100.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 100)
        self.assertEqual([r["id"] for r in rows], list(range(100)))
        self.assertTrue(all(isinstance(r["answer"], (int, float)) and r["question"] for r in rows))


class Anls(unittest.TestCase):
    def test_scores(self):
        self.assertEqual(score_anls.anls_one("Hello  World", ["hello world"]), 1.0)
        self.assertEqual(score_anls.anls_one("abc", ["xyz"]), 0.0)
        self.assertAlmostEqual(score_anls.anls_one("abcd", ["abce"]), 0.75)
        # Below the 0.5 threshold scores 0; the best answer counts.
        self.assertEqual(score_anls.anls_one("ab", ["abcde"]), 0.0)
        self.assertEqual(score_anls.anls_one("abcde", ["x", "abcde"]), 1.0)
        self.assertEqual(score_anls.anls_one("", [""]), 1.0)

    def test_levenshtein(self):
        self.assertEqual(score_anls.levenshtein("kitten", "sitting"), 3)
        self.assertEqual(score_anls.levenshtein("", "abc"), 3)


class AccCommon(unittest.TestCase):
    def rows(self):
        return [
            {"id": 1, "pred": 3, "pred_raw": "#### 3", "finish_reason": "stop", "latency_s": 2.0,
             "completion_tokens": 10},
            {"id": 2, "pred": None, "pred_raw": "no", "finish_reason": "stop"},
            {"id": 3, "pred": None, "error": "timeout", "error_detail": "t"},
            {"id": 4, "pred": None, "pred_raw": "<think>x", "reasoning_only": True, "finish_reason": "length"},
        ]

    def test_outcomes_and_tally(self):
        rows = self.rows()
        self.assertEqual([ac.outcome(r) for r in rows], ["answered", "unparsed", "failed", "reasoning_only"])
        t = ac.tally(rows)
        self.assertEqual((t["total"], t["unanswered"], t["failed"], t["unparsed"], t["reasoning_only"],
                          t["truncated"]), (4, 2, 1, 1, 1, 1))
        self.assertEqual(t["failures"], {"timeout": 1})

    def test_strip_think_and_reasoning_only(self):
        self.assertEqual(ac.strip_think("<think>a</think> b"), "b")
        self.assertEqual(ac.strip_think("a</think>b"), "b")
        self.assertEqual(ac.strip_think("x<think>cut"), "x")
        self.assertTrue(ac.reasoning_only({"content": "", "finish_reason": "length"}, None))
        self.assertFalse(ac.reasoning_only({"content": "", "finish_reason": "stop"}, None))
        self.assertTrue(ac.reasoning_only({"content": "<think>only"}, None))

    def test_gate(self):
        s = dict(ac.tally(self.rows()))
        self.assertEqual(quiet(ac.gate, "x", "accuracy", 25.0, s, 5), 0)
        self.assertTrue(s["usable"])
        s = dict(ac.tally(self.rows()))
        self.assertEqual(quiet(ac.gate, "x", "accuracy", 25.0, s, 1), 1)
        self.assertFalse(s["usable"])
        self.assertIn("over --max-unanswered=1", s["problems"][0])
        # A score of 0 only warns; no predictions fails.
        s = dict(ac.tally(self.rows()[:1]))
        self.assertEqual(quiet(ac.gate, "x", "accuracy", 0.0, s, 5), 0)
        self.assertIn("warnings", s)
        s = dict(ac.tally([]))
        self.assertEqual(quiet(ac.gate, "x", "accuracy", 0.0, s, 5), 1)

    def test_report_escapes(self):
        rows = self.rows()
        rows[1]["pred_raw"] = "<script>| `x`"
        s = {**ac.tally(rows), "server": "v", "config": {"n": 4}}
        with tempfile.TemporaryDirectory() as d:
            text = Path(ac.write_report(d, "T", "Accuracy 25%", s, rows, "id", "gt")).read_text(encoding="utf-8")
        self.assertIn("&lt;script&gt;", text)
        self.assertNotIn("<script>", text)
        self.assertIn("3 question(s) worth a look", text)

    def test_resume_drops_failures(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for r in self.rows():
                    f.write(json.dumps(r) + "\n")
            kept = ac.load_resumable(path, "id")
        self.assertEqual(kept, {"1", "2", "4"})

    def test_time_budget(self):
        asker = ac.Asker("http://x", 64, 10, time_budget_s=0.0)
        self.assertTrue(asker.time_left())
        asker = ac.Asker("http://x", 64, 10, time_budget_s=1e-9)
        asker.deadline -= 1
        self.assertFalse(asker.time_left())
        self.assertTrue(asker.stats()["out_of_time"])


class Machine(unittest.TestCase):
    def test_kind(self):
        self.assertEqual(machine.kind_from("NPU Strix"), "strix")
        self.assertEqual(machine.kind_from("NPU Strix Halo"), "strix-halo")
        self.assertEqual(machine.kind_from("NPU Krackan"), "krackan")
        self.assertEqual(machine.kind_from("", "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"), "strix-halo")
        self.assertEqual(machine.kind_from("", "AMD Ryzen AI 9 HX 370 w/ Radeon 890M"), "strix")
        self.assertEqual(machine.kind_from("", "AMD Ryzen AI 7 350 w/ Radeon 860M"), "krackan")
        self.assertEqual(machine.kind_from("", "AMD Ryzen AI 5 340"), "krackan")
        self.assertEqual(machine.kind_from("", "Intel Core i9"), "unknown")
        self.assertFalse(machine.valid_kind("unknown"))
        self.assertFalse(machine.valid_kind("a b"))
        self.assertTrue(machine.valid_kind("strix-halo"))


CSV = ("context_length_k,ttft_avg_s,ttft_std_s,ttft_min_s,ttft_max_s,prefill_avg_toks_per_s,prefill_std_toks_per_s,"
       "prefill_min_toks_per_s,prefill_max_toks_per_s,decoding_avg_toks_per_s,decoding_std_toks_per_s,"
       "decoding_min_toks_per_s,decoding_max_toks_per_s\n"
       "1,0.5,0,0,0,{p1},0,0,0,{d1},0,0,0\n2,0.9,0,0,0,{p2},0,0,0,{d2},0,0,0\n")


class Compare(unittest.TestCase):
    def test_parse_perf_csv_validates(self):
        pts = compare.parse_perf_csv(CSV.format(p1=1000, d1=20, p2=900, d2=19) + "x,1,1\n4,nan,0,0,0,inf\n")
        self.assertEqual(sorted(pts), [1.0, 2.0, 4.0])
        self.assertEqual(pts[1.0]["decoding_avg_toks_per_s"], 20.0)
        self.assertEqual(pts[4.0], {})

    def test_perf_ab_warns_past_threshold(self):
        new = compare.parse_perf_csv(CSV.format(p1=1000, d1=18.0, p2=900, d2=18.9))
        old = compare.parse_perf_csv(CSV.format(p1=1000, d1=20.0, p2=900, d2=19.0))
        res = compare.compare_perf(new, old, warn=True)
        self.assertEqual(len(res["warnings"]), 1)
        self.assertIn("decode at 1k is -10.0%", res["warnings"][0])
        self.assertEqual(compare.compare_perf(new, old, warn=False)["warnings"], [])
        self.assertEqual(compare.compare_perf(new, {}, warn=True)["rows"][0]["decode"], (None, 18.0, None))

    def test_perf_rows_roundtrip(self):
        pts = compare.parse_perf_csv(CSV.format(p1=1000, d1=20, p2=900, d2=19))
        self.assertEqual(compare.perf_rows_from_json(compare.perf_rows_to_json(pts)), pts)
        self.assertEqual(compare.perf_rows_from_json("x"), {})

    def summary(self, **kw):
        base = {"accuracy": 80.0, "total": 50, "unanswered": 0, "failed": 0, "failures": {}, "unparsed": 0,
                "reasoning_only": 0, "truncated": 0, "usable": True, "server": '{"version":"1"}',
                "config": {"n": 50, "asked": 50}}
        base.update(kw)
        return base

    def test_clean_acc_summary(self):
        s = compare.clean_acc_summary(self.summary(failures={"timeout": 2, "evil<b>": 1}), "gsm8k")
        self.assertEqual(s["failures"], {"timeout": 2})
        self.assertIsNone(compare.clean_acc_summary(self.summary(accuracy="NaN"), "gsm8k"))
        self.assertIsNone(compare.clean_acc_summary(self.summary(accuracy=101), "gsm8k"))
        self.assertIsNone(compare.clean_acc_summary([], "gsm8k"))
        self.assertTrue(compare.clean_acc_summary(self.summary(config={"n": 50, "asked": 20}), "gsm8k")["partial"])
        self.assertIsNotNone(compare.clean_acc_summary({"anls": 70, "total": 50}, "docvqa"))

    def test_acc_delta_and_soft_gate(self):
        pr = compare.clean_acc_summary(self.summary(accuracy=60.0, unanswered=7, failed=7,
                                                    failures={"timeout": 7}, usable=False), "gsm8k")
        base = compare.clean_acc_summary(self.summary(server='{"version":"0"}'), "gsm8k")
        res = compare.compare_acc(pr, base)
        self.assertEqual(res["delta"], -20.0)
        self.assertEqual(len(res["warnings"]), 1)
        notes = "; ".join(res["notes"])
        self.assertIn("failed the bench gate", notes)
        self.assertIn("⚠️ 7 unanswered (timeout ×7)", notes)
        self.assertIn("FLM version differs", notes)
        ok = compare.compare_acc(compare.clean_acc_summary(self.summary(accuracy=75.0), "gsm8k"), base)
        self.assertEqual(ok["warnings"], [])
        self.assertIsNone(compare.compare_acc(pr, None)["delta"])


def write_result(root: Path, key: str, model: str, *, kind="NPU Strix", devkit=None, base=None,
                 gsm8k=None, status="ok", vlm=False):
    d = root / f"devkit-bench-result-{key}-{model.replace(':', '_')}"
    (d / "perf").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"key": key, "model": model, "vlm": vlm}))
    (d / "machine.json").write_text(json.dumps({"kind": machine.kind_from(kind)}))
    variants = {"devkit": {"status": status}}
    if devkit:
        (d / "perf" / "devkit.csv").write_text(devkit)
    if base:
        (d / "perf" / "base.csv").write_text(base)
        variants["base"] = {"status": "ok"}
    (d / "perf" / "status.json").write_text(json.dumps({"variants": variants}))
    if gsm8k:
        (d / "gsm8k").mkdir()
        (d / "gsm8k" / "summary.json").write_text(json.dumps(gsm8k))
    return d


class Report(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def gsm(self, acc, **kw):
        return {"accuracy": acc, "total": 50, "unanswered": 0, "failed": 0, "failures": {}, "unparsed": 0,
                "reasoning_only": 0, "truncated": 0, "usable": True, "config": {"n": 50, "asked": 50}, **kw}

    def test_render_and_upsert(self):
        write_result(self.root, "lin-xrt", "qwen3:0.6b", devkit=CSV.format(p1=900, d1=18, p2=800, d2=17),
                     base=CSV.format(p1=1000, d1=20, p2=820, d2=17.5), gsm8k=self.gsm(40.0))
        write_result(self.root, "lin-hrx", "qwen3:0.6b", status="failed (exit 3)",
                     gsm8k=self.gsm(10.0, usable=False, unanswered=9, failed=9, failures={"timeout": 9}))
        # Junk an attacker could upload is ignored.
        bad = self.root / "devkit-bench-result-lin-xrt-x"
        bad.mkdir()
        (bad / "meta.json").write_text(json.dumps({"key": "lin-xrt", "model": "<img src=x>"}))
        results = report.load_results(self.root)
        self.assertEqual(len(results), 2)
        old = report.parse_baseline({"schema": 1, "sha": SHA_A, "run_id": 7, "updated": "2026-11-19", "entries": {
            "lin-xrt": {"qwen3:0.6b": {"gsm8k": {"score": 55.0, "total": 50, "n": 50},
                                       "perf": {"strix": {"rows": [{"context_length_k": 1,
                                                                    "decoding_avg_toks_per_s": 19}]}}},
                        "llama3.2:1b": {"gsm8k": {"score": 30.0, "total": 50}}},
            "lin-hrx": {"qwen3:0.6b": {"gsm8k": {"score": 50.0, "total": 50}}},
            "bogus": {}, }})
        matrix = [{"key": "lin-xrt", "model": "qwen3:0.6b"}, {"key": "lin-hrx", "model": "qwen3:0.6b"},
                  {"key": "lin-hrx", "model": "gemma3:1b"}]
        text = report.render(results, old, matrix, mode="pr", head_sha=SHA_B, build_run_id=12, baseline_run_id=7)
        self.assertIn("## Dev-kit bench (pr) on `bbbbbbbbbb`", text)
        self.assertIn("nightly baseline: run 7 on `aaaaaaaaaa` (2026-11-19)", text)
        self.assertIn("Linux XRT qwen3:0.6b: decode at 1k is -10.0%", text)
        self.assertIn("gsm8k: -15.0 points", text)
        self.assertIn("Linux HRX qwen3:0.6b gsm8k: failed the bench gate", text)
        self.assertIn("no results from the bench job of gemma3:1b", text)
        self.assertIn("| qwen3:0.6b | strix | 1k | 20.0 → 18.0 | -10.0% |", text)
        self.assertIn("| -5.3% |", text)  # decode vs the nightly's 19 for strix
        self.assertNotIn("<img", text)
        new = report.upsert_baseline(old, results, SHA_B, 99)
        xrt = new["entries"]["lin-xrt"]
        self.assertEqual(xrt["qwen3:0.6b"]["gsm8k"]["score"], 40.0)
        self.assertEqual(xrt["qwen3:0.6b"]["gsm8k"]["sha"], SHA_B)
        self.assertEqual(xrt["qwen3:0.6b"]["perf"]["strix"]["rows"][0]["decoding_avg_toks_per_s"], 18.0)
        self.assertEqual(xrt["llama3.2:1b"]["gsm8k"]["score"], 30.0)
        # Unusable results keep the old numbers.
        self.assertEqual(new["entries"]["lin-hrx"]["qwen3:0.6b"]["gsm8k"]["score"], 50.0)
        self.assertNotIn("bogus", new["entries"])
        self.assertEqual(report.parse_baseline(json.loads(json.dumps(new)))["entries"], new["entries"])

    def test_cap(self):
        text = report.cap("line\n" * 30000, limit=1000)
        self.assertLessEqual(len(text.encode()), 1000)
        self.assertIn("cut", text)
        self.assertEqual(report.cap("short\n"), "short\n")

    def test_parse_baseline_rejects_garbage(self):
        self.assertEqual(report.parse_baseline("x")["entries"], {})
        self.assertEqual(report.parse_baseline({"schema": 2})["entries"], {})
        b = report.parse_baseline({"schema": 1, "sha": "nope", "entries": {"lin-xrt": {
            "bad model|": {"gsm8k": {"score": 1}}, "ok:1b": {"gsm8k": {"score": "x"}, "perf": {"a b": {}}}}}})
        self.assertEqual(b["sha"], "")
        self.assertEqual(b["entries"], {})


CATALOG = {"models": {
    "qwen3": {"0.6b": {"size": 6e8, "details": {"family": "qwen3"}, "supported_platforms": ["aie2p"]},
              "8b": {"size": 8e9, "details": {"family": "qwen3"}}},
    "qwen3-it": {"4b": {"size": 4e9, "details": {"family": "qwen3-it"}}},
    "llama3.2": {"1b": {"size": 1e9, "details": {"family": "llama3"}}},
    "embed-gemma": {"300m": {"size": 3e8, "details": {"family": "embed-gemma"}}},
    "qwen3.5-oga-hipep": {"9b": {"size": 9e9, "details": {"family": "qwen3.5"}}},
    "gemma3": {"4b": {"size": 4e9, "details": {"family": "gemma3"}}},
}}
ENGINES = {"shared": ["gemm"], "engines": {
    "qwen3_npu": {"families": ["qwen3", "qwen3-it"], "models": ["qwen3:0.6b", "qwen3:8b", "qwen3-it:4b"]},
    "llama_npu": {"families": ["llama3"], "models": ["llama3.2:1b"]},
    "gemma_embedding": {"families": ["embed-gemma"], "models": ["embed-gemma:300m"]},
    "gemma_npu": {"families": ["gemma3"], "models": ["gemma3:4b"]},
}}
ASSIGNMENT = {"models": {"qwen3:0.6b": {}, "qwen3:8b": {}, "qwen3-it:4b": {}, "llama3.2:1b": {},
                         "embed-gemma:300m": {}, "gemma3:4b": {"vlm": True}, "smolvla:500m": {}}}


class SelectModels(unittest.TestCase):
    def select(self, **kw):
        engines = select_models.changes.DevkitEngines.parse(json.dumps(ENGINES))
        catalog = select_models.changes.load_catalog(json.dumps(CATALOG))
        return select_models.select(catalog=catalog, engines=engines, assignment=ASSIGNMENT, **kw)

    def names(self, res):
        return [m["model"] for m in res["models"]]

    def test_pr_engine_families(self):
        res = self.select(mode="pr", families=["qwen3"])
        self.assertEqual(self.names(res), ["qwen3:0.6b", "qwen3:8b"])
        res = self.select(mode="pr", families=["qwen3", "qwen3-it"], cap=2)
        self.assertEqual(self.names(res), ["qwen3:0.6b", "qwen3-it:4b"])
        self.assertEqual(res["dropped"], ["qwen3:8b"])

    def test_pr_every_family_one_per_engine(self):
        res = self.select(mode="pr", families=["qwen3", "qwen3-it", "llama3", "embed-gemma", "gemma3"])
        self.assertEqual(self.names(res), ["qwen3:0.6b", "llama3.2:1b", "gemma3:4b"])

    def test_pr_kernels_and_engines_take_turns(self):
        res = self.select(mode="pr", families=["qwen3", "qwen3-it"], xclbin_models=["gemma3:4b"],
                          xclbin_bench_models=["gemma3:4b"], cap=2)
        self.assertEqual(self.names(res), ["gemma3:4b", "qwen3:0.6b"])
        self.assertEqual(res["dropped"], ["qwen3-it:4b", "qwen3:8b"])

    def test_pr_kernels_worth_benching(self):
        res = self.select(mode="pr", xclbin_models=["gemma3:4b", "llama3.2:1b", "smolvla:500m"],
                          xclbin_bench_models=["gemma3:4b", "smolvla:500m"])
        self.assertEqual(res["models"], [{"model": "gemma3:4b", "vlm": True}])
        self.assertEqual(res["skipped"], ["smolvla:500m"])

    def test_nightly_and_dispatch(self):
        res = self.select(mode="nightly")
        self.assertEqual(self.names(res), ["qwen3:0.6b", "llama3.2:1b", "gemma3:4b", "qwen3-it:4b", "qwen3:8b"])
        self.assertIn("embed-gemma:300m", res["skipped"])
        # In turns: two a night, every model within three nights.
        seen = set()
        for day in range(3):
            res = self.select(mode="nightly", cap=2, rotation=day)
            self.assertEqual(len(res["models"]), 2)
            seen |= set(self.names(res))
        self.assertEqual(len(seen), 5)
        res = self.select(mode="dispatch", requested=["qwen3:8b", "qwen3.5-oga-hipep:9b", "qwen3:0.6b"])
        self.assertEqual(self.names(res), ["qwen3:8b", "qwen3:0.6b"])
        self.assertEqual(res["skipped"], ["qwen3.5-oga-hipep:9b"])

    def test_real_maps_load(self):
        engines = select_models.changes.DevkitEngines.parse(
            (REPO / "flm-dev-kit/FLM_DLL/engines.json").read_text(encoding="utf-8"))
        catalog = select_models.changes.load_catalog((REPO / "src/model_list.json").read_text(encoding="utf-8"))
        assignment = json.loads((REPO / "flm-dev-kit/FLM_Xclbin/kernel_assignment.json").read_text(encoding="utf-8"))
        res = select_models.select(mode="nightly", catalog=catalog, engines=engines, assignment=assignment)
        self.assertGreater(len(res["models"]), 10)
        self.assertLessEqual(len(res["models"]), select_models.NIGHTLY_CAP)
        res = select_models.select(mode="pr", catalog=catalog, engines=engines, assignment=assignment,
                                   families=engines.all_families())
        self.assertLessEqual(len(res["models"]), select_models.PR_CAP)


def run(**kw):
    base = {"id": 5, "run_number": 3, "run_attempt": 1, "path": resolve.BUILD_WORKFLOW,
            "repository": {"full_name": "ROCm/FastFlowLM"}, "head_repository": {"full_name": "ROCm/FastFlowLM"},
            "head_sha": SHA_A, "event": "pull_request", "head_branch": "feature", "status": "completed",
            "conclusion": "success"}
    base.update(kw)
    return base


class Resolve(unittest.TestCase):
    MATCH = {"repo": "ROCm/FastFlowLM", "head_sha": SHA_A, "head_repo": "ROCm/FastFlowLM", "event": "pull_request"}

    def test_pick_build_run_pins_everything(self):
        runs = [run(id=1, run_number=1), run(id=2, run_number=9, path=".github/workflows/evil.yml"),
                run(id=3, run_number=8, head_repository={"full_name": "fork/FastFlowLM"}),
                run(id=4, run_number=7, repository={"full_name": "fork/FastFlowLM"}),
                run(id=6, run_number=6, head_sha=SHA_B), run(id=7, run_number=5, event="push"),
                run(id=8, run_number=2)]
        self.assertEqual(resolve.pick_build_run(runs, **self.MATCH)["id"], 8)
        self.assertIsNone(resolve.pick_build_run(runs[1:6], **self.MATCH))
        # Nightly: the newest scheduled run on main, whatever its commit.
        nightly = [run(id=20, run_number=20, event="schedule", head_branch="main", head_sha=SHA_B),
                   run(id=21, run_number=21, event="schedule", head_branch="main", head_sha="bad")]
        self.assertEqual(resolve.pick_build_run(nightly, **{**self.MATCH, "event": "schedule",
                                                           "head_sha": ""})["id"], 20)

    def test_schedule_needs_main(self):
        m = {**self.MATCH, "event": "schedule"}
        self.assertFalse(resolve.run_matches(run(event="schedule", head_branch="main-fork"), **m))
        self.assertTrue(resolve.run_matches(run(event="schedule", head_branch="main"), **m))
        # A fork's branch named main does not pass for this repository's.
        self.assertFalse(resolve.run_matches(run(event="schedule", head_branch="main",
                                                 head_repository={"full_name": "x/FastFlowLM"}), **m))

    def test_pick_baseline(self):
        good = run(id=10, path=resolve.BENCH_WORKFLOW, event="schedule", head_branch="main")
        runs = {10: good, 11: {**good, "id": 11, "conclusion": "cancelled"},
                12: {**good, "id": 12, "head_repository": {"full_name": "fork/FastFlowLM"}},
                13: {**good, "id": 13, "path": ".github/workflows/copy.yml"},
                14: {**good, "id": 14, "event": "workflow_dispatch"}}

        def art(i, run_id, when, **kw):
            return {"id": i, "name": resolve.BASELINE_ARTIFACT, "expired": False, "created_at": when,
                    "workflow_run": {"id": run_id}, **kw}
        arts = [art(1, 10, "2026-11-01"), art(2, 11, "2026-11-05"), art(3, 12, "2026-11-06"),
                art(4, 13, "2026-11-07"), art(5, 14, "2026-11-08"), art(6, 10, "2026-11-09", expired=True)]
        self.assertEqual(resolve.pick_baseline(arts, runs, "ROCm/FastFlowLM")["artifact_id"], 1)
        self.assertIsNone(resolve.pick_baseline(arts[1:], runs, "ROCm/FastFlowLM"))

    def plan(self, **kw):
        p = {"schema": 1, "bench": "true", "head_sha": SHA_A, "families": "qwen3",
             "xclbin_models": '["gemma3:4b"]', "bench_models": '["gemma3:4b"]', "xclbins": "devkit-xclbins",
             "builds": {"lin-xrt": {"ok": "true", "package": "portable", "libs": "devkit-libs-lin-xrt"},
                        "lin-hrx": {"ok": "false", "package": "HRX-portable", "libs": ""}}}
        p.update(kw)
        return p

    def test_parse_plan(self):
        p = resolve.parse_plan(self.plan())
        self.assertTrue(p["bench"])
        self.assertEqual(p["xclbin_models"], ["gemma3:4b"])
        for bad in ({"schema": 2}, {"head_sha": "x"}, {"families": "a b"}, {"xclbin_models": '["$(x)"]'},
                    {"builds": {"lin-arm": {}}}, {"xclbins": "../x"},
                    {"builds": {"lin-xrt": {"ok": True, "package": "a/b", "libs": ""}}}):
            with self.subTest(bad=bad):
                with self.assertRaises(resolve.PlanError):
                    resolve.parse_plan(self.plan(**bad))

    def test_build_matrix(self):
        p = resolve.parse_plan(self.plan())
        index = {"portable": {"expired": False}, "devkit-libs-lin-xrt": {"expired": False},
                 "devkit-xclbins": {"expired": False}}
        models = [{"model": "qwen3:0.6b", "vlm": False}]
        entries, notes = resolve.build_matrix(p, index, ["lin-xrt", "lin-hrx", "win-xrt"], models,
                                              {"qwen3", "llama3"}, overlay_all=False)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["families"], "qwen3")
        self.assertEqual(entries[0]["safe"], "qwen3_0.6b")
        self.assertEqual(len(notes), 2)
        entries, _ = resolve.build_matrix(p, index, ["lin-xrt"], models, {"qwen3", "llama3"}, overlay_all=True)
        self.assertEqual(entries[0]["families"], "llama3,qwen3")
        index["devkit-xclbins"]["expired"] = True
        entries, notes = resolve.build_matrix(p, index, ["lin-xrt"], models, set(), overlay_all=False)
        self.assertEqual(entries, [])
        self.assertIn("expired", notes[0])


class DocVqa(unittest.TestCase):
    def test_pinned(self):
        self.assertRegex(docvqa_fetch.REVISION, r"^[0-9a-f]{40}$")
        self.assertIn(f"/resolve/{docvqa_fetch.REVISION}/", docvqa_fetch.shard_url())
        self.assertEqual(docvqa_fetch.sample([str(i) for i in range(10)], 3),
                         docvqa_fetch.sample([str(i) for i in range(10)], 3))

    def test_verify(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f")
            Path(path).write_bytes(b"abc")
            self.assertIn("bytes", docvqa_fetch.verify(path, sha256="0" * 64, size=4))
            self.assertIn("SHA-256", docvqa_fetch.verify(path, sha256="0" * 64, size=3))
            self.assertEqual(docvqa_fetch.verify(
                path, sha256="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", size=3), "")


def workflow(name: str) -> str:
    return (REPO / ".github/workflows" / name).read_text(encoding="utf-8")


class Workflows(unittest.TestCase):
    """The comment workflow runs with a write token on artifacts a fork can
    write: hold it to its invariants (devkit_check.py checks them too)."""

    def test_comment_workflow_invariants(self):
        import devkit_check
        errors: list = []
        devkit_check.check_bench_workflows(REPO, errors)
        self.assertEqual(errors, [])

    def test_check_catches_violations(self):
        import devkit_check
        bench, comment = workflow("devkit-bench.yml"), workflow("devkit-bench-comment.yml")
        cases = {
            "renamed": (bench.replace("name: Dev-kit Bench\n", "name: Dev-kit Benches\n", 1), comment),
            "write": (bench, comment.replace("  actions: read\n", "  actions: read\n  contents: write\n", 1)),
            "job perms": (bench, comment.replace("    runs-on: ubuntu-latest", "    permissions: {}\n"
                                                 "    runs-on: ubuntu-latest", 1)),
            "checkout": (bench, comment + "      - uses: actions/checkout@" + "0" * 40 + "\n"),
            "path": (bench, comment.replace("github.event.workflow_run.path == '.github/workflows/devkit-bench.yml'",
                                            "true", 1)),
            "inline": (bench, comment.replace('echo "number=$(jq', 'echo "${{ github.event.workflow_run.head_branch }}"\n'
                                              '          echo "number=$(jq', 1)),
            "bench write": (bench.replace("  actions: read\n", "  actions: read\n  pull-requests: write\n", 1),
                            comment),
        }
        for name, (b, c) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                (Path(d) / ".github/workflows").mkdir(parents=True)
                (Path(d) / devkit_check.BENCH_WORKFLOW).write_text(b, encoding="utf-8")
                (Path(d) / devkit_check.BENCH_COMMENT_WORKFLOW).write_text(c, encoding="utf-8")
                errors: list = []
                devkit_check.check_bench_workflows(Path(d), errors)
                self.assertTrue(errors, name)

    def test_bench_workflow_has_no_secrets_or_writes(self):
        text = workflow("devkit-bench.yml")
        self.assertNotIn("secrets.", text)
        self.assertNotRegex(text, r"(?m)^\s+(contents|pull-requests|issues|packages): write")
        self.assertNotIn("GITHUB_ENV", text)
        for line in text.splitlines():
            if "uses:" in line and "./" not in line:
                self.assertRegex(line, r"@[0-9a-f]{40}", line)


if __name__ == "__main__":
    unittest.main()
