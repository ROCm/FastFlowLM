"""Tests for npu_test/plan.py's tables: python3 .github/script/ci/test_plan.py

Here rather than beside npu_test so the changes job runs them with the
change-detection tests: both tables must follow src/model_list.json."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import changes  # noqa: E402
from npu_test.plan import NON_CHAT_FAMILIES, PR_CHAT, Model, Plan  # noqa: E402


def real_catalog() -> dict[str, dict]:
    return changes.load_catalog((changes.REPO_ROOT / changes.MODEL_LIST).read_text(encoding="utf-8"))


class Tables(unittest.TestCase):
    def test_pr_chat_names_every_chat_family(self):
        """A family missing from PR_CHAT gets its smallest model on every pull
        request, however large: name it, with None to leave it to the changes
        that touch it."""
        cat = real_catalog()
        npu = {changes.family_of(e) for mid, e in cat.items() if not changes.runs_on_gpu(mid, e)}
        self.assertEqual(npu - NON_CHAT_FAMILIES - set(PR_CHAT), set())
        self.assertEqual(set(PR_CHAT) - npu, set())

    def test_pr_chat_models_are_npu_models_of_their_family(self):
        cat = real_catalog()
        for family, model in PR_CHAT.items():
            if model is None:
                continue
            with self.subTest(family=family):
                self.assertIn(model, cat)
                self.assertEqual(changes.family_of(cat[model]), family)
                self.assertFalse(changes.runs_on_gpu(model, cat[model]))


def plan(tier="pr", families=(), **kw) -> Plan:
    cat = {m.id: m for m in (
        Model(id="qwen3:0.6b", family="qwen3", size=0.6e9),
        Model(id="big:120b", family="big", size=120e9),
        Model(id="big:8b", family="big", size=8e9),
        Model(id="d3:lite", family="d3", size=0.8e9),
        Model(id="d3:huge", family="d3", size=60e9),
        Model(id="new-oga-hipep:70b", family="qwen3", size=70e9, backend="oga-hipep"),
        Model(id="qwen3.5-oga-hipep:9b", family="qwen3.5", size=7.5e9, backend="oga-hipep"),
    )}
    return Plan(cat, tier, list(families), **kw)


class Required(unittest.TestCase):
    def test_a_new_model_runs_even_when_its_family_is_not_touched(self):
        p = plan(required=["big:120b", "d3:huge"])
        # "big" is in no PR_CHAT entry, so it gets its smallest model too.
        self.assertEqual(p.chat(), ["big:8b", "qwen3:0.6b", "big:120b"])
        self.assertEqual(plan().chat(), ["big:8b", "qwen3:0.6b"])
        self.assertEqual(p.suite("decision"), ["d3:lite", "d3:huge"])
        self.assertEqual(p.missing(["chat", "decision"])[-1], "big:120b")

    def test_every_tier_tests_a_new_model(self):
        for tier in ("install", "pr", "full"):
            with self.subTest(tier=tier):
                p = plan(tier, required=["big:120b", "d3:huge"])
                self.assertIn("big:120b", p.planned(["chat"]))
                self.assertIn("d3:huge", p.planned(["decision"]))
        self.assertEqual(plan("install").planned(["chat", "decision"]), ["qwen3:0.6b", "d3:lite"])

    def test_a_new_model_runs_past_models(self):
        p = plan(only_models=["qwen3:0.6b"], required=["big:120b"])
        self.assertEqual(p.chat(), ["qwen3:0.6b", "big:120b"])

    def test_a_new_gpu_model_runs_beside_the_default_one(self):
        p = plan(only_models=["qwen3.5-oga-hipep:9b"], backend="oga-hipep",
                 required=["new-oga-hipep:70b"])
        self.assertEqual(p.chat(), ["qwen3.5-oga-hipep:9b", "new-oga-hipep:70b"])
        self.assertEqual(p.suite("api"), p.chat())
        # The NPU run leaves the GPU model to its own job.
        npu = plan(required=["new-oga-hipep:70b"])
        self.assertNotIn("new-oga-hipep:70b", npu.chat())
        self.assertFalse(npu.backend_serves("new-oga-hipep:70b"))

    def test_a_required_gpu_model_runs_beside_each_familys_pick(self):
        """A GPU backend run without --models tests PR_GPU's pick of each
        family (gemma3's only when touched), and the models the change adds or
        edits, on every tier."""
        cat = {m.id: m for m in (
            Model(id="qwen3.8-oga-hipep:27b", family="qwen3.5", size=17e9, backend="oga-hipep"),
            Model(id="qwen3.5-oga-hipep:9b", family="qwen3.5", size=7.5e9, backend="oga-hipep"),
            Model(id="gemma3-oga-hipep:4b", family="gemma3", size=4e9, backend="oga-hipep"),
            Model(id="gemma3-oga-hipep:27b", family="gemma3", size=20e9, backend="oga-hipep"),
            Model(id="qwen3:0.6b", family="qwen3", size=0.6e9),
        )}

        def chat(tier, families=(), required=()):
            return Plan(cat, tier, list(families), backend="oga-hipep", required=list(required)).chat()

        self.assertEqual(chat("pr"), ["qwen3.8-oga-hipep:27b"])
        self.assertEqual(chat("pr", ["gemma3"]), ["gemma3-oga-hipep:4b", "qwen3.8-oga-hipep:27b"])
        for tier in ("install", "pr"):
            with self.subTest(tier=tier):
                self.assertEqual(chat(tier, required=["gemma3-oga-hipep:27b", "qwen3:0.6b"]),
                                 ["qwen3.8-oga-hipep:27b", "gemma3-oga-hipep:27b"])
        self.assertEqual(len(chat("full")), 4)
        # A PR_GPU model gone from the catalog: the family's smallest stands in.
        del cat["qwen3.8-oga-hipep:27b"]
        self.assertEqual(chat("pr"), ["qwen3.5-oga-hipep:9b"])

if __name__ == "__main__":
    unittest.main()
