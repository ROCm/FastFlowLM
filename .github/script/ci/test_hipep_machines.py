"""Tests for hipep_machines.py. Run: python3 .github/script/ci/test_hipep_machines.py"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hipep_machines  # noqa: E402

SCRIPT = HERE / "hipep_machines.py"


def config(*machines: dict) -> str:
    return json.dumps({"machines": list(machines)})


def machine(mid: str, *labels: str, **fields) -> dict:
    return {"id": mid, "runs-on": ["self-hosted", "windows", "x64", "oga-hipep", *labels], **fields}


NPU_RUNNERS = {"id": "npu-runners", "runs-on": ["self-hosted", "windows", "x64", "npu", "sweep", "oga-hipep"],
               "max-model-gb": 32, "max-tier": "pr", "changed-models": False}


class Load(unittest.TestCase):
    def test_repository_config(self):
        """The committed config loads, and enables at least one kind."""
        text = (hipep_machines.REPO_ROOT / hipep_machines.CONFIG).read_text(encoding="utf-8")
        machines = hipep_machines.load(text)
        self.assertTrue(any(m["enabled"] for m in machines))

    def test_defaults(self):
        (m,) = hipep_machines.load(config(machine("stx-halo", "stx-halo")))
        self.assertEqual((m["enabled"], m["max-model-gb"], m["max-tier"], m["changed-models"],
                          m["standby"]), (True, 0, "full", True, False))

    def test_shares_the_npu_runners(self):
        (m,) = hipep_machines.load(config(NPU_RUNNERS))
        self.assertEqual((m["runs-on"][3:], m["max-model-gb"], m["max-tier"], m["changed-models"]),
                         (["npu", "sweep", "oga-hipep"], 32, "pr", False))

    def test_rejects(self):
        bad = {
            "no self-hosted": {"id": "stx", "runs-on": ["windows", "oga-hipep"]},
            "labels not a list": {"id": "stx", "runs-on": "self-hosted"},
            "upper-case id": machine("STX", "stx"),
            "string limit": machine("stx", "stx", **{"max-model-gb": "16"}),
            "negative limit": machine("stx", "stx", **{"max-model-gb": -1}),
            "enabled as a string": machine("stx", "stx", enabled="false"),
            "unknown max-tier": machine("stx", "stx", **{"max-tier": "nightly"}),
            "changed-models as a string": machine("stx", "stx", **{"changed-models": "false"}),
            "standby as a string": machine("stx", "stx", standby="true"),
        }
        for why, entry in bad.items():
            with self.subTest(why):
                with self.assertRaises(ValueError):
                    hipep_machines.load(config(entry))
        with self.assertRaises(ValueError):
            hipep_machines.load(config(machine("stx", "stx"), machine("stx", "stx")))


class Pick(unittest.TestCase):
    MACHINES = hipep_machines.load(config(
        machine("stx-halo", "stx-halo"),
        machine("stx", "stx", enabled=False, **{"max-model-gb": 16}),
        NPU_RUNNERS))

    def ids(self, only: str) -> tuple[list[str], list[str]]:
        chosen, unknown = hipep_machines.pick(self.MACHINES, only)
        return [m["id"] for m in chosen], unknown

    def tiers(self, tier: str) -> list[tuple[str, str]]:
        chosen, _ = hipep_machines.pick(self.MACHINES, "")
        return [(m["id"], m["tier"]) for m in json.loads(hipep_machines.matrix(chosen, tier))]

    def test_enabled_by_default(self):
        self.assertEqual(self.ids(""), (["stx-halo", "npu-runners"], []))
        self.assertEqual(self.ids(" , "), (["stx-halo", "npu-runners"], []))

    def test_only(self):
        self.assertEqual(self.ids("stx"), (["stx"], []))
        self.assertEqual(self.ids("npu-runners, stx-halo"), (["stx-halo", "npu-runners"], []))
        self.assertEqual(self.ids("none"), ([], []))
        self.assertEqual(self.ids("stx,strix"), (["stx"], ["strix"]))

    def test_matrix(self):
        chosen, _ = hipep_machines.pick(self.MACHINES, "npu-runners")
        self.assertEqual(json.loads(hipep_machines.matrix(chosen, "pr")),
                         [{"id": "npu-runners",
                           "runs-on": ["self-hosted", "windows", "x64", "npu", "sweep", "oga-hipep"],
                           "max-model-gb": 32, "tier": "pr", "changed-models": ""}])
        self.assertEqual(hipep_machines.matrix([], "full"), "[]")

    def test_changed_models(self):
        """Only the kinds that test the change's models get them."""
        chosen, _ = hipep_machines.pick(self.MACHINES, "")
        rows = json.loads(hipep_machines.matrix(chosen, "pr", "gemma3-oga-hipep:4b,gpt-oss-oga-hipep:120b"))
        self.assertEqual([(m["id"], m["changed-models"]) for m in rows],
                         [("stx-halo", "gemma3-oga-hipep:4b,gpt-oss-oga-hipep:120b"), ("npu-runners", "")])

    def test_changed_models_go_to_every_kind_when_no_taker_runs(self):
        """A change's models are required: with only kinds that leave them to
        others running, those kinds test them (one larger than max-model-gb
        is a skip there)."""
        chosen, _ = hipep_machines.pick(self.MACHINES, "npu-runners")
        rows = json.loads(hipep_machines.matrix(chosen, "pr", "gemma3-oga-hipep:4b"))
        self.assertEqual([(m["id"], m["changed-models"]) for m in rows],
                         [("npu-runners", "gemma3-oga-hipep:4b")])

    def test_max_tier(self):
        self.assertEqual(self.tiers("full"), [("stx-halo", "full"), ("npu-runners", "pr")])
        self.assertEqual(self.tiers("pr"), [("stx-halo", "pr"), ("npu-runners", "pr")])
        self.assertEqual(self.tiers("install"), [("stx-halo", "install"), ("npu-runners", "install")])


# A catalog of oga-hipep models, as src/model_list.json has them (size in bytes).
CATALOG = json.dumps({"models": {
    "qwen3.8-oga-hipep": {"27b": {"name": "q38", "size": 16.9e9, "details": {"family": "qwen3.5"}}},
    "qwen3.5-oga-hipep": {"122b-a10b": {"name": "q35", "size": 71.8e9, "details": {"family": "qwen3.5"}}},
    "gemma4-it-oga-hipep": {"26b-a4b": {"name": "g4", "size": 16.6e9, "details": {"family": "gemma4e"}}},
    "gemma3-oga-hipep": {"4b": {"name": "g3", "size": 4.4e9, "details": {"family": "gemma3"}}},
    "gpt-oss-oga-hipep": {"120b": {"name": "go", "size": 69.3e9, "details": {"family": "gpt-oss"}}},
    "lfm2": {"1b": {"name": "lfm2", "size": 1e9, "details": {"family": "lfm2"}}},
}})


class Standby(unittest.TestCase):
    MACHINES = hipep_machines.load(config(machine("stx-halo", "stx-halo", standby=True), NPU_RUNNERS))

    def ids(self, tier="pr", changed="", families=(), models=(), only="", catalog=CATALOG):
        chosen, _ = hipep_machines.pick(self.MACHINES, only)
        kept, notes = hipep_machines.standby(chosen, tier, changed, set(families), list(models), catalog)
        return [m["id"] for m in kept]

    def test_the_shared_runners_test_a_plain_pull_request(self):
        self.assertEqual(self.ids(), ["npu-runners"])
        # A family whose model they hold.
        self.assertEqual(self.ids(families=["gemma3", "qwen3.5", "lfm2"]), ["npu-runners"])

    def test_the_full_tier_runs_every_kind(self):
        self.assertEqual(self.ids("full"), ["stx-halo", "npu-runners"])

    def test_new_weights_need_the_standby_kind(self):
        self.assertEqual(self.ids(changed="gemma3-oga-hipep:4b"), ["stx-halo", "npu-runners"])
        rows = json.loads(hipep_machines.matrix(
            [m for m in self.MACHINES], "pr", "gemma3-oga-hipep:4b"))
        self.assertEqual([(m["id"], m["changed-models"]) for m in rows],
                         [("stx-halo", "gemma3-oga-hipep:4b"), ("npu-runners", "")])

    def test_a_model_larger_than_the_others_hold_needs_it(self):
        self.assertEqual(self.ids(families=["gpt-oss"]), ["stx-halo", "npu-runners"])
        self.assertEqual(self.ids(models=["qwen3.5-oga-hipep:122b-a10b"]), ["stx-halo", "npu-runners"])
        # A dispatch naming only small models leaves it out, whatever the families.
        self.assertEqual(self.ids(families=["gpt-oss"], models=["gemma3-oga-hipep:4b"]), ["npu-runners"])

    def test_it_runs_when_no_other_kind_does(self):
        self.assertEqual(self.ids(only="stx-halo"), ["stx-halo"])

    def test_an_unreadable_catalog_keeps_it(self):
        self.assertEqual(self.ids(catalog=None), ["stx-halo", "npu-runners"])

    def test_a_family_missing_from_pr_gpu_gets_its_smallest_model(self):
        cat = json.loads(CATALOG)
        cat["models"]["new-oga-hipep"] = {"200b": {"name": "n", "size": 200e9, "details": {"family": "newfam"}}}
        self.assertEqual(self.ids(catalog=json.dumps(cat)), ["stx-halo", "npu-runners"])

    def test_a_pr_gpu_model_gone_from_the_catalog_falls_back_to_the_smallest(self):
        cat = json.loads(CATALOG)
        del cat["models"]["qwen3.8-oga-hipep"]  # PR_GPU's qwen3.5 pick; the 122B is left
        self.assertEqual(self.ids(catalog=json.dumps(cat)), ["stx-halo", "npu-runners"])

    def test_the_repository_catalog_and_config(self):
        """With the committed files a plain pull request runs on the shared
        runners alone, and one touching gpt-oss on the STX Halo too."""
        machines = hipep_machines.load(
            (hipep_machines.REPO_ROOT / hipep_machines.CONFIG).read_text(encoding="utf-8"))
        text = (hipep_machines.REPO_ROOT / hipep_machines.MODEL_LIST).read_text(encoding="utf-8")
        chosen, _ = hipep_machines.pick(machines, "")

        def ids(**kw):
            kept, _ = hipep_machines.standby(chosen, kw.get("tier", "pr"), kw.get("changed", ""),
                                             set(kw.get("families", ())), [], text)
            return [m["id"] for m in kept]

        self.assertEqual(ids(), ["npu-runners"])
        self.assertEqual(ids(families=["qwen2", "gemma3", "gemma4-12b", "laguna", "nemotron-h"]),
                         ["npu-runners"])
        self.assertEqual(ids(families=["gpt-oss"]), ["stx-halo", "npu-runners"])
        self.assertEqual(ids(families=["step3.5"]), ["stx-halo", "npu-runners"])
        self.assertEqual(ids(tier="full"), ["stx-halo", "npu-runners"])


class Cli(unittest.TestCase):
    def run_cli(self, *args: str, text: str) -> tuple[int, str, list[str]]:
        with tempfile.TemporaryDirectory() as tmp:
            path, target = Path(tmp) / "machines.json", Path(tmp) / "out"
            path.write_text(text, encoding="utf-8")
            res = subprocess.run([sys.executable, str(SCRIPT), "--config", str(path), "--github-output", *args],
                                 capture_output=True, text=True,
                                 env={**os.environ, "GITHUB_OUTPUT": str(target)})
            lines = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
        return res.returncode, res.stdout, lines

    def test_github_output(self):
        code, _, lines = self.run_cli("--tier", "full", "--only", "", "--changed-models", "gemma3-oga-hipep:4b",
                                      text=config(machine("stx-halo", "stx-halo"), NPU_RUNNERS))
        self.assertEqual(code, 0)
        self.assertEqual(len(lines), 1)
        key, value = lines[0].split("=", 1)
        self.assertEqual((key, [(m["id"], m["tier"], m["changed-models"]) for m in json.loads(value)]),
                         ("machines", [("stx-halo", "full", "gemma3-oga-hipep:4b"), ("npu-runners", "pr", "")]))

    def test_changed_models_without_a_taker_warn(self):
        code, out, lines = self.run_cli("--only", "npu-runners", "--changed-models", "gemma3-oga-hipep:4b",
                                        text=config(machine("stx-halo", "stx-halo"), NPU_RUNNERS))
        self.assertEqual(code, 0)
        self.assertEqual([m["changed-models"] for m in json.loads(lines[0].split("=", 1)[1])],
                         ["gemma3-oga-hipep:4b"])
        self.assertIn("::warning::", out)
        code, out, lines = self.run_cli("--only", "none", "--changed-models", "gemma3-oga-hipep:4b",
                                        text=config(machine("stx-halo", "stx-halo")))
        self.assertEqual((code, lines), (0, ["machines=[]"]))
        self.assertIn("::warning::", out)

    def test_standby_prints_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat = Path(tmp) / "model_list.json"
            cat.write_text(CATALOG, encoding="utf-8")
            text = config(machine("stx-halo", "stx-halo", standby=True), NPU_RUNNERS)
            code, out, lines = self.run_cli("--catalog", str(cat), text=text)
            self.assertEqual([m["id"] for m in json.loads(lines[0].split("=", 1)[1])], ["npu-runners"])
            self.assertIn("stx-halo (standby): not needed", out)
            code, out, lines = self.run_cli("--catalog", str(cat), "--families", "gpt-oss", text=text)
            self.assertEqual([m["id"] for m in json.loads(lines[0].split("=", 1)[1])],
                             ["stx-halo", "npu-runners"])
            self.assertIn("gpt-oss-oga-hipep:120b (69 GB) is larger", out)

    def test_none_writes_an_empty_matrix(self):
        code, _, lines = self.run_cli("--only", "none", text=config(machine("stx-halo", "stx-halo")))
        self.assertEqual((code, lines), (0, ["machines=[]"]))

    def test_unknown_name_warns(self):
        code, out, lines = self.run_cli("--only", "strix", text=config(machine("stx-halo", "stx-halo")))
        self.assertEqual((code, lines), (0, ["machines=[]"]))
        self.assertIn("::warning::", out)

    def test_bad_config_fails(self):
        code, out, lines = self.run_cli(text=config(machine("stx", "stx", **{"max-tier": "nightly"})))
        self.assertEqual((code, lines), (2, []))
        self.assertIn("::error", out)


if __name__ == "__main__":
    unittest.main()
