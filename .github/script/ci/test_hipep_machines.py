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
        self.assertEqual((m["enabled"], m["max-model-gb"], m["max-tier"], m["changed-models"]),
                         (True, 0, "full", True))

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

    def test_max_tier(self):
        self.assertEqual(self.tiers("full"), [("stx-halo", "full"), ("npu-runners", "pr")])
        self.assertEqual(self.tiers("pr"), [("stx-halo", "pr"), ("npu-runners", "pr")])
        self.assertEqual(self.tiers("install"), [("stx-halo", "install"), ("npu-runners", "install")])


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
