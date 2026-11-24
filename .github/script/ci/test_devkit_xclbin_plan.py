# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for devkit_xclbin_plan.py: python3 -m unittest discover -s ci -p 'test_*.py' -t ci"""

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
import devkit_xclbin_plan as xp  # noqa: E402

X = xp.XCLBIN
UNITS = {
    "Dequant": [],
    "MM/host": ["GENERIC", "VISION"],
    "Fam/decoding": ["A", "B", "C"],
    "Fam/prefill": [],
    "Other/deep/unit": [],
}
ASSIGNMENT = {
    "m:1b": {"folder": "M-1B", "kernels": {"dequant.xclbin": "xclbins-Dequant",
                                            "mm.xclbin": "xclbins-MM_host_GENERIC",
                                            "layer.xclbin": "xclbins-Fam_decoding_A"}},
    "m:3b": {"folder": "M-3B", "kernels": {"layer.xclbin": "xclbins-Fam_decoding_B",
                                            "attn.xclbin": "xclbins-Fam_prefill"}},
}
CANARY = [("Dequant", None), ("MM/host", "GENERIC")]


def plan(mode="changed", files=None, **kw):
    kw.setdefault("canary", CANARY)
    kw.setdefault("base_units", UNITS)
    kw.setdefault("base_assignment", ASSIGNMENT)
    return xp.make_plan(mode, UNITS, ASSIGNMENT, files=files, **kw)


def builds(result):
    return [(b["unit"], b["model_type"]) for b in result["units"]]


class Modes(unittest.TestCase):
    def test_all_builds_every_model_type(self):
        r = plan("all")
        self.assertEqual(len(r["units"]), 1 + 2 + 3 + 1 + 1)
        self.assertIn({"unit": "MM/host", "model_type": "VISION", "artifact_name": "xclbins-MM_host_VISION"},
                      r["units"])
        self.assertIn({"unit": "Dequant", "model_type": None, "artifact_name": "xclbins-Dequant"}, r["units"])
        self.assertEqual(r["touched_models"], ["m:1b", "m:3b"])

    def test_assigned_builds_only_assigned_artifacts(self):
        self.assertEqual(builds(plan("assigned")), [
            ("Dequant", None), ("Fam/decoding", "A"), ("Fam/decoding", "B"),
            ("Fam/prefill", None), ("MM/host", "GENERIC")])

    def test_unknown_files_plan_all(self):
        r = plan("changed", None)
        self.assertEqual(r["mode"], "all")
        self.assertEqual(len(r["units"]), 8)
        # Sharded as mode all, not packed into the changed mode's few shards.
        r = plan("changed", None, max_shards=2, per_shard=8, fallback_shards=16)
        self.assertEqual(len(r["shards"]), 8)
        self.assertEqual(r["bench_models"], r["touched_models"])


class Changed(unittest.TestCase):
    def test_file_in_unit_builds_its_assigned_model_types(self):
        r = plan(files=[f"{X}/Fam/decoding/kernel/layer.cc"])
        self.assertEqual(builds(r), [("Fam/decoding", "A"), ("Fam/decoding", "B")])
        self.assertEqual(r["touched_models"], ["m:1b", "m:3b"])

    def test_unit_with_no_assigned_model_type_builds_all(self):
        r = plan(files=[f"{X}/Other/deep/unit/Makefile", f"{X}/Dequant/iron/x.py"])
        self.assertEqual(builds(r), [("Dequant", None), ("Other/deep/unit", None)])
        self.assertEqual(r["touched_models"], ["m:1b"])

    def test_shared_dir_selects_every_unit_below(self):
        # Fam/makefiles has no unit: everything under Fam/.
        r = plan(files=[f"{X}/Fam/makefiles/common.mk"])
        self.assertEqual(builds(r), [("Fam/decoding", "A"), ("Fam/decoding", "B"), ("Fam/prefill", None)])
        # Other/deep/common/: nothing below it or Other/deep but the unit.
        self.assertEqual(builds(plan(files=[f"{X}/Other/deep/common/h.hpp"])), [("Other/deep/unit", None)])
        # A shared file at the root: every unit (MM/host only for GENERIC,
        # the one MODEL_TYPE of it the assignment uses).
        self.assertEqual(len(plan(files=[f"{X}/some_shared.py"])["units"]), 6)

    def test_files_no_bitstream_reads_select_nothing(self):
        for path in (f"{X}/README.md", f"{X}/Fam/decoding/README.md", f"{X}/kernel_assignment.md",
                     f"{X}/.gitignore", f"{X}/Fam/decoding/.gitignore",
                     f"{X}/host_common/buffer.hpp", f"{X}/setup.sh", f"{X}/dimhelper.py",
                     "src/server/server.cpp", "flm-dev-kit/FLM_DLL/detail/q4nx/q4nx.cpp"):
            with self.subTest(path=path):
                self.assertEqual(plan(files=[path])["units"], [])

    def test_toolchain_and_ci_build_the_canary(self):
        for path in (f"{X}/toolchain.lock", f"{X}/toolchain.image", "docker/aie-build.Dockerfile",
                     ".github/workflows/devkit-xclbins.yml", ".github/actions/devkit-build-xclbins/action.yml",
                     ".github/script/ci/devkit_xclbin_plan.py", ".github/workflows/build-and-test.yml"):
            with self.subTest(path=path):
                r = plan(files=[path])
                self.assertEqual(builds(r), [("Dequant", None), ("MM/host", "GENERIC")])
                # The canary serves m:1b, but nothing of it changed.
                self.assertEqual((r["touched_models"], r["bench_models"]), (["m:1b"], []))

    def test_ci_no_build_uses_adds_no_canary(self):
        # changes.py OUTSIDE_BUILD_AND_TEST: no dev-kit area there, no canary here.
        for path in (".github/script/ci/merge_reports.py", ".github/script/ci/test_changes.py",
                     ".github/script/ci/check_internal_refs.py"):
            with self.subTest(path=path):
                self.assertEqual(builds(plan(files=[path])), [])
                self.assertEqual(builds(plan(files=[path, f"{X}/Fam/prefill/kernel/k.cc"])), [("Fam/prefill", None)])

    def test_assignment_change_builds_exactly_the_changed_models_artifacts(self):
        base = json.loads(json.dumps(ASSIGNMENT))
        base["m:3b"]["kernels"]["layer.xclbin"] = "xclbins-Fam_decoding_C"
        r = plan(files=[f"{X}/kernel_assignment.json"], base_assignment=base)
        self.assertEqual(builds(r), [("Fam/decoding", "B"), ("Fam/prefill", None)])
        self.assertEqual(r["touched_models"], ["m:3b"])
        self.assertEqual(r["bench_models"], ["m:3b"])
        # A model new at head (no file at the base: every model is new).
        r = plan(files=[f"{X}/kernel_assignment.json"], base_assignment={})
        self.assertEqual(len(r["units"]), 5)
        # A changed key order is no change.
        same = {m: dict(reversed(list(e.items()))) for m, e in ASSIGNMENT.items()}
        self.assertEqual(plan(files=[f"{X}/kernel_assignment.json"], base_assignment=same)["units"], [])

    def test_rtp_lock_change_builds_the_changed_stanzas_artifacts(self):
        lock = f"{X}/rtp_addresses.lock.txt"
        self.assertEqual(lock, xp.RTP_LOCK)
        base = "[xclbins-Fam_decoding_A/rtp_addresses.txt]\nR 1\n\n[xclbins-Fam_decoding_B/rtp_addresses.txt]\nR 2\n"
        head = base.replace("R 2", "R 3") + "\n[xclbins-Fam_prefill/rtp_addresses.txt]\nR 4\n"
        r = plan(files=[lock], base_lock=base, head_lock=head)
        # B changed, Fam/prefill added; A unchanged. Not a source change.
        self.assertEqual(builds(r), [("Fam/decoding", "B"), ("Fam/prefill", None)])
        self.assertEqual(r["bench_models"], [])
        # No lock at the base: every stanza; a removed stanza still rebuilds.
        self.assertEqual(len(plan(files=[lock], base_lock=None, head_lock=head)["units"]), 3)
        self.assertEqual(builds(plan(files=[lock], base_lock=base, head_lock=base.split("\n\n")[0])),
                         [("Fam/decoding", "B")])
        self.assertEqual(plan(files=[lock], base_lock=base, head_lock=base)["units"], [])

    def test_build_units_change_builds_changed_units(self):
        base = dict(UNITS, **{"Fam/decoding": ["A", "B"]})
        del base["Fam/prefill"]
        r = plan(files=[f"{X}/build_units.json"], base_units=base)
        # Fam/decoding: its new C; Fam/prefill: new, its one build.
        self.assertEqual(builds(r), [("Fam/decoding", "C"), ("Fam/prefill", None)])
        self.assertEqual(r["bench_models"], ["m:3b"])

    def test_build_units_new_model_type_is_built(self):
        # Fam/decoding gains C, which no model uses yet: C is the build the
        # change exists for, not the assigned A and B.
        base = dict(UNITS, **{"Fam/decoding": ["A", "B"]})
        r = plan(files=[f"{X}/build_units.json"], base_units=base)
        self.assertEqual(builds(r), [("Fam/decoding", "C")])
        # A removed MODEL_TYPE adds none: the unit's assigned ones.
        base = dict(UNITS, **{"Fam/decoding": ["A", "B", "C", "D"]})
        r = plan(files=[f"{X}/build_units.json"], base_units=base)
        self.assertEqual(builds(r), [("Fam/decoding", "A"), ("Fam/decoding", "B")])

    def test_removed_unit_selects_nothing(self):
        base = dict(UNITS, **{"Fam/gone": [], "conv1d": []})
        for path in (f"{X}/Fam/gone/Makefile", f"{X}/Fam/gone/kernel/x.cc", f"{X}/conv1d/Makefile"):
            with self.subTest(path=path):
                self.assertEqual(plan(files=[path], base_units=base)["units"], [])
        # Without the base, the old fallback: every unit under Fam/.
        self.assertEqual(len(plan(files=[f"{X}/Fam/gone/Makefile"], base_units={})["units"]), 3)
        self.assertEqual(xp.unit_for(f"{X}/Fam/gone/x/y.cc", UNITS, {"Fam/gone"}), [])

    def test_bench_models_are_source_and_assignment_changes(self):
        # A source change: every user of the rebuilt build.
        r = plan(files=[f"{X}/MM/host/kernel/mm.cc"])
        self.assertEqual(r["bench_models"], ["m:1b"])
        # A new model reusing built kernels: only that model, though its
        # artifacts serve others too.
        head = dict(ASSIGNMENT, **{"m:new": {"folder": "NEW", "kernels": {"mm.xclbin": "xclbins-MM_host_GENERIC"}}})
        r = xp.make_plan("changed", UNITS, head, files=[f"{X}/kernel_assignment.json"],
                         base_units=UNITS, base_assignment=ASSIGNMENT, canary=CANARY)
        self.assertEqual(builds(r), [("MM/host", "GENERIC")])
        self.assertEqual((r["touched_models"], r["bench_models"]), (["m:1b", "m:new"], ["m:new"]))

    def test_shipped_kernel_builds_its_assigned_artifact(self):
        # Every model using the folder; models sharing it put the same
        # artifact in a shared slot (devkit_check enforces it).
        assignment = dict(ASSIGNMENT, **{"f:1b": {"folder": "M-1B", "kernels": {
            "layer.xclbin": "xclbins-Fam_decoding_A", "extra.xclbin": "xclbins-Fam_decoding_C"}}})
        r = xp.make_plan("changed", UNITS, assignment, files=["src/xclbins/M-1B/layer.xclbin"],
                         canary=CANARY, base_units=UNITS, base_assignment=assignment)
        self.assertEqual(builds(r), [("Fam/decoding", "A")])
        self.assertEqual(r["touched_models"], ["f:1b", "m:1b"])
        self.assertEqual(r["bench_models"], [])  # no source changed
        r = xp.make_plan("changed", UNITS, assignment, files=["src/xclbins/M-1B/extra.xclbin"],
                         canary=CANARY, base_units=UNITS, base_assignment=assignment)
        self.assertEqual(builds(r), [("Fam/decoding", "C")])
        # A slot or folder no model assigns, and files that are no kernel: nothing.
        for path in ("src/xclbins/M-1B/conv.xclbin", "src/xclbins/Nope/layer.xclbin",
                     "src/xclbins/M-1B/notes.txt", "src/xclbins/layer.xclbin"):
            with self.subTest(path=path):
                self.assertEqual(builds(plan(files=[path])), [])
        self.assertTrue(any("no " in x and "assigns: nothing to build" in x
                            for x in plan(files=["src/xclbins/M-1B/conv.xclbin"])["reasons"]))
        # Beside a source change: both, and only the source's users bench.
        r = plan(files=["src/xclbins/M-3B/attn.xclbin", f"{X}/Dequant/kernel/k.cc"])
        self.assertEqual(builds(r), [("Dequant", None), ("Fam/prefill", None)])
        self.assertEqual(r["bench_models"], ["m:1b"])

    def test_unit_for_walks_up_to_the_nearest_unit(self):
        self.assertEqual(xp.unit_for(f"{X}/MM/host/kernel/a/b.cc", UNITS), ["MM/host"])
        self.assertEqual(xp.unit_for(f"{X}/MM/host/Makefile", UNITS), ["MM/host"])
        self.assertEqual(xp.unit_for(f"{X}/MM/kernel/mm.cc", UNITS), ["MM/host"])


class Sharding(unittest.TestCase):
    def items(self, n):
        return [{"unit": f"u{i:03}", "model_type": None, "artifact_name": f"xclbins-u{i:03}"} for i in range(n)]

    def test_shards(self):
        def sizes(n, most, per=0):
            return [len(s) for s in xp.shard(self.items(n), most, per)]
        self.assertEqual(sizes(0, 4), [])
        self.assertEqual(sizes(1, 4), [1])
        self.assertEqual(sizes(10, 4), [3, 3, 2, 2])
        self.assertEqual(sizes(3, 4), [1, 1, 1])
        self.assertEqual(sizes(10, 4, 8), [5, 5])
        self.assertEqual(sizes(8, 4, 8), [8])
        self.assertEqual(sizes(184, 16), [12] * 8 + [11] * 8)
        self.assertEqual(sizes(100, 4, 8), [25] * 4)

    def test_shards_keep_order_and_cover_everything(self):
        items = self.items(37)
        self.assertEqual([b for s in xp.shard(items, 5) for b in s], items)


class Canary(unittest.TestCase):
    def test_canary_from_pins(self):
        pins = "# comment\nabc 1 Dequant - build/xclbins/dequant.xclbin\nabd 2 MM/host GENERIC build/x.xclbin\n" \
               "abe 3 Gone - build/x.xclbin\n"
        self.assertEqual(xp.load_canary(pins, UNITS), [("Dequant", None), ("MM/host", "GENERIC")])
        self.assertEqual(xp.load_canary("", {"Dequant": []}), [])

    def test_real_pins_name_real_builds(self):
        units = xp.load_units(xp.read(xp.REPO_ROOT, xp.BUILD_UNITS))
        canary = xp.load_canary(xp.read(xp.REPO_ROOT, xp.PINS), units)
        self.assertTrue(canary)
        for unit, mt in canary:
            self.assertIn(unit, units)
            self.assertTrue(mt is None or mt in units[unit])


class Cli(unittest.TestCase):
    """End to end in a scratch repository: base...head, GITHUB_OUTPUT."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "ci@example.com")
        self.git("config", "user.name", "ci")
        self.write(xp.BUILD_UNITS, json.dumps({"units": {u: {"model_types": m} for u, m in UNITS.items()}}))
        self.write(xp.ASSIGNMENT, json.dumps({"models": ASSIGNMENT}))
        for unit in UNITS:
            self.write(f"{X}/{unit}/Makefile", "bitstream:\n")
        self.commit("base")
        self.base = self.git("rev-parse", "HEAD").strip()

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              capture_output=True, text=True).stdout

    def write(self, path, text):
        p = self.repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def commit(self, msg):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg)

    def run_cli(self, *args, env=None):
        res = subprocess.run([sys.executable, str(HERE / "devkit_xclbin_plan.py"), "--repo", str(self.repo), *args],
                             capture_output=True, text=True, env=env)
        self.assertEqual(res.returncode, 0, res.stderr)
        return json.loads(res.stdout)

    def test_diff_from_merge_base(self):
        self.git("checkout", "-q", "-b", "pr")
        self.write(f"{X}/Fam/prefill/kernel.cc", "x")
        changed = json.loads(json.dumps(ASSIGNMENT))
        changed["m:1b"]["kernels"]["mm.xclbin"] = "xclbins-MM_host_VISION"
        self.write(xp.ASSIGNMENT, json.dumps({"models": changed}))
        self.commit("pr")
        head = self.git("rev-parse", "HEAD").strip()
        # main moves on after the branch point: not the pull request's change.
        self.git("checkout", "-q", "main")
        self.write(f"{X}/Dequant/kernel.cc", "y")
        self.commit("main")
        main = self.git("rev-parse", "HEAD").strip()
        r = self.run_cli("--mode", "changed", "--base", main, "--head", head)
        self.assertEqual(r["mode"], "changed")
        self.assertEqual(builds(r), [("Dequant", None), ("Fam/decoding", "A"), ("Fam/prefill", None),
                                     ("MM/host", "VISION")])
        self.assertEqual(r["touched_models"], ["m:1b", "m:3b"])

    def test_unknown_base_and_github_output(self):
        out = self.repo / "gh_output"
        env = dict(os.environ, GITHUB_OUTPUT=str(out))
        r = self.run_cli("--mode", "changed", "--base", "0" * 40, "--shards", "2", "--per-shard", "1",
                         "--fallback-shards", "3", "--github-output", env=env)
        self.assertEqual((r["mode"], len(r["units"]), len(r["shards"])), ("all", 8, 3))
        lines = dict(l.split("=", 1) for l in out.read_text().splitlines())
        self.assertEqual(json.loads(lines["matrix"]), {"shard": [0, 1, 2]})
        self.assertEqual((lines["count"], lines["empty"], lines["mode"]), ("8", "false", "all"))
        self.assertEqual(json.loads(lines["plan"]), r)
        self.assertEqual(json.loads(lines["touched-models"]), ["m:1b", "m:3b"])
        self.assertEqual(json.loads(lines["bench-models"]), ["m:1b", "m:3b"])
        # Nothing to build: an empty matrix the caller skips.
        out.unlink()
        self.run_cli("--mode", "changed", "--files", "README.md", "--github-output", env=env)
        lines = dict(l.split("=", 1) for l in out.read_text().splitlines())
        self.assertEqual((lines["empty"], lines["matrix"]), ("true", '{"shard":[]}'))


if __name__ == "__main__":
    unittest.main()
