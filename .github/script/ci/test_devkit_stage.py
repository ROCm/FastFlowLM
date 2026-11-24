# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for devkit_stage.py: python3 -m unittest discover -s ci -p 'test_*.py' -t ci

A throwaway checkout (the dev kit's JSON and Makefiles, src/lib with real
prebuilt libraries from this repository, src/xclbins, the catalog), the
dev-kit build artifacts as the stage job downloads them, and
devkit_stage.main."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import devkit_stage as st  # noqa: E402
import devkit_swap as ds  # noqa: E402

REPO = HERE.parents[2]
HEAD, BASE = "a" * 40, "b" * 40


def prebuilt(runtime: str, name: str) -> Path:
    path = REPO / "src" / "lib" / runtime / name
    if not path.is_file():
        raise unittest.SkipTest(f"{path} is not in the checkout")
    return path


def makefile(lib: str) -> str:
    return (f"ifeq ($(OS),Linux)\nTARGET = $(LIB_DIR)/lib{lib}.so\nelse\nTARGET = $(BUILD_DIR)/{lib}.dll\n"
            "endif\nLDFLAGS += -Wl,-Bsymbolic-functions\n")


ENGINES = {"shared": ["mha"], "engines": {
    "qwen3_npu": {"families": ["qwen3"], "models": ["qwen3:0.6b"], "uses": []}},
    "served_elsewhere": {"smolvla:500m": "a pybind module"}}
ASSIGNMENT = {"models": {
    "qwen3:0.6b": {"folder": "Qwen3-0.6B-NPU2", "kernels": {"attn.xclbin": "xclbins-a", "mm.xclbin": "xclbins-m"}},
    "smolvla:500m": {"folder": "SmolVLA", "kernels": {"expert.xclbin": "xclbins-s"}}}}
CATALOG = {"models": {
    "qwen3": {"0.6b": {"name": "Qwen3-0.6B-NPU2", "details": {"family": "qwen3"}}},
    "d3": {"lite": {"name": "D3-Lite-NPU2", "details": {"family": "d3"}}}}}


class StageCase(unittest.TestCase):
    """qwen3_npu and the shared mha ship in src/lib, Qwen3-0.6B-NPU2's kernels
    in src/xclbins. add_new_model() adds newm_npu (a new engine) and newm:1b
    (a new model, kernels in NewM-1B-NPU2), whose binaries src/ lacks."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.repo = self.tmp / "repo"
        self.write_json("flm-dev-kit/FLM_DLL/engines.json", ENGINES)
        self.write_json("flm-dev-kit/FLM_Xclbin/kernel_assignment.json", ASSIGNMENT)
        self.write_json("src/model_list.json", CATALOG)
        for unit in ("qwen3_npu", "mha"):
            self.write(f"flm-dev-kit/FLM_DLL/detail/{unit}/Makefile", makefile(unit))
        for rt in ("xrt", "hrx"):
            for name in ("libmha.so", "libqwen3_npu.so", "mha.dll", "mha.lib", "qwen3_npu.dll", "qwen3_npu.lib"):
                self.copy(prebuilt("xrt", name), f"src/lib/{rt}/{name}")
        self.copy(prebuilt("xrt", "libaiebu.a"), "src/lib/xrt/libaiebu.a")
        self.write("src/xclbins/Qwen3-0.6B-NPU2/attn.xclbin", "old attn")
        self.write("src/xclbins/Qwen3-0.6B-NPU2/mm.xclbin", "old mm")
        # The dev-kit builds' artifacts, as download-artifact lays them out.
        self.libs = self.tmp / "devkit-libs"
        for short, rt in (("lin", "xrt"), ("lin", "hrx"), ("win", "xrt"), ("win", "hrx")):
            d = self.libs / f"devkit-libs-{short}-{rt}"
            d.mkdir(parents=True)
            names = (("libmha.so", "libqwen3_npu.so") if short == "lin"
                     else ("mha.dll", "mha.lib", "qwen3_npu.dll", "qwen3_npu.lib"))
            for name in names:
                shutil.copyfile(prebuilt("xrt", name), d / name)
            # The new engine: the build of another engine stands in for it.
            for name, src in ((("libnewm_npu.so", "libqwen3_npu.so"),) if short == "lin" else
                              (("newm_npu.dll", "qwen3_npu.dll"), ("newm_npu.lib", "qwen3_npu.lib"))):
                shutil.copyfile(prebuilt("xrt", src), d / name)
        self.xclbins = self.tmp / "devkit-xclbins"
        for rel, text in (("Qwen3-0.6B-NPU2/attn.xclbin", "new attn"), ("NewM-1B-NPU2/layer.xclbin", "new layer"),
                          ("NewM-1B-NPU2/stray.xclbin", "not assigned"), ("SmolVLA/expert.xclbin", "expert")):
            p = self.xclbins / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        self.out = self.tmp / "out"

    def add_new_model(self) -> None:
        engines, assignment, catalog = (json.loads(json.dumps(x)) for x in (ENGINES, ASSIGNMENT, CATALOG))
        engines["engines"]["newm_npu"] = {"families": ["newm"], "models": ["newm:1b"], "uses": []}
        assignment["models"]["newm:1b"] = {"folder": "NewM-1B-NPU2", "kernels": {"layer.xclbin": "xclbins-n"}}
        catalog["models"]["newm"] = {"1b": {"name": "NewM-1B-NPU2", "details": {"family": "newm"}}}
        self.write_json("flm-dev-kit/FLM_DLL/engines.json", engines)
        self.write_json("flm-dev-kit/FLM_Xclbin/kernel_assignment.json", assignment)
        self.write_json("src/model_list.json", catalog)
        self.write("flm-dev-kit/FLM_DLL/detail/newm_npu/Makefile", makefile("newm_npu"))

    def write(self, rel: str, text: str) -> None:
        p = self.repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def write_json(self, rel: str, data: dict) -> None:
        self.write(rel, json.dumps(data))

    def copy(self, src: Path, rel: str) -> None:
        p = self.repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, p)

    def stage(self, *extra: str, families: str = "qwen3", kernels: list | None = None,
              libs: bool = True, xclbins: bool = True) -> tuple[int, str, dict]:
        if self.out.exists():
            shutil.rmtree(self.out)
        args = ["--repo", str(self.repo), "--out", str(self.out), "--head-sha", HEAD, "--base-sha", BASE,
                "--families", families, "--kernel-models", json.dumps(kernels if kernels is not None else [])]
        if libs:
            for d in sorted(self.libs.iterdir()):
                args += ["--libs", f"{d.name[len('devkit-libs-'):]}={d}"]
        if xclbins:
            args += ["--xclbins", str(self.xclbins)]
        buf = io.StringIO()
        env = {"GITHUB_RUN_ID": "100", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_STEP_SUMMARY": ""}
        with contextlib.redirect_stdout(buf), mock.patch.dict(os.environ, env):
            code = st.main(args + list(extra))
        text = buf.getvalue()
        outputs = dict(line.split("=", 1) for line in text.splitlines()
                       if "=" in line and line.split("=", 1)[0] in ("staged", "suites", "hrx-suites", "families",
                                                                    "required", "count"))
        return code, text, outputs

    def manifest(self, os_name: str, runtime: str) -> ds.Manifest:
        path = self.out / f"manifest-{os_name}-{runtime}" / ds.MANIFEST_FILE
        return ds.parse_manifest(path.read_bytes(), os_name, runtime)

    def tree(self, short: str, runtime: str) -> dict[str, str]:
        """repo path -> sha256 of the stage tree, checked against its list."""
        root = self.out / f"stage-{short}-{runtime}"
        listed = {}
        for line in (root / st.SUMS_FILE).read_text(encoding="utf-8").splitlines():
            digest, path = line.split("  ", 1)
            listed[path] = digest
        present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {st.SUMS_FILE}
        self.assertEqual(present, set(listed))
        for path, digest in listed.items():
            self.assertEqual(hashlib.sha256((root / path).read_bytes()).hexdigest(), digest, path)
        return listed


class Engines(StageCase):
    def test_replaces_the_engines_of_the_families(self):
        code, text, out = self.stage(families="")
        self.assertEqual((code, out["staged"]), (0, "false"), text)
        code, text, out = self.stage()
        self.assertEqual(code, 0, text)
        self.assertEqual(out["staged"], "true")
        # Every family (qwen3 is the only one here): every engine and the
        # shared unit, and the pr tier picks one model per family.
        self.assertEqual(sorted(self.tree("lin", "xrt")), ["src/lib/xrt/libmha.so", "src/lib/xrt/libqwen3_npu.so"])
        self.assertEqual(sorted(self.tree("win", "hrx")), ["src/lib/hrx/mha.dll", "src/lib/hrx/mha.lib",
                                                           "src/lib/hrx/qwen3_npu.dll", "src/lib/hrx/qwen3_npu.lib"])
        self.assertEqual(out["families"], "")
        m = self.manifest("windows", "xrt")
        self.assertEqual((m.run_id, m.head_sha, m.base_sha, m.full_refresh), ("100", HEAD, BASE, False))
        got = {e.repo_path: (e.artifact, e.artifact_path, e.new) for e in m.files}
        self.assertEqual(got["src/lib/xrt/qwen3_npu.dll"], ("devkit-libs-win-xrt", "qwen3_npu.dll", False))
        self.assertEqual(got["src/lib/xrt/qwen3_npu.lib"], ("devkit-libs-win-xrt", "qwen3_npu.lib", False))
        e = next(f for f in m.files if f.repo_path == "src/lib/xrt/qwen3_npu.dll")
        self.assertEqual(e.sha256, hashlib.sha256(prebuilt("xrt", "qwen3_npu.dll").read_bytes()).hexdigest())
        self.assertNotIn(b"\r\n", (self.out / "manifest-windows-xrt" / ds.MANIFEST_FILE).read_bytes())
        # The swap reads exactly these: both systems of each runtime.
        manifests = {k: self.manifest(*k) for k in ds.TEST_JOBS}
        entries, _, errors = ds.plan_swap(manifests, "100", HEAD)
        self.assertEqual(errors, [])
        self.assertEqual(len(entries), 12)

    def test_only_the_engines_serving_the_families(self):
        self.add_new_model()
        code, text, out = self.stage(families="newm", kernels=["newm:1b"])
        self.assertEqual(code, 0, text)
        self.assertEqual(sorted(self.tree("lin", "hrx")), ["src/lib/hrx/libnewm_npu.so",
                                                           "src/xclbins/NewM-1B-NPU2/layer.xclbin"])
        self.assertEqual(out["families"], "newm")

    def test_a_new_model(self):
        """newm_npu has no file in src/lib and newm:1b none in src/xclbins
        yet: allowed, as engines.json and kernel_assignment.json declare
        them, and recorded as new; the new model is required."""
        self.add_new_model()
        code, text, out = self.stage(families="newm", kernels=["newm:1b"])
        self.assertEqual(code, 0, text)
        self.assertEqual({e.repo_path: e.new for e in self.manifest("linux", "hrx").files},
                         {"src/lib/hrx/libnewm_npu.so": True, "src/xclbins/NewM-1B-NPU2/layer.xclbin": True})
        # The unassigned file beside the new kernel is not staged.
        self.assertEqual(sorted(self.tree("win", "xrt")), ["src/lib/xrt/newm_npu.dll", "src/lib/xrt/newm_npu.lib",
                                                           "src/xclbins/NewM-1B-NPU2/layer.xclbin"])
        self.assertEqual(out["required"], "newm:1b")
        self.assertIn("7 file(s) new to src/ (pending their swap)", text)

    def test_missing_library_stages_nothing(self):
        (self.libs / "devkit-libs-win-hrx" / "qwen3_npu.lib").unlink()
        code, text, _ = self.stage()
        self.assertEqual(code, 1)
        self.assertIn("windows hrx: qwen3_npu.lib is not in the dev-kit libraries (devkit-libs-win-hrx)", text)
        self.assertFalse(self.out.exists())

    def test_no_libraries_for_a_runtime(self):
        shutil.rmtree(self.libs / "devkit-libs-lin-hrx")
        code, text, _ = self.stage()
        self.assertEqual(code, 1)
        self.assertIn("linux hrx: engines to stage (mha, qwen3_npu) but no dev-kit libraries", text)

    def test_not_a_library(self):
        (self.libs / "devkit-libs-lin-xrt" / "libqwen3_npu.so").write_bytes(b"MZ not an ELF")
        (self.libs / "devkit-libs-win-xrt" / "qwen3_npu.lib").write_bytes(b"MZ not an archive")
        code, text, _ = self.stage()
        self.assertEqual(code, 1)
        self.assertIn("linux xrt: the dev-kit libqwen3_npu.so is not a linux library", text)
        self.assertIn("windows xrt: the dev-kit qwen3_npu.lib is not a windows import library", text)

    def test_symbolic_link_is_refused(self):
        # Seen as a link (creating one needs a privilege Windows runners lack).
        link = self.libs / "devkit-libs-lin-xrt" / "libqwen3_npu.so"
        real = Path.is_symlink
        with mock.patch.object(Path, "is_symlink", lambda p: p == link or real(p)):
            code, text, _ = self.stage()
        self.assertEqual(code, 1)
        self.assertIn("libqwen3_npu.so is not in the dev-kit libraries", text)

    def test_abi_check(self):
        # A libmha.so that lost what libqwen3_npu.so takes from it.
        shutil.copyfile(prebuilt("xrt", "libgemm.so"), self.libs / "devkit-libs-lin-xrt" / "libmha.so")
        code, text, _ = self.stage()
        self.assertEqual(code, 1)
        self.assertIn("linux xrt: the dev-kit libmha.so lacks", text)
        self.assertFalse(self.out.exists())


class Kernels(StageCase):
    def test_declared_slots_only(self):
        code, text, out = self.stage(families="", kernels=["qwen3:0.6b", "newm:1b", "smolvla:500m"])
        self.assertEqual(code, 0, text)
        tree = self.tree("lin", "xrt")
        # A rebuilt slot replaces the shipped one (mm.xclbin, not rebuilt,
        # stays); every system and runtime gets the same kernels.
        self.assertEqual(sorted(tree), ["src/xclbins/Qwen3-0.6B-NPU2/attn.xclbin"])
        self.assertEqual(sorted(self.tree("win", "hrx")), sorted(tree))
        self.assertEqual({e.repo_path: (e.new, e.artifact, e.artifact_path)
                          for e in self.manifest("windows", "hrx").files},
                         {"src/xclbins/Qwen3-0.6B-NPU2/attn.xclbin":
                          (False, "devkit-xclbins", "Qwen3-0.6B-NPU2/attn.xclbin")})
        # Not assigned (newm:1b here), or served elsewhere: never staged.
        self.assertIn("newm:1b: no kernel_assignment.json entry", text)
        self.assertIn("smolvla:500m: not an NPU catalog model", text)
        # The staged kernels' models are required.
        self.assertEqual(out["required"], "qwen3:0.6b")

    def test_unsafe_folder(self):
        bad = json.loads(json.dumps(ASSIGNMENT))
        bad["models"]["qwen3:0.6b"]["folder"] = "../Qwen3"
        self.write_json("flm-dev-kit/FLM_Xclbin/kernel_assignment.json", bad)
        code, text, _ = self.stage(families="", kernels=["qwen3:0.6b"])
        self.assertEqual(code, 1)
        self.assertIn("'src/xclbins/../Qwen3/attn.xclbin' is no plain relative path", text)

    def test_no_kernels_artifact(self):
        code, text, _ = self.stage(families="", kernels=["qwen3:0.6b"], xclbins=False)
        self.assertEqual(code, 1)
        self.assertIn("no dev-kit kernels (devkit-xclbins)", text)


class Pending(StageCase):
    """What src/ lacks (devkit_check.py's pending binaries) must be staged."""

    def test_the_new_engine_must_be_staged(self):
        self.add_new_model()
        code, text, _ = self.stage(families="", kernels=["newm:1b"])
        self.assertEqual(code, 1)
        self.assertIn("src/lib/hrx/libnewm_npu.so is required (an assigned slot or an engine library) and not "
                      "in src/, and this run does not stage it", text)
        self.assertFalse(self.out.exists())

    def test_the_new_kernels_must_be_staged(self):
        self.add_new_model()
        code, text, _ = self.stage(families="newm", kernels=[])
        self.assertEqual(code, 1)
        self.assertIn("src/xclbins/NewM-1B-NPU2/layer.xclbin is required", text)

    def test_nothing_pending_on_a_complete_tree(self):
        code, text, _ = self.stage(families="", kernels=[])
        self.assertEqual(code, 0, text)
        self.assertNotIn("pending their swap", text)


class Selection(StageCase):
    def test_union_with_the_product(self):
        code, text, out = self.stage("--suites", "asr", "--product-families", "d3", "--new-models", "d3:lite",
                                     families="", kernels=["qwen3:0.6b"])
        self.assertEqual(code, 0, text)
        self.assertEqual(out["suites"], "chat,prefix_cache,decision,asr")
        self.assertEqual(out["hrx-suites"], "chat,prefix_cache,decision,asr")
        self.assertEqual(out["families"], "d3")
        self.assertEqual(out["required"], "d3:lite,qwen3:0.6b")
        # The staged engines' families choose suites too; HRX skips api.
        self.add_new_model()
        code, text, out = self.stage("--suites", "api", "--product-families", "d3", families="newm",
                                     kernels=["newm:1b"])
        self.assertEqual(code, 0, text)
        self.assertEqual((out["suites"], out["hrx-suites"], out["families"]),
                         ("chat,prefix_cache,api", "chat,prefix_cache", "d3,newm"))

    def test_every_suite_stays_every_suite(self):
        code, _, out = self.stage("--suites", "all")
        self.assertEqual(code, 0)
        self.assertEqual((out["suites"], out["hrx-suites"]), ("all", st.HRX_ALL))

    def test_suite_helpers(self):
        self.assertEqual(st.hrx_suites("chat,api,serving"), "chat")
        self.assertEqual(st.union_suites("api", "", True), "api")
        # Something staged always tests: flm loads every engine at start-up.
        self.assertEqual(st.union_suites("none", "", True), "chat")
        self.assertEqual(st.union_suites("none", "", False), "none")

    def test_full_refresh_requires_every_catalog_model(self):
        code, text, out = self.stage("--require-all", "--full-refresh", kernels=["qwen3:0.6b"])
        self.assertEqual(code, 0, text)
        self.assertEqual(out["required"], "d3:lite,qwen3:0.6b")
        self.assertTrue(self.manifest("linux", "xrt").full_refresh)

    def test_nothing_to_stage(self):
        code, text, out = self.stage(families="", kernels=["smolvla:500m"])
        self.assertEqual(code, 0, text)
        self.assertEqual((out["staged"], out["count"], out["suites"]), ("false", "0", "none"))
        self.assertEqual(self.manifest("linux", "xrt").files, [])

    def test_out_must_be_empty(self):
        (self.out / "x").mkdir(parents=True)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            st.main(["--repo", str(self.repo), "--out", str(self.out), "--head-sha", HEAD])


if __name__ == "__main__":
    unittest.main()
