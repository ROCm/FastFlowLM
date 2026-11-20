# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for devkit_overlay.py: python3 -m unittest discover -s ci -p 'test_*.py' -t ci"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import changes  # noqa: E402
import devkit_overlay as do  # noqa: E402

REPO = HERE.parents[2]
PREBUILT = REPO / "src" / "lib"
ENGINES = changes.load_devkit_engines()


def prebuilt(runtime: str, name: str) -> Path:
    path = PREBUILT / runtime / name
    if not path.is_file():
        raise unittest.SkipTest(f"{path} is not in the checkout")
    return path


def make_pe(exports: list[str] = (), imports: dict[str, list[str]] | None = None) -> bytes:
    """A minimal PE32+ image: one section at RVA 0x1000 holding an export
    directory and an import directory."""
    imports = imports or {}
    base, sec = 0x1000, bytearray()

    def put(blob: bytes) -> int:
        rva = base + len(sec)
        sec.extend(blob)
        while len(sec) % 8:
            sec.append(0)
        return rva

    names = [put(n.encode() + b"\0") for n in exports]
    names_rva = put(b"".join(struct.pack("<I", r) for r in names)) if names else 0
    export_rva = put(struct.pack("<IIHHIIIIIII", 0, 0, 0, 0, 0, 1, 0, len(names), 0, names_rva, 0))
    descs = []
    for dll, funcs in imports.items():
        hints = [put(b"\0\0" + f.encode() + b"\0") for f in funcs]
        thunks = put(b"".join(struct.pack("<Q", h) for h in hints) + struct.pack("<Q", 1 << 63 | 7)
                     + b"\0" * 8)
        descs.append((thunks, put(dll.encode() + b"\0")))
    import_rva = put(b"".join(struct.pack("<IIIII", t, 0, 0, n, t) for t, n in descs) + b"\0" * 20)

    dos = bytearray(0x40)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x40)
    coff = b"PE\0\0" + struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 240, 0x2022)
    opt = bytearray(240)
    struct.pack_into("<H", opt, 0, 0x20B)
    struct.pack_into("<I", opt, 108, 16)
    struct.pack_into("<II", opt, 112, export_rva, 40)
    struct.pack_into("<II", opt, 120, import_rva, 20 * (len(descs) + 1))
    header = struct.pack("<8sIIIIIIHHI", b".edata", len(sec), base, len(sec), 0x200, 0, 0, 0, 0, 0)
    image = bytes(dos) + coff + bytes(opt) + header
    return image + b"\0" * (0x200 - len(image)) + bytes(sec)


class LibraryNames(unittest.TestCase):
    def test_every_unit_names_a_prebuilt_library(self):
        # The overlay replaces the package's file of the same name, so every
        # engine and shared unit must map to one the product ships.
        self.assertIsNotNone(ENGINES)
        units = (set(ENGINES.engines) | set(ENGINES.shared)) - do.NO_LIBRARY
        for runtime, os_name in (("xrt", "linux"), ("hrx", "linux"), ("xrt", "windows")):
            shipped = {p.name for p in (PREBUILT / runtime).glob("*")}
            if not shipped:
                self.skipTest(f"src/lib/{runtime} is not in the checkout")
            for unit in sorted(units):
                with self.subTest(runtime=runtime, unit=unit):
                    self.assertIn(do.lib_name(unit, os_name), shipped)

    def test_names(self):
        self.assertEqual(do.lib_name("q4nx", "linux"), "libq4_npu_eXpress.so")
        self.assertEqual(do.lib_name("qwen3_5_omni", "windows"), "qwen3_5_omni_npu.dll")
        self.assertEqual(do.lib_name("qwen3_npu", "linux"), "libqwen3_npu.so")
        self.assertEqual(do.lib_dir(Path("r"), "linux"), Path("r/lib"))
        self.assertEqual(do.lib_dir(Path("r"), "windows"), Path("r"))


class Units(unittest.TestCase):
    def test_one_engine_and_its_users(self):
        units, every = do.units_to_overlay(ENGINES, ENGINES.families("qwen3_npu"))
        self.assertFalse(every)
        self.assertIn("qwen3_npu", units)
        self.assertNotIn("mha", units)
        self.assertNotIn("llama_npu", units)

    def test_every_family_overlays_everything(self):
        units, every = do.units_to_overlay(ENGINES, ENGINES.all_families())
        self.assertTrue(every)
        self.assertTrue({"q4nx", "mha", "gemm", "dequant", "lm_head", "qwen3_5_omni"} <= set(units))
        self.assertNotIn("vision_common", units)

    def test_no_family_no_engine(self):
        self.assertEqual(do.units_to_overlay(ENGINES, set()), ([], False))


class Symbols(unittest.TestCase):
    def test_elf(self):
        defined, undefined = do.elf_symbols(prebuilt("xrt", "libqwen3_npu.so").read_bytes())
        self.assertTrue(any(s.startswith("_ZN9qwen3_npu") for s in defined))
        # The engine takes MHA from libmha.so without naming it (DT_NEEDED).
        self.assertIn("_ZN3MHAD1Ev", undefined)
        self.assertIn("_ZN3MHAD1Ev", do.elf_symbols(prebuilt("xrt", "libmha.so").read_bytes())[0])

    def test_pe(self):
        mha = do.pe_exports(prebuilt("xrt", "mha.dll").read_bytes())
        imports = do.pe_imports(prebuilt("xrt", "qwen3_npu.dll").read_bytes())
        self.assertTrue(imports["mha.dll"])
        self.assertLessEqual(imports["mha.dll"], mha)

    def test_synthetic_pe(self):
        image = make_pe(["?a@@YAXXZ", "b"], {"MHA.dll": ["x", "y"], "kernel32.dll": ["Sleep"]})
        self.assertEqual(do.pe_exports(image), {"?a@@YAXXZ", "b"})
        # Lower-cased DLL names; by-ordinal imports left out.
        self.assertEqual(do.pe_imports(image), {"mha.dll": {"x", "y"}, "kernel32.dll": {"Sleep"}})

    def test_not_a_binary(self):
        with self.assertRaises(do.BinaryError):
            do.elf_symbols(b"MZ" + b"\0" * 100)
        with self.assertRaises(do.BinaryError):
            do.pe_exports(b"\x7fELF" + b"\0" * 100)


class Abi(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def package(self, os_name: str, names: list[str]) -> Path:
        root = self.tmp / os_name
        lib = do.lib_dir(root, os_name)
        lib.mkdir(parents=True)
        for name in names:
            shutil.copyfile(prebuilt("xrt", name), lib / name)
        return root

    def test_linux(self):
        root = self.package("linux", ["libmha.so", "libgemm.so", "libqwen3_npu.so"])
        lib = root / "lib"
        self.assertEqual(do.abi_errors(root, "linux", {lib / "libmha.so": prebuilt("xrt", "libmha.so")}), [])
        # A library that stopped exporting what the engines take from it.
        errors = do.abi_errors(root, "linux", {lib / "libmha.so": prebuilt("xrt", "libgemm.so")})
        self.assertEqual(len(errors), 1)
        self.assertIn("_ZN3MHAD1Ev", errors[0])
        # A dev-kit engine needing what no package library defines (here the
        # Dequant unit, which this package lacks).
        errors = do.abi_errors(root, "linux", {lib / "libqwen3_npu.so": prebuilt("xrt", "libllama_npu.so")})
        self.assertEqual(len(errors), 1)
        self.assertIn("libqwen3_npu.so needs", errors[0])
        self.assertIn("_ZN7DequantD1Ev", errors[0])

    def test_system_symbols_are_not_the_package_s(self):
        for name in ("_ZNKRSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE3strEv",
                     "_ZTTNSt7__cxx1118basic_stringstreamIwSt11char_traitsIwESaIwEEE",
                     "_ZNSo9_M_insertIbEERSoT_", "_ZGVbN2v_exp", "_Znwm", "_ZdlPvm",
                     "_ZTVN10__cxxabiv117__class_type_infoE"):
            self.assertTrue(do.STD_SYMBOL.match(name) or do.RUNTIME_SYMBOL.match(name), name)
        for name in ("_ZN3MHAD1Ev", "_ZN7DequantD1Ev", "_ZGVZN3foo3barEvE1x"):
            self.assertFalse(do.STD_SYMBOL.match(name) or do.RUNTIME_SYMBOL.match(name), name)

    def test_windows(self):
        root = self.package("windows", ["mha.dll", "gemm.dll", "qwen3_npu.dll"])
        self.assertEqual(do.abi_errors(root, "windows", {root / "mha.dll": prebuilt("xrt", "mha.dll")}), [])
        errors = do.abi_errors(root, "windows", {root / "mha.dll": prebuilt("xrt", "gemm.dll")})
        self.assertEqual(len(errors), 1)
        self.assertIn("qwen3_npu.dll imports", errors[0])
        # A dev-kit engine importing what the package's shared unit lacks.
        (self.tmp / "new.dll").write_bytes(make_pe(["e"], {"mha.dll": ["??1MHA@@QEAA@XZ", "?new@@YAXXZ"]}))
        errors = do.abi_errors(root, "windows", {root / "qwen3_npu.dll": self.tmp / "new.dll"})
        self.assertEqual(errors, ["qwen3_npu.dll imports 1 name(s) the package mha.dll does not export: "
                                  "?new@@YAXXZ"])


CATALOG = {"models": {
    "qwen3": {"0.6b": {"name": "Qwen3-0.6B-NPU2", "details": {"family": "qwen3"}},
              "8b": {"name": "Qwen3-8B-NPU2", "details": {"family": "qwen3"},
                     "supported_platforms": ["aie2p"]}},
    "qwen3-old": {"1b": {"name": "Q-old", "details": {"family": "qwen3"}, "supported_platforms": ["aie2"]}},
    "qwen3-oga-hipep": {"8b": {"name": "Q-gpu", "details": {"family": "qwen3"}}},
    "llama3.2": {"1b": {"name": "Llama-3.2-1B-NPU2", "details": {"family": "llama3"}}},
    "d3": {"lite": {"name": "D3-Lite-NPU2", "details": {"family": "d3"}}},
}}


class Selection(unittest.TestCase):
    def catalog(self) -> dict:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        (root / "model_list.json").write_text(json.dumps(CATALOG), encoding="utf-8")
        return do.npu_catalog(root)

    def test_npu_catalog(self):
        self.assertEqual(sorted(self.catalog()), ["d3:lite", "llama3.2:1b", "qwen3:0.6b", "qwen3:8b"])

    def test_families(self):
        out, notes = do.select_tests(self.catalog(), {"qwen3"}, False, [], ["llama3.2:1b"])
        self.assertEqual(out, {"models": "qwen3:0.6b,qwen3:8b", "families": "qwen3",
                               "suites": "chat,prefix_cache", "required": "llama3.2:1b", "test": "true"})
        self.assertEqual(notes, [])

    def test_kernels_only(self):
        out, notes = do.select_tests(self.catalog(), set(), False, ["d3:lite", "smolvla:500m"], [])
        self.assertEqual(out, {"models": "d3:lite", "families": "", "suites": "decision",
                               "required": "d3:lite", "test": "true"})
        self.assertIn("smolvla:500m", notes[0])

    def test_every_family_leaves_the_pick_to_the_pr_tier(self):
        out, _ = do.select_tests(self.catalog(), {"qwen3", "llama3", "d3"}, True, [], [])
        self.assertEqual((out["models"], out["families"], out["suites"], out["test"]),
                         ("", "", "chat,prefix_cache,decision", "true"))

    def test_new_models_choose_suites(self):
        out, _ = do.select_tests(self.catalog(), {"qwen3"}, False, [], ["d3:lite"])
        self.assertEqual((out["suites"], out["required"]), ("chat,prefix_cache,decision", "d3:lite"))
        out, _ = do.select_tests(self.catalog(), set(), False, [], ["d3:lite"])
        self.assertEqual((out["suites"], out["test"]), ("decision", "true"))

    def test_replaced_engines_other_families_are_tested(self):
        out, _ = do.select_tests(self.catalog(), {"qwen3"}, False, [], [], {"qwen3", "llama3"})
        self.assertEqual((out["models"], out["families"]),
                         ("llama3.2:1b,qwen3:0.6b,qwen3:8b", "qwen3"))

    def test_nothing(self):
        out, _ = do.select_tests(self.catalog(), set(), False, ["smolvla:500m"], [])
        self.assertEqual(out["test"], "false")


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.root = self.tmp / "pkg"
        (self.root / "lib").mkdir(parents=True)
        (self.root / "xclbins" / "Qwen3-0.6B-NPU2").mkdir(parents=True)
        (self.root / "xclbins" / "Qwen3-0.6B-NPU2" / "attn.xclbin").write_text("old")
        (self.root / "xclbins" / "Qwen3-0.6B-NPU2" / "mm.xclbin").write_text("old")
        (self.root / "model_list.json").write_text(json.dumps(CATALOG), encoding="utf-8")
        for name in ("libmha.so", "libqwen3_npu.so"):
            shutil.copyfile(prebuilt("xrt", name), self.root / "lib" / name)
        self.libs = self.tmp / "libs"
        self.libs.mkdir()
        self.xclbins = self.tmp / "xclbins"
        (self.xclbins / "Qwen3-0.6B-NPU2").mkdir(parents=True)
        (self.xclbins / "Qwen3-0.6B-NPU2" / "attn.xclbin").write_text("new")

    def run_main(self, *extra: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = do.main(["--root", str(self.root), "--os", "linux", "--libs", str(self.libs),
                            "--xclbins", str(self.xclbins), *extra])
        return code, out.getvalue()

    def test_overlays_engines_and_kernels(self):
        shutil.copyfile(prebuilt("hrx", "libqwen3_npu.so"), self.libs / "libqwen3_npu.so")
        code, out = self.run_main("--families", "qwen3,qwen3-it,qwen3-tk,deepseek-r1-0528",
                                  "--xclbin-models", '["qwen3:0.6b"]')
        self.assertEqual(code, 0, out)
        self.assertEqual((self.root / "lib" / "libqwen3_npu.so").read_bytes(),
                         prebuilt("hrx", "libqwen3_npu.so").read_bytes())
        # The package's tree stays complete: the rebuilt kernel over it.
        folder = self.root / "xclbins" / "Qwen3-0.6B-NPU2"
        self.assertEqual(((folder / "attn.xclbin").read_text(), (folder / "mm.xclbin").read_text()), ("new", "old"))
        self.assertIn("models=qwen3:0.6b,qwen3:8b\n", out)
        self.assertIn("required=qwen3:0.6b\n", out)

    def test_untested_required_model_fails(self):
        code, out = self.run_main("--new-models", "nosuch:1b")
        self.assertEqual(code, 1)
        self.assertIn("would go untested", out)
        self.assertEqual((self.root / "xclbins" / "Qwen3-0.6B-NPU2" / "attn.xclbin").read_text(), "old")

    def test_missing_library_copies_nothing(self):
        code, out = self.run_main("--families", "qwen3")
        self.assertEqual(code, 1)
        self.assertIn("libqwen3_npu.so is not in the dev-kit libraries", out)
        self.assertEqual((self.root / "xclbins" / "Qwen3-0.6B-NPU2" / "attn.xclbin").read_text(), "old")

    def test_broken_abi_copies_nothing(self):
        # A libmha.so that lost what the package's libqwen3_npu.so takes from it.
        shutil.copyfile(prebuilt("xrt", "libgemm.so"), self.libs / "libmha.so")
        code, out = do_units(self, ["mha"])
        self.assertEqual(code, 1)
        self.assertIn("libmha.so lacks", out)
        self.assertEqual((self.root / "lib" / "libmha.so").read_bytes(),
                         prebuilt("xrt", "libmha.so").read_bytes())


def do_units(case: Cli, units: list[str]) -> tuple[int, str]:
    """Runs the overlay with every family, restricted to these units."""
    original = do.units_to_overlay
    do.units_to_overlay = lambda engines, families: (units, True)
    try:
        return case.run_main("--families", ",".join(sorted(ENGINES.all_families())))
    finally:
        do.units_to_overlay = original


WORKFLOW = REPO / ".github" / "workflows" / "build-and-test.yml"
ACTION = REPO / ".github" / "actions" / "devkit-npu-overlay" / "action.yml"
GATE = """          || (needs.npu-gate.result == 'success'
              && (needs.npu-gate.outputs.trusted == 'true' || needs.npu-approve.result == 'success')))"""


def jobs(text: str) -> dict[str, str]:
    """Job id -> its block of build-and-test.yml (text between job headers)."""
    body = text[text.index("\njobs:\n"):]
    parts = re.split(r"(?m)^  ([A-Za-z0-9_-]+):\n", body)
    return dict(zip(parts[1::2], parts[2::2]))


class Workflow(unittest.TestCase):
    """The overlay jobs keep the NPU jobs' gate and count in summary."""

    def test_overlay_jobs(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        blocks = jobs(text)
        overlay = sorted(j for j in blocks if j.endswith("-devkit"))
        self.assertEqual(overlay, ["test-lin-hrx-devkit", "test-lin-xrt-devkit",
                                   "test-win-hrx-devkit", "test-win-xrt-devkit"])
        for job in overlay:
            with self.subTest(job=job):
                block = blocks[job]
                self.assertIn(GATE, block)
                self.assertIn("github.event_name == 'pull_request'", block)
                self.assertIn("needs.changes.outputs.devkit-overlay == 'true'", block)
                self.assertRegex(block, r"runs-on: \[self-hosted, (linux|windows), x64, npu, sweep\]")
                self.assertIn("uses: ./.github/actions/devkit-npu-overlay", block)
                self.assertIn(f"      - {job}\n", blocks["summary"])
                self.assertIn(f"{job}=${{{{ needs.{job}.result }}}}", blocks["summary"])
        self.assertIn("persist-credentials: false", blocks["test-lin-xrt-devkit"])
        self.assertIn("devkit-overlay == 'true'", blocks["npu-approve"])
        for build in ("build-lin-xrt-portable", "build-lin-hrx-portable", "build-win-xrt", "build-win-hrx"):
            self.assertIn("needs.changes.outputs.devkit-overlay == 'true'", blocks[build], build)

    def test_harness_is_the_base(self):
        text = ACTION.read_text(encoding="utf-8")
        self.assertIn("ref: ${{ inputs.harness-ref }}", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn('clean-env: "true"', text)
        self.assertIn("harness-dir: ${{ github.workspace }}/devkit-harness", text)
        self.assertIn("harness-ref: ${{ github.event.pull_request.base.sha }}", WORKFLOW.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
