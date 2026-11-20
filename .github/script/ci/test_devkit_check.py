# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for devkit_check.py: python3 -m unittest discover -s ci -p 'test_*.py' -t ci"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import devkit_check as dc  # noqa: E402

LOCK = """# header

[xclbins-Fam_decoding_M_1B/rtp_addresses.txt]
RTP_A 1024
RTP_B 2048
"""
MAKEFILE = "LDFLAGS += -Wl,-Bsymbolic-functions\n"
NOTICE = "// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.\n"

MD = """# Kernel Assignment

### Family

#### `m:1b` → `M-1B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Attn` | Peano |
| `layer.xclbin` | `xclbins-Fam_decoding_M_1B` | Peano |

#### No tag yet

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Attn` | Peano |
| `mm.xclbin` | — | — |
"""


class Tree:
    """A minimal repository with every file devkit_check reads."""

    def __init__(self, root: Path):
        self.root = root
        self.units = {"Attn": [], "Fam/decoding": ["M_1B", "M_2B"]}
        for unit in self.units:
            self.write(f"{dc.XCLBIN}/{unit}/Makefile", "all:\n")
        self.write(f"{dc.XCLBIN}/Fam/decoding/makefiles/Makefile", "")  # not a unit
        self.write(f"{dc.XCLBIN}/Attn/build/Makefile", "")               # not a unit
        self.write_units()
        self.assignment = {"models": {"m:1b": {"folder": "M-1B-NPU2", "kernels": {
            "attn.xclbin": "xclbins-Attn", "layer.xclbin": "xclbins-Fam_decoding_M_1B"}}}}
        self.write_json(dc.ASSIGNMENT, self.assignment)
        self.write(dc.ASSIGNMENT_MD, MD)
        for d in ("q4nx", "m_npu", "m_flash"):
            self.write(f"{dc.DLL}/detail/{d}/Makefile", MAKEFILE)
        self.write(f"{dc.DLL}/detail/vision_common/x.cpp", "")
        self.write(f"{dc.DLL}/detail/m_flash/m_flash.cpp", '#include "models/m/m_npu.hpp"\n')
        # A comment naming another engine is no dependency.
        self.write(f"{dc.DLL}/detail/m_npu/m_npu.cpp", "// forked from ../m_flash/ and models/m_flash/\n")
        self.write(f"{dc.DLL}/include/models/m/m_npu.hpp", "")
        self.engines = {"shared": ["q4nx", "vision_common"], "engines": {
            "m_npu": {"families": ["m"], "models": ["m:1b"], "uses": []},
            "m_flash": {"families": ["m-flash"], "models": [], "uses": ["m_npu"]}}}
        self.write_json(dc.ENGINES, self.engines)
        self.write(dc.RTP_LOCK, LOCK)

    def write(self, rel: str, text: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def write_json(self, rel: str, data: dict) -> None:
        self.write(rel, json.dumps(data, indent=2))

    def write_units(self) -> None:
        self.write_json(dc.BUILD_UNITS, {"units": {u: {"model_types": t, "artifacts": dc.artifact_names(u, t)}
                                                   for u, t in self.units.items()}})

    def errors(self) -> list[str]:
        return dc.run(self.root)[0]


class Synthetic(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tree = Tree(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def assertError(self, *fragments: str) -> None:
        errors = self.tree.errors()
        self.assertTrue(any(all(f in e for f in fragments) for e in errors),
                        f"no error with {fragments}: {errors}")

    def test_clean_tree_passes(self):
        self.assertEqual(self.tree.errors(), [])

    def test_artifact_names(self):
        self.assertEqual(dc.artifact_names("Attn", []), ["xclbins-Attn"])
        self.assertEqual(dc.artifact_names("Fam/decoding", ["A", "B"]),
                         ["xclbins-Fam_decoding_A", "xclbins-Fam_decoding_B"])

    def test_unlisted_and_missing_units(self):
        self.tree.write(f"{dc.XCLBIN}/New/Makefile", "")
        self.assertError("New has a Makefile but is not listed")
        shutil.rmtree(self.tmp / dc.XCLBIN / "Attn")
        self.assertError("Attn is listed but")

    def test_artifacts_must_follow_model_types(self):
        data = json.loads((self.tmp / dc.BUILD_UNITS).read_text())
        data["units"]["Fam/decoding"]["artifacts"] = ["xclbins-Fam_decoding_M_1B"]
        self.tree.write_json(dc.BUILD_UNITS, data)
        self.assertError("Fam/decoding: artifacts are")

    def test_assignment_artifact_must_resolve(self):
        self.tree.units["Fam/decoding"] = ["M_2B"]  # M_1B no longer built
        self.tree.write_units()
        self.assertError("m:1b: layer.xclbin -> 'xclbins-Fam_decoding_M_1B' is no build unit's artifact")

    def test_assignment_folder_and_slot(self):
        index = {"xclbins-Attn": ("Attn", None), "xclbins-Other": ("Other", None)}

        def errors(models):
            out = []
            dc.check_assignment({"models": models}, index, out)
            return out

        ok = {"a": {"folder": "F-NPU2", "kernels": {"attn.xclbin": "xclbins-Attn"}},
              "b": {"folder": "F-NPU2", "kernels": {"attn.xclbin": "xclbins-Attn", "mm.xclbin": "xclbins-Other"}}}
        self.assertEqual(errors(ok), [])
        clash = dict(ok, c={"folder": "F-NPU2", "kernels": {"attn.xclbin": "xclbins-Other"}})
        self.assertTrue(any("F-NPU2/attn.xclbin is 'xclbins-Other', but a puts 'xclbins-Attn'" in e
                            for e in errors(clash)), errors(clash))
        for folder in ("/home/runner/work/_temp", "../../x", "a/../b", "a//b", "a b"):
            with self.subTest(folder=folder):
                e = errors({"a": {"folder": folder, "kernels": {"attn.xclbin": "xclbins-Attn"}}})
                self.assertTrue(any("is not a plain relative path" in x for x in e), e)
        e = errors({"a": {"folder": "F", "kernels": {"../attn.xclbin": "xclbins-Attn"}}})
        self.assertTrue(any("is not a plain file name" in x for x in e), e)

    def test_markdown_must_agree(self):
        cases = {
            MD.replace("`M-1B-NPU2`", "`M-1B`"): "folder M-1B, but",
            MD.replace("`xclbins-Fam_decoding_M_1B`", "`xclbins-Attn`"): "the table has layer.xclbin -> xclbins-Attn",
            MD.replace("| `layer.xclbin` | `xclbins-Fam_decoding_M_1B` | Peano |\n", ""):
                "layer.xclbin -> xclbins-Fam_decoding_M_1B is in",
            MD.replace("| `layer.xclbin` | `xclbins-Fam_decoding_M_1B` | Peano |", "| `layer.xclbin` | — | — |"):
                "no CI artifact (—) for layer.xclbin",
            MD.replace("| `layer.xclbin` |", "| `vision.xclbin` | `xclbins-Attn` | Peano |\n| `layer.xclbin` |"):
                "vision.xclbin -> xclbins-Attn, which",
            MD.replace("`m:1b`", "`m:2b`"): "no \"#### `m:1b`",
            MD.replace("| `mm.xclbin` | — | — |", "| `mm.xclbin` | `xclbins-Gone` | Peano |"):
                "mm.xclbin -> xclbins-Gone is no build unit's artifact",
        }
        for md, fragment in cases.items():
            with self.subTest(fragment=fragment):
                self.tree.write(dc.ASSIGNMENT_MD, md)
                self.assertError(fragment)

    def test_engines(self):
        self.tree.write(f"{dc.DLL}/detail/other_npu/Makefile", MAKEFILE)
        self.assertError("detail/other_npu has a Makefile but is neither an engine nor shared")
        shutil.rmtree(self.tmp / dc.DLL / "detail" / "other_npu")
        self.tree.engines["engines"]["m_flash"]["uses"] = []
        self.tree.engines["engines"]["m_npu"]["models"] = ["m:9b"]
        self.tree.engines["shared"].append("gone")
        self.tree.write_json(dc.ENGINES, self.tree.engines)
        self.assertError("m_flash uses [], but its sources and headers reference ['m_npu']")
        self.assertError("m_npu: model m:9b is not in")
        self.assertError("model m:1b is served by no engine")
        self.assertError("gone is listed but")

    def test_makefile_reference_counts_as_use(self):
        for name, text in (("Makefile", MAKEFILE + "SOURCES += ../m_flash/a.cpp # ../q4nx/\n"),
                           ("Makefile", MAKEFILE + "CXX_FLAGS += -I../m_flash\n"),
                           ("Makefile", MAKEFILE + "MF := ../m_flash\n"),
                           ("CMakeLists.txt", "add_library(m ../m_flash/a.cpp) # ../q4nx\n")):
            with self.subTest(name=name, text=text):
                self.tree.write(f"{dc.DLL}/detail/m_npu/Makefile", MAKEFILE)
                self.tree.write(f"{dc.DLL}/detail/m_npu/{name}", text)
                self.assertError("m_npu uses [], but its sources and headers reference ['m_flash']")
                (self.tmp / dc.DLL / "detail" / "m_npu" / "CMakeLists.txt").unlink(missing_ok=True)
        # A file name such as ../m_flash.cpp names no directory.
        self.tree.write(f"{dc.DLL}/detail/m_npu/Makefile", MAKEFILE + "X := ../m_flash.cpp\n")
        self.assertEqual(self.tree.errors(), [])

    def test_engine_without_makefile(self):
        (self.tmp / dc.DLL / "detail" / "m_npu" / "Makefile").unlink()
        self.assertError("engine m_npu has no", "so build.sh skips it")

    def test_model_served_twice(self):
        self.tree.engines["engines"]["m_flash"]["models"] = ["m:1b"]
        self.tree.engines["served_elsewhere"] = {"m:1b": "elsewhere"}
        self.tree.write_json(dc.ENGINES, self.tree.engines)
        self.assertError("m_npu: model m:1b is also served by m_flash")
        self.assertError("m:1b is served by an engine and listed under served_elsewhere")

    def test_empty_json_objects_are_errors(self):
        for rel, fragment in ((dc.BUILD_UNITS, 'no "units" object'), (dc.ASSIGNMENT, 'no "models" object'),
                              (dc.ENGINES, 'no "engines" object')):
            with self.subTest(rel=rel):
                original = (self.tmp / rel).read_text()
                self.tree.write(rel, "{}")
                self.assertError(fragment)
                self.tree.write(rel, original)
        self.tree.write_json(dc.ASSIGNMENT, {"models": {}})
        self.assertError("m:1b: not a model in")  # the md comparison still runs

    def test_duplicate_keys_are_errors(self):
        text = (self.tmp / dc.ASSIGNMENT).read_text()
        entry = json.dumps(self.tree.assignment["models"]["m:1b"])
        self.tree.write(dc.ASSIGNMENT, text.replace('"models": {', '"models": {"m:1b": ' + entry + ",", 1))
        self.assertError("not valid JSON: duplicate key(s) m:1b")

    def test_bsymbolic(self):
        self.tree.write(f"{dc.DLL}/detail/m_npu/Makefile", "# LDFLAGS += -Wl,-Bsymbolic-functions\n")
        self.assertError("m_npu/Makefile: no -Wl,-Bsymbolic-functions")

    def test_notice_status(self):
        self.assertEqual(dc.notice_status(NOTICE + "int x;\n"), "amd")
        self.assertEqual(dc.notice_status("# Copyright (C) 2024, Advanced Micro Devices, Inc.\n"), "amd")
        self.assertEqual(dc.notice_status("// Copyright (c) 2019 Someone\n"), "third-party")
        self.assertEqual(dc.notice_status("// Generated by tools/x.py. Do not edit.\n"), "generated")
        self.assertEqual(dc.notice_status("#include <x>\n"), "missing")
        self.assertEqual(dc.notice_status("// Copyright 2020 Someone\n"), "third-party")
        self.assertEqual(dc.notice_status("// SPDX-FileCopyrightText: 2013 Someone\n"), "third-party")
        for text in ("# TODO: add the copyright notice\n", "// no Copyright yet\n",
                     "// see https://example.com/copyright\n"):
            self.assertEqual(dc.notice_status(text), "missing", text)
        for path in ("a/b/Makefile", "a/k.cc", "a/b.md", "a/b.inc", "a/b.html", "a/pyproject.toml"):
            self.assertTrue(dc.needs_notice(path), path)
        self.assertFalse(dc.needs_notice("a/b.json"))

    def test_notices_of_added_files(self):
        if shutil.which("git") is None:
            self.skipTest("no git")

        def git(*args):
            subprocess.run(["git", "-C", str(self.tmp), *args], check=True, capture_output=True)

        git("init", "-q")
        git("add", "-A")  # the fixture is the base; only what follows is added
        git("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")
        self.tree.write(f"{dc.XCLBIN}/Attn/kernel/new.cc", "#include <x>\n")
        self.tree.write(f"{dc.XCLBIN}/Attn/kernel/ok.cc", NOTICE)
        self.tree.write(f"{dc.DLL}/include/nlohmann/json.hpp", "// vendored\n")
        self.tree.write(f"{dc.DLL}/include/third.h", "/* Copyright 2020 Someone */\n")
        git("add", "-A")
        git("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "head")
        errors, notes = dc.run(self.tmp, "HEAD~1", "HEAD")
        notice_errors = [e for e in errors if "copyright notice" in e]
        self.assertEqual(len(notice_errors), 1, errors)
        self.assertIn("Attn/kernel/new.cc", notice_errors[0])
        self.assertTrue(any("third.h: third-party" in n for n in notes), notes)
        # Without a base the check is skipped.
        self.assertFalse(any("copyright notice" in e for e in dc.run(self.tmp)[0]))


    def test_rtp_lock_must_parse_and_resolve(self):
        self.tree.write(dc.RTP_LOCK, LOCK.replace("Fam_decoding_M_1B", "Gone"))
        self.assertError("xclbins-Gone is no build unit's artifact")
        self.tree.write(dc.RTP_LOCK, LOCK.replace("[xclbins-", "xclbins-"))
        self.assertError(dc.RTP_LOCK, "expected [<artifact>/<file>]")
        # A stanza of an artifact no model uses: nothing would enforce it.
        self.tree.write(dc.RTP_LOCK, LOCK + "\n[xclbins-Fam_decoding_M_2B/rtp_addresses.txt]\nRTP_C 1\n")
        self.assertError("xclbins-Fam_decoding_M_2B is used by no")
        (self.tmp / dc.RTP_LOCK).unlink()
        self.assertError(dc.RTP_LOCK, "cannot read")

    def test_rtp_lock_change_needs_an_engine_change(self):
        if shutil.which("git") is None:
            self.skipTest("no git")

        def git(*args):
            subprocess.run(["git", "-C", str(self.tmp), *args], check=True, capture_output=True)

        def commit(msg):
            git("add", "-A")
            git("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                "commit", "-q", "--allow-empty", "-m", msg)

        def lock_errors(base="HEAD~1"):
            errors, notes = dc.run(self.tmp, base, "HEAD")
            return [e for e in errors if "stanza of" in e], notes

        # Two more models on the same kernel: f:1b served by m_flash (which
        # uses m_npu), o:1b by o_npu (which uses nothing).
        t = self.tree
        for model, folder in (("f:1b", "F-1B-NPU2"), ("o:1b", "O-1B-NPU2")):
            t.assignment["models"][model] = {"folder": folder, "kernels": {
                "layer.xclbin": "xclbins-Fam_decoding_M_1B"}}
        t.write_json(dc.ASSIGNMENT, t.assignment)
        t.write(dc.ASSIGNMENT_MD, MD.replace("#### No tag yet", "".join(
            f"#### `{m}` → `{f}`\n\n| Kernels | Artifacts | Compiler |\n|---|---|---|\n"
            "| `layer.xclbin` | `xclbins-Fam_decoding_M_1B` | Peano |\n\n"
            for m, f in (("f:1b", "F-1B-NPU2"), ("o:1b", "O-1B-NPU2"))) + "#### No tag yet"))
        t.engines["engines"]["m_flash"]["models"] = ["f:1b"]
        t.engines["engines"]["o_npu"] = {"families": ["o"], "models": ["o:1b"], "uses": []}
        t.write_json(dc.ENGINES, t.engines)
        t.write(f"{dc.DLL}/detail/o_npu/Makefile", MAKEFILE)
        self.assertEqual(dc.run(self.tmp)[0], [])

        lock = self.tmp / dc.RTP_LOCK
        lock.unlink()
        git("init", "-q")
        commit("no lock")
        lock.write_text(LOCK, encoding="utf-8")
        commit("lock added")
        errors, notes = lock_errors()
        self.assertEqual(errors, [])
        self.assertTrue(any("is new: no engine change required" in n for n in notes), notes)

        # A stanza moved, only m_flash touched: m_npu and o_npu are missing.
        lock.write_text(LOCK.replace("RTP_B 2048", "RTP_B 4096"), encoding="utf-8")
        t.write(f"{dc.DLL}/detail/m_flash/m_flash.cpp", '#include "models/m/m_npu.hpp"\n// x\n')
        commit("moved")
        errors, _ = lock_errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("xclbins-Fam_decoding_M_1B", errors[0])
        self.assertIn("m:1b", errors[0])
        missing = errors[0].split("did not:", 1)[1]
        self.assertIn(f"{dc.DLL}/detail/m_npu/", missing)
        self.assertIn(f"{dc.DLL}/detail/o_npu/", missing)
        self.assertNotIn("m_flash", missing)

        # m_npu changed too (which also covers m_flash, as it uses m_npu):
        # one engine of three serving the kernel's models is still missing.
        lock.write_text(LOCK.replace("RTP_B 2048", "RTP_B 8192"), encoding="utf-8")
        t.write(f"{dc.DLL}/detail/m_npu/m_npu.cpp", "// constants moved\n")
        commit("moved with one engine")
        errors, _ = lock_errors()
        self.assertEqual(len(errors), 1, errors)
        missing = errors[0].split("did not:", 1)[1]
        self.assertIn(f"{dc.DLL}/detail/o_npu/", missing)
        self.assertNotIn("m_npu", missing)
        self.assertNotIn("m_flash", missing)

        # Every serving engine changed: a note.
        lock.write_text(LOCK.replace("RTP_B 2048", "RTP_B 16384"), encoding="utf-8")
        t.write(f"{dc.DLL}/detail/m_npu/m_npu.cpp", "// constants moved again\n")
        t.write(f"{dc.DLL}/detail/o_npu/o_npu.cpp", "// constants moved\n")
        commit("moved with every engine")
        errors, notes = lock_errors()
        self.assertEqual(errors, [])
        self.assertTrue(any("changed with" in n and "o_npu" in n and "m_npu" in n for n in notes), notes)


class RealTree(unittest.TestCase):
    def test_tree_passes(self):
        errors, _ = dc.run(dc.REPO_ROOT)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
