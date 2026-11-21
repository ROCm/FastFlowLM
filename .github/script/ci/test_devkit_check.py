# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for devkit_check.py: python3 -m unittest discover -s ci -p 'test_*.py' -t ci"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
        # What the product ships: the catalog and every assigned kernel.
        self.catalog = {"model_path": "models", "models": {}}
        self.sync_catalog()

    def sync_catalog(self) -> None:
        """src/model_list.json and src/xclbins for every assignment model
        (served_elsewhere skipped), named after its folder."""
        elsewhere = self.engines.get("served_elsewhere") or {}
        for model, entry in self.assignment["models"].items():
            if model in elsewhere:
                continue
            family, size = model.split(":")
            self.catalog["models"].setdefault(family, {}).setdefault(size, {
                "supported_platforms": ["aie2p"], "name": entry["folder"], "details": {"family": family}})
            for slot in entry["kernels"]:
                self.write(f"{dc.SHIPPED_XCLBINS}/{entry['folder']}/{slot}", "")
        self.write_json(dc.MODEL_LIST, self.catalog)

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
        # The real allow-list names real files: in a fixture it is all stale.
        patcher = mock.patch.dict(dc.CATALOG_ALLOW, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

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
        t.sync_catalog()
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


class Catalog(Synthetic):
    """src/ (model_list.json, xclbins, lib) against the dev kit's JSON."""

    def entry(self, tag: str, **fields) -> None:
        family, size = tag.split(":")
        self.tree.catalog["models"].setdefault(family, {})[size] = {
            "supported_platforms": ["aie2p"], "details": {"family": family}, **fields}
        self.tree.write_json(dc.MODEL_LIST, self.tree.catalog)

    def engines(self, **models) -> None:
        for engine, ms in models.items():
            self.tree.engines["engines"][engine]["models"] = ms
        self.tree.write_json(dc.ENGINES, self.tree.engines)

    def library(self, unit: str, name: str, cmake: str = "") -> None:
        self.tree.write(f"{dc.DLL}/detail/{unit}/Makefile",
                        MAKEFILE + f"TARGET = $(LIB_DIR)/lib{name}.so\nelse\nTARGET = $(BUILD_DIR)/{name}.dll\n")
        self.tree.write(f"{dc.DLL}/detail/{unit}/CMakeLists.txt",
                        f"# add_library(old SHARED)\nadd_library({cmake or name} SHARED ${{SOURCES}})\n")

    def ship(self, *rels: str) -> None:
        for rel in rels:
            self.tree.write(rel, "")

    def test_npu_catalog_follows_flm_filter(self):
        cat = {"models": {
            "m": {"1b": {"supported_platforms": ["aie2p"]}, "2b": {"supported_platforms": ["dev04"]},
                  "3b": {}, "4b": {"supported_platforms": ["aie2p"], "backend": "oga-hipep"}},
            "m-core": {"1b": {"supported_platforms": ["dev04"]}, "2b": {"supported_platforms": ["aie2p"]}},
            "m-oga-hipep": {"1b": {}, "2b": {"backend": "flm"}},
            "e": {"1b": {"supported_platforms": ["aie2p"], "model_kind": "embedding", "backend": "x"}}}}
        self.assertEqual(sorted(dc.npu_catalog(cat)), ["e:1b", "m-oga-hipep:2b", "m:1b", "m:3b"])
        self.assertEqual(dc.xclbin_folder({"name": "A"}), "A")
        self.assertEqual(dc.xclbin_folder({"name": "A", "xclbin_folder": "B"}), "B")

    def test_gpu_and_other_platform_entries_need_nothing(self):
        self.entry("m:2b", name="M-2B")  # no supported_platforms: an NPU model
        self.assertError("model m:2b has no", "add \"m:2b\": {\"folder\": \"M-2B\"")
        del self.tree.catalog["models"]["m"]["2b"]
        self.entry("m:3b", name="M-3B", supported_platforms=["dev04"])
        self.entry("m-core:1b", name="M-Core", supported_platforms=["aie2p"])
        self.entry("m-oga-hipep:1b", name="M-GPU", supported_platforms=None)
        self.assertEqual(self.tree.errors(), [])

    def test_rule1_npu_model_needs_an_assignment(self):
        self.entry("n:1b", name="N-1B-NPU2")
        self.assertError(f"{dc.MODEL_LIST}: NPU (aie2p) model n:1b has no {dc.ASSIGNMENT} entry",
                         '"folder": "N-1B-NPU2"', dc.ASSIGNMENT_MD)
        self.assertError("NPU model n:1b of", "is served by no engine", "detail/<engine>/")

    def test_rule1_alias_reuses_a_folder(self):
        self.entry("n:1b", name="N-1B-NPU2", xclbin_folder="M-1B-NPU2")
        self.assertEqual(dc.catalog_aliases(self.tree.catalog, self.tree.assignment["models"]), {"n:1b"})
        # An alias still needs an engine (or an allow-list entry).
        self.assertError("NPU model n:1b of", "is served by no engine")
        self.engines(m_npu=["m:1b", "n:1b"])
        self.assertEqual(self.tree.errors(), [])
        self.entry("n:1b", name="N-1B-NPU2", xclbin_folder="Nope-NPU2")
        self.assertError("n:1b runs the kernels of xclbin_folder Nope-NPU2, but no", "model has that folder")

    def test_rule1_alias_allow_listed(self):
        self.entry("n:1b", name="N-1B-NPU2", xclbin_folder="M-1B-NPU2")
        with mock.patch.dict(dc.CATALOG_ALLOW, {"n:1b": "runs elsewhere"}):
            self.assertEqual(self.tree.errors(), [])
        # Only an alias can be excused; a model with no kernels at all cannot.
        self.entry("o:1b", name="O-1B-NPU2")
        with mock.patch.dict(dc.CATALOG_ALLOW, {"o:1b": "no"}):
            self.assertError("model o:1b has no")
            self.assertError("CATALOG_ALLOW", "o:1b excuses nothing any more")

    def test_rule2_assignment_must_be_a_catalog_model(self):
        del self.tree.catalog["models"]["m"]
        self.tree.write_json(dc.MODEL_LIST, self.tree.catalog)
        self.assertError(f"{dc.ASSIGNMENT}: m:1b is no NPU (aie2p) model of {dc.MODEL_LIST}", "served_elsewhere")
        self.assertError("m_npu lists m:1b, which is no NPU (aie2p) model")

    def test_rule2_and_4_served_elsewhere(self):
        t = self.tree
        t.assignment["models"]["vla:1b"] = {"folder": "VLA", "kernels": {"attn.xclbin": "xclbins-Attn"}}
        t.write_json(dc.ASSIGNMENT, t.assignment)
        t.write(dc.ASSIGNMENT_MD, MD + "\n#### `vla:1b` → `VLA`\n\n| Kernels | Artifacts |\n|---|---|\n"
                "| `attn.xclbin` | `xclbins-Attn` |\n")
        self.assertError("vla:1b is no NPU (aie2p) model")
        t.engines["served_elsewhere"] = {"vla:1b": "a pybind module"}
        t.write_json(dc.ENGINES, t.engines)
        # Not in the catalog and nothing shipped under src/xclbins/VLA: fine.
        self.assertEqual(t.errors(), [])

    def test_rule3_folder_is_the_catalog_folder(self):
        self.entry("m:1b", name="M-1B")
        self.assertError(f"{dc.ASSIGNMENT}: m:1b has folder M-1B-NPU2, but", "xclbin folder M-1B")
        self.entry("m:1b", name="M-1B", xclbin_folder="M-1B-NPU2")
        self.assertEqual(self.tree.errors(), [])

    def test_rule4_assigned_slot_ships(self):
        (self.tmp / dc.SHIPPED_XCLBINS / "M-1B-NPU2" / "layer.xclbin").unlink()
        self.assertError("m:1b assigns layer.xclbin, but src/xclbins/M-1B-NPU2/layer.xclbin is not shipped")

    def test_rule5_shipped_kernel_is_assigned(self):
        self.ship(f"{dc.SHIPPED_XCLBINS}/M-1B-NPU2/conv.xclbin", f"{dc.SHIPPED_XCLBINS}/M-1B-NPU2/notes.txt")
        self.assertError("src/xclbins/M-1B-NPU2/conv.xclbin is shipped, but no", "with folder M-1B-NPU2 assigns "
                         "conv.xclbin", "CATALOG_ALLOW")
        with mock.patch.dict(dc.CATALOG_ALLOW, {"src/xclbins/M-1B-NPU2/conv.xclbin": "unused"}):
            self.assertEqual(self.tree.errors(), [])
        self.ship(f"{dc.SHIPPED_XCLBINS}/X-NPU2/a.xclbin")
        self.assertError("src/xclbins/X-NPU2/a.xclbin is shipped, but no", "model has folder X-NPU2")

    def test_rule5_union_over_models_sharing_a_folder(self):
        # f:1b shares M-1B-NPU2 and adds a slot m:1b does not assign.
        t = self.tree
        t.assignment["models"]["f:1b"] = {"folder": "M-1B-NPU2", "kernels": {"mm.xclbin": "xclbins-Attn"}}
        t.write_json(dc.ASSIGNMENT, t.assignment)
        t.write(dc.ASSIGNMENT_MD, MD + "\n#### `f:1b` → `M-1B-NPU2`\n\n| Kernels | Artifacts |\n|---|---|\n"
                "| `mm.xclbin` | `xclbins-Attn` |\n")
        t.engines["engines"]["m_flash"]["models"] = ["f:1b"]
        t.write_json(dc.ENGINES, t.engines)
        t.sync_catalog()
        self.entry("f:1b", name="F-1B-NPU2", xclbin_folder="M-1B-NPU2")
        self.assertEqual(t.errors(), [])

    def test_stale_allow_list_entry(self):
        with mock.patch.dict(dc.CATALOG_ALLOW, {"src/xclbins/M-1B-NPU2/gone.xclbin": "was unused"}):
            self.assertError("src/xclbins/M-1B-NPU2/gone.xclbin excuses nothing any more (was unused)")

    def test_rule6_engine_model_must_be_in_the_catalog(self):
        self.engines(m_flash=["m:9b"])
        self.assertError("m_flash lists m:9b, which is no NPU (aie2p) model of")

    def test_rule7_library_mapping(self):
        self.assertEqual(self.tree.errors(), [])  # no TARGET, no library: nothing to ship
        self.library("q4nx", "q4_npu_eXpress")
        self.library("m_npu", "m_npu")
        self.assertError(f"{dc.ENGINES}: q4nx builds q4_npu_eXpress, but src/lib/xrt/ lacks "
                         "libq4_npu_eXpress.so, q4_npu_eXpress.dll, q4_npu_eXpress.lib")
        self.assertError("m_npu builds m_npu, but src/lib/hrx/ lacks")
        for rt in ("xrt", "hrx"):
            self.ship(*(f"src/lib/{rt}/{f}" for lib in ("q4_npu_eXpress", "m_npu") for f in dc.shipped_names(lib)))
        self.assertEqual(self.tree.errors(), [])
        (self.tmp / "src/lib/hrx/m_npu.lib").unlink()
        self.assertError("m_npu builds m_npu, but src/lib/hrx/ lacks m_npu.lib")
        self.ship("src/lib/hrx/m_npu.lib", "src/lib/xrt/libnova_npu.so", "src/lib/xrt/libaiebu.a")
        self.assertError("src/lib/xrt/libnova_npu.so is no library of an", "detail/<dir>/")
        self.assertError("src/lib/xrt/libaiebu.a is no library of an")
        with mock.patch.dict(dc.CATALOG_ALLOW, {"src/lib/xrt/libnova_npu.so": "x", "src/lib/xrt/libaiebu.a": "y"}):
            self.assertEqual(self.tree.errors(), [])
        # The Makefile and CMakeLists.txt must build the same library.
        self.library("m_npu", "m_npu", cmake="m_other")
        self.assertError("detail/m_npu: its Makefile TARGET and CMakeLists.txt add_library name different "
                         "libraries (m_npu, m_other)")


    def test_pending_on_a_pull_request(self):
        """A new model's kernels and a new engine's libraries, not in src/
        yet: errors anywhere, pending (for the run to stage) with
        --pending-ok."""
        (self.tmp / dc.SHIPPED_XCLBINS / "M-1B-NPU2" / "layer.xclbin").unlink()
        self.library("m_npu", "m_npu")
        for rt in ("xrt", "hrx"):
            self.ship(f"src/lib/{rt}/m_npu.dll", f"src/lib/{rt}/m_npu.lib")
        self.assertError("src/xclbins/M-1B-NPU2/layer.xclbin is not shipped")
        self.assertError("m_npu builds m_npu, but src/lib/xrt/ lacks libm_npu.so")
        pending: list = []
        errors, _ = dc.run(self.tmp, pending=pending)
        self.assertEqual(errors, [])
        self.assertEqual(sorted(pending), ["src/lib/hrx/libm_npu.so", "src/lib/xrt/libm_npu.so",
                                           "src/xclbins/M-1B-NPU2/layer.xclbin"])
        # Only missing binaries are pending: anything else stays an error.
        self.ship(f"{dc.SHIPPED_XCLBINS}/M-1B-NPU2/stray.xclbin")
        errors, _ = dc.run(self.tmp, pending=[])
        self.assertTrue(any("stray.xclbin is shipped, but no" in e for e in errors), errors)

    def test_pending_cli(self):
        (self.tmp / dc.SHIPPED_XCLBINS / "M-1B-NPU2" / "layer.xclbin").unlink()
        target = self.tmp / "out.txt"
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(target)}), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(dc.main(["--repo", str(self.tmp), "--pending-ok", "--github-output"]), 0)
            self.assertEqual(target.read_text(encoding="utf-8"), "pending=src/xclbins/M-1B-NPU2/layer.xclbin\n")
            self.assertIn("pending swap: src/xclbins/M-1B-NPU2/layer.xclbin", out.getvalue())
            target.unlink()
            self.assertEqual(dc.main(["--repo", str(self.tmp), "--github-output"]), 1)
            self.assertEqual(target.read_text(encoding="utf-8"), "pending=\n")


class RealTree(unittest.TestCase):
    def test_tree_passes(self):
        errors, _ = dc.run(dc.REPO_ROOT)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
