"""Tests for changes.py: python3 .github/script/ci/test_changes.py"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import changes  # noqa: E402

SCRIPT = HERE / "changes.py"


def cli(*args: str, env: dict | None = None) -> dict[str, str]:
    """Runs changes.py and returns its key=value lines."""
    res = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
                         env=env, check=True)
    return dict(l.split("=", 1) for l in res.stdout.splitlines() if "=" in l and ":" not in l.split("=")[0])


def families(out: dict[str, str]) -> set[str]:
    return set(filter(None, out["families"].split(",")))


class FakeGit:
    """Stands in for changes.Git: fixed file contents and changed lines."""

    def __init__(self, base=None, head=None, lines=""):
        self.contents = {"base": base or {}, "head": head or {}}
        self.lines = lines

    def show(self, side, path):
        return self.contents[side].get(path)

    def changed_lines(self, paths):
        return self.lines


def catalog(**entries) -> str:
    """model_list.json text from id=(name, family) pairs, ids spelled name__tag."""
    models: dict = {}
    for mid, (name, family) in entries.items():
        base, tag = mid.split("__")
        models.setdefault(base, {})[tag] = {"name": name, "details": {"family": family}}
    return json.dumps({"model_path": "models", "models": models})


class Cli(unittest.TestCase):
    def test_docs_only(self):
        out = cli("--files", "docs/docs/models/d3.md", "README.md", "src/README.md",
                  "assets/logo.png", "src/tb_files/panda.png", "src/iceCream.jpg",
                  "src/home_install.sh~", ".github/ISSUE_TEMPLATE/bug.yml", "src/wix/.gitignore")
        self.assertEqual(out["run_tests"], "false")
        self.assertEqual(out["families"], "")
        self.assertEqual(out["asr_full"], "false")

    def test_packaged_text_runs(self):
        for f in ("LICENSE_BINARY.txt", "THIRD_PARTY_NOTICES.txt", "Dockerfile", "debian/control",
                  ".github/workflows/ci.yml", "src/CMakeLists.txt", "hrx-integration/x.patch"):
            self.assertEqual(cli("--files", f)["run_tests"], "true", f)

    def test_mixed(self):
        out = cli("--files", "docs/index.md", "src/runner/runner.cpp")
        self.assertEqual(out["run_tests"], "true")
        self.assertEqual(families(out), set())
        self.assertEqual(out["asr_full"], "false")

    def test_engine_library_maps_to_every_family_it_serves(self):
        out = cli("--files", "src/lib/hrx/libqwen3_npu.so")
        self.assertEqual(families(out), {"qwen3", "qwen3-it", "qwen3-tk", "deepseek-r1-0528"})
        self.assertEqual(families(cli("--files", "src/lib/xrt/llama_npu.dll")), {"llama3", "deepseek-r1"})

    def test_engine_paths(self):
        cases = {
            "src/include/AutoModel/modeling_lfm2.hpp": {"lfm2", "lfm2.5-tk"},
            "src/common/AutoModel/modeling_qwen3vl_image.cpp": {"qwen3vl", "qwen3vl-flash"},
            "src/include/models/gemma_text/flm/aie2p/gemma_text_gemm.hpp": {"gemma3-text"},
            "src/include/models/qwen3/core/x.hpp": {"qwen3"},
            "src/common/models/qwen35/core/dev04/a.cpp": {"qwen3.5"},
            "src/test/gemma4_core/t.cpp": {"gemma4e"},
            "src/test/qwen3_8_mtp_npu/t.cpp": {"qwen3.8-mtp", "qwen3.8-dflash2"},
            "src/detail/gemm/gemm.cpp": {"gemma4e", "gemma4e-flash"},
            "src/xclbins/Gemma3-4B-NPU2/a.xclbin": {"gemma3"},
            "src/common/AutoEmbeddingModel/modeling_gemma_embedding.cpp": {"embed-gemma"},
            "src/common/decision/py_json.cpp": {"d3"},
            "src/test/d3_contract/t.cpp": {"d3"},
            "src/include/whisper/whisper_npu.hpp": {"whisper-v3"},
            "src/test/phi4_core/test_phi4_frontend.cpp": {"phi4"},
            # Shared by every engine: the pr tier's one model per engine covers it.
            "src/lib/xrt/libgemm.so": set(),
            "src/common/AutoModel/automodel.cpp": set(),
            "src/test/phi4_core/test_model_downloader.cpp": set(),
            # The oga-hipep backend, which no NPU model runs.
            "src/common/oga/oga_runtime_install.cpp": set(),
            "src/include/oga/oga_api.hpp": set(),
            "src/test/oga_hipep/frontend/CMakeLists.txt": set(),
            # ...but for the header the qwen3.5 frontend includes in every package.
            "src/include/oga/oga_vision.hpp": {"qwen3.5"},
            # A built engine no catalog family uses.
            "src/test/qwen3_5_omni_npu/t.cpp": set(),
        }
        for path, expected in cases.items():
            self.assertEqual(families(cli("--files", path)), expected, path)

    def test_asr_paths(self):
        for f in ("src/common/whisper/modeling_whisper.cpp", "src/xclbins/Whisper-V3-Turbo-NPU2/x",
                  "src/lib/hrx/libwhisper_npu.so", "src/common/AutoModel/modeling_gemma4e_audio.cpp",
                  "src/detail/gemma4e_npu/gemma4e_audio.cpp", "src/server/multipart.cpp",
                  ".github/script/npu_test/assets/asr_eval/expected.json"):
            self.assertEqual(cli("--files", f)["asr_full"], "true", f)
        self.assertEqual(cli("--files", "src/detail/gemma4e_npu/gemma4e_image.cpp")["asr_full"], "false")
        # No git to read the changed lines from: assume they matter.
        self.assertEqual(cli("--files", "src/server/server.cpp")["asr_full"], "true")

    def test_unknown_base_runs_everything(self):
        out = cli("--base", "0" * 40, "--head", "HEAD")
        self.assertEqual((out["run_tests"], out["asr_full"], out["families"]), ("true", "true", ""))

    def test_github_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out"
            cli("--files", "docs/a.md", "--github-output", env={**os.environ, "GITHUB_OUTPUT": str(target)})
            lines = target.read_text(encoding="utf-8").splitlines()
        self.assertIn("run_tests=false", lines)
        self.assertEqual([l.split("=")[0] for l in lines],
                         ["run_tests", "families", "asr_full", "suites", "hipep",
                          "new_models", "hipep_models", "devkit", "devkit_areas", "devkit_families",
                          "devkit_overlay", "summary"])

    def test_suites_scoped_to_the_change(self):
        cases = {
            ("src/common/whisper/whisper.cpp",): "asr",
            (".github/script/npu_test/assets/asr_eval/manifest.json",): "asr",
            ("src/common/AutoDecisionModel/auto_decision_model.cpp",): "decision",
            (".github/script/npu_test/assets/d3_goldens.jsonl",): "decision",
            (".github/script/npu_test/suites/serving.py",): "serving",
            ("src/test/openai_api_test/test_errors.py",): "api",
            ("src/lib/xrt/libqwen3_npu.so",): "chat,prefix_cache",
            # The oga-hipep backend alone runs no NPU suite, only Test Windows HIPEP.
            ("src/common/oga/oga_hipep_backend.cpp",): "none",
            # Nor does shutdown, which tests no NPU model.
            (".github/script/npu_test/suites/shutdown.py",): "none",
            (".github/script/npu_test/console.py",): "none",
            (".github/script/npu_test/suites/shutdown.py", ".github/script/npu_test/suites/serving.py"):
                "serving",
            ("src/test/oga_hipep/test_oga_hipep.cpp", "src/lib/xrt/libqwen3_npu.so"): "chat,prefix_cache",
            ("src/include/oga/oga_vision.hpp",): "chat,prefix_cache",
            # Model-extension contracts, fixture and schemas: chat runs the extension jobs.
            ("src/test/model_frontend/fake_model_extension.c",): "chat",
            ("src/test/model_extension_gemma4e/model-list.json",): "chat",
            ("src/test/model_list_platform/test_model_list_platform.cpp",): "chat",
            ("src/schemas/model-catalog-overlay-v1.schema.json",): "chat",
            ("src/cmake/FastFlowLMModelExtension.cmake",): "chat",
            # An engine with an audio encoder is also tested by asr.
            ("src/lib/xrt/libgemma4e_npu.so",): "chat,prefix_cache,asr",
            ("src/test/openai_api_test/test_errors.py", "src/lib/xrt/libd3_npu.so"): "api,decision",
            # Shared code, or the harness itself, runs everything.
            ("src/server/server.cpp",): "all",
            (".github/script/npu_test/harness.py",): "all",
            ("src/common/whisper/whisper.cpp", "src/CMakeLists.txt"): "all",
        }
        for files, expected in cases.items():
            with self.subTest(files=files):
                self.assertEqual(cli("--files", *files)["suites"], expected)

    def test_oga_hipep_code_runs_only_hipep(self):
        out = cli("--files", "src/common/oga/oga_engine.cpp", "src/test/oga_hipep/test_oga_hipep.cpp")
        self.assertEqual((out["run_tests"], out["families"], out["suites"], out["hipep"]),
                         ("true", "", "none", "true"))

    def test_hipep_runs_when_the_change_reaches_its_models(self):
        cases = {
            # The backend, and the frontends of the families with oga-hipep models.
            ("src/test/oga_hipep/test_qwen_frontend.cpp",): "true",
            ("src/include/oga/oga_vision.hpp",): "true",
            ("src/common/AutoModel/modeling_qwen3_5vl.cpp",): "true",
            ("src/include/AutoModel/modeling_qwen3_5vl.hpp",): "true",
            ("src/common/AutoModel/modeling_qwen3_5vl_image.cpp",): "true",
            ("src/common/AutoModel/modeling_qwen2.cpp",): "true",
            ("src/common/AutoModel/modeling_gemma3.cpp",): "true",
            ("src/common/AutoModel/modeling_gemma4e.cpp",): "true",
            ("src/common/AutoModel/modeling_gpt_oss.cpp",): "true",
            # The tests of the suites it runs, and anything that runs every suite.
            (".github/script/npu_test/suites/chat.py",): "true",
            (".github/script/npu_test/suites/shutdown.py",): "true",
            (".github/script/npu_test/console.py",): "true",
            (".github/script/npu_test/assets/vision/cat.png",): "true",
            ("src/test/openai_api_test/test_errors.py",): "true",
            ("src/server/server.cpp",): "true",
            # The qwen3.5 NPU engine, other frontends, other suites, docs.
            ("src/lib/xrt/qwen3_5vl_npu.dll",): "false",
            ("src/common/models/qwen35/core/dev04/a.cpp",): "false",
            ("src/common/AutoModel/modeling_lfm2.cpp",): "false",
            ("src/common/AutoModel/modeling_gemma3_text.cpp",): "false",
            (".github/script/npu_test/suites/serving.py",): "false",
            ("src/common/whisper/whisper.cpp",): "false",
            ("docs/index.md",): "false",
        }
        for files, expected in cases.items():
            with self.subTest(files=files):
                self.assertEqual(cli("--files", *files)["hipep"], expected)


class Decide(unittest.TestCase):
    def test_model_list_change_names_changed_families(self):
        base = catalog(qwen3__0_6b=("Qwen3-0.6B-NPU2", "qwen3"), lfm2__1b=("LFM2", "lfm2"))
        head = json.loads(base)
        head["models"]["lfm2"]["1b"]["default_context_length"] = 4096
        head["models"]["d3"] = {"lite": {"name": "D3", "details": {"family": "d3"}}}
        head = json.dumps(head)
        git = FakeGit(base={changes.MODEL_LIST: base}, head={changes.MODEL_LIST: head})
        d = changes.decide([changes.MODEL_LIST], git, head)
        self.assertEqual(d.families, {"lfm2", "d3"})

    def test_oga_hipep_entries_name_no_family(self):
        base = catalog(qwen3__0_6b=("Qwen3-0.6B-NPU2", "qwen3"), lfm2__1b=("LFM2", "lfm2"))
        head = json.loads(base)
        head["models"]["gemma3-oga-hipep"] = {"4b": {"name": "Gemma3-4B_gpuep", "details": {"family": "gemma3"}}}
        gpu_only = json.dumps(head)
        git = FakeGit(base={changes.MODEL_LIST: base}, head={changes.MODEL_LIST: gpu_only})
        d = changes.decide([changes.MODEL_LIST], git, gpu_only)
        self.assertEqual((d.families, d.suites_value(), d.runs_hipep()), (set(), "none", True))
        self.assertEqual(d.outputs()["hipep_models"], "gemma3-oga-hipep:4b")
        # An NPU entry changed beside it still names its family.
        head["models"]["lfm2"]["1b"]["default_context_length"] = 4096
        both = json.dumps(head)
        git = FakeGit(base={changes.MODEL_LIST: base}, head={changes.MODEL_LIST: both})
        d = changes.decide([changes.MODEL_LIST], git, both)
        self.assertEqual((d.families, d.suites_value(), d.runs_hipep()), ({"lfm2"}, "chat,prefix_cache", True))
        # A removed one still runs HIPEP, but is no model to test.
        git = FakeGit(base={changes.MODEL_LIST: gpu_only}, head={changes.MODEL_LIST: base})
        d = changes.decide([changes.MODEL_LIST], git, base)
        self.assertEqual((d.runs_hipep(), d.outputs()["hipep_models"]), (True, ""))

    def test_backend_key_overrides_the_oga_hipep_suffix(self):
        base = catalog(**{"gemma3-oga-hipep__4b": ("Gemma3-4B_gpuep", "gemma3"), "lfm2__1b": ("LFM2", "lfm2")})

        def outcome(base, head):
            git = FakeGit(base={changes.MODEL_LIST: base}, head={changes.MODEL_LIST: head})
            d = changes.decide([changes.MODEL_LIST], git, head)
            return d.families, d.suites_value(), d.outputs()["hipep_models"]

        # An -oga-hipep entry moved to an NPU backend runs on the NPU, and
        # Test Windows HIPEP still runs, since it ran there before, though
        # no longer on that model.
        head = json.loads(base)
        head["models"]["gemma3-oga-hipep"]["4b"]["backend"] = "flm"
        moved = json.dumps(head)
        self.assertEqual(outcome(base, moved)[0], {"gemma3"})
        self.assertNotEqual(outcome(base, moved)[1], "none")
        self.assertEqual(outcome(base, moved)[2], "")
        git = FakeGit(base={changes.MODEL_LIST: base}, head={changes.MODEL_LIST: moved})
        self.assertTrue(changes.decide([changes.MODEL_LIST], git, moved).runs_hipep())
        # And an edit to it, once there, names its family too.
        head["models"]["gemma3-oga-hipep"]["4b"]["default_context_length"] = 4096
        self.assertEqual(outcome(moved, json.dumps(head))[0], {"gemma3"})
        # An entry that names the oga-hipep backend runs on the GPU whatever its name.
        head = json.loads(base)
        head["models"]["lfm2"]["1b"]["backend"] = "oga-hipep"
        self.assertEqual(outcome(json.dumps(head), json.dumps(head).replace("LFM2", "LFM2b")),
                         (set(), "none", "lfm2:1b"))

    def test_model_info_of_an_entry_moved_off_oga_hipep_names_its_family(self):
        base = catalog(**{"gemma3-oga-hipep__4b": ("x", "gemma3"), "lfm2__1b": ("y", "lfm2")})
        head = json.loads(base)
        head["models"]["gemma3-oga-hipep"]["4b"]["backend"] = "flm"
        head = json.dumps(head)
        git = FakeGit(base={changes.MODEL_LIST: base, changes.MODEL_INFO: '{}'},
                      head={changes.MODEL_LIST: head, changes.MODEL_INFO: '{"gemma3-oga-hipep:4b": 2}'})
        d = changes.decide([changes.MODEL_INFO], git, head)
        self.assertEqual(d.families, {"gemma3"})
        self.assertNotEqual(d.suites_value(), "none")
        self.assertTrue(d.runs_hipep())

    def test_catalog_fields_outside_models_run_every_suite(self):
        base = catalog(qwen3__0_6b=("Qwen3-0.6B-NPU2", "qwen3"), lfm2__1b=("LFM2", "lfm2"))

        def outcome(head):
            text = json.dumps(head)
            git = FakeGit(base={changes.MODEL_LIST: base}, head={changes.MODEL_LIST: text})
            d = changes.decide([changes.MODEL_LIST], git, text)
            return d.families, d.suites_value(), d.runs_hipep()

        head = json.loads(base)
        head["model_path"] = "models2"
        self.assertEqual(outcome(head), (set(), "all", True))
        # model_path is the folder of every model: an entry changed beside it narrows nothing.
        head["models"]["gemma3-oga-hipep"] = {"4b": {"name": "Gemma3-4B_gpuep", "details": {"family": "gemma3"}}}
        self.assertEqual(outcome(head), (set(), "all", True))
        head["models"]["lfm2"]["1b"]["default_context_length"] = 4096
        self.assertEqual(outcome(head), ({"lfm2"}, "all", True))
        # A field only one side has counts too.
        head["model_path"] = "models"
        head["schema"] = 2
        self.assertEqual(outcome(head), ({"lfm2"}, "all", True))

    def test_model_info_of_oga_hipep_entries_names_no_family(self):
        cat = catalog(**{"gemma3-oga-hipep__4b": ("x", "gemma3"), "lfm2__1b": ("y", "lfm2")})
        git = FakeGit(base={changes.MODEL_LIST: cat, changes.MODEL_INFO: '{"lfm2:1b": 1}'},
                      head={changes.MODEL_LIST: cat, changes.MODEL_INFO: '{"lfm2:1b": 1, "gemma3-oga-hipep:4b": 2}'})
        d = changes.decide([changes.MODEL_INFO], git, cat)
        self.assertEqual((d.families, d.suites_value(), d.runs_hipep()), (set(), "none", True))
        self.assertEqual(d.outputs()["hipep_models"], "gemma3-oga-hipep:4b")

    def test_model_list_without_git_adds_nothing(self):
        d = changes.decide([changes.MODEL_LIST], None, catalog(a__b=("A", "qwen3")))
        self.assertTrue(d.run_tests)
        self.assertEqual(d.families, set())

    def test_model_info_change_maps_through_catalog(self):
        cat = catalog(**{"qwen3-core__4b": ("x", "qwen3"), "lfm2__1b": ("y", "lfm2")})
        git = FakeGit(base={changes.MODEL_LIST: cat, changes.MODEL_INFO: '{"lfm2:1b": 1, "q": 2}'},
                      head={changes.MODEL_LIST: cat, changes.MODEL_INFO: '{"lfm2:1b": 3, "q": 2}'})
        self.assertEqual(changes.decide([changes.MODEL_INFO], git, cat).families, {"lfm2"})

    def test_shared_server_files_need_audio_lines(self):
        files = ["src/server/rest_handler.cpp"]
        self.assertFalse(changes.decide(files, FakeGit(lines="+  tag = \"qwen3\";")).asr_full)
        self.assertTrue(changes.decide(files, FakeGit(lines="-  // handle audio upload")).asr_full)

    def test_r2t2_engine_runs_the_whole_asr_set_on_r2t2(self):
        for f in ("src/lib/xrt/libqwen3_asr_npu.so", "src/lib/hrx/qwen3_asr_npu.dll",
                  "src/include/models/qwen3_asr/flm/aie2p/qwen3_asr_npu.hpp",
                  "src/include/AutoModel/modeling_qwen3_asr.hpp",
                  "src/common/AutoModel/modeling_qwen3_asr_audio.cpp",
                  "src/common/AutoModel/modeling_qwen3_asr_stream.cpp",
                  "src/xclbins/Confucius4-R2T2-NPU2/a.xclbin"):
            with self.subTest(f=f):
                out = cli("--files", f)
                self.assertEqual((families(out), out["suites"], out["asr_full"]),
                                 ({"r2t2"}, "asr", "true"))

    def test_new_models_are_entries_with_new_weights(self):
        base = json.loads(catalog(qwen3__0_6b=("Qwen3-0.6B-NPU2", "qwen3"), lfm2__1b=("LFM2", "lfm2"),
                                  **{"qwen3.5-oga-hipep__9b": ("Q35-9B_gpuep", "qwen3.5")}))
        for tags in base["models"].values():
            for e in tags.values():
                e["supported_platforms"] = ["aie2p"]
        base["models"]["qwen3.5-oga-hipep"]["9b"]["supported_platforms"] = None
        head = json.loads(json.dumps(base))
        # Metadata only: the runners already have these weights.
        head["models"]["lfm2"]["1b"]["default_context_length"] = 4096
        # New weights for an existing id.
        head["models"]["qwen3"]["0_6b"]["name"] = "Qwen3-0.6B-NPU2-v2"
        # A new NPU model, a new corelib-only one no runner can run, a new GPU one.
        head["models"]["big"] = {"120b": {"name": "Big-120B-NPU2", "supported_platforms": ["aie2p"],
                                          "details": {"family": "big"}},
                                 "120b-core": {"name": "Big-core", "supported_platforms": ["dev04"],
                                               "details": {"family": "big"}}}
        head["models"]["big-oga-hipep"] = {"70b": {"name": "Big_gpuep", "details": {"family": "big"}}}
        base, head = json.dumps(base), json.dumps(head)
        git = FakeGit(base={changes.MODEL_LIST: base}, head={changes.MODEL_LIST: head})
        d = changes.decide([changes.MODEL_LIST], git, head)
        self.assertEqual(d.new_models, ["big:120b", "qwen3:0_6b"])
        # A GPU entry is required on Test Windows HIPEP whether its weights are
        # new or it is only edited (its output is hipep_models).
        self.assertEqual(d.outputs()["hipep_models"], "big-oga-hipep:70b")
        self.assertEqual(d.outputs()["new_models"], "big:120b,qwen3:0_6b")
        self.assertTrue(d.runs_hipep())
        self.assertIn("big", d.families)
        # Without a base to compare, nothing is new: the run tests everything anyway.
        self.assertEqual(changes.decide([changes.MODEL_LIST], None, head).new_models, [])

    def test_republished_weights_are_new(self):
        """Weights republished under the same name: another file list, or a raised
        flm_min_version (which makes flm pull fetch them again)."""
        base = catalog(qwen3__0_6b=("Qwen3-0.6B-NPU2", "qwen3"), lfm2__1b=("LFM2", "lfm2"),
                       gemma3__1b=("Gemma3-1B", "gemma3-text"))
        for field_, value in (("files", ["model.q4nx", "extra.bin"]), ("file_url", "https://x"),
                              ("flm_min_version", "9.9.9")):
            with self.subTest(field=field_):
                head = json.loads(base)
                head["models"]["lfm2"]["1b"][field_] = value
                head = json.dumps(head)
                git = FakeGit(base={changes.MODEL_LIST: base}, head={changes.MODEL_LIST: head})
                self.assertEqual(changes.decide([changes.MODEL_LIST], git, head).new_models, ["lfm2:1b"])
        # New file hashes in model_info.json, by id or by model_info_key.
        cat = json.loads(base)
        cat["models"]["gemma3"]["1b"]["model_info_key"] = "g1"
        cat = json.dumps(cat)
        git = FakeGit(base={changes.MODEL_LIST: cat,
                            changes.MODEL_INFO: '{"lfm2:1b": [{"oid": "a"}], "g1": [{"oid": "b"}]}'},
                      head={changes.MODEL_LIST: cat,
                            changes.MODEL_INFO: '{"lfm2:1b": [{"oid": "c"}], "g1": [{"oid": "d"}]}'})
        d = changes.decide([changes.MODEL_INFO], git, cat)
        self.assertEqual((d.new_models, d.families), (["gemma3:1b", "lfm2:1b"], {"lfm2", "gemma3-text"}))

    def test_families_missing_from_catalog_are_dropped(self):
        d = changes.decide(["src/lib/hrx/libqwen3_npu.so"], None, catalog(a__b=("A", "qwen3")))
        self.assertEqual(d.families, {"qwen3"})

    def test_table_matches_catalog(self):
        """Every family the tables name exists in the real catalog, and every
        catalog family is reachable from some engine path."""
        text = (changes.REPO_ROOT / changes.MODEL_LIST).read_text(encoding="utf-8")
        known = {changes.family_of(e) for e in changes.load_catalog(text).values()}
        named = {f for fams in changes.ENGINES.values() for f in fams}
        named |= {f for fams in changes.CORE_FAMILIES.values() for f in fams}
        named |= {f for _, _, fams in changes.SUBSYSTEMS for f in fams}
        self.assertEqual(named - known, set())
        self.assertEqual(known - named, set())

    def test_gpu_families_come_from_the_head_catalog(self):
        """A frontend reaches Test Windows HIPEP when its family has a GPU entry
        in the head catalog, and any frontend does when there is no catalog."""
        cat = catalog(**{"gemma3-oga-hipep__4b": ("Gemma3-4B_gpuep", "gemma3"), "lfm2__1b": ("LFM2", "lfm2")})
        self.assertEqual(changes.gpu_families(changes.load_catalog(cat)), {"gemma3"})
        self.assertTrue(changes.decide(["src/common/AutoModel/modeling_gemma3.cpp"], None, cat).runs_hipep())
        self.assertFalse(changes.decide(["src/common/AutoModel/modeling_lfm2.cpp"], None, cat).runs_hipep())
        self.assertTrue(changes.decide(["src/common/AutoModel/modeling_lfm2.cpp"], None, None).runs_hipep())
        # The NPU engine of a family with a GPU entry does not reach it.
        self.assertFalse(changes.decide(["src/lib/xrt/gemma3_npu.dll"], None, cat).runs_hipep())


class DevKit(unittest.TestCase):
    """flm-dev-kit/ selects dev-kit areas, never product builds or suites."""

    def test_devkit_only_runs_no_product_job(self):
        out = cli("--files", "flm-dev-kit/FLM_Xclbin/MM/iron/mm_aie.py",
                  "flm-dev-kit/FLM_DLL/detail/qwen2_npu/qwen2_npu.cpp")
        self.assertEqual((out["run_tests"], out["families"], out["asr_full"], out["hipep"], out["new_models"]),
                         ("false", "", "false", "false", ""))
        self.assertEqual((out["devkit"], out["devkit_areas"]), ("true", "xclbin,dll"))
        self.assertIn("flm-dev-kit only", out["summary"])

    def test_each_area(self):
        cases = {
            "flm-dev-kit/FLM_Xclbin/Gemma4/gemma4_decoding/kernel/k.cc": "xclbin",
            "flm-dev-kit/FLM_DLL/detail/qwen2_npu/qwen2_npu.cpp": "dll",
            "flm-dev-kit/FLM_Q4NX_Converter/q4nx/model_converter.py": "converter",
            "flm-dev-kit/FLM_Pybind/SmolVLA/Makefile": "pybind",
            "flm-dev-kit/aie2p-launchlib/launchlib/tools/txn_dump.py": "launchlib",
            # In no area: every area, as an unknown product path runs every suite.
            "flm-dev-kit/env.sh": "xclbin,dll,converter,pybind,launchlib",
            # Files another area builds from (DEVKIT_ALSO), one per row.
            "flm-dev-kit/FLM_Xclbin/host_common/buffer.hpp": "xclbin,pybind",
            "flm-dev-kit/FLM_Xclbin/Whisper/encoder_MM/iron/mm.py": "xclbin,launchlib",
            "flm-dev-kit/FLM_DLL/cmake/flm_xrt_win.cmake": "dll,launchlib",
            "flm-dev-kit/FLM_DLL/detail/whisper_npu/whisper_encoder.cpp": "dll,launchlib",
            "flm-dev-kit/FLM_DLL/include/buffer.hpp": "xclbin,dll,launchlib",
            "flm-dev-kit/FLM_DLL/detail/q4nx/q4_npu_eXpress.cpp": "xclbin,dll,launchlib",
            "flm-dev-kit/FLM_DLL/detail/hunyuan_npu/hunyuan_npu.cpp": "xclbin,dll",
            "flm-dev-kit/FLM_DLL/detail/minicpm_ocr_npu/image/npu/x.cpp": "xclbin,dll",
            # The RTP lock goes with an engine change.
            "flm-dev-kit/FLM_Xclbin/rtp_addresses.lock.txt": "xclbin,dll",
        }
        for path, areas in cases.items():
            with self.subTest(path=path):
                out = cli("--files", path)
                self.assertEqual((out["run_tests"], out["devkit"], out["devkit_areas"]), ("false", "true", areas))

    def test_devkit_docs_are_nothing(self):
        out = cli("--files", "flm-dev-kit/README.md", "flm-dev-kit/AGENTS.md",
                  "flm-dev-kit/FLM_Xclbin/kernel_assignment.md", "flm-dev-kit/FLM_DLL/.gitignore")
        self.assertEqual((out["run_tests"], out["devkit"], out["devkit_areas"]), ("false", "false", ""))
        self.assertEqual(out["summary"], "documentation only: NPU tests skipped")

    def test_mixed_keeps_product_rules_for_product_paths(self):
        out = cli("--files", "flm-dev-kit/FLM_DLL/detail/qwen3_npu/qwen3_npu.cpp", "src/lib/xrt/libqwen3_npu.so")
        self.assertEqual((out["run_tests"], out["suites"], out["devkit"], out["devkit_areas"]),
                         ("true", "chat,prefix_cache", "true", "dll"))
        self.assertEqual(families(out), {"qwen3", "qwen3-it", "qwen3-tk", "deepseek-r1-0528"})
        out = cli("--files", "flm-dev-kit/FLM_Xclbin/MM/Makefile", "src/server/server.cpp")
        self.assertEqual((out["run_tests"], out["suites"], out["devkit_areas"]), ("true", "all", "xclbin"))
        # Product documentation beside dev-kit code still builds no product.
        out = cli("--files", "flm-dev-kit/FLM_Xclbin/MM/Makefile", "docs/index.md")
        self.assertEqual((out["run_tests"], out["devkit_areas"]), ("false", "xclbin"))

    def test_product_files_the_devkit_links_select_its_areas(self):
        # The prebuilt aiebu: every area that links it, and the product rules
        # (a shared library: every suite) still apply.
        for path in ("src/lib/xrt/libaiebu.a", "src/lib/xrt/aiebu_static.lib"):
            with self.subTest(path=path):
                out = cli("--files", path)
                self.assertEqual((out["run_tests"], out["suites"], out["devkit"], out["devkit_areas"]),
                                 ("true", "all", "true", "xclbin,dll,pybind,launchlib"))
        # Anything else in src/lib/xrt is on the dev-kit engines' -L path.
        out = cli("--files", "src/lib/xrt/libqwen3_npu.so")
        self.assertEqual((out["run_tests"], out["devkit_areas"]), ("true", "dll"))
        self.assertEqual(families(out), {"qwen3", "qwen3-it", "qwen3-tk", "deepseek-r1-0528"})
        # The HRX engine builds fetch the pinned package through hrx-integration/:
        # the dll area and every dev-kit engine's families; the product rules
        # still run everything.
        for path in ("hrx-integration/hrx-release.env", "hrx-integration/fetch-hrx-release.ps1"):
            with self.subTest(path=path):
                out = cli("--files", path)
                self.assertEqual((out["run_tests"], out["devkit"], out["devkit_areas"]),
                                 ("true", "true", "dll"))
                self.assertEqual(devkit_families(out), changes.every_engine_family(changes.load_devkit_engines()))
        # setup_windows_deps.ps1 builds xrt_coreutil.lib from .github/windows-deps/,
        # which the dev-kit Windows engines link: the dll area too.
        out = cli("--files", ".github/windows-deps/xrt_coreutil.def")
        self.assertEqual((out["devkit"], out["devkit_areas"]), ("true", "dll"))
        self.assertEqual(devkit_families(out), changes.every_engine_family(changes.load_devkit_engines()))
        # The dev kit reads nothing from src/lib/hrx or src/lib itself.
        for path in ("src/lib/hrx/libaiebu.a", "src/lib/libcurl.dll", "src/server/server.cpp"):
            with self.subTest(path=path):
                self.assertEqual(cli("--files", path)["devkit"], "false")

    def test_devkit_ci_selects_every_area(self):
        for path in (".github/script/ci/changes.py", ".github/workflows/build-and-test.yml",
                     ".github/workflows/release.yml", ".github/actions/detect-changes/action.yml",
                     ".github/actions/devkit-build-lib-lin/action.yml",
                     ".github/actions/devkit-build-lib-win/action.yml", ".github/workflows/devkit-engines.yml",
                     ".github/workflows/devkit-xclbins.yml", ".github/actions/devkit-xclbin-plan/action.yml",
                     ".github/actions/devkit-build-xclbins/action.yml",
                     ".github/actions/devkit-assemble-xclbins/action.yml",
                     ".github/script/ci/devkit_xclbin_plan.py", ".github/script/ci/devkit_assemble_xclbins.py"):
            with self.subTest(path=path):
                out = cli("--files", path)
                self.assertEqual((out["run_tests"], out["devkit"], out["devkit_areas"]),
                                 ("true", "true", "xclbin,dll,converter,pybind,launchlib"))
        self.assertEqual(cli("--files", ".github/workflows/lin-build.yml")["devkit"], "false")

    def test_kernel_image_selects_nothing(self):
        # The aie-build image, its check, its pins and the workflow that
        # builds it: devkit-aie-image.yml tests them, so no dev-kit area (the
        # kernel jobs build in the published image), no product build or
        # suite, no engine family.
        paths = ("docker/aie-build.Dockerfile", "docker/aie-build-check.sh", "docker/aie-build.pdi.sha256",
                 "flm-dev-kit/FLM_Xclbin/toolchain.lock", ".github/workflows/devkit-aie-image.yml")
        for path in paths:
            with self.subTest(path=path):
                out = cli("--files", path)
                self.assertEqual((out["run_tests"], out["families"], out["devkit"], out["devkit_areas"],
                                  out["devkit_families"]), ("false", "", "false", "", ""))
                self.assertIn("kernel build image only", out["summary"])
        out = cli("--files", *paths)
        self.assertEqual((out["run_tests"], out["devkit"], out["devkit_areas"]), ("false", "false", ""))
        # The pinned digest is what the kernel jobs build in: the xclbin area.
        out = cli("--files", "flm-dev-kit/FLM_Xclbin/toolchain.image", "flm-dev-kit/FLM_Xclbin/toolchain.lock")
        self.assertEqual((out["run_tests"], out["devkit_areas"]), ("false", "xclbin"))
        self.assertIn("flm-dev-kit only", out["summary"])
        # Beside a kernel source, the kernel source still selects its area.
        out = cli("--files", "docker/aie-build.Dockerfile", "flm-dev-kit/FLM_Xclbin/MM/Makefile")
        self.assertEqual((out["run_tests"], out["devkit_areas"]), ("false", "xclbin"))
        # Beside product code, the product rules still see the product file.
        out = cli("--files", "docker/aie-build.Dockerfile", "src/server/server.cpp")
        self.assertEqual((out["run_tests"], out["suites"], out["devkit_areas"]), ("true", "all", ""))
        # The product's own build image is product code, not the dev kit's.
        out = cli("--files", "Dockerfile")
        self.assertEqual((out["run_tests"], out["devkit"]), ("true", "false"))
        # Other dev-kit workflows are still the dev-kit CI: every area.
        out = cli("--files", ".github/workflows/devkit-xclbins.yml")
        self.assertEqual(out["devkit_areas"], "xclbin,dll,converter,pybind,launchlib")

    def test_overlay(self):
        # A dev-kit engine: the overlay runs, and no product suite (run_tests
        # false: only the portable packages it starts from build).
        out = cli("--files", "flm-dev-kit/FLM_DLL/detail/qwen3_npu/qwen3_npu.cpp")
        self.assertEqual((out["run_tests"], out["suites"], out["hipep"], out["devkit_overlay"]),
                         ("false", "none", "false", "true"))
        self.assertIn("portable packages built for the overlay", out["summary"])
        # A kernel: the kernel build names the models, so the overlay runs.
        out = cli("--files", "flm-dev-kit/FLM_Xclbin/MM/iron/mm_aie.py")
        self.assertEqual((out["run_tests"], out["suites"], out["devkit_families"], out["devkit_overlay"]),
                         ("false", "none", "", "true"))
        # Areas with no NPU test, and dll files that build no engine: none.
        for path in ("flm-dev-kit/FLM_Q4NX_Converter/q4nx/model_converter.py",
                     "flm-dev-kit/FLM_Pybind/SmolVLA/Makefile",
                     "flm-dev-kit/aie2p-launchlib/launchlib/tools/txn_dump.py",
                     "flm-dev-kit/FLM_DLL/build_qwen3vl.sh",
                     "flm-dev-kit/FLM_DLL/model_list.json"):
            with self.subTest(path=path):
                out = cli("--files", path)
                self.assertEqual((out["run_tests"], out["suites"], out["devkit_overlay"]),
                                 ("false", "none", "false"))
                self.assertIn("product builds and NPU tests skipped", out["summary"])
        # Beside product code the product rules still decide the suites.
        out = cli("--files", "flm-dev-kit/FLM_DLL/detail/qwen3_npu/qwen3_npu.cpp", "src/server/server.cpp")
        self.assertEqual((out["run_tests"], out["suites"], out["devkit_overlay"]), ("true", "all", "true"))
        # The dev-kit CI and unknown changes: every family, so the overlay too.
        self.assertEqual(cli("--files", ".github/workflows/build-and-test.yml")["devkit_overlay"], "true")
        self.assertEqual(cli("--base", "0" * 40, "--head", "HEAD")["devkit_overlay"], "true")
        # Documentation and product-only changes: none.
        self.assertEqual(cli("--files", "docs/a.md")["suites"], "none")
        self.assertEqual(cli("--files", "src/server/server.cpp")["devkit_overlay"], "false")

    def test_unknown_files_select_every_area(self):
        out = cli("--base", "0" * 40, "--head", "HEAD")
        self.assertEqual((out["devkit"], out["devkit_areas"]), ("true", "xclbin,dll,converter,pybind,launchlib"))

    def test_no_changed_files_select_no_area(self):
        d = changes.decide([])
        self.assertEqual(d.outputs()["devkit"], "false")


def devkit_families(out: dict[str, str]) -> set[str]:
    return set(filter(None, out["devkit_families"].split(",")))


class DevKitFamilies(unittest.TestCase):
    """devkit_families: the families of the dev-kit engines a change touches,
    through flm-dev-kit/FLM_DLL/engines.json."""

    @classmethod
    def setUpClass(cls):
        cls.engines = changes.load_devkit_engines()
        assert cls.engines is not None, changes.DEVKIT_ENGINES
        cls.all = cls.engines.all_families()

    def test_engine_dir_maps_to_its_families(self):
        out = cli("--files", "flm-dev-kit/FLM_DLL/detail/qwen3_npu/qwen3_npu.cpp")
        self.assertEqual(devkit_families(out), {"qwen3", "qwen3-it", "qwen3-tk", "deepseek-r1-0528"})
        self.assertEqual((out["run_tests"], out["families"]), ("false", ""))
        # r2t2's engine (#1068): ASR only, through either spelling of its dirs.
        for path in ("flm-dev-kit/FLM_DLL/detail/qwen3_asr_npu/qwen3_asr_npu.cpp",
                     "flm-dev-kit/FLM_DLL/include/models/qwen3_asr/qwen3_asr_npu.hpp"):
            self.assertEqual(devkit_families(cli("--files", path)), {"r2t2"}, path)

    def test_engines_that_use_another_add_their_families(self):
        # d3_npu compiles qwen3_5vl_npu's vision sources; gemma4e_flash,
        # gemma4e_dflash and gemma4e_dflash_flash include gemma4e's headers.
        self.assertEqual(devkit_families(cli("--files", "flm-dev-kit/FLM_DLL/detail/qwen3_5vl_npu/rot_pos_emb.cpp")),
                         {"qwen3.5", "d3"})
        self.assertEqual(devkit_families(cli("--files", "flm-dev-kit/FLM_DLL/include/models/gemma4e/gemma4e_npu.hpp")),
                         {"gemma4e", "gemma4e-flash", "gemma4e-dflash", "gemma4e-dflash-flash"})
        # A user does not pull in what it uses.
        self.assertEqual(devkit_families(cli("--files", "flm-dev-kit/FLM_DLL/detail/d3_npu/d3_npu.cpp")), {"d3"})

    def test_shared_code_maps_to_every_family(self):
        for path in ("flm-dev-kit/FLM_DLL/detail/q4nx/q4_npu_eXpress.cpp",
                     "flm-dev-kit/FLM_DLL/detail/mha/mha.cpp",
                     "flm-dev-kit/FLM_DLL/detail/gemm/gemm.cpp",
                     "flm-dev-kit/FLM_DLL/detail/dequant/dequant.cpp",
                     "flm-dev-kit/FLM_DLL/detail/lm_head/lm_head.cpp",
                     "flm-dev-kit/FLM_DLL/detail/vision_common/conv3dPatchEmbedding.cpp",
                     "flm-dev-kit/FLM_DLL/include/buffer.hpp",
                     "flm-dev-kit/FLM_DLL/include/models/nosuch/x.hpp",
                     "flm-dev-kit/FLM_DLL/detail/nosuch/x.cpp",
                     "flm-dev-kit/FLM_DLL/common/modules/sampler.cpp",
                     "flm-dev-kit/FLM_DLL/cmake/flm_hrx.cmake",
                     "flm-dev-kit/FLM_DLL/build.sh",
                     "flm-dev-kit/FLM_DLL/build_manylinux.sh",
                     "flm-dev-kit/FLM_DLL/hrx.mk",
                     # Not dev-kit engine code, but the engines link or run under them.
                     "src/lib/xrt/libaiebu.a",
                     "src/lib/xrt/libqwen3_npu.so",
                     ".github/script/ci/changes.py",
                     "flm-dev-kit/env.sh"):
            with self.subTest(path=path):
                self.assertEqual(devkit_families(cli("--files", path)), self.all)
        self.assertEqual(devkit_families(cli("--base", "0" * 40, "--head", "HEAD")), self.all)

    def test_no_engine_change_gives_no_families(self):
        for path in ("flm-dev-kit/FLM_DLL/build_qwen3vl.sh",
                     "flm-dev-kit/FLM_DLL/model_list.json",
                     "flm-dev-kit/FLM_DLL/README.md",
                     "flm-dev-kit/FLM_DLL/detail/qwen3_npu/README.md",
                     "flm-dev-kit/FLM_Xclbin/MM/host7_mt_variant_bias/iron/mm.py",
                     "flm-dev-kit/FLM_Q4NX_Converter/convert.py",
                     "src/lib/hrx/libqwen3_npu.so",
                     "src/server/server.cpp"):
            with self.subTest(path=path):
                self.assertEqual(cli("--files", path)["devkit_families"], "")

    def test_engines_json_change_tests_the_changed_entries(self):
        head = json.loads((changes.REPO_ROOT / changes.DEVKIT_ENGINES).read_text(encoding="utf-8"))
        base = json.loads(json.dumps(head))
        base["engines"]["qwen2_npu"]["models"] = []
        del base["engines"]["lfm3_npu"]
        git = FakeGit(base={changes.DEVKIT_ENGINES: json.dumps(base)})
        d = changes.decide([changes.DEVKIT_ENGINES], git, None, changes.DevkitEngines.parse(json.dumps(head)))
        self.assertEqual(d.devkit_families, {"qwen2", "lfm3"})
        # Without the base to compare with: every family.
        d = changes.decide([changes.DEVKIT_ENGINES], None, None, changes.DevkitEngines.parse(json.dumps(head)))
        self.assertEqual(d.devkit_families, self.all)

    def test_malformed_engines_json_reads_as_unreadable(self):
        good = {"shared": ["q4nx"], "engines": {"a_npu": {"families": ["a"], "models": [], "uses": []}}}
        self.assertEqual(changes.DevkitEngines.parse(json.dumps(good)).families("a_npu"), {"a"})
        for field, value in (("families", "qwen2"), ("uses", "b_npu"), ("models", [1])):
            bad = json.loads(json.dumps(good))
            bad["engines"]["a_npu"][field] = value
            with self.subTest(field=field):
                self.assertIsNone(changes.DevkitEngines.parse(json.dumps(bad)))
        self.assertIsNone(changes.DevkitEngines.parse(json.dumps({**good, "engines": {"a_npu": []}})))
        self.assertIsNone(changes.DevkitEngines.parse(json.dumps({**good, "shared": "q4nx"})))

    def test_engines_json_agrees_with_engine_table(self):
        """Every ENGINES row naming a dev-kit engine (<name> or <name>_npu
        under flm-dev-kit/FLM_DLL/detail) has that engine's families, every
        dev-kit engine has an ENGINES row, and its families are catalog
        families of the models it serves."""
        text = (changes.REPO_ROOT / changes.MODEL_LIST).read_text(encoding="utf-8")
        cat = changes.load_catalog(text)
        known = {changes.family_of(e) for e in cat.values()}
        engines = self.engines.engines
        for names, fams in changes.ENGINES.items():
            for name in names:
                eng = self.engines.header_owner(name)
                if eng is None:
                    continue
                with self.subTest(engine=eng, names=names):
                    self.assertEqual(sorted(engines[eng]["families"]), sorted(fams))
        for eng, entry in engines.items():
            with self.subTest(engine=eng):
                name = eng[:-4] if eng.endswith("_npu") else eng
                self.assertIn(name, changes.ENGINE_FAMILIES, f"{eng} has no row in changes.ENGINES")
                self.assertEqual(set(entry["families"]) - known, set())
                served = {changes.family_of(cat[m]) for m in entry["models"] if m in cat}
                self.assertEqual(served - set(entry["families"]), set())


class Prebuilts(unittest.TestCase):
    """src/lib and src/xclbins map to families; shared or unknown files run all."""

    def test_engine_library_maps_to_its_family(self):
        cases = {
            "src/lib/xrt/libgemma_text_npu.so": {"gemma3-text"},
            "src/lib/hrx/gemma4e_dflash.dll": {"gemma4e-dflash"},
            "src/lib/hrx/gemma4e_dflash.lib": {"gemma4e-dflash"},
            "src/lib/xrt/libqwen3vl_flash.so": {"qwen3vl-flash"},
            "src/lib/hrx/libhy_mem_npu.so": {"hy-memory"},
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                out = cli("--files", path)
                self.assertEqual(families(out), expected)
                self.assertNotEqual(out["suites"], "all")

    def test_shared_libraries_run_every_suite(self):
        for path in ("src/lib/xrt/libaiebu.a", "src/lib/xrt/aiebu_static.lib", "src/lib/hrx/libmha.so",
                     "src/lib/xrt/q4_npu_eXpress.dll", "src/lib/hrx/libhrx_runtime.so", "src/lib/libcurl.dll",
                     "src/lib/xrt/sub/libqwen3_npu.so"):
            with self.subTest(path=path):
                d = changes.decide([path])
                self.assertEqual((d.families, d.suites_value()), (set(), "all"))
                self.assertTrue(any("not one engine's" in r for r in d.reasons), d.reasons)

    def test_xclbin_folder_maps_to_the_families_that_load_it(self):
        cat = json.dumps({"model_path": "m", "models": {
            "base": {"1b": {"name": "Base-NPU2", "details": {"family": "fam-a"}}},
            "draft": {"1b": {"name": "Draft-NPU2", "base_model": "base:1b", "xclbin_folder": "Base-NPU2",
                             "details": {"family": "fam-b"}}},
            # xclbin_folder counts on any entry, dependant or not, as in
            # AutoModel::_shared_initialize_model_state.
            "other": {"1b": {"name": "Other-NPU2", "xclbin_folder": "Base-NPU2",
                             "details": {"family": "fam-c"}}}}})
        d = changes.decide(["src/xclbins/Base-NPU2/attn.xclbin"], None, cat)
        self.assertEqual(d.families, {"fam-a", "fam-b", "fam-c"})
        for folder in ("Draft-NPU2", "Other-NPU2"):
            with self.subTest(folder=folder):
                d = changes.decide([f"src/xclbins/{folder}/attn.xclbin"], None, cat)
                self.assertEqual((d.families, d.suites_value()), (set(), "all"))
                self.assertTrue(any("no catalog entry loads this folder" in r for r in d.reasons), d.reasons)
        # The real catalog: qwen3.8-dflash2 runs Qwen3.8-27B-NPU2's kernels.
        out = cli("--files", "src/xclbins/Qwen3.8-27B-NPU2/layer.xclbin")
        self.assertEqual(families(out), {"qwen3.8-mtp", "qwen3.8-dflash2"})
        self.assertEqual(out["suites"], "chat,prefix_cache")

    def test_unknown_xclbin_paths_run_every_suite(self):
        for path in ("src/xclbins/No-Such-Model-NPU2/a.xclbin", "src/xclbins/README.txt"):
            with self.subTest(path=path):
                out = cli("--files", path)
                self.assertEqual((families(out), out["suites"]), (set(), "all"))

    def test_every_prebuilt_in_the_tree_is_mapped(self):
        """Every src/xclbins folder is some catalog entry's, and every engine
        library in src/lib/{xrt,hrx} is in ENGINES, bar the shared units."""
        cat = changes.load_catalog((changes.REPO_ROOT / changes.MODEL_LIST).read_text(encoding="utf-8"))
        folders = changes.xclbin_folders(cat)
        xclbins = changes.REPO_ROOT / "src/xclbins"
        if xclbins.is_dir():
            for d in sorted(p.name for p in xclbins.iterdir() if p.is_dir()):
                self.assertTrue(folders.get(d), f"src/xclbins/{d} is no catalog entry's folder")
        shared = {"q4_npu_eXpress", "gemm", "mha", "dequant", "lm_head", "aiebu", "aiebu_static"}
        for runtime in ("xrt", "hrx"):
            lib = changes.REPO_ROOT / "src/lib" / runtime
            if not lib.is_dir():
                continue
            for f in sorted(p.name for p in lib.iterdir() if p.is_file()):
                why, fams = changes.engine_families(f"src/lib/{runtime}/{f}", {})
                stem = re.sub(r"^lib|\.(so|dll|lib|a)$", "", f)
                if stem in shared:
                    self.assertEqual(fams, (), f)
                elif stem.removesuffix("_npu") != "qwen3_5_omni":
                    self.assertTrue(fams, f"src/lib/{runtime}/{f}: {why}")


@unittest.skipUnless(shutil.which("git"), "git not installed")
class RealGit(unittest.TestCase):
    """base...head through a throwaway repository."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.git("init", "-q")
        self.write(changes.MODEL_LIST, catalog(qwen3__0_6b=("Qwen3-0.6B-NPU2", "qwen3"),
                                               gemma3__1b=("Gemma3-1B-NPU2", "gemma3-text")))
        self.write("src/server/server.cpp", "int main() {}\n")
        self.base = self.commit()

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=t", "-c", "user.email=t@t",
                               *args], capture_output=True, text=True, check=True).stdout.strip()

    def write(self, rel, text):
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "c")
        return self.git("rev-parse", "HEAD")

    def run_cli(self):
        return cli("--repo", str(self.repo), "--base", self.base, "--head", self.commit())

    def test_model_list_and_server_lines(self):
        cat = json.loads((self.repo / changes.MODEL_LIST).read_text(encoding="utf-8"))
        cat["models"]["gemma3"]["1b"]["flm_min_version"] = "0.9.40"
        self.write(changes.MODEL_LIST, json.dumps(cat))
        self.write("src/server/server.cpp", "int main() { /* new tag */ }\n")
        out = self.run_cli()
        self.assertEqual((out["run_tests"], out["families"], out["asr_full"]), ("true", "gemma3-text", "false"))

    def test_docs_only_and_audio_line(self):
        self.write("docs/index.md", "hi\n")
        self.assertEqual(self.run_cli()["run_tests"], "false")
        self.write("src/server/server.cpp", "int main() { transcribe(); }\n")
        self.assertEqual(self.run_cli()["asr_full"], "true")


if __name__ == "__main__":
    unittest.main()
