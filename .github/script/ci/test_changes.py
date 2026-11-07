"""Tests for changes.py: python3 .github/script/ci/test_changes.py"""

from __future__ import annotations

import json
import os
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
            "src/test/qwen3_8_mtp_npu/t.cpp": {"qwen3.8-mtp"},
            "src/detail/gemm/gemm.cpp": {"gemma4e", "gemma4e-flash"},
            "src/xclbins/Gemma3-4B-NPU2/a.xclbin": {"gemma3"},
            "src/common/AutoEmbeddingModel/modeling_gemma_embedding.cpp": {"embed-gemma"},
            "src/common/decision/py_json.cpp": {"d3"},
            "src/test/d3_contract/t.cpp": {"d3"},
            "src/include/whisper/whisper_npu.hpp": {"whisper-v3"},
            # Shared by every engine: the pr tier's one model per engine covers it.
            "src/lib/xrt/libgemm.so": set(),
            "src/common/AutoModel/automodel.cpp": set(),
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
        self.assertEqual([l.split("=")[0] for l in lines], ["run_tests", "families", "asr_full", "suites", "summary"])

    def test_suites_scoped_to_the_change(self):
        cases = {
            ("src/common/whisper/whisper.cpp",): "asr",
            (".github/script/npu_test/assets/asr_eval/manifest.json",): "asr",
            ("src/common/AutoDecisionModel/auto_decision_model.cpp",): "decision",
            (".github/script/npu_test/assets/d3_goldens.jsonl",): "decision",
            (".github/script/npu_test/suites/serving.py",): "serving",
            ("src/test/openai_api_test/test_errors.py",): "api",
            ("src/lib/xrt/libqwen3_npu.so",): "chat,prefix_cache",
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
