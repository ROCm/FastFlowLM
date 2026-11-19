#!/usr/bin/env python3
"""Decides, from the files a pull request or push changes, what CI must test.

    changes.py --base <sha> --head <sha> [--github-output]
    changes.py --files src/server/server.cpp docs/index.md

Outputs (printed, and appended to $GITHUB_OUTPUT with --github-output):
  run_tests  false only when every changed file is documentation or otherwise
             never built or packaged, so no NPU runner is reserved for it
  families   comma-separated model families (details.family in
             src/model_list.json) whose engine or NPU catalog entry changed;
             the pr tier tests every model in them instead of one per engine
  asr_full   true when speech-recognition code changed, so the pr tier runs
             the whole ASR evaluation set
  suites     "all", or the comma-separated npu_test suites the change can
             affect when every changed file is scoped to some (an ASR-only
             change runs asr, a d3-only change runs decision); "none" when
             every changed file is oga-hipep code, a GPU catalog entry or a
             test of the shutdown suite, which no NPU job runs; a change to a
             model_list.json field outside its model entries, such as
             model_path, gives "all"
  hipep      true when the change can reach an oga-hipep model, which Test
             Windows HIPEP runs on the GPU (one per family): when every suite
             runs, or a changed file is oga-hipep code, a GPU catalog entry,
             the frontend of a family with a GPU entry, or a test of the chat,
             api and shutdown suites that job runs
  new_models comma-separated NPU catalog entries (model ids) the change adds,
             or whose weights it points at another package (name or url):
             entries no runner has yet. The test jobs download them whatever
             their size, outside the pr tier's download budget, and fail if
             they cannot; only entries an NPU2 (aie2p) runner can run
  hipep_models
             comma-separated oga-hipep entries (model ids) the change adds or
             edits in model_list.json or model_info.json, new weights or not
             (an edit names no family for the GPU, so this is how one gets
             tested). Test Windows HIPEP requires them, as the NPU jobs do
             new_models: downloaded whatever their size, and a failure if
             they cannot be
  devkit     true when a change needs a dev-kit job: code under flm-dev-kit/
             (the kernel, engine, converter and pybind sources, built by their
             own jobs), a src/lib/xrt file the dev kit links (libaiebu.a,
             aiebu_static.lib, any library its engines' -L order can pick
             up), hrx-integration/ (the HRX engine builds fetch the pinned
             package with it), or the CI that runs the dev-kit jobs
             (including .github/workflows/devkit-*.yml); documentation under
             flm-dev-kit/ counts as nothing, as for the product. Nothing in
             the product consumes flm-dev-kit/ yet, so those paths select no
             product build, suite or family: a change only there builds and
             tests no product package (run_tests false). The product files
             above keep their product rules
  devkit_areas
             comma-separated dev-kit areas the change touches, from xclbin,
             dll, converter, pybind and launchlib, plus the areas that build
             from a changed file of another area, of src/lib/xrt or of
             hrx-integration/; every
             area when the changed files are unknown, a changed dev-kit file
             is in none of them, or the dev-kit CI changed
  devkit_families
             comma-separated catalog families whose dev-kit engines the change
             touches, for the dev-kit overlay tests: from
             flm-dev-kit/FLM_DLL/engines.json, a change under
             detail/<engine>/ or its include/models/<x>/ headers gives the
             engine's families and those of every engine that uses it; a
             change to anything else that builds into the engines (the
             shared units, the common headers, common/, cmake/, the build
             scripts, a src/lib/xrt library they link), to the dev-kit CI, or
             an unknown change gives every engine's families. Per-model
             helper scripts (build_<model>.sh) and model_list*.json give
             none; an engines.json change gives the engines whose entry
             changed. Empty when no dev-kit engine is touched
  summary    one line for the job log

When in doubt it tests more: an all-zero or missing base (a new branch), or a
range git cannot diff (shallow clone, force-push), counts as "everything
changed" (every dev-kit area too). It still names no families then: the pr tier already runs one model
of every engine, and testing every model of every family is the nightly job.

Standard library only; --files skips git for the file list, for tests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_LIST = "src/model_list.json"
MODEL_INFO = "src/model_info.json"

# Files that are never built, packaged or read by a test. A change made only
# of these skips the NPU tests. LICENSE_*.txt and THIRD_PARTY_NOTICES.txt are
# not here: the packages ship them.
NON_CODE = re.compile(
    r"^(docs/"
    r"|assets/"                 # README images and the like, at the repo root
    r"|src/tb_files/"
    r"|src/iceCream\.jpg$"
    r"|\.github/ISSUE_TEMPLATE/"
    r"|\.github/(PULL_REQUEST_TEMPLATE[^/]*|CODEOWNERS|FUNDING\.yml)$"
    r"|(.*/)?\.gitignore$|\.mailmap$"
    r"|.*\.md$"
    r"|.*~$)",                  # editor backups such as src/home_install.sh~
    re.IGNORECASE)

# The FLM dev kit: kernel, engine, converter and pybind sources imported from
# FastFlowLM_IRON, built and tested by dev-kit jobs of their own. No product
# build reads them yet, so they select no product build or suite (a product
# change that starts consuming one must add it to the product rules here).
# Area -> the prefix that selects it, in output order.
DEVKIT_ROOT = "flm-dev-kit/"
DEVKIT_AREAS = (
    ("xclbin", "flm-dev-kit/FLM_Xclbin/"),
    ("dll", "flm-dev-kit/FLM_DLL/"),
    ("converter", "flm-dev-kit/FLM_Q4NX_Converter/"),
    ("pybind", "flm-dev-kit/FLM_Pybind/"),
    ("launchlib", "flm-dev-kit/aie2p-launchlib/"),
)


ALL_DEVKIT_AREAS = frozenset(area for area, _ in DEVKIT_AREAS)

# Areas that build from another area's files, beyond the one their prefix
# selects: dev-kit path pattern -> the extra areas it selects.
DEVKIT_ALSO = (
    # FLM_Pybind/SmolVLA/Makefile: HOST_COMMON_DIR := ../../FLM_Xclbin/host_common.
    (re.compile(r"^flm-dev-kit/FLM_Xclbin/host_common/"), {"pybind"}),
    # aie2p-launchlib/whisper_encoder/tools build from FLM_Xclbin/Whisper.
    (re.compile(r"^flm-dev-kit/FLM_Xclbin/Whisper/"), {"launchlib"}),
    # launchlib and its trials include FLM_DLL/include and
    # FLM_DLL/cmake/flm_xrt_win.cmake; whisper_encoder/Makefile builds
    # FLM_DLL/detail/{q4nx,whisper_npu}.
    (re.compile(r"^flm-dev-kit/FLM_DLL/(include/|cmake/|detail/(q4nx|whisper_npu)/)"), {"launchlib"}),
    # The host builds of MiniCPM_V/decoding, Hy_MT2_1.8B/{decoding,
    # prefill_overlay} and minicpm_ocr/{decoding,vision_overlay} compile
    # against FLM_DLL/include and detail/{q4nx,hunyuan_npu,minicpm_ocr_npu}.
    (re.compile(r"^flm-dev-kit/FLM_DLL/(include/|detail/(q4nx|hunyuan_npu|minicpm_ocr_npu)/)"), {"xclbin"}),
)

# Product files the dev kit builds from (it reads them by relative path), so
# a change to one also selects these dev-kit areas. The prebuilt aiebu is
# linked by the dev-kit engines (libaiebu.a; aiebu_static.lib through
# FLM_DLL/cmake/flm_xrt_win.cmake), every FLM_Xclbin host build, SmolVLA and
# launchlib. The dev-kit engine Makefiles also put -L../../../../src/lib/xrt
# ahead of their own build/lib, so any library there can stand in for a unit
# they link. This lasts until libaiebu.a is vendored into
# flm-dev-kit/FLM_DLL/external_libs.
DEVKIT_FROM_PRODUCT = (
    (re.compile(r"^src/lib/xrt/(libaiebu\.a|aiebu_static\.lib)$"), {"xclbin", "dll", "pybind", "launchlib"}),
    (re.compile(r"^src/lib/xrt/[^/]+$"), {"dll"}),
    # The HRX engine builds fetch the pinned package with these scripts
    # (build_manylinux.sh, build.ps1 -Runtime HRX, hrx.mk, cmake/flm_hrx.cmake).
    (re.compile(r"^hrx-integration/"), {"dll"}),
    # setup_windows_deps.ps1 builds xrt_coreutil.lib from these files, and
    # devkit-engines.yml's Windows jobs key their C:\dev cache on them.
    (re.compile(r"^\.github/windows-deps/"), {"dll"}),
)

# The CI that decides and runs the dev-kit jobs: a change to it runs every
# dev-kit area, so it is exercised before it merges.
DEVKIT_CI = re.compile(
    r"^\.github/(workflows/(build-and-test|release|devkit-[^/]+)\.yml$"
    r"|actions/(detect-changes|devkit[^/]*)/|script/ci/)")


# The dev-kit engines (flm-dev-kit/FLM_DLL/detail/<engine>) and what they
# serve: engine -> families, models, uses (the engines whose code it
# compiles), plus the shared detail/ units. ENGINES below maps the product's
# src/ engine names; this maps the dev kit's directories, and test_changes.py
# checks the two agree for every engine both have.
DEVKIT_DLL = "flm-dev-kit/FLM_DLL/"
DEVKIT_ENGINES = "flm-dev-kit/FLM_DLL/engines.json"
# FLM_DLL files that build into no engine library: the per-model dev helpers
# (CI runs build.sh and build_manylinux.sh) and the model registries.
DEVKIT_DLL_NO_BUILD = re.compile(
    r"^flm-dev-kit/FLM_DLL/(build_(?!manylinux\.sh$)[A-Za-z0-9_]+\.sh|model_list[^/]*\.json"
    r"|(.*/)?\.gitattributes)$")


@dataclass
class DevkitEngines:
    """flm-dev-kit/FLM_DLL/engines.json, parsed."""
    engines: dict  # engine -> {"families": [...], "models": [...], "uses": [...]}
    shared: frozenset

    @classmethod
    def parse(cls, text: Optional[str]) -> Optional["DevkitEngines"]:
        try:
            data = json.loads(text) if text else None
        except ValueError:
            return None
        if not isinstance(data, dict) or not isinstance(data.get("engines"), dict):
            return None

        def str_list(v) -> bool:
            return isinstance(v, list) and all(isinstance(x, str) for x in v)

        # A malformed entry (devkit_check reports it) makes the whole file
        # unreadable, so callers fall back to every family rather than test
        # a wrong set (a string "families" would split into characters).
        fields = ("families", "models", "uses")
        if not all(isinstance(v, dict) and all(str_list(v.get(f, [])) for f in fields)
                   for v in data["engines"].values()):
            return None
        shared = data.get("shared", [])
        if not str_list(shared):
            return None
        engines = {k: {f: list(v.get(f, [])) for f in fields} for k, v in data["engines"].items()}
        return cls(engines, frozenset(shared))

    def families(self, engine: str) -> set:
        return set(self.engines.get(engine, {}).get("families") or [])

    def all_families(self) -> set:
        return {f for e in self.engines for f in self.families(e)}

    def with_users(self, engines: set) -> set:
        """These engines and every engine that uses one of them, transitively."""
        out, todo = set(), list(engines)
        while todo:
            e = todo.pop()
            if e in out:
                continue
            out.add(e)
            todo += [u for u, entry in self.engines.items() if e in (entry.get("uses") or [])]
        return out

    def header_owner(self, name: str) -> Optional[str]:
        """include/models/<name> pairs with detail/<name> or detail/<name>_npu."""
        return next((c for c in (name, f"{name}_npu") if c in self.engines), None)

    def engines_of(self, path: str) -> Optional[set]:
        """The engines a flm-dev-kit/FLM_DLL path builds into, before adding
        their users: a set, or None for every engine. Call only for paths
        under FLM_DLL that are not DEVKIT_DLL_NO_BUILD."""
        parts = path[len(DEVKIT_DLL):].split("/")
        if parts[0] == "detail" and len(parts) >= 3:
            if parts[1] in self.engines:
                return {parts[1]}
            return None  # a shared unit, or a directory engines.json does not know
        if parts[:2] == ["include", "models"] and len(parts) >= 4:
            owner = self.header_owner(parts[2])
            return {owner} if owner else None
        return None


def devkit_area(path: str) -> Optional[str]:
    """The dev-kit area of a path under flm-dev-kit/, None outside all areas."""
    return next((area for area, prefix in DEVKIT_AREAS if path.startswith(prefix)), None)


def devkit_extra_areas(path: str) -> set:
    """The areas besides its own that a dev-kit path selects (DEVKIT_ALSO)."""
    out: set = set()
    for rx, areas in DEVKIT_ALSO:
        if rx.match(path):
            out |= areas
    return out


# Engine -> the catalog families it serves, from the registry in
# src/common/AutoModel/builtin_backends.cpp (RegisterFlm<engine>(family)) and
# the frontends in src/include/AutoModel/modeling_*.hpp. An engine's name
# appears in several spellings across the tree; every spelling below is the
# engine's name with "_npu" and a "lib" prefix stripped:
#   src/lib/{xrt,hrx}/[lib]<name>_npu.{so,dll,lib}  prebuilt engine libraries
#   src/include/models/<name>/flm/**              their headers
#   src/include/AutoModel/modeling_<name>.hpp     frontends (and the .cpp
#   src/common/AutoModel/modeling_<name>[_image|_audio].cpp  beside them)
#   src/test/<name>[_npu]/**                      engine unit tests
ENGINES = {
    # names                                  families
    ("llama", "llama3"):                     ("llama3", "deepseek-r1"),
    ("qwen3",):                              ("qwen3", "qwen3-it", "qwen3-tk", "deepseek-r1-0528"),
    ("qwen2",):                              ("qwen2",),
    ("qwen2vl",):                            ("qwen2vl",),
    # Qwen3VL_Flash derives from Qwen3VL and its engine includes qwen3vl_npu.
    ("qwen3vl",):                            ("qwen3vl", "qwen3vl-flash"),
    ("qwen3vl_flash",):                      ("qwen3vl-flash",),
    ("qwen3_5vl",):                          ("qwen3.5",),
    ("minicpm_v_4_7", "minicpm_v"):          ("minicpm-v-4.7",),
    ("minicpm_v_4_7_moe", "minicpm_v_moe"): ("minicpm-v-4.7-moe",),
    ("minicpm_ocr",):                        ("minicpm-ocr",),
    ("qwen3_6_moe",):                        ("qwen3.6-moe",),
    # qwen3.8-dflash2 is the same engine with a draft package (Qwen3_8DFlash2 derives from Qwen3_8MTP).
    ("qwen3_8mtp", "qwen3_8_mtp"):           ("qwen3.8-mtp", "qwen3.8-dflash2"),
    ("gemma", "gemma3"):                     ("gemma3",),
    ("gemma_text", "gemma3_text"):           ("gemma3-text",),
    # Gemma4e_Flash, Gemma4e_DFlash and Gemma4e_DFlash_Flash derive from Gemma4e and their engines include gemma4e_npu.
    ("gemma4e",):                            ("gemma4e", "gemma4e-flash", "gemma4e-dflash", "gemma4e-dflash-flash"),
    ("gemma4e_flash",):                      ("gemma4e-flash",),
    ("gemma4e_dflash",):                     ("gemma4e-dflash",),
    ("gemma4e_dflash_flash",):               ("gemma4e-dflash-flash",),
    ("gemma4_12b",):                         ("gemma4-12b",),
    ("hunyuan",):                            ("hunyuan",),
    ("hy_mem", "hy_memory"):                 ("hy-memory",),
    ("gpt_oss",):                            ("gpt-oss",),
    ("lfm2",):                               ("lfm2", "lfm2.5-tk"),
    ("lfm3",):                               ("lfm3",),
    ("nanbeige",):                           ("nanbeige",),
    ("phi4",):                               ("phi4",),
    ("gemma_embedding",):                    ("embed-gemma",),
    ("d3",):                                 ("d3",),
    ("whisper",):                            ("whisper-v3",),
    # r2t2 (Confucius4-R2T2) runs on the qwen3_asr engine; qwen3_asr_stream is
    # its streaming frontend.
    ("qwen3_asr", "qwen3_asr_stream"):       ("r2t2",),
    # Built but registered for no catalog family.
    ("qwen3_5_omni",):                       (),
}
ENGINE_FAMILIES = {name: fams for names, fams in ENGINES.items() for name in names}

# The corelib backends (src/common/models/<name>/core, src/include/models/
# <name>/core, src/test/<name>_core) serve only the family they register for
# in builtin_backends.cpp, not every family of the flm engine of that name.
CORE_FAMILIES = {
    "gemma4": ("gemma4e",),
    "phi4": ("phi4",),
    "qwen3": ("qwen3",),
    "qwen35": ("qwen3.5",),
    "qwen36": ("qwen3.6-moe",),
}

# Subsystems outside the per-engine layout, each serving one family.
SUBSYSTEMS = (
    ("whisper engine", r"src/(common|include)/whisper/", ("whisper-v3",)),
    ("embedding engine", r"src/(common|include)/AutoEmbeddingModel/|src/include/embedding_model\.hpp$",
     ("embed-gemma",)),
    ("decision engine", r"src/(common|include)/(decision|AutoDecisionModel)/"
     r"|src/include/decision_model\.hpp$|src/test/d3_\w+/", ("d3",)),
    # src/detail builds the gemma4e engine from source; gemma4e_flash
    # includes it.
    ("gemma4e engine source", r"src/detail/", ("gemma4e", "gemma4e-flash")),
    # An oga-hipep header that the qwen3.5 frontend includes in every package.
    ("header the qwen3.5 frontend includes", r"src/include/oga/oga_vision\.hpp$", ("qwen3.5",)),
)

# The oga-hipep backend, in the Windows XRT package only, runs on the GPU the
# catalog entries whose effective backend is oga-hipep: their "backend" key,
# or without one, a name ending in GPU_SUFFIX (model_list::backend_for_family).
# No NPU model runs its code (bar the header above) or those entries, so they
# name no family and run no NPU job: Test Windows HIPEP tests them, on every
# GPU entry.
GPU_ONLY = re.compile(r"^(src/(common|include)/oga/|src/test/oga_hipep/)")
GPU_SUFFIX = "-oga-hipep"
# An entry's "backend" key (flm::backend::kBackendKey), when present, overrides
# the suffix rule; GPU_BACKEND is the oga-hipep backend's id.
GPU_BACKEND_KEY = "backend"
GPU_BACKEND = "oga-hipep"
# A family's frontend also drives its GPU entries, and its NPU engine does not:
# a change to the frontend of a family with one (gpu_families) reaches them.
GPU_FRONTEND = re.compile(r"^src/(include|common)/AutoModel/")
# The suites Test Windows HIPEP runs.
GPU_SUITES = {"chat", "api", "shutdown"}
# Those that test no NPU model, so a change to their tests runs no NPU job.
GPU_ONLY_SUITES = {"shutdown"}

# Tests of shared code kept in an engine's test directory, which name no family:
# test_model_downloader tests src/pull, beside the phi4 corelib tests it came
# with. A change to phi4's own entries shows in the catalog.
SHARED_TESTS = re.compile(r"^src/test/phi4_core/test_model_downloader\.cpp$")

# Files that only serve speech recognition: any change runs the whole ASR
# set. Ported from the paths: filter of the old asr-test.yml and asr_gate.sh,
# with the test scripts moved to npu_test.
ASR_ONLY = re.compile(
    r"^(src/(common|include)/whisper/"
    r"|src/xclbins/Whisper-V3-Turbo-NPU2/"
    r"|src/lib/(xrt|hrx)/[^/]*whisper_npu"
    r"|src/(common|include)/models/qwen3_asr/"
    r"|src/(common|include)/AutoModel/modeling_qwen3_asr[^/]*\.(cpp|hpp)$"
    r"|src/xclbins/Confucius4-R2T2-NPU2/"
    r"|src/lib/(xrt|hrx)/[^/]*qwen3_asr_npu"
    r"|src/test/qwen3_asr[^/]*/"
    r"|src/(common|include)/audio_process_utils/"
    r"|src/(common|include)/audio/"
    r"|src/common/AutoModel/[^/]*_audio\.cpp$"
    r"|src/detail/gemma4e_npu/gemma4e_audio"
    r"|src/include/(pocketfft|dr_libs)/"
    r"|src/cmake/FlmAudioCodecs\.cmake$"
    r"|src/server/multipart\.(cpp|hpp)$"
    r"|src/test/whisper_npu/"
    r"|\.github/script/npu_test/suites/(asr|whisper)[^/]*\.py$"
    r"|\.github/script/npu_test/tools/build_asr_eval\.py$"
    r"|\.github/script/npu_test/assets/asr_eval/)")
# These host /v1/audio/transcriptions but also every other endpoint, so they
# count only when a changed line is about audio or speech.
ASR_SHARED = re.compile(r"^src/server/(rest_handler|server)\.cpp$")
ASR_WORDS = re.compile(r"audio|transcri|whisper|multipart|speech|\basr\b|\bwav\b|mp3|flac"
                       r"|\bogg\b|opus|m4a|\bpcm\b|resampl", re.IGNORECASE)


# The npu_test suites, in run order; "all" when a change is not scoped.
SUITES = ("chat", "serving", "prefix_cache", "api", "embedding", "decision", "asr", "shutdown")

# Test-side files that matter to one suite only. Anything else in npu_test
# (the client, server, harness, plan) runs every suite.
TEST_SCOPES = (
    (r"\.github/script/npu_test/suites/(\w+)\.py$", None),  # the suite itself
    (r"\.github/script/npu_test/console\.py$", {"shutdown"}),
    (r"\.github/script/npu_test/assets/d3_goldens[^/]*\.jsonl$", {"decision"}),
    (r"\.github/script/npu_test/assets/vision/|\.github/script/npu_test/tools/make_vision_assets\.py$",
     {"chat"}),
    (r"src/test/(openai_api_test|openai_usage)/", {"api"}),
    # The model-extension contracts run in cpu-tests.yml on every change; the
    # Gemma4-open bundle and its NPU tests run with chat.
    (r"src/test/(model_frontend|model_backend|model_list_platform|model_extension_\w+)/"
     r"|src/schemas/|src/cmake/FastFlowLMModelExtension[^/]*\.cmake(\.in)?$", {"chat"}),
)


def family_suites(families: Iterable[str], catalog: dict[str, dict]) -> set:
    """The suites that exercise models of these families."""
    out = set()
    for fam in families:
        if fam == "d3":
            out.add("decision")
        elif fam == "embed-gemma":
            out.add("embedding")
        elif fam == "whisper-v3":
            out.add("asr")
        else:
            out |= {"chat", "prefix_cache"}
            # Chat models with an audio encoder are tested by the asr suite.
            if any(family_of(e) == fam and e.get("asr") for e in catalog.values()):
                out.add("asr")
    return out


def is_non_code(path: str) -> bool:
    return bool(NON_CODE.match(path))


class Git:
    """The diff range; None from any method means git could not answer."""

    def __init__(self, repo: Path, base: str, head: str):
        self.repo, self.base, self.head = repo, base, head
        # Diff from the merge base, as base...head does, so a pull request
        # is judged by its own commits rather than by what main gained since.
        self.merge_base = self._git("merge-base", base, head)
        if self.merge_base:
            self.merge_base = self.merge_base.strip()

    def _git(self, *args: str) -> Optional[str]:
        try:
            res = subprocess.run(["git", "-c", "core.quotePath=false", "-C", str(self.repo), *args],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
        except OSError:
            return None
        return res.stdout if res.returncode == 0 else None

    def files(self) -> Optional[list[str]]:
        # --no-renames lists both sides of a rename, so a file moved out of
        # an engine directory still counts against that engine.
        out = self._git("diff", "--name-only", "--no-renames", f"{self.base}...{self.head}")
        return None if out is None else [l for l in out.splitlines() if l.strip()]

    def changed_lines(self, paths: list[str]) -> Optional[str]:
        out = self._git("diff", "-U0", f"{self.base}...{self.head}", "--", *paths)
        if out is None:
            return None
        return "\n".join(l for l in out.splitlines()
                         if l[:1] in ("+", "-") and not l.startswith(("+++", "---")))

    def show(self, rev_side: str, path: str) -> Optional[str]:
        rev = self.merge_base if rev_side == "base" else self.head
        return self._git("show", f"{rev}:{path}") if rev else None


@dataclass
class Decision:
    run_tests: bool = True
    families: set = field(default_factory=set)
    asr_full: bool = False
    # None: every suite. Otherwise the suites every changed file is scoped to.
    suites: Optional[set] = field(default_factory=set)
    # A changed file reaches an oga-hipep model (every suite running does too).
    hipep: bool = False
    # NPU catalog entries no runner has yet (see new_downloads).
    new_models: list = field(default_factory=list)
    # The oga-hipep entries added or edited in the head catalog, which include
    # those with new weights (new_downloads' GPU half).
    gpu_models: set = field(default_factory=set)
    # Dev-kit areas (DEVKIT_AREAS) a changed flm-dev-kit/ file needs.
    devkit_areas: set = field(default_factory=set)
    # Catalog families of the dev-kit engines the change touches.
    devkit_families: set = field(default_factory=set)
    reasons: list = field(default_factory=list)

    def all_suites(self, path: str, why: str = "") -> None:
        if self.suites is not None:
            why = f" ({why})" if why else ""
            self.reasons.append(f"{path}{why} is not scoped to particular suites: running them all")
        self.suites = None

    def add_suites(self, names: set) -> None:
        if self.suites is not None:
            self.suites |= names

    def add_new_downloads(self, path: str, npu: list, gpu: list) -> None:
        self.new_models = sorted(set(self.new_models) | set(npu))
        self.gpu_models |= set(gpu)
        if npu or gpu:
            self.reasons.append(f"{path}: new weights, tested whatever their size: "
                                + ", ".join(npu + gpu))

    def reach_hipep(self, reason: str) -> None:
        self.hipep = True
        self.reasons.append(reason)

    def runs_hipep(self) -> bool:
        return self.run_tests and (self.hipep or self.suites_value() == "all")

    def devkit_value(self) -> str:
        return ",".join(area for area, _ in DEVKIT_AREAS if area in self.devkit_areas)

    def summary(self) -> str:
        devkit = f"; dev kit: {self.devkit_value().replace(',', ', ')}" if self.devkit_areas else ""
        if self.devkit_families:
            devkit += f" (engine families: {len(self.devkit_families)})"
        if not self.run_tests:
            if self.devkit_areas:
                return f"flm-dev-kit only: product builds and NPU tests skipped{devkit}"
            return "documentation only: NPU tests skipped"
        return self._product_summary() + devkit

    def _product_summary(self) -> str:
        hipep = "HIPEP: " + ("runs" if self.runs_hipep() else "skipped")
        if self.suites_value() == "none":
            return f"oga-hipep only: NPU tests skipped; {hipep}"
        parts = ["NPU tests run"]
        parts.append("families in full: " + (", ".join(sorted(self.families)) or "none"))
        parts.append("ASR: whole set" if self.asr_full else "ASR: pr subset")
        parts.append("suites: " + self.suites_value().replace(",", ", "))
        if self.new_models:
            parts.append("new models: " + ", ".join(self.new_models))
        parts.append(hipep)
        if self.gpu_models:
            parts.append("GPU models added or edited: " + ", ".join(sorted(self.gpu_models)))
        return "; ".join(parts)

    def suites_value(self) -> str:
        if self.suites is None:
            return "all"
        if not self.suites:
            return "none" if self.hipep else "all"
        return ",".join(s for s in SUITES if s in self.suites)

    def outputs(self) -> dict[str, str]:
        return {"run_tests": str(self.run_tests).lower(),
                "families": ",".join(sorted(self.families)),
                "asr_full": str(self.asr_full).lower(),
                "suites": self.suites_value(),
                "hipep": str(self.runs_hipep()).lower(),
                "new_models": ",".join(self.new_models),
                "hipep_models": ",".join(sorted(self.gpu_models)),
                "devkit": str(bool(self.devkit_areas)).lower(),
                "devkit_areas": self.devkit_value(),
                "devkit_families": ",".join(sorted(self.devkit_families)),
                "summary": self.summary()}


def load_catalog(text: Optional[str]) -> dict[str, dict]:
    """model id ("name:tag") -> entry, from model_list.json text."""
    if not text:
        return {}
    try:
        models = json.loads(text).get("models", {})
    except (ValueError, AttributeError):
        return {}
    return {f"{name}:{tag}": entry for name, tags in models.items()
            if isinstance(tags, dict) for tag, entry in tags.items() if isinstance(entry, dict)}


def family_of(entry: dict) -> str:
    details = entry.get("details")
    return details.get("family", "") if isinstance(details, dict) else ""


def runs_on_gpu(model_id: str, entry: Optional[dict] = None) -> bool:
    """Whether the oga-hipep backend, not an NPU one, runs this catalog entry.
    As in model_list::apply_support_filter, an entry's "backend" key overrides
    the backend its name implies."""
    backend = (entry or {}).get(GPU_BACKEND_KEY)
    if backend is not None:
        return backend == GPU_BACKEND
    return model_id.split(":", 1)[0].endswith(GPU_SUFFIX)


def gpu_families(catalog: dict[str, dict]) -> Optional[set]:
    """The families with a GPU entry, whose frontends drive the oga-hipep
    backend too; None without a catalog, when any frontend might be one."""
    if not catalog:
        return None
    return {family_of(e) for mid, e in catalog.items() if runs_on_gpu(mid, e)} - {""}


def catalog_diff(base: dict[str, dict], head: dict[str, dict]) -> tuple[set, list]:
    """Families with an NPU entry added, removed or edited between two catalogs,
    and the GPU entries that were. An entry is a GPU one only if it is on
    every side that has it."""
    fams, gpu = set(), []
    for mid in sorted(set(base) | set(head)):
        if base.get(mid) == head.get(mid):
            continue
        sides = [e for e in (base.get(mid), head.get(mid)) if e is not None]
        split_entry(mid, sides, fams, gpu)
    return fams - {""}, gpu


# The NPU the self-hosted test runners have. A new entry for another NPU only
# (a dev04 corelib entry) is one `flm list` leaves out there, so no runner can
# test it.
RUNNER_PLATFORMS = {"aie2p"}
# The fields that decide what an entry downloads: a change to any is weights
# no runner has yet. A raised flm_min_version marks a downloaded copy outdated
# (model_downloader.cpp), so flm pull fetches the files again; that and new
# model_info.json hashes are how republished weights ship.
DOWNLOAD_FIELDS = ("name", "url", "file_url", "files", "flm_min_version")


def split_downloads(ids: Iterable[str], head: dict[str, dict]) -> tuple[list, list]:
    """(NPU, GPU) ids of head catalog entries, NPU ones only if a runner's NPU
    can run them."""
    npu, gpu = [], []
    for mid in sorted(set(ids)):
        entry = head.get(mid)
        if entry is None:
            continue
        if runs_on_gpu(mid, entry):
            gpu.append(mid)
            continue
        platforms = entry.get("supported_platforms")
        if not platforms or RUNNER_PLATFORMS & set(platforms):
            npu.append(mid)
    return npu, gpu


def new_downloads(base: dict[str, dict], head: dict[str, dict]) -> tuple[list, list]:
    """(NPU, GPU) model ids whose weights no runner has yet: entries the head
    catalog adds, or whose download fields it changes."""
    return split_downloads(
        (mid for mid, entry in head.items()
         if base.get(mid) is None
         or any(base[mid].get(k) != entry.get(k) for k in DOWNLOAD_FIELDS)), head)


def split_entry(mid: str, sides: list, fams: set, gpu: list) -> None:
    """Files one changed entry: under GPU entries if it is on the GPU on any
    side (Test Windows HIPEP runs it), and under its NPU family unless it is
    on the GPU on every side."""
    on_gpu = [runs_on_gpu(mid, e) for e in sides]
    if any(on_gpu):
        gpu.append(mid)
    if not all(on_gpu):
        fams.update(family_of(e) for e in sides if e)


def catalog_fields_diff(base_text: str, head_text: str) -> list[str]:
    """The top-level fields of two model_list.json texts, other than models,
    that differ. Every model reads them: model_path is the folder of all."""
    base, head = json.loads(base_text), json.loads(head_text)
    return sorted(k for k in base.keys() | head.keys() if k != "models" and base.get(k) != head.get(k))


def model_info_diff(base_text: Optional[str], head_text: Optional[str],
                    catalogs: Iterable[dict[str, dict]]) -> Optional[tuple[set, list, list]]:
    """Families with an NPU entry whose model_info.json entry changed, the GPU
    entries whose one did, and every catalog id whose one did (new file
    hashes: weights no runner has). Keys are model ids or a catalog entry's
    model_info_key."""
    try:
        base = json.loads(base_text) if base_text else {}
        head = json.loads(head_text) if head_text else {}
    except ValueError:
        return None
    if not isinstance(base, dict) or not isinstance(head, dict):
        return None
    keys = {k for k in set(base) | set(head) if base.get(k) != head.get(k)}
    hits: dict[str, list] = {}
    for catalog in catalogs:
        for mid, entry in catalog.items():
            if mid in keys or entry.get("model_info_key") in keys:
                hits.setdefault(mid, []).append(entry)
    fams, gpu = set(), []
    for mid, entries in sorted(hits.items()):
        split_entry(mid, entries, fams, gpu)
    return fams - {""}, gpu, sorted(hits)


def add_catalog_change(d: Decision, path: str, fams: set, gpu: list,
                       catalog: dict[str, dict], neither: str) -> None:
    """A catalog file's changed entries: their NPU families in full, and Test
    Windows HIPEP for GPU entries, which also tests those the head catalog
    still runs on the GPU. A change to neither is not scoped."""
    d.families |= fams
    if fams:
        d.add_suites(family_suites(fams, catalog))
    if not fams and not gpu:
        d.all_suites(path)
    what = ", ".join(sorted(fams)) or ("no NPU family" if gpu else neither)
    if gpu:
        d.hipep = True
        d.gpu_models |= {mid for mid in gpu if mid in catalog and runs_on_gpu(mid, catalog[mid])}
        what += (f"; {len(gpu)} oga-hipep entr{'y' if len(gpu) == 1 else 'ies'}, e.g. {gpu[0]},"
                 " which only Test Windows HIPEP runs")
    d.reasons.append(f"{path}: entries changed in {what}")


def engine_families(path: str, xclbin_families: dict[str, set]) -> Optional[tuple[str, tuple]]:
    """(why, families) for a path inside one engine's code, else None."""
    parts = path.split("/")
    if len(parts) < 3 or parts[0] != "src":
        return None
    top, sub = parts[1], parts[2]
    if top == "lib":
        # Prebuilt engines: src/lib/{xrt,hrx}/[lib]<engine>[_npu].{so,dll,lib}.
        # Anything else there is shared by every engine (the q4nx, gemm, mha,
        # dequant and lm_head units, libaiebu.a, aiebu_static.lib, runtime
        # libraries) or by the whole package (the Windows DLLs in src/lib):
        # no families, so every suite runs.
        if sub in ("xrt", "hrx") and len(parts) == 4:
            stem = parts[3].rsplit(".", 1)[0]
            if parts[3].endswith(".so") and stem.startswith("lib"):
                stem = stem[3:]
            name = stem[:-4] if stem.endswith("_npu") else stem
            if name in ENGINE_FAMILIES:
                return f"engine library {name}", ENGINE_FAMILIES[name]
        return f"library {parts[-1]}, not one engine's", ()
    if top == "include" and sub == "models" and len(parts) >= 5:
        name, kind = parts[3], parts[4]
        if kind == "core" and name in CORE_FAMILIES:
            return f"{name} core backend", CORE_FAMILIES[name]
        if name in ENGINE_FAMILIES:
            return f"engine headers {name}", ENGINE_FAMILIES[name]
        return None
    if top == "common" and sub == "models" and len(parts) >= 5 and parts[3] in CORE_FAMILIES:
        return f"{parts[3]} core backend", CORE_FAMILIES[parts[3]]
    if top in ("include", "common") and sub == "AutoModel" and len(parts) == 4:
        m = re.match(r"modeling_(\w+?)(_image|_audio)?\.(hpp|cpp)$", parts[3])
        if m and m.group(1) in ENGINE_FAMILIES:
            return f"frontend modeling_{m.group(1)}", ENGINE_FAMILIES[m.group(1)]
        return None
    if top == "test" and len(parts) >= 4:
        name = sub[:-4] if sub.endswith("_npu") else sub
        if name.endswith("_core") and name[:-5] in CORE_FAMILIES:
            return f"{name} tests", CORE_FAMILIES[name[:-5]]
        if name in ENGINE_FAMILIES:
            return f"{sub} tests", ENGINE_FAMILIES[name]
        return None
    if top == "xclbins":
        # src/xclbins/<folder>/**: the families of the catalog entries that
        # load their kernels from that folder (see xclbin_folders).
        fams = tuple(sorted(xclbin_families.get(sub, ()))) if len(parts) >= 4 else ()
        if fams:
            return f"xclbins {sub}", fams
        return (f"xclbins {sub}: no catalog entry loads this folder" if len(parts) >= 4
                else f"{sub}: at the top of src/xclbins, no model's folder"), ()
    return None


def xclbin_folders(head_catalog: dict[str, dict]) -> dict[str, set]:
    """xclbin folder under src/xclbins -> the families of the catalog entries
    that load it. As AutoModel::_shared_initialize_model_state
    (src/common/AutoModel/automodel.cpp): an entry's xclbin_folder when it is
    a non-empty string, on any entry, else its own name (model_list::
    resolve_source only defaults it for dependants); qwen3.8-dflash2 runs the
    kernels of Qwen3.8-27B-NPU2."""
    out: dict[str, set] = {}
    for entry in head_catalog.values():
        own = entry.get("xclbin_folder")
        folder = own if isinstance(own, str) and own else entry.get("name")
        if isinstance(folder, str) and folder:
            out.setdefault(folder, set()).add(family_of(entry))
    return {k: v - {""} for k, v in out.items()}


def load_devkit_engines(text: Optional[str] = None) -> Optional[DevkitEngines]:
    """engines.json from text, else from the checkout; None if unreadable."""
    if text is None:
        try:
            text = (REPO_ROOT / DEVKIT_ENGINES).read_text(encoding="utf-8")
        except OSError:
            return None
    return DevkitEngines.parse(text)


def every_engine_family(engines: Optional[DevkitEngines]) -> set:
    """Every dev-kit engine's families; ENGINES' if engines.json is unreadable."""
    if engines is not None:
        return engines.all_families()
    return {f for fams in ENGINES.values() for f in fams}


def add_devkit_families(d: Decision, files: list[str], code: list[str],
                        engines: Optional[DevkitEngines], git: Optional[Git]) -> None:
    """d.devkit_families for the changed dev-kit files and the product files
    that select the dll area (see devkit_families in the docstring)."""
    everything = None  # the first path that needs every engine's families
    hit: set = set()
    for path in files:
        if not path.startswith(DEVKIT_DLL) or is_non_code(path) or DEVKIT_DLL_NO_BUILD.match(path):
            continue
        if path == DEVKIT_ENGINES:
            changed = devkit_engines_diff(git, engines)
            if changed is None:
                everything = everything or f"{path} (cannot compare it with the base)"
            else:
                hit |= changed
            continue
        if engines is None:
            everything = everything or f"{path} ({DEVKIT_ENGINES} unreadable)"
            continue
        owners = engines.engines_of(path)
        if owners is None:
            everything = everything or f"{path} (builds into every engine)"
        else:
            hit |= owners
    for path in files:
        if path.startswith(DEVKIT_ROOT) and not is_non_code(path) and devkit_area(path) is None:
            everything = everything or f"{path} (in no dev-kit area)"
    for path in code:
        if DEVKIT_CI.match(path):
            everything = everything or f"{path} (CI that runs the dev-kit jobs)"
        elif any(rx.match(path) and "dll" in areas for rx, areas in DEVKIT_FROM_PRODUCT):
            everything = everything or f"{path} (linked by the dev-kit engines)"
    if everything:
        d.devkit_families = every_engine_family(engines)
        d.reasons.append(f"{everything}: every dev-kit engine's families")
        return
    if hit and engines is not None:
        users = engines.with_users(hit)
        d.devkit_families = {f for e in users for f in engines.families(e)}
        more = sorted(users - hit)
        d.reasons.append(f"dev-kit engine(s) {', '.join(sorted(hit))}"
                         + (f", used by {', '.join(more)}" if more else "")
                         + f" -> families {', '.join(sorted(d.devkit_families)) or 'none'}")


def devkit_engines_diff(git: Optional[Git], head: Optional[DevkitEngines]) -> Optional[set]:
    """Engines whose engines.json entry differs between base and head (an
    engine on one side only included); None when the two cannot be read."""
    if git is None or head is None:
        return None
    base = DevkitEngines.parse(git.show("base", DEVKIT_ENGINES))
    if base is None:
        return None
    if base.shared != head.shared:
        return None
    return {e for e in set(base.engines) | set(head.engines)
            if base.engines.get(e) != head.engines.get(e)}


def decide(files: Optional[list[str]], git: Optional[Git] = None,
           head_catalog_text: Optional[str] = None,
           devkit_engines: Optional[DevkitEngines] = None) -> Decision:
    """The decision for a list of changed files (None: git could not tell).

    git supplies the changed lines and the base and head versions of the
    catalogs; without it those checks fall back as described inline.
    devkit_engines is the head engines.json; None reads the checkout's.
    """
    d = Decision(suites=None)
    if devkit_engines is None:
        devkit_engines = load_devkit_engines()
    if files is None:
        d.asr_full = True
        d.devkit_areas = set(ALL_DEVKIT_AREAS)
        d.devkit_families = every_engine_family(devkit_engines)
        d.reasons.append("changed files unknown (new branch, shallow clone or force-push): "
                         "running everything, ASR set in full, every dev-kit area and engine family")
        return d
    if not files:
        d.reasons.append("no changed files: running the tests anyway")
        return d

    # flm-dev-kit/ code selects dev-kit areas only; the product rules below
    # see the other files. Its documentation counts as nothing.
    devkit = [f for f in files if f.startswith(DEVKIT_ROOT) and not is_non_code(f)]
    for path in devkit:
        area = devkit_area(path)
        if area is None:
            if not d.devkit_areas >= ALL_DEVKIT_AREAS:
                d.reasons.append(f"{path} is in no dev-kit area: every dev-kit area")
            d.devkit_areas |= ALL_DEVKIT_AREAS
            continue
        d.devkit_areas.add(area)
        extra = devkit_extra_areas(path) - d.devkit_areas
        if extra:
            d.devkit_areas |= extra
            d.reasons.append(f"{path}: {area}, and built by {', '.join(sorted(extra))}")
    if devkit:
        d.reasons.append(f"{len(devkit)} changed flm-dev-kit file(s), e.g. {devkit[0]}: dev-kit areas "
                         f"{d.devkit_value()}; no product build or suite for them")

    code = [f for f in files if not is_non_code(f) and not f.startswith(DEVKIT_ROOT)]
    add_devkit_families(d, files, code, devkit_engines, git)
    d.suites = set()  # narrowed per file below; None once any file is unscoped
    if not code:
        d.run_tests = False
        if devkit:
            d.reasons.append("no product code changed: product builds and NPU tests skipped")
        else:
            d.reasons.append(f"all {len(files)} changed file(s) are documentation or non-code")
        return d
    d.reasons.append(f"{len(code)} of {len(files)} changed file(s) are product code, e.g. {code[0]}")

    # Product files the dev kit builds from, and the CI that runs it, also
    # select dev-kit areas; their product rules below are unchanged.
    for path in code:
        if DEVKIT_CI.match(path):
            new = ALL_DEVKIT_AREAS - d.devkit_areas
            d.devkit_areas |= ALL_DEVKIT_AREAS
            if new:
                d.reasons.append(f"{path}: CI that runs the dev-kit jobs -> every dev-kit area")
            continue
        for rx, areas in DEVKIT_FROM_PRODUCT:
            if rx.match(path):
                new = areas - d.devkit_areas
                d.devkit_areas |= areas
                if new:
                    d.reasons.append(f"{path}: linked by the dev kit -> {', '.join(sorted(new))}")
                break

    head_catalog = load_catalog(head_catalog_text)
    gpu_fams = gpu_families(head_catalog)
    xclbin_families = xclbin_folders(head_catalog)

    gpu_code = []
    for path in code:
        hit = None if SHARED_TESTS.match(path) else engine_families(path, xclbin_families)
        if hit is None:
            hit = next(((why, fams) for why, pattern, fams in SUBSYSTEMS
                        if re.match(pattern, path)), None)
        if GPU_ONLY.match(path):
            gpu_code.append(path)
            if hit is None:
                continue
        if hit and hit[1]:
            new = set(hit[1]) - d.families
            if new:
                d.families |= new
                d.reasons.append(f"{path}: {hit[0]} -> {', '.join(sorted(new))}")
            if GPU_FRONTEND.match(path) and (gpu_fams is None or gpu_fams & set(hit[1])):
                d.reach_hipep(f"{path}: {hit[0]} also runs oga-hipep models,"
                              " which Test Windows HIPEP tests")

        # Which suites this file can affect.
        scope = None
        for pattern, names in TEST_SCOPES:
            m = re.match(pattern, path)
            if m:
                scope = names if names is not None else (
                    {m.group(1)} if m.group(1) in SUITES else None)
                break
        if scope and scope & GPU_SUITES:
            d.reach_hipep(f"{path}: tests the {', '.join(sorted(scope & GPU_SUITES))} suite,"
                          " which Test Windows HIPEP runs")
        if scope:
            scope = scope - GPU_ONLY_SUITES
        if scope is None and ASR_ONLY.match(path):
            scope = {"asr"}
        if scope is None and hit and hit[1]:
            scope = family_suites(hit[1], head_catalog)
        if scope is None and path not in (MODEL_LIST, MODEL_INFO):
            d.all_suites(path, hit[0] if hit else "")
        elif scope is not None:
            d.add_suites(scope)
    if gpu_code:
        more = f" and {len(gpu_code) - 1} more" if len(gpu_code) > 1 else ""
        d.reach_hipep(f"{gpu_code[0]}{more}: oga-hipep backend, which Test Windows HIPEP tests")

    if MODEL_LIST in code:
        base_text = git.show("base", MODEL_LIST) if git else None
        base = load_catalog(base_text)
        if git is None or not base or not head_catalog:
            d.reasons.append(f"{MODEL_LIST} changed but cannot be compared; no family added for it")
            d.all_suites(MODEL_LIST)
        else:
            fields = catalog_fields_diff(base_text, head_catalog_text)
            if fields:
                d.reasons.append(f"{MODEL_LIST}: {', '.join(fields)} changed, outside its model entries")
                d.all_suites(MODEL_LIST)
            add_catalog_change(d, MODEL_LIST, *catalog_diff(base, head_catalog), head_catalog,
                               "no family (top-level fields only)")
            d.add_new_downloads(MODEL_LIST, *new_downloads(base, head_catalog))
    if MODEL_INFO in code:
        diff = None
        if git:
            diff = model_info_diff(git.show("base", MODEL_INFO), git.show("head", MODEL_INFO),
                                   [load_catalog(git.show("base", MODEL_LIST)), head_catalog])
        if diff is None:
            d.reasons.append(f"{MODEL_INFO} changed but cannot be compared; no family added for it")
            d.all_suites(MODEL_INFO)
        else:
            fams, gpu, ids = diff
            add_catalog_change(d, MODEL_INFO, fams, gpu, head_catalog, "no family")
            d.add_new_downloads(MODEL_INFO, *split_downloads(ids, head_catalog))

    # A family the head catalog lacks (removed, or never listed) has no model
    # the plan could run.
    if head_catalog:
        known = {family_of(e) for e in head_catalog.values()}
        gone = d.families - known
        if gone:
            d.families -= gone
            d.reasons.append(f"not in the head catalog, so not tested: {', '.join(sorted(gone))}")

    asr = [f for f in code if ASR_ONLY.match(f)]
    shared = [f for f in code if ASR_SHARED.match(f)]
    if asr:
        d.asr_full = True
        d.reasons.append(f"ASR-only file(s) changed, e.g. {asr[0]}: whole ASR set")
    elif shared:
        lines = git.changed_lines(shared) if git else None
        if lines is None:
            d.asr_full = True
            d.reasons.append(f"{', '.join(shared)} changed and the lines cannot be read: whole ASR set")
        else:
            words = [l for l in lines.splitlines() if ASR_WORDS.search(l)]
            if words:
                d.asr_full = True
                d.reasons.append(f"changed lines in {', '.join(shared)} mention audio or speech: "
                                 f"{words[0][:100]}")
            else:
                d.reasons.append(f"{', '.join(shared)} changed, but no changed line is about audio")
    return d


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="", help="Base commit; empty or all-zero means unknown.")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--files", nargs="*",
                        help="Changed files, instead of diffing base...head. With --base the "
                             "catalog diffs and changed lines still come from git.")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--github-output", action="store_true",
                        help="Also append the outputs to $GITHUB_OUTPUT.")
    args = parser.parse_args(argv)

    known_base = bool(args.base.strip("0 "))
    git = Git(args.repo, args.base, args.head) if known_base else None
    if args.files is not None:
        files: Optional[list[str]] = [f.replace("\\", "/") for f in args.files]
    else:
        files = git.files() if git else None
    head_catalog = git.show("head", MODEL_LIST) if git else None
    if head_catalog is None:
        try:
            head_catalog = (args.repo / MODEL_LIST).read_text(encoding="utf-8")
        except OSError:
            head_catalog = None

    engines_text = git.show("head", DEVKIT_ENGINES) if git else None
    if engines_text is None:
        try:
            engines_text = (args.repo / DEVKIT_ENGINES).read_text(encoding="utf-8")
        except OSError:
            engines_text = None
    devkit_engines = DevkitEngines.parse(engines_text)
    decision = decide(files, git, head_catalog, devkit_engines)
    for reason in decision.reasons:
        print(f"changes: {reason}")
    outputs = decision.outputs()
    for key, value in outputs.items():
        print(f"{key}={value}")
    if args.github_output:
        target = os.environ.get("GITHUB_OUTPUT")
        if not target:
            print("--github-output given but $GITHUB_OUTPUT is not set", file=sys.stderr)
            return 2
        with open(target, "a", encoding="utf-8") as fh:
            fh.writelines(f"{k}={v}\n" for k, v in outputs.items())
    return 0


if __name__ == "__main__":
    sys.exit(main())
