#!/usr/bin/env python3
"""Decides, from the files a pull request or push changes, what CI must test.

    changes.py --base <sha> --head <sha> [--github-output]
    changes.py --files src/server/server.cpp docs/index.md

Outputs (printed, and appended to $GITHUB_OUTPUT with --github-output):
  run_tests  false only when every changed file is documentation or otherwise
             never built or packaged, so no NPU runner is reserved for it
  families   comma-separated model families (details.family in
             src/model_list.json) whose engine or catalog entry changed; the
             pr tier tests every model in them instead of one per engine
  asr_full   true when speech-recognition code changed, so the pr tier runs
             the whole ASR evaluation set
  suites     "all", or the comma-separated npu_test suites the change can
             affect when every changed file is scoped to some (an ASR-only
             change runs asr, a d3-only change runs decision)
  summary    one line for the job log

When in doubt it tests more: an all-zero or missing base (a new branch), or a
range git cannot diff (shallow clone, force-push), counts as "everything
changed". It still names no families then: the pr tier already runs one model
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
    ("qwen3_6_moe",):                        ("qwen3.6-moe",),
    ("qwen3_8mtp", "qwen3_8_mtp"):           ("qwen3.8-mtp",),
    ("gemma", "gemma3"):                     ("gemma3",),
    ("gemma_text", "gemma3_text"):           ("gemma3-text",),
    # Gemma4e_Flash and Gemma4e_DFlash derive from Gemma4e and their engines include gemma4e_npu.
    ("gemma4e",):                            ("gemma4e", "gemma4e-flash", "gemma4e-dflash"),
    ("gemma4e_flash",):                      ("gemma4e-flash",),
    ("gemma4e_dflash",):                     ("gemma4e-dflash",),
    ("gemma4_12b",):                         ("gemma4-12b",),
    ("hunyuan",):                            ("hunyuan",),
    ("gpt_oss",):                            ("gpt-oss",),
    ("lfm2",):                               ("lfm2", "lfm2.5-tk"),
    ("lfm3",):                               ("lfm3",),
    ("nanbeige",):                           ("nanbeige",),
    ("phi4",):                               ("phi4",),
    ("gemma_embedding",):                    ("embed-gemma",),
    ("d3",):                                 ("d3",),
    ("whisper",):                            ("whisper-v3",),
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
)

# Files that only serve speech recognition: any change runs the whole ASR
# set. Ported from the paths: filter of the old asr-test.yml and asr_gate.sh,
# with the test scripts moved to npu_test.
ASR_ONLY = re.compile(
    r"^(src/(common|include)/whisper/"
    r"|src/xclbins/Whisper-V3-Turbo-NPU2/"
    r"|src/lib/(xrt|hrx)/[^/]*whisper_npu"
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
SUITES = ("chat", "serving", "prefix_cache", "api", "embedding", "decision", "asr")

# Test-side files that matter to one suite only. Anything else in npu_test
# (the client, server, harness, plan) runs every suite.
TEST_SCOPES = (
    (r"\.github/script/npu_test/suites/(\w+)\.py$", None),  # the suite itself
    (r"\.github/script/npu_test/assets/d3_goldens[^/]*\.jsonl$", {"decision"}),
    (r"\.github/script/npu_test/assets/vision/|\.github/script/npu_test/tools/make_vision_assets\.py$",
     {"chat"}),
    (r"src/test/(openai_api_test|openai_usage)/", {"api"}),
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
            res = subprocess.run(["git", "-C", str(self.repo), *args],
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
    reasons: list = field(default_factory=list)

    def all_suites(self, path: str) -> None:
        if self.suites is not None:
            self.reasons.append(f"{path} is not scoped to particular suites: running them all")
        self.suites = None

    def add_suites(self, names: set) -> None:
        if self.suites is not None:
            self.suites |= names

    def summary(self) -> str:
        if not self.run_tests:
            return "documentation only: NPU tests skipped"
        parts = ["NPU tests run"]
        parts.append("families in full: " + (", ".join(sorted(self.families)) or "none"))
        parts.append("ASR: whole set" if self.asr_full else "ASR: pr subset")
        parts.append("suites: " + self.suites_value().replace(",", ", "))
        return "; ".join(parts)

    def suites_value(self) -> str:
        if self.suites is None or not self.suites:
            return "all"
        return ",".join(s for s in SUITES if s in self.suites)

    def outputs(self) -> dict[str, str]:
        return {"run_tests": str(self.run_tests).lower(),
                "families": ",".join(sorted(self.families)),
                "asr_full": str(self.asr_full).lower(),
                "suites": self.suites_value(),
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


def catalog_diff(base: dict[str, dict], head: dict[str, dict]) -> set:
    """Families with an entry added, removed or edited between two catalogs."""
    fams = set()
    for mid in set(base) | set(head):
        if base.get(mid) != head.get(mid):
            fams.update(family_of(e) for e in (base.get(mid), head.get(mid)) if e)
    return fams - {""}


def model_info_diff(base_text: Optional[str], head_text: Optional[str],
                    catalogs: Iterable[dict[str, dict]]) -> Optional[set]:
    """Families whose model_info.json entry changed. Keys are model ids or a
    catalog entry's model_info_key."""
    try:
        base = json.loads(base_text) if base_text else {}
        head = json.loads(head_text) if head_text else {}
    except ValueError:
        return None
    if not isinstance(base, dict) or not isinstance(head, dict):
        return None
    keys = {k for k in set(base) | set(head) if base.get(k) != head.get(k)}
    fams = set()
    for catalog in catalogs:
        for mid, entry in catalog.items():
            if mid in keys or entry.get("model_info_key") in keys:
                fams.add(family_of(entry))
    return fams - {""}


def engine_families(path: str, xclbin_families: dict[str, set]) -> Optional[tuple[str, tuple]]:
    """(why, families) for a path inside one engine's code, else None."""
    parts = path.split("/")
    if len(parts) < 3 or parts[0] != "src":
        return None
    top, sub = parts[1], parts[2]
    if top == "lib" and sub in ("xrt", "hrx") and len(parts) == 4:
        stem = parts[3].rsplit(".", 1)[0]
        if parts[3].endswith(".so") and stem.startswith("lib"):
            stem = stem[3:]
        name = stem[:-4] if stem.endswith("_npu") else stem
        if name in ENGINE_FAMILIES:
            return f"engine library {name}", ENGINE_FAMILIES[name]
        return None
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
    if top == "xclbins" and len(parts) >= 4:
        return f"xclbins {sub}", tuple(sorted(xclbin_families.get(sub, ())))
    return None


def decide(files: Optional[list[str]], git: Optional[Git] = None,
           head_catalog_text: Optional[str] = None) -> Decision:
    """The decision for a list of changed files (None: git could not tell).

    git supplies the changed lines and the base and head versions of the
    catalogs; without it those checks fall back as described inline.
    """
    d = Decision(suites=None)
    if files is None:
        d.asr_full = True
        d.reasons.append("changed files unknown (new branch, shallow clone or force-push): "
                         "running everything, ASR set in full")
        return d
    if not files:
        d.reasons.append("no changed files: running the tests anyway")
        return d

    code = [f for f in files if not is_non_code(f)]
    d.suites = set()  # narrowed per file below; None once any file is unscoped
    if not code:
        d.run_tests = False
        d.reasons.append(f"all {len(files)} changed file(s) are documentation or non-code")
        return d
    d.reasons.append(f"{len(code)} of {len(files)} changed file(s) are code, e.g. {code[0]}")

    head_catalog = load_catalog(head_catalog_text)
    xclbin_families: dict[str, set] = {}
    for entry in head_catalog.values():
        if entry.get("name"):
            xclbin_families.setdefault(entry["name"], set()).add(family_of(entry))

    for path in code:
        hit = engine_families(path, xclbin_families)
        if hit is None:
            hit = next(((why, fams) for why, pattern, fams in SUBSYSTEMS
                        if re.match(pattern, path)), None)
        if hit and hit[1]:
            new = set(hit[1]) - d.families
            if new:
                d.families |= new
                d.reasons.append(f"{path}: {hit[0]} -> {', '.join(sorted(new))}")

        # Which suites this file can affect.
        scope = None
        for pattern, names in TEST_SCOPES:
            m = re.match(pattern, path)
            if m:
                scope = names if names is not None else (
                    {m.group(1)} if m.group(1) in SUITES else None)
                break
        if scope is None and ASR_ONLY.match(path):
            scope = {"asr"}
        if scope is None and hit and hit[1]:
            scope = family_suites(hit[1], head_catalog)
        if scope is None and path not in (MODEL_LIST, MODEL_INFO):
            d.all_suites(path)
        elif scope is not None:
            d.add_suites(scope)

    if MODEL_LIST in code:
        base = load_catalog(git.show("base", MODEL_LIST)) if git else {}
        if git is None or not base or not head_catalog:
            d.reasons.append(f"{MODEL_LIST} changed but cannot be compared; no family added for it")
            d.all_suites(MODEL_LIST)
        else:
            fams = catalog_diff(base, head_catalog)
            d.families |= fams
            if fams:
                d.add_suites(family_suites(fams, head_catalog))
            else:
                d.all_suites(MODEL_LIST)
            d.reasons.append(f"{MODEL_LIST}: entries changed in "
                             + (", ".join(sorted(fams)) or "no family (top-level fields only)"))
    if MODEL_INFO in code:
        fams = None
        if git:
            fams = model_info_diff(git.show("base", MODEL_INFO), git.show("head", MODEL_INFO),
                                   [load_catalog(git.show("base", MODEL_LIST)), head_catalog])
        if fams is None:
            d.reasons.append(f"{MODEL_INFO} changed but cannot be compared; no family added for it")
            d.all_suites(MODEL_INFO)
        else:
            d.families |= fams
            if fams:
                d.add_suites(family_suites(fams, head_catalog))
            else:
                d.all_suites(MODEL_INFO)
            d.reasons.append(f"{MODEL_INFO}: entries changed in " + (", ".join(sorted(fams)) or "no family"))

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

    decision = decide(files, git, head_catalog)
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
