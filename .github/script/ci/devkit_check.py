#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Fast static checks of the FLM dev kit's CI data (flm-dev-kit/).

    devkit_check.py [--repo <root>] [--base <sha> [--head <sha>]] [--pending-ok [--github-output]]

Checks, each failure printed as "devkit_check: error: ..." (exit 1):

  build units   flm-dev-kit/FLM_Xclbin/build_units.json lists exactly the
                directories under FLM_Xclbin that own a Makefile (build/,
                makefiles/ and hidden directories skipped), as CI finds build
                units; each unit's artifacts are the names its model_types
                produce (xclbins-<unit, / as _>[_<MODEL_TYPE>]), and no two
                units produce the same artifact
  assignment    every artifact in FLM_Xclbin/kernel_assignment.json is one a
                build unit produces (a unit, and a MODEL_TYPE of it when the
                unit fans out); its schema is models -> {folder, kernels
                {<slot>.xclbin: artifact}, optional vlm}; folder is a plain
                relative path and slot a plain file name (CI writes
                <folder>/<slot>), and models sharing a folder put the same
                artifact in a slot they share
  assignment md FLM_Xclbin/kernel_assignment.md, hand-written, agrees with the
                JSON as far as it can be parsed: every "#### `<model>` →
                `<folder>`" heading is a JSON model with that folder and the
                table under it lists the same slot -> artifact pairs ("—" for
                a slot with no CI build, which the JSON must then not list);
                every JSON model has such a heading. Tables under other
                headings (models without a tag, such as Qwen3.5-Omni) are not
                compared, but their artifacts must resolve to a build unit.
                The Compiler column and the vlm flag are not checked
  engines       flm-dev-kit/FLM_DLL/engines.json: every engine and shared
                directory exists under FLM_DLL/detail; every detail/<dir>
                with a Makefile is an engine or shared; families are
                non-empty strings; every model is a kernel_assignment.json
                model and every assignment model is served by an engine or
                listed under served_elsewhere, and no model is served twice
                (by two engines, or by one and served_elsewhere); every
                engine's detail/<engine> has a Makefile (build.sh builds only
                those); each engine's uses are exactly the other engines whose
                detail/ sources ("../<engine>", also as "-I../<engine>") or
                include/models/<x>/ headers its Makefile or CMakeLists.txt
                (outside # comments) or the #include lines of its sources and
                own headers name; a model with no assignment of its own
                may be listed when it is a catalog alias (see catalog).
                The JSON files must not repeat a key
  catalog       what the product ships (src/) is what the dev kit describes.
                The NPU models of src/model_list.json are the entries flm
                keeps on an aie2p machine (supported_platforms naming aie2p
                or absent, backend flm); a model's xclbin folder is its
                xclbin_folder, else its name. (1) every NPU model has a
                kernel_assignment.json entry, unless it is an alias: no entry
                but an xclbin_folder that some entry's folder is; (2) every
                assignment model is an NPU model, unless engines.json lists
                it under served_elsewhere; (3) an entry's folder is its
                model's xclbin folder; (4) every assigned slot ships as
                src/xclbins/<folder>/<slot> (served_elsewhere skipped); (5)
                every src/xclbins/<folder>/*.xclbin is a slot some model with
                that folder assigns; (6) every NPU model is listed by an
                engine and every engine model is an NPU model; (7) every
                file in src/lib/{xrt,hrx} is lib<name>.so, <name>.dll or
                <name>.lib of a library an engine or shared directory builds
                (its Makefile TARGET and CMakeLists.txt add_library, which
                must agree), and each such library ships all three in both.
                CATALOG_ALLOW excuses named files and alias tags, each with
                its reason; an entry that excuses nothing is an error.
                Pending swap: with --pending-ok (Build and Test's
                devkit-static, pull requests only) an assigned slot (4) or
                an engine library file (7) src/ lacks is no error but
                "pending": printed, and written as pending=<csv> with
                --github-output. The pull request's run must then stage
                every one (devkit_stage.py --pending fails otherwise, and
                summary fails a run with pending binaries that staged
                nothing), so a new model or engine can get through CI
                before its swap commits the binaries. Without --pending-ok
                (pushes, the nightly, a release) they are errors, so main
                never lacks a binary
  rtp lock      FLM_Xclbin/rtp_addresses.lock.txt (devkit_rtp_lock.py) parses
                and names only artifacts a build unit produces and some
                kernel_assignment.json model uses. With --base, when the pull
                request changes the lock: every artifact whose stanza it adds,
                changes or removes must come with a change under
                FLM_DLL/detail/<engine>/ of every engine serving a model that
                uses the artifact (engines.json), since each engine keeps its
                own copy of those addresses; an engine also counts as changed
                when an engine it uses (its "uses") changed. The engines are
                reported. A lock new at the base needs no engine change
  Bsymbolic     every FLM_DLL/detail/*/Makefile has -Wl,-Bsymbolic-functions
                on a line that is not a comment
  bench comment .github/workflows/devkit-bench-comment.yml, which runs with a
                write token on artifacts a fork can write: it triggers on
                workflow_run of exactly build-and-test.yml's name (Build
                and Test calls devkit-bench.yml, which uploads the
                summary, in its own run); its permissions are exactly
                actions: read and pull-requests: write, at the workflow
                level and nowhere else; it checks out nothing; its job
                condition pins the triggering run to build-and-test.yml's
                path in this repository, pull_request events; no
                ${{ github.event.* }} expression appears in a run: script.
                devkit-bench.yml keeps contents: read and actions: read
                and grants no write; build-and-test.yml's jobs calling it
                (devkit-bench, and devkit-bench-nightly, whose condition
                starts with github.event_name == 'schedule' while
                devkit-bench's starts with github.event_name !=
                'schedule', so a run calls it once) grant exactly contents:
                read and actions: read and pass no secrets. Skipped when the
                files are absent
  swap wiring   the names devkit_swap.py reads another Build and Test run
                by match the CI that makes them: build-and-test.yml has a
                devkit-stage job (STAGE_JOB) that runs devkit_stage.py and
                uploads manifest-<os>-<runtime>/manifest.json as
                devkit-stage-manifest-<os>-<runtime> (MANIFEST_ARTIFACT,
                MANIFEST_FILE) for each os and runtime; one job named as
                TEST_JOBS gives for each, downloading that runtime's product
                package; a summary job with a step named PENDING_STEP that
                runs devkit_swap.py pending; the devkit-engines and
                devkit-xclbins jobs; and a changes job that runs
                devkit_swap.py verify and check with actions: read. Skipped
                when the workflow is absent
  notices       with --base: every file added under flm-dev-kit/ between the
                merge base of --base and --head (default HEAD), of a source or
                script type, has an AMD copyright notice in its first 15
                lines (Markdown and HTML included); vendored trees (nlohmann,
                aiebu, external_libs) are skipped, and a file with a
                third-party notice ("Copyright (c)", "Copyright ©",
                "Copyright <year>" or an SPDX line) or a "Generated by ... Do
                not edit" first line passes, as AGENTS.md's notice rules give
                those no AMD notice. NOTICE_HELD lists the files held
                without any notice pending review, each with its reason

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from devkit_rtp_lock import LOCK_REL as RTP_LOCK, assigned, changed_artifacts, parse_lock, users  # noqa: E402
DEVKIT = "flm-dev-kit"
XCLBIN = f"{DEVKIT}/FLM_Xclbin"
DLL = f"{DEVKIT}/FLM_DLL"
BUILD_UNITS = f"{XCLBIN}/build_units.json"
ASSIGNMENT = f"{XCLBIN}/kernel_assignment.json"
ASSIGNMENT_MD = f"{XCLBIN}/kernel_assignment.md"
ENGINES = f"{DLL}/engines.json"

# Directories that are never build units: build output and the per-unit
# makefile fragments (common.mk and friends).
SKIP_DIRS = {"build", "makefiles"}
SOURCE_EXT = (".cpp", ".hpp", ".h", ".cc", ".c", ".inl")
BSYMBOLIC = "-Wl,-Bsymbolic-functions"

# Files that must carry the notice when added: source, script and build files.
NOTICE_EXT = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl", ".inc",
              ".py", ".sh", ".bash", ".ps1", ".psm1", ".mk", ".cmake", ".yml", ".yaml",
              ".toml", ".md", ".html")
NOTICE_NAMES = ("Makefile", "CMakeLists.txt")
NOTICE = re.compile(r"Copyright.{0,40}Advanced Micro Devices", re.IGNORECASE)
NOTICE_LINES = 15
# Third-party code keeps its own notice and gets no AMD one on top (an
# adapted file adds an AMD "Modifications Copyright" line, which NOTICE
# matches); vendored trees are third-party throughout. Generated files get
# no notice.
THIRD_PARTY_NOTICE = re.compile(r"Copyright\s*(?:\(c\)|©|\d{4})|SPDX-(?:License-Identifier|FileCopyrightText):",
                                re.IGNORECASE)
GENERATED = re.compile(r"\bGenerated by\b.*\bDo not edit\b", re.IGNORECASE)
# Files held without any notice pending review (path -> reason). A held file is
# reported as a note, not an error; drop it here once its notice lands.
NOTICE_HELD = {
    f"{DEVKIT}/FLM_Xclbin/Gemma4/test_top.py":
        "the Gemma 4 model-card example (Hugging Face transformers usage); kept without a "
        "notice pending review",
}
VENDORED = (f"{DEVKIT}/FLM_DLL/include/nlohmann/", f"{DEVKIT}/FLM_DLL/include/aiebu/",
            f"{DEVKIT}/FLM_DLL/external_libs/", f"{DEVKIT}/FLM_Xclbin/host_common/aiebu/")

INCLUDE = re.compile(r"^\s*#\s*include\b")

MD_MODEL = re.compile(r"^#### `([^`]+)` → `([^`]+)`\s*$")
MD_HEADING = re.compile(r"^#{1,6}\s")
MD_ROW = re.compile(r"^\|\s*`([^`]+\.xclbin)`\s*\|\s*(?:`([^`]+)`|(—))\s*\|")


def artifact_names(unit: str, model_types: list) -> list[str]:
    """The CI artifact names a unit's builds produce."""
    safe = unit.replace("/", "_")
    return [f"xclbins-{safe}_{t}" for t in model_types] if model_types else [f"xclbins-{safe}"]


def find_units(xclbin_root: Path) -> set[str]:
    """Directories under xclbin_root (relative, / separated) owning a Makefile."""
    units = set()
    for dirpath, dirnames, filenames in os.walk(xclbin_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        if "Makefile" in filenames and Path(dirpath) != xclbin_root:
            units.add(Path(dirpath).relative_to(xclbin_root).as_posix())
    return units


def _no_dupes(pairs: list) -> dict:
    keys = [k for k, _ in pairs]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise ValueError(f"duplicate key(s) {', '.join(dup)}")
    return dict(pairs)


def load_json(repo: Path, rel: str, errors: list) -> Optional[dict]:
    try:
        data = json.loads((repo / rel).read_text(encoding="utf-8"), object_pairs_hook=_no_dupes)
    except OSError as exc:
        errors.append(f"{rel}: cannot read: {exc}")
        return None
    except ValueError as exc:
        errors.append(f"{rel}: not valid JSON: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{rel}: top level is not an object")
        return None
    return data


def check_build_units(repo: Path, data: dict, errors: list) -> dict[str, tuple[str, str]]:
    """Checks build_units.json; returns artifact -> (unit, model_type)."""
    units = data.get("units")
    if not isinstance(units, dict):
        errors.append(f"{BUILD_UNITS}: no \"units\" object")
        return {}
    found = find_units(repo / XCLBIN)
    for unit in sorted(found - set(units)):
        errors.append(f"{BUILD_UNITS}: {XCLBIN}/{unit} has a Makefile but is not listed")
    for unit in sorted(set(units) - found):
        errors.append(f"{BUILD_UNITS}: {unit} is listed but {XCLBIN}/{unit}/Makefile does not exist")
    index: dict[str, tuple[str, str]] = {}
    for unit, entry in sorted(units.items()):
        where = f"{BUILD_UNITS}: {unit}"
        if not isinstance(entry, dict):
            errors.append(f"{where}: entry is not an object")
            continue
        mts = entry.get("model_types")
        if not isinstance(mts, list) or not all(isinstance(t, str) and t for t in mts):
            errors.append(f"{where}: model_types is not a list of non-empty strings")
            continue
        if len(set(mts)) != len(mts):
            errors.append(f"{where}: model_types lists a MODEL_TYPE twice")
        want = artifact_names(unit, mts)
        if entry.get("artifacts") != want:
            errors.append(f"{where}: artifacts are {entry.get('artifacts')}, but its model_types produce {want}")
        for art, mt in zip(want, mts or [""]):
            if art in index:
                errors.append(f"{where}: artifact {art} is also produced by {index[art][0]}")
            index[art] = (unit, mt)
    return index


# A model's folder and its slots: devkit_assemble_xclbins.py writes
# <folder>/<slot> under its output, and refuses anything else.
SAFE_FOLDER = re.compile(r"[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)*")
SAFE_SLOT = re.compile(r"[A-Za-z0-9_.+-]+\.xclbin")


def check_assignment(data: dict, index: dict, errors: list) -> dict[str, dict]:
    """Checks kernel_assignment.json; returns its models."""
    models = data.get("models")
    if not isinstance(models, dict):
        errors.append(f"{ASSIGNMENT}: no \"models\" object")
        return {}
    placed: dict[tuple[str, str], tuple[str, object]] = {}  # (folder, slot) -> (model, artifact)
    for model, entry in models.items():
        where = f"{ASSIGNMENT}: {model}"
        if not isinstance(entry, dict):
            errors.append(f"{where}: entry is not an object")
            continue
        extra = set(entry) - {"folder", "kernels", "vlm"}
        if extra:
            errors.append(f"{where}: unknown key(s) {', '.join(sorted(extra))}")
        folder = entry.get("folder")
        if not isinstance(folder, str) or not folder:
            errors.append(f"{where}: folder is not a non-empty string")
            folder = None
        elif not SAFE_FOLDER.fullmatch(folder) or any(p in (".", "..") for p in folder.split("/")):
            errors.append(f"{where}: folder {folder!r} is not a plain relative path "
                          "([A-Za-z0-9_.+-] parts separated by /, no . or ..)")
        if "vlm" in entry and not isinstance(entry["vlm"], bool):
            errors.append(f"{where}: vlm is not true or false")
        kernels = entry.get("kernels")
        if not isinstance(kernels, dict):
            errors.append(f"{where}: kernels is not an object")
            continue
        for slot, art in kernels.items():
            if not slot.endswith(".xclbin"):
                errors.append(f"{where}: kernel slot {slot!r} does not end in .xclbin")
            elif not SAFE_SLOT.fullmatch(slot):
                errors.append(f"{where}: kernel slot {slot!r} is not a plain file name ([A-Za-z0-9_.+-])")
            if folder is not None:
                other = placed.setdefault((folder, slot), (model, art))
                if other[1] != art:
                    errors.append(f"{where}: {folder}/{slot} is {art!r}, but {other[0]} puts {other[1]!r} "
                                  "there (models sharing a folder must agree on a shared slot)")
            if not isinstance(art, str) or art not in index:
                errors.append(f"{where}: {slot} -> {art!r} is no build unit's artifact "
                              f"(see {BUILD_UNITS}; a fanned-out unit needs the MODEL_TYPE there)")
    return models


def parse_assignment_md(text: str) -> tuple[dict, list]:
    """(model -> {"folder", "kernels": {slot: artifact or None}, "line"},
    [(line, slot, artifact) of rows under other headings])."""
    models: dict[str, dict] = {}
    other: list = []
    current: Optional[dict] = None
    in_section = False
    for n, line in enumerate(text.splitlines(), 1):
        if MD_HEADING.match(line):
            m = MD_MODEL.match(line)
            current = None
            in_section = line.startswith("####")
            if m:
                if m.group(1) in models:
                    models[m.group(1)]["dupes"].append(n)
                    current = models[m.group(1)]
                else:
                    current = models[m.group(1)] = {"folder": m.group(2), "kernels": {}, "line": n,
                                                    "dupes": [], "slot_dupes": []}
            continue
        row = MD_ROW.match(line)
        if not row:
            continue
        slot, art = row.group(1), row.group(2)
        if current is not None:
            if slot in current["kernels"]:
                current["slot_dupes"].append((n, slot))
            current["kernels"][slot] = art
        elif in_section:
            other.append((n, slot, art))
    return models, other


def check_assignment_md(text: str, models: dict, index: dict, errors: list) -> None:
    md, other = parse_assignment_md(text)
    where = ASSIGNMENT_MD
    for model in sorted(set(models) - set(md)):
        errors.append(f"{where}: no \"#### `{model}` → `<folder>`\" section for {ASSIGNMENT} model {model}")
    for model, entry in sorted(md.items()):
        at = f"{where}:{entry['line']}: {model}"
        for n in entry["dupes"]:
            errors.append(f"{where}:{n}: second section for {model}")
        for n, slot in entry["slot_dupes"]:
            errors.append(f"{where}:{n}: {model} lists {slot} twice")
        if model not in models:
            errors.append(f"{at}: not a model in {ASSIGNMENT}")
            continue
        if entry["folder"] != models[model].get("folder"):
            errors.append(f"{at}: folder {entry['folder']}, but {ASSIGNMENT} has {models[model].get('folder')}")
        json_kernels = models[model].get("kernels") or {}
        md_kernels = {s: a for s, a in entry["kernels"].items() if a is not None}
        for slot in sorted(set(json_kernels) | set(entry["kernels"])):
            in_md, in_json = md_kernels.get(slot), json_kernels.get(slot)
            if in_md == in_json:
                continue
            if slot not in entry["kernels"]:
                errors.append(f"{at}: {slot} -> {in_json} is in {ASSIGNMENT} but not in the table")
            elif in_md is None:
                errors.append(f"{at}: the table has no CI artifact (—) for {slot}, but {ASSIGNMENT} has {in_json}")
            elif in_json is None:
                errors.append(f"{at}: the table has {slot} -> {in_md}, which {ASSIGNMENT} does not list")
            else:
                errors.append(f"{at}: the table has {slot} -> {in_md}, {ASSIGNMENT} has {in_json}")
    for n, slot, art in other:
        if art is not None and art not in index:
            errors.append(f"{where}:{n}: {slot} -> {art} is no build unit's artifact")


def header_owner(name: str, engines: Iterable[str]) -> Optional[str]:
    """include/models/<name> pairs with detail/<name> or detail/<name>_npu."""
    engines = set(engines)
    return next((c for c in (name, f"{name}_npu") if c in engines), None)


def _read_tree(base: Path, header_only: bool = False) -> str:
    """The #include lines of the sources under base, and (unless
    header_only) the non-comment text of its Makefiles and CMakeLists.txt."""
    chunks = []
    if not base.is_dir():
        return ""
    for dirpath, _, files in os.walk(base):
        for f in sorted(files):
            is_source = f.endswith(SOURCE_EXT)
            if not is_source and (f not in ("Makefile", "CMakeLists.txt") or header_only):
                continue
            try:
                lines = (Path(dirpath) / f).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            # Only what the build reads: #include lines of sources, and the
            # Makefile and CMakeLists.txt outside # comments (a comment naming
            # a fork's origin is no dependency).
            if is_source:
                chunks += [l for l in lines if INCLUDE.match(l)]
            else:
                chunks += [l.split("#", 1)[0] for l in lines]
    return "\n".join(chunks)


def scan_engine_uses(dll_root: Path | str, engines: set, shared: set) -> dict[str, set]:
    """engine -> the other engines whose detail/ directories ("../<engine>",
    with or without a trailing / or a -I in front) or include/models/<x>/
    headers its Makefile or CMakeLists.txt (outside # comments) or the
    #include lines of its sources and own headers name.
    Shared units are not engines and are never in the result."""
    dll_root = Path(dll_root)
    models_dir = dll_root / "include" / "models"
    header_dirs = sorted(p.name for p in models_dir.iterdir() if p.is_dir()) if models_dir.is_dir() else []
    out = {}
    for engine in engines:
        text = _read_tree(dll_root / "detail" / engine)
        text += "\n".join(_read_tree(models_dir / x, header_only=True)
                          for x in header_dirs if header_owner(x, engines) == engine)
        uses = {d for d in re.findall(r"\.\./([A-Za-z0-9_]+)(?![A-Za-z0-9_.])", text) if d in engines}
        uses |= {o for x in re.findall(r"models/([A-Za-z0-9_]+)/", text)
                 for o in [header_owner(x, engines)] if o}
        out[engine] = uses - {engine} - set(shared)
    return out


def check_engines(repo: Path, data: dict, models: dict, errors: list,
                  aliases: frozenset | set = frozenset()) -> None:
    """aliases: catalog tags with no kernel_assignment.json entry that run
    another model's kernels (catalog_aliases); an engine may list them."""
    detail = repo / DLL / "detail"
    shared = data.get("shared")
    engines = data.get("engines")
    elsewhere = data.get("served_elsewhere", {})
    if not isinstance(shared, list) or not all(isinstance(s, str) for s in shared):
        errors.append(f"{ENGINES}: \"shared\" is not a list of directory names")
        shared = []
    if not isinstance(engines, dict):
        errors.append(f"{ENGINES}: no \"engines\" object")
        return
    if not isinstance(elsewhere, dict):
        errors.append(f"{ENGINES}: \"served_elsewhere\" is not an object")
        elsewhere = {}
    for name in sorted(set(shared) & set(engines)):
        errors.append(f"{ENGINES}: {name} is both shared and an engine")
    for name in sorted(set(shared) | set(engines)):
        if not (detail / name).is_dir():
            errors.append(f"{ENGINES}: {name} is listed but {DLL}/detail/{name} does not exist")
    with_makefile = {p.name for p in detail.iterdir() if (p / "Makefile").is_file()} if detail.is_dir() else set()
    for name in sorted(with_makefile - set(shared) - set(engines)):
        errors.append(f"{ENGINES}: {DLL}/detail/{name} has a Makefile but is neither an engine nor shared")
    # Shared units are exempt: vision_common is a source pool with no Makefile.
    for name in sorted(engines):
        if (detail / name).is_dir() and not (detail / name / "Makefile").is_file():
            errors.append(f"{ENGINES}: engine {name} has no {DLL}/detail/{name}/Makefile, so build.sh skips it")

    owner: dict[str, str] = {}
    for name, entry in sorted(engines.items()):
        where = f"{ENGINES}: {name}"
        if not isinstance(entry, dict):
            errors.append(f"{where}: entry is not an object")
            continue
        extra = set(entry) - {"families", "models", "uses"}
        if extra:
            errors.append(f"{where}: unknown key(s) {', '.join(sorted(extra))}")
        for key in ("families", "models", "uses"):
            val = entry.get(key)
            if not isinstance(val, list) or not all(isinstance(v, str) and v for v in val):
                errors.append(f"{where}: {key} is not a list of non-empty strings")
        for m in entry.get("models") or []:
            if isinstance(m, str):
                if m in owner and owner[m] != name:
                    errors.append(f"{where}: model {m} is also served by {owner[m]}")
                owner.setdefault(m, name)
                if models and m not in models and m not in aliases:
                    errors.append(f"{where}: model {m} is not in {ASSIGNMENT}")
        for u in entry.get("uses") or []:
            if u not in engines:
                errors.append(f"{where}: uses {u}, which is not an engine")
    served = set(owner)
    for m in sorted(served & set(elsewhere)):
        errors.append(f"{ENGINES}: {m} is served by an engine and listed under served_elsewhere")
    for m in sorted(set(elsewhere) - set(models)):
        errors.append(f"{ENGINES}: served_elsewhere names {m}, which is not in {ASSIGNMENT}")
    for m in sorted(set(models) - served - set(elsewhere)):
        errors.append(f"{ENGINES}: {ASSIGNMENT} model {m} is served by no engine "
                      "(add it to an engine's models, or to served_elsewhere)")

    existing = {e for e in engines if (detail / e).is_dir()}
    scanned = scan_engine_uses(repo / DLL, existing, set(shared))
    for name in sorted(existing):
        declared = set(engines[name].get("uses") or []) if isinstance(engines[name], dict) else set()
        if declared != scanned[name]:
            errors.append(f"{ENGINES}: {name} uses {sorted(declared)}, but its sources and headers "
                          f"reference {sorted(scanned[name])}")


# ── catalog: what the product ships (src/) against the dev kit ──────────────
# The product builds nothing from flm-dev-kit/: it ships prebuilt engine
# libraries (src/lib/<runtime>/) and kernels (src/xclbins/<folder>/<slot>).
# These rules keep the dev kit the source of both, so a model cannot ship
# under src/ without its engine, its engines.json entry and its
# kernel_assignment.json entry.
MODEL_LIST = "src/model_list.json"
SHIPPED_XCLBINS = "src/xclbins"
# src/CMakeLists.txt links the engines from src/lib/${FLM_RUNTIME_NAME}, xrt or
# hrx: lib<name>.so on Linux, the <name>.lib import library on Windows, whose
# <name>.dll .github/workflows/win-build.yml copies from the same directory
# into the package. Each runtime ships all three, built by the dev-kit dir.
SHIPPED_RUNTIMES = ("xrt", "hrx")
NPU_PLATFORM = "aie2p"           # the generation the FLM_DLL engines and kernels are built for
FLM_BACKEND = "flm"              # flm::backend::kFlmBackendId
BACKEND_SUFFIXES = {"-core": "core", "-oga-hipep": "oga-hipep"}  # catalog::backend_for_model_type
# Shipped files and catalog tags the catalog rules accept without a dev-kit
# source (key: a src/ path, or a model tag) -> why. An entry that no longer
# excuses anything is an error, so the list cannot go stale.
CATALOG_ALLOW = {
    f"{SHIPPED_XCLBINS}/MiniCPM-OCR-1.0-NPU2/conv.xclbin":
        "loaded by no engine: minicpm_ocr_npu registers conv8.xclbin (and only when built with "
        "PREFILL_CONV=npu); conv.xclbin is the older conv1d kernel it does not work with "
        "(FLM_DLL/detail/minicpm_ocr_npu/README.md). Kept: binaries under src/ are not removed here",
    "src/lib/xrt/libaiebu.a":
        "the prebuilt AIE assembler the dev-kit engines and kernel hosts link, no engine library",
    "src/lib/xrt/aiebu_static.lib":
        "the prebuilt AIE assembler (Windows) the dev-kit engines link, no engine library",
}

MK_TARGET = re.compile(r"^\s*TARGET\s*[:?]?=\s*\$\((?:LIB_DIR|BUILD_DIR)\)/(?:lib)?([A-Za-z0-9_]+)\.(?:so|dll)\s*$")
CMAKE_LIBRARY = re.compile(r"\badd_library\s*\(\s*([A-Za-z0-9_]+)\s+SHARED\b")


def entry_backend(family: str, entry: dict) -> str:
    """The backend that runs a catalog entry: its "backend" key, else the
    family's suffix (model_list::backend_for_family)."""
    backend = entry.get("backend")
    if isinstance(backend, str) and backend:
        return backend
    return next((b for sfx, b in BACKEND_SUFFIXES.items()
                 if family.endswith(sfx) and len(family) > len(sfx)), FLM_BACKEND)


def npu_catalog(data: dict) -> dict[str, dict]:
    """tag -> entry for the catalog models the FLM_DLL engines and kernels run:
    those flm on an NPU2 (aie2p) machine keeps (model_list::
    apply_support_filter): backend flm (any backend for an embedding model)
    and supported_platforms naming aie2p, or absent (every generation)."""
    out = {}
    for family, sizes in (data.get("models") or {}).items():
        if not isinstance(sizes, dict):
            continue
        for size, entry in sizes.items():
            if not isinstance(entry, dict):
                continue
            platforms = entry.get("supported_platforms")
            if platforms is not None and (not isinstance(platforms, list) or NPU_PLATFORM not in platforms):
                continue
            if entry.get("model_kind") != "embedding" and entry_backend(family, entry) != FLM_BACKEND:
                continue
            out[f"{family}:{size}"] = entry
    return out


def xclbin_folder(entry: dict) -> str:
    """The src/xclbins folder the engine loads: xclbin_folder, else name
    (AutoModel::_shared_initialize_model_state, model_list::resolve_source)."""
    folder = entry.get("xclbin_folder")
    return folder if isinstance(folder, str) and folder else str(entry.get("name") or "")


def catalog_aliases(catalog: dict, assignment: dict) -> set[str]:
    """NPU tags without a kernel_assignment.json entry that name an
    xclbin_folder: they run the kernels of whichever model uses it."""
    return {t for t, e in npu_catalog(catalog).items()
            if t not in assignment and isinstance(e.get("xclbin_folder"), str) and e["xclbin_folder"]}


def unit_library(dll: Path, unit: str, errors: list) -> Optional[str]:
    """The library a detail/<unit> builds: its Makefile's TARGET
    (lib<name>.so, <name>.dll) and its CMakeLists.txt add_library(<name>
    SHARED), which must agree; None when it builds none (vision_common)."""
    names = set()
    for f, rx in (("Makefile", MK_TARGET), ("CMakeLists.txt", CMAKE_LIBRARY)):
        try:
            lines = (dll / "detail" / unit / f).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        names |= {m.group(1) for l in lines for m in [rx.search(l.split("#", 1)[0])] if m}
    if len(names) > 1:
        errors.append(f"{DLL}/detail/{unit}: its Makefile TARGET and CMakeLists.txt add_library name "
                      f"different libraries ({', '.join(sorted(names))}): make them build one")
        return None
    return next(iter(names), None)


def shipped_names(lib: str) -> tuple[str, ...]:
    """The files of library lib each runtime directory ships."""
    return (f"lib{lib}.so", f"{lib}.dll", f"{lib}.lib")


def check_catalog(repo: Path, catalog: dict, assignment: dict, engines: dict,
                  errors: list, allow: Optional[dict] = None, pending: Optional[list] = None) -> str:
    """src/model_list.json, src/xclbins and src/lib against kernel_assignment
    .json and engines.json; returns a note. allow: CATALOG_ALLOW. pending:
    a list (--pending-ok) gets the paths of the binaries rules 4 and 7 find
    missing, instead of errors."""
    allow = CATALOG_ALLOW if allow is None else allow
    used: set[str] = set()

    def excused(key: str) -> bool:
        if key in allow:
            used.add(key)
            return True
        return False

    npu = npu_catalog(catalog)
    elsewhere = engines.get("served_elsewhere") or {}
    elsewhere = elsewhere if isinstance(elsewhere, dict) else {}
    eng = engines.get("engines") or {}
    eng = eng if isinstance(eng, dict) else {}
    folders: dict[str, set] = {}  # folder -> the slots its kernel_assignment.json models assign
    for m, e in assignment.items():
        if isinstance(e, dict) and isinstance(e.get("folder"), str):
            folders.setdefault(e["folder"], set()).update((e.get("kernels") or {}) if isinstance(
                e.get("kernels"), dict) else ())
    aliases = catalog_aliases(catalog, assignment)
    md = f"and its \"#### `<tag>` → `<folder>`\" section to {ASSIGNMENT_MD}"

    # 1. Every NPU tag has kernels: its own entry, or another model's folder.
    for tag, entry in sorted(npu.items()):
        folder = xclbin_folder(entry)
        if tag in assignment:
            # 3. The folder the assignment fills is the one the engine loads.
            ka = assignment[tag].get("folder") if isinstance(assignment[tag], dict) else None
            if ka != folder:
                errors.append(f"{ASSIGNMENT}: {tag} has folder {ka}, but {MODEL_LIST} gives it xclbin folder "
                              f"{folder} (its xclbin_folder, else its name), which the engine loads: "
                              f"make the two agree (and the {ASSIGNMENT_MD} heading)")
        elif tag in aliases:
            if folder not in folders:
                errors.append(f"{MODEL_LIST}: {tag} runs the kernels of xclbin_folder {folder}, but no "
                              f"{ASSIGNMENT} model has that folder: point it at a model's folder, or give "
                              f"{tag} an entry of its own in {ASSIGNMENT} {md}")
        else:
            errors.append(f"{MODEL_LIST}: NPU ({NPU_PLATFORM}) model {tag} has no {ASSIGNMENT} entry: add "
                          f"\"{tag}\": {{\"folder\": \"{folder}\", \"kernels\": {{\"<slot>.xclbin\": "
                          f"\"<artifact of {BUILD_UNITS}>\", ...}}}} and its \"#### `{tag}` → `{folder}`\" "
                          f"section to {ASSIGNMENT_MD} (a tag that runs another "
                          "model's kernels names that model's folder in xclbin_folder instead)")

    # 2. Every assignment is an NPU model, or runs outside FLM_DLL.
    for m in sorted(set(assignment) - set(npu) - set(elsewhere)):
        errors.append(f"{ASSIGNMENT}: {m} is no NPU ({NPU_PLATFORM}) model of {MODEL_LIST}: drop its entry "
                      f"and section in {ASSIGNMENT_MD}, fix the tag, or list it under served_elsewhere in "
                      f"{ENGINES} if no FLM_DLL engine runs it")

    # 4. Every assigned slot ships.
    for m, e in sorted(assignment.items()):
        if m in elsewhere or not isinstance(e, dict) or not isinstance(e.get("folder"), str):
            continue
        for slot in sorted(e.get("kernels") or {}):
            if not (repo / SHIPPED_XCLBINS / e["folder"] / slot).is_file():
                if pending is not None:
                    pending.append(f"{SHIPPED_XCLBINS}/{e['folder']}/{slot}")
                    continue
                errors.append(f"{ASSIGNMENT}: {m} assigns {slot}, but {SHIPPED_XCLBINS}/{e['folder']}/{slot} "
                              f"is not shipped: add the kernel there, or drop the slot from {ASSIGNMENT} "
                              f"and {ASSIGNMENT_MD}")

    # 5. Every shipped kernel is assigned by a model using its folder.
    root = repo / SHIPPED_XCLBINS
    shipped = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.xclbin") if p.is_file()) \
        if root.is_dir() else []
    for rel in shipped:
        folder, slot = rel.rsplit("/", 1) if "/" in rel else ("", rel)
        path = f"{SHIPPED_XCLBINS}/{rel}"
        if slot in folders.get(folder, ()) or excused(path):
            continue
        whose = (f"no {ASSIGNMENT} model with folder {folder} assigns {slot}" if folder in folders
                 else f"no {ASSIGNMENT} model has folder {folder}")
        errors.append(f"{path} is shipped, but {whose}: add the slot and the build-unit artifact that "
                      f"builds it to {ASSIGNMENT} {md}; if no engine loads it, say why in CATALOG_ALLOW "
                      "(.github/script/ci/devkit_check.py)")

    # 6. Every NPU tag is served by exactly one engine (check_engines reports
    # two), and engines serve catalog models.
    served: dict[str, list] = {}
    for name, e in sorted(eng.items()):
        for m in (e.get("models") or []) if isinstance(e, dict) else []:
            served.setdefault(m, []).append(name)
            if isinstance(m, str) and m not in npu and m not in elsewhere:
                errors.append(f"{ENGINES}: {name} lists {m}, which is no NPU ({NPU_PLATFORM}) model of "
                              f"{MODEL_LIST}: drop it or fix the tag")
    for tag in sorted(set(npu) - set(served)):
        if tag in aliases and excused(tag):
            continue
        errors.append(f"{ENGINES}: NPU model {tag} of {MODEL_LIST} is served by no engine: add it to the "
                      f"models of the engine that runs it; a new model family needs its engine under "
                      f"{DLL}/detail/<engine>/ (Makefile with -Wl,-Bsymbolic-functions, CMakeLists.txt, "
                      f"sources, include/models/<x>/ header) and an entry in {ENGINES}")

    # 7. Every shipped engine library is a dev-kit dir's, and every dir's
    # library ships for each runtime.
    dirs = [d for d in (engines.get("shared") or []) if isinstance(d, str)] + sorted(eng)
    libs: dict[str, str] = {}
    for d in dirs:
        lib = unit_library(repo / DLL, d, errors)
        if lib:
            libs[lib] = d
    owned = {f: lib for lib in libs for f in shipped_names(lib)}
    count = 0
    for rt in SHIPPED_RUNTIMES:
        lib_dir = repo / "src" / "lib" / rt
        files = sorted(p.name for p in lib_dir.iterdir() if p.is_file()) if lib_dir.is_dir() else []
        count += len(files)
        for f in files:
            path = f"src/lib/{rt}/{f}"
            if f in owned or excused(path):
                continue
            errors.append(f"{path} is no library of an {ENGINES} engine or shared unit (their "
                          f"{DLL}/detail/<dir>/Makefile TARGET and CMakeLists.txt add_library name it): add "
                          f"the sources that build it under {DLL}/detail/<dir>/ and the dir to {ENGINES}")
        for lib, d in sorted(libs.items()):
            missing = [f for f in shipped_names(lib) if f not in files]
            if missing and pending is not None:
                pending += [f"src/lib/{rt}/{f}" for f in missing]
            elif missing:
                errors.append(f"{ENGINES}: {d} builds {lib}, but src/lib/{rt}/ lacks {', '.join(missing)}, "
                              f"which src/CMakeLists.txt links for {rt} (lib<name>.so on Linux, <name>.lib "
                              "and <name>.dll on Windows): ship them, or drop the dir from "
                              f"{ENGINES} and {DLL}/detail/")

    for key in sorted(set(allow) - used):
        errors.append(f"CATALOG_ALLOW (.github/script/ci/devkit_check.py): {key} excuses nothing any more "
                      f"({allow[key]}): drop the entry")
    return (f"catalog: {len(npu)} NPU models ({len(aliases)} reusing another's kernels), "
            f"{len(shipped)} shipped kernels, {count} prebuilt library files, {len(libs)} dev-kit libraries, "
            f"{len(used)} allow-listed")


def required_binaries(repo: Path) -> set[str]:
    """The src/ paths check_catalog needs to exist: every assigned slot
    (src/xclbins/<folder>/<slot>, served_elsewhere models skipped, rule 4)
    and every file of every dev-kit library for each runtime
    (src/lib/<rt>/lib<name>.so, <name>.dll, <name>.lib, rule 7). A
    deletion of anything else loses nothing the catalog asks for
    (devkit_swap.py check). Unreadable data gives what could be read."""
    errors: list = []
    assignment = (load_json(repo, ASSIGNMENT, errors) or {}).get("models") or {}
    engines = load_json(repo, ENGINES, errors) or {}
    elsewhere = engines.get("served_elsewhere") or {}
    out = set()
    for m, e in (assignment.items() if isinstance(assignment, dict) else ()):
        if m in elsewhere or not isinstance(e, dict) or not isinstance(e.get("folder"), str):
            continue
        out |= {f"{SHIPPED_XCLBINS}/{e['folder']}/{slot}" for slot in (e.get("kernels") or {})}
    eng = engines.get("engines") or {}
    dirs = [d for d in (engines.get("shared") or []) if isinstance(d, str)]
    dirs += sorted(eng) if isinstance(eng, dict) else []
    for d in dirs:
        lib = unit_library(repo / DLL, d, errors)
        if lib:
            out |= {f"src/lib/{rt}/{f}" for rt in SHIPPED_RUNTIMES for f in shipped_names(lib)}
    return out


def check_bsymbolic(repo: Path, errors: list) -> int:
    detail = repo / DLL / "detail"
    count = 0
    for mk in sorted(detail.glob("*/Makefile")):
        count += 1
        lines = mk.read_text(encoding="utf-8", errors="replace").splitlines()
        if not any(BSYMBOLIC in l.split("#", 1)[0] for l in lines):
            errors.append(f"{mk.relative_to(repo).as_posix()}: no {BSYMBOLIC} outside comments "
                          "(every engine must bind its own copy of the shared runtime)")
    return count


def check_rtp_lock(repo: Path, index: dict, assignment: Optional[dict], errors: list) -> int:
    """The lock parses and names known artifacts some model uses; returns its
    stanza count."""
    try:
        entries, _ = parse_lock((repo / RTP_LOCK).read_text(encoding="utf-8"))
    except OSError as exc:
        errors.append(f"{RTP_LOCK}: cannot read: {exc}")
        return 0
    except ValueError as exc:
        errors.append(f"{RTP_LOCK}: {exc}")
        return 0
    if index:
        for artifact in sorted({a for a, _ in entries} - set(index)):
            errors.append(f"{RTP_LOCK}: {artifact} is no build unit's artifact (build_units.json)")
    if assignment:
        used = assigned(assignment)
        for artifact in sorted({a for a, _ in entries} - used):
            errors.append(f"{RTP_LOCK}: {artifact} is used by no {ASSIGNMENT} model, so no engine "
                          "loads it: drop its stanza (devkit_rtp_lock.py generate leaves it out)")
    return len(entries)


def git_out(repo: Path, *args: str) -> Optional[str]:
    try:
        res = subprocess.run(["git", "-c", "core.quotePath=false", "-C", str(repo), *args],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError:
        return None
    return res.stdout if res.returncode == 0 else None


def check_rtp_lock_change(repo: Path, base: str, head: str, assignment: dict, engines: dict,
                          errors: list, notes: list) -> None:
    """A changed lock stanza needs a change to every engine serving a model
    that uses it (or to an engine that engine uses)."""
    changed = git_out(repo, "diff", "--name-only", "--no-renames", f"{base}...{head}")
    mb = git_out(repo, "merge-base", base, head)
    if changed is None or mb is None:
        errors.append(f"rtp lock: git cannot diff {base}...{head} (fetch the history: fetch-depth: 0)")
        return
    files = [l for l in changed.splitlines() if l.strip()]
    if RTP_LOCK not in files:
        notes.append("rtp lock: unchanged")
        return
    base_text = git_out(repo, "show", f"{mb.strip()}:{RTP_LOCK}")
    if base_text is None:
        notes.append(f"rtp lock: {RTP_LOCK} is new: no engine change required")
        return
    head_text = git_out(repo, "show", f"{head}:{RTP_LOCK}") or ""
    for artifact in changed_artifacts(base_text, head_text):
        models, serving = users(artifact, assignment, engines)
        dirs = [e for e in serving if e.startswith(f"{DLL}/detail/")]
        if not dirs:
            notes.append(f"rtp lock: {artifact} changed; no engine serves a model using it "
                         f"(models: {', '.join(models) or 'none'})")
            continue
        missing = [d for d in dirs if not engine_touched(d, engines, files)]
        if missing:
            errors.append(f"{RTP_LOCK}: the stanza of {artifact} changed, but these engines serving its "
                          f"models ({', '.join(models)}) did not: {', '.join(missing)}: update their "
                          "hard-coded RTP addresses in the same pull request")
        else:
            notes.append(f"rtp lock: {artifact} changed with {', '.join(dirs)}")


def engine_touched(engine_dir: str, engines: dict, files: list[str]) -> bool:
    """files change engine_dir (FLM_DLL/detail/<engine>/) or an engine it uses."""
    eng = engine_dir[len(f"{DLL}/detail/"):].rstrip("/")
    entry = (engines.get("engines") or {}).get(eng) or {}
    deps = {eng, *(entry.get("uses") or [])}
    return any(f.startswith(f"{DLL}/detail/{x}/") for x in deps for f in files)


def needs_notice(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name in NOTICE_NAMES or name.endswith(NOTICE_EXT)


def added_files(repo: Path, base: str, head: str) -> Optional[list[str]]:
    try:
        res = subprocess.run(["git", "-c", "core.quotePath=false", "-C", str(repo), "diff", "--name-only", "--no-renames",
                              "--diff-filter=A", f"{base}...{head}", "--", f"{DEVKIT}/"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError:
        return None
    if res.returncode != 0:
        return None
    return [l for l in res.stdout.splitlines() if l.strip()]


def file_head(repo: Path, head: str, path: str) -> Optional[str]:
    try:
        res = subprocess.run(["git", "-C", str(repo), "show", f"{head}:{path}"],
                             capture_output=True, encoding="utf-8", errors="replace")
    except OSError:
        return None
    return res.stdout if res.returncode == 0 else None


def notice_status(text: str) -> str:
    """"amd", "third-party", "generated" or "missing" for a file's text."""
    top = text.splitlines()[:NOTICE_LINES]
    if any(NOTICE.search(l) for l in top):
        return "amd"
    if any(THIRD_PARTY_NOTICE.search(l) for l in top):
        return "third-party"
    if any(GENERATED.search(l) for l in top[:3]):
        return "generated"
    return "missing"


def check_notices(repo: Path, base: str, head: str, errors: list, notes: list) -> int:
    added = added_files(repo, base, head)
    if added is None:
        errors.append(f"notices: git cannot diff {base}...{head} (fetch the history: fetch-depth: 0)")
        return 0
    checked = 0
    for path in added:
        if not needs_notice(path) or path.startswith(VENDORED):
            continue
        if path in NOTICE_HELD:
            notes.append(f"notices: {path}: held without a notice: {NOTICE_HELD[path]}")
            continue
        checked += 1
        text = file_head(repo, head, path)
        if text is None:
            errors.append(f"{path}: cannot read it at {head}")
            continue
        status = notice_status(text)
        if status in ("third-party", "generated"):
            notes.append(f"notices: {path}: {status}, no AMD notice expected")
        elif status == "missing":
            errors.append(f"{path}: added without an AMD copyright notice in its first {NOTICE_LINES} lines "
                          "(\"Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights "
                          "reserved.\", or for adapted third-party code its notice plus an AMD "
                          "\"Modifications Copyright\" line)")
    return checked


BENCH_WORKFLOW = ".github/workflows/devkit-bench.yml"
BENCH_COMMENT_WORKFLOW = ".github/workflows/devkit-bench-comment.yml"
# Calls BENCH_WORKFLOW; the comment workflow follows its runs.
BUILD_WORKFLOW = ".github/workflows/build-and-test.yml"


def _top_block(text: str, key: str) -> list[str]:
    """The lines of a top-level YAML mapping (key: on its own line), unindented."""
    lines = text.splitlines()
    out, inside = [], False
    for line in lines:
        if re.match(rf"^{re.escape(key)}:\s*(#.*)?$", line):
            inside = True
            continue
        if inside:
            if line and not line[0].isspace() and not line.startswith("#"):
                break
            if line.strip() and not line.strip().startswith("#"):
                out.append(line)
    return out


def _caller_jobs(text: str, workflow: str, with_ids: bool = False):
    """The jobs (their lines) of a workflow that call ./<workflow>: a list,
    or with_ids a dict job id -> lines."""
    jobs, current = [], None
    for line in text.splitlines():
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*(#.*)?$", line)
        if m:
            current = []
            jobs.append((m.group(1), current))
        elif current is not None and line and not line[0].isspace() and not line.startswith("#"):
            current = None
        elif current is not None:
            current.append(line)
    found = [(k, j) for k, j in jobs if any(re.match(rf"^    uses:\s*\./{re.escape(workflow)}\s*(#.*)?$", line)
                                            for line in j)]
    return dict(found) if with_ids else [j for _, j in found]


def _job_condition(job: list[str]) -> str:
    """A job's if: expression (one line, or the first line of a >- block)."""
    for k, line in enumerate(job):
        m = re.match(r"^    if:\s*(.*)$", line)
        if m:
            value = m.group(1).strip()
            if value in (">-", ">", "|", "|-") and k + 1 < len(job):
                value = job[k + 1].strip()
            return value
    return ""


def _run_scripts(text: str) -> list[str]:
    """The bodies of run: | blocks (and one-line run: values)."""
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        m = re.match(r"^(\s*)(- )?run:\s*(.*)$", lines[i])
        if m:
            indent = len(m.group(1)) + (2 if m.group(2) else 0)
            rest = m.group(3)
            if rest and rest[0] not in "|>":
                out.append(rest)
            else:
                i += 1
                while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent):
                    out.append(lines[i])
                    i += 1
                continue
        i += 1
    return out


def check_bench_workflows(repo: Path, errors: list) -> int:
    """The bench comment workflow's invariants (see the docstring); returns
    how many files were checked."""
    try:
        bench = (repo / BENCH_WORKFLOW).read_text(encoding="utf-8")
        comment = (repo / BENCH_COMMENT_WORKFLOW).read_text(encoding="utf-8")
        build = (repo / BUILD_WORKFLOW).read_text(encoding="utf-8")
    except OSError:
        return 0
    where = BENCH_COMMENT_WORKFLOW
    name = re.search(r"^name:\s*(.+?)\s*$", build, re.M)
    build_name = name.group(1).strip("'\"") if name else ""
    on = _top_block(comment, "on")
    wf = re.search(r"workflows:\s*\[\s*['\"]?([^'\"\]]+?)['\"]?\s*\]", "\n".join(on))
    if not build_name or not wf or wf.group(1) != build_name:
        errors.append(f"{where}: on.workflow_run.workflows must be [\"{build_name}\"], the name of "
                      f"{BUILD_WORKFLOW}, which runs the benches (workflow_run matches names, so a rename "
                      f"breaks it silently)")
    if not any(re.match(r"^[ \t]+workflow_run:", line) for line in on) or \
            any(re.match(r"^[ \t]+(pull_request|pull_request_target|push|issue_comment):", line) for line in on):
        errors.append(f"{where}: must trigger on workflow_run only")
    perms = sorted(line.strip() for line in _top_block(comment, "permissions"))
    if perms != ["actions: read", "pull-requests: write"]:
        errors.append(f"{where}: workflow permissions must be exactly actions: read and pull-requests: write, "
                      f"got {perms}")
    if re.search(r"^[ \t]+permissions:", comment, re.M):
        errors.append(f"{where}: no job may set its own permissions")
    if re.search(r"actions/checkout@", comment):
        errors.append(f"{where}: must not check anything out")
    if re.search(r"secrets\.", comment):
        errors.append(f"{where}: must use no secrets")
    for needle in (f"github.event.workflow_run.path == '{BUILD_WORKFLOW}'",
                   "github.event.workflow_run.repository.full_name == github.repository",
                   "github.event.workflow_run.event == 'pull_request'"):
        if needle not in comment:
            errors.append(f"{where}: the job condition must include {needle}")
    for text, path in ((comment, where), (bench, BENCH_WORKFLOW)):
        for line in _run_scripts(text):
            if re.search(r"\$\{\{\s*(github\.event\.|github\.head_ref|inputs\.)", line):
                errors.append(f"{path}: an event field or input is expanded inside a run: script "
                              f"(pass it through env:): {line.strip()[:80]}")
    bench_perms = sorted(line.strip() for line in _top_block(bench, "permissions"))
    if bench_perms != ["actions: read", "contents: read"]:
        errors.append(f"{BENCH_WORKFLOW}: workflow permissions must be exactly contents: read and actions: read, "
                      f"got {bench_perms}")
    if re.search(r"^[ \t]+[a-z-]+: write[ \t]*$", bench, re.M):
        errors.append(f"{BENCH_WORKFLOW}: no job may be granted a write permission")
    callers = _caller_jobs(build, BENCH_WORKFLOW, with_ids=True)
    ids = sorted(callers)
    if ids not in (["devkit-bench"], ["devkit-bench", "devkit-bench-nightly"]):
        errors.append(f"{BUILD_WORKFLOW}: the jobs calling ./{BENCH_WORKFLOW} must be devkit-bench and optionally "
                      f"devkit-bench-nightly (found {', '.join(ids) or 'none'}): the comment workflow follows "
                      f"{BUILD_WORKFLOW}'s runs")
    elif len(ids) == 2:
        # One call per run: the two conditions exclude each other.
        for job_id, needle in (("devkit-bench", "${{ github.event_name != 'schedule' && "),
                               ("devkit-bench-nightly", "${{ github.event_name == 'schedule' && ")):
            cond = _job_condition(callers[job_id])
            if not cond.startswith(needle.strip()):
                errors.append(f"{BUILD_WORKFLOW}: {job_id}'s condition must start with {needle.strip()}, so a run "
                              f"calls ./{BENCH_WORKFLOW} once")
    for job in callers.values():
        if any(re.match(r"^    secrets:", line) for line in job):
            errors.append(f"{BUILD_WORKFLOW}: the job calling {BENCH_WORKFLOW} must pass no secrets")
        perms, inside = [], False
        for line in job:
            if re.match(r"^    permissions:\s*(#.*)?$", line):
                inside = True
            elif inside and re.match(r"^      \S", line):
                perms.append(line.strip())
            elif inside and line.strip() and not line.strip().startswith("#"):
                inside = False
        if sorted(perms) != ["actions: read", "contents: read"]:
            errors.append(f"{BUILD_WORKFLOW}: the job calling {BENCH_WORKFLOW} must grant exactly contents: read "
                          f"and actions: read, got {sorted(perms)}")
    return 3


def _jobs(text: str) -> dict[str, list[str]]:
    """Job id -> its lines, for the jobs: block of a workflow."""
    jobs: dict[str, list[str]] = {}
    current, inside = None, False
    for line in text.splitlines():
        if re.match(r"^jobs:\s*(#.*)?$", line):
            inside = True
            continue
        if inside and line and not line[0].isspace() and not line.startswith("#"):
            break
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*(#.*)?$", line) if inside else None
        if m:
            current = jobs.setdefault(m.group(1), [])
        elif current is not None:
            current.append(line)
    return jobs


def check_swap_wiring(repo: Path, errors: list) -> int:
    """The names devkit_swap.py reads another run by agree with the CI that
    makes them (see the docstring); returns how many files were checked."""
    try:
        build = (repo / BUILD_WORKFLOW).read_text(encoding="utf-8")
    except OSError:
        return 0
    import devkit_swap  # here: devkit_swap imports this module
    where = "devkit_swap.py"
    if devkit_swap.WORKFLOW_PATH != BUILD_WORKFLOW:
        errors.append(f"{where}: WORKFLOW_PATH is {devkit_swap.WORKFLOW_PATH}, not {BUILD_WORKFLOW}")
    jobs = _jobs(build)
    for job in (devkit_swap.SUMMARY_JOB, devkit_swap.STAGE_JOB) + tuple(
            p.split(" / ")[0] for p in devkit_swap.BUILD_JOB_PREFIXES):
        if job not in jobs:
            errors.append(f"{BUILD_WORKFLOW}: no job {job}, which {where} reads a run's results by")
    stage = "\n".join(jobs.get(devkit_swap.STAGE_JOB, []))
    if stage and ("devkit_stage.py" not in stage or any(re.match(r"^    name:", l) for l in
                                                         jobs[devkit_swap.STAGE_JOB])):
        errors.append(f"{BUILD_WORKFLOW}: the {devkit_swap.STAGE_JOB} job must run devkit_stage.py and keep "
                      f"its id as its name ({where} STAGE_JOB)")
    for (os_name, runtime), name in devkit_swap.TEST_JOBS.items():
        artifact = devkit_swap.manifest_artifact(os_name, runtime)
        upload = re.search(rf"name:[ \t]*{re.escape(artifact)}[ \t]*\n\s*path:[ \t]*(.+?)[ \t]*$", stage, re.M)
        if not upload or not upload.group(1).endswith(
                f"/manifest-{os_name}-{runtime}/{devkit_swap.MANIFEST_FILE}"):
            errors.append(f"{BUILD_WORKFLOW}: the {devkit_swap.STAGE_JOB} job must upload "
                          f".../manifest-{os_name}-{runtime}/{devkit_swap.MANIFEST_FILE} as {artifact}, the "
                          f"artifact {where} reads (MANIFEST_ARTIFACT, MANIFEST_FILE)")
        found = [lines for lines in jobs.values() if f"    name: {name}" in lines]
        if len(found) != 1:
            errors.append(f"{BUILD_WORKFLOW}: {where} TEST_JOBS expects exactly one job named \"{name}\" "
                          f"(the product test job of {os_name} {runtime})")
    summary = jobs.get(devkit_swap.SUMMARY_JOB, [])
    step = next((i for i, l in enumerate(summary) if l.strip() == f"- name: {devkit_swap.PENDING_STEP}"), None)
    if step is None or "devkit_swap.py pending" not in "\n".join(summary[step:step + 30]):
        errors.append(f"{BUILD_WORKFLOW}: the summary job needs a step named \"{devkit_swap.PENDING_STEP}\" "
                      f"running devkit_swap.py pending ({where} PENDING_STEP: fetch accepts a run whose only "
                      "failure is that step)")
    changes_job = "\n".join(jobs.get("changes", []))
    for sub in ("verify", "check"):
        if f"devkit_swap.py {sub}" not in changes_job:
            errors.append(f"{BUILD_WORKFLOW}: the changes job must run devkit_swap.py {sub}")
    if not re.search(r"^      actions: read\s*$", changes_job, re.M):
        errors.append(f"{BUILD_WORKFLOW}: the changes job needs actions: read (devkit_swap.py verify reads "
                      "another run's manifests)")
    return 1


def run(repo: Path, base: str = "", head: str = "HEAD",
        pending: Optional[list] = None) -> tuple[list[str], list[str]]:
    """(errors, notes) for the tree at repo; pending: see check_catalog."""
    errors: list[str] = []
    notes: list[str] = []
    units = load_json(repo, BUILD_UNITS, errors)
    index = check_build_units(repo, units, errors) if units is not None else {}
    if units is not None:
        notes.append(f"build units: {len(units.get('units') or {})} units, {len(index)} artifacts")
    assignment = load_json(repo, ASSIGNMENT, errors)
    models = check_assignment(assignment, index, errors) if assignment is not None else {}
    if assignment is not None:
        notes.append(f"assignment: {len(models)} models")
    try:
        md_text = (repo / ASSIGNMENT_MD).read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"{ASSIGNMENT_MD}: cannot read: {exc}")
        md_text = None
    if md_text is not None and assignment is not None:
        check_assignment_md(md_text, models, index, errors)
        notes.append(f"assignment md: {len(parse_assignment_md(md_text)[0])} model sections compared")
    catalog = load_json(repo, MODEL_LIST, errors)
    engines = load_json(repo, ENGINES, errors)
    if engines is not None:
        check_engines(repo, engines, models, errors,
                      catalog_aliases(catalog, models) if catalog is not None else set())
        notes.append(f"engines: {len(engines.get('engines') or {})} engines, "
                     f"{len(engines.get('shared') or [])} shared")
    if catalog is not None and engines is not None and assignment is not None:
        notes.append(check_catalog(repo, catalog, models, engines, errors, pending=pending))
    notes.append(f"rtp lock: {check_rtp_lock(repo, index, assignment, errors)} stanzas")
    notes.append(f"Bsymbolic: {check_bsymbolic(repo, errors)} Makefiles")
    notes.append(f"bench workflows: {check_bench_workflows(repo, errors)} checked")
    notes.append(f"swap wiring: {check_swap_wiring(repo, errors)} files checked")
    if base.strip("0 "):
        notes.append(f"notices: {check_notices(repo, base, head, errors, notes)} added source/script files")
        check_rtp_lock_change(repo, base, head, assignment or {}, engines or {}, errors, notes)
    else:
        notes.append("notices: skipped (no --base)")
    return errors, notes


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--base", default="",
                        help="Check the notices of files added, and the RTP lock's changes, since the "
                             "merge base with this commit; "
                             "empty or all-zero skips that check.")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--pending-ok", action="store_true",
                        help="a pull request's run: missing assigned slots and engine libraries are pending "
                             "(its run must stage them), not errors")
    parser.add_argument("--github-output", action="store_true", help="append pending=<csv> to $GITHUB_OUTPUT")
    args = parser.parse_args(argv)
    pending: Optional[list] = [] if args.pending_ok else None
    errors, notes = run(args.repo, args.base, args.head, pending)
    for note in notes:
        print(f"devkit_check: {note}")
    for path in pending or []:
        print(f"devkit_check: pending swap: {path} is not in src/ yet; this pull request's run must stage it, "
              "and its swap commit it")
    if args.github_output:
        target = os.environ.get("GITHUB_OUTPUT")
        if not target:
            print("--github-output given but $GITHUB_OUTPUT is not set", file=sys.stderr)
            return 2
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(f"pending={','.join(sorted(set(pending or [])))}\n")
    for err in errors:
        print(f"devkit_check: error: {err}")
    if errors:
        print(f"devkit_check: {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("devkit_check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
