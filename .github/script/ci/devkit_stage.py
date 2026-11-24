#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Stages a pull request's dev-kit engines and kernels into the product builds.

    devkit_stage.py --out <dir> --head-sha <sha> --base-sha <sha> [--repo <checkout>]
                    [--families <csv>] [--libs <short>-<runtime>=<dir> ...]
                    [--xclbins <dir>] [--kernel-models <json list, default $DEVKIT_KERNEL_MODELS>]
                    [--suites <changes suites>] [--product-families <csv>]
                    [--new-models <csv>] [--require-all] [--full-refresh]
                    [--runtimes xrt,hrx] [--github-output]

Build and Test's devkit-stage job runs this once, on a hosted runner, after
the dev-kit builds (devkit-engines, devkit-xclbins) and before the product
builds, on a pull request whose dev-kit engine or kernel source changed
(changes.py devkit_stage) and on the dev-kit full refresh. It decides which
dev-kit files go into which product build, checks them, and writes, for
each system and runtime (<short> lin or win, <runtime> xrt or hrx):

  <out>/stage-<short>-<runtime>/   the files at their repository paths
                                   (src/lib/<runtime>/..., src/xclbins/...)
                                   and devkit-stage.sha256 (sha256sum's
                                   format) listing exactly them; uploaded as
                                   devkit-stage-<short>-<runtime>, which
                                   .github/actions/devkit-stage-apply copies
                                   over the build's checkout before CMake runs
  <out>/manifest-<os>-<runtime>/manifest.json
                                   the record devkit_swap.py reads (below);
                                   uploaded as devkit-stage-manifest-<os>-
                                   <runtime>, kept 30 days

Nothing is committed: the packages built from the staged tree are the state
after a swap (flm-dev-kit/AGENTS.md, "Prebuilt binaries"), so the product
NPU tests test exactly the bytes devkit_swap.py fetch would commit.

Engines (from --libs, the devkit-libs-<short>-<runtime> artifacts), for
the runtimes of --runtimes (changes.py devkit_runtimes: "hrx" alone when
only the pinned HRX package changed; the other runtime's systems get no
engine, and kernels as usual): those
whose engines.json families meet --families (changes.py devkit_families),
or every engine and shared unit when --families holds every engine's
families (devkit_overlay.units_to_overlay); for each, its library (the
Makefile TARGET devkit_check.py reads, devkit_check.unit_library): Linux
lib<name>.so, Windows <name>.dll and the <name>.lib import library the
build links against. A file src/lib/<runtime>/ has is replaced; a file it
lacks is new, which is allowed because the unit is an engines.json entry
(a new engine). The ABI check of devkit_overlay.py then runs on the
runtime's libraries as they will be (src/lib/<runtime>/, and on Windows the
runtime libraries under src/lib/ the package ships beside them): a replaced
library's exports and imports against the one it replaces and the rest; a
new library has nothing to compare with. flm itself is relinked by the
build, so the linker checks its imports.

Kernels (from --xclbins, the devkit-xclbins artifact, <folder>/<slot>):
for each --kernel-models model (devkit-xclbins' bench-models that also got
a kernel: the models whose kernel source or kernel_assignment.json entry
changed; not the canary builds a CI change makes, nor shipped kernels
rebuilt only to compare), the slots its kernel_assignment.json entry
assigns, into src/xclbins/<folder>/<slot> in every build. Only assigned
slots are staged, so a new file is always a declared one. A model outside
the NPU catalog (served elsewhere, e.g. SmolVLA) is left out: no product
test would load its kernels.

Every path is checked as devkit_swap.py fetch checks a manifest's
(devkit_swap.check_entry_path), every source is a regular file of the
expected format, never a symbolic link. Every binary the catalog needs
that the checkout lacks (devkit_check.required_binaries: an assigned slot,
an engine library of either runtime; what devkit_check.py --pending-ok
reports as pending on a pull request, a new model's or engine's) must be
staged here; one that is not is an error, as it is without a pull
request.

Test selection for the four product test jobs (outputs): the union of the
product's (--suites, --product-families, --new-models) and the dev kit's
(devkit_overlay.select_tests: the families of the staged engines, the
staged kernels' models, required): suites, hrx-suites (less api and
serving, as .github/actions/detect-changes gives them), families, required
(the new models and the staged kernels' catalog models; with
--require-all, the dev-kit full refresh, every NPU catalog model). When
something is staged, suites and hrx-suites are never empty: flm links
every engine at start-up, so any chat run loads a staged engine.

Manifest, schema 2:
  schema, os, runtime, run_id and run_attempt ($GITHUB_RUN_ID,
  $GITHUB_RUN_ATTEMPT), head_sha (--head-sha: the commit built), base_sha
  (--base-sha: the main commit the run built on), full_refresh, and files,
  one entry per staged file:
    kind          engine or kernel
    repo_path     src/lib/<runtime>/<file> or src/xclbins/<folder>/<slot>
    artifact, artifact_path
                  where the bytes came from (devkit-libs-<short>-<runtime>
                  or devkit-xclbins, and the path inside it), for fetch
    size, sha256, git_sha1 (the git blob id of the bytes)
    new           true when the checkout has no file at repo_path
The product test job of that system and runtime must pass for its files to
count as tested (devkit_swap.py TEST_JOBS).

Outputs (printed, and appended to $GITHUB_OUTPUT with --github-output):
staged (true when any file is staged), suites, hrx-suites, families,
required, count. A Markdown summary goes to $GITHUB_STEP_SUMMARY. Exit 1 on
an error (nothing written then), 2 on bad arguments. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import changes  # noqa: E402
import devkit_check  # noqa: E402
import devkit_overlay  # noqa: E402
import devkit_swap  # noqa: E402

REPO_ROOT = HERE.parents[2]
MANIFEST_SCHEMA = 2
SUMS_FILE = "devkit-stage.sha256"
# (os, runtime) pairs, in output order.
TARGETS = (("linux", "xrt"), ("linux", "hrx"), ("windows", "xrt"), ("windows", "hrx"))
# detect-changes' HRX suites for "all": the HRX packages skip api and serving.
HRX_ALL = "chat,prefix_cache,embedding,decision,asr"
HRX_SKIP = {"api", "serving", "none"}
# A Windows import library is an ar archive.
AR_MAGIC = b"!<arch>\n"


def short(os_name: str) -> str:
    return devkit_swap.SHORT_OS[os_name]


def stage_artifact(os_name: str, runtime: str) -> str:
    return f"devkit-stage-{short(os_name)}-{runtime}"


@dataclass
class Staged:
    """One file to stage: where it comes from and the repository path it becomes."""
    kind: str
    repo_path: str
    source: Path
    artifact: str
    artifact_path: str


@dataclass
class Plan:
    files: dict = field(default_factory=dict)   # (os, runtime) -> [Staged]
    notes: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)


def csv(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def magic_ok(path: Path, os_name: str) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(8)
    if path.suffix == ".lib":
        return head == AR_MAGIC
    return head[:2] == b"MZ" if os_name == "windows" else head[:4] == b"\x7fELF"


def regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def library_of(repo: Path, unit: str) -> str:
    """The library detail/<unit> builds (devkit_check's reading of its
    Makefile TARGET), else devkit_overlay's name for it."""
    found = devkit_check.unit_library(repo / devkit_check.DLL, unit, [])
    return found or devkit_overlay.lib_stem(unit)


def engine_files(repo: Path, units: list[str], os_name: str) -> list[str]:
    out = []
    for unit in units:
        lib = library_of(repo, unit)
        out += [f"{lib}.dll", f"{lib}.lib"] if os_name == "windows" else [f"lib{lib}.so"]
    return out


def runtime_binaries(repo: Path, os_name: str, runtime: str) -> dict[str, Path]:
    """Library name (lower case on Windows) -> its file, for the ABI check:
    src/lib/<runtime>/, and on Windows also the runtime libraries directly
    under src/lib/ the package ships beside them (the runtime's own win)."""
    lib = repo / "src" / "lib"
    pattern = "*.dll" if os_name == "windows" else "*.so"
    dirs = [lib, lib / runtime] if os_name == "windows" else [lib / runtime]
    out: dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob(pattern)):
            if regular_file(p) and devkit_overlay.readable(p, os_name):
                out[p.name.lower() if os_name == "windows" else p.name] = p
    return out


def plan_engines(repo: Path, plan: Plan, os_name: str, runtime: str, units: list[str],
                 libs: Optional[Path]) -> list[Staged]:
    if not units:
        return []
    label = f"{os_name} {runtime}"
    if libs is None or not libs.is_dir():
        plan.errors.append(f"{label}: engines to stage ({', '.join(units)}) but no dev-kit libraries "
                           f"({devkit_swap.libs_artifact(os_name, runtime)})")
        return []
    artifact = devkit_swap.libs_artifact(os_name, runtime)
    out = []
    for name in engine_files(repo, units, os_name):
        repo_path = f"src/lib/{runtime}/{name}"
        why = devkit_swap.check_entry_path("engine", repo_path, os_name, runtime)
        src = libs / name
        if why:
            plan.errors.append(f"{label}: {why}")
        elif not regular_file(src):
            plan.errors.append(f"{label}: {name} is not in the dev-kit libraries ({artifact})")
        elif not magic_ok(src, os_name):
            plan.errors.append(f"{label}: the dev-kit {name} is not a {os_name} "
                               f"{'import library' if name.endswith('.lib') else 'library'}")
        else:
            out.append(Staged("engine", repo_path, src, artifact, name))
    if plan.errors:
        return out
    # The ABI check, on the runtime's libraries as they will be.
    current = runtime_binaries(repo, os_name, runtime)
    final: dict[Path, Path] = {p: p for p in current.values()}
    replaced = set()
    for s in out:
        if s.repo_path.endswith(".lib"):
            continue
        dest = repo.joinpath(*s.repo_path.split("/"))
        key = dest.name.lower() if os_name == "windows" else dest.name
        place = current.get(key, dest)
        final[place] = s.source
        if place in current.values() and place.is_file():
            replaced.add(place)
    try:
        for e in devkit_overlay.abi_errors_of(final, os_name, replaced):
            plan.errors.append(f"{label}: {e}")
    except (devkit_overlay.BinaryError, struct.error, ValueError, IndexError) as exc:
        plan.errors.append(f"{label}: cannot read the symbol tables of src/lib/{runtime} or the dev-kit "
                           f"libraries: {exc}")
    return out


def plan_kernels(repo: Path, plan: Plan, xclbins: Optional[Path], models: list[str],
                 assignment: dict, catalog: dict) -> tuple[list[Staged], list[str]]:
    """(the kernels to stage, the catalog models they serve)."""
    if not models:
        return [], []
    if xclbins is None or not xclbins.is_dir():
        plan.errors.append(f"kernels to stage for {', '.join(models)} but no dev-kit kernels "
                           f"({devkit_swap.XCLBINS_ARTIFACT})")
        return [], []
    out, served, seen = [], [], set()
    for model in models:
        entry = assignment.get(model)
        if not isinstance(entry, dict) or not isinstance(entry.get("folder"), str):
            plan.notes.append(f"{model}: no kernel_assignment.json entry; nothing staged for it")
            continue
        if model not in catalog:
            plan.notes.append(f"{model}: not an NPU catalog model, so no product test would load its "
                              "kernels; not staged")
            continue
        folder, got = entry["folder"], 0
        for slot in sorted(entry.get("kernels") or {}):
            repo_path = f"src/xclbins/{folder}/{slot}"
            why = devkit_swap.check_entry_path("kernel", repo_path, "linux", "xrt")
            if why:
                plan.errors.append(why)
                continue
            src = xclbins.joinpath(folder, slot)
            if not regular_file(src):
                continue  # not rebuilt: the shipped one stays
            got += 1
            if repo_path not in seen:
                seen.add(repo_path)
                out.append(Staged("kernel", repo_path, src, devkit_swap.XCLBINS_ARTIFACT, f"{folder}/{slot}"))
        if got:
            served.append(model)
        else:
            plan.notes.append(f"{model}: the dev-kit kernels hold none of its slots; nothing staged for it")
    return out, served


def union_suites(product: str, devkit: str, staged: bool) -> str:
    """The suites of both selections ("all" wins; "none" is nothing)."""
    product = product or "none"
    if product == "all":
        return "all"
    names = set(csv(product)) - {"none"} | set(csv(devkit))
    if not names and staged:
        names = {"chat"}
    return ",".join(s for s in changes.SUITES if s in names) or "none"


def hrx_suites(suites: str) -> str:
    if suites == "all":
        return HRX_ALL
    return ",".join(s for s in csv(suites) if s not in HRX_SKIP)


def select(plan: Plan, catalog: dict, engines: changes.DevkitEngines, families: set, every: bool,
           units: list[str], kernel_models: list[str], args) -> None:
    test_families = set(families) if every else families | {f for u in units for f in engines.families(u)}
    new_models = csv(args.new_models)
    if args.require_all:
        new_models = list(dict.fromkeys(new_models + sorted(catalog)))
    sel, notes = devkit_overlay.select_tests(catalog, families, every, kernel_models, new_models,
                                             test_families)
    plan.notes += notes
    staged = any(plan.files.values())
    suites = union_suites(args.suites, sel["suites"], staged)
    hrx = hrx_suites(suites) or ("chat" if staged else "")
    fams = sorted(set(csv(args.product_families)) | set(csv(sel["families"])))
    plan.outputs = {
        "staged": str(staged).lower(),
        "suites": suites,
        "hrx-suites": hrx,
        "families": ",".join(fams),
        "required": sel["required"],
        "count": str(sum(len(v) for v in plan.files.values())),
    }


def load(repo: Path, rel: str, plan: Plan) -> dict:
    try:
        data = json.loads((repo / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        plan.errors.append(f"{rel}: cannot read: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def make_plan(args) -> Plan:
    plan = Plan()
    repo: Path = args.repo
    engines = changes.load_devkit_engines((repo / changes.DEVKIT_ENGINES).read_text(encoding="utf-8")
                                          if (repo / changes.DEVKIT_ENGINES).is_file() else "")
    if engines is None:
        plan.errors.append(f"{changes.DEVKIT_ENGINES} is missing or unreadable")
        return plan
    assignment = load(repo, devkit_check.ASSIGNMENT, plan).get("models") or {}
    catalog = devkit_overlay.npu_catalog(repo / "src")
    families = set(csv(args.families))
    unknown = families - engines.all_families()
    if unknown:
        plan.notes.append(f"families no dev-kit engine serves: {', '.join(sorted(unknown))}")
    units, every = devkit_overlay.units_to_overlay(engines, families)
    kernels, served = plan_kernels(repo, plan, args.xclbins, json.loads(args.kernel_models or "[]"),
                                   assignment if isinstance(assignment, dict) else {}, catalog)
    runtimes = set(csv(args.runtimes))
    if runtimes != {"xrt", "hrx"}:
        plan.notes.append(f"engines of the {', '.join(sorted(runtimes))} runtime(s) only (changes.py "
                          "devkit_runtimes): the other runtime's libraries are neither built nor staged")
    for os_name, runtime in TARGETS:
        libs = args.libs.get(f"{short(os_name)}-{runtime}")
        plan.files[(os_name, runtime)] = (plan_engines(repo, plan, os_name, runtime,
                                                       units if runtime in runtimes else [], libs) + kernels)
    # The pending binaries (devkit_check.py --pending-ok): what src/ lacks
    # must come from here.
    staged = {s.repo_path for files in plan.files.values() for s in files}
    pending = sorted(p for p in devkit_check.required_binaries(repo) if not (repo / p).is_file())
    for path in pending:
        if path not in staged:
            plan.errors.append(f"{path} is required (an assigned slot or an engine library) and not in "
                               "src/, and this run does not stage it: add the dev-kit source that builds "
                               "it, or drop the entry")
    if pending:
        plan.notes.append(f"{len(pending)} file(s) new to src/ (pending their swap), e.g. `{pending[0]}`")
    select(plan, catalog, engines, families, every, units, served, args)
    return plan


def digests(path: Path) -> dict:
    data = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "git_sha1": blob}


def write(args, plan: Plan) -> None:
    out: Path = args.out
    for (os_name, runtime), files in plan.files.items():
        tree = out / f"stage-{short(os_name)}-{runtime}"
        tree.mkdir(parents=True, exist_ok=True)
        sums, entries = [], []
        for s in sorted(files, key=lambda f: f.repo_path):
            d = digests(s.source)
            dest = tree.joinpath(*s.repo_path.split("/"))
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s.source, dest)
            sums.append(f"{d['sha256']}  {s.repo_path}\n")
            entries.append({"kind": s.kind, "repo_path": s.repo_path, "artifact": s.artifact,
                            "artifact_path": s.artifact_path, **d,
                            "new": not (args.repo / s.repo_path).is_file()})
        # Bytes, so the files are the same on every runner (no CRLF).
        (tree / SUMS_FILE).write_bytes("".join(sums).encode("utf-8"))
        manifest = {
            "schema": MANIFEST_SCHEMA, "os": os_name, "runtime": runtime,
            "run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
            "head_sha": args.head_sha, "base_sha": args.base_sha,
            "full_refresh": bool(args.full_refresh), "files": entries,
        }
        mdir = out / f"manifest-{os_name}-{runtime}"
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / devkit_swap.MANIFEST_FILE).write_bytes((json.dumps(manifest, indent=1) + "\n").encode("utf-8"))


def summary(plan: Plan) -> str:
    lines = ["### Dev-kit files staged into the product builds", ""]
    lines += [f"- ❌ {e}" for e in plan.errors]
    for (os_name, runtime), files in plan.files.items():
        engines = [s.repo_path.rsplit("/", 1)[1] for s in files if s.kind == "engine"]
        kernels = sorted({s.repo_path.split("/")[2] for s in files if s.kind == "kernel"})
        lines.append(f"- {os_name} {runtime}: {len(engines)} engine file(s)"
                     + (f" ({', '.join(f'`{e}`' for e in engines[:12])}{', ...' if len(engines) > 12 else ''})"
                        if engines else "")
                     + f"; kernels in {len(kernels)} folder(s)"
                     + (f" ({', '.join(f'`{k}`' for k in kernels)})" if kernels else ""))
    lines += [f"- {n}" for n in plan.notes]
    if plan.outputs:
        o = plan.outputs
        lines.append(f"- Product tests: suites `{o['suites']}` (HRX `{o['hrx-suites'] or 'none'}`); families "
                     f"`{o['families'] or 'one model per family (pr tier)'}`; required `{o['required'] or 'none'}`")
    return "\n".join(lines) + "\n"


def parse_libs(values: list[str], parser: argparse.ArgumentParser) -> dict[str, Path]:
    out = {}
    for v in values:
        key, sep, path = v.partition("=")
        if not sep or key not in {f"{short(o)}-{r}" for o, r in TARGETS}:
            parser.error(f"--libs {v!r}: expected <lin|win>-<xrt|hrx>=<dir>")
        out[key] = Path(path)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--families", default="")
    parser.add_argument("--libs", action="append", default=[])
    parser.add_argument("--xclbins", type=Path, default=None)
    parser.add_argument("--kernel-models", default=os.environ.get("DEVKIT_KERNEL_MODELS", "[]"),
                        help="JSON list; default $DEVKIT_KERNEL_MODELS")
    parser.add_argument("--suites", default="none", help="changes' suites for the product")
    parser.add_argument("--product-families", default="")
    parser.add_argument("--new-models", default="")
    parser.add_argument("--require-all", action="store_true",
                        help="every NPU catalog model is required (the dev-kit full refresh)")
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--runtimes", default="xrt,hrx",
                        help="runtimes whose engines devkit-engines built (changes.py devkit_runtimes); "
                             "the others' engines are not staged (kernels still are)")
    parser.add_argument("--github-output", action="store_true")
    args = parser.parse_args(argv)
    args.libs = parse_libs(args.libs, parser)
    if not csv(args.runtimes) or set(csv(args.runtimes)) - {"xrt", "hrx"}:
        parser.error(f"--runtimes {args.runtimes!r}: expected a comma-separated subset of xrt, hrx")
    if args.full_refresh and set(csv(args.runtimes)) != {"xrt", "hrx"}:
        parser.error("--full-refresh stages every runtime's engines: --runtimes must be xrt,hrx")
    try:
        if not isinstance(json.loads(args.kernel_models or "[]"), list):
            raise ValueError
    except ValueError:
        parser.error("--kernel-models must be a JSON list")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"--out {args.out} is not empty")

    plan = make_plan(args)
    if not plan.errors:
        write(args, plan)
    text = summary(plan)
    print(text, end="")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    for e in plan.errors:
        print(f"::error::devkit stage: {e}")
    if plan.errors:
        return 1
    for key, value in plan.outputs.items():
        print(f"{key}={value}")
    if args.github_output:
        target = os.environ.get("GITHUB_OUTPUT")
        if not target:
            print("--github-output given but $GITHUB_OUTPUT is not set", file=sys.stderr)
            return 2
        with open(target, "a", encoding="utf-8") as fh:
            fh.writelines(f"{k}={v}\n" for k, v in plan.outputs.items())
    return 0


if __name__ == "__main__":
    sys.exit(main())
