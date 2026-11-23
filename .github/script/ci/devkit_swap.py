#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Commits the dev-kit binaries CI built and tested, and checks that it did.

    devkit_swap.py fetch   [--pr <n>] [--run <id>] [--repo <owner/name>] [--remote origin]
                           [--dry-run] [--no-git-fetch]
    devkit_swap.py verify  --head <sha> --base <sha> --pr <n> --head-ref <branch>
                           --head-repo <owner/name> [--repo <owner/name>] [--github-output]
    devkit_swap.py check   --head <sha> --base <sha> --pr <n> --head-ref <branch>
                           --head-repo <owner/name> [--repo <owner/name>]
    devkit_swap.py pending --manifests <dir> --head <sha> --run <id> [--tree <commit>]
                           [--base <sha>] [--summary <file>]

The policy (flm-dev-kit/AGENTS.md, "Prebuilt binaries"): the prebuilt engine
libraries (src/lib/{xrt,hrx}/*.so, *.dll, *.lib) and kernels
(src/xclbins/<folder>/<slot>.xclbin) are never edited by hand. Only a pull
request that changes dev-kit source updates them, and the bytes committed
are exactly the bytes its Build and Test run built (devkit-engines,
devkit-xclbins), staged into its product builds before they compiled
(devkit-stage, devkit_stage.py) and NPU-tested in its product test jobs
(TEST_JOBS). Kernel builds are not byte-reproducible, so nothing is rebuilt
after the fact: the staging step records what it put into each system's
and runtime's build (the devkit-stage-manifest-<os>-<runtime> artifacts,
schema 2, documented in devkit_stage.py), and the pull request's author
copies exactly those files.

fetch (the author, locally; read access is enough: the gh CLI's login, or
$GH_TOKEN / $GITHUB_TOKEN): checks that the working tree is clean, that the
local HEAD is the pull request's head (nothing unpushed, nothing after the
tested commit) and the run's head, that the run is this repository's Build
and Test for that pull request (or a dev-kit full refresh dispatched on its
branch, see below), completed, that every job of it passed (check_jobs: one
by one, not the run's conclusion, so that the one failure a staged run
always has, summary's PENDING_STEP saying the swap is pending, is allowed,
and so is a failed bench job, which summary only warns about; any other
failed job or summary step is not), that the product test job of
every system and runtime the run staged files into passed, that the pull
request is not from a fork, and that main gained no commit touching the
dev-kit engine or kernel sources or the prebuilt binaries since the run's
base (documentation there does not count, see main_moved; else merge main
and let CI run again). Then it downloads the manifests
and the artifacts holding the files that differ from the branch, checks
every file's size and SHA-256 against the manifests, writes them under
src/lib and src/xclbins, and makes ONE commit (no push) whose message ends
with the trailers

    Devkit-Swap-Run: <Build and Test run id>
    Devkit-Swap-Tested: <the commit that run tested>

--dry-run prints the plan and changes nothing. A run's devkit-libs-* and
devkit-xclbins artifacts are kept 7 days, the manifests 30: once they are
gone, re-run the workflow.

verify (Build and Test's changes job, pull requests): whether the pull
request's head is a swap commit CI can trust without testing again: it has
the trailers (whoever wrote it), its only parent is the tested commit, it
changes only paths the run's manifests list, each to exactly the manifest's
bytes, and carries every file the run staged (no half swap: one system's
build of an engine, part of the kernels), the recorded run passes the
checks fetch makes (read with the job's token, actions: read: only the
small manifests are downloaded), main gained no dev-kit source or binary
change (documentation does not count) between the run's base and the pull
request's base, and the pull request changes none of SWAP_CI_PATHS (the
code that stages, writes and checks the manifests: it runs from the pull
request's merge commit, so a swap must not come with changes to it). The bytes are compared by git blob
id (the manifest records it beside the SHA-256; both hash exactly the
bytes), read from the trees alone, so the changes job's blob:none clone
downloads no binary. Prints the reasons and writes swap-verified=true|false
with --github-output. Build and Test then runs no product build, NPU test,
dev-kit build, staging or bench for a verified swap; devkit-static still
runs. verify and check run the pull request's own code: they keep honest
authors on the flow, the guarantee is review of any .github/ change and
branch protection.

check (every pull request): every path under src/lib/ or src/xclbins/ the
pull request changes against its merge base must be (a) byte-equal (blob
id) to the manifest entry for that path of a swap commit in the pull request that
verifies against its run, (b) a deletion of a file no assigned kernel slot
or shipped engine library needs (devkit_check.required_binaries), or (c) a
third-party file (third_party(): the src/lib entries of devkit_check's
CATALOG_ALLOW, and the runtime libraries directly under src/lib/).
Anything else is a hand edit: an error (SWAP_ENFORCE), which Build and
Test's summary reports.

pending (Build and Test's summary job, its step PENDING_STEP): from this
run's manifests, the staged files whose blob ids differ from those of
--tree (the commit built): a swap is pending. Prints nothing and exits 0
when there is none; otherwise writes the fetch command to stdout and to
--summary (the job summary) and exits 1, so the pull request cannot merge
without its swap. With --base, also says when the pull request changes
SWAP_CI_PATHS, which no verified swap can come with (split it).

The dev-kit full refresh: a workflow_dispatch of Build and Test on a pull
request's branch with devkit-full-refresh (build-and-test.yml) builds every
engine and every assigned kernel, stages all of them into the product
builds, runs the product test jobs on the dispatch's tier (default pr) with every NPU catalog
model required, and marks its manifests full_refresh. fetch and verify
accept such a run for the pull request whose head is the run's head.

Untrusted data: the artifacts are written by code from the pull request.
Names, sizes and zip entries are checked before anything is read; only
regular zip members named by a manifest are read (no absolute paths, no
"..", no backslashes, no symlinks), every byte is hashed before anything is
written, and only src/lib/{xrt,hrx}/*.{so,dll,lib} and
src/xclbins/<folder>/<slot>.xclbin are ever written, through no symlink.
The token goes to api.github.com only: API calls follow no redirect, and
artifact downloads follow GitHub's redirect without it. Standard library only; every GitHub call goes through
Api, which the tests replace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import changes  # noqa: E402
import devkit_check  # noqa: E402

# check fails a pull request with a hand-edited binary (False: it only warns).
SWAP_ENFORCE = True

WORKFLOW_PATH = ".github/workflows/build-and-test.yml"
DEFAULT_REPO = "ROCm/FastFlowLM"
TRAILER_RUN = "Devkit-Swap-Run"
TRAILER_TESTED = "Devkit-Swap-Tested"

# (os, runtime) -> the product test job in build-and-test.yml that tests
# the package that system's and runtime's staged files went into.
# devkit_check.py checks these names against the workflow.
TEST_JOBS = {
    ("linux", "xrt"): "Test Linux XRT portable",
    ("linux", "hrx"): "Test Linux HRX portable",
    ("windows", "xrt"): "Test Windows XRT",
    ("windows", "hrx"): "Test Windows HRX",
}
MANIFEST_ARTIFACT = "devkit-stage-manifest-{os}-{runtime}"
MANIFEST_FILE = "manifest.json"
MANIFEST_SCHEMA = 2
LIBS_ARTIFACT = "devkit-libs-{short}-{runtime}"
SHORT_OS = {"linux": "lin", "windows": "win"}
XCLBINS_ARTIFACT = "devkit-xclbins"
SUMMARY_JOB = "summary"
STAGE_JOB = "devkit-stage"
# The summary step that fails while a swap is pending: the one failure a
# run may have and still be swapped in.
PENDING_STEP = "Dev-kit swap pending"
# Jobs of the called dev-kit builds ("<caller job> / <job>"), which
# devkit_check.py checks exist.
BUILD_JOB_PREFIXES = ("devkit-engines / ", "devkit-xclbins / ")
# Jobs of the called dev-kit benches. Their failures do not count (summary's
# Dev-kit checks only warns about them) except a model failing the accuracy
# hard gate, which fails that summary step and so the run.
BENCH_JOB_PREFIX = "devkit-bench / "

BINARY_ROOTS = ("src/lib/", "src/xclbins/")
# What a change on main since the run's base must not touch: the dev-kit
# engine and kernel sources and build inputs, and the binaries themselves
# (main_moved: documentation under them, changes.py is_non_code, does not
# count; anything under BINARY_ROOTS does).
SOURCE_PATHS = ("flm-dev-kit/FLM_DLL/", "flm-dev-kit/FLM_Xclbin/", "src/lib/", "src/xclbins/")
THIRD_PARTY = {k: v for k, v in devkit_check.CATALOG_ALLOW.items() if k.startswith("src/lib/")}


def third_party(path: str) -> Optional[str]:
    """Why a src/lib file is no dev-kit build, or None: THIRD_PARTY, and the
    runtime libraries directly under src/lib/ (curl, protobuf, the MSVC
    runtime: the Windows package ships them; only src/lib/<runtime>/ holds
    engines)."""
    if path in THIRD_PARTY:
        return THIRD_PARTY[path]
    if path.startswith("src/lib/") and "/" not in path[len("src/lib/"):]:
        return "a third-party runtime library the Windows package ships, not a dev-kit build"
    return None

MAX_MANIFEST_BYTES = 1 << 20
MAX_FILE_BYTES = 1 << 30
MAX_ARTIFACT_BYTES = 4 << 30
CHUNK = 1 << 20

SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RUN_ID = re.compile(r"^[1-9][0-9]{0,19}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_+-][A-Za-z0-9_.+-]*$")
ENGINE_FILE = {"linux": re.compile(r"^lib[A-Za-z0-9_+-]+\.so$"),
               "windows": re.compile(r"^[A-Za-z0-9_+-]+\.(dll|lib)$")}
KERNEL_SLOT = re.compile(r"^[A-Za-z0-9_+-][A-Za-z0-9_.+-]*\.(xclbin|switches)$")


class SwapError(Exception):
    """A refusal or a failed check, with the message the user sees."""


# ── Manifests ────────────────────────────────────────────────────────────────
# The one reader of devkit_stage.py's manifest (schema 2, documented
# there). Everything else here works on Manifest and Entry.

class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    kind: str           # engine or kernel
    repo_path: str
    artifact: str
    artifact_path: str
    size: int
    sha256: str
    git_sha1: str
    new: bool           # no file at repo_path in the commit built


@dataclass
class Manifest:
    os: str
    runtime: str
    run_id: str
    head_sha: str
    base_sha: str
    full_refresh: bool
    files: list = field(default_factory=list)


def libs_artifact(os_name: str, runtime: str) -> str:
    return LIBS_ARTIFACT.format(short=SHORT_OS[os_name], runtime=runtime)


def manifest_artifact(os_name: str, runtime: str) -> str:
    return MANIFEST_ARTIFACT.format(os=os_name, runtime=runtime)


def safe_relpath(path: str) -> bool:
    """A plain relative POSIX path: no absolute path, drive, backslash, empty
    part, "." or ".."; every part SAFE_NAME."""
    if not isinstance(path, str) or not path or "\\" in path or path.startswith("/") or ":" in path:
        return False
    return all(part not in (".", "..") and SAFE_NAME.match(part) for part in path.split("/"))


def check_entry_path(kind: str, repo_path: str, os_name: str, runtime: str) -> Optional[str]:
    """Why repo_path is no file a manifest of this os and runtime may name; None when it is."""
    if not safe_relpath(repo_path):
        return f"{repo_path!r} is no plain relative path"
    parts = repo_path.split("/")
    if kind == "engine":
        if len(parts) != 4 or parts[:2] != ["src", "lib"] or parts[2] != runtime:
            return f"{repo_path}: an engine file must be src/lib/{runtime}/<file>"
        if not ENGINE_FILE[os_name].match(parts[3]):
            return f"{repo_path}: not a {os_name} engine library name"
        if third_party(repo_path):
            return f"{repo_path}: a third-party file, no dev-kit build"
        return None
    if kind == "kernel":
        if len(parts) != 4 or parts[:2] != ["src", "xclbins"] or not KERNEL_SLOT.match(parts[-1]):
            return f"{repo_path}: a kernel must be src/xclbins/<folder>/<slot>.xclbin (or .switches)"
        return None
    return f"unknown kind {kind!r}"


def parse_manifest(data: bytes, os_name: str, runtime: str) -> Manifest:
    """The staging step's manifest for os_name and runtime, validated field
    by field; ManifestError on anything unexpected."""
    if len(data) > MAX_MANIFEST_BYTES:
        raise ManifestError("too large")
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ManifestError(f"not JSON: {exc}") from None
    if not isinstance(raw, dict) or raw.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"not a schema {MANIFEST_SCHEMA} manifest")
    if raw.get("os") != os_name or raw.get("runtime") != runtime:
        raise ManifestError(f"says {raw.get('os')!r}/{raw.get('runtime')!r}, its artifact {os_name}/{runtime}")
    run_id, head, base = raw.get("run_id"), raw.get("head_sha"), raw.get("base_sha")
    if not (isinstance(run_id, str) and RUN_ID.match(run_id)):
        raise ManifestError("bad run_id")
    if not (isinstance(head, str) and SHA1.match(head)) or not (isinstance(base, str) and SHA1.match(base)):
        raise ManifestError("bad head_sha or base_sha")
    if not isinstance(raw.get("full_refresh"), bool):
        raise ManifestError("bad full_refresh")
    files = raw.get("files")
    if not isinstance(files, list):
        raise ManifestError("files is not a list")
    out = Manifest(os_name, runtime, run_id, head, base, raw["full_refresh"])
    seen = set()
    for f in files:
        if not isinstance(f, dict):
            raise ManifestError("a files entry is not an object")
        kind, path = f.get("kind"), f.get("repo_path")
        why = check_entry_path(kind, path, os_name, runtime)
        if why:
            raise ManifestError(why)
        if path in seen:
            raise ManifestError(f"{path} listed twice")
        seen.add(path)
        want = libs_artifact(os_name, runtime) if kind == "engine" else XCLBINS_ARTIFACT
        if f.get("artifact") != want:
            raise ManifestError(f"{path}: artifact {f.get('artifact')!r}, expected {want}")
        # The artifact's own layout: the libraries flat, the kernels
        # <folder>/<slot> as under src/xclbins.
        prefix = "src/lib/" + runtime + "/" if kind == "engine" else "src/xclbins/"
        if f.get("artifact_path") != path[len(prefix):]:
            raise ManifestError(f"{path}: artifact_path {f.get('artifact_path')!r} does not match")
        size = f.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_FILE_BYTES:
            raise ManifestError(f"{path}: bad size")
        if not (isinstance(f.get("sha256"), str) and SHA256.match(f["sha256"])):
            raise ManifestError(f"{path}: bad sha256")
        if not (isinstance(f.get("git_sha1"), str) and SHA1.match(f["git_sha1"])):
            raise ManifestError(f"{path}: bad git_sha1")
        if not isinstance(f.get("new"), bool):
            raise ManifestError(f"{path}: bad new")
        out.files.append(Entry(kind, path, f["artifact"], f["artifact_path"], size, f["sha256"],
                               f["git_sha1"], f["new"]))
    # A Windows engine is staged as its DLL and its import library together:
    # the build links the one and the package ships the other.
    for e in out.files:
        name = e.repo_path.rsplit("/", 1)[1]
        for mine, other in ((".lib", ".dll"), (".dll", ".lib")):
            if e.repo_path.endswith(mine) and e.repo_path[:-4] + other not in seen:
                raise ManifestError(f"{e.repo_path}: no {name[:-4]}{other} beside it")
    return out


def plan_swap(manifests: dict, run_id: str, head_sha: str) -> tuple[dict, list[str], list[str]]:
    """(repo path -> Entry of the staged files, notes, errors) from a run's
    manifests ((os, runtime) -> Manifest). Whether the test jobs that
    tested them passed is check_recorded_run's."""
    errors, notes = [], []
    entries: dict[str, Entry] = {}
    bases = {m.base_sha for m in manifests.values()}
    if len(bases) > 1:
        errors.append(f"the manifests name different bases: {', '.join(sorted(bases))}")
    if len({m.full_refresh for m in manifests.values()}) > 1:
        errors.append("some manifests are of a full refresh and some are not")
    stems: dict[str, dict[str, set]] = {}
    for (os_name, runtime), m in sorted(manifests.items()):
        label = f"{os_name} {runtime}"
        if m.run_id != run_id or m.head_sha != head_sha:
            errors.append(f"the {label} manifest is of run {m.run_id} at {m.head_sha[:10]}, "
                          f"not run {run_id} at {head_sha[:10]}")
            continue
        stems.setdefault(runtime, {})[os_name] = {
            re.sub(r"^lib|\.(so|dll|lib)$", "", e.repo_path.rsplit("/", 1)[1])
            for e in m.files if e.kind == "engine"}
        for e in m.files:
            other = entries.get(e.repo_path)
            if other is not None and other.sha256 != e.sha256:
                errors.append(f"{e.repo_path}: the manifests disagree on its bytes")
            entries.setdefault(e.repo_path, e)
    # An engine ships for both systems (src/lib/<runtime>/lib<x>.so beside
    # <x>.dll and <x>.lib): swapping one system's build alone would ship two
    # builds of different sources.
    for runtime, by_os in sorted(stems.items()):
        linux, windows = by_os.get("linux", set()), by_os.get("windows", set())
        if linux != windows:
            diff = sorted(linux ^ windows)
            errors.append(f"the Linux and Windows {runtime} builds did not both stage the engines "
                          f"{', '.join(diff)}: the swap would leave {runtime} half updated")
    return entries, notes, errors


# ── GitHub ───────────────────────────────────────────────────────────────────

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Api:
    """The GitHub REST calls this script makes, and nothing else. The tests
    replace it with a fake of the same methods."""

    def __init__(self, repo: str, token: str, base: str = "https://api.github.com"):
        self.repo, self.token, self.base = repo, token, base

    def _request(self, path: str, params: Optional[dict] = None):
        url = f"{self.base}/repos/{self.repo}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json", "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "devkit-swap"})
        return req

    def _get(self, path: str, params: Optional[dict] = None):
        # No redirect is followed: urllib would send the token to wherever
        # it points (a moved repository answers 301).
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(self._request(path, params), timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            moved = " (a redirect, not followed: is --repo the repository's current name?)"                 if 300 <= exc.code < 400 else ""
            raise SwapError(f"GitHub API {path}: HTTP {exc.code}{moved}") from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise SwapError(f"GitHub API {path}: {exc}") from None

    def _paged(self, path: str, key: str, params: Optional[dict] = None, limit: int = 1000) -> list:
        out, page = [], 1
        while len(out) < limit:
            data = self._get(path, {**(params or {}), "per_page": 100, "page": page})
            items = data.get(key, []) if isinstance(data, dict) else data
            if not isinstance(items, list):
                raise SwapError(f"GitHub API {path}: unexpected answer")
            out += items
            if len(items) < 100:
                break
            page += 1
        return out

    def pull(self, number: int) -> dict:
        return self._get(f"pulls/{number}")

    def pulls_for_branch(self, owner: str, branch: str) -> list:
        return self._paged("pulls", "", {"head": f"{owner}:{branch}", "state": "open"})

    def run(self, run_id: str) -> dict:
        return self._get(f"actions/runs/{run_id}")

    def runs_for_sha(self, sha: str) -> list:
        return self._paged("actions/workflows/build-and-test.yml/runs", "workflow_runs",
                           {"head_sha": sha}, limit=200)

    def jobs(self, run_id: str) -> list:
        return self._paged(f"actions/runs/{run_id}/jobs", "jobs", {"filter": "latest"})

    def artifacts(self, run_id: str) -> list:
        return self._paged(f"actions/runs/{run_id}/artifacts", "artifacts")

    def download_artifact(self, artifact_id: int, dest: Path, limit: int) -> None:
        """The artifact's zip to dest, at most limit bytes. GitHub answers
        with a redirect to its storage, followed without the token."""
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            opener.open(self._request(f"actions/artifacts/{int(artifact_id)}/zip"), timeout=60)
            raise SwapError(f"artifact {artifact_id}: no redirect to its download")
        except urllib.error.HTTPError as exc:
            if exc.code not in (301, 302, 303, 307, 308):
                raise SwapError(f"artifact {artifact_id}: HTTP {exc.code} (expired? re-run the workflow)") from None
            location = exc.headers.get("Location", "")
        if not location.startswith("https://"):
            raise SwapError(f"artifact {artifact_id}: unexpected download location")
        req = urllib.request.Request(location, headers={"User-Agent": "devkit-swap"})
        got = 0
        try:
            with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as out:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    got += len(chunk)
                    if got > limit:
                        raise SwapError(f"artifact {artifact_id}: larger than {limit} bytes")
                    out.write(chunk)
        except (urllib.error.URLError, OSError) as exc:
            raise SwapError(f"artifact {artifact_id}: download failed: {exc}") from None


def find_token() -> str:
    """$GH_TOKEN, $GITHUB_TOKEN, else the gh CLI's login."""
    for var in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        res = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        res = None
    if res is not None and res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip()
    raise SwapError("no GitHub access: log in with `gh auth login`, or set GH_TOKEN (read access is enough)")


# ── Git ──────────────────────────────────────────────────────────────────────

class Git:
    def __init__(self, repo: Path):
        self.repo = repo

    def run(self, *args: str, binary: bool = False, check: bool = True):
        res = subprocess.run(["git", "-c", "core.quotePath=false", "-C", str(self.repo), *args],
                             capture_output=True)
        if check and res.returncode != 0:
            raise SwapError(f"git {' '.join(args[:3])}: {res.stderr.decode('utf-8', 'replace').strip()}")
        if not check and res.returncode != 0:
            return None
        return res.stdout if binary else res.stdout.decode("utf-8", "replace")

    def rev(self, rev: str) -> Optional[str]:
        out = self.run("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", check=False)
        return out.strip() if out else None

    def parents(self, commit: str) -> list[str]:
        return self.run("rev-list", "--parents", "-n", "1", commit).split()[1:]

    def trailers(self, commit: str) -> dict[str, list[str]]:
        out = self.run("show", "-s", "--format=%(trailers:only,unfold)", commit)
        found: dict[str, list[str]] = {}
        for line in out.splitlines():
            key, sep, value = line.partition(":")
            if sep:
                found.setdefault(key.strip(), []).append(value.strip())
        return found

    def changes(self, a: str, b: str, paths: Iterable[str] = ()) -> list[tuple[str, str]]:
        """(status letter, path) of a..b, renames as delete and add."""
        out = self.run("diff", "--name-status", "--no-renames", "-z", a, b, "--", *paths)
        parts = [p for p in out.split("\0") if p]
        return [(parts[i][0], parts[i + 1]) for i in range(0, len(parts) - 1, 2)]

    def is_ancestor(self, a: str, b: str) -> bool:
        return self.run("merge-base", "--is-ancestor", a, b, check=False) is not None

    def blob_ids(self, commit: str, paths: Iterable[str]) -> dict[str, str]:
        """path -> blob id at commit, for the files under paths (trees only:
        no blob is read, so a blob:none clone fetches nothing)."""
        out = self.run("ls-tree", "-r", "-z", commit, "--", *paths)
        ids = {}
        for item in out.split("\0"):
            if item:
                meta, _, path = item.partition("\t")
                parts = meta.split()
                if parts[1] == "blob":
                    ids[path] = parts[2]
        return ids

    def sha256s(self, commit: str, paths: Iterable[str]) -> dict[str, Optional[str]]:
        """path -> SHA-256 of its blob at commit (None: no such file), read
        one at a time through one git cat-file --batch, hashed as it
        streams."""
        out: dict[str, Optional[str]] = {}
        proc = subprocess.Popen(["git", "-C", str(self.repo), "cat-file", "--batch"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            for path in paths:
                proc.stdin.write(f"{commit}:{path}\n".encode("utf-8"))
                proc.stdin.flush()
                header = proc.stdout.readline().decode("utf-8", "replace").split()
                if len(header) != 3 or header[1] != "blob":
                    out[path] = None
                    continue
                left, h = int(header[2]), hashlib.sha256()
                while left:
                    chunk = proc.stdout.read(min(left, CHUNK))
                    if not chunk:
                        raise SwapError(f"git cat-file: {path} at {commit[:10]} cut short")
                    h.update(chunk)
                    left -= len(chunk)
                proc.stdout.read(1)  # the newline after the content
                out[path] = h.hexdigest()
        finally:
            proc.stdin.close()
            proc.stdout.close()
            proc.wait()
        return out


def trailer_values(git: Git, commit: str) -> Optional[tuple[str, str]]:
    """(run id, tested sha) of a swap commit; None when the commit has neither
    trailer; SwapError when they are there but malformed."""
    t = git.trailers(commit)
    run, tested = t.get(TRAILER_RUN, []), t.get(TRAILER_TESTED, [])
    if not run and not tested:
        return None
    if len(run) != 1 or len(tested) != 1 or not RUN_ID.match(run[0]) or not SHA1.match(tested[0]):
        raise SwapError(f"{commit[:10]}: needs exactly one {TRAILER_RUN}: <run id> and one "
                        f"{TRAILER_TESTED}: <40-hex sha> trailer")
    return run[0], tested[0]


# ── The run ──────────────────────────────────────────────────────────────────

@dataclass
class Pr:
    number: int
    head_ref: str
    head_sha: str
    head_repo: str


def check_run(run: dict, repo: str, sha: str, pr: Pr) -> list[str]:
    """Why run is not this repository's completed Build and Test of sha for
    pr (a pull_request run, or a workflow_dispatch on its branch, whose
    manifests must then be a full refresh's); empty when it is."""
    why = []
    full = lambda key: ((run.get(key) or {}).get("full_name") or "")  # noqa: E731
    if run.get("path") != WORKFLOW_PATH:
        why.append(f"run {run.get('id')} is not of {WORKFLOW_PATH}")
    if full("repository") != repo or full("head_repository") != repo:
        why.append(f"run {run.get('id')} is not of {repo} on a branch of {repo}")
    if run.get("head_sha") != sha:
        why.append(f"run {run.get('id')} tested {str(run.get('head_sha'))[:10]}, not {sha[:10]}")
    if run.get("status") != "completed":
        why.append(f"run {run.get('id')} has not completed ({run.get('status')}): wait for it")
    event = run.get("event")
    if event not in ("pull_request", "workflow_dispatch"):
        why.append(f"run {run.get('id')} is a {event} run, not the pull request's or a full refresh's")
    elif run.get("head_branch") != pr.head_ref:
        why.append(f"run {run.get('id')} is of branch {run.get('head_branch')}, not {pr.head_ref}")
    elif event == "pull_request":
        numbers = [p.get("number") for p in run.get("pull_requests") or [] if isinstance(p, dict)]
        if numbers and pr.number not in numbers:
            why.append(f"run {run.get('id')} is of pull request(s) {numbers}, not #{pr.number}")
    return why


def summary_failures(job: dict) -> list[str]:
    """The summary job's failed steps other than PENDING_STEP (its result
    when its steps are not listed)."""
    steps = [st for st in job.get("steps") or [] if isinstance(st, dict)]
    if not steps:
        return [f"its result {job.get('conclusion')}"]
    return [str(st.get("name")) for st in steps
            if st.get("conclusion") not in ("success", "skipped", None) and st.get("name") != PENDING_STEP]


def check_jobs(jobs: list) -> tuple[list[str], dict]:
    """(why the run's jobs do not allow a swap, (os, runtime) -> the
    conclusion of its product test job). Every job must have passed or been
    skipped, one by one, except the summary job (the required check), which
    may have failed at PENDING_STEP alone: that is the swap fetch makes, and
    the bench jobs (BENCH_JOB_PREFIX), which summary only warns about unless
    a model failed the accuracy hard gate (then its Dev-kit checks step
    failed, which counts). The staging job must have passed."""
    why = []
    by_name = {j.get("name"): j for j in jobs if isinstance(j, dict)}
    summary = by_name.get(SUMMARY_JOB)
    if summary is None:
        why.append(f"no {SUMMARY_JOB} job")
    elif summary.get("conclusion") != "success":
        other = summary_failures(summary)
        if summary.get("conclusion") != "failure" or other:
            why.append(f"its {SUMMARY_JOB} job did not pass ({summary.get('conclusion')}): "
                       f"{', '.join(other[:5]) or 'see the run'}")
    bad = sorted(f"{j.get('name')} ({j.get('conclusion')})" for j in jobs
                 if isinstance(j, dict) and j.get("name") != SUMMARY_JOB
                 and not str(j.get("name")).startswith(BENCH_JOB_PREFIX)
                 and j.get("conclusion") not in ("success", "skipped", "neutral"))
    if bad:
        why.append(f"{len(bad)} job(s) did not pass: {', '.join(bad[:5])}{', ...' if len(bad) > 5 else ''}: "
                   "fix them or re-run the failed jobs, then fetch")
    stage = by_name.get(STAGE_JOB)
    if stage is None or stage.get("conclusion") != "success":
        why.append(f"its {STAGE_JOB} job did not run or pass "
                   f"({stage.get('conclusion') if stage else 'not found'}): it staged nothing to swap")
    tests = {key: (by_name.get(name) or {}).get("conclusion") or "not run" for key, name in TEST_JOBS.items()}
    return why, tests


def read_zip_member(zip_path: Path, member: str, size: Optional[int], limit: int = MAX_FILE_BYTES) -> bytes:
    """One regular file of an untrusted zip, after checking every entry name:
    exactly size bytes, or (size None) whatever it holds, up to limit."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            infos = zf.infolist()
            for info in infos:
                # orig_filename: the name as stored (filename has a
                # backslash turned into a slash on Windows).
                raw = info.orig_filename
                name = raw.rstrip("/") if info.is_dir() else raw
                if not safe_relpath(name):
                    raise SwapError(f"{zip_path.name}: unsafe entry name {raw!r}")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise SwapError(f"{zip_path.name}: {raw} is a symbolic link")
            matches = [i for i in infos if i.orig_filename == member and not i.is_dir()]
            if len(matches) != 1:
                raise SwapError(f"{zip_path.name}: {member} is missing or listed twice")
            info = matches[0]
            if size is None:
                size = info.file_size
                if size > limit:
                    raise SwapError(f"{zip_path.name}: {member} is larger than {limit} bytes")
            if info.file_size != size:
                raise SwapError(f"{zip_path.name}: {member} is {info.file_size} bytes, the manifest says {size}")
            with zf.open(info) as fh:
                data = fh.read(size + 1)
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        raise SwapError(f"{zip_path.name}: unreadable: {exc}") from None
    if len(data) != size:
        raise SwapError(f"{zip_path.name}: {member} is not {size} bytes")
    return data


def usable(artifact: Optional[dict], limit: int) -> bool:
    return (isinstance(artifact, dict) and not artifact.get("expired")
            and isinstance(artifact.get("size_in_bytes"), int) and artifact["size_in_bytes"] <= limit)


def load_manifests(api, run_id: str, artifacts: list, tmp: Path) -> tuple[dict, list[str]]:
    """((os, runtime) -> the staging step's Manifest for each, errors)."""
    index = {a.get("name"): a for a in artifacts if isinstance(a, dict)}
    out, errors = {}, []
    for os_name, runtime in sorted(TEST_JOBS):
        name = manifest_artifact(os_name, runtime)
        art = index.get(name)
        if not usable(art, MAX_MANIFEST_BYTES):
            errors.append(f"{name} is missing, expired or too large: re-run the workflow "
                          "(manifests are kept 30 days)")
            continue
        dest = tmp / f"{name}.zip"
        try:
            api.download_artifact(art["id"], dest, MAX_MANIFEST_BYTES)
            with zipfile.ZipFile(dest) as zf:
                names = [i.orig_filename for i in zf.infolist()]
                if names != [MANIFEST_FILE]:
                    raise SwapError(f"{name}: holds {names[:5]}, not just {MANIFEST_FILE}")
            data = read_zip_member(dest, MANIFEST_FILE, None, MAX_MANIFEST_BYTES)
            out[(os_name, runtime)] = parse_manifest(data, os_name, runtime)
        except (SwapError, ManifestError, zipfile.BadZipFile, OSError) as exc:
            errors.append(f"{name}: {exc}")
    return out, errors


@dataclass
class RunCheck:
    run: dict
    manifests: dict
    entries: dict
    notes: list
    errors: list


def check_recorded_run(api, repo: str, run_id: str, tested: str, pr: Pr, tmp: Path) -> RunCheck:
    """Everything fetch and verify ask of the run: check_run, check_jobs,
    its manifests, that the product test job of every system and runtime
    with staged files passed, and the swap they plan."""
    run = api.run(run_id)
    errors = check_run(run, repo, tested, pr)
    if errors:
        return RunCheck(run, {}, {}, [], errors)
    why, tests = check_jobs(api.jobs(run_id))
    errors += why
    if why and any(w.startswith(f"its {STAGE_JOB} job") for w in why):
        return RunCheck(run, {}, {}, [], errors)
    manifests, more = load_manifests(api, run_id, api.artifacts(run_id), tmp)
    errors += more
    for key, m in sorted(manifests.items()):
        if m.files and tests[key] != "success":
            errors.append(f"the {key[0]} {key[1]} build staged {len(m.files)} file(s), but {TEST_JOBS[key]} "
                          f"did not pass ({tests[key]}): they are untested")
    if manifests and not any(m.files for m in manifests.values()):
        errors.append(f"run {run_id} staged no dev-kit file: nothing to swap")
    entries, notes, more = plan_swap(manifests, run_id, tested)
    errors += more
    full = {m.full_refresh for m in manifests.values()}
    if run.get("event") == "workflow_dispatch":
        if full != {True}:
            errors.append(f"run {run_id} is a dispatch but not a dev-kit full refresh")
        elif any(v != "success" for v in tests.values()):
            errors.append(f"run {run_id} is a dev-kit full refresh, but not all four product test jobs passed")
    if run.get("event") == "pull_request" and True in full:
        errors.append(f"run {run_id} is a pull request run with full-refresh manifests")
    return RunCheck(run, manifests, entries, notes, errors)


def counts_for_swap(path: str) -> bool:
    """Whether a change to path (under SOURCE_PATHS) on main makes a run's
    swap stale: every binary does, and every source or build input but
    documentation (changes.py is_non_code, so that the two agree)."""
    return path.startswith(BINARY_ROOTS) or not changes.is_non_code(path)


def main_moved(git: Git, since: str, until: str) -> list[str]:
    """Commits on until since since that change a SOURCE_PATHS file that
    counts_for_swap (a commit changing only documentation there does not
    count; a SwapError when since is unknown here, or is no commit of
    until's history: a run's base is a commit of main). Trees only (no
    blob is read): a blob:none clone fetches nothing."""
    if not git.rev(since):
        raise SwapError(f"the run's base {since[:10]} is not in this clone: fetch main")
    if not git.is_ancestor(since, until):
        raise SwapError(f"the run's base {since[:10]} is not a commit of {until}")
    moved = []
    for commit in git.run("rev-list", f"{since}..{until}", "--", *SOURCE_PATHS).split():
        # -c: a merge lists the files it changed against every parent (what
        # a side brought in is listed by the side's own commits); --root:
        # a root commit lists its files.
        out = git.run("diff-tree", "-r", "-c", "--root", "--no-commit-id", "--name-only", "-z",
                      commit, "--", *SOURCE_PATHS)
        if any(counts_for_swap(p) for p in out.split("\0") if p):
            moved.append(commit)
    return moved


# ── verify ───────────────────────────────────────────────────────────────────

# What stages the dev-kit files into the builds, writes, reads or acts on
# the manifests a swap is checked against, builds the packages the staged
# files are tested in, and the NPU test harness that decides whether they
# pass (the product test jobs run the pull request's own: npu_test, the
# npu-test and install-package actions). A pull request that changes any of
# them gets no verified swap: its own code would vouch for it (the checks
# run from the pull request's merge commit). changes.py is not here: a new
# model family needs its engine table (ENGINES) and comes with its swap;
# what it decides runs from the pull request's code whatever this says.
# npu_test's pr-tier table (plan.py PR_CHAT) needs no entry for a new
# family (it gets its smallest model), so a new model needs no harness edit.
SWAP_CI_PATHS = (
    ".github/script/ci/devkit_stage.py", ".github/script/ci/devkit_overlay.py",
    ".github/script/ci/devkit_swap.py", ".github/script/ci/devkit_check.py",
    ".github/actions/devkit-stage-apply/", ".github/actions/detect-changes/",
    ".github/actions/lin-build/", ".github/actions/win-build/",
    ".github/workflows/build-and-test.yml", ".github/workflows/lin-build.yml",
    ".github/workflows/win-build.yml", ".github/workflows/devkit-engines.yml",
    ".github/workflows/devkit-xclbins.yml",
    ".github/script/npu_test/", ".github/actions/npu-test/", ".github/actions/install-package/",
)


def swap_ci_changed(git: Git, base: str, head: str) -> list[str]:
    """The SWAP_CI_PATHS files head changes against its merge base with base."""
    merge_base = git.run("merge-base", base, head).strip()
    return sorted({p for _, p in git.changes(merge_base, head, SWAP_CI_PATHS)})


def verify_swap(api, git: Git, repo: str, commit: str, pr: Pr, tmp: Path,
                base: str = "") -> tuple[bool, list[str], dict]:
    """(verified, reasons, repo path -> Entry it swapped) for one commit."""
    try:
        values = trailer_values(git, commit)
    except SwapError as exc:
        return False, [str(exc)], {}
    if values is None:
        return False, [f"{commit[:10]} is no swap commit (no {TRAILER_RUN} trailer)"], {}
    run_id, tested = values
    reasons = []
    parents = git.parents(commit)
    if parents != [tested]:
        reasons.append(f"{commit[:10]}'s parent is {', '.join(p[:10] for p in parents) or 'none'}, "
                       f"not the tested {tested[:10]}")
        return False, reasons, {}
    try:
        rc = check_recorded_run(api, repo, run_id, tested, pr, tmp)
    except SwapError as exc:
        return False, [f"run {run_id}: {exc}"], {}
    reasons += rc.errors
    changed = git.changes(tested, commit)
    if not changed:
        reasons.append(f"{commit[:10]} changes nothing")
    candidates = []
    for status, path in changed:
        entry = rc.entries.get(path)
        if entry is None:
            reasons.append(f"{commit[:10]} changes {path}, which run {run_id}'s manifests do not list as tested")
        elif status not in ("A", "M"):
            reasons.append(f"{commit[:10]} {'deletes' if status == 'D' else 'changes the mode of'} {path}")
        else:
            candidates.append(path)
    if reasons:
        return False, reasons, {}
    # The bytes, by their git blob ids (a hash of exactly the bytes, which
    # the manifest records beside the SHA-256): read from the trees alone,
    # so the changes job's blob:none clone downloads no binary.
    ids = git.blob_ids(commit, BINARY_ROOTS)
    for path in candidates:
        if ids.get(path) != rc.entries[path].git_sha1:
            reasons.append(f"{path} at {commit[:10]} is not the bytes run {run_id} tested")
    # All of them: a commit carrying part of what the run tested (one
    # system's build of an engine, some of the kernels) is no swap of it.
    missing = sorted(p for p, e in rc.entries.items() if p not in candidates and ids.get(p) != e.git_sha1)
    if missing:
        reasons.append(f"run {run_id} tested {len(missing)} file(s) {commit[:10]} does not carry "
                       f"(e.g. {missing[0]}): run devkit_swap.py fetch for the whole swap")
    swapped = {} if reasons else {p: rc.entries[p] for p in candidates}
    if base and rc.manifests and not reasons:
        try:
            moved = main_moved(git, next(iter(rc.manifests.values())).base_sha, base)
        except SwapError as exc:
            moved, reasons = [], reasons + [str(exc)]
        if moved:
            reasons.append(f"main gained {len(moved)} commit(s) touching the dev-kit sources or binaries "
                           f"since run {run_id}'s base (e.g. {moved[0][:10]}): merge main and let CI test again")
    return not reasons, reasons, swapped


def cmd_verify(args, api, git: Git) -> int:
    pr = Pr(args.pr, args.head_ref, args.head, args.head_repo)
    ci = swap_ci_changed(git, args.base, args.head) if args.head_repo == args.repo else []
    if args.head_repo != args.repo:
        ok, reasons = False, ["the pull request is from a fork: no swap to trust"]
    elif ci and trailer_values(git, args.head) is not None:
        ok, reasons = False, [f"the pull request also changes the dev-kit CI that checks a swap ({', '.join(ci[:3])}"
                              f"{', ...' if len(ci) > 3 else ''}): keep swaps and CI changes in separate pull "
                              "requests"]
    else:
        with tempfile.TemporaryDirectory() as tmp:
            ok, reasons, swapped = verify_swap(api, git, args.repo, args.head, pr, Path(tmp), args.base)
        if ok:
            reasons = [f"{len(swapped)} file(s) are exactly the bytes run "
                       f"{trailer_values(git, args.head)[0]} built and tested"]
    for r in reasons:
        print(f"devkit_swap: {r}")
    print(f"swap-verified={str(ok).lower()}")
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            fh.write(f"swap-verified={str(ok).lower()}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and (ok or not reasons[0].endswith("(no Devkit-Swap-Run trailer)")):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### Dev-kit swap: {'verified' if ok else 'not verified'}\n\n"
                     + "".join(f"- {r}\n" for r in reasons)
                     + ("\nNo product build, NPU test, dev-kit build or staging runs for it.\n" if ok
                        else "\nBuild and Test runs as for any other commit.\n"))
    return 0


# ── check ────────────────────────────────────────────────────────────────────

def cmd_check(args, api, git: Git) -> int:
    merge_base = git.run("merge-base", args.base, args.head).strip()
    changed = git.changes(merge_base, args.head, BINARY_ROOTS)
    if not changed:
        print("devkit_swap: check: no prebuilt binary changed")
        return 0
    pr = Pr(args.pr, args.head_ref, args.head, args.head_repo)
    accepted: dict[str, set] = {}
    notes = []
    commits = git.run("rev-list", f"{merge_base}..{args.head}").split()
    ci = swap_ci_changed(git, args.base, args.head)
    with tempfile.TemporaryDirectory() as tmp:
        for commit in commits:
            try:
                is_swap = trailer_values(git, commit) is not None
            except SwapError as exc:
                notes.append(str(exc))
                continue
            if not is_swap:
                continue
            if args.head_repo != args.repo:
                notes.append(f"{commit[:10]}: a swap commit on a fork's branch is not trusted")
                continue
            if ci:
                notes.append(f"{commit[:10]}: not trusted, the pull request also changes the dev-kit CI "
                             f"that checks swaps ({', '.join(ci[:3])})")
                continue
            ok, reasons, swapped = verify_swap(api, git, args.repo, commit, pr, Path(tmp))
            if ok:
                for path, entry in swapped.items():
                    accepted.setdefault(path, set()).add(entry.git_sha1)
                notes.append(f"swap commit {commit[:10]}: {len(swapped)} file(s) verified")
            else:
                notes += [f"swap commit {commit[:10]} does not verify: {r}" for r in reasons]
    required = devkit_check.required_binaries(Path(args.repo_root))
    # Blob ids from the trees alone (see verify_swap): no binary is downloaded.
    ids = git.blob_ids(args.head, BINARY_ROOTS) if accepted else {}
    hand = []
    for status, path in changed:
        if third_party(path):
            notes.append(f"{path}: third-party ({third_party(path)})")
        elif status == "D":
            if path in required:
                hand.append(f"{path} is deleted, but the dev kit still assigns or builds it")
        elif ids.get(path) in accepted.get(path, ()):
            continue
        else:
            hand.append(f"{path} is not the bytes a verified swap commit of this pull request brought in")
    for n in notes:
        print(f"devkit_swap: check: {n}")
    if not hand:
        print(f"devkit_swap: check: {len(changed)} changed prebuilt file(s), all from verified swaps, "
              "allowed deletions or third-party")
        return 0
    level = "error" if SWAP_ENFORCE else "warning"
    for h in hand:
        print(f"::{level}::{h}")
    print(f"devkit_swap: check: {len(hand)} prebuilt file(s) edited by hand. src/lib and src/xclbins are "
          "never edited by hand: drop the edit, change the dev-kit source (flm-dev-kit/) instead, push, let "
          "Build and Test build, stage and NPU-test it, then run `python .github/script/ci/devkit_swap.py "
          "fetch` and push the swap commit it makes (flm-dev-kit/AGENTS.md, \"Prebuilt binaries\").")
    return 1 if SWAP_ENFORCE else 0


# ── fetch ────────────────────────────────────────────────────────────────────

def repo_from_remote(git: Git, remote: str) -> str:
    url = (git.run("remote", "get-url", remote, check=False) or "").strip()
    m = re.search(r"github\.com[:/]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$", url)
    return m.group(1) if m else DEFAULT_REPO


def find_pr(api, git: Git, repo: str, number: Optional[int]) -> dict:
    if number is not None:
        return api.pull(number)
    branch = (git.run("symbolic-ref", "--quiet", "--short", "HEAD", check=False) or "").strip()
    if not branch:
        raise SwapError("HEAD is detached: check out the pull request's branch, or name it with --pr")
    pulls = api.pulls_for_branch(repo.split("/")[0], branch)
    if len(pulls) != 1:
        raise SwapError(f"{len(pulls)} open pull requests of {repo} have head {branch}: name one with --pr")
    return pulls[0]


def find_run(api, sha: str, run_id: Optional[str]) -> str:
    if run_id:
        if not RUN_ID.match(run_id):
            raise SwapError(f"--run {run_id!r} is not a run id")
        return run_id
    runs = [r for r in api.runs_for_sha(sha) if isinstance(r, dict) and r.get("path") == WORKFLOW_PATH
            and r.get("event") in ("pull_request", "workflow_dispatch")]
    if not runs:
        raise SwapError(f"no Build and Test run for {sha[:10]} yet: push, and wait for it")
    runs.sort(key=lambda r: (str(r.get("created_at")), int(r.get("id") or 0)), reverse=True)
    latest = runs[0]
    if latest.get("status") != "completed":
        raise SwapError(f"Build and Test run {latest.get('id')} for {sha[:10]} is still {latest.get('status')}: "
                        "wait for it")
    return str(latest.get("id"))


def safe_target(repo: Path, rel: str) -> Path:
    """repo/rel, refusing a path through a symbolic link or out of the tree."""
    target = repo.joinpath(*rel.split("/"))
    p = repo
    for part in rel.split("/"):
        p = p / part
        if p.is_symlink():
            raise SwapError(f"{rel}: {p.relative_to(repo).as_posix()} is a symbolic link")
    if os.path.commonpath([repo.resolve(), target.resolve()]) != str(repo.resolve()):
        raise SwapError(f"{rel} is outside the repository")
    return target


def file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def commit_message(run_id: str, tested: str, entries: list) -> str:
    engines = sum(1 for e in entries if e.kind == "engine")
    lines = [f"devkit: swap in the binaries Build and Test run {run_id} tested", "",
             f"The dev-kit engine libraries and kernels that run {run_id} built from",
             f"{tested[:12]}, staged into its product builds and tested on the NPU",
             f"(devkit_swap.py fetch): {engines} engine file(s), {len(entries) - engines} kernel(s).",
             "CI checks these bytes against the run's staging manifests.", ""]
    lines += [f"  {e.repo_path}" for e in entries[:60]]
    if len(entries) > 60:
        lines.append(f"  ... and {len(entries) - 60} more")
    lines += ["", f"{TRAILER_RUN}: {run_id}", f"{TRAILER_TESTED}: {tested}", ""]
    return "\n".join(lines)


def cmd_fetch(args, api, git: Git) -> int:
    repo_root = Path(git.run("rev-parse", "--show-toplevel").strip())
    git = Git(repo_root)
    if git.run("status", "--porcelain", "--untracked-files=no").strip():
        raise SwapError("the working tree has uncommitted changes: commit or stash them first")
    if git.run("status", "--porcelain", "--untracked-files=all", "--", *BINARY_ROOTS).strip():
        raise SwapError("there are untracked files under src/lib or src/xclbins: remove them first")
    head = git.rev("HEAD")
    pull = find_pr(api, git, args.repo, args.pr)
    pr = Pr(int(pull.get("number") or 0), str((pull.get("head") or {}).get("ref") or ""),
            str((pull.get("head") or {}).get("sha") or ""),
            str(((pull.get("head") or {}).get("repo") or {}).get("full_name") or ""))
    if pull.get("state") != "open":
        raise SwapError(f"pull request #{pr.number} is not open")
    if pr.head_repo != args.repo:
        raise SwapError(f"pull request #{pr.number} comes from {pr.head_repo or 'a deleted fork'}: only a branch "
                        f"of {args.repo} can carry a swap")
    if pr.head_sha != head:
        raise SwapError(f"local HEAD {head[:10]} is not pull request #{pr.number}'s head {pr.head_sha[:10]}: "
                        "push your commits (or pull), let Build and Test run on them, then fetch")
    run_id = find_run(api, head, args.run)
    print(f"devkit_swap: pull request #{pr.number} at {head[:10]}, Build and Test run {run_id}")
    with tempfile.TemporaryDirectory(prefix="devkit-swap-") as tmp_name:
        tmp = Path(tmp_name)
        rc = check_recorded_run(api, args.repo, run_id, head, pr, tmp)
        for n in rc.notes:
            print(f"devkit_swap: note: {n}")
        if rc.errors:
            raise SwapError(f"run {run_id} cannot be swapped in:\n  " + "\n  ".join(rc.errors))
        base = next(iter(rc.manifests.values())).base_sha
        if not args.no_git_fetch:
            git.run("fetch", "--quiet", args.remote, "main")
        moved = main_moved(git, base, f"{args.remote}/main")
        if moved:
            raise SwapError(f"main gained {len(moved)} commit(s) touching the dev-kit sources or binaries "
                            f"since the run's base {base[:10]} (e.g. {moved[0][:10]}): merge main, push, "
                            "let Build and Test run again, then fetch")
        todo = sorted((e for e in rc.entries.values()
                       if file_sha256(safe_target(repo_root, e.repo_path)) != e.sha256),
                      key=lambda e: e.repo_path)
        if not todo:
            raise SwapError("nothing to swap: the branch already has every file the run staged")
        index = {a.get("name"): a for a in api.artifacts(run_id) if isinstance(a, dict)}
        needed = sorted({e.artifact for e in todo})
        gone = [n for n in needed if not usable(index.get(n), MAX_ARTIFACT_BYTES)]
        if gone:
            raise SwapError(f"artifacts expired or missing: {', '.join(gone)} (kept 7 days): re-run the "
                            "Build and Test workflow on this commit, then fetch")
        print(f"devkit_swap: {len(todo)} file(s) to swap from {', '.join(needed)}:")
        for e in todo:
            print(f"  {e.repo_path}  {e.size} bytes  sha256 {e.sha256[:16]}")
        if args.dry_run:
            print("devkit_swap: --dry-run: nothing downloaded or written")
            return 0
        staged = tmp / "staged"
        staged.mkdir()
        for name in needed:
            art = index[name]
            dest = tmp / f"artifact-{int(art['id'])}.zip"
            print(f"devkit_swap: downloading {name} ({art['size_in_bytes']} bytes)")
            api.download_artifact(art["id"], dest, MAX_ARTIFACT_BYTES)
            for e in (e for e in todo if e.artifact == name):
                data = read_zip_member(dest, e.artifact_path, e.size)
                if hashlib.sha256(data).hexdigest() != e.sha256:
                    raise SwapError(f"{e.repo_path}: {name}/{e.artifact_path} is not the bytes the manifest "
                                    "records: nothing written")
                (staged / hashlib.sha256(e.repo_path.encode()).hexdigest()).write_bytes(data)
            dest.unlink()
        for e in todo:
            target = safe_target(repo_root, e.repo_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            safe_target(repo_root, e.repo_path)
            part = target.with_name(target.name + ".devkit-swap")
            shutil.copyfile(staged / hashlib.sha256(e.repo_path.encode()).hexdigest(), part)
            os.replace(part, target)
    paths = [e.repo_path for e in todo]
    git.run("add", "--force", "--", *paths)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as fh:
        fh.write(commit_message(run_id, head, todo))
        msg = fh.name
    try:
        git.run("commit", "--quiet", "-F", msg)
    finally:
        os.unlink(msg)
    new = git.rev("HEAD")
    if git.parents(new) != [head] or trailer_values(git, new) != (run_id, head):
        raise SwapError(f"the commit {new[:10]} is not the swap commit expected (a commit hook?): check it, "
                        f"or drop it with `git reset --hard {head[:12]}`")
    # What git stored, as CI will read it (a clean or smudge filter, or a
    # line-ending setting, must not have touched a byte).
    stored = git.sha256s(new, paths)
    bad = [e.repo_path for e in todo if stored.get(e.repo_path) != e.sha256]
    if bad or {p for _, p in git.changes(head, new)} != set(paths):
        raise SwapError(f"the commit {new[:10]} does not hold exactly the tested bytes "
                        f"({', '.join(bad[:3]) or 'other files changed'}): drop it with "
                        f"`git reset --hard {head[:12]}` and check your git filters")
    print(f"devkit_swap: committed {new[:10]} ({len(paths)} file(s)). Next: `git push`. Build and Test "
          "verifies the bytes against run " + run_id + " and skips the NPU tests for that commit.")
    return 0


# ── pending ──────────────────────────────────────────────────────────────────

def pending_text(todo: list[str], run_id: str, head: str, full: bool, ci: list[str]) -> str:
    lines = ["### Dev-kit swap pending", "",
             f"This run built {len(todo)} dev-kit file(s) under src/lib and src/xclbins from this pull "
             f"request's dev-kit source (e.g. `{todo[0]}`), staged them into its product builds and "
             "NPU-tested those packages, but the branch does not have them yet. Without them main would "
             "ship binaries out of step with the source (or, for a new engine, fail to build), so this "
             "pull request cannot merge until they are committed. They are never edited by hand: from a "
             f"clean checkout of the branch at `{head[:12]}`"
             f"{' (the dev-kit full refresh)' if full else ''}, run", "",
             f"```\npython .github/script/ci/devkit_swap.py fetch --run {run_id}\ngit push\n```", "",
             "fetch commits exactly the bytes this run tested. Build and Test then verifies the swap "
             "commit against this run and runs no NPU test again for it (flm-dev-kit/AGENTS.md, "
             "\"Prebuilt binaries\")."]
    if ci:
        lines += ["", f"This pull request also changes the CI that stages and checks swaps "
                      f"({', '.join(f'`{c}`' for c in ci[:3])}{', ...' if len(ci) > 3 else ''}), which no "
                      "verified swap can come with: move that change to a pull request of its own first."]
    return "\n".join(lines) + "\n"


def cmd_pending(args, git: Git) -> int:
    """PENDING_STEP: exit 1, with the fetch command, while the run staged
    files the branch (--tree) does not have. With --expect-staged (the run's
    devkit-stage said it staged), every system's and runtime's manifest must
    be there: a lost download must not read as nothing pending."""
    manifests = {}
    for key in TEST_JOBS:
        path = args.manifests / manifest_artifact(*key) / MANIFEST_FILE
        if not path.is_file() and args.expect_staged:
            print(f"::error::devkit_swap pending: {manifest_artifact(*key)}/{MANIFEST_FILE} is missing, but "
                  f"devkit-stage staged files: re-run the failed jobs")
            return 1
        if path.is_file():
            try:
                manifests[key] = parse_manifest(path.read_bytes(), *key)
            except ManifestError as exc:
                print(f"::error::devkit_swap pending: {path.parent.name}: {exc}")
                return 1
    if not manifests:
        print("devkit_swap: pending: no staging manifest: nothing staged")
        return 0
    entries, _, errors = plan_swap(manifests, args.run, args.head)
    if errors:
        for e in errors:
            print(f"::error::devkit_swap pending: {e}")
        return 1
    # Blob ids from the trees alone: a blob:none checkout reads no blob.
    have = git.blob_ids(args.tree, BINARY_ROOTS)
    todo = sorted(p for p, e in entries.items() if have.get(p) != e.git_sha1)
    if not todo:
        print(f"devkit_swap: pending: the branch has all {len(entries)} staged file(s): no swap pending")
        return 0
    ci = swap_ci_changed(git, args.base, args.head) if args.base else []
    text = pending_text(todo, args.run, args.head, any(m.full_refresh for m in manifests.values()), ci)
    print(text, end="")
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    print(f"::error title={PENDING_STEP}::{len(todo)} staged dev-kit file(s) are not committed: run "
          f"`python .github/script/ci/devkit_swap.py fetch --run {args.run}` on a clean checkout of "
          f"{args.head[:12]} and push; CI then verifies the swap and runs no NPU test again"
          + (". First move the swap CI change to its own pull request." if ci else "."))
    return 1


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None, api=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch", help="commit the binaries a passed Build and Test run tested")
    p.add_argument("--pr", type=int, default=None)
    p.add_argument("--run", default=None)
    p.add_argument("--repo", default=None, help="owner/name (default: the remote's)")
    p.add_argument("--remote", default="origin")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-git-fetch", action="store_true", help="do not fetch main first")
    for name in ("verify", "check"):
        p = sub.add_parser(name)
        p.add_argument("--head", required=True)
        p.add_argument("--base", required=True)
        p.add_argument("--pr", type=int, required=True)
        p.add_argument("--head-ref", required=True)
        p.add_argument("--head-repo", required=True)
        p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO)
        p.add_argument("--repo-root", default=str(REPO_ROOT))
        if name == "verify":
            p.add_argument("--github-output", action="store_true")
    p = sub.add_parser("pending")
    p.add_argument("--manifests", type=Path, required=True)
    p.add_argument("--head", required=True, help="the commit the run built")
    p.add_argument("--run", required=True)
    p.add_argument("--tree", default="", help="the commit whose files to compare (default --head)")
    p.add_argument("--base", default="", help="the pull request's base, to name a swap CI change")
    p.add_argument("--summary", default="", help="also append the Markdown here")
    p.add_argument("--expect-staged", action="store_true",
                   help="the run staged files: fail when any manifest is missing")
    p.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args(argv)
    try:
        if args.cmd == "pending":
            args.tree = args.tree or args.head
            return cmd_pending(args, Git(Path(args.repo_root)))
        if args.cmd == "fetch":
            git = Git(Path.cwd())
            args.repo = args.repo or repo_from_remote(git, args.remote)
            return cmd_fetch(args, api or Api(args.repo, find_token()), git)
        if args.github_output if args.cmd == "verify" else False:
            if not os.environ.get("GITHUB_OUTPUT"):
                parser.error("--github-output needs $GITHUB_OUTPUT")
        git = Git(Path(args.repo_root))
        if api is None:
            api = Api(args.repo, os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "")
        return {"verify": cmd_verify, "check": cmd_check}[args.cmd](args, api, git)
    except Exception as exc:  # noqa: BLE001 -- verify and check never break the changes job
        if not isinstance(exc, SwapError) and args.cmd not in ("verify", "check"):
            raise
        if not isinstance(exc, SwapError):
            exc = f"unexpected {type(exc).__name__}: {exc}"
        print(f"devkit_swap: {args.cmd}: {exc}", file=sys.stderr)
        if args.cmd == "verify":
            print("swap-verified=false")
            if args.github_output:
                with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
                    fh.write("swap-verified=false\n")
            return 0
        if args.cmd == "check" and not SWAP_ENFORCE:
            print(f"::warning::devkit_swap check could not run: {exc}")
            return 0
        return 1


if __name__ == "__main__":
    sys.exit(main())
