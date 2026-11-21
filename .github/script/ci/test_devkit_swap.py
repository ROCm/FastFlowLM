# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for devkit_swap.py (the manifests devkit_stage.py writes are
tested in test_devkit_stage.py):
python3 -m unittest discover -s ci -p 'test_*.py' -t ci

The GitHub API is FakeApi (devkit_swap.Api's methods over in-memory runs,
jobs and artifact zips); git is real, on throwaway repositories with a bare
"origin"."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import devkit_check as dc  # noqa: E402
import devkit_swap as ds  # noqa: E402

REPO = "o/r"
RUN = "100"
PR = 7
XRT_LIN, XRT_WIN, KERNEL = "src/lib/xrt/libfoo_npu.so", "src/lib/xrt/foo_npu.dll", "src/xclbins/M-NPU2/attn.xclbin"
XRT_LIB = "src/lib/xrt/foo_npu.lib"


def rmtree(path: Path) -> None:
    """shutil.rmtree that also removes read-only files (git objects on Windows)."""
    def fix(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=fix)
    else:
        shutil.rmtree(path, onerror=fix)


def blob(tag: str, size: int = 64) -> bytes:
    """Binary-looking bytes (NUL bytes: no line-ending conversion applies)."""
    return (tag.encode() + b"\0\x7f\r\n") * (size // (len(tag) + 4) + 1)


def digests(data: bytes) -> dict:
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "git_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}


def zip_of(files: dict[str, bytes], symlinks: tuple = ()) -> bytes:
    """A zip holding exactly these entry names (ZipInfo would turn a
    backslash into a slash on Windows; an artifact need not)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in list(files.items()) + [(n, b"target") for n in symlinks]:
            info = zipfile.ZipInfo("x")
            info.filename = info.orig_filename = name
            if name in symlinks:
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, data)
    return buf.getvalue()


def manifest(os_name: str, runtime: str, files: dict[str, bytes], *, head: str, base: str,
             run_id: str = RUN, full_refresh: bool = False) -> dict:
    """A schema 2 manifest as devkit_stage.py writes it, for repo path -> bytes."""
    entries = []
    for path, data in sorted(files.items()):
        kind = "kernel" if path.startswith("src/xclbins/") else "engine"
        prefix = "src/xclbins/" if kind == "kernel" else f"src/lib/{runtime}/"
        artifact = ds.XCLBINS_ARTIFACT if kind == "kernel" else ds.libs_artifact(os_name, runtime)
        entries.append({"kind": kind, "repo_path": path, "artifact": artifact,
                        "artifact_path": path[len(prefix):], **digests(data), "new": False})
    return {"schema": 2, "os": os_name, "runtime": runtime, "run_id": run_id, "run_attempt": "1",
            "head_sha": head, "base_sha": base, "full_refresh": full_refresh, "files": entries}


def summary_job(pending: bool) -> dict:
    """Build and Test's summary job as the jobs API lists it: failed at
    PENDING_STEP alone while a swap is pending."""
    steps = [{"name": "Set up job", "conclusion": "success"}, {"name": "Gate and verdict", "conclusion": "success"},
             {"name": "Dev-kit checks", "conclusion": "success"},
             {"name": ds.PENDING_STEP, "conclusion": "failure" if pending else "success"},
             {"name": "Complete job", "conclusion": "success"}]
    return {"name": "summary", "conclusion": "failure" if pending else "success", "steps": steps}


class FakeApi:
    """devkit_swap.Api over dictionaries; download_artifact writes the zip's bytes."""

    def __init__(self):
        self.pulls, self.runs, self.jobs_by_run, self.artifacts_by_run = {}, {}, {}, {}
        self.zips: dict[int, bytes] = {}
        self.downloads: list[int] = []

    def pull(self, number):
        if number not in self.pulls:
            raise ds.SwapError(f"GitHub API pulls/{number}: HTTP 404")
        return self.pulls[number]

    def pulls_for_branch(self, owner, branch):
        return [p for p in self.pulls.values() if p["head"]["ref"] == branch]

    def run(self, run_id):
        if str(run_id) not in self.runs:
            raise ds.SwapError(f"GitHub API actions/runs/{run_id}: HTTP 404")
        return self.runs[str(run_id)]

    def runs_for_sha(self, sha):
        return [r for r in self.runs.values() if r["head_sha"] == sha]

    def jobs(self, run_id):
        return self.jobs_by_run.get(str(run_id), [])

    def artifacts(self, run_id):
        return self.artifacts_by_run.get(str(run_id), [])

    def download_artifact(self, artifact_id, dest, limit):
        self.downloads.append(artifact_id)
        data = self.zips[artifact_id]
        if len(data) > limit:
            raise ds.SwapError("too large")
        Path(dest).write_bytes(data)

    def add_artifact(self, run_id: str, name: str, data: bytes, expired: bool = False) -> None:
        aid = len(self.zips) + 1
        self.zips[aid] = data
        self.artifacts_by_run.setdefault(run_id, []).append(
            {"id": aid, "name": name, "size_in_bytes": len(data), "expired": expired})


class Repo:
    """A git repository with a bare origin: main (base) and feature (tested)."""

    def __init__(self, root: Path):
        self.root, self.path, self.origin = root, root / "work", root / "origin.git"
        hooks = root / "hooks"
        hooks.mkdir()
        self.git_raw("init", "--bare", "-b", "main", str(self.origin), cwd=root)
        self.path.mkdir()
        self.git("init", "-b", "main")
        for key, value in (("user.name", "t"), ("user.email", "t@example.com"), ("commit.gpgsign", "false"),
                           ("core.autocrlf", "false"), ("core.hooksPath", str(hooks))):
            self.git("config", key, value)
        self.git("remote", "add", "origin", str(self.origin))

    def git_raw(self, *args: str, cwd: Path) -> str:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout

    def git(self, *args: str) -> str:
        return self.git_raw(*args, cwd=self.path).strip()

    def write(self, files: dict[str, bytes]) -> None:
        for rel, data in files.items():
            p = self.path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def commit(self, files: dict[str, bytes], message: str = "change") -> str:
        self.write(files)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")


DEVKIT_DATA = {
    "flm-dev-kit/FLM_Xclbin/kernel_assignment.json": json.dumps(
        {"models": {"m:1": {"folder": "M-NPU2", "kernels": {"attn.xclbin": "xclbins-a"}}}}).encode(),
    "flm-dev-kit/FLM_DLL/engines.json": json.dumps(
        {"shared": [], "engines": {"foo_npu": {"families": ["m"], "models": ["m:1"], "uses": []}}}).encode(),
    "flm-dev-kit/FLM_DLL/detail/foo_npu/Makefile": b"TARGET = $(LIB_DIR)/libfoo_npu.so\n",
}
OLD = {XRT_LIN: blob("old-so"), XRT_WIN: blob("old-dll"), XRT_LIB: blob("old-lib"), KERNEL: blob("old-k"),
       "src/lib/xrt/libaiebu.a": blob("aiebu"), "src/lib/zlib1.dll": blob("zlib"),
       "src/xclbins/M-NPU2/extra.xclbin": blob("extra")}
NEW = {XRT_LIN: blob("new-so", 200), XRT_WIN: blob("new-dll", 300), XRT_LIB: blob("new-lib", 90),
       KERNEL: blob("new-k", 150)}


class SwapCase(unittest.TestCase):
    """A pull request (feature, #7) whose head changed dev-kit source and whose
    Build and Test run 100 staged the Linux and Windows XRT builds' files,
    passed every job and failed only at summary's PENDING_STEP (the swap is
    pending, as it always is before fetch)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(rmtree, self.tmp)
        self.repo = Repo(self.tmp)
        self.base = self.repo.commit({**DEVKIT_DATA, **OLD}, "base")
        self.repo.git("push", "-q", "origin", "main")
        self.repo.git("checkout", "-q", "-b", "feature")
        self.head = self.repo.commit({"flm-dev-kit/FLM_DLL/detail/foo_npu/foo.cpp": b"int f();\n"}, "source")
        self.repo.git("push", "-q", "origin", "feature")
        self.repo.git("fetch", "-q", "origin")
        self.api = FakeApi()
        self.api.pulls[PR] = {"number": PR, "state": "open",
                              "head": {"ref": "feature", "sha": self.head, "repo": {"full_name": REPO}}}
        self.add_run(RUN)
        cwd = os.getcwd()
        os.chdir(self.repo.path)
        self.addCleanup(os.chdir, cwd)

    def add_run(self, run_id: str, *, event: str = "pull_request", head: str | None = None,
                full_refresh: bool = False, staged: tuple = (("linux", "xrt"), ("windows", "xrt")),
                new: dict | None = None, manifests: dict | None = None, jobs: list | None = None,
                pending: bool = True, **run_fields) -> None:
        head = head or self.head
        new = NEW if new is None else new
        self.api.runs[run_id] = {
            "id": int(run_id), "path": ds.WORKFLOW_PATH, "repository": {"full_name": REPO},
            "head_repository": {"full_name": REPO}, "head_sha": head, "head_branch": "feature",
            "status": "completed", "conclusion": "failure" if pending else "success", "event": event,
            "pull_requests": [{"number": PR}] if event == "pull_request" else [], **run_fields}
        self.api.jobs_by_run[run_id] = jobs if jobs is not None else (
            [summary_job(pending), {"name": ds.STAGE_JOB, "conclusion": "success"},
             {"name": "devkit-engines / Linux XRT", "conclusion": "success"},
             {"name": "devkit-xclbins / assemble", "conclusion": "success"}]
            + [{"name": name, "conclusion": "success"} for name in ds.TEST_JOBS.values()])
        # The staging step uploads a manifest for every system and runtime,
        # empty where it staged nothing.
        for os_name, runtime in ds.TEST_JOBS:
            mine = {p: d for p, d in new.items() if (os_name, runtime) in staged and (
                p.startswith("src/xclbins/")
                or (p.startswith(f"src/lib/{runtime}/") and (p.endswith(".so") == (os_name == "linux"))))}
            m = (manifests or {}).get((os_name, runtime)) or manifest(
                os_name, runtime, mine, head=head, base=self.base, run_id=run_id, full_refresh=full_refresh)
            self.api.add_artifact(run_id, ds.manifest_artifact(os_name, runtime),
                                  zip_of({ds.MANIFEST_FILE: json.dumps(m).encode()}))
            libs = {p.rsplit("/", 1)[1]: d for p, d in mine.items() if p.startswith("src/lib/")}
            if libs and not any(a["name"] == ds.libs_artifact(os_name, runtime)
                                for a in self.api.artifacts(run_id)):
                self.api.add_artifact(run_id, ds.libs_artifact(os_name, runtime), zip_of(libs))
        kernels = {p[len("src/xclbins/"):]: d for p, d in new.items() if p.startswith("src/xclbins/")}
        if kernels:
            self.api.add_artifact(run_id, ds.XCLBINS_ARTIFACT, zip_of(kernels))

    def main(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ds.main(list(argv), api=self.api)
        return code, out.getvalue(), err.getvalue()

    def fetch(self, *extra: str) -> tuple[int, str, str]:
        return self.main("fetch", "--repo", REPO, "--no-git-fetch", *extra)

    def refused(self, needle: str, *extra: str) -> None:
        before = self.repo.git("rev-parse", "HEAD")
        code, out, err = self.fetch(*extra)
        self.assertEqual(code, 1, out + err)
        self.assertIn(needle, err)
        self.assertEqual(self.repo.git("rev-parse", "HEAD"), before)
        self.assertEqual(self.repo.git("status", "--porcelain"), "")

    def verify(self, head: str | None = None, base: str | None = None) -> tuple[bool, str]:
        code, out, err = self.main("verify", "--head", head or self.repo.git("rev-parse", "HEAD"),
                                   "--base", base or self.base, "--pr", str(PR), "--head-ref", "feature",
                                   "--head-repo", REPO, "--repo", REPO, "--repo-root", str(self.repo.path))
        self.assertEqual(code, 0)
        return "swap-verified=true" in out, out + err

    def check(self, base: str | None = None) -> tuple[int, str]:
        code, out, err = self.main("check", "--head", self.repo.git("rev-parse", "HEAD"),
                                   "--base", base or self.base, "--pr", str(PR), "--head-ref", "feature",
                                   "--head-repo", REPO, "--repo", REPO, "--repo-root", str(self.repo.path))
        return code, out + err


class Fetch(SwapCase):
    def test_swap_commit(self):
        code, out, err = self.fetch()
        self.assertEqual(code, 0, out + err)
        new = self.repo.git("rev-parse", "HEAD")
        self.assertEqual(self.repo.git("rev-parse", "HEAD^"), self.head)
        message = self.repo.git("log", "-1", "--format=%B")
        self.assertIn(f"{ds.TRAILER_RUN}: {RUN}", message)
        self.assertIn(f"{ds.TRAILER_TESTED}: {self.head}", message)
        changed = sorted(self.repo.git("diff", "--name-only", self.head, new).splitlines())
        self.assertEqual(changed, sorted(NEW))
        for path, data in NEW.items():
            self.assertEqual((self.repo.path / path).read_bytes(), data, path)
        self.assertEqual(self.repo.git("status", "--porcelain"), "")
        self.assertIn("git push", out)
        # Nothing is pushed.
        self.assertEqual(self.repo.git("rev-parse", "origin/feature"), self.head)

    def test_defaults_find_the_pull_request_and_run(self):
        code, out, err = self.fetch()
        self.assertEqual(code, 0, out + err)
        self.assertIn(f"pull request #{PR} at {self.head[:10]}, Build and Test run {RUN}", out)

    def test_dry_run_changes_nothing(self):
        code, out, err = self.fetch("--dry-run")
        self.assertEqual(code, 0, out + err)
        self.assertIn("--dry-run", out)
        for path in NEW:
            self.assertIn(path, out)
        self.assertEqual(self.repo.git("rev-parse", "HEAD"), self.head)
        self.assertEqual(self.repo.git("status", "--porcelain"), "")
        self.assertEqual(sorted(self.api.downloads), [a["id"] for a in self.api.artifacts(RUN)
                                                      if a["name"].startswith("devkit-stage-manifest-")])

    def test_pending_is_the_one_failure_allowed(self):
        # The run failed, at summary's PENDING_STEP alone: that is the swap.
        self.assertEqual(self.api.runs[RUN]["conclusion"], "failure")
        code, out, err = self.fetch()
        self.assertEqual(code, 0, out + err)

    def test_run_without_pending_failure(self):
        self.api.runs.clear()
        self.api.artifacts_by_run.clear()
        self.add_run(RUN, pending=False)
        code, out, err = self.fetch()
        self.assertEqual(code, 0, out + err)

    def test_full_refresh_run(self):
        self.add_run("200", event="workflow_dispatch", full_refresh=True, staged=tuple(ds.TEST_JOBS),
                     new={**NEW, "src/lib/hrx/libfoo_npu.so": blob("h-so"), "src/lib/hrx/foo_npu.dll": blob("h-dll"),
                          "src/lib/hrx/foo_npu.lib": blob("h-lib")})
        code, out, err = self.fetch("--run", "200")
        self.assertEqual(code, 0, out + err)
        self.assertIn("src/lib/hrx/foo_npu.lib", self.repo.git("show", "--name-only", "--format=", "HEAD"))
        ok, why = self.verify(base=self.base)
        self.assertTrue(ok, why)

    # ── Refusals ──

    def test_dirty_tree(self):
        (self.repo.path / XRT_LIN).write_bytes(b"edited")
        code, _, err = self.fetch()
        self.assertEqual(code, 1)
        self.assertIn("uncommitted changes", err)

    def test_untracked_binary(self):
        (self.repo.path / "src/xclbins/M-NPU2/stray.xclbin").write_bytes(b"x")
        code, _, err = self.fetch()
        self.assertEqual(code, 1)
        self.assertIn("untracked files under src/lib or src/xclbins", err)

    def test_head_after_the_tested_commit(self):
        self.repo.commit({"README.md": b"more\n"})
        self.refused("is not pull request #7's head")

    def test_pull_request_moved_on(self):
        self.api.pulls[PR]["head"]["sha"] = "f" * 40
        self.refused("is not pull request #7's head")

    def test_fork(self):
        self.api.pulls[PR]["head"]["repo"] = {"full_name": "someone/r"}
        self.refused("comes from someone/r")

    def test_closed(self):
        self.api.pulls[PR]["state"] = "closed"
        self.refused("is not open")

    def test_run_of_another_commit(self):
        self.api.runs[RUN]["head_sha"] = "e" * 40
        self.refused(f"tested {'e' * 10}", "--run", RUN)

    def test_run_of_another_workflow(self):
        self.api.runs[RUN]["path"] = ".github/workflows/release.yml"
        self.refused("is not of .github/workflows/build-and-test.yml", "--run", RUN)

    def test_run_still_going(self):
        self.api.runs[RUN]["status"] = "in_progress"
        self.refused("is still in_progress")

    def test_push_run(self):
        self.api.runs[RUN]["event"] = "push"
        self.refused("is a push run", "--run", RUN)

    def test_run_of_another_pull_request(self):
        self.api.runs[RUN]["pull_requests"] = [{"number": 8}]
        self.refused("not #7", "--run", RUN)

    def test_summary_failed(self):
        # Another summary step failed beside the pending one.
        self.api.jobs_by_run[RUN][0]["steps"][1]["conclusion"] = "failure"
        self.refused("its summary job did not pass (failure): Gate and verdict")

    def test_summary_cancelled(self):
        self.api.jobs_by_run[RUN][0]["conclusion"] = "cancelled"
        self.refused("its summary job did not pass (cancelled)")

    def test_test_job_failed(self):
        name = ds.TEST_JOBS[("linux", "xrt")]
        next(j for j in self.api.jobs_by_run[RUN] if j["name"] == name)["conclusion"] = "failure"
        self.refused(f"{name} (failure)")

    def test_staged_but_untested(self):
        name = ds.TEST_JOBS[("windows", "xrt")]
        next(j for j in self.api.jobs_by_run[RUN] if j["name"] == name)["conclusion"] = "skipped"
        self.refused(f"the windows xrt build staged 3 file(s), but {name} did not pass (skipped): they are untested")

    def test_build_failed(self):
        self.api.jobs_by_run[RUN].append({"name": "devkit-xclbins / build (3)", "conclusion": "failure"})
        self.refused("devkit-xclbins / build (3) (failure)")

    def test_a_failed_bench_alone_does_not_block_the_swap(self):
        # summary only warns about it, so fetch does not count it either.
        self.api.jobs_by_run[RUN].append({"name": "devkit-bench / Bench lin-xrt m:1", "conclusion": "failure"})
        code, out, err = self.fetch()
        self.assertEqual(code, 0, out + err)

    def test_bench_hard_gate_failure_is_refused(self):
        # A model failing the accuracy hard gate fails summary's Dev-kit checks.
        self.api.jobs_by_run[RUN].append({"name": "devkit-bench / Report", "conclusion": "failure"})
        self.api.jobs_by_run[RUN][0]["steps"][2]["conclusion"] = "failure"
        self.refused("its summary job did not pass (failure): Dev-kit checks")

    def test_nothing_staged(self):
        self.api.jobs_by_run[RUN] = [j for j in self.api.jobs_by_run[RUN] if j["name"] != ds.STAGE_JOB]
        self.refused("its devkit-stage job did not run or pass (not found)")

    def test_empty_manifests(self):
        self.api.artifacts_by_run[RUN] = []
        self.add_run(RUN, staged=())
        self.refused("staged no dev-kit file: nothing to swap")

    def test_manifest_expired(self):
        for a in self.api.artifacts(RUN):
            if a["name"].startswith("devkit-stage-manifest-"):
                a["expired"] = True
        self.refused("is missing, expired or too large")

    def test_artifacts_expired(self):
        for a in self.api.artifacts(RUN):
            if a["name"] == ds.XCLBINS_ARTIFACT:
                a["expired"] = True
        self.refused("re-run the Build and Test workflow")

    def test_main_moved(self):
        self.repo.git("checkout", "-q", "main")
        self.repo.commit({"flm-dev-kit/FLM_Xclbin/x/kernel.cc": b"//\n"})
        self.repo.git("push", "-q", "origin", "main")
        self.repo.git("checkout", "-q", "feature")
        self.repo.git("fetch", "-q", "origin")
        self.refused("main gained 1 commit(s) touching the dev-kit sources or binaries")

    def test_main_moved_elsewhere_is_fine(self):
        self.repo.git("checkout", "-q", "main")
        self.repo.commit({"src/server/x.cpp": b"//\n"})
        self.repo.git("push", "-q", "origin", "main")
        self.repo.git("checkout", "-q", "feature")
        self.repo.git("fetch", "-q", "origin")
        code, out, err = self.fetch()
        self.assertEqual(code, 0, out + err)

    def test_nothing_to_swap(self):
        self.repo.commit(NEW, "already")
        self.repo.git("push", "-q", "origin", "feature")
        new_head = self.repo.git("rev-parse", "HEAD")
        self.api.pulls[PR]["head"]["sha"] = new_head
        self.api.artifacts_by_run.clear()
        self.add_run("300", head=new_head)
        self.refused("nothing to swap")

    def test_hash_mismatch_writes_nothing(self):
        for a in self.api.artifacts(RUN):
            if a["name"] == ds.XCLBINS_ARTIFACT:
                self.api.zips[a["id"]] = zip_of({"M-NPU2/attn.xclbin": b"y" * len(NEW[KERNEL])})
        self.refused("is not the bytes the manifest records: nothing written")
        self.assertEqual((self.repo.path / KERNEL).read_bytes(), OLD[KERNEL])

    def test_size_mismatch(self):
        for a in self.api.artifacts(RUN):
            if a["name"] == ds.XCLBINS_ARTIFACT:
                self.api.zips[a["id"]] = zip_of({"M-NPU2/attn.xclbin": b"short"})
        self.refused("the manifest says")

    def test_unsafe_zip_entry(self):
        for a in self.api.artifacts(RUN):
            if a["name"] == ds.XCLBINS_ARTIFACT:
                self.api.zips[a["id"]] = zip_of({"M-NPU2/attn.xclbin": NEW[KERNEL], "../evil.xclbin": b"x"})
        self.refused("unsafe entry name")
        self.assertFalse((self.repo.path.parent / "evil.xclbin").exists())

    def test_symlink_in_zip(self):
        for a in self.api.artifacts(RUN):
            if a["name"] == ds.XCLBINS_ARTIFACT:
                self.api.zips[a["id"]] = zip_of({"M-NPU2/attn.xclbin": NEW[KERNEL]}, symlinks=("M-NPU2/l",))
        self.refused("is a symbolic link")

    def test_manifest_naming_outside_the_binaries(self):
        bad = manifest("linux", "xrt", {XRT_LIN: NEW[XRT_LIN]}, head=self.head, base=self.base)
        bad["files"][0]["repo_path"] = "src/server/server.cpp"
        self.api.artifacts_by_run[RUN] = []
        self.add_run(RUN, manifests={("linux", "xrt"): bad})
        self.refused("an engine file must be src/lib/xrt/<file>")

    def test_half_updated_runtime(self):
        self.api.artifacts_by_run[RUN] = []
        self.add_run(RUN, staged=(("linux", "xrt"),))
        self.refused("did not both stage the engines foo_npu")

    def test_dispatch_that_is_no_full_refresh(self):
        self.add_run("200", event="workflow_dispatch", staged=tuple(ds.TEST_JOBS))
        self.refused("is a dispatch but not a dev-kit full refresh", "--run", "200")

    def test_full_refresh_needs_all_four_test_jobs(self):
        self.add_run("200", event="workflow_dispatch", full_refresh=True)
        name = ds.TEST_JOBS[("linux", "hrx")]
        next(j for j in self.api.jobs_by_run["200"] if j["name"] == name)["conclusion"] = "skipped"
        self.refused("not all four product test jobs passed", "--run", "200")


class Verify(SwapCase):
    def swapped(self) -> str:
        code, out, err = self.fetch()
        self.assertEqual(code, 0, out + err)
        return self.repo.git("rev-parse", "HEAD")

    def test_verified(self):
        self.swapped()
        ok, why = self.verify()
        self.assertTrue(ok, why)
        self.assertIn("4 file(s) are exactly the bytes run 100 built and tested", why)

    def test_github_output(self):
        self.swapped()
        out = self.tmp / "out.txt"
        with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(out)}):
            code, _, _ = self.main("verify", "--head", self.repo.git("rev-parse", "HEAD"), "--base", self.base,
                                   "--pr", str(PR), "--head-ref", "feature", "--head-repo", REPO, "--repo", REPO,
                                   "--repo-root", str(self.repo.path), "--github-output")
        self.assertEqual(code, 0)
        self.assertEqual(out.read_text(encoding="utf-8"), "swap-verified=true\n")

    def test_not_a_swap_commit(self):
        ok, why = self.verify()
        self.assertFalse(ok)
        self.assertIn("no Devkit-Swap-Run trailer", why)
        self.assertEqual(self.api.downloads, [])

    def test_author_does_not_matter_but_parent_does(self):
        self.swapped()
        self.repo.git("commit", "-q", "--amend", "--no-edit", "--author", "Someone <s@example.com>")
        ok, why = self.verify()
        self.assertTrue(ok, why)
        self.repo.commit({"README.md": b"x\n"}, f"again\n\n{ds.TRAILER_RUN}: {RUN}\n{ds.TRAILER_TESTED}: {self.head}")
        ok, why = self.verify()
        self.assertFalse(ok)
        self.assertIn("not the tested", why)

    def test_extra_file(self):
        self.swapped()
        self.repo.write({"src/runner/x.cpp": b"//\n"})
        self.repo.git("add", "-A")
        self.repo.git("commit", "-q", "--amend", "--no-edit")
        ok, why = self.verify()
        self.assertFalse(ok)
        self.assertIn("changes src/runner/x.cpp, which run 100's manifests do not list", why)

    def test_other_bytes(self):
        self.swapped()
        self.repo.write({KERNEL: blob("tampered", 150)})
        self.repo.git("add", "-A")
        self.repo.git("commit", "-q", "--amend", "--no-edit")
        ok, why = self.verify()
        self.assertFalse(ok)
        self.assertIn(f"{KERNEL} at", why)

    def test_deletion(self):
        self.swapped()
        self.repo.git("rm", "-q", XRT_LIB)
        self.repo.git("commit", "-q", "--amend", "--no-edit")
        ok, why = self.verify()
        self.assertFalse(ok)
        self.assertIn(f"deletes {XRT_LIB}", why)

    def test_malformed_trailers(self):
        self.repo.commit({"README.md": b"x\n"}, f"x\n\n{ds.TRAILER_RUN}: soon\n{ds.TRAILER_TESTED}: {self.head}")
        ok, why = self.verify()
        self.assertFalse(ok)
        self.assertIn("needs exactly one", why)

    def test_run_failed_after_all(self):
        self.swapped()
        self.api.jobs_by_run[RUN][0]["steps"][2]["conclusion"] = "failure"
        ok, why = self.verify()
        self.assertFalse(ok)
        self.assertIn("summary job did not pass (failure): Dev-kit checks", why)

    def test_fork(self):
        self.swapped()
        code, out, _ = self.main("verify", "--head", self.repo.git("rev-parse", "HEAD"), "--base", self.base,
                                 "--pr", str(PR), "--head-ref", "feature", "--head-repo", "x/r", "--repo", REPO,
                                 "--repo-root", str(self.repo.path))
        self.assertIn("swap-verified=false", out)
        self.assertIn("from a fork", out)

    def test_main_moved_since(self):
        self.swapped()
        self.repo.git("checkout", "-q", "main")
        later = self.repo.commit({"src/xclbins/M-NPU2/attn.xclbin": blob("main-k")})
        self.repo.git("checkout", "-q", "feature")
        ok, why = self.verify(base=later)
        self.assertFalse(ok)
        self.assertIn("merge main and let CI test again", why)

    def test_partial_swap(self):
        # The Windows half of the engine left out: Linux would ship a new
        # build beside an old Windows one.
        self.swapped()
        self.repo.git("checkout", "-q", self.head, "--", XRT_WIN, XRT_LIB)
        self.repo.git("commit", "-q", "--amend", "--no-edit")
        self.assertEqual(sorted(self.repo.git("diff", "--name-only", self.head, "HEAD").splitlines()),
                         sorted([XRT_LIN, KERNEL]))
        ok, why = self.verify()
        self.assertFalse(ok)
        self.assertIn("tested 2 file(s)", why)
        self.assertIn("does not carry", why)
        code, out = self.check()
        self.assertEqual(code, 1)
        self.assertIn(f"::error::{XRT_LIN}", out)

    def test_pull_request_that_changes_swap_ci(self):
        # The pull request's own code writes and checks the manifests: a
        # swap that comes with a change to it is not trusted.
        self.head = self.repo.commit({".github/script/ci/devkit_swap.py": b"# changed\n"}, "ci")
        self.repo.git("push", "-q", "origin", "feature")
        self.api.pulls[PR]["head"]["sha"] = self.head
        self.api.runs.clear()
        self.api.artifacts_by_run.clear()
        self.add_run(RUN)
        self.swapped()
        self.api.downloads.clear()
        ok, why = self.verify()
        self.assertFalse(ok)
        self.assertIn("also changes the dev-kit CI that checks a swap (.github/script/ci/devkit_swap.py)", why)
        self.assertEqual(self.api.downloads, [])
        code, out = self.check()
        self.assertIn("not trusted, the pull request also changes the dev-kit CI", out)
        self.assertIn(f"::error::{XRT_LIN}", out)

    def test_unexpected_error_is_not_verified(self):
        self.swapped()
        with mock.patch.object(ds, "verify_swap", side_effect=AttributeError("'list' object has no attribute 'get'")):
            ok, why = self.verify()
        self.assertFalse(ok)
        self.assertIn("unexpected AttributeError", why)

    def test_api_unreachable(self):
        self.swapped()
        self.api.runs.clear()
        ok, why = self.verify()
        self.assertFalse(ok)
        self.assertIn("HTTP 404", why)


class Check(SwapCase):
    def test_nothing_changed(self):
        code, out = self.check()
        self.assertEqual(code, 0)
        self.assertIn("no prebuilt binary changed", out)
        self.assertEqual(self.api.downloads, [])

    def test_hand_edit_fails(self):
        self.assertTrue(ds.SWAP_ENFORCE)
        self.repo.commit({XRT_LIN: b"hand"})
        code, out = self.check()
        self.assertEqual(code, 1)
        self.assertIn(f"::error::{XRT_LIN} is not the bytes a verified swap commit", out)
        self.assertIn("devkit_swap.py fetch", out)

    def test_not_enforced_warns(self):
        self.repo.commit({XRT_LIN: b"hand"})
        with mock.patch.object(ds, "SWAP_ENFORCE", False):
            code, out = self.check()
        self.assertEqual(code, 0)
        self.assertIn(f"::warning::{XRT_LIN}", out)

    def test_verified_swap_passes(self):
        code, out, err = self.fetch()
        self.assertEqual(code, 0, out + err)
        code, out = self.check()
        self.assertEqual(code, 0)
        self.assertNotIn("::error::", out)
        self.assertIn("4 file(s) verified", out)

    def test_edit_after_the_swap(self):
        self.fetch()
        self.repo.commit({KERNEL: blob("later", 150)})
        code, out = self.check()
        self.assertEqual(code, 1)
        self.assertIn(f"::error::{KERNEL}", out)
        self.assertNotIn(f"::error::{XRT_LIN}", out)

    def test_deletions(self):
        self.repo.git("rm", "-q", "src/xclbins/M-NPU2/extra.xclbin")
        self.repo.git("commit", "-q", "-m", "drop an unassigned kernel")
        code, out = self.check()
        self.assertEqual(code, 0, out)
        self.repo.git("rm", "-q", KERNEL)
        self.repo.git("commit", "-q", "-m", "drop an assigned kernel")
        code, out = self.check()
        self.assertIn(f"::error::{KERNEL} is deleted, but the dev kit still assigns or builds it", out)

    def test_third_party(self):
        self.repo.commit({"src/lib/xrt/libaiebu.a": b"new aiebu", "src/lib/zlib1.dll": b"new zlib"})
        code, out = self.check()
        self.assertEqual(code, 0, out)
        self.assertIn("third-party", out)

    def test_unexpected_error_fails_when_enforced(self):
        self.repo.commit({XRT_LIN: b"hand"})
        with mock.patch.object(ds.devkit_check, "required_binaries", side_effect=ValueError("bad data")):
            code, out = self.check()
        self.assertEqual(code, 1)
        self.assertIn("unexpected ValueError: bad data", out)
        with mock.patch.object(ds.devkit_check, "required_binaries", side_effect=ValueError("bad data")), \
                mock.patch.object(ds, "SWAP_ENFORCE", False):
            code, out = self.check()
        self.assertEqual(code, 0)
        self.assertIn("::warning::devkit_swap check could not run: unexpected ValueError: bad data", out)

    def test_fake_trailers_do_not_excuse(self):
        self.repo.commit({XRT_LIN: b"hand"}, f"x\n\n{ds.TRAILER_RUN}: {RUN}\n{ds.TRAILER_TESTED}: {self.head}")
        code, out = self.check()
        self.assertIn(f"::error::{XRT_LIN}", out)
        self.assertIn("does not verify", out)


class Pending(SwapCase):
    """summary's PENDING_STEP: fails while the branch lacks the staged files."""

    def write_manifests(self, new: dict) -> Path:
        d = self.tmp / "manifests"
        for os_name, runtime in ds.TEST_JOBS:
            mine = {p: v for p, v in new.items() if runtime == "xrt" and (
                p.startswith("src/xclbins/") or p.endswith(".so") == (os_name == "linux"))}
            m = manifest(os_name, runtime, mine, head=self.head, base=self.base)
            (d / ds.manifest_artifact(os_name, runtime)).mkdir(parents=True)
            (d / ds.manifest_artifact(os_name, runtime) / ds.MANIFEST_FILE).write_text(json.dumps(m))
        return d

    def pending(self, d: Path, *extra: str) -> tuple[int, str]:
        summary = self.tmp / "summary.md"
        code, out, _ = self.main("pending", "--manifests", str(d), "--head", self.head, "--run", RUN,
                                 "--summary", str(summary), "--repo-root", str(self.repo.path), *extra)
        self.summary = summary.read_text(encoding="utf-8") if summary.is_file() else ""
        return code, out

    def test_fails_with_the_command_when_files_differ(self):
        code, out = self.pending(self.write_manifests(NEW))
        self.assertEqual(code, 1)
        self.assertIn(f"python .github/script/ci/devkit_swap.py fetch --run {RUN}", out)
        self.assertIn("4 dev-kit file(s)", out)
        self.assertIn(f"::error title={ds.PENDING_STEP}::", out)
        self.assertIn("runs no NPU test again", out)
        self.assertIn("### Dev-kit swap pending", self.summary)
        self.assertNotIn("pull request of its own", out)

    def test_passes_when_the_branch_has_them(self):
        self.repo.commit(NEW)
        self.head = self.repo.git("rev-parse", "HEAD")
        code, out = self.pending(self.write_manifests(NEW))
        self.assertEqual(code, 0, out)
        self.assertIn("no swap pending", out)
        self.assertEqual(self.summary, "")

    def test_passes_without_manifests(self):
        code, _ = self.pending(self.tmp / "none")
        self.assertEqual(code, 0)

    def test_names_a_swap_ci_change(self):
        self.head = self.repo.commit({".github/workflows/build-and-test.yml": b"x: 1\n"}, "ci")
        code, out = self.pending(self.write_manifests(NEW), "--base", self.base)
        self.assertEqual(code, 1)
        self.assertIn("`.github/workflows/build-and-test.yml`", out)
        self.assertIn("move that change to a pull request of its own first", out)

    def test_broken_manifest_fails(self):
        d = self.write_manifests(NEW)
        (d / ds.manifest_artifact("linux", "xrt") / ds.MANIFEST_FILE).write_text("{")
        code, out = self.pending(d)
        self.assertEqual(code, 1)
        self.assertIn("::error::devkit_swap pending:", out)


class Manifests(unittest.TestCase):
    """parse_manifest: the manifest is data from the pull request's own job."""

    HEAD, BASE = "a" * 40, "b" * 40

    def good(self) -> dict:
        return manifest("windows", "xrt", {XRT_WIN: b"d", XRT_LIB: b"l", KERNEL: b"k"}, head=self.HEAD,
                        base=self.BASE)

    def parse(self, m: dict, os_name: str = "windows", runtime: str = "xrt") -> ds.Manifest:
        return ds.parse_manifest(json.dumps(m).encode(), os_name, runtime)

    def test_good(self):
        m = self.parse(self.good())
        self.assertEqual(sorted(e.repo_path for e in m.files), sorted([XRT_WIN, XRT_LIB, KERNEL]))
        self.assertFalse(any(e.new for e in m.files))

    def test_bad(self):
        cases = {
            "absolute": ("repo_path", "/src/lib/xrt/x.dll"),
            "dotdot": ("repo_path", "src/lib/xrt/../../../x.dll"),
            "backslash": ("repo_path", "src\\lib\\xrt\\x.dll"),
            "drive": ("repo_path", "C:/src/lib/xrt/x.dll"),
            "other runtime": ("repo_path", "src/lib/hrx/foo_npu.dll"),
            "nested lib": ("repo_path", "src/lib/xrt/sub/foo_npu.dll"),
            "extension": ("repo_path", "src/lib/xrt/foo_npu.exe"),
            "linux name on windows": ("repo_path", "src/lib/xrt/libfoo.so"),
            "artifact": ("artifact", "devkit-libs-lin-xrt"),
            "artifact path": ("artifact_path", "../foo_npu.dll"),
            "size": ("size", -1),
            "bool size": ("size", True),
            "sha256": ("sha256", "zz"),
            "kind": ("kind", "script"),
            "new": ("new", "yes"),
        }
        for why, (key, value) in cases.items():
            with self.subTest(why):
                m = self.good()
                entry = next(e for e in m["files"] if e["repo_path"] == XRT_WIN)
                entry[key] = value
                with self.assertRaises(ds.ManifestError):
                    self.parse(m)
        with self.assertRaises(ds.ManifestError):
            self.parse(self.good(), "linux")
        m = self.good()
        m["files"].append(dict(m["files"][0]))
        with self.assertRaisesRegex(ds.ManifestError, "listed twice"):
            self.parse(m)
        for path in ("src/xclbins/attn.xclbin", "src/xclbins/M-NPU2/sub/attn.xclbin"):
            with self.subTest(kernel=path):
                bad_kernel = self.good()
                entry = next(e for e in bad_kernel["files"] if e["repo_path"] == KERNEL)
                entry["repo_path"], entry["artifact_path"] = path, path[len("src/xclbins/"):]
                with self.assertRaisesRegex(ds.ManifestError, "src/xclbins/<folder>/<slot>.xclbin"):
                    self.parse(bad_kernel)
        # A third-party file is no dev-kit build, whatever its name.
        third = manifest("windows", "xrt", {"src/lib/xrt/aiebu_static.lib": b"a"}, head=self.HEAD, base=self.BASE)
        with self.assertRaisesRegex(ds.ManifestError, "third-party"):
            self.parse(third)
        # An import library comes with its DLL.
        lone = manifest("windows", "xrt", {XRT_LIB: b"l", KERNEL: b"k"}, head=self.HEAD, base=self.BASE)
        with self.assertRaisesRegex(ds.ManifestError, "no foo_npu.dll beside it"):
            self.parse(lone)
        # And a DLL with its import library: the build links the one.
        lone = manifest("windows", "xrt", {XRT_WIN: b"d"}, head=self.HEAD, base=self.BASE)
        with self.assertRaisesRegex(ds.ManifestError, "no foo_npu.lib beside it"):
            self.parse(lone)
        # A manifest from before the staging (schema 1).
        old = self.good()
        old["schema"] = 1
        with self.assertRaisesRegex(ds.ManifestError, "not a schema 2 manifest"):
            self.parse(old)
        with self.assertRaises(ds.ManifestError):
            ds.parse_manifest(b"{", "windows", "xrt")
        with self.assertRaises(ds.ManifestError):
            ds.parse_manifest(b" " * (ds.MAX_MANIFEST_BYTES + 1), "windows", "xrt")

    def test_every_staged_entry_is_swapped(self):
        lin = manifest("linux", "xrt", {KERNEL: b"k", "src/xclbins/N-NPU2/a.xclbin": b"n"}, head=self.HEAD,
                       base=self.BASE)
        entries, notes, errors = ds.plan_swap({("linux", "xrt"): self.parse(lin, "linux")}, RUN, self.HEAD)
        self.assertEqual((sorted(entries), notes, errors), ([KERNEL, "src/xclbins/N-NPU2/a.xclbin"], [], []))

    def test_disagreeing_manifests(self):
        lin = manifest("linux", "xrt", {KERNEL: b"k1"}, head=self.HEAD, base=self.BASE)
        win = manifest("windows", "xrt", {KERNEL: b"k2"}, head=self.HEAD, base=self.BASE)
        _, _, errors = ds.plan_swap({("linux", "xrt"): self.parse(lin, "linux"),
                                     ("windows", "xrt"): self.parse(win)}, RUN, self.HEAD)
        self.assertIn(f"{KERNEL}: the manifests disagree on its bytes", errors)

    def test_other_run(self):
        m = self.parse(self.good())
        _, _, errors = ds.plan_swap({("windows", "xrt"): m}, "999", self.HEAD)
        self.assertTrue(errors and "not run 999" in errors[0])


class Zips(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(rmtree, self.tmp)

    def read(self, data: bytes, member: str = "a/b.xclbin", size: int | None = 3) -> bytes:
        path = self.tmp / "z.zip"
        path.write_bytes(data)
        return ds.read_zip_member(path, member, size)

    def test_reads_one_member(self):
        self.assertEqual(self.read(zip_of({"a/b.xclbin": b"abc", "a/c.xclbin": b"x"})), b"abc")

    def test_rejects(self):
        # Any unsafe entry fails the whole zip, even beside a good member.
        for name in ("/abs.xclbin", "a/../b", "a\\b", "C:/x", "a//b", "./a"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ds.SwapError, "unsafe entry name"):
                    self.read(zip_of({"a/b.xclbin": b"abc", name: b"x"}))
        with self.assertRaisesRegex(ds.SwapError, "the manifest says"):
            self.read(zip_of({"a/b.xclbin": b"abcd"}))
        with self.assertRaisesRegex(ds.SwapError, "missing"):
            self.read(zip_of({"other": b"x"}))
        with self.assertRaisesRegex(ds.SwapError, "symbolic link"):
            self.read(zip_of({"a/b.xclbin": b"abc"}, symlinks=("a/l",)))
        with self.assertRaisesRegex(ds.SwapError, "unreadable"):
            self.read(b"not a zip")


class Redirects(unittest.TestCase):
    """Api sends its token to the API host only: no redirect is followed."""

    def test_api_call_does_not_follow_a_redirect(self):
        import http.server
        import threading
        seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append((self.path, self.headers.get("Authorization")))
                if self.path.startswith("/elsewhere"):
                    body = b"{}"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(301)
                    self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/elsewhere")
                    self.send_header("Content-Length", "0")
                    self.end_headers()

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        api = ds.Api(REPO, "secret", base=f"http://127.0.0.1:{server.server_port}")
        with self.assertRaisesRegex(ds.SwapError, r"HTTP 301 \(a redirect, not followed"):
            api.run("1")
        self.assertEqual([p for p, _ in seen], [f"/repos/{REPO}/actions/runs/1"])


class Targets(unittest.TestCase):
    def test_symlinked_directory_is_refused(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(rmtree, tmp)
        (tmp / "outside").mkdir()
        (tmp / "repo" / "src").mkdir(parents=True)
        try:
            os.symlink(tmp / "outside", tmp / "repo" / "src" / "xclbins", target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create a symbolic link here")
        with self.assertRaisesRegex(ds.SwapError, "symbolic link"):
            ds.safe_target(tmp / "repo", "src/xclbins/M/a.xclbin")


class Wiring(unittest.TestCase):
    """devkit_check.py's swap wiring rule, on this tree and on a broken copy."""

    def test_tree(self):
        errors: list = []
        self.assertEqual(dc.check_swap_wiring(dc.REPO_ROOT, errors), 1)
        self.assertEqual(errors, [])

    def test_broken(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(rmtree, tmp)
        rel = dc.BUILD_WORKFLOW
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        text = (dc.REPO_ROOT / rel).read_text(encoding="utf-8")
        text = text.replace("name: devkit-stage-manifest-windows-hrx", "name: stage-manifest-windows-hrx")
        text = text.replace("    name: Test Windows HRX\n", "    name: Test Win HRX\n")
        text = text.replace(f"- name: {ds.PENDING_STEP}", "- name: Swap pending")
        text = text.replace("devkit_swap.py check", "devkit_swap.py chek")
        (tmp / rel).write_text(text, encoding="utf-8")
        errors: list = []
        dc.check_swap_wiring(tmp, errors)
        joined = "\n".join(errors)
        self.assertIn("as devkit-stage-manifest-windows-hrx", joined)
        self.assertIn('exactly one job named "Test Windows HRX"', joined)
        self.assertIn(f'a step named "{ds.PENDING_STEP}"', joined)
        self.assertIn("must run devkit_swap.py check", joined)

    def test_build_and_test(self):
        text = (dc.REPO_ROOT / dc.BUILD_WORKFLOW).read_text(encoding="utf-8")
        jobs = {k: "\n".join(v) for k, v in dc._jobs(text).items()}
        self.assertIn("      devkit-full-refresh:\n", text)
        changes_job = jobs["changes"]
        self.assertLess(changes_job.index("devkit_swap.py verify"),
                        changes_job.index("uses: ./.github/actions/detect-changes"))
        self.assertIn("swap-verified: ${{ steps.swap.outputs.swap-verified }}", changes_job)
        self.assertIn("if: github.event_name == 'pull_request'", changes_job)
        # The full refresh tests on the full tier, every catalog model required.
        self.assertIn("tier: ${{ github.event_name == 'workflow_dispatch' && inputs.devkit-full-refresh "
                      "&& 'full' || inputs.tier }}", changes_job)
        stage = jobs[ds.STAGE_JOB]
        for line in ('[ "${FULL_REFRESH}" = true ] && args+=(--require-all --full-refresh)',
                     "HEAD_SHA: ${{ github.event.pull_request.head.sha || github.sha }}",
                     "BASE_SHA: ${{ needs.changes.outputs.devkit-base }}",
                     # Only the pull request's dev-kit builds: pushes and the
                     # nightly (the -main callers) leave the product builds
                     # waiting for nothing.
                     "    needs: [changes, devkit-engines, devkit-xclbins]"):
            self.assertIn(line, stage)
        for job in ("devkit-engines-main", "devkit-xclbins-main"):
            self.assertIn("github.event_name != 'pull_request'", jobs[job])
        for job in ("build-lin-xrt-portable", "build-lin-hrx-deb", "build-win-xrt", "build-win-source"):
            with self.subTest(job=job):
                self.assertIn("    needs: [changes, devkit-stage]", jobs[job])
                self.assertIn("(needs.devkit-stage.result == 'success' || needs.devkit-stage.result == 'skipped')",
                              jobs[job])
                self.assertIn("devkit-stage: ${{ needs.devkit-stage.outputs.staged == 'true' }}", jobs[job])
        # The binaries check fails summary, not the changes job.
        self.assertIn("continue-on-error: true", changes_job)
        self.assertIn("binaries-check: ${{ steps.binaries.outcome }}", changes_job)
        self.assertIn("&& 'refresh'", jobs["devkit-bench"])
        summary = jobs["summary"]
        self.assertIn('[ "${SWAP_VERIFIED}" = true ] && STAGE_REQUESTED=false', summary)
        self.assertIn('if [ "${BINARIES}" = failure ]; then', summary)
        self.assertIn("devkit_swap.py pending", summary)
        # Nothing here writes: no grant of a write permission.
        self.assertNotRegex(text, r"(?m)^\s+[a-z-]+: write\s*$")

    def test_required_binaries(self):
        req = dc.required_binaries(dc.REPO_ROOT)
        self.assertIn("src/lib/hrx/qwen3_npu.lib", req)
        self.assertIn("src/lib/xrt/libq4_npu_eXpress.so", req)
        self.assertTrue(any(p.startswith("src/xclbins/") for p in req))
        self.assertNotIn("src/lib/xrt/libaiebu.a", req)
        self.assertNotIn("src/xclbins/MiniCPM-OCR-1.0-NPU2/conv.xclbin", req)


if __name__ == "__main__":
    unittest.main()
