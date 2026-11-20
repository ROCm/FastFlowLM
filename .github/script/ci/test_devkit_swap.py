# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for devkit_swap.py and the manifests devkit_overlay.py writes:
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
import devkit_overlay as do  # noqa: E402
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
             run_id: str = RUN, tests_ran: bool = True, full_refresh: bool = False,
             untested: tuple = ()) -> dict:
    """A schema 1 manifest as devkit_overlay.py writes it, for repo path -> bytes."""
    entries = []
    for path, data in sorted(files.items()):
        kind = "kernel" if path.startswith("src/xclbins/") else "engine"
        prefix = "src/xclbins/" if kind == "kernel" else f"src/lib/{runtime}/"
        artifact = ds.XCLBINS_ARTIFACT if kind == "kernel" else ds.libs_artifact(os_name, runtime)
        entries.append({"kind": kind, "package_path": path[len(prefix):], "repo_path": path,
                        "artifact": artifact, "artifact_path": path[len(prefix):], **digests(data),
                        "tested": tests_ran and path not in untested})
    return {"schema": 1, "os": os_name, "runtime": runtime, "run_id": run_id, "run_attempt": "1",
            "head_sha": head, "base_sha": base, "full_refresh": full_refresh, "tests_ran": tests_ran,
            "files": entries}


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
    Build and Test run 100 passed, with the Linux and Windows XRT overlays."""

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
                full_refresh: bool = False, overlays: tuple = (("linux", "xrt"), ("windows", "xrt")),
                new: dict | None = None, manifests: dict | None = None, jobs: list | None = None,
                **run_fields) -> None:
        head = head or self.head
        new = NEW if new is None else new
        self.api.runs[run_id] = {
            "id": int(run_id), "path": ds.WORKFLOW_PATH, "repository": {"full_name": REPO},
            "head_repository": {"full_name": REPO}, "head_sha": head, "head_branch": "feature",
            "status": "completed", "conclusion": "success", "event": event,
            "pull_requests": [{"number": PR}] if event == "pull_request" else [], **run_fields}
        self.api.jobs_by_run[run_id] = jobs if jobs is not None else (
            [{"name": "summary", "conclusion": "success"},
             {"name": "devkit-engines / Linux XRT", "conclusion": "success"},
             {"name": "devkit-xclbins / assemble", "conclusion": "success"}]
            + [{"name": ds.OVERLAYS[k], "conclusion": "success"} for k in overlays])
        for os_name, runtime in overlays:
            mine = {p: d for p, d in new.items() if p.startswith("src/xclbins/")
                    or (p.startswith(f"src/lib/{runtime}/") and (p.endswith(".so") == (os_name == "linux")))}
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
        self.assertEqual(self.api.downloads, [a["id"] for a in self.api.artifacts(RUN)
                                              if a["name"].startswith("devkit-overlay-manifest-")])

    def test_full_refresh_run(self):
        self.add_run("200", event="workflow_dispatch", full_refresh=True, overlays=tuple(ds.OVERLAYS),
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
        self.api.jobs_by_run[RUN][0]["conclusion"] = "failure"
        self.api.runs[RUN]["conclusion"] = "failure"
        self.refused("its summary job did not pass")

    def test_overlay_failed(self):
        self.api.jobs_by_run[RUN].append({"name": ds.OVERLAYS[("linux", "hrx")], "conclusion": "failure"})
        self.refused("Test Linux HRX dev-kit overlay did not pass")

    def test_build_failed(self):
        self.api.jobs_by_run[RUN].append({"name": "devkit-xclbins / build (3)", "conclusion": "failure"})
        self.refused("devkit-xclbins / build (3) did not pass")

    def test_bench_failure_alone_is_accepted(self):
        self.api.runs[RUN]["conclusion"] = "failure"
        self.api.jobs_by_run[RUN].append({"name": "devkit-bench / Bench lin-xrt m:1", "conclusion": "failure"})
        code, out, err = self.fetch()
        self.assertEqual(code, 0, out + err)

    def test_no_overlay_ran(self):
        self.api.jobs_by_run[RUN] = [j for j in self.api.jobs_by_run[RUN] if "overlay" not in j["name"]]
        self.refused("ran no dev-kit overlay test")

    def test_tests_did_not_run(self):
        self.api.artifacts_by_run[RUN] = []
        self.add_run(RUN, manifests={("linux", "xrt"): manifest(
            "linux", "xrt", {XRT_LIN: NEW[XRT_LIN]}, head=self.head, base=self.base, tests_ran=False)})
        self.refused("tested nothing")

    def test_manifest_expired(self):
        for a in self.api.artifacts(RUN):
            if a["name"].startswith("devkit-overlay-manifest-"):
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
        self.add_run(RUN, overlays=(("linux", "xrt"),))
        self.api.jobs_by_run[RUN].append({"name": ds.OVERLAYS[("windows", "xrt")], "conclusion": "skipped"})
        self.refused("did not both test the engines foo_npu")

    def test_dispatch_that_is_no_full_refresh(self):
        self.add_run("200", event="workflow_dispatch", overlays=tuple(ds.OVERLAYS))
        self.refused("is a dispatch but not a dev-kit full refresh", "--run", "200")

    def test_full_refresh_needs_all_four_overlays(self):
        self.add_run("200", event="workflow_dispatch", full_refresh=True)
        self.refused("not all four overlay jobs passed", "--run", "200")


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
        self.api.jobs_by_run[RUN][0]["conclusion"] = "failure"
        ok, why = self.verify()
        self.assertFalse(ok)
        self.assertIn("summary job did not pass", why)

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

    def test_hand_edit_warns(self):
        self.repo.commit({XRT_LIN: b"hand"})
        code, out = self.check()
        self.assertEqual(code, 0)
        self.assertIn(f"::warning::{XRT_LIN} is not the bytes a verified swap commit", out)
        self.assertIn("devkit_swap.py fetch", out)

    def test_enforced(self):
        self.repo.commit({XRT_LIN: b"hand"})
        with mock.patch.object(ds, "SWAP_ENFORCE", True):
            code, out = self.check()
        self.assertEqual(code, 1)
        self.assertIn(f"::error::{XRT_LIN}", out)

    def test_verified_swap_passes(self):
        code, out, err = self.fetch()
        self.assertEqual(code, 0, out + err)
        code, out = self.check()
        self.assertEqual(code, 0)
        self.assertNotIn("::warning::", out)
        self.assertIn("4 file(s) verified", out)

    def test_edit_after_the_swap(self):
        self.fetch()
        self.repo.commit({KERNEL: blob("later", 150)})
        code, out = self.check()
        self.assertIn(f"::warning::{KERNEL}", out)
        self.assertNotIn(f"::warning::{XRT_LIN}", out)

    def test_deletions(self):
        self.repo.git("rm", "-q", "src/xclbins/M-NPU2/extra.xclbin")
        self.repo.git("commit", "-q", "-m", "drop an unassigned kernel")
        code, out = self.check()
        self.assertNotIn("::warning::", out)
        self.repo.git("rm", "-q", KERNEL)
        self.repo.git("commit", "-q", "-m", "drop an assigned kernel")
        code, out = self.check()
        self.assertIn(f"::warning::{KERNEL} is deleted, but the dev kit still assigns or builds it", out)

    def test_third_party(self):
        self.repo.commit({"src/lib/xrt/libaiebu.a": b"new aiebu", "src/lib/zlib1.dll": b"new zlib"})
        code, out = self.check()
        self.assertNotIn("::warning::", out)
        self.assertIn("third-party", out)

    def test_fake_trailers_do_not_excuse(self):
        self.repo.commit({XRT_LIN: b"hand"}, f"x\n\n{ds.TRAILER_RUN}: {RUN}\n{ds.TRAILER_TESTED}: {self.head}")
        code, out = self.check()
        self.assertIn(f"::warning::{XRT_LIN}", out)
        self.assertIn("does not verify", out)


class Guide(SwapCase):
    def write_manifests(self, new: dict) -> Path:
        d = self.tmp / "manifests"
        for os_name, runtime in (("linux", "xrt"), ("windows", "xrt")):
            mine = {p: v for p, v in new.items() if p.startswith("src/xclbins/")
                    or p.endswith(".so") == (os_name == "linux")}
            m = manifest(os_name, runtime, mine, head=self.head, base=self.base)
            (d / ds.manifest_artifact(os_name, runtime)).mkdir(parents=True)
            (d / ds.manifest_artifact(os_name, runtime) / ds.MANIFEST_FILE).write_text(json.dumps(m))
        return d

    def guide(self, d: Path) -> str:
        code, out, _ = self.main("guide", "--manifests", str(d), "--head", self.head, "--run", RUN,
                                 "--repo-root", str(self.repo.path))
        self.assertEqual(code, 0)
        return out

    def test_command_when_files_differ(self):
        out = self.guide(self.write_manifests(NEW))
        self.assertIn(f"devkit_swap.py fetch --run {RUN}", out)
        self.assertIn("4 dev-kit file(s)", out)

    def test_nothing_when_the_branch_has_them(self):
        self.repo.commit(NEW)
        self.assertEqual(self.guide(self.write_manifests(NEW)), "")

    def test_nothing_without_manifests(self):
        self.assertEqual(self.guide(self.tmp / "none"), "")


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
        self.assertTrue(all(e.tested for e in m.files))

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
        bad_kernel = self.good()
        bad_kernel["files"][-1]["repo_path"] = "src/xclbins/attn.xclbin"
        with self.assertRaises(ds.ManifestError):
            self.parse(bad_kernel)
        with self.assertRaises(ds.ManifestError):
            ds.parse_manifest(b"{", "windows", "xrt")
        with self.assertRaises(ds.ManifestError):
            ds.parse_manifest(b" " * (ds.MAX_MANIFEST_BYTES + 1), "windows", "xrt")

    def test_untested_entries_are_left_out(self):
        lin = manifest("linux", "xrt", {KERNEL: b"k", "src/xclbins/N-NPU2/a.xclbin": b"n"}, head=self.HEAD,
                       base=self.BASE, untested=("src/xclbins/N-NPU2/a.xclbin",))
        entries, notes, errors = ds.plan_swap({("linux", "xrt"): self.parse(lin, "linux")}, RUN, self.HEAD)
        self.assertEqual((sorted(entries), errors), ([KERNEL], []))
        self.assertIn("not swapped", notes[0])

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


class OverlayManifest(unittest.TestCase):
    """devkit_overlay.py's side: what the overlay jobs record."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(rmtree, self.tmp)
        self.root = self.tmp / "pkg"
        self.libs = self.tmp / "libs"
        self.xclbins = self.tmp / "xclbins"
        for d in (self.root, self.libs, self.xclbins / "M-NPU2", self.xclbins / "Other-NPU2"):
            d.mkdir(parents=True)
        (self.libs / "foo_npu.dll").write_bytes(b"dll")
        (self.libs / "foo_npu.lib").write_bytes(b"lib")
        (self.libs / "libfoo_npu.so").write_bytes(b"so")
        (self.xclbins / "M-NPU2" / "attn.xclbin").write_bytes(b"k")
        (self.xclbins / "Other-NPU2" / "x.xclbin").write_bytes(b"o")
        self.kernels = [(self.xclbins / "M-NPU2" / "attn.xclbin", self.root / "xclbins" / "M-NPU2" / "attn.xclbin"),
                        (self.xclbins / "Other-NPU2" / "x.xclbin", self.root / "xclbins" / "Other-NPU2" / "x.xclbin")]

    def entries(self, os_name: str, pairs: list, tests_ran: bool = True) -> dict:
        out = do.manifest_entries(self.root, os_name, "xrt", pairs, self.libs, "devkit-libs-win-xrt"
                                  if os_name == "windows" else "devkit-libs-lin-xrt", self.xclbins,
                                  "devkit-xclbins", tests_ran, {"M-NPU2"})
        return {e["repo_path"]: e for e in out}

    def test_windows_records_the_import_library(self):
        got = self.entries("windows", [(self.libs / "foo_npu.dll", self.root / "foo_npu.dll")] + self.kernels)
        self.assertEqual(sorted(got), ["src/lib/xrt/foo_npu.dll", "src/lib/xrt/foo_npu.lib",
                                       "src/xclbins/M-NPU2/attn.xclbin", "src/xclbins/Other-NPU2/x.xclbin"])
        self.assertEqual(got["src/lib/xrt/foo_npu.dll"]["package_path"], "foo_npu.dll")
        self.assertIsNone(got["src/lib/xrt/foo_npu.lib"]["package_path"])
        self.assertEqual(got["src/lib/xrt/foo_npu.lib"]["sha256"], hashlib.sha256(b"lib").hexdigest())
        self.assertEqual(got["src/lib/xrt/foo_npu.lib"]["git_sha1"], digests(b"lib")["git_sha1"])
        # A kernel no tested model loads is recorded, but not as tested.
        self.assertTrue(got["src/xclbins/M-NPU2/attn.xclbin"]["tested"])
        self.assertFalse(got["src/xclbins/Other-NPU2/x.xclbin"]["tested"])

    def test_linux_paths(self):
        got = self.entries("linux", [(self.libs / "libfoo_npu.so", self.root / "lib" / "libfoo_npu.so")])
        self.assertEqual(got["src/lib/xrt/libfoo_npu.so"]["package_path"], "lib/libfoo_npu.so")
        self.assertEqual(got["src/lib/xrt/libfoo_npu.so"]["artifact_path"], "libfoo_npu.so")
        self.assertEqual(len(got), 1)

    def test_nothing_tested(self):
        got = self.entries("linux", self.kernels, tests_ran=False)
        self.assertFalse(any(e["tested"] for e in got.values()))

    def test_written_manifest_parses(self):
        # Kernels only (no ABI check needed): the overlay writes the
        # manifest, and devkit_swap.py reads it back.
        (self.root / "xclbins" / "M-NPU2").mkdir(parents=True)
        (self.root / "model_list.json").write_text(json.dumps(
            {"models": {"qwen3": {"0.6b": {"name": "M-NPU2", "details": {"family": "qwen3"}}}}}),
            encoding="utf-8")
        out_file = self.tmp / "m" / "manifest.json"
        buf = io.StringIO()
        env = {"GITHUB_RUN_ID": RUN, "GITHUB_RUN_ATTEMPT": "2"}
        with contextlib.redirect_stdout(buf), mock.patch.dict(os.environ, env):
            code = do.main(["--root", str(self.root), "--os", "linux", "--xclbins", str(self.xclbins),
                            "--xclbin-models", '["qwen3:0.6b"]', "--manifest", str(out_file), "--runtime", "hrx",
                            "--xclbins-artifact", "devkit-xclbins", "--head-sha", "a" * 40,
                            "--base-sha", "b" * 40])
        self.assertEqual(code, 0, buf.getvalue())
        raw = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertEqual((raw["run_id"], raw["run_attempt"], raw["tests_ran"], raw["full_refresh"]),
                         (RUN, "2", True, False))
        m = ds.parse_manifest(out_file.read_bytes(), "linux", "hrx")
        tested = {e.repo_path: e.tested for e in m.files}
        self.assertEqual(tested, {"src/xclbins/M-NPU2/attn.xclbin": True,
                                  "src/xclbins/Other-NPU2/x.xclbin": False})
        self.assertNotIn(b"\r\n", out_file.read_bytes())

    def test_manifest_needs_its_arguments(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            do.main(["--root", str(self.root), "--os", "linux", "--manifest", str(self.tmp / "m.json")])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            do.main(["--root", str(self.root), "--os", "linux", "--xclbins", str(self.xclbins),
                     "--manifest", str(self.tmp / "m.json"), "--runtime", "xrt"])


class Wiring(unittest.TestCase):
    """devkit_check.py's side of the swap."""

    def test_required_binaries(self):
        req = dc.required_binaries(dc.REPO_ROOT)
        self.assertIn("src/lib/hrx/qwen3_npu.lib", req)
        self.assertIn("src/lib/xrt/libq4_npu_eXpress.so", req)
        self.assertTrue(any(p.startswith("src/xclbins/") for p in req))
        self.assertNotIn("src/lib/xrt/libaiebu.a", req)
        self.assertNotIn("src/xclbins/MiniCPM-OCR-1.0-NPU2/conv.xclbin", req)


if __name__ == "__main__":
    unittest.main()
