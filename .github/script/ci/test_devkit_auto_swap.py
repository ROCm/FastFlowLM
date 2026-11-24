# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for devkit_auto_swap.py:
python3 -m unittest discover -s ci -p 'test_*.py' -t ci

The GitHub API is FakeApi (in-memory pull requests, reviews, runs, jobs and
comments); gate runs on a real throwaway git repository."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import devkit_auto_swap as das  # noqa: E402
import devkit_swap as ds  # noqa: E402

REPO = "o/r"
HEAD = "a" * 40
OLD = "b" * 40
BASE = "c" * 40
SWAP = "d" * 40
BRANCH = "feat/kernel"


def summary_job(pending: bool) -> dict:
    steps = [{"name": "Gate and verdict", "conclusion": "success"}, {"name": "Dev-kit checks", "conclusion": "success"},
             {"name": ds.PENDING_STEP, "conclusion": "failure" if pending else "success"}]
    return {"name": ds.SUMMARY_JOB, "conclusion": "failure" if pending else "success", "steps": steps}


def jobs(pending: bool = True, extra_failure: bool = False) -> list:
    out = [{"name": ds.STAGE_JOB, "conclusion": "success"}, summary_job(pending)]
    out += [{"name": name, "conclusion": "success"} for name in ds.TEST_JOBS.values()]
    if extra_failure:
        out.append({"name": "Test Windows HIPEP (stx-halo)", "conclusion": "failure"})
    return out


class FakeApi:
    def __init__(self):
        self.pulls = {7: {"number": 7, "state": "open", "draft": False, "user": {"login": "author"},
                          "head": {"ref": BRANCH, "sha": HEAD, "repo": {"full_name": REPO}},
                          "base": {"ref": "main"}}}
        self.runs = [{"id": 100, "path": ds.WORKFLOW_PATH, "event": "pull_request", "head_sha": HEAD,
                      "head_branch": BRANCH, "head_repository": {"full_name": REPO},
                      "pull_requests": [{"number": 7, "base": {"ref": "main", "sha": BASE}}],
                      "status": "completed", "conclusion": "failure", "created_at": "2026-11-21T10:00:00Z"}]
        self.commits = {}
        self.main_commits = {BASE}
        self.jobs_by_run = {"100": jobs()}
        self.review_list = [{"user": {"login": "maint"}, "state": "APPROVED", "commit_id": HEAD}]
        self.perms = {"maint": "write", "maint2": "admin", "reader": "read", "author": "write"}
        self.comments: list = []
        self.next_id = 1

    def pull(self, number):
        return self.pulls[number]

    def pulls_for_branch(self, owner, branch):
        return [p for p in self.pulls.values() if p["head"]["ref"] == branch and p["state"] == "open"]

    def runs_for_sha(self, sha):
        return [r for r in self.runs if r["head_sha"] == sha]

    def jobs(self, run_id):
        return self.jobs_by_run.get(str(run_id), [])

    def reviews(self, number):
        return self.review_list

    def permission(self, login):
        if self.perms.get(login) == "error":
            raise ds.SwapError("GitHub API collaborators: HTTP 502")
        return self.perms.get(login, "none")

    def commit(self, sha):
        return self.commits.get(sha, {"sha": sha, "parents": [{"sha": OLD}], "commit": {"message": "work"}})

    def compare_status(self, base, head):
        return "ahead" if base in self.main_commits else "diverged"

    def issue_comments(self, number):
        return self.comments

    def create_comment(self, number, body):
        self.comments.append({"id": self.next_id, "user": {"login": "swapbot[bot]"}, "body": body})
        self.next_id += 1

    def update_comment(self, comment_id, body):
        next(c for c in self.comments if c["id"] == comment_id)["body"] = body


def plan(api, **kw):
    args = {"repo": REPO, "head_repo": REPO, "branch": BRANCH, "event_pr": 7}
    args.update(kw)
    return das.plan(api, args["repo"], args["head_repo"], args["branch"], args["event_pr"])


class Plan(unittest.TestCase):
    def test_approved_pending_swaps(self):
        out = plan(FakeApi())
        self.assertEqual(out["action"], "swap", out)
        self.assertEqual((out["pr"], out["run"], out["head"], out["branch"]), ("7", "100", HEAD, BRANCH))
        self.assertIn("maint", out["reason"])

    def test_event_pr_is_a_hint(self):
        self.assertEqual(plan(FakeApi(), event_pr=None)["action"], "swap")
        self.assertEqual(plan(FakeApi(), event_pr=8)["action"], "none")

    def test_fork_gets_nothing(self):
        out = plan(FakeApi(), head_repo="fork/r")
        self.assertEqual(out["action"], "none")
        self.assertIn("fork", out["reason"])
        api = FakeApi()
        api.pulls[7]["head"]["repo"]["full_name"] = "fork/r"
        self.assertEqual(plan(api)["action"], "none")

    def test_bad_branch_name(self):
        self.assertEqual(plan(FakeApi(), branch="x;rm -rf /")["action"], "none")

    def test_draft_closed_or_other_base(self):
        for change in ({"draft": True}, {"state": "closed"}, {"base": {"ref": "release"}}):
            api = FakeApi()
            api.pulls[7].update(change)
            self.assertEqual(plan(api)["action"], "none", change)

    def test_no_approval(self):
        api = FakeApi()
        api.review_list = []
        out = plan(api)
        self.assertEqual(out["action"], "none")
        self.assertIn("no approval", out["reason"])

    def test_approval_of_an_older_commit(self):
        api = FakeApi()
        api.review_list = [{"user": {"login": "maint"}, "state": "APPROVED", "commit_id": OLD}]
        out = plan(api)
        self.assertEqual(out["action"], "none")
        self.assertIn("other commits: maint", out["reason"])

    def test_approval_by_a_reader_or_the_author(self):
        for login in ("reader", "author", "stranger"):
            api = FakeApi()
            api.review_list = [{"user": {"login": login}, "state": "APPROVED", "commit_id": HEAD}]
            self.assertEqual(plan(api)["action"], "none", login)

    def test_changes_requested_blocks(self):
        api = FakeApi()
        api.review_list.append({"user": {"login": "maint2"}, "state": "CHANGES_REQUESTED", "commit_id": HEAD})
        out = plan(api)
        self.assertEqual(out["action"], "none")
        self.assertIn("changes requested by maint2", out["reason"])

    def test_a_readers_change_request_does_not_block(self):
        api = FakeApi()
        api.review_list.append({"user": {"login": "reader"}, "state": "CHANGES_REQUESTED", "commit_id": HEAD})
        self.assertEqual(plan(api)["action"], "swap")

    def test_latest_review_counts(self):
        api = FakeApi()
        api.review_list = [{"user": {"login": "maint"}, "state": "APPROVED", "commit_id": HEAD},
                           {"user": {"login": "maint"}, "state": "COMMENTED", "commit_id": HEAD},
                           {"user": {"login": "maint"}, "state": "DISMISSED", "commit_id": HEAD}]
        self.assertEqual(plan(api)["action"], "none")
        api.review_list = [{"user": {"login": "maint"}, "state": "CHANGES_REQUESTED", "commit_id": OLD},
                           {"user": {"login": "maint"}, "state": "APPROVED", "commit_id": HEAD}]
        self.assertEqual(plan(api)["action"], "swap")

    def test_run_not_pending(self):
        api = FakeApi()
        api.runs[0]["conclusion"] = "success"
        api.jobs_by_run["100"] = jobs(pending=False)
        self.assertEqual(plan(api)["action"], "none")

    def test_run_with_another_failure(self):
        api = FakeApi()
        api.jobs_by_run["100"] = jobs(extra_failure=True)
        out = plan(api)
        self.assertEqual(out["action"], "none")
        self.assertIn("did not pass", out["reason"])

    def test_summary_failed_elsewhere(self):
        api = FakeApi()
        api.jobs_by_run["100"][1]["steps"][1]["conclusion"] = "failure"
        self.assertEqual(plan(api)["action"], "none")

    def test_run_in_progress_or_missing(self):
        api = FakeApi()
        api.runs[0]["status"] = "in_progress"
        self.assertEqual(plan(api)["action"], "none")
        api.runs = []
        self.assertEqual(plan(api)["action"], "none")

    def test_newest_run_decides(self):
        api = FakeApi()
        api.runs.append({**api.runs[0], "id": 101, "created_at": "2026-11-21T11:00:00Z", "status": "in_progress"})
        out = plan(api)
        self.assertEqual(out["action"], "none")
        self.assertIn("101", out["reason"])

    def test_other_workflows_and_events_ignored(self):
        api = FakeApi()
        api.runs = [{**api.runs[0], "event": "push"}, {**api.runs[0], "id": 5, "path": ".github/workflows/x.yml"}]
        self.assertEqual(plan(api)["action"], "none")

    def test_full_refresh_dispatch_is_the_newest_run(self):
        api = FakeApi()
        api.runs.append({**api.runs[0], "id": 101, "event": "workflow_dispatch", "pull_requests": [],
                         "created_at": "2026-11-21T11:00:00Z"})
        api.jobs_by_run["101"] = jobs()
        out = plan(api)
        self.assertEqual((out["action"], out["run"]), ("swap", "101"))
        api.runs[1]["status"] = "in_progress"
        self.assertEqual(plan(api)["action"], "none")

    def test_run_against_another_base(self):
        for prs in ([], [{"number": 8, "base": {"ref": "main", "sha": BASE}}],
                    [{"number": 7, "base": {"ref": "evil", "sha": BASE}}],
                    [{"number": 7, "base": {"ref": "main", "sha": OLD}}]):
            api = FakeApi()
            api.runs[0]["pull_requests"] = prs
            self.assertEqual(plan(api)["action"], "none", prs)

    def test_second_pull_request_from_the_branch(self):
        api = FakeApi()
        api.pulls[8] = {**api.pulls[7], "number": 8, "base": {"ref": "evil"}}
        out = plan(api)
        self.assertEqual(out["action"], "none")
        self.assertIn("2 open pull requests", out["reason"])

    def test_permission_error_is_no_silent_none(self):
        api = FakeApi()
        api.perms["maint2"] = "error"
        api.review_list.append({"user": {"login": "maint2"}, "state": "CHANGES_REQUESTED", "commit_id": HEAD})
        with self.assertRaises(ds.SwapError):
            plan(api)


def swap_commit(api, tested=HEAD, parents=None, trailers=None):
    """The pull request's head becomes a swap commit on tested whose run passed."""
    lines = trailers if trailers is not None else [f"{ds.TRAILER_RUN}: 100", f"{ds.TRAILER_TESTED}: {tested}"]
    api.commits[SWAP] = {"sha": SWAP, "parents": [{"sha": p} for p in (parents or [tested])],
                         "commit": {"message": "devkit: swap in\n\n  src/lib/x\n\n" + "\n".join(lines)}}
    api.pulls[7]["head"]["sha"] = SWAP
    api.runs.append({**api.runs[0], "id": 200, "head_sha": SWAP, "conclusion": "success",
                     "created_at": "2026-11-21T12:00:00Z"})
    api.jobs_by_run["200"] = jobs(pending=False)


class Merge(unittest.TestCase):
    def test_passed_swap_approved_at_tested_merges(self):
        api = FakeApi()
        swap_commit(api)
        out = plan(api)
        self.assertEqual((out["action"], out["head"], out["run"]), ("merge", SWAP, "200"), out)

    def test_approval_of_the_swap_itself(self):
        api = FakeApi()
        swap_commit(api)
        api.review_list = [{"user": {"login": "maint"}, "state": "APPROVED", "commit_id": SWAP}]
        self.assertEqual(plan(api)["action"], "merge")

    def test_not_approved(self):
        api = FakeApi()
        swap_commit(api)
        api.review_list = [{"user": {"login": "maint"}, "state": "APPROVED", "commit_id": OLD}]
        self.assertEqual(plan(api)["action"], "none")

    def test_push_after_the_swap_is_not_merged(self):
        api = FakeApi()
        swap_commit(api)
        later = "e" * 40
        api.pulls[7]["head"]["sha"] = later
        api.commits[later] = {"sha": later, "parents": [{"sha": SWAP}], "commit": {"message": "review fix"}}
        api.runs.append({**api.runs[0], "id": 300, "head_sha": later, "conclusion": "success"})
        out = plan(api)
        self.assertEqual(out["action"], "none")
        self.assertIn("no swap commit", out["reason"])

    def test_swap_run_failed(self):
        api = FakeApi()
        swap_commit(api)
        api.runs[-1]["conclusion"] = "failure"
        api.jobs_by_run["200"] = jobs(pending=False, extra_failure=True)
        self.assertEqual(plan(api)["action"], "none")

    def test_malformed_swap_commits(self):
        for kw in ({"parents": [OLD]}, {"parents": [HEAD, OLD]}, {"trailers": [f"{ds.TRAILER_RUN}: 100"]},
                   {"trailers": [f"{ds.TRAILER_RUN}: 100", f"{ds.TRAILER_RUN}: 101", f"{ds.TRAILER_TESTED}: {HEAD}"]},
                   {"trailers": [f"{ds.TRAILER_RUN}: x", f"{ds.TRAILER_TESTED}: {HEAD}"]}):
            api = FakeApi()
            swap_commit(api, **kw)
            self.assertEqual(plan(api)["action"], "none", kw)

    def test_dispatch_run_of_the_swap_is_no_merge(self):
        api = FakeApi()
        swap_commit(api)
        api.runs[-1].update({"event": "workflow_dispatch", "pull_requests": []})
        out = plan(api)
        self.assertEqual(out["action"], "none")
        self.assertIn("workflow_dispatch", out["reason"])

    def test_changes_requested_blocks_merge(self):
        api = FakeApi()
        swap_commit(api)
        api.review_list.append({"user": {"login": "maint2"}, "state": "CHANGES_REQUESTED", "commit_id": SWAP})
        self.assertEqual(plan(api)["action"], "none")


class Recheck(unittest.TestCase):
    def run_main(self, api, *extra):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return das.main(["plan", "--repo", REPO, "--head-repo", REPO, "--head-branch", BRANCH, *extra], api=api)

    def test_same_decision_passes(self):
        self.assertEqual(self.run_main(FakeApi(), "--expect-action", "swap", "--expect-head", HEAD), 0)

    def test_changed_decision_fails(self):
        api = FakeApi()
        api.review_list.append({"user": {"login": "maint2"}, "state": "CHANGES_REQUESTED", "commit_id": HEAD})
        self.assertEqual(self.run_main(api, "--expect-action", "swap", "--expect-head", HEAD), 1)
        self.assertEqual(self.run_main(FakeApi(), "--expect-action", "swap", "--expect-head", OLD), 1)


class Gate(unittest.TestCase):
    def git(self, root, *args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.git(self.root, "init", "-q", "-b", "main")
        self.git(self.root, "config", "user.email", "t@example.com")
        self.git(self.root, "config", "user.name", "t")
        (self.root / "README").write_text("x\n")
        self.git(self.root, "add", ".")
        self.git(self.root, "commit", "-qm", "base")
        self.git(self.root, "checkout", "-qb", "pr")

    def tearDown(self):
        self.tmp.cleanup()

    def commit(self, rel):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("changed\n")
        self.git(self.root, "add", ".")
        self.git(self.root, "commit", "-qm", rel)

    def test_source_only_swaps(self):
        self.commit("flm-dev-kit/FLM_DLL/detail/foo/foo.cpp")
        out = das.gate(ds.Git(self.root), "main", "HEAD")
        self.assertEqual(out["action"], "swap")

    def test_swap_ci_change_splits(self):
        self.commit("flm-dev-kit/FLM_DLL/detail/foo/foo.cpp")
        self.commit(".github/script/ci/devkit_stage.py")
        out = das.gate(ds.Git(self.root), "main", "HEAD")
        self.assertEqual(out["action"], "split")
        self.assertEqual(out["ci_paths"], ".github/script/ci/devkit_stage.py")

    def test_main_ci_change_is_not_the_pull_requests(self):
        self.commit("flm-dev-kit/FLM_DLL/detail/foo/foo.cpp")
        self.git(self.root, "checkout", "-q", "main")
        self.commit(".github/script/ci/devkit_stage.py")
        self.git(self.root, "checkout", "-q", "pr")
        self.assertEqual(das.gate(ds.Git(self.root), "main", "HEAD")["action"], "swap")


class Comment(unittest.TestCase):
    def test_sticky(self):
        api = FakeApi()
        das.comment(api, 7, "swapbot[bot]", "first")
        das.comment(api, 7, "swapbot[bot]", "second", "detail")
        self.assertEqual(len(api.comments), 1)
        self.assertIn("second", api.comments[0]["body"])
        self.assertTrue(api.comments[0]["body"].startswith(das.MARKER))

    def test_other_authors_marker_is_not_reused(self):
        api = FakeApi()
        api.comments.append({"id": 50, "user": {"login": "mallory"}, "body": das.MARKER})
        das.comment(api, 7, "swapbot[bot]", "mine")
        self.assertEqual(len(api.comments), 2)
        self.assertEqual(api.comments[0]["body"], das.MARKER)

    def test_detail_is_fenced_and_defused(self):
        api = FakeApi()
        das.comment(api, 7, "swapbot[bot]", "title", "````\n@everyone </details> ``` done")
        body = api.comments[0]["body"]
        self.assertIn("`````\n", body)
        self.assertNotIn("@everyone", body)


if __name__ == "__main__":
    unittest.main()
