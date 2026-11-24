#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Decides when the dev-kit auto-swap may swap in, then merge, an approved pull request.

    devkit_auto_swap.py plan    --repo <owner/name> --head-repo <owner/name> --head-branch <branch>
                                [--event-pr <n>] [--default-branch main] [--github-output]
                                [--expect-action swap|merge --expect-head <sha>]
    devkit_auto_swap.py gate    --repo-root <checkout> --base <rev> --head <rev> [--github-output]
    devkit_auto_swap.py comment --repo <owner/name> --pr <n> --bot-login <login> --title <text>
                                [--body-file <file>]

.github/workflows/devkit-auto-swap.yml runs this from the default branch
(workflow_run: it never runs a pull request's version of this file). The
swap itself is devkit_swap.py fetch, unchanged, with every check it makes
(flm-dev-kit/AGENTS.md, "Prebuilt binaries"); this only decides whether a
maintainer has accepted the pull request, so that fetch may run for its
author, and when the swapped pull request may be merged. The merge is the
bot's own (not GitHub's auto-merge, which merges at once where main
requires no check, and stays on after later pushes): only a head that is
a swap commit whose Build and Test run passed.

plan (read-only token: pull-requests: read, actions: read, contents:
read), else action=none with the reason:
  - the pull request: the one open pull request of this repository (not a
    fork's) with the event's head branch, not a draft, based on the default
    branch. The event's pull request number is a hint only; the rest is read
    from the API
  - the run: its head's newest Build and Test run of the kinds fetch takes
    (pull_request, or a full-refresh workflow_dispatch on its branch),
    completed; a pull_request run must name this pull request with the
    default branch as its base, at a commit of it (a pull_request run's
    workflow comes from the merge with its base)
  - accepted: an APPROVED review by someone other than the author with
    write, maintain or admin permission, and no such person's latest
    review requesting changes (a failed permission lookup is an error,
    never "no permission")
  action=swap: the run passed every job and failed only at summary's
    devkit_swap.PENDING_STEP (check_jobs), approved at exactly the head
  action=merge: the run is a pull_request run that passed and the head
    looks like a swap commit (both trailers, its only parent the tested
    commit), approved at the tested commit or at the swap commit. Trailers
    are the author's text: the merge job runs main's devkit_swap.py verify
    on the head (the commit's bytes are exactly its run's tested ones, no
    swap CI change, main not moved) before it merges
plan writes pr, run, head and branch. With --expect-action and
--expect-head (the swap job before it pushes, the merge job before it
merges) it exits 1 unless it decides the same again.

gate (on a blob:none checkout of the head): action=split when the pull
request changes devkit_swap.SWAP_CI_PATHS against its merge base with
--base (verify would refuse the swap: the pull request has to be split),
with the paths; else action=swap.

comment (the bot account's token): one sticky comment, the newest by --bot-login
holding MARKER, created or replaced: --title (the workflow's own text)
and the --body-file, untrusted (fetch's messages can quote artifact names),
in a code fence longer than any backtick run in it, with @ defused.

Standard library only; the GitHub calls go through Api, which the tests
replace.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import devkit_swap as ds  # noqa: E402

MARKER = "<!-- devkit-auto-swap -->"
WRITERS = ("admin", "maintain", "write")
BRANCH = re.compile(r"^[A-Za-z0-9._/+-]{1,200}$")
REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MAX_BODY = 60000


class Api(ds.Api):
    """devkit_swap.Api plus the review, permission and comment calls."""

    def reviews(self, number: int) -> list:
        return self._paged(f"pulls/{int(number)}/reviews", "")

    def permission(self, login: str) -> str:
        """The login's permission; "none" only when GitHub answers 404 (no
        collaborator). Any other failure raises."""
        try:
            data = self._get(f"collaborators/{urllib.request.quote(login, safe='')}/permission")
        except ds.SwapError as exc:
            if "HTTP 404" in str(exc):
                return "none"
            raise
        return str((data or {}).get("permission") or "none")

    def commit(self, sha: str) -> dict:
        return self._get(f"commits/{sha}")

    def compare_status(self, base: str, head: str) -> str:
        """ahead, identical, behind or diverged: head against base."""
        return str((self._get(f"compare/{base}...{urllib.request.quote(head, safe='')}", {"per_page": 1}) or {})
                   .get("status") or "")

    def issue_comments(self, number: int) -> list:
        return self._paged(f"issues/{int(number)}/comments", "")

    def _send(self, method: str, path: str, body: dict) -> dict:
        req = self._request(path)
        req.method = method
        req.data = json.dumps(body).encode("utf-8")
        req.add_header("Content-Type", "application/json")
        opener = urllib.request.build_opener(ds._NoRedirect)
        try:
            with opener.open(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            raise ds.SwapError(f"GitHub API {method} {path}: HTTP {exc.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ds.SwapError(f"GitHub API {method} {path}: {exc}") from None

    def create_comment(self, number: int, body: str) -> dict:
        return self._send("POST", f"issues/{int(number)}/comments", {"body": body})

    def update_comment(self, comment_id: int, body: str) -> dict:
        return self._send("PATCH", f"issues/comments/{int(comment_id)}", {"body": body})


# ── plan ─────────────────────────────────────────────────────────────────────

def find_pull(api, repo: str, head_repo: str, branch: str, event_pr: Optional[int]) -> tuple[Optional[dict], str]:
    """(the pull request, "") or (None, why not). Every open pull request
    with this head branch counts, whatever its base: two make it ambiguous."""
    if head_repo != repo:
        return None, f"the run is of {head_repo or 'a deleted fork'}, not {repo}: forks get no swap"
    if not BRANCH.match(branch or ""):
        return None, f"head branch {branch!r} is no branch name this handles"
    pulls = [p for p in api.pulls_for_branch(repo.split("/")[0], branch) if isinstance(p, dict)
             and (p.get("head") or {}).get("ref") == branch]
    if len(pulls) != 1:
        return None, f"{len(pulls)} open pull requests of {repo} have head {branch}"
    if event_pr is not None and pulls[0].get("number") != event_pr:
        return None, f"the event names pull request #{event_pr}, the branch's is #{pulls[0].get('number')}"
    # The listing can lag: read the pull request itself.
    return api.pull(int(pulls[0]["number"])), ""


def accepted(api, pull: dict, commits: set) -> tuple[bool, str]:
    """Whether a writer other than the author approved one of commits and no
    writer's latest review requests changes. A permission lookup that fails
    other than with 404 raises: it must not drop a change request."""
    author = str((pull.get("user") or {}).get("login") or "")
    latest: dict[str, dict] = {}
    for r in api.reviews(int(pull["number"])):
        if not isinstance(r, dict):
            continue
        login = str((r.get("user") or {}).get("login") or "")
        # COMMENTED and PENDING change no reviewer's verdict.
        if login and r.get("state") in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED"):
            latest[login] = r
    perms: dict[str, str] = {}

    def writer(login: str) -> bool:
        if login not in perms:
            perms[login] = api.permission(login)
        return perms[login] in WRITERS

    blocking = sorted(l for l, r in latest.items() if r.get("state") == "CHANGES_REQUESTED" and writer(l))
    if blocking:
        return False, f"changes requested by {', '.join(blocking)}"
    approvers = sorted(l for l, r in latest.items() if r.get("state") == "APPROVED" and l != author
                       and r.get("commit_id") in commits and writer(l))
    shown = "/".join(sorted(c[:10] for c in commits))
    if not approvers:
        stale = sorted(l for l, r in latest.items() if r.get("state") == "APPROVED" and r.get("commit_id") not in commits)
        return False, (f"no approval of {shown} by a maintainer other than the author"
                       + (f" (approvals of other commits: {', '.join(stale)})" if stale else ""))
    return True, f"approved at {shown} by {', '.join(approvers)}"


def newest_run(api, head: str) -> tuple[Optional[dict], str]:
    """head's newest Build and Test run of the kinds fetch takes (a pull
    request's, or a full refresh dispatched on its branch), as find_run
    picks it."""
    runs = [r for r in api.runs_for_sha(head) if isinstance(r, dict)
            and r.get("path") == ds.WORKFLOW_PATH and r.get("event") in ("pull_request", "workflow_dispatch")]
    if not runs:
        return None, f"no Build and Test run of {head[:10]}"
    runs.sort(key=lambda r: (str(r.get("created_at")), int(r.get("id") or 0)), reverse=True)
    run = runs[0]
    if not ds.RUN_ID.match(str(run.get("id"))):
        return None, f"run id {run.get('id')!r} is no run id"
    if run.get("status") != "completed":
        return None, f"Build and Test run {run.get('id')} is {run.get('status')}"
    return run, ""


def run_base_ok(api, run: dict, pull: dict, repo: str, default_branch: str) -> str:
    """Why run is not of this pull request against default_branch ("" when it
    is). A pull_request run's workflow comes from the merge with its base,
    so that base must be the default branch, at a commit of it. A dispatch
    runs the branch's own workflow (gate checks the swap CI it changes)."""
    if run.get("head_branch") != (pull.get("head") or {}).get("ref") \
            or ((run.get("head_repository") or {}).get("full_name") or repo) != repo:
        return f"run {run.get('id')} is not of the pull request's branch"
    if run.get("event") != "pull_request":
        return ""
    mine = [p for p in run.get("pull_requests") or [] if isinstance(p, dict) and p.get("number") == pull.get("number")]
    if len(mine) != 1:
        return f"run {run.get('id')} does not name pull request #{pull.get('number')}"
    base = mine[0].get("base") or {}
    if base.get("ref") != default_branch or not ds.SHA1.match(str(base.get("sha") or "")):
        return f"run {run.get('id')} ran against {base.get('ref')}, not {default_branch}"
    if api.compare_status(str(base["sha"]), default_branch) not in ("ahead", "identical"):
        return f"run {run.get('id')}'s base {str(base['sha'])[:10]} is no commit of {default_branch}"
    return ""


def asks_for_swap(api, run: dict) -> str:
    """Why run does not pass everything but ask for a swap ("" when it does)."""
    run_id = str(run.get("id"))
    jobs = api.jobs(run_id)
    why, _ = ds.check_jobs(jobs)
    if why:
        return f"Build and Test run {run_id}: " + "; ".join(why)
    summary = next((j for j in jobs if isinstance(j, dict) and j.get("name") == ds.SUMMARY_JOB), {})
    if not any(isinstance(st, dict) and st.get("name") == ds.PENDING_STEP and st.get("conclusion") == "failure"
               for st in summary.get("steps") or []):
        return f"Build and Test run {run_id} asks for no swap"
    return ""


TRAILER = re.compile(r"^([A-Za-z-]+):\s*(\S+)\s*$")


def swap_parent(api, sha: str) -> Optional[str]:
    """The tested commit when sha is a swap commit on it (exactly one
    Devkit-Swap-Run and one Devkit-Swap-Tested trailer, the tested commit
    its only parent); None otherwise. verify, in the run that passed on
    sha, checked its bytes."""
    data = api.commit(sha)
    parents = [p.get("sha") for p in data.get("parents") or [] if isinstance(p, dict)]
    message = str((data.get("commit") or {}).get("message") or "")
    last = message.strip().split("\n\n")[-1].splitlines()
    found: dict[str, list[str]] = {}
    for line in last:
        m = TRAILER.match(line)
        if m:
            found.setdefault(m.group(1), []).append(m.group(2))
    run, tested = found.get(ds.TRAILER_RUN, []), found.get(ds.TRAILER_TESTED, [])
    if len(run) != 1 or len(tested) != 1 or not ds.RUN_ID.match(run[0]) or not ds.SHA1.match(tested[0]):
        return None
    return tested[0] if parents == [tested[0]] else None


def plan(api, repo: str, head_repo: str, branch: str, event_pr: Optional[int],
         default_branch: str = "main") -> dict:
    pull, why = find_pull(api, repo, head_repo, branch, event_pr)
    if pull is None:
        return {"action": "none", "reason": why}
    number = int(pull.get("number") or 0)
    head = str((pull.get("head") or {}).get("sha") or "")
    out = {"action": "none", "pr": str(number), "head": head, "branch": branch}
    if pull.get("state") != "open" or pull.get("draft"):
        return {**out, "reason": f"pull request #{number} is not open or is a draft"}
    if ((pull.get("head") or {}).get("repo") or {}).get("full_name") != repo:
        return {**out, "reason": f"pull request #{number} comes from a fork"}
    if (pull.get("base") or {}).get("ref") != default_branch:
        return {**out, "reason": f"pull request #{number} is not based on {default_branch}"}
    if not ds.SHA1.match(head):
        return {**out, "reason": f"pull request #{number} has no head commit"}
    run, why = newest_run(api, head)
    if run is None:
        return {**out, "reason": why}
    out["run"] = str(run["id"])
    why = run_base_ok(api, run, pull, repo, default_branch)
    if why:
        return {**out, "reason": why}
    if run.get("conclusion") == "success":
        # Passed: merge when the head is a swap commit and its tested
        # commit (or the swap itself) is approved. Only a pull_request run
        # runs verify and check; the merge job verifies the head again with
        # main's devkit_swap.py before it merges.
        if run.get("event") != "pull_request":
            return {**out, "reason": f"Build and Test run {run['id']} is a {run.get('event')} run: a merge needs "
                                     "a pull_request run, which verifies the swap"}
        tested = swap_parent(api, head)
        if tested is None:
            return {**out, "reason": f"Build and Test run {run['id']} passed and {head[:10]} is no swap commit: "
                                     "nothing to swap or merge"}
        ok, why = accepted(api, pull, {tested, head})
        if not ok:
            return {**out, "reason": why}
        return {**out, "action": "merge", "reason": f"swap commit {head[:10]} passed run {run['id']}; {why}"}
    why = asks_for_swap(api, run)
    if why:
        return {**out, "reason": why}
    ok, why = accepted(api, pull, {head})
    if not ok:
        return {**out, "reason": why}
    return {**out, "action": "swap", "reason": f"run {run['id']} asks for a swap; {why}"}


# ── gate ─────────────────────────────────────────────────────────────────────

def gate(git: ds.Git, base: str, head: str) -> dict:
    ci = ds.swap_ci_changed(git, base, head)
    if ci:
        return {"action": "split", "ci_paths": ",".join(ci),
                "reason": f"the pull request changes the CI that checks swaps ({', '.join(ci)})"}
    return {"action": "swap", "ci_paths": "", "reason": "no swap CI change"}


# ── comment ──────────────────────────────────────────────────────────────────

def fenced(text: str) -> str:
    """text, @ defused, in a code fence no backtick run in it can close."""
    text = text[:MAX_BODY].replace("@", "@​")
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def comment(api, number: int, bot_login: str, title: str, detail: str = "") -> None:
    full = f"{MARKER}\n{title}\n" + (f"\n{fenced(detail.strip())}\n" if detail.strip() else "")
    mine = [c for c in api.issue_comments(number) if isinstance(c, dict)
            and (c.get("user") or {}).get("login") == bot_login and MARKER in str(c.get("body") or "")]
    if mine:
        api.update_comment(int(max(mine, key=lambda c: int(c.get("id") or 0))["id"]), full)
    else:
        api.create_comment(number, full)


def write_outputs(values: dict, github_output: bool) -> None:
    for k, v in values.items():
        print(f"{k}={v}")
    if github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            for k, v in values.items():
                fh.write(f"{k}={str(v).replace(chr(10), ' ')}\n")


def main(argv: Optional[list[str]] = None, api=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--repo", required=True)
    p.add_argument("--head-repo", required=True)
    p.add_argument("--head-branch", required=True)
    p.add_argument("--event-pr", default="")
    p.add_argument("--default-branch", default="main")
    p.add_argument("--github-output", action="store_true")
    p.add_argument("--expect-action", choices=("swap", "merge"), default=None)
    p.add_argument("--expect-head", default="")
    p = sub.add_parser("gate")
    p.add_argument("--repo-root", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--github-output", action="store_true")
    p = sub.add_parser("comment")
    p.add_argument("--repo", required=True)
    p.add_argument("--pr", type=int, required=True)
    p.add_argument("--bot-login", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--body-file", type=Path, default=None)
    args = parser.parse_args(argv)
    if getattr(args, "github_output", False) and not os.environ.get("GITHUB_OUTPUT"):
        parser.error("--github-output needs $GITHUB_OUTPUT")
    try:
        if args.cmd == "gate":
            write_outputs(gate(ds.Git(Path(args.repo_root)), args.base, args.head), args.github_output)
            return 0
        if not REPO.match(args.repo):
            parser.error(f"--repo {args.repo!r} is no owner/name")
        api = api or Api(args.repo, os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "")
        if args.cmd == "plan":
            event_pr = int(args.event_pr) if args.event_pr.isdigit() else None
            out = plan(api, args.repo, args.head_repo, args.head_branch, event_pr, args.default_branch)
            write_outputs(out, args.github_output)
            if args.expect_action and (out["action"], out.get("head")) != (args.expect_action, args.expect_head):
                print(f"::error::devkit_auto_swap plan: no longer {args.expect_action} at "
                      f"{args.expect_head[:10]}: {out.get('reason')}")
                return 1
            return 0
        detail = args.body_file.read_text(encoding="utf-8", errors="replace") if args.body_file else ""
        comment(api, args.pr, args.bot_login, args.title, detail)
        return 0
    except ds.SwapError as exc:
        print(f"::error::devkit_auto_swap {args.cmd}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
