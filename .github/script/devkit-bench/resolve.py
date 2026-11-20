#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Decides what devkit-bench.yml benches, on a GitHub-hosted runner.

    resolve.py find-build    --repo R --event pull_request|schedule|any
                             [--head-sha S] [--head-repo R] [--run-id N]
                             [--require-success] [--wait-minutes 200]
                             [--artifacts-out artifacts.json]
    resolve.py find-baseline --repo R
    resolve.py plan          --plan devkit-bench-plan.json
                             --artifacts artifacts.json --mode pr|nightly|dispatch
                             [--runtimes lin-xrt,lin-hrx] [--models a,b]
                             [--baseline baseline.json] [--force]

find-build finds the Build and Test run whose artifacts the bench uses and
waits (polling) for it to complete. The run must be of
.github/workflows/build-and-test.yml in this repository, for exactly this
head commit and head repository, of the given event (schedule: on main),
the newest such run; a run id given (dispatch) is checked the same way.
Nightly (schedule, no --head-sha) it is the newest completed scheduled run
on main. It
must have uploaded devkit-bench-plan (its summary job does when the change
benches, and nightly) and the artifact must not have expired. A pull
request also needs the run to have succeeded (--require-success); a
nightly run may have failed on a product test, as long as the plan says
the builds the bench needs succeeded. Outputs run-id, ok and reason; the
run's artifact list goes to --artifacts-out.

find-baseline finds the newest unexpired devkit-bench-baseline artifact
uploaded by a scheduled devkit-bench.yml run on main of this repository
that completed (success, or failure: see baseline_run_ok), not a fork's or
a branch's run with the same workflow name.
Outputs baseline-run-id ("" when there is none).

plan reads the plan (validated field by field: it is an artifact), picks
the models (select_models.py) and builds the bench matrix: one entry per
runtime and model, with the package, dev-kit libraries and kernels that
runtime's overlay takes. A runtime whose package or libraries did not
build, or whose artifacts expired, is left out with a reason. Nightly, the
bench runs only if dev-kit engine or kernel source changed since the commit
the latest baseline benched (changes.py devkit_bench between the two), or
with no baseline; otherwise the previous baseline stands. --force (dispatch)
benches whatever the plan says. Outputs bench, matrix, docvqa-models,
reason, and a summary on $GITHUB_STEP_SUMMARY.

The token (GITHUB_TOKEN, actions: read) is sent to api.github.com only.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import select_models  # noqa: E402

API = "https://api.github.com"
BUILD_WORKFLOW = ".github/workflows/build-and-test.yml"
BENCH_WORKFLOW = ".github/workflows/devkit-bench.yml"
PLAN_ARTIFACT = "devkit-bench-plan"
BASELINE_ARTIFACT = "devkit-bench-baseline"
SHA = re.compile(r"^[0-9a-f]{40}$")
REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
ARTIFACT = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
FAMILY = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# key -> (os, runtime). The keys of the plan's "builds".
RUNTIMES = {"lin-xrt": ("linux", "xrt"), "lin-hrx": ("linux", "hrx"),
            "win-xrt": ("windows", "xrt"), "win-hrx": ("windows", "hrx")}
PR_RUNTIMES = ("lin-xrt", "lin-hrx")
NIGHTLY_RUNTIMES = ("lin-xrt", "lin-hrx")
WEEKLY_RUNTIMES = ("win-xrt",)


class PlanError(ValueError):
    pass


# ── GitHub API ───────────────────────────────────────────────────────────────

def api_get(path: str, params: Optional[dict] = None) -> dict:
    url = f"{API}/{path.lstrip('/')}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN', '')}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_artifacts(repo: str, run_id: int) -> list[dict]:
    out, page = [], 1
    while True:
        data = api_get(f"repos/{repo}/actions/runs/{run_id}/artifacts", {"per_page": 100, "page": page})
        batch = data.get("artifacts") or []
        out += batch
        if len(batch) < 100:
            return out
        page += 1


# ── Pure selection ───────────────────────────────────────────────────────────

def _full_name(obj) -> str:
    return (obj or {}).get("full_name", "") if isinstance(obj, dict) else ""


def run_matches(run: dict, *, repo: str, head_sha: str, head_repo: str, event: str,
                workflow: str = BUILD_WORKFLOW) -> bool:
    """Whether a run is this repository's workflow file for this exact head
    commit (any, when head_sha is empty: the newest nightly), head
    repository and event (schedule: on main)."""
    if (run.get("path") != workflow or _full_name(run.get("repository")) != repo
            or _full_name(run.get("head_repository")) != head_repo
            or not SHA.match(str(run.get("head_sha") or ""))
            or (head_sha and run.get("head_sha") != head_sha)):
        return False
    if event != "any" and run.get("event") != event:
        return False
    if event == "schedule" and run.get("head_branch") != "main":
        return False
    return True


def pick_build_run(runs: list[dict], **match) -> Optional[dict]:
    hits = [r for r in runs if run_matches(r, **match)]
    return max(hits, key=lambda r: (r.get("run_number", 0), r.get("run_attempt", 0))) if hits else None


def artifact_index(artifacts: list[dict]) -> dict[str, dict]:
    """name -> the newest artifact of that name (id, expired, size)."""
    out: dict[str, dict] = {}
    for a in sorted(artifacts, key=lambda a: a.get("id", 0)):
        name = a.get("name")
        if isinstance(name, str):
            out[name] = {"id": a.get("id"), "expired": bool(a.get("expired")),
                         "size": a.get("size_in_bytes", 0)}
    return out


def usable_artifact(index: dict, name: str) -> bool:
    return bool(name) and name in index and not index[name]["expired"]


def baseline_run_ok(run: dict, repo: str) -> bool:
    """A nightly devkit-bench.yml run on this repository's main. A failed one
    counts: a model's failed bench fails the run, and report.py only takes a
    model's usable results into the baseline (an upsert), so its artifact is
    as good as a successful run's; a cancelled run's is not."""
    return (run.get("path") == BENCH_WORKFLOW and _full_name(run.get("repository")) == repo
            and _full_name(run.get("head_repository")) == repo and run.get("event") == "schedule"
            and run.get("head_branch") == "main" and run.get("conclusion") in ("success", "failure"))


def pick_baseline(artifacts: list[dict], runs: dict[int, dict], repo: str) -> Optional[dict]:
    """The newest unexpired baseline artifact whose run baseline_run_ok."""
    for a in sorted(artifacts, key=lambda a: a.get("created_at") or "", reverse=True):
        if a.get("name") != BASELINE_ARTIFACT or a.get("expired"):
            continue
        run_id = (a.get("workflow_run") or {}).get("id")
        run = runs.get(run_id)
        if run is not None and baseline_run_ok(run, repo):
            return {"artifact_id": a.get("id"), "run_id": run_id, "head_sha": run.get("head_sha")}
    return None


def _str(plan: dict, key: str, pattern: re.Pattern, allow_empty: bool = True) -> str:
    value = plan.get(key, "")
    if not isinstance(value, str) or (value and not pattern.match(value)) or (not value and not allow_empty):
        raise PlanError(f"{key}: unexpected value {str(value)[:80]!r}")
    return value


def _list(value, pattern: re.Pattern, what: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value or "[]")
        except ValueError as exc:
            raise PlanError(f"{what}: not JSON") from exc
    if not isinstance(value, list) or not all(isinstance(v, str) and pattern.match(v) for v in value):
        raise PlanError(f"{what}: expected a list of names")
    return value


def parse_plan(plan) -> dict:
    """The plan, validated; raises PlanError."""
    if not isinstance(plan, dict) or plan.get("schema") != 1:
        raise PlanError("not a schema 1 devkit-bench plan")
    out = {
        "bench": plan.get("bench") in (True, "true"),
        "head_sha": _str(plan, "head_sha", SHA, allow_empty=False),
        "families": [f for f in str(plan.get("families") or "").split(",") if f],
        "xclbin_models": _list(plan.get("xclbin_models", []), select_models.MODEL_ID, "xclbin_models"),
        "bench_models": None,
        "xclbins": _str(plan, "xclbins", ARTIFACT),
        "builds": {},
    }
    if not all(FAMILY.match(f) for f in out["families"]):
        raise PlanError("families: unexpected value")
    if plan.get("bench_models") not in (None, ""):
        out["bench_models"] = _list(plan["bench_models"], select_models.MODEL_ID, "bench_models")
    builds = plan.get("builds")
    if not isinstance(builds, dict):
        raise PlanError("builds: expected an object")
    for key, b in builds.items():
        if key not in RUNTIMES or not isinstance(b, dict):
            raise PlanError(f"builds: unexpected entry {str(key)[:40]!r}")
        out["builds"][key] = {"ok": b.get("ok") in (True, "true"),
                              "package": _str(b, "package", ARTIFACT, allow_empty=False),
                              "libs": _str(b, "libs", ARTIFACT)}
    return out


def build_matrix(plan: dict, index: dict, runtimes: list[str], models: list[dict],
                 all_families: set, overlay_all: bool) -> tuple[list[dict], list[str]]:
    """(matrix entries, notes about runtimes left out). overlay_all
    (nightly): every engine's library goes over the package, as the plan's
    families are those of a pull request."""
    notes, entries = [], []
    for key in runtimes:
        b = plan["builds"].get(key)
        if b is None or not b["ok"]:
            notes.append(f"{key}: its package or dev-kit libraries did not build; not benched")
            continue
        missing = [n for n in (b["package"], b["libs"], plan["xclbins"]) if n and not usable_artifact(index, n)]
        if missing:
            notes.append(f"{key}: artifacts missing or expired ({', '.join(missing)}): push or re-run "
                         f"Build and Test; not benched")
            continue
        os_name, runtime = RUNTIMES[key]
        # The overlay replaces the engines serving these families; without
        # libraries (a kernel-only change) it replaces none.
        families = sorted(all_families) if overlay_all else plan["families"]
        for m in models:
            entries.append({"key": key, "os": os_name, "runtime": runtime, "package": b["package"],
                            "libs": b["libs"], "xclbins": plan["xclbins"],
                            "families": ",".join(families) if b["libs"] else "",
                            "model": m["model"], "safe": select_models_safe(m["model"]), "vlm": m["vlm"]})
    return entries, notes


def select_models_safe(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def source_changed(repo_root: Path, base: str, head: str) -> tuple[bool, str]:
    """changes.py devkit_bench between two commits."""
    proc = subprocess.run([sys.executable, str(repo_root / ".github/script/ci/changes.py"),
                           "--repo", str(repo_root), "--base", base, "--head", head],
                          capture_output=True, text=True)
    out = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
    if proc.returncode != 0 or "devkit_bench" not in out:
        return True, "change detection failed: benching"
    return out["devkit_bench"] == "true", out.get("summary", "")


# ── Commands ─────────────────────────────────────────────────────────────────

def write_outputs(values: dict) -> None:
    for k, v in values.items():
        print(f"{k}={v}")
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.writelines(f"{k}={v}\n" for k, v in values.items())


def step_summary(text: str) -> None:
    print(text)
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def cmd_find_build(args) -> int:
    if (not REPO.match(args.repo) or not REPO.match(args.head_repo or args.repo)
            or not (SHA.match(args.head_sha) or (not args.head_sha and args.event == "schedule"))):
        print("::error::bad --repo, --head-sha or --head-repo")
        return 2
    match = {"repo": args.repo, "head_sha": args.head_sha, "head_repo": args.head_repo or args.repo,
             "event": args.event}
    deadline = time.monotonic() + args.wait_minutes * 60
    run = None
    while True:
        if args.run_id:
            candidate = api_get(f"repos/{args.repo}/actions/runs/{int(args.run_id)}")
            run = candidate if run_matches(candidate, **{**match, "event": "any"}) else None
            if run is None:
                write_outputs({"ok": "false", "run-id": "", "reason": "run is not this repository's "
                               "Build and Test run for this commit"})
                return 0
        else:
            params = {"per_page": 100}
            if args.head_sha:
                params["head_sha"] = args.head_sha
            else:
                params["status"] = "completed"
            if args.event != "any":
                params["event"] = args.event
            if args.event == "schedule":
                params["branch"] = "main"
            runs = api_get(f"repos/{args.repo}/actions/workflows/build-and-test.yml/runs", params)
            run = pick_build_run(runs.get("workflow_runs") or [], **match)
        if run is not None and run.get("status") == "completed":
            break
        if time.monotonic() >= deadline:
            write_outputs({"ok": "false", "run-id": "", "reason": "no completed Build and Test run for "
                           f"{args.head_sha[:10]} within {args.wait_minutes} min"})
            return 0
        time.sleep(args.poll_seconds)
    conclusion = run.get("conclusion")
    allowed = ("success",) if args.require_success else ("success", "failure")
    if conclusion not in allowed:
        write_outputs({"ok": "false", "run-id": str(run["id"]),
                       "reason": f"Build and Test run {run['id']} concluded {conclusion}"})
        return 0
    artifacts = list_artifacts(args.repo, run["id"])
    if args.artifacts_out:
        Path(args.artifacts_out).write_text(json.dumps(artifact_index(artifacts)), encoding="utf-8")
    index = artifact_index(artifacts)
    if not usable_artifact(index, PLAN_ARTIFACT):
        why = "expired" if PLAN_ARTIFACT in index else "not uploaded (the change benches nothing)"
        write_outputs({"ok": "false", "run-id": str(run["id"]),
                       "reason": f"Build and Test run {run['id']}: {PLAN_ARTIFACT} {why}"})
        return 0
    write_outputs({"ok": "true", "run-id": str(run["id"]), "head-sha": run["head_sha"],
                   "reason": f"Build and Test run {run['id']}"})
    return 0


def cmd_find_baseline(args) -> int:
    data = api_get(f"repos/{args.repo}/actions/artifacts", {"name": BASELINE_ARTIFACT, "per_page": 30})
    artifacts = data.get("artifacts") or []
    runs: dict[int, dict] = {}
    for a in artifacts:
        run_id = (a.get("workflow_run") or {}).get("id")
        if isinstance(run_id, int) and run_id not in runs and not a.get("expired"):
            runs[run_id] = api_get(f"repos/{args.repo}/actions/runs/{run_id}")
    found = pick_baseline(artifacts, runs, args.repo)
    write_outputs({"baseline-run-id": str(found["run_id"]) if found else "",
                   "baseline-sha": (found or {}).get("head_sha") or ""})
    return 0


def cmd_plan(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    no = {"bench": "false", "matrix": "[]", "docvqa": "false"}
    try:
        plan = parse_plan(json.loads(Path(args.plan).read_text(encoding="utf-8")))
        index = json.loads(Path(args.artifacts).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        write_outputs({**no, "reason": f"unreadable plan: {exc}"})
        step_summary(f"### Dev-kit bench\n\nNot benched: unreadable plan ({exc}).")
        return 0
    if args.head_sha and plan["head_sha"] != args.head_sha:
        write_outputs({**no, "reason": f"the plan is for {plan['head_sha'][:10]}, not {args.head_sha[:10]}"})
        step_summary("### Dev-kit bench\n\nNot benched: Build and Test's plan is for another commit.")
        return 0
    if not plan["bench"] and not args.force:
        write_outputs({**no, "reason": "Build and Test's plan: no dev-kit engine or kernel source to bench"})
        step_summary("### Dev-kit bench\n\nNot benched: no dev-kit engine or kernel source changed.")
        return 0

    notes = []
    if args.mode == "nightly" and not args.force:
        base = ""
        if args.baseline and Path(args.baseline).is_file():
            try:
                base = str(json.loads(Path(args.baseline).read_text(encoding="utf-8")).get("sha") or "")
            except ValueError:
                base = ""
        if SHA.match(base):
            changed, why = source_changed(repo_root, base, plan["head_sha"])
            if not changed:
                write_outputs({**no, "reason": f"no dev-kit engine or kernel source change since the "
                               f"baseline's {base[:10]}: the previous baseline stands"})
                step_summary(f"### Dev-kit bench (nightly)\n\nNot benched: nothing to bench changed "
                             f"between the baseline's `{base[:10]}` and `{plan['head_sha'][:10]}`; "
                             f"the previous baseline stands.")
                return 0
            notes.append(f"changed since the baseline's {base[:10]}: {why}")
        else:
            notes.append("no baseline yet: benching the assigned set")

    try:
        catalog = select_models.changes.load_catalog(
            (repo_root / select_models.changes.MODEL_LIST).read_text(encoding="utf-8"))
        engines = select_models.changes.DevkitEngines.parse(
            (repo_root / select_models.changes.DEVKIT_ENGINES).read_text(encoding="utf-8"))
        assignment = json.loads((repo_root / "flm-dev-kit/FLM_Xclbin/kernel_assignment.json")
                                .read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        write_outputs({**no, "reason": f"cannot read the catalog or dev-kit maps: {exc}"})
        return 1
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    if requested:
        mode = "dispatch"
    elif args.mode == "dispatch":
        # What the build changed, else the assigned set.
        mode = "pr" if plan["families"] or plan["xclbin_models"] else "nightly"
    else:
        mode = args.mode
    chosen = select_models.select(mode=mode, catalog=catalog, engines=engines, assignment=assignment,
                                  families=plan["families"], xclbin_models=plan["xclbin_models"],
                                  xclbin_bench_models=plan["bench_models"], requested=requested,
                                  rotation=int(time.time() // 86400))
    runtimes = [r.strip() for r in args.runtimes.split(",") if r.strip()]
    bad = [r for r in runtimes if r not in RUNTIMES]
    if bad:
        write_outputs({**no, "reason": f"unknown runtimes: {', '.join(bad)}"})
        return 2
    all_families = engines.all_families() if engines else set()
    entries, more = build_matrix(plan, index, runtimes, chosen["models"], all_families,
                                 overlay_all=mode != "pr")
    notes += more
    if chosen["dropped"]:
        what = "other nights' turn" if mode == "nightly" else "over the cap, not benched"
        notes.append(f"{what}: {', '.join(chosen['dropped'])}")
    if chosen["skipped"]:
        notes.append(f"not benchable here (not in the NPU catalog, or embedding/ASR/decision): "
                     f"{', '.join(chosen['skipped'])}")
    bench = bool(entries)
    lines = [f"### Dev-kit bench ({args.mode})", "",
             f"- Build and Test head `{plan['head_sha'][:10]}`; runtimes: {', '.join(runtimes)}",
             f"- Models: {', '.join(m['model'] for m in chosen['models']) or 'none'}"]
    lines += [f"- {n}" for n in notes]
    if not bench:
        lines.append("- Nothing to bench.")
    step_summary("\n".join(lines))
    write_outputs({"bench": str(bench).lower(), "matrix": json.dumps(entries, separators=(",", ":")),
                   "docvqa": str(any(e["vlm"] for e in entries)).lower(),
                   "head-sha": plan["head_sha"],
                   "reason": "; ".join(notes) or "benching"})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("find-build")
    p.add_argument("--repo", required=True)
    p.add_argument("--event", choices=("pull_request", "schedule", "any"), required=True)
    p.add_argument("--head-sha", required=True)
    p.add_argument("--head-repo", default="")
    p.add_argument("--run-id", default="")
    p.add_argument("--require-success", action="store_true")
    p.add_argument("--wait-minutes", type=float, default=200)
    p.add_argument("--poll-seconds", type=float, default=60)
    p.add_argument("--artifacts-out", default="")
    p = sub.add_parser("find-baseline")
    p.add_argument("--repo", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--plan", required=True)
    p.add_argument("--artifacts", required=True)
    p.add_argument("--mode", choices=("pr", "nightly", "dispatch"), required=True)
    p.add_argument("--runtimes", default=",".join(PR_RUNTIMES))
    p.add_argument("--models", default="")
    p.add_argument("--baseline", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--head-sha", default="", help="the commit the plan must be for")
    p.add_argument("--repo-root", default=str(HERE.parents[2]))
    args = ap.parse_args(argv)
    return {"find-build": cmd_find_build, "find-baseline": cmd_find_baseline, "plan": cmd_plan}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
