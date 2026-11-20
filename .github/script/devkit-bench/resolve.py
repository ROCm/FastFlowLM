#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Decides what the dev-kit benches (devkit-bench.yml) bench, on a
GitHub-hosted runner.

    resolve.py write-plan    --out plan.json --head-sha S --package-sha S
                             --builds lin-xrt=success,lin-hrx=failure,...
                             --engines success|failure|skipped
                             [--families a,b] [--xclbin-models JSON]
                             [--bench-models JSON] [--xclbins devkit-xclbins]
    resolve.py find-build    --repo R --head-sha S [--run-id N]
                             [--wait-minutes 200] --plan-out plan.json
                             [--artifacts-out artifacts.json]
    resolve.py find-baseline --repo R
    resolve.py plan          --plan plan.json [--artifacts artifacts.json]
                             --mode pr|nightly|dispatch
                             [--runtimes lin-xrt,lin-hrx] [--models a,b]
                             [--baseline baseline.json] [--force]

Build and Test (build-and-test.yml) calls devkit-bench.yml in the same run,
after its builds: write-plan turns what it passes (the touched families,
the models whose kernels were rebuilt, which portable packages and dev-kit
builds succeeded) into the plan; their artifacts are in this run.

find-build is the dispatch path only (rebenching an earlier Build and Test
run): it finds that run and waits (polling) for it to complete. The run must
be of .github/workflows/build-and-test.yml in this repository, for exactly
this head commit (a run id given is checked the same way), the newest such
run, completed with success or failure. Its plan is derived from the
artifacts it uploaded (plan_from_artifacts): the packages and dev-kit
libraries present, and devkit-xclbins if present. Outputs run-id, ok,
head-sha and reason; the run's artifact list goes to --artifacts-out.

find-baseline finds the newest unexpired devkit-bench-baseline artifact
uploaded by a scheduled run on main of this repository (Build and Test's,
or devkit-bench.yml's before it was called from there) that completed
(success, or failure: see baseline_run_ok), not a fork's or a branch's run
with the same workflow name. Outputs baseline-run-id ("" when there is
none).

plan reads the plan (validated field by field), picks the models
(select_models.py) and builds the bench matrix: one entry per runtime and
model, with the package, dev-kit libraries and kernels that runtime's
overlay takes. A runtime whose package or libraries did not build, or
(dispatch, with --artifacts) whose artifacts expired, is left out with a
reason. Nightly and dispatch, the bench runs only if dev-kit engine or
kernel source changed since the commit the latest baseline benched
(changes.py devkit_bench between the two), or with no baseline; otherwise
the previous baseline stands. --force (dispatch) benches regardless.
Outputs bench, matrix, docvqa, head-sha, reason, and a summary on
$GITHUB_STEP_SUMMARY.

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
# Whose scheduled runs upload the nightly baseline: Build and Test's, which
# calls devkit-bench.yml, and devkit-bench.yml's own from before that (their
# artifacts last 90 days).
BASELINE_WORKFLOWS = (BUILD_WORKFLOW, BENCH_WORKFLOW)
BASELINE_ARTIFACT = "devkit-bench-baseline"
SHA = re.compile(r"^[0-9a-f]{40}$")
REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
ARTIFACT = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
FAMILY = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# key -> (os, runtime). The keys of the plan's "builds".
RUNTIMES = {"lin-xrt": ("linux", "xrt"), "lin-hrx": ("linux", "hrx"),
            "win-xrt": ("windows", "xrt"), "win-hrx": ("windows", "hrx")}
# key -> Build and Test's portable package artifact ({sha}: that run's
# github.sha, the merge commit on a pull request).
PACKAGES = {"lin-xrt": "portable", "lin-hrx": "HRX-portable",
            "win-xrt": "fastflowlm-windows-{sha}", "win-hrx": "HRX-fastflowlm-windows-{sha}"}
XCLBINS_ARTIFACT = "devkit-xclbins"
JOB_RESULTS = ("success", "failure", "cancelled", "skipped")
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


def libs_artifact(key: str) -> str:
    return f"devkit-libs-{key}"


def run_matches(run: dict, *, repo: str, head_sha: str) -> bool:
    """Whether a run is this repository's Build and Test, from this
    repository's branch, for exactly this head commit."""
    return (run.get("path") == BUILD_WORKFLOW and _full_name(run.get("repository")) == repo
            and _full_name(run.get("head_repository")) == repo
            and SHA.match(str(run.get("head_sha") or "")) is not None
            and run.get("head_sha") == head_sha)


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
    """A scheduled run on this repository's main of a workflow that uploads
    the baseline (BASELINE_WORKFLOWS). A failed one counts: a model's failed
    bench (or a product test) fails the run, and report.py only takes a
    model's usable results into the baseline (an upsert), so its artifact is
    as good as a successful run's; a cancelled run's is not."""
    return (run.get("path") in BASELINE_WORKFLOWS and _full_name(run.get("repository")) == repo
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


def write_plan(*, head_sha: str, package_sha: str, builds: dict, engines: str, families: str = "",
               xclbin_models: str = "[]", bench_models: str = "", xclbins: str = "") -> dict:
    """The plan of this Build and Test run, from what it passes to
    devkit-bench.yml: builds maps a runtime key to its portable package
    build's result, engines is devkit-engines' result. A runtime is ok when
    its package built and the engines built (all four, or none) or were not
    asked for; its libraries count only when the engines built."""
    libs_ok = engines == "success"
    out = {"schema": 1, "bench": True, "head_sha": head_sha, "families": families,
           "xclbin_models": xclbin_models or "[]", "bench_models": bench_models, "xclbins": xclbins,
           "builds": {}}
    for key, result in builds.items():
        if key not in RUNTIMES or result not in JOB_RESULTS:
            raise PlanError(f"builds: unexpected entry {str(key)[:40]!r}={str(result)[:20]!r}")
        out["builds"][key] = {"ok": result == "success" and engines in ("success", "skipped"),
                              "package": PACKAGES[key].format(sha=package_sha),
                              "libs": libs_artifact(key) if libs_ok else ""}
    return out


def plan_from_artifacts(index: dict, head_sha: str) -> dict:
    """The plan of an earlier Build and Test run (dispatch), from what it
    uploaded: a runtime whose package is there is benched, with its dev-kit
    libraries when those are there, and the kernels when devkit-xclbins is.
    Which families and kernels the change touched is not known: dispatch
    names its models, or benches the assigned set."""
    builds = {}
    for key, pattern in PACKAGES.items():
        names = [n for n in index if re.fullmatch(pattern.format(sha="[0-9a-f]{40}"), n)]
        if not names:
            continue
        name = sorted(names)[-1]
        builds[key] = {"ok": usable_artifact(index, name), "package": name,
                       "libs": libs_artifact(key) if usable_artifact(index, libs_artifact(key)) else ""}
    return {"schema": 1, "bench": True, "head_sha": head_sha, "families": "", "xclbin_models": "[]",
            "bench_models": "", "builds": builds,
            "xclbins": XCLBINS_ARTIFACT if usable_artifact(index, XCLBINS_ARTIFACT) else ""}


def build_matrix(plan: dict, index: Optional[dict], runtimes: list[str], models: list[dict],
                 all_families: set, overlay_all: bool) -> tuple[list[dict], list[str]]:
    """(matrix entries, notes about runtimes left out). index is the run's
    artifacts (dispatch: another run's, which may have expired); None for
    this run's own, which its builds just uploaded. overlay_all (nightly,
    dispatch): every engine's library goes over the package, as the plan's
    families are those of a pull request."""
    notes, entries = [], []
    for key in runtimes:
        b = plan["builds"].get(key)
        if b is None or not b["ok"]:
            notes.append(f"{key}: its package or dev-kit libraries did not build; not benched")
            continue
        missing = [n for n in (b["package"], b["libs"], plan["xclbins"])
                   if n and index is not None and not usable_artifact(index, n)]
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


def cmd_write_plan(args) -> int:
    """This run's plan, from the inputs Build and Test passed."""
    try:
        if not SHA.match(args.head_sha) or not SHA.match(args.package_sha):
            raise PlanError("bad --head-sha or --package-sha")
        if args.engines not in JOB_RESULTS:
            raise PlanError(f"--engines: unexpected value {args.engines[:20]!r}")
        builds = dict(item.split("=", 1) for item in args.builds.split(",") if "=" in item)
        plan = write_plan(head_sha=args.head_sha, package_sha=args.package_sha, builds=builds,
                          engines=args.engines, families=args.families, xclbin_models=args.xclbin_models,
                          bench_models=args.bench_models, xclbins=args.xclbins)
        parse_plan(plan)
    except PlanError as exc:
        print(f"::error::bench plan: {exc}")
        return 2
    Path(args.out).write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print(json.dumps(plan, indent=1))
    return 0


def cmd_find_build(args) -> int:
    """Dispatch: the Build and Test run whose artifacts to bench."""
    if not REPO.match(args.repo) or not SHA.match(args.head_sha) or (args.run_id and not args.run_id.isdigit()):
        print("::error::bad --repo, --head-sha or --run-id")
        return 2
    deadline = time.monotonic() + args.wait_minutes * 60
    while True:
        if args.run_id:
            candidate = api_get(f"repos/{args.repo}/actions/runs/{int(args.run_id)}")
            run = candidate if run_matches(candidate, repo=args.repo, head_sha=args.head_sha) else None
            if run is None:
                write_outputs({"ok": "false", "run-id": "", "reason": "run is not this repository's "
                               "Build and Test run for this commit"})
                return 0
        else:
            runs = api_get(f"repos/{args.repo}/actions/workflows/build-and-test.yml/runs",
                           {"per_page": 100, "head_sha": args.head_sha})
            run = pick_build_run(runs.get("workflow_runs") or [], repo=args.repo, head_sha=args.head_sha)
        if run is not None and run.get("status") == "completed":
            break
        if time.monotonic() >= deadline:
            write_outputs({"ok": "false", "run-id": "", "reason": "no completed Build and Test run for "
                           f"{args.head_sha[:10]} within {args.wait_minutes:g} min"})
            return 0
        time.sleep(args.poll_seconds)
    conclusion = run.get("conclusion")
    if conclusion not in ("success", "failure"):
        write_outputs({"ok": "false", "run-id": str(run["id"]),
                       "reason": f"Build and Test run {run['id']} concluded {conclusion}"})
        return 0
    index = artifact_index(list_artifacts(args.repo, run["id"]))
    if args.artifacts_out:
        Path(args.artifacts_out).write_text(json.dumps(index), encoding="utf-8")
    plan = plan_from_artifacts(index, run["head_sha"])
    if not plan["builds"]:
        write_outputs({"ok": "false", "run-id": str(run["id"]),
                       "reason": f"Build and Test run {run['id']} uploaded no portable package"})
        return 0
    Path(args.plan_out).write_text(json.dumps(plan, indent=1), encoding="utf-8")
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
        index = json.loads(Path(args.artifacts).read_text(encoding="utf-8")) if args.artifacts else None
    except (OSError, ValueError) as exc:
        write_outputs({**no, "reason": f"unreadable plan: {exc}"})
        step_summary(f"### Dev-kit bench\n\nNot benched: unreadable plan ({exc}).")
        return 0
    if not plan["bench"] and not args.force:
        write_outputs({**no, "reason": "no dev-kit engine or kernel source to bench"})
        step_summary("### Dev-kit bench\n\nNot benched: no dev-kit engine or kernel source changed.")
        return 0

    notes = []
    # A pull request benches because its change is source (Build and Test's
    # changes devkit-bench); the nightly and a dispatch only when the source
    # changed since the baseline.
    if args.mode in ("nightly", "dispatch") and not args.force:
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
                step_summary(f"### Dev-kit bench ({args.mode})\n\nNot benched: nothing to bench changed "
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
    p = sub.add_parser("write-plan")
    p.add_argument("--out", required=True)
    p.add_argument("--head-sha", required=True, help="the commit benched")
    p.add_argument("--package-sha", required=True, help="the run's github.sha (Windows package names)")
    p.add_argument("--builds", required=True, help="key=result,... of the portable package builds")
    p.add_argument("--engines", required=True, help="devkit-engines' result")
    p.add_argument("--families", default="")
    p.add_argument("--xclbin-models", default="[]")
    p.add_argument("--bench-models", default="")
    p.add_argument("--xclbins", default="")
    p = sub.add_parser("find-build")
    p.add_argument("--repo", required=True)
    p.add_argument("--head-sha", required=True)
    p.add_argument("--run-id", default="")
    p.add_argument("--wait-minutes", type=float, default=200)
    p.add_argument("--poll-seconds", type=float, default=60)
    p.add_argument("--plan-out", required=True)
    p.add_argument("--artifacts-out", default="")
    p = sub.add_parser("find-baseline")
    p.add_argument("--repo", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--plan", required=True)
    p.add_argument("--artifacts", default="", help="another run's artifact list (dispatch)")
    p.add_argument("--mode", choices=("pr", "nightly", "dispatch"), required=True)
    p.add_argument("--runtimes", default=",".join(PR_RUNTIMES))
    p.add_argument("--models", default="")
    p.add_argument("--baseline", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--repo-root", default=str(HERE.parents[2]))
    args = ap.parse_args(argv)
    return {"write-plan": cmd_write_plan, "find-build": cmd_find_build, "find-baseline": cmd_find_baseline,
            "plan": cmd_plan}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
