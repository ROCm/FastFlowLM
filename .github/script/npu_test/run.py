"""Runs the NPU test suites against one flm binary and writes a report.

    python -m npu_test --flm-bin path/to/flm --label linux-xrt-portable \
        --tier pr --output-dir results

Run from .github/script (or with it on PYTHONPATH). Exits 1 if any case
failed; warnings and skips do not fail the run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from .harness import FAIL, PASS, SKIP, WARN, Context, Result, configure_stdout
from .model_cache import ModelCache
from .plan import GPU_BACKENDS, TIERS, Plan, load_catalog
from .suites import SUITES, ORDER

# Per model. A GPU backend's models are large (qwen3.5-oga-hipep:9b is 7.5 GB)
# and new to most runners, some of which download at a few MB/s.
PULL_TIMEOUT = 90 * 60

# All NPU downloads together, on the install and pr tiers. A model new to the
# catalog is on no runner yet, and the first job to want it would otherwise
# spend its time limit on the download (a 30B model at a few MB/s takes over
# an hour). What the budget leaves undownloaded is a warning, and goes
# untested on that runner; the full tier and --backend runs wait for every
# download.
PULL_BUDGET_MIN = 30.0

# All the models the change adds (--required-models) together. They are
# pulled first and outside PULL_BUDGET_MIN, since testing them is the point of
# the change: a 120 GB package at a few MB/s takes hours. Build and Test gives
# such a job 600 minutes, so this leaves three hours for the other downloads
# and the suites, and the report is still written. A new model this leaves
# undownloaded is a failure, not a warning.
REQUIRED_PULL_BUDGET_MIN = 420.0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flm-bin", default=os.environ.get("FLM_BIN", "flm"),
                        help="The flm binary under test (default: $FLM_BIN, else flm).")
    parser.add_argument("--label", required=True,
                        help="Names the package under test in the report, "
                             "e.g. linux-xrt-portable.")
    parser.add_argument("--tier", choices=TIERS, default="pr")
    parser.add_argument("--suites", default="all",
                        help=f"Comma-separated subset of: {', '.join(ORDER)} (default: all).")
    parser.add_argument("--families", default="",
                        help="Comma-separated model families the change touches; "
                             "the pr tier tests every model in them, or with --backend "
                             "one of each that it tests only when touched.")
    parser.add_argument("--asr-full", action="store_true",
                        help="Run the whole ASR evaluation set even on the pr tier.")
    parser.add_argument("--required-models", default="",
                        help="Comma-separated model ids the change adds (with --backend: "
                             "gives new weights). Every tier tests them, they are downloaded "
                             "whatever their size, and the run fails if the package does not "
                             "list them or they do not pull.")
    parser.add_argument("--required-pull-budget", type=float, default=REQUIRED_PULL_BUDGET_MIN,
                        metavar="MINUTES",
                        help="Time for downloading the --required-models, all together; one "
                             f"left undownloaded fails the run (default: {REQUIRED_PULL_BUDGET_MIN:g}).")
    parser.add_argument("--models", default="",
                        help="Comma-separated model ids; restricts every suite to these.")
    parser.add_argument("--backend", choices=["", *sorted(GPU_BACKENDS)], default="",
                        help="Test this GPU backend's models instead of the NPU models: one "
                             "per family, or every one on the full tier (default: the NPU "
                             "models).")
    parser.add_argument("--max-model-gb", type=float, default=0.0, metavar="GB",
                        help="Skip the models larger than this (catalog size), which this "
                             "runner's memory cannot hold, --required-models too (default: 0, "
                             "no limit).")
    parser.add_argument("--model-extension", action="append", default=[],
                        help="Model-extension manifest passed to `flm list` and every "
                             "server (repeatable).")
    parser.add_argument("--asr-expected-key", default="",
                        help="Row of asr_eval/expected.json to check WER against, "
                             "e.g. linux-hrx (default: none, core checks only).")
    parser.add_argument("--output-dir", default="npu-test-results")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--pull-budget", type=float, default=None, metavar="MINUTES",
                        help="Time for downloading the models this machine lacks, all "
                             "together; 0 for no limit (default: "
                             f"{PULL_BUDGET_MIN:.0f} on the install and pr tiers without "
                             "--backend, else no limit).")
    # Not $FLM_SERVE_PORT: a runner may set it for an flm of its own, which then
    # holds that port (it did on a Windows runner, failing every server start).
    parser.add_argument("--port", type=int, default=None,
                        help="Port for the servers this run starts (default: a free port "
                             "for each server).")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def pull_budget(args: argparse.Namespace) -> float | None:
    """The download budget in seconds, None for no limit."""
    minutes = args.pull_budget
    if minutes is None:
        minutes = 0.0 if args.tier == "full" or args.backend else PULL_BUDGET_MIN
    return minutes * 60 if minutes > 0 else None


def check_required(ctx: Context, plan: Plan, names: list[str], skipped: set[str]) -> None:
    """Records why a model the change adds would go untested in this run.

    One the package does not list is a failure: the change added it to the
    catalog, so a package built from that change must offer it. One of
    another backend (a GPU model in an NPU run) is not this job's to test,
    nor is one larger than this runner holds (skipped, by --max-model-gb,
    and already recorded), which a HIPEP machine without that limit tests.
    """
    planned = set(plan.planned(names))
    for model in plan.required:
        if model in skipped:
            continue
        if model not in plan.catalog:
            ctx.record(Result("setup", "new model", FAIL, model,
                              "the change adds this model, but `flm list --json --filter all` "
                              "does not list it on this machine"))
        elif not plan.backend_serves(model):
            ctx.record(Result("setup", "new model", SKIP, model,
                              "another backend runs it; its own job tests it"))
        elif model not in planned:
            ctx.record(Result("setup", "new model", WARN, model,
                              f"no suite this job runs ({', '.join(names)}) tests it"))


def stamp(cache: ModelCache | None, plan: Plan, pulled: list[str]) -> None:
    """Remembers the models a GPU backend run just pulled, and so verified."""
    if cache is None:
        return
    for model_id in pulled:
        why = "" if model_id in plan.unavailable else cache.stamp(model_id)
        if why:
            print(f"  [setup] {model_id}: not remembered as verified ({why})", flush=True)


def main(argv=None) -> int:
    configure_stdout()
    args = parse_args(argv)
    started = time.monotonic()

    names = ORDER if args.suites == "all" else [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        print(f"unknown suite(s): {', '.join(unknown)}; known: {', '.join(ORDER)}")
        return 2

    version = subprocess.run([args.flm_bin, "version"], capture_output=True, text=True)
    print(version.stdout.strip() or version.stderr.strip())
    if version.returncode != 0:
        print(f"`{args.flm_bin} version` failed with code {version.returncode}")
        return 1

    # Without the catalog nothing can run, but the failure still belongs in the
    # report, so a broken package shows up as a failed case, not a traceback.
    catalog_error = ""
    try:
        catalog = load_catalog(args.flm_bin, args.model_extension)
    except Exception as exc:
        catalog, catalog_error = {}, f"`flm list --json` failed: {type(exc).__name__}: {exc}"
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    only = [m.strip() for m in args.models.split(",") if m.strip()] or None
    required = [m.strip() for m in args.required_models.split(",") if m.strip()]
    plan = Plan(catalog, args.tier, families, only_models=only, backend=args.backend,
                required=required)
    ctx = Context(flm_bin=args.flm_bin, label=args.label, tier=args.tier, plan=plan,
                  out_dir=args.output_dir, request_timeout=args.request_timeout,
                  port=args.port, verbose=args.verbose,
                  model_extensions=args.model_extension)
    ctx.asr_full = args.asr_full or args.tier == "full"
    ctx.asr_expected_key = args.asr_expected_key

    print(f"{args.label}: {args.tier} tier, {len(catalog)} catalog models, "
          f"suites: {', '.join(names)}" + (f", touched families: {', '.join(families)}"
                                           if families else "")
          + (f", backend: {args.backend}" if args.backend else "")
          + (f", new models: {', '.join(required)}" if required else ""))
    try:
        if catalog_error:
            print(catalog_error)
            ctx.record(Result("setup", "catalog", FAIL, detail=catalog_error))
        else:
            # A model larger than the runner's memory would only fail to load.
            # A required one too is a skip, not a failure: a HIPEP machine
            # without the limit (stx-halo) tests it.
            too_big = set()
            if args.max_model_gb > 0:
                for model_id in plan.planned(names):
                    size_gb = plan.model(model_id).size / 1e9
                    if size_gb > args.max_model_gb:
                        ctx.record(Result("setup", "size", SKIP, model_id,
                                          f"{size_gb:.1f} GB, larger than this runner holds "
                                          f"(--max-model-gb {args.max_model_gb:g})"))
                        plan.unavailable.add(model_id)
                        too_big.add(model_id)
            # Pulled first, a download counts against no request's timeout: flm
            # serve downloads a missing model inside the request that names it,
            # and every request behind that one waits too.
            # A GPU backend's models are pulled even when present: the pull
            # checks their files against the pinned hashes and replaces any
            # that do not match, which `flm list` does not. That takes minutes
            # per model, so one a pull on this runner verified before, whose
            # pins and files have not changed since, is not pulled again;
            # except a required one: the change adds or republishes it, perhaps in
            # a way the pins do not cover (a raised flm_min_version).
            check_required(ctx, plan, names, too_big)
            missing = plan.missing(names, everything=bool(args.backend))
            cache = ModelCache.open(args.flm_bin) if args.backend else None
            verified = {}
            if cache is not None:
                for model_id in missing:
                    if model_id in plan.required:
                        continue
                    when = cache.verified(model_id) if plan.model(model_id).installed else None
                    if when:
                        verified[model_id] = when
            if verified:
                print(f"\n== setup: verified on this runner before and unchanged since, so not "
                      f"pulled: {', '.join(verified)}", flush=True)
                for model_id, when in verified.items():
                    ctx.record(Result("setup", "pull", PASS, model_id,
                                      f"not pulled: verified {when}, and neither its pins nor "
                                      f"its files changed since", data={"cached": True}))
                missing = [m for m in missing if m not in verified]
            # The models the change adds first, with no budget: a large new
            # model must not go untested because it is new to the runner.
            new = [m for m in missing if m in plan.required]
            missing = [m for m in missing if m not in plan.required]
            if new:
                limit = args.required_pull_budget * 60
                print(f"\n== setup: download the models this change adds "
                      f"({args.required_pull_budget:g} min for all): {', '.join(new)}", flush=True)
                started_pull = time.monotonic()
                ctx.pull(new, limit, limit, required=True)
                print(f"== setup done in {time.monotonic() - started_pull:.0f}s", flush=True)
                stamp(cache, plan, new)
            if missing:
                budget = pull_budget(args)
                limit = f"{budget / 60:g}-min budget" if budget else "no time limit"
                what = ("pull (and verify) the models" if args.backend
                        else "download the models this machine lacks")
                print(f"\n== setup: {what} ({limit}): {', '.join(missing)}", flush=True)
                started_pull = time.monotonic()
                ctx.pull(missing, PULL_TIMEOUT, budget)
                print(f"== setup done in {time.monotonic() - started_pull:.0f}s", flush=True)
                stamp(cache, plan, missing)
            for name in names:
                ctx.run_suite(SUITES[name])
    finally:
        # The version the servers reported, which is what the report should name.
        ctx.version = (version.stdout.strip().split()[-1] if version.stdout.strip()
                       else "unknown")
        path = ctx.write_report(names, time.monotonic() - started)
        print(f"\nReport: {path}")

    counts = {s: sum(1 for r in ctx.results if r.status == s) for s in ("pass", "fail", "warn", "skip")}
    print(f"{counts['pass']} passed, {counts['fail']} failed, {counts['warn']} warnings, "
          f"{counts['skip']} skipped")
    return 1 if ctx.failed else 0


if __name__ == "__main__":
    sys.exit(main())
