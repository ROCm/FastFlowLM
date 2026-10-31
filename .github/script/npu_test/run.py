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

from .harness import FAIL, Context, Result, configure_stdout
from .plan import TIERS, Plan, load_catalog
from .suites import SUITES, ORDER


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
                             "the pr tier tests every model in them.")
    parser.add_argument("--asr-full", action="store_true",
                        help="Run the whole ASR evaluation set even on the pr tier.")
    parser.add_argument("--models", default="",
                        help="Comma-separated model ids; restricts every suite to these.")
    parser.add_argument("--asr-expected-key", default="",
                        help="Row of asr_eval/expected.json to check WER against, "
                             "e.g. linux-hrx (default: none, core checks only).")
    parser.add_argument("--output-dir", default="npu-test-results")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    # Not $FLM_SERVE_PORT: a runner may set it for an flm of its own, which then
    # holds that port (it did on a Windows runner, failing every server start).
    parser.add_argument("--port", type=int, default=None,
                        help="Port for the servers this run starts (default: a free port "
                             "for each server).")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


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
        catalog = load_catalog(args.flm_bin)
    except Exception as exc:
        catalog, catalog_error = {}, f"`flm list --json` failed: {type(exc).__name__}: {exc}"
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    only = [m.strip() for m in args.models.split(",") if m.strip()] or None
    plan = Plan(catalog, args.tier, families, only_models=only)
    ctx = Context(flm_bin=args.flm_bin, label=args.label, tier=args.tier, plan=plan,
                  out_dir=args.output_dir, request_timeout=args.request_timeout,
                  port=args.port, verbose=args.verbose)
    ctx.asr_full = args.asr_full or args.tier == "full"
    ctx.asr_expected_key = args.asr_expected_key

    print(f"{args.label}: {args.tier} tier, {len(catalog)} catalog models, "
          f"suites: {', '.join(names)}" + (f", touched families: {', '.join(families)}"
                                           if families else ""))
    try:
        if catalog_error:
            print(catalog_error)
            ctx.record(Result("setup", "catalog", FAIL, detail=catalog_error))
        else:
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
