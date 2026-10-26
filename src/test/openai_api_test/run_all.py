#!/usr/bin/env python3
"""Run every OpenAI API test module against one running server.

Usage::

    python3 run_all.py <model> [--always-thinks] [--endpoint URL] [--timeout T] [-k NAME] [-v]

Exits 0 when nothing failed (skips are fine), 1 otherwise.
"""

from __future__ import annotations

import sys

import flm_api
import test_errors
import test_logprobs
import test_reasoning
import test_response_format
import test_response_shape
import test_tool_calls

MODULES = [test_response_shape, test_reasoning, test_tool_calls, test_response_format, test_logprobs, test_errors]


def main() -> None:
    args = flm_api.parse_args(__doc__.splitlines()[0])
    api = flm_api.make_api(args)
    total = flm_api.Tally()
    for module in MODULES:
        print(f"\n== {module.__name__}")
        total.add(flm_api.run_cases(module.CASES, api, only=args.only, verbose=args.verbose))
    sys.exit(flm_api.summarize(total))


if __name__ == "__main__":
    main()
