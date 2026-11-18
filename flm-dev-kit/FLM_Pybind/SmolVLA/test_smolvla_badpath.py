#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Check that a bad model path raises instead of killing the interpreter.

Needs its own process: the singleton guard in test_smolvla_py.py fires before
the path check while a valid instance is alive, so the path handling can only be
exercised from a clean state. The underlying model code calls exit(1) on a
missing config.json / model.safetensors, which would take the interpreter down
with it -- the binding validates the paths up front to turn that into an
exception.

    /path/to/FastFlowLM_IRON/models_simple/.venv/bin/python test_smolvla_badpath.py
"""

import sys

import smolvla_npu


def main():
    failures = []
    for name, path in [
        ("nonexistent directory", "/nonexistent/model/path"),
        ("directory without config.json", "/tmp"),
    ]:
        try:
            smolvla_npu.SmolVLA(path)
            print(f"  [FAIL] {name}: construction unexpectedly succeeded")
            failures.append(name)
        except RuntimeError as e:
            print(f"  [PASS] {name} -- {e}")

    if failures:
        print(f"FAILED ({len(failures)})")
        return 1
    print("all checks passed (interpreter survived)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
