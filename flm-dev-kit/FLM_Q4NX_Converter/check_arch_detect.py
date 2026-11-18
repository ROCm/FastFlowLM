# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Arch auto-detection check: each GGUF given must resolve to the expected converter family.

    python check_arch_detect.py qwen35-2B=2b-q4k.gguf qwen35-2B=2b-mmproj-bf16.gguf qwen3.8-27B=tiny-r3.gguf
Exit status 0 only if every GGUF resolves as expected (name given as one of the ModelArchNames aliases, or
`error` when detection must refuse with a ValueError instead of guessing).
"""
import sys

from gguf import GGUFReader

from q4nx.constants import ModelArchNames
from q4nx.model_converter import get_model_arch_from_gguf

fails = 0
for spec in sys.argv[1:]:
    want_alias, path = spec.split("=", 1)
    want = "error" if want_alias == "error" else next(a for a, names in ModelArchNames.items() if want_alias in names)
    try:
        got = get_model_arch_from_gguf(GGUFReader(path))
    except ValueError as e:
        got = "error"
        print(f"      refused: {str(e)[:110]}")
    ok = got == want
    fails += not ok
    print(("PASS " if ok else "FAIL ") + f"{path}: {got!r} (expected {want!r})")
sys.exit(1 if fails else 0)
