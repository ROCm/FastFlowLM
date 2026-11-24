#!/usr/bin/env python3
"""Hold the exact-retry family lists of test_response_shape.py to the model catalog.

The exact-retry cases run only on the oga-hipep models whose frontend takes a prompt-end checkpoint,
and the frontend is chosen by the catalog's ``details.family``. This check needs no server and no
model. It fails when:

* a family that has an oga-hipep model in ``src/model_list.json`` is in none of the three lists
  (a new family: decide whether its exact retry is verified, known to be missing, or unverified);
* a family is in more than one list;
* a list names a family that no oga-hipep model has (a typo, or a family that was removed).

Usage::

    python check_exact_retry_families.py [path/to/model_list.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import test_response_shape as shape

LISTS = {
    "EXACT_RETRY_FAMILIES": shape.EXACT_RETRY_FAMILIES,
    "EXACT_RETRY_MISSING_FAMILIES": shape.EXACT_RETRY_MISSING_FAMILIES,
    "EXACT_RETRY_UNVERIFIED_FAMILIES": shape.EXACT_RETRY_UNVERIFIED_FAMILIES,
}


def oga_families(catalog: dict) -> dict[str, list[str]]:
    """family -> the oga-hipep models that have it. An oga-hipep model's type key ends in -oga-hipep."""
    found: dict[str, list[str]] = {}
    for model_type, sizes in catalog["models"].items():
        if not model_type.endswith("-oga-hipep"):
            continue
        for size, entry in sizes.items():
            family = (entry.get("details") or {}).get("family")
            found.setdefault(str(family), []).append(f"{model_type}:{size}")
    return found


def problems(catalog: dict) -> list[str]:
    found = oga_families(catalog)
    out = []
    for family, models in sorted(found.items()):
        named = [name for name, families in LISTS.items() if family in families]
        if not named:
            out.append(f"family {family!r} ({', '.join(models)}) is in none of the exact-retry lists")
        elif len(named) > 1:
            out.append(f"family {family!r} is in more than one list: {', '.join(named)}")
    for name, families in LISTS.items():
        for family in families:
            if family not in found:
                out.append(f"{name} names {family!r}, which no oga-hipep model in the catalog has")
    return out


def main() -> int:
    default = Path(__file__).resolve().parents[2] / "model_list.json"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    with open(path, encoding="utf-8") as f:
        catalog = json.load(f)
    found = oga_families(catalog)
    if not found:
        print(f"FAIL: no oga-hipep model in {path}")
        return 1
    bad = problems(catalog)
    for line in bad:
        print("FAIL: " + line)
    if bad:
        print("Fix the lists in test_response_shape.py (EXACT_RETRY_*).")
        return 1
    print(f"OK: {len(found)} oga-hipep families, each in one exact-retry list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
