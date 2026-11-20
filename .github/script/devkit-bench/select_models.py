#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Which models the dev-kit benches run.

The inputs are the bench plan (resolve.py: what Build and Test passed to
devkit-bench.yml, or a dispatched run's artifacts) and this checkout:

  pull request  the models of the dev-kit engines serving the touched
                families (engines.json "models", as IRON's
                dll_assignment.json; only those whose catalog family was
                touched), plus the models whose kernels Build and Test
                rebuilt (devkit-xclbins.yml touched-models) that are worth
                benching (its bench-models: users of a kernel whose source
                changed, or whose assignment changed). When every family is
                touched (shared engine code), the smallest model of each
                engine instead. Capped at PR_CAP, smallest first.
  nightly       the assigned set (every kernel_assignment.json model), in
                turns: NIGHTLY_CAP models a night, the window moving by
                NIGHTLY_CAP each day (rotation, the day number), so the
                whole set is benched every few nights; the baseline keeps
                each model's latest numbers.
  dispatch      the models named, uncapped.

Only models the NPU catalog (src/model_list.json, aie2p, not oga-hipep)
lists are benched, and not the families flm bench and the chat benches
cannot measure (NOT_BENCHED: embedding, ASR, the d3 decision models);
SmolVLA is no catalog model. A model is "vlm" (DocVQA too) when
kernel_assignment.json says so. Models over a cap are reported, not benched.

Standard library only.
"""
from __future__ import annotations

import re
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
import changes  # noqa: E402

# Per runtime. A pull request benches each model on Linux XRT and HRX, a
# per-model job each, three at a time per runtime: 4 models keep a PR's
# benches to about two hours of six NPU runners.
PR_CAP = 4
# The assigned set is about 50 models, 40-odd of them benchable; GSM8K alone
# took IRON ~1050 runner-minutes over 49 models per runtime. 16 a night, on
# two runtimes four at a time, is about four hours.
NIGHTLY_CAP = 16
NOT_BENCHED = {"embed-gemma", "whisper-v3", "r2t2", "d3"}
MODEL_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def npu_catalog(catalog: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for mid, entry in catalog.items():
        if changes.runs_on_gpu(mid, entry):
            continue
        platforms = entry.get("supported_platforms")
        if platforms and not changes.RUNNER_PLATFORMS & set(platforms):
            continue
        out[mid] = entry
    return out


def benchable(mid: str, catalog: dict[str, dict]) -> bool:
    return (bool(MODEL_ID.match(mid)) and mid in catalog
            and changes.family_of(catalog[mid]) not in NOT_BENCHED)


def size_of(mid: str, catalog: dict[str, dict]) -> float:
    try:
        return float(catalog.get(mid, {}).get("size") or 0)
    except (TypeError, ValueError):
        return 0.0


def ordered(models: Iterable[str], catalog: dict[str, dict]) -> list[str]:
    return sorted(set(models), key=lambda m: (size_of(m, catalog), m))


def engine_models(engines: changes.DevkitEngines, families: set, catalog: dict[str, dict]) -> list[str]:
    """The benchable models of the engines serving the touched families."""
    every = bool(families) and families >= engines.all_families()
    out = []
    for name in sorted(engines.engines):
        if not engines.families(name) & families:
            continue
        models = [m for m in engines.engines[name]["models"] if benchable(m, catalog)]
        if every:
            # Shared code: one model per engine, the smallest.
            out += ordered(models, catalog)[:1]
        else:
            out += [m for m in models if changes.family_of(catalog[m]) in families]
    return out


def select(*, mode: str, catalog: dict[str, dict], engines: Optional[changes.DevkitEngines],
           assignment: dict, families: Iterable[str] = (), xclbin_models: Iterable[str] = (),
           xclbin_bench_models: Optional[Iterable[str]] = None, requested: Iterable[str] = (),
           cap: Optional[int] = None, rotation: int = 0) -> dict:
    """{"models": [{"model", "vlm"}], "dropped": [...], "skipped": [...]}.
    mode: pr, nightly or dispatch."""
    catalog = npu_catalog(catalog)
    assigned = assignment.get("models", {}) if isinstance(assignment, dict) else {}
    families = set(families)
    if mode == "dispatch":
        wanted = list(dict.fromkeys(requested))
    elif mode == "nightly":
        wanted = list(assigned)
        cap = NIGHTLY_CAP if cap is None else cap
    elif mode == "pr":
        kernels = list(xclbin_models)
        if xclbin_bench_models is not None:
            worth = set(xclbin_bench_models)
            kernels = [m for m in kernels if m in worth]
        wanted = kernels + (engine_models(engines, families, catalog) if engines else [])
        cap = PR_CAP if cap is None else cap
        lists = [ordered([m for m in kernels if benchable(m, catalog)], catalog),
                 ordered([m for m in wanted[len(kernels):] if benchable(m, catalog)], catalog)]
        turns = [m for pair in zip_longest(*lists) for m in pair if m is not None]
        chosen = list(dict.fromkeys(turns))
        skipped = sorted({m for m in wanted if not benchable(m, catalog)})
        return {"models": [{"model": m, "vlm": bool(assigned.get(m, {}).get("vlm"))} for m in chosen[:cap]],
                "dropped": chosen[cap:], "skipped": skipped}
    else:
        raise ValueError(f"unknown mode {mode!r}")
    skipped = sorted({m for m in wanted if not benchable(m, catalog)})
    chosen = [m for m in wanted if benchable(m, catalog)]
    chosen = list(dict.fromkeys(chosen)) if mode == "dispatch" else ordered(chosen, catalog)
    if mode == "nightly" and cap and len(chosen) > cap:
        start = (rotation * cap) % len(chosen)
        chosen = chosen[start:] + chosen[:start]
    dropped = chosen[cap:] if cap is not None else []
    chosen = chosen[:cap] if cap is not None else chosen
    return {"models": [{"model": m, "vlm": bool(assigned.get(m, {}).get("vlm"))} for m in chosen],
            "dropped": dropped, "skipped": skipped}
