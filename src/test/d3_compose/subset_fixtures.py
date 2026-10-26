#!/usr/bin/env python3
"""Cut the full fixtures of gen_compose_fixtures.py down to the compact set that is committed.

    python subset_fixtures.py <full fixtures dir> fixtures [--every 5]

Every N-th record of compose_requests.jsonl, compose_offsets.jsonl and of each frequent kind of compose_units.jsonl
is kept (kinds with fewer than 50 records are kept whole), so every kind of case stays covered with the same mix.
The small files (decision_config.json, readout.json, vocab_bytes.json) are copied unchanged.
The tests take the fixtures directory as an argument; the full set is used with
`cmake -DFLM_D3_COMPOSE_FIXTURES=<full dir>` (or ctest's command line `test_d3_compose <dir> - <fuzz.jsonl>`).
"""
import argparse
import collections
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=5)
    a = ap.parse_args()
    full, out = Path(a.full), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("decision_config.json", "readout.json", "vocab_bytes.json"):
        shutil.copyfile(full / name, out / name)
    for name in ("compose_requests.jsonl", "compose_offsets.jsonl"):
        lines = (full / name).read_text().split("\n")[:-1]
        (out / name).write_text("".join(l + "\n" for i, l in enumerate(lines) if i % a.every == 0))
    units = (full / "compose_units.jsonl").read_text().split("\n")[:-1]
    kinds = collections.Counter(json.loads(l)["kind"] for l in units)
    seen = collections.Counter()
    keep = []
    for l in units:
        k = json.loads(l)["kind"]
        if kinds[k] < 50 or seen[k] % a.every == 0:
            keep.append(l)
        seen[k] += 1
    (out / "compose_units.jsonl").write_text("".join(l + "\n" for l in keep))


if __name__ == "__main__":
    main()
