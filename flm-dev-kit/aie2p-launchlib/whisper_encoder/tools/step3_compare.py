#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# step3_compare.py REF VARIANT... -- step 3: compares the npu_test ASR reports of
# build/step3/<variant>/ (tools/step3_asr.sh): error rate and wall time per set,
# and per clip whether the transcript equals REF's. Prints the clips that differ.

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def report(variant):
    return json.loads((HERE / "build/step3" / variant / "asr-whisper-v3_turbo.json").read_text())


ref_name, names = sys.argv[1], sys.argv[1:]
reports = {n: report(n) for n in names}
ref = reports[ref_name]

print("error rate / wall s per set:")
print(f"  {'set':12}" + "".join(f"{n:>22}" for n in names))
for s in ref["sets"]:
    row = "".join(f"{reports[n]['sets'][s]['error_rate']:>12.4f} {reports[n]['sets'][s]['wall_s']:>8.1f} s" for n in names)
    print(f"  {s:12}{row}")
print(f"  {'overall':12}" + "".join(f"{reports[n]['overall']['error_rate']:>12.4f}{'':>10}" for n in names))

for n in names[1:]:
    items = reports[n]["items"]
    same = [k for k in ref["items"] if items.get(k, {}).get("text") == ref["items"][k]["text"]]
    print(f"\n{n} vs {ref_name}: {len(same)} of {len(ref['items'])} transcripts identical")
    for k in ref["items"]:
        if k in same:
            continue
        a, b = ref["items"][k], items.get(k, {})
        print(f"  {k}: edits {a['edits']} -> {b.get('edits')} (of {a['length']})")
        print(f"    {ref_name}: {a['text'][:160]}")
        print(f"    {n}: {b.get('text', '')[:160]}")
