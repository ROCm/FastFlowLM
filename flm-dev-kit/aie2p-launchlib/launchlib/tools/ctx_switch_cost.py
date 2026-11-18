#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# ctx_switch_cost.py PROBE XCLBIN_DIR [N] -- the cost of a hw_context switch
# between the xclbins of one model folder (e.g. FastFlowLM's
# xclbins/Qwen3-0.6B-NPU2): for each pair, PROBE (whisper_encoder's
# stream_probe.exe) opens one hw_context on each and runs a do-nothing
# instruction stream (one mask write of mask 0) N times in one context, then N
# times alternating between the two. The cost of a switch: the median of the
# alternating runs minus that of the single-context ones. Prints a table and
# the mean over the pairs.

import itertools
import re
import statistics
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

probe, folder = sys.argv[1], Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
xclbins = sorted(folder.glob("*.xclbin"))

# TXN 0.1, device generation 4 (npu2), 6 rows, 8 columns, 1 mem-tile row; one
# maskwrite32 of mask 0 to shim (0, 0) BD 0: no register changes.
noop_ops = [3, 0, 0x1D000, 0, 0, 0, 28]
header = [0 | 1 << 8 | 4 << 16 | 6 << 24, 8 | 1 << 8, 1, (4 + len(noop_ops)) * 4]  # = 0x06040100, 0x108
noop = Path(tempfile.mkdtemp()) / "noop.txn"
noop.write_bytes(struct.pack(f"<{len(header) + len(noop_ops)}I", *header, *noop_ops))


def run_ms(a, b, streams):
    out = subprocess.run([probe, str(a), "--other", str(b), *streams], capture_output=True, text=True,
                         timeout=600).stdout
    ms = [float(x) for x in re.findall(r"state 4 after ([\d.]+) ms", out)]
    if len(ms) != len(streams):
        raise RuntimeError(f"{a.name} / {b.name}: {len(ms)} of {len(streams)} runs completed")
    return ms


print(f"{folder.name}: {len(xclbins)} xclbins, {n} runs each way")
costs = []
for a, b in itertools.combinations(xclbins, 2):
    same = run_ms(a, b, [str(noop)] * n)
    alt = run_ms(a, b, [str(noop), f"other:{noop}"] * (n // 2))
    cost = statistics.median(alt) - statistics.median(same)
    costs.append(cost)
    print(f"  {a.stem:>22} <-> {b.stem:<22} one context {statistics.median(same):6.3f} ms, "
          f"alternating {statistics.median(alt):6.3f} ms: switch {cost:6.3f} ms")
print(f"  mean switch {statistics.mean(costs):.3f} ms, min {min(costs):.3f}, max {max(costs):.3f}")
