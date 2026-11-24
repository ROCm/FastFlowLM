#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Describes the NPU machine a bench ran on, so perf is only compared like
with like.

    machine.py --out machine.json [--runtime xrt|hrx]
               [--meta meta.json --key lin-xrt --model <tag> --vlm true|false]

The NPU runners carry no kind label, so a job lands on whichever Strix,
Strix Halo or Krackan box is free, and their tokens/s differ far more than
the 7% a regression warning looks for. kind is what perf baselines and
comparisons are keyed on:

  Linux    /sys/class/accel/accel*/device/vbnv (driver-provided, no XRT
           needed, so HRX too: "NPU Strix", "NPU Strix Halo", "NPU Krackan"),
           else the CPU brand from /proc/cpuinfo
  Windows  the CPU brand (Win32_Processor.Name)

CPU brand to kind: "Ryzen AI MAX" is Strix Halo, "Ryzen AI 9 (HX) 36x/37x"
Strix, "Ryzen AI 5/7 33x-35x" Krackan; anything else "unknown", which is
compared with nothing. Also recorded: CPU, PCI ids and revision, OS,
runtime and the runner's name (the artifact only; summaries show the kind).
--meta also writes the bench job's identity (runtime key, model, vlm),
which report.py reads.

Standard library only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import subprocess
import sys
from typing import Optional

CPU_KINDS = (
    (re.compile(r"Ryzen\s+AI\s+MAX", re.I), "strix-halo"),
    (re.compile(r"Ryzen\s+AI\s+9\s+(HX\s+)?3[67]\d", re.I), "strix"),
    (re.compile(r"Ryzen\s+AI\s+[57]\s+(PRO\s+)?3[345]\d", re.I), "krackan"),
)
KIND = re.compile(r"^[a-z0-9-]{1,32}$")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:32]


def kind_from(vbnv: str = "", cpu: str = "") -> str:
    """The machine kind from the NPU's name (vbnv), else the CPU brand."""
    name = (vbnv or "").strip()
    if name:
        name = re.sub(r"^(AMD\s+)?NPU\s*", "", name, flags=re.I)
        if name:
            return slug(name)
    for pattern, kind in CPU_KINDS:
        if pattern.search(cpu or ""):
            return kind
    return "unknown"


def valid_kind(kind) -> bool:
    return isinstance(kind, str) and bool(KIND.match(kind)) and kind != "unknown"


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return ""


def linux_info() -> dict:
    info: dict = {}
    for dev in sorted(glob.glob("/sys/class/accel/accel*/device")):
        vbnv = _read(os.path.join(dev, "vbnv"))
        if vbnv:
            info.update(vbnv=vbnv, pci_vendor=_read(os.path.join(dev, "vendor")),
                        pci_device=_read(os.path.join(dev, "device")),
                        pci_revision=_read(os.path.join(dev, "revision")))
            break
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            info["cpu"] = line.split(":", 1)[1].strip()
            break
    return info


def windows_info() -> dict:
    try:
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        cpu = ""
    return {"cpu": cpu}


def describe(runtime: str = "", os_name: Optional[str] = None) -> dict:
    os_name = os_name or ("windows" if os.name == "nt" else "linux")
    info = windows_info() if os_name == "windows" else linux_info()
    return {"kind": kind_from(info.get("vbnv", ""), info.get("cpu", "")), "os": os_name,
            "runtime": runtime, "runner": os.environ.get("RUNNER_NAME", platform.node()), **info}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--runtime", default="")
    ap.add_argument("--meta", default="")
    ap.add_argument("--key", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--vlm", default="false")
    args = ap.parse_args(argv)
    if args.meta:
        with open(args.meta, "w", encoding="utf-8") as f:
            json.dump({"key": args.key, "model": args.model, "vlm": args.vlm == "true"}, f, indent=2)
    info = describe(args.runtime)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"machine kind: {info['kind']} ({info.get('vbnv') or info.get('cpu') or 'nothing detected'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
