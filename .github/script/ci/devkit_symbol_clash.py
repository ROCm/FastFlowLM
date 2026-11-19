#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Fail if two dev-kit engine libraries export the same process-wide singleton.

    devkit_symbol_clash.py <lib dir> [--allow NAME ...]

On Linux flm loads every engine library into one process. A GNU-unique symbol
(an `inline` namespace-scope variable, or a function-local static of an
`inline` function; `nm` type `u`) is ONE object for the whole process: if two
libraries define the same name, both use the copy the dynamic loader
initialised first. It compiles and links without a warning; at run time one
engine runs with the other's data. -Wl,-Bsymbolic-functions does not change
this, and Windows DLLs do not share symbols this way, so only Linux breaks.

Seen 2026-11-07: MiniCPM-OCR's vision overlay reused the Qwen3.5-2B flash
overlay's global names (op_mm, op_attn, all_channels, the
vision_overlay::gen_switch() statics), and every flash image request hung
(FastFlowLM PR #983). Fix a clash by putting the engine's own tables and
helpers in an engine-specific namespace. ALLOWED_PREFIXES lists names every
engine compiles from the same shared header with the same value; add one only
after checking that it is the same in every library.

Run by devkit-engines.yml on flm-dev-kit/FLM_DLL/build/lib (Linux XRT and
HRX). Ported from the internal FLM_IRON CI. Standard library and binutils' nm.
"""
import argparse
import collections
import glob
import os
import subprocess
import sys

# Same name in several engine libraries, reviewed and known to hold the same value in all of them.
ALLOWED_PREFIXES = {
    "std::",
    "hrx::",
    "npu_app::_setup_kernel()::dump_idx",
    "cfg_sub",
    "Dequant::Impl::IT",
    "Gemm::Impl::shim_tiles",
    "LMHead::Impl::lm_head_tiles",
    "dfl::kTapLayers",
    "qwen3_5vl_desc::inv_freq",
    "qwen3vl_desc::inv_freq",
    "gemma4e_cpu_func::inv_freq_swa",
    "gemma4e_cpu_func::inv_freq_global",
    # gemma4e_npu/ and gemma4e_dflash/ each carry a gemma4e_bfp_gemm.hpp; identical on 2026-11-08.
    "gemma4e_bfp::",
    # One thread-local from include/npu_utils/runtime_wait.hpp, included by every
    # engine. It starts empty. An HRX wait on this thread fills it and the
    # completion check on that same thread reads it back.
    "flm::npu::ert_failure_detail",
}
SKIPPED_LIBS = ("xrt", "boost")


def unique_symbols(lib_path):
    out = subprocess.run(["nm", "-D", "-C", "--defined-only", lib_path],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        kind, name = parts[1], parts[2]
        if kind == "u" and not name.startswith("guard variable"):
            yield name


def is_allowed(name, extra_prefixes):
    return any(name.startswith(prefix) for prefix in ALLOWED_PREFIXES | set(extra_prefixes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("libdir")
    parser.add_argument("--allow", nargs="*", default=[])
    args = parser.parse_args()

    owners = collections.defaultdict(set)
    libs = sorted(glob.glob(os.path.join(args.libdir, "lib*.so")))
    for lib in libs:
        base = os.path.basename(lib)
        if any(word in base for word in SKIPPED_LIBS):
            continue
        for name in unique_symbols(lib):
            owners[name].add(base)

    clashes = {}
    for name, libs_with_it in owners.items():
        if len(libs_with_it) > 1 and not is_allowed(name, args.allow):
            clashes[name] = sorted(libs_with_it)

    for name in sorted(clashes):
        print(f"CLASH {name}\n      defined in: {', '.join(clashes[name])}")
    print(f"checked {len(libs)} libraries, {len(owners)} unique symbols, {len(clashes)} clashes")
    if clashes:
        print("Rename or namespace these per engine (README.md next to this script).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
