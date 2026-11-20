# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Top-level ``smolvla_npu`` shim: loads the prebuilt extension and becomes it.

The compiled ``.so`` lives inside ``smolvla_npu_pkg/`` so the wheel owns every
file it installs, but callers (and the C++ module's own ``PYBIND11_MODULE``
name) expect a top-level ``smolvla_npu``. This module loads the extension from
the package directory and replaces itself with it in ``sys.modules``, so
``import smolvla_npu`` yields the real extension and ``from smolvla_npu import
SmolVLA`` works unchanged.

The extension's init symbol is ``PyInit_smolvla_npu``, so the name passed to the
loader below must stay ``smolvla_npu`` -- loading it under any other name fails
with a missing-init error.

Import failures here are the single most likely thing a tester hits, and the
raw loader message ("cannot open shared object file") names the missing library
without saying what to do about it. Each known cause is re-raised with the fix
attached; run ``smolvla-doctor`` for the full environment check.
"""

import importlib.machinery
import importlib.util
import os
import sys
import sysconfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT_DIR = os.path.join(_HERE, "smolvla_npu_pkg")
_EXT_NAME = "smolvla_npu.cpython-312-x86_64-linux-gnu.so"
_EXT_PATH = os.path.join(_EXT_DIR, _EXT_NAME)


def _abi_hint():
    """Explain an interpreter mismatch, or None if the interpreter is fine."""
    want = "cpython-312-x86_64-linux-gnu"
    have = (sysconfig.get_config_var("EXT_SUFFIX") or "").removesuffix(".so").lstrip(".")
    if have and have != want:
        return (f"this extension is built for {want} but you are running "
                f"{have} (python {sys.version.split()[0]}). A CPython "
                f"extension only loads under the interpreter it was built "
                f"for -- use python3.12.")
    return None


def _load():
    if not os.path.exists(_EXT_PATH):
        hint = _abi_hint()
        raise ImportError(
            f"the smolvla_npu extension is missing from {_EXT_DIR}."
            + (f" {hint}" if hint else " The wheel may be corrupt; reinstall it.")
        )

    loader = importlib.machinery.ExtensionFileLoader("smolvla_npu", _EXT_PATH)
    spec = importlib.util.spec_from_loader("smolvla_npu", loader, origin=_EXT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Publish before exec_module: the extension's init runs during exec, and
    # anything it imports that reaches back for smolvla_npu must find it.
    sys.modules["smolvla_npu"] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        # Undo the optimistic registration so a retry re-runs this path rather
        # than handing out a half-initialised module.
        sys.modules.pop("smolvla_npu", None)
        raise ImportError(_diagnose(str(e))) from e
    return module


def _diagnose(msg):
    """Turn a dynamic-loader error into one that names the fix."""
    lines = [f"failed to load the smolvla_npu extension from {_EXT_PATH}", "", str(msg), ""]

    if "libxrt_coreutil" in msg:
        lines += [
            "Cause: the XRT runtime is not installed, or not on the loader path.",
            "Fix:   install XRT (it provides libxrt_coreutil.so.2), or point the",
            "       loader at an existing install:",
            "           export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH",
        ]
    elif "GLIBCXX" in msg or "CXXABI" in msg:
        lines += [
            "Cause: the system libstdc++ is older than the one this extension was",
            "       built against (gcc-13 / GLIBCXX_3.4.32).",
            "Fix:   install a newer libstdc++ (Ubuntu: `sudo apt install libstdc++6`",
            "       from a 24.04-or-newer release, or `sudo apt install gcc-13`).",
        ]
    elif "GLIBC_" in msg:
        lines += [
            "Cause: the system glibc is older than the build host's (needs >= 2.38).",
            "Fix:   this build cannot run here; it needs a rebuild on your distro.",
        ]
    elif "libgomp" in msg:
        lines += [
            "Cause: the OpenMP runtime is missing.",
            "Fix:   sudo apt install libgomp1",
        ]
    else:
        hint = _abi_hint()
        if hint:
            lines += [f"Cause: {hint}"]

    lines += ["", "Run `smolvla-doctor` for a full environment check."]
    return "\n".join(lines)


_load()
