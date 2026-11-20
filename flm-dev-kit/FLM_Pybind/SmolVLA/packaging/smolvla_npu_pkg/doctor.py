# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Environment check for the prebuilt smolvla_npu extension.

The extension ships as a binary, so the tester cannot read the build flags to
find out why it will not load or why it crashed. Every check here corresponds
to one way that goes wrong, in the order the failures actually happen:
interpreter ABI -> CPU instruction set -> system libraries -> XRT device ->
python imports -> bitstreams -> model directory.

The CPU check is the one worth reading twice. The extension is compiled with
-march=native on a Zen 4 host, so it contains AVX-512 (including vdpbf16ps).
Those instructions are not guarded at runtime -- on a CPU without them the
process dies with SIGILL and no python traceback, which looks like a crash in
the model rather than a portability problem. This reports it up front.

Run as `smolvla-doctor`. Exit status is 0 when nothing blocking was found.

--model-path is optional: the model directory is an argument to smolvla-verify,
not part of the install, so it is skipped rather than flagged when absent.
"""

import argparse
import ctypes.util
import os
import platform
import shutil
import subprocess
import sys
import sysconfig

# Built with -march=native on znver4. avx512_bf16 backs vdpbf16ps, which is
# present in the shipped binary; the rest come with any AVX-512 capable core.
REQUIRED_CPU_FLAGS = ["avx2", "fma", "avx512f", "avx512bw", "avx512dq", "avx512vl", "avx512_bf16"]

# gcc-13 / Ubuntu 24.04 build host.
REQUIRED_GLIBC = (2, 38)
REQUIRED_GLIBCXX = (3, 4, 32)

MODEL_FILES = ["config.json", "model.safetensors"]

# The bitstreams ship inside this package (see build_package.sh), not with the
# model directory: they are artifacts of the kernel build and go stale against
# the extension, not against the checkpoint. The extension looks for them in
# XCLBINS/ beside itself, which is this directory.
XCLBINS = ["vision.xclbin", "text.xclbin", "expert.xclbin"]
XCLBIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "XCLBINS")

OK, INFO, WARN, FAIL = "ok", "info", "warn", "FAIL"


class Report:
    """Collects check results so the summary can count what actually blocks."""

    def __init__(self):
        self.rows = []

    def add(self, status, name, detail, fix=None):
        self.rows.append((status, name, detail, fix))
        mark = {OK: "  ok  ", INFO: " ---- ", WARN: " warn ", FAIL: " FAIL "}[status]
        print(f"[{mark}] {name}: {detail}")
        if fix and status not in (OK, INFO):
            for line in fix.splitlines():
                print(f"          {line}")

    @property
    def checked(self):
        """Rows that are a verdict on the environment. INFO is a note about
        something that was not checked, so it neither passes nor warns."""
        return [r for r in self.rows if r[0] != INFO]

    @property
    def failures(self):
        return [r for r in self.rows if r[0] == FAIL]

    @property
    def warnings(self):
        return [r for r in self.rows if r[0] == WARN]


def check_interpreter(rep):
    want = "cpython-312-x86_64-linux-gnu"
    have = (sysconfig.get_config_var("EXT_SUFFIX") or "").removesuffix(".so").lstrip(".")
    ver = platform.python_version()
    if have == want:
        rep.add(OK, "interpreter", f"python {ver} ({have})")
    else:
        rep.add(FAIL, "interpreter", f"python {ver} ({have or 'unknown'}), need {want}",
                "A CPython extension only loads under the interpreter ABI it was\n"
                "built for. Recreate the venv with python3.12.")


def _cpu_flags():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def check_cpu(rep):
    flags = _cpu_flags()
    if not flags:
        rep.add(WARN, "cpu", "could not read /proc/cpuinfo; skipping ISA check")
        return
    missing = [f for f in REQUIRED_CPU_FLAGS if f not in flags]
    if not missing:
        rep.add(OK, "cpu", "AVX-512 + bf16 present")
    else:
        rep.add(FAIL, "cpu", f"missing {', '.join(missing)}",
                "This build is compiled with -march=native on Zen 4 and issues these\n"
                "instructions unguarded. On this CPU it will die with SIGILL (often\n"
                "reported as 'Illegal instruction' with no traceback) rather than\n"
                "raising a python error. It needs a rebuild targeting this machine.")


def _libc_version():
    try:
        out = subprocess.run(["ldd", "--version"], capture_output=True, text=True, timeout=10)
        first = out.stdout.splitlines()[0]
        tok = first.split()[-1]
        return tuple(int(x) for x in tok.split(".")[:2])
    except Exception:
        return None


def check_glibc(rep):
    ver = _libc_version()
    if ver is None:
        rep.add(WARN, "glibc", "could not determine version")
        return
    s = ".".join(map(str, ver))
    if ver >= REQUIRED_GLIBC:
        rep.add(OK, "glibc", s)
    else:
        rep.add(FAIL, "glibc", f"{s}, need >= {'.'.join(map(str, REQUIRED_GLIBC))}",
                "The extension references newer glibc symbols than this system\n"
                "provides. It cannot be made to load here without a rebuild.")


def _libstdcxx_path():
    # ctypes.util.find_library goes through the loader cache, which is what the
    # extension itself will use, so this resolves the same file it would get.
    name = ctypes.util.find_library("stdc++")
    if not name:
        return None
    try:
        lib = ctypes.CDLL(name)
        return getattr(lib, "_name", name)
    except OSError:
        return None


def check_libstdcxx(rep):
    """Verify GLIBCXX_3.4.32 is present -- the gcc-13 ABI level."""
    want = "GLIBCXX_" + ".".join(map(str, REQUIRED_GLIBCXX))
    path = _libstdcxx_path()
    if not path:
        rep.add(WARN, "libstdc++", "not found via the loader; skipping ABI check")
        return
    real = path
    for cand in (path, os.path.join("/lib/x86_64-linux-gnu", os.path.basename(path))):
        if os.path.exists(cand):
            real = cand
            break
    if not shutil.which("strings") or not os.path.exists(real):
        rep.add(WARN, "libstdc++", f"found {path}, could not verify {want}")
        return
    try:
        out = subprocess.run(["strings", "-a", real], capture_output=True, text=True, timeout=60)
        if want in out.stdout:
            rep.add(OK, "libstdc++", f"{want} present")
        else:
            rep.add(FAIL, "libstdc++", f"{want} not found in {real}",
                    "Built with gcc-13. Install a newer libstdc++6 (Ubuntu 24.04+),\n"
                    "or `sudo apt install gcc-13`.")
    except Exception:
        rep.add(WARN, "libstdc++", f"found {real}, could not verify {want}")


def check_xrt(rep):
    """libxrt_coreutil.so.2 is a hard link-time dependency of the extension."""
    found = ctypes.util.find_library("xrt_coreutil")
    if found:
        rep.add(OK, "XRT runtime", f"libxrt_coreutil found ({found})")
        return
    alt = "/opt/xilinx/xrt/lib/libxrt_coreutil.so.2"
    if os.path.exists(alt):
        rep.add(WARN, "XRT runtime", f"present at {alt} but not on the loader path",
                "export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH\n"
                "(or source the XRT setup script: . /opt/xilinx/xrt/setup.sh)")
        return
    rep.add(FAIL, "XRT runtime", "libxrt_coreutil.so.2 not found",
            "The extension links XRT directly and will not import without it.\n"
            "Install XRT with NPU (XDNA) support, then re-run.")


def check_npu_device(rep):
    """A loadable extension still needs an actual NPU to run forward()."""
    nodes = [p for p in ("/dev/accel/accel0", "/dev/accel/accel1") if os.path.exists(p)]
    if nodes:
        rep.add(OK, "NPU device", ", ".join(nodes))
        return
    if os.path.isdir("/dev/accel"):
        rep.add(WARN, "NPU device", "/dev/accel exists but has no accel* node",
                "The XDNA driver is loaded but no NPU was enumerated. Check that the\n"
                "NPU is enabled in BIOS.")
    else:
        rep.add(WARN, "NPU device", "/dev/accel not present",
                "No XDNA driver. Import will succeed but constructing SmolVLA() will\n"
                "fail. Install the amdxdna driver for your kernel.")


def check_imports(rep):
    """The python side: torch/lerobot for the harness, then the extension."""
    for mod, label in (("torch", "torch"), ("lerobot", "lerobot")):
        try:
            m = __import__(mod)
            rep.add(OK, label, getattr(m, "__version__", "installed"))
        except Exception as e:
            rep.add(FAIL, label, f"import failed: {type(e).__name__}: {e}",
                    "Re-run setup.sh, or activate the venv it created.")

    try:
        import torch
        if torch.version.cuda:
            rep.add(WARN, "torch build", f"CUDA build ({torch.version.cuda})",
                    "Expected the CPU wheel. Harmless, but it pulled in several GB of\n"
                    "unused nvidia-* packages.")
    except Exception:
        pass

    try:
        import smolvla_npu
        rep.add(OK, "smolvla_npu", f"loaded (MAX_IMAGES={smolvla_npu.MAX_IMAGES})")
    except Exception as e:
        # The shim already attaches a diagnosis; show its first two lines.
        head = "; ".join(str(e).splitlines()[:2]).strip()
        rep.add(FAIL, "smolvla_npu", f"import failed: {head}")


def check_xclbins(rep):
    """The NPU bitstreams, which are part of the install rather than the model.

    A missing one is a broken install, not a bad --model-path, and it does not
    surface until NPU bring-up several seconds into a run -- so name it here.
    SMOLVLA_XCLBIN_DIR overrides the search; report that when it is set, since
    a stale value is otherwise invisible.
    """
    override = os.environ.get("SMOLVLA_XCLBIN_DIR")
    xdir = override or XCLBIN_DIR
    missing = [x for x in XCLBINS if not os.path.exists(os.path.join(xdir, x))]

    if not missing:
        where = f"{xdir} (SMOLVLA_XCLBIN_DIR)" if override else xdir
        rep.add(OK, "bitstreams", f"all {len(XCLBINS)} present in {where}")
        return

    detail = f"{xdir} missing " + ", ".join(missing)
    if override:
        rep.add(FAIL, "bitstreams", detail,
                "SMOLVLA_XCLBIN_DIR points at a directory that does not hold all\n"
                "three bitstreams. Unset it to use the ones shipped in the wheel.")
    else:
        rep.add(FAIL, "bitstreams", detail,
                "The bitstreams ship inside this package, so this is a broken\n"
                "install rather than a missing model file. Reinstall the wheel.")


def check_model(rep, model_path):
    # Not a failure and not a warning: the model directory is a run-time
    # argument, so an environment without one is still a correct install. Say
    # it was skipped rather than implying something is wrong.
    if not model_path:
        rep.add(INFO, "model dir", "not checked (pass --model-path to validate one)")
        return
    if not os.path.isdir(model_path):
        rep.add(FAIL, "model dir", f"{model_path} is not a directory")
        return

    missing = [f for f in MODEL_FILES if not os.path.exists(os.path.join(model_path, f))]
    if not missing:
        rep.add(OK, "model dir", f"{model_path} complete")
        return
    rep.add(FAIL, "model dir", f"{model_path} missing " + ", ".join(missing),
            "The directory must hold config.json and model.safetensors. Copy it\n"
            "from the machine that produced it -- it is not part of this package.\n"
            "An XCLBINS/ subdirectory, if it has one, is left over from an older\n"
            "layout and is ignored: the bitstreams ship with the wheel now.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="check the smolvla_npu runtime environment")
    ap.add_argument("--model-path", default=os.environ.get("SMOLVLA_MODEL_PATH"),
                    help="NPU model directory to validate. Optional -- it is a "
                         "run-time argument to smolvla-verify, not part of the "
                         "install (default: $SMOLVLA_MODEL_PATH)")
    args = ap.parse_args(argv)

    print(f"smolvla-npu doctor -- {platform.platform()}\n")
    rep = Report()
    check_interpreter(rep)
    check_cpu(rep)
    check_glibc(rep)
    check_libstdcxx(rep)
    check_xrt(rep)
    check_npu_device(rep)
    check_imports(rep)
    check_xclbins(rep)
    check_model(rep, args.model_path)

    print()
    if rep.failures:
        print(f"{len(rep.failures)} blocking problem(s): "
              + ", ".join(r[1] for r in rep.failures))
        return 1
    if rep.warnings:
        print(f"no blocking problems; {len(rep.warnings)} warning(s): "
              + ", ".join(r[1] for r in rep.warnings))
        return 0
    print(f"all {len(rep.checked)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
