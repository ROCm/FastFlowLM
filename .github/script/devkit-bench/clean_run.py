#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Runs a command in an emptied environment, as npu-test's clean-env does.

    clean_run.py -- <command> [args...]

Keeps only what flm and the bench scripts need: PATH, the user's home,
temp and locale variables (on Windows also the system ones a process
needs: SystemRoot, ProgramFiles, ...), PYTHONUTF8, the proxy and CA
variables (flm downloads weights through the runners' proxy) and every
FLM_*, XRT_*, XILINX_* and HRX_* variable. No token, runner or GitHub
variable reaches the code under test. Exits with the command's code.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

KEEP = {
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "PYTHONUTF8",
    "http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE",
    # Windows
    "Path", "PATHEXT", "SystemRoot", "SystemDrive", "windir", "ComSpec", "OS", "TEMP", "TMP",
    "USERPROFILE", "USERNAME", "USERDOMAIN", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
    "ProgramData", "ALLUSERSPROFILE", "PUBLIC", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432",
    "CommonProgramFiles", "CommonProgramFiles(x86)", "CommonProgramW6432", "COMPUTERNAME",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
}
KEEP_PREFIX = re.compile(r"^(FLM|XRT|XILINX|HRX)_")


def clean_env(env: dict) -> dict:
    # Windows names are case-insensitive: compare them so.
    keep = {k.upper() for k in KEEP} if os.name == "nt" else KEEP
    return {k: v for k, v in env.items()
            if (k.upper() if os.name == "nt" else k) in keep or KEEP_PREFIX.match(k)}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        print("usage: clean_run.py -- <command> [args...]", file=sys.stderr)
        return 2
    env = clean_env(dict(os.environ))
    print(f"Clean environment: {' '.join(sorted(env))}", flush=True)
    return subprocess.run(argv, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
