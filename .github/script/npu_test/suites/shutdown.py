"""Ctrl+C and /bye end flm, run in a console of its own as a person runs it.

A process that had loaded hip-ep (the oga-hipep backend) used to hang on its
way out for good: ExitProcess waited in the HIP runtime's DLL teardown, and the
process kept its GPU memory, unkillable, until a reboot. Each case starts flm
in a hidden console of its own (console.py), so Ctrl+C reaches flm alone and
typed keys reach its prompt, loads the model, quits, and expects flm gone
within QUIT_TIMEOUT seconds:

  serve-ctrl-c  flm serve, after a chat request, then Ctrl+C
  run-bye       flm run at its prompt, then /bye
  run-ctrl-c    flm run at its prompt, then Ctrl+C

Windows only. Only a GPU backend run (--backend) gives it models: NPU processes
quit through the same paths and always have. A process that survives a kill
holds the GPU until the runner reboots, so the cases after one are skipped.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import time
import urllib.error

from .. import console
from ..client import Client
from ..harness import expect, skip

NAME = "shutdown"
DESCRIPTION = "Ctrl+C and /bye end flm serve and flm run within seconds"

# flm ends well within a second; the hang never ends.
QUIT_TIMEOUT = 30
PROMPT = ">>>"

# (case, flm command, how to quit)
CASES = (
    ("serve-ctrl-c", "serve", "Ctrl+C"),
    ("run-bye", "run", "/bye"),
    ("run-ctrl-c", "run", "Ctrl+C"),
)


def run(ctx) -> None:
    models = ctx.plan.suite(NAME)
    stuck = ""
    for model in models:
        for name, command, how in CASES:
            with ctx.case(NAME, name, model) as case:
                if os.name != "nt":
                    skip("needs a Windows console")
                if stuck:
                    skip(f"not run after {stuck}, whose flm could not be ended")
                (serve if command == "serve" else run_prompt)(ctx, model, how, case)
            if case.data.get("stuck"):
                stuck = f"{name} [{model}]"


def serve(ctx, model: str, how: str, case) -> None:
    port = ctx.port or free_port()
    log_path = ctx.out_dir / f"server-{NAME}-{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}.log"
    case.data["log"] = log_path.name
    command = [ctx.flm_bin, "serve", "--host", "127.0.0.1", "--port", str(port),
               "-a", "0", "-e", "0", *extension_args(ctx)]
    with open(log_path, "wb") as log:
        proc = console.start(command, stdout=log)
        try:
            client = Client(f"http://127.0.0.1:{port}", timeout=ctx.request_timeout)
            wait_ready(proc, client, ctx.request_timeout)
            client.chat(model, [{"role": "user", "content": "Reply with one word: hello"}],
                        stream=False, max_tokens=16)
            quit_flm(proc, how, case)
        finally:
            end(proc, case)


def run_prompt(ctx, model: str, how: str, case) -> None:
    proc = console.start([ctx.flm_bin, "run", model, *extension_args(ctx)])
    try:
        wait_prompt(proc, ctx.request_timeout)
        quit_flm(proc, how, case)
    finally:
        end(proc, case)


def quit_flm(proc, how: str, case) -> None:
    """Quits flm the way a person does and expects it gone within QUIT_TIMEOUT."""
    if how == "/bye":
        console.type_line(proc.pid, "/bye")
        codes = (0,)
    else:
        console.ctrl_c(proc.pid)
        # The default handler's code; a handler of flm's own may also end it with 0.
        codes = (console.STATUS_CONTROL_C_EXIT, 0)
    started = time.monotonic()
    try:
        proc.wait(QUIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(15)
            left = "it ended when killed"
        except subprocess.TimeoutExpired:
            case.data["stuck"] = True
            left = f"pid {proc.pid} survived a kill too and holds its memory until a reboot"
        raise AssertionError(f"still running {QUIT_TIMEOUT}s after {how}; {left}") from None
    case.data.update(exit_code=f"0x{exit_code(proc):08X}",
                     seconds_to_exit=round(time.monotonic() - started, 1))
    expect(exit_code(proc) in codes,
           f"exit code 0x{exit_code(proc):08X} after {how}, expected "
           + " or ".join(f"0x{code:08X}" for code in codes))


def end(proc, case) -> None:
    """Kills an flm a failed case left running; marks the case if even that fails."""
    if proc.poll() is not None:
        return
    proc.kill()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        case.data["stuck"] = True


def wait_ready(proc, client: Client, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"flm serve exited with code 0x{exit_code(proc):08X} "
                                 f"before it was ready")
        try:
            client.get_json("/v1/version", timeout=5)
            return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    raise AssertionError(f"flm serve did not answer within {timeout:.0f}s")


def wait_prompt(proc, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    text = ""
    while time.monotonic() < deadline:
        try:
            text = console.screen(proc.pid)
        except RuntimeError:
            if proc.poll() is None:
                raise
        if proc.poll() is not None:
            raise AssertionError(f"flm run exited with code 0x{exit_code(proc):08X} "
                                 f"before its prompt; screen: {last_lines(text)}")
        if PROMPT in text:
            return
        time.sleep(2)
    raise AssertionError(f"no {PROMPT} prompt within {timeout:.0f}s; screen: {last_lines(text)}")


def exit_code(proc) -> int:
    return proc.returncode & 0xFFFFFFFF


def last_lines(text: str, count: int = 3) -> str:
    return " | ".join(text.splitlines()[-count:])[-300:]


def extension_args(ctx) -> list[str]:
    return [arg for manifest in ctx.model_extensions for arg in ("--model-extension", manifest)]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
