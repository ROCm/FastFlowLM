"""Starts, watches and stops one `flm serve` process.

Each suite gets its own server with the flags it needs (`-a 1` for Whisper,
`--prefix-slots 2`, ...), so a crash in one suite cannot leak into the next,
and its log is kept as a separate file in the report.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .client import Client

IS_WINDOWS = os.name == "nt"
# Matches utils::get_server_port() in src/common/utils.cpp.
DEFAULT_PORT = 52625


class ServerError(RuntimeError):
    pass


class FlmServer:
    def __init__(self, flm_bin: str, *, name: str, log_dir: Path,
                 args: list[str] | None = None, host: str = "127.0.0.1",
                 port: int | None = None, ready_timeout: float = 300.0,
                 request_timeout: float = 600.0, max_restarts: int = 3):
        self.flm_bin = flm_bin
        self.name = name
        self.args = list(args or [])
        self.host = host
        # None: a free port per start. A server stuck in the NPU driver cannot
        # be killed and keeps its port, which would otherwise stop every later
        # suite from starting.
        self.fixed_port = port
        self.port = port or DEFAULT_PORT
        self.unkillable = False
        self.ready_timeout = ready_timeout
        self.max_restarts = max_restarts
        self.restarts = 0
        self.log_path = Path(log_dir) / f"server-{name}.log"
        self.proc: subprocess.Popen | None = None
        self._log = None
        self.client = Client(f"http://{host}:{self.port}", timeout=request_timeout)
        self.version = "unknown"

    @property
    def base_url(self) -> str:
        return self.client.base_url

    @property
    def command(self) -> list[str]:
        # -a / -e default to 0 so a suite only gets the sidecars it asks for.
        cmd = [self.flm_bin, "serve", "--host", self.host, "--port", str(self.port)]
        if "-a" not in self.args and "--asr" not in self.args:
            cmd += ["-a", "0"]
        if "-e" not in self.args and "--embed" not in self.args:
            cmd += ["-e", "0"]
        return cmd + self.args

    # -- lifecycle -------------------------------------------------------------

    def __enter__(self) -> "FlmServer":
        self.start()
        return self

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

    def _port_answers(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/v1/version", timeout=3):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.host, 0))
            return sock.getsockname()[1]

    def start(self, append: bool = False) -> None:
        if self.fixed_port is None:
            self.port = self._free_port()
        # Suites hold on to self.client, so move it rather than replace it.
        self.client.base_url = f"http://{self.host}:{self.port}"
        # A server left behind by an earlier job (or anything else on the port)
        # would answer every request in place of the one under test.
        if self._port_answers():
            raise ServerError(f"port {self.port} already answers /v1/version; another "
                              f"server is running, so the package under test cannot start")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab" if append else "wb")
        if append:
            self._log.write(f"\n=== restart {self.restarts} ===\n".encode())
            self._log.flush()
        print(f"[server:{self.name}] {' '.join(self.command)}", flush=True)
        kwargs: dict = {"stdout": self._log, "stderr": subprocess.STDOUT}
        if IS_WINDOWS:
            # So Ctrl-Break can stop the server without also stopping us.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(self.command, **kwargs)
        self._wait_ready()
        self.version = self.client.version()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise ServerError(f"flm serve exited with code {self.proc.returncode} "
                                  f"before it was ready; see {self.log_path.name}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/v1/version", timeout=5):
                    pass
            except (urllib.error.URLError, OSError):
                time.sleep(2)
                continue
            # Something answered; make sure it is our process and not one that
            # took the port between the check in start() and now.
            time.sleep(1)
            if self.proc is not None and self.proc.poll() is not None:
                raise ServerError(f"flm serve exited with code {self.proc.returncode}; "
                                  f"port {self.port} is answered by another process")
            return
        raise ServerError(f"flm serve did not answer within {self.ready_timeout:.0f}s; "
                          f"see {self.log_path.name}")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.CTRL_BREAK_EVENT if IS_WINDOWS else signal.SIGINT)
                self.proc.wait(timeout=30)
            except (subprocess.TimeoutExpired, OSError, ValueError):
                self.proc.kill()
                try:
                    self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    # Still there after SIGKILL: stuck in the kernel, which for
                    # flm means the NPU driver. Nothing here can free it.
                    self.unkillable = True
                    print(f"[server:{self.name}] pid {self.proc.pid} survived SIGKILL; "
                          f"it is likely stuck in the NPU driver", flush=True)
        if self._log is not None:
            self._log.close()
            self._log = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def died(self, grace: float = 5.0) -> bool:
        """True once the process has exited, waiting briefly for one on its way out.

        A request that kills the server often fails before the process is gone,
        and calling it alive there would blame the next model for the crash.
        """
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not self.alive():
                return True
            time.sleep(0.2)
        return not self.alive()

    def restart(self) -> bool:
        """Brings the server back after a crash. False once the budget is spent."""
        if self.restarts >= self.max_restarts:
            return False
        self.restarts += 1
        self.stop()
        try:
            self.start(append=True)
        except ServerError as exc:
            print(f"[server:{self.name}] restart failed: {exc}", flush=True)
            return False
        return True

    def log_tail(self, lines: int = 40) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])
