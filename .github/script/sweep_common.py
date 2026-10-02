"""Shared plumbing for the FLM model sweep scripts.

The sweep is split into four independent entry points so that a failure in one
modality does not block or invalidate the others, and so a single failed run
can be restarted on its own:

    sweep_llm.py         text chat completions
    sweep_vision.py      chat completions with images
    sweep_embedding.py   the /v1/embeddings endpoint
    sweep_audio.py       the /v1/audio/transcriptions endpoint

Each entry point owns its own `flm serve` process, discovers the models it
cares about, writes one CSV, and exits non-zero if any model failed. Nothing is
shared between them at runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# A sweep runs for tens of minutes and its progress output *is* the live status
# on a CI runner. Python block-buffers stdout whenever it is not a terminal, so
# under Actions every print would otherwise sit in an 8 KB buffer and arrive in
# one burst when the job ends -- no way to tell which model is being tested, or
# whether anything is happening at all. Line buffering here covers every print
# in every task script, so individual calls do not have to remember flush=True.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - surfaced immediately at runtime
    print("The 'openai' package is required. Install it with: pip install openai")
    raise

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"

# Matches utils::get_server_port() in src/common/utils.cpp.
DEFAULT_PORT = 52625

IS_WINDOWS = os.name == "nt"


def usage_to_dict(usage) -> dict:
    """Flattens an SDK usage object into a plain dict.

    FLM returns non-standard fields alongside the OpenAI ones -- the prefill
    and decode durations and rates, KV occupancy. The SDK's models allow extra
    fields, so model_dump() keeps them, but going through a dict means nothing
    here depends on that or on any particular SDK version.
    """
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    for method in ("model_dump", "to_dict", "dict"):
        dump = getattr(usage, method, None)
        if callable(dump):
            try:
                result = dump()
            except Exception:
                continue
            if isinstance(result, dict):
                return result
    return dict(getattr(usage, "__dict__", {}) or {})


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------

def build_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser carrying the options every sweep script understands."""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--flm-bin",
        default=os.environ.get("FLM_BIN", "flm"),
        help="Path to the flm binary (default: $FLM_BIN, else 'flm' on PATH).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Value to export as FLM_MODEL_PATH before starting the server.",
    )
    parser.add_argument(
        "--platform",
        default="windows" if IS_WINDOWS else "linux",
        choices=["linux", "windows"],
        help="Label recorded in the output filename (default: autodetected).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server bind address.")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FLM_SERVE_PORT", DEFAULT_PORT)),
        help=f"Server port (default: $FLM_SERVE_PORT, else {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--output-dir",
        default="sweep-results",
        help="Directory for the CSV, summary JSON and server log.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="TAG",
        help="Test only these model tags instead of the discovered set.",
    )
    parser.add_argument(
        "--skip-models",
        nargs="+",
        default=[],
        metavar="TAG",
        help="Model tags to exclude from the discovered set.",
    )
    parser.add_argument(
        "--installed-only",
        action="store_true",
        help="Only test already-downloaded models, so the sweep pulls nothing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many models. 0 means no cap.",
    )

    parser.add_argument(
        "--gen-lim",
        type=int,
        default=512,
        help="max_completion_tokens per request. 0 or less leaves it unset.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help=(
            "Per-request timeout in seconds (default: 300). Applied both as the "
            "HTTP timeout and as a wall-clock deadline on streamed responses."
        ),
    )
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=300.0,
        help="How long to wait for the server to answer /v1/version.",
    )
    parser.add_argument(
        "--max-server-restarts",
        type=int,
        default=3,
        help=(
            "How many times to restart the server after it dies mid-sweep "
            "(default: 3). 0 disables restarting."
        ),
    )
    parser.add_argument(
        "--wedged-after",
        type=int,
        default=3,
        help=(
            "Treat the server as wedged once this many consecutive models fail "
            "while it is still running, and restart it (default: 3). "
            "0 disables the check."
        ),
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Attach to an already-running server instead of starting one.",
    )

    return parser


# --------------------------------------------------------------------------
# Base task
# --------------------------------------------------------------------------

class SweepTask:
    """Owns a server process, a model selection and one CSV of results."""

    # Overridden by subclasses.
    name = "sweep"
    needs_asr = False
    needs_embed = False
    csv_header: list[str] = []

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """Subclasses override this to register task-specific flags."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base_url = f"http://{args.host}:{args.port}/v1"
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if args.model_path:
            os.environ["FLM_MODEL_PATH"] = str(args.model_path)

        self.server: subprocess.Popen | None = None
        self._server_log_handle = None
        self.version = "unknown"
        self.rows_written = 0
        self.failures: list[str] = []
        # Models the sweep gave up on after the server stopped coming back.
        # Distinct from failures: these were never actually exercised.
        self.not_attempted: list[str] = []
        self.server_restarts = 0
        # Kept on the instance so a summary can still be written if run() dies.
        self.models: list[str] = []
        # Audit trail of every server-lifecycle decision, with its reason.
        self.events: list[dict] = []
        self._started_at = time.monotonic()
        self._consecutive_failures = 0
        # Usage block from the most recent chat(), consumed by format_usage().
        self.last_usage: dict = {}
        self._catalog_cache: list[dict] | None = None
        self._labels_by_model: dict[str, list[str]] | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "SweepTask":
        self.start_server()
        self.version = self.fetch_version()
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="flm",
            timeout=self.args.request_timeout,
            max_retries=0,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop_server()
        return False

    @property
    def server_log_path(self) -> Path:
        return self.output_dir / f"{self.name}_server.log"

    def log_event(self, event: str, *, reason: str = "", detail: str = "", **fields) -> None:
        """Records one lifecycle decision, on stdout and in the summary JSON.

        Console output dies with the runner's log retention and cannot be
        correlated with the CSV, so anything the sweep decides on its own --
        above all why it restarted the server -- has to explain itself in the
        artifact as well.
        """
        entry = {
            "event": event,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_s": round(time.monotonic() - self._started_at, 1),
        }
        if reason:
            entry["reason"] = reason
        if detail:
            entry["detail"] = detail
        entry.update(fields)
        self.events.append(entry)

        message = f"[{self.name}] {event}"
        if reason:
            message += f" ({reason})"
        if detail:
            message += f": {detail}"
        print(message, flush=True)

    def start_server(self, append_log: bool = False) -> None:
        if self.args.no_server:
            print(f"[{self.name}] Attaching to existing server at {self.base_url}")
            self.wait_until_ready()
            return

        cmd = [
            self.args.flm_bin,
            "serve",
            "--host", self.args.host,
            "--port", str(self.args.port),
            "-a", "1" if self.needs_asr else "0",
            "-e", "1" if self.needs_embed else "0",
        ]
        self.log_event(
            "server_start",
            detail=" ".join(cmd),
            attempt=self.server_restarts,
            log=str(self.server_log_path),
        )

        # Appending on a restart keeps the log of the crash being recovered from.
        self._server_log_handle = open(self.server_log_path, "ab" if append_log else "wb")
        if append_log:
            banner = f"\n=== server restart {self.server_restarts} ===\n"
            self._server_log_handle.write(banner.encode("utf-8"))
            self._server_log_handle.flush()

        popen_kwargs: dict = {
            "stdout": self._server_log_handle,
            "stderr": subprocess.STDOUT,
        }
        if IS_WINDOWS:
            # Needed so we can deliver Ctrl-Break for a graceful shutdown
            # without also killing this script.
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        self.server = subprocess.Popen(cmd, **popen_kwargs)
        self.wait_until_ready()

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.args.server_timeout
        while time.monotonic() < deadline:
            if self.server is not None and self.server.poll() is not None:
                raise RuntimeError(
                    f"flm serve exited with code {self.server.returncode} "
                    f"before becoming ready. See {self.server_log_path}"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/version", timeout=5):
                    print(f"[{self.name}] Server is ready at {self.base_url}")
                    return
            except (urllib.error.URLError, OSError):
                time.sleep(2)
        raise TimeoutError(
            f"Server did not respond within {self.args.server_timeout:.0f}s. "
            f"See {self.server_log_path}"
        )

    def stop_server(self) -> None:
        if self.server is not None and self.server.poll() is None:
            try:
                if IS_WINDOWS:
                    self.server.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.server.send_signal(signal.SIGINT)
                self.server.wait(timeout=30)
                self.log_event("server_stop", reason="graceful",
                               detail=f"exit code {self.server.returncode}")
            except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
                # Worth recording: a server that will not answer Ctrl-C is a
                # different kind of broken from one that exits on request.
                self.log_event("server_stop", reason="killed",
                               detail=f"graceful stop failed: {exc}")
                self.server.kill()
                try:
                    self.server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.log_event("server_stop", reason="unresponsive",
                                   detail="still running 15s after SIGKILL")
        if self._server_log_handle is not None:
            self._server_log_handle.close()
            self._server_log_handle = None

    def server_died(self) -> bool:
        """True when this script started the server and it has since exited.

        Only catches the server going away. A server that is still running but
        wedged looks alive here, and is caught by the consecutive-failure check
        in iter_models instead.
        """
        return self.server is not None and self.server.poll() is not None

    def await_server_exit(self, timeout: float = 5.0) -> bool:
        """Waits briefly for a server that may still be on its way out.

        A process killed by the request that just failed is often mid-exit when
        poll() first runs. Returning "alive" there would blame the *next* model
        for the crash, so give it a moment to actually go away before deciding.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.server_died():
                return True
            time.sleep(0.2)
        return self.server_died()

    def restart_server(self, reason: str, detail: str = "") -> bool:
        """Brings the server back mid-sweep. False once the budget is spent."""
        if self.args.no_server:
            self.log_event("restart_declined", reason="--no-server",
                           detail="the server is managed outside this script")
            return False
        if self.server_restarts >= self.args.max_server_restarts:
            self.log_event(
                "restart_declined",
                reason="budget exhausted",
                detail=f"--max-server-restarts is {self.args.max_server_restarts}",
            )
            return False

        self.server_restarts += 1
        # A wedged server is still running, so take it down before starting
        # another one on the same port. Already-dead servers no-op here.
        self.stop_server()
        try:
            self.start_server(append_log=True)
        except Exception as exc:
            # A restart can fail for reasons the sweep cannot fix, such as the
            # old process still holding the port. Give up cleanly so the rows
            # collected so far still get written.
            self.log_event("server_restart", reason=reason, detail=detail,
                           attempt=self.server_restarts, outcome=f"failed: {exc}")
            return False

        self.log_event("server_restart", reason=reason, detail=detail,
                       attempt=self.server_restarts, outcome="ready")
        return True

    def iter_models(self, models: list[str]):
        """Yields models, keeping the server alive for the rest of the sweep.

        Two ways the server ruins a sweep, both handled the same way. It can
        exit, which poll() sees directly. Or it can stay up and stop answering,
        which looks exactly like every model failing in turn -- so a run of
        consecutive failed models is treated as the server being wedged and
        earns a forced restart.

        Either way the sweep resumes at the *next* model: the culprit is not
        retried, since it would just break the server again. Once restarts run
        out the remaining models are recorded as not attempted rather than
        reported as failures they never got to be. Every decision here lands in
        self.events with its reason.
        """
        total = len(models)
        for index, model in enumerate(models):
            # Every task funnels through here, so one progress line covers all
            # four. Position and elapsed time are what a watcher actually wants
            # from a sweep that runs for tens of minutes: which model is up now,
            # and how much of the list is left.
            elapsed = int(time.monotonic() - self._started_at)
            print(
                f"[{self.name}] ({index + 1}/{total}) {model} "
                f"-- {elapsed // 60}m{elapsed % 60:02d}s elapsed",
                flush=True,
            )
            failures_before = len(self.failures)
            yield model

            model_failed = len(self.failures) > failures_before
            if model_failed:
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0

            # Only pay the grace period when something actually went wrong.
            crashed = self.await_server_exit() if model_failed else self.server_died()
            wedged = (
                not crashed
                and self.args.wedged_after > 0
                and self._consecutive_failures >= self.args.wedged_after
            )
            if not (crashed or wedged):
                continue

            if crashed:
                returncode = self.server.returncode if self.server else "unknown"
                reason = "crash"
                detail = f"exited with code {returncode} while testing {model}"
                self.failures.append(f"{model}: flm server {detail}")
                self.log_event("server_crashed", reason="process exited",
                               detail=detail, model=model)
            else:
                culprits = models[index - self._consecutive_failures + 1: index + 1]
                reason = "wedged"
                detail = (
                    f"{self._consecutive_failures} consecutive models failed "
                    f"while the server was still running"
                )
                self.log_event("server_wedged", reason="consecutive failures",
                               detail=detail, models=culprits)

            self._consecutive_failures = 0
            remaining = models[index + 1:]
            if not remaining:
                return
            if not self.restart_server(reason=reason, detail=detail):
                self.log_event(
                    "models_not_attempted",
                    reason=f"server unavailable after {reason}",
                    detail=f"{len(remaining)} model(s) skipped",
                    models=remaining,
                )
                self.not_attempted.extend(remaining)
                return

    # -- discovery ---------------------------------------------------------

    def fetch_version(self) -> str:
        try:
            with urllib.request.urlopen(f"{self.base_url}/version", timeout=10) as resp:
                version = json.loads(resp.read().decode("utf-8")).get("version")
                if version:
                    print(f"[{self.name}] flm version {version}")
                    return str(version)
        except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as exc:
            print(f"[{self.name}] Could not read version: {exc}")
        return "unknown"

    def catalog(self) -> list[dict]:
        """Model catalog, loaded once per run."""
        if self._catalog_cache is None:
            self._catalog_cache = self._load_catalog()
        return self._catalog_cache

    def _load_catalog(self) -> list[dict]:
        """Full model catalog with the vlm/asr flags the sweep filters on.

        `flm list --json` is the only source that carries those flags;
        /v1/models omits them and hides the embedding and Whisper models
        entirely. Falls back to /v1/models when the binary is unavailable.
        """
        list_filter = "installed" if self.args.installed_only else "all"
        cmd = [self.args.flm_bin, "list", "--json", "--filter", list_filter]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, check=True
            )
            # Tolerate banner lines printed before the JSON document.
            start = proc.stdout.find("{")
            if start < 0:
                raise ValueError("no JSON object in 'flm list' output")
            entries = json.loads(proc.stdout[start:]).get("models", [])
            return [
                {
                    "id": entry.get("model") or entry.get("name"),
                    "vlm": bool(entry.get("vlm", False)),
                    "asr": bool(entry.get("asr", False)),
                    "family": (entry.get("details") or {}).get("family", ""),
                    # model_list.json's label array, passed through verbatim by
                    # `flm list --json`. Carries "single-turn" among others.
                    "labels": list(entry.get("label") or []),
                    "installed": bool(entry.get("installed", True)),
                }
                for entry in entries
                if entry.get("model") or entry.get("name")
            ]
        except (subprocess.SubprocessError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[{self.name}] 'flm list' unavailable ({exc}); falling back to /v1/models")

        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8")).get("data", [])
            return [
                {"id": m["id"], "vlm": False, "asr": False, "family": "",
                 "labels": [], "installed": True}
                for m in data
                if m.get("id")
            ]
        except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as exc:
            print(f"[{self.name}] Could not list models: {exc}")
            return []

    def select_models(self, catalog: list[dict]) -> list[str]:
        """Subclasses narrow the catalog to the models they exercise."""
        raise NotImplementedError

    def model_labels(self, model: str) -> list[str]:
        """Catalog labels for one model, empty when the catalog had none."""
        if self._labels_by_model is None:
            self._labels_by_model = {
                entry["id"]: entry.get("labels") or []
                for entry in self.catalog()
                if entry.get("id")
            }
        return self._labels_by_model.get(model, [])

    def is_single_turn(self, model: str) -> bool:
        """True for models model_list.json labels "single-turn".

        These are built for one exchange -- dedicated translation models, the
        flash variants -- and cap context at 1k. Sending a follow-up turn gets
        rejected by design, so a sweep that counted that as a failure would be
        reporting the model working correctly as a defect.

        Unknown models are treated as multi-turn: /v1/models carries no labels,
        so guessing here would silently drop the follow-up for every model
        whenever the flm binary was unavailable.
        """
        return "single-turn" in self.model_labels(model)

    def resolve_models(self) -> list[str]:
        if self.args.models:
            models = list(self.args.models)
            print(f"[{self.name}] Using {len(models)} model(s) from --models")
        else:
            models = self.select_models(self.catalog())

        skip = set(self.args.skip_models)
        if skip:
            models = [m for m in models if m not in skip]
        if self.args.limit > 0:
            models = models[: self.args.limit]

        # Warmed here so a catalog that cannot be read complains during setup
        # rather than in the middle of the first model's output.
        single_turn = [m for m in models if self.is_single_turn(m)]

        print(f"[{self.name}] {len(models)} model(s) selected:")
        for index, model in enumerate(models, 1):
            suffix = " [single-turn]" if model in single_turn else ""
            print(f"  {index:>3}. {model}{suffix}")
        if single_turn:
            print(
                f"[{self.name}] {len(single_turn)} model(s) are single-turn; "
                "their follow-up round is skipped, not failed."
            )
        self.models = models
        return models

    # -- output ------------------------------------------------------------

    @property
    def csv_path(self) -> Path:
        return self.output_dir / (
            f"{self.name}_{self.args.platform}_v{self.version}.csv"
        )

    @property
    def summary_path(self) -> Path:
        return self.output_dir / (
            f"{self.name}_{self.args.platform}_v{self.version}_summary.json"
        )

    def open_csv(self):
        handle = open(self.csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(handle)
        writer.writerow(self.csv_header)
        return handle, writer

    def record(self, writer, row: list, *, model: str, error: str | None = None) -> None:
        writer.writerow(row)
        self.rows_written += 1
        if error is not None:
            self.failures.append(f"{model}: {error}")

    def write_summary(self, models: list[str]) -> None:
        summary = {
            "task": self.name,
            "platform": self.args.platform,
            "flm_version": self.version,
            "models_tested": len(models) - len(self.not_attempted),
            "models_selected": len(models),
            "rows_written": self.rows_written,
            "failures": self.failures,
            "not_attempted": self.not_attempted,
            # Models whose follow-up round was deliberately not sent. Recorded
            # so a reader can tell a deliberately shortened test from one that
            # silently lost a round.
            "single_turn": [m for m in models if self.is_single_turn(m)],
            "server_restarts": self.server_restarts,
            "events": self.events,
            "csv": self.csv_path.name,
            "server_log": self.server_log_path.name,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # -- helpers -----------------------------------------------------------

    def completion_kwargs(self) -> dict:
        """Token cap, omitted entirely when the caller disabled it."""
        if self.args.gen_lim and self.args.gen_lim > 0:
            return {"max_completion_tokens": self.args.gen_lim}
        return {}

    def collect_stream(self, response) -> tuple[str, str]:
        """Accumulates streamed chunks into (reasoning_content, output_content).

        The HTTP timeout only bounds the wait for the next chunk, so a model
        that keeps emitting tokens can stream well past it. This enforces
        --request-timeout as a deadline on the whole response.
        """
        reasoning_content, output_content = "", ""
        usage = None
        deadline = time.monotonic() + self.args.request_timeout
        for chunk in response:
            if time.monotonic() > deadline:
                try:
                    response.close()
                except Exception:
                    pass
                raise TimeoutError(
                    f"stream exceeded the {self.args.request_timeout:.0f}s "
                    f"request timeout after {len(output_content)} characters"
                )
            # FLM attaches usage to the final chunk, which also carries a
            # choices entry with a null delta. Read it before the choices
            # guard below, so a server that instead sends usage on its own
            # terminal chunk (the stock OpenAI shape) still gets picked up.
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "reasoning_content", None):
                reasoning_content += delta.reasoning_content
            if delta.content:
                output_content += delta.content
        return reasoning_content, output_content, usage

    def chat(self, model: str, messages: list, stream: bool = True) -> tuple[str, str]:
        """One chat completion, returned as (reasoning_content, output_content).

        The server's usage block lands on self.last_usage rather than in the
        return value: callers unpack a 2-tuple, and timing data is something
        only the reporting cares about.
        """
        self.last_usage = {}
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            **self.completion_kwargs(),
        )
        if stream:
            reasoning, output, usage = self.collect_stream(response)
        else:
            message = response.choices[0].message
            reasoning = getattr(message, "reasoning_content", None) or ""
            output = message.content or ""
            usage = getattr(response, "usage", None)
        self.last_usage = usage_to_dict(usage)
        return reasoning, output

    def format_usage(self) -> str:
        """The server's own prefill/decode accounting, as one line.

        FLM reports far more than the OpenAI-standard token counts: it breaks
        the request into prefill and decode, each with a duration and a rate,
        and says how much of the prompt the KV cache served. That is the part
        worth seeing per request, since wall-clock latency alone cannot
        distinguish a slow prefill from a slow decode.

        Everything is optional. A server that reports only the standard fields
        still gets a token count, and one that reports no usage at all gets an
        empty string, leaving the caller's plain timing line as it was.
        """
        usage = self.last_usage
        if not usage:
            return ""

        def num(key):
            value = usage.get(key)
            return value if isinstance(value, (int, float)) else None

        segments = []

        prompt = num("prompt_tokens")
        if prompt is not None:
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            part = f"prefill {prompt} tok"
            if isinstance(cached, (int, float)) and cached:
                part += f" ({cached} cached)"
            ttft = num("prefill_duration_ttft")
            if ttft is not None:
                part += f" in {ttft:.2f}s"
            rate = num("prefill_speed_tps")
            if rate is not None:
                part += f" @ {rate:.1f} tps"
            segments.append(part)

        completion = num("completion_tokens")
        if completion is not None:
            part = f"decode {completion} tok"
            duration = num("decoding_duration")
            if duration is not None:
                part += f" in {duration:.2f}s"
            rate = num("decoding_speed_tps")
            if rate is not None:
                part += f" @ {rate:.1f} tps"
            segments.append(part)

        # Only meaningful on the first request against a model, where it is
        # the difference between "the model is slow" and "it was still loading".
        load = num("load_duration")
        if load:
            segments.append(f"load {load:.2f}s")

        occupancy = num("kv_token_occupancy_rate_percentage")
        if occupancy is not None:
            active = num("active_kv_tokens")
            capacity = num("max_kv_token_capacity")
            if active is not None and capacity:
                segments.append(f"kv {active}/{capacity} ({occupancy:.0f}%)")

        return " | ".join(segments)

    @staticmethod
    def asset(filename: str) -> Path:
        path = ASSETS_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Test asset missing: {path}")
        return path

    # -- entry point -------------------------------------------------------

    def run(self) -> None:
        raise NotImplementedError


def run_task(task_cls, argv: list[str] | None = None) -> int:
    """Parses arguments, runs one sweep task, and maps failures to an exit code."""
    parser = build_parser(task_cls.__doc__ or task_cls.name)
    task_cls.add_arguments(parser)
    args = parser.parse_args(argv)

    task = task_cls(args)
    try:
        try:
            with task:
                task.run()
        except KeyboardInterrupt:
            task.log_event("aborted", reason="interrupted")
            return 130
        except Exception as exc:
            task.log_event("aborted", reason="fatal error", detail=str(exc))
            return 1
    finally:
        # Rewritten here rather than only at the end of run(), so the summary
        # also captures how the server was shut down and any fatal error that
        # cut the sweep short. write_summary overwrites, so the extra call is
        # just the authoritative one.
        try:
            task.write_summary(task.models)
        except Exception as summary_exc:
            print(f"[{task.name}] Could not write summary: {summary_exc}")

    if task.not_attempted:
        print(
            f"\n[{task.name}] {len(task.not_attempted)} model(s) not attempted "
            f"after the server stopped coming back:"
        )
        for model in task.not_attempted:
            print(f"  - {model}")

    if task.failures:
        print(f"\n[{task.name}] {len(task.failures)} failure(s):")
        for failure in task.failures:
            print(f"  - {failure}")
        print(f"[{task.name}] Results: {task.csv_path}")
        return 1

    print(f"\n[{task.name}] All checks passed. Results: {task.csv_path}")
    return 0
