"""What every suite shares: the run context, cases, and the report.

A suite is a module in suites/ with NAME, a one-line DESCRIPTION, and
run(ctx). It records each check as a case:

    with ctx.case("chat", "stream-two-turns", model) as case:
        reply = server.client.chat(model, messages)
        expect(reply.content, "empty content")
        case.data["tokens"] = reply.usage.get("completion_tokens")

An AssertionError (or expect()) in the block records FAIL, skip() records
SKIP, warn() records WARN, and any other exception records FAIL with its type.
The block never raises, so one broken model cannot stop the suite. When the
server dies under a case, it is restarted for the next one; once restarts run
out, ServerGone ends the suite and its remaining models are reported as not
attempted.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .plan import Plan
from .server import FlmServer, ServerError

PASS, FAIL, SKIP, WARN = "pass", "fail", "skip", "warn"

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS = SCRIPT_DIR / "assets"
REPO_ROOT = SCRIPT_DIR.parents[2]


class Skip(Exception):
    """Raised inside a case: the check does not apply here."""


class Warn(Exception):
    """Raised inside a case: worth a look, but not a failure."""


class ServerGone(Exception):
    """The suite's server died and could not be restarted."""


def expect(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def skip(message: str):
    raise Skip(message)


def warn(message: str):
    raise Warn(message)


@dataclass
class Result:
    suite: str
    case: str
    status: str
    model: str = ""
    detail: str = ""
    seconds: float = 0.0
    data: dict = field(default_factory=dict)


class Case:
    def __init__(self, ctx: "Context", suite: str, name: str, model: str,
                 server: FlmServer | None):
        self.ctx = ctx
        self.suite = suite
        self.name = name
        self.model = model
        self.server = server
        self.data: dict = {}
        self.status = PASS
        self._started = 0.0

    def __enter__(self) -> "Case":
        self._started = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        seconds = time.monotonic() - self._started
        detail = ""
        if exc_type is None:
            self.status = PASS
        elif issubclass(exc_type, Skip):
            self.status, detail = SKIP, str(exc)
        elif issubclass(exc_type, Warn):
            self.status, detail = WARN, str(exc)
        elif issubclass(exc_type, ServerGone):
            return False  # let the suite end
        elif issubclass(exc_type, AssertionError):
            self.status, detail = FAIL, str(exc) or "assertion failed"
        elif issubclass(exc_type, KeyboardInterrupt):
            return False
        else:
            self.status = FAIL
            detail = f"{exc_type.__name__}: {exc}"
            if self.ctx.verbose:
                traceback.print_exception(exc_type, exc, tb)
        self.ctx.record(Result(self.suite, self.name, self.status, self.model, detail,
                               round(seconds, 2), self.data))
        if self.status == FAIL and self.server is not None and self.server.died():
            self.ctx.record(Result(self.suite, "server-crash", FAIL, self.model,
                                   f"flm serve exited (code {self.server.proc.returncode}) "
                                   f"during {self.name}",
                                   data={"log_tail": self.server.log_tail()}))
            if not self.server.restart():
                raise ServerGone(f"server for {self.suite} could not be restarted")
        return True

    @property
    def ok(self) -> bool:
        return self.status in (PASS, WARN)


class Context:
    def __init__(self, *, flm_bin: str, label: str, tier: str, plan: Plan,
                 out_dir: Path, request_timeout: float, port: int | None = None,
                 verbose: bool = False):
        self.flm_bin = flm_bin
        self.label = label
        self.tier = tier
        self.plan = plan
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.request_timeout = request_timeout
        self.port = port
        self.verbose = verbose
        self.results: list[Result] = []
        self.servers: list[FlmServer] = []
        self.version = "unknown"
        self.assets = ASSETS
        self.repo_root = REPO_ROOT

    # -- servers and cases -----------------------------------------------------

    def server(self, name: str, args: list[str] | None = None, **kwargs) -> FlmServer:
        """A server for one suite, not yet started; use it as a context manager."""
        kwargs.setdefault("port", self.port)
        server = FlmServer(self.flm_bin, name=name, log_dir=self.out_dir, args=args,
                           request_timeout=self.request_timeout, **kwargs)
        self.servers.append(server)
        return server

    def case(self, suite: str, name: str, model: str = "",
             server: FlmServer | None = None) -> Case:
        return Case(self, suite, name, model, server)

    def record(self, result: Result) -> None:
        self.results.append(result)
        mark = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP", WARN: "WARN"}[result.status]
        where = f" [{result.model}]" if result.model else ""
        line = f"  {mark} {result.suite}/{result.case}{where} ({result.seconds:.1f}s)"
        if result.detail:
            line += f": {result.detail}"
        print(line, flush=True)

    def not_attempted(self, suite: str, models: list[str], reason: str) -> None:
        for model in models:
            self.record(Result(suite, "not-attempted", FAIL, model, reason))

    def run_suite(self, module) -> None:
        """Runs one suite module, turning a fatal error into a recorded failure."""
        print(f"\n== {module.NAME}: {module.DESCRIPTION}", flush=True)
        started = time.monotonic()
        try:
            module.run(self)
        except ServerGone as exc:
            self.record(Result(module.NAME, "server-gone", FAIL, detail=str(exc)))
        except ServerError as exc:
            self.record(Result(module.NAME, "server-start", FAIL, detail=str(exc)))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            traceback.print_exc()
            self.record(Result(module.NAME, "suite-error", FAIL,
                               detail=f"{type(exc).__name__}: {exc}"))
        for server in self.servers:
            if server.unkillable:
                server.unkillable = False  # report it once
                self.record(Result(module.NAME, "server-stop", FAIL,
                                   detail=f"flm serve (pid {server.proc.pid}) survived SIGKILL, "
                                          f"likely stuck in the NPU driver; later suites may "
                                          f"find the NPU busy",
                                   data={"log_tail": server.log_tail()}))
        print(f"== {module.NAME} done in {time.monotonic() - started:.0f}s", flush=True)

    def asset(self, relative: str) -> Path:
        path = self.assets / relative
        if not path.is_file():
            raise FileNotFoundError(f"test asset missing: {path}")
        return path

    # -- report ----------------------------------------------------------------

    @property
    def failed(self) -> bool:
        return any(r.status == FAIL for r in self.results)

    def write_report(self, suites: list[str], seconds: float) -> Path:
        counts = {s: sum(1 for r in self.results if r.status == s)
                  for s in (PASS, FAIL, WARN, SKIP)}
        report = {
            "label": self.label,
            "tier": self.tier,
            "flm_version": self.version,
            "suites": suites,
            "seconds": round(seconds, 1),
            "counts": counts,
            "results": [asdict(r) for r in self.results],
        }
        path = self.out_dir / "results.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        (self.out_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
        return path


def render_markdown(report: dict) -> str:
    """One package's results as Markdown: counts, then every non-pass case."""
    c = report["counts"]
    icon = "❌" if c["fail"] else ("⚠️" if c["warn"] else "✅")
    lines = [
        f"### {icon} {report['label']} ({report['tier']} tier, flm {report['flm_version']})",
        "",
        f"{c['pass']} passed, {c['fail']} failed, {c['warn']} warnings, {c['skip']} skipped "
        f"in {report['seconds'] / 60:.1f} min. Suites: {', '.join(report['suites'])}.",
        "",
    ]
    notable = [r for r in report["results"] if r["status"] in (FAIL, WARN)]
    if notable:
        lines += ["| | Suite | Case | Model | Detail |", "|---|---|---|---|---|"]
        for r in notable:
            mark = "❌" if r["status"] == FAIL else "⚠️"
            detail = r["detail"].replace("|", "\\|").replace("\n", " ")[:300]
            lines.append(f"| {mark} | {r['suite']} | {r['case']} | {r['model']} | {detail} |")
        lines.append("")
    per_suite = {}
    for r in report["results"]:
        s = per_suite.setdefault(r["suite"], {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0, "s": 0.0})
        s[r["status"]] += 1
        s["s"] += r["seconds"]
    if per_suite:
        lines += ["| Suite | Pass | Fail | Warn | Skip | Time |", "|---|---|---|---|---|---|"]
        for name, s in per_suite.items():
            lines.append(f"| {name} | {s[PASS]} | {s[FAIL]} | {s[WARN]} | {s[SKIP]} "
                         f"| {s['s'] / 60:.1f} min |")
        lines.append("")
    return "\n".join(lines)


def configure_stdout() -> None:
    # The live log is the only progress a CI watcher sees; block buffering would
    # hold it until the job ends. Model output can be any script, which the
    # Windows console code page cannot always encode, and a print must never
    # be what ends a suite.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True, encoding="utf-8", errors="backslashreplace")
