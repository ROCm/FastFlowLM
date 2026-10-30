"""The OpenAI API contract: runs the live tests in src/test against our server.

src/test/openai_api_test (output parsing, streaming, tools, response_format,
logprobs, sampling, continue_final_message, errors) and
src/test/openai_usage (usage and cache token counts) are standalone scripts
that developers run by hand. Running them here, as subprocesses, keeps one
copy of each check. Every PASS/FAIL/SKIP line they print becomes its own case,
so a failure shows by name in the report rather than as "run_all.py exited 1".
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..harness import FAIL, PASS, SKIP, Result, ServerGone, expect

NAME = "api"
DESCRIPTION = "OpenAI API and usage contract tests from src/test, per model"

# A run_all.py pass is about a minute on qwen3:0.6b; the budget is for large
# models on the full tier, whose answers come after up to 1024 reasoning tokens.
RUN_ALL_TIMEOUT = 30 * 60
USAGE_TIMEOUT = 10 * 60

# flm_api.run_cases(): "PASS  name (1.2s)", "FAIL  name (1.2s): why",
# "FAIL  name: cannot reach the server: why", "SKIP  name: why".
CASE_LINE = re.compile(r"^(PASS|FAIL|SKIP)  (\S+)(?: \(([\d.]+)s\))?(?:: ?(.*))?$")
MODULE_LINE = re.compile(r"^== (\S+)$")
SUMMARY_LINE = re.compile(r"^(\d+) passed, (\d+) failed, (\d+) skipped$")
STATUS = {"PASS": PASS, "FAIL": FAIL, "SKIP": SKIP}

# test_usage_contract.py: "Test 1: ...", "Test 1 PASSED", "FAIL: why".
USAGE_TESTS = {"1": "usage.non_stream", "2": "usage.stream"}
USAGE_START = re.compile(r"^Test (\d+): ")
USAGE_PASSED = re.compile(r"^Test (\d+) PASSED$")
USAGE_FAIL = re.compile(r"^FAIL: (.*)$")


def run(ctx) -> None:
    models = ctx.plan.suite(NAME)
    if not models:
        return
    tests = ctx.repo_root / "src" / "test"
    run_all = tests / "openai_api_test" / "run_all.py"
    usage = tests / "openai_usage" / "test_usage_contract.py"
    with ctx.case(NAME, "scripts-present") as case:
        missing = [str(p) for p in (run_all, usage) if not p.is_file()]
        # The workflow sparse-checks-out these directories; a path change there
        # would otherwise look like every model passing nothing.
        expect(not missing, f"missing: {', '.join(missing)}")
    if case.status != PASS:
        ctx.not_attempted(NAME, models, "API test scripts missing")
        return

    with ctx.server(NAME) as server:
        for i, model in enumerate(models):
            try:
                test_model(ctx, server, model, run_all, usage)
            except ServerGone:
                ctx.not_attempted(NAME, models[i + 1:], "server could not be restarted")
                raise


def test_model(ctx, server, model: str, run_all: Path, usage: Path) -> None:
    with ctx.case(NAME, "load", model, server) as case:
        # Loading (and downloading) gets its own case and timeout, so a model
        # that cannot load is one failure rather than every case timing out.
        server.client.chat(model, [{"role": "user", "content": "Hi"}], stream=False,
                           max_tokens=1, temperature=0, top_k=1)
    if case.status != PASS:
        return

    common = [model, "--endpoint", server.base_url, "--timeout", str(ctx.request_timeout)]
    always_thinks = ["--always-thinks"] if ctx.plan.model(model).always_thinks else []
    with ctx.case(NAME, "openai_api_test", model, server) as case:
        parser = RunAllParser(ctx, model, server)
        out = run_script(ctx, run_all, common + always_thinks, model, RUN_ALL_TIMEOUT,
                         on_line=parser.feed)
        case.data.update(out.data())
        parser.flush()
        # Before the output checks: after a crash, they would only blame run_all.py.
        expect(server.alive(), f"flm serve exited during run_all.py, in "
                               f"{parser.module} after {parser.last_case or 'no case'}")
        parser.check(out)
    with ctx.case(NAME, "openai_usage", model, server) as case:
        out = run_script(ctx, usage, common, model, USAGE_TIMEOUT)
        case.data.update(out.data())
        check_usage(ctx, model, out)
        expect(server.alive(), f"flm serve exited during test_usage_contract.py "
                               f"({out.status_text()})")


class ScriptOutput:
    def __init__(self, script: Path, log: Path):
        self.script = script
        self.log = log
        self.lines: list[str] = []
        self.returncode: int | None = None
        self.killed = ""  # why run_script() stopped it, if it did
        self.seconds = 0.0

    def tail(self, count: int = 40) -> str:
        return "\n".join(self.lines[-count:])

    def last_line(self) -> str:
        return next((l.strip() for l in reversed(self.lines) if l.strip()), "(no output)")

    def status_text(self) -> str:
        if self.killed:
            return f"killed: {self.killed}"
        return f"exit code {self.returncode}"

    def data(self) -> dict:
        data = {"log": self.log.name, "exit_code": self.returncode,
                "seconds": round(self.seconds, 1)}
        if self.killed or self.returncode != 0:
            data["output_tail"] = self.tail()
        return data


def run_script(ctx, script: Path, args: list[str], model: str, timeout: float,
               on_line=None) -> ScriptOutput:
    """Runs one test script to completion or timeout, keeping its output.

    on_line sees each line as it comes and may return a reason to stop the
    script early. The output also goes to a log file next to the server logs, since the
    report keeps only a tail of it.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    out = ScriptOutput(script, ctx.out_dir / f"api-{script.stem}-{safe}.log")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-u", str(script)] + args
    print(f"  [{NAME}] {model}: {script.name} (log: {out.log.name})", flush=True)
    started = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            cwd=str(script.parent), env=env)

    def kill(reason: str) -> None:
        if not out.killed:
            out.killed = reason
        proc.kill()

    # A timer rather than communicate(timeout=) so lines are seen as they come.
    timer = threading.Timer(timeout, kill, args=(f"timed out after {timeout:.0f}s",))
    timer.start()
    try:
        with open(out.log, "w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n\n")
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                out.lines.append(line)
                log.write(line + "\n")
                log.flush()
                stop = on_line(line) if on_line is not None else None
                if stop:
                    kill(stop)
        out.returncode = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    out.seconds = time.monotonic() - started
    return out


class RunAllParser:
    """Turns run_all.py output into one case per result line, as it arrives.

    A FAIL's message can span lines, so each result is recorded when the
    next marker line (or the end of output) closes it.
    """

    def __init__(self, ctx, model: str, server):
        self.ctx = ctx
        self.model = model
        self.server = server
        self.module = ""
        self.summary = None
        self.counts = {PASS: 0, FAIL: 0, SKIP: 0}
        self.last_case = ""
        self._pending: Result | None = None

    def feed(self, line: str) -> str | None:
        m = CASE_LINE.match(line)
        if m:
            self.flush()
            mark, name, seconds, detail = m.groups()
            self._pending = Result(NAME, name, STATUS[mark], self.model,
                                   (detail or "").strip(), float(seconds or 0))
            if "cannot reach the server" in self._pending.detail and self.server.died(grace=3):
                # Every later case would fail the same way; the crash is the
                # one failure worth reporting, and the server needs a restart.
                return "flm serve died"
            return None
        m = MODULE_LINE.match(line) or SUMMARY_LINE.match(line)
        if m:
            self.flush()
            if len(m.groups()) == 1:
                self.module = m.group(1)
            else:
                self.summary = tuple(int(g) for g in m.groups())
            return
        if line.startswith("Traceback (most recent call last)"):
            # run_cases() lets anything but an assertion or a connection error
            # escape, which ends the run; check() reports that, not the case
            # that happened to print last.
            self.flush()
        elif self._pending is not None and self._pending.status == FAIL and line.strip():
            self._pending.detail += "\n" + line

    def flush(self) -> None:
        if self._pending is not None:
            self.ctx.record(self._pending)
            self.counts[self._pending.status] += 1
            self.last_case = self._pending.case
            self._pending = None

    def check(self, out: ScriptOutput) -> None:
        """Checks what the per-case lines cannot show: a timeout, a crash
        between cases, or output this parser did not understand."""
        total = sum(self.counts.values())
        expect(not out.killed, f"run_all.py {out.status_text()}, in {self.module or 'startup'} "
                               f"after {self.last_case or 'no case'}")
        expect(total, f"run_all.py printed no result lines ({out.status_text()}): "
                      f"{out.last_line()}")
        expect(self.summary is not None,
               f"run_all.py stopped in {self.module} after {self.last_case} "
               f"({out.status_text()}): {out.last_line()}")
        parsed = (self.counts[PASS], self.counts[FAIL], self.counts[SKIP])
        expect(parsed == self.summary, f"parsed {parsed} pass/fail/skip but run_all.py "
                                       f"reported {self.summary}")
        expect((out.returncode != 0) == (self.counts[FAIL] > 0),
               f"run_all.py exit code {out.returncode} with {self.counts[FAIL]} failed cases")


def check_usage(ctx, model: str, out: ScriptOutput) -> None:
    """Records the usage script's two tests as cases.

    It stops at the first failure, printing "FAIL: why" for a contract
    violation or a traceback for anything else, so a test that started and
    did not pass failed with that message, and a later one never ran.
    """
    started, passed = [], set()
    reason = ""
    for line in out.lines:
        m = USAGE_START.match(line)
        if m:
            started.append(m.group(1))
        m = USAGE_PASSED.match(line)
        if m:
            passed.add(m.group(1))
        m = USAGE_FAIL.match(line)
        if m:
            reason = m.group(1)
    expect(started, f"test_usage_contract.py printed no tests ({out.status_text()}): "
                    f"{out.last_line()}")
    blocked = ""
    for number, name in USAGE_TESTS.items():
        if number in passed:
            ctx.record(Result(NAME, name, PASS, model))
        elif number in started:
            why = reason or out.last_line()
            if out.killed:
                why = f"{out.status_text()}; last line: {out.last_line()}"
            ctx.record(Result(NAME, name, FAIL, model, why,
                              data={"output_tail": out.tail(15)}))
            blocked = blocked or name
        elif blocked:
            ctx.record(Result(NAME, name, SKIP, model, f"not run: {blocked} failed first"))
        else:
            ctx.record(Result(NAME, name, FAIL, model,
                              f"never started ({out.status_text()}): {out.last_line()}"))
    expect(not out.killed, f"test_usage_contract.py {out.status_text()}")
    all_passed = all(n in passed for n in USAGE_TESTS)
    expect((out.returncode == 0) == all_passed,
           f"test_usage_contract.py exit code {out.returncode} but "
           f"{len(passed)} of {len(USAGE_TESTS)} tests passed")
