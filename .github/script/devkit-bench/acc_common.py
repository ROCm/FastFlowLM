# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Shared plumbing for the two dev-kit accuracy benches.

gsm8k_run.py (GSM8K) and run_vqa.py (DocVQA) both stream questions through
an already-running `flm serve` over /v1/chat/completions. They used to handle
a bad question differently -- GSM8K counted a request failure, DocVQA silently
scored it as an empty answer -- so a broken run could look like a real score.
Everything here is the one behaviour both runners and both scorers share:

Every question ends in exactly one OUTCOME:
  answered        a usable reply came back and an answer was parsed from it
  reasoning_only  the model was still reasoning when it stopped: the final
                  answer is empty but it spent its tokens thinking (a reasoning
                  field, a <think> block -- closed or cut off -- and nothing
                  after it, or an empty reply that hit max_tokens). It tried, so
                  it is a WRONG answer, not an unanswered one.
  failed          no usable reply; `error` is set to one of ERROR_KINDS
  unparsed        a final answer came back, but no answer could be parsed from
                  it (GSM8K: no number in it)
  unanswered = failed + unparsed, in both benches. reasoning_only, failed and
  unparsed all score as wrong; only failed + unparsed count toward the
  --max-unanswered gate.

Per-question record fields both runners write (on top of each bench's own):
  pred_raw           message.content as returned (None if no usable response)
  reasoning_raw      message.reasoning_content / .reasoning, only if sent
  reasoning_only     True for the reasoning_only outcome above
  pred               the parsed answer; None when there is none
  error              ERROR_KINDS key; absent when the request succeeded
  error_detail       human-readable detail for `error`
  finish_reason      choices[0].finish_reason ("length" = hit max_tokens)
  prompt_tokens, completion_tokens, latency_s, attempts
  timeout_s          the per-request timeout this question ran under

Timeouts adapt per model (see Asker): a model that decodes slower than
MIN_DECODE_TOKENS_PER_SEC gets a longer timeout once its rate is measured, and
a run gives up after MAX_CONSECUTIVE_TIMEOUTS in a row instead of waiting out
every remaining question on a hung server. A runner's --time-budget-s stops
asking new questions once spent (Asker.out_of_time): the questions not asked
are left out of the run (config.asked < config.n), not scored as failures, so
a slow model's partial run is marked as such rather than failing the gate.

Each runner also writes run_meta.json beside its predictions (server version,
request config, measured decode rate) and each scorer folds it into
summary.json, so a baseline entry records which FLM build produced it.
"""
import http.client
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

# Why each kind is (or isn't) retried. Decoding is greedy, so re-asking for a
# reply that came back empty would only produce the same empty reply.
ERROR_KINDS = {
    # The server accepted the request and is still grinding on it -- retrying
    # spends another full timeout window and queues behind the abandoned one.
    "timeout": 0,
    "connection_error": 3,
    "http_error": 3,
    "bad_response_shape": 3,
    # HTTP 200, but message.content is null/empty/whitespace and there is no
    # sign the model reasoned (no reasoning field, didn't hit max_tokens) --
    # otherwise it's the reasoning_only outcome, not a failure.
    "empty_reply": 0,
    # Never sent: the run gave up after MAX_CONSECUTIVE_TIMEOUTS (see Asker).
    "aborted": 0,
}
RETRY_BACKOFF_BASE_S = 3.0

# The slowest decode rate a request is allowed before it counts as a timeout,
# until Asker has measured the model's own rate.
MIN_DECODE_TOKENS_PER_SEC = 5

# Asker's adaptive timeout: a full max_tokens reply at the median decode rate
# measured so far, times this headroom. The median, so one stalled reply can't
# drag every later timeout up to the cap. Replies shorter than
# MIN_RATE_SAMPLE_TOKENS are mostly prefill, so they aren't a rate sample.
TIMEOUT_HEADROOM = 1.5
MIN_RATE_SAMPLE_TOKENS = 64
MAX_TIMEOUT_S = 3600.0

# Timeouts in a row before the server is presumed hung and the rest of the run
# is skipped -- otherwise 50 questions x a ~10 min timeout is 8 h of nothing.
MAX_CONSECUTIVE_TIMEOUTS = 3

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)

META_FILE = "run_meta.json"


def default_timeout(max_tokens, overhead_s):
    """Scale the per-request timeout with max_tokens so slow models don't time out mid-generation."""
    return max_tokens / MIN_DECODE_TOKENS_PER_SEC + overhead_s


def strip_think(text):
    """The final answer: the reply with all reasoning removed.

    Drops closed <think>...</think> blocks; everything before an orphan
    </think> (templates that open the block inside the prompt); and
    everything after an unclosed <think> (cut off mid-thought by max_tokens).
    """
    text = THINK_BLOCK.sub("", text or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def reasoning_only(rep, pred):
    """True when a reply has no final answer because the model was still reasoning."""
    if rep.get("error") or pred not in (None, ""):
        return False
    content = rep.get("content") or ""
    if not content.strip():
        return bool(rep.get("reasoning")) or rep.get("finish_reason") == "length"
    return not strip_think(content)


def annotate(level, msg):
    """GitHub Actions annotation (::warning:: / ::error::) on stderr."""
    print(f"::{level}::{msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Runner side
# --------------------------------------------------------------------------

def server_info(base_url):
    """GET /v1/version, or exit -- no point streaming questions at a dead server."""
    url = base_url.rstrip("/") + "/v1/version"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            raw = resp.read().decode("utf-8", "replace").strip()
    except Exception as exc:
        sys.exit(f"[error] cannot reach flm server at {base_url} ({exc!r}).\n"
                 f"        Start it first, e.g.:  flm serve <model> --port 52625")
    try:
        info = json.dumps(json.loads(raw), separators=(",", ":"))
    except ValueError:
        info = raw
    print(f"[server] {base_url} version: {info}", flush=True)
    return info[:300]


def _parse_reply(raw, rep):
    """Fill `rep` from a 200 response body. Returns (error_kind, detail), or (None, None) if usable."""
    try:
        body = json.loads(raw)
    except ValueError:
        return "bad_response_shape", f"non-JSON body: {raw[:300]!r}"
    try:
        choice = body["choices"][0]
        msg = choice["message"]
        content = msg["content"]
    except (KeyError, IndexError, TypeError):
        return "bad_response_shape", f"no choices[0].message.content; body: {json.dumps(body)[:500]}"

    usage = body.get("usage") or {}
    rep["finish_reason"] = choice.get("finish_reason")
    rep["prompt_tokens"] = usage.get("prompt_tokens")
    rep["completion_tokens"] = usage.get("completion_tokens")
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        rep["reasoning"] = reasoning
    if content is not None and not isinstance(content, str):
        return "bad_response_shape", f"message.content is {type(content).__name__}, not a string"
    rep["content"] = content

    # An empty answer from a model that reasoned (reasoning field, or spent
    # the whole budget) is a reply -- the reasoning_only outcome, a wrong
    # answer -- not a failure.
    if not (content or "").strip() and not reasoning and rep["finish_reason"] != "length":
        return "empty_reply", (f"message.content is {'null' if content is None else 'empty'} "
                               f"(finish_reason={rep['finish_reason']}, "
                               f"completion_tokens={rep['completion_tokens']})")
    return None, None


def chat(base_url, payload, timeout, qid):
    """POST one chat completion, retrying per ERROR_KINDS.

    Returns a reply dict: content, reasoning, finish_reason, prompt_tokens,
    completion_tokens, latency_s, attempts -- plus error/error_detail when no
    usable reply came back (already logged as a ::warning::).
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    attempt = 0
    while True:
        attempt += 1
        rep = {"attempts": attempt}
        t0 = time.time()
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            kind, detail = _parse_reply(raw, rep)
        # HTTPError and TimeoutError before the OSError catch-all: both are
        # OSErrors. A connect timeout arrives wrapped in a URLError instead.
        except urllib.error.HTTPError as exc:
            body = exc.read()[:500].decode("utf-8", "replace")
            kind, detail = "http_error", f"HTTP {exc.code} {exc.reason}: {body}"
        except TimeoutError as exc:
            kind, detail = "timeout", f"{exc!r} (timeout={timeout:.0f}s)"
        except (OSError, http.client.HTTPException) as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                kind, detail = "timeout", f"{exc!r} (timeout={timeout:.0f}s)"
            else:
                kind, detail = "connection_error", repr(exc)
        rep["latency_s"] = round(time.time() - t0, 2)

        if kind is None:
            return rep
        if attempt > ERROR_KINDS[kind]:
            rep["error"], rep["error_detail"] = kind, detail
            annotate("warning", f"id={qid}: no usable reply after {attempt} attempt(s) [{kind}]: {detail}")
            return rep
        backoff = RETRY_BACKOFF_BASE_S * attempt
        print(f"  [warn] id={qid}: [{kind}] {detail} (retry {attempt}/{ERROR_KINDS[kind]}, "
              f"backoff {backoff:.0f}s)", flush=True)
        time.sleep(backoff)


class Asker:
    """chat() with a per-model timeout and a give-up rule for a hung server.

    Timeout: starts at default_timeout(max_tokens, overhead_s). Every reply of
    at least MIN_RATE_SAMPLE_TOKENS gives a decode-rate sample (tokens/latency
    -- prefill included, so it errs slow); the timeout then becomes a full
    max_tokens reply at the median rate, x TIMEOUT_HEADROOM, never below the
    default nor above MAX_TIMEOUT_S. A fixed timeout (--timeout) disables this.

    Give-up: after max_consecutive_timeouts timeouts in a row (0 = never),
    `aborted` is set and the runner records every remaining question as an
    "aborted" failure instead of asking it.
    """

    def __init__(self, base_url, max_tokens, overhead_s, fixed_timeout=None,
                 max_consecutive_timeouts=MAX_CONSECUTIVE_TIMEOUTS, time_budget_s=0.0):
        self.base_url = base_url
        # 0: no budget. Otherwise no new question starts once it is spent.
        self.deadline = time.monotonic() + time_budget_s if time_budget_s else None
        self.out_of_time = False
        self.max_tokens = max_tokens
        self.adaptive = not fixed_timeout
        self.base_timeout = fixed_timeout or default_timeout(max_tokens, overhead_s)
        self.timeout = self.base_timeout
        self.max_consecutive_timeouts = max_consecutive_timeouts
        self.rates = []
        self.consecutive_timeouts = 0
        self.aborted = False

    def time_left(self):
        """False (and out_of_time set) once the run's time budget is spent."""
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.out_of_time = True
        return not self.out_of_time

    def ask(self, payload, qid):
        # A question started before the budget ran out may run past it by up
        # to its timeout; the job's timeout has to allow for that.
        timeout = self.timeout
        rep = chat(self.base_url, payload, timeout, qid)
        rep["timeout_s"] = round(timeout)
        if rep.get("error") == "timeout":
            self.consecutive_timeouts += 1
            if self.max_consecutive_timeouts and self.consecutive_timeouts >= self.max_consecutive_timeouts:
                self.aborted = True
                annotate("error", f"{self.consecutive_timeouts} timeouts in a row (timeout={timeout:.0f}s) -- "
                                  "the server looks hung; skipping the remaining questions.")
        else:
            self.consecutive_timeouts = 0
        self._observe(rep)
        return rep

    def _observe(self, rep):
        tok, lat = rep.get("completion_tokens"), rep.get("latency_s")
        if rep.get("error") or not tok or not lat or tok < MIN_RATE_SAMPLE_TOKENS:
            return
        self.rates.append(tok / lat)
        if not self.adaptive:
            return
        rate = self.decode_rate()
        new = min(MAX_TIMEOUT_S, max(self.base_timeout, self.max_tokens / rate * TIMEOUT_HEADROOM))
        if abs(new - self.timeout) > 0.1 * self.timeout:
            print(f"[timeout] median decode {rate:.2f} tok/s over {len(self.rates)} reply(s) -> "
                  f"per-request timeout {self.timeout:.0f}s -> {new:.0f}s", flush=True)
        self.timeout = new

    def decode_rate(self):
        return statistics.median(self.rates) if self.rates else None

    def aborted_reply(self):
        return {"attempts": 0, "timeout_s": round(self.timeout), "error": "aborted",
                "error_detail": f"not asked: the run gave up after {self.max_consecutive_timeouts} "
                                "timeouts in a row"}

    def stats(self):
        """Fields for run_meta.json once the run is over."""
        rate = self.decode_rate()
        return {"timeout_s": round(self.base_timeout), "timeout_adaptive": self.adaptive,
                "final_timeout_s": round(self.timeout),
                "decode_tok_per_s": None if rate is None else round(rate, 2),
                "aborted": self.aborted, "out_of_time": self.out_of_time}


def reply_fields(rep):
    """The shared per-question record fields for one chat() reply."""
    out = {"pred_raw": rep.get("content")}
    if rep.get("reasoning"):
        out["reasoning_raw"] = rep["reasoning"]
    for k in ("finish_reason", "prompt_tokens", "completion_tokens", "latency_s", "attempts", "timeout_s"):
        out[k] = rep.get(k)
    if rep.get("error"):
        out["error"] = rep["error"]
        out["error_detail"] = rep["error_detail"]
    return out


def load_resumable(out_path, id_key):
    """Resume support: keep records that got a usable reply, drop failed ones.

    Rewrites `out_path` without the failed records, so a re-run retries them
    instead of appending a duplicate line per question. Returns the kept ids.
    """
    keep = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("error") or r.get("pred_raw") is None:
                    continue
                keep[str(r[id_key])] = r
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in keep.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return set(keep)


def write_meta(out_path, **meta):
    path = os.path.join(os.path.dirname(os.path.abspath(out_path)), META_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def progress(i, n, t_start, qid, rep, pred, gt):
    el = time.time() - t_start
    status = rep.get("error") or ("ok" if pred not in (None, "") else
                                  "reasoning_only" if reasoning_only(rep, pred) else "unparsed")
    lat = rep.get("latency_s")
    shown = "None" if pred is None else repr(str(pred)[:40])
    print(f"[{i}/{n}] {el / i:.1f}s/q eta {(n - i) * el / i / 60:.0f}m id={qid} {status} "
          f"finish={rep.get('finish_reason')} out_tok={rep.get('completion_tokens')} "
          f"{'' if lat is None else f'{lat:.1f}s '}pred={shown} gt={str(gt)[:40]!r}",
          flush=True)


# --------------------------------------------------------------------------
# Scorer side
# --------------------------------------------------------------------------

def load_predictions(path, id_key):
    """Records from a predictions.jsonl, last record per id winning."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[str(r[id_key])] = r
    return list(rows.values())


def outcome(r):
    if r.get("error"):
        return "failed"
    if r.get("pred") not in (None, ""):
        return "answered"
    return "reasoning_only" if r.get("reasoning_only") else "unparsed"


def tally(rows):
    """Outcome counters shared by both benches' summary.json."""
    failures = {}
    for r in rows:
        if r.get("error"):
            failures[r["error"]] = failures.get(r["error"], 0) + 1
    failed = sum(failures.values())
    unparsed = sum(1 for r in rows if outcome(r) == "unparsed")
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    tok = [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
    return {
        "total": len(rows),
        "unanswered": failed + unparsed,
        "failed": failed,
        "failures": dict(sorted(failures.items())),
        "unparsed": unparsed,
        "reasoning_only": sum(1 for r in rows if outcome(r) == "reasoning_only"),
        "truncated": sum(1 for r in rows if r.get("finish_reason") == "length"),
        "mean_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
        "mean_completion_tokens": round(sum(tok) / len(tok), 1) if tok else None,
    }


def load_meta(pred_path):
    path = os.path.join(os.path.dirname(os.path.abspath(pred_path)), META_FILE)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def failures_text(counts):
    return ", ".join(f"{kind} ×{n}" for kind, n in counts["failures"].items())


def _cell(text, limit=160):
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[:limit] + "…"
    # Escape < and > so a <think> tag shows up instead of vanishing as HTML.
    return text.replace("|", "\\|").replace("`", "'").replace("<", "&lt;").replace(">", "&gt;")


def issue_detail(r):
    """One-line explanation of why a question is worth a look."""
    oc = outcome(r)
    if oc == "failed":
        return r.get("error_detail", "")
    reasoning = r.get("reasoning_raw")
    rs = f" · reasoning field: {len(reasoning)} chars" if reasoning else ""
    raw = r.get("pred_raw") or ""
    if oc == "reasoning_only":
        where = (f"reasoning field ({len(reasoning)} chars)" if reasoning else
                 "a <think> block" if raw.strip() else "an empty reply")
        return (f"no final answer -- still reasoning; scored wrong. {r.get('completion_tokens')} tokens, "
                f"finish={r.get('finish_reason')}, in {where}; tail: {(reasoning or raw)[-120:]!r}")
    if oc == "unparsed":
        what = "only a <think> block" if raw.strip() and not strip_think(raw) else "no answer found"
        return f"{what}; reply tail: {raw[-160:]!r}{rs}"
    return f"hit max_tokens; pred={r.get('pred')!r}{rs}"


def write_report(out_dir, title, headline, summary, rows, id_key, gt_key):
    """report.md: the run's counters plus a row per question worth a look.

    Listed: every failed, unparsed and reasoning_only question, and every
    answered one that hit max_tokens (its answer may be a cut-off guess).
    """
    issues = [r for r in rows if outcome(r) != "answered" or r.get("finish_reason") == "length"]
    meta_cfg = summary.get("config") or {}
    verdict = ("❌ **Not usable as a baseline:** " + "; ".join(summary["problems"])
               if summary.get("problems") else
               "⚠️ **Warning:** " + "; ".join(summary["warnings"]) if summary.get("warnings") else "")
    lines = [
        f"### {title}",
        "",
        *([verdict, ""] if verdict else []),
        f"**{headline}** · unanswered **{summary['unanswered']}**/{summary['total']} "
        f"(failed {summary['failed']}{' — ' + failures_text(summary) if summary['failed'] else ''}; "
        f"unparsed {summary['unparsed']}) · still reasoning, scored wrong "
        f"{summary.get('reasoning_only', 0)} · hit max_tokens {summary['truncated']}",
        "",
        f"Server `{summary.get('server') or 'unknown'}` · "
        + " · ".join(f"{k}={v}" for k, v in meta_cfg.items())
        + (f" · mean latency {summary['mean_latency_s']}s" if summary.get("mean_latency_s") is not None else "")
        + (f" · mean output tokens {summary['mean_completion_tokens']}"
           if summary.get("mean_completion_tokens") is not None else ""),
        "",
    ]
    if not issues:
        lines.append("No failed, unparsed, still-reasoning or truncated questions.")
    else:
        lines += [
            f"<details{' open' if summary['unanswered'] else ''}><summary>{len(issues)} question(s) worth a look"
            "</summary>",
            "",
            "| id | outcome | finish | tokens in / out | latency | gt | detail |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in issues:
            oc = outcome(r)
            oc = r["error"] if oc == "failed" else oc
            lat = r.get("latency_s")
            lines.append(
                f"| {_cell(r[id_key], 40)} | {oc} | {r.get('finish_reason') or '—'} | "
                f"{r.get('prompt_tokens') if r.get('prompt_tokens') is not None else '—'} / "
                f"{r.get('completion_tokens') if r.get('completion_tokens') is not None else '—'} | "
                f"{'—' if lat is None else f'{lat:.1f}s'} | {_cell(r.get(gt_key, ''), 40)} | "
                f"{_cell(issue_detail(r))} |")
        lines += ["", "</details>"]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def gate(label, metric_name, metric, summary, max_unanswered):
    """Fail the run (return 1) when its score can't be trusted as a baseline.

    No predictions or a pile of unanswered questions is almost always a broken
    run (server, harness, FLM build) rather than the model's real accuracy.
    Sets summary["usable"] (and "problems" when not): this is the benches'
    one hard gate, and the nightly baseline (report.py) keeps a model's
    previous entry instead of one whose "usable" is false.

    A score of 0 with the questions answered is the model's real result, so it
    only warns (summary["warnings"]): the run passes and stays usable.
    """
    problems, warnings = [], []
    if summary["total"] == 0:
        problems.append("no predictions were recorded")
    elif metric == 0:
        warnings.append(f"{metric_name} is 0 on {summary['total']} questions")
    if max_unanswered is not None and summary["unanswered"] > max_unanswered:
        why = []
        if summary["failed"]:
            why.append(f"failed {summary['failed']}: {failures_text(summary)}")
        if summary["unparsed"]:
            why.append(f"unparsed {summary['unparsed']}")
        problems.append(f"{summary['unanswered']}/{summary['total']} unanswered ({'; '.join(why)}), "
                        f"over --max-unanswered={max_unanswered}")
    summary["usable"] = not problems
    if problems:
        summary["problems"] = problems
    if warnings:
        summary["warnings"] = warnings
    for p in problems:
        annotate("error", f"{label}: {p}. See report.md in this model's artifact or the job summary.")
    for w in warnings:
        annotate("warning", f"{label}: {w}. See report.md in this model's artifact or the job summary.")
    if not problems and summary["unanswered"]:
        annotate("warning", f"{label}: {summary['unanswered']}/{summary['total']} unanswered, "
                            f"within --max-unanswered={max_unanswered}; each counts as wrong.")
    return 1 if problems else 0
