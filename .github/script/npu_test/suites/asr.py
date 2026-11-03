"""Whisper through /v1/audio/transcriptions, and chat models fed audio.

Runs the FLEURS evaluation set in assets/asr_eval/ and scores it: WER per set
(CER for ja and zh) against human transcripts, NFKC-normalised, each file's
edits capped at its reference length so one runaway transcript cannot swamp a
set, plus a pooled rate over every set. Around that it checks feature groups:

    verbose_json     segments ordered, inside the clip, non-empty, joining into
                     the text; duration equal to the clip length
    timestamp_rules  segments never overlap; the first starts within 1 s
    window_tail      no phrase repeated across 30 s window boundaries
    language         forcing the language is honoured; unknown codes rejected
    prompt           a prompt naming proper nouns does not hurt and helps them
    quiet_log        transcripts stay out of the server log

Core checks, verbose_json, timestamp_rules and prompt fail the run. The others
only warn: window_tail because main's HRX build loops on long-form audio,
language and quiet_log because main never required them.

The eval set is chosen by tier: install sends 2 en-short clips and skips the
groups that cost extra requests; pr sends every en-short clip and the first 3
of every other set; full (or --asr-full) sends all 84.
"""

from __future__ import annotations

import base64
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

from ..client import HttpError
from ..harness import ServerGone, expect, skip, warn

NAME = "asr"
DESCRIPTION = "Whisper transcription accuracy and features, and chat models fed audio"

# verbose_json's language, as OpenAI reports it: openai-whisper's lower-case names.
LANGUAGE_NAMES = {
    "en": "english", "de": "german", "es": "spanish", "fr": "french",
    "ja": "japanese", "zh": "chinese", "ar": "arabic",
}
CHAT_PROMPT = ("Transcribe the speech in this audio exactly, in its original language. "
               "Reply with the transcript only.")
CHAT_CLIPS = 5
# MP3 encoder padding and the server's own rounding.
DURATION_TOLERANCE_S = 0.1
# The encoder is not bit-exact between passes, so two requests for the same
# clip can differ in a word or two (see the PR #794 notes).
SAME_TEXT_MAX_WER = 0.02
REPEAT_NGRAM = 6
RUNAWAY_RATIO = 1.5
LONG_SETS = ("en-long", "en-longform")
# expected.json holds main's full-set rates from one machine per platform.
# The pooled rate gets 2 points of slack. A set gets 5, and only with 10 or
# more files: one flipped decode moves a ten-file set by a point or more, and
# a single long-form file by many, which HRX builds do between identical runs.
POOLED_TOLERANCE = 0.02
SET_TOLERANCE = 0.05
SET_CHECK_MIN_FILES = 10
PR_CLIPS_PER_SET = 3
INSTALL_CLIPS = 2
# A sanity bound that holds on every platform (main is at 8-10% on en-short),
# so a broken decoder fails even when no expected.json row is given.
EN_SHORT_MAX_WER = 0.5
ARABIC_DIACRITICS = re.compile("[\u064b-\u065f\u0670\u06d6-\u06ed]")


# -- scoring ------------------------------------------------------------------

def normalize(text: str, language: str) -> str:
    """Lowercase, no punctuation or symbols, single spaces, NFKC."""
    text = unicodedata.normalize("NFKC", text).lower()
    if language == "ar":
        text = ARABIC_DIACRITICS.sub("", text)
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def tokens(text: str, language: str, metric: str) -> list[str]:
    text = normalize(text, language)
    return list(text.replace(" ", "")) if metric == "cer" else text.split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, 1):
        current = [i]
        for j, hyp_token in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ref_token != hyp_token)))
        previous = current
    return previous[-1]


def error_rate(ref: str, hyp: str, language: str, metric: str) -> tuple[int, int]:
    """(edits, reference length), so rates can be pooled over a set."""
    ref_tokens = tokens(ref, language, metric)
    return edit_distance(ref_tokens, tokens(hyp, language, metric)), len(ref_tokens)


def pooled(pairs) -> float:
    """Error rate over many files, each file's edits capped at its length."""
    pairs = list(pairs)
    edits = sum(min(e, n) for e, n in pairs)
    length = sum(n for _, n in pairs)
    return edits / length if length else 0.0


def repeats(text: str, n: int = REPEAT_NGRAM) -> int:
    """n-word sequences that occur more than once."""
    words = normalize(text, "en").split()
    counts = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return sum(c - 1 for c in counts.values() if c > 1)


def same_text(a: str, b: str) -> tuple[bool, str]:
    edits, length = error_rate(a, b, "en", "wer")
    rate = edits / length if length else float(bool(b.strip()))
    return rate <= SAME_TEXT_MAX_WER, f"{rate:.1%} apart" if rate else "identical"


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def settle(problems: list[str], required: bool) -> None:
    """Ends a case that collected several checks: fail, warn, or pass."""
    if problems:
        message = "; ".join(problems)
        if required:
            raise AssertionError(message)
        warn(message)


# -- selection ----------------------------------------------------------------

def select_items(manifest_items: list[dict], tier: str, full: bool) -> list[dict]:
    """The clips this tier sends, in manifest order (so the subset is stable)."""
    if full:
        return list(manifest_items)
    taken: Counter = Counter()
    out = []
    for item in manifest_items:
        name = item["set"]
        if tier == "install":
            keep = name == "en-short" and taken[name] < INSTALL_CLIPS
        else:
            keep = name == "en-short" or taken[name] < PR_CLIPS_PER_SET
        if keep:
            taken[name] += 1
            out.append(item)
    return out


# -- requests -----------------------------------------------------------------

def transcribe(client, model: str, path: Path, **fields) -> tuple[dict, float]:
    """(parsed body, seconds). Anything but a JSON object without "error" raises."""
    form = {"model": model, **{k: v for k, v in fields.items() if v is not None}}
    started = time.monotonic()
    _, raw = client.post_multipart("/v1/audio/transcriptions", form,
                                   {"file": (path.name, path.read_bytes(), "audio/mpeg")})
    seconds = time.monotonic() - started
    try:
        body = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        body = None
    # The server answers with an empty body when no ASR model is loaded.
    expect(isinstance(body, dict), f"not a JSON object: {raw[:120]!r}")
    expect("error" not in body, f"error body: {json.dumps(body)[:200]}")
    return body, seconds


class WhisperRun:
    """One Whisper model's run: its eval results, shared by the later cases."""

    def __init__(self, ctx, server, model: str, items: list[dict], all_items: list[dict]):
        self.ctx = ctx
        self.server = server
        self.client = server.client
        self.model = model
        self.items = items
        self.all_items = all_items
        self.eval_dir = ctx.assets / "asr_eval"
        self.results: dict = {"items": {}, "sets": {}, "overall": {}, "language": {},
                              "prompt": {}}
        self.segment_failures: Counter = Counter()
        self.verbose_items = 0

    def case(self, name: str):
        return self.ctx.case(NAME, name, self.model, self.server)

    def path(self, item: dict) -> Path:
        return self.ctx.asset(f"asr_eval/{item['file']}")

    # -- formats -------------------------------------------------------------

    def formats(self) -> None:
        """json, no response_format and verbose_json on one clip must agree."""
        if self.ctx.tier == "install":
            item = self.items[0]
        else:
            item = next((i for i in self.items if i["set"] == "en-long"), self.items[0])
        path = self.path(item)
        plain = verbose = None
        with self.case("formats") as case:
            case.data["file"] = item["file"]
            plain, _ = transcribe(self.client, self.model, path, response_format="json")
            text = plain.get("text") or ""
            expect(text.strip(), "json text is empty")
            default, _ = transcribe(self.client, self.model, path)
            expect(set(default) == {"model", "text"},
                   f"default format returns {', '.join(sorted(default))}, not model and text")
            ok, detail = same_text(text, default.get("text") or "")
            expect(ok, f"default text differs from json: {detail}")
        with self.case("verbose-json") as case:
            case.data["file"] = item["file"]
            if plain is None:
                skip("the json request failed")
            verbose, _ = transcribe(self.client, self.model, path,
                                    response_format="verbose_json")
            expect("segments" in verbose, "no segments: the server ignores verbose_json")
            ok, detail = same_text(plain.get("text") or "", verbose.get("text") or "")
            expect(ok, f"verbose_json text differs from json: {detail}")

    # -- evaluation ----------------------------------------------------------

    def check_segments(self, body: dict, duration: float) -> None:
        """Tallies the verbose_json and timestamp_rules problems of one response."""
        segments = body.get("segments") or []
        text = body.get("text") or ""
        problems = {
            ("verbose_json", "duration is the clip length"):
                not isinstance(body.get("duration"), (int, float))
                or abs(body["duration"] - duration) > DURATION_TOLERANCE_S,
            ("verbose_json", "segment ids count up from 0"):
                [s.get("id") for s in segments] != list(range(len(segments))),
            ("verbose_json", "segments lie inside the clip"):
                any(not 0 <= s["start"] <= s["end"] <= duration + DURATION_TOLERANCE_S
                    for s in segments),
            ("verbose_json", "segments are in order"):
                any(b["start"] < a["start"] for a, b in zip(segments, segments[1:])),
            ("verbose_json", "no empty or zero-length segments"):
                any(s["end"] - s["start"] < 0.01 or not s["text"].strip() for s in segments),
            ("verbose_json", "segments join into the text"):
                "".join(s["text"] for s in segments).strip() != text.strip(),
            ("timestamp_rules", "segments never overlap"):
                any(b["start"] < a["end"] - 0.005 for a, b in zip(segments, segments[1:])),
            ("timestamp_rules", "first segment starts within 1 s"):
                bool(segments) and segments[0]["start"] > 1.0,
        }
        for key, bad in problems.items():
            self.segment_failures[key] += bool(bad)

    def eval_set(self, set_name: str, set_items: list[dict]) -> None:
        """One verbose_json request per clip: accuracy, speed, segments."""
        with self.case(f"eval-{set_name}") as case:
            pairs, audio_s, wall_s, detected, runaways = [], 0.0, 0.0, 0, []
            errors = []
            for item in set_items:
                try:
                    body, seconds = transcribe(self.client, self.model, self.path(item),
                                               response_format="verbose_json")
                except Exception as exc:  # one bad file must not hide the rest
                    errors.append(f"{item['file']}: {type(exc).__name__}: {exc}"[:200])
                    if not self.server.alive():
                        break  # the case fails and the server is restarted
                    continue
                text = body.get("text") or ""
                edits, length = error_rate(item["reference"], text, item["language"],
                                           item["metric"])
                runaway = (len(tokens(text, item["language"], item["metric"]))
                           > RUNAWAY_RATIO * length)
                pairs.append((edits, length))
                audio_s += item["duration"]
                wall_s += seconds
                if runaway:
                    runaways.append(item["file"])
                self.results["items"][item["file"]] = {
                    "set": set_name, "language": item["language"], "metric": item["metric"],
                    "text": text, "seconds": round(seconds, 2), "edits": edits,
                    "length": length, "runaway": runaway}
                if body.get("language") == LANGUAGE_NAMES[item["language"]]:
                    detected += 1
                if "segments" in body:
                    self.verbose_items += 1
                    self.check_segments(body, item["duration"])
            rate = pooled(pairs)
            stats = {
                "metric": set_items[0]["metric"], "error_rate": round(rate, 4),
                "files": len(pairs), "of": len(set_items), "runaways": len(runaways),
                "audio_s": round(audio_s, 1), "wall_s": round(wall_s, 1),
                "rtf": round(wall_s / audio_s, 4) if audio_s else None,
                "language_detected": detected,
            }
            self.results["sets"][set_name] = stats
            case.data.update(stats)
            if errors:
                raise AssertionError(f"{len(pairs)}/{len(set_items)} transcribed; {errors[0]}")
            if set_name == "en-short" and pairs:
                expect(rate < EN_SHORT_MAX_WER, f"WER {pct(rate)} on en-short, "
                       f"over the {pct(EN_SHORT_MAX_WER)} sanity bound")
            # Long-form runaways are window_tail's to report. Elsewhere one is
            # worth a look, but a single odd decode is not a regression.
            if runaways and set_name not in LONG_SETS:
                warn(f"runaway transcripts: {', '.join(runaways)}")

    def evaluate(self) -> None:
        by_set: dict[str, list[dict]] = {}
        for item in self.items:
            by_set.setdefault(item["set"], []).append(item)
        for set_name, set_items in by_set.items():
            self.eval_set(set_name, set_items)
        scored = self.results["items"].values()
        self.results["overall"] = {
            "error_rate": round(pooled((r["edits"], r["length"]) for r in scored), 4),
            "runaways": sum(r["runaway"] for r in scored),
            "files": len(self.results["items"]),
        }

    def segment_cases(self) -> None:
        for group, name in (("verbose_json", "segments"),
                            ("timestamp_rules", "timestamp-rules")):
            with self.case(name) as case:
                case.data["files"] = self.verbose_items
                if not self.verbose_items:
                    skip("no verbose_json response with segments to check")
                bad = [f"{check}: {count} of {self.verbose_items} files"
                       for (g, check), count in sorted(self.segment_failures.items())
                       if g == group and count]
                expect(not bad, "; ".join(bad))

    def window_tail(self) -> None:
        long_items = [f for f, r in self.results["items"].items() if r["set"] in LONG_SETS]
        if not long_items:
            return
        reference = {i["file"]: i["reference"] for i in self.items}
        with self.case("window-tail") as case:
            problems = []
            for name in long_items:
                text = self.results["items"][name]["text"]
                counts = {"reference": repeats(reference[name]), "text": repeats(text)}
                case.data[name] = counts
                if counts["text"] > counts["reference"]:
                    problems.append(f"{name}: {counts['text']} repeated "
                                    f"{REPEAT_NGRAM}-word phrases, reference {counts['reference']}")
            settle(problems, required=False)

    # -- feature probes ------------------------------------------------------

    def language(self) -> None:
        """Forcing the language: no worse than detecting it, and reported back."""
        for language in sorted({i["language"] for i in self.items} - {"en"}):
            probe = [i for i in self.items if i["language"] == language][:PR_CLIPS_PER_SET]
            with self.case(f"language-{language}") as case:
                pairs, reported, problems, failed = [], [], [], []
                for item in probe:
                    try:
                        body, _ = transcribe(self.client, self.model, self.path(item),
                                             response_format="verbose_json",
                                             language=language)
                    except (HttpError, AssertionError) as exc:
                        # A valid language must be accepted and transcribed, as
                        # the old asr_test.py required; only how well it does
                        # below is a warning.
                        failed.append(f"{item['file']}: {type(exc).__name__}: {exc}"[:200])
                        continue
                    pairs.append(error_rate(item["reference"], body.get("text") or "",
                                            language, item["metric"]))
                    if "segments" in body:
                        reported.append(body.get("language"))
                auto = pooled((self.results["items"][i["file"]]["edits"],
                               self.results["items"][i["file"]]["length"])
                              for i in probe if i["file"] in self.results["items"])
                forced = pooled(pairs)
                case.data.update({"forced": round(forced, 4), "detected": round(auto, 4)})
                self.results["language"][language] = case.data.copy()
                if pairs and forced > auto + SET_TOLERANCE:
                    problems.append(f"forcing it hurts: {pct(forced)} forced, "
                                    f"{pct(auto)} detected")
                # Only verbose_json reports the language. Forcing a wrong
                # language is no test: Whisper often keeps transcribing what
                # it hears.
                if any(r != LANGUAGE_NAMES[language] for r in reported):
                    problems.append("response reports "
                                    + ", ".join(sorted({str(r) for r in reported})))
                expect(not failed, "forced-language requests failed: " + "; ".join(failed))
                settle(problems, required=False)

        english = next((i for i in self.items if i["set"] == "en-short"), None)
        if english is None:
            return
        with self.case("language-invalid") as case:
            try:
                body, _ = transcribe(self.client, self.model, self.path(english),
                                     language="xx")
            except HttpError as exc:
                case.data["status"] = exc.status
                if exc.status >= 500:
                    warn(f"language 'xx' got HTTP {exc.status}, not a 4xx")
                return
            except AssertionError as exc:  # an error body is a rejection too
                case.data["status"] = str(exc)[:120]
                return
            warn(f"language 'xx' was accepted: {json.dumps(body)[:120]}")

    def prompt(self) -> None:
        """A prompt naming the clip's proper nouns should not hurt, and should help them."""
        probe = [i for i in self.items if i["set"] == "en-short"][:5]
        with self.case("prompt") as case:
            hits_without = hits_with = names_total = 0
            pairs_without, pairs_with = [], []
            for item in probe:
                names = sorted({w.strip(".,;:!?\"'()") for w in item["raw_reference"].split()[1:]
                                if w[:1].isupper()} - {""})
                before = self.results["items"].get(item["file"])
                if not names or before is None:
                    continue
                body, _ = transcribe(self.client, self.model, self.path(item),
                                     prompt=" ".join(names))
                after = body.get("text") or ""
                pairs_with.append(error_rate(item["reference"], after, "en", "wer"))
                pairs_without.append((before["edits"], before["length"]))
                names_total += len(names)
                hits_without += sum(n in before["text"] for n in names)
                hits_with += sum(n in after for n in names)
            if not names_total:
                skip("no en-short clip with proper nouns was transcribed")
            with_rate, without_rate = pooled(pairs_with), pooled(pairs_without)
            case.data.update({"names": names_total, "hits_with": hits_with,
                              "hits_without": hits_without, "wer_with": round(with_rate, 4),
                              "wer_without": round(without_rate, 4)})
            self.results["prompt"] = dict(case.data)
            problems = []
            if with_rate > without_rate + SET_TOLERANCE:
                problems.append(f"a prompt hurts accuracy: WER {pct(with_rate)} with, "
                                f"{pct(without_rate)} without")
            if hits_with < hits_without:
                problems.append(f"names spelled as given {hits_with}/{names_total} with the "
                                f"prompt, {hits_without}/{names_total} without")
            settle(problems, required=True)

    def quiet_log(self) -> None:
        with self.case("quiet-log") as case:
            try:
                log = self.server.log_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                skip(f"cannot read the server log: {exc}")
            # Streamed output interleaves timestamp markers with the text.
            log = "".join(re.sub(r"<\|[\d.]+\|>", "", log).split())
            leaked = []
            for name, r in self.results["items"].items():
                squashed = "".join(r["text"].split())
                if len(squashed) >= 20 and squashed[:20] in log:
                    leaked.append(name)
            case.data["leaked"] = len(leaked)
            if leaked:
                warn(f"{len(leaked)} of {len(self.results['items'])} transcripts found in "
                     f"the server log, e.g. {leaked[0]}")

    # -- expected rates ------------------------------------------------------

    def compare_expected(self) -> None:
        """Rates against expected.json's row for this platform.

        Those rates are main's over the full set. A set that ran in full is
        compared as it is. The pooled rate of a subset is not comparable to
        the full pooled rate, so it is compared to the expected set rates
        weighted by this run's reference lengths per set, and only warns:
        three clips of a set are far noisier than ten.
        """
        key = self.ctx.asr_expected_key
        if not key or not self.results["items"]:
            return
        with self.case("wer-expected") as case:
            platforms = json.loads(self.ctx.asset("asr_eval/expected.json")
                                   .read_text(encoding="utf-8"))["platforms"]
            expected = platforms.get(key)
            if expected is None:
                warn(f"expected.json has no row for {key!r}; accuracy is not checked")
            set_limits = expected.get("sets", {})
            # Judged by what was sent, not what came back: a failed file is
            # already a failure of its eval case.
            selected = Counter(i["set"] for i in self.items)
            complete = {name for name, n in Counter(i["set"] for i in self.all_items).items()
                        if selected[name] == n}
            subset = len(self.items) < len(self.all_items)
            overall = self.results["overall"]["error_rate"]
            failures, warnings = [], []

            if subset:
                lengths: Counter = Counter()
                for r in self.results["items"].values():
                    lengths[r["set"]] += r["length"]
                total = sum(lengths[s] for s in lengths if s in set_limits)
                limit = (sum(set_limits[s] * n for s, n in lengths.items() if s in set_limits)
                         / total if total else None)
                tolerance, bucket = SET_TOLERANCE, warnings
            else:
                limit, tolerance, bucket = expected.get("overall"), POOLED_TOLERANCE, failures
            case.data.update({"key": key, "subset": subset, "pooled": overall,
                              "pooled_expected": None if limit is None else round(limit, 4),
                              "sets": {}})
            if limit is not None and overall > limit + tolerance:
                bucket.append(f"pooled {pct(overall)} vs expected {pct(limit)}")

            for name, current in self.results["sets"].items():
                limit = set_limits.get(name)
                if limit is None:
                    continue
                case.data["sets"][name] = {"error_rate": current["error_rate"],
                                           "expected": limit, "rtf": current["rtf"]}
                if (name in complete and current["files"] >= SET_CHECK_MIN_FILES
                        and current["error_rate"] > limit + SET_TOLERANCE):
                    failures.append(f"{name} {pct(current['error_rate'])} vs expected "
                                    f"{pct(limit)}")
            expect(not failures, "; ".join(failures + warnings))
            settle(warnings, required=False)

    def save(self) -> None:
        """Every transcript and rate, for a reader chasing a WER change."""
        safe = re.sub(r"[^\w.-]", "_", self.model)
        path = self.ctx.out_dir / f"asr-{safe}.json"
        path.write_text(json.dumps({"model": self.model, **self.results}, indent=1,
                                   ensure_ascii=False), encoding="utf-8")

    def run(self) -> None:
        try:
            self.formats()
            self.evaluate()
            self.segment_cases()
            self.window_tail()
            if self.ctx.tier != "install":
                self.language()
                self.prompt()
            self.quiet_log()
            self.compare_expected()
        finally:
            self.save()


# -- chat audio ---------------------------------------------------------------

def run_chat(ctx, models: list[str], clips: list[dict]) -> None:
    """Chat models with an audio encoder transcribe a few en-short clips."""
    audio = [(item, base64.b64encode(ctx.asset(f"asr_eval/{item['file']}").read_bytes())
              .decode("ascii")) for item in clips]
    with ctx.server("asr-chat") as server:
        for index, model_id in enumerate(models):
            model = ctx.plan.model(model_id)
            # Sampling and think state persist on the loaded model, so pin both.
            extra = {"temperature": 0, "top_k": 1,
                     "max_tokens": 1024 if model.always_thinks else 256}
            if model.think:
                extra["think"] = False
            try:
                with ctx.case(NAME, "chat-audio", model_id, server) as case:
                    pairs, errors, seconds = [], [], 0.0
                    for item, encoded in audio:
                        try:
                            reply = server.client.chat(model_id, [{"role": "user", "content": [
                                {"type": "text", "text": CHAT_PROMPT},
                                {"type": "input_audio",
                                 "input_audio": {"data": encoded, "format": "mp3"}},
                            ]}], stream=False, **extra)
                            text = reply.content
                            seconds += reply.seconds
                        except Exception as exc:
                            errors.append(f"{item['file']}: {type(exc).__name__}: {exc}"[:200])
                            if not server.alive():
                                break  # the case fails and the server is restarted
                            text = ""
                        # An empty transcript counts as every word missed.
                        pairs.append(error_rate(item["reference"], text, "en", "wer"))
                    rate = pooled(pairs)
                    case.data.update({"wer": round(rate, 4), "clips": len(audio),
                                      "transcribed": len(audio) - len(errors),
                                      "seconds": round(seconds, 1)})
                    if errors:
                        raise AssertionError(f"{len(audio) - len(errors)}/{len(audio)} "
                                             f"transcribed; {errors[0]}")
                    expect(rate < 0.5, f"WER {pct(rate)}, not under 50%")
            except ServerGone as exc:
                ctx.not_attempted(NAME, models[index + 1:], str(exc))
                return


# -- entry point --------------------------------------------------------------

def run(ctx) -> None:
    whisper = ctx.plan.suite("whisper")
    chat = ctx.plan.suite("asr_chat")
    if not whisper and not chat:
        return
    manifest = json.loads(ctx.asset("asr_eval/manifest.json").read_text(encoding="utf-8"))
    all_items = manifest["items"]
    items = select_items(all_items, ctx.tier, bool(getattr(ctx, "asr_full", False)))
    print(f"[asr] {len(items)} of {len(all_items)} clips, expected rates: "
          f"{getattr(ctx, 'asr_expected_key', '') or 'none'}", flush=True)

    if whisper:
        # -a 1 loads the ASR sidecar, which serves every Whisper tag.
        with ctx.server("asr", ["-a", "1"]) as server:
            for index, model in enumerate(whisper):
                try:
                    WhisperRun(ctx, server, model, items, all_items).run()
                except ServerGone as exc:
                    # This model's remaining cases are lost too.
                    ctx.not_attempted(NAME, whisper[index:], str(exc))
                    break
    if chat:
        clips = [i for i in all_items if i["set"] == "en-short"][:CHAT_CLIPS]
        run_chat(ctx, chat, clips)
