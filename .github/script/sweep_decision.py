#!/usr/bin/env python3
"""Sweep the decision models through POST /v1/systemone.

Per model (served with `--decision <tag>`):
  goldens     stored requests answered by the upstream BF16 model (assets/d3_goldens.jsonl): every answer has
              the upstream shape, top-1 agrees on at least --min-agreement of them, mean |dp| at most --max-dp
  types       one choice, one noul, one score, and all three in one request
  two-letter  a 27-option choice (the 27th option's code is "AA")
  invalid     a malformed question -> 422 {"detail": {"invalid_questions": ...}}, nothing run
  too-long    a prompt over the input limit -> max_length_exceeded inside a 200 reply
  image       a request with a PNG image (yellow box on blue) answered, and a broken image -> 422 "images[0]: ..."
  docs        the example request of docs/docs/models/d3.md: the server's reply must decide what the documented
              reply decides (choice, noul side, most likely score), so the docs cannot drift from the server

Decision models are absent from /v1/models, so the model set comes from `flm list` (family "d3"). Each decision
model gets its own server (--decision takes one tag).
"""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from sweep_common import ASSETS_DIR, SweepTask, run_task

REPO = Path(__file__).resolve().parents[2]
STATE = "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today."


def answer_shape_ok(answer: dict) -> bool:
    kind = answer.get("type")
    if "error" in answer:
        return set(answer) == {"type", "error", "message"}
    if kind == "choice":
        return list(answer) == ["type", "choice", "probabilities", "confidence"]
    if kind == "noul":
        return list(answer) == ["type", "noul"]
    if kind == "score":
        return list(answer) == ["type", "score", "probabilities", "confidence", "legend"]
    return False


class DecisionSweep(SweepTask):
    name = "decision"
    csv_header = ["Model", "Case", "Detail", "Latency (s)", "Status", "Error"]

    def __init__(self, args):
        super().__init__(args)
        # One server per decision model: the tag is the server's --decision argument. run() restarts
        # the server for every further model.
        self.needs_decision = (args.models or ["d3:lite"])[0]

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument("--min-agreement", type=float, default=0.85,
                            help="Share of golden answers whose top-1 must match the upstream BF16 model.")
        parser.add_argument("--max-dp", type=float, default=0.05,
                            help="Largest allowed mean |probability difference| against the upstream BF16 model.")

    def select_models(self, catalog: list[dict]) -> list[str]:
        return [entry["id"] for entry in catalog if entry["family"] == "d3"] or ["d3:lite"]

    # -- http -------------------------------------------------------------

    @staticmethod
    def _json_or_raw(raw: bytes) -> dict:
        """The reply body as JSON; a body that is not JSON (a proxy's or crash page's 5xx) becomes
        {"_raw": its start}, so the case fails instead of the whole sweep."""
        text = raw.decode("utf-8", errors="replace")
        try:
            body = json.loads(text or "{}")
        except ValueError:
            return {"_raw": text[:200]}
        return body if isinstance(body, dict) else {"_raw": text[:200]}

    def post(self, body) -> tuple[int, dict, float]:
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/systemone", data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.args.request_timeout) as resp:
                return resp.status, self._json_or_raw(resp.read()), time.monotonic() - started
        except urllib.error.HTTPError as err:
            return err.code, self._json_or_raw(err.read()), time.monotonic() - started

    def case(self, writer, model: str, case: str, ok: bool, detail: str, elapsed: float, error: str = "") -> None:
        self.record(writer, [model, case, detail, f"{elapsed:.2f}", "ok" if ok else "error", "" if ok else error],
                    model=model, error=None if ok else f"{case}: {error or detail}")
        print(f"  {case}: {detail} -> {'ok' if ok else 'FAILED ' + error}")

    # -- cases ------------------------------------------------------------

    def _goldens(self, writer, model: str) -> None:
        agree = total = 0
        dps: list[float] = []
        bad_shape: list[str] = []
        elapsed = 0.0
        # every d3 size has its own BF16 answers: d3_goldens.jsonl is d3:lite's, d3_goldens_<size>.jsonl the others'
        size = model.split(":", 1)[1]
        goldens = "d3_goldens.jsonl" if size == "lite" else f"d3_goldens_{size}.jsonl"
        if not (ASSETS_DIR / goldens).is_file():
            # e.g. d3_goldens_turbo.jsonl exists once the upstream BF16 reference has answered the requests
            self.case(writer, model, "goldens", False, f"{goldens} is missing", 0.0, f"no BF16 goldens for {model} ({goldens})")
            return
        for line in open(self.asset(goldens), encoding="utf-8"):
            rec = json.loads(line)
            status, body, t = self.post(rec["request"])
            elapsed += t
            if status != 200:
                bad_shape.append(f"{rec['id']}: status {status}")
                continue
            for key, want in rec["response"]["answers"].items():
                got = body["answers"].get(key, {})
                if not answer_shape_ok(got) or list(body["answers"]) != list(rec["response"]["answers"]):
                    bad_shape.append(f"{rec['id']}/{key}")
                    continue
                if "probabilities" in want:
                    p, q = list(want["probabilities"].values()), list(got["probabilities"].values())
                    dps.append(sum(abs(a - b) for a, b in zip(p, q)) / len(p))
                    agree += max(range(len(p)), key=p.__getitem__) == max(range(len(q)), key=q.__getitem__)
                    total += 1
                elif "noul" in want:
                    dps.append(abs(want["noul"] - got["noul"]))
                    agree += (want["noul"] >= 0.5) == (got["noul"] >= 0.5)
                    total += 1
        share = agree / max(total, 1)
        mdp = sum(dps) / max(len(dps), 1)
        ok = not bad_shape and share >= self.args.min_agreement and mdp <= self.args.max_dp
        self.case(writer, model, "goldens", ok, f"top-1 {agree}/{total}, mean |dp| {mdp:.4f}", elapsed,
                  "; ".join(bad_shape[:5]) or f"agreement {share:.3f} / mean |dp| {mdp:.4f} outside limits")

    def _types(self, writer, model: str) -> None:
        questions = {
            "route": {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": {"returns": "Refunds and replacements", "billing": "Payments", "general": None}},
            "urgent": {"type": "noul", "instructions": "Does this need a reply today?"},
            "anger": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]},
        }
        status, body, t = self.post({"state": STATE, "questions": questions})
        ok = (status == 200 and list(body.get("answers", {})) == list(questions)
              and all(answer_shape_ok(a) and "error" not in a for a in body["answers"].values())
              and body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] == 0)
        self.case(writer, model, "types", ok, f"status {status}, {len(body.get('answers', {}))} answers", t, json.dumps(body)[:200])

    def _two_letter(self, writer, model: str) -> None:
        criteria = {f"option{i:02d}": f"Option number {i}" for i in range(27)}
        status, body, t = self.post({"state": STATE, "questions": {"q": {"type": "choice", "criteria": criteria}}})
        a = body.get("answers", {}).get("q", {})
        ok = status == 200 and answer_shape_ok(a) and len(a.get("probabilities", {})) == 27
        self.case(writer, model, "two-letter", ok, f"status {status}, {len(a.get('probabilities', {}))} options", t, json.dumps(body)[:200])

    def _invalid(self, writer, model: str) -> None:
        status, body, t = self.post({"questions": {"ok": {"type": "noul"}, "bad": {"type": "rank"}}})
        ok = status == 422 and body == {"detail": {"invalid_questions": {
            "bad": "unsupported question type 'rank' (choice, noul or score)"}}}
        self.case(writer, model, "invalid", ok, f"status {status}", t, json.dumps(body)[:200])
        status, body, t = self.post(b'{"questions": ')
        ok = status == 422 and str(body.get("detail", "")).startswith("could not read the body as JSON")
        self.case(writer, model, "not-json", ok, f"status {status}", t, json.dumps(body)[:200])

    def _too_long(self, writer, model: str) -> None:
        status, body, t = self.post({"state": "word " * 33000, "questions": {"q": {"type": "noul"}}})
        a = body.get("answers", {}).get("q", {})
        ok = status == 200 and a.get("error") == "max_length_exceeded" and body["usage"]["input_tokens"] == 0
        self.case(writer, model, "too-long", ok, f"status {status}, {a.get('error')}", t, json.dumps(body)[:200])

    def _prefix_bidir(self, writer, model: str) -> None:
        """d3:turbo only: a box question is answered when /v1/systemone/models lists `box` (the engine returns the
        four hidden rows a box reads) and refused inline (`box_unsupported`) when it does not; input_tokens counts
        the state block once (4 questions cost far less than 4 single-question requests)."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/systemone/models", timeout=30) as resp:
                entry = json.loads(resp.read())["data"][0]
                listed = entry.get("question_types") or []
        except (urllib.error.URLError, ValueError, KeyError, IndexError) as err:
            self.case(writer, model, "models", False, "GET /v1/systemone/models failed", 0.0, str(err))
            return
        status, body, t = self.post({"state": STATE, "questions": {"b": {"type": "box", "instructions": "The customer's name"}}})
        a = body.get("answers", {}).get("b", {})
        if "box" in listed:
            box = a.get("box", {})
            ok = (status == 200 and list(a) == ["type", "box", "confidence"] and sorted(box) == ["bottom", "left", "right", "top"]
                  and all(0.0 <= v <= 1.0 for v in box.values()) and box["top"] <= box["bottom"] and box["left"] <= box["right"])
        else:
            ok = status == 200 and a.get("error") == "invalid_question" and a.get("reason") == "box_unsupported"
        self.case(writer, model, "box", ok, f"status {status}, box listed: {'box' in listed}", t, json.dumps(body)[:200])
        long_state = "The customer wrote: " + "the parcel is late, again. " * 40
        one = {"q1": {"type": "noul", "instructions": "Is the customer upset?"}}
        four = {f"q{i}": {"type": "noul", "instructions": f"Is the customer upset? ({i})"} for i in range(1, 5)}
        s1, b1, t1 = self.post({"state": long_state, "questions": one})
        s4, b4, t4 = self.post({"state": long_state, "questions": four})
        n1 = b1.get("usage", {}).get("input_tokens", 0)
        n4 = b4.get("usage", {}).get("input_tokens", 0)
        ok = s1 == 200 and s4 == 200 and n1 > 0 and n1 < n4 < 4 * n1
        self.case(writer, model, "state-once-usage", ok, f"1 question {n1} tokens, 4 questions {n4}", t1 + t4,
                  f"input_tokens {n1} / {n4}")
        # what the build lists is what it answers: boxes on an image, assets, a track over an encoded video
        # (needs FFmpeg), long_context refused
        if "box" in listed:
            status, body, t = self.post({"state": "A small test picture.", "images": [self.IMAGE],
                                         "questions": {"b": {"type": "box", "instructions": "The yellow rectangle"}}})
            a = body.get("answers", {}).get("b", {})
            box = a.get("box", {})
            ok = (status == 200 and list(a) == ["type", "box", "confidence"] and sorted(box) == ["bottom", "left", "right", "top"]
                  and all(0.0 <= v <= 1.0 for v in box.values()) and body["usage"]["input_tokens"] > 100)
            self.case(writer, model, "box-image", ok, f"status {status}, box {box}", t, json.dumps(body)[:200])
        if entry.get("assets"):
            status, body, t = self.post({"state": "Picture: <pic>", "assets": {"<pic>": self.IMAGE},
                                         "questions": {"bg": {"type": "choice", "instructions": "What is the main colour of the background?",
                                                              "criteria": {"blue": "Blue", "red": "Red", "green": "Green"}}}})
            a = body.get("answers", {}).get("bg", {})
            ok = status == 200 and answer_shape_ok(a) and "error" not in a and body["usage"]["input_tokens"] > 100
            self.case(writer, model, "assets", ok, f"status {status}, choice {a.get('choice')}", t, json.dumps(body)[:200])
        if "track" in listed:
            # a track question needs an encoded video (a frame list is only read by plain questions): the 4 s,
            # 48 x 32 H.264 clip of the video tests. Without the FFmpeg libraries the server refuses encoded
            # videos with 422 "video decoding backend not available": the case is then skipped, not failed.
            clip = REPO / "src/test/d3_video/fixtures/h264.mp4"
            if not clip.is_file():
                self.case(writer, model, "track", True, "skipped: the clip fixture is not in this checkout", 0.0)
            else:
                video = "data:video/mp4;base64," + base64.b64encode(clip.read_bytes()).decode("ascii")
                status, body, t = self.post({"state": "A short clip.", "videos": [video],
                                             "questions": {"t": {"type": "track", "instructions": "The moving object", "rate": 1}}})
                a = body.get("answers", {}).get("t", {})
                entries = a.get("track") or []
                if status == 422 and "decoding backend not available" in json.dumps(body):
                    self.case(writer, model, "track", True, "skipped: no FFmpeg libraries on this machine", t)
                else:
                    ok = (status == 200 and a.get("type") == "track" and len(entries) >= 1
                          and all(sorted(e) == ["box", "confidence", "time", "visible"] and 0.0 <= e["visible"] <= 1.0 for e in entries))
                    self.case(writer, model, "track", ok, f"status {status}, {len(entries)} moments", t, json.dumps(body)[:200])
        status, body, t = self.post({"state": STATE, "long_context": True, "questions": {"q": {"type": "noul"}}})
        ok = (status == 422) == (entry.get("long_context") is False)
        self.case(writer, model, "long-context", ok, f"status {status}, listed long_context {entry.get('long_context')}", t,
                  json.dumps(body)[:200])

    # 96 x 64 PNG: a yellow box on a blue background
    IMAGE = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAABACAIAAABqVuVZAAAAiElEQVR42u3bMRXAIAxAQcKrjk5V1QkZlYAM"
             "5FQEYmoApi4Z7u8ZuJeMxHm/RfsqAkCAAAECBAgQIK06/gzPp+V81dWHDXJigAABAiRAgAABAgQIECBAAgQIECBAgAABEiBAgAABAgQIkAAB"
             "AgQIECBAgABpXfiSaYMAAQIECBAgQAIECBCgbH3ZCwYBFiTfPwAAAABJRU5ErkJggg==")

    def _image(self, writer, model: str) -> None:
        q = {"type": "choice", "instructions": "What is the main colour of the background?",
             "criteria": {"blue": "Blue", "red": "Red", "green": "Green"}}
        status, body, t = self.post({"questions": {"bg": q}, "images": [self.IMAGE]})
        a = body.get("answers", {}).get("bg", {})
        ok = status == 200 and answer_shape_ok(a) and "error" not in a and body["usage"]["input_tokens"] > 100
        self.case(writer, model, "image", ok, f"status {status}, choice {a.get('choice')}", t, json.dumps(body)[:200])
        status, body, t = self.post({"questions": {"bg": q}, "images": [self.IMAGE[:120]]})
        ok = status == 422 and str(body.get("detail", "")).startswith("images[0]: ")
        self.case(writer, model, "image-broken", ok, f"status {status}", t, json.dumps(body)[:200])

    @staticmethod
    def _decision(answer: dict):
        """What an answer decides: the choice, the side of a noul, the most likely score level."""
        if answer.get("type") == "noul":
            return answer.get("noul", 0) >= 0.5
        probabilities = answer.get("probabilities") or {}
        return max(probabilities, key=probabilities.get) if probabilities else None

    def _same_decision(self, documented: dict, got: dict) -> bool:
        """choice / noul: exactly the documented decision. score: the documented reply is one model's and its example is
        a borderline one (levels 0 and 1 near each other for another model), so the neighbouring level may rank first;
        the documented level must still carry at least 0.3 of the server's probability."""
        if self._decision(documented) == self._decision(got):
            return True
        if documented.get("type") == "score" and got.get("type") == "score":
            level = self._decision(documented)
            return got.get("probabilities", {}).get(level, 0.0) >= 0.3
        return False

    def _docs(self, writer, model: str) -> None:
        doc = (REPO / "docs/docs/models/d3.md").read_text(encoding="utf-8")
        m = re.search(r"-d '(\{.*?\})'", doc, re.S)
        r = re.search(r"```json\s*(\{.*?\})\s*```", doc[m.end():], re.S) if m else None
        if not m or not r:
            self.case(writer, model, "docs", False, "no example request and reply in docs/docs/models/d3.md", 0.0, "missing")
            return
        request, documented = json.loads(m.group(1)), json.loads(r.group(1))
        status, body, t = self.post(request)
        answers = body.get("answers", {})
        ok = (status == 200 and list(answers) == list(request["questions"])
              and all(answer_shape_ok(a) and "error" not in a for a in answers.values()))
        # the documented reply must decide what the server decides; its numbers are rounded
        drift = [k for k, a in documented.get("answers", {}).items() if not self._same_decision(a, answers.get(k, {}))]
        ok = ok and not drift
        self.case(writer, model, "docs", ok, f"status {status}" + (f", docs decide differently on {drift}" if drift else ""),
                  t, json.dumps(body)[:200])

    def run(self) -> None:
        models = self.resolve_models()
        if self.args.no_server and len(models) > 1:
            raise SystemExit("--no-server attaches to one server, which serves one decision model; pass --models <tag>")
        handle, writer = self.open_csv()
        try:
            for model in self.iter_models(models):
                if model != self.needs_decision:   # each decision model needs its own server
                    self.stop_server()
                    self.needs_decision = model
                    self.start_server(append_log=True)
                print(f"\n--- Decision: {model} ---")
                self._goldens(writer, model)
                self._types(writer, model)
                self._two_letter(writer, model)
                self._invalid(writer, model)
                self._too_long(writer, model)
                if model == "d3:turbo":
                    self._prefix_bidir(writer, model)
                self._image(writer, model)
                self._docs(writer, model)
        finally:
            handle.close()
        self.write_summary(models)


if __name__ == "__main__":
    raise SystemExit(run_task(DecisionSweep))
