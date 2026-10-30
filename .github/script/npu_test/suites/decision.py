"""The decision (d3) models through POST /v1/systemone.

Each model gets its own server, since --decision takes one tag. Per model:
  goldens     stored requests answered by the upstream BF16 model: every answer has the
              upstream shape, top-1 agrees on >= 85% of them, mean |dp| <= 0.05
  types       one choice, one noul, one score, and all three in one request
  two-letter  a 27-option choice (the 27th option's code is "AA")
  invalid     a malformed question -> 422 naming it, nothing run; a body that is not JSON -> 422
  too-long    a prompt over the input limit -> max_length_exceeded inside a 200 reply
  image       a PNG answered, a broken one -> 422 "images[0]: ..."
  d3:turbo and d3:ultra also get the prefix-bidirectional checks (box, state counted once,
  assets, track, long_context), each gated on what /v1/systemone/models lists.
The install tier runs goldens and types only.
"""

from __future__ import annotations

import base64
import json
import time

from ..client import HttpError
from ..harness import FAIL, Result, ServerGone, expect, skip, warn
from ..server import ServerError

NAME = "decision"
DESCRIPTION = "/v1/systemone on each d3 model: BF16 goldens, question types, errors, images"

MIN_AGREEMENT = 0.85
MAX_MEAN_DP = 0.05
PREFIX_MODELS = ("d3:turbo", "d3:ultra")

STATE = ("The order arrived damaged yesterday. The customer has a receipt and asks for "
         "a replacement today.")

# 96 x 64 PNG: a yellow box on a blue background
IMAGE = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAABACAIAAABqVuVZAAAAiElEQVR42u3bMRXAIAxAQcKrjk5V1QkZlYAM"
         "5FQEYmoApi4Z7u8ZuJeMxHm/RfsqAkCAAAECBAgQIK06/gzPp+V81dWHDXJigAABAiRAgAABAgQIECBAAgQIECBAgAABEiBAgAABAgQIkAAB"
         "AgQIECBAgABpXfiSaYMAAQIECBAgQAIECBCgbH3ZCwYBFiTfPwAAAABJRU5ErkJggg==")

BACKGROUND_QUESTION = {"type": "choice", "instructions": "What is the main colour of the background?",
                       "criteria": {"blue": "Blue", "red": "Red", "green": "Green"}}

# The 4 s, 48 x 32 H.264 clip of the d3 video tests.
TRACK_CLIP = "src/test/d3_video/fixtures/h264.mp4"


def answer_shape_ok(answer: dict) -> bool:
    """The upstream reply shape, key order included: clients index these positionally."""
    if not isinstance(answer, dict):
        return False
    if "error" in answer:
        return set(answer) == {"type", "error", "message"}
    return list(answer) == {
        "choice": ["type", "choice", "probabilities", "confidence"],
        "noul": ["type", "noul"],
        "score": ["type", "score", "probabilities", "confidence", "legend"],
    }.get(answer.get("type"), None)


def post(server, body) -> tuple[int, dict]:
    """POST /v1/systemone, returning (status, JSON body) for 2xx and errors alike.

    A body that is not JSON (a proxy's or crash page's 5xx) comes back as
    {"_raw": its start}, so the case fails on it rather than raising.
    """
    data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    try:
        status, raw = server.client.post_raw("/v1/systemone", data, "application/json")
        text = raw.decode("utf-8", "replace")
    except HttpError as exc:
        status, text = exc.status, exc.body
    try:
        parsed = json.loads(text or "{}")
    except ValueError:
        return status, {"_raw": text[:200]}
    return status, parsed if isinstance(parsed, dict) else {"_raw": text[:200]}


def brief(body: dict) -> str:
    return json.dumps(body)[:200]


def server_name(model: str) -> str:
    # The log file is named after the server; ':' is not allowed in Windows file names.
    return f"{NAME}-{model.replace(':', '-')}"


def run(ctx) -> None:
    for model in ctx.plan.suite(NAME):
        # One server per model: a model whose server will not start or stay up
        # must not cost the other models their run.
        try:
            with ctx.server(server_name(model), ["--decision", model]) as server:
                check_model(ctx, server, model)
        except ServerError as exc:
            ctx.record(Result(NAME, "server-start", FAIL, model, str(exc)))
        except ServerGone as exc:
            ctx.record(Result(NAME, "server-gone", FAIL, model, str(exc)))


def check_model(ctx, server, model: str) -> None:
    goldens(ctx, server, model)
    types(ctx, server, model)
    if ctx.tier == "install":
        return
    two_letter(ctx, server, model)
    invalid(ctx, server, model)
    too_long(ctx, server, model)
    if model in PREFIX_MODELS:
        prefix_bidirectional(ctx, server, model)
    image(ctx, server, model)


# -- cases -------------------------------------------------------------------

def goldens(ctx, server, model: str) -> None:
    with ctx.case(NAME, "goldens", model, server) as case:
        # Each d3 size has its own BF16 answers: d3_goldens.jsonl is d3:lite's,
        # d3_goldens_<size>.jsonl the others'.
        size = model.split(":", 1)[-1]
        name = "d3_goldens.jsonl" if size == "lite" else f"d3_goldens_{size}.jsonl"
        path = ctx.assets / name
        if not path.is_file():
            # A model added to the catalog before its goldens: nothing to
            # compare against yet, and the other cases still cover it.
            warn(f"no BF16 goldens for {model} yet ({name}); add them to compare its answers")
        agree = total = 0
        dps: list[float] = []
        bad: list[str] = []
        started = time.monotonic()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            status, body = post(server, rec["request"])
            if status != 200:
                bad.append(f"{rec['id']}: status {status}")
                continue
            wanted = rec["response"]["answers"]
            got_all = body.get("answers") or {}
            if list(got_all) != list(wanted):
                bad.append(f"{rec['id']}: answers {list(got_all)}")
                continue
            for key, want in wanted.items():
                got = got_all[key]
                if not answer_shape_ok(got) or "error" in got:
                    bad.append(f"{rec['id']}/{key}")
                    continue
                if "probabilities" in want:
                    p = list(want["probabilities"].values())
                    q = list(got["probabilities"].values())
                    if len(p) != len(q):
                        bad.append(f"{rec['id']}/{key}: {len(q)} probabilities, want {len(p)}")
                        continue
                    dps.append(sum(abs(a - b) for a, b in zip(p, q)) / len(p))
                    agree += max(range(len(p)), key=p.__getitem__) == max(range(len(q)), key=q.__getitem__)
                    total += 1
                elif "noul" in want:
                    dps.append(abs(want["noul"] - got["noul"]))
                    agree += (want["noul"] >= 0.5) == (got["noul"] >= 0.5)
                    total += 1
        share = agree / max(total, 1)
        mean_dp = sum(dps) / max(len(dps), 1)
        case.data.update(agree=agree, total=total, mean_dp=round(mean_dp, 4),
                         seconds_per_request=round((time.monotonic() - started) / max(total, 1), 3))
        expect(not bad, f"bad replies: {'; '.join(bad[:5])}")
        expect(total > 0, f"{name} has no answer to compare")
        expect(share >= MIN_AGREEMENT and mean_dp <= MAX_MEAN_DP,
               f"top-1 {agree}/{total} (min {MIN_AGREEMENT}), mean |dp| {mean_dp:.4f} "
               f"(max {MAX_MEAN_DP})")


def types(ctx, server, model: str) -> None:
    with ctx.case(NAME, "types", model, server):
        questions = {
            "route": {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": {"returns": "Refunds and replacements", "billing": "Payments",
                                   "general": None}},
            "urgent": {"type": "noul", "instructions": "Does this need a reply today?"},
            "anger": {"type": "score", "instructions": "How upset is the customer?",
                      "criteria": ["calm", "annoyed", "angry"]},
        }
        status, body = post(server, {"state": STATE, "questions": questions})
        expect(status == 200, f"status {status}: {brief(body)}")
        answers = body.get("answers") or {}
        expect(list(answers) == list(questions), f"answers {list(answers)}, want {list(questions)}")
        wrong = [k for k, a in answers.items() if not answer_shape_ok(a) or "error" in a]
        expect(not wrong, f"bad answers {wrong}: {brief(body)}")
        usage = body.get("usage") or {}
        # A decision is read off the prefill; nothing is generated.
        expect(usage.get("input_tokens", 0) > 0 and usage.get("output_tokens") == 0,
               f"usage {usage}")


def two_letter(ctx, server, model: str) -> None:
    with ctx.case(NAME, "two-letter", model, server):
        criteria = {f"option{i:02d}": f"Option number {i}" for i in range(27)}
        status, body = post(server, {"state": STATE,
                                     "questions": {"q": {"type": "choice", "criteria": criteria}}})
        expect(status == 200, f"status {status}: {brief(body)}")
        answer = (body.get("answers") or {}).get("q", {})
        expect(answer_shape_ok(answer) and "error" not in answer, f"bad answer: {brief(body)}")
        expect(len(answer.get("probabilities", {})) == 27,
               f"{len(answer.get('probabilities', {}))} options, want 27")


def invalid(ctx, server, model: str) -> None:
    with ctx.case(NAME, "invalid", model, server):
        status, body = post(server, {"questions": {"ok": {"type": "noul"}, "bad": {"type": "rank"}}})
        expect(status == 422, f"status {status}: {brief(body)}")
        want = {"detail": {"invalid_questions": {
            "bad": "unsupported question type 'rank' (choice, noul or score)"}}}
        expect(body == want, f"reply {brief(body)}")
    with ctx.case(NAME, "not-json", model, server):
        status, body = post(server, b'{"questions": ')
        expect(status == 422, f"status {status}: {brief(body)}")
        expect(str(body.get("detail", "")).startswith("could not read the body as JSON"),
               f"reply {brief(body)}")


def too_long(ctx, server, model: str) -> None:
    with ctx.case(NAME, "too-long", model, server):
        status, body = post(server, {"state": "word " * 33000, "questions": {"q": {"type": "noul"}}})
        expect(status == 200, f"status {status}: {brief(body)}")
        answer = (body.get("answers") or {}).get("q", {})
        expect(answer.get("error") == "max_length_exceeded", f"answer {brief(answer)}")
        expect((body.get("usage") or {}).get("input_tokens") == 0,
               f"usage {body.get('usage')}: nothing should have run")


def image(ctx, server, model: str) -> None:
    with ctx.case(NAME, "image", model, server) as case:
        status, body = post(server, {"questions": {"bg": BACKGROUND_QUESTION}, "images": [IMAGE]})
        expect(status == 200, f"status {status}: {brief(body)}")
        answer = (body.get("answers") or {}).get("bg", {})
        expect(answer_shape_ok(answer) and "error" not in answer, f"bad answer: {brief(body)}")
        case.data["choice"] = answer.get("choice")
        # The image tokens must have reached the prefill, not just the text.
        expect((body.get("usage") or {}).get("input_tokens", 0) > 100, f"usage {body.get('usage')}")
    with ctx.case(NAME, "image-broken", model, server):
        status, body = post(server, {"questions": {"bg": BACKGROUND_QUESTION}, "images": [IMAGE[:120]]})
        expect(status == 422, f"status {status}: {brief(body)}")
        expect(str(body.get("detail", "")).startswith("images[0]: "), f"reply {brief(body)}")


def box_ok(answer: dict) -> bool:
    box = answer.get("box") or {}
    return (list(answer) == ["type", "box", "confidence"]
            and sorted(box) == ["bottom", "left", "right", "top"]
            and all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in box.values()))


def prefix_bidirectional(ctx, server, model: str) -> None:
    """d3:turbo / d3:ultra: what /v1/systemone/models lists is what the build answers."""
    entry = None
    with ctx.case(NAME, "models", model, server) as case:
        data = server.client.get_json("/v1/systemone/models", timeout=30).get("data") or []
        expect(data and isinstance(data[0], dict), f"no model entry: {data!r}"[:200])
        entry = data[0]
        case.data.update(question_types=entry.get("question_types"),
                         assets=entry.get("assets"), long_context=entry.get("long_context"))
    if entry is None:
        return
    listed = entry.get("question_types") or []

    with ctx.case(NAME, "box", model, server):
        # Listed: the engine returns the four hidden rows a box reads. Not
        # listed: refused inline, the rest of the request still answered.
        status, body = post(server, {"state": STATE, "questions": {
            "b": {"type": "box", "instructions": "The customer's name"}}})
        expect(status == 200, f"status {status}: {brief(body)}")
        answer = (body.get("answers") or {}).get("b", {})
        if "box" in listed:
            box = answer.get("box") or {}
            expect(box_ok(answer) and box.get("top", 1) <= box.get("bottom", 0)
                   and box.get("left", 1) <= box.get("right", 0), f"bad box: {brief(answer)}")
        else:
            expect(answer.get("error") == "invalid_question"
                   and answer.get("reason") == "box_unsupported", f"answer {brief(answer)}")

    with ctx.case(NAME, "state-once-usage", model, server) as case:
        # The state block is prefilled once, however many questions share it.
        long_state = "The customer wrote: " + "the parcel is late, again. " * 40
        one = {"q1": {"type": "noul", "instructions": "Is the customer upset?"}}
        four = {f"q{i}": {"type": "noul", "instructions": f"Is the customer upset? ({i})"}
                for i in range(1, 5)}
        s1, b1 = post(server, {"state": long_state, "questions": one})
        s4, b4 = post(server, {"state": long_state, "questions": four})
        expect(s1 == 200 and s4 == 200, f"status {s1} / {s4}")
        n1 = (b1.get("usage") or {}).get("input_tokens", 0)
        n4 = (b4.get("usage") or {}).get("input_tokens", 0)
        case.data.update(one_question=n1, four_questions=n4)
        expect(0 < n1 < n4 < 4 * n1, f"input_tokens 1 question {n1}, 4 questions {n4}")

    if "box" in listed:
        with ctx.case(NAME, "box-image", model, server):
            status, body = post(server, {"state": "A small test picture.", "images": [IMAGE],
                                         "questions": {"b": {"type": "box",
                                                             "instructions": "The yellow rectangle"}}})
            expect(status == 200, f"status {status}: {brief(body)}")
            answer = (body.get("answers") or {}).get("b", {})
            expect(box_ok(answer), f"bad box: {brief(answer)}")
            expect((body.get("usage") or {}).get("input_tokens", 0) > 100, f"usage {body.get('usage')}")

    if entry.get("assets"):
        with ctx.case(NAME, "assets", model, server):
            status, body = post(server, {"state": "Picture: <pic>", "assets": {"<pic>": IMAGE},
                                         "questions": {"bg": BACKGROUND_QUESTION}})
            expect(status == 200, f"status {status}: {brief(body)}")
            answer = (body.get("answers") or {}).get("bg", {})
            expect(answer_shape_ok(answer) and "error" not in answer, f"bad answer: {brief(body)}")
            expect((body.get("usage") or {}).get("input_tokens", 0) > 100, f"usage {body.get('usage')}")

    if "track" in listed:
        with ctx.case(NAME, "track", model, server) as case:
            # A track question needs an encoded video (a frame list is only read
            # by plain questions).
            clip = ctx.repo_root / TRACK_CLIP
            if not clip.is_file():
                skip(f"{TRACK_CLIP} is not in this checkout")
            video = "data:video/mp4;base64," + base64.b64encode(clip.read_bytes()).decode("ascii")
            status, body = post(server, {"state": "A short clip.", "videos": [video],
                                         "questions": {"t": {"type": "track",
                                                             "instructions": "The moving object",
                                                             "rate": 1}}})
            if status == 422 and "decoding backend not available" in json.dumps(body):
                skip("no FFmpeg libraries on this machine")
            expect(status == 200, f"status {status}: {brief(body)}")
            answer = (body.get("answers") or {}).get("t", {})
            moments = answer.get("track") or []
            case.data["moments"] = len(moments)
            expect(answer.get("type") == "track" and moments, f"no track: {brief(answer)}")
            expect(all(isinstance(m, dict) and sorted(m) == ["box", "confidence", "time", "visible"]
                       and 0.0 <= m["visible"] <= 1.0 for m in moments),
                   f"bad track entries: {json.dumps(moments)[:200]}")

    with ctx.case(NAME, "long-context", model, server):
        # Refused exactly when the model entry says long_context is unavailable.
        status, body = post(server, {"state": STATE, "long_context": True,
                                     "questions": {"q": {"type": "noul"}}})
        refused_listed = entry.get("long_context") is False
        expect((status == 422) == refused_listed,
               f"status {status} but long_context listed {entry.get('long_context')}: {brief(body)}")
