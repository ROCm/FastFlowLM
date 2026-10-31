"""The prefix cache: saved prompts are found again, and evicted by the
documented policy, across interleaved conversations.

Each chat request reports what the cache did in usage.flm_prefix_cache
(lookup hit/miss, the slot it resumed from, what it saved and dropped) and
how many prompt tokens it skipped in usage.prompt_tokens_details.cached_tokens.
The server runs with two slots, so a third conversation must push one out,
and with a --prefix-min-gain larger than any follow-up turn here, so each
conversation keeps exactly one snapshot: that of its first turn. Without that,
a follow-up would replace its own conversation's older snapshot ("superseded")
and the case would no longer tell least-recently-used from first-in-first-out.

A model whose responses carry no flm_prefix_cache has no slot cache and is
skipped; which models do is the engine's business, not a list kept here.
"""

from __future__ import annotations

from ..harness import ServerGone, expect, skip, warn
from ..server import ServerError

NAME = "prefix_cache"
DESCRIPTION = "prefix-slot hits, LRU eviction with 2 slots, cached vs cold reply"

SLOTS = 2
# Far more tokens than a follow-up turn adds here, so follow-ups are never
# saved (see the module docstring).
MIN_GAIN = 1000

GREEDY = {"temperature": 0, "top_k": 1, "top_p": 1.0}
REPLY_TOKENS = 24
THINKING_TOKENS = 512

# Distinct system prompts, so no conversation's snapshot is a prefix of
# another's. All are far below the 512-token minimum for a shared-head
# ("boundary") snapshot, so only end-of-prompt snapshots are saved.
CONVERSATIONS = {
    "A": ("You are a geography tutor. Keep every answer to one short sentence.",
          ["What is the capital of France?", "And the capital of Germany?",
           "And the capital of Spain?"]),
    "B": ("You are a zoology tutor. Keep every answer to one short sentence.",
          ["Name one large mammal.", "Name one bird that cannot fly."]),
    "C": ("You are a cooking tutor. Keep every answer to one short sentence.",
          ["Name one citrus fruit."]),
    "D": ("You are an astronomy tutor. Keep every answer to one short sentence.",
          ["Name the planet closest to the Sun."]),
}


def run(ctx) -> None:
    models = ctx.plan.suite(NAME)
    if not models:
        return
    args = ["--prefix-slots", str(SLOTS), "--prefix-min-gain", str(MIN_GAIN)]
    try:
        with ctx.server(NAME, args) as server:
            for index, model_id in enumerate(models):
                try:
                    _model_cases(ctx, server, ctx.plan.model(model_id))
                except ServerGone as exc:
                    ctx.not_attempted(NAME, models[index + 1:], str(exc))
                    raise
    except ServerError as exc:
        ctx.not_attempted(NAME, models, f"server did not start: {exc}")
        raise


class _Conversation:
    """One conversation's history, and the end snapshot the cache holds for it."""

    def __init__(self, name: str, system: str, questions: list):
        self.name = name
        self.messages = [{"role": "system", "content": system}]
        self.questions = list(questions)
        self.asked: list = []   # the messages of each request sent, in order
        self.replies: list = []
        self.slot = None        # id of its newest end snapshot


class _Session:
    def __init__(self, server, model):
        self.server = server
        self.model = model
        self.conversations = {k: _Conversation(k, *v) for k, v in CONVERSATIONS.items()}
        self.boundary_seen = False
        self.trace: list = []

    def ask(self, name: str, stream: bool = False, probe: bool = False):
        """Sends the next question of a conversation; returns (reply, report).

        probe: the model's first request, which skips the model when the
        engine reports no prefix cache at all.
        """
        conv = self.conversations[name]
        messages = conv.messages + [{"role": "user", "content": conv.questions[len(conv.asked)]}]
        reply = self.send(messages, stream)
        conv.asked.append(messages)
        conv.replies.append(reply)
        conv.messages = messages + [{"role": "assistant",
                                     "content": reply.content or reply.reasoning[:300]}]
        report = self.report(reply, f"{name}{len(conv.asked)}", probe)
        for saved in report.get("saved") or []:
            if saved.get("slot_type") == "end":
                conv.slot = saved.get("slot")
        return reply, report

    def send(self, messages: list, stream: bool = False):
        # think=False and greedy on every request: FLM keeps both on the
        # loaded model, and the think setting changes the rendered prompt.
        budget = THINKING_TOKENS if self.model.always_thinks else REPLY_TOKENS
        return self.server.client.chat(self.model.id, messages, stream=stream, think=False,
                                       max_tokens=budget, **GREEDY)

    def report(self, reply, label: str, probe: bool = False) -> dict:
        report = reply.usage.get("flm_prefix_cache")
        if probe and not isinstance(report, dict):
            skip("no usage.flm_prefix_cache: this model's engine has no prefix-slot cache")
        expect(isinstance(report, dict), f"{label}: usage has no flm_prefix_cache")
        if any(s.get("slot_type") == "boundary" for s in report.get("saved") or []):
            self.boundary_seen = True
        self.trace.append({
            "request": label, "lookup": report.get("lookup"), "slot": report.get("slot"),
            "matched": report.get("matched_tokens"), "miss_reason": report.get("miss_reason"),
            "saved": [s.get("slot") for s in report.get("saved") or []],
            "dropped": report.get("dropped"), "slots_used": report.get("slots_used"),
            "cached_tokens": reply.cached_tokens, "prompt_tokens": reply.usage.get("prompt_tokens"),
        })
        _check_usage(reply, report, label)
        return report


def _check_usage(reply, report: dict, label: str) -> None:
    """cached_tokens and the cache's own report must tell the same story."""
    cached, prompt = reply.cached_tokens, reply.usage.get("prompt_tokens")
    expect(report.get("lookup") in ("hit", "miss"), f"{label}: lookup is {report.get('lookup')!r}")
    expect(isinstance(report.get("slots_used"), int) and report["slots_used"] <= SLOTS,
           f"{label}: slots_used {report.get('slots_used')!r} with --prefix-slots {SLOTS}")
    if report["lookup"] == "hit":
        matched = report.get("matched_tokens")
        expect(isinstance(matched, int) and matched > 0, f"{label}: hit with matched_tokens {matched!r}")
        expect(cached == matched, f"{label}: cached_tokens {cached!r} != matched_tokens {matched}")
        expect(isinstance(prompt, int) and cached < prompt,
               f"{label}: cached_tokens {cached} not below prompt_tokens {prompt!r}")
    else:
        expect(report.get("miss_reason"), f"{label}: miss without a miss_reason")
        expect(cached == 0, f"{label}: miss but cached_tokens is {cached!r}")


def _model_cases(ctx, server, model) -> None:
    session = _Session(server, model)
    later = ("hit-after-switch", "lru-eviction", "cached-matches-cold")

    with ctx.case(NAME, "cold-miss", model.id, server) as case:
        if model.single_turn:
            skip("single-turn model: no conversation to resume")
        reports = {"A1": session.ask("A", probe=True)[1], "B1": session.ask("B")[1]}
        case.data["trace"] = session.trace
        for label, report in reports.items():
            expect(report["lookup"] == "miss", f"{label}: {report['lookup']} on a new conversation")
            expect(report.get("slots_max") == SLOTS,
                   f"{label}: slots_max {report.get('slots_max')!r}, started with --prefix-slots {SLOTS}")
            expect(session.conversations[label[0]].slot is not None,
                   f"{label} saved no end snapshot (saved: {report.get('saved')})")
        expect(reports["B1"].get("slots_used") == 2,
               f"B1: slots_used {reports['B1'].get('slots_used')}, expected 2")
    if not case.ok:
        reason = (ctx.results[-1].detail if case.status == "skip" else "needs cold-miss to pass")
        for name in later:
            with ctx.case(NAME, name, model.id, server):
                skip(reason)
        return

    a = session.conversations["A"]
    with ctx.case(NAME, "hit-after-switch", model.id, server) as case:
        a_slot = a.slot
        reply, report = session.ask("A")
        case.data["trace"] = session.trace[-1:]
        expect(report["lookup"] == "hit",
               f"A2 after B1: {report['lookup']} ({report.get('miss_reason')}), expected a hit on A1's snapshot")
        expect(report.get("slot") == a_slot,
               f"A2 resumed from slot {report.get('slot')}, A1's snapshot is slot {a_slot}")
        expect(report.get("slot_type") == "end", f"A2 resumed from a {report.get('slot_type')} slot")
        expect(reply.content.strip() or reply.reasoning.strip(), "A2: empty reply")
    hit_ok = case.ok

    with ctx.case(NAME, "lru-eviction", model.id, server) as case:
        if not hit_ok:
            skip("needs hit-after-switch to pass")
        if session.boundary_seen:
            skip("the model saved a shared-head snapshot; this case assumes end snapshots only")
        # Slots now: A1 (used by A2) and B1 (saved after A1, unused since).
        # C1 needs room: least recently used is B1, though A1 is older.
        a_slot, b_slot = a.slot, session.conversations["B"].slot
        _, c1 = session.ask("C")
        expect(c1["lookup"] == "miss", f"C1: {c1['lookup']} on a new conversation")
        expect(c1.get("dropped") == 1, f"C1 dropped {c1.get('dropped')} snapshots, expected 1")
        # Streamed, so the streamed usage block is checked as well.
        _, a3 = session.ask("A", stream=True)
        _, b2 = session.ask("B")
        case.data["trace"] = session.trace[-3:]
        expect(a3["lookup"] == "hit" and a3.get("slot") == a_slot,
               f"A3: {a3['lookup']} slot {a3.get('slot')} ({a3.get('miss_reason')}); A was used "
               f"more recently than B, so A's slot {a_slot} should have survived C1")
        expect(not (b2["lookup"] == "hit" and b2.get("slot") == b_slot),
               f"B2 resumed from B1's slot {b_slot}, which C1 should have evicted as least recently used")
        expect(b2["lookup"] == "miss", f"B2: hit on slot {b2.get('slot')}, expected a miss")

    with ctx.case(NAME, "cached-matches-cold", model.id, server) as case:
        if not hit_ok:
            skip("needs hit-after-switch to pass")
        # D1 pushes out A's snapshot (least recently used again), so replaying
        # A2 prefills it from scratch.
        session.ask("D")
        cold = session.send(a.asked[1])
        cold_report = session.report(cold, "A2-replay")
        cached = a.replies[1]
        case.data.update(trace=session.trace[-2:], cached=cached.content[:200], cold=cold.content[:200])
        expect(cold.content.strip() or cold.reasoning.strip(), "A2 replay: empty reply")
        if cold_report["lookup"] != "miss":
            warn(f"the A2 replay resumed from slot {cold_report.get('slot')}; no cold run to compare")
        if cold.content.strip() != cached.content.strip():
            # docs/.../prefix_cache.md: resuming from a snapshot is not
            # bit-identical to a cold prefill; about 1 in 5 greedy answers
            # differ in wording. Worth a look, not a failure.
            warn(f"cached reply {cached.content[:80]!r} != cold reply {cold.content[:80]!r}")

