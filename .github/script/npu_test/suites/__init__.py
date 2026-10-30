"""The test suites, in the order a run executes them.

Each module defines NAME, DESCRIPTION and run(ctx); see harness.py. The fast,
broad suites come first so a broken build shows up within minutes.
"""

from . import api, asr, chat, decision, embedding, prefix_cache, serving

ORDER = ["chat", "serving", "prefix_cache", "api", "embedding", "decision", "asr"]
SUITES = {m.NAME: m for m in (chat, serving, prefix_cache, api, embedding, decision, asr)}
