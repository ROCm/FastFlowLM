"""The test suites, in the order a run executes them.

Each module defines NAME, DESCRIPTION and run(ctx); see harness.py. The fast,
broad suites come first so a broken build shows up within minutes. shutdown
comes last: an flm it cannot end would hold the device for every suite after it.
"""

from . import api, asr, chat, decision, embedding, prefix_cache, serving, shutdown

ORDER = ["chat", "serving", "prefix_cache", "api", "embedding", "decision", "asr", "shutdown"]
SUITES = {m.NAME: m for m in (chat, serving, prefix_cache, api, embedding, decision, asr,
                              shutdown)}
