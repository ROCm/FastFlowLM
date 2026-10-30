"""The model catalog, and which models each suite runs at each tier.

Tiers:
  install  a handful of requests, to show an installed .deb or MSI works
  pr       one model per engine family, plus every model of a family the
           change touches (--families)
  full     every model in the catalog (nightly)

The chat suite covers language models family by family, so there is no
separate sweep of every model on a pull request: one model per engine family
exercises the engine, and the families a change touches run in full.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field

TIERS = ("install", "pr", "full")

# One model per engine family for the pr tier, the smallest that still covers
# the engine. None leaves a family out of the pr tier unless the change
# touches it: those are the large models whose load alone costs minutes, on
# engines no other model shares. A family missing from this table (one added
# to the catalog later) gets its smallest model, so it is never silently
# untested.
PR_CHAT = {
    "hunyuan": "hy-mt2-flash:1.8b",
    "nanbeige": "nanbeige4.1:3b",
    "qwen3vl": "qwen3vl-it:4b",
    "qwen3vl-flash": "qwen3vl-flash:4b",
    "gemma4e": "gemma4-it:e2b",
    "gemma4-12b": "gemma4-it:12b",
    "gemma4e-flash": "gemma4e-flash:e2b",
    "qwen3.5": "qwen3.5:0.8b",
    "minicpm-v-4.7": "minicpm-v-4.7:1b",
    "qwen3.6-moe": None,      # 35 GB
    "qwen3.8-mtp": None,      # 27 GB, ~2 tok/s
    "lfm2": "lfm2:1.2b",
    "lfm3": "lfm3:1.7b",
    "lfm2.5-tk": "lfm2.5-tk:1.2b",
    "phi4": "phi4-mini-it:4b",
    "gemma3-text": "gemma3:1b",
    "gemma3": "gemma3:4b",
    "llama3": "llama3.2:1b",
    "deepseek-r1": None,      # the llama3 engine, already covered
    "deepseek-r1-0528": None, # the qwen3 engine, already covered
    "qwen3": "qwen3:0.6b",
    "qwen3-tk": None,         # the qwen3 engine, already covered
    "qwen3-it": None,         # the qwen3 engine, already covered
    "gpt-oss": None,          # 20 GB
    "qwen2": "qwen2.5-it:3b",
    "qwen2vl": "qwen2.5vl-it:3b",
}

# Families that are not chat models; each has its own suite.
NON_CHAT_FAMILIES = {"embed-gemma", "d3", "whisper-v3"}

# Models that reason even when asked not to. Their answer comes after the
# reasoning, so they need a larger token budget before any content appears.
ALWAYS_THINKS_FAMILIES = {"qwen3-tk", "lfm2.5-tk", "deepseek-r1", "deepseek-r1-0528",
                          "nanbeige", "gpt-oss"}

# Per-suite model lists for the tiers that do not derive from PR_CHAT.
SUITE_MODELS = {
    "api": {
        "install": [],
        "pr": ["qwen3:0.6b", "gemma4-it:e2b"],
        "full": ["qwen3:0.6b", "qwen3:4b", "qwen3-it:4b", "gemma4-it:e2b", "qwen3.5:2b",
                 "llama3.2:3b", "gpt-oss:20b", "lfm2.5-tk:1.2b"],
    },
    "prefix_cache": {
        "install": [],
        "pr": ["qwen3:0.6b", "gemma4-it:e2b"],
        "full": ["qwen3:0.6b", "gemma4-it:e2b", "qwen3.5:0.8b", "llama3.2:1b",
                 "lfm2:1.2b", "phi4-mini-it:4b", "gemma3:1b"],
    },
    "serving": {
        "install": [],
        "pr": ["qwen3:0.6b", "llama3.2:1b"],
        "full": ["qwen3:0.6b", "llama3.2:1b"],
    },
    # Chat models fed audio through /v1/chat/completions. Whisper itself is
    # always tested; these are the chat models with an audio encoder.
    "asr_chat": {
        "install": [],
        # 12b has its own audio front end, separate from the e2b/e4b one.
        "pr": ["gemma4-it:e2b", "gemma4-it:e4b", "nemotron3.5-bolt:e4b", "gemma4-it:12b"],
        "full": None,  # every catalog model with asr=true, Whisper aside
    },
    "embedding": {
        "install": ["embed-gemma:300m"],
        "pr": ["embed-gemma:300m"],
        "full": None,  # every embed-gemma model
    },
    "decision": {
        "install": ["d3:lite"],
        "pr": ["d3:lite"],
        "full": None,  # every d3 model
    },
    "whisper": {
        "install": ["whisper-v3:turbo"],
        "pr": ["whisper-v3:turbo"],
        "full": None,  # every whisper-v3 model
    },
}

INSTALL_CHAT = ["qwen3:0.6b"]


@dataclass
class Model:
    id: str
    family: str = ""
    vlm: bool = False
    asr: bool = False
    think: bool = False
    labels: list = field(default_factory=list)
    size: float = 0.0

    @property
    def single_turn(self) -> bool:
        return "single-turn" in self.labels

    @property
    def always_thinks(self) -> bool:
        return self.family in ALWAYS_THINKS_FAMILIES

    @property
    def translation(self) -> bool:
        # Not lfm2-trans: that is LFM2-2.6B-Transcript, a meeting summarizer.
        return "translation" in self.labels or self.id.startswith("translategemma")


def load_catalog(flm_bin: str) -> dict[str, Model]:
    """The catalog as `flm list --json --filter all` reports it on this machine.

    That is the only listing that carries the vlm/asr flags and families, and it
    already leaves out models this NPU cannot run.
    """
    proc = subprocess.run([flm_bin, "list", "--json", "--filter", "all"],
                          capture_output=True, text=True, timeout=300, check=True)
    start = proc.stdout.find("{")  # tolerate banner lines before the JSON
    if start < 0:
        raise RuntimeError("no JSON in `flm list --json` output")
    models = {}
    for entry in json.loads(proc.stdout[start:]).get("models", []):
        model_id = entry.get("model") or entry.get("name")
        if not model_id:
            continue
        details = entry.get("details") or {}
        models[model_id] = Model(
            id=model_id,
            family=details.get("family", ""),
            vlm=bool(entry.get("vlm")),
            asr=bool(entry.get("asr")),
            think=bool(details.get("think")),
            labels=list(entry.get("label") or []),
            size=float(entry.get("size") or 0),
        )
    return models


class Plan:
    def __init__(self, catalog: dict[str, Model], tier: str, families: list[str],
                 only_models: list[str] | None = None):
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}")
        self.catalog = catalog
        self.tier = tier
        self.families = set(families)
        self.only_models = only_models

    def _known(self, ids) -> list[str]:
        """Keeps catalog models, in order, dropping ones this machine lacks."""
        out = []
        for model_id in ids:
            if model_id in self.catalog and model_id not in out:
                if not self.only_models or model_id in self.only_models:
                    out.append(model_id)
        return out

    def _family(self, family: str) -> list[str]:
        return [m.id for m in self.catalog.values() if m.family == family]

    def _touched(self, predicate) -> list[str]:
        return [m.id for m in self.catalog.values()
                if m.family in self.families and predicate(m)]

    def chat(self) -> list[str]:
        # --models names exactly what to test, whatever the tier would pick.
        if self.only_models:
            return [m for m in self.only_models if m in self.catalog
                    and self.catalog[m].family not in NON_CHAT_FAMILIES]
        if self.tier == "install":
            return self._known(INSTALL_CHAT)
        chat_models = [m for m in self.catalog.values() if m.family not in NON_CHAT_FAMILIES]
        if self.tier == "full":
            return self._known(m.id for m in chat_models)
        picked = []
        for family in sorted({m.family for m in chat_models}):
            if family in PR_CHAT:
                if PR_CHAT[family]:
                    picked.append(PR_CHAT[family])
            else:
                members = sorted((m for m in chat_models if m.family == family),
                                 key=lambda m: (m.size, m.id))
                picked.append(members[0].id)
        picked += self._touched(lambda m: m.family not in NON_CHAT_FAMILIES)
        return self._known(picked)

    def suite(self, name: str) -> list[str]:
        """Models for a suite with a fixed list in SUITE_MODELS."""
        table = SUITE_MODELS[name]
        listed = table[self.tier]
        if listed is None:  # full tier: derive from the catalog
            listed = {
                "asr_chat": [m.id for m in self.catalog.values()
                             if m.asr and m.family != "whisper-v3"],
                "embedding": self._family("embed-gemma"),
                "decision": self._family("d3"),
                "whisper": self._family("whisper-v3"),
            }[name]
        extra = []
        if self.tier == "pr":
            extra = {
                "asr_chat": self._touched(lambda m: m.asr and m.family != "whisper-v3"),
                "decision": self._touched(lambda m: m.family == "d3"),
                "embedding": self._touched(lambda m: m.family == "embed-gemma"),
                "whisper": self._touched(lambda m: m.family == "whisper-v3"),
            }.get(name, [])
        return self._known(list(listed) + extra)

    def model(self, model_id: str) -> Model:
        return self.catalog.get(model_id) or Model(id=model_id)
