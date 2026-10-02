#!/usr/bin/env python3
"""Sweep the embedding models through the /v1/embeddings endpoint.

Covers three things per model: a single-string request, a batched request,
and a semantic sanity check. The sanity check embeds a sentence alongside a
paraphrase and an unrelated sentence, then asserts the paraphrase scores
higher. A model that returns well-formed vectors carrying no meaning passes
the first two checks and fails this one.

Embedding models are absent from /v1/models, so the server needs `-e 1` and
the model set comes from `flm list`.
"""

from __future__ import annotations

import math
import sys
import time

from sweep_common import SweepTask, run_task

SINGLE_INPUT = "Hi, everyone!"

BATCH_INPUTS = [
    "The NPU accelerates matrix multiplication.",
    "Quantized weights reduce memory bandwidth.",
    "She poured the tea into a chipped porcelain cup.",
]

# (anchor, paraphrase, unrelated)
SIMILARITY_TRIPLE = (
    "A man is playing a guitar on the street.",
    "A busker performs guitar music on the sidewalk.",
    "The quarterly earnings report exceeded analyst expectations.",
)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingSweep(SweepTask):
    name = "embedding"
    needs_embed = True
    csv_header = [
        "Model", "Case", "Input", "Dimensions", "L2 Norm",
        "Detail", "Latency (s)", "Status", "Error",
    ]

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument(
            "--min-similarity-margin",
            type=float,
            default=0.05,
            help="How far the paraphrase must outscore the unrelated sentence.",
        )

    def select_models(self, catalog: list[dict]) -> list[str]:
        models = [entry["id"] for entry in catalog if entry["family"] == "embed-gemma"]
        if not models:
            print(
                "[embedding] No embedding models found. The /v1/models fallback "
                "hides them, so pass --models explicitly if 'flm list' failed."
            )
        return models

    def _embed(self, model: str, text_or_list):
        response = self.client.embeddings.create(model=model, input=text_or_list)
        if not response.data:
            raise RuntimeError("server returned no embedding data")
        return [item.embedding for item in response.data]

    def _case_single(self, writer, model: str) -> None:
        print(f"  single input: {SINGLE_INPUT!r}")
        started = time.monotonic()
        try:
            vectors = self._embed(model, SINGLE_INPUT)
        except Exception as exc:
            self._fail(writer, model, "single", SINGLE_INPUT, started, exc)
            return

        elapsed = f"{time.monotonic() - started:.2f}"
        vector = vectors[0]
        norm = math.sqrt(sum(x * x for x in vector))
        preview = ", ".join(f"{x:.4f}" for x in vector[:8])
        self.record(
            writer,
            [model, "single", SINGLE_INPUT, len(vector), f"{norm:.4f}",
             f"first 8: [{preview}]", elapsed, "ok", ""],
            model=model,
        )
        print(f"    {len(vector)} dims, norm {norm:.4f}, {elapsed}s")

    def _case_batch(self, writer, model: str) -> None:
        label = f"{len(BATCH_INPUTS)} inputs"
        print(f"  batch: {label}")
        started = time.monotonic()
        try:
            vectors = self._embed(model, BATCH_INPUTS)
        except Exception as exc:
            self._fail(writer, model, "batch", label, started, exc)
            return

        elapsed = f"{time.monotonic() - started:.2f}"
        if len(vectors) != len(BATCH_INPUTS):
            error = f"expected {len(BATCH_INPUTS)} vectors, got {len(vectors)}"
            self.record(
                writer,
                [model, "batch", label, "", "", "", elapsed, "error", error],
                model=model,
                error=f"batch: {error}",
            )
            print(f"    ERROR: {error}")
            return

        dims = {len(v) for v in vectors}
        if len(dims) != 1:
            error = f"inconsistent dimensions across batch: {sorted(dims)}"
            self.record(
                writer,
                [model, "batch", label, "", "", "", elapsed, "error", error],
                model=model,
                error=f"batch: {error}",
            )
            print(f"    ERROR: {error}")
            return

        dim = dims.pop()
        self.record(
            writer,
            [model, "batch", label, dim, "",
             f"{len(vectors)} vectors returned", elapsed, "ok", ""],
            model=model,
        )
        print(f"    {len(vectors)} vectors, {dim} dims, {elapsed}s")

    def _case_similarity(self, writer, model: str) -> None:
        anchor, paraphrase, unrelated = SIMILARITY_TRIPLE
        print("  similarity: paraphrase should outscore unrelated")
        started = time.monotonic()
        try:
            vectors = self._embed(model, list(SIMILARITY_TRIPLE))
        except Exception as exc:
            self._fail(writer, model, "similarity", anchor, started, exc)
            return

        elapsed = f"{time.monotonic() - started:.2f}"
        if len(vectors) != 3:
            error = f"expected 3 vectors, got {len(vectors)}"
            self.record(
                writer,
                [model, "similarity", anchor, "", "", "", elapsed, "error", error],
                model=model,
                error=f"similarity: {error}",
            )
            print(f"    ERROR: {error}")
            return

        near = cosine(vectors[0], vectors[1])
        far = cosine(vectors[0], vectors[2])
        margin = near - far
        detail = f"paraphrase {near:.4f} vs unrelated {far:.4f}, margin {margin:.4f}"
        passed = margin >= self.args.min_similarity_margin

        self.record(
            writer,
            [model, "similarity", anchor, len(vectors[0]), "",
             detail, elapsed, "ok" if passed else "error",
             "" if passed else f"margin below {self.args.min_similarity_margin}"],
            model=model,
            error=None if passed else f"similarity margin {margin:.4f} too small",
        )
        print(f"    {detail} -> {'ok' if passed else 'FAILED'}")

    def _fail(self, writer, model: str, case: str, label: str,
              started: float, exc: Exception) -> None:
        elapsed = f"{time.monotonic() - started:.2f}"
        print(f"    ERROR: {exc}")
        self.record(
            writer,
            [model, case, label, "", "", "", elapsed, "error", str(exc)],
            model=model,
            error=f"{case}: {exc}",
        )

    def run(self) -> None:
        models = self.resolve_models()
        handle, writer = self.open_csv()
        try:
            for model in self.iter_models(models):
                print(f"\n--- Embedding: {model} ---")
                self._case_single(writer, model)
                self._case_batch(writer, model)
                self._case_similarity(writer, model)
                time.sleep(1)
        finally:
            handle.close()
        print(f"\nEmbedding sweep complete. Saved to {self.csv_path}")
        self.write_summary(models)


if __name__ == "__main__":
    sys.exit(run_task(EmbeddingSweep))
