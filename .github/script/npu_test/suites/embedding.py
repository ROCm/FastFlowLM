"""The embedding model through /v1/embeddings.

Well-formed vectors are not enough: a model whose vectors carry no meaning
passes the shape checks, so the similarity case asserts that a paraphrase
outscores an unrelated sentence. The server loads the embedding model only
with `-e 1`, and that one model answers whatever "model" a request names.
"""

from __future__ import annotations

import math

from ..harness import FAIL, Result, ServerGone, expect

NAME = "embedding"
DESCRIPTION = "/v1/embeddings: shape, batching, determinism and semantic similarity"

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
MIN_SIMILARITY_MARGIN = 0.05
# The same text through the same weights; anything below this is not noise.
MIN_DETERMINISTIC_COSINE = 0.999


def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed(server, model: str, text_or_list) -> list[list[float]]:
    """The vectors of one request, checked for shape, in the server's order.

    The handler reports its own failures as a 200 with an error body, so a
    missing "data" list is the failure signal, not the status.
    """
    body = server.client.post_json("/v1/embeddings", {"model": model, "input": text_or_list})
    data = body.get("data") if isinstance(body, dict) else None
    expect(isinstance(data, list) and data, f"no embedding data in reply: {str(body)[:200]}")
    vectors = []
    for i, item in enumerate(data):
        vector = item.get("embedding")
        expect(isinstance(vector, list) and vector, f"data[{i}] has no embedding vector")
        expect(all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector),
               f"data[{i}] has a non-finite or non-numeric value")
        expect(item.get("index") == i, f"data[{i}] has index {item.get('index')!r}")
        vectors.append(vector)
    return vectors


def run(ctx) -> None:
    models = ctx.plan.suite(NAME)
    if not models:
        return
    with ctx.server(NAME, ["-e", "1"]) as server:
        for done, model in enumerate(models):
            try:
                check(ctx, server, model)
            except ServerGone as exc:
                ctx.record(Result(NAME, "server-gone", FAIL, model, str(exc)))
                ctx.not_attempted(NAME, models[done + 1:], str(exc))
                return


def check(ctx, server, model: str) -> None:
    with ctx.case(NAME, "single", model, server) as case:
        vectors = embed(server, model, SINGLE_INPUT)
        expect(len(vectors) == 1, f"expected 1 vector, got {len(vectors)}")
        vector = vectors[0]
        norm = math.sqrt(sum(x * x for x in vector))
        expect(norm > 0.0, "zero vector")
        case.data.update(dim=len(vector), norm=round(norm, 4))

    with ctx.case(NAME, "batch", model, server) as case:
        vectors = embed(server, model, BATCH_INPUTS)
        expect(len(vectors) == len(BATCH_INPUTS),
               f"expected {len(BATCH_INPUTS)} vectors, got {len(vectors)}")
        dims = sorted({len(v) for v in vectors})
        expect(len(dims) == 1, f"inconsistent dimensions across batch: {dims}")
        case.data["dim"] = dims[0]

    with ctx.case(NAME, "similarity", model, server) as case:
        vectors = embed(server, model, list(SIMILARITY_TRIPLE))
        expect(len(vectors) == 3, f"expected 3 vectors, got {len(vectors)}")
        near = cosine(vectors[0], vectors[1])
        far = cosine(vectors[0], vectors[2])
        case.data.update(paraphrase=round(near, 4), unrelated=round(far, 4))
        expect(near - far >= MIN_SIMILARITY_MARGIN,
               f"paraphrase {near:.4f} vs unrelated {far:.4f}: margin "
               f"{near - far:.4f} below {MIN_SIMILARITY_MARGIN}")

    with ctx.case(NAME, "deterministic", model, server) as case:
        first = embed(server, model, SINGLE_INPUT)[0]
        second = embed(server, model, SINGLE_INPUT)[0]
        expect(len(first) == len(second),
               f"dimensions differ between runs: {len(first)} vs {len(second)}")
        similarity = cosine(first, second)
        case.data["cosine"] = round(similarity, 6)
        expect(similarity > MIN_DETERMINISTIC_COSINE,
               f"same input twice gave cosine {similarity:.6f}")
