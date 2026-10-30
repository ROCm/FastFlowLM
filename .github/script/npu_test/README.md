# NPU tests

Tests one FastFlowLM package on an NPU machine, in server mode: it starts
`flm serve` with the flags each suite needs, drives it over HTTP, and writes a
report. Python 3.9+ standard library only, so a runner needs nothing installed.

CI runs it from `.github/workflows/build-and-test.yml` through the
`.github/actions/npu-test` action, once per package (XRT and HRX portable, the
Ubuntu 24.04 .deb, the MSI, the models-from-source builds).

## Running it

From `.github/script`:

```bash
python3 -m npu_test --flm-bin /path/to/flm --label my-machine --tier pr
python3 -m npu_test --flm-bin flm --label dev --suites chat --models qwen3:0.6b,llama3.2:1b
```

| Option | Meaning |
|---|---|
| `--tier install\|pr\|full` | How much to run; see below |
| `--suites a,b` | Only these suites (default: all) |
| `--models a,b` | Restrict every suite to these model ids |
| `--families a,b` | Families a change touches; the pr tier tests all their models |
| `--asr-full` | The whole ASR evaluation set on the pr tier |
| `--asr-expected-key KEY` | Check WER against this row of `assets/asr_eval/expected.json` |
| `--output-dir DIR` | Where `results.json`, `summary.md` and the server logs go |

Models download on first use into `$FLM_MODEL_PATH`. Exit code 1 means a case
failed; warnings and skips do not fail a run.

## Tiers

| Tier | Runs on | Covers |
|---|---|---|
| `install` | an installed .deb or MSI | one chat model, Whisper on two clips, embeddings, `d3:lite` |
| `pr` | pull requests, pushes, release tags | one model per engine family, every model of a touched family, the feature suites, a subset of the ASR set |
| `full` | nightly | every catalog model, the whole ASR set, every d3 size |

Which models each suite runs at each tier is in `plan.py`, in one place.

On a pull request, `.github/script/ci/changes.py` also picks the suites: only
those the changed files can affect (an ASR-only change runs `asr`, a d3-only
change `decision`, an engine change `chat` and `prefix_cache`), or all of them
when any changed file is shared code. The HRX packages never run `api` or
`serving`, which test server logic the backend does not touch.

## Suites

| Suite | Checks |
|---|---|
| `chat` | Per engine family: a known answer, a streamed two-turn conversation, a non-streamed reply and its usage; for vision models, the colour of a generated test image |
| `serving` | Queue full (503), concurrent requests, client disconnect releasing the NPU, cancel, model switching, unknown model, `/v1/models` |
| `prefix_cache` | Multi-slot KV cache (`--prefix-slots 2`): a hit when a conversation returns, LRU eviction, cached and cold replies agree |
| `api` | `src/test/openai_api_test` (tools, logprobs, response_format, reasoning, sampling, continue_final_message, errors) and `src/test/openai_usage` |
| `embedding` | Single and batch vectors, similarity ordering, determinism |
| `decision` | d3 goldens against the upstream model, answer types, errors, the turbo/ultra prefix features |
| `asr` | Whisper WER/CER on the FLEURS set (`assets/asr_eval`), response formats, segments, language, prompt; chat-model transcription (Gemma 4, Nemotron 3.5) |

## Layout

```
run.py       entry point and arguments
plan.py      catalog and per-tier model selection
harness.py   context, cases (pass/fail/warn/skip), report
server.py    one flm serve process
client.py    HTTP: JSON, SSE streams, multipart
suites/      one module per suite: NAME, DESCRIPTION, run(ctx)
assets/      images, audio, the ASR evaluation set, d3 goldens
tools/       scripts that regenerate assets
```

## Adding a suite

Write `suites/<name>.py` with `NAME`, `DESCRIPTION` and `run(ctx)`, record each
check with `with ctx.case(NAME, "<case>", model, server):`, and add it to
`suites/__init__.py`. Use `expect()` for what a correct server always does,
`warn()` where the ground truth is uncertain, and `skip()` where a check does
not apply to a model.
