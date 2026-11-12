# NPU tests

Tests one FastFlowLM package on an NPU machine, in server mode: it starts
`flm serve` with the flags each suite needs, drives it over HTTP, and writes a
report. Python 3.9+ standard library only, so a runner needs nothing installed.

CI runs it from `.github/workflows/build-and-test.yml` through the
`.github/actions/npu-test` action, once per package (XRT and HRX portable, the
Ubuntu 24.04 .deb, the MSI, the models-from-source builds), and once more for
the Windows XRT zip's GPU models (Test Windows HIPEP).

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
| `--backend oga-hipep` | Test every model of this GPU backend instead of the NPU models |
| `--families a,b` | Families a change touches; the pr tier tests all their models |
| `--asr-full` | The whole ASR evaluation set on the pr tier |
| `--asr-expected-key KEY` | Check WER against this row of `assets/asr_eval/expected.json` |
| `--pull-budget MINUTES` | Time for downloading missing models, all together (default 30 on `install`/`pr`, else no limit; 0 for no limit) |
| `--output-dir DIR` | Where `results.json`, `summary.md` and the server logs go |

Models live in `$FLM_MODEL_PATH`. Before any suite, the run pulls the models
it will use that `flm list --json` reports as not installed (`flm pull`, one
`setup`/`pull` case each, smallest download first; a `--backend` run pulls all
of its models, which also checks their files against the pinned hashes), so a
download counts against no
request's timeout: `flm serve` would otherwise download inside the request
that names the model, and every request behind it would wait out its own
timeout. A model that does not pull is not tested. On the `install` and `pr`
tiers the downloads share a 30-minute budget (`--pull-budget MINUTES`, 0 for no
limit), so a model new to the catalog cannot spend a job's time limit on one
runner; a model the budget leaves undownloaded is a warning and goes untested
on that runner. The `full` tier and `--backend` runs wait for every download.

A request that times out restarts that suite's server, since flm keeps working
on a request its client gave up on, and the model's remaining cases are
dropped. Timeouts count against the server's restart budget (3), so a run of
them ends the suite rather than the job. Exit code 1 means a case failed;
warnings and skips do not fail a run.

## Tiers

| Tier | Runs on | Covers |
|---|---|---|
| `install` | an installed .deb or MSI | one chat model, Whisper on two clips, embeddings, `d3:lite` |
| `pr` | pull requests, pushes, release tags | one model per engine family, every model of a touched family, the feature suites, a subset of the ASR set |
| `full` | nightly | every catalog model, the whole ASR set, every d3 size |

Which models each suite runs at each tier is in `plan.py`, in one place.

The tiers pick NPU models only. A model whose `backend` in `flm list --json` runs
on the GPU (`oga-hipep`, in the Windows XRT package) is tested by a run with
`--backend oga-hipep`, which tests all of them whatever the tier. The Test
Windows HIPEP job runs the `chat`, `api` and `shutdown` suites that way on
`qwen3.5-oga-hipep:9b` alone, since the other packages are 4 to 124 GB each,
unless a manual run names the models (below). Such a run gives `api` the
backend's models in place of its NPU list, so the OpenAI contract tests, tool
calls included, run on the GPU too.

On a pull request, `.github/script/ci/changes.py` also picks the suites: only
those the changed files can affect (an ASR-only change runs `asr`, a d3-only
change `decision`, an engine change `chat` and `prefix_cache`), or all of them
when any changed file is shared code. The HRX packages never run `api` or
`serving`, which test server logic the backend does not touch. Test Windows
HIPEP runs when every suite does, or when the change reaches
`qwen3.5-oga-hipep:9b`: the oga-hipep backend or its catalog entries, the
qwen3.5 frontend, the `chat`, `api` and `shutdown` tests. A change only to the
backend, its entries or the `shutdown` tests touches no NPU family and runs no
NPU test job.

A manual run of Build and Test can name the models to test, for example
`gh workflow run build-and-test.yml --ref <branch> -f models=qwen3.5-oga-hipep:9b`:
every package then runs only those, and one whose catalog lacks them runs
nothing. Add `-f fresh-hip-ep=true` to have Test Windows HIPEP install hip-ep
from scratch, instead of reusing the runner's copy.

## Suites

| Suite | Checks |
|---|---|
| `chat` | Per engine family: a known answer, a streamed two-turn conversation, a non-streamed reply and its usage; for vision models, the colour of a generated test image, and a resent two-turn conversation whose turn-1 image is swapped for a same-size one (the answer must name the new colour; with the original image it must still name the old one) |
| `serving` | Queue full (503), concurrent requests, client disconnect releasing the NPU, cancel, model switching, unknown model, `/v1/models` |
| `prefix_cache` | Multi-slot KV cache (`--prefix-slots 2`): a hit when a conversation returns, LRU eviction, cached and cold replies agree |
| `api` | `src/test/openai_api_test` (tools, logprobs, response_format, reasoning, sampling, continue_final_message, errors) and `src/test/openai_usage` |
| `embedding` | Single and batch vectors, similarity ordering, determinism |
| `decision` | d3 goldens against the upstream model, answer types, errors, the turbo/ultra prefix features |
| `asr` | Whisper WER/CER on the FLEURS set (`assets/asr_eval`), response formats, segments, language, prompt; chat-model transcription (Gemma 4, Nemotron 3.5) |
| `shutdown` | GPU backend runs on Windows only: Ctrl+C ends `flm serve` with its model loaded, and `/bye` and Ctrl+C end `flm run` at its prompt, each within 30 s, with flm in a console of its own as a person runs it (a process that had loaded hip-ep used to hang on its way out, unkillable) |

## Layout

```
run.py       entry point and arguments
plan.py      catalog and per-tier model selection
harness.py   context, cases (pass/fail/warn/skip), report
server.py    one flm serve process
client.py    HTTP: JSON, SSE streams, multipart
console.py   a program in a hidden console of its own: Ctrl+C, typed lines, the screen (Windows)
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
