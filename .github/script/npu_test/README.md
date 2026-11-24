# NPU tests

Tests one FastFlowLM package on an NPU machine, in server mode: it starts
`flm serve` with the flags each suite needs, drives it over HTTP, and writes a
report. Python 3.9+ standard library only, so a runner needs nothing installed.

CI runs it from `.github/workflows/build-and-test.yml` through the
`.github/actions/npu-test` action, once per package (XRT and HRX portable, the
Ubuntu 24.04 .deb, the MSI, the models-from-source builds), and once per kind
of GPU machine for the Windows XRT zip's GPU models (Test Windows HIPEP; see
[HIPEP machines](#hipep-machines)).

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
| `--backend oga-hipep` | Test this GPU backend's models instead of the NPU models: one per family, every one on the `full` tier |
| `--max-model-gb GB` | Skip the models larger than this (catalog size), which the runner's memory cannot hold, `--required-models` too (default 0, no limit) |
| `--families a,b` | Families a change touches; the pr tier tests all their models (with `--backend`, the model of each family it tests only when touched) |
| `--required-models a,b` | Models a change adds (with `--backend`, adds, republishes or moves onto it): tested on every tier (even past `--models`), downloaded whatever their size, and a failure if the package does not list them or they do not download |
| `--required-pull-budget MINUTES` | Time for downloading the `--required-models`, all together (default 420) |
| `--asr-full` | The whole ASR evaluation set on the pr tier |
| `--asr-expected-key KEY` | Check WER against this row of `assets/asr_eval/expected.json` |
| `--pull-budget MINUTES` | Time for downloading missing models, all together (default 30 on `install`/`pr`, else no limit; 0 for no limit) |
| `--output-dir DIR` | Where `results.json`, `summary.md` and the server logs go |

Models live in `$FLM_MODEL_PATH`. Before any suite, the run pulls the models
it will use that `flm list --json` reports as not installed (`flm pull`, one
`setup`/`pull` case each, smallest download first), so a download counts
against no request's timeout: `flm serve` would otherwise download inside the
request that names the model, and every request behind it would wait out its
own timeout. A model that does not pull is not tested. On the `install` and
`pr` tiers the downloads share a 30-minute budget (`--pull-budget MINUTES`, 0
for no limit), so a model new to the catalog cannot spend a job's time limit
on one runner; a model the budget leaves undownloaded is a warning and goes
untested on that runner. The `full` tier and `--backend` runs wait for every
download.
The models a change adds or republishes (`--required-models`, from
`changes.py`'s `new_models` in CI; for Test Windows HIPEP its `hipep_models`,
the oga-hipep entries with weights no runner has yet) are pulled first and outside
that budget, with 7 hours for all of them (`--required-pull-budget MINUTES`),
so a new model is tested however large it is; one left undownloaded is a
failure. Build and Test gives its test jobs 600 minutes when a change adds
one. A required model larger than `--max-model-gb` is a `setup`/`size` skip,
not a failure: a HIPEP machine without that limit tests it.

A `--backend` run pulls every model it tests, which also checks their files
against the pinned hashes: minutes per oga-hipep package, 14 for the 124 GB
one. So once a pull succeeds, `model_cache.py` stamps the model in
`$FLM_MODEL_PATH/npu-test-verified`, with its pins in the catalog (Hugging Face
revision, files, file sources) and the size and modification time of each of
its files. A later run does not pull a model while `flm list` calls it
installed and neither its pins nor its files changed; its `setup`/`pull` case
says so. A new revision in the catalog, or a file changed on the runner, pulls
it again, which downloads what changed and verifies the rest. An entry pinned
to a branch rather than a revision is pulled every time. Delete a stamp, or
the folder, to have the next run verify again. A required model is pulled
whatever its stamp says: the change adds or republishes it.

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

The tiers above pick NPU models. A model whose `backend` in `flm list --json`
runs on the GPU (`oga-hipep`, in the Windows XRT package) is tested by a run
with `--backend oga-hipep`, and the Test Windows HIPEP job runs the `chat`,
`api` and `shutdown` suites that way. On the `install` and `pr` tiers it tests
one model per family (`PR_GPU` in `plan.py`): a family's models share its
frontend and features, and differ only in weights. That is the newest Qwen
(`qwen3.8-oga-hipep:27b`, for the Qwen3.5, 3.6 and 3.8 models) and Gemma 4
(`gemma4-it-oga-hipep:26b-a4b`), about 33 GB; Qwen2.5, Gemma 3, the Gemma 4
12B on its own frontend (`gemma4-12b`), gpt-oss (69 GB) and the families on
the TemplateChat frontend (`laguna`, `nemotron-h`; 18 GB each)
run there only when the change touches their family (`--families`).
An oga-hipep entry the change edits without new weights (a capability, a
context length) counts its family as touched (`changes.py`'s
`hipep_families`, passed as `--families`), so that family's one model runs.
The run also tests the models with new weights: added, republished or moved
onto the GPU (`changes.py`'s `hipep_models`, passed as `--required-models`), on
the kinds of machine that test them (see [HIPEP machines](#hipep-machines)).
The `full` tier tests every oga-hipep model (4 to 124 GB each, about 500 GB
in all).
Models larger than the machine holds are skipped either way, and a manual run
that names the models (below) tests only those. Such a run gives `api` the
backend's models in place of its NPU list, so the OpenAI contract tests, tool
calls included, run on the GPU too.

On a pull request, `.github/script/ci/changes.py` also picks the suites: only
those the changed files can affect (an ASR-only change runs `asr`, a d3-only
change `decision`, an engine change `chat` and `prefix_cache`), or all of them
when any changed file is shared code. The HRX packages never run `api` or
`serving`, which test server logic the backend does not touch. Test Windows
HIPEP runs when every suite does, or when the change reaches an oga-hipep
model: the oga-hipep backend or its catalog entries, the frontend of a family
with an oga-hipep entry (qwen3.5, qwen2, gemma3, gemma4e, gemma4-12b, gpt-oss,
laguna, nemotron-h today, read from the catalog), the `chat`, `api` and `shutdown` tests. A change only to
the backend, its entries or the `shutdown` tests touches no NPU family and
runs no NPU test job.

A manual run of Build and Test can name the models to test, for example
`gh workflow run build-and-test.yml --ref <branch> -f models=qwen3.5-oga-hipep:9b`:
every package then runs only those, and one whose catalog lacks them runs
nothing. Add `-f fresh-hip-ep=true` to have Test Windows HIPEP install hip-ep
from scratch, instead of reusing the runner's copy.

## HIPEP machines

Test Windows HIPEP runs one job per kind of machine listed in
`.github/hipep-machines.json`, each reporting as `windows-hipep-<id>`;
`.github/script/ci/hipep_machines.py` checks the file and turns it into the
job matrix. The job first prints the runner's GPUs, its memory, and the
memory dedicated to each GPU, which Windows leaves out of its own figure
(96 GB of an STX Halo's 128 GB). Two kinds today:

- `stx-halo`: an STX Halo of its own (xconucstrhalo86), labelled `oga-hipep`
  and `stx-halo`, which no NPU job asks for, so it runs beside the NPU tests.
  It tests every model nightly. It is on `standby` below the `full` tier, so
  that one machine does not test every pull request: it runs only when
  `npu-runners` cannot test the change, that is when the change gives models
  new weights (added, republished or moved onto the GPU), which it tests as
  required models with 10 hours instead of 2 (their new revisions download
  within the job, 7 hours for all of them, a failure if one does not, and a
  change may add or re-pin many); when a model the run tests is larger than
  `npu-runners` holds (gpt-oss, for a change to its family); or when
  `npu-runners` does not run (`FLM_HIPEP_MACHINES`). The changes job prints
  why it runs, or that it is not needed.
- `npu-runners`: the shared NPU test runners (`npu`, `sweep`: STX, Krackan and
  STX Halo machines) that also carry `oga-hipep`, whichever is free. It holds
  that runner for its run, as an NPU test job does. It tests the `pr` tier's
  models even nightly (`max-tier`), since each runner would otherwise
  download all 500 GB and the NPU nightly needs them, and skips the models
  above 32 GB (`max-model-gb`): gpt-oss and the 120B-class Qwen and Laguna
  models. It
  leaves the changed entries' models to `stx-halo` (`changed-models`), so a
  change to many of them does not hold an NPU runner for hours. When
  `FLM_HIPEP_MACHINES` leaves `stx-halo` out, it tests them itself, those up
  to 32 GB, with a warning in the changes job, so they are not silently
  untested.

GitHub assigns a job by labels alone, without looking at a runner's memory or
drivers, so a repository admin adds `oga-hipep` (Settings > Actions >
Runners, then the runner's labels) only to the shared runners that can run
the `pr` tier's models (about 17 GB each, loaded one at a time; 33 GB in
all): at least 32 GB the GPU can use, which is the
memory dedicated to it plus half the memory Windows reports (64 GB on a
machine that dedicates little; the job prints both), a current NPU driver,
and room for those models in `FLM_MODEL_PATH`. NPU jobs do not ask for the
label, so they still use every runner. Never add `stx-halo` to a shared
runner: the nightly's every-model job would then land on it.

| Field | Meaning |
|---|---|
| `id` | Short name, in the job's name and its report's label |
| `runs-on` | The labels its runners register with; several runners may share them, and a run takes whichever is free. The NPU test runners' own (`npu`, `sweep`) share those runners, and a label only some of them carry (`oga-hipep`) picks those |
| `enabled` | `false` leaves the kind out of every run |
| `max-model-gb` | The largest model, by catalog size, its memory holds; larger ones are skipped there, each a `setup`/`size` case (0: no limit) |
| `max-tier` | The highest tier its job runs (`install`, `pr` or `full`, the default); a run of a higher tier runs this one there |
| `changed-models` | `false` leaves the models the change gives new weights (added, republished or moved onto the GPU) to the kinds that test them (`true`, the default), whose job requires them (`--required-models`) and gets 600 minutes; when no kind that runs is `true`, every kind that runs tests them |
| `standby` | `true`: below the `full` tier the kind runs only when the other kinds that run cannot test the change: none of them runs, the change has models it takes (`changed-models`), or a model the run tests (one per family the change touches, as `PR_GPU` picks it, or those a dispatch names) is larger than every one of their `max-model-gb`. `false` (the default): it runs on every tier |

A runner of a kind of its own (xconucstrhalo86 for `stx-halo`):

1. Windows 11 with the NPU driver (the package's `flm.exe` loads XRT from it),
   an AMD GPU driver hip-ep supports, and Python 3.9 or later on `PATH`.
2. A GitHub Actions runner registered for the repository with the labels
   `self-hosted`, `windows`, `x64`, `oga-hipep` and the kind's label
   (`stx-halo`), and neither `npu` nor `sweep`, or NPU test jobs would take it.
3. `FLM_MODEL_PATH`, in the runner's `.env`, on a disk with room for every
   model it tests (about 500 GB for all of them, which the nightly tests, and
   room for a new revision); otherwise they go to the machine's own
   `FLM_MODEL_PATH`, which installing flm may have set, or else to
   `%USERPROFILE%\.flm`.
4. Optionally, the models pulled once by hand (`flm pull <model>`): a CI pull
   gets 90 minutes, and the 124 GB model needs about 23 MB/s for that. An
   interrupted pull resumes on the next run. Either way each model is verified
   once more, by the first run's pull, before it is stamped.

Another kind of its own is another entry with its own label and the largest
model its memory holds, its runners registered with that label. Another
runner of a kind already listed needs no change.

A job no runner takes waits a day before it fails, and the run's verdict
waits with it. While a kind has no runner up (`stx-halo` until xconucstrhalo86
is registered, `npu-runners` until a shared runner carries `oga-hipep`), set
the repository variable `FLM_HIPEP_MACHINES` to the kinds to run instead of
the enabled ones, or to `none`; unset, the enabled ones run. While only
`npu-runners` runs, no job tests the changed entries' models.

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
run.py          entry point and arguments
plan.py         catalog and per-tier model selection
model_cache.py  stamps of the GPU models a pull verified, so later runs skip the pull
harness.py      context, cases (pass/fail/warn/skip), report
server.py       one flm serve process
client.py       HTTP: JSON, SSE streams, multipart
console.py      a program in a hidden console of its own: Ctrl+C, typed lines, the screen (Windows)
suites/         one module per suite: NAME, DESCRIPTION, run(ctx)
assets/         images, audio, the ASR evaluation set, d3 goldens
tools/          scripts that regenerate assets
```

## Adding a suite

Write `suites/<name>.py` with `NAME`, `DESCRIPTION` and `run(ctx)`, record each
check with `with ctx.case(NAME, "<case>", model, server):`, and add it to
`suites/__init__.py`. Use `expect()` for what a correct server always does,
`warn()` where the ground truth is uncertain, and `skip()` where a check does
not apply to a model.
