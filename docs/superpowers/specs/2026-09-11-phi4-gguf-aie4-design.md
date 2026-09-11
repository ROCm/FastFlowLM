# Phi-4 Q8_0 GGUF on AIE4 Design

## Summary

Add one catalog model, `phi4-mini-it-aie4:4b`, that FastFlowLM can pull and run through `ryzenai-corelib` on AIE4. The implementation starts from FastFlowLM `main`, supports only the validated Phi-4 Mini Instruct Q8_0 GGUF, and uses corelib's explicit lossy Q8_0-to-group-64 requantization APIs.

The model is read directly from GGUF. FastFlowLM will not generate or ship an ONNX initializer manifest, an ONNX model, or converted weight files. The Phi-4 architecture and tensor-name contract live in a model-specific C++ adapter, while shape, dtype, offset, and model metadata come from the GGUF file.

This PR produces an AIE4-enabled developer build of the normal `flm.exe`. It does not package the AIE4 runtime in MSI, WiX, or Inno Setup.

## Fixed inputs

### Model

- FLM tag: `phi4-mini-it-aie4:4b`
- GGUF repository: `unsloth/Phi-4-mini-instruct-GGUF`
- GGUF revision: `78eb92a46fc37e6b524df991ed9aca9bc6aa7b80`
- GGUF file: `Phi-4-mini-instruct.Q8_0.gguf`
- Supported quantization: GGML `Q8_0` only

### Tokenizer and configuration

- Repository: `microsoft/Phi-4-mini-instruct`
- Revision: `cfbefacb99257ffa30c83adab238a50856ac3083`
- Files: `tokenizer.json`, `tokenizer_config.json`, and `config.json`

The tokenizer files come from a second repository because the selected Unsloth repository does not publish the files FastFlowLM's existing tokenizer frontend consumes. Model loading cross-checks the tokenizer/config contract against GGUF metadata rather than assuming the two fixed sources agree.

### Corelib

- Repository: `VitisAI/ryzenai-corelib`
- Commit: `3c35aebdefa3f0c2255668bab1be5648ece320f8`
- ABI version: `0.3.0`

Because corelib remains pre-1.0, FastFlowLM requires an exact runtime version match: major, minor, and patch must all be `0.3.0`.

## Goals

1. `flm pull phi4-mini-it-aie4:4b` downloads the pinned GGUF and the pinned tokenizer/config files.
2. An AIE4-enabled `flm.exe` runs the model through corelib from the CLI and REST APIs.
3. The existing Phi-4 NPU2/Q4NX model continues to use its existing backend.
4. A build without AIE4 support, or an AIE4 build with no corelib DLL, still starts and runs non-AIE4 models.
5. The implementation validates the model and tokenizer contracts before creating device state.
6. No silent CPU or NPU2 fallback is possible for the AIE4 tag.
7. The completed implementation is exercised on real AIE4 hardware before the PR is considered complete.

## Non-goals

This PR does not add:

- ONNX model loading for AIE4;
- a JSON tensor manifest or manifest generator;
- a generic GGUF runtime or arbitrary local GGUF support;
- Q4_0, Q4_K, Q6_K, or mixed-quantization support;
- another model family;
- a packed-weight disk cache;
- parallel Q8_0 weight creation;
- Python as a runtime dependency;
- automatic corelib/runtime download;
- MSI, WiX, or Inno Setup packaging;
- a new corelib API;
- CPU or NPU2 fallback for the AIE4 model.

## Architecture

```text
flm.exe
  └── Phi4 frontend
      ├── existing chat, tokenizer, sampling, and server integration
      └── phi4_corelib_aie4
          ├── phi4_corelib_gguf       GGUF v3 parsing and Phi-4 tensor mapping
          ├── phi4_corelib_shape_plan corelib padding and buffer extents
          ├── corelib_runtime         DLL lifetime, version, and availability
          └── corelib_api             dynamically resolved C ABI
                                      │
                                      └── ryzenai_corelib.dll → AIE4
```

### Backend selection

The new catalog entry sets `details.execution_backend` to `corelib_aie4_gguf`. `Phi4::load_model()` selects the new backend only for that explicit value. An absent backend field retains the current NPU2 behavior. An unknown value is an error.

There is no automatic hardware or format detection and no fallback. This makes a request for the AIE4 tag observable and testable: it either runs through corelib or fails.

### Component boundaries

#### Phi-4 frontend

`modeling_phi4.cpp` remains responsible for:

- backend selection;
- tokenizer setup and chat-template application;
- sampling;
- request capacity checks;
- translating the existing `AutoModel` interface to the selected causal-LM engine.

It does not parse GGUF or call individual corelib operators.

#### Corelib API and runtime

A small dynamic adapter resolves only the C ABI symbols used by this backend:

- version, dependency self-test, device-context query, errors, and cleanup;
- object lifecycle;
- stream creation and synchronization;
- tensor creation, windows, reads, and writes;
- matmul padding, Q8_0 requantized weight creation, and dispatch;
- SSMLP padding, Q8_0 requantized weight creation, and dispatch;
- RMSNorm weight creation, padding, and dispatch;
- flat-MHA padding and dispatch.

The adapter owns no model policy. It converts failed statuses into exceptions that retain the corelib status, call name, and thread-local detail message. RAII wrappers release every returned object.

Corelib is not loaded during process startup. It is loaded when an AIE4 model is selected. Runtime lookup order is:

1. the absolute DLL named by `FLM_AIE4_CORELIB_PATH`;
2. `<flm executable directory>/aie4/ryzenai_corelib.dll`.

The loader does not search the current working directory. After loading, it resolves the version functions first, requires ABI `0.3.0`, resolves the remaining symbols, runs the dependency self-test, and verifies an AIE4 device context exists.

#### GGUF package

`Phi4GgufPackage` owns a read-only mapping of the single GGUF file and exposes validated, non-owning tensor views whose lifetime cannot exceed the mapping. It parses only the GGUF v3 facilities used by the pinned model:

- little-endian header;
- metadata scalar, string, and array encodings;
- tensor names, dimensions, GGML types, and relative offsets;
- model alignment and tensor-data start.

The parser performs checked arithmetic for every count, offset, alignment, and byte-length calculation. A truncated directory, duplicate tensor name, unsupported value type that cannot be skipped safely, out-of-range tensor, overlapping invalid range, or malformed string is a load error.

The model-facing API is intentionally narrow:

```cpp
TensorView RequireQ8(name, expected_shape);
FloatTensorView RequireF32(name, expected_shape);
ProjectionViews AttentionQkv(layer);
ProjectionViews GateUp(layer);
GgufPhi4Metadata Metadata();
```

The adapter hardcodes Phi-4 Mini's expected tensor names and architecture. QKV and gate/up tensors are fused in this GGUF; the adapter splits each into row-aligned byte-range views without dequantizing or copying it. Q8_0 rows consist of complete 34-byte blocks for 32 weights, so every allowed split must fall on a complete-row boundary.

#### Phi-4 AIE4 engine

The execution engine selectively carries forward the hardware-validated structure from PR #706:

- one corelib stream;
- padded tensors sized from corelib's helper APIs;
- fixed-size K/V caches;
- prefill and one-token decode;
- explicit synchronization at producer/consumer boundaries;
- host-side lazy embedding lookup;
- Phi-4 partial rotary tables;
- corelib matmul, fused SSMLP, standalone RMSNorm, and flat MHA dispatch;
- a maximum sequence length of 4096 and maximum usable decode window of 4095.

The source path changes completely: no ONNX initializers and no manifest are accepted.

For each quantized projection, the engine passes a raw Q8_0 block view to `ryzenai_corelib_*_weights_create_gguf_requantized` with group size 64. Weight objects are created serially. The corelib API documents an open, unattributed all-zero-output incident correlated with concurrent requantized creates; avoiding concurrency is the measured safe configuration and is required for this first implementation.

GGUF stores norms as F32. The adapter converts only the required norm vectors and epsilon to BF16 at model load. Embedding rows are decoded lazily for requested token IDs instead of materializing the full 200064-by-3072 embedding. RoPE tables are derived once from the GGUF's Phi-3/Phi-4 rope metadata and uploaded as FP32.

## Model contract validation

Validation occurs before device weight creation wherever possible. The package must match all of these constraints:

- the expected Phi-3/Phi-4 GGUF architecture identifier;
- 32 decoder layers;
- hidden size 3072;
- intermediate size 8192;
- 24 attention heads;
- 8 key/value heads;
- head size 128;
- vocabulary size 200064;
- partial rotary dimension 96;
- RMS epsilon `1e-5`;
- maximum sequence length 4096;
- `phi3.rope.dimension_count` equal to 96;
- finite, positive `phi3.rope.freq_base` and `phi3.rope.scaling.attn_factor`;
- `phi3.rope.scaling.original_context_length` equal to 4096;
- `rope_factors_short.weight`, when present, is F32 with exactly 48 elements;
- the long-rope branch is rejected because this backend supports only the original 4096-token window;
- every required projection present with the exact expected logical shape;
- every projection, tied embedding, and LM head source is Q8_0;
- every required norm present in the supported floating type;
- `output.weight` is absent and `token_embd.weight` is used for both embedding and LM head, as in the pinned model;
- tokenizer vocabulary size agrees with GGUF;
- `tokenizer.json` maps `<|end|>` to 200020 and `<|endoftext|>` to 199999;
- GGUF identifies 200020 as its EOS token and `config.json` identifies 199999, so the frontend stop set is their explicit union `{200020, 199999}`;
- `tokenizer_config.json` has `add_bos_token == false` and the frontend does not prepend `config.json`'s BOS token;
- the chat template contains the required Phi-4 user, end, and assistant markers.

The error names the model field or tensor, its actual value, and the expected value. The loader does not repair, reinterpret, or silently accept a mismatch.

## Pull and catalog design

The existing catalog format assumes one base repository per model. Retain the existing string-only `files` array and add an optional `file_sources` object keyed by those file names. Each override contains `url` and `revision`; files without an override continue to use the model's base URL unchanged. `model_info.json` remains the source of expected remote size and content hash for `pull` and `check`. No existing catalog entry needs migration.

For each file, pull:

1. constructs a URL from that file's fixed repository and revision;
2. downloads or resumes into a temporary path;
3. validates expected size and SHA-256;
4. atomically renames the completed file into the model directory.

A model is available only when all required files validate. `flm check` uses the same per-file records. No generated overlay is copied into the model directory.

The final directory is:

```text
models/phi4-mini-it-aie4/
├── Phi-4-mini-instruct.Q8_0.gguf
├── tokenizer.json
├── tokenizer_config.json
└── config.json
```

## Build and runtime configuration

The feature is disabled by default. An AIE4 developer build enables:

```text
FLM_ENABLE_CORELIB_AIE4=ON
RYZENAI_CORELIB_INCLUDE_DIR=<corelib install/include>
```

The build consumes the public header from the pinned corelib commit but does not link its import library. Calls go through the dynamically resolved function table. The normal `flm.exe` is produced and supports `pull`, `run`, and `serve`.

This PR does not copy runtime DLLs. The developer supplies `ryzenai_corelib.dll` and its DynamicDispatch, XRT, and RyzenMM dependency closure. `FLM_AIE4_CORELIB_PATH` or the executable-relative `aie4` directory identifies corelib itself; its dependent DLL directory must be available to the Windows loader.

A default build has no corelib compile or runtime requirement. An AIE4-enabled build with a missing runtime still starts and can run ordinary models; selecting `phi4-mini-it-aie4:4b` reports the missing runtime.

## Request lifecycle and error policy

A process-wide AIE4 access manager serializes AIE4 generation. One model instance handles one active generation at a time, matching the stream and mutable KV-cache ownership model.

Failures before any operation is submitted are recoverable model/request errors. Failures after submission, or during synchronization, leave device completion uncertain. The model instance is then marked poisoned, its conversational state is cleared, and subsequent requests are refused until the model is unloaded and recreated. FastFlowLM does not continue on potentially inconsistent KV state.

A cancellation is checked before prefill and between decode steps. It never destroys a stream while work is outstanding; submitted work is synchronized before the request releases model state.

Capacity checks happen before submission. They account for the rendered prompt and requested generation budget and enforce the AIE4 decode limit of 4095. Unbounded/sentinel generation requests are capped rather than allowed to reach an unsupported attention window.

## Expected file changes

### Existing files

```text
src/CMakeLists.txt
src/CMakePresets.json
src/common/AutoModel/automodel.cpp
src/common/AutoModel/modeling_phi4.cpp
src/include/AutoModel/automodel.hpp
src/include/AutoModel/modeling_phi4.hpp
src/pull/model_downloader.cpp
src/pull/model_downloader.hpp
src/model_list.json
src/model_info.json
src/runner/runner.cpp
src/server/rest_handler.cpp
src/server/server.cpp
src/src/main.cpp
```

Only files proven necessary during implementation should be changed. In particular, downloader changes are limited to optional per-file sources, and shared frontend changes are limited to behavior the AIE4 route requires.

### New product files

```text
src/include/corelib/corelib_api.hpp
src/include/corelib/corelib_object.hpp
src/include/corelib/corelib_runtime.hpp
src/common/corelib/corelib_api.cpp
src/common/corelib/corelib_runtime.cpp
src/common/corelib/corelib_sources.cmake
src/include/models/phi4/phi4_corelib_aie4.hpp
src/include/models/phi4/phi4_corelib_constants.hpp
src/include/models/phi4/phi4_corelib_gguf.hpp
src/include/models/phi4/phi4_corelib_shape_plan.hpp
src/include/models/phi4/phi4_corelib_host.hpp
src/common/corelib/phi4_corelib_aie4.cpp
src/common/corelib/phi4_corelib_gguf.cpp
src/common/corelib/phi4_corelib_shape_plan.cpp
src/common/corelib/phi4_corelib_host.cpp
```

The host component owns lazy Q8_0 embedding-row decode, F32-to-BF16 norm conversion, and FP32 RoPE-table derivation. The engine owns only model state and operator sequencing.

## Testing

### Unit tests without AIE4 hardware

Tests cover:

- valid GGUF v3 metadata and tensor-directory parsing;
- truncation, arithmetic overflow, bad alignment, duplicate names, and out-of-file ranges;
- missing tensors and incorrect dtype, shape, or byte length;
- zero-copy QKV and gate/up splits at exact row boundaries;
- Phi-4 architecture validation;
- tokenizer/config/GGUF disagreement;
- missing corelib symbols and exact ABI mismatch;
- object release and cleanup through a fake corelib;
- corelib call descriptors, group size 64, Q8_0 type, sequencing, and synchronization;
- per-file pull URLs, revisions, hashes, resume behavior, and atomic completion;
- unchanged behavior for existing single-source catalog entries;
- frontend routing and the absence of fallback;
- request bounds, cancellation, and poisoned-instance behavior.

### Runtime integration

Against the real DLL, tests verify:

- ABI `0.3.0`;
- every required symbol resolves;
- the dependency self-test succeeds;
- device-context reporting agrees with the test environment.

### Required AIE4 acceptance run

Before completion, run the produced `flm.exe` on a real AIE4 system:

```powershell
flm pull phi4-mini-it-aie4:4b
flm check phi4-mini-it-aie4:4b
flm run phi4-mini-it-aie4:4b
flm serve phi4-mini-it-aie4:4b
```

The acceptance run must include:

1. `What is 2+2?`, with a correct, self-terminated answer;
2. `What does AMD do?`, with a relevant answer;
3. at least ten prompts in one loaded process;
4. `/api/chat` and `/v1/chat/completions`;
5. request cancellation;
6. prompt and generation limit boundaries;
7. at least ten complete load-and-generate cycles, checking for empty or all-zero token output;
8. backend evidence proving corelib/AIE4 execution and no CPU/NPU2 fallback;
9. model load time, cold and warm TTFT, and decode tokens/second.

The record identifies the machine, power mode, FastFlowLM commit, corelib commit, corelib ABI, GGUF revision, commands, and outcomes. Performance numbers are descriptive, not a pass/fail gate, unless a regression threshold is agreed separately.

## Commit structure

Keep the work in one PR with reviewable commits:

1. `build: add optional dynamic corelib 0.3.0 runtime`
2. `feat: add validated Phi-4 Q8_0 GGUF reader`
3. `feat: add corelib-backed Phi-4 AIE4 engine`
4. `feat: route Phi-4 GGUF models through AIE4`
5. `feat: pull Phi-4 GGUF and tokenizer from pinned sources`
6. `test: validate Phi-4 GGUF AIE4 integration`
7. `docs: document developer setup and hardware results`

Each of commits 1–5 must compile before the next product commit is added. The test and documentation commits may depend on the completed product path. No commit adds a generated tensor manifest.

## Completion criteria

The PR is complete only when:

- default builds and existing model behavior remain unchanged;
- an AIE4-enabled build produces the normal `flm.exe`;
- ordinary models remain usable when corelib is absent;
- the new tag pulls and checks all files from their pinned sources;
- the installed model contains no manifest, ONNX model, or converted weights;
- the GGUF is mapped directly and Q8_0 projection views are passed to corelib's explicit requantized APIs;
- runtime ABI is exactly `0.3.0`;
- automated unit and fake-corelib tests pass;
- the real-DLL integration checks pass;
- the required real-AIE4 acceptance run passes;
- no CPU or NPU2 fallback exists;
- documentation states that Q8_0-to-group-64 conversion is lossy and records the tested revisions and hardware results.
