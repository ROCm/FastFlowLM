# Adding a model on the core backend

How to bring up a new model family on **core**, the backend that reaches its
kernels through ryzenai-corelib. There are two ways, and the first is the one
to reach for:

- **corelib runs the model whole** (it ships `ryzenai_corelib_model_<name>`):
  write one `BackendSpec` (its `.model` is the `ModelSpec`) and hand it to
  `flm::core::Factory` — see §2a. Every core model in the tree — Qwen3, Qwen3.5,
  Qwen3.6, Gemma 4 and Phi-4 — is built this way, each in one short
  `<model>_core_backend.cpp`.
- **corelib has only the ops**: write the engine yourself, op by op — the rest
  of this document, from §2 on, walks through it. No model in the tree does
  this any more; the last one, Phi-4's, is under
  `src/common/models/phi4/core/dev04/` in the history before Phi-4 moved to
  corelib's `phi4mini` model, and the files §2–§4 name are its.

A backend names *where the kernels come from*: `flm` is FastFlowLM's own kernel
flow, `core` is corelib. That is a separate axis from which silicon the host has,
which is `utils::npu_platform` (`aie2p`, `dev04`) and is answered at run time
by `utils::get_device()`. They are genuinely independent: a build links every
flow it was configured with, and several flows can serve one generation. Keying
either axis off the other is how adding corelib once took the FastFlowLM models
away.

`get_device()` asks the machine, once per process. A build that links corelib
asks corelib what NPU it has (`CorelibRuntime::NpuArch`, corelib's
`ryzenai_corelib_testing_npu_arch` until its public header names one): `AIEM`
is `dev04`, and `AIE2P`, no NPU, or a corelib that cannot be loaded is
`default_npu_platform()`, `aie2p`. corelib reads the device itself, so
FastFlowLM holds no table of devices and names none. A build without corelib
answers `aie2p`: every `dev04` model is a corelib one. Linking corelib still
says nothing about the silicon — it is what lets the build ask.

This is a contributor document. For *using* a backend once it exists — `--backend`,
`FLM_BACKEND`, precedence — see [`docs/docs/instructions/cli.md`](../../../docs/docs/instructions/cli.md).

---

## 0. Before you start

| | |
|---|---|
| **Platform** | Windows and Linux. Both configure and build; only Windows has been run on hardware, and the `src/test/phi4_core` suite is still Windows-only. |
| **Hardware** | An dev04 NPU. There is no simulator; a wrong shape shows up as garbage output, not an error. |
| **corelib headers** | The corelib ABI is still pre-1.0, so FastFlowLM requires an **exact `0.53.0`** match on major, minor and patch. [`corelib_api.hpp`](../../include/core/corelib_api.hpp) `#error`s on any other version — deliberately, because the C ABI has changed shape between patch releases. The version is queried before any other entry point, so a mismatched runtime reports a version error rather than a missing symbol. corelib's default build gives one `ryzenai_corelib_model_<name>` library per model, carrying its kernels, which must sit beside corelib. A per-op engine enables its model's kernel bundle on the stream it opens; a model corelib runs whole enables its own. |
| **Weights** | A GGUF corelib's model reads: a Q8_0 one, which it requantizes on every load, or a Q4_G128 or Q4_G32 one, whose codes it packs as they are. The catalog names which; see `preferred_source` in §6. |

Configure with:

```shell
cmake -B build -S src -DFLM_ENABLE_CORE=ON
# RYZENAI_CORELIB_INCLUDE_DIR / RYZENAI_CORELIB_LIB_DIR are found automatically
# when they are on the default paths; otherwise pass them. The library lookup
# searches lib/ and lib64/ beside the headers as well as RYZENAI_CORELIB_LIB_DIR.
```

Boost is looked up only on Windows: XRT's `xrt/detail/any.h` falls back to
`boost::any` whenever `__cplusplus` reads below 201703L, which MSVC does
unless handed `/Zc:__cplusplus`. GCC and Clang report C++20 honestly, so that
branch is never taken and there is nothing to find.

That builds the `flm_core` static library and links it into `flm`. The
library carries `FLM_ENABLE_CORE=1` as a **PUBLIC** compile definition, so
everything that links it sees the `#if` guards flip.

---

## 1. Pick the family

You need one name, not two.

- **family** — `details.family` in [`model_list.json`](../../model_list.json), e.g. `"phi4"`.
  It selects the *frontend* (the chat template, the tokenizer contract, the sampler).
- **backend id** — already decided: `core`. It names the kernel provider, and the
  constant is [`flm::backend::kCoreBackendId`](../../include/AutoModel/model_backend.hpp).
  You do not mint one.

So a corelib engine for a family that already exists (Phi-4 today) needs no new
name at all — it registers the family's `core` backend. A genuinely new
architecture needs a new family.

A family has at most one backend per provider. If you find yourself wanting two
engines behind the same provider, the choice belongs in the catalog as two
entries, not in the backend id.

---

## 2a. A model corelib runs whole

[`core/corelib_model.hpp`](../../include/core/corelib_model.hpp) is a `causal_lm`
over `ryzenai_corelib_model_*`; [`core/corelib_model_backend.hpp`](../../include/core/corelib_model_backend.hpp)
is the `ModelBackend` around it. A family needs only its spec:

```cpp
flm::backend::BackendFactory core_factory() {
    return core::Factory({
        .model = {"qwen3", "Qwen3", kCoreContextLimit, core::Continuation::kFromStart,
                  core::Rewind::kSeek},
        .context_rule = "1..4096",
        .eos_ids = {151645, 151643},
    });
}
```

Two fields say how the model behaves, and both come from what corelib's driver
accepts (read its `models/<Name>.cpp`):

| field | value | when |
|---|---|---|
| `continuation` | `kAnywhere` | the driver takes `rows > 1` at any position |
| | `kFromStart` | only at position 0; a later prompt re-runs the conversation |
| | `kFromPage` | at 0 or on a 4096-row page boundary of a paged cache |
| `rewind` | `kSeek` | attention only: each pass states its position |
| | `kSnapshot` | recurrent state in state slots: `model_state_save/restore` |
| | `kReplay` | recurrent state outside any slot: reset and replay |

Leave `eos_ids` empty to read the stop tokens from the package, and set
`read_bos` when `tokenizer_config.json` names a BOS but no id states it.
corelib packs the weights on every load; there is no packed-weight cache on
this path.

The header (`<model>_core_backend.hpp`) keeps the `inline core_traits()` and the
context constants; registration in `builtin_backends.cpp` is unchanged.
Tests: [`test/core_model/`](../../test/core_model/) covers the engine and the
backend against the fake corelib; a new family adds no engine test of its own.

---

## 2. Lay out the files

```
src/include/models/<model>/<backend>/<platform>/  headers
src/common/models/<model>/<backend>/<platform>/   the implementation
```

Both axes from the top of this document appear in the path, in that order: the
family owns the directory, the kernel provider comes next, and the silicon
generation last — `phi4/core/dev04/` for the corelib implementation, and
`phi4/flm/aie2p/` for the FastFlowLM one. Nothing keys one axis off the other;
the path just records both answers, so a family that gains a second provider or
a second generation grows a sibling folder instead of a conditional.

FastFlowLM's own engines have no *sources* here — they ship as prebuilt libraries
under `lib/<runtime>/` — but their headers follow the same three levels. Headers
mirror the source split, so an `#include` says which provider and which silicon
it belongs to (`models/phi4/core/dev04/phi4_core_backend.hpp` against
`models/phi4/flm/aie2p/phi4_npu.hpp`), and a grep for `models/<model>/core/` finds
everything the core path pulls in.

**Do not edit any `CMakeLists.txt` for this.** [`models_sources.cmake`](models_sources.cmake)
globs `*/core/*/*.cpp` — every platform under every family's core folder — and that
glob is what `flm_core` compiles. Creating the folder is the whole registration step.

A per-op engine is six translation units, as Phi-4's was, and the split is worth
copying:

| file | what belongs in it |
|---|---|
| `<model>_core_gguf.cpp` | Open and validate the GGUF. Tensor lookup by name, shape checks, metadata, and the cross-validation against `config.json` / `tokenizer.json` / `tokenizer_config.json`. **No device code.** |
| `<model>_core_shape_plan.cpp` | Ask corelib which kernels it ships (`*_enum_kernels`), pick the smallest one covering each live row count (`CoveringRows`, in `core/kernel_grid.hpp` at `f827ca9a`), and cache one row-extent record per row count. Built once per engine. |
| `<model>_core_weight_cache.cpp` | The on-disk cache of packed weights: write after a fresh pack, reload as one `role_packed` source per weight. |
| `<model>_core_host.cpp` | The math corelib does not do: Q8 embedding-row decode, RMS norm, f32→bf16, RoPE tables. Plain CPU, unit-testable, no corelib types in the signatures. |
| `<model>_core.cpp` | The `causal_lm` subclass. Owns the device tensors, the stream and the decode loop. |
| `<model>_core_backend.cpp` | The `ModelBackend`. Construction order and execution policy. |

Keeping the GGUF and host layers free of corelib types is what lets you test them
without hardware.

---

## 3. The engine: a `causal_lm` subclass

Model it on `phi4_core.hpp` at `f827ca9a`. Two rules matter more than the rest.

### `causal_lm.hpp` is a frozen ABI

`src/lib/xrt` and `src/lib/hrx` ship ~20 **prebuilt** engine libraries compiled
against today's [`causal_lm.hpp`](../../include/causal_lm.hpp). Their vtables are
emitted inside those binaries. Adding, removing or reordering a virtual there
shifts vtable slots and corrupts dispatch **at runtime, with no compiler error**.

So: you implement `causal_lm` as it stands. You do not change it. The same
applies to the layout and virtuals of `buffer.hpp`, `tensor_2d.hpp`,
`lm_config.hpp`, `q4_npu_eXpress.hpp` and `npu_utils/*` (a fix inside an inline
function body, such as `LM_Config`'s config normalization, leaves both alone). If a change to any of those looks necessary, the seam you
actually want is `ModelBackend` (§4), which sits above `causal_lm` and is compiled
from source.

### `load_weights(Q4NX&)` is a shim

It is pure virtual in the frozen header, but it describes FastFlowLM's own weight
format, which a GGUF engine does not have. Implement it as a throwing stub with a
comment saying why:

```cpp
// An ABI shim, not a capability. load_weights is pure virtual in causal_lm.hpp,
// which is frozen because the engine libraries in src/lib/<runtime> are prebuilt
// against it. Nothing calls this: FlmBackend loads weights through the
// concrete engine type, and this engine's weights come from the GGUF package it
// was constructed with. See AutoModel/model_backend.hpp.
void <model>_core::load_weights(Q4NX&) { throw std::runtime_error(...); }
```

Nothing calls it: `FlmBackend` calls `load_weights` through the *concrete*
engine type, never through a `causal_lm*`.

### Everything else

- Take the GGUF package and the `CorelibRuntime` by `shared_ptr` in the constructor
  and hold both for the engine's lifetime. corelib objects must not outlive the
  API they were created from.
- Create weights concurrently. Requantizing the 161 weights is effectively the
  whole of model load, and the creates are independent — each reads its own
  mapped range and produces its own object — so they run across a pool
  (`kWeightCreateConcurrency`). The per-create thread hint
  (`kRequantizeThreads`) goes to corelib as `fast_packer_threads`: 0 lets corelib
  choose (8 threads), 1 keeps each create serial. The two multiply,
  so pick them together. See `phi4_core_constants.hpp` at `f827ca9a`.
- Expose `bool poisoned() const noexcept`. corelib fails only the call that
  failed, so `CorelibModel` resets the session and runs everything it held again
  from position 0, once; a second failure poisons the engine, the backend
  surfaces this, and `AutoModel` turns it into a 500 that asks for an
  unload/reload.
- Put every magic number in a `<model>_core_constants.hpp` with a comment on
  where it came from.

---

## 4. The backend: policy, not just construction

[`ModelBackend`](../../include/AutoModel/model_backend.hpp) owns one engine **and
every rule for driving it**. Its defaults describe the FastFlowLM NPU engines, so
you override only what differs. A per-op backend overrides six, the values
here being the per-op Phi-4 one's (the shared backend in
[`corelib_model_backend.cpp`](../core/corelib_model_backend.cpp) states the same
ones from its spec):

| override | corelib value | why |
|---|---|---|
| `id()` | `"core"` | the kernel provider; what `--backend` matches |
| `detail()` | `runtime_->loaded_library_path()` | provenance line in `flm show` |
| `max_decode_length()` | `4095` | corelib's own decode window |
| `supports_preemption()` | `false` | no checkpoint/restore on this path |
| `forwards_past_eos()` | `false` | the extra post-EOS `forward()` the FLM engines want would exceed the window |
| `forced_eos_ids()` | `{200020, 199999}` | `<\|end\|>` and `<\|endoftext\|>`: `tokenizer_config.json` names only the second |

`poisoned()` forwards to the engine.

### Constructor order

Validate everything *before* you touch the device, so a mismatched package fails
while nothing has been allocated:

1. reject preemption / an out-of-range context length,
2. read `config.json`, `tokenizer.json`, `tokenizer_config.json`,
3. `Open` the GGUF and cross-validate it against all three,
4. `CorelibRuntime::GetOrCreate(...)`,
5. construct the engine, `clear_context()`.

Declare the runtime `shared_ptr` **before** the engine member so it is destroyed
after it, and reset the engine explicitly in the destructor.

### Traits: what the frontend must know *before* the backend exists

`BackendTraits` cannot be a virtual on `ModelBackend` — the frontend consults it
while assembling the `BackendContext`. Keep it `inline` in the header so tests
and the registry can read it without linking the engine:

```cpp
inline flm::backend::BackendTraits core_traits() {
    flm::backend::BackendTraits traits;
    traits.needs_npu_xclbin   = false;   // no xclbin manager is built
    traits.supports_preemption = false;  // rejected before the factory runs
    traits.max_context_length = 4096;    // rejected before the factory runs
    return traits;
}
```

### Register it

One line in [`builtin_backends.cpp`](../AutoModel/builtin_backends.cpp), under the
guard:

```cpp
#if defined(FLM_ENABLE_CORE)
    registry.register_backend("<family>", flm::backend::kCoreBackendId,
                              flm::<model>::core_factory(),
                              flm::<model>::core_traits());
#endif
```

This is the only file that knows both family names and engine types.

---

## 5. The frontend (new families only)

If the family already exists, you are done with C++ — `AutoModel::_shared_load_backend`
resolves the id, checks the traits, builds your backend and points `lm_engine` at
its engine. No frontend change.

For a new family, add `modeling_<family>.{hpp,cpp}` under `AutoModel/` and wire it
into [`all_models.hpp`](../../include/AutoModel/all_models.hpp) — the enum, the
`modelFamilyMap` entry, and the `switch` case. Keep it backend-agnostic:
[`modeling_phi4.cpp`](../AutoModel/modeling_phi4.cpp) has no `#if
FLM_ENABLE_CORE` anywhere. It loads the backend, sets up the tokenizer
(honouring `backend_->forced_eos_ids()`), applies the sampler, and delegates the
decode loop to `_shared_generate`.

Also register the FLM-side engine if the family has one:
`RegisterFlm<<family>_npu>(registry, "<family>")`.

---

## 6. The catalog

Two files, and both of them must agree.

**[`model_list.json`](../../model_list.json)** is the one catalog. Every install
ships it and every build reads it; what differs between installs is how much of
it survives [`model_list::apply_support_filter`](../../include/model_list.hpp),
which drops every entry this machine or this build cannot run. A model that is
offered nowhere is the same to a user as a model that does not exist, so both
axes prune:

| the entry is dropped when | said by |
|---|---|
| the host is not silicon the entry names | `"supported_platforms"` on the entry |
| the build did not link the kernel flow the entry needs | the **family name**: a family ending in `-core` is corelib's, everything else is FastFlowLM's |

So the corelib flavour of a model is **its own top-level family**, named
`<family>-core`, sitting beside the stock one rather than patching it:

```jsonc
"<family>-core": {
  "<size>": {
    "supported_platforms": ["dev04"],
    "name": "<dir name under models/>",
    "url": "...", "file_url": "...", "size": 4100140571,
    "default_context_length": 4096,
    "files": ["<weights>.gguf", "tokenizer.json", "tokenizer_config.json", "config.json"],
    "file_sources": { "tokenizer.json": { "url": "...", "revision": "..." } }
  }
}
```

Points that are easy to get wrong:

- **The family name is the mechanism.** Nothing in the file says `core`;
  `model_list` reads the `-core` suffix off the tag and stamps the answer onto
  the entry as `"backend"`, which is what
  [`AutoModel`](../AutoModel/automodel.cpp) later reads. Do not write `backend`
  by hand — it is derived, and a hand-written one is overwritten.
  `--backend` and `FLM_BACKEND` still override it, and are checked against the
  registry. The plural `supported_backends` and `details.execution_backend` are
  retired, as are `supported_backend` and `platform_overrides`.
- **`supported_platforms` is required on every shipped entry**, and
  `src/test/model_list_platform` fails if one is missing or disagrees with its
  family name. Omitting it is legal — it means *every* generation, which is what
  a catalog written before the key meant — but a shipped entry should say what
  it was built for. Every FastFlowLM entry is `["aie2p"]`; every `-core` entry is
  `["dev04"]`.
- Separate families mean **separate tags**, so no `model_info_key` redirect is
  needed: `model_info.json` keys the corelib records under `<family>-core:<size>`
  directly, which is also the tag a user types.
- `file_sources` pins a per-file origin + revision when the weights and the
  tokenizer come from different repos (very common with GGUF mirrors).
- `preferred_source` is a JSON merge patch over the entry naming a source
  `flm pull` tries first — today the AMD Q4 GGUFs under `huggingface.co/amd/`,
  over the public Q8_0 mirror the entry itself names. It carries its own `url`,
  `files`, `size` and `model_info_key` (records keyed `<tag>@amd`), and
  `"use_hf_token": true` sends the user's Hugging Face token (`HF_TOKEN`, or
  the file `hf auth login` writes) with the files that `url` serves, and with
  nothing else. A pull uses it when its files are already on disk or the
  first of them answers a HEAD request, and falls back to the entry otherwise;
  outside a pull, the entry resolves to the preferred source exactly when every
  file only it has is on disk.
- The two keys are checked independently, and **the platform is checked first
  and for everyone**. On a `dev04` NPU a corelib build offers the `-core` tags
  and nothing else; on an `aie2p` one, or none, every build offers the
  FastFlowLM tags and none of the `-core` ones.
- An empty catalog is therefore an ordinary state, not a crash: `model_list`
  prints one line naming the generation and the linked kernels, `flm list` says
  it found nothing, `flm run` says the tag is not found, and `flm --help` still
  works. A build that aborted here could not even tell you why.

**[`model_info.json`](../../model_info.json)** — one record per file, with `size`
and `sha256`. The downloader refuses anything it cannot match
([`model_downloader.cpp`](../../pull/model_downloader.cpp)).

---

## 7. Tests

Three tiers, and the first two do not need an NPU:

- **Pure logic** — for a per-op engine, the GGUF and host layers. Shape
  rejection, metadata parsing, contract cross-validation, RMS norm and RoPE
  against reference values. A spec has none: [`test/core_model/`](../../test/core_model/)
  runs it against the fake corelib, as `test_phi4_core_backend.cpp` does Phi-4's.
- **Frontend against a stub backend** — see
  [`test/phi4_core/test_phi4_frontend.cpp`](../../test/phi4_core/test_phi4_frontend.cpp).
  It defines its own empty `flm::backend::register_builtin_backends` (so it never
  links the prebuilt engine libraries) and installs stubs with
  `BackendRegistry::replace_backend`. The stub repeats the real backend's
  pre-device work, which keeps the "nothing is opened before validation"
  and "no device is created for a bad package" assertions meaningful.
- **Registry and selection** — [`test/model_backend/`](../../test/model_backend/),
  Linux-buildable: registration, duplicate ids, resolution precedence, and the
  error text for an id this build does not have.

A per-op engine adds a `test/<model>_core/` directory following the Phi-4 one.
Note the trick that one uses at configure time: it synthesizes a version-bumped copy of
`ryzenai/corelib.h` and asserts the adapter **fails** to compile against it, which
is how the version pin is actually enforced.

---

## 8. Verify

```powershell
flm pull  <family>:<size>
flm run   <family>:<size> --backend core
flm run   <family>:<size> --backend bogus        # must exit 1, listing the real ids
```

Check, in order:

1. `show_profile` prints `Backend: core` and a `Backend detail:` line
   with the loaded corelib path.
2. Generation is coherent. Garbage output on core is almost always a shape-plan or
   requantization-threading bug, not a tokenizer bug.
3. Decode stops at the window limit without a corelib error.
4. The prompt-length ceiling and the preemption rejection fire *before* any device
   allocation — they are traits, checked by the frontend.
5. `git diff` shows **no** layout or virtual changes to `causal_lm.hpp`,
   `buffer.hpp`, `tensor_2d.hpp`, `lm_config.hpp`, `q4_npu_eXpress.hpp` and
   `npu_utils/`.
   A violation here fails at runtime, not at build time, which is why it is a
   checklist item rather than a compiler's job.

---

## Checklist

- [ ] `src/common/models/<model>/core/<platform>/` created (no CMake edit)
- [ ] GGUF/host layers hold no corelib types
- [ ] `load_weights` is a documented throwing shim
- [ ] weight creates run across the `kWeightCreateConcurrency` pool, with the
      thread hint passed
- [ ] runtime `shared_ptr` declared before the engine member
- [ ] all validation happens before `GetOrCreate`
- [ ] `BackendTraits` is `inline` in the header
- [ ] registered in `builtin_backends.cpp` under `#if defined(FLM_ENABLE_CORE)`
- [ ] no `#if FLM_ENABLE_CORE` anywhere in the frontend
- [ ] `model_list.json` and `model_info.json` agree, the corelib family is
      named `-core` and its `supported_platforms` says `dev04`
- [ ] frozen headers: no layout or virtual changes
