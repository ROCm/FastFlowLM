# Adding a model on the rai backend

How to bring up a new model family on **rai**, the backend that reaches its
kernels through ryzenai-corelib, using Phi-4 as the worked example. Phi-4 is the
only model on this path today, so every file named below has a `phi4` counterpart
you can read straight through.

A backend names *where the kernels come from*: `flm` is FastFlowLM's own kernel
flow, `rai` is corelib. That is a separate axis from which silicon the host has,
which is `utils::npu_platform` (`aie2p`, `aie_next`) and is answered at run time
by `utils::get_device()`. They are genuinely independent: a build links every
flow it was configured with, and several flows can serve one generation. Keying
either axis off the other is how adding corelib once took the FastFlowLM models
away.

`get_device()` is a **stand-in** until the real probe lands, and it consults
nothing: it returns `default_npu_platform()`. Not the environment, and *not*
whether corelib was linked — linking corelib says what kernels this binary has,
never what silicon it is running on, and an install that lists a model it
cannot run is worse than one that lists nothing. So the constant *is* the
answer, for every build: change it and rebuild to move a whole install to the
other generation, and the probe replaces the one function.

It is **`aie_next`** today, while the corelib path is being brought up. Read
the next paragraph before being surprised by what `flm list` shows.

This is a contributor document. For *using* a backend once it exists — `--backend`,
`FLM_BACKEND`, precedence — see [`docs/docs/instructions/cli.md`](../../../docs/docs/instructions/cli.md).

---

## 0. Before you start

| | |
|---|---|
| **Platform** | Windows and Linux. Both configure and build; only Windows has been run on hardware, and the `src/test/phi4_rai` suite is still Windows-only. |
| **Hardware** | An aie_next NPU. There is no simulator; a wrong shape shows up as garbage output, not an error. |
| **corelib headers** | Exactly **0.5.0**. [`corelib_api.hpp`](../../include/rai/corelib_api.hpp) `#error`s on any other version — deliberately, because the C ABI has changed shape between patch releases. |
| **Weights** | A GGUF the vendor kernels can requantize. Phi-4 uses Q8_0; the corelib entry points are `*_create_gguf_requantized`. |

Configure with:

```shell
cmake -B build -S src -DFLM_ENABLE_RAI=ON
# RYZENAI_CORELIB_INCLUDE_DIR / RYZENAI_CORELIB_LIB_DIR are found automatically
# when they are on the default paths; otherwise pass them. The library lookup
# searches lib/ and lib64/ beside the headers as well as RYZENAI_CORELIB_LIB_DIR.
```

Boost is looked up only on Windows: XRT's `xrt/detail/any.h` falls back to
`boost::any` whenever `__cplusplus` reads below 201703L, which MSVC does
unless handed `/Zc:__cplusplus`. GCC and Clang report C++20 honestly, so that
branch is never taken and there is nothing to find.

That builds the `flm_rai` static library and links it into `flm`. The
library carries `FLM_ENABLE_RAI=1` as a **PUBLIC** compile definition, so
everything that links it sees the `#if` guards flip.

---

## 1. Pick the family

You need one name, not two.

- **family** — `details.family` in [`model_list.json`](../../model_list.json), e.g. `"phi4"`.
  It selects the *frontend* (the chat template, the tokenizer contract, the sampler).
- **backend id** — already decided: `rai`. It names the kernel provider, and the
  constant is [`flm::backend::kRaiBackendId`](../../include/AutoModel/model_backend.hpp).
  You do not mint one.

So a corelib engine for a family that already exists (Phi-4 today) needs no new
name at all — it registers the family's `rai` backend. A genuinely new
architecture needs a new family.

A family has at most one backend per provider. If you find yourself wanting two
engines behind the same provider, the choice belongs in the catalog as two
entries, not in the backend id.

---

## 2. Lay out the files

```
src/include/models/<model>/rai/  headers for the corelib implementation
src/common/models/<model>/rai/   the corelib implementation
```

The folders are named after the kernel provider, not the silicon. FastFlowLM's
own engines have no folder here — they ship as prebuilt libraries under
`lib/<runtime>/` and their headers stay at `models/<model>/`. Headers mirror the
source split, so an `#include` says which provider it belongs to
(`models/phi4/rai/phi4_rai_gguf.hpp` against `models/phi4/phi4_npu.hpp`), and a
grep for `models/<model>/rai/` finds everything the rai path pulls in.

**Do not edit any `CMakeLists.txt` for this.** [`models_sources.cmake`](models_sources.cmake)
globs `*/rai/*.cpp`, and that glob is what `flm_rai` compiles. Creating the
folder is the whole registration step.

Phi-4's rai side is five translation units, and the split is worth copying:

| file | what belongs in it |
|---|---|
| `<model>_rai_gguf.cpp` | Open and validate the GGUF. Tensor lookup by name, shape checks, metadata, and the cross-validation against `config.json` / `tokenizer.json` / `tokenizer_config.json`. **No device code.** |
| `<model>_rai_shape_plan.cpp` | Ask corelib how it wants each operator padded (`matmul_pad_shape`, `ssmlp_pad_rows`, …) and cache one row-extent record per live row count. Built once per engine. |
| `<model>_rai_host.cpp` | The math corelib does not do: Q8 embedding-row decode, RMS norm, f32→bf16, RoPE tables. Plain CPU, unit-testable, no corelib types in the signatures. |
| `<model>_rai.cpp` | The `causal_lm` subclass. Owns the device tensors, the stream and the decode loop. |
| `<model>_rai_backend.cpp` | The `ModelBackend`. Construction order and execution policy. |

Keeping the GGUF and host layers free of corelib types is what lets you test them
without hardware.

---

## 3. The engine: a `causal_lm` subclass

Model it on [`phi4_rai.hpp`](../../include/models/phi4/rai/phi4_rai.hpp).
Two rules matter more than the rest.

### `causal_lm.hpp` is a frozen ABI

`src/lib/xrt` and `src/lib/hrx` ship ~20 **prebuilt** engine libraries compiled
against today's [`causal_lm.hpp`](../../include/causal_lm.hpp). Their vtables are
emitted inside those binaries. Adding, removing or reordering a virtual there
shifts vtable slots and corrupts dispatch **at runtime, with no compiler error**.

So: you implement `causal_lm` as it stands. You do not change it. The same
applies to `buffer.hpp`, `tensor_2d.hpp`, `lm_config.hpp`, `q4_npu_eXpress.hpp`
and `npu_utils/*`. If a change to any of those looks necessary, the seam you
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
void <model>_rai::load_weights(Q4NX&) { throw std::runtime_error(...); }
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
  (`kRequantizeThreads`) stays at corelib's default of one so the two forms of
  parallelism do not multiply into an oversubscribed machine. See
  [`phi4_rai_constants.hpp`](../../include/models/phi4/rai/phi4_rai_constants.hpp).
- Expose `bool poisoned() const noexcept`. A corelib failure mid-decode usually
  leaves device state that only a reload can clear; the backend surfaces this and
  `AutoModel` turns it into a 500 that asks for an unload/reload.
- Put every magic number in a `<model>_rai_constants.hpp` with a comment on
  where it came from.

---

## 4. The backend: policy, not just construction

[`ModelBackend`](../../include/AutoModel/model_backend.hpp) owns one engine **and
every rule for driving it**. Its defaults describe the FastFlowLM NPU engines, so
you override only what differs. Phi-4's corelib backend
([`phi4_rai_backend.cpp`](phi4/rai/phi4_rai_backend.cpp)) overrides six:

| override | corelib value | why |
|---|---|---|
| `id()` | `"rai"` | the kernel provider; what `--backend` matches |
| `detail()` | `runtime_->loaded_library_path()` | provenance line in `flm show` |
| `max_decode_length()` | `4095` | corelib's own decode window |
| `supports_preemption()` | `false` | no checkpoint/restore on this path |
| `forwards_past_eos()` | `false` | the extra post-EOS `forward()` the FLM engines want would exceed the window |
| `forced_eos_ids()` | `{200020, 199999}` | proven by three sources agreeing in `ValidatePhi4Contract`, which beats `tokenizer_config.json` alone |

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
inline flm::backend::BackendTraits rai_traits() {
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
#if defined(FLM_ENABLE_RAI)
    registry.register_backend("<family>", flm::backend::kRaiBackendId,
                              flm::<model>::rai_factory(),
                              flm::<model>::rai_traits());
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
FLM_ENABLE_RAI` anywhere. It loads the backend, sets up the tokenizer
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
| the build did not link the kernel flow the entry needs | the **family name**: a family ending in `-rai` is corelib's, everything else is FastFlowLM's |

So the corelib flavour of a model is **its own top-level family**, named
`<family>-rai`, sitting beside the stock one rather than patching it:

```jsonc
"<family>-rai": {
  "<size>": {
    "supported_platforms": ["aie_next"],
    "name": "<dir name under models/>",
    "url": "...", "file_url": "...", "size": 4100140571,
    "default_context_length": 4096,
    "files": ["<weights>.gguf", "tokenizer.json", "tokenizer_config.json", "config.json"],
    "file_sources": { "tokenizer.json": { "url": "...", "revision": "..." } }
  }
}
```

Points that are easy to get wrong:

- **The family name is the mechanism.** Nothing in the file says `rai`;
  `model_list` reads the `-rai` suffix off the tag and stamps the answer onto
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
  it was built for. All 42 FastFlowLM entries are `["aie2p"]`; the corelib one
  is `["aie_next"]`.
- Separate families mean **separate tags**, so no `model_info_key` redirect is
  needed: `model_info.json` keys the corelib records under `<family>-rai:<size>`
  directly, which is also the tag a user types.
- `file_sources` pins a per-file origin + revision when the weights and the
  tokenizer come from different repos (very common with GGUF mirrors).
- The two keys are checked independently, and **the platform is checked first
  and for everyone**. On today's `aie_next` default that means a corelib build
  offers `phi4-mini-it-rai` and nothing else, and a build *without* corelib
  offers **nothing at all** — the 42 FastFlowLM entries are pruned by the
  generation, and the corelib entry by the kernels it would need. That is the
  cost of bringing up the next generation before its probe exists, and it is
  reversed by setting `default_npu_platform()` back to `aie2p`, which gives
  both builds the same 42 tags and neither the corelib one.
- An empty catalog is therefore an ordinary state, not a crash: `model_list`
  prints one line naming the generation and the linked kernels, `flm list` says
  it found nothing, `flm run` says the tag is not found, and `flm --help` still
  works. A build that aborted here could not even tell you why.

**[`model_info.json`](../../model_info.json)** — one record per file, with `size`
and `sha256`. The downloader refuses anything it cannot match
([`model_downloader.cpp:46`](../../pull/model_downloader.cpp#L46)).

---

## 7. Tests

Three tiers, and the first two do not need an NPU:

- **Pure logic** — the GGUF and host layers. Shape rejection, metadata parsing,
  contract cross-validation, RMS norm and RoPE against reference values.
- **Frontend against a stub backend** — see
  [`test/phi4_rai/test_phi4_frontend.cpp`](../../test/phi4_rai/test_phi4_frontend.cpp).
  It defines its own empty `flm::backend::register_builtin_backends` (so it never
  links the prebuilt engine libraries) and installs stubs with
  `BackendRegistry::replace_backend`. The stub repeats the real backend's
  pre-device work verbatim, which keeps the "nothing is opened before validation"
  and "no device is created for a bad package" assertions meaningful.
- **Registry and selection** — [`test/model_backend/`](../../test/model_backend/),
  Linux-buildable: registration, duplicate ids, resolution precedence, and the
  error text for an id this build does not have.

Add a `test/<model>_rai/` directory following the Phi-4 one. Note the
trick it uses at configure time: it synthesizes a version-bumped copy of
`ryzenai/corelib.h` and asserts the adapter **fails** to compile against it, which
is how the version pin is actually enforced.

---

## 8. Verify

```powershell
flm pull  <family>:<size>
flm run   <family>:<size> --backend rai
flm run   <family>:<size> --backend bogus        # must exit 1, listing the real ids
```

Check, in order:

1. `show_profile` prints `Backend: rai` and a `Backend detail:` line
   with the loaded corelib path.
2. Generation is coherent. Garbage output on rai is almost always a shape-plan or
   requantization-threading bug, not a tokenizer bug.
3. Decode stops at the window limit without a corelib error.
4. The prompt-length ceiling and the preemption rejection fire *before* any device
   allocation — they are traits, checked by the frontend.
5. `git diff --stat` shows **zero** changes to `causal_lm.hpp`, `buffer.hpp`,
   `tensor_2d.hpp`, `lm_config.hpp`, `q4_npu_eXpress.hpp` and `npu_utils/`.
   A violation here fails at runtime, not at build time, which is why it is a
   checklist item rather than a compiler's job.

---

## Checklist

- [ ] `src/common/models/<model>/rai/` created (no CMake edit)
- [ ] GGUF/host layers hold no corelib types
- [ ] `load_weights` is a documented throwing shim
- [ ] weight creates are serialized, with the thread hint passed
- [ ] runtime `shared_ptr` declared before the engine member
- [ ] all validation happens before `GetOrCreate`
- [ ] `BackendTraits` is `inline` in the header
- [ ] registered in `builtin_backends.cpp` under `#if defined(FLM_ENABLE_RAI)`
- [ ] no `#if FLM_ENABLE_RAI` anywhere in the frontend
- [ ] `model_list.json` and `model_info.json` agree, the corelib family is
      named `-rai` and its `supported_platforms` says `aie_next`
- [ ] frozen headers untouched
