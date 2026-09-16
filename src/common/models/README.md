# Adding a model on the corelib backend

How to bring up a new model family (or a new backend for an existing one) on
**ryzenai-corelib / AIE4**, using Phi-4 as the worked example. Phi-4 is the only
model on this path today, so every file named below has a `phi4` counterpart you
can read straight through.

This is a contributor document. For *using* a backend once it exists — `--backend`,
`FLM_BACKEND`, precedence — see [`docs/docs/instructions/cli.md`](../../../docs/docs/instructions/cli.md).

---

## 0. Before you start

| | |
|---|---|
| **Platform** | Windows only. `FLM_ENABLE_CORELIB_AIE4` is rejected at configure time elsewhere ([`CMakeLists.txt:54`](../../CMakeLists.txt#L54)). |
| **Hardware** | An AIE4 NPU. There is no simulator; a wrong shape shows up as garbage output, not an error. |
| **corelib headers** | Exactly **0.3.0**. [`corelib_api.hpp`](../../include/corelib/corelib_api.hpp) `#error`s on any other version — deliberately, because the C ABI has changed shape between patch releases. |
| **Weights** | A GGUF the vendor kernels can requantize. Phi-4 uses Q8_0; the corelib entry points are `*_create_gguf_requantized`. |

Configure with:

```powershell
cmake -B build -S src -DFLM_ENABLE_CORELIB_AIE4=ON
# RYZENAI_CORELIB_INCLUDE_DIR / RYZENAI_CORELIB_LIB_DIR are found automatically
# when they are on the default paths; otherwise pass them.
```

That builds the `flm_corelib_aie4` static library and links it into `flm`. The
library carries `FLM_ENABLE_CORELIB_AIE4=1` as a **PUBLIC** compile definition, so
everything that links it sees the `#if` guards flip.

---

## 1. Pick the two names

You need a **family** and a **backend id**, and they are not the same thing.

- **family** — `details.family` in [`model_list.json`](../../model_list.json), e.g. `"phi4"`.
  It selects the *frontend* (the chat template, the tokenizer contract, the sampler).
- **backend id** — the key a user types after `--backend`, e.g. `"corelib_aie4_gguf"`.
  It selects the *engine*. One family can have several.

A new corelib model that is an existing family (a second Phi-4 quantization, say)
needs only a new backend id. A genuinely new architecture needs both.

Declare the id as an `inline constexpr` next to its traits, the way
[`phi4_corelib_backend.hpp`](../../include/models/phi4/corelib/phi4_corelib_backend.hpp) does —
not as a bare string literal at the registration site.

---

## 2. Lay out the files

```
src/include/models/<model>/flm/      headers for the FastFlowLM NPU implementation
src/include/models/<model>/corelib/  headers for the corelib implementation
src/common/models/<model>/flm/       FastFlowLM's own NPU implementation
src/common/models/<model>/corelib/   the corelib implementation
```

Headers mirror the source split, so an `#include` says which engine it belongs to
(`models/phi4/corelib/phi4_corelib_gguf.hpp`, `models/phi4/flm/phi4_npu.hpp`) and a
grep for `models/<model>/corelib/` finds everything the corelib path pulls in.

**Do not edit any `CMakeLists.txt` for this.** [`models_sources.cmake`](models_sources.cmake)
globs `*/flm/*.cpp` and `*/corelib/*.cpp`; the second glob is what
`flm_corelib_aie4` compiles. Creating the folder is the whole registration step.

Phi-4's corelib side is five translation units, and the split is worth copying:

| file | what belongs in it |
|---|---|
| `<model>_corelib_gguf.cpp` | Open and validate the GGUF. Tensor lookup by name, shape checks, metadata, and the cross-validation against `config.json` / `tokenizer.json` / `tokenizer_config.json`. **No device code.** |
| `<model>_corelib_shape_plan.cpp` | Ask corelib how it wants each operator padded (`matmul_pad_shape`, `ssmlp_pad_rows`, …) and cache one row-extent record per live row count. Built once per engine. |
| `<model>_corelib_host.cpp` | The math corelib does not do: Q8 embedding-row decode, RMS norm, f32→bf16, RoPE tables. Plain CPU, unit-testable, no corelib types in the signatures. |
| `<model>_corelib_aie4.cpp` | The `causal_lm` subclass. Owns the device tensors, the stream and the decode loop. |
| `<model>_corelib_backend.cpp` | The `ModelBackend`. Construction order and execution policy. |

Keeping the GGUF and host layers free of corelib types is what lets you test them
without hardware.

---

## 3. The engine: a `causal_lm` subclass

Model it on [`phi4_corelib_aie4.hpp`](../../include/models/phi4/corelib/phi4_corelib_aie4.hpp).
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
// against it. Nothing calls this: FlmNpuBackend loads weights through the
// concrete engine type, and this engine's weights come from the GGUF package it
// was constructed with. See AutoModel/model_backend.hpp.
void <model>_corelib_aie4::load_weights(Q4NX&) { throw std::runtime_error(...); }
```

Nothing calls it: `FlmNpuBackend` calls `load_weights` through the *concrete*
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
  [`phi4_corelib_constants.hpp`](../../include/models/phi4/corelib/phi4_corelib_constants.hpp).
- Expose `bool poisoned() const noexcept`. A corelib failure mid-decode usually
  leaves device state that only a reload can clear; the backend surfaces this and
  `AutoModel` turns it into a 500 that asks for an unload/reload.
- Put every magic number in a `<model>_corelib_constants.hpp` with a comment on
  where it came from.

---

## 4. The backend: policy, not just construction

[`ModelBackend`](../../include/AutoModel/model_backend.hpp) owns one engine **and
every rule for driving it**. Its defaults describe the FastFlowLM NPU engines, so
you override only what differs. Phi-4's corelib backend
([`phi4_corelib_backend.cpp`](phi4/corelib/phi4_corelib_backend.cpp)) overrides six:

| override | corelib value | why |
|---|---|---|
| `id()` | `"corelib_aie4_gguf"` | what `--backend` matches |
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
inline flm::backend::BackendTraits corelib_aie4_traits() {
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
#if defined(FLM_ENABLE_CORELIB_AIE4)
    registry.register_backend("<family>", flm::<model>::kCorelibAie4BackendId,
                              flm::<model>::corelib_aie4_factory(),
                              flm::<model>::corelib_aie4_traits());
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
FLM_ENABLE_CORELIB_AIE4` anywhere. It loads the backend, sets up the tokenizer
(honouring `backend_->forced_eos_ids()`), applies the sampler, and delegates the
decode loop to `_shared_generate`.

Also register the FLM-side engine if the family has one:
`RegisterFlmNpu<<family>_npu>(registry, "<family>")`.

---

## 6. The catalog

Two files, and both must agree.

**[`model_list.json`](../../model_list.json)** — the entry. Phi-4 shares one tag
across both NPU generations, so the aie4 artifacts arrive as a
`platform_overrides.aie4` patch:

```jsonc
"<family>": {
  "<size>": {
    "supported_backends": ["flm_npu"],
    "supported_platforms": ["aie2p", "aie4"],
    "platform_overrides": {
      "aie4": {
        "name": "<dir name under models/>",
        "supported_backends": ["corelib_aie4_gguf"],
        "url": "...", "file_url": "...", "size": 4100140571,
        "default_context_length": 4096,
        "details": { "execution_backend": "corelib_aie4_gguf" },
        "files": ["<weights>.gguf", "tokenizer.json", "tokenizer_config.json", "config.json"],
        "file_sources": { "tokenizer.json": { "url": "...", "revision": "..." } },
        "model_info_key": "<family>-aie4:<size>",
        "ms_url": null
      }
    }
  }
}
```

Points that are easy to get wrong:

- The patch is a **JSON merge-patch**: arrays replace wholesale, and `null`
  *deletes* a key — that is what `"ms_url": null` is doing.
- `supported_platforms` is pruned at load; an entry that omits it means aie2p-only.
- `supported_backends` is what `--backend` is validated against; absent, it falls
  back to `[details.execution_backend]` or `["flm_npu"]`.
- `details.execution_backend` is the *default* for the entry — precedence is
  `--backend` > `FLM_BACKEND` > this > `flm_npu`.
- `file_sources` pins a per-file origin + revision when the weights and the
  tokenizer come from different repos (very common with GGUF mirrors).
- `model_info_key` redirects the downloader to a differently-named record set,
  which is needed exactly because the tag is shared across platforms.

**[`model_info.json`](../../model_info.json)** — one record per file, with `size`
and `sha256`. The downloader refuses anything it cannot match
([`model_downloader.cpp:46`](../../pull/model_downloader.cpp#L46)).

---

## 7. Tests

Three tiers, and the first two do not need an NPU:

- **Pure logic** — the GGUF and host layers. Shape rejection, metadata parsing,
  contract cross-validation, RMS norm and RoPE against reference values.
- **Frontend against a stub backend** — see
  [`test/phi4_corelib_aie4/test_phi4_frontend.cpp`](../../test/phi4_corelib_aie4/test_phi4_frontend.cpp).
  It defines its own empty `flm::backend::register_builtin_backends` (so it never
  links the prebuilt engine libraries) and installs stubs with
  `BackendRegistry::replace_backend`. The stub repeats the real backend's
  pre-device work verbatim, which keeps the "nothing is opened before validation"
  and "no device is created for a bad package" assertions meaningful.
- **Registry and selection** — [`test/model_backend/`](../../test/model_backend/),
  Linux-buildable: registration, duplicate ids, resolution precedence, and the
  error text for an unregistered or disallowed id.

Add a `test/<model>_corelib_aie4/` directory following the Phi-4 one. Note the
trick it uses at configure time: it synthesizes a version-bumped copy of
`ryzenai/corelib.h` and asserts the adapter **fails** to compile against it, which
is how the 0.3.0 pin is actually enforced.

---

## 8. Verify

```powershell
flm pull  <family>:<size>
flm run   <family>:<size> --backend corelib_aie4_gguf
flm run   <family>:<size> --backend bogus        # must exit 1, listing the real ids
```

Check, in order:

1. `show_profile` prints `Backend: corelib_aie4_gguf` and a `Backend detail:` line
   with the loaded corelib path.
2. Generation is coherent. Garbage output on AIE4 is almost always a shape-plan or
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

- [ ] `src/common/models/<model>/corelib/` created (no CMake edit)
- [ ] GGUF/host layers hold no corelib types
- [ ] `load_weights` is a documented throwing shim
- [ ] weight creates are serialized, with the thread hint passed
- [ ] runtime `shared_ptr` declared before the engine member
- [ ] all validation happens before `GetOrCreate`
- [ ] `BackendTraits` is `inline` in the header
- [ ] registered in `builtin_backends.cpp` under `#if defined(FLM_ENABLE_CORELIB_AIE4)`
- [ ] no `#if FLM_ENABLE_CORELIB_AIE4` anywhere in the frontend
- [ ] `model_list.json` + `model_info.json` agree, including `model_info_key`
- [ ] frozen headers untouched
