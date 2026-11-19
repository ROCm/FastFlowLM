<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# SmolVLA UI Host

C++ host that loads a [SmolVLA](https://huggingface.co/lerobot/smolvla_base)
Vision-Language-Action policy and runs one policy step: an observation (two
camera images + language instruction + robot state) in, a denoised action chunk
out. The target accelerator is an AMD NPU (NPU2 / Ryzen AI) via XRT + xclbin
kernels. The port is **partial**: the vision encoder's projections/MHA and the
language tower's q/k/v projections run on the NPU, while everything else
(norms, rope, attention math, the MLPs, and the whole action expert) still runs
on the CPU (`cpu_functions.hpp`, AVX-512 + OpenMP). The CPU path is an explicit
stand-in — each block already owns its `npu_app` handles, so swapping a block's
`cpu_func::*` calls for a real kernel is a local change.

What is complete and verified against `ref.safetensors`:

- the full **model description** (tensor names, dtypes, shapes) mirroring
  `model.safetensors`;
- the **weight-loading path** (single-file streaming into device buffers, with
  padded expert layouts);
- the **VLM prefill** (vision → connector → language) producing the language kv
  cache;
- the **action expert** (interleaved self/cross attention + SwiGLU) and the
  **flow-matching denoise** loop with its F32 adapters.

`main.cpp` reconstructs a captured observation, runs prefill, and checks the
language kv cache and the deterministic expert core (`expert.final_norm` and the
step-0 velocity `suffix.v_t`) against the reference tensors.

## Model architecture

SmolVLA is built on SmolVLM2-500M. Three sub-networks plus small adapters:

```
SmolVLA (top level, model.* BF16 adapters)
├── vlm  (model.vlm_with_expert.vlm — SmolVLM2, BF16)
│   ├── vision    (SigLIP encoder)
│   ├── language  (LLaMA-style text model, GQA; caches per-layer kv)
│   └── non-layer: connector (modality projection) + embed_tokens
└── expert (model.vlm_with_expert.lm_expert — action expert, 16 layers)
```

### Dimensions (from `config.json`)

| module   | hidden (D) | Q / K / V dim | heads / kv | head_dim | inter | layers | dtype |
|----------|-----------:|--------------:|-----------:|---------:|------:|-------:|-------|
| vision   | 768        | 768/768/768   | 12 / 12    | 64       | 3072  | 12     | BF16  |
| language | 960        | 960/320/320   | 15 / 5     | 64       | 2560  | 16     | BF16  |
| expert   | 720        | 960/320/320   | 15 / 5     | 64       | 2048  | 16     | BF16  |

- `chunk_size` = 50 action tokens; `max_action_dim` = `max_state_dim` = 32.
- Language and expert use **grouped-query attention** (GQA): Q spans all heads,
  K/V span the smaller kv-head set.
- Norms differ per tower, and `cpu_functions.hpp` carries all three:
  vision uses **biased LayerNorm** (`_layer_norm_biased`, eps 1e-6), language
  uses **LayerNorm** (`_layer_norm`, eps 1e-5), the expert uses **RMSNorm**
  (`_rms_norm`, eps 1e-5, no mean subtraction). All three eps values and every
  shape constant below live in `SmolVLA/SmolVLA_config.hpp`.

### One policy step (`SmolVLA::forward`)

```
SmolVLA::predict(observation)              // prefill, once per observation
├── state_proj(state) -> state token
└── vlm->forward
    ├── vision->forward       (SigLIP encoder over both cameras)
    ├── connector GEMM        (12288 = 768×16 -> D_text) -> image tokens
    └── language->forward     (text stack; fills kv cache, returns a view)
                              sets valid_prefix_len P (= image + text + 1 state)

SmolVLA::forward(observation)               // NUM_DENOISE_STEPS Euler steps
  predict(observation)                      // unless skip_prefill
  x = noise or sample_noise([50, 32])       // Box-Muller from rand(), srand(0)
  for step 0..N-1:  t = 1 + step*dt         // dt = DENOISE_DT = -1/N
    h = embed_actions(x, t)                 // adapter: in_proj + time MLP
    h = expert_core(h)                      // 16 interleaved expert layers
    v = read_velocity(h)                    // adapter: out_proj -> [50, 32]
    x += dt * v
  return x                                  // denoised action chunk
```

### The VLM → expert kv handoff

During prefill each language layer caches its keys and values into one shared
buffer laid out `(2, num_layers, MAX_L, DK)` — all layer **keys** first, then all
layer **values** (`MAX_L = 256`, `DK == DV == 320`). Two decisions (verified
numerically, confirmed with the user) make the cache directly consumable by the
expert:

- **Keys are cached post-RoPE**, values **pre-RoPE** (V is never roped).
- The expert reads cache layer `l` for expert layer `l`, same index.

`SmolVLA_language::valid_prefix_len` surfaces P (= 151 in the reference) so the
expert can offset its RoPE and size its attention.

### The action expert: interleaved self / cross attention

16 layers over the fixed 50-token action chunk. `self_attn_every_n_layers = 2`:

- **Even layers → self-attention.** q/k/v projected from the expert hidden state
  (720→960/320/320); q & k get RoPE with **position offset P** (the chunk sits
  after the VLM prefix). Attention runs over `concat(VLM_cache_kv[:P], expert_kv[:L])`
  — **block-causal**: query i sees all P prefix keys plus expert keys 0..i.
  Implemented as `_gqa_mha(causal=true, q_offset=P, L_kv=P+L)`. `k_proj`/`v_proj`
  are `[720→320]`.
- **Odd layers → cross-attention.** q from the expert hidden state (roped, offset
  P); k/v are the VLM cache **re-projected** through this layer's k/v proj
  (`[320→320]`), with **no RoPE on k** (the cache is already post-rope). Attends
  the P prefix only, **non-causal**: `_gqa_mha(causal=false, L_kv=P)`.

`q_proj[720→960]`, `o_proj[960→720]`, and the SwiGLU MLP (`gate/up[720→2048]`,
`down[2048→720]`) are identical across both layer types. A final `expert.norm`
(RMSNorm) closes the stack. Self and cross layers carry the same tensor set and
differ only in k/v shape, so one `SmolVLA_expert_layer` struct describes both;
they are still packed to distinct byte sizes (`self_layer_byte_size` /
`cross_layer_byte_size`).

**Padding.** Expert weights are stored padded to `PAD_ALIGN = 256` (pD=768,
pDQ=1024, pDK=pDV=512, pinter=2048); activations stay unpadded. Every expert GEMM
passes the padded input dim as `_cpu_mm`'s `w_stride` while K/N stay at the real
dims, so only the real columns are read.

### Top-level adapters (all BF16, `model.*`)

Bridge the robot action/state space and the expert/text hidden space, and add the
flow-matching time conditioning:

- `action_in_proj [32→720]`, `action_out_proj [720→32]` (biased);
- `action_time_mlp_in [1440→720]`, `action_time_mlp_out [720→720]` — the noisy
  action embedding is concatenated with a sinusoidal time embedding
  (min_period=0.004, max_period=4.0, dim=720) then run through in → silu → out;
- `state_proj [32→960]` — projects the robot state into the single state token
  appended after the last real text token.

These live on the top-level `SmolVLA` object (they are `model.*` tensors, not
under `lm_expert`) and are implemented in `SmolVLA.cpp`
(`embed_actions` / `read_velocity`), not inside the expert.

## Source layout (`src/`)

```
main.cpp                        entry: build + load + prefill + verify kv/expert
SmolVLA/
  SmolVLA.{hpp,cpp}             top level: vlm + expert + adapters; predict,
                                forward, embed_actions, expert_core, read_velocity
  SmolVLA_config.hpp            shared constants (smolvla_cfg::): image/patch
                                geometry, token counts, MAX_PREFIX_LEN, PAD_ALIGN,
                                denoise schedule, per-tower norm eps, rope theta
  cpu_functions.hpp             CPU kernels (cpu_func::): _cpu_mm, _gqa_mha,
                                _rope, _rope_inv_freq, _layer_norm{,_biased},
                                _rms_norm, _linear_biased, _sigmoid,
                                _elementwise_mul, _residual_add, _load_w16
  avx512_util.hpp               AVX-512 helper: _mm512_exp_ps_corrected
  vlm/
    SmolVLA_vlm.{hpp,cpp}       owns vision + language; connector GEMM
    SmolVLA_vlm_def.hpp         non-layer vlm descriptors (connector, embed_tokens)
    vision/
      SmolVLA_vision.{hpp,cpp}  vision module (alloc + load + forward)
      SmolVLA_vision_def.hpp    vision tensor descriptors
      SmolVLA_vision_exec.hpp   SigLIP encoder blocks + conv patch embedding
      seq_gen.hpp               smolvla_vlm_vision_seq:: NPU sequence builders
    language/
      SmolVLA_language.{hpp,cpp} text model; prefill fills the kv cache
      SmolVLA_language_def.hpp   text tensor descriptors
      SmolVLA_language_exec.hpp  GQA attn + SwiGLU MLP blocks
      seq_gen.hpp                smolvla_vlm_text_seq:: NPU sequence builders
  expert/
    SmolVLA_expert.{hpp,cpp}    action expert (alloc + load + forward)
    SmolVLA_expert_def.hpp      expert tensor descriptors (self/cross split, padded)
    SmolVLA_expert_exec.hpp     self/cross attn + MLP blocks + stack driver
python/
  bindings.cpp                  pybind11 module (smolvla_npu); the `make python`
                                entry point, in place of main.cpp
```

Outside `src/`: `verify_smolvla.py` (NPU-vs-torch harness),
`test_smolvla_py.py` / `test_smolvla_badpath.py` (extension tests), and
`packaging/` (the redistributable archive — see
[Packaging](#packaging-for-distribution)).

Shared infrastructure (`buffer`, `weight_desc`, `SafeTensors`, `npu_utils`,
`npu_xclbin_manager`) lives in `../../host_common`, included via
`-I$(HOST_COMMON_DIR)`.

### `_def` vs `_exec` vs module split

- **`*_def.hpp`** — pure model description. Holds one `weight_desc_t` per tensor
  (dtype, shape as `{in, out}`, HF name, byte offset), computes packed byte sizes,
  and knows how to `load()` a tensor into a device buffer at the right offset. This
  is where tensor **names** and **dtypes** are pinned to `model.safetensors`. The
  expert def additionally pads every dim to `PAD_ALIGN` and loads row-by-row into
  the wider stride (`load_padded`).
- **`*.{hpp,cpp}`** (the module class) — owns the `_def`, the device buffers, and
  splits work into `alloc_weights()` and `load_weights(SafeTensors&)`. All modules
  share one open `SafeTensors` handle.
- **`*_exec.hpp`** — runtime compute. Per-block structs (`*_attn_block`,
  `*_mlp_block`, self/cross attn, ...) own their scratch buffers and a `forward()`;
  a driver struct loops the layers, owns the `npu_app_manager*` for its xclbin,
  and constructs the blocks from it. Blocks that have no kernel yet take only
  their `_def` and run entirely through `cpu_func::*`.

## Weight loading

- Tensors are stored in `model.safetensors` as `[out, in]`; descriptors declare
  shape as `{in, out}` (the transpose).
- Descriptor names are the **raw HuggingFace** paths, e.g.
  `model.vlm_with_expert.lm_expert.layers.3.self_attn.q_proj.weight`.
- Everything loads from a **single** `SafeTensors(model_path + "/model.safetensors")`
  opened once in `SmolVLA::load_weights()` and passed to each sub-module.
- Expert layer buffers use `PAD_ALIGN = 256` row padding (`load_padded` reads each
  unpadded row into the wider padded stride, zeroing the pad). Vision and language
  load unpadded via direct offsets.
- `SafeTensors::load_weights` calls `exit(1)` on any missing tensor or size
  mismatch, so a clean run proves the description matches the file.

## Build & run

Two targets share every object except their entry point: `run.exe` takes
`main.cpp`, the python module takes `src/python/bindings.cpp`. Everything is
compiled `-fPIC`, so one `build/` tree feeds both and neither forces a rebuild
of the other.

First build the three xclbins (`vision`, `text`, `expert`) and put them in `XCLBINS/`, where
`run.exe` and the python module look for them. They are not stored in git. Set up the kernel
toolchain once as in [`../../FLM_Xclbin/README.md`](../../FLM_Xclbin/README.md), then:

```sh
source ../../FLM_Xclbin/setup.sh       # set MLIR_AIE_DIR and PEANO_INSTALL_DIR first
for k in vision text expert; do make -C ../../FLM_Xclbin/SMOLVLA/$k bitstream; done
mkdir -p XCLBINS && cp ../../FLM_Xclbin/SMOLVLA/{vision,text,expert}/build/xclbins/*.xclbin XCLBINS/
```

Then build and run:

```sh
make                  # -> run.exe (find src -name '*.cpp' auto-includes new files)
make python           # -> smolvla_npu.cpython-312-x86_64-linux-gnu.so
make clean            # removes build/, run.exe, smolvla_npu*.so
./run.exe -n SmolVLA          # model dir: $FLM_MODEL_PATH/models/SmolVLA (flm's models folder)
./run.exe -n SmolVLA -i 1     # -n model name, -i iterations
```

- Compiler: `g++-13`, C++20, AVX2/FMA (`-march=native`), OpenMP.
- Links against XRT (`-lxrt_coreutil -laiebu`) and Boost program_options
  (`run.exe` only — `main.cpp` is the sole user of program_options, and the
  extension resolves interpreter symbols at load time, so it links neither
  that nor `-lpython`).
- Parallel by default (`JOBS=4`); an explicit `-j` on the command line wins.

| variable | default | effect |
|---|---|---|
| `PROFILE` | 0 | `-DFLM_PROFILE`; every `forward()` prints its per-stage timing table |
| `VERBOSE` | 0 | `-DVERBOSE=1` |
| `DIST` | 0 | redistribution build: `-fvisibility=hidden -DNDEBUG -s` (see [Packaging](#packaging-for-distribution)) |
| `JOBS` | 4 | parallel compile width |
| `PYTHON` | `./.venv/bin/python` | interpreter the extension is built against |

**Objects are keyed by source path only**, so flipping `PROFILE`, `DIST`, or
`VERBOSE` between two builds does *not* trigger a rebuild — the stale objects
are silently reused. `make clean` first whenever you change one.

`main.cpp` fixes the RNG seed (`srand(0)`), builds the model, loads all weights,
prints the resolved dims / packed byte sizes, runs prefill, and prints error
metrics for the language kv cache and the expert core.

### The python module

`make python` builds `smolvla_npu` against whatever `PYTHON` points at. The
extension is named with that interpreter's ABI tag and **will not load under any
other version**, so point it at the interpreter you will import from:

```sh
./setup.sh                                  # dev venv: ./.venv from requirements.txt
make clean && make python PROFILE=0         # PROFILE=1 buries harness output
source activate.sh                          # activates .venv
python -c "import smolvla_npu; print(smolvla_npu.SmolVLA)"
```

The `.so` is copied to the directory root, so `import smolvla_npu` resolves from
here with no install step.

Note there are two `setup.sh`: this one builds the **development** venv from
`requirements.txt`, while `packaging/setup.sh` is the **recipient's** installer
and is a different script with different flags.

`requirements.txt` is not a second package list. It includes
`packaging/requirements-runtime.txt` with a `-r` line, so the development venv
gets exactly the CPU-only set a recipient gets, and adds only the four build
tools (`pybind11` for `make python`; `build`, `setuptools`, `wheel` for
`packaging/build_package.sh`). Its `--extra-index-url` line points pip at the
PyTorch CPU wheel index, which the `+cpu` torch pins need; without it pip would
not find them. Change a runtime version in `requirements-runtime.txt`, never
here, so the two venvs cannot drift apart.

## Testing

Three levels, cheapest first. All of them need an NPU device and a model
directory; run them from this directory (`FLM_Pybind/SmolVLA`) with
`source activate.sh` first — it activates `.venv`.

**1. C++ numerics** — `./run.exe -n SmolVLA`. Checks the language kv cache and
the expert core against `ref.safetensors`; see [Verification](#verification).

**2. Extension behaviour** — the two python tests cover what the CLI cannot:

```sh
python test_smolvla_py.py        # imports, singleton rule, torch tensor I/O,
                                 # bit-identical repeat with the same noise
python test_smolvla_badpath.py   # a bad model path raises instead of exit(1)
```

`test_smolvla_badpath.py` needs its own process: the singleton guard fires
before the path check while a valid instance is alive, so the path handling is
only reachable from a clean state.

**3. End-to-end vs PyTorch** — `verify_smolvla.py` runs the same observation
through both the NPU and the lerobot reference policy, driving both with the
same per-step noise so the residual is attributable to this implementation:

```sh
python verify_smolvla.py --steps 8                            # libero, 2 cameras
python verify_smolvla.py --dataset lerobot/pusht --steps 4    # 1 camera
python verify_smolvla.py --dataset lerobot/abc_130k_v3_smoke \
                         --cameras top,left_wrist             # 3 cameras, pick 2
```

| flag | default | |
|---|---|---|
| `--model_path` | `$FLM_MODEL_PATH/models/SmolVLA`, else `~/.config/flm/models/SmolVLA` | NPU model dir |
| `--model_id` | `lerobot/smolvla_base` | reference torch checkpoint |
| `--dataset` / `--episode` | `lerobot/libero` / 0 | observation source |
| `--cameras` | dataset order | override the camera→slot pairing |
| `--steps` / `--seed` | 16 / 0 | frames compared; step *t* uses `seed+t` |
| `--tol` | 0.8 | per-step error threshold — **parsed but not yet enforced** (see below) |
| `--torch_threads` | 8 | matches `cpu_func::max_prefill_thread` so timings compare |
| `--order` | `interleaved`, or `phased` under `--power` | how the two backends are sequenced (see [Ordering](#ordering-interleaved-vs-phased)) |
| `--skip_torch` | off | phased only: NPU timings with no accuracy comparison |

Any lerobot dataset works: its `meta.camera_keys` are paired **positionally**
with the checkpoint's `camera1..cameraN` slots, extra cameras are dropped with a
warning, and a state/action width that differs from the checkpoint's is
padded/sliced with a warning (both paths see the same padded vector, so the
comparison stays valid). The harness docstring lists the datasets this has been
run against, grouped by camera count with a star on the best in each group, plus
two (`lerobot/svla_so100_pickplace`, `lerobot/austin_sailor_dataset`) that are
known to compare poorly.

Datasets download to the Hugging Face cache (`$HF_HOME`, default `~/.cache/huggingface`)
on first use — set `HF_HOME` yourself to put it elsewhere — and a first run on a new
dataset can take a while before any output appears.

Output per step, and the closing summary:

```
 frame |     torch |       npu | speedup |      nMSE      cos
------------------------------------------------------------------------------
     0 |    0.479s |    0.375s |   1.28x | 2.602e-04  0.99995
       ref action[0]: [-0.41661, 0.07236, 0.58138, 2.12471, 0.29294, -0.07247]
       npu action[0]: [-0.40430, 0.08252, 0.58984, 2.09375, 0.29102, -0.07910]
     1 |    0.456s |    0.355s |   1.28x | 7.550e-04  0.99975

3 steps | action[0]: worst nMSE 8.580e-04, worst cos 0.99957

timing, mean over 2 step(s) (step 0 excluded as warmup):
  preprocess       0.6 ms   (both paths)
  prepare_*        1.3 ms   (npu path; torch does this internally)
  model          456.1 ms torch /   358.4 ms npu
  end to end     456.7 ms torch /   360.3 ms npu   (1.27x)
```

Step 0 is printed but excluded from the timing means — it pays for lazily built
caches on both sides. The `ref`/`npu action[0]` dump is printed for step 0 only.

A non-zero residual is expected and comes from the prefill gap described in
[Verification](#verification). **`--tol` is currently parsed and documented but
never compared against**: the run fails only on non-finite output, so the
"worst nMSE / worst cos" line is informational and you have to judge it
yourself. Wiring it into the exit status is open work.

#### Ordering: interleaved vs phased

`--order interleaved` (the default) alternates torch → NPU one frame at a time,
so each frame's comparison prints as it is computed. `--order phased` runs every
torch step first, then every NPU step, giving the NPU one contiguous window an
external meter can isolate; the torch policy is freed (`del policy`) before the
NPU phase begins, so nothing but the NPU path is resident during it.

**`--power` selects phased on its own** — it is the only order it can measure.
Passing `--power --order interleaved` is an **error, not a silent override**:
alternating backends re-heat each other every step, so neither is ever on a
settled rail, and quietly ignoring the flag would hand back numbers that do not
answer what was asked.

Ordering does not change peak memory. Both orders stage every step's inputs up
front (~6 MB/step for a 2-camera checkpoint) so that the dataset decode,
`preprocess()` and `prepare_*` land outside every measurement window — by the
time a phase starts, the only work left is the forward. A few hundred steps is
fine; a few thousand is not.

Phased-only flags — `--npu_repeat`, `--cooldown`, `--warmup*`, `--skip_torch` —
are rejected under an explicit `--order interleaved`. The check runs against
what you actually typed, before defaults are filled in, so `--power` implying a
warmup never trips it.

### Power measurement

`--power` adds mean/peak watts per frame and a steady-state summary:

```sh
python verify_smolvla.py --steps 16 --power
```

The rail is amdgpu hwmon **PPT** — the same package number `htop`, `btop` and
`sensors` show. It covers the whole APU (CPU + iGPU + NPU together). This
platform exposes no NPU-only rail: `xrt-smi` reports `Estimated Power : N/A`
and RAPL's `energy_uj` is root-only. So read the two columns **against each
other**, not as either backend's own draw.

`--power` switches `--order` to phased for you (see
[Ordering](#ordering-interleaved-vs-phased)) — that is what makes the two
numbers separable, since each backend then owns one contiguous window.
`--npu_repeat N` replays the NPU pass to widen that window if your meter needs
a longer sample; accuracy still comes from the first pass.

#### Warmup: why, and how long

The PPT rail does not step to its steady value when a phase begins — it ramps.
Measuring across that transient reports the ramp rather than the backend: a
cold-started phase reads low early, a hot-started one reads high early, and the
two means drift toward each other for reasons that have nothing to do with
either path. Without warmup the raw run showed torch *climbing* 27 → 58 W while
the NPU *fell* 44 → 23 W; both curves were the package's response, not the
backends.

The two phases settle on **different timescales because they settle for
different reasons**, which is why they get separate defaults:

| phase | mechanism | observed | default |
|---|---|---|---|
| torch | boost/thermal ramp — cores go to work immediately, package power follows | ~27 → ~58 W over 8–10 s, flat after | `--warmup_torch 15` |
| npu | **cpu frequency governor** (`amd-pstate-epp`) raising the host clocks the NPU path drives | 23 W plateau for 35–40 s, then steps to ~46 W | `--warmup_npu 20` |

⚠️ **The 20 s NPU default is below the observed 35–40 s step.** At 20 s the NPU
phase reads the *low* plateau (~23.8 W, verified: flat across all frames), not
the ~46 W it settles at. That is a stable, repeatable reading of the pre-ramp
state, not a noisy one — but it is a different operating point than the torch
phase is measured at, and it roughly doubles the reported torch/NPU ratio
(2.46x at 20 s vs 1.26x at 45 s). Pass `--warmup_npu 45` to measure the
post-step plateau; keep 20 s if you want the shorter run and know which
plateau you are quoting.

The NPU case is not thermal. At the two plateaus the CPU runs 2586 vs 3064 MHz
while busy% is flat (−1.2% at the *higher* plateau) and frame time is unchanged
at ~0.375 s; watts correlate +0.78 with CPU MHz and −0.31 with busy%. A
pure-CPU control run moved the opposite way while heating 39 → 53 °C, which
rules temperature out. In active mode the SMU firmware picks the operating
point from a filtered view of utilization, and the NPU path's bursty
submit/wait pattern reads as low-urgency until tens of seconds of it accumulate.

> **Note — what actually runs during warmup.** Real forwards, with the output
> thrown away. It is the same `run_torch` / `run_npu` the measured loop calls,
> so the rail sees exactly the load the measurement will see. Three details
> matter:
>
> - It **rotates over all staged frames** (`scratch[n % len(scratch)]`), not one
>   frame repeatedly — replaying a single frame would keep caches hotter than
>   the measured loop ever gets them.
> - Results go to a **scratch dict**, so warmup timings never reach `s["t_npu"]`
>   and cannot drag the reported per-frame mean toward the ramp it exists to
>   exclude.
> - The loop is deadline-based and a forward cannot be interrupted, so the
>   actual warmup **overshoots by up to one forward** (~0.5 s). That is the
>   intent: it should end on a settled rail, not exactly on the clock.
>
> Each phase prints its warmup start and the discarded-forward count, with the
> live rail reading on both lines, so you can see the ramp being absorbed.

#### Setting it

Three levels, most specific wins:

```sh
python verify_smolvla.py --power                              # 15s torch, 20s npu
python verify_smolvla.py --power --warmup_npu 45              # 15s torch, 45s npu (past the governor step)
python verify_smolvla.py --power --warmup 30                  # 30s both
python verify_smolvla.py --power --warmup 30 --warmup_npu 60  # 30s torch, 60s npu
python verify_smolvla.py --power --warmup_torch 5             # 5s torch, 20s npu
python verify_smolvla.py --power --warmup 0 --cooldown 0 --power_skip 0   # raw
```

| flag | default | |
|---|---|---|
| `--power` | off | per-frame watts + steady-state summary |
| `--warmup_torch S` | 15s with `--power`, else 0 | discarded torch forwards before the first measured torch step |
| `--warmup_npu S` | 20s with `--power`, else 0 | the same for the NPU phase |
| `--warmup S` | — | sets both at once; either per-phase flag still overrides it |
| `--cooldown S` | 20s with `--power`, else 0 | idle before and after the NPU phase (the rail decays under ~25 W in ~15 s) |
| `--power_skip N` | 2 with `--power`, else 0 | drop each phase's first N frames from the summary; backstop for whatever the warmup missed |
| `--power_period S` | 0.1 | sysfs sampling interval |
| `--idle_probe S` | 6.0 | idle tail used for the package floor; free when a `--cooldown` window already supplies one |
| `--npu_repeat N` | 1 | replay the NPU pass N times to widen its window; accuracy from pass 1 only |

The conditioning flags apply only under `--power` — a plain run keeps its old
wall-clock speed — and all require phased order, which `--power` supplies.
Budget roughly **+55 s** per `--power` run at the defaults (15 s + 20 s warmup,
20 s cooldown), or +80 s with `--warmup_npu 45`.

Per-frame rows gain three columns under `--power`, and the summary follows the
timing block:

```
 frame |     torch   mean    peak |       npu   mean    peak | speedup |      nMSE      cos
     0 |    0.464s  58.4W   59.0W |    0.381s  23.8W   24.0W |   1.22x | 2.602e-04  0.99995

SoC package power (/sys/class/hwmon/hwmon7/power1_input, the rail htop/sensors read):
  torch    58.5 W mean    59.0 W peak  net   55.0 W   (over 2 frame(s))
  npu      23.8 W mean    24.0 W peak  net   20.4 W   (over 2 frame(s))
  idle      3.4 W mean                     (over 310 sample(s), package doing nothing)
  ratio  2.46x  (torch / npu, package while each backend runs),  2.70x net of idle
  Measured after: 15s torch / 20s npu warmup, 20s cooldown between phases, first 2 frame(s) dropped
```

That is a real `--warmup_npu 20` run, so it shows the low-plateau NPU figure
and the resulting 2.46x. With `--warmup_npu 45` the same machine reports
`npu 46.1 W` and `ratio 1.26x`.

The **net** column subtracts the idle floor. Both backend figures are
package-wide and so include the machine simply being powered on — ~3–4 W here,
which is an eighth of a 24 W reading — so net is what running the model
actually costs. The trailing `Measured after:` line records the conditioning
that produced the numbers; treat a summary without it as unconditioned. The
frame counts in parentheses are post-`--power_skip`, which is why a 4-step run
summarises 2 frames.

**Reading the result.** `mean == peak` with a flat per-frame column means the
measurement is clean — but clean is not the same as post-ramp: a run that warms
up short of the governor step reads the low plateau just as flatly. A column
that *rises mid-run* is the unambiguous failure, and means the warmup landed
inside the step; raise `--warmup_npu` and re-run.

Every number here is fit to *this* host under `powersave` /
`balance_performance`. Another machine, governor, or EPP setting has its own
settling time and its own plateaus, so re-derive the warmups rather than
inheriting them — log the rail during a long fixed-load run and look for where
it flattens. `/tmp/rail2.py`-style sampling of `power1_input` alongside
`/proc/cpuinfo` MHz is enough to see it.

## Packaging for distribution

`packaging/` builds a self-contained archive for someone who gets the **binary
but not the source**. `packaging/README.md` is the recipient-facing document;
this section is how to produce and check the thing they receive.

```sh
cd packaging
./build_package.sh                  # -> dist/smolvla-npu-<version>.tar.gz
./build_package.sh --skip-build     # package the .so that is already there
./build_package.sh --offline-wheels # also vendor every dependency wheel (several GB)
```

What goes in: the compiled `.so`, the three xclbins from `XCLBINS/` (build them first), the python
harness, the doctor, `setup.sh`, `requirements-runtime.txt`, `README.md`. What
does not: any `.cpp`/`.hpp`, the Makefile, or the model directory.

| file | role |
|---|---|
| `pyproject.toml` | wheel metadata; `dependencies = []` deliberately (see the comment there) |
| `setup.py` | forces a platform wheel tag — setuptools sees no `ext_modules` and would otherwise tag `py3-none-any` |
| `smolvla_npu.py` | top-level shim; loads the `.so` out of `smolvla_npu_pkg/` and maps import errors to fixes |
| `smolvla_npu_pkg/doctor.py` | preflight checks, exposed as `smolvla-doctor` |
| `smolvla_npu_pkg/verify_smolvla.py` | the harness; `--model_path` falls back to `$SMOLVLA_MODEL_PATH` |
| `setup.sh` | recipient-side installer: venv, CPU-only torch, wheel, generated `activate.sh` |
| `requirements-runtime.txt` | the slim CPU-only set, not a pip freeze |

### Keeping the source out of the binary

The archive holding no `.cpp` is not sufficient — a normal build leaks the
source through the binary in two ways, and both are silent:

- the **dynamic symbol table** carries ~500 internal names (`SmolVLA_*`,
  `cpu_func::*`, `npu_app*`);
- every `assert()` embeds `__FILE__` plus its expression text, so `strings` on
  the `.so` prints real paths like `src/SmolVLA/expert/seq_gen.hpp`.

`strip --strip-all` removes neither. `DIST=1` does: `-fvisibility=hidden`
(504 → 175 dynamic symbols, none naming internals) and `-DNDEBUG` (7 → 0 source
paths). `build_package.sh` builds with

```sh
make clean && make python PROFILE=0 DIST=1
```

— the `clean` is mandatory, since objects are keyed by source path only and a
stale non-DIST `build/` would be reused without a word. It then runs three
guards and aborts on any of them: the wheel actually contains the `.so`, the
archive contains no source or bitstream files, and `nm -D` / `strings` on the
shipped `.so` find no internal symbols and no source paths.

`NDEBUG` also compiles out the shape asserts, which is why `DIST` is off by
default — locally you want a bad shape to fail loudly rather than corrupt
output quietly. It also means a `DIST=1` build should be re-verified
numerically rather than assumed identical to the one you tested.

Expect `-Wunused-variable` warnings under `DIST=1`: those variables only fed
asserts that `NDEBUG` removed.

### Testing the archive

Test it the way the recipient will — from the tarball, in a directory with no
source tree in sight, not from `packaging/`:

```sh
mkdir /tmp/pkgtest && tar xzf packaging/dist/smolvla-npu-0.1.0.tar.gz -C /tmp/pkgtest
cd /tmp/pkgtest/smolvla-npu-0.1.0
./setup.sh
source activate.sh
smolvla-verify --model_path $FLM_MODEL_PATH/models/SmolVLA \
               --dataset lerobot/libero --steps 2
```

`setup.sh` ends by running the doctor and propagates its exit code, so a clean
install is `RC=0` with all nine checks `[ ok ]`. Then confirm the numbers match
the source-tree run above — that is the check that the hardened build did not
change behaviour. `./setup.sh --check-only` re-runs the diagnostics against an
existing install without reinstalling anything.

The install takes no model path: the model directory is an argument to
`smolvla-verify`, so installing on a machine that does not have a copy yet is
still a complete install. The doctor reports it as `not checked` rather than a
warning, and validates one on request with `smolvla-doctor --model-path DIR`.

The doctor's CPU check is the one that matters most: `-march=native` emits
unguarded AVX-512 bf16, so a host without `avx512_bf16` dies by SIGILL with no
traceback at all. It also checks the interpreter ABI (cp312 exactly), glibc
≥ 2.38, `GLIBCXX_3.4.32` (gcc-13), XRT, `/dev/accel/accel0`, and the imports.

### Gotchas

- **The harness is copied, not linked.** `build_package.sh` copies
  `verify_smolvla.py` into `smolvla_npu_pkg/` only if one is not already there,
  because the packaged copy has its own docstring (it documents the installed
  entry points, not `make python`). After changing the harness in the source
  tree, copy it across by hand and re-apply that docstring, or the archive
  ships the old one.
- **Bump `version` in `pyproject.toml`** before re-issuing — the archive name
  comes from it, and `setup.sh` picks the newest wheel it finds, so two builds
  at the same version are indistinguishable to the recipient.
- The model directory is never packaged and never recorded at install time. It
  is validated at `smolvla-verify` startup instead, before the multi-minute
  torch reference load.

## Verification

Noise is not captured, so the full denoise output can't be bit-matched. `main.cpp`
verifies the deterministic pieces instead:

- **language kv** — post-rope K vs `vlm.layer.{l}.k_rope`, pre-rope V vs
  `vlm.layer.{l}.v_proj`, plus a per-region (image / text / state) key breakdown.
- **expert core** — feed the captured `suffix.embeds`
  (== `expert.layer.0.residual_in`) through the expert stack over the fresh kv
  cache, compare `expert.final_norm`, then `action_out_proj(final_norm)` vs
  `suffix.v_t` (denoise step 0, t=1.0).

The residual error is dominated by an upstream VLM language attention-mask
choice that is being reworked separately; the expert logic itself matches the
reference to bf16 tolerance in per-layer isolation.

## Status & TODO

- [x] Full model description mirroring `model.safetensors`.
- [x] Single-file weight loading (padded expert layouts) into device buffers.
- [x] VLM prefill (vision + connector + language) filling the kv cache.
- [x] Action expert (self/cross attention + SwiGLU) on CPU, verified.
- [x] Adapters + flow-matching denoise loop (`embed_actions` / `read_velocity` /
      `forward`).
- [x] Shape/geometry constants centralized in `SmolVLA_config.hpp`.
- [ ] Finalize the VLM language attention mask (currently non-causal; being reworked).
- [ ] Finish the NPU port: the vision blocks and the language q/k/v projections
      run on xclbin kernels; the language MHA/MLP and the whole expert are still
      `cpu_func::*`.
- [ ] Real image/text tokenization feeding `SmolVLA::predict` (currently replays
      captured reference inputs).

### Notes

- The connector input width (`smolvla_cfg::CONNECTOR_IN` = 12288 = 768 × 4²) is
  a constant in `SmolVLA_config.hpp` because `config.json` does not carry it.
- `cpu_func::_rope` takes an optional `pos_offset` so the expert can rope its
  chunk at positions P..P+L-1 while the language caller uses the default 0.

## Conventions

**Constants.** Anything that is a property of the *architecture* rather than of
one block — image/patch geometry, tokens per image, `MAX_PREFIX_LEN`,
`PAD_ALIGN`, the denoise schedule, per-tower norm eps, rope theta — lives in
`SmolVLA/SmolVLA_config.hpp` under `namespace smolvla_cfg`, and is derived from
its neighbours where possible (`TOKENS_PER_IMAGE = PATCH_GRID²`,
`DENOISE_DT = -1/NUM_DENOISE_STEPS`). Dims that `config.json` does carry (`D`,
head counts, layer counts, `chunk_size`) are read from it in the `_def` build
and never duplicated as literals.

**Naming.**

- `cpu_func::` kernels are prefixed with `_` (`_cpu_mm`, `_gqa_mha`, `_rope`).
- Descriptor accessors on a `_def` use the `*_all()` suffix: `all()` (the
  NPU-uploaded per-layer weights, in packed order), `norm_all()` (the CPU-only
  norm weights), `non_layer_all()`.
- Buffers/members holding norm weights are named `norm_*`, never `rms_*`: only
  the expert actually uses RMSNorm.
- Exec structs are `SmolVLA_<tower>_<role>_block` (e.g.
  `SmolVLA_expert_cross_attn_block`), driven by a `SmolVLA_<tower>_exec`.
- Sequence builders live in `smolvla_vlm_<tower>_seq`.
- Header guards follow the `__NAME__HPP__` double-underscore style.

## NPU execution model

Everything that runs on the accelerator goes through the same four-layer stack
from `host_common/npu_utils`:

```
npu_xclbin_manager   one per process, owns the xrt::device + hw contexts
  └── npu_app_manager    one per registered xclbin (register_xclbin(path))
        └── npu_app          one per kernel invocation site (create_app())
              └── npu_sequence   the instruction stream that app runs (app.seq())
```

Three xclbins are registered, each by the `_exec` that owns it:
`vision.xclbin` (in `SmolVLA_vision_exec`), `text.xclbin` (in
`SmolVLA_language_exec`) and `expert.xclbin` (in `SmolVLA_expert_exec`).

**Where they come from.** The bitstreams are artifacts of the kernel build, not
of the checkpoint — they go stale against the host code, never against the
weights — so they ship with the binary rather than with the model directory.
`utils::xclbin_path(name)` (`host_common/utils/xclbin_path.hpp`) resolves each
one, taking the first that exists of:

1. `$SMOLVLA_XCLBIN_DIR/<name>` — for testing a rebuilt bitstream in place.
2. `<binary dir>/XCLBINS/<name>` — the normal case. That is this directory's
   `XCLBINS/` (you build them, see [Build & run](#build--run); they are not stored in git) in the source tree, and
   `site-packages/smolvla_npu_pkg/XCLBINS/` for the installed wheel.
3. `<binary dir>/<name>` — a flat, hand-assembled install.

"Binary dir" comes from `dladdr`, not `/proc/self/exe`: under the python
extension the executable is the interpreter, so `/proc/self/exe` would point at
the venv's `python3`. Nothing here consults `model_path` — a model directory is
`config.json` + `model.safetensors`, and an `XCLBINS/` subdirectory inside one
is a leftover from the older layout that is now ignored. Failing to resolve
throws with every candidate listed, so the python binding raises instead of
killing the interpreter.

**Buffers.** A device-visible buffer comes from `create_bo_buffer<T>(n)` on
either the manager or an `npu_app`; the app it is created from determines which
context it is mapped into. Host and device views are explicitly synchronized:
`sync_to_device()` after the CPU writes a buffer the NPU will read,
`sync_from_device()` after the NPU writes one the CPU will read. Buffers that
never touch the NPU (norm weights, CPU scratch) are plain `buffer<T>(n)` and
are never synced — `SmolVLA_language::alloc_weights` allocates
`norm_weights[l]` this way while `layer_weights[l]` goes through the manager.

**Launching.** Three shapes are in use, in increasing order of control:

- `app(out, in, w)` — the `operator()` form. Builds a run, starts it, waits.
- `auto r = app.create_run(out, in, w); r.start(); … r.wait();` — split so the
  CPU can do useful work between start and wait. The language attention block
  uses this to overlap `rope(q)` with the `k_proj` launch.
- `rl.add(app.create_run(...)); rl.execute(); rl.wait();` — an `xrt::runlist`
  batches independent launches into one submission. The vision attention block
  submits q/k/v as one runlist since they share an input and have no ordering
  constraint between them.

Argument order for every mm kernel is `(output, input, weights)`.

### Sequence generation and RTP state

An `npu_app`'s sequence is generated once (at construction, or on the first call
that changes a shape) and then re-executed unchanged on every layer — the
per-layer weight buffer is a launch *argument*, not part of the sequence. So
sequence generation happens in the block constructor when the shape is fixed
(the language blocks always run at `MAX_L`) and lazily behind a `last_L` guard
when it is not (`SmolVLA_vision_attn_block` regenerates only when the image
count changes).

Runtime parameters (RTPs) are written into compute-tile data memory and persist
there until overwritten. Because the sequences are static and execute in a fixed
cycle, `smolvla_vlm_text_seq` tracks the last value it emitted per RTP slot in
function-local statics and skips writes that would be redundant. This makes the
generation order load-bearing:

```
q_proj, k_proj, v_proj, [mha], o_proj, gate_proj, up_proj, down_proj  -> loop
```

Two consequences worth knowing before touching that code:

- The cycle must **close** — whatever `down_proj` leaves in the tiles has to be
  what `q_proj` assumes. `q_proj` therefore always emits its writes on the first
  generation pass.
- A launch cannot be dropped just because its output is dead. Under
  `early_stop` the language attention block still launches `q_proj`, because
  `q_proj`'s sequence is the one carrying the `K_div_k` write that `k_proj` and
  `v_proj` depend on; skipping it leaves `down_proj`'s value from the previous
  layer in the tiles and silently corrupts k/v.

`force_rtp=true` opts out of the tracking (needed if two model instances ever
generate concurrently, since the trackers describe one NPU's tile state).

The mm and attn generators do **not** share RTP state: mm targets tile columns
2–5, attn targets 0, 1, 6, 7, at disjoint addresses. That is why `mha` running
between `v_proj` and `o_proj` cannot clobber the mm parameters.

## Profiling

`host_common/utils/prof_utils.hpp`. Compiled out entirely unless `-DFLM_PROFILE`
(set by the Makefile's `PROFILE=1`, which is the default).

`PROF_TICK` does not print — it *accumulates* into a process-wide registry keyed
by stage name, so a stage that runs once per layer is summed under one name
instead of producing 16 log lines. The shape at every call site is:

```cpp
PROF_INIT();                    // declares a local cursor __prof_t
... work ...
PROF_TICK("language.attn:qkv"); // add (now - cursor) to that stage, move cursor
... work ...
PROF_TICK("language.attn:mha");
```

`PROF_INIT()` declares a **local**, so every function that uses `PROF_TICK` must
call it first — a `PROF_TICK` without one is a compile error naming `__prof_t`.
Nested scopes each keep their own cursor, so timings do not bleed across
function boundaries.

Stage names are `<tower>.<block>:<step>` (`vision.attn:qkv`,
`expert.cross_attn:kv_reproj`, `denoise:expert_core`) so the report groups
readably.

**Registry ownership.** The registry is process-wide and `PROF_REPORT()` prints
whatever is currently in it. Each tower claims it for the span of its own
report: `SmolVLA_vision::forward` and `SmolVLA_language::forward` call
`PROF_RESET()` on entry and `PROF_REPORT()` on exit, and `SmolVLA::forward`
does the same around the whole denoise loop (so the 10 Euler steps report as
one aggregated `10x` table rather than ten copies).

That is also why `SmolVLA_vlm::forward` and `SmolVLA::predict` use the
`PROF_ACC_DECL` / `PROF_ACC` / `PROF_ACC_PRINT` triple instead: those measure
spans that *contain* a child reset, so their totals have to live in a local that
the child cannot wipe.

The registry is not thread-safe — tick from the driver thread, never from inside
an OpenMP parallel region.

## Verification harness

`main.cpp` is a fixed-input test driver, not a general entry point. It seeds
`srand(0)`, replays a captured observation out of `ref.safetensors`
(`vision.camera{1,2}.pixel_values`, the tokenizer-padded
`input.processed.observation.language.tokens`, the padded
`observation.state`), and compares against captured reference tensors with
`get_error_metrics` (cosine similarity + relative L2).

Two modes, selected by whether the prefill runs:

- **End-to-end** — `model.predict(input)` fills the kv cache from our own VLM,
  then `model.forward(input, &noise, /*skip_prefill=*/false)` denoises over it.
- **Expert-isolation** — overwrite the kv cache with the reference states
  (`vlm.layer.{l}.k_rope` into the key half, `vlm.layer.{l}.v_proj` into the
  value half) and call `forward(..., /*skip_prefill=*/true)`. This measures the
  expert + denoise loop against a ground-truth prefix, independent of the VLM
  attention-mask issue upstream.

**Gotcha when skipping prefill.** `valid_prefix_len` is set inside
`SmolVLA_language::forward`, so with `skip_prefill=true` it is still 0 unless
the prefill ran earlier in the same process. `SmolVLA::expert_core` reads it for
the RoPE offset and the attention length, so a stale 0 makes every expert layer
attend an empty prefix — the denoise still runs and still produces numbers, it
just produces the wrong ones. When injecting a cache without prefilling, set
`valid_prefix_len` to the injected P explicitly.

Since the reference noise **is** captured (`input.noise` == `suffix.x_t`, the
step-0 denoise input), the full 10-step integration is reproducible; nothing
here depends on the Box-Muller sampler in `forward`, which is only the fallback
when no noise is supplied.

Deterministic intermediates that can also be checked directly:
`expert.final_norm` (expert stack output before the adapters) and `suffix.v_t`
(step-0 velocity, i.e. `action_out_proj(final_norm)` at t=1.0).

Normalization stats are identity for this checkpoint, so the raw denoise output
compares directly against `result.pred_action` with no post-processing.

## Building on this

Adding a block, in the order the code expects:

1. **Describe the weights** in the tower's `_def.hpp` — a `weight_desc_t` per
   tensor with the raw HF name, dtype, and `{in, out}` shape. Add it to the
   right accessor (`all()` for NPU-uploaded, `norm_all()` for CPU-only,
   `non_layer_all()` for tower-level), which is what sizes the buffer and fixes
   the packing offsets. `all()`'s order **is** the byte layout, and the mm
   sequences assume each bias immediately precedes its weight.
2. **Write the block** in `_exec.hpp` as a `SmolVLA_<tower>_<role>_block` that
   owns its scratch and its `npu_app` handles, takes the manager in its
   constructor, and exposes `forward()`. A CPU-only block takes just the `_def`.
3. **Drive it** from the tower's `_exec` struct, which owns the
   `npu_app_manager*` and constructs the blocks.

Porting a CPU block to the NPU is local by design: the block already holds its
`npu_app` handles, so it is create the apps → generate the sequence → replace
the `cpu_func::*` calls with launches → add the `sync_*` calls at the boundary.
Nothing outside the block changes.

Any new architecture-level constant goes in `SmolVLA_config.hpp`, derived from
its neighbours where one exists; anything `config.json` already carries is read
in the `_def` build instead.

## Known issues

- **VLM language attention mask.** The prefix currently attends
  bidirectionally (`causal=false`). This is the dominant term in the end-to-end
  residual error; the expert path matches to bf16 tolerance when fed the
  reference kv cache instead.
- **`gen_text_attn_seq` is unused.** Retained deliberately — it is the sequence
  the language MHA port will need.
