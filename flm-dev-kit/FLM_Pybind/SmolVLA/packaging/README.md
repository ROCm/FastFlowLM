<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# SmolVLA on the AMD NPU

Prebuilt python package for running the [SmolVLA](https://huggingface.co/lerobot/smolvla_base)
vision-language-action policy on an AMD NPU (Ryzen AI / XDNA), plus a harness
that compares it against the PyTorch reference.

The NPU extension ships **compiled** — there is no build step and no compiler
required.

## Requirements

| | |
|---|---|
| OS | Linux x86-64, glibc ≥ 2.38 (Ubuntu 24.04 or newer) |
| Python | **3.12 exactly** — the extension is a cp312 binary |
| CPU | AVX-512 with `avx512_bf16` (Zen 4 / Ryzen AI, or newer) |
| NPU | XDNA driver (`/dev/accel/accel0`) + XRT runtime |
| Model | a SmolVLA model directory, supplied separately (~3.3 GB) |

The CPU requirement is not negotiable: the extension is compiled with
`-march=native` on Zen 4 and issues AVX-512 bf16 instructions unguarded. On a
CPU without them it dies with `Illegal instruction` (SIGILL) and no traceback.
`smolvla-doctor` checks this before you hit it.

## Install

```sh
tar xzf smolvla-npu-0.1.0.tar.gz && cd smolvla-npu-0.1.0
./setup.sh
```

That creates a venv, installs a **CPU-only** torch (no CUDA wheels), installs
the package, writes `activate.sh`, and runs the environment check. The model
directory is not needed here — nothing in the install reads it.

Useful flags: `--venv DIR`, `--python /usr/bin/python3.12`, `--check-only`
(re-run diagnostics on an existing install), `--offline` (install from a
bundled `wheels/` directory on an air-gapped host). `./setup.sh --help` lists
them all.

## The model directory

Not included — it is several GB. Copy it from whoever sent you this package. It
must look like:

```
SmolVLA/
├── config.json
└── model.safetensors
```

That is all of it. The NPU bitstreams are **not** part of the model directory —
they ship inside the wheel, because they track the kernel build rather than the
checkpoint. If your copy has an `XCLBINS/` subdirectory it is left over from an
older layout and is ignored.

It is a **run-time argument**: pass `--model_path` on each run. If you would
rather not repeat it, `export SMOLVLA_MODEL_PATH=/path/to/SmolVLA` (the
commented line in `activate.sh`) supplies the default. Either way the layout is
checked at startup, before the multi-minute torch load, so a wrong path fails
in a second with the list of what is missing.

To check a copy without running anything:

```sh
smolvla-doctor --model-path /path/to/SmolVLA
```

## Run

```sh
source activate.sh
smolvla-verify --model_path /path/to/SmolVLA --dataset lerobot/libero --steps 4
```

Each step prints the NPU-vs-torch difference for the first action of the chunk
and the wall-clock for both paths:

```
[ 0] frame     0  action[0] max 0.10021 cos 0.99747 nmse 0.005  e2e 0.492s torch / 0.440s npu
```

`max` is max absolute error, `cos` cosine similarity, `nmse` error energy
relative to the reference. A non-zero residual is expected — see
`smolvla-verify --help` for why, and for the `--tol` threshold that separates
"known prefill gap" from "something changed".

Datasets are downloaded from the HuggingFace Hub on first use, into
`$HF_HOME` (set by `activate.sh` to a cache inside the install directory).

### Datasets

Any lerobot dataset works. Its cameras are paired positionally with the
checkpoint's `camera1..cameraN` slots; `--cameras top,left_wrist` overrides the
pairing. A state/action width that differs from the checkpoint's is padded or
sliced, with a warning. Some that have been run:

| cameras | dataset |
|---|---|
| 1 | `lerobot/pusht`, `lerobot/berkeley_mvp`, `lerobot/aloha_sim_insertion_human` |
| 2 | `lerobot/libero`, `lerobot/libero_plus`, `lerobot/austin_buds_dataset` |
| 3 | `lerobot/droid_100`, `lerobot/abc_130k_v3_smoke`, `lerobot/aloha_mobile_wipe_wine` |

`lerobot/svla_so100_pickplace` and `lerobot/austin_sailor_dataset` are known to
compare poorly.

## Using the module directly

```python
import torch, smolvla_npu

model = smolvla_npu.SmolVLA("/path/to/SmolVLA")
chunk = model.forward(
    images,   # list of float32 [3,512,512] tensors, one per camera, in [-1,1]
    tokens,   # prompt token ids
    state,    # float32 robot state
    noise,    # optional float32 [chunk_size, action_dim]
)            # -> float32 [chunk_size, action_dim]
```

**One model per process.** The NPU hardware context cannot be shared, and it
cannot be re-initialised after a `forward()` — constructing a second instance
raises, even after deleting the first. Use a fresh process.

## Troubleshooting

Run `smolvla-doctor` (or `./setup.sh --check-only`). It checks the interpreter
ABI, CPU instruction set, glibc/libstdc++ versions, XRT, the NPU device node and
the python imports, and prints the fix for whatever it finds. Add
`--model-path DIR` to validate a model directory as well.

| symptom | cause |
|---|---|
| `Illegal instruction` / SIGILL, no traceback | CPU lacks AVX-512 bf16; needs a rebuild for your CPU |
| `libxrt_coreutil.so.2: cannot open shared object file` | XRT not installed or not on `LD_LIBRARY_PATH` |
| `GLIBCXX_3.4.32 not found` | libstdc++ older than gcc-13 |
| `undefined symbol: PyInit_...` / won't import | not running python 3.12 |
| constructing `SmolVLA()` fails | no NPU device, or a second instance in one process |
