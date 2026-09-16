# FLMGEMM override

Runs Gemma4 E2B's prefill projections on the `flm.GEMM` and `flm.DequantBFP`
operators from [IRON](https://github.com/amd/IRON/tree/devel/iron/operators/flm),
in place of the
engine's `mm.xclbin` and `dequant.xclbin`. Both operators are open source and
compiled out of tree, so this is also the path by which a rebuilt or retuned
operator reaches a running model: build it, drop the artifacts next to the
model's xclbins, and point `FLM_PLUGIN` here.

By default it keeps the engine's own structure. Weights are dequantized per
layer per prefill chunk, exactly where `dequant.xclbin` ran, straight into the
packed bfp16ebs8 order the GEMM reads. Nothing is dequantized ahead of time: the
weights stay 4-bit in DRAM, there is no extra weight file, and the staging
buffers come to 75 MiB.

## Modes

`FLM_GEMM_MODE` trades memory for prefill time, and lets the two replacements be
measured apart. Prefill of a 247-token prompt on Strix, against 922 ms for the
untouched engine:

| mode | weights | dequant | GEMM | median | resident |
|---|---|---|---|---|---|
| `dequant` (default) | q4, per chunk on device | `DequantBFP` | `flm.GEMM` | 880 ms | 75 MiB |
| `bf16` | `model.dq_bf16`, resident | none | shipped `mm` | 732 ms | 3501 MiB |
| `bfp16` | `model.dq_bfp`, resident | none | `flm.GEMM` | 674 ms | 1969 MiB |

Read down the table: dropping the dequant from the timed path is worth about
190 ms, and replacing the kernel a further 59 ms. The default gives most of the
first back in exchange for keeping the weights 4-bit, because the prefill loop
touches every weight exactly once per request, so a per-chunk dequant does the
same total work a resident one does -- just on every request instead of once.

The resident modes need a sidecar built by `tools/dequantize.py`; see below.
`bf16` also drives the shipped `mm` overlay directly through `Gemm`, which costs
no extra hardware context because the engine has already registered that
xclbin.

## What it needs

Next to the model's xclbins, in `xclbins/<model>/`:

```
FLM_GEMM_<config>.xclbin
FLM_GEMM_M<M>_K<K>_N<N>_<config>.bin              one per shape
FLM_GEMM_M<M>_K<K>_N<N>_<config>_epigelu.bin      gate projection, fused GeLU
FLM_DequantBFP_<tag>.xclbin
FLM_DequantBFP_K<K>_N<N>_engine_<tag>.bin         one per shape
FLM_DequantBFP_K<K>_N<N>_engine_run<R>p<P>_<tag>.bin   gate and up, interleaved
```

Both operators take their shape at run time, so each needs only one xclbin and
E2B's ten weight shapes fit comfortably inside the driver's budget of 16
hardware contexts.

The `<config>` tag names the tuning an operator was built at and gains a field
whenever that gains a knob, so the plugin reads it off whichever xclbin it finds
rather than spelling it out. A stream is matched by that stem followed by
`_M<M>_K<K>_N<N>`.

A layer the plugin would otherwise serve but whose shape has no instruction
stream is a **hard error** at load, naming the layer, the projection and the
shape. Falling back silently would leave the model correct and merely slower,
which is the kind of build mistake that goes unnoticed.

The dequant streams must be built with **`qw_layout=engine`**. The engine's
loader runs `reorder_cpy` over each projection, which interleaves pairs of
32-row block-rows, and that is byte for byte the order the operator's fill would
otherwise gather out of the weights file. In engine mode the descriptor reads it
straight through, so the operator consumes the layer's weight buffer in place.

`gate` and `up` need the `run`/`period` variants because the loader alternates
them 512 out-features at a time. Which of the pair a dispatch reads is set by
where the plugin starts the buffer view: `up` at the region base, `gate` one run
later.

`q`, `k` and `v` are dequantized in **one** dispatch at their combined width,
and each GEMM reads its share through a sub-buffer -- the same thing the shipped
`mm` does with a `weight_offset`. The packed order is column-block-major over N
and the three are adjacent out-features, so they land one after another. So the
shape a layer needs is `K=hidden, N=DQ+DK+DV`, not three separate ones; layers
whose buffer holds no k or v use `N=DQ` instead.

A layer is served only if every one of its projections has both a dequant shape
and a GEMM instruction stream for the M being dispatched. Otherwise the whole
layer falls back to the engine, dequant included: splitting a layer between two
xclbins puts a hardware context switch at every crossing, which costs more than
the operators save.

## Environment

| variable | effect |
|---|---|
| `FLM_GEMM_CONFIG` | pick one GEMM xclbin by stem, when the directory holds several |
| `FLM_GEMM_OFF` | leave every projection on the engine's own operators |
| `FLM_DEQUANT_VERIFY` | compare every dequantized buffer against `model.dq_bfp` |

## Building the sidecars

```bash
cd tools
python3 dequantize.py <model_dir> --mode bf16 --only ""
IRON_PATH=<iron> python3 dequantize.py <model_dir> --mode bfp --only "mlp."
IRON_PATH=<iron> python3 dequantize.py <model_dir> --mode bfp --only "self_attn." \
    --out model.dq_bfp_attn
```

`bf16` writes what `dequant.xclbin` would have written, in the same buffer
layout, so the shipped `mm` reads it unchanged. `bfp` writes the packed form
`flm.GEMM` takes, calling the operator's own packer so the layout cannot drift
from the kernel's -- which is why that mode needs an IRON checkout and `bf16`
needs only numpy.

Both use the same container format as `model.q4nx`, so they could later be
merged into it by concatenating tensor entries.

## Checking a new shape

`FLM_DEQUANT_VERIFY=1` needs the packed sidecars `model.dq_bfp` and
`model.dq_bfp_attn` next to the model's weights; the plugin does not otherwise
read them. It compares each buffer the dequant writes against the sidecar entry
for the same tensor and reports per tensor. On E2B that is 205 checks — every
projection of every layer, the skip layers having no k or v in the engine's
buffer.

Worth doing whenever a shape, an offset or an interleave changes, because those
failures are silent: a wrong stride yields a buffer of the right size holding
real weight values in the wrong order, and the model still generates fluent text.

## Shapes and offsets

Neither is read from a manifest. One side of every projection is the hidden size
and the other follows from the q4nx tensor's byte count, so the plugin needs only
`hidden_size` from `config.json` and never has to work out whether a layer is
sliding-window or global. The one property it does infer is the double-wide MLP,
because that is what tells it the layer's weight buffer holds no k or v.

Offsets follow the order the engine's loader writes: q, then k and v where the
layer has them, o, then up and gate interleaved, then down.

M is the padded chunk length, which the engine publishes with each call. The
plugin serves the M values it found GEMM streams for and declines the rest — and
when it declines a chunk it declines that chunk's dequant too, so the engine's
own pipeline runs intact.
