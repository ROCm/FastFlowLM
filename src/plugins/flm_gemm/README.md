# FLMGEMM override

Runs Gemma4 E2B's prefill projections on the FLMGEMM operator from
[IRON](https://github.com/Xilinx/mlir-aie), in place of the engine's own GEMM.
The operator is open source and compiled out of tree, so this is also the path
by which a rebuilt or retuned operator reaches a running model: build it, drop
the artifacts next to the model's xclbins, and point `FLM_PLUGIN` here.

The override reads weights it packed itself, which means the engine's
dequantization step has nothing left to feed and the plugin switches it off.
That accounts for most of what it saves; the operator itself accounts for the
rest.

## What it needs

Next to the model's xclbins, in `xclbins/<model>/`:

```
FLM_GEMM_<config>.xclbin
FLM_GEMM_M<M>_K<K>_N<N>_<config>.bin          one per shape
FLM_GEMM_M<M>_K<K>_N<N>_<config>_epigelu.bin  gate projection, fused GeLU
```

Next to the model's weights:

```
model.dq_bfp        packed mlp weights, one U8 tensor per projection
model.dq_bfp_attn   packed attention weights
```

Tensors are named `model.layers.<i>.<proj>.weight.dq_bfp`, holding bfp16ebs8:
nine bytes per eight values, a shared exponent followed by eight mantissas, in
the order the operator reads B. `tools/replace-gemm/sidecar.py` in the internal
tree produces them.

A layer is served only if every one of its seven projections has both a packed
weight and an instruction stream for the M being dispatched. Otherwise the whole
layer falls back: splitting a layer between two xclbins puts a hardware context
switch at every crossing, which costs more than the operator saves.

## Environment

| variable | effect |
|---|---|
| `FLM_GEMM_CONFIG` | operator configuration tag, default `tn64_ma32_emf_floor_npu2` |
| `FLM_GEMM_OFF` | leave every projection on the engine's own GEMM |

## Shapes

K and N are not read from a manifest. One side of every projection is the
hidden size, and the other follows from the packed byte count, so the plugin
needs only `hidden_size` from `config.json` and never has to work out whether a
layer is sliding-window or global, skip or not.

M is the padded chunk length, which the engine publishes with each call. The
plugin serves the M values it found instruction streams for and declines the
rest.
