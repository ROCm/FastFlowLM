# Operator plugins

This directory demonstrates the plugin mechanism for the FastFlowLM engine.
Plugins let you override individual operators in the end-to-end model without
rebuilding `flm` or any engine library.

We demonstrate it with a plugin in [`iron_gemm/`](iron_gemm) that overrides two
operators in Gemma4 E2B: dequant and matrix multiplication. It adds two options,
independently selectable:

- **Offline dequant.** Stops dequantizing the prefill weights at run time.
  Already-dequantized weights (bf16 or bfp16, depending on the option below) are
  loaded from disk and the dequant step is overridden with a no-op. It trades
  disk and memory for prefill time.
- **IRON bfp16 GEMM.** Runs the projections on an open-source matrix
  multiplication from IRON, whose source is
  [here](https://github.com/amd/IRON/tree/devel/iron/operators/flm/gemm). It
  operates on bfp16 (block floating point) rather than bf16, so it needs its
  weights in that format and the dequant operator is replaced along with it.
  The FastFlowLM v1.0.5 matrix multiplication emulates bf16 multiplication 
  using bfp16 internally, so swapping the open-source IRON bfp16
  implementation retains bit-equivalence.

Prefill of a 247-token prompt on Strix, for each combination, measured on
FastFlowLM v1.0.5:

![prefill medians](assets/prefill.png)

## What the mechanism consists of

- **Common infrastructure**, public: [`flm_plugin.hpp`](../include/flm_plugin.hpp)
  (plugin loading, `flm::plugin_context`, the `FLM_PLUGIN` macro),
  [`npu_utils/op_override.hpp`](../include/npu_utils/op_override.hpp)
  (`op_override`, `op_result`, `op_call`) and
  [`npu_utils/op_registry.hpp`](../include/npu_utils/op_registry.hpp)
  (`op_registry` and its key matching).
- **Operator names**, per-model and public: each model's header names the
  operators it dispatches — for Gemma4, `gemma4e_ops::op::*` and the
  `gemma4e_ops::key()` that composes a layer index with one, in
  [`models/gemma4e/gemma4e_npu.hpp`](../include/models/gemma4e/gemma4e_npu.hpp).
  A plugin includes that header and hooks the keys it wants.
- **The binding of those names to dispatch sites**, inside the per-model engine
  library. That is the only part a plugin author never sees.
- **The plugin**, a standalone shared library loaded at run time.

Loading is generic across models. *Declaring* operators is opt-in per model:
`causal_lm::ops()` returns `nullptr` by default, and Gemma4 E2B/E4B is the only
engine that declares any so far. A plugin loaded against any other model
registers nothing.

If `FLM_PLUGIN` is unset, no operator is overridden and behavior is unchanged.
The engine builds its registry either way; a dispatch with no hook installed
costs one null check.

## A minimal plugin

```cpp
#include <memory>

#include "flm_plugin.hpp"                   // op_override, plugin_context, FLM_PLUGIN
#include "models/gemma4e/gemma4e_npu.hpp"   // gemma4e_ops::key, gemma4e_ops::op

class my_dequant : public flm::op_override {
public:
    flm::op_result create_run(const flm::op_call& call) override {
        // ...
    }
};

void register_my_plugin(const flm::plugin_context& ctx) {
    int layer = 0;
    ctx.ops->override_op(gemma4e_ops::key(layer, gemma4e_ops::op::dequant_qkv),
                         std::make_shared<my_dequant>());
}

FLM_PLUGIN(register_my_plugin)
```

It needs no engine source and no engine library to link against:

```bash
g++ -std=c++20 -fPIC -shared -O2 \
    -mavx -mavx2 -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma \
    -I<flm>/src/include -I/opt/xilinx/xrt/include \
    -Wl,-Bsymbolic-functions \
    my_plugin.cpp -o my_plugin.so
```

Neither flag group is optional. `typedef.hpp` has inline functions returning
`__m256`/`__m512`, so compiling without the AVX flags changes their ABI relative
to the engine. And `npu_app` and the classes around it are header only, so every
engine library carries its own weak copy of their inline code;
`-Wl,-Bsymbolic-functions` is what keeps the plugin's calls bound to its own.

Then load it — `:`-separated (`;` on Windows), so several may load at once:

```bash
FLM_PLUGIN=/path/to/my_plugin.so flm serve gemma4-it:e2b
```

## Overriding an operator

A plugin is a shared library with one entry point. `FLM_PLUGIN` names the
function the engine calls once the model exists and before its weights load:

```cpp
void register_overrides(const flm::plugin_context& ctx) {
    auto hook = std::make_shared<iron_gemm_override>(ctx);
    const size_t bound = hook->bind(*ctx.ops, hook);
    ...
}

FLM_PLUGIN(register_overrides)
```

Operators are named by the model. `iron_gemm` binds itself to each projection of
each layer, and to the dequant steps that feed them:

```cpp
bound += ops.override_op(gemma4e_ops::key((int)layer, names[r]), self);

for (std::string_view dq : { gemma4e_ops::op::dequant_qkv, gemma4e_ops::op::dequant_o,
                             gemma4e_ops::op::dequant_gate, gemma4e_ops::op::dequant_up,
                             gemma4e_ops::op::dequant_down }) {
    ops.override_op(gemma4e_ops::key((int)layer, dq), self);
}
```

An override implements one method. It is handed the buffers the engine's own
operator would have received, and runs whatever it likes:

```cpp
flm::op_result create_run(const flm::op_call& call) override {
    ...
    // IRON's argument order is A, B, C; the engine's is C, A, B.
    app(*call.args[1], b, *call.args[0]);
    return flm::op_result();
}
```

Returning `flm::op_result::decline()` instead hands the call back to the engine,
per dispatch — which is how `iron_gemm` restricts itself to the shapes it has
instruction streams for.

Turning an operator *off* is the same mechanism with nothing in it. With offline
dequant selected, the weights are already in place, so the dequant override
returns having done nothing:

```cpp
void _dequant(const flm::op_call& call, layer_entry& layer) {
    if (this->mode_ != weight_mode::dequant) return;  // the weights are already there
    ...
}
```

## What it needs on disk

Next to the model's xclbins, both operators' artifacts — one xclbin each, since
they take their shape at run time, plus one instruction stream per shape:

```
FLM_GEMM_<config>.xclbin                    FLM_DequantBFP_<config>.xclbin
FLM_GEMM_<config>_M<M>_K<K>_N<N>.bin        FLM_DequantBFP_K<K>_N<N>_engine_<config>.bin
```

`<config>` names the tuning the operator was built at, so the plugin reads it
off whichever xclbin it finds rather than spelling it out. A shape the plugin
would otherwise serve but has no stream for is a hard error at load; an M
nothing was built for is a run-time fallback to the engine.

| variable | effect |
|---|---|
| `IRON_GEMM_MODE` | `dequant` (default), `bf16` or `bfp16` — which weights the GEMM reads |
| `IRON_GEMM_OFF` | register nothing, leaving FastFlowLM's operators in place |
| `IRON_GEMM_CONFIG` | pick one GEMM xclbin by stem, when several are present |
| `IRON_DEQUANT_VERIFY` | check every dequantized buffer against the sidecar, byte for byte |

## Prefill weights

The two offline modes read weights a converter dequantized ahead of time, next
to `model.q4nx`. `--mode bf16` writes what `dequant.xclbin` would have written,
in the layout FastFlowLM's GEMM reads (3.45 GiB); `--mode bfp` writes the packed
form the IRON bfp16 GEMM takes (1.94 GiB), calling that operator's own packer so
the layout cannot drift from its kernel's.

They are also the reference for `IRON_DEQUANT_VERIFY=1`, which checks every
buffer the dequant produces against them, byte for byte.

## Reproducing the numbers

With an IRON checkout set up and its environment sourced:

```bash
IRON=<iron checkout>  MODEL=<model dir>  FLM=<this checkout>/src
TOOLS=$FLM/plugins/iron_gemm/tools     # the three scripts below live here

# 1. build both operators, 22 shapes for Gemma4 E2B
cd $TOOLS && IRON_PATH=$IRON python3 $TOOLS/build_artifacts.py
cp build/FLM_*.xclbin build/FLM_*.bin $FLM/xclbins/Gemma4-E2B-IT-NPU2/

# 2. only for the offline modes: pre-dequantized weights
python3 $TOOLS/dequantize.py $MODEL --mode bf16 --only ""
IRON_PATH=$IRON python3 $TOOLS/dequantize.py $MODEL --mode bfp --only "mlp."
IRON_PATH=$IRON python3 $TOOLS/dequantize.py $MODEL --mode bfp --only "self_attn." \
    --out model.dq_bfp_attn

# 3. build flm with the plugin, and serve
cd $FLM/build && cmake -DFLM_BUILD_PLUGINS=ON .. && ninja flm iron_gemm_plugin
FLM_PLUGIN=$PWD/plugins/iron_gemm/iron_gemm_plugin.so ./flm serve gemma4-it:e2b
```

Send a 247-token prompt to `/api/generate` and read `prompt_eval_duration`,
setting `IRON_GEMM_MODE` for each configuration and `IRON_GEMM_OFF=1` for the
unmodified engine. Discard the first response and take the median of the rest; run-to-run spread is about 2%, so differences below
~40 ms need several runs to see.

`$TOOLS/plot_prefill.py` draws the chart above from one `<label>: median <ms>`
line per configuration on stdin.

## What a plugin can do

An override owns its dispatch completely. It registers its own xclbins through
the `npu_xclbin_manager` it is handed, creates its own apps, loads its own
instruction streams and allocates its own weights — all through public headers,
so it is built against the public tree and needs no engine source. It may return
a run for the caller to wait on, return having already done the work, or decline
and let the engine proceed.

What the engine keeps is the schedule: which operators exist, in what order they
run, and which buffers they are given. A plugin changes what happens at a step,
not the shape of the model.

Operators are addressed by key, with `*` matching a whole dot-separated segment.
The key space is the model's own — Gemma4 spells it `layers.<i>.<operator>`, so
`layers.*.mlp.up_proj` takes that projection on every layer — but nothing in the
API requires layers, or any particular shape. The set is declared by the engine,
so `override_op` on an unknown key throws at registration rather than silently
never firing; the message lists every declared key, which is the quickest way to
discover a model's vocabulary. `list_ops()` returns it.
