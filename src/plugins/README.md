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
  (plugin loading, `flm::plugin_context`, the `FLM_PLUGIN` macro) and
  [`npu_utils/hook_registry.hpp`](../include/npu_utils/hook_registry.hpp)
  (`flm::hook_registry`, `flm::hook_result`).
- **Operator names and signatures**, per-model and public: each model's header
  names the operators it dispatches and the exact function signature each one
  resolves to — for Gemma4, `gemma4e_ops::op::*` and the `gemma4e_ops::*_sig_t`
  typedefs, in
  [`models/gemma4e/gemma4e_npu.hpp`](../include/models/gemma4e/gemma4e_npu.hpp).
  A plugin includes that header and binds the names it wants.
- **The resolving of those names at dispatch sites**, inside the per-model
  engine library, in the engine's own constructor. That is the only part a
  plugin author never sees.
- **The plugin**, a standalone shared library loaded at run time.

Loading is generic across models: `flm::plugin_context::npu` carries a
`flm::hook_registry` (`npu->hooks`) that exists before any model engine does,
so plugins load and bind names to it first, and each engine resolves whichever
names it declares against that same registry when it is constructed. An
engine that declares no operators is unaffected either way. Gemma4 E2B/E4B is
the only engine that declares any so far.

If `FLM_PLUGIN` is unset, no operator is overridden and behavior is unchanged:
`hook_registry::resolve()` returns the engine's own default callable untouched
when nothing was bound under that name.

## A minimal plugin

```cpp
#include "flm_plugin.hpp"                   // plugin_context, FLM_PLUGIN
#include "models/gemma4e/gemma4e_npu.hpp"   // gemma4e_ops::op::*, *_sig_t

void register_my_plugin(const flm::plugin_context& ctx) {
    ctx.npu->hooks.override_op<gemma4e_ops::dequant_sig_t>(
        gemma4e_ops::op::dequant_qkv[0],  // e_gemma4e_swa_layer
        [](bytes& dequantized, bytes& quantized, int64_t layer, int64_t padded) {
            // ...
            return flm::hook_result<ert_cmd_state>(ERT_CMD_STATE_COMPLETED);
        });
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
function the engine calls once the NPU device exists and before any model
engine is constructed:

```cpp
void register_overrides(const flm::plugin_context& ctx) {
    auto state = std::make_shared<iron_gemm_state>(ctx);
    if (!state->ready()) return;

    flm::hook_registry& hooks = ctx.npu->hooks;
    hooks.override_op<gemma4e_ops::proj_sig_t>(gemma4e_ops::op::q_swa_proj,
        [state](bytes& out, bytes& in, bytes& w, int64_t layer, int64_t padded) {
            return state->run_proj_sync(R_Q, out, in, w, layer, padded);
        });
    ...
}

FLM_PLUGIN(register_overrides)
```

Operators are named by the model, one name per app: sliding-window and global
attention run different kernels, so `iron_gemm` binds the same role to both
their names rather than one name shared across both. `k`/`v` and the fused
decode layer's `.async` names build a run to start and wait on later, so they
resolve to a different signature (`ert_cmd_state` vs `xrt::run`) than the rest.

An override is a callable matching the exact signature its name resolves to.
It is handed the buffers the engine's own operator would have received, plus
the layer index and the M or context length that call ran at, and returns a
real result, `skip()` (nothing further for the caller to start or wait on), or
`defer()` (fall through to the engine's own implementation):

```cpp
flm::hook_result<ert_cmd_state> run_proj_sync(size_t r, bytes& out, bytes& in, bytes& weights,
                                              int64_t layer_idx, int64_t padded_arg) {
    const uint32_t padded = (uint32_t)padded_arg;
    layer_entry& layer = this->layers_[(size_t)layer_idx];
    if (!layer.served_m.count(padded) || !layer.slots[r].has_value()) {
        return flm::hook_result<ert_cmd_state>::defer();
    }
    const slot& s = *layer.slots[r];
    bytes& b = this->_weights(layer, r);
    npu_app& app = this->apps_.at(shape_key{ padded, s.k, s.n, wants_gelu(r) });
    // IRON's argument order is A, B, C; the engine's is C, A, B.
    return app(in, b, out);
}
```

`defer()` is how `iron_gemm` restricts itself to the shapes it has instruction
streams for — every other layer or M falls back to the engine's own operator,
per dispatch.

Turning an operator *off* is the same mechanism with nothing in it. With
offline dequant selected, the weights are already in place, so the dequant
override defers immediately:

```cpp
flm::hook_result<ert_cmd_state> run_dequant(dequant_matrix m, bytes& /*dequantized*/, bytes& quantized,
                                            int64_t layer_idx, int64_t /*padded*/) {
    if (this->mode_ != weight_mode::dequant) return flm::hook_result<ert_cmd_state>::defer();  // the weights are already there
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
so it is built against the public tree and needs no engine source. It may
return a run for the caller to wait on, return having already done the work
(`skip()`), or defer and let the engine proceed.

What the engine keeps is the schedule: which operators exist, in what order
they run, and which buffers they are given. A plugin changes what happens at
a step, not the shape of the model.

Operators are addressed by a plain name, one per app the engine dispatches —
sliding-window and global attention are different names, since they are
different kernels, not the same one distinguished by an argument. The set of
names and their signatures is declared by the model's own public header;
`override_op<Sig>` throws at bind time if a signature does not match what the
engine later resolves under that name, and `hook_registry::check_all_resolved()`
throws if a plugin bound a name the engine never resolved at all — both are
the quickest way to find a typo.
