# Operator plugins

A plugin replaces individual operations of a model engine with your own
implementation, without rebuilding `flm` or any of the engine libraries. The
engine declares the operations it runs; a plugin binds an override to the ones
it wants and dispatches them however it likes.

```
flm serve gemma4-it:e2b            # engine's own operators
FLM_PLUGIN=./my_plugin.so flm serve gemma4-it:e2b   # yours, where you bound them
```

`FLM_PLUGIN` takes several paths separated by `:` (`;` on Windows). They are
loaded in order, after the engine is constructed and before its weights are
read, which is the only window in which overrides may be registered.

## Writing one

```cpp
#include "flm_plugin.hpp"

class my_up_proj : public flm::op_override {
public:
    flm::op_result create_run(const flm::op_call& call) override {
        // call.args are the buffers the default receives, in its order.
        // call.layer and call.extent say which layer and how many rows.
        my_app_(*call.args[1], my_weights_[call.layer], *call.args[0]);
        return flm::op_result();          // done, nothing to wait on
    }
};

static void register_overrides(const flm::plugin_context& ctx) {
    ctx.ops->override_op("layers.*.mlp.up_proj", std::make_shared<my_up_proj>(ctx));
}
FLM_PLUGIN(register_overrides)
```

An override owns everything it needs: it registers its own xclbin through
`ctx.npu`, creates its own apps, loads its own instruction streams and allocates
its own weights. `flm_gemm/` is a complete worked example.

### Keys

A key is `layers.<i>.<role>`, or just `<role>` for a model-level operation. `*`
in place of a whole dot-separated segment matches every value of it. The roles
an engine may declare are listed in `flm::role`; which of them it actually
declares is per model and documented on the model's header. `list_ops()` returns
the set at run time, and `override_op` throws on a key that matches nothing, so
a typo fails at registration rather than running unaccelerated.

### Returning from `create_run`

| return | meaning |
|---|---|
| `op_result(run)` | the caller starts and waits on this run |
| `op_result()` | the work is already finished |
| `op_result::decline()` | not served, run the default for this call |

`decline()` is how an override restricts itself to the shapes it supports; the
engine falls back per call, with no configuration.

`call.blocking` tells you the caller will do nothing until the work completes,
so you may run it synchronously and return an empty result instead of building a
run object.

### Disabling a step

Overriding a step with something that returns an empty `op_result` removes it.
`flm::no_op_override` does exactly that. This is how a plugin that brings its own
weights switches off the engine's dequantization — and note the hazard: if
anything still reads the buffers that step filled, it reads stale data. An
override that declines some calls must therefore keep the dequant step alive for
exactly those calls, which is easiest to get right by having one object serve
both, as `flm_gemm/` does.

Removing a step does not free the buffers it wrote. The engine cannot know that
an override will never decline, so the staging stays allocated.

## Building one

Follow `flm_gemm/CMakeLists.txt`. Two requirements:

**Link with `-Wl,-Bsymbolic-functions`.** `npu_app` and the classes around it are
header only, so every engine library carries its own copy of their inline code
as a weak symbol. Without this flag your calls bind to whichever library the
loader saw first, which may have been built against a different revision of
those headers, and the objects you construct are laid out for someone else. The
failure is a crash inside XRT with no obvious cause.

**Match the toolchain.** A plugin and `flm` exchange C++ objects across the
`dlopen` boundary, so both must be built with the same compiler and standard
library — the same constraint that already holds between `flm` and the engine
libraries it links. `flm_plugin_abi_version` catches a plugin built against a
different override interface, but nothing catches a toolchain mismatch.
