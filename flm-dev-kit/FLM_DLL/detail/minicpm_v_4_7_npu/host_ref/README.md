<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# minicpm_v_4_7_npu host reference and scoring harness

The CPU path of `libminicpm_v_4_7_npu.so` (ADDING_A_NEW_MODEL.md S5 step 1) and the tools that
score it. One command runs both S3 gates:

```bash
bash FLM_DLL/detail/minicpm_v_4_7_npu/host_ref/run_gate.sh
```

It builds the engine with every stage on the host at `DEBUG_LEVEL=1`, regenerates S2's dumps
and the q4 oracle, runs the engine on S2's prompt, and scores it. It exits non-zero on any
failure. Takes about a minute.

## Backend switches

Defined in `../minicpm_v_prefill_cpu.hpp`. They are opt-out: a plain `make` is all-NPU.

| make flag | macro | stage | overlay | owner |
|---|---|---|---|---|
| `PREFILL_MM=cpu` | `MINICPM_V_MM_ON_NPU` | every prefill projection | `mm.xclbin` | S5 |
| `PREFILL_ATTN=cpu` | `MINICPM_V_ATTN_ON_NPU` | full-attention layers' attention | `attn.xclbin` | S5 |
| `PREFILL_GDN=cpu` | `MINICPM_V_GDN_ON_NPU` | gated delta rule | `GateDeltaNet_prefill.xclbin` | S5 |
| `PREFILL_CONV=cpu` | `MINICPM_V_CONV_ON_NPU` | causal conv1d + silu + q/k L2 norm | `conv.xclbin` | S5 |
| `DECODE=cpu` | `MINICPM_V_DECODE_ON_NPU` | decode **and the lm_head** | `layer.xclbin`, `lm_head.xclbin` | S4 |

Each stage is a pair, `_<stage>_npu()` and `_<stage>_host()`, over the same buffers. The NPU
helper returns `false` when its macro is 0 or when it declines at run time, and the caller
then runs the host kernel. Both leave the stage's output valid on host and device, so any
single macro can be set to 0 by hand. All 32 combinations are meant to work; the nine checked
are the default, each flag alone, all five together at `DEBUG_LEVEL=0` and `1`, and the four
prefill flags together. All nine build.

`MINICPM_V_SCRATCH_ON_DEVICE` is MM || ATTN || GDN || CONV. When it is 0, the prefill scratch
is host memory and `scratch_push/pull()` are no-ops. The weights, caches, `x` and `logits` are
**device BOs in every configuration**. `minicpm_v_desc::load_*_weights()` (a frozen contract)
calls `sync_to_device()` on them unconditionally. So the all-host build still opens the device
but registers no xclbin and opens no hw context.

The Makefile stamps the flag set into `build/minicpm_v_4_7_npu/lib/.cxx_flags`. Changing any flag
rebuilds every object instead of relinking stale ones.

`DECODE=cpu` decodes a token by running the prefill graph at `L = 1`, then a host lm_head. The
conv state, recurrent state and kv cache it reads and writes are the array's buffers in the
array's layout, so it is the golden reference for S4.

## Files

| file | what |
|---|---|
| `../minicpm_v_prefill_cpu.hpp` | the switches, scratch discipline, and every host kernel (`namespace minicpm_v_cpu`) |
| `../minicpm_v_prefill.hpp` | the prefill graph, split into `_npu` / `_host` stage pairs |
| `run_engine.cpp` | drives the `.so`: full prefill, greedy decode, prefill+decode split, multi-turn chunks |
| `q4_oracle.py` | S2's reference re-run on the weights in `model.q4nx` → `models_simple/dumps/minicpm_v_q4/` |
| `compare_engine.py` | scores the engine's dumps against both references |
| `run_gate.sh` | all of the above |

`run_engine` with `FLM_DUMP_DIR=<dir>`, against a `DEBUG_LEVEL >= 1` engine, writes every
manifest tensor it has as `<dir>/layers/NN/<name>.npy`. Dumps are hooked at the shared stage
boundaries, so they work with the NPU stages on too. The exceptions are `linear_conv`,
`linear_z`, `mlp_up` and `mlp_gate`, which only the host stages expose.

## What "matches" means

The engine runs Q4_K weights; S2's dumps come from bf16 weights. Quantization alone caps the
`residual_out` cosine at 0.958 by layer 23 and the logits at 0.947 (the Q4_K quantization floor). So the gate scores against the
q4 oracle (same weights, fp32 activations) and prints the bf16 column next to it:

| | vs q4 oracle | vs bf16 dumps | q4 floor |
|---|---|---|---|
| worst `residual_out` (layer 23) | 0.99986 | 0.95838 | 0.95825 |
| last-row logits | 0.99963 | 0.90150 | 0.89812 |
| decode rows 11..19 (prefill 12, decode 8) | ≥ 0.99926, argmax all equal | | |
| multi-turn 12 + 2 + 6 | 0.99998 / 0.99963, argmax equal | | |

The q4-floor column scores the q4 oracle's own logits against the bf16 dumps. For the full
`[L, V]` logits that is the 0.947 above; for the last row alone it is 0.898.

Greedy continuation: *"Hmm, the user wants a short introduction to large language models. I
should keep it concise but informative. …"* The first token, 77264, matches HF `generate`.

## Numerics that are not obvious

* **Round, don't truncate.** `typedef.hpp`'s `f32o_bf16_512` truncates. Every host store here
  goes through `cpu_func::f32_bf16_rne_512` instead. With truncation the same run scored
  0.9977 / 0.995 (`f32o_bf16_512` truncates instead of rounding).
* **The conv overlay does the L2 norm.** `conv.cc` applies silu and then `l2_norm` over the q
  and k heads, so `conv_qkv` holds normalized q/k. The GDN overlay scales q by `1/sqrt(128)`
  itself. The host stages split the same way. `linear_conv.npy` is dumped between silu and the
  norm, which is where the reference takes it.
* **`alpha_beta` holds `exp(g)` and `sigmoid(b)`**, not `g` and `b`. The recurrent state is
  fp32 `[16][K=128][V=128]` at bf16 element offset `get_s_offset()` of the cache slot, behind
  the `[3][6144]` conv state.
* **Offsets are bytes**, not elements, in the weight descriptors.
* **Threads.** `prefill_threads()` defaults to one per physical core (16 here) and can be
  overridden with `FLM_PREFILL_THREADS` (32 threads, one per SMT thread, oversubscribe the
  host kernels). All-host, `DEBUG_LEVEL=0`:
  a 1k prompt prefills in about 1.0 s and decode is about 47 ms/token.

## Not covered by the NPU-vs-host cross reference

These stages are the same `cpu_func` call on both backends, so a cross reference cannot see a
bug in them. Only the S2 comparison above covers them:

* the RMS norms, q/k norm + RoPE (`_rope_rms_batch`), the gated norm, the alpha/beta
  projections and their activation, the attention gate sigmoid and multiply, the residual adds;
* `fill_kv_cache` and the conv-state carry in the linear block;
* on the NPU MLP path, `_exp_glu` over the array's interleaved up/gate slabs. The host path
  de-interleaves instead (`dequant_mm_upgate`, then `glu_mul`), so the two MLP paths differ in
  the GLU step.
