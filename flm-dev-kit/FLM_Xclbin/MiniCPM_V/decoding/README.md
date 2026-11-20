<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# FLM Decoding Layer generator

> **Status:** this design is built for the MiniCPM-V 4.7 1B shape (hidden 1024, the 0.8B Qwen3.5 layout) and nothing else is
> assigned from it. It started as the Qwen3.5-2B decoding layer generator (it was `FLM_Xclbin/Qwen3_5_2B/qwen3_5_decoding`) and was moved
> here when the 2B model was removed; the sources are unchanged by the move, so the 2B shape and the model headers stay in the
> design family. `MODEL_TYPE=QWEN3_5_08B_SL` builds `layer_lm_sl.xclbin` (the decode with the lm_head as a phase of the layer design;
> every run takes x from and returns it to DDR, so a firmware design reload between runs cannot change the result). This is the only
> variant CI builds and ships. `MODEL_TYPE=QWEN3_5_08B` still builds the stateful variant (`layer_lm.xclbin`, x stays on the array
> inside a layer group) for local experiments; it is neither built by CI nor assigned to a model folder. See
> `FLM_Xclbin/kernel_assignment.md` for the built variants and where the artifacts go.

Only for generateing decoding layers on AMD Ryzen AI Strix+ chips.


## Constraints

- Must be traditional layer, sliding window is allowed

## Models

| Model Series | Head Dim | Num Heads | Num KV Heads | Num Heads per group | Num heads per CU | KV Bias | QK Norm | intrinsic mm | Pad Q | Note                       |
|--------------|----------|-----------|--------------|---------------------|------------------|---------|---------|--------------|-------|----------------------------|
| Llama        | 64, 128  | 32        | 8            | 8                   | 2 x (4 -> 1)     | No      | No      | 8x8x8        | No    | single 8x8 q, different k  |
| Llama-3B     | 256      | 24        | 8            | 8                   | 2 x (3 -> 1)     | No      | No      | 8x8x8        | Yes   | row 3, 7 are removed       |
| Qwen-small   | 128      | 16        | 8            | 2                   | 2 x (2 -> 1)     | No      | Yes     | 8x8x8        | Yes   | row 2,3,6,7 are removed    |
| Qwen-large   | 128      | 32        | 8            | 8                   | 2 x (4 -> 1)     | No      | Yes     | 8x8x8        | No    | single 8x8 q, different k  |
| *Gemma-small | 256      | 4         | 1            | 4                   | 4 -> 1           | No      | Yes     | 4x8x8        | No    | Only one CU is used        |
| Gemma-large  | 256      | 8         | 4            | 2                   | 2 -> 1           | No      | Yes     | 4x8x8        | Yes   | zero pad to 4x8 in kernel  |


## Attention implements

For kernels

| Type | Kernel spec  | intrinsic mm |
|------|--------------|--------------| 
| 1    | 2 x (4 -> 1) | 8x8x8        |
| 2    | 4 -> 1       | 4x8x8        |

Dataflow of Q:

- Always from (NH, DH) -> 4 x (NH // 4, DH)
- NH always pads to multiple of 4
- DIM: ((DH // 8, 8), (NH // 4, DH),(8, 1))

Inside Kernel:

- Requires process padding and remove useless data according to parameters

## Compiler (2026-11-03): Peano 22

Already Peano. Rebuilt with the CI pin (Peano = `llvm-aie 22.0.0.2026090201+a36c62b9`; the June 21.0 used before is older.): prefill logits bit-identical to the June build; decode not bit-identical
but as close to the CPU reference (`--check`: worst cos 0.998865 both, 0.998100 vs 0.997611 at `--kv-scale 8`), and
10K decode 31.4-31.5 vs 31.2 tok/s. The shipped layer.xclbin is this build.

## Other shapes: `MODEL_TYPE=QWEN3_5_08B` (MiniCPM-V-4.7 1B, `layer_lm.xclbin`)

The design is not tied to the 2B: `make bitstream MODEL_TYPE=QWEN3_5_08B` builds it for hidden 1024 / ffn 3584 (the Qwen3.5-0.8B shape that
`minicpm-v-4.7:1b` uses). The stateful result is `layer_lm.xclbin` (not shipped; the shipped variant is `layer_lm_sl.xclbin` of `MiniCPM-V-4.7-1B-NPU2`): the FLM_DLL engine
`minicpm_v_4_7_npu` runs the Q8_0 lm_head as a run of this design (mode 2), so a decode token stays in one xclbin / hw context (the old layer.xclbin +
lm_head.xclbin pair switched twice per token, ~2 x 2.4 ms). The engine uses it when the folder has it (`FLM_MINICPM_LM_IN_LAYER=0` forces the
two-xclbin path); the old `layer.xclbin` stays untouched because an engine that does not write the mode RTPs would wait on lock 6 forever.

* `RTP_rms_mode` is pinned (`iron/rms_residual.py`, `use_address`): with hidden 1024 the rms core's buffers are smaller and the RTP would land below
  36864, where the host writes it for every model size. The allocator adds 32 to a pinned address, so the pin is 36832; the build checks the final address.
* Needs the runners' mlir-aie (`aie.helpers.dialects.ext.scf` API); the newer mlir-aie moved several modules and `aie.core(link_with=...)`.
* The lm_head weights are re-laid for the projection engine (`reorder_cpy_lm_head_in_layer`, 8 streams, Q8_0 half chunks padded to Q4_K chunks, 294 MB
  for 248,320 x 1024 instead of 271 MB).

### `MODEL_TYPE=QWEN3_5_08B_SL`: every run self-contained (`layer_lm_sl.xclbin`)

`make bitstream MODEL_TYPE=QWEN3_5_08B_SL` builds the same design with `-DSTATELESS_X`. In the plain design the residual stream stays on the array across
the runs of a layer group and the gate core counts the layers of a period itself, so a firmware design reload between two runs (any other process's
command) silently changes the attention run. In the `_SL` build every run takes x from DDR and returns it, and the gate core reads the kind of layer from an
RTP (`RTP_gate_mode`, 36864; the build checks the address). The host (FLM_DLL `minicpm_v_4_7_npu`) must send x with, and receive x from, every run and write
the gate RTP; this folder's own `host/` (run.exe) does not, so `_SL` builds are for the engine only. The cause: the gate core counts the layer period, so a design reload in the middle of a layer group breaks the attention run.
