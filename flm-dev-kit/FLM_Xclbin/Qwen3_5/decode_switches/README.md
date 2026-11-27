<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Qwen3_5/decode_switches

The control-packet switches of the Qwen3.5-4B decode (`FLM_DLL/detail/qwen3_5vl_npu`, `qwen3_5vl_npu.cpp` and
`qwen3_5vl_launch.hpp`).

Each decode token runs the 32 layers on `layer.xclbin` (`Qwen3_5/qwen3_5_decoding`, `MODEL_TYPE` `QWEN3_5_4B`) and
then the q4_k lm_head on `lm_head_q4k.xclbin` (`Qwen3_5/lm_head_npu_bin`, `MODEL_TYPE` `QWEN3_5_4B_Q4K`). As two
xclbins they need two hardware contexts and every token switches twice, about 2.7 ms per switch. With
aie2p-launchlib the two kernels are apps of one package and lm_head runs as a guest of the layer hardware context.
Each token is one window: `layer_then_lm` → lm_head → `lm_then_layer`, built once and executed again every token
(lm_head's arguments never change).

`make bitstream` builds the two kernels as their own units do for those `MODEL_TYPE`s (kernel objects, IRON design
with the unit's `IRON_ARGS`, `aiecc.py` with their flags plus `--aie-generate-txn`), packs them with
`aie2p-launchlib/launchlib/tools/make_package.py`, and runs `../prefill_switches/switch_files.cpp` over the package:
`build/xclbins/qwen3_5_decode.switches` holds `cold_layer`, `layer_then_lm` and `lm_then_layer` (the `Q35SW1` format
of `qwen3_5vl_launch.hpp`). `qwen3_5_decode_switches_base.xclbin` is the layer build's xclbin, there because every
unit's output needs an xclbin; nothing ships it. `lm_then_layer` restores the layer build, so the build checks that
its attention RTP buffers sit where the engine writes the context length (53248 / 53536).

`kernel_assignment.json` places `qwen3_5_decode.switches` in the `Qwen3.5-4B-NPU2` folder. The engine uses it when it
is there, the q4_k lm_head is in use and preemption is off: `lm_head_q4k.xclbin` is then not loaded as its own
context. `QWEN35_DECODE_NO_SWITCH=1` turns it off, `QWEN35_DECODE_SWITCH_DIR=<dir>` loads loose switch files instead
(rig). Without the file nothing changes; the other Qwen3.5 sizes have none.

The switch file carries copies of both kernels, and nothing checks it against the shipped `layer.xclbin` /
`lm_head_q4k.xclbin`. CI's changed-mode plan has no dependency map, so a change to `qwen3_5_decoding` or
`lm_head_npu_bin` alone does not rebuild this unit: such a change must also touch this unit (a file here or its
`build_units.json` entry), or the package ships switches that restore the old layer.

The decode window holds the system-wide array lock (`qwen3_5vl_array_lock`, see `../prefill_switches/README.md`)
whenever the switches are in use.

Not covered: HRX (the engine keeps lm_head in its own context).
