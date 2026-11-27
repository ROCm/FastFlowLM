<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Qwen3_5/vision_switches

The control-packet switches of the Qwen3.5-4B vision encoder (`FLM_DLL/detail/qwen3_5vl_npu`, `qwen3_5vl_image.cpp`
and `qwen3_5vl_launch.hpp`).

The encoder's projections and MLP run on `vision_mm.xclbin` (`MM/host7_mt_variant_bias`, `MODEL_TYPE`
`QWEN3_5_4B-VISION-MM`), its attention on `vision_attn.xclbin` (`VisionAttention/hostAttention_whoel_array`,
`MODEL_TYPE` `QWEN3_5_4B-VISION-ATTENTION`). As two xclbins they need two hardware contexts, and each block switches
twice (mm → attention → mm), about 2.6 ms per switch: 48 per image on the 4B's 24 blocks. With aie2p-launchlib the
two kernels are apps of one package and the array switches between them with control packets inside the vision mm
hardware context.

`make bitstream` builds the two kernels as their own units do for those `MODEL_TYPE`s (kernel objects, IRON design
with the unit's `IRON_ARGS`, `aiecc.py` with their flags plus `--aie-generate-txn`), packs them with
`aie2p-launchlib/launchlib/tools/make_package.py`, and runs `../prefill_switches/switch_files.cpp` over the package:
`build/xclbins/qwen3_5_vision.switches` holds `cold_mm`, `mm_then_attn` and `attn_then_mm` (the `Q35SW1` format of
`qwen3_5vl_launch.hpp`). `qwen3_5_vision_switches_base.xclbin` is the mm build's xclbin, there because every unit's
output needs an xclbin; nothing ships it.

`kernel_assignment.json` places `qwen3_5_vision.switches` in the `Qwen3.5-4B-NPU2` folder. The engine uses it when it
is there and preemption is off: per block one submission runs mm (cold) → attention → mm, and `vision_attn.xclbin` is
not loaded. `QWEN35_VISION_NO_SWITCH=1` turns it off, `QWEN35_VISION_SWITCH_DIR=<dir>` loads loose switch files
instead (rig). Without the file nothing changes; the other Qwen3.5 sizes have none.

The image is encoded inside the prefill, which holds the system-wide array lock (`qwen3_5vl_array_lock`, see
`../prefill_switches/README.md`) whenever either set of switches is in use.

Not covered: HRX (the engine falls back to one context per kernel).
