<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# SMOLVLA/vision_switches

The warm switches of the SmolVLA vision tower (`FLM_Pybind/SmolVLA`, `vlm/vision/seq_gen.hpp`).

The tower's matmuls run on `SMOLVLA/vision_mm8` (8 columns, 32 cores, x multicast to both column groups) and its
attention on `SMOLVLA/vision_attn32` (4 compute units of 8 cores). They are two designs of one array; with
aie2p-launchlib both are apps of one package and the array switches between them inside the mm8 hardware context.
A switch is about 165 us as control packets (about 1,100 configuration ops).

`make bitstream` builds both kernels as their own units do (kernel objects, IRON design, `aiecc.py` with their flags
plus `--aie-generate-txn`), packs them with `aie2p-launchlib/launchlib/tools/make_package.py` and runs
`switch_files.cpp` (host only, no NPU or XRT needed) over the package. `build/xclbins/vision.switches` holds four
kernels: `to_attn32` and `to_mm8` as instruction streams, `cp_to_attn32` and `cp_to_mm8` as control packets (the
instruction stream only starts the packet DMA; the packets are one buffer at XRT argument 7 of the run that carries
the switch). The format is in `switch_files.cpp` (`SVSW1`). `vision_switches_base.xclbin` is the mm8 build's xclbin,
there because every unit's output needs an xclbin; nothing ships it.

`kernel_assignment.json` places `vision.switches` and `vision_mm8.xclbin` (from `SMOLVLA/vision_mm8`) next to
`text.xclbin`, `expert.xclbin` and `vision.xclbin`. The engine uses them by default; `SMOLVLA_VISION_L32=0` runs the
vision tower on `vision.xclbin` as before, `SMOLVLA_VISION_CTRLPKT=0` loads the switches by instruction streams
only.
