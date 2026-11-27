<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Qwen3_5/prefill_switches

The control-packet switches of the Qwen3.5 prefill (`FLM_DLL/detail/qwen3_5vl_npu`, `qwen3_5vl_launch.hpp`).

The prefill's projections run on `mm_rtm2_glu.xclbin`. conv1d (`Qwen3_5/conv1d_prefill`), the gated delta net
(`Qwen3_5/gate_delta_net_prefill/hostDeltaNetPrefill`) and gated attention (`Qwen3_5/gated_attention_prefill`) are
other kernels; as separate xclbins each needs its own hardware context, and a switch between contexts costs about
2.6 ms: 88 of them per prompt on the 4B (three per linear-attention layer, two per attention layer). With
aie2p-launchlib the four kernels are apps of one package and the array switches between them with control packets
inside the mm hardware context, about 0.1 ms per switch.

`make bitstream` builds the kernels as their own units do (kernel objects, IRON design, `aiecc.py` with their
flags plus `--aie-generate-txn`), packs them with `aie2p-launchlib/launchlib/tools/make_package.py`, and runs
`switch_files.cpp` (host only, no NPU or XRT needed) over the package: `build/xclbins/qwen3_5.switches` holds, for
the cold start of mm and for each switch the engine takes (`--pairs`), the instruction stream and the control packets.
The format is in `qwen3_5vl_launch.hpp` (`Q35SW1`).

The delta net is there twice, from the same unit and with the same buffers and DMA: `gdn`, built with
`GDN_CHUNK_MODE=1` (the chunked core, `kernel/gdn_chunked.h`, about 2.9 times faster), which the engine takes, and
`gdn0`, the per-token recurrence of `GateDeltaNet_prefill.xclbin`, which it takes with `QWEN35_GDN_RECURRENT=1`. `qwen3_5_switches_base.xclbin` is the mm build's xclbin, there because every
unit's output needs an xclbin; nothing ships it.

`kernel_assignment.json` places `qwen3_5.switches` in the `Qwen3.5-4B-NPU2` folder next to `mm_rtm2_glu.xclbin`. The
engine uses it when it is there, the mm kernel is `mm_rtm2_glu.xclbin` and preemption is off (the switches load the mm
app themselves with `cold_mm`, so the context's xclbin only has to have the same partition): per linear-attention
layer one submission runs mm → conv1d → delta net → mm, per attention layer mm → attention → mm, and conv.xclbin,
GateDeltaNet_prefill.xclbin and attn.xclbin are not loaded. `QWEN35_NO_SWITCH=1` turns it off,
`QWEN35_SWITCH_DIR=<dir>` loads loose switch files instead (rig). Without the file nothing changes.

Sharing the NPU: as for d3, each array turn is one `xrt::runlist` (`qwen3_5vl_launch::window`) and a prefill holds
the system-wide lock d3 and Inkling2 take (`qwen3_5vl_array_lock`: flock on `$TMPDIR/flm_d3_array.lock`, the named
mutex `Global\flm_d3_array` on Windows) from its vision tower to its last layer.

Not covered: HRX (the engine falls back to one context per kernel).
