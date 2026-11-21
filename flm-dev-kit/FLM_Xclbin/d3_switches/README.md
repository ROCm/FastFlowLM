<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# d3_switches

The control-packet switches of the d3 engine (`FLM_DLL/detail/d3_npu`, `d3_launch.hpp`).

d3's prefill runs on `mm_rtm2_glu.xclbin`. Attention (`Qwen3_5/bidirectional_attention_prefill`) and conv1d
(`Qwen3_8/conv1d_prefill`) are other kernels; as separate xclbins each needs its own hardware context, and a switch
between contexts costs 2.2-2.8 ms, more than they save below ~2,000 tokens. With aie2p-launchlib the three kernels are
apps of one package and the array switches between them with control packets inside the mm hardware context, about
0.12 ms per switch.

`make bitstream` builds the three kernels as their own units do (kernel objects, IRON design, `aiecc.py` with their
flags plus `--aie-generate-txn`), packs them with `aie2p-launchlib/launchlib/tools/make_package.py`, and runs
`switch_files.cpp` (host only, no NPU or XRT needed) over the package: `build/xclbins/d3.switches` holds, for every
ordered pair of apps and for the cold start of mm, the instruction stream and the control packets. The format is in
`d3_launch.hpp` (`D3SW1`). `d3_switches_base.xclbin` is the mm build's xclbin, there because every unit's output needs
an xclbin; nothing ships it.

`kernel_assignment.json` places `d3.switches` in the `D3-Edge-NPU2` folder next to `mm_rtm2_glu.xclbin`. The engine uses
it when it is there and the mm kernel is `mm_rtm2_glu.xclbin` (the switches load the mm app themselves with
`cold_mm`, so the context's xclbin only has to have the same partition): attention from 512 tokens, conv1d from 1,536
rows (single non-dependent segment); `D3_NO_SWITCH=1` turns it off, `D3_ATTN_ARRAY_MIN` and `D3_CONV_ARRAY_MIN` move the
thresholds. Without the file nothing changes (host attention below 1,536 tokens, host conv1d).

Sharing the NPU: warm switches rely on the array staying as the last launch left it, and a second process's hardware
context takes the array at any command boundary (measured: two FLM servers on one NPU hung the device for every client
until the commands timed out, over and over). So each array turn is one submission (`cold_mm`, switch, guest kernel,
switch back as one `xrt::runlist`, `d3_launch::window`), and a request holds a system-wide lock (`d3_array_lock`: flock
on `$TMPDIR/flm_d3_array.lock`, a named mutex on Windows) from its vision tower to its last layer. Another FLM process
that does not take this lock (another model's engine) can still take the array; run d3 with `D3_NO_SWITCH=1` next to it.

Not covered: HRX (the engine falls back), prefix_bidir and multi-segment prompts' conv1d (host).
