<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# r2t2_switches

The control-packet switches of the r2t2 engine (`FLM_DLL/detail/qwen3_asr_npu`, `prefill/qwen3_asr_launch.hpp`).

r2t2's prompt prefill runs its matrix products on `prefill.xclbin` (`dequant_mm_mix_bfp16`). The prompt attention
(`generic_prefill_attention/attention_DH_128_GQA_1_2`) is another kernel; as `attn.xclbin` it needs its own hardware
context, and the two switches per layer between contexts (~2.5 ms each) cost more than the NPU saves below ~1,000 tokens.
With aie2p-launchlib the two kernels are apps of one package and the array switches between them with control packets
inside the prefill hardware context.

`make bitstream` builds the two kernels as their own units do (kernel objects, IRON design, `aiecc.py` with their flags
plus `--aie-generate-txn`), packs them with `aie2p-launchlib/launchlib/tools/make_package.py`, and runs
`switch_files.cpp` (host only, no NPU or XRT needed) over the package: `build/xclbins/r2t2.switches` holds the cold start
of mm and the switches mm to attn and attn to mm, each as an instruction stream and its control packets. The format is in
`qwen3_asr_launch.hpp` (`R2SW1`). `r2t2_switches_base.xclbin` is the mm build's xclbin, there because every unit's output
needs an xclbin; nothing ships it. The mm build's RTP addresses are those of `dequant_mm_mix_bfp16` in
`rtp_addresses.lock.txt`, so the engine's prefill sequences are valid after `cold_mm`.

`kernel_assignment.json` places `r2t2.switches` in the `Confucius4-R2T2-NPU2` folder next to `prefill.xclbin`. The engine
uses it when it is there: the attention of prompts from 384 tokens then runs on the NPU, one submission per layer
(`cold_mm`, switch, attention, switch back as one `xrt::runlist`), under the same system-wide lock as d3's switches
(flock on `$TMPDIR/flm_d3_array.lock`, a named mutex on Windows), so the two engines' windows never interleave.
`QWEN3_ASR_NO_SWITCH=1` turns it off (then `attn.xclbin` in its own context, from 1,024 tokens);
`QWEN3_ASR_NPU_ATTENTION_MIN` moves the threshold. Without the file nothing changes.

Measured (Ryzen AI 9 HX 370, Linux, prompt attention of all 28 layers / whole prefill):

| prompt tokens | CPU attention | `attn.xclbin`, own context | switches |
|---:|---:|---:|---:|
| 512 | 53 / 233 ms | 93 / 346 ms | 42 / 209 ms |
| 1,152 | 189 / 590 ms | 151 / 575 ms | 103 / 458 ms |
| 2,048 | 693 / 1,334 ms | 271 / 919 ms | 222 / 815 ms |

The output tokens are the same as with `attn.xclbin` in its own context (the same kernel).

Not covered: HRX (the engine falls back to `attn.xclbin`).
