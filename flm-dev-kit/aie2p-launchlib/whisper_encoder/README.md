<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# whisper_encoder: the launch library on a real model (trial)

Whisper-V3-Turbo's encoder alternates two xclbins 64 times per 30 s audio window: `encoder_mm.xclbin` (matmul)
and `encoder_attn.xclbin` (attention), `FLM_Xclbin/Whisper/encoder_MM` and `encoder_attention_888`. FLM keeps one
hw_context per xclbin, so each change is a context switch (~2.65 ms, `FLM_Xclbin/Whisper/README.md`: ~170 ms of
~600 ms per window). The trial: switch with launchlib's warm kernels inside one hw_context instead.

| Step | What | Status |
|---|---|---|
| 1 | Package the two kernels; time a warm switch against today's context switch | **done**: 0.19 ms per launch with the switch instead of 2.7 ms (below) |
| 2 | One encoder layer's real launches (FLM's instructions + the warm switch) in one context; same output as today | **done** after a library fix (below): the hang was a pending lock acquire |
| 3 | The whole encoder in a local build of the Whisper engine: same transcript | **done** with switch files of command-processor writes (below): same transcripts. The engine's control-packet path is open |
| 4 | Report: gain, and what shipping would need | **done** (below) |

## Use

```
../vadd_vmul_bf16/env.sh make package     # the two kernels -> build/pkg (aiecc + Peano, ~1 min)
make check                                # no NPU: register model of cold/warm, binaries round trip
make switch                               # step 1 on the NPU
make compare MODEL=.../Whisper-V3-Turbo-NPU2   # step 2: the engine without and with the switch (needs the engine change)
```

`tools/build_packages.sh` builds each kernel as its Makefile does (Peano object, IRON MLIR, aiecc with the
flags of its `bitstream.mk`) plus `--aie-generate-txn`, then `../launchlib/tools/make_package.py`. FLM's designs
have no IRON runtime sequence (the engine writes its instructions in C++), so the packages have no `insts.bin`.

## Step 1: the switch

Both kernels use all 32 cores. Register model (`make check`): cold and warm kernels in both directions give
aiecc's configuration and leave no route, mux or channel setting of the other kernel; `test_ctrlpkt` replays
both warm kernels on the switch model.

| Kernel (`Transport::ctrlpkt`) | Command-processor stream | Control packets |
|---|---|---|
| warm(mm, attn) | 1,202 ops, 34.6 KB | 56.9 KB (attention's program: ~8.7 KB per core, rows broadcast) |
| warm(attn, mm) | 1,159 ops, 31.7 KB | 4.8 KB |

`make switch` (one start + wait per launch, empty runtime sequences, 400 launches; Windows: the `dll` CI runner,
Krackan):

| µs per launch | Linux mean | Linux p50 | Windows mean | Windows p50 |
|---|---|---|---|---|
| one context, rerun (no switch: the floor) | 77.3 | 83.1 | 68.7 | 65.9 |
| **one context, warm mm ↔ attn, alternating** | **189.9** | 173.4 | **186.1** | 189.1 |
| of them mm → attn | 185.7 | 172.6 | 191.5 | 190.6 |
| of them attn → mm | 173.2 | 162.5 | 184.5 | 179.6 |
| **two contexts, alternating (FLM today)** | **2,705.7** | 2,756.4 | **2,772.8** | 2,740.6 |

So a switch costs ~0.1 ms beyond the rerun instead of ~2.6 ms: **14× less per launch**. Per window (32 switches
each way): ~7 ms instead of ~168 ms, about **160 ms less of ~600 ms**, if the real launches behave as these
empty ones (step 2 checks the data). Windows ran 20,000 launches per kind on top: all pass.

## Step 2: the real launches in one context

What was built:

- `FLM_DLL/detail/whisper_npu/whisper_encoder.{hpp,cpp}` (outside this folder, on branch `aie2-launchlib` only:
  it changes a shipping engine, so it comes in its own PR after step 3; without it `make compare` stops with a
  message instead of comparing today's path with itself): with `FLM_WHISPER_SWITCH_DIR=DIR` the attention app is made
  in the matmul's hw_context, and `prepend_switch` puts `DIR/to_attn.txn` in front of `mha_engine`'s instructions and
  `DIR/to_mm.txn` in front of `o_half[0]`'s (in a layer the attention always follows a matmul and `o_half[0]` always
  follows the attention). Not set: nothing changes. Also, under the flag only, the attention waits for its last input
  transfers (`drain_inputs`). Debug: `FLM_WHISPER_SWITCH_DUMP=DIR` writes the joined streams,
  `FLM_WHISPER_SWITCH_TRACE=1` prints each switched launch.
- `host/make_switch.cpp` writes the two switch files from the packages (`--kind cold` for whole-array switches).
- `host/encoder_compare.cpp` runs the engine of this tree on a fixed random 30 s window, writes the encoder output
  (cross-attention K/V) and times each window: `make compare` runs it without and with the flag and compares.
- `tools/build_xroot.sh` builds the other three Whisper xclbins of this tree and assembles the engine's xclbin folder.
- `host/stream_probe.cpp`, `tools/probe_rounds.sh`: run dumped instruction streams as ELFs (as FLM's npu_app), in
  one hw_context or two, repeated.

Results before the fix:

| Run | Result |
|---|---|
| Today's path (flag off), 3 windows | identical output each window |
| Flag on | layer 0 completes; in layer 1 a matmul after the switch back hangs (`qds_device::wait() unexpected command state`) |
| Probe, switch streams alone, 3 rounds | pass |
| Probe, matmul + switches without the attention, 3 rounds | pass |
| Probe, matmul, switch, attention, switch back (separate launches), 15 rounds | hang in round 2 |
| The same with a **cold** (whole-array) switch back, 15 rounds | hang in round 3 |
| Joined streams (switch + op in one launch) | hang after 1-4 rounds, at random |
| A delay between the switch and the op | no change |
| The attention waiting for all its input transfers (`drain_inputs`) | they all finish; the hang stays |
| **Control: the same streams in two hw_contexts (as FLM today), 15 rounds** | **pass** |

The first guess (state no instruction stream can clear: shim DMA, stream-switch buffers) was wrong.

### Root cause: a core stopped in a lock acquire leaves the acquire pending

The cores loop forever, so after a launch every core waits in a lock acquire. The reset part stops it
(`Core_Control` reset), but the acquire stays pending in the lock's tile. The load then writes the next app's lock
values, the pending acquire is granted, and one token of the next app is gone: its cores and DMAs get out of step.
The result is wrong data or, in the engine, a hang. Two hw_contexts never see it: the firmware resets the columns.

How it was found (`tools/`, `host/stream_probe.cpp`; `seed:N` inputs, `PROBE_DUMP` output dumps, status snapshots
with `../launchlib/tools/aie_status`):

| Experiment | Result |
|---|---|
| matmul, switch, attention (one context) | attention output wrong, a different hash each time; a second switch to the attention after it: right |
| the attention after a matmul load whose cores were never enabled | right |
| the attention after a matmul load whose 32 cores were enabled but got no data (they only wait on a lock) | wrong 3/3; with 1 core enabled: wrong sometimes |
| ruled out: core and mem-tile data memory zeroed, core registers reset (`txn_core_scrub.py`: the writes land, `aie_status` shows them), all locks zeroed, flushed switch FIFOs, delays, module resets (ignored by the firmware), split switch block writes | still wrong |
| **after the DMA reset: every core-tile lock 63, then 0** (`txn_lock_drain.py ... 63 0 load core`) | **right, every run**; mem-tile locks instead: wrong; the same before the DMA reset: wrong every run (the old DMA chains take the tokens) |

The outputs looked only slightly wrong (a few ulp) because the probe inputs are uniform in [1, 2): attention over
the wrong block of such data is close to the right result. The launchlib examples did not show it in ~1.8 M
launches (why not was not measured: their apps are small and use the same few locks).

The fix is step 5 of the reset part (`../launchlib/include/launch/kernel/reset.hpp`): after the DMA channels are
reset, the core-tile locks are written 63 (the largest value: every pending "acquire >= n" is granted), then 0;
the load writes the app's values after that. Reset and cold kernels drain every core tile, warm kernels the
locks `prev`'s cores acquire (the package's census of the core ELFs, 3 per core here). `test_build` checks the
drain and its place. Not covered: a core waiting for a
lock to equal a value (IRON's object FIFOs wait for >= n).

### Results with the fix

`build/switch/` made by `make_switch.exe` with the fixed library (no patch scripts), npu5:

| Run | Result |
|---|---|
| Probe: matmul, switch, attention, switch back, 30 rounds (120 launches, one context) | all complete; every matmul and attention output equal to the healthy one (was: hang in round 1-4) |
| Probe: the always-wrong cases above (attention after a matmul with data, after 32 or 1 waiting matmul cores) | right, 15/15 |
| `make compare` (engine, 3 windows) | **K/V IDENTICAL** to today's path |
| `tools/compare_repeat.sh 5` | 5/5 IDENTICAL |

So the encoder in one context gives the same K/V. The other examples' stress and pressure runs pass with the fix.

## Step 3: the whole engine in FastFlowLM

The engine of this tree inside a shipped FastFlowLM: `build/rt/stock` is a copy of a v1.0.12 portable build (XRT,
with the Whisper speed-up of IRON#155), `build/rt/tree` the same with this tree's `libwhisper_npu.so` and Whisper
xclbins (`build/xroot`). Three variants: **stock** (as shipped), **tree** (this tree, the switch off) and
**switch** (`FLM_WHISPER_SWITCH_DIR=build/switch`). The transcripts come from FastFlowLM's own ASR test
(`npu_test --suites asr`, `flm serve -a 1`): the FLEURS set of 84 clips in 7 languages, 4 of them longer than one
30 s window, scored against human transcripts, plus its checks of formats, segments, window tails, timestamps,
language and prompt.

| Run | stock | tree | switch |
|---|---|---|---|
| `tools/step3_asr.sh VARIANT` (84 clips, Whisper alone on the NPU) | 23/23 cases pass | 23/23, **84/84 transcripts equal to stock** | 23/23, **84/84 equal to stock** |
| WER / CER per set (en-short ... ar), overall | 3.4 ... 15.5%, 7.53% | the same in every set | the same in every set |
| `tools/step3_mixed.py VARIANT N` (a chat model in the same server, chat requests without pause; English clips) | 48/48 equal | | **72/72 equal**; chat 73/73 |

Every transcript of these runs is the same in all three variants. This trial's switch files are streams of
command-processor writes (`make_switch.exe`, `prepend_switch`): the engine has no control-packet path yet (it
would pass each switch's packet buffer as a kernel argument), so its timing is not the one of step 1.

With **another process** on the NPU (`tools/step3_two_process.sh VARIANT`: a stock `flm serve qwen3:0.6b` in its
own process answering chat requests without pause, 8 short English clips) both paths are unreliable on this
machine (npu5, amdxdna 7.0.0-rc1): every request takes ~65 s instead of ~1.3 s, and transcripts change.

| Variant | Identical to the stock transcript | Errors (`qds_device::wait() unexpected command state`) | Wrong text |
|---|---|---|---|
| stock (two runs) | 9 of 16 | 1 | small word changes; one "Thank you." (Whisper on garbage) |
| switch | 6 of 8 | 1 | one runaway repeated phrase |

That is today's behaviour, not the switch's; how the two compare in rate would need many more runs (each takes
~10 minutes). Notes from setting it up: two servers that each load a chat model and Whisper run out of
hw_contexts (`DRM_IOCTL_AMDXDNA_CREATE_HWCTX ... Invalid argument` for the second one); after a hang or a killed
server the NPU fails the next commands for a while, so the script checks the NPU first; a process started in
the background by a script ignores SIGINT, and so do its children (`tools/chat_load.py` restores it).

## Step 4: the gain, and what shipping it would take

**Gain.** The encoder's 64 context switches per 30 s window (~2.6 ms each, ~168 ms) become warm switches of
~0.1 ms each (~7 ms) with control packets: ~160 ms of ~600 ms per window, the same on Linux and Windows (step 1).
The rest of the request (decoder, the other encoder work) does not change. The engine trial (step 3) gives the
same transcripts.

**Cost at run time: none of the library.** The engine reads the switch kernels from files and puts them in front
of two apps' instruction streams (`prepend_switch`, ~80 lines in `FLM_DLL/detail/whisper_npu`; with control
packets it also passes their packet buffers, one more kernel argument); `launchlib` runs at build time only, to make those files. So Windows
needs no C++20 header library, only the files.

What shipping would take:

| Item | What | Why |
|---|---|---|
| Switch files with the xclbins | CI builds `to_attn` and `to_mm` (stream and packets; `tools/build_packages.sh` + `make_switch.exe`) in the same job as `encoder_MM` and `encoder_attention_888`, and ships them next to them in `xclbins/Whisper-V3-Turbo-NPU2/` | they write those exact programs, BDs and locks: a file made from another build of either kernel runs the wrong configuration (no error, wrong output) |
| A check that they match | the engine compares a hash of the two xclbins (recorded in the switch files at build time) and falls back to two hw_contexts on a mismatch | the binding is otherwise implicit, as between `.q4nx` and xclbins (nothing checks that a `.q4nx` matches the xclbins it is paired with; a mismatch gives wrong output, not an error) |
| The engine change | the packet buffer of each switch as a kernel argument (`make_switch.exe` writes it too); `FLM_WHISPER_SWITCH_DIR` becomes "switch files present": on when they are there, an env variable to turn it off; the trial's debug switches (`_DUMP`, `_TRACE`) removed | today it is a trial flag with command-processor writes only |
| HRX | run step 3 on the HRX runtime (FastFlowLM's `HRX-portable`) | only XRT was tested; HRX runs the same instruction streams, but its ELF and command path differ |
| Windows | run step 3 on Windows (XRT and the MSI) | Windows uses the same files; the NPU firmware and driver differ |
| CI | PR Check already benches Whisper on an `FLM_DLL/detail/whisper_npu` change (XRT and HRX) and npu_test's `asr` suite checks the transcripts | no new jobs; `FLM_Xclbin/kernel_assignment.md` lists the two files with the two kernels |
| Risk owner | the switch writes registers a hw_context load normally writes (`launchlib/README.md`): new firmware or a new driver could break it | a kernel or firmware change must rerun step 2's probe and step 3; the fallback above limits the damage to "slower" only if the hash catches it |

Not solved by shipping, and not caused by it: with another process on the NPU, Whisper is ~50× slower and
sometimes wrong or failing, with or without the switch (step 3). That belongs to the driver or firmware.

Next: the engine's control-packet path (`make_switch.exe` writing the packets, the engine passing them), then
step 3 again for its time per window. Models with more switches per request gain more (`../flm_survey/`).

