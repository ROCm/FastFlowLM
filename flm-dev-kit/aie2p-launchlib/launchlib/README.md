<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# launchlib (AIE2P): make the launch kernels of a list at run time

`launchlib` is a header-only C++20 library. The host gives it a list of configurations and a runtime sequence per
app. The library reads the package of each app and makes one **instruction stream** (TXN binary) for each
configuration. It is the AIE2P counterpart of `aie4-launchlib/launchlib` (branch `flm_aie4_double_pm`), with the
same concepts and without the PM ping-pong. Control packets (`Transport::ctrlpkt`, section 4.1) are the primary
transport and the default: the stream sends the bulk of a load as packets, with a packet buffer next to it.
`Transport::txn` (command-processor writes only) is the fallback.

The library does not start a tool, does not load a PDI, and writes no file.

```cpp
#include "launch/launch.hpp"

launch::ConfigList list{"pkg", launch::Transport::ctrlpkt,
                        {launch::cold("vadd"), launch::warm("vadd", "vmul"), launch::rerun("vmul")}};
launch::SequenceMap seqs = {{"vadd", vadd_sequence}, {"vmul", vmul_sequence}};   // section 5
launch::Result r = launch::build(list, seqs);
// r.kernels[i].words: the instruction stream of kernel i; r.base_xclbin: an xclbin for the hw_context
```

The host makes one `xrt::hw_context` on `r.base_xclbin` and runs kernel `i` as `xrt::run` of the xclbin's kernel
`MLIR_AIE`: argument 0 = 3 (opcode), 1 = an instruction BO with `r.kernels[i].words`, 2 = its word count, 3.. =
the buffers of the runtime sequence. Or, FLM's way, `launch/txn/elf.hpp` turns the words into an ELF for
`xrt::elf` → `xrt::module` → `xrt::ext::kernel` (arguments 3, 0, 0, buffers); that is the default
(`SessionOptions::flow = Flow::elf`, and `--flow elf` in the examples) and costs 8–13 µs less per launch
(`../vadd_vmul_bf16/README.md`, "ELF flow vs xclbin flow"). A full host program is
`../vadd_vmul_bf16/host/host.cpp` (`include/flow.hpp` there runs either flow).

Or let **`launch::Session`** (`launch/xrt/session.hpp`, the one header with XRT) do all of that, safely:

```cpp
#include "launch/xrt/session.hpp"

launch::Session s(list, seqs);                    // build, hw_context; options: flow, policy, ...
xrt::bo a = s.buffer(bytes, 0), b = s.buffer(bytes, 1), c = s.buffer(bytes, 2);
launch::RunReport r = s.run({{"cold_vadd", {a, b, c}}, {"vadd_then_vmul", {a, b, c}}, {"rerun_vmul", {a, b, c}}});
if (r.status != launch::RunReport::Status::ok) { /* r.message; run them again */ }
```

| What the Session adds | How |
|---|---|
| Order check | It knows which app is in the array and throws `std::logic_error`, before anything runs, for a warm kernel whose `prev` is not that app, a rerun of another app, or a rerun after a reset. (Between different floorplans such a kernel hangs or silently leaves shim DMA tasks that break later kernels, section 3.2.) |
| Sharing guard | Before and after each run it asks for every hw_context on the NPU (`launch/xrt/npu_watch.hpp`: on Linux the amdxdna driver, `DRM_IOCTL_AMDXDNA_GET_ARRAY`; on Windows XRT's internal list, built with `LAUNCH_NPU_WATCH_XRT=1`, public XRT headers only, `../vadd_vmul_bf16/WINDOWS.md`). While another context exists, warm and rerun launches run as the cold kernel of their app and sequence (`SharedPolicy::cold_only`, the default; also `refuse`, `allow`). If another context came, went or submitted a command during a run that depended on the array, the report is `other_context_ran` and the Session opens a new hw_context. After such a change, or more than `max_gap` (2 ms; 1 ms on Windows) since it last asked, the array is "unknown": the next warm or rerun launch runs as the cold kernel. |
| Recovery | A runlist that throws or does not finish in `timeout` (10 s): report `failed`, and a new hw_context (the only way back from a hang or from stuck shim tasks). Buffers made before stay valid. After `max_failures` failed runlists in a row (2 on Windows, no limit elsewhere) the Session closes the hw_context instead, and every run reports `stopped` until `reopen()`. |
| ELF chunks | In the ELF flow, long lists of launches run as several runlists (at most `elf_budget` = 24 MB of instructions and `elf_max_runs` = 1,024 runs each), and the runs are used again. |

What the guard cannot see: a context of another process that opens, runs and closes **inside one of our
runlists** (one that runs a launch stays in the list at least ~5.8 ms on Linux, ~2.2 ms on Windows), and one that
runs nothing and lives less than `max_gap`. The second one is not harmless: on Linux, a process that opens and
closes empty contexts (stress S6, `--child churn-empty`) gave wrong results the Session reported ok, and made
runlists fail; on Windows it made launches much slower. Do not share the NPU with such a process. XRT's
`access_mode::exclusive` does not keep other contexts out (measured).
Without a context list (Windows without `LAUNCH_NPU_WATCH_XRT`, or on an XRT whose list it does not recognize, or `accel`
empty; `Session::watch_why()` says which) the Session cannot see other contexts, so it takes the NPU as shared: by the policy, every warm and rerun launch runs as cold (`cold_only`), unless `SessionOptions::exclusive`
says no other process uses the NPU. On Windows a warm or rerun launch after another app's context hangs the NPU, and
repeated hangs can make the machine unresponsive (reboot to recover). The order check and the recovery still work.

What it costs: one driver query per run, ~1 µs with no context on the NPU but 35–55 µs once any context exists
(ours too: the driver asks the firmware). Within `max_gap` of its last query the Session asks only after the run (a
context that opens in that gap is then seen after the run: `other_context_ran` instead of a cold launch). In a
runlist of N launches it is paid once, not N times (`../pm_runtime_load_singletile/results/pressure_32core.md`, P4).

**Requirements:** C++20; `nlohmann/json.hpp` (config.hpp, the JSON form); XRT (`xrt/session.hpp` only); aiebu only
for `txn/elf_aiebu.hpp` (a comparison; `make test_elf`); Python 3 (tools). Device: npu2 (AIE2P). Linux
was measured.

---

## 0. Terms

| Term | Meaning here |
|---|---|
| app, design | One IRON design (`vadd`). "App" is its name in a configuration; "design" the built files. |
| package | The folder `tools/make_package.py` writes: one subfolder per app. |
| configuration | One entry of `ConfigList`; it gives one kernel. |
| kernel | One instruction stream of `Result::kernels`; the host launches it. |
| instruction stream, TXN | The ops the NPU's command processor runs in order: 4 header words, then the ops (section 6). |
| command processor | The NPU firmware that runs a stream. One for the whole array (AIE4 has 6 uCs). |
| part | One part of a kernel: reset, load, program, sequence. The kind of a kernel says which parts it has. |
| floorplan | Everything of an app but its program: tiles, BDs, locks, chain starts, routes, mux. |
| chain start | The Start_Queue and Ctrl writes that start a DMA channel's BD chain. |

## 1. Data flow

```mermaid
flowchart LR
  subgraph Offline["Offline (per design)"]
    IRON[IRON design .py] --> AIECC[aiecc --aie-generate-xclbin<br/>--aie-generate-npu-insts --aie-generate-txn]
    AIECC --> MP[tools/make_package.py]
  end
  MP --> PKG[(package/app: init.txn, design.xclbin,<br/>manifest.txt, insts.bin)]
  subgraph Build["launch::build (C++, run time)"]
    PKG --> LP[load_package: sort init writes<br/>per tile and class]
    CL[ConfigList] --> V[validate]
    V --> KP
    LP --> KP[KernelPipeline::build_kernel<br/>reset, load or program, sequence]
    TS[TransportStrategy: txn or ctrlpkt] --> KP
    SQ[SequenceMap: host C++] --> RT[Rt] --> KP
    KP --> ENC[txn::encode]
  end
  ENC --> RES[(Result: kernels[i].words,<br/>base_xclbin)]
  RES --> HOST[host: XRT hw_context, runlist]
```

`build()` validates the list, loads the package of each app once, makes the transport, runs the pipeline for
each configuration, and encodes each kernel's ops.

## 2. Files

| File | Role |
|---|---|
| `include/launch/launch.hpp` | `build()`, `Result`, `Kernel`, `KernelStats` |
| `include/launch/config.hpp` | `ConfigList`, `Config` (kinds), `reset()`, `cold()`, `warm()`, `rerun()`, `cycle()`, `validate()`, JSON form |
| `include/launch/kernel/pipeline.hpp` | `Recipe` (the table of section 4), `KernelPipeline::build_kernel` (Template Method) |
| `include/launch/kernel/reset.hpp` | The reset part: WHICH registers (`ResetScope`, `reset_targets`), HOW (`reset_part`) |
| `include/launch/kernel/context.hpp` | `KernelContext` (the one argument of a part), `write_runs`, core hold / start |
| `include/launch/transport/strategy.hpp` | `TransportStrategy` (Strategy): `load`, `load_program` |
| `include/launch/transport/txn_strategy.hpp` | `TxnStrategy`: command-processor writes |
| `include/launch/transport/ctrlpkt_strategy.hpp` | `CtrlPktStrategy`: the state by control packets down one lane per column (section 4.1) |
| `include/launch/transport/ctrlpkt_lane.hpp` | `ctrlpkt::plan`: the lane of each column (shim MM2S channel, packet routes, the app routes it toggles) |
| `include/launch/transport/ctrlpkt_packets.hpp` | Control-packet encoding, packet ids, the shim BD of a chain |
| `include/launch/transport/ctrlpkt_check.hpp` | `ctrlpkt::check_kernel`: replays a kernel on a switch model and compares its end state with its TXN twin |
| `include/launch/hw/switch.hpp` | A model of the stream switches: register file, packet routes (`route`) |
| `include/launch/rt/rt.hpp` | `Rt` (the aiex.py calls), `Arg`, `Memcpy`, `SequenceFn`, `SequenceMap`; the shim BD encoding |
| `include/launch/formats/package.hpp` | `load_package` (the only file I/O; "APP@STATE" states and "APP[A-B]" columns, section 3.1), `Package`, `TileInit`, `floorplan_difference` |
| `include/launch/txn/txn.hpp` | The instruction stream: `Op` (IR), `encode`, `parse` |
| `include/launch/txn/elf.hpp` | `to_elf`: a stream as the ELF of XRT's ELF flow (the AddressPatch ops as relocations; aiebu's ELF without aiebu) |
| `include/launch/txn/elf_aiebu.hpp` | `to_elf_aiebu`: the same by aiebu (needs `libaiebu.a`; for `make test_elf` and `SessionOptions::elf_writer`) |
| `include/launch/xrt/session.hpp` | `Session`: build, hw_context, runs with order check, sharing guard, recovery, ELF chunks (needs XRT) |
| `include/launch/xrt/packets.hpp` | `packet_buffers`, `set_packets`: the packet buffer of a ctrlpkt kernel, for hosts that run kernels themselves |
| `include/launch/xrt/npu_watch.hpp` | `NpuWatch`: every hw_context on the NPU from the amdxdna driver (Linux) or XRT's internal list (`LAUNCH_NPU_WATCH_XRT=1`: public XRT headers, query found by RTTI name, entry layout checked at run time) |
| `include/launch/hw/aie2p.hpp` | AIE2P facts (sizes, offsets) with their aie-rt names; `classify(address)` |
| `include/launch/hw/hw_gen.hpp` | **Generated** by `tools/gen_hw.py`: the stream-switch routing registers per tile kind |
| `tools/make_package.py` | aiecc design folder → package (with the lock census of the core ELFs: llvm-objdump of Peano); `--state` adds a state |
| `tools/op_cost.cpp` | `make op_cost`: what one TXN op costs on the NPU (write32 ~60 ns, maskwrite32 ~110, block write ~60 + ~4 per word, ~5 in program memory); the numbers of `launch::CostModel` |
| `tools/gen_hw.py` | aie-rt `xaie2pgbl_params.h` → `hw_gen.hpp`; `--check` |
| `tools/txn_dump.py` | Print a TXN binary |
| `tools/lane_report.cpp` | `make lane_report PKG=dir`: the lanes the ctrlpkt transport plans for every app |
| `tests/` | Section 8 |

## 3. Input

### 3.1 Package of a design (`tools/make_package.py -o PKG NAME=DESIGN_FOLDER ...`)

DESIGN_FOLDER is where aiecc ran with `--aie-generate-xclbin --aie-generate-npu-insts --aie-generate-txn`.

| File | Content |
|---|---|
| `init.txn` | aiecc's `configure` sequence (`aie-translate -aie-npu-to-binary -aie-sequence-name=configure`): every init write of the design, as aie-rt writes it (core disable, DMA reset, program into PM, locks, BDs, chain starts, routes, mux, core enable). |
| `design.xclbin` | aiecc's xclbin. The host makes the hw_context on one of them. |
| `manifest.txt` | `design NAME`, `tile COL ROW PKT_ID` (controller id), `core COL ROW`, `buffer NAME COL ROW ADDRESS BYTES`, `lock NAME COL ROW ID INIT`, `alloc NAME DIR CHANNEL COL` (shim_dma_allocation), from `input_with_addresses.mlir`. |
| `insts.bin` | aiecc's runtime sequence; tests only (the C++ sequence must equal it). |

`load_package` sorts the writes of `init.txn` per tile and register class (`hw::classify`):

| Class | Kept as |
|---|---|
| program, data, bd, mux | the writes, in order |
| lock, route | the last value of each register |
| channel | the chain starts (Start_Queue, Ctrl value writes); aiecc's reset mask writes are dropped |
| core_control | dropped; marks a core with a program |

A write without a class, a Sync or patch in `init.txn`, or a block write across a class border stops the load
with an error naming the file. The library checks that the device is npu2 (header of `init.txn`).

**States** (`make_package.py -o PKG --state NAME@STATE=FILE.txn`, also with apps in the same call): the app NAME
loaded set up for one of its runtime ops. FILE.txn holds that op's configuration as its host writes it (BDs,
routes, locks of the op); the tool copies it to `PKG/NAME/state_STATE.txn`. Any configuration may then name the
app `"NAME@STATE"`: `load_package` loads NAME and sorts the state's writes after `init.txn`'s, so
`warm(other, "ovl@P")` switches back to the overlay `ovl` with its op `P` ready (the host's next `P` launch writes
no configuration), and `warm("ovl@P", other)` leaves it. The load refuses, naming the file, op and address:

| A state may not | because |
|---|---|
| write a program (PM) or set up a core `init.txn` has not | the state is a configuration, not another design |
| write a register or data word that `init.txn` sets to another nonzero value (locks excepted) | the app's other ops expect `init.txn`'s value |
| hold a Sync, patch or a write without a class | as `init.txn` |

Channel reset masks are dropped, as in `init.txn`. A state the host's switches used to merge into a copy of the
package by hand (an "ovlP" folder) is the same kernel this way: the Qwen4 attention switch gives byte-identical
`warm(attn, ovl@P)` / `warm(ovl@P, attn)` streams with both transports.

**Columns** (`"NAME[A-B]"`, also `"NAME@STATE[A-B]"`): the part of NAME in columns A..B, its tiles there and
nothing else. A warm kernel between it and a smaller app inside those columns (a guest) resets and loads those
columns only: `warm("gemv[0-1]", attnd)` in, `warm(attnd, "gemv[0-1]")` back leave NAME's other columns as its
last run left them, cores waiting, and reload only what the guest took. The array must hold NAME before (as for
any warm kernel). NAME's columns must not depend on each other's progress (a DMA chain, a lock or data in flight
across the border), since A..B start again from the init values and the other columns do not. The load checks the
census (a core that acquires a lock across the border stops it); the rest is the host's knowledge of NAME, as with
states. Measured on the Qwen4 decode GEMV (32 cores, the x vector broadcast from column 0, every column its own
weight and y chains) with a 4-core attention guest in columns 0-1: 309 + 313 ops and 190 KB of packets per round
trip instead of 1133 + 1250 ops and 760 KB.

### 3.2 List of configurations

| Kind | Builder | Default name |
|---|---|---|
| reset | `reset()` | `reset` |
| cold | `cold(app)` | `cold_<app>` |
| warm | `warm(prev, app)` | `<prev>_then_<app>` |
| warm, program only | `warm(prev, app, {.program_only = true})` | `<prev>_then_<app>_pm` |
| rerun | `rerun(app)` | `rerun_<app>` |

`sequence` (C++ field or JSON key) picks another runtime sequence than the app's. `cycle({a, b, c})` gives
`cold(a), warm(a,b), warm(b,c), warm(c,a)`. `validate()` checks names (unique, not empty), apps, `prev` of warm
kernels. JSON: `config.hpp` header. `build()` also checks: the packages, the sequences, and that a program-only
warm kernel's two apps share a floorplan (`floorplan_difference`).

**The order contract** (as AIE4): a warm kernel assumes `prev` is in the array, a rerun kernel that its app is.
The streams do not check it; the runlist order makes it true. reset and cold kernels can follow anything that
kept the contract. A program-only warm kernel also assumes every launch of `prev` left the ping/pong DMA chains
at ping (an even number of blocks per core per launch).

Breaking it costs more between apps of **different floorplans** (measured, `../pm_runtime_load_singletile`
stress S4 and S5c): a warm kernel with the wrong `prev` makes its runlist fail (`ERT_CMD_STATE_NEW`, after the
driver's timeout, about a minute), and the hw_context fails every later runlist; a rerun of the wrong app runs
its sequence on routes that are not its own and leaves shim DMA tasks that never finish. **No kernel can clear
those**: the AIE2P shim DMA channel has no reset bit (Ctrl bit 1 is Pause_Mem), so the next app that routes that
channel gets their data first, cold kernels included. A new hw_context clears both (the firmware resets the
partition when it loads a context). Between apps of one floorplan a wrong `prev` gave right data
(`../vadd_vmul_bf16` S4).

**The one-context contract** (measured, the examples' stress tests S5): the state the kernels leave in the array
lasts only while this hw_context is the only one on the NPU. When the NPU runs another hw_context (this process
or another) and comes back, this context's array is its xclbin again — also between two launches of one runlist.
After anything else may have used the NPU, start with `cold` (or a warm kernel whose `prev` is the base xclbin's
app). A full warm kernel between apps of one floorplan does not depend on the array's state; reruns and
program-only warm kernels do, and between different floorplans every warm kernel does (see above: run one on
the wrong array and only a new hw_context recovers).

## 4. The launch

| Kind | reset | load | program | sequence |
|---|---|---|---|---|
| reset | whole_array | | | |
| cold | whole_array | x | | x |
| warm | used_by (prev, run) | x | | x |
| warm, program only | | | x | x |
| rerun | | | | x |

The cores **loop forever** (as the AIE4 designs): a kernel ends with them waiting for the next block. A rerun
runs the sequence again; every other kind stops them.

| Part | Ops | Why |
|---|---|---|
| reset 1 | `Core_Control` reset on, enable off (cores of the scope) | Stops the old program. A core waiting on a lock stops too, but its acquire stays pending: see reset 5. |
| reset 2 | DMA channel Ctrl reset bit on (core and mem-tile channels of the scope) | Stops the old chains. |
| reset 3 | Switch registers: run's value, or 0 (block writes of consecutive runs); shim mux and demux: 0 | A design sets only its own routes. A warm kernel skips what prev already set to the same value. aiecc sets the mux with mask writes, so without the 0 the old app's mux bits stayed (register model, `test_build` between different floorplans). |
| reset 4 | DMA channel Ctrl = 0 | Reset bit off, and the bits the old app set on a channel the new one does not use (aiecc writes Ctrl = 1 on the channels it uses). |
| reset 5 | Core-tile locks: 63, then 0 (the lock drain: every core tile for reset and cold; for warm the locks `prev`'s cores acquire, the package's census, which `make_package.py` takes from the core ELFs: 64 locks on vadd_vmul_bf16's 32 cores instead of 512; a package without a census: every lock aiecc writes in `prev`'s core tiles) | A core stopped while it waits in a lock acquire leaves the acquire pending; the load's lock writes would grant it and one token of the next app would be gone (wrong data or a hang: measured on the Whisper encoder, `whisper_encoder/README.md`). 63, the largest value, grants every pending acquire; it must come after reset 4, else the old DMA chains take the tokens (measured). Not covered: a core waiting for a lock to *equal* a value (IRON's object FIFOs wait for >= n). |
| load | program (PM), data, **all lock values**, BDs, mux, chain starts, then per core unreset + enable | After a DMA reset the locks must be written: an idle S2MM channel holds the lock of its next BD and the reset drops it (without the lock writes the next launch hangs, measured). |
| program | per core: reset, program, unreset, enable | Nothing else differs between apps of one floorplan. |
| sequence | the host's `Rt` calls | |

`launch::breakdown(kernel)` lists what each part writes, by register class, with an estimated time
(`launch::CostModel`, `launch::estimate_us`: the op costs `tools/op_cost.cpp` measured). The example's README
compares a full and a program-only switch (command-processor ops with control packets: 1×1 75 vs 29; 8×2 777 vs
257).

Lock writes, since the census (vadd_vmul_bf16 8 × 4, `warm(vadd, vmul)`): the drain writes the 64
locks the cores acquire, not all 512 of their tiles, and the load writes the locks the app declares (every
`aie.lock`, manifest `declare`) or its cores acquire, and any lock aiecc sets to a value other than 0: 272 of the
656 aiecc writes (it also zeroes every other lock of a core tile, which no one uses).

Scopes: `whole_array` = all 32 cores, every core and mem-tile DMA channel, every routing register and shim mux
of the 8 columns (`hw_gen.hpp`: 143 shim, 113 mem tile, 154 core). `used_by` = the cores, started channels,
routing registers and shim muxes of `prev` and `run`, and the core-tile locks `prev` declares (aiecc declares
all 16 of each core tile it uses). Shim channels are never reset: each sequence writes and
waits for its own, and the hardware has no shim channel reset (section 3.2).

The command processor runs the ops in order: a write has landed before the next op starts. No barrier, no
answer, no token is needed (AIE4 needs them for its six uCs and its control packets).

### 4.1 Transport::ctrlpkt

`Transport::ctrlpkt`, the default (`ConfigList::transport`; JSON `"transport"`, options `"ctrlpkt": {...}`;
the hosts' `--transport`), sends the bulk of the load and program parts as control packets: per column a
shim MM2S channel streams them down a **lane** of packet routes (shim, mem tile, core rows) into the tiles'
control ports, all columns at once, while the command processor goes on with what packets may not write. The
packets of a kernel are one buffer, kernel argument `Kernel::packet_arg` (the highest one the sequence leaves
free); `Session` passes it, other hosts use `xrt/packets.hpp`. Reset and rerun kernels have no packets.

- The order of a kernel and why: the top of `ctrlpkt_strategy.hpp`. The lane takes ports the app does not
  use; app routes that block it are **toggled** to lane values for the chains and back after.
- Each column's chain ends in a `Sync` on its channel. A packet that goes nowhere would leave that Sync waiting
  forever (a wait that never ends can wedge amdxdna on Linux until a reboot), so nothing is left to the device:
  - `CtrlPktOptions::check` (default on): `build()` replays every kernel on a switch model, follows every
    packet to the tile it names, refuses TXN writes into a chain in flight, and compares the end state with
    the same kernel built by `TxnStrategy`. A kernel that fails it throws.
  - `CtrlPktOptions::rewrite_lanes` (default on): warm and program-only kernels write every lane register,
    so a lane another hw_context's switch took away gives wrong data (as with txn), not a hang. Off saves
    ~23 ops per column; then the lane must still be in the array.
- Words the runtime sequence writes too (an RTP lock) go by TXN after the chains: a packet still in flight
  could land after the sequence's write.
- A column without a lane, or without enough free shim BDs, loads by TXN writes; so does every column of a
  kernel whose sequence uses all five kernel arguments.
- `memtile_data` (default off): mem-tile data by packet too. `starts_by_packet` (default on): DMA and core
  starts by packet when every toggle is in the shim.

Measured, µs (Linux: npu5 Strix Halo; Windows: Krackan; the examples' READMEs have the full tables):

| | Linux | Windows | two hw_contexts (FLM today), Linux |
|---|---|---|---|
| vadd_vmul_bf16 8 × 4, program-only switch (beyond a rerun) | 24.5 | 26.2 | 2,196 |
| vadd_vmul_bf16 8 × 4, full warm switch (beyond a rerun) | 77.5 | 89.7 | 2,196 |
| vadd_vmul_bf16 8 × 2, program-only switch | 16.3 | | 1,584 |
| Whisper mm ↔ attn, whole launch, alternating | 190 | 186 | 2,706 |

The rest of a program-only switch is mostly the lane rewrite (~23 TXN ops per column) and the TXN op floor.

Use `Transport::txn` instead:

- on 1 × 1 designs: one column has nothing to run in parallel, so packets are slower than TXN writes there;
- next to other hw_contexts (other processes, the sharing guard's modes): ctrlpkt has not been tested there.
  The example hosts' modes with other contexts (stress `ctx`, `churn`, `guard`, `--robust`, ...) default to
  txn and take ctrlpkt only with `--hang-risk`; the pressure scripts run their two-process parts with txn.

## 5. Runtime sequences (`rt/rt.hpp`)

The C++ form of the IRON `runtime_sequence`, same names:

| Call | Ops |
|---|---|
| `npu_dma_memcpy_nd({.metadata, .bd_id, .mem = Arg{index, elem_bytes}, .offsets, .sizes, .strides, .issue_token, .burst_length, .push})` | BlockWrite of the 8-word shim BD, AddressPatch (kernel argument + byte offset), then the push (unless `.push = false`) |
| `npu_push_bd(channel, bd, token)` | [MaskWrite32 of the shim's controller id into the channel Ctrl, if token] Write32 of the Start_Queue |
| `npu_dma_wait(channel)` | Sync on the channel's task-complete token |
| `npu_write32`, `npu_maskwrite32` | Write32, MaskWrite32 |
| `npu_rtp_write("buffer", i, v)`, `set_lock_value("lock", v)` | Write32 by name (manifest) |
| `alloc("lhs0")` | the shim channel of an object FIFO (manifest) |

The encoding follows mlir-aie (`AIEDmaToNpu.cpp`, `getHardwareStridesWraps`; npu2 burst encodings). `tests/test_rt`
checks the example's sequence against aiecc's `insts.bin` word for word (1 and 8 columns). Only linear transfers
were checked against aiecc; strided ones follow the same formulas but are not checked yet.

## 6. Instruction stream (`txn/txn.hpp`)

Header: `rows << 24 | device generation << 16 | minor << 8 | major`, `mem-tile rows << 8 | columns`, op count,
bytes. Ops: Write32 (6 words), MaskWrite32 (7), BlockWrite (4 + n), Sync (4), AddressPatch (12), Preempt (1).
Source: mlir-aie `AIETargetNPU.cpp`. `parse(encode(s)) == s`, and `encode(parse(w)) == w` for every aiecc
binary tested.

## 7. Measured facts (npu5)

| Fact | Detail |
|---|---|
| Program-only switch (ctrlpkt) | 3.4 µs (1 core), 16.3 µs (16 cores), 24.5 µs (32 cores, 23 KB of program). |
| Full warm switch (ctrlpkt) | 7.3 µs (1 core), 64.3 µs (16 cores), 77.5 µs (32 cores). |
| Run in a runlist | ~20 µs per run with almost no work: the floor per launch. |
| hw_context switch (FLM way) | 143 µs (1 core) to 2.2 ms (32 cores). |
| DMA reset without lock writes | the next launch hangs (`ERT_CMD_STATE_TIMEOUT`). |
| A reset mask write on a Start_Queue register | pushes a BD (wrong data). The library resets Ctrl registers only (`channel_ctrl_of`, tested). |
| aiecc's IRON task API | writes the byte offset into BD word 1 and into the patch; the patch overwrites the word. |
| The NPU gives +0 where the CPU gives −0, and flushes denormals to 0 | Checks compare values. |
| Another hw_context on the NPU | this context comes back as its xclbin: the kernels' state is gone (one-context contract, section 3.2). |
| Stress and pressure (ctrlpkt) | about 1.8 million launches on Linux (`../vadd_vmul_bf16` stress and pressure, `../pm_runtime_load_singletile` pressure) and 0.95 million on Windows, every output checked: 0 wrong, 0 hangs. |
| fp32 add, subtract, multiply on the cores | not always IEEE round to nearest: a sum is within 1 ulp of the largest of its inputs and result, a product within 1 ulp (`../pm_runtime_load_singletile`, `designs/design.hpp`). |
| Three floorplans (`../pm_runtime_load_singletile`, 32 cores) | see its README: every row of the AIE4 host test, the pressure runs. |
| AIE2P TXN ops | write, block write, block set, mask write, mask poll, and the custom ops (task-token sync, DDR patch, register dump, timer, merged sync). No branch, no register-to-register copy: AIE4's `vpick_nocp` (the uC copies BD words the core wrote) cannot be written; `vpick` needs control packets from the core. |

The example's README has the full table.

## 8. Tests

| Where | What | Needs |
|---|---|---|
| `make test` | `gen_hw.py --check`; `test_txn` (round trip, classes; aiecc's binaries of a package); `test_rt` (C++ sequence == aiecc's, 8 columns); `test_build` (the kernels on a register model: cold = aiecc's configure, cold + warm = the next app's configure, program-only too; the parts of each kind; `breakdown`; JSON; `channel_ctrl_of`); `test_errors` (every mistake throws and names the configuration, file or op: bad Rt calls, bad lists, a throwing sequence, 11 kinds of broken package); `test_ctrlpkt` (every kernel under every `CtrlPktOptions` set passes its check and converts to the ELF flow; the check finds corrupted or cut packets, a missing Sync, a TXN write into a chain in flight, lost lanes without `rewrite_lanes`; an RTP lock goes by TXN); `test_state` (a state of the first app: the Package fields, cold/warm/back/rerun with both transports, what it leaves; the bad names, a missing file, and each write a state may not make); `test_columns` (`APP[A-B]` of the first app: its tiles, warm kernels in and out of it write only those columns with both transports and less than the whole app's, bad ranges, a census acquire across the border; skipped for a one-column app) | a package (default: the example's precompiled 8 × 2) |
| `test_build PKG A B` with A, B of different floorplans | warm(A, B, program_only) is refused; cold(A), warm(A, B), warm(B, A) twice each give aiecc's state and leave no route, mux or channel Ctrl of the other app | a package (`../pm_runtime_load_singletile`: `make check`, all six pairs) |
| `make test_elf` (or `build/test_elf FILE.txn ...`) | `txn::to_elf` against aiebu: BDs and relocations of hand-made streams, the streams it refuses; on every kernel of a package (and on TXN files, e.g. FLM's dumped streams) the same .ctrltext as aiebu's without its patch ops, the same relocations, `.dynamic` and header fields; host time of both | `libaiebu.a` (XRT), a package |
| `../vadd_vmul_bf16`: `make check` | the same on the example's current package | |
| `../vadd_vmul_bf16`: `make run` | the device: every kind, 1,600-launch loops, checks, the ELF flow, timing | the NPU |
| `../vadd_vmul_bf16`: `make stress` | the device: random walks over every kind, runlist lengths, odd lengths, wrong prev, other contexts and processes, special values, the largest length | the NPU |

## 9. Not done

- Control-packet transport (section 4.1): lanes are planned at build time, not built into the xclbin (an
  xclbin with the lanes would not need `rewrite_lanes`). Not tested next to other processes on the NPU.
- npu1 (Phoenix): another device table and TXN generation 3.
- Strided runtime-sequence transfers are not compared with aiecc yet.
- Windows: built and measured (`../vadd_vmul_bf16/WINDOWS.md`, `../pm_runtime_load_singletile`). The sharing guard
  reads XRT's internal list through a mirrored vtable (`LAUNCH_NPU_WATCH_XRT=1`); one XRT version (2.21.0) tested
  on Windows. If the list is not recognized, warm and rerun launches run as cold unless `exclusive`. Tested next to one other process; not next to many other NPU
  users, and not with a context that arrives inside one of our runlists.
- Trace is not written. (RTP and lock-by-name calls are used by `../pm_runtime_load_singletile`'s vminus.)
