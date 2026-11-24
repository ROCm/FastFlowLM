<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Development notes of the 8-row verify layer

Written while the design was built and measured (October 2026). Numbers are from the box and build named in each note;
script, log and workspace names refer to the development workspace and are not part of this repo.

| Note | Content |
|---|---|
| `fused8_allocation.md` | tile / memtile / shim channel / packet id allocation of the fused layer |
| `fused8_results.md` | whole-layer results and the engine interface (argument and BO layouts, RTPs, L_max rules, host precompute) |
| `proj_engine_L8_results.md`, `proj_speedup.md` | the 8-token Q4 projection core and the memtile x replay |
| `rmsglu_L8_results.md` | norms with the resident residual, GLU rounds |
| `ple8_L8_results.md` | the per-layer-input (PLE) stage and the host `pli` contract |
| `attn8_L8_results.md`, `attn_longctx.md` | the first attention (K/V replayed per token) and `attnx` (K/V read once per verify) |
| `swa_linear_buffer.md` | the sliding-window K/V cache as a linear buffer with host compaction (engine contract) |
| `hang_rootcause.md` | the dispatch-count hang: one unconsumed shim task-completion token per dispatch |
| `phase8_results.md`, `al_and_host.md`, `e2e_results.md` | engine integration, acceptance length analysis, end-to-end speed and accuracy |
