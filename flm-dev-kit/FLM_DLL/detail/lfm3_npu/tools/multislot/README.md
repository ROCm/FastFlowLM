<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# LFM3 multi-slot test rig

Run before a PR that touches the saved-state table (`lfm3_npu_state_api`) or the LFM3 wrapper's prefix-slot path.

1. **Engine level** (`lfm3_engine_run --state-test`, needs a device and the xclbins): read/write round trip, split restore
   (`write_state_kv` + tail state with `kv_from`), a state read mid-decode (conv ring and K/V rows on the device), restore after
   `update_max_length`, warm-resume vs cold-prefill logits, rejection of an unfilled state. Prints PASS/FAIL per gate.

   `LFM3_DECODE=full|conv|host LFM3_PREFILL_MIN=1 lfm3_engine_run --pkg <package> --ids-file <1k-token prompt> --greedy 24 --npu --xclbin-root <root> --max-length 2048 --state-test`

2. **Server level** (`rig.py`): starts its own `flm serve lfm3:1.7b` processes (own ports, never kills others) and asserts on
   `usage.flm_prefix_cache`: history growth, branching, boundary sharing, replay after unrelated traffic (stale rows), think on/off,
   tools, ~10k-token conversations (cache growth), `--prefix-slots 2` eviction, `--prefix-cache-mb 1`, `--prefix-slots 0`, and
   fresh-server determinism.

   `python3 rig.py --flm <flm> --env FLM_XCLBIN_PATH=<root> FLM_MODEL_PATH=... [--decode full|conv|host] [--only S1,S3]`

Lessons applied (see the multislot plan): greedy requests pass `top_k=1`; warm-vs-cold text equality is not a gate (chunked
prefill numerics), the gates are fresh-server replay and replay-after-interference; trim is 0 (the generation prompt is a token
prefix of how the next turn renders the reply header, with thinking on and off).
