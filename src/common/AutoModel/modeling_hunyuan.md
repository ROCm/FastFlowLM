# Hunyuan (Hy-MT2-1.8B) frontend notes

## Reply cap and loop stop (2026-11-10)

Junk input (哈 x 200, a base64 string) made the model repeat itself until the context was full,
~20 s per request. Two stops in `modeling_hunyuan.cpp`:

* **Cap, only when the caller sent no `max_tokens`.** The server passes 4096 as `length_limit`
  when the request has no `max_tokens`, so `generate()` cannot tell "no limit" from "4096". It
  reads `input.requested_max_new_tokens` in `insert()` instead: that is only set when the client
  sent one. Then the reply is capped at `2 x prompt tokens + 64` (the whole templated prompt,
  system turn included). A translation is about as long as its source, so this only hits runaway
  replies. A caller's own `max_tokens` is never changed.
* **Loop stop, always on.** After every token, `_stop_generation()` checks whether the reply ends in
  one block of 1..32 tokens repeated at least 4 times and covering at least 64 tokens. Checked from
  the shortest block up; the needed span only grows with the block length, so it stops at the
  first span longer than the reply. Cost is at most ~4K int compares per token.
  Both report `finish_reason: "length"` (a truncation, not the model choosing to stop).

The hook is `AutoModel::_stop_generation()`, asked by `_shared_generate` after each token; its
default returns false, so other models are unchanged.

Measured on Krackan (`flm serve`, no `max_tokens`): 哈 x 200 stops after 66 tokens in 1.4 s;
a base64 string is copied back and ends by itself (439 tokens); normal translations up to a
3,980-token prompt end with `stop`.

## Context length 8192 (2026-11-10)

Single-turn models always load `default_context_length` from `model_list.json`; `--ctx-len` is
ignored for them. Hy-MT2 was 1024 because the engine used to cap the whole context at 1024; the
engine handles 32K+ since the long-context work, so the catalog now says 8192 (KV cache 64 KB per
token = 512 MB). `max_prefill_len` stays 1024: the app feeds the engine 1024-token calls and the
engine splits them into 512-token pieces itself.
