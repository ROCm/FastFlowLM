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
a base64 string is copied back and ends by itself (439 tokens). Neither stop fires on normal
translations (130-line test unchanged). A reply that ends with `stop` is not proof that the whole
input was translated: see the next section.

## Early stop on long or repeated input (2026-11-11)

The model can end with its own end token (`finish_reason: "stop"`) before the whole input is
translated. Found with English -> Chinese, greedy, on Krackan:

* **Repeated text** (15 sentences x 4): the original bf16 model (transformers, CPU) translates one
  pass and stops (14/60 sentences); the NPU build does two passes (28/60). Same with repetition
  penalty 1.05 or 1.0. This is the model's own habit.
* **Long unique text** (LoCoMo dialogue): 1,000 words (1.3K tokens) come out complete; at 2,000
  words the NPU build stops at about two thirds of the source, after inventing a few lines, while
  the bf16 model translates all of it.
* Cause of the long-text case, measured with teacher forcing (every setup is fed the same token
  ids; the NPU through `test_long --decode`): the NPU picks the same top token as bf16 on about
  93% of positions, flat from 2.6K to 5K context, and bf16 weights rounded to Q4_0 with exact CPU
  math give the same 92%. So the drift comes from the 4-bit weights, not from long-context error.
  Once the text has drifted, even the bf16 model stops at the same place (P(end) 0.58).
* Open lead: at the true end of a source that stops mid-sentence, the NPU gives the end token 0.08%
  where bf16 gives 98.9% and simulated Q4_0 83% (one position only).

What clients can do: send long documents in pieces of about 1K tokens, and do not rely on a
repeated passage being translated more than once.

## Context length 8192 (2026-11-10)

Single-turn models always load `default_context_length` from `model_list.json`; `--ctx-len` is
ignored for them. Hy-MT2 was 1024 because the engine used to cap the whole context at 1024; the
engine handles 32K+ since the long-context work, so the catalog now says 8192 (KV cache 64 KB per
token = 512 MB). `max_prefill_len` stays 1024: the app feeds the engine 1024-token calls and the
engine splits them into 512-token pieces itself.
