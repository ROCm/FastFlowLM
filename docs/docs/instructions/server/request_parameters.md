---
layout: docs
title: Request Parameters & Long Prompts
nav_order: 13
parent: Local Server (Server Mode)
---

# 🎛️ Request Parameters & Long Prompts

This page says what FLM does with the fields of a `/v1/chat/completions` or `/v1/completions` request: which sampling fields it applies and how, what happens to `seed` and to fields it doesn't know, and how it reads a prompt that is longer than one prefill chunk.

> 📘 **Related pages:** [OpenAI API + Examples](https://fastflowlm.com/docs/instructions/server/openapi/) · [Reasoning, Tool Calls & Streaming](https://fastflowlm.com/docs/instructions/server/reasoning_and_tools/) (`reasoning_content`, `think`) · [Structured Output & tool_choice](https://fastflowlm.com/docs/instructions/server/structured_output/) · [Prefix Cache](https://fastflowlm.com/docs/instructions/server/prefix_cache/)

---

# 📑 Table of Contents

- [Sampling Fields](#️-sampling-fields)
- [Presence, Frequency and Repetition Penalties](#-presence-frequency-and-repetition-penalties)
- [The Penalty Window](#-the-penalty-window)
- [`seed` and Repeatable Output](#-seed-and-repeatable-output)
- [Reasoning in `reasoning_content`](#-reasoning-in-reasoning_content)
- [Fields FLM Doesn't Know](#-fields-flm-doesnt-know)
- [Long Prompts: Prefill Chunking](#-long-prompts-prefill-chunking)

---

## 🎚️ Sampling Fields

| Field | Values FLM accepts | What it does |
|---|---|---|
| `temperature` | ≥ 0 | Sharpens (below 1) or flattens (above 1) the distribution. `0` always picks the most likely token. |
| `top_p` | 0 – 1 | Keeps the smallest set of most likely tokens whose probabilities add up to `top_p`. |
| `top_k` | ≥ 1 | FLM extension. Samples among the `top_k` most likely tokens. `1` is greedy decoding. |
| `presence_penalty` | ≥ 0 | See [Penalties](#-presence-frequency-and-repetition-penalties). |
| `frequency_penalty` | ≥ 0 | See [Penalties](#-presence-frequency-and-repetition-penalties). |
| `repetition_penalty` | > 0 | FLM extension. See [Penalties](#-presence-frequency-and-repetition-penalties). |
| `repeat_last_n` | integer ≥ −1 | FLM extension. How many generated tokens the penalties look at. See [The Penalty Window](#-the-penalty-window). |
| `seed` | any integer | Makes sampling repeatable. See [`seed`](#-seed-and-repeatable-output). |

Each token is chosen in this order: penalties → the `tool_choice` / `response_format` grammar, if any → `top_k` → `top_p` and the model's min-p cut-off → `temperature` → sample.

The OpenAI Python SDK sends `top_k`, `repetition_penalty` and `repeat_last_n` through `extra_body`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="flm")

response = client.chat.completions.create(
    model="qwen3.5:4b",
    messages=[{"role": "user", "content": "List ten animals, no repeats."}],
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={"top_k": 20, "repetition_penalty": 1.0},
)
print(response.choices[0].message.content)
```

> ⚠️ **Settings stick.** Sampling fields are applied to the loaded model and stay in effect for every later request that doesn't send them, including requests from other clients. They return to the model's defaults only when the model is reloaded. If your application mixes settings, send every sampling field on **every** request. `seed` is the exception: it applies only to the request that sends it. (`think` and `reasoning_effort` [stick the same way](https://fastflowlm.com/docs/instructions/server/reasoning_and_tools/#-turning-thinking-on-and-off).)

> ⚠️ **Out-of-range values are ignored, not rejected.** A negative `temperature` or `presence_penalty`, or a `top_p` above 1, gets HTTP 200: the server logs a warning and keeps the value it had. OpenAI accepts penalties from −2 to 2, but FLM ignores negative ones, so a penalty can't be used to *encourage* repetition. `repeat_last_n` and `seed` are stricter: a value that isn't an integer, or a `repeat_last_n` below −1, is rejected with HTTP 400.

### Model defaults

A request that sends no sampling fields uses the model's own defaults, which differ by model. The presence penalty is the one that surprises people most:

| Models | `presence_penalty` at load |
|---|---|
| `qwen3.5:*`, `qwen3.5-core:*`, `qwen3.6-moe:35b-a3b`, `qwen3.6-moe-core:35b-a3b`, `qwen3.8-mtp:27b`, `minicpm-v-4.7:1b` | 1.5 |
| `gemma4-it:*`, `gemma4-it-core:*`, `gemma4e-flash:*`, `nemotron3.5-bolt:e4b` | 1.0 |
| every other model | 0 |

Sending `presence_penalty=0` turns the default off, and it stays off until the model is reloaded (see *Settings stick* above).

---

## 🔂 Presence, Frequency and Repetition Penalties

All three push the model away from tokens it has just generated. They change the logit of each token found in the [penalty window](#-the-penalty-window), where `count` is how many times that token appears in the window:

| Field | Change to the token's logit | Off when |
|---|---|---|
| `repetition_penalty` | divided by the penalty if the logit is positive, multiplied by it if negative | `1` |
| `frequency_penalty` | minus `frequency_penalty × count` | `0` |
| `presence_penalty` | minus `presence_penalty`, once, whatever the count | `0` |

`repetition_penalty` is applied first, then the other two are subtracted.

- **`presence_penalty`** steers the model toward tokens it hasn't used recently. Use it against a model that keeps coming back to the same words or ideas.
- **`frequency_penalty`** grows with each repeat, so it acts on loops: the same phrase over and over.
- **`repetition_penalty`** below 1 *rewards* repeats, and the server warns that it can make the model loop.

Penalties also act on reasoning tokens, and on tokens that a `tool_choice` or `response_format` grammar requires. They can make a required token less likely, but they never let the model leave the grammar.

The [log probabilities](https://fastflowlm.com/docs/instructions/server/openapi/#-example-token-log-probabilities) a response reports are taken **before** any penalty is applied.

> ⚠️ **`qwen3.8-mtp:27b`:** tokens accepted from the speculative draft head are compared with the model's own most likely token and skip the sampler. Penalties, `temperature`, `top_k` and `top_p` only shape the tokens the model samples itself, and drafted tokens don't enter the penalty window.

---

## 🪟 The Penalty Window

Penalties only look at the **last 64 tokens the model generated**, unless you set `repeat_last_n`:

- **The prompt doesn't count.** That includes the system prompt, the user's messages and earlier assistant turns you send back in `messages`. A word that appears only in the prompt is not penalised. One exception: when the server continues a conversation it still holds in its KV cache, the end of its previous reply can still be in the window. A request that sends [`seed`](#-seed-and-repeatable-output) always starts with an empty window.
- **The window slides.** Once a token falls more than `repeat_last_n` generated tokens back, `presence_penalty` and `frequency_penalty` stop applying to it. With the default of 64, the penalties in a long reply keep the model from repeating itself within a stretch of about 64 tokens, not across the whole reply.

### Changing the window size

`repeat_last_n` is an FLM extension, sent at the top level of the request (`extra_body` in the OpenAI SDK). It works on `/v1/chat/completions` and `/v1/completions`:

| `repeat_last_n` | Penalties look at |
|---|---|
| `64` (default) | the last 64 generated tokens |
| any positive integer `N` | the last `N` generated tokens |
| `-1` | every token generated so far in the reply |
| `0` | nothing: all three penalties are off, whatever their values |

```python
response = client.chat.completions.create(
    model="qwen3.5:4b",
    messages=[{"role": "user", "content": "Write a 500-word story without repeating an adjective."}],
    presence_penalty=1.0,
    extra_body={"repeat_last_n": -1},   # penalise repeats across the whole reply
)
```

Like the other sampling fields, the window size [sticks](#️-sampling-fields) until the model is reloaded, which sets it back to 64. Making the window smaller takes effect at once: tokens it no longer covers stop being penalised from the first token of the request. Ollama-style `options.repeat_last_n` is not read by the `/v1` endpoints: send `repeat_last_n` at the top level.

---

## 🎲 `seed` and Repeatable Output

Send an integer `seed` and FLM samples from it: the same request with the same `seed` gives the same text, on `/v1/chat/completions` and `/v1/completions`, streamed or not. Any 64-bit integer works, negative ones included.

```python
response = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "Invent a name for a pet dragon."}],
    temperature=0.8,
    top_p=0.9,
    presence_penalty=0,     # send every sampling field: they stick across requests
    frequency_penalty=0,
    seed=42,
)
```

What `seed` does:

- **It applies to its own request only.** The sampler's random generator restarts from the seed when the request starts. A request without `seed` carries on from wherever the generator was, so it doesn't disturb the next seeded one.
- **It empties the [penalty window](#-the-penalty-window).** Without this, a conversation the server continues from its KV cache would start with the end of the previous reply in the window, and the same seeded request would sample differently depending on what ran before it.
- **It doesn't pin the other sampling fields.** They [stick](#️-sampling-fields) across requests, so another client can change them between your two calls. Send `temperature`, `top_p`, `top_k`, the penalties and `repeat_last_n` with every seeded request.

There is no `system_fingerprint` to compare: non-streaming responses have none, and each stream gets a random one.

Greedy decoding (`temperature=0` or `top_k=1`) doesn't use the random generator, so it is repeatable without a seed. Two caveats hold for seeded and greedy output alike:

- **Penalties still apply.** They change *which* token is picked, but in the same way every time, as long as each request sends the same values.
- **A cached prompt can change the wording.** A prompt resumed from the [prefix cache](https://fastflowlm.com/docs/instructions/server/prefix_cache/#️-limits-to-know-about) or the prompt cache is not bit-for-bit identical to the same prompt read from scratch, so the same request can differ between a cold run and a cached one.

If the same request doesn't repeat, send it twice with `logprobs: true`. If the `logprob` values differ, the model read the prompt differently (usually because one run was cached), not because of the seed.

---

## 🧠 Reasoning in `reasoning_content`

On `/v1/chat/completions`, a reasoning model's thinking is returned in `reasoning_content`, never in `content`:

- non-streaming: `choices[0].message.reasoning_content`
- streaming: `choices[0].delta.reasoning_content`, in deltas before the answer's `content` deltas

The tags (`<think>…</think>`, Gemma4's thought channel, gpt-oss's `analysis` channel) are removed. The field is left out when the model didn't reason. The OpenAI SDK doesn't declare it, so read it with `getattr(msg, "reasoning_content", None)`.

Reasoning tokens count toward `max_tokens` and `completion_tokens`, and they enter the [penalty window](#-the-penalty-window) like any other generated token. `/v1/completions` returns the raw text, tags included.

Turning thinking on and off, streaming examples and the full message format: [Reasoning, Tool Calls & Streaming](https://fastflowlm.com/docs/instructions/server/reasoning_and_tools/).

---

## 📨 Fields FLM Doesn't Know

**FLM ignores fields it doesn't read. It doesn't reject them.** OpenAI SDKs, LangChain and agent frameworks send fields that FLM has no use for, and they still work against FLM. The downside is that a misspelt field (`temprature`) is dropped without any error or log line, so check your spelling if a setting seems to have no effect.

| Kind of field | What FLM does |
|---|---|
| A field FLM doesn't know | Ignored. |
| A field sent as `null` | Treated as not sent. Some clients send `null` for every option they leave unset. |
| `stop`, `n`, `logit_bias`, `user` | Ignored, as above. `stop` sequences are not applied, and there is always exactly one choice. |
| An `options` object (Ollama style) on a `/v1` endpoint | Ignored. Put sampling fields at the top level of the request. |
| A sampling field outside its range | Ignored with a warning in the server log. The previous value stays in effect ([see above](#️-sampling-fields)). |
| A sampling field of the wrong JSON type, such as `"temperature": "0.7"` | The request fails with an error. Send numbers as JSON numbers. |
| A malformed `messages`, `tool_choice`, `parallel_tool_calls`, `response_format`, `logprobs`, `top_logprobs`, `seed` or `repeat_last_n` | Rejected with HTTP 400 and `"type": "invalid_request_error"` before anything is generated, for streaming requests too. |

FLM's own extensions to the OpenAI request, all top-level fields (`extra_body` in the OpenAI SDK):

| Field | See |
|---|---|
| `think` | [Turning Thinking On and Off](https://fastflowlm.com/docs/instructions/server/reasoning_and_tools/#-turning-thinking-on-and-off) |
| `top_k`, `repetition_penalty` | [Sampling Fields](#️-sampling-fields) |
| `repeat_last_n` | [The Penalty Window](#-the-penalty-window) |
| `backend` | [Choose an Execution Backend](https://fastflowlm.com/docs/instructions/cli/#-choose-an-execution-backend) |

---

## 📏 Long Prompts: Prefill Chunking

Before generating, FLM reads the prompt into the model's KV cache. This is the **prefill**, and it is done in chunks of at most `max_prefill_len` tokens:

| Models | `max_prefill_len` |
|---|---|
| Flash models (`gemma4e-flash:*`, `qwen3vl-flash:4b`, `hy-mt2-flash:1.8b`) | 1024 |
| Every other chat model | 4096 |

**`max_prefill_len` is not a limit on the prompt.** A longer prompt just takes more chunks. A 32,000-token prompt on a model with 4096-token chunks is read in eight passes, seven of 4096 tokens and one of 3,328, and the server log shows each one:

```text
[FLM]  Prefill chunk 1/8 with 4096 tokens
[FLM]  Prefill chunk 2/8 with 4096 tokens
...
[FLM]  Prefill chunk 8/8 with 3328 tokens
```

### What limits the prompt: the context length

The prompt has to fit in the **context length**, with room left over to generate:

- A prompt of as many tokens as the context length, or more, is **rejected whole** with HTTP 400. On most models the message is `"Max length reached!"` (`"type": "model_error"`). Models on the `core` backend check before reading anything, and they also reject a request whose `max_tokens` doesn't fit in what is left. FLM never truncates a prompt.
- A prompt that fits starts generating, and generation stops when the context is full, with `finish_reason: "length"`.

So for a 32k-token prompt, start the server with a context comfortably above 32k. The model must support that length (see its [model card](https://fastflowlm.com/docs/models/)):

```shell
flm serve llama3.2:1b --ctx-len 65536
```

`usage.prompt_tokens` in each response tells you how long a prompt really was, and `usage.kv_token_occupancy_rate_percentage` how full the context is.

### Things worth knowing

- **Only the new part is read.** When the start of the prompt is already in the KV cache (the prompt cache, or the [prefix cache](https://fastflowlm.com/docs/instructions/server/prefix_cache/) on Qwen3.5 / Qwen3.6), only the rest is chunked and read. `usage.prompt_tokens_details.cached_tokens` says how much was reused.
- **Cancelling stops between chunks.** If the client disconnects during a long prefill, FLM stops before the next chunk instead of reading the whole prompt.
- **Images can make the first chunk larger.** On some vision models the image encoder runs with the first chunk, so that chunk is widened to hold every image token. That chunk size, rounded up to a power of two, is then used for the rest of that prompt.
- **On the `core` backend chunks are page-aligned.** They end on multiples of `max_prefill_len` counted from the start of the conversation, not from the start of the new part ([details](https://fastflowlm.com/docs/models/qwen/#-qwen3-06b--17b--4b--8b-on-the-core-backend-developer-preview)).

### Changing the chunk size

```shell
flm serve llama3.2:1b --prefill-chunk-len 8192
```

`--prefill-chunk-len` replaces `max_prefill_len` for every model the server loads. Values below 512 are raised to 512, and the value is rounded up to a power of two (`5000` → `8192`). It changes how a prompt is split into passes. It doesn't change how long a prompt can be: that is `--ctx-len`.
