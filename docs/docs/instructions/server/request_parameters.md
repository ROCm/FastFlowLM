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
- [Continuing an Assistant Message (`continue_final_message`)](#-continuing-an-assistant-message-continue_final_message)
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

## ⏩ Continuing an Assistant Message (`continue_final_message`)

> Available from FLM v1.0.11, on `/v1/chat/completions`.

By default, a request whose last message is from the assistant gets a **new** reply: the chat template closes that message and opens a fresh assistant turn. Set `continue_final_message: true` and the model **continues** the last message instead, from exactly where its text ends. The field is vLLM's, and Hugging Face's `apply_chat_template` has the same flag.

```python
response = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[
        {"role": "user", "content": "Is Paris the capital of France? Answer yes or no."},
        {"role": "assistant", "content": "The answer is:"},
    ],
    max_tokens=1,
    logprobs=True,
    top_logprobs=5,
    extra_body={"continue_final_message": True},
)
first = response.choices[0].logprobs.content[0]
print(repr(first.token), first.logprob)   # ' Yes' -0.098 on llama3.2:1b
```

Use it to score what the model would say at an exact point, through the model's own chat template: classification, multiple-choice evaluation, or checking how confident the model is in an answer. With `logprobs`, the first entry is the model's distribution for the token right after your text (see [the example](https://fastflowlm.com/docs/instructions/server/openapi/#-example-score-the-next-token-after-a-partial-answer)).

What you get back:

- **`content` is only the continuation.** Your text isn't repeated, so `prefix + content` is the whole message. Its whitespace is kept as the model wrote it (`' Yes'`, not `'Yes'`), even when the continuation is only whitespace. In a stream, trailing whitespace arrives once the text after it, or the end of the reply, does.
- **`logprobs` cover only the continued tokens.** Your text is part of the prompt, so it has no entries.
- **`usage.prompt_tokens` includes your text**, and `completion_tokens` counts the continuation.
- **The prompt is cut the way Hugging Face cuts it.** FLM renders the chat with no generation prompt, finds the last occurrence of the final message's text, and stops right after it. One exception: a few templates write a finished message differently from how they open a reply (Qwen3.5 leaves out the empty `<think></think>` block, Gemma4 12B with thinking off its empty thought channel, LFM2.5-Thinking its opening `<think>`). Cut that way, the prompt looks like a finished turn and the model ends it, so for those templates FLM puts your text after the reply's opening instead. Templates that trim message content also drop trailing whitespace from your text, so `"The answer is: "` is read as `"The answer is:"`. A template that keeps the space keeps it, unless your text also starts with whitespace (Hugging Face drops the trailing space then too), and the model then sees a separate space token, which changes what it predicts next.
- **Models that reason first may read your text as reasoning.** `deepseek-r1`, `deepseek-r1-0528`, `qwen3-tk`, `lfm2.5-tk`, `nanbeige4.1`, and any model with `think` on, start every reply by reasoning. They can treat the message you continue as the start of that reasoning: they go on, close it with `</think>`, then answer. FLM then returns the continuation in `reasoning_content` and the answer after it in `content`, as vLLM's reasoning parsers do. A stream from such a model sends nothing until it either writes that tag or ends. To score the next token, use a model with `think` off: a reasoning model's first token is reasoning, not the answer.

Rejected with HTTP 400 (`invalid_request_error`), for streaming requests too:

| Request | Why |
|---|---|
| `continue_final_message` that isn't a boolean | |
| The last message isn't from the assistant | There is nothing to continue. |
| `add_generation_prompt: true` together with it | They ask for opposite things; vLLM refuses this too. Sending `add_generation_prompt: false` is fine. |
| The last message has `tool_calls`, images or audio, or no text | Only text can be continued. |
| `response_format` (JSON), or `tool_choice` `"required"` / a named function | The grammar for these starts at the beginning of a reply. `tool_choice` `"auto"` and `"none"` work. |
| The chat template rewrites the last message, for example by moving a closed `<think>…</think>` block out of it | Its text no longer appears as you sent it, so there is no place to continue from. An unclosed `<think>` is kept, and the model goes on reasoning. |
| The model can't continue a message: single-turn (flash) models and `gpt-oss` | The message is `continue_final_message is not supported by <model>`. |

`continue_final_message: false` or `null` keeps the default behaviour. FLM does not read `add_generation_prompt` otherwise.

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
| A malformed `messages`, `tool_choice`, `parallel_tool_calls`, `response_format`, `logprobs`, `top_logprobs`, `seed`, `repeat_last_n` or `continue_final_message` | Rejected with HTTP 400 and `"type": "invalid_request_error"` before anything is generated, for streaming requests too. |

FLM's own extensions to the OpenAI request, all top-level fields (`extra_body` in the OpenAI SDK):

| Field | See |
|---|---|
| `think` | [Turning Thinking On and Off](https://fastflowlm.com/docs/instructions/server/reasoning_and_tools/#-turning-thinking-on-and-off) |
| `top_k`, `repetition_penalty` | [Sampling Fields](#️-sampling-fields) |
| `repeat_last_n` | [The Penalty Window](#-the-penalty-window) |
| `backend` | [Choose an Execution Backend](https://fastflowlm.com/docs/instructions/cli/#-choose-an-execution-backend) |
| `continue_final_message` (vLLM's name) | [Continuing an Assistant Message](#-continuing-an-assistant-message-continue_final_message) |

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
