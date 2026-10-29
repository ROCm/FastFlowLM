---
layout: docs
title: Prefix Cache (Prompt Reuse)
nav_order: 12
parent: Local Server (Server Mode)
---

# 🗃️ Prefix Cache (Prompt Reuse)

Agents and chat apps send the same long system prompt and tool list with every request. Without a cache, FLM reads that whole prompt again each time, and on a 14k-token prompt that is 10 to 60 seconds before the first word appears.

The **prefix cache** lets `flm serve` remember what it already read. When a new request starts with text FLM has seen before, FLM picks up from where it left off and only reads what is new.

| Model | Same system prompt + tools, new user message (14k-token prompt) | |
|---|---|---|
| `qwen3.5:2b` | 12.1 s | **1.6 s** |
| `qwen3.5:4b` | 26.4 s | **2.0 s** |
| `qwen3.5:9b` | 37.7 s | **3.0 s** |
| `qwen3.6-moe:35b-a3b` | 63.0 s | **3.5 s** |

(Time to first token, measured on an NPU2 Linux machine: the first number is a cold prompt, the second is the same prompt with the shared part cached.)

The other supported models gain the same way. Measured on shorter prompts (about 1 to 2 k tokens), a cold prompt against the same prompt with its shared start cached:

| Model | Prompt | Cold | Cached |
|---|---|---|---|
| `gemma4-it:12b` | 2.2 k tokens | 9.3 s | **1.7 s** |
| `gpt-oss:20b` | 2.1 k tokens | 7.7 s | **1.5 s** |
| `gemma3:4b` | 2.2 k tokens | 3.3 s | **1.0 s** |
| `lfm2.5-tk:1.2b` | 2.0 k tokens | 1.4 s | **0.5 s** |
| `minicpm-v-4.7:1b` | 1.6 k tokens, with a picture | 1.3 s | **0.2 s** |
| `qwen3.8-mtp:27b` | 1.3 k tokens (shared system prompt only) | 16.5 s | **3.6 s** |

---

## ✅ Where it applies

| | Prefix cache |
|---|---|
| Models | **Qwen:** `qwen3.5:0.8b` / `2b` / `4b` / `9b`, `qwen3.6-moe:35b-a3b`, `qwen3:0.6b` / `1.7b` / `4b` / `8b`, `qwen3-it:4b`, `qwen3-tk:4b`, `qwen2.5-it:3b`, `qwen3vl-it:4b`, `qwen2.5vl-it:3b`, `qwen3.8-mtp:27b` (the shared system prompt only, see below)<br>**Llama and DeepSeek:** `llama3.2:1b` / `3b`, `llama3.1:8b`, `deepseek-r1:8b`, `deepseek-r1-0528:8b`<br>**Phi and Nanbeige:** `phi4-mini-it:4b`, `nanbeige4.1:3b`<br>**LFM:** `lfm2:1.2b`, `lfm2:2.6b`, `lfm2-trans:2.6b`, `lfm2.5-it:1.2b`, `lfm2.5-tk:1.2b`<br>**gpt-oss:** `gpt-oss:20b`, `gpt-oss-sg:20b`<br>**Gemma:** `gemma3:1b`, `gemma3:4b`, `medgemma:4b`, `medgemma1.5:4b`, `translategemma:4b`, `gemma4-it:e2b` / `e4b` / `12b`, `nemotron3.5-bolt:e4b`<br>**MiniCPM:** `minicpm-v-4.7:1b` |
| Endpoint | `POST /v1/chat/completions` of `flm serve` (what Hermes, Open WebUI and most agents use), streaming or not |
| Not covered | `flm run` (CLI mode) and the Ollama endpoints `/api/chat` and `/api/generate`, which keep the older single-checkpoint reuse. A request on an Ollama endpoint also empties the prefix cache, so a client that mixes both APIs loses its cached prompts. |

Not covered: the single-turn models (`hy-mt2-flash`, `qwen3vl-flash`, `gemma4e-flash`: they take one request and pin their system prompt, so there is nothing to reuse), the developer-preview `*-core` backends, `lfm3`, and the speech, embedding and robotics models. They are unchanged.

It is on by default. Nothing to configure for a single chat or agent.

---

## ⚙️ Options

```shell
flm serve qwen3.5:4b --prefix-cache-mb 8192 --prefix-slots 16
```

| Option | Default | What it does |
|---|---|---|
| `--prefix-cache-mb N` | `4096` | The most host (CPU) memory, in MiB, that saved prompts may use. It is a limit, not a reservation: memory is taken only when a prompt is saved, and given back when it is dropped. |
| `--prefix-slots N` | `8` (at most `32`) | How many saved prompts to keep. `0` turns the prefix cache off. |
| `--prefix-min-gain N` | `128` | After a request, its prompt is saved only if it is at least `N` tokens longer than a saved prompt that already covers it; otherwise the next request resumes from the older one and re-reads the difference. `0` saves every prompt. See [Tuning `--prefix-min-gain`](#tuning---prefix-min-gain). |
| `FLM_DISABLE_PREFIX_SLOTS=1` | off | Environment variable that also turns it off (useful to compare against a cold run). |

When the first prompt is saved, the server logs what a prompt costs and the limits in force:

```text
[FLM]  Prefix cache: up to 8 snapshots in 4096 MiB (a saved row costs 32 KiB, a snapshot's state 49 MiB)
```

### Tuning `--prefix-min-gain`

Every saved prompt costs a copy of the model's state out of the NPU and host memory, and when memory or slots run out it pushes an older prompt out. When a conversation grows by only a few tokens per turn, saving every turn would fill the cache with near-copies of each other. `--prefix-min-gain N` skips a save unless the new prompt adds at least `N` tokens over a saved prompt that already covers it. The server logs it when it skips:

```text
[FLM]  Prefix cache: not saved, only 70 tokens past slot 3 (needs 128, --prefix-min-gain)
```

The price of a skip is that the next request resumes from the older saved prompt and has to read the difference again, which is always **less than `N` tokens** (about a second at most for `N = 512` on a 2B model reading about 700 tokens/s).

| `N` | Effect | Use it when |
|---|---|---|
| `0` | Every prompt is saved: least to re-read per turn, most copying and evictions (83 saves in the test below, as fast as `64`) | A few conversations, plenty of memory and slots |
| `64` | As fast as `0` with a few fewer saves | Same as `0` |
| **`128` (default)** | About 20% fewer saves than `64` for about 0.5% more turn time | Most setups |
| `256` | About 30% fewer saves, about 2% more turn time | Many agents or large prompts, memory or `--prefix-slots` is tight |
| `512` and above | About 45% fewer saves but 7% or more turn time | Rarely worth it: only when copying the state is slower than re-reading `N` tokens (very large models, very slow memory) |

Measured with 6 interleaved agents (own 2.5k-token system prompt, 10 turns each, tool results of 30 to 1500 tokens, plus title-style side calls) on `qwen3.5:2b`, 16 slots, two runs averaged. Turn time is relative to `N = 64`:

| `N` | Saves | Evictions | Cache hit | Turn time |
|---|---|---|---|---|
| 0 (one run) | 83 | 67 | 86.0% | 1.00 |
| 64 | 78 | 62 | 84.3% | 1.00 |
| 128 | 64 | 48 | 83.5% | 1.006 |
| 256 | 56 | 40 | 82.6% | 1.02 |
| 512 | 44 | 28 | 79.8% | 1.07 |
| 1024 | 37 | 21 | 75.3% | 1.15 |

With fewer slots than conversations (8 slots for 6 agents here) the cache thrashes and `N` made no difference to the time. A synthetic workload on one model: measure your own agents (`usage.prompt_tokens_details.cached_tokens` and the time to first token, see below) before moving far from the default.

### How much memory does it need?

A saved prompt costs a fixed amount per model plus an amount per token:

| Model | Per saved token | Fixed per saved prompt | One 15k-token conversation |
|---|---|---|---|
| `qwen3.5:0.8b` / `2b` | 12 KiB | 18 MiB | about 195 MiB |
| `qwen3.5:4b` / `9b` | 32 KiB | 49 MiB | about 511 MiB |
| `qwen3.6-moe:35b-a3b` | 20 KiB | 61 MiB | about 350 MiB |

The other models:

| Model | Per saved token | Fixed per saved prompt | One 4 k-token prompt |
|---|---|---|---|
| `llama3.2:1b` | 32 KiB | none | about 128 MiB |
| `llama3.2:3b` | 112 KiB | none | about 448 MiB |
| `llama3.1:8b`, `deepseek-r1:8b` | 128 KiB | none | about 512 MiB |
| `deepseek-r1-0528:8b`, `qwen3:4b` / `8b`, `qwen3-it:4b`, `qwen3-tk:4b`, `qwen3vl-it:4b` | 144 KiB | none | about 576 MiB |
| `qwen3:0.6b` / `1.7b` | 112 KiB | none | about 448 MiB |
| `qwen2.5-it:3b`, `qwen2.5vl-it:3b` | 36 KiB | none | about 144 MiB |
| `phi4-mini-it:4b` | 128 KiB | none | about 512 MiB |
| `nanbeige4.1:3b` | 64 KiB | none | about 256 MiB |
| `lfm2:1.2b`, `lfm2.5-it:1.2b`, `lfm2.5-tk:1.2b` | 12 KiB | under 1 MiB | about 48 MiB |
| `lfm2:2.6b`, `lfm2-trans:2.6b` | 16 KiB | under 1 MiB | about 64 MiB |
| `gpt-oss:20b`, `gpt-oss-sg:20b` | 24 KiB | 3 MiB | about 99 MiB |
| `gemma3:1b` | 4 KiB | 11 MiB | about 27 MiB |
| `gemma3:4b`, `medgemma:4b`, `medgemma1.5:4b`, `translategemma:4b` | 20 KiB | 116 MiB | about 196 MiB |
| `gemma4-it:e2b` | 6 KiB | 6 MiB | about 30 MiB |
| `gemma4-it:e4b`, `nemotron3.5-bolt:e4b` | 16 KiB | 20 MiB | about 84 MiB |
| `gemma4-it:12b` | 16 KiB | 320 MiB | about 384 MiB |
| `minicpm-v-4.7:1b` | 12 KiB | 18 MiB | about 66 MiB |
| `qwen3.8-mtp:27b` | 84 KiB | 151 MiB | about 487 MiB |

(Measured from the server log for the models we ran; the 3B and 8B Llama and the 0.6B to 8B Qwen3 sizes follow from the model shape.) `qwen3.8-mtp:27b` keeps fewer and smaller defaults, 4 saved prompts in 1536 MiB, because each saved prompt is large; `--prefix-slots` and `--prefix-cache-mb` still override them.

Conversations that share a system prompt share its saved copy, so one agent with several sub-agents costs much less than the same number of unrelated conversations.

Measured with four unrelated 15k-token conversations taking turns and default options, the whole server process (model included) peaked at about 9 GB on `qwen3.5:4b` and about 12 GB on `qwen3.5:9b`.

**Rules of thumb**

- One agent, or a few agents sharing a system prompt: the defaults are enough.
- Several *unrelated* long conversations (different system prompts): each needs up to two saved prompts (its system + tools, and its latest turn). With the default 8 slots about four conversations stay cached; raise `--prefix-slots` (and `--prefix-cache-mb`) for more.
- If one prompt is bigger than `--prefix-cache-mb`, it is not saved and the server says so:

  ```text
  [WARNING]  Prefix cache: a 140000-token prompt needs 4424 MiB, more than --prefix-cache-mb (4096 MiB); it is not saved. Raise --prefix-cache-mb to reuse prompts this long.
  ```

When a limit is reached, FLM drops the least useful saved prompt first: older replies of a conversation that has a newer one, then the least recently used reply, and only then a shared system-prompt copy that nothing depends on.

---

## 🔍 How to see that it works

The response reports how much of the prompt came from the cache:

```json
"usage": {
  "prompt_tokens": 14046,
  "prompt_tokens_details": { "cached_tokens": 14012 }
}
```

The server log shows each decision:

```text
[FLM]  No saved prefix matches, prefilling 13742 tokens          <- first request: nothing to reuse
[FLM]  Prefix slot 3: resuming at 9391 of 9408 tokens            <- a sub-agent reuses the shared system prompt
[FLM]  Prefix cache: dropping the least recently used end snapshot (9420 tokens)
```

---

## 🧠 How it works

1. While FLM reads a prompt it saves a **snapshot** of the model's state at the end of the system prompt and tools, and another one at the end of the prompt.
2. For a new request, FLM looks for the **longest saved snapshot whose text is exactly the beginning of the new prompt**. It restores that snapshot and reads only the rest.
3. Snapshots are matched on the exact tokens, so two different conversations that start with the same system prompt reuse the same snapshot of it. Images are matched by their content: the same picture is reused, a different picture is read again (the text before it is still reused).
4. A short request with its own system prompt (for example the title call some agents send after each turn) does not erase the snapshots of the main conversation.

Real **Hermes** runs (official Docker image), average time to first token per request, stock FLM v1.0.10 against this cache:

| Model | Multi-agent (3 parallel sub-agents) | Chat, 4 turns |
|---|---|---|
| `qwen3.5:2b` | 8.4 s → **4.6 s** | 5.4 s → **3.2 s** |
| `qwen3.5:4b` | 17.8 s → **7.1 s** | 12.1 s → **6.9 s** |
| `qwen3.5:9b` | 27.4 s → **12.8 s** | 17.4 s → **9.8 s** |
| `qwen3.6-moe:35b-a3b` | 48.4 s → **24.7 s** | 30.7 s → **18.6 s** |

In the multi-agent runs the first sub-agent reads the shared system prompt and its siblings reuse it, and the parent agent resumes its own earlier prompt after the sub-agents finish (Qwen3.6: 9.5 s instead of 61 s).

---

## ⚠️ Limits to know about

- **The match is exact and from the start.** Anything that changes the early part of the prompt (a timestamp in the system prompt, tools listed in a different order, an edited system message) breaks the match at that point and everything after it is read again. Keep the system prompt and tool list **identical and in the same order** between requests, and put anything that changes (dates, per-request data) at the end.
- **A repeated identical request** does not resume at its full length: FLM must read at least one token to produce an answer, so a retry resumes at the end of the system prompt + tools.
- **Short side calls are not saved.** A reply is only saved if it adds at least 64 tokens over a snapshot that already covers it, so short side requests cannot push out a long conversation. In exchange, the next turn of a very short chat may read up to 64 more tokens.
- **Qwen3.6 after a tool loop.** Qwen3.6's chat template rewrites the earlier assistant messages (their reasoning blocks) as soon as a newer user message arrives. After a tool-using turn, a new user message can therefore only reuse the system prompt and the first user message, not the whole history. No prompt cache can do better with this template; without the prefix cache nothing is reused.
- **Answers are not bit-for-bit identical to a cold run.** Resuming from a snapshot gives the same quality, but with greedy decoding some answers differ in wording (for example the order of a list) from what the same prompt gives when read from scratch: about 1 in 5 on Qwen3.5 and about 1 in 3 on Qwen3.6 in our replay tests. The cache that came before it behaves the same way.
- **Hermes sub-agents.** Sub-agents reuse the parent's system-prompt snapshot only when every task's `context` field is empty. A per-task context is placed in the sub-agent's system prompt, which makes each sub-agent a different prompt that is read from scratch. Larger models (4B and up) leave it empty in practice; `qwen3.5:2b` sometimes fills it in.
- **Qwen3.8-27B keeps the shared system prompt only.** A saved prompt of this model is large (about 150 MiB plus 84 KiB per token), so it saves the shared start of a conversation (system prompt + tools) and nothing after it, for text prompts. A conversation that keeps going is still continued from the end of its last turn with the older single-checkpoint reuse, as before; the saved system prompt is used when another conversation ran in between, so several agents that share one system prompt do not read it again each time.
- **Gemma 3 has no shared system-prompt snapshot.** Its chat template puts the system prompt inside the first user turn, so two conversations that differ in their first message have no text in common to save as a shared start. Each conversation still reuses its own earlier turns.
- **Models with a sliding window or a convolution state** (Gemma 3 and 4, gpt-oss, the LFM2 family) save a state that cannot be rewound, so a saved prompt is resumed only when the new prompt starts with all of it, and a prompt that shares just the first part of a saved one resumes at the saved system-prompt start instead.
- **Pictures and audio.** The Qwen-VL models, Qwen3.5 and MiniCPM-V reuse prompts that contain pictures. The Gemma 4 models (and the `nemotron3.5-bolt` model built on them) reuse text prompts through the prefix cache; a prompt with a picture or audio keeps the older single-checkpoint reuse. `qwen3.8-mtp:27b` is text only.
- **Platforms.** Tested on Linux with the XRT backend. The Windows and HRX builds contain the same code but have not been run.

---

## 🛠️ Troubleshooting

| You see | Likely cause | What to do |
|---|---|---|
| `cached_tokens` is always `0` | The start of the prompt changes between requests, or the request went to an Ollama endpoint / `flm run` | Compare two requests' system prompt and tools; use `/v1/chat/completions` |
| Every request logs `No saved prefix matches` after a few conversations | More conversations than `--prefix-slots` or `--prefix-cache-mb` allow | Raise `--prefix-slots` (up to 32) and `--prefix-cache-mb` |
| `[WARNING] Prefix cache: a N-token prompt needs ...` | One prompt alone is bigger than `--prefix-cache-mb` | Raise `--prefix-cache-mb` |
| Want to compare against no cache | | `flm serve <model> --prefix-slots 0` |
