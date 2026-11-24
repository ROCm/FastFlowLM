---
layout: docs
title: "Under the Hood: Output Parsing & Constrained Decoding"
nav_order: 11
parent: Local Server (Server Mode)
---

# 🔬 Under the Hood: Output Parsing & Constrained Decoding

This page is for contributors. It explains how `/v1/chat/completions` turns raw model text into `reasoning_content` / `content` / `tool_calls`, and how `tool_choice` and `response_format` restrict decoding. For the user-facing behaviour, see [Reasoning, Tool Calls & Streaming](https://fastflowlm.com/docs/instructions/server/reasoning_and_tools/) and [Structured Output & tool_choice](https://fastflowlm.com/docs/instructions/server/structured_output/).

All of it lives in one shared library, `src/common/chat/` with headers in `src/include/chat/`. Models don't carry their own parsers any more.

---

## 🗺️ The Big Picture

```text
              request                                              response
                 │                                                     ▲
                 ▼                                                     │
 chat_constraint_request_from_openai()          ┌──────── SSE chunks / JSON body
   tool_choice, parallel_tool_calls,            │                      ▲
   response_format, think/reasoning_effort      │       chat_compute_deltas()   (stream only)
                 │                              │                      ▲
                 ▼                              │       chat_parse(spec, prefill + text)
 AutoModel::make_constraint()                   │                      ▲
   └─ chat_make_constraint(spec, …)             │                      │ generated text
        ├─ chat_structural_tag() ──► chat_grammar_engine (xgrammar) ──► decode_constraint
        └─ token_ban_constraint (tool_choice=none)                     │
                                                                       ▼
                         decode loop:  _sample():  fill_mask ─► Sampler::sample(logits, mask)
                                                   accept(token)
```

Both halves are driven by one `chat_format_spec`, detected once per model. **The grammar that constrains decoding and the parser that reads the output are built from the same spec, so they can't disagree about the syntax.**

| File | Role |
|---|---|
| `chat_format.{hpp,cpp}` | `chat_format_spec`, detection from the template, prefill computation |
| `chat_parser.{hpp,cpp}` | `chat_parse()`: text → `chat_msg` for every format |
| `chat_args.{hpp,cpp}` | Prefix-stable conversion of XML / Python / Gemma4 arguments to JSON |
| `chat_msg.{hpp,cpp}` | `chat_msg`, `chat_msg_delta`, `chat_compute_deltas()` |
| `chat_stream.{hpp,cpp}` | Per-request streaming state, tool call ids |
| `chat_constraint.{hpp,cpp}` | OpenAI request → `chat_constraint_request` → `decode_constraint` |
| `chat_grammar.{hpp,cpp}` | xgrammar structural tags per format, `chat_grammar_engine` |
| `decode_constraint.hpp` | The interface the decode loop talks to, plus `token_ban_constraint` |

---

## 1️⃣ Format Detection

`AutoModel::_shared_setup_tokenizer()` calls `_detect_chat_format()`, which runs `chat_format_detect(template_src, is_token)`. The detector checks markers from most specific to least, and looks at **both** the Jinja template and the tokenizer. Some templates never render the markers the model emits: LFM2.5-Thinking prints tool calls as plain content, and the R1 distills never mention `<think>`.

| Check (in order) | Result |
|---|---|
| `<|channel|>` and `<|message|>` known | `HARMONY` |
| `<|tool_call>` and `<|channel>` known | `GEMMA4` |
| `<|tool_call_start|>` known | `LFM2_PYTHONIC` |
| Template contains `<function=` and `<parameter=` | `QWEN_XML` |
| Template contains `<tool_call>` | `HERMES_JSON` (the token alone isn't trusted: some vocabularies carry it without training) |
| otherwise | `CONTENT_ONLY`; `reasoning_forced` for DeepSeek-R1 style templates |

"Known" means present in the template or a single token in the vocabulary. `<think>` / `</think>` turn on reasoning for the tag-based kinds.

`extra_stops` holds markers that end the turn without being in the model's eos list (`<|call|>` for gpt-oss, `<|tool_response>` for Gemma4). They're added to `eos_token_ids` at load time.

The server log prints the result: `Chat format: hermes_json with reasoning`.

### Prefill

The generation prompt often opens a block the model then continues. Qwen3 with thinking on ends the prompt with `<think>\n`, and Harmony ends it with `<|start|>assistant`. `_begin_turn()` decodes the last 32 prompt tokens into `meta_info.prompt_tail`, and `chat_format_prefill(spec, prompt_tail)` returns the part the parser and grammar must see first:

- tag formats: `reasoning_start` if the tail opened a reasoning block that it didn't close, else `""` (or `reasoning_start` when `reasoning_forced`);
- Harmony: everything after the last `<|start|>`.

The parser reads `prefill + generated_text`. The constraint gets the same prefill in `start()`.

---

## 2️⃣ The Parser

```cpp
chat_msg chat_parse(const chat_format_spec& spec, const std::string& text,
                    bool is_partial, const chat_parse_options& options);
```

One function serves both streaming and non-streaming. The design depends on one invariant:

> **Prefix property.** For a partial text `t` and any continuation `t + u`, every field of `chat_parse(t, partial=true)` is a prefix of the same field of `chat_parse(t + u, …)`.

That's what lets the stream send plain suffixes as deltas. Three techniques keep it:

1. **Partial markers are held back.** `cut_partial_marker()` drops a tail that could be the start of a marker (`<thi`, `</tool_ca`, …). The next parse either completes the marker or shows the text.
2. **Text is trimmed.** Each reasoning/content segment is trimmed and segments are joined with `\n\n`, so trailing whitespace only shows up once something follows it.
3. **Arguments are converted prefix-stably** (`chat_args.cpp`). Strings stream as they arrive. A bare number or `true`/`false` is held until it ends, because its JSON form depends on text not seen yet. XML, Python and Gemma4 argument syntaxes go through `chat_relaxed_args_to_json()` or `parse_xml_arguments()`. When the syntax doesn't type a value (Qwen XML), the tool's declared JSON-schema type decides.

`parse_tagged()` handles all tag formats with one scanner: content until the next `reasoning_start` / `tool_start`, reasoning until `reasoning_end`, a tool block until `tool_end`, dispatched by kind (`parse_hermes_block`, `parse_qwen_xml_block`, `parse_lfm2_block`, `parse_gemma4_block`). `stop_marker` truncates the text. `parse_harmony()` walks `<|start|>…<|channel|>…<|message|>…<|end|>` messages: `analysis` becomes reasoning, `to=functions.x` becomes a call, and everything else becomes content.

`chat_parse_options.parse_tool_calls` is false when the request has no tools or `tool_choice` is `"none"`. Tool markup then stays in content.

---

## 3️⃣ Streaming

`chat_turn_streambuf` (in `src/server/chat_core.hpp`) owns a `chat_stream`. On every flush of complete UTF-8:

```cpp
std::vector<chat_msg_delta> chat_stream::update(const std::string& text, bool is_final);
```

1. appends the text and **re-parses the whole turn** (a scan of a few KB per token, which is noise next to a decode step);
2. assigns a stable `call_<24 chars>` id to each new tool call;
3. returns `chat_compute_deltas(prev, cur)`: the new reasoning suffix, the new content suffix, and per tool call either a first delta (id + name + arguments so far) or an arguments suffix.

If a field ever fails to extend its previous value, that's a parser bug. `chat_compute_deltas` logs `chat parser rewrote already streamed …` and skips the field rather than sending a contradiction.

Each delta goes to the request's `chat_sink`; for `/v1/chat/completions` that is `openai_chat_sink` (in `src/server/protocol_openai_chat.hpp`), which writes it as a chunk.

`finish()` runs a last `update("", true)` so held-back text is released. `RestHandler::run_chat_turn` then promotes the finish reason to `tool_calls` only on a clean EOT stop and reports the end of the turn; the sink sends the final chunk, the usage chunk when `stream_options.include_usage` was set, then `[DONE]`.

The non-streaming path of `RestHandler::run_chat_turn` calls `chat_parse(…, is_partial=false, …)` once and hands the sink the whole message, so the two modes agree by construction.

---

## 4️⃣ Constrained Decoding

### From request to constraint

`chat_constraint_request_from_openai()` validates and normalizes `tool_choice`, `parallel_tool_calls`, `response_format` and thinking (`think: false` or `reasoning_effort: "none"` → `allow_reasoning = false`). A field set to `null` reads as missing. Errors become HTTP 400.

`AutoModel::make_constraint()` also clears `allow_reasoning` when the *model* has thinking off (`thinking_enabled()`). When the model leaves tools out of its prompt (`tools_enabled()` false, e.g. Qwen3 up to 1.7B), it rejects `required` and the function form with a 400 and drops the tools for `auto`. It then builds a `chat_vocab` and calls `chat_make_constraint()`, which returns one of:

- an **xgrammar constraint** when `chat_structural_tag()` returns a grammar;
- a **`token_ban_constraint`** for `tool_choice: "none"` when the tool opener is a single token;
- **null**: nothing to constrain, and the decode loop pays nothing.

### Structural tags

`chat_structural_tag(spec, request)` emits an [xgrammar](https://github.com/mlc-ai/xgrammar) structural tag:

| Request | Grammar shape (tag formats) |
|---|---|
| `auto` + tools | `[reasoning?] triggered_tags(trigger = tool opener, tags = one per tool)`: free text, but every call is fully constrained |
| `required` | same, with `at_least_one` |
| function form | `[reasoning?] ws <one tag for that tool>` |
| `parallel_tool_calls: false` | `stop_after_first` on the triggered tags |
| `response_format` | `[reasoning?] ws json_schema(schema)`: `{"type": "object"}` for `json_object` |

`[reasoning?]` is an optional `reasoning_start any_text reasoning_end ws` block. It's there only when reasoning is allowed or the model forces it. It's optional so that a prefill that already opened it is accepted.

Per-tool tags follow xgrammar's builtin styles where one exists, and are hand-written otherwise:

| Format | Tag |
|---|---|
| `HERMES_JSON` | `<tool_call>\n{"name": "f", "arguments": ` + JSON schema + `}\n</tool_call>` (xgrammar `qwen_3`) |
| `QWEN_XML` | `<tool_call>\n<function=f>\n` + schema in `qwen_xml` style + `\n</function>\n</tool_call>`; booleans widened to `True/False/true/false` |
| `GEMMA4` | `<|tool_call>call:f` + EBNF from `gemma4_arguments_grammar()` + `<tool_call|>` |
| `LFM2_PYTHONIC` | one tag `<|tool_call_start|>[` + EBNF `python_calls_grammar()` + `]<|tool_call_end|>` holding all calls |
| `HARMONY` | `harmony_format()`: analysis / final / `commentary to=functions.f` messages joined by `<|start|>assistant` (xgrammar `harmony` style, with both spacings gpt-oss uses) |

The EBNF argument grammars (`argument_list_rules()`) put declared properties in one fixed order, require `required` ones, make optional ones skippable, and type values from the schema (string enums spelled out, array `items` followed). They bound numbers to 16 digits and whitespace to a few characters, because a constrained model otherwise tends to pad forever.

### The engine

`chat_grammar_engine` is built lazily, once per model (`Grammar engine ready in … ms`). It reads `tokenizer.json` for the token strings, and builds xgrammar's `TokenizerInfo` and `GrammarCompiler`. Stop tokens that the grammar matches *as text* (`<|call|>`, `<|return|>`, `<|end|>` for Harmony) are kept out of xgrammar's stop list.

`make()` compiles the request's tag into an `xgrammar_constraint`:

- `start(prefill)` feeds the prefill (Harmony drops the leading `assistant` role) into the matcher;
- `fill_mask()` fills a `uint32` bitset over the vocabulary;
- `accept(token)` advances; `rollback(n)` undoes;
- `accept_forced(token)` handles a token the model is *made* to emit;
- if thinking was off in the grammar but the model reasons anyway, because the template opened `<think>` in the prefill or the model is forced to emit it, `start()` / `accept_forced()` switch to a precompiled grammar that has reasoning and replay the prefill and history into it;
- if the matcher ever rejects a token, `give_up()` logs `Constrained decoding disabled for this request: …` and stops masking. A desynchronised grammar would only produce garbage.

### The decode loop

`AutoModel::_sample(logits, meta_info)` is the only way a token gets sampled:

1. `constraint->fill_mask(constraint_mask_)`;
2. `sampler->sample(logits, &mask)`: `Sampler::sampler_allowed_apply()` sets masked logits to `-inf` on the sampler's **own fp32 copy**, after repetition penalties and before top-k/top-p. The engine's device-mapped logits buffer is never written. Greedy sampling copies too when a mask is present;
3. if the sample still lands outside the mask (only with broken logits, e.g. all NaN), the best allowed token is used instead;
4. `constraint->accept(token)`.

`insert()` samples the first token through `_sample()` and sets `meta_info.insert_sample_pending`. Models whose `generate()` feeds preamble tokens before decoding have to keep the constraint in step:

| Helper | When |
|---|---|
| `_force_token(meta_info, tok)` | A preamble token that **is** part of the output (e.g. a forced `<think>\n`). It rolls back the unused insert sample and calls `accept_forced()`. |
| `_skip_insert_sample(meta_info)` | Preamble tokens that are **not** output (the empty think block fed when thinking is off). It rolls back the unused insert sample only. |

See `modeling_qwen3_5vl.cpp` and `modeling_minicpm_v.cpp` for examples, and `src/create_new_model.md` §2f–2g.

---

## ➕ Adding a New Format

1. Add a `chat_format_kind` and its name in `FORMAT_NAMES` (`chat_format.cpp`).
2. Give it markers in `chat_format_make()`, and a detection rule in `chat_format_detect()` placed by specificity.
3. Parse its tool block in `parse_tool_block()`. Keep the prefix property: hold partial markers back, and stream strings but hold scalars. If the arguments aren't JSON, add a dialect to `chat_args`.
4. Add its tags in `tool_tags()` (`chat_grammar.cpp`). Prefer an xgrammar builtin style. Otherwise write an EBNF grammar on `argument_list_rules()`.
5. If the grammar matches any eos token as text, list it in `text_markers()`.
6. Verify with `chat_grammar_engine::accepts(spec, request, prefill, text, error)`. It feeds a whole output through the grammar and reports the byte where it fails, so you can check that real model outputs are accepted.
7. Run the live API tests in `src/test/openai_api_test/` against a model of that family.

---

## 🧪 Tests

`src/test/openai_api_test/` holds stdlib-only Python scripts that drive a running server. They check the response contract, stream/non-stream agreement, the thinking switch, every `tool_choice` mode, `response_format` schema conformance, and the 400 errors. See its `README.md`.
