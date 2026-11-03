---
layout: docs
title: Reasoning, Tool Calls & Streaming
nav_order: 9
parent: Local Server (Server Mode)
---

# 🧠 Reasoning, Tool Calls & Streaming Output

Every model writes its answer in its own markup. Qwen3 wraps its thinking in `<think>…</think>` and tool calls in `<tool_call>…</tool_call>`. Gemma4 thinks in a `<|channel>thought` channel and writes calls as `call:f{…}`. gpt-oss uses Harmony channels. LFM2 writes Python calls.

FLM's `/v1/chat/completions` endpoint removes that markup for you. Whatever the model, you get the same OpenAI-shaped message:

| Field | What it holds |
|---|---|
| `reasoning_content` | The model's thinking, without the tags. Left out when the model didn't reason. |
| `content` | The answer the user should see. `null` when the turn is only tool calls. |
| `tool_calls` | The function calls the model asked for. Each one has `arguments` as a **JSON string**, whatever syntax the model used. |

This page covers how that output looks with and without streaming, how to turn thinking on and off, and how to read it from the OpenAI Python SDK.

> 📘 To *force* a tool call, ban one, or make the model answer in JSON that matches a schema, see [Structured Output & tool_choice](https://fastflowlm.com/docs/instructions/server/structured_output/).

---

# 📑 Table of Contents

- [Supported Model Formats](#-supported-model-formats)
- [Non-Streaming Responses](#-non-streaming-responses)
- [Streaming Responses](#-streaming-responses)
- [Turning Thinking On and Off](#-turning-thinking-on-and-off)
- [Use Case: Show the Model's Thinking Live](#-use-case-show-the-models-thinking-live)
- [Use Case: A Streaming Tool-Calling Agent Loop](#-use-case-a-streaming-tool-calling-agent-loop)
- [Use Case: Parallel Tool Calls](#-use-case-parallel-tool-calls)
- [Rules Worth Knowing](#-rules-worth-knowing)
- [Troubleshooting](#-troubleshooting)

---

## 🗂️ Supported Model Formats

FLM detects the format once, when the model loads, from its chat template and tokenizer. The server log names the format it detected:

```text
[FLM]  Chat format: hermes_json with reasoning
```

| Format (log name) | Model families | Reasoning | Tool call syntax the model writes |
|---|---|---|---|
| `hermes_json` | Qwen3, Qwen2.5, Nanbeige | `<think>…</think>` | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` |
| `qwen_xml` | Qwen3.5 and later, LFM3 | `<think>…</think>` | `<tool_call><function=f><parameter=k>v</parameter></function></tool_call>` |
| `lfm2_pythonic` | LFM2, LFM2.5 | `<think>…</think>` (thinking variants) | `<|tool_call_start|>[f(a=1, b="x")]<|tool_call_end|>` |
| `gemma4` | Gemma4 | `<|channel>thought … <channel|>` | `<|tool_call>call:f{a:<|"|>x<|"|>}<tool_call|>` |
| `harmony` | gpt-oss | `analysis` channel | `commentary to=functions.f` channel |
| `content_only` | everything else | `<think>…</think>` if the model uses it (e.g. DeepSeek-R1) | none: the model has no tool call syntax |

Your code doesn't depend on any of this. The table explains why a model can or can't call tools, and what its raw output looks like if you ever read the server log.

> 💡 Check the [model card](https://fastflowlm.com/docs/models/) for which models are labelled `tool-calling` and `reasoning`.

---

## 📦 Non-Streaming Responses

A request with tools:

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="flm")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]

resp = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
)

choice = resp.choices[0]
msg = choice.message
print("finish_reason:", choice.finish_reason)                  # "tool_calls"
print("reasoning:", getattr(msg, "reasoning_content", None))   # the <think> text, tags removed
print("content:", msg.content)                                 # None: the turn is only a call
for call in msg.tool_calls or []:
    print(call.id, call.function.name, json.loads(call.function.arguments))
```

The raw JSON body looks like this:

```json
{
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "reasoning_content": "The user wants the weather in Paris, I should call get_weather…",
      "content": null,
      "tool_calls": [{
        "index": 0,
        "id": "call_QbsRMJjkvHumJZVPdR0Gkqsm",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {"prompt_tokens": 180, "completion_tokens": 64, "total_tokens": 244, "...": "..."}
}
```

How to read it:

- **`reasoning_content`** is a FLM extension that other OpenAI-compatible servers also use. The OpenAI SDK keeps unknown fields, so read it with `getattr(msg, "reasoning_content", None)` or `msg.model_extra`.
- **`content` is `null`** when the model only called tools, as with OpenAI. When the model writes text *and* calls tools, both are set.
- **`arguments` is always a JSON string**, even when the model wrote XML (Qwen3.5), Python (`f(a=1)`, LFM2) or Gemma4's relaxed syntax. FLM converts it. When the model's syntax doesn't say whether a value is a string or a number (Qwen XML), FLM uses the type declared in your tool's `parameters`.
- **`id`** is generated by FLM as `call_` plus 24 random characters. Send it back as `tool_call_id` in the `tool` message.
- **`finish_reason`** is:
  - `"tool_calls"` when the model stopped normally after calling at least one tool,
  - `"stop"` when it stopped normally without calls,
  - `"length"` when it hit `max_tokens` or the context limit. A truncated generation is never reported as `"tool_calls"`, even if a call was started. Treat the call's arguments as incomplete.

---

## 🌊 Streaming Responses

With `stream=True` the same message arrives in small deltas. FLM runs the **same parser** for both modes. When the stream ends, the deltas add up to exactly the message a non-streaming request would have returned.

Each chunk's `delta` carries **one** of these:

| Delta | Meaning |
|---|---|
| `{"reasoning_content": "…"}` | More thinking text. Append it. |
| `{"content": "…"}` | More answer text. Append it. |
| `{"tool_calls": [{"index": 0, "id": "call_…", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}` | **A new call starts.** Only the first delta of a call has `id`, `type` and `name`. |
| `{"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\": \"Par"}}]}` | More argument text for call `index`. Append it to that call's `arguments`. |

Then a final chunk with `"delta": {"content": null}` and the `finish_reason`, then `data: [DONE]`.

Here is a real stream from `qwen3:0.6b` (abridged) that calls `get_weather` twice:

```text
data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_803aPw…","type":"function","function":{"name":"get_weather","arguments":""}}]}, …}]}
data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"function":{"arguments":"{\""}}]}, …}]}
data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"function":{"arguments":"city"}}]}, …}]}
data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"function":{"arguments":"\": \"Paris\"}"}}]}, …}]}
data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":1,"id":"call_gDVP0q…","type":"function","function":{"name":"get_weather","arguments":""}}]}, …}]}
data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":1,"function":{"arguments":"{\"city\": \"Tokyo\"}"}}]}, …}]}
data: {"choices":[{"delta":{"content":null},"finish_reason":"tool_calls"}], …}
data: [DONE]
```

### ✅ Accumulating a stream correctly

Merge tool call deltas **by `index`**. Don't append every delta as a new call. Later deltas have no `id` and no `name`.

```python
def collect_stream(stream):
    """Fold a chat.completion.chunk stream into one assistant message."""
    reasoning, content = [], []
    calls = {}          # index -> {"id", "type", "function": {"name", "arguments"}}
    finish_reason = None
    usage = None

    for chunk in stream:
        if chunk.usage:                 # only with stream_options={"include_usage": True}
            usage = chunk.usage
        if not chunk.choices:           # the usage chunk has no choices
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if choice.finish_reason:
            finish_reason = choice.finish_reason

        text = getattr(delta, "reasoning_content", None)
        if text:
            reasoning.append(text)
        if delta.content:
            content.append(delta.content)
        for tc in delta.tool_calls or []:
            call = calls.setdefault(tc.index, {
                "id": None, "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            if tc.id:
                call["id"] = tc.id
            if tc.function and tc.function.name:
                call["function"]["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                call["function"]["arguments"] += tc.function.arguments

    message = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return message, "".join(reasoning), finish_reason, usage
```

`message` can go straight back into `messages` for the next turn. Leave the reasoning out: FLM strips `reasoning_content` from earlier assistant turns before it builds the prompt anyway.

### 📊 Usage in streams

As with OpenAI, a stream reports token usage only when you ask for it:

```python
stream = client.chat.completions.create(..., stream=True,
                                        stream_options={"include_usage": True})
```

Every chunk then has `"usage": null`, and one extra chunk before `[DONE]` has `"choices": []` and the full usage, including FLM's timing fields (`prefill_speed_tps`, `decoding_speed_tps`, …). Check `chunk.choices` before you index it.

---

## 🔀 Turning Thinking On and Off

Reasoning models can be switched per request. These are FLM extensions, so the OpenAI SDK sends them through `extra_body`:

```python
# Turn thinking off: answers come faster, reasoning_content is absent
client.chat.completions.create(model="qwen3:4b", messages=msgs, extra_body={"think": False})

# Turn it back on
client.chat.completions.create(model="qwen3:4b", messages=msgs, extra_body={"think": True})

# The OpenAI-style knob works too: "none" turns thinking off,
# "low" / "medium" / "high" turn it on (gpt-oss also uses it as its effort level)
client.chat.completions.create(model="gpt-oss:20b", messages=msgs, reasoning_effort="low")
```

> ⚠️ **The setting sticks.** `think` and `reasoning_effort` are applied to the loaded model and stay in effect for later requests that don't mention them. If your application mixes thinking and non-thinking calls, send the setting on **every** request.

What changes when thinking is off:

- `reasoning_content` is left out of the response.
- With [constrained decoding](https://fastflowlm.com/docs/instructions/server/structured_output/) active (`tool_choice`, `response_format`), the grammar leaves no room for a reasoning block either, so the model can't slip into one.
- Models that can't switch thinking off (for example `qwen3-tk`, `lfm2.5-tk`, `deepseek-r1`) keep reasoning. Their thinking still arrives in `reasoning_content`, never in `content`.

---

## 💭 Use Case: Show the Model's Thinking Live

A chat UI usually shows the thinking dimmed or folded away, and the answer normally:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="flm")

DIM, RESET = "\033[2m", "\033[0m"
in_thinking = False

stream = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "A bat and a ball cost $1.10 in total. "
                                          "The bat costs $1.00 more than the ball. "
                                          "How much is the ball?"}],
    stream=True,
    extra_body={"think": True},
)
for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    thinking = getattr(delta, "reasoning_content", None)
    if thinking:
        if not in_thinking:
            print(f"{DIM}🤔 ", end="")
            in_thinking = True
        print(thinking, end="", flush=True)
    if delta.content:
        if in_thinking:
            print(f"{RESET}\n\n", end="")
            in_thinking = False
        print(delta.content, end="", flush=True)
print(RESET)
```

You never see `<think>` or `</think>` in either field, even half-written. FLM holds back a few characters whenever the text might be the start of a tag, until it can tell.

---

## 🔁 Use Case: A Streaming Tool-Calling Agent Loop

This loop keeps calling the model until it answers without asking for a tool. It uses `collect_stream()` from [above](#-accumulating-a-stream-correctly).

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="flm")
MODEL = "qwen3:4b"

tools = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "convert_currency",
        "description": "Convert an amount between two currencies",
        "parameters": {"type": "object",
                       "properties": {"amount": {"type": "number"},
                                      "from": {"type": "string"},
                                      "to": {"type": "string"}},
                       "required": ["amount", "from", "to"]}}},
]

def get_weather(city):
    return {"city": city, "temperature_c": 18, "sky": "cloudy"}

def convert_currency(amount, **kw):
    return {"amount": round(amount * 1.08, 2), "currency": kw["to"]}

IMPLEMENTATIONS = {"get_weather": get_weather, "convert_currency": convert_currency}

messages = [{"role": "user",
             "content": "I'm flying to Paris with 200 USD. What's the weather there, "
                        "and how many euros is that?"}]

for step in range(5):                       # never loop forever
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, tools=tools, stream=True,
        extra_body={"think": False},
    )
    message, _reasoning, finish_reason, _ = collect_stream(stream)
    messages.append(message)

    if finish_reason != "tool_calls":
        print(message["content"])
        break

    for call in message["tool_calls"]:
        fn = IMPLEMENTATIONS[call["function"]["name"]]
        args = json.loads(call["function"]["arguments"])
        result = fn(**args)
        print(f"🔧 {call['function']['name']}({args}) -> {result}")
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps(result),
        })
```

Things this loop relies on:

1. **`finish_reason == "tool_calls"`** is the signal to run tools. It's only set when the model stopped normally, so a `"length"` cut-off falls through instead of running a half-written call.
2. **The assistant message goes back with its `tool_calls`**, and each result goes back as a `tool` message with the matching `tool_call_id`. FLM rebuilds the model's native markup from these for the next prompt.
3. **`json.loads(arguments)` is safe** once the stream has ended. While streaming, `arguments` is a growing prefix of the final JSON. Don't parse it before the call is complete.

---

## 🧵 Use Case: Parallel Tool Calls

Ask for two things and a capable model calls the same tool twice. Each call has its own `index` and `id`:

```python
resp = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Weather in Paris and in Tokyo?"}],
    tools=tools,
)
for call in resp.choices[0].message.tool_calls or []:
    print(call.index, call.id, call.function.arguments)
# 0 call_Gce8tV… {"city": "Paris"}
# 1 call_sCOOJo… {"city": "Tokyo"}
```

Answer **every** call with its own `tool` message before the next request. To get at most one call per turn, send `parallel_tool_calls=False` (see [Structured Output & tool_choice](https://fastflowlm.com/docs/instructions/server/structured_output/#-parallel_tool_calls)).

---

## 📏 Rules Worth Knowing

- **Tools are only parsed when the request has `tools`.** Without them, any tool markup the model writes stays in `content` as plain text.
- **`tool_choice: "none"` hides the tools from the model** entirely. They're not put in the prompt, so the model answers in prose.
- **Text and calls can interleave.** A model may write a sentence, call a tool, then write more. Both parts are returned: the text pieces are joined in `content` and the calls are in `tool_calls`. When separate text pieces are joined, they're separated by a blank line (`\n\n`).
- **Whitespace at the edges is trimmed.** `content` and `reasoning_content` never start or end with blank lines. While streaming, trailing whitespace is held back until more text follows it.
- **Text after the model hands over to a tool is dropped.** Gemma4 marks the end of its turn with `<|tool_response>`, and gpt-oss with `<|call|>`. Generation stops there.
- **Only `/v1/chat/completions` does this parsing.** `/v1/completions` returns the raw text.

---

## 🩺 Troubleshooting

| Symptom | Likely cause |
|---|---|
| Tool markup like `<tool_call>` shows up in `content` | The request had no `tools`, or the model's format is `content_only` (check the `Chat format:` log line). |
| My client creates one tool call per streamed chunk | Merge deltas by `index`, as in [`collect_stream()`](#-accumulating-a-stream-correctly). |
| `json.loads(arguments)` fails | You parsed while still streaming, or `finish_reason` is `"length"`: raise `max_tokens`. |
| `reasoning_content` is empty but I asked for thinking | A previous request sent `think: false` and the setting stuck. Send `think: true` explicitly. |
| Long pause before the first content token | The model is thinking. Stream it and read `reasoning_content`, or turn thinking off. |
| `usage` is missing in streams | Pass `stream_options={"include_usage": True}`. |
