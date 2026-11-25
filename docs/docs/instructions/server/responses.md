---
layout: docs
title: Stateless Responses
parent: Local Server (Server Mode)
---

# Stateless Responses

`POST /v1/responses` generates a turn with the same local model as Chat
Completions. The client owns the history and executes any functions. FLM does
not store a response for later retrieval, even when the request says `store: true`.

| Field | What FLM does |
|---|---|
| `model`, `input` | Supply a local chat model and either a string or a nonempty array of ordered text messages, reasoning items, function calls and results. No image, audio, file or item-ID references. |
| `instructions` | Prepend a system message for this request. Resend it on the next turn if it should still apply. |
| `tools` | Accept only flat `{"type":"function","name":...}` declarations. Any other type, even in a mixed list, rejects the whole request with HTTP 400 and `error.param` such as `tools[1].type`. A client function may have any supported function name, including `web_search`. |
| `tool_choice`, `parallel_tool_calls` | Use the same constraints as [Chat Completions](../structured_output/). A named choice has the form `{"type":"function","name":"read"}`. |
| `max_output_tokens`, `temperature`, `top_p` | Pass the output budget and sampling settings to the chat engine. |
| `reasoning.effort` | Pass the effort to the model. Support and accepted levels depend on the [model](../reasoning_and_tools/). The client can resend reasoning items as part of `input`; the chat core removes past reasoning before rendering the next prompt. Encrypted reasoning is rejected. |
| `text.format` | `text`, `json_object`, or `json_schema`. For `json_schema`, put `name` and `schema` directly in `format`. Model and schema limits match Chat Completions. |
| `stream` | `false` returns one `response` object. `true` returns named SSE events, not `chat.completion.chunk` objects. |
| `store` | Accept either boolean; always report `store: false`. There is no GET or DELETE by response ID. |
| `previous_response_id`, `conversation`, `background` | HTTP 400 naming the supplied field. Omit the field rather than passing `null` or `false`. |

The `output` array can contain assistant text (`message`), model reasoning
(`reasoning` summary text), and `function_call` items. The stream identifies
each item by `output_index` and item ID. Check the terminal status before
using the output: `completed`, `incomplete` with `reason: "max_output_tokens"`,
or `failed` with an error. A partial function call stays in the terminal snapshot
with `status: "incomplete"`; it does not receive a callable `done` event.
Errors before streaming use an HTTP error response.
Usage counts include reasoning in output tokens; a separate reasoning-token
count is not measured and is reported as zero.

## Continue after a function call

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="local")
history = [{"role": "user", "content": "Read the counter and tell me its value."}]
tools = [{"type": "function", "name": "read_counter",
          "parameters": {"type": "object", "properties": {}, "required": []}}]
first = client.responses.create(model="qwen3:4b", input=history, tools=tools,
                                tool_choice={"type": "function", "name": "read_counter"})
assert first.status == "completed"
history.extend(item.model_dump(exclude_none=True) for item in first.output)
for call in first.output:
    if call.type == "function_call" and call.name == "read_counter":
        history.append({"type": "function_call_output", "call_id": call.call_id,
                        "output": "42"})  # The client reads its own counter here.
answer = client.responses.create(model="qwen3:4b", input=history)
print(answer.output_text)
```

Set a Responses client's base URL to the server's `/v1`. For Pi, select the
`openai-responses` provider API. For Codex, use a Responses provider and turn
off web search and multi-agent namespaces if the client version offers them
(`web_search="disabled"` and `features.multi_agent=false`). The request itself
must declare only functions; asking the model not to search does not remove a
tool declaration. FLM does not run hosted tools, remote MCP services, or background jobs.
