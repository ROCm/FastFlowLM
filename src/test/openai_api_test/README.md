# OpenAI API tests

Live tests for `/v1/chat/completions`: output parsing (reasoning, content, tool calls), streaming, and constrained decoding (`tool_choice`, `parallel_tool_calls`, `response_format`). They drive a FastFlowLM server that is already running and need only the Python 3 standard library.

User docs for what's tested here:

- `docs/docs/instructions/server/reasoning_and_tools.md`
- `docs/docs/instructions/server/structured_output.md`
- `docs/docs/instructions/server/chat_output_internals.md` (how it's built)

## Running

```bash
flm serve qwen3:4b                     # in another terminal
python3 run_all.py qwen3:4b            # everything
python3 test_tool_calls.py qwen3:4b    # one module
python3 run_all.py qwen3:4b -k schema  # only cases whose name contains "schema"
```

| Option | Meaning |
|---|---|
| `--endpoint URL` | Server base URL, default `http://127.0.0.1:52625` |
| `--timeout T` | Per-request timeout in seconds, default 300 |
| `--always-thinks` | The model reasons even with `think: false` (`qwen3-tk`, `lfm2.5-tk`, `deepseek-r1`, `deepseek-r1-0528`, `nanbeige4.1`, `gpt-oss`). Don't require reasoning to disappear, and give every answer room for the reasoning before it. |
| `--reasoning-budget N` | With `--always-thinks`, tokens added to each request that expects an answer (`max_tokens` ≥ 32), default 1024 |
| `-k NAME` | Run only matching cases |
| `-v` | Print tracebacks for failures |

Each case prints `PASS`, `FAIL` or `SKIP`. The exit code is non-zero if anything failed. A full run takes about a minute on `qwen3:0.6b`.

## What's covered

| Module | Checks |
|---|---|
| `test_response_shape.py` | Message and chunk shape, one delta kind per chunk, `finish_reason`, `usage` only with `stream_options.include_usage`, stream folds to the non-stream message |
| `test_reasoning.py` | `think: true` → `reasoning_content`, `think: false` / `reasoning_effort: "none"` → none, reasoning streamed before content, switch is per request, reasoning in history is accepted |
| `test_tool_calls.py` | `auto` / `required` / forced function / `none`, `parallel_tool_calls: false`, ids, arguments are JSON matching the tool schema (typed: integer, number, boolean, enum, arrays), `strict: false`, streamed call deltas, a tool round trip |
| `test_response_format.py` | `json_object`, `json_schema` (flat, enum, Pydantic `$defs`/`$ref`, `const`/`anyOf`/bounds/`pattern`), streamed, with and without thinking, schema vs. `auto` tools, `text` |
| `test_logprobs.py` | `logprobs` / `top_logprobs` shape in both modes, greedy token heads its `top_logprobs`, entries spell the content and nothing else (no reasoning, no tool call markup), stream and non-stream report the same tokens, `max_tokens` N writes at most N tokens and `completion_tokens` counts each, malformed values get 400 |
| `test_sampling.py` | The same `seed` samples the same text (with an unseeded request in between, streamed or not), `repeat_last_n: 0` turns penalties off, `-1` covers the whole reply, malformed values get 400 |
| `test_continue_final_message.py` | `continue_final_message` continues a trailing assistant message instead of answering it: content is only the new text, whitespace as written (a model that reasons first may read the prefix as reasoning, and close it), `logprobs` cover exactly the continued tokens (the first entry scores the next token), stream matches non-stream, a cache-resumed continuation matches a cold one, a normal turn can follow, malformed combinations get 400 |
| `test_errors.py` | Every malformed `tool_choice` / `response_format` gets HTTP 400 with an OpenAI error body, streaming too, and the server keeps serving |

Some cases skip by design:

- **Tool cases** skip on models whose chat format has no tool call syntax (the server answers `has no tool call syntax`), and on models too small to be shown tools (`does not support tool calling`).
- **Thinking-on cases** skip on models that never reason.
- **Multi-turn cases** skip on single-turn (flash) models.
- **`continue_final_message` cases** skip on models that cannot continue a message (single-turn models, `gpt-oss`); the server answers `continue_final_message is not supported`.
- **Stream/non-stream parity** compares greedy outputs. Greedy decoding on the NPU isn't always bit-reproducible between identical requests (mostly over long outputs), so the check retries and only fails if the non-streaming output reproduced itself while the stream never matched it. Otherwise it skips.
- **Seed and penalty-window comparisons** in `test_sampling.py` retry the same way, and skip when two identical greedy requests report different logprobs, since then no output of the model's is repeatable.

## Notes

- Requests pin `think`, `top_k: 1`, `temperature: 0` and `repeat_last_n: 64` explicitly. FLM applies sampling and thinking settings to the loaded model and **keeps them for later requests**, so after a run the server is left greedy with thinking as the last case set it. Send your own values, or restart the server, before using it interactively.
- The tests check guarantees, not model quality. Under `auto` a model may or may not call a tool, and the tests only check that what it returns is well formed. Under `required`, a forced function, or `response_format`, the grammar makes the shape a guarantee, so any violation is a server bug.
- Shared helpers live in `flm_api.py`: the HTTP/SSE client, `fold_stream()` (merges tool call deltas by `index`, as clients should), the contract checks, and a small JSON Schema validator. Tools and schemas are in `fixtures.py`.

## Adding a case

```python
@case
def my_case(api: Api) -> None:
    """One line saying what is guaranteed."""
    choice = api.ok(api.body(user("..."), tools=WEATHER_TOOLS, tool_choice="required"))
    flm_api.check_message_contract(choice)
    ...
```

Functions marked `@case` are collected in definition order. Raise `AssertionError` (or use `expect()`) to fail, and `Skip` when the case doesn't apply to the model.
