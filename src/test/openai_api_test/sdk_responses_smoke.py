"""Optional SDK smoke: python sdk_responses_smoke.py MODEL --endpoint http://127.0.0.1:52625

Requires openai-python and a loaded model that supports forced function calls.
The normal HTTP test runner does not depend on the SDK.
"""
import argparse
import secrets

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--endpoint", default="http://127.0.0.1:52625")
    args = parser.parse_args()
    client = OpenAI(base_url=args.endpoint.rstrip("/") + "/v1", api_key="local", max_retries=0)
    opts = dict(model=args.model, temperature=0, max_output_tokens=1024, store=False)
    prompt = "Say OK."
    one = client.responses.create(input=prompt, **opts)
    with client.responses.stream(input=prompt, **opts) as stream:
        streamed = stream.get_final_response()
    for response in (one, streamed):
        assert response.object == "response" and response.status == "completed" and response.output_text.strip()

    tools = [{"type":"function", "name":"read_marker", "description":"Read the current marker.",
              "parameters":{"type":"object","properties":{},"required":[]}}]
    history = [{"role":"user","content":"Call read_marker and reply with its exact result."}]
    first = client.responses.create(input=history, tools=tools, tool_choice={"type":"function","name":"read_marker"}, **opts)
    call, = (item for item in first.output if item.type == "function_call")
    assert first.status == "completed" and call.name == "read_marker" and call.call_id
    marker = "fixture-" + secrets.token_hex(4)
    history.extend(item.model_dump(exclude_none=True) for item in first.output)
    history.append({"type":"function_call_output","call_id":call.call_id,"output":marker})
    second = client.responses.create(input=history, **opts)
    assert second.status == "completed" and marker in second.output_text
    print("PASS SDK create, accumulated stream, and function result replayed by call_id")


if __name__ == "__main__":
    main()
