"""A small HTTP client for the FLM server, standard library only.

The NPU runners sit behind a proxy that intermittently blocks downloads, so the
harness avoids anything that needs `pip install`. This covers what the suites
need: JSON requests, Server-Sent Events streams, and multipart uploads.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field


class HttpError(Exception):
    """A response with a non-2xx status. Carries the status and the body."""

    def __init__(self, status: int, body: str, url: str):
        self.status = status
        self.body = body
        self.url = url
        super().__init__(f"HTTP {status} from {url}: {body[:500]}")

    def json(self):
        try:
            return json.loads(self.body)
        except json.JSONDecodeError:
            return None


@dataclass
class ChatResult:
    """One chat completion, streamed or not, folded into a single shape."""

    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    tool_calls: list = field(default_factory=list)
    chunks: int = 0
    seconds: float = 0.0
    raw: dict | None = None  # the full body of a non-streamed response

    @property
    def cached_tokens(self) -> int | None:
        details = self.usage.get("prompt_tokens_details") or {}
        value = details.get("cached_tokens")
        return value if isinstance(value, int) else None

    def usage_line(self) -> str:
        """The server's prefill/decode accounting as one line, for the log."""
        u = self.usage
        parts = []
        if isinstance(u.get("prompt_tokens"), int):
            part = f"prefill {u['prompt_tokens']} tok"
            if self.cached_tokens:
                part += f" ({self.cached_tokens} cached)"
            parts.append(part)
        if isinstance(u.get("completion_tokens"), int):
            part = f"decode {u['completion_tokens']} tok"
            if isinstance(u.get("decoding_speed_tps"), (int, float)):
                part += f" @ {u['decoding_speed_tps']:.1f} tps"
            parts.append(part)
        if u.get("load_duration"):
            parts.append(f"load {u['load_duration']:.2f}s")
        return " | ".join(parts)


class Client:
    def __init__(self, base_url: str, timeout: float = 600.0):
        # base_url is the server root, e.g. http://127.0.0.1:52625
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- plumbing ------------------------------------------------------------

    def _open(self, method: str, path: str, body: bytes | None, headers: dict,
              timeout: float | None):
        url = f"{self.base_url}{path}"
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            return urllib.request.urlopen(request, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            raise HttpError(exc.code, text, url) from None

    def get_json(self, path: str, timeout: float | None = None):
        with self._open("GET", path, None, {}, timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post_json(self, path: str, body: dict, timeout: float | None = None):
        data = json.dumps(body).encode("utf-8")
        with self._open("POST", path, data, {"Content-Type": "application/json"},
                        timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post_raw(self, path: str, data: bytes, content_type: str,
                 timeout: float | None = None) -> tuple[int, bytes]:
        """POSTs arbitrary bytes and returns (status, body); non-2xx raises."""
        with self._open("POST", path, data, {"Content-Type": content_type},
                        timeout) as resp:
            return resp.status, resp.read()

    def post_multipart(self, path: str, fields: dict, files: dict,
                       timeout: float | None = None) -> tuple[int, bytes]:
        """POSTs multipart/form-data. files maps name -> (filename, bytes, mime)."""
        boundary = f"----flm{uuid.uuid4().hex}"
        out = bytearray()
        for name, value in fields.items():
            if value is None:
                continue
            out += f"--{boundary}\r\n".encode()
            out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            out += str(value).encode("utf-8") + b"\r\n"
        for name, (filename, payload, mime) in files.items():
            out += f"--{boundary}\r\n".encode()
            out += (f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n').encode()
            out += f"Content-Type: {mime}\r\n\r\n".encode()
            out += payload + b"\r\n"
        out += f"--{boundary}--\r\n".encode()
        return self.post_raw(path, bytes(out), f"multipart/form-data; boundary={boundary}",
                             timeout)

    def stream_events(self, path: str, body: dict, timeout: float | None = None,
                      deadline: float | None = None):
        """POSTs body and yields each SSE `data:` payload, parsed, until [DONE].

        The socket timeout bounds the wait for the next line only; deadline
        (seconds) bounds the whole stream, so a model that never stops
        producing tokens still ends the request.
        """
        data = json.dumps(body).encode("utf-8")
        resp = self._open("POST", path, data, {"Content-Type": "application/json",
                                               "Accept": "text/event-stream"}, timeout)
        end = time.monotonic() + (deadline or timeout or self.timeout)
        try:
            for raw_line in resp:
                if time.monotonic() > end:
                    raise TimeoutError(f"stream exceeded {deadline or timeout or self.timeout:.0f}s")
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    return
                yield json.loads(payload)
        finally:
            resp.close()

    # -- endpoints -----------------------------------------------------------

    def version(self) -> str:
        try:
            return str(self.get_json("/v1/version", timeout=10).get("version", "unknown"))
        except Exception:
            return "unknown"

    def chat(self, model: str, messages: list, *, stream: bool = True,
             timeout: float | None = None, **extra) -> ChatResult:
        """One /v1/chat/completions request. extra is merged into the body.

        Streamed requests ask for usage, as an OpenAI client must, so both
        modes come back with the server's usage block.
        """
        body = {"model": model, "messages": messages, "stream": stream, **extra}
        if stream:
            body.setdefault("stream_options", {"include_usage": True})
        started = time.monotonic()
        result = ChatResult()
        if stream:
            calls: dict[int, dict] = {}
            for chunk in self.stream_events("/v1/chat/completions", body, timeout=timeout):
                result.chunks += 1
                if chunk.get("usage"):
                    result.usage = chunk["usage"]
                if "error" in chunk:
                    raise HttpError(500, json.dumps(chunk), "/v1/chat/completions (stream)")
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    result.content += delta.get("content") or ""
                    result.reasoning += delta.get("reasoning_content") or ""
                    for call in delta.get("tool_calls") or []:
                        slot = calls.setdefault(call.get("index", 0),
                                                {"function": {"name": "", "arguments": ""}})
                        fn = call.get("function") or {}
                        slot["function"]["name"] += fn.get("name") or ""
                        slot["function"]["arguments"] += fn.get("arguments") or ""
                        if call.get("id"):
                            slot["id"] = call["id"]
                    if choice.get("finish_reason"):
                        result.finish_reason = choice["finish_reason"]
            result.tool_calls = [calls[i] for i in sorted(calls)]
        else:
            data = self.post_json("/v1/chat/completions", body, timeout=timeout)
            result.raw = data
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            result.content = message.get("content") or ""
            result.reasoning = message.get("reasoning_content") or ""
            result.tool_calls = message.get("tool_calls") or []
            result.finish_reason = choice.get("finish_reason")
            result.usage = data.get("usage") or {}
        result.seconds = time.monotonic() - started
        return result
