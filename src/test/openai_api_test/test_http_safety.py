#!/usr/bin/env python3
"""HTTP layer safety: bad requests get a 4xx and never take the server down, and CORS,
Origin and Host follow the server's flags.

No model is needed: every request goes to a route that answers before generation (a missing
field, /api/show, /v1/models, a refused Origin), so the model argument is only a label.

The CORS cases adapt to how the server was started and skip what does not apply:

* ``--cors 0``: no response carries any ``Access-Control-*`` header;
* ``--cors 1`` (default): pages on this machine (localhost, 127.0.0.1, [::1], any port) get
  their Origin echoed, a browser SDK preflight (``x-stainless-*``) passes, other origins 403;
* ``--cors-origins "*"``: every origin is served with ``Access-Control-Allow-Origin: *``.

Run it once per mode to cover all three. A loopback-bound server also refuses a foreign
``Host`` (DNS rebinding) with 403.

Detecting the mode cannot tell a server started with ``--cors-origins "*"`` from one that has
no Origin policy at all, so a CI run declares what it started: ``FLM_TEST_CORS=off|local|any``
(the mode) and ``FLM_TEST_HOST_CHECK=1`` (bound to loopback). A server in another mode then
fails the CORS cases instead of skipping them.

Usage::

    python3 test_http_safety.py <any-label> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import time
import urllib.parse
from typing import Any

import flm_api
from flm_api import Api, Skip, case, expect

LOCAL_ORIGIN = "http://localhost:3000"
FOREIGN_ORIGIN = "http://evil.example"


def address(api: Api) -> tuple[str, int]:
    u = urllib.parse.urlparse(api.url)
    return u.hostname or "127.0.0.1", u.port or 80


def call(api: Api, method: str, path: str, body: Any = None, *, content_type: str = "application/json",
         headers: dict[str, str] | None = None, timeout: float = 10.0) -> tuple[int, dict[str, str], Any]:
    """One request on its own connection; returns status, lower-cased headers, JSON (or text) body."""
    host, port = address(api)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
        hdrs.setdefault("Content-Type", content_type)
    skip_host = "Host" in hdrs
    conn.putrequest(method, path, skip_host=skip_host, skip_accept_encoding=True)
    for k, v in hdrs.items():
        conn.putheader(k, v)
    if data is not None:
        conn.putheader("Content-Length", str(len(data)))
    conn.endheaders(data)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", "replace")
    resp_headers = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    try:
        parsed: Any = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        parsed = raw
    return resp.status, resp_headers, parsed


def cors_of(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.startswith("access-control-")}


def alive(api: Api) -> bool:
    try:
        return call(api, "GET", "/v1/models", timeout=5)[0] == 200
    except OSError:
        return False


def expect_400(status: int, body: Any, param: str | None, where: str) -> None:
    expect(status == 400, f"{where}: status {status}, want 400 (body {body!r:.200})")
    err = body.get("error") if isinstance(body, dict) else None
    expect(isinstance(err, dict), f"{where}: error is not an object: {body!r:.200}")
    expect(err.get("code") == 400 and err.get("type") == "invalid_request_error", f"{where}: {err}")
    if param is not None:
        expect(err.get("param") == param, f"{where}: param {err.get('param')!r}, want {param!r}")


def cors_mode(api: Api) -> str:
    """'off', 'local' or 'any', from how the server answers a preflight and a foreign origin.

    With FLM_TEST_CORS set, a server in another mode is a failure, not a skip."""
    status, headers, _ = call(api, "OPTIONS", "/v1/chat/completions",
                              headers={"Origin": LOCAL_ORIGIN, "Access-Control-Request-Method": "POST"})
    if status != 200 or not cors_of(headers):
        mode = "off"
    else:
        status, headers, _ = call(api, "GET", "/v1/models", headers={"Origin": FOREIGN_ORIGIN})
        mode = "any" if status == 200 else "local"
    declared = os.environ.get("FLM_TEST_CORS")
    if declared:
        expect(mode == declared, f"FLM_TEST_CORS={declared} but the server behaves as '{mode}'")
    return mode


# ---------------------------------------------------------------- missing fields, request_id

MISSING = [
    ("/api/show", {}, "model"),
    ("/api/generate", {}, "prompt"),
    ("/api/generate", {"model": "x"}, "prompt"),
    ("/api/chat", {}, "messages"),
    ("/api/chat", {"model": "x", "messages": "hi"}, "messages"),
    ("/api/embeddings", {}, "model"),
    ("/api/embeddings", {"model": "x"}, "input"),
    ("/api/embeddings", {"model": "x", "input": [1, 2]}, "input"),
    ("/v1/embeddings", {"input": "hi"}, "model"),
]


@case
def missing_fields_are_400_and_the_server_lives(api: Api) -> None:
    """{} or a missing/mistyped field is a 400 naming it, as JSON or text/plain, never a crash."""
    for path, body, param in MISSING:
        for ctype in ("application/json", "text/plain"):
            status, _, parsed = call(api, "POST", path, body, content_type=ctype)
            expect_400(status, parsed, param, f"POST {path} {body} [{ctype}]")
    for path in ("/api/show", "/api/chat", "/v1/embeddings"):
        status, _, parsed = call(api, "POST", path, "", content_type="text/plain")
        expect(400 <= status < 500, f"POST {path} empty body: status {status} {parsed!r:.200}")
    expect(alive(api), "server stopped answering after missing-field requests")


@case
def non_string_request_id_keeps_threads(api: Api) -> None:
    """More non-string request_ids than there are I/O threads, then the server still answers."""
    for i in range(12):  # the server runs 10 I/O threads; once killed each, it would stop answering
        for path in ("/api/show", "/api/cancel"):
            status, _, parsed = call(api, "POST", path, {"model": "x", "request_id": 123})
            expect(status < 500, f"{path} request {i + 1}: status {status} {parsed!r:.200}")
    expect(alive(api), "server stopped answering after non-string request_id requests")


@case
def cancel_rejects_non_string_request_id(api: Api) -> None:
    """/api/cancel with a numeric request_id is a 400, whatever the Content-Type."""
    for ctype in ("application/json", "text/plain"):
        status, _, parsed = call(api, "POST", "/api/cancel", '{"request_id": 123}', content_type=ctype)
        expect_400(status, parsed, "request_id", f"/api/cancel [{ctype}]")


# ---------------------------------------------------------------- routing


@case
def query_and_trailing_slash_route(api: Api) -> None:
    """/v1/models?x=1 and /v1/models/ reach /v1/models."""
    for path in ("/v1/models?x=1", "/v1/models/"):
        status, _, body = call(api, "GET", path)
        expect(status == 200 and isinstance(body, dict) and body.get("object") == "list", f"GET {path}: {status}")


@case
def not_found_is_json(api: Api) -> None:
    """An unknown route is a 404 with a JSON body and Content-Type, with or without a body."""
    for method, body in (("GET", None), ("POST", {"a": 1})):
        status, headers, parsed = call(api, method, "/no/such/route", body)
        expect(status == 404, f"{method} 404: status {status}")
        expect(headers.get("content-type", "").startswith("application/json"), f"{method} 404: {headers.get('content-type')}")
        expect(isinstance(parsed, dict) and "error" in parsed, f"{method} 404 body: {parsed!r:.200}")


# ---------------------------------------------------------------- CORS, Origin, Host


@case
def cors_off_writes_no_cors_headers(api: Api) -> None:
    """--cors 0: no Access-Control-* on a 200, a 400, a 404, or the preflight."""
    if cors_mode(api) != "off":
        raise Skip("server runs with CORS on")
    for method, path, body in (("GET", "/v1/models", None), ("POST", "/api/show", {}),
                               ("GET", "/no/such/route", None), ("OPTIONS", "/v1/chat/completions", None)):
        _, headers, _ = call(api, method, path, body, headers={"Origin": LOCAL_ORIGIN})
        expect(not cors_of(headers), f"{method} {path}: {cors_of(headers)}")


@case
def local_page_preflight_passes(api: Api) -> None:
    """--cors 1: a page on localhost gets its Origin echoed and the SDK headers allowed."""
    if cors_mode(api) != "local":
        raise Skip("server is not in the default CORS mode")
    asked = "content-type, authorization, x-stainless-os, x-stainless-lang, openai-beta"
    status, headers, _ = call(api, "OPTIONS", "/v1/chat/completions", headers={
        "Origin": LOCAL_ORIGIN, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": asked})
    expect(status == 200, f"preflight status {status}")
    expect(headers.get("access-control-allow-origin") == LOCAL_ORIGIN, f"ACAO {headers.get('access-control-allow-origin')}")
    allowed = headers.get("access-control-allow-headers", "").lower()
    for h in ("x-stainless-os", "openai-beta", "authorization"):
        expect(h in allowed, f"preflight does not allow {h}: {allowed}")
    status, headers, _ = call(api, "GET", "/v1/models", headers={"Origin": LOCAL_ORIGIN})
    expect(status == 200 and headers.get("access-control-allow-origin") == LOCAL_ORIGIN, f"GET: {status} {cors_of(headers)}")
    expect("origin" in headers.get("vary", "").lower(), f"Vary: {headers.get('vary')}")
    ext = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"  # browser-extension UIs, as Ollama allows
    status, headers, _ = call(api, "GET", "/v1/models", headers={"Origin": ext})
    expect(status == 200 and headers.get("access-control-allow-origin") == ext, f"extension: {status} {cors_of(headers)}")


@case
def foreign_origin_refused(api: Api) -> None:
    """--cors 0 or 1: a page from another origin gets a 403 before anything runs."""
    if cors_mode(api) == "any":
        raise Skip("server allows every origin")
    for method, path, body in (("GET", "/v1/models", None), ("POST", "/api/cancel", {"request_id": "req_0"}),
                               ("POST", "/api/pull", {"model": "x"})):
        status, headers, parsed = call(api, method, path, body, content_type="text/plain",
                                       headers={"Origin": FOREIGN_ORIGIN})
        expect(status == 403, f"{method} {path} from {FOREIGN_ORIGIN}: {status}")
        expect(isinstance(parsed, dict) and parsed.get("error", {}).get("code") == 403, f"body {parsed!r:.200}")
        expect("access-control-allow-origin" not in headers, f"403 grants the origin: {cors_of(headers)}")


@case
def any_origin_mode_is_wildcard(api: Api) -> None:
    """--cors-origins "*": every origin is served with ACAO "*" (the behavior before WP-06)."""
    if cors_mode(api) != "any":
        raise Skip('server not started with --cors-origins "*"')
    status, headers, _ = call(api, "GET", "/v1/models", headers={"Origin": FOREIGN_ORIGIN})
    expect(status == 200 and headers.get("access-control-allow-origin") == "*", f"{status} {cors_of(headers)}")


@case
def foreign_host_refused(api: Api) -> None:
    """A loopback-bound server refuses a request naming a foreign Host (DNS rebinding)."""
    host, port = address(api)
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise Skip("the endpoint is not a loopback address")
    if cors_mode(api) == "any":
        raise Skip('--cors-origins "*" turns the Host check off')
    status, _, _ = call(api, "GET", "/v1/models", headers={"Host": f"localhost:{port}"})
    expect(status == 200, f"Host localhost: {status}")
    status, _, parsed = call(api, "GET", "/v1/models", headers={"Host": f"evil.example:{port}"})
    if status == 200 and os.environ.get("FLM_TEST_HOST_CHECK") != "1":
        raise Skip("server is bound to a non-loopback address (no Host check)")
    expect(status == 403, f"Host evil.example: {status} {parsed!r:.200}")


@case
def over_connection_cap_reply_follows_cors(api: Api) -> None:
    """The 503 for a connection over --socket carries no ACAO unless every origin is allowed."""
    mode = cors_mode(api)
    host, port = address(api)
    held: list[socket.socket] = []
    try:
        reply = b""
        for _ in range(64):
            sock = socket.create_connection((host, port), timeout=5)
            held.append(sock)
            time.sleep(0.05)
            probe = socket.create_connection((host, port), timeout=5)
            probe.sendall(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nOrigin: " + LOCAL_ORIGIN.encode() + b"\r\n\r\n")
            reply = probe.recv(4096)
            probe.close()
            if reply.startswith(b"HTTP/1.1 503"):
                break
        if not reply.startswith(b"HTTP/1.1 503"):
            raise Skip("could not reach the connection cap with 64 idle sockets")
        has_acao = b"access-control-allow-origin" in reply.lower()
        expect(has_acao == (mode == "any"), f"mode {mode}: 503 reply ACAO present={has_acao}")
    finally:
        for s in held:
            s.close()
        time.sleep(1.0)
    expect(alive(api), "server stopped answering after the connection-cap probe")


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
