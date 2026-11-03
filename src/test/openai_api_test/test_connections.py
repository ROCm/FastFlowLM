#!/usr/bin/env python3
"""Connection slots: every connection must give its slot back, however it ends.

The server accepts at most ``--socket`` connections at once (default 10) and
closes the rest on accept. A client that hangs up on a streaming reply used to
keep its slot forever, so after that many dropped streams the server turned
every connection away until it was restarted. These cases drop more clients
than the default limit allows, then check the limit is still exactly what it
should be: ``limit - 1`` idle sockets leave room for one request, ``limit``
idle sockets do not. A leaked slot fails the first check, a slot released twice
fails the second.

Needs a server started with the default ``--socket 10``.

Usage::

    python3 test_connections.py <model> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

import json
import socket
import time
import urllib.parse
import urllib.request
from typing import Any

import flm_api
from flm_api import Api, case, expect
from fixtures import user

LIMIT = 10  # the default --socket
DROPS = LIMIT + 2  # more than the limit, so a leak per drop cannot go unnoticed
BARRIER_S = 180.0  # most we wait for the server to work through the dropped requests


def address(api: Api) -> tuple[str, int]:
    parts = urllib.parse.urlsplit(api.url)
    return parts.hostname or "127.0.0.1", parts.port or 80


def send_chat(api: Api, body: dict[str, Any]) -> socket.socket:
    payload = json.dumps(body).encode("utf-8")
    sock = socket.create_connection(address(api), timeout=api.timeout)
    sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                 b"Content-Type: application/json\r\nContent-Length: " + str(len(payload)).encode()
                 + b"\r\n\r\n" + payload)
    return sock


def drop_clients(api: Api, *, stream: bool, wait_for_reply: bool) -> None:
    """Send DROPS long requests and hang up on each one."""
    body = api.body(user("Write a long story about a dragon."), max_tokens=200, stream=stream)
    for _ in range(DROPS):
        sock = send_chat(api, body)
        if wait_for_reply:
            sock.recv(4096)  # the first streamed bytes, or the whole non-streaming reply
        else:
            time.sleep(0.05)
        sock.close()
        time.sleep(0.2)
    drain(api)


def drain(api: Api) -> None:
    """Wait until the server has unwound every dropped request.

    A dropped request keeps its slot until it is dequeued and cancelled, and the
    server runs one request at a time, so a short request that completes proves
    all the earlier ones are gone. (It may be turned away while the queue is full.)
    """
    deadline = time.monotonic() + BARRIER_S
    while time.monotonic() < deadline:
        try:
            if api.post(api.body(user("Say hi."), max_tokens=4)).status == 200:
                time.sleep(0.5)
                return
        except OSError:
            pass  # turned away (limit reached) or reset: try again
        time.sleep(1.0)
    expect(False, f"server did not get through the dropped requests within {BARRIER_S:.0f}s")


def models_ok(api: Api) -> bool:
    """GET /v1/models; False when the server turns the connection away."""
    url = api.url.replace("/chat/completions", "/models")
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            resp.read()
            return resp.status == 200
    except OSError:
        return False


def idle_sockets(api: Api, count: int) -> list[socket.socket]:
    socks = [socket.create_connection(address(api), timeout=5) for _ in range(count)]
    time.sleep(0.3)  # let the server accept them all
    return socks


def expect_exact_limit(api: Api) -> None:
    """The limit still admits LIMIT connections in total, and no more."""
    expect(models_ok(api), "server turns away a lone request")
    for idle, admitted in ((LIMIT - 1, True), (LIMIT, False)):
        socks = idle_sockets(api, idle)
        try:
            got = models_ok(api)
        finally:
            for sock in socks:
                sock.close()
        time.sleep(0.5)
        expect(got == admitted, f"with {idle} idle connections a request was "
                                f"{'served' if got else 'turned away'}, expected "
                                f"{'served' if admitted else 'turned away'} (leaked or double-released slots)")


@case
def dropped_streams_release_their_slot(api: Api) -> None:
    """More hang-ups mid-stream than the limit allows leave the limit intact."""
    drop_clients(api, stream=True, wait_for_reply=True)
    expect_exact_limit(api)


@case
def dropped_streams_before_first_chunk(api: Api) -> None:
    """Hanging up straight after sending a streaming request (during prefill)."""
    drop_clients(api, stream=True, wait_for_reply=False)
    expect_exact_limit(api)


@case
def dropped_non_streaming_requests(api: Api) -> None:
    """Hanging up on a non-streaming request while it generates."""
    drop_clients(api, stream=False, wait_for_reply=False)
    expect_exact_limit(api)


@case
def finished_requests_release_their_slot(api: Api) -> None:
    """Completed streaming and non-streaming requests release exactly once."""
    for stream in (True, False):
        for _ in range(DROPS):
            body = api.body(user("Say hi."), max_tokens=8)
            if stream:
                api.ok_stream(body)
            else:
                api.ok(body)
    expect_exact_limit(api)


@case
def idle_and_malformed_connections(api: Api) -> None:
    """Connections that open and close, or send garbage, release their slot."""
    host, port = address(api)
    for _ in range(DROPS):
        socket.create_connection((host, port), timeout=5).close()
    for junk in (b"GARBAGE\r\n\r\n", b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 5\r\n\r\nabc"):
        for _ in range(DROPS):
            sock = socket.create_connection((host, port), timeout=5)
            sock.sendall(junk)
            time.sleep(0.05)
            sock.close()
    time.sleep(1.0)
    expect_exact_limit(api)


@case
def server_still_serves(api: Api) -> None:
    """After all the hang-ups a normal request still succeeds."""
    flm_api.check_message_contract(api.ok(api.body(user("Say hi."), max_tokens=16)))


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
