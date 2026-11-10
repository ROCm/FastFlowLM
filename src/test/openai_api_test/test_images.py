#!/usr/bin/env python3
"""Image parts: a usable image reaches the model, anything else gets a 400.

An ``image_url`` part the server could not use used to be dropped without a
word: the request got a 200 and the model answered about an image it never saw.
Malformed parts (no ``url``, ``image_url`` given as a string, non-base64 data,
an http URL) got a 500 instead. Now every image is checked before generation
starts, and one that is not a PNG, JPEG or WebP image the server can decode,
within the pixel limit, refuses the whole request with an OpenAI style 400
naming the problem. So does an image outside a user message, which OpenAI does
not accept and the chat templates drop.

The refusals happen before the model sees anything, so those cases run on any
model. The cases that check an image reaches the model need a vision model:
they send a red and a blue image, and skip when the model cannot name the
colours of a red and a blue PNG (a text model may well guess "red", but not
both). Two colours also keep a dropped image from passing on a lucky guess.

All images are built here, so the module needs only the standard library.

Usage::

    python3 test_images.py <model> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any

import flm_api
from flm_api import Api, case, expect

QUESTION = "What color is this image? Answer in one word."


# ---------------------------------------------------------------- images


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def png(width: int, height: int, rgb: tuple[int, int, int] = (255, 0, 0), *, pixels: bool = True) -> bytes:
    """A solid-colour 8-bit RGB PNG. Without ``pixels`` the IDAT is empty: only the header is honest."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = (b"\0" + bytes(rgb) * width) * height if pixels else b""
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b"")


RED_PNG = png(64, 64)
BLUE_PNG = png(64, 64, (0, 0, 255))
# 64 x 64 solid red / blue, lossless WebP (ffmpeg -c:v libwebp -lossless 1)
RED_WEBP = base64.b64decode("UklGRiIAAABXRUJQVlA4TBUAAAAvP8APAAcQ9Y/+B4CE8P+9FtH/1A8A")
BLUE_WEBP = base64.b64decode("UklGRiIAAABXRUJQVlA4TBUAAAAvP8APAAcQ0f/+B4CE8P+9FtH/1A8A")
GIF = (b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00"
       b",\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;")
BMP = (b"BM" + struct.pack("<IHHI", 58, 0, 0, 54) + struct.pack("<IiiHHIIiiII", 40, 1, 1, 1, 24, 0, 4, 0, 0, 0, 0)
       + b"\x00\x00\xff\x00")
TIFF = b"II*\x00\x08\x00\x00\x00" + b"\x00" * 16


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def data_url(data: bytes, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64,{b64(data)}"


def image_part(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def ask(*parts: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": QUESTION}, *parts]}]


# ---------------------------------------------------------------- checks


def expect_400(api: Api, messages: list[dict[str, Any]], message_part: str) -> None:
    """Refused with an OpenAI style 400 that names the problem, streaming too, before any chunk."""
    body = api.body(messages, max_tokens=8)
    resp = api.post(body)
    expect(resp.status == 400, f"expected HTTP 400, got {resp.status}: {resp.body}")
    err = resp.body.get("error")
    expect(isinstance(err, dict), f"error body is not an object: {resp.body}")
    expect(err.get("type") == "invalid_request_error", f"error type is {err.get('type')!r}")
    expect(message_part in str(err.get("message")), f"message {err.get('message')!r} lacks {message_part!r}")

    s = api.stream(body)
    expect(s.status == 400, f"streaming: expected HTTP 400, got {s.status}")
    expect(not s.chunks, "streaming: chunks were sent before the rejection")


def answer(api: Api, messages: list[dict[str, Any]]) -> str:
    return str(api.ok(api.body(messages, max_tokens=16))["message"].get("content") or "")


def names_colours(api: Api, red_url: str, blue_url: str) -> tuple[bool, str]:
    """Whether the model names both colours right, and what it said."""
    red = answer(api, ask(image_part(red_url)))
    blue = answer(api, ask(image_part(blue_url)))
    ok = "red" in red.lower() and "blue" not in red.lower() and "blue" in blue.lower() and "red" not in blue.lower()
    return ok, f"red image -> {red!r}, blue image -> {blue!r}"


def require_vision(api: Api) -> None:
    """Skip unless the model sees PNGs: a text model answers 200 but cannot tell a red one from a blue one."""
    ok, said = names_colours(api, data_url(RED_PNG), data_url(BLUE_PNG))
    if not ok:
        raise flm_api.Skip(f"the model cannot name the colours of PNG images (not a vision model?): {said}")


def expect_seen(api: Api, red_url: str, blue_url: str) -> None:
    require_vision(api)
    ok, said = names_colours(api, red_url, blue_url)
    expect(ok, f"the images did not reach the model: {said}")


# ---------------------------------------------------------------- accepted


@case
def webp_data_url(api: Api) -> None:
    """A WebP data URL reaches the model (only png/jpeg prefixes used to be recognised)."""
    expect_seen(api, data_url(RED_WEBP, "image/webp"), data_url(BLUE_WEBP, "image/webp"))


@case
def media_type_any_case(api: Api) -> None:
    """Media types are case-insensitive: data:image/PNG reaches the model."""
    expect_seen(api, data_url(RED_PNG, "image/PNG"), data_url(BLUE_PNG, "IMAGE/png"))


@case
def bare_base64(api: Api) -> None:
    """Bare base64 without a data: prefix is still accepted."""
    expect_seen(api, b64(RED_WEBP), b64(BLUE_WEBP))


# ---------------------------------------------------------------- refused


@case
def unsupported_formats(api: Api) -> None:
    """GIF, BMP and TIFF are refused by name."""
    for data, media_type, name in ((GIF, "image/gif", "GIF"), (BMP, "image/bmp", "BMP"), (TIFF, "image/tiff", "TIFF")):
        expect_400(api, ask(image_part(data_url(data, media_type))), f"unsupported image format {name}")


@case
def truncated_png(api: Api) -> None:
    """A PNG cut short."""
    expect_400(api, ask(image_part(data_url(RED_PNG[: len(RED_PNG) // 2]))), "truncated or corrupt")


@case
def not_an_image(api: Api) -> None:
    """Bytes that are no image at all."""
    junk = bytes((i * 151 + 7) % 256 for i in range(2048))
    expect_400(api, ask(image_part(data_url(junk))), "not a PNG, JPEG or WebP image")


@case
def over_pixel_limit(api: Api) -> None:
    """An image larger than the pixel limit is refused from its header, before any decoding."""
    expect_400(api, ask(image_part(data_url(png(20000, 20000, pixels=False)))), "over the limit")


@case
def empty_url(api: Api) -> None:
    expect_400(api, ask(image_part("")), "is empty")


@case
def not_base64(api: Api) -> None:
    expect_400(api, ask(image_part("data:image/png;base64,!!!not*base64###")), "not valid base64")


@case
def remote_url(api: Api) -> None:
    """http(s) URLs are not fetched."""
    expect_400(api, ask(image_part("https://example.com/cat.png")), "remote image URLs are not supported")


@case
def malformed_parts(api: Api) -> None:
    """image_url without url, image_url as a string, a non-image or non-base64 data URL."""
    expect_400(api, ask({"type": "image_url", "image_url": {}}), "'image_url.url' is missing")
    expect_400(api, ask({"type": "image_url", "image_url": data_url(RED_PNG)}), "'image_url' must be an object")
    expect_400(api, ask(image_part(data_url(b"hello world", "text/plain"))), "is not an image")
    expect_400(api, ask(image_part("data:image/png,abc")), "must be base64-encoded")


@case
def image_outside_user_message(api: Api) -> None:
    """Images in a system or assistant message."""
    system = [{"role": "system", "content": [{"type": "text", "text": "Be brief."}, image_part(data_url(RED_PNG))]},
              {"role": "user", "content": "What color is the image?"}]
    expect_400(api, system, "images are only supported in user messages")
    assistant = [{"role": "user", "content": "Hi."},
                 {"role": "assistant", "content": [image_part(data_url(RED_PNG))]},
                 {"role": "user", "content": "What color was it?"}]
    expect_400(api, assistant, "images are only supported in user messages")


@case
def one_bad_image_refuses_the_request(api: Api) -> None:
    """A good image next to a bad one: the whole request is refused, naming the bad part."""
    expect_400(api, ask(image_part(data_url(RED_PNG)), image_part(data_url(GIF, "image/gif"))), "content[2]")


@case
def serves_after_refusals(api: Api) -> None:
    """The refusals leave the server serving."""
    api.ok(api.body([{"role": "user", "content": "Say hi."}], max_tokens=4))


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
