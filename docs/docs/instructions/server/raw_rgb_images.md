---
layout: docs
title: Raw RGB Images (video frames)
nav_order: 13
parent: Local Server (Server Mode)
---

# 🎞️ Raw RGB Images (video frames)

A program that already holds decoded pixels, for example a video pipeline, does not need to encode them to JPEG or PNG just so FLM can decode them again. `minicpm-v-4.7:1b` accepts **raw 8-bit RGB frames** directly, as an addition to the normal `image_url` input, on `POST /v1/chat/completions`.

Every frame is treated as an **independent picture**, exactly as if it had been sent as a PNG. MiniCPM-V 4.7 has no video mode, so nothing about time is passed to the model: to describe order or timing, say so in the prompt. Other models answer a raw frame with HTTP 400.

## Request

Add an `image_raw` part to the message content:

```json
{
  "type": "image_raw",
  "image_raw": {
    "pixel_format": "RGB8",
    "layout": "THWC",
    "dtype": "uint8",
    "shape": [120, 224, 398, 3],
    "max_slice_nums": 1,
    "data": "<base64 of the 32,094,720 bytes>"
  }
}
```

- `shape` is `[frames, height, width, 3]`; `data` is the pixels in that order (frame by frame, row by row, R, G, B per pixel), base64 encoded.
- Nothing is converted: `pixel_format`, `layout` and `dtype` must be exactly `RGB8`, `THWC` and `uint8`, and the number of bytes must match `shape`. Anything else is a 400 that says what is wrong.
- `max_slice_nums` is optional, see below.
- Instead of `data`, a frame file can be named with `"path"`, see below.
- A message may hold several `image_raw` parts, and may mix them with `image_url` parts; pictures keep the order of the parts.

```python
import base64, numpy as np, requests

frames = np.fromfile("frames.rgb", dtype=np.uint8).reshape(120, 224, 398, 3)
part = {"type": "image_raw", "image_raw": {
    "pixel_format": "RGB8", "layout": "THWC", "dtype": "uint8",
    "shape": list(frames.shape), "data": base64.b64encode(frames.tobytes()).decode()}}
r = requests.post("http://localhost:52625/v1/chat/completions", json={
    "model": "minicpm-v-4.7:1b", "temperature": 0, "max_tokens": 512,
    "messages": [{"role": "user", "content": [part, {"type": "text", "text": "Describe the video."}]}]})
print(r.json()["choices"][0]["message"]["content"])
```

For the same pixels, the answer is identical to sending each frame as a PNG (lossless). A JPEG of the same frame is lossy, so its answer can differ.

## Frames from a file (`path`)

If the frames are already in a file, or your program can write them to one, send its path instead of the base64 `data`; the request then carries a few hundred bytes instead of tens of megabytes and nothing is base64-encoded:

```json
{"type": "image_raw", "image_raw": {"pixel_format": "RGB8", "layout": "THWC", "dtype": "uint8",
                                    "shape": [120, 224, 398, 3], "path": "/data/frames/clip42.rgb"}}
```

The file holds exactly the bytes `data` would decode to. Give exactly one of `data` and `path`. This is **off by default**, because it lets a request make the server read a file: start the server with `--allow-raw-path <directory>` and only regular files inside that directory (after symlinks and `..` are resolved) are read. Anything else, including a missing file, gets the same 400. The path must be local: network and device paths (`\\server\share\...`, `//server/share/...`, `\\?\UNC\...`, `\\.\...`) are refused before anything is opened; a relative path is read from the server's working directory. The file size must match `shape` exactly. FLM reads the file into memory during the request, so it may be rewritten once the response is out. Frames are the same, so the answer is identical to the `data` form; for 120 frames of 398 × 224 it saves about 0.8 s of the 20 s.

## Limits

| Limit | Default | Option |
|---|---|---|
| Frames in one request | `128` | `--max-raw-frames N` |
| Raw pixel bytes in one request | `128` MiB | `--max-raw-mb N` |
| Prompt tokens of the pictures | about `17300` | not adjustable |

The server accepts a request body of up to 256 MB, and base64 makes the pixels a third bigger.

The token limit is the one that usually bites. MiniCPM-V scales a picture to about a 448 × 448 pixel area, so a frame costs about **73 prompt tokens**, whatever its size (measured at 398 × 224 and 1280 × 720). The prompt, with all the pictures in it, has to fit one prefill, and the engine's buffer for that stops at about 17,300 tokens: **about 235 frames**. A request over the limit gets a 400 that names the token count; send fewer frames. `--ctx-len` must also be large enough for the pictures plus the answer (`--ctx-len 32768` fits the full limit).

### Slicing: `max_slice_nums`

A large picture is normally cut into up to 9 extra slices to keep detail, which costs about 64 more tokens each (a 1280 × 720 PNG is 477 tokens). Video frames are usually better as one view each, so raw frames are **not sliced by default** (`max_slice_nums` is 1). Add `"max_slice_nums": N` (1 to 9) to the `image_raw` object to allow slicing; with `9` the result is identical to sending the frame as an `image_url`.

Prefill runs at about 430 to 500 tokens per second on the NPU, so 120 frames of 398 × 224 (8,937 prompt tokens) take about 18 to 21 seconds before the first word.
