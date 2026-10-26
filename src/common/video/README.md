# Video input for d3:turbo (CPU side)

What this directory is: the part of FastFlowLM that turns a request's `videos` into what the d3 engine reads, byte for
byte as the upstream runtime (`d3_runtime.py` pin v1.1, `d3_compose.py`) does. No NPU code. The engine integration
(vision tower per temporal group, window loop, box / track) consumes `video/video_preprocess.hpp`'s `VideoPayload`.

| file | content |
|---|---|
| `video_sampling.cpp` | `video_sample_indices`, frame-list thinning, even padding, `window_plan`, `group_times`, moments, `_video_box` refusals, row budget |
| `video_preprocess.cpp` | video `smart_resize` (0.2 MP per frame, banker's rounding), torchvision-exact bicubic resize, normalize, patchify, timestamp text, pad runs, token budget, 3D M-RoPE positions |
| `video_decode.cpp` | `_decode_video` / `_decode_windows` / `_frames_video` on top of a `FrameReader` |
| `ffmpeg_dl.cpp` | the FFmpeg `FrameBackend`, libraries loaded at run time |
| `video_input.cpp` | one `videos[i]` value (data URL, frame list) -> `VideoClip` |

## Accepted inputs (`videos`: a list of any number)

* `"data:video/<mp4|webm|quicktime|x-matroska>[;params];base64,<data>"`, at most 32,000,000 bytes, 8,294,400 pixels per
  frame, **60 seconds** (design decision A4; upstream's server allows 300). Decoded with FFmpeg.
* extension: `"data:image/gif;base64,..."` (or `data:video/gif`), decoded with FFmpeg.
* extension: `{"frames": ["data:image/png;base64,...", ...]}`, PNG / JPEG / WebP frames of one size (at most 2,048, thinned
  to 32, read as a clip at 2 frames per second). Needs no video decoder, always available.
* anything else (paths, http(s) URLs) is refused with upstream's strict-mode text.

All videos of a request together may take 16,384 input tokens (one token per 32 x 32 pixels per two frames).

## Decoder: FFmpeg, loaded at run time

There is no build-time dependency. The first request with an encoded video loads `libavformat`, `libavcodec`,
`libavutil` and `libswscale` (FFmpeg 6.x, 7.x or 8.x: sonames 60/58/7, 61/59/8, 62/60/9) with `dlopen` (`LoadLibrary`
on Windows: `avformat-60.dll`, ...). Without them the request is refused with 422
`video inputs are not available: video decoding backend not available (<why>)`; frame lists keep working.

* Ubuntu / Debian: `sudo apt install libavformat60 libavcodec60 libavutil58 libswscale7` (FFmpeg 6.1, Ubuntu 24.04);
  newer releases ship the 7.x / 8.x sonames. Fedora: `sudo dnf install ffmpeg-free-libs` (or the RPM Fusion `ffmpeg-libs`).
* Windows: put an LGPL shared build (`avformat-6x.dll`, `avcodec-6x.dll`, `avutil-5x.dll`, `swscale-x.dll`) next to
  `flm.exe`, or set `FLM_FFMPEG_DIR`.
* `FLM_FFMPEG_DIR=<dir>` is searched first. CMake option `FLM_VIDEO_FFMPEG` (default ON) controls run-time loading; `-DFLM_VIDEO_FFMPEG=OFF` leaves the decoder out (the build then answers encoded videos with 422 and does not list `video` / `track`).

**Licensing.** FLM does not contain, link against, bundle or redistribute FFmpeg: no FFmpeg code or header is compiled into `flm`,
`libd3_npu.so` / `d3_npu.dll` or the xclbins (the binaries only carry the library names above and our own wrapper code, which
talks to the public C ABI through `dlopen`). The FFmpeg libraries are the user's own installation and stay under their own
licence (LGPL-2.1+, or GPL when the distribution built them with GPL options). Do not add FFmpeg binaries or headers to FLM's
installers, packages or repository; if a package ever has to ship them, use an LGPL-only shared build and include its licence
text and source offer. The only file that includes FFmpeg headers is the optional test `test_ffmpeg_layout.cpp`, which builds
against the system headers and is not part of any shipped binary.

Decoding follows OpenCV's FFmpeg capture (what upstream uses): first video stream, sequential decode, swscale
`SWS_BICUBIC` to RGB24, frame count = `nb_frames` else duration x fps + 0.5, display-matrix rotation applied. Against OpenCV 5.0 (its
bundled FFmpeg) the test clips (H.264 mp4 / mov / mkv, VP9 webm, GIF, rotated) decode bit-identically.

## Tests

`src/test/d3_video` (`cmake -S src/test/d3_video -B build && cmake --build build && ctest --test-dir build`).
Goldens come from running the real `Qwen3VLVideoProcessor` and upstream functions (`gen_fixtures.py`, a Python with transformers, torchvision and OpenCV, CPU).
The FFmpeg tests skip when the libraries are missing.
