"""Builds the test clips from PNGs already in the FastFlowLM repo.

The customer demo (120 frames of 398x224 RGB8) is licensed material and the repo is
public, so it is not shipped. This clip has the same shape: three 40-frame "shots",
each a slow pan across one repo image. Integer math only, so Linux and Windows build
the same bytes (checked by sha256 in the results).
"""

import hashlib
import os

from pngio import read_rgb

FW, FH = 398, 224
SHOTS = ["src/tb_files/sekiro.png", "src/tb_files/panda.png", "docs/assets/architecture.png"]


def _crop(img, x, y, w, h):
    iw, _, rgb = img
    return b"".join(rgb[((y + r) * iw + x) * 3:((y + r) * iw + x + w) * 3] for r in range(h))


def _pan(img, n, w, h):
    iw, ih, _ = img
    return [_crop(img, (iw - w) * k // (n - 1), (ih - h) * k // (2 * (n - 1)), w, h) for k in range(n)]


def _upscale(img, factor_num, factor_den):
    iw, ih, rgb = img
    ow, oh = iw * factor_num // factor_den, ih * factor_num // factor_den
    cols = [x * factor_den // factor_num for x in range(ow)]
    rows = []
    for y in range(oh):
        sy = y * factor_den // factor_num
        src = rgb[sy * iw * 3:(sy + 1) * iw * 3]
        px = [src[c * 3:c * 3 + 3] for c in range(iw)]
        rows.append(b"".join(px[c] for c in cols))
    return ow, oh, b"".join(rows)


class Clips:
    def __init__(self, repo, cache_dir):
        self.repo, self.cache = repo, cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._imgs = {}

    def _img(self, rel):
        if rel not in self._imgs:
            self._imgs[rel] = read_rgb(os.path.join(self.repo, rel))
        return self._imgs[rel]

    def _cached(self, name, build):
        path = os.path.join(self.cache, name)
        if os.path.exists(path):
            return open(path, "rb").read()
        data = build()
        with open(path, "wb") as f:
            f.write(data)
        return data

    def main(self):
        """120 x 224 x 398 x 3 bytes, frame-major (THWC)."""
        return self._cached("clip120.rgb", lambda: b"".join(
            b"".join(_pan(self._img(s), 40, FW, FH)) for s in SHOTS))

    def frames(self, n):
        """n frames of the main clip, wrapping past 120 (for the 121..128 limit cases)."""
        clip = self.main()
        size = FW * FH * 3
        return b"".join(clip[(i % 120) * size:(i % 120 + 1) * size] for i in range(n))

    def portrait(self, n):
        return self._cached(f"portrait{n}.rgb", lambda: b"".join(_pan(self._img(SHOTS[0]), n, FH, FW)))

    def hd(self, n):
        """n frames of 1920x1080 from a 2x upscale of the first shot."""
        def build():
            big = _upscale(self._img(SHOTS[0]), 2, 1)
            return b"".join(_pan(big, n, 1920, 1080))
        return self._cached(f"hd{n}.rgb", build)

    def digest(self):
        return hashlib.sha256(self.main()).hexdigest()
