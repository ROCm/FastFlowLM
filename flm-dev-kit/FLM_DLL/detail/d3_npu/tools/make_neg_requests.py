# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Request files for the negative tests of encode_state_once's M3 / M4 validation (run on any package with d3_req_run --keep-going).

    python make_neg_requests.py <out dir>      -> <out dir>/*.req and list.txt

Expected errors (printed as "error,<file>,<what>"): pad without image_ids, image id outside the table, run shorter / longer than its
image, image_ids without runs, video pad in a state without media, odd grid, state run not matching media, state-image pad in a block
without question images, readout offset out of range; ok_two_images (one block with the same table image twice and another one between,
plus a state image) must run.
"""
import os
import sys

import numpy as np

from req_file import to_bf16_bits, write_request

IMG, VID = 248056, 248057
rng = np.random.default_rng(3)


def pix(n):
    return to_bf16_bits(rng.normal(0, 0.5, (n, 1536)).astype(np.float32))


def txt(n):
    return [int(x) for x in rng.integers(1000, 2000, n)]


def q(ids, im=None):
    return {"ids": ids, "offsets": [0], "image_ids": im or []}


cases = {
    "pad_no_ids": dict(state=txt(10), media=None, qi=None, qs=[q(txt(3) + [IMG] * 4 + txt(2))]),
    "id_range": dict(state=txt(10), media=None, qi=([[4, 4]], pix(16)), qs=[q(txt(3) + [IMG] * 4 + txt(2), im=[1])]),
    "run_short": dict(state=txt(10), media=None, qi=([[4, 4]], pix(16)), qs=[q(txt(3) + [IMG] * 3 + txt(2), im=[0])]),
    "run_long": dict(state=txt(10), media=None, qi=([[4, 4]], pix(16)), qs=[q(txt(3) + [IMG] * 5 + txt(2), im=[0])]),
    "fewer_runs": dict(state=txt(10), media=None, qi=([[4, 4]], pix(16)), qs=[q(txt(5), im=[0])]),
    "video_no_media": dict(state=txt(5) + [VID] * 4 + txt(2), media=None, qi=None, qs=[q(txt(3))]),
    "odd_grid": dict(state=txt(10), media=None, qi=([[3, 4]], pix(12)), qs=[q(txt(3) + [IMG] * 3, im=[0])]),
    "state_run": dict(state=txt(2) + [IMG] * 3 + txt(2), media=([[4, 4]], pix(16)), qi=None, qs=[q(txt(3))]),
    "media_pad_in_block": dict(state=txt(2) + [IMG] * 4 + txt(2), media=([[4, 4]], pix(16)), qi=None, qs=[q(txt(3) + [IMG] * 4)]),
    "readout_range": dict(state=txt(10), media=None, qi=None, qs=[{"ids": txt(3), "offsets": [3], "image_ids": []}]),
    "ok_two_images": dict(state=txt(2) + [IMG] * 4 + txt(2), media=([[4, 4]], pix(16)),
                          qi=([[4, 4], [4, 6]], np.concatenate([pix(16), pix(24)])),
                          qs=[q(txt(2) + [IMG] * 4 + txt(1) + [IMG] * 6 + txt(1) + [IMG] * 4 + txt(2), im=[0, 1, 0])]),
}
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "list.txt"), "w") as lst:
    for k, c in cases.items():
        path = os.path.abspath(os.path.join(out, f"{k}.req"))
        write_request(path, IMG, VID, c["state"], c["media"], c["qi"], c["qs"])
        lst.write(path + "\n")
