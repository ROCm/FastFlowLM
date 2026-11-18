# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Writer of the request files tools/d3_req_run.cpp reads (see the header of d3_req_run.cpp for the layout).

    write_request(path, image_id, video_id, state_ids, state_media, qimages, questions)

state_media / qimages: (grids [[gh, gw], ...], pixels uint16 array [sum gh*gw, 1536] holding bf16 bits) or None.
questions: list of dicts {"ids": [...], "offsets": [...], "image_ids": [...]}.
"""
import struct

import numpy as np


def to_bf16_bits(x) -> np.ndarray:
    """float32 -> bf16 bits, round to nearest even (torch's .to(bfloat16))."""
    bits = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _ints(f, v):
    v = np.asarray(v, dtype="<i4")
    f.write(struct.pack("<i", len(v)))
    f.write(v.tobytes())


def _payload(f, media, validate=True):
    grids, pix = media if media else ([], np.zeros((0, 1536), np.uint16))
    f.write(struct.pack("<i", len(grids)))
    for gh, gw in grids:
        f.write(struct.pack("<ii", gh, gw))
    assert not validate or pix.shape == (sum(gh * gw for gh, gw in grids), 1536), (pix.shape, grids)
    f.write(np.ascontiguousarray(pix, dtype="<u2").tobytes())


def write_request(path, image_id, video_id, state_ids, state_media, qimages, questions, validate=True):
    with open(path, "wb") as f:
        f.write(b"D3RQ" + struct.pack("<I", 1))
        f.write(struct.pack("<ii", image_id, video_id))
        _ints(f, state_ids)
        _payload(f, state_media, validate)
        _payload(f, qimages, validate)
        f.write(struct.pack("<i", len(questions)))
        for q in questions:
            _ints(f, q["ids"])
            _ints(f, q.get("offsets", []))
            _ints(f, q.get("image_ids", []))
