# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""cmp_q4nx.py A.q4nx B.q4nx : compare the safetensors-style headers (names, dtype, shape, bytes)."""
import json
import struct
import sys


def header(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n))


a, b = header(sys.argv[1]), header(sys.argv[2])
ka, kb = set(a) - {'__metadata__'}, set(b) - {'__metadata__'}
print('only in A:', sorted(ka - kb)[:20], len(ka - kb))
print('only in B:', sorted(kb - ka)[:20], len(kb - ka))
bad = 0
for k in sorted(ka & kb):
    ea, eb = a[k], b[k]
    sa = ea['data_offsets'][1] - ea['data_offsets'][0]
    sb = eb['data_offsets'][1] - eb['data_offsets'][0]
    if ea['dtype'] != eb['dtype'] or ea['shape'] != eb['shape'] or sa != sb:
        bad += 1
        if bad <= 25:
            print('DIFF', k, ea['dtype'], ea['shape'], sa, '|', eb['dtype'], eb['shape'], sb)
print('tensors', len(ka), len(kb), 'diff', bad)
print('metadata A', a.get('__metadata__'))
print('metadata B', b.get('__metadata__'))
