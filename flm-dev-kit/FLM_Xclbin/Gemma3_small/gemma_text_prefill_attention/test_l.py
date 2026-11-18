# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import sys


def main(l_begin, l_in):
    L_MIN = 256
    L_CHUNK_SIZE = 64

    l_end = l_begin + l_in
    l_begin_chunked = (l_begin // L_CHUNK_SIZE) * L_CHUNK_SIZE
    l_offset = l_begin - l_begin_chunked
    l_end_chunked = l_begin_chunked + (l_in + l_offset + L_MIN - 1) // L_MIN * L_MIN

    l_effective = l_end_chunked - l_begin_chunked

    sliding_l_begin = (l_begin_chunked - 1024) if (l_begin_chunked - 1024) > 0 else 0
    sliding_l_length = (l_end_chunked - sliding_l_begin)
    in_memory = (l_begin - 1024) if (l_begin - 1024) > 0 else 0
    l_idx = in_memory % 1024
    local_offset = in_memory - (sliding_l_begin)

    print(f"l_begin, l_in: {l_begin, l_in}")
    print(f"l_begin_chunked: {l_begin_chunked}")
    print(f"l_offset: {l_offset}")
    print(f"l_end_chunked: {l_end_chunked}")
    print(f"l_effective: {l_effective}")
    print(f"sliding_l_begin: {sliding_l_begin}")
    print(f"sliding_l_length: {sliding_l_length}")
    print(f"in_memory: {in_memory}")
    print(f"l_idx: {l_idx}")
    print(f"local_offset: {local_offset}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <l_begin> <l_in>")
        sys.exit(1)
    l_begin = int(sys.argv[1])
    l_in = int(sys.argv[2])
    main(l_begin, l_in)
