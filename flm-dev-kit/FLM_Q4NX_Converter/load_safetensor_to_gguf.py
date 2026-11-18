#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Copy every tensor of a .safetensors file into a .gguf file, keeping each tensor's dtype.

    python load_safetensor_to_gguf.py model.safetensors [out.gguf] [--arch NAME] [--exclude-prefix P]...

Needs: pip install gguf numpy
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as GGType
from gguf import GGUFWriter

# safetensors dtype -> (numpy dtype that holds the raw bytes, GGUF tensor type)
DTYPES = {
    "F64": (np.float64, GGType.F64),
    "F32": (np.float32, GGType.F32),
    "F16": (np.float16, GGType.F16),
    "BF16": (np.uint16, GGType.BF16),
    "I64": (np.int64, GGType.I64),
    "I32": (np.int32, GGType.I32),
    "I16": (np.int16, GGType.I16),
    "I8": (np.int8, GGType.I8),
    "U8": (np.uint8, GGType.I8),
    "BOOL": (np.uint8, GGType.I8),
}


def read_index(file):
    (index_size,) = struct.unpack("<Q", file.read(8))
    index = json.loads(file.read(index_size))
    data_base = 8 + index_size
    return index, data_base


def gguf_key(metadata_key):
    if "." in metadata_key:
        return metadata_key
    return "general." + metadata_key


def read_tensor(file, data_base, name, entry):
    if entry["dtype"] not in DTYPES:
        raise ValueError(f"tensor {name!r}: dtype {entry['dtype']!r} is not supported")
    holder, gguf_type = DTYPES[entry["dtype"]]
    start, stop = entry["data_offsets"]
    file.seek(data_base + start)
    values = np.frombuffer(file.read(stop - start), dtype=holder)
    shape = entry["shape"]
    if shape:
        values = values.reshape(shape)
    return values, shape, gguf_type


def convert(src, dst, arch, skip_prefixes):
    with open(src, "rb") as file:
        index, data_base = read_index(file)
        metadata = index.pop("__metadata__", {})

        writer = GGUFWriter(str(dst), arch=arch)
        for key, value in metadata.items():
            writer.add_string(gguf_key(key), str(value))

        in_file_order = sorted(index.items(), key=lambda item: item[1]["data_offsets"][0])
        for name, entry in in_file_order:
            if skip_prefixes and name.startswith(skip_prefixes):
                print(f"  [skip] {name}")
                continue
            values, shape, gguf_type = read_tensor(file, data_base, name, entry)
            writer.add_tensor(name, values, raw_shape=shape or None, raw_dtype=gguf_type)
            print(f"  {name:<50} {entry['dtype']:<5} {tuple(shape)}")

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="Copy a .safetensors file into a .gguf file, keeping dtypes.")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path, nargs="?", help="default: src with a .gguf suffix")
    parser.add_argument("--arch", default="conversion", help="value of general.architecture")
    parser.add_argument("--exclude-prefix", metavar="PREFIX", action="append", default=[],
                        help="skip tensors whose name starts with PREFIX (can be repeated)")
    args = parser.parse_args()

    dst = args.dst or args.src.with_suffix(".gguf")
    print(f"Converting {args.src} -> {dst}")
    convert(args.src, dst, args.arch, tuple(args.exclude_prefix))
    print("Done.")


if __name__ == "__main__":
    main()
