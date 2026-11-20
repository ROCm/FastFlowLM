#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Compares the dev kit's assembled kernels with the ones FastFlowLM ships.

    devkit_xclbin_compare.py --devkit <dir> [--shipped src/xclbins] [--json <file>]
                             [--github-output]

<devkit> is the devkit-xclbins artifact (devkit_assemble_xclbins.py's tree,
<model folder>/<slot>.xclbin); <shipped> is FastFlowLM's src/xclbins/, laid
out the same. Every <folder>/<slot>.xclbin of either tree is one of:

  identical      the same bytes (sha256)
  same-pdi       different files whose AIE_PARTITION PDI images are all byte
                 for byte the same. The rest may still differ: the row lists
                 every other section that does (other_sections_differ: by
                 name, e.g. IP_LAYOUT, EMBEDDED_METADATA), and
                 "aie_partition (non-PDI)" when the AIE_PARTITION section
                 differs outside its PDI images and their uuids (cdo groups,
                 kernel ids, start columns, ...). An empty list means only
                 the xclbin header and PDI uuids differ
  differs        PDI images differ (sizes and differing bytes reported), or
                 a file has no AIE_PARTITION section, cannot be parsed or
                 cannot be read
  only-devkit    the dev kit built it; FastFlowLM ships no such file
  only-shipped   FastFlowLM ships it in a model folder this run built, but this
                 run did not build this slot
  not-built      FastFlowLM ships it in a model folder this run did not build
                 at all (a pull request builds only the kernels it touches)

The comparison is only a report: it never fails (exit 0 whatever it finds,
2 for bad arguments). Some units (generic_decoding_layer and the like) do not
build bit for bit the same twice (packet-switch register values, unzeroed ELF
data), so for them "differs" is expected and says nothing on its own; the
Dequant and MM kernels do build the same, and should be identical or
same-pdi. Prints the table and counts; --json writes every row; with
--github-output the table goes to $GITHUB_STEP_SUMMARY when that is set.

The xclbin (axlf) and AIE_PARTITION layouts are read in pure Python, from
XRT's xclbin.h: no xclbinutil.

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Optional

MAGIC = b"xclbin2\0"
# struct axlf: m_magic[8], m_signature_length (i32), reserved[28],
# m_keyBlock[256], m_uniqueId (u64), then struct axlf_header.
HEADER_OFF = 8 + 4 + 28 + 256 + 8
# axlf_header: m_length, m_timeStamp, m_featureRomTimeStamp (u64 each),
# m_versionPatch (u16), m_versionMajor, m_versionMinor (u8), m_mode,
# m_actionMask (u16), m_interface_uuid[16], m_platformVBNV[64], uuid[16],
# m_debug_bin[16], m_numSections (u32), 4 bytes padding; then the section
# headers.
NUM_SECTIONS_OFF = HEADER_OFF + 24 + 8 + 16 + 64 + 16 + 16
SECTIONS_OFF = NUM_SECTIONS_OFF + 8
# axlf_section_header: m_sectionKind (u32), m_sectionName[16], 4 bytes
# padding, m_sectionOffset, m_sectionSize (u64).
SECTION_HEADER = 40
AIE_PARTITION = 32
# struct aie_partition: schema_version, padding, mpo_name, operations_per_cycle,
# padding, two u64 fingerprints (32 bytes), aie_partition_info (88), then
# array_offset aie_pdi {u32 size, u32 offset}.
PARTITION_PDIS_OFF = 32 + 88
PARTITION_SIZE = 184
# struct aie_pdi: uuid[16], array_offset pdi_image, array_offset cdo_groups,
# reserved[64].
PDI_SIZE = 96

CLASSES = ("identical", "same-pdi", "differs", "only-devkit", "only-shipped", "not-built")


class XclbinError(ValueError):
    pass


def sections(data: bytes) -> list[tuple[int, str, bytes]]:
    """(kind, name, payload) of every section of an xclbin."""
    if data[:8] != MAGIC:
        raise XclbinError("not an xclbin (no xclbin2 magic)")
    if len(data) < SECTIONS_OFF:
        raise XclbinError("truncated header")
    (count,) = struct.unpack_from("<I", data, NUM_SECTIONS_OFF)
    if SECTIONS_OFF + count * SECTION_HEADER > len(data):
        raise XclbinError(f"{count} section headers do not fit in {len(data)} bytes")
    out = []
    for i in range(count):
        off = SECTIONS_OFF + i * SECTION_HEADER
        (kind,) = struct.unpack_from("<I", data, off)
        name = data[off + 4:off + 20].split(b"\0", 1)[0].decode("ascii", "replace")
        start, size = struct.unpack_from("<QQ", data, off + 24)
        if start + size > len(data):
            raise XclbinError(f"section {i} ({kind}) runs past the end of the file")
        out.append((kind, name, data[start:start + size]))
    return out


def split_partition(sec: bytes) -> tuple[list[bytes], bytes]:
    """(PDI images, the section with every aie_pdi uuid and PDI image
    zeroed) of one AIE_PARTITION section."""
    if len(sec) < PARTITION_SIZE:
        raise XclbinError("AIE_PARTITION section too small")
    count, first = struct.unpack_from("<II", sec, PARTITION_PDIS_OFF)
    if first + count * PDI_SIZE > len(sec):
        raise XclbinError("AIE_PARTITION aie_pdi array runs past the section")
    images, rest = [], bytearray(sec)
    for i in range(count):
        entry = first + i * PDI_SIZE
        size, off = struct.unpack_from("<II", sec, entry + 16)
        if off + size > len(sec):
            raise XclbinError(f"PDI {i} runs past the AIE_PARTITION section")
        images.append(sec[off:off + size])
        rest[entry:entry + 16] = bytes(16)
        rest[off:off + size] = bytes(size)
    return images, bytes(rest)


def pdi_images(data: bytes) -> list[bytes]:
    """The PDI images of every AIE_PARTITION section, in order."""
    return [img for kind, _, sec in sections(data) if kind == AIE_PARTITION
            for img in split_partition(sec)[0]]


def other_differences(a: bytes, b: bytes) -> list[str]:
    """The sections that differ between two xclbins besides the PDI images
    and their uuids: by name (or kind), "aie_partition (non-PDI)" for the
    rest of an AIE_PARTITION section."""
    def keyed(data: bytes) -> dict:
        out, seen = {}, {}
        for kind, name, sec in sections(data):
            if kind == AIE_PARTITION:
                sec = split_partition(sec)[1]
            label = "aie_partition (non-PDI)" if kind == AIE_PARTITION else (name or f"kind {kind}")
            n = seen[(kind, label)] = seen.get((kind, label), 0) + 1
            out[(kind, label, n)] = sec
        return out
    da, db = keyed(a), keyed(b)
    return sorted({k[1] for k in da.keys() | db.keys() if da.get(k) != db.get(k)})


def differing_bytes(a: bytes, b: bytes) -> int:
    """Bytes that differ between two equally long buffers."""
    x = (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).to_bytes(len(a), "little")
    return len(x) - x.count(0)


def compare_files(devkit: bytes, shipped: bytes) -> dict:
    row: dict = {}
    if hashlib.sha256(devkit).digest() == hashlib.sha256(shipped).digest():
        row["class"] = "identical"
        return row
    try:
        a, b = pdi_images(devkit), pdi_images(shipped)
    except XclbinError as e:
        row.update({"class": "differs", "note": f"unparsable: {e}"})
        return row
    row["devkit_pdi_sizes"] = [len(p) for p in a]
    row["shipped_pdi_sizes"] = [len(p) for p in b]
    if not a or not b:
        row.update({"class": "differs", "note": "no AIE_PARTITION PDI in " +
                    ("either" if not a and not b else "devkit" if not a else "shipped")})
    elif a == b:
        row["class"] = "same-pdi"
        row["other_sections_differ"] = other_differences(devkit, shipped)
    else:
        row["class"] = "differs"
        if len(a) != len(b):
            row["note"] = f"{len(a)} vs {len(b)} PDIs"
        elif all(len(x) == len(y) for x, y in zip(a, b)):
            row["pdi_differing_bytes"] = [differing_bytes(x, y) for x, y in zip(a, b)]
    return row


def xclbins(root: Path) -> set[str]:
    # A broken symlink is listed too, and then reported as unreadable.
    return {p.relative_to(root).as_posix() for p in root.rglob("*.xclbin")
            if p.is_file() or p.is_symlink()} if root.is_dir() else set()


def compare_trees(devkit: Path, shipped: Path) -> list[dict]:
    rows = []
    a, b = xclbins(devkit), xclbins(shipped)
    built_folders = {rel.split("/", 1)[0] for rel in a}
    for rel in sorted(a | b):
        if rel not in b:
            rows.append({"path": rel, "class": "only-devkit"})
        elif rel not in a:
            folder = rel.split("/", 1)[0]
            rows.append({"path": rel, "class": "only-shipped" if folder in built_folders else "not-built"})
        else:
            try:
                row = compare_files((devkit / rel).read_bytes(), (shipped / rel).read_bytes())
            except OSError as e:
                row = {"class": "differs", "note": f"unreadable: {e.strerror or e}"}
            rows.append({"path": rel, **row})
    return rows


def counts(rows: list[dict]) -> dict[str, int]:
    return {c: sum(1 for r in rows if r["class"] == c) for c in CLASSES}


def detail(row: dict) -> str:
    parts = []
    if "devkit_pdi_sizes" in row:
        parts.append(f"PDI bytes {'/'.join(map(str, row['devkit_pdi_sizes'])) or '-'} vs "
                     f"{'/'.join(map(str, row['shipped_pdi_sizes'])) or '-'}")
    if row.get("other_sections_differ"):
        parts.append(f"also differs: {', '.join(row['other_sections_differ'])}")
    if "pdi_differing_bytes" in row:
        parts.append(f"{'/'.join(map(str, row['pdi_differing_bytes']))} bytes differ")
    if "note" in row:
        parts.append(row["note"])
    return "; ".join(parts)


def markdown(rows: list[dict], shipped: str) -> str:
    c = counts(rows)
    out = ["### Dev-kit xclbins vs FastFlowLM's", "",
           f"Each `<folder>/<slot>.xclbin` of this run's assembled kernels against `{shipped}`. "
           "Report only: kernels such as generic_decoding_layer do not build bit for bit the same "
           "twice, so \"differs\" alone is no failure.", "",
           " · ".join(f"{k}: **{v}**" for k, v in c.items()), "",
           "| Slot | Result | Detail |", "|---|---|---|"]
    # The not-built rows of a pull request that built a few kernels would
    # bury the rest: counted above, listed in the JSON.
    for r in rows:
        if r["class"] != "not-built":
            out.append(f"| `{r['path']}` | {r['class']} | {detail(r)} |")
    if c["not-built"]:
        out += ["", f"{c['not-built']} shipped slots in model folders this run did not build are not listed."]
    return "\n".join(out) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--devkit", type=Path, required=True, help="The assembled devkit-xclbins tree.")
    parser.add_argument("--shipped", type=Path, default=Path("src/xclbins"))
    parser.add_argument("--json", type=Path, help="Write every row here.")
    parser.add_argument("--github-output", action="store_true", help="Also write $GITHUB_STEP_SUMMARY.")
    args = parser.parse_args(argv)
    for label, d in (("--devkit", args.devkit), ("--shipped", args.shipped)):
        if not d.is_dir():
            print(f"::warning::xclbin comparison skipped: {label} {d} is not a directory")
            return 0
    try:
        rows = compare_trees(args.devkit, args.shipped)
    except OSError as e:
        # A report: a tree it cannot walk is a note, not a failure (files it
        # cannot read are rows of their own).
        print(f"::warning::xclbin comparison skipped: {e}")
        return 0
    c = counts(rows)
    for r in rows:
        print(f"  {r['class']:<12} {r['path']}{'  ' + detail(r) if detail(r) else ''}")
    print(json.dumps(c))
    try:
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps({"counts": c, "rows": rows}, indent=1) + "\n", encoding="utf-8")
        summary = os.environ.get("GITHUB_STEP_SUMMARY") if args.github_output else None
        if summary:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(markdown(rows, args.shipped.as_posix()))
    except OSError as e:
        print(f"::warning::xclbin comparison: could not write output: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
