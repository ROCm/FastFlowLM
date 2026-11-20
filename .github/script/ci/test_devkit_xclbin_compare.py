# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Tests for devkit_xclbin_compare.py: python3 -m unittest discover -s ci -p 'test_*.py' -t ci"""

from __future__ import annotations

import io
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import devkit_xclbin_compare as xc  # noqa: E402

REPO_ROOT = HERE.parents[2]


def partition(pdis: list[bytes], uuid: bytes = b"\x11" * 16) -> bytes:
    """An AIE_PARTITION payload: the header, the PDI images, the aie_pdi array."""
    head = bytearray(xc.PARTITION_SIZE)
    images = b"".join(pdis)
    table_off = xc.PARTITION_SIZE + len(images)
    struct.pack_into("<II", head, xc.PARTITION_PDIS_OFF, len(pdis), table_off)
    table, off = bytearray(), xc.PARTITION_SIZE
    for p in pdis:
        entry = bytearray(xc.PDI_SIZE)
        entry[:16] = uuid
        struct.pack_into("<II", entry, 16, len(p), off)
        table += entry
        off += len(p)
    return bytes(head) + images + bytes(table)


def xclbin(sections: list[tuple[int, bytes]], stamp: int = 1) -> bytes:
    """An axlf file with these (kind, payload) sections; stamp goes in m_timeStamp."""
    data = bytearray(xc.SECTIONS_OFF + len(sections) * xc.SECTION_HEADER)
    data[:8] = xc.MAGIC
    struct.pack_into("<Q", data, xc.HEADER_OFF + 8, stamp)
    struct.pack_into("<I", data, xc.NUM_SECTIONS_OFF, len(sections))
    for i, (kind, payload) in enumerate(sections):
        h = xc.SECTIONS_OFF + i * xc.SECTION_HEADER
        struct.pack_into("<I", data, h, kind)
        data[h + 4:h + 4 + len(b"sec")] = b"sec"
        struct.pack_into("<QQ", data, h + 24, len(data), len(payload))
        data += payload
    return bytes(data)


PDI = b"\xdd\0\0\0" + bytes(range(60))


class Parsing(unittest.TestCase):
    def test_pdi_images(self):
        x = xclbin([(6, b"topology"), (xc.AIE_PARTITION, partition([PDI, PDI[::-1]]))])
        self.assertEqual(xc.pdi_images(x), [PDI, PDI[::-1]])
        self.assertEqual([k for k, _, _ in xc.sections(x)], [6, xc.AIE_PARTITION])

    def test_bad_files(self):
        with self.assertRaisesRegex(xc.XclbinError, "magic"):
            xc.pdi_images(b"ELF" + bytes(600))
        x = bytearray(xclbin([(xc.AIE_PARTITION, partition([PDI]))]))
        struct.pack_into("<Q", x, xc.SECTIONS_OFF + 32, 10 ** 6)  # section size
        with self.assertRaisesRegex(xc.XclbinError, "past the end"):
            xc.pdi_images(bytes(x))

    def test_classes(self):
        base = xclbin([(xc.AIE_PARTITION, partition([PDI]))])
        self.assertEqual(xc.compare_files(base, base)["class"], "identical")
        # Another timestamp and PDI uuid: same PDI.
        other = xclbin([(xc.AIE_PARTITION, partition([PDI], b"\x22" * 16))], stamp=2)
        self.assertEqual(xc.compare_files(other, base)["class"], "same-pdi")
        changed = PDI[:10] + b"\xff\xfe" + PDI[12:]
        row = xc.compare_files(xclbin([(xc.AIE_PARTITION, partition([changed]))]), base)
        self.assertEqual((row["class"], row["pdi_differing_bytes"]), ("differs", [2]))
        row = xc.compare_files(xclbin([(xc.AIE_PARTITION, partition([PDI + b"xx"]))]), base)
        self.assertEqual((row["class"], row["devkit_pdi_sizes"], row["shipped_pdi_sizes"]),
                         ("differs", [len(PDI) + 2], [len(PDI)]))
        self.assertIn(f"PDI bytes {len(PDI) + 2} vs {len(PDI)}", xc.detail(row))
        row = xc.compare_files(xclbin([(6, b"x")]), base)
        self.assertEqual((row["class"], row["note"]), ("differs", "no AIE_PARTITION PDI in devkit"))
        row = xc.compare_files(b"junk", base)
        self.assertTrue(row["note"].startswith("unparsable"), row)

    def test_same_pdi_lists_other_differences(self):
        base = xclbin([(8, b"ip layout"), (xc.AIE_PARTITION, partition([PDI]))])
        # Only the header and the PDI uuid: nothing else listed.
        row = xc.compare_files(xclbin([(8, b"ip layout"), (xc.AIE_PARTITION, partition([PDI], b"\x22" * 16))],
                                      stamp=2), base)
        self.assertEqual((row["class"], row["other_sections_differ"]), ("same-pdi", []))
        # Another section differs.
        row = xc.compare_files(xclbin([(8, b"IP LAYOUT"), (xc.AIE_PARTITION, partition([PDI]))]), base)
        self.assertEqual((row["class"], row["other_sections_differ"]), ("same-pdi", ["sec"]))
        self.assertIn("also differs: sec", xc.detail(row))
        # The partition outside its PDIs (here a start column) differs.
        part = bytearray(partition([PDI]))
        part[40] = 3
        row = xc.compare_files(xclbin([(8, b"ip layout"), (xc.AIE_PARTITION, bytes(part))]), base)
        self.assertEqual((row["class"], row["other_sections_differ"]), ("same-pdi", ["aie_partition (non-PDI)"]))
        # A section only one file has.
        row = xc.compare_files(xclbin([(xc.AIE_PARTITION, partition([PDI]))]), base)
        self.assertEqual(row["other_sections_differ"], ["sec"])


class Trees(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def put(self, rel: str, data: bytes) -> None:
        p = self.tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def test_trees_and_report(self):
        base = xclbin([(xc.AIE_PARTITION, partition([PDI]))])
        self.put("devkit/A-NPU2/attn.xclbin", base)
        self.put("shipped/A-NPU2/attn.xclbin", base)
        self.put("devkit/A-NPU2/mm.xclbin", xclbin([(xc.AIE_PARTITION, partition([PDI]))], stamp=7))
        self.put("shipped/A-NPU2/mm.xclbin", base)
        self.put("devkit/A-NPU2/layer.xclbin", xclbin([(xc.AIE_PARTITION, partition([PDI[::-1]]))]))
        self.put("shipped/A-NPU2/layer.xclbin", base)
        self.put("devkit/B-NPU2/new.xclbin", base)
        self.put("shipped/A-NPU2/vision.xclbin", base)  # a folder the run built, slot it did not
        self.put("shipped/C-NPU2/old.xclbin", base)     # a folder the run did not build
        summary = self.tmp / "summary.md"
        out = io.StringIO()
        os.environ["GITHUB_STEP_SUMMARY"] = str(summary)
        try:
            with redirect_stdout(out):
                rc = xc.main(["--devkit", str(self.tmp / "devkit"), "--shipped", str(self.tmp / "shipped"),
                              "--json", str(self.tmp / "out/compare.json"), "--github-output"])
        finally:
            del os.environ["GITHUB_STEP_SUMMARY"]
        self.assertEqual(rc, 0)
        report = json.loads((self.tmp / "out/compare.json").read_text())
        self.assertEqual(report["counts"], {"identical": 1, "same-pdi": 1, "differs": 1,
                                            "only-devkit": 1, "only-shipped": 1, "not-built": 1})
        self.assertEqual({r["path"]: r["class"] for r in report["rows"]}, {
            "A-NPU2/attn.xclbin": "identical", "A-NPU2/layer.xclbin": "differs",
            "A-NPU2/mm.xclbin": "same-pdi", "B-NPU2/new.xclbin": "only-devkit",
            "A-NPU2/vision.xclbin": "only-shipped", "C-NPU2/old.xclbin": "not-built"})
        md = summary.read_text()
        self.assertIn("| `A-NPU2/mm.xclbin` | same-pdi |", md)
        self.assertIn("| `A-NPU2/vision.xclbin` | only-shipped |", md)
        self.assertNotIn("C-NPU2/old.xclbin", md)  # counted, not listed
        self.assertIn("1 shipped slots in model folders this run did not build", md)

    def run_main(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            rc = xc.main(list(args))
        return rc, out.getvalue()

    def test_missing_trees_are_a_warning_not_a_report(self):
        self.put("devkit/A-NPU2/attn.xclbin", b"x")
        for devkit, shipped, bad in (("none", "none2", "--devkit"), ("devkit", "none2", "--shipped")):
            with self.subTest(bad=bad):
                rc, out = self.run_main("--devkit", str(self.tmp / devkit), "--shipped", str(self.tmp / shipped),
                                        "--json", str(self.tmp / "out.json"))
                self.assertEqual(rc, 0)
                self.assertIn(f"::warning::xclbin comparison skipped: {bad}", out)
                self.assertFalse((self.tmp / "out.json").exists())

    def test_unreadable_files_are_rows(self):
        base = xclbin([(xc.AIE_PARTITION, partition([PDI]))])
        self.put("devkit/A-NPU2/attn.xclbin", base)
        self.put("shipped/A-NPU2/attn.xclbin", base)
        self.put("shipped/A-NPU2/mm.xclbin", base)
        (self.tmp / "devkit/A-NPU2/mm.xclbin").symlink_to(self.tmp / "gone.xclbin")
        rc, out = self.run_main("--devkit", str(self.tmp / "devkit"), "--shipped", str(self.tmp / "shipped"),
                                "--json", str(self.tmp / "out.json"))
        self.assertEqual(rc, 0)
        rows = {r["path"]: r for r in json.loads((self.tmp / "out.json").read_text())["rows"]}
        self.assertEqual(rows["A-NPU2/attn.xclbin"]["class"], "identical")
        self.assertEqual(rows["A-NPU2/mm.xclbin"]["class"], "differs")
        self.assertTrue(rows["A-NPU2/mm.xclbin"]["note"].startswith("unreadable"), rows)

    def test_output_it_cannot_write_is_a_warning(self):
        base = xclbin([(xc.AIE_PARTITION, partition([PDI]))])
        self.put("devkit/A-NPU2/attn.xclbin", base)
        self.put("shipped/A-NPU2/attn.xclbin", base)
        self.put("file", b"")
        rc, out = self.run_main("--devkit", str(self.tmp / "devkit"), "--shipped", str(self.tmp / "shipped"),
                                "--json", str(self.tmp / "file/out.json"))
        self.assertEqual(rc, 0)
        self.assertIn("::warning::xclbin comparison: could not write output", out)


class RealFiles(unittest.TestCase):
    def test_shipped_xclbins_parse(self):
        found = sorted((REPO_ROOT / "src/xclbins").glob("*/*.xclbin"))[:20]
        if not found:
            self.skipTest("no src/xclbins")
        for p in found:
            with self.subTest(path=p.name):
                data = p.read_bytes()
                self.assertTrue(xc.pdi_images(data))
                self.assertEqual(xc.compare_files(data, data)["class"], "identical")


if __name__ == "__main__":
    unittest.main()
