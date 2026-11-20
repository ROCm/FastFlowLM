# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import collections
import re
import sys

STEP = {"East": (1, 0, "West"), "West": (-1, 0, "East"), "North": (0, 1, "South"), "South": (0, -1, "North")}


def read_flows(logical_mlir):
    text = open(logical_mlir).read()
    tiles = {}
    for name, col, row in re.findall(r"%(\w+) = aie\.tile\((\d+), (\d+)\)", text):
        tiles[name] = (int(col), int(row))

    def ports(kind, body):
        found = re.findall(rf"aie\.packet_{kind}<%(\w+), (\w+) : (\d+)>", body)
        return [(tiles[tile], port, int(ch)) for tile, port, ch in found]

    flows = []
    for pkt_id, body in re.findall(r"aie\.packet_flow\((\d+)\)\s*\{(.*?)\}", text, re.S):
        dests = set(ports("dest", body))
        for src in ports("source", body):
            flows.append((int(pkt_id), src, dests))
    return flows


def read_switchboxes(physical_mlir):
    text = open(physical_mlir).read()
    boxes = {}
    pattern = r"%switchbox_(\d+)_(\d+) = aie\.switchbox\(%\w+\) \{(.*?)\n    \}"
    for col, row, body in re.findall(pattern, text, re.S):
        masters = collections.defaultdict(list)
        for port, ch, amsels in re.findall(r"%\w+ = aie\.masterset\((\w+) : (\d+), ([^)]*)\)", body):
            for amsel in re.findall(r"%(\w+)", amsels):
                masters[amsel].append((port, int(ch)))
        rules = {}
        for port, ch, rule_body in re.findall(r"aie\.packet_rules\((\w+) : (\d+)\) \{(.*?)\}", body, re.S):
            found = re.findall(r"aie\.rule\((\d+), (\d+), %(\w+)\)", rule_body)
            rules[(port, int(ch))] = [(int(mask), int(value), amsel) for mask, value, amsel in found]
        boxes[(int(col), int(row))] = (masters, rules)

    mux_in, mux_out = {}, {}
    for col, row, body in re.findall(r"%shim_mux_(\d+)_(\d+) = aie\.shim_mux\(%\w+\) \{(.*?)\}", text, re.S):
        tile = (int(col), int(row))
        for a, a_ch, b, b_ch in re.findall(r"aie\.connect<(\w+) : (\d+), (\w+) : (\d+)>", body):
            if a == "DMA":
                mux_in[(tile, int(a_ch))] = int(b_ch)
            else:
                mux_out[(tile, int(a_ch))] = int(b_ch)
    return boxes, mux_in, mux_out


def trace(pkt_id, src, boxes, mux_in, mux_out):
    src_tile, src_port, src_ch = src
    start = (src_port, src_ch)
    if src_tile[1] == 0 and src_port == "DMA":
        start = ("South", mux_in.get((src_tile, src_ch), -1))

    delivered, seen = set(), set()
    todo = [(src_tile, start)]
    while todo:
        tile, slave = todo.pop()
        if (tile, slave) in seen or tile not in boxes:
            continue
        seen.add((tile, slave))
        masters, rules = boxes[tile]
        for mask, value, amsel in rules.get(slave, []):
            if (pkt_id & mask) != (value & mask):
                continue
            for port, ch in masters.get(amsel, []):
                if port == "South" and tile[1] == 0:
                    delivered.add((tile, "DMA", mux_out.get((tile, ch), -1)))
                elif port in STEP:
                    d_col, d_row, back = STEP[port]
                    todo.append(((tile[0] + d_col, tile[1] + d_row), (back, ch)))
                else:
                    delivered.add((tile, port, ch))
            break
    return delivered


def main():
    flows = read_flows(sys.argv[1])
    boxes, mux_in, mux_out = read_switchboxes(sys.argv[2])
    wrong = 0
    for pkt_id, src, dests in flows:
        got = trace(pkt_id, src, boxes, mux_in, mux_out)
        extra = sorted(got - dests)
        missing = sorted(dests - got)
        if extra or missing:
            wrong += 1
            print(f"error: packet id {pkt_id} from {src}: extra {extra}, missing {missing}")
    print(f"{len(flows)} packet flow sources traced, {wrong} wrong")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
