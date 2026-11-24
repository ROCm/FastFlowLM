# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Turn each op's load-time configuration into a C++ table the host can replay.

aiecc --aie-generate-txn writes a design's whole configuration as register
writes. For every op variant we keep what changes between ops:

  sw     stream switch registers (core / shim 0x3F000.., mem tile 0xB0000..)
  mux    shim mux / demux mask writes (0x1F000, 0x1F004 on row 0)
  bds    core tile / mem tile BD blockwrites
  locks  final (= init) lock values (core 0x1F000.., mem tile 0xC0000..)
  starts per DMA channel: start queue value and the control value written after it
  resets every DMA channel control register (to reset before a switch)

The core program, program memory and data memory come from the base xclbin.

    python3 extract_txn2.py --base base/.../input_with_addresses.mlir --rtp ct_rtp:2:0 \
        --op A=A/txn.mlir --op B=B/txn.mlir -o op_configs.hpp
"""
import argparse
import re


def split_addr(addr):
    """NPU2 register address = col << 25 | row << 20 | offset"""
    return (addr >> 25) & 0x7F, (addr >> 20) & 0x1F, addr & 0xFFFFF


def kind_of(row):
    return "shim" if row == 0 else "mem" if row == 1 else "core"


RANGES = {
    "core": dict(sw=(0x3F000, 0x40000), bd=(0x1D000, 0x1D200), ctrl=(0x1DE00, 0x1DE20), lock=(0x1F000, 0x1F100)),
    "mem": dict(sw=(0xB0000, 0xB1000), bd=(0xA0000, 0xA0600), ctrl=(0xA0600, 0xA0660), lock=(0xC0000, 0xC0400)),
    "shim": dict(sw=(0x3F000, 0x40000), mux=(0x1F000, 0x1F010)),
}


def in_range(off, r):
    return r is not None and r[0] <= off < r[1]


def parse_globals(text):
    out = {}
    for name, body in re.findall(r'memref\.global "private" constant @(\w+)\s*:\s*memref<\d+xi32>\s*=\s*dense<(.*?)>', text):
        body = body.strip()
        if body.startswith('"0x'):
            raw = bytes.fromhex(body[3:-1])
            out[name] = [int.from_bytes(raw[i:i + 4], "little") for i in range(0, len(raw), 4)]
        else:
            out[name] = [int(v) & 0xFFFFFFFF for v in body.strip("[]").split(",")]
    return out


def parse_txn(path):
    text = open(path).read()
    glob = parse_globals(text)
    ssa = dict(re.findall(r'(%\w+)\s*=\s*memref\.get_global\s+@(\w+)', text))
    cfg = dict(sw={}, mux=[], bds=[], locks={}, starts=[], ctrls=set())
    pending = {}
    lock_writes = {}
    for line in text.splitlines():
        m = re.search(r'aiex\.npu\.blockwrite\((%\w+)\)\s*\{address = (\d+)', line)
        if m:
            col, row, off = split_addr(int(m.group(2)))
            words = glob[ssa[m.group(1)]]
            r = RANGES[kind_of(row)]
            if in_range(off, r.get("bd")):
                cfg["bds"].append((col, row, off, words))
            elif in_range(off, r.get("sw")):
                for i, w in enumerate(words):
                    cfg["sw"][(col, row, off + 4 * i)] = w
            continue
        m = re.search(r'aiex\.npu\.maskwrite32 \{address = (\d+) : ui32, mask = (\d+) : ui32, value = (\d+)', line)
        if m:
            col, row, off = split_addr(int(m.group(1)))
            r = RANGES[kind_of(row)]
            if in_range(off, r.get("mux")):
                cfg["mux"].append((col, row, off, int(m.group(2)), int(m.group(3))))
            elif in_range(off, r.get("ctrl")):
                cfg["ctrls"].add((col, row, off))
            continue
        m = re.search(r'aiex\.npu\.write32 \{address = (\d+) : ui32, value = (\d+)', line)
        if not m:
            continue
        col, row, off = split_addr(int(m.group(1)))
        value = int(m.group(2))
        r = RANGES[kind_of(row)]
        if in_range(off, r.get("sw")):
            cfg["sw"][(col, row, off)] = value
        elif in_range(off, r.get("lock")):
            cfg["locks"][(col, row, off)] = value
            lock_writes[(col, row, off)] = lock_writes.get((col, row, off), 0) + 1
        elif in_range(off, r.get("ctrl")):
            if off % 8 == 4:
                pending[(col, row, off - 4)] = value
            elif (col, row, off) in pending:
                cfg["starts"].append((col, row, off, off + 4, pending.pop((col, row, off)), value))
                cfg["ctrls"].add((col, row, off))
    assert not pending, pending
    # aie-rt zeroes all 16 locks of a core tile, then writes the declared ones
    # again with their init value; mem tiles only get the declared ones. Keep
    # only declared locks: the others stay 0 whichever op runs.
    cfg["locks"] = {key: v for key, v in cfg["locks"].items()
                    if key[1] == 1 or lock_writes[key] >= 2}
    return cfg


def rtp_address(path, sym):
    m = re.search(r'aie\.buffer\(%\w+\) \{address = (\d+) : i32[^}]*sym_name = "' + sym + '"', open(path).read())
    assert m, f"{sym} not found"
    return int(m.group(1))


def emit(cfg, name, shared_bds):
    L = [f"inline const op_config {name} = {{",
         "    // stream switch registers: (col, row, offset, value)",
         "    {"]
    for (c, r, o), v in sorted(cfg["sw"].items()):
        L.append(f"        {{{c}, {r}, 0x{o:05X}, 0x{v:08X}}},")
    L += ["    },", "    // shim mux / demux mask writes: (col, row, offset, mask, value)", "    {"]
    for c, r, o, mask, v in cfg["mux"]:
        L.append(f"        {{{c}, {r}, 0x{o:05X}, 0x{mask:08X}, 0x{v:08X}}},")
    L += ["    },", "    // BD registers: (col, row, offset, shared, words)", "    {"]
    for c, r, o, words in cfg["bds"]:
        sh = "true" if (c, r, o) in shared_bds else "false"
        L.append(f"        {{{c}, {r}, 0x{o:05X}, {sh}, {{{', '.join(f'0x{w:08X}' for w in words)}}}}},")
    L += ["    },", "    // lock values: (col, row, offset, value)", "    {"]
    for (c, r, o), v in sorted(cfg["locks"].items()):
        L.append(f"        {{{c}, {r}, 0x{o:05X}, {v}}},")
    L += ["    },", "    // channels: (col, row, control offset, start queue offset, queue value, control value)", "    {"]
    for c, r, ctrl, q, qv, cv in cfg["starts"]:
        L.append(f"        {{{c}, {r}, 0x{ctrl:05X}, 0x{q:05X}, 0x{qv:08X}, {cv}}},")
    L += ["    },", "};", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", action="append", required=True, help="NAME=txn.mlir")
    ap.add_argument("--base", required=True, help="base input_with_addresses.mlir")
    ap.add_argument("--rtp", action="append", default=[], help="buffer symbol to export as an address")
    ap.add_argument("-o", required=True)
    args = ap.parse_args()

    ops = []
    for spec in args.op:
        name, path = spec.split("=")
        ops.append((name, parse_txn(path)))

    L = ["// Generated by tools/extract_txn2.py from aiecc --aie-generate-txn. Do not edit.",
         "#pragma once",
         "#include <cstdint>",
         "#include <vector>",
         "",
         "struct reg_write { uint32_t col, row, offset, value; };",
         "struct mask_write { uint32_t col, row, offset, mask, value; };",
         "// shared: another op uses this BD register too -> rewrite it at every switch;",
         "// otherwise it can stay loaded (write it once after the xclbin is loaded)",
         "struct reg_block { uint32_t col, row, offset; bool shared; std::vector<uint32_t> words; };",
         "struct chain_start { uint32_t col, row, ctrl_offset, queue_offset, queue_value, ctrl_value; };",
         "struct op_config {",
         "    std::vector<reg_write> sw;",
         "    std::vector<mask_write> mux;",
         "    std::vector<reg_block> bds;",
         "    std::vector<reg_write> locks;",
         "    std::vector<chain_start> starts;",
         "};",
         ""]
    for spec in args.rtp:
        sym = spec.split(":")[0]
        L.append(f"constexpr uint32_t ADDR_{sym} = 0x{rtp_address(args.base, sym):05X};")
    L.append("")
    # every channel control register any op touches: reset these before a switch
    ctrls = sorted(set().union(*[cfg["ctrls"] for _, cfg in ops]))
    L += ["// all DMA channel control registers used by any op: (col, row, offset)",
          "inline const std::vector<reg_write> all_channels = {"]
    for c, r, o in ctrls:
        L.append(f"    {{{c}, {r}, 0x{o:05X}, 0}},")
    L += ["};", ""]
    # BD registers more than one op writes
    seen = {}
    for _, cfg in ops:
        for c, r, o, _w in cfg["bds"]:
            seen[(c, r, o)] = seen.get((c, r, o), 0) + 1
    shared_bds = {key for key, n in seen.items() if n > 1}
    for name, cfg in ops:
        L += emit(cfg, f"op_{name}", shared_bds)
        n_sh = sum(1 for c, r, o, _w in cfg["bds"] if (c, r, o) in shared_bds)
        print(f"op {name}: {len(cfg['sw'])} switch regs, {len(cfg['mux'])} mux writes, {len(cfg['bds'])} BDs ({n_sh} shared) "
              f"({sum(len(b[3]) for b in cfg['bds'])} words), {len(cfg['locks'])} locks, {len(cfg['starts'])} channels")
    open(args.o, "w").write("\n".join(L))
    print(f"-> {args.o}")


if __name__ == "__main__":
    main()
