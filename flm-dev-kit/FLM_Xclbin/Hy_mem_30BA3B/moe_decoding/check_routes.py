#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Routing check for the routed design (input_physical.mlir), run after every xclbin build.

The packet-switched network can deadlock in ways the compiler does not see: a
switch arbiter carries one packet at a time from start to end, so a packet that
stalls halfway (its receiver is full) blocks every other flow on that arbiter,
even ones leaving through other ports. If the stalled flow is waiting for one of
those blocked flows, nothing moves again. Seen on hardware with the old x relay
design (rms -> relay packets vs MT1's per-column x packets, and proj outputs vs
those x packets).

The check traces every packet flow hop by hop through the switch rules and fails if
  1. a flow does not arrive exactly where it was asked to, or
  2. two flows from FORBIDDEN_PAIRS groups share an arbiter in some switch.

Usage: check_routes.py <input_physical.mlir> <iron layer.mlir>     (exit 1 on a problem)
       The routed file has the switch settings; the IRON file lists the flows asked
       for (routing turns circuit flows into plain switch connections).
"""
import re
import sys
from collections import defaultdict

# ---- flow groups, by (source tile (col, row), source DMA channel, packet id) ----
# Tile positions follow iron/layer_aie.py.
PROJ_COLS = (0, 1, 6, 7)
GROUPS = {
    # packets into the proj engine's x input (MT1 S2MM 3); they stall when MT1's
    # x buffers are full, i.e. until the proj cores use up the x they have
    "x_in": lambda xy, ch, pid: (xy, ch, pid) in {((2, 2), 0, 0),      # rms
                                                  ((3, 1), 4, 2),      # attention O (MT3)
                                                  ((5, 5), 0, 9)},     # GLU
    # proj results on their way out: core -> gather mem tiles -> MT1 -> consumers;
    # the proj cores stop using x when these cannot leave
    "proj_out": lambda xy, ch, pid: (xy[0] in PROJ_COLS and xy[1] >= 2 and ch == 0)
                                    or (xy, ch) in {((0, 1), 4), ((6, 1), 4), ((1, 1), 5)},
    # the router's control packets that start the expert weight streams
    "router_ctrl": lambda xy, ch, pid: (xy, ch) == ((5, 4), 1),
    # flows that may share an arbiter with x_in because they can never be part of a wait
    # loop with it: IT2's sends to rms and rope (layer input, rms weights) finish at the
    # start of a call before any x_in packet exists; router control packets go to shim
    # control registers, which always accept them, so they never stall
    "x_safe": lambda xy, ch, pid: (xy, ch) in {((2, 0), 0), ((2, 0), 1), ((5, 4), 1)},
}
# x_in may share an arbiter only with x_in and x_safe. A wider proj_out-only rule was
# not enough: in one build GLU x packets shared with MT2 -> rms (o_proj / shared-down
# results on their way to rms) and every call hung. Anything that must move before proj
# can use its next x closes the same loop, so other flows are forbidden by default.
X_ONLY_WITH = ("x_in", "x_safe")
FORBIDDEN_PAIRS = [("x_in", "proj_out")]
# Reported but not failing yet: the shipping Qwen3.6 build has this sharing and
# runs; it matters once the routed experts run (to be fixed in the MoE port).
WARN_PAIRS = [("proj_out", "router_ctrl")]

# ---- parse ------------------------------------------------------------------------
text = open(sys.argv[1]).read()
tile_xy = {m.group(1): (int(m.group(2)), int(m.group(3)))
           for m in re.finditer(r"%(\w+)\s*=\s*aie\.tile\((\d+),\s*(\d+)\)", text)}

def blocks(kind):
    """Bodies of aie.<kind>(%tile) { ... }, brace matched."""
    for m in re.finditer(r"aie\.%s\(%%(\w+)\)\s*\{" % kind, text):
        depth, i = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        yield tile_xy[m.group(1)], text[m.end():i - 1]

port = lambda s: tuple(x.strip() for x in s.split(":"))   # "East : 2" -> ("East", "2")
switch = {}                                                # xy -> (rules, amsel masters, amsel arbiter, connects)
for xy, body in blocks("switchbox"):
    masters = defaultdict(list)
    for m in re.finditer(r"aie\.masterset\(([^,]+:[^,]+),\s*([^)]*)\)", body):
        for a in m.group(2).split(","):
            masters[a.strip()].append(port(m.group(1)))
    rules = defaultdict(list)
    for m in re.finditer(r"aie\.packet_rules\(([^)]*)\)\s*\{(.*?)\}", body, re.S):
        for r in re.finditer(r"aie\.rule\((\d+),\s*(\d+),\s*(%\w+)\)", m.group(2)):
            rules[port(m.group(1))].append((int(r.group(1)), int(r.group(2)), r.group(3)))
    arbiter = {m.group(1): int(m.group(2))
               for m in re.finditer(r"(%\w+)\s*=\s*aie\.amsel<(\d+)>\s*\((\d+)\)", body)}
    connects = defaultdict(list)
    for a, b in re.findall(r"aie\.connect<([^,]+),\s*([^>]+)>", body):
        connects[port(a)].append(port(b))
    switch[xy] = (rules, masters, arbiter, connects)
shim_mux = {xy: [(port(a), port(b)) for a, b in re.findall(r"aie\.connect<([^,]+),\s*([^>]+)>", body)]
            for xy, body in blocks("shim_mux")}

STEP = {"East": (1, 0, "West"), "West": (-1, 0, "East"), "North": (0, 1, "South"), "South": (0, -1, "North")}

def start_port(xy, ch):
    """Where a DMA MM2S channel enters its switchbox (shims go through the mux)."""
    if xy[1] == 0:
        for a, b in shim_mux.get(xy, []):
            if a == ("DMA", str(ch)):
                return ("South", b[1])
        return None
    return ("DMA", str(ch))

def arrive(xy, mport):
    """A master port that ends a flow: a DMA, or a shim's South into the mux."""
    if mport[0] == "DMA":
        return (xy, "DMA", mport[1])
    if xy[1] == 0 and mport[0] == "South":
        for a, b in shim_mux.get(xy, []):
            if a == ("North", mport[1]):
                return (xy, b[0], b[1])
    return None

def next_hop(xy, mport):
    if mport[0] in STEP:
        dx, dy, back = STEP[mport[0]]
        return (xy[0] + dx, xy[1] + dy), (back, mport[1])
    return None

def trace_packet(xy, slave, pid, arbs, dests, depth=0):
    if depth > 64 or xy not in switch:
        return
    rules, masters, arbiter, _ = switch[xy]
    for mask, value, amsel in rules.get(slave, []):
        if pid & mask == value & mask:                     # first matching rule wins
            arbs.add((xy, arbiter.get(amsel)))
            for mport in masters[amsel]:
                hop = next_hop(xy, mport)
                end = arrive(xy, mport)
                if end:
                    dests.add(end)
                elif hop:
                    trace_packet(hop[0], hop[1], pid, arbs, dests, depth + 1)
            return

def trace_circuit(xy, slave, dests, depth=0):
    if depth > 64 or xy not in switch:
        return
    for mport in switch[xy][3].get(slave, []):
        end = arrive(xy, mport)
        hop = next_hop(xy, mport)
        if end:
            dests.add(end)
        elif hop:
            trace_circuit(hop[0], hop[1], dests, depth + 1)

# ---- what was asked (from the IRON MLIR) ------------------------------------------
asked = open(sys.argv[2]).read()
asked_xy = {m.group(1): (int(m.group(2)), int(m.group(3)))
            for m in re.finditer(r"%(\w+)\s*=\s*aie\.tile\((\d+),\s*(\d+)\)", asked)}
want_packet = defaultdict(set)
for m in re.finditer(r"aie\.packet_flow\((\d+)\)\s*\{(.*?)\n\s*\}", asked, re.S):
    srcs = re.findall(r"packet_source<%(\w+),\s*(\w+)\s*:\s*(\d+)>", m.group(2))
    dsts = re.findall(r"packet_dest<%(\w+),\s*(\w+)\s*:\s*(\d+)>", m.group(2))
    for s in srcs:
        for d in dsts:
            want_packet[(asked_xy[s[0]], s[1], int(s[2]), int(m.group(1)))].add((asked_xy[d[0]], d[1], d[2]))
want_circuit = defaultdict(set)
for s, sb, sc, d, db, dc in re.findall(r"aie\.flow\(%(\w+),\s*(\w+)\s*:\s*(\d+),\s*%(\w+),\s*(\w+)\s*:\s*(\d+)\)", asked):
    want_circuit[(asked_xy[s], sb, int(sc))].add((asked_xy[d], db, dc))

# ---- check ------------------------------------------------------------------------
problems, warnings = [], []
arb_users = defaultdict(set)
for key, want in sorted(want_packet.items()):
    xy, bundle, ch, pid = key
    if bundle != "DMA":
        continue          # control-port sources (shim TileControl) are not DMA channels
    arbs, dests = set(), set()
    sp = start_port(xy, ch)
    if sp:
        trace_packet(xy, sp, pid, arbs, dests)
    want_dma = {w for w in want if w[1] == "DMA"}
    got_dma = {d for d in dests if d[1] == "DMA"}
    if got_dma != want_dma:
        problems.append(f"packet flow src {xy} ch{ch} id{pid}: asked {sorted(want_dma)}, routed to {sorted(got_dma)}")
    for a in arbs:
        arb_users[a].add(key[:1] + key[2:])                # (xy, ch, pid)
for key, want in sorted(want_circuit.items()):
    xy, bundle, ch = key
    if bundle != "DMA":
        continue
    dests = set()
    sp = start_port(xy, ch)
    if sp:
        trace_circuit(xy, sp, dests)
    want_dma = {w for w in want if w[1] == "DMA"}
    if {d for d in dests if d[1] == "DMA"} != want_dma:
        problems.append(f"circuit flow src {xy} ch{ch}: asked {sorted(want_dma)}, routed to {sorted(dests)}")

for (xy, arb), users in sorted(arb_users.items(), key=lambda t: (t[0][0], t[0][1] or 0)):
    groups = defaultdict(set)
    for u in users:
        for name, member in GROUPS.items():
            if member(*u):
                groups[name].add(u)
    for pairs, out in ((FORBIDDEN_PAIRS, problems), (WARN_PAIRS, warnings)):
        for a, b in pairs:
            if groups[a] and groups[b] and groups[a] != groups[b]:
                out.append(f"switch {xy} arbiter {arb} shared by {a} {sorted(groups[a])} and {b} {sorted(groups[b])}")
    # strict x_in rule: everything else on an x_in arbiter must be in X_ONLY_WITH
    allowed = set().union(*(groups[g] for g in X_ONLY_WITH))
    # flows from the same DMA channel as an x_in flow are fine: a channel sends one
    # packet at a time, so while its x packet is stalled it sends nothing else anyway
    x_channels = {u[:2] for u in groups["x_in"]}
    allowed |= {u for u in users if u[:2] in x_channels}
    if groups["x_in"] and users - allowed:
        problems.append(f"switch {xy} arbiter {arb} shared by x_in {sorted(groups['x_in'])} and {sorted(users - allowed)}")

for w in warnings:
    print("check_routes: WARNING " + w)
if problems:
    print("check_routes: FAILED")
    for p in problems:
        print("  " + p)
    sys.exit(1)
print(f"check_routes: OK ({len(want_packet)} packet flows, {len(want_circuit)} circuit flows)")
