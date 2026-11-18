# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import sys, statistics as st, collections
fn = sys.argv[1]; NK = int(sys.argv[2]) if len(sys.argv) > 2 else 4; P = 46; OFF = 2   # samples: 0 entry, 1 after zero, then NK*P, then epi_start, chunks 0,7,15,23,31
calls = {}
for line in open(fn):
    f = line.strip().split(',')
    if f[4] == 'BAD': continue
    blk, rnd, r, c = map(int, f[:4]); base = int(f[4]); n = int(f[5]); d = list(map(int, f[6:6+n-1]))
    calls[(r, c, blk*12 + rnd)] = (base, n, d)
print("calls", len(calls), "samples/call", collections.Counter(v[1] for v in calls.values()).most_common(2))
names = ['qwacq0','dq0','qwacq1','dq1','barA'] + sum([[f'xacq0_g{g}',f'conv0_g{g}',f'xacq1_g{g}',f'conv1_g{g}',f'mac_g{g}'] for g in range(8)], []) + ['barB']
def cat(nm):
    for k in ['qwacq','dq','barA','xacq','conv','mac','barB']:
        if nm.startswith(k): return k
cyc = collections.defaultdict(list)
for (r, c, rd), (base, n, d) in calls.items():
    if rd == 0: continue
    s = collections.defaultdict(int)
    s['zero_acc'] = d[0]            # sample1 - sample0
    s['first_qw_wait'] = d[1]       # sample2 (after first qw acquire) - sample1
    for q in range(OFF + 1, OFF + NK*P):
        s[cat(names[(q - OFF) % P])] += d[q-1]
    base_q = OFF + NK*P            # epi_start sample
    s['epi_start_gap'] = d[base_q - 1]
    for i, lab in enumerate(['epi_c0', 'epi_c7', 'epi_c15', 'epi_c23', 'epi_c31']):
        s[lab] = d[base_q + i]
    for k, v in s.items(): cyc[k].append(v)
tot = 0
print("\ncycles per call (mean | median), core rows/cols pooled, round>0:")
order = ['zero_acc','first_qw_wait','qwacq','dq','barA','xacq','conv','mac','barB','epi_start_gap','epi_c0','epi_c7','epi_c15','epi_c23','epi_c31']
for k in order:
    v = cyc[k]; tot += st.mean(v)
    print(f"{k:14s} {st.mean(v):9.0f} | {st.median(v):9.0f}")
print("sum", round(tot), "cycles =", round(tot/1810, 1), "us")
# inter-call spacing from base (entry sample)
sp = []
for r in range(4):
    for c in range(8):
        for k in range(2, 47):
            a = calls.get((r, c, k-1)); b = calls.get((r, c, k))
            if a and b and a[0] and b[0]: sp.append((b[0] - a[0]) & 0xFFFFFFFF)
print("period entry->entry: mean %.0f median %.0f (%.1f us)" % (st.mean(sp), st.median(sp), st.median(sp)/1810))
# per column view of the epilogue and first wait
print("\nper column (mean over rows/rounds): zero_acc, first_qw_wait, epi_total(c31), xacq total")
for c in range(8):
    ent = [calls[(r, c, rd)] for r in range(4) for rd in range(1, 48) if (r, c, rd) in calls]
    fq = st.mean([e[2][1] for e in ent]); epi = st.mean([sum(e[2][OFF+NK*P-1:OFF+NK*P+5]) for e in ent])
    xa = st.mean([sum(e[2][OFF+q-1] for q in range(NK*P) if 'xacq' in names[q % P]) for e in ent])
    print(c, "first_qw_wait %7.0f  epilogue %7.0f  xacq %7.0f" % (fq, epi, xa))
