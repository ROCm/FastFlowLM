import sys, re, collections
blocks = []
cur = None
for line in open(sys.argv[1]):
    m = re.match(r"VISION L (\d+) \(padded (\d+)\) probe (\d): encode ([\d.]+) ms", line)
    if m:
        cur = {"L": int(m[1]), "probe": int(m[3]), "encode": float(m[4]), "st": {}}
        blocks.append(cur)
        continue
    m = re.match(r"VISION (\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
    if m and cur:
        cur["st"][m[1]] = (int(m[2]), float(m[3]), float(m[4]))
groups = collections.defaultdict(list)
for b in blocks:
    groups[(b["probe"], b["L"])].append(b)
for key, bs in sorted(groups.items()):
    measured = bs[1:]  # first block is warmup
    measured.sort(key=lambda b: b["encode"])
    b = measured[len(measured) // 2]
    print(f"probe {key[0]} L {key[1]}: encode {b['encode']:.1f} ms (runs {[round(x['encode']) for x in measured]})")
    for name, (calls, total, med) in b["st"].items():
        print(f"   {name:16s} {calls:3d} total {total:8.2f} ms  median {med:8.1f} us")
