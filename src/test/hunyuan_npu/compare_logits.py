"""Compare two test_long --dump folders (reference first), one row per prompt length.

Usage: python3 compare_logits.py <ref_dir> <test_dir>
"""
import glob
import os
import re
import sys

import numpy as np


def load_bf16(path):
    raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32)
    return (raw << 16).view(np.float32).astype(np.float64)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def top_k(x, k):
    return set(np.argpartition(-x, k)[:k].tolist())


def lengths_in(folder):
    found = set()
    for path in glob.glob(os.path.join(folder, "L*_prefill.bin")):
        found.add(int(re.match(r"L(\d+)_", os.path.basename(path)).group(1)))
    return sorted(found)


def decode_steps(folder, n):
    steps = glob.glob(os.path.join(folder, f"L{n}_decode*.bin"))
    return len(steps)


def compare_pair(ref_path, test_path):
    ref = load_bf16(ref_path)
    test = load_bf16(test_path)
    same_top1 = int(np.argmax(ref) == np.argmax(test))
    top5_overlap = len(top_k(ref, 5) & top_k(test, 5))
    return cosine(ref, test), same_top1, top5_overlap


def main():
    ref_dir, test_dir = sys.argv[1], sys.argv[2]
    print(f"{'tokens':>8} {'prefill cos':>12} {'top1':>5} {'top5':>5}"
          f" {'decode cos min':>15} {'mean':>9} {'top1 same':>10}")
    for n in lengths_in(test_dir):
        name = f"L{n}_prefill.bin"
        if not os.path.exists(os.path.join(ref_dir, name)):
            continue
        p_cos, p_top1, p_top5 = compare_pair(os.path.join(ref_dir, name), os.path.join(test_dir, name))

        steps = min(decode_steps(ref_dir, n), decode_steps(test_dir, n))
        cosines = []
        same = 0
        for i in range(steps):
            name = f"L{n}_decode{i}.bin"
            c, t1, _ = compare_pair(os.path.join(ref_dir, name), os.path.join(test_dir, name))
            cosines.append(c)
            same += t1
        d_min = min(cosines) if cosines else float("nan")
        d_mean = sum(cosines) / len(cosines) if cosines else float("nan")
        print(f"{n:>8} {p_cos:>12.5f} {'yes' if p_top1 else 'NO':>5} {p_top5:>4}/5"
              f" {d_min:>15.5f} {d_mean:>9.5f} {same:>6}/{steps}")


if __name__ == "__main__":
    main()
