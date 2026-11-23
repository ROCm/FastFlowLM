# Compile every probe case: build/<name>/{final.xclbin,insts.bin}, and cases.tsv for run.sh / run.ps1.
import os, subprocess
from concurrent.futures import ThreadPoolExecutor
import pages

CASES = []
for P in (8, 16, 32, 64, 128, 256, 1024):
    CASES.append(("stream", 256, P))
    for mode in ("scatter", "gather"):
        for R in (128, 256):
            CASES.append((mode, R, P))

cols, fifos, total = 8, 2, 1 << 20
nch = cols * fifos
rows, jobs = [], []
for mode, R, P in CASES:
    name = f"{mode}_r{R}_p{P}"
    d = os.path.join("build", name)
    os.makedirs(d, exist_ok=True)
    mod, ib, ob, passes = pages.design(mode, R, P, cols, fifos, total)
    open(os.path.join(d, "aie.mlir"), "w").write(mod)
    if not os.path.exists(os.path.join(d, "final.xclbin")):
        jobs.append(d)
    rows.append(f"{name}\t{mode}\t{R}\t{P}\t{ib}\t{ob}\t{passes * P * pages.PAGE}\t{nch}")


def compile_one(d):
    subprocess.run(["aiecc.py", "--aie-generate-xclbin", "--no-compile-host", "--xclbin-name=final.xclbin",
                    "--no-xchesscc", "--no-xbridge", "--aie-generate-npu-insts", "--npu-insts-name=insts.bin",
                    "aie.mlir"], cwd=d, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


with ThreadPoolExecutor(int(os.environ.get("JOBS", "12"))) as ex:
    list(ex.map(compile_one, jobs))
open("build/cases.tsv", "w").write("\n".join(rows) + "\n")
