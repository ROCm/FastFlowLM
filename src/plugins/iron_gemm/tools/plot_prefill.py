"""Bar chart of the prefill medians, read from bench3.sh output.

    ./bench3.sh stock                              >  runs.txt
    ./bench3.sh dequant $P                         >> runs.txt
    ./bench3.sh bf16    $P IRON_GEMM_MODE=bf16      >> runs.txt
    ./bench3.sh bfp16   $P IRON_GEMM_MODE=bfp16     >> runs.txt
    python3 plot_prefill.py < runs.txt
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Bench label -> how the bar reads. Two independent options, so four points.
BASE = "FastFlowLM v1.0.5"
NAMES = {
    "stock": BASE,
    "bf16": f"{BASE}\n+ offline dequant",
    "dequant": "IRON bfp16 GEMM",
    "bfp16": "IRON bfp16 GEMM\n+ offline dequant",
}

runs = dict(re.findall(r"^(\S+): median ([\d.]+)", sys.stdin.read(), re.M))
bars = [(NAMES[k], float(runs[k])) for k in NAMES if k in runs]
if not bars:
    sys.exit("no '<label>: median <ms>' lines on stdin")

labels = [b[0] for b in bars]
values = [b[1] for b in bars]

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(4.8, 2.6), dpi=220)
ax.barh(range(len(values)), values, color=sns.color_palette("mako_r", len(values)),
        height=0.66, edgecolor="none", zorder=3)

for i, value in enumerate(values):
    ax.text(value + max(values) * 0.02, i, f"{value:.0f}", va="center", ha="left",
            fontsize=8, fontweight="bold")

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=7, linespacing=1.4)
ax.invert_yaxis()
ax.set_xlim(0, max(values) * 1.16)
ax.set_xlabel("prefill (ms, median)", fontsize=7.5)
ax.set_title("Gemma4-E2B, 247-token prompt", fontsize=8.5, pad=8, loc="left")
ax.tick_params(axis="both", labelsize=7.5, length=0)
ax.grid(axis="x", alpha=0.18, linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)

fig.tight_layout()
out = os.environ.get("IRON_PLOT", "prefill.png")
fig.savefig(out, facecolor=fig.get_facecolor())
print(f"wrote {out}")
