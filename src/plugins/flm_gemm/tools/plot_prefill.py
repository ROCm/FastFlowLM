"""Bar chart of the prefill medians for each combination of the plugin's options.

Medians are literals: fill them in from your own bench3.sh runs.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Two independent options, so four points: the stock kernel or the IRON one,
# each with the dequant left in the prefill loop or moved offline.
BARS = [
    ("stock", 939.3),
    ("stock\n+ offline dequant", 733.6),
    ("new bfp16 GEMM", 874.7),
    ("new bfp16 GEMM\n+ offline dequant", 669.9),
]


def main():
    labels = [b[0] for b in BARS]
    values = [b[1] for b in BARS]

    plt.style.use("dark_background")
    colors = sns.color_palette("mako_r", len(values))

    fig, ax = plt.subplots(figsize=(4.4, 2.6), dpi=220)
    y = range(len(values))
    ax.barh(list(y), values, color=colors, height=0.66, edgecolor="none", zorder=3)

    for i, value in enumerate(values):
        ax.text(value + max(values) * 0.02, i, f"{value:.0f}", va="center",
                ha="left", fontsize=8, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7.5, linespacing=1.4)
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
    out = os.environ.get("FLM_PLOT", "prefill.png")
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
