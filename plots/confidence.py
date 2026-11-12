from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text
if __package__:
    from .typography import calibrate_print_text, configure_serif_fonts
else:
    from typography import calibrate_print_text, configure_serif_fonts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    with args.data.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The interval file is empty.")
    aliases = {"configuration": "comparison", "mean": "delta_mcc",
               "ci_lower": "ci_low", "ci_upper": "ci_high", "holm_p": "p_holm"}
    for row in rows:
        for name, alias in aliases.items():
            if name not in row:
                if alias not in row:
                    raise ValueError(f"Missing column: {name} or {alias}")
                row[name] = row[alias]
    configure_serif_fonts()
    fig, ax = plt.subplots(figsize=(7.25, max(3.6, 0.4 * len(rows))))
    blue, gray = "#0057B8", "#59636E"
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=gray, linestyle="--", linewidth=1.0)
    limits = [0.0]
    for yi, row in zip(y, rows):
        mean, low, high, p = [float(row[key]) for key in ("mean", "ci_lower", "ci_upper", "holm_p")]
        if not np.isfinite([mean, low, high, p]).all() or not 0 <= p <= 1 or low > high:
            raise ValueError(f"Invalid interval: {row['configuration']}")
        color = blue if p < 0.05 else gray

        ax.hlines(yi, low, high, color=color, linewidth=1.6)
        ax.plot([low, high], [yi, yi], "|", color=color, markersize=7)
        ax.plot(mean, yi, "o", color=color, markersize=6)
        ax.text(1.03, yi, f"Holm p={p:.3f}", va="center", ha="left",
                transform=ax.get_yaxis_transform(), fontsize=11.5)
        limits.extend((mean, low, high))
    ax.set_yticks(y, [row["configuration"] for row in rows], fontsize=11.5)
    ax.set_xlabel("MCC配对差值及95%置信区间", fontsize=12)
    padding = max((max(limits) - min(limits)) * 0.12, 0.001)
    ax.set_xlim(min(limits) - padding, max(limits) + padding)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    ax.grid(True, axis="x", color="#C9D2DC", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    legend = [Line2D([0], [0], marker="o", color=blue, label="Holm校正后 p<0.05"),
              Line2D([0], [0], marker="o", color=gray, label="Holm校正后 p≥0.05")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.53, 0.02),
               ncol=2, frameon=False, fontsize=11.5)
    fig.subplots_adjust(left=0.30, right=0.765, bottom=0.24, top=0.96)
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_color("#000000")
        text.set_alpha(1.0)
    calibrate_print_text(fig)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output / f"confidence.{extension}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
