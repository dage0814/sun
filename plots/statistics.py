from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.text import Text
if __package__:
    from .typography import calibrate_print_text, configure_serif_fonts
else:
    from typography import calibrate_print_text, configure_serif_fonts

STEMS = ["gp", "fusion", "paths", "models", "efficiency", "window", "loss", "ranking"]
BLUE, YELLOW, GREEN = "#397CB7", "#E6C85C", "#6FAF8E"
GRAY, PALE = "#81858A", "#E5E6E8"
NUMBERS = {}
OUT = Path("figures")

mpl.rcParams.update({
    "font.size": 12.0, "axes.titlesize": 13.0, "axes.labelsize": 12.5,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5,
    "axes.unicode_minus": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "grid.color": "#D4D6D9", "grid.linewidth": 0.65, "grid.alpha": 0.75,
})
configure_serif_fonts()


def read_data(file):
    with Path(file).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The summary file is empty.")
    result = {}
    for row in rows:
        key = (row["figure"], row["configuration"], row["metric"])
        mean, sd = float(row["mean"]), float(row["sd"])
        if not np.isfinite([mean, sd]).all() or sd < 0:
            raise ValueError(f"Invalid mean or SD: {key}")
        if key in result:
            raise ValueError(f"Duplicate summary: {key}")
        result[key] = (mean, sd)
    return result


def series(stem, rows, metric):
    pairs = [NUMBERS[(stem, name, metric)] for name in rows]
    return np.asarray([pair[0] for pair in pairs]), np.asarray([pair[1] for pair in pairs])


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_color("#000000")
        text.set_alpha(1.0)
    calibrate_print_text(fig)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style(ax, axis="x"):
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")


def complexity():
    stem = STEMS[0]
    labels = ["完整CSF-GP", "移除复杂度约束", "移除相似度过滤", "移除精英保留", "移除Hoist变异"]
    nodes, nodes_sd = series(stem, labels, "平均公式节点数")
    rankic, rankic_sd = series(stem, labels, "RankIC")
    corr, _ = series(stem, labels, "因子平均绝对相关系数")
    norm = Normalize(vmin=min(0.35, corr.min()), vmax=max(0.70, corr.max()))
    cmap = mpl.colormaps["viridis_r"]
    fig, ax = plt.subplots(figsize=(7.2, 3.92))
    for i, label in enumerate(labels):
        color = cmap(norm(corr[i]))
        ax.errorbar(nodes[i], rankic[i], xerr=nodes_sd[i], yerr=rankic_sd[i],
                    fmt="none", ecolor=color, elinewidth=1.1, capsize=2.5)
        ax.scatter(nodes[i], rankic[i], s=100 if i == 0 else 66, marker="D" if i == 0 else "o",
                   color=color, edgecolor="black", linewidth=0.7, zorder=4)
        offsets = [(8, 23), (-134, -42), (20, -40), (16, -27), (12, 14)]
        ax.annotate(f"{label}\n平均|ρ|={corr[i]:.2f}", (nodes[i], rankic[i]), xytext=offsets[i],
                    textcoords="offset points", fontsize=11.5,
                    arrowprops={"arrowstyle": "-", "color": "#606060", "linewidth": 0.65},
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5})
    ax.set(xlabel="平均公式节点数（越低越简洁）", ylabel="测试集RankIC（越高越好）")
    ax.margins(x=0.18, y=0.25)
    style(ax, "both")
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("因子平均绝对相关系数（越低越好）")
    fig.subplots_adjust(left=0.13, right=0.87, bottom=0.16, top=0.97)
    save(fig, stem)


def fusion():
    stem = STEMS[1]
    names = ["线性加权", "特征拼接+MLP", "CA1", "CA1+CA2", "CA1+CA2+CA3"]
    labels = ["线性加权", "拼接+MLP", "CA1", "CA1+CA2", "完整模型\n(CA1+CA2+CA3)"]
    y = np.arange(5)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.58), sharey=True)
    for ax, metric in zip(axes.flat, ["ACC", "F1", "MCC", "AUROC"]):
        mean, sd = series(stem, names, metric)
        ax.plot(mean, y, color=BLUE, linewidth=1.4, zorder=2)
        ax.errorbar(mean, y, xerr=sd, fmt="o", color=BLUE, markersize=5, elinewidth=1, capsize=2.5)
        ax.scatter([mean[-1]], [y[-1]], s=55, color="#D55E00", edgecolor="black", linewidth=0.5, zorder=5)
        ax.set_title(metric, pad=9)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.margins(x=0.16, y=0.08)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_yticks(y, labels)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    fig.subplots_adjust(left=0.285, right=0.985, bottom=0.08, top=0.94, hspace=0.38, wspace=0.28)
    save(fig, stem)


def component_matrix(ax, active_rows, labels, components, y):
    ax.set(xlim=(-0.55, len(components)-0.45), ylim=(-0.55, len(labels)-0.45))
    for row_index, active in enumerate(active_rows):
        for column in range(len(components)):
            ax.scatter(column, y[row_index], s=54, color=BLUE if column in active else PALE,
                       edgecolor="white", linewidth=0.5, zorder=3)
        if len(active) > 1:
            ax.plot([min(active), max(active)], [y[row_index]]*2, color=BLUE, lw=1.4, zorder=2)
    ax.set_xticks(range(len(components)), components, rotation=35, ha="right")
    ax.set_yticks(y, labels)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def component_plot(stem, names, labels, active_rows, components, height):
    y = np.arange(len(labels))[::-1]
    fig = plt.figure(figsize=(7.2, height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.27)
    matrix = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, 1], sharey=matrix), fig.add_subplot(gs[0, 2], sharey=matrix)]
    component_matrix(matrix, active_rows, labels, components, y)
    for ax, metric in zip(axes, ["MCC", "AUROC"]):
        means, sds = series(stem, names, metric)
        for i, (yi, mean, sd) in enumerate(zip(y, means, sds)):
            ax.errorbar(mean, yi, xerr=sd, fmt="D" if i == 0 else "o",
                        color="#D55E00" if i == 0 else BLUE, markersize=5, elinewidth=1, capsize=2.5)
        ax.set_xlabel(f"{metric}\n（均值 ± 标准差）")
        ax.margins(x=0.15)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
        ax.tick_params(labelleft=False)
        style(ax)
    fig.subplots_adjust(left=0.275, right=0.985, bottom=0.24, top=0.97)
    save(fig, stem)


def paths():
    component_plot(STEMS[2],
        ["完整三路径模型", "量化路径+新闻路径", "新闻路径+关系图路径", "量化路径+关系图路径", "仅新闻路径", "仅量化路径"],
        ["完整三路径", "量化+新闻", "新闻+关系图", "量化+关系图", "仅新闻", "仅量化"],
        [[0, 1, 2], [0, 1], [1, 2], [0, 2], [1], [0]], ["量化", "新闻", "关系图"], 3.50)


def external():
    stem = STEMS[3]
    methods = ["Adv-SHNets", "MagicNet", "MSGCA", "AST-GTN"]
    data = {metric: series(stem, methods, metric) for metric in ("ACC", "F1", "MCC", "AUROC")}
    y = np.arange(4)[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.48), sharey=True)
    for ax, (metric, values) in zip(axes.flat, data.items()):
        mean, sd = [np.array(value, dtype=float) for value in values]
        for i, (yi, m, s, method) in enumerate(zip(y, mean, sd, methods)):
            ax.errorbar(m, yi, xerr=s, fmt="D" if i == 3 else "o", color="#D55E00" if i == 3 else BLUE,
                        markersize=5.5, elinewidth=1, capsize=2.5)
        ax.set_title(metric, pad=8)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.set_xlabel("均值 ± 标准差")
        ax.margins(x=0.18)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_yticks(y, methods)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    fig.subplots_adjust(left=0.235, right=0.985, bottom=0.12, top=0.95, hspace=0.65, wspace=0.24)
    save(fig, stem)


def efficiency():
    stem = STEMS[4]
    labels = ["ReLU", "LeakyReLU", "ELU", "GELU", "SGFRELU"]
    time, time_sd = series(stem, labels, "训练耗时")
    auroc, auroc_sd = series(stem, labels, "AUROC")
    mcc, _ = series(stem, labels, "MCC")
    denominator = mcc.max() - mcc.min()
    size = lambda value: 80 + 850 * (value - mcc.min()) / denominator if denominator > 0 else 300
    colors = ["#606060", "#0072B2", "#E69F00", "#009E73", "#D55E00"]
    fig, ax = plt.subplots(figsize=(7.2, 3.70))
    for i, label in enumerate(labels):
        ax.errorbar(time[i], auroc[i], xerr=time_sd[i], yerr=auroc_sd[i], fmt="none",
                    ecolor=colors[i], elinewidth=1.1, capsize=2.5, zorder=3)
        ax.scatter(time[i], auroc[i], s=size(mcc[i]), color=colors[i], alpha=0.82,
                   edgecolor="black" if i == 4 else "white", linewidth=0.6, zorder=4)
        offsets = [(8, 16), (-57, 20), (20, -13), (-70, 11), (-96, 11)]
        ax.annotate(label, (time[i], auroc[i]), xytext=offsets[i], textcoords="offset points", fontsize=11.5, zorder=6)
    order = np.argsort(time)
    ax.plot(time[order], auroc[order], color=GRAY, linestyle="--", linewidth=1, zorder=1)
    ax.set(xlabel="端到端训练耗时（s，越低越好）", ylabel="AUROC（越高越好）")
    ax.margins(x=0.15, y=0.2)
    style(ax, "both")
    ax.text(0.97, 0.07, "气泡大小表示 MCC", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11.5, color="#000000")
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.18, top=0.96)
    save(fig, stem)


def window():
    stem = STEMS[5]
    windows = np.array([4, 6, 8, 10, 12])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.85), sharex=True)
    for ax, metric in zip(axes.flat, ["ACC", "F1", "MCC", "AUROC"]):
        means, sds = series(stem, [str(w) for w in windows], metric)
        ax.errorbar(windows, means, yerr=sds, color=BLUE, marker="o", markersize=5.2,
                    linewidth=1.8, elinewidth=1, capsize=3, zorder=3)
        ax.axvline(8, color=GREEN, linestyle="--", linewidth=1.1)
        ax.scatter([8], [means[2]], s=45, color="#D55E00", edgecolor="black", linewidth=0.5, zorder=5)
        ax.set(title=metric, ylabel="均值 ± 标准差")
        ax.margins(y=0.18)
        style(ax, "y")
    for ax in axes[1, :]:
        ax.set_xlabel("历史窗口长度 T（交易日）")
        ax.set_xticks(windows)
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.14, top=0.95, hspace=0.38, wspace=0.48)
    save(fig, stem)


def objective():
    component_plot(STEMS[6],
        ["Focal+Soft-MCC+时序Mixup", "Focal+Soft-MCC", "Focal+时序Mixup", "Focal Loss", "Cross-Entropy"],
        ["完整三机制", "Focal+Soft-MCC", "Focal+Mixup", "Focal Loss", "Cross-Entropy"],
        [[0, 1, 2], [0, 1], [0, 2], [0], []], ["Focal", "Soft-MCC", "Mixup"], 3.32)


def ranking():
    stem = STEMS[7]
    names = ["TF-IDF", "Word2Vec-Mean", "BERT-Base", "FinBERT-Sentiment", "完整FinFE-BERT"]
    display = ["TF-IDF", "Word2Vec-Mean", "BERT-Base", "FinBERT-Sentiment", "FinFE-BERT"]
    data = {metric: series(stem, names, metric) for metric in ("NDCG@3", "NDCG@9", "RankIC")}
    y = np.arange(5)[::-1]
    fig = plt.figure(figsize=(7.2, 3.49))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.29)
    first = fig.add_subplot(gs[0, 0])
    second = fig.add_subplot(gs[0, 1], sharey=first)
    for metric, color, offset, marker in [("NDCG@3", "#0072B2", 0.11, "o"), ("NDCG@9", "#D55E00", -0.11, "s")]:
        mean, sd = map(np.asarray, data[metric])
        first.errorbar(mean, y + offset, xerr=sd, fmt=marker, color=color,
                       markersize=5, elinewidth=1, capsize=2.5, label=metric)
    mean, sd = map(np.asarray, data["RankIC"])
    for index, (yi, m, s) in enumerate(zip(y, mean, sd)):
        second.errorbar(m, yi, xerr=s, fmt="D" if index == 4 else "o",
                        color="#D55E00" if index == 4 else BLUE, markersize=5, elinewidth=1, capsize=2.5)
    first.set_yticks(y, display)
    first.set(xlabel="NDCG（均值 ± 标准差）")
    first.margins(x=0.12)
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.985), ncol=2,
               frameon=False, columnspacing=1.6, handletextpad=0.6)
    second.tick_params(labelleft=False)
    second.set(xlabel="RankIC（均值 ± 标准差）")
    second.margins(x=0.12)
    first.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
    second.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
    for ax in (first, second):
        style(ax)
    fig.subplots_adjust(left=0.275, right=0.985, bottom=0.18, top=0.85)
    save(fig, stem)


def main():
    global NUMBERS, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--figure", choices=STEMS + ["all"], default="all")
    args = parser.parse_args()
    NUMBERS, OUT = read_data(args.data), args.output
    functions = dict(zip(STEMS, (complexity, fusion, paths, external, efficiency, window, objective, ranking)))
    selected = functions.values() if args.figure == "all" else [functions[args.figure]]
    for function in selected:
        function()


if __name__ == "__main__":
    main()
